# train_layered_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import math 
# --- DDP and System Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR

from dataset2 import TrajectoryDataset
# <<< 1. 导入新的分层模型 >>>
from predictor_model_layered import SingleLayerPredictorMLP

# ==============================================================================
# --- 超参数和配置 ---
# ==============================================================================
ZARR_PATH = 'opt13b_trajectories_scores_2t'
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 8
USE_AMP = True

# OPT-13B Specifics
SEQ_LEN = 2048
NUM_LAYERS = 40
D_MODEL = 5120

# --- 新增: 预测器特定超参数 ---
# 我们只为前5层训练独立的预测器
NUM_PREDICTOR_LAYERS = 5 
# <<< 2. 改进损失函数: 使用更小的k，让模型专注于最重要的token >>>
PREDICTOR_K = 50 
# <<< 3. 改进模型: MLP内部结构和Dropout >>>
PREDICTOR_HIDDEN_DIMS = [4096, 2048]
PREDICTOR_DROPOUT = 0.2
MARGIN = 0.5


# ==============================================================================
# --- DDP 设置函数 (无变化) ---
# ==============================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

# ==============================================================================
# --- 辅助函数: 排序归一化 ---
# ==============================================================================
# <<< 4. 改进训练目标: 对真实分数进行排序归一化 >>>
def rank_normalize(scores):
    """
    将分数转换为基于排序的归一化值。
    最高的分数变为1.0，最低的变为接近0。
    输入 shape: [B, S]
    输出 shape: [B, S]
    """
    ranks = scores.argsort(dim=-1).argsort(dim=-1)
    normalized_ranks = ranks.float() / (scores.shape[-1] - 1)
    return normalized_ranks

# ==============================================================================
# --- 损失函数 (无变化) ---
# ==============================================================================
def pairwise_ranking_loss(predicted_scores, true_scores, attention_mask=None, margin=1.0, k=34, sampling_ratio=0.1):
    fill_value = torch.finfo(predicted_scores.dtype).min
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(1).expand_as(predicted_scores)
        predicted_scores = predicted_scores.clone()
        predicted_scores.masked_fill_(mask, fill_value)
    else:
        predicted_scores = predicted_scores.clone()
    B, L, S = predicted_scores.shape
    with torch.no_grad():
        _, topk_indices = torch.topk(true_scores, k=k, dim=-1)
        positive_mask = torch.zeros_like(true_scores, dtype=torch.bool).scatter_(-1, topk_indices, True)
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.unsqueeze(1).expand_as(predicted_scores)
            negative_mask = ~positive_mask & ~expanded_attention_mask
        else:
            negative_mask = ~positive_mask
    all_losses = []
    for b in range(B):
        for l in range(L):
            p_scores = predicted_scores[b, l][positive_mask[b, l]]
            n_scores = predicted_scores[b, l][negative_mask[b, l]]
            num_positives = p_scores.shape[0]
            num_negatives = n_scores.shape[0]
            if num_positives == 0 or num_negatives == 0: continue
            num_samples = int(num_negatives * sampling_ratio)
            if num_samples == 0 and num_negatives > 0: num_samples = 1
            if num_samples == 0: continue
            sampled_indices = torch.randperm(num_negatives, device=n_scores.device)[:num_samples]
            sampled_n_scores = n_scores[sampled_indices]
            margin_violations = margin - (p_scores.unsqueeze(1) - sampled_n_scores.unsqueeze(0))
            loss_for_pair = torch.clamp(margin_violations, min=0).mean()
            all_losses.append(loss_for_pair)
    if not all_losses:
        return torch.tensor(0.0, device=predicted_scores.device, requires_grad=True)
    return torch.stack(all_losses).mean()

# ==============================================================================
# --- 主函数 ---
# ==============================================================================
def main():
    setup_ddp()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print("--- Starting DDP Training with 5 Layered MLP Predictors ---")
        print(f"Hyperparameters: LR={LEARNING_RATE}, K={PREDICTOR_K}, Margin={MARGIN}")
        print(f"MLP Structure: Hidden Dims={PREDICTOR_HIDDEN_DIMS}, Dropout={PREDICTOR_DROPOUT}")

    # 1. 数据加载 (无变化)
    full_dataset = TrajectoryDataset(ZARR_PATH)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    if rank == 0:
        print(f"Data loaded: {train_size} training samples, {val_size} validation samples.")

    # <<< 5. 核心修改: 初始化5个独立的模型 >>>
    models = nn.ModuleList([
        DDP(
            SingleLayerPredictorMLP(
                input_dim=D_MODEL,
                output_dim=SEQ_LEN,
                hidden_dims=PREDICTOR_HIDDEN_DIMS,
                dropout=PREDICTOR_DROPOUT
            ).to(device),
            device_ids=[rank],
            find_unused_parameters=False
        ) for _ in range(NUM_PREDICTOR_LAYERS)
    ])
    
    # 将所有模型的参数组合给优化器
    all_params = []
    for model in models:
        all_params.extend(model.parameters())
    
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE)
    
    # <<< 关键修复 1: 精确计算每个 epoch 的优化器步数 >>>
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION_STEPS)
    
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                           total_steps=steps_per_epoch * NUM_EPOCHS) # 使用 total_steps 更明确
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    if rank == 0:
        print(f"--- Starting Training for {NUM_PREDICTOR_LAYERS} models ---")
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        for model in models:
            model.train()
        train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", disable=(rank != 0))
        
        for i, (embeddings, true_scores) in enumerate(pbar):
            embeddings = embeddings.to(device)
            true_scores = true_scores.to(device)
            
            # --- 假设: 我们总是为序列中的最后一个token进行预测 ---
            # 这是LLM生成任务中最常见的场景
            query_embedding = embeddings[:, -1, :] # Shape: [B, D_MODEL]
            
            total_loss = 0.0
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                # <<< 6. 核心修改: 循环遍历每个模型和对应的层数据 >>>
                for layer_idx in range(NUM_PREDICTOR_LAYERS):
                    predictor_model = models[layer_idx]
                    
                    # 获取当前层对应的真实分数 Shape: [B, SEQ_LEN]
                    target_scores_for_layer = true_scores[:, layer_idx, :]
                    
                    # 对目标分数进行排序归一化
                    normalized_targets = rank_normalize(target_scores_for_layer)
                    
                    # 预测分数 Shape: [B, SEQ_LEN]
                    predicted_scores_for_layer = predictor_model(query_embedding)
                    
                    # Loss函数需要3D输入 [B, L, S], 我们这里 L=1
                    # unsqueeze(1) 将 [B, S] -> [B, 1, S]
                    src_padding_mask = (embeddings.sum(dim=-1) == 0) # Shape: [B, S]

                    loss = pairwise_ranking_loss(
                        predicted_scores_for_layer.unsqueeze(1), 
                        normalized_targets.unsqueeze(1), 
                        attention_mask=src_padding_mask,
                        margin=MARGIN,
                        k=PREDICTOR_K,
                        sampling_ratio=0.1
                    )
                    total_loss += loss

            # 对所有层的平均损失进行反向传播
            avg_loss = total_loss / NUM_PREDICTOR_LAYERS
            scaled_loss = avg_loss / GRAD_ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()
            
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # <<< 关键修复 2: 在优化器更新后，无条件地 step scheduler >>>
                scheduler.step() 

            if rank == 0:
                pbar.set_postfix({'avg_loss': f'{avg_loss.item():.4f}'})

        # --- 验证阶段 ---
        for model in models:
            model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(rank != 0))
            for embeddings, true_scores in pbar_val:
                embeddings = embeddings.to(device)
                true_scores = true_scores.to(device)
                query_embedding = embeddings[:, -1, :]
                
                total_val_loss_for_batch = 0.0
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    for layer_idx in range(NUM_PREDICTOR_LAYERS):
                        target_scores_for_layer = true_scores[:, layer_idx, :]
                        normalized_targets = rank_normalize(target_scores_for_layer)
                        predicted_scores_for_layer = models[layer_idx](query_embedding)
                        
                        src_padding_mask = (embeddings.sum(dim=-1) == 0)

                        loss = pairwise_ranking_loss(
                            predicted_scores_for_layer.unsqueeze(1), 
                            normalized_targets.unsqueeze(1),
                            attention_mask=src_padding_mask,
                            margin=MARGIN, k=PREDICTOR_K
                        )
                        total_val_loss_for_batch += loss.item()
                
                avg_val_loss_for_batch = total_val_loss_for_batch / NUM_PREDICTOR_LAYERS
                loss_tensor = torch.tensor([avg_val_loss_for_batch], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
        
        if rank == 0:
            # 计算平均训练损失需要更精细的记录，这里简化一下
            avg_val_loss = val_loss / len(val_loader)
            print(f"\nEpoch {epoch+1} Summary: Avg Val Loss: {avg_val_loss:.6f}")
            
            # 保存所有模型的状态字典
            state_dicts = {f'model_layer_{i}': models[i].module.state_dict() for i in range(NUM_PREDICTOR_LAYERS)}
            torch.save(state_dicts, f'layered_predictors_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()