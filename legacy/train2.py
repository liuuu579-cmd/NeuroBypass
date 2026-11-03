# train2.py 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
# --- DDP and System Imports ---
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset2 import TrajectoryDataset
from predictor_model2 import PredictorTransformer # 使用你之前的新模型

# ==============================================================================
# --- 超参数和配置 ---
# ==============================================================================
ZARR_PATH = 'opt13b_trajectories_scores_2t'
# <<< CRITICAL >>> 这个BATCH_SIZE现在是 PER-GPU 的批次大小
# 例如，如果你有4个GPU，总的批次大小将是 BATCH_SIZE * 4
BATCH_SIZE = 4  # 大幅减小，因为每个GPU都会加载一个批次
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
# <<< NEW >>> 梯度累积，模拟更大的批次大小
# Effective Batch Size = BATCH_SIZE * NUM_GPUS * GRAD_ACCUMULATION_STEPS
GRAD_ACCUMULATION_STEPS = 8 # 例如：2 * 4 GPUs * 8 = 64 的有效批次大小

# <<< NEW >>> 是否使用混合精度训练
USE_AMP = True

# OPT-13B Specifics
SEQ_LEN = 2048
NUM_LAYERS = 40
D_MODEL = 5120

# 新模型的超参数
PREDICTOR_NHEAD = 16
PREDICTOR_ENCODER_LAYERS = 8
PREDICTOR_D_HID = 4096

# ==============================================================================
# --- DDP 设置函数 ---
# ==============================================================================
def setup_ddp():
    """初始化DDP"""
    dist.init_process_group(backend="nccl") # NCCL是NVIDIA GPU推荐的后端
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()

# ==============================================================================
# --- 损失函数---
# ==============================================================================
def direct_attention_kldiv_loss(predicted_scores, true_scores, attention_mask=None, temperature=1.0):
    """
    直接计算预测分数和真实分数之间的KL散度。
    Args:
        predicted_scores (torch.Tensor): 模型输出的分数, shape (B, L, S)
        true_scores (torch.Tensor): 从数据加载的真实分数, shape (B, L, S)
        attention_mask (torch.Tensor): 填充部分的掩码, shape (B, S)
        temperature (float): Softmax温度系数
    """
    # --- 1. 应用掩码 ---
    if attention_mask is not None:
        # 扩展掩码以匹配分数的形状 (B, L, S)
        mask = attention_mask.unsqueeze(1).expand_as(predicted_scores)
        
        # 获取对应数据类型的最小值，以安全地填充
        fill_value = torch.finfo(predicted_scores.dtype).min
        predicted_scores.masked_fill_(mask, fill_value)
        true_scores.masked_fill_(mask, fill_value)

    # --- 2. 计算KL散度 ---
    # Log-Softmax on predicted scores
    log_pred_dist = F.log_softmax(predicted_scores / temperature, dim=-1)
    
    # Softmax on true scores (确保在float32上计算以保证数值稳定性)
    with torch.no_grad():
        true_dist = F.softmax(true_scores.float() / temperature, dim=-1)
    
    # KL Divergence Loss
    loss = F.kl_div(log_pred_dist, true_dist, reduction='batchmean', log_target=False)
    
    return loss

def pairwise_ranking_loss(predicted_scores, true_scores, attention_mask=None, margin=1.0, k=34, sampling_ratio=0.1):
    """
    一个高效的成对排序损失函数，专注于Top-K和随机负样本。
    (已修复AMP兼容性问题)
    """
    # <<< FIX STARTS HERE >>>
    
    # 1. 安全地获取填充值
    # 使用 torch.finfo 来获取 predicted_scores 数据类型 (可能是 float16) 的最小值
    # 这是处理混合精度的标准做法
    fill_value = torch.finfo(predicted_scores.dtype).min
    
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(1).expand_as(predicted_scores)
        # 克隆张量以避免原地修改输入
        predicted_scores = predicted_scores.clone()
        predicted_scores.masked_fill_(mask, fill_value)
    else:
        # 即使没有 attention_mask，我们最好也克隆一下，以保持函数行为一致
        predicted_scores = predicted_scores.clone()
        
    # <<< FIX ENDS HERE >>>

    B, L, S = predicted_scores.shape
    
    # ... (函数的其余部分保持不变) ...
    # 1. 识别正样本和负样本
    with torch.no_grad():
        # 找到每个真实分布中的 top-k 索引
        _, topk_indices = torch.topk(true_scores, k=k, dim=-1)
        
        # 创建正样本和负样本的掩码
        positive_mask = torch.zeros_like(true_scores, dtype=torch.bool).scatter_(-1, topk_indices, True)
        
        # 检查 attention_mask 是否存在，并相应地构建 negative_mask
        if attention_mask is not None:
            # 这里的 mask 还没有被 unsqueeze，所以形状是 (B, S)
            # 我们需要把它扩展到 (B, L, S)
            expanded_attention_mask = attention_mask.unsqueeze(1).expand_as(predicted_scores)
            negative_mask = ~positive_mask & ~expanded_attention_mask
        else:
            negative_mask = ~positive_mask
    
    # 2. 获取对应的预测分数
    all_losses = []
    for b in range(B):
        for l in range(L):
            p_scores = predicted_scores[b, l][positive_mask[b, l]]
            n_scores = predicted_scores[b, l][negative_mask[b, l]]

            num_positives = p_scores.shape[0]
            num_negatives = n_scores.shape[0]
            
            if num_positives == 0 or num_negatives == 0:
                continue

            # 3. 高效地构建成对比较
            num_samples = int(num_negatives * sampling_ratio)
            if num_samples == 0 and num_negatives > 0:
                num_samples = 1 
            
            # 确保有负样本可以采样
            if num_samples == 0:
                continue

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
    # --- DDP初始化 ---
    setup_ddp()
    # 获取当前进程的rank和总进程数
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"--- Starting DDP Training on {world_size} GPUs ---")
        print(f"Per-GPU Batch Size: {BATCH_SIZE}")
        print(f"Gradient Accumulation Steps: {GRAD_ACCUMULATION_STEPS}")
        print(f"Effective Global Batch Size: {BATCH_SIZE * world_size * GRAD_ACCUMULATION_STEPS}")
        print(f"Using Mixed Precision (AMP): {USE_AMP}")

    # 1. 数据加载
    # 只有主进程需要下载/准备数据，但所有进程都需要创建dataset实例
    full_dataset = TrajectoryDataset(ZARR_PATH)
    # --- 数据集分割在DDP中需要保证所有进程分割结果一致，设置随机种子 ---
    torch.manual_seed(42)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # --- 使用 DistributedSampler ---
    # 它能确保每个进程在每个epoch拿到不重复的数据子集
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0, pin_memory=True)
    
    if rank == 0:
        print(f"Data loaded: {train_size} training samples, {val_size} validation samples.")

    # 2. 模型初始化
    model = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(device)
    
    # --- 将模型包装在DDP中 ---
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 3. 优化器 & AMP GradScaler
    #optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    #scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                       steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, 
                       epochs=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    if rank == 0:
        print("--- Starting Training ---")
        
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", disable=(rank != 0))
        
        ### <<< 修改: 更新数据加载和损失函数调用
        for i, (embeddings, true_scores) in enumerate(pbar):
            embeddings = embeddings.to(device)
            true_scores = true_scores.to(device) # <--- 加载真实分数
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                src_padding_mask = (embeddings.sum(dim=-1) == 0)
                predicted_scores = model(embeddings, src_padding_mask=src_padding_mask)
                # <--- 调用新的、直接的损失函数
                topk_val = int(SEQ_LEN * 0.3) 
                loss = pairwise_ranking_loss(
                    predicted_scores, 
                    true_scores, 
                    attention_mask=src_padding_mask,
                    margin=0.5, # margin 是一个可以调整的超参数，可以从 1.0, 0.5, 0.1 开始尝试
                    k=topk_val,
                    sampling_ratio=0.1 # 采样10%的负样本进行比较，以加速训练
                )
                loss = loss / GRAD_ACCUMULATION_STEPS
                #loss = direct_attention_kldiv_loss(predicted_scores, true_scores, attention_mask=src_padding_mask)
                #loss = loss / GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            
            #if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
             #   scaler.step(optimizer)
             #   scaler.update()
             #   optimizer.zero_grad()
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm=1.0 是一个常用的默认值
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}'})

        if (len(train_loader) % GRAD_ACCUMULATION_STEPS != 0):
             scaler.step(optimizer)
             scaler.update()
             optimizer.zero_grad()

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(rank != 0))
            ### <<< 修改: 更新数据加载和损失函数调用
            for embeddings, true_scores in pbar_val:
                embeddings = embeddings.to(device)
                true_scores = true_scores.to(device) # <--- 加载真实分数
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    src_padding_mask = (embeddings.sum(dim=-1) == 0)
                    predicted_scores = model(embeddings, src_padding_mask=src_padding_mask)
                    # <--- 调用新的、直接的损失函数
                    loss = pairwise_ranking_loss(
                        predicted_scores, 
                        true_scores, 
                        attention_mask=src_padding_mask,
                        margin=0.5,
                        k=topk_val, # 使用 topk_val
                        sampling_ratio=0.1 # 验证时也可以用采样，或者不采样设为1.0
                    )
                
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
                
                if rank == 0:
                    pbar_val.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
            torch.save(model.module.state_dict(), f'predictor_direct_scores_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()