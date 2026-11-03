# train2A_final.py
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

# <<< MODIFIED >>> 导入原始的数据集类，无需改动
from dataset2 import TrajectoryDataset
from predictor_model2A import PredictorTransformer

# ==============================================================================
# --- 超参数和配置 ---
# ==============================================================================
ZARR_PATH = 'opt13b_trajectories_scores_2t'
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 8
USE_AMP = True

# OPT-13B Specifics
SEQ_LEN = 2048
# <<< MODIFIED >>> 我们的新目标是预测前5层，这是一个新的关键超参数
NUM_PREDICT_LAYERS = 5
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
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()

# ==============================================================================
# --- 损失函数  ---
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
            if num_positives == 0 or num_negatives == 0:
                continue
            num_samples = int(num_negatives * sampling_ratio)
            if num_samples == 0 and num_negatives > 0:
                num_samples = 1
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
    setup_ddp()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"--- Starting DDP Training on {world_size} GPUs ---")
        print(f"Targeting prediction for the first {NUM_PREDICT_LAYERS} layers.")
        print(f"Effective Global Batch Size: {BATCH_SIZE * world_size * GRAD_ACCUMULATION_STEPS}")

    # 1. 数据加载
    # <<< MODIFIED >>> 使用原始的数据集类，无需任何改动
    full_dataset = TrajectoryDataset(ZARR_PATH)
    
    torch.manual_seed(42)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0, pin_memory=True)
    
    if rank == 0:
        print(f"Data loaded: {train_size} training samples, {val_size} validation samples.")

    # 2. 模型初始化
    # <<< MODIFIED >>> 将 `num_opt_layers` 设置为新的目标层数
    model = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_PREDICT_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 3. 优化器 & AMP GradScaler
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
        
        for i, (embeddings, true_scores_all_layers) in enumerate(pbar):
            embeddings = embeddings.to(device)
            # <<< CRITICAL MODIFICATION >>>
            # 在这里对加载的 true_scores 进行切片，只保留前 `NUM_PREDICT_LAYERS` 层
            true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                src_padding_mask = (embeddings.sum(dim=-1) == 0)
                predicted_scores = model(embeddings, src_padding_mask=src_padding_mask)
                
                topk_val = int(SEQ_LEN * 0.3) 
                loss = pairwise_ranking_loss(
                    predicted_scores, 
                    true_scores, 
                    attention_mask=src_padding_mask,
                    margin=0.5,
                    k=topk_val,
                    sampling_ratio=0.1
                )
                loss = loss / GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    scheduler.step()

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(rank != 0))
            for embeddings, true_scores_all_layers in pbar_val:
                embeddings = embeddings.to(device)
                # <<< CRITICAL MODIFICATION >>>
                # 在验证阶段同样进行切片
                true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    src_padding_mask = (embeddings.sum(dim=-1) == 0)
                    predicted_scores = model(embeddings, src_padding_mask=src_padding_mask)
                    loss = pairwise_ranking_loss(
                        predicted_scores, 
                        true_scores, 
                        attention_mask=src_padding_mask,
                        margin=0.5,
                        k=topk_val,
                        sampling_ratio=0.1
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
            torch.save(model.module.state_dict(), f'predictor_5layers_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()