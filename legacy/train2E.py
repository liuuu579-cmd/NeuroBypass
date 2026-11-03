# train2D.py (Fixed Again)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
# --- DDP and System Imports ---
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset2 import TrajectoryDataset
from predictor_model2A import PredictorTransformer

# ==============================================================================
# --- 超参数和配置 ---
# ==============================================================================
ZARR_PATH = 'opt13b_trajectories_scores_multi_point' 
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 16
USE_AMP = True

# OPT-13B Specifics
SEQ_LEN = 2048
NUM_PREDICT_LAYERS = 5
D_MODEL = 5120
BUDGET_RATIO = 0.3 
NUM_RECENT_TOKENS = 10 

PREDICTOR_NHEAD = 4
PREDICTOR_ENCODER_LAYERS = 1
PREDICTOR_D_HID = 2048

# ==============================================================================
# --- DDP 设置函数 ---
# ==============================================================================
def setup_ddp(local_rank):
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=2)
    )
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ==============================================================================
# --- 主函数 ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    local_rank = int(os.environ['LOCAL_RANK'])
    setup_ddp(local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"--- Starting DDP Training on {world_size} GPUs ---")
        print(f"Using D_MODEL = {D_MODEL}")

    if rank == 0:
        print("Initializing model on all ranks...")
        
    model = PredictorTransformer(
        input_dim=D_MODEL, num_opt_layers=NUM_PREDICT_LAYERS, nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS, d_hid=PREDICTOR_D_HID
    )
    model.to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        print("Model initialization and DDP wrapping complete.")

    if rank == 0:
        print("Starting synchronized data loading...")
        full_dataset = TrajectoryDataset(ZARR_PATH)
        print("Rank 0 dataset loaded.")

    dist.barrier() 

    if rank != 0:
        full_dataset = TrajectoryDataset(ZARR_PATH)

    dist.barrier() 
    if rank == 0:
        print("All datasets loaded successfully.")
    
    torch.manual_seed(42)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, 
                            num_workers=0, pin_memory=True, drop_last=True)
    
    if rank == 0:
        print(f"Data loaded: {train_size} training samples, {val_size} validation samples.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                       steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, 
                       epochs=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    if rank == 0:
        print("--- Starting Training ---")
        
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        # <--- 修复 2：重新引入 train_loss 用于周期总结 ---
        train_loss = 0.0 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", disable=(rank != 0))
        
        for i, (embeddings, true_scores_all_layers) in enumerate(pbar):
            embeddings = embeddings.to(device)
            true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                src_padding_mask = (embeddings.sum(dim=-1) == 0)
                predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                
                with torch.no_grad():
                    effective_lengths = (~src_padding_mask).sum(dim=1)
                    k_values = (effective_lengths * BUDGET_RATIO).int().clamp(min=1)

                    masked_scores = true_scores.masked_fill(src_padding_mask.unsqueeze(1), -torch.inf)
                    
                    k_max = k_values.max().item()
                    _, topk_indices = torch.topk(masked_scores, k=k_max, dim=-1)
                    
                    seq_range = torch.arange(k_max, device=device).view(1, k_max)
                    k_mask = seq_range < k_values.unsqueeze(1)
                    
                    src_values_base = k_mask.unsqueeze(1).float()
                    src_values_expanded = src_values_base.expand_as(topk_indices)

                    target_topk = torch.zeros_like(true_scores, dtype=torch.float)
                    target_topk.scatter_(-1, topk_indices, src_values_expanded)

                    seq_indices = torch.arange(true_scores.shape[-1], device=device).unsqueeze(0)
                    start_indices = (effective_lengths - NUM_RECENT_TOKENS).clamp(min=0).unsqueeze(1)
                    recent_mask = (seq_indices >= start_indices) & (seq_indices < effective_lengths.unsqueeze(1))
                    target_recent = recent_mask.unsqueeze(1).float()

                    target = (target_topk + target_recent).clamp(max=1.0)
                
                loss_mask = (~src_padding_mask).unsqueeze(1).float()
                num_positives = (target * loss_mask).sum()
                num_negatives = ((1 - target) * loss_mask).sum()
                epsilon = 1e-7
                pos_weight = num_negatives / (num_positives + epsilon)
                
                base_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                unreduced_loss = base_loss_fn(predicted_logits, target)

                with torch.no_grad():
                    # 1. 找出所有正样本的分数
                    scores_for_positives = true_scores.masked_fill(target == 0, -torch.inf)
                    # 2. 计算 Softmax 得到相对概率
                    importance_weights_dist = F.softmax(scores_for_positives, dim=-1)
                    
                    # 3. 缩放权重
                    curr_bs = embeddings.shape[0] # 使用当前 batch 的大小，比全局 BATCH_SIZE 更安全
                    k_reshaped = k_values.view(curr_bs, 1, 1)
                    scaled_importance_weights = importance_weights_dist * k_reshaped
                    
                    # 4. 初始化最终权重矩阵
                    final_loss_weights = torch.ones_like(target)
                    final_loss_weights[target == 1] = scaled_importance_weights[target == 1]

                    # ============================================================
                    # >>> 核心修复开始 <<<
                    # 问题：最近的 token 分数可能很低，导致 softmax 后权重接近 0。
                    # 解决：强制要求最近的 token (target_recent == 1) 的权重至少为 1.0。
                    
                    # target_recent 是 float (0.或1.)，转为 bool 掩码
                    recent_bool_mask = (target_recent > 0.5)
                    recent_bool_mask = recent_bool_mask.expand_as(final_loss_weights)
                    # 对这些位置，取 max(当前权重, 1.0)
                    final_loss_weights[recent_bool_mask] = torch.max(
                        final_loss_weights[recent_bool_mask],
                        torch.tensor(1.0, device=device)
                    )

                weighted_loss = unreduced_loss * final_loss_weights
                loss = (weighted_loss * loss_mask).sum() / loss_mask.sum()
                loss = loss / GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            # <--- 修复 2：累加 train_loss ---
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
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}', 
                                 'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                                 'pos_w': f'{pos_weight.item():.2f}'})
        
        # 验证阶段...
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(rank != 0))
            for embeddings, true_scores_all_layers in pbar_val:
                embeddings = embeddings.to(device)
                true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                    src_padding_mask = (embeddings.sum(dim=-1) == 0)
                    predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                    
                    # <--- 核心修复 1：将训练循环中正确的 'target' 计算逻辑复制到这里 ---
                    # --- START of COPIED LOGIC ---
                    effective_lengths = (~src_padding_mask).sum(dim=1)
                    k_values = (effective_lengths * BUDGET_RATIO).int().clamp(min=1)
                    masked_scores = true_scores.masked_fill(src_padding_mask.unsqueeze(1), -torch.inf)
                    
                    # 增加一个对空batch的保护
                    k_max = k_values.max().item() if k_values.numel() > 0 else 1 
                    _, topk_indices = torch.topk(masked_scores, k=k_max, dim=-1)
                    
                    seq_range = torch.arange(k_max, device=device).view(1, k_max)
                    k_mask = seq_range < k_values.unsqueeze(1)
                    
                    # 关键步骤：创建和训练循环中一样的 expanded src 张量
                    src_values_base = k_mask.unsqueeze(1).float()
                    src_values_expanded = src_values_base.expand_as(topk_indices)

                    target_topk = torch.zeros_like(true_scores, dtype=torch.float)
                    # 使用形状匹配的 src_values_expanded 进行 scatter 操作
                    target_topk.scatter_(-1, topk_indices, src_values_expanded)
    
                    seq_indices = torch.arange(true_scores.shape[-1], device=device).unsqueeze(0)
                    start_indices = (effective_lengths - NUM_RECENT_TOKENS).clamp(min=0).unsqueeze(1)
                    recent_mask = (seq_indices >= start_indices) & (seq_indices < effective_lengths.unsqueeze(1))
                    target_recent = recent_mask.unsqueeze(1).float()
    
                    target = (target_topk + target_recent).clamp(max=1.0)
                    # --- END of COPIED LOGIC ---
    
                    loss_mask = (~src_padding_mask).unsqueeze(1).float()
                    num_positives = (target * loss_mask).sum()
                    num_negatives = ((1 - target) * loss_mask).sum()
                    epsilon = 1e-7
                    pos_weight = num_negatives / (num_positives + epsilon)
                    
                    # 这里直接使用标量pos_weight即可，BCEWithLogitsLoss会自动处理
                    base_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                    unreduced_loss = base_loss_fn(predicted_logits, target)
                    
                    scores_for_positives = true_scores.masked_fill(target == 0, -torch.inf)
                    importance_weights_dist = F.softmax(scores_for_positives, dim=-1)
                    
                    val_batch_size = embeddings.shape[0]
                    k_reshaped = k_values.view(val_batch_size, 1, 1)
                    scaled_importance_weights = importance_weights_dist * k_reshaped
                    
                    final_loss_weights = torch.ones_like(target)
                    final_loss_weights[target == 1] = scaled_importance_weights[target == 1]
                    
                    # ============================================================
                    # >>> 核心修复开始 (验证集) <<<
                    # 同样的逻辑应用到验证集，确保 Loss 计算标准一致
                    recent_bool_mask = (target_recent > 0.5)
                    recent_bool_mask = recent_bool_mask.expand_as(final_loss_weights)
                    final_loss_weights[recent_bool_mask] = torch.max(
                        final_loss_weights[recent_bool_mask],
                        torch.tensor(1.0, device=device)
                    )
                    # >>> 核心修复结束 <<<
                    # ============================================================
                    
                    weighted_loss = unreduced_loss * final_loss_weights
                    loss = (weighted_loss * loss_mask).sum() / loss_mask.sum()
                
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
                
                if rank == 0:
                    pbar_val.set_postfix({'val_loss': f'{loss.item():.4f}', 'pos_w': f'{pos_weight.item():.2f}'})
        
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
            torch.save(model.module.state_dict(), f'predictor_topk_{BUDGET_RATIO}_iwl_epoch_{epoch+1}.pth')
    
    cleanup_ddp()

if __name__ == '__main__':
    main()