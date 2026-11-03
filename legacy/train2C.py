# train.py (完整修改版, with Dynamic pos_weight)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
# --- DDP and System Imports ---
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset2 import TrajectoryDataset
from predictor_model2A import PredictorTransformer

# ==============================================================================
# --- 超参数和配置 ---
# ==============================================================================
ZARR_PATH = 'opt13b_trajectories_scores_multi_point' 
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 8
USE_AMP = True

# OPT-13B Specifics
SEQ_LEN = 2048
NUM_PREDICT_LAYERS = 5
D_MODEL = 5120

# ALPHA-based training
ALPHA = 2
BUDGET_RATIO = 0.3 

# 新模型的超参数
PREDICTOR_NHEAD = 16
PREDICTOR_ENCODER_LAYERS = 8
PREDICTOR_D_HID = 4096

# ==============================================================================
# --- DDP 设置函数 (保持不变) ---
# ==============================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

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
        print(f"Approach: Multi-Label Classification with DYNAMIC targets.")
        print(f"Using ALPHA-based thresholding with ALPHA={ALPHA}.")
        if BUDGET_RATIO:
            print(f"Applying a hard budget cap of {BUDGET_RATIO*100}%.")
        # <<< NEW >>>
        print("Using DYNAMIC pos_weight for BCEWithLogitsLoss.")
        print(f"Effective Global Batch Size: {BATCH_SIZE * world_size * GRAD_ACCUMULATION_STEPS}")

    # 1. 数据加载 (保持不变)
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

    # 2. 模型初始化 (保持不变)
    model = PredictorTransformer(
        input_dim=D_MODEL, num_opt_layers=NUM_PREDICT_LAYERS, nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS, d_hid=PREDICTOR_D_HID
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 3. 优化器, 调度器, AMP GradScaler (保持不变)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                       steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, 
                       epochs=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # <<< MODIFIED >>> 移除固定的损失函数定义
    # 我们将在循环内部动态创建它

    if rank == 0:
        print("--- Starting Training ---")
        
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", disable=(rank != 0))
        
        for i, (embeddings, true_scores_all_layers) in enumerate(pbar):
            embeddings = embeddings.to(device)
            true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                src_padding_mask = (embeddings.sum(dim=-1) == 0)
                predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                
                # --- 动态生成二值目标张量 (逻辑不变) ---
                with torch.no_grad():
                    masked_scores = true_scores.masked_fill(src_padding_mask.unsqueeze(1), -torch.inf)
                    max_scores, _ = torch.max(masked_scores, dim=-1, keepdim=True)
                    thresholds = max_scores - ALPHA
                    target = (masked_scores >= thresholds).float()
                    if BUDGET_RATIO is not None and BUDGET_RATIO < 1.0:
                        effective_lengths = (~src_padding_mask).sum(dim=1) # Shape: (B)
                        # Shape: (B) -> (B, 1) -> (B, L)
                        k_cap = (effective_lengths * BUDGET_RATIO).int().unsqueeze(1).expand(-1, NUM_PREDICT_LAYERS)
                        
                        num_selected = target.sum(dim=-1) # Shape: (B, L)
                        
                        # 找出哪些样本-层对超过了上限
                        exceeded_mask = num_selected > k_cap

                        # 如果有任何超出的情况，需要用topk重新生成这些目标
                        if torch.any(exceeded_mask):
                            # 对超出的部分用 topk 重新生成 target
                            # 这是一个略复杂但精确的操作，为避免循环，我们可以在整个张量上操作
                            # 这里提供一个简化但效果近似的思路：如果超过了，就直接用 topk
                            # 完整的批处理实现比较复杂，我们先用一个可行的简化方案
                            k = (effective_lengths * BUDGET_RATIO).clamp(min=1).int() # Shape: (B)
                            # 遍历batch中的每个样本 (因为它们的k不同)
                            for b in range(true_scores.shape[0]):
                                # 检查该样本是否有任何层超限
                                if torch.any(exceeded_mask[b]):
                                    # 对该样本的所有层都应用 k_cap
                                    kb = k[b].item()
                                    length = effective_lengths[b].item()
                                    _, topk_indices = torch.topk(true_scores[b, :, :length], k=kb, dim=-1)
                                    # 重置并重新填充目标
                                    target[b].fill_(0.0)
                                    target[b].scatter_(-1, topk_indices, 1.0)
                
                # --- <<< NEW: 动态计算 pos_weight >>> ---
                loss_mask = (~src_padding_mask).unsqueeze(1).float()
                
                # 在非padding区域内计算正负样本数量
                num_positives = (target * loss_mask).sum()
                num_negatives = ((1 - target) * loss_mask).sum()
                
                # 防止除以零
                epsilon = 1e-7
                pos_weight = num_negatives / (num_positives + epsilon)
                pos_weight_tensor = torch.tensor([pos_weight], device=device)
                
                # 动态创建损失函数
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')
                # --- <<< END NEW >>> ---

                # --- 计算损失 ---
                unmasked_loss = loss_fn(predicted_logits, target)
                loss = (unmasked_loss * loss_mask).sum() / loss_mask.sum()
                
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
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}', 
                                 'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                                 'pos_w': f'{pos_weight.item():.2f}'}) # 在进度条中显示权重

        # --- 验证阶段 (同样应用动态 pos_weight) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(rank != 0))
            for embeddings, true_scores_all_layers in pbar_val:
                embeddings = embeddings.to(device)
                true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    src_padding_mask = (embeddings.sum(dim=-1) == 0)
                    predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                    
                    masked_scores = true_scores.masked_fill(src_padding_mask.unsqueeze(1), -torch.inf)
                    max_scores, _ = torch.max(masked_scores, dim=-1, keepdim=True)
                    thresholds = max_scores - ALPHA
                    target = (masked_scores >= thresholds).float()

                    loss_mask = (~src_padding_mask).unsqueeze(1).float()
                    num_positives = (target * loss_mask).sum()
                    num_negatives = ((1 - target) * loss_mask).sum()
                    epsilon = 1e-7
                    pos_weight = num_negatives / (num_positives + epsilon)
                    pos_weight_tensor = torch.tensor([pos_weight], device=device)

                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')
                    
                    unmasked_loss = loss_fn(predicted_logits, target)
                    loss = (unmasked_loss * loss_mask).sum() / loss_mask.sum()
                
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
                
                if rank == 0:
                    pbar_val.set_postfix({'val_loss': f'{loss.item():.4f}', 'pos_w': f'{pos_weight.item():.2f}'})
        
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
            torch.save(model.module.state_dict(), f'predictor_alpha_{ALPHA}_dyn_posw_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()