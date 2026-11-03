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
ZARR_PATH = 'opt13b_trajectories_scores_2t'
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 8
USE_AMP = True

# OPT-13B Specifics
SEQ_LEN = 2048
NUM_PREDICT_LAYERS = 5
D_MODEL = 5120

# <<< NEW >>> 设定每层要预取的token数量 (Top-K)
# 这是一个关键超参数，决定了我们的正样本数量。可以根据实际需求调整。
# 例如，256 约占总序列长度的 12.5%
K_PREFETCH = 256 

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
# --- <<< REMOVED >>> 损失函数 ---
# ==============================================================================
# pairwise_ranking_loss 函数已被移除，我们将使用 BCEWithLogitsLoss

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
        print(f"Approach: Multi-Label Classification")
        print(f"Targeting prediction for the first {NUM_PREDICT_LAYERS} layers.")
        print(f"Prefetching Top-{K_PREFETCH} indices per layer.")
        print(f"Effective Global Batch Size: {BATCH_SIZE * world_size * GRAD_ACCUMULATION_STEPS}")

    # 1. 数据加载
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
    model = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_PREDICT_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 3. 优化器, 调度器, AMP GradScaler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                       steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, 
                       epochs=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # <<< NEW >>> 初始化损失函数，处理类别不平衡问题
    # 计算正样本的权重，公式为 (负样本数 / 正样本数)
    # 这会加大对正样本(应该预取的token)预测错误的惩罚
    pos_weight_value = math.sqrt((SEQ_LEN - K_PREFETCH) / K_PREFETCH)
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if rank == 0:
        print(f"Using BCEWithLogitsLoss with pos_weight = {pos_weight_value:.2f} to handle class imbalance.")


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
            
            # <<< MODIFIED >>> 核心逻辑修改：从Ranking Loss到Multi-Label Classification
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                src_padding_mask = (embeddings.sum(dim=-1) == 0)
                predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                
                # --- 动态生成二值目标张量 ---
                with torch.no_grad():
                    _, topk_indices = torch.topk(true_scores, k=K_PREFETCH, dim=-1)
                    target = torch.zeros_like(true_scores)
                    target.scatter_(-1, topk_indices, 1.0)
                
                # --- 计算损失 ---
                # 为了防止padding部分对loss产生影响，我们将这些位置的loss权重设为0
                # 创建一个权重掩码，padding位置为0，非padding位置为1
                loss_mask = (~src_padding_mask).unsqueeze(1).float()
                
                # 计算原始loss，BCEWithLogitsLoss默认reduction是'mean'
                unmasked_loss = loss_fn(predicted_logits, target)
                
                # 应用掩码，只计算非padding部分的平均loss
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
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(rank != 0))
            for embeddings, true_scores_all_layers in pbar_val:
                embeddings = embeddings.to(device)
                true_scores = true_scores_all_layers[:, :NUM_PREDICT_LAYERS, :].to(device)
                
                # <<< MODIFIED >>> 验证逻辑同样修改
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    src_padding_mask = (embeddings.sum(dim=-1) == 0)
                    predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                    
                    # --- 动态生成二值目标张量 ---
                    _, topk_indices = torch.topk(true_scores, k=K_PREFETCH, dim=-1)
                    target = torch.zeros_like(true_scores)
                    target.scatter_(-1, topk_indices, 1.0)

                    # --- 计算损失 ---
                    loss_mask = (~src_padding_mask).unsqueeze(1).float()
                    unmasked_loss = loss_fn(predicted_logits, target)
                    loss = (unmasked_loss * loss_mask).sum() / loss_mask.sum()
                
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
                
                if rank == 0:
                    pbar_val.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
            # <<< MODIFIED >>> 更新保存的文件名以反映新方法
            torch.save(model.module.state_dict(), f'predictor_multilabel_k{K_PREFETCH}_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()