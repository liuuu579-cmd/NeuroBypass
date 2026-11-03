import os
import argparse
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset_router import RouterDataset
from model_router import RouterTransformer

# ==============================================================================
# --- 配置 ---
# ==============================================================================
# 你的 8 个 Zarr 文件夹的基础名称
ZARR_BASE_NAME = '/home/liushenyu/test/opt13b_greedy_routing_labels_part_' 
NUM_PARTS = 8

BATCH_SIZE = 16  
LEARNING_RATE = 2e-4  # SFT 调大了 LR，让模型快速跳出全猜1的陷阱
NUM_EPOCHS = 10
GRAD_ACCUMULATION_STEPS = 4
USE_AMP = True

NUM_OPT_LAYERS = 40
D_MODEL = 5120
K_TO_SKIP = 6  # 我们数据生成时设定了跳过 6 层

# ==============================================================================
def setup_ddp(local_rank):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    setup_ddp(local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"--- Starting Router SFT Training on {world_size} GPUs ---")

    # 1. 模型初始化
    model = RouterTransformer(
        input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4,
        num_encoder_layers=2, d_hid=2048
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # 2. 读取数据
    if rank == 0:
        print("Loading and concatenating 8 datasets...")
    
    datasets = []
    for i in range(NUM_PARTS):
        path = f"{ZARR_BASE_NAME}{i}"
        if os.path.exists(path):
            datasets.append(RouterDataset(path))
        else:
            if rank == 0: print(f"Warning: {path} not found.")
            
    full_dataset = ConcatDataset(datasets)
    dist.barrier()
    
    if rank == 0:
        print(f"Successfully loaded total {len(full_dataset)} trajectories.")
    
    # 3. 划分数据集
    torch.manual_seed(42)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, drop_last=True)
    
    # 4. 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, epochs=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # 【核心修改点】：引入 pos_weight 解决类别不平衡问题
    # 目标变量 target 中，1.0(Keep) 有 34 个，0.0(Skip) 有 6 个。
    # pos_weight = 负样本数 / 正样本数 = 6.0 / 34.0
    pos_weight = torch.tensor([6.0 / 34.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    
    # 5. 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0 
        
        train_hits = 0
        train_total_skip_targets = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", disable=(rank != 0))
        
        for i, (embeddings, target_masks) in enumerate(pbar):
            embeddings = embeddings.to(device)
            target_masks = target_masks.to(device) 
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                src_padding_mask = (embeddings.sum(dim=-1) == 0)
                predicted_logits = model(embeddings, src_padding_mask=src_padding_mask) 
                
                loss = loss_fn(predicted_logits, target_masks)
                loss = loss / GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            
            with torch.no_grad():
                _, pred_skip_indices = torch.topk(predicted_logits, k=K_TO_SKIP, dim=1, largest=False)
                _, target_skip_indices = torch.topk(target_masks, k=K_TO_SKIP, dim=1, largest=False)
                
                batch_hits = 0
                for b in range(target_masks.size(0)):
                    pred_set = set(pred_skip_indices[b].tolist())
                    target_set = set(target_skip_indices[b].tolist())
                    batch_hits += len(pred_set.intersection(target_set))
                
                train_hits += batch_hits
                train_total_skip_targets += target_masks.size(0) * K_TO_SKIP
            
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}', 
                                  'hit_rate': f'{train_hits/train_total_skip_targets:.3f}'})
        
        model.eval()
        val_loss = 0.0
        val_hits = 0
        val_total_skip_targets = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", disable=(rank != 0))
            for embeddings, target_masks in pbar_val:
                embeddings = embeddings.to(device)
                target_masks = target_masks.to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                    src_padding_mask = (embeddings.sum(dim=-1) == 0)
                    predicted_logits = model(embeddings, src_padding_mask=src_padding_mask)
                    loss = loss_fn(predicted_logits, target_masks)
                
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
                
                _, pred_skip_indices = torch.topk(predicted_logits, k=K_TO_SKIP, dim=1, largest=False)
                _, target_skip_indices = torch.topk(target_masks, k=K_TO_SKIP, dim=1, largest=False)
                
                batch_hits = 0
                for b in range(target_masks.size(0)):
                    pred_set = set(pred_skip_indices[b].tolist())
                    target_set = set(target_skip_indices[b].tolist())
                    batch_hits += len(pred_set.intersection(target_set))
                
                hit_stats = torch.tensor([batch_hits, target_masks.size(0) * K_TO_SKIP], dtype=torch.float32, device=device)
                dist.all_reduce(hit_stats, op=dist.ReduceOp.SUM)
                
                val_hits += hit_stats[0].item()
                val_total_skip_targets += hit_stats[1].item()
                
                if rank == 0:
                    pbar_val.set_postfix({'val_loss': f'{loss.item():.4f}', 
                                          'val_hit_rate': f'{val_hits/val_total_skip_targets:.3f}'})
        
        if rank == 0:
            epoch_train_hit = train_hits / train_total_skip_targets
            epoch_val_hit = val_hits / val_total_skip_targets
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Hit Rate: {epoch_train_hit:.4f} ({train_hits}/{train_total_skip_targets})")
            print(f"  Val Loss:   {val_loss/len(val_loader):.4f} | Val Hit Rate:   {epoch_val_hit:.4f} ({val_hits}/{val_total_skip_targets})")
            torch.save(model.module.state_dict(), f'router_sft_epoch_{epoch+1}.pth')
    
    cleanup_ddp()

if __name__ == '__main__':
    main()