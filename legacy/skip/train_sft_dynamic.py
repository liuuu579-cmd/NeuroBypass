import os
import argparse
import datetime
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
from transformers import AutoModelForCausalLM

from dataset_router_dynamic import DynamicRouterDataset, pad_dynamic_collate_fn
from model_router_v4 import DynamicConditionalRouter

# ==============================================================================
# --- 实验配置 ---
# ==============================================================================
ZARR_BASE_NAME = '/home/liushenyu/test/opt13b_greedy_routing_labels_v3_part_'
MODEL_PATH = "/home/liushenyu/opt-13b/"
NUM_PARTS = 4

BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 8
USE_AMP = True

NUM_OPT_LAYERS = 40
D_MODEL = 5120
MAX_SKIP_LAYERS = 16
class EarlyExitException(Exception):
    """用于在提取完 Layer 1 特征后立即中断前向计算，跳过剩余 38 层"""
    pass
def setup_ddp(local_rank):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def extract_layer1_embeddings(base_model, input_ids, attention_mask=None):
    """
    通过 forward hook 在 Layer 1 结束后立即拦截特征并打断前向
    避免运行后续 38 层，节省 90% 显存并保证底层 mask 与位置编码完全原生兼容
    """
    saved_hidden_state = []

    def hook_fn(module, args, output):
        # OPTDecoderLayer 输出格式通常是 (hidden_states, ...)
        if isinstance(output, tuple):
            saved_hidden_state.append(output[0])
        else:
            saved_hidden_state.append(output)
        # 截获特征后直接抛出异常中断 forward
        raise EarlyExitException()

    # 在 Layer 1 (第 2 层) 上注册临时钩子
    handle = base_model.model.decoder.layers[1].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True
            )
    except EarlyExitException:
        pass
    finally:
        handle.remove()  # 确保钩子必定被移除

    if not saved_hidden_state:
        raise RuntimeError("未能成功截获 Layer 1 的特征，请检查模型结构。")

    return saved_hidden_state[0]

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
        print(f"--- [A+B Route] Starting Conditional Dynamic Router SFT on {world_size} GPUs ---")

    # 1. 挂载冻结的浅层主干抽取器
    if rank == 0: print("Loading frozen backbone feature extractor (Layers 0-1)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16
    ).to(device)
    #tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    #if len(tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
    #    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    # 2. 初始化条件化动态 Router
    model = DynamicConditionalRouter(
        embedding_dim=D_MODEL, 
        hidden_dim=512, 
        num_layers=NUM_OPT_LAYERS, 
        max_skip=MAX_SKIP_LAYERS
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # 3. 聚合 Zarr 数据集
    datasets = []
    for i in range(NUM_PARTS):
        path = f"{ZARR_BASE_NAME}{i}"
        if os.path.exists(path):
            datasets.append(DynamicRouterDataset(path, max_skip_steps=MAX_SKIP_LAYERS, num_layers=NUM_OPT_LAYERS))
        else:
            if rank == 0: print(f"Warning: Dataset shard {path} not found.")
            
    full_dataset = ConcatDataset(datasets)
    dist.barrier()
    
    torch.manual_seed(42)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
        pin_memory=True, collate_fn=pad_dynamic_collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, 
        pin_memory=True, collate_fn=pad_dynamic_collate_fn, drop_last=True
    )
    
    # 4. 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, 
        epochs=NUM_EPOCHS
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # 注意：BCEWithLogitsLoss 不设固定常数 pos_weight，直接对全层进行拟合
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    # 5. 主训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        train_hits = 0
        train_total_targets = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", disable=(rank != 0))
        
        for i, (input_ids, padding_mask, target_k, target_masks) in enumerate(pbar):
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            target_k = target_k.to(device)
            target_masks = target_masks.to(device)
            attention_mask = (~padding_mask).long()
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                # 传入 attention_mask 保证位置编码计算准确无误
                embeddings = extract_layer1_embeddings(base_model, input_ids, attention_mask=attention_mask)
                predicted_logits = model(embeddings, target_k, src_padding_mask=padding_mask.to(device))
                loss = loss_fn(predicted_logits, target_masks) / GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            
            # 统计不同 k 条件下的动态命中率 (Hit Rate)
            with torch.no_grad():
                batch_hits = 0
                total_in_batch = 0
                for b in range(target_masks.size(0)):
                    k_val = target_k[b].item()
                    _, pred_skip_idx = torch.topk(predicted_logits[b], k=k_val, largest=False)
                    _, gt_skip_idx = torch.topk(target_masks[b], k=k_val, largest=False)
                    
                    pred_set = set(pred_skip_idx.tolist())
                    gt_set = set(gt_skip_idx.tolist())
                    batch_hits += len(pred_set.intersection(gt_set))
                    total_in_batch += k_val
                    
                train_hits += batch_hits
                train_total_targets += total_in_batch
            
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            if rank == 0:
                curr_hit = train_hits / max(train_total_targets, 1)
                pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}', 'hit_rate': f'{curr_hit:.3f}'})
                
        # 6. 验证循环
        model.eval()
        val_loss = 0.0
        val_hits = 0
        val_total_targets = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", disable=(rank != 0))
            for input_ids, padding_mask, target_k, target_masks in pbar_val:
                input_ids = input_ids.to(device)
                padding_mask = padding_mask.to(device)
                target_k = target_k.to(device)
                target_masks = target_masks.to(device)
                attention_mask = (~padding_mask).long()
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                    embeddings = extract_layer1_embeddings(base_model, input_ids, attention_mask=attention_mask)
                    predicted_logits = model(embeddings, target_k, src_padding_mask=padding_mask.to(device))
                    loss = loss_fn(predicted_logits, target_masks)
                
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
                
                batch_hits = 0
                total_in_batch = 0
                for b in range(target_masks.size(0)):
                    k_val = target_k[b].item()
                    _, pred_skip_idx = torch.topk(predicted_logits[b], k=k_val, largest=False)
                    _, gt_skip_idx = torch.topk(target_masks[b], k=k_val, largest=False)
                    batch_hits += len(set(pred_skip_idx.tolist()).intersection(set(gt_skip_idx.tolist())))
                    total_in_batch += k_val
                    
                stats = torch.tensor([batch_hits, total_in_batch], dtype=torch.float32, device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                val_hits += stats[0].item()
                val_total_targets += stats[1].item()
                
        if rank == 0:
            epoch_train_hit = train_hits / max(train_total_targets, 1)
            epoch_val_hit = val_hits / max(val_total_targets, 1)
            print(f"\n[Summary Epoch {epoch+1}]")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Hit Rate: {epoch_train_hit:.4f}")
            print(f"  Val Loss:   {val_loss/len(val_loader):.4f} | Val Hit Rate:   {epoch_val_hit:.4f}")
            torch.save(model.module.state_dict(), f'dynamic_router_sft_epoch_{epoch+1}.pth')
            
    cleanup_ddp()

if __name__ == '__main__':
    main()