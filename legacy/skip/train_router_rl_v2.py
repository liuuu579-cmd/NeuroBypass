import os
import argparse
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset_router import RouterDataset
from model_router import RouterTransformer

# ==============================================================================
# --- 配置 ---
# ==============================================================================
ZARR_BASE_NAME = '/home/liushenyu/test/opt13b_greedy_routing_labels_part_' 
NUM_PARTS = 8

MODEL_PATH = "/home/liushenyu/opt-13b/"
SFT_WEIGHT_PATH = "router_sft_epoch_10.pth" # 拿你刚才最好的权重

BATCH_SIZE = 1       
LEARNING_RATE = 1e-5 
NUM_EPOCHS = 5
GRAD_ACCUMULATION_STEPS = 16
USE_AMP = True

NUM_OPT_LAYERS = 40
D_MODEL = 5120
K_TO_SKIP = 6

# 【新增超级参数】：针对同一句话探索的次数 (建议 4 或 8，不增加显存，只增加一点点前向时间)
NUM_SAMPLES = 4 

class BatchSkipHook:
    def __init__(self, skip_mask):
        self.skip_mask = skip_mask.view(-1, 1, 1)

    def __call__(self, module, args, output):
        hidden_states_in = args[0]
        if isinstance(output, tuple):
            hidden_states_out = output[0]
            mixed_out = self.skip_mask * hidden_states_in + (1 - self.skip_mask) * hidden_states_out
            return (mixed_out,) + output[1:]
        else:
            mixed_out = self.skip_mask * hidden_states_in + (1 - self.skip_mask) * output
            return mixed_out

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
    
    if rank == 0: print(f"--- Starting Router RL (Multi-Sample) on {world_size} GPUs ---")

    from transformers import AutoModelForCausalLM
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    opt_model.eval()
    for param in opt_model.parameters():
        param.requires_grad = False
    transformer_layers = opt_model.model.decoder.layers

    router = RouterTransformer(
        input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048
    ).to(device)
    
    state_dict = torch.load(SFT_WEIGHT_PATH, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    router.load_state_dict(new_state_dict)
    router = DDP(router, device_ids=[rank], find_unused_parameters=False)
    
    datasets = [RouterDataset(f"{ZARR_BASE_NAME}{i}") for i in range(NUM_PARTS) if os.path.exists(f"{ZARR_BASE_NAME}{i}")]
    full_dataset = ConcatDataset(datasets)
    train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    optimizer = optim.AdamW(router.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    for epoch in range(NUM_EPOCHS):
        router.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [RL]", disable=(rank != 0))
        
        for i, (embeddings, _) in enumerate(pbar):
            embeddings = embeddings.to(device, dtype=torch.float16)
            B, S, D = embeddings.shape
            
            src_padding_mask = (embeddings.sum(dim=-1) == 0)
            seq_lengths = (~src_padding_mask).sum(dim=1) - 1
            hf_attn_mask = (~src_padding_mask).long()
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                logits = router(embeddings, src_padding_mask=src_padding_mask) 
                skip_probs = torch.sigmoid(-logits) 
            
            # 【核心修改 1】：针对这一个 Batch，产生 NUM_SAMPLES 组不同的动作
            # skip_probs shape: [B, 40] -> 扩展为 [B, NUM_SAMPLES, 40]
            skip_probs_expanded = skip_probs.unsqueeze(1).expand(B, NUM_SAMPLES, NUM_OPT_LAYERS)
            M_skip = torch.bernoulli(skip_probs_expanded).detach() # [B, N, 40]
            
            # 记录每一次采样跳过了多少层
            num_skipped = M_skip.sum(dim=2) # [B, N]
            
            # 准备存放 N 次探索的 KL 散度
            kl_divs = torch.zeros(B, NUM_SAMPLES, device=device)
            
            with torch.no_grad():
                # 无论跳哪层，Baseline (不跳层) 永远只需算一次！极其省时间！
                base_out = opt_model(inputs_embeds=embeddings, attention_mask=hf_attn_mask)
                base_logits = base_out.logits[torch.arange(B), seq_lengths, :]
                base_probs = F.softmax(base_logits, dim=-1)
                
                # 遍历这 N 种组合，分别测试 KL
                for n in range(NUM_SAMPLES):
                    handles = []
                    for layer_idx in range(NUM_OPT_LAYERS):
                        layer_mask = M_skip[:, n, layer_idx] # 取出第 n 次采样的掩码
                        h = transformer_layers[layer_idx].register_forward_hook(BatchSkipHook(layer_mask))
                        handles.append(h)
                    
                    skip_out = opt_model(inputs_embeds=embeddings, attention_mask=hf_attn_mask)
                    skip_logits = skip_out.logits[torch.arange(B), seq_lengths, :]
                    skip_log_probs = F.log_softmax(skip_logits, dim=-1)
                    
                    for h in handles:
                        h.remove()
                        
                    kl_divs[:, n] = F.kl_div(skip_log_probs, base_probs, reduction='none').sum(dim=-1)
                
            # 【核心修改 2】：计算局部 Advantage
            # 计算这句话专属的平均 KL 作为 Baseline
            baseline_kl = kl_divs.mean(dim=1, keepdim=True) # [B, 1]
            
            # 奖励设计：因为 KL 越低越好，所以当你的 KL 小于 Baseline 时，Advantage 为正！
            # 相比全局滑动平均，这个局部 Advantage 能够极其精准地评价“这次跳层组合到底好不好”
            advantages = (baseline_kl - kl_divs) # [B, N]
            
            # 长度软惩罚：鼓励尽量跳 6 层
            LENGTH_PENALTY_COEF = 0.1 
            length_penalty = LENGTH_PENALTY_COEF * ((num_skipped - K_TO_SKIP) ** 2) # [B, N]
            advantages = advantages - length_penalty

            # 计算 Policy Loss
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                # action_log_probs: [B, N, 40]
                action_log_probs = M_skip * torch.log(skip_probs_expanded + 1e-9) + (1 - M_skip) * torch.log(1 - skip_probs_expanded + 1e-9)
                sum_log_probs = action_log_probs.sum(dim=2) # [B, N]
                
                # 对 N 次采样求平均
                pg_loss = -(advantages * sum_log_probs).mean()
                
                entropy = -(skip_probs * torch.log(skip_probs + 1e-9) + (1 - skip_probs) * torch.log(1 - skip_probs + 1e-9))
                mean_entropy = entropy.sum(dim=1).mean() 
                
                ENTROPY_COEF = 0.05 
                
                rl_loss = pg_loss - ENTROPY_COEF * mean_entropy
                rl_loss = rl_loss / GRAD_ACCUMULATION_STEPS

            scaler.scale(rl_loss).backward()
            
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if rank == 0:
                pbar.set_postfix({
                    'AvgKL': f'{baseline_kl.mean().item():.4f}', 
                    'Skips': f'{num_skipped.float().mean().item():.1f}', 
                    'Ent': f'{mean_entropy.item():.2f}' 
                })
        
        if rank == 0:
            torch.save(router.module.state_dict(), f'router_rl_epoch_{epoch+1}.pth')
            print(f"Epoch {epoch+1} Saved. Avg KL: {baseline_kl.mean().item():.4f}")
    
    cleanup_ddp()

if __name__ == '__main__':
    main()