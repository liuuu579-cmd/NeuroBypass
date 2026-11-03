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
SFT_WEIGHT_PATH = "router_sft_epoch_10.pth" 

BATCH_SIZE = 1       
LEARNING_RATE = 1e-5 
NUM_EPOCHS = 5
GRAD_ACCUMULATION_STEPS = 16
USE_AMP = True

NUM_OPT_LAYERS = 40
D_MODEL = 5120
K_TO_SKIP = 6

# ==============================================================================
# >>> 核心黑科技：支持 Batch 差异化的跳层 Hook <<<
# ==============================================================================
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
        print(f"--- Starting Router RL Fine-tuning on {world_size} GPUs ---")

    if rank == 0: print("Loading OPT-13B as Environment...")
    from transformers import AutoModelForCausalLM
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    opt_model.eval()
    for param in opt_model.parameters():
        param.requires_grad = False
    transformer_layers = opt_model.model.decoder.layers

    if rank == 0: print("Loading Router and SFT Weights...")
    router = RouterTransformer(
        input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4,
        num_encoder_layers=2, d_hid=2048
    ).to(device)
    
    state_dict = torch.load(SFT_WEIGHT_PATH, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    router.load_state_dict(new_state_dict)
    
    router = DDP(router, device_ids=[rank], find_unused_parameters=False)
    
    datasets = []
    for i in range(NUM_PARTS):
        path = f"{ZARR_BASE_NAME}{i}"
        if os.path.exists(path): datasets.append(RouterDataset(path))
            
    full_dataset = ConcatDataset(datasets)
    train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    optimizer = optim.AdamW(router.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    running_baseline = 0.0

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
            
            # ==========================================================
            # 步骤 A：Agent 出牌 (基于伯努利分布的独立采样)
            # ==========================================================
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                logits = router(embeddings, src_padding_mask=src_padding_mask) 
                skip_probs = torch.sigmoid(-logits) 
            
            # 【修改点 1】：放弃 Top-K 硬约束，让每一层完全根据自己的概率决定是否跳过
            # 这样不仅保证了梯度的数学正确性，还给了模型极大的探索空间
            M_skip = torch.bernoulli(skip_probs).detach() # [B, 40] 离散采样，不需要梯度流过
            num_skipped = M_skip.sum(dim=1) # 计算当前跳过了多少层 [B]
            
            # ==========================================================
            # 步骤 B：环境反馈 (OPT-13B 给出 KL 散度)
            # ==========================================================
            with torch.no_grad():
                base_out = opt_model(inputs_embeds=embeddings, attention_mask=hf_attn_mask)
                base_logits = base_out.logits[torch.arange(B), seq_lengths, :]
                base_probs = F.softmax(base_logits, dim=-1)
                
                handles = []
                for layer_idx in range(NUM_OPT_LAYERS):
                    layer_mask = M_skip[:, layer_idx]
                    h = transformer_layers[layer_idx].register_forward_hook(BatchSkipHook(layer_mask))
                    handles.append(h)
                
                skip_out = opt_model(inputs_embeds=embeddings, attention_mask=hf_attn_mask)
                skip_logits = skip_out.logits[torch.arange(B), seq_lengths, :]
                skip_log_probs = F.log_softmax(skip_logits, dim=-1)
                
                for h in handles:
                    h.remove()
                    
                kl_divs = F.kl_div(skip_log_probs, base_probs, reduction='none').sum(dim=-1) # [B]
                
            # ==========================================================
            # 步骤 C：计算 Reward (对数奖励 + 长度惩罚)
            # ==========================================================
            # 【修改点 2】：引入长度软惩罚。因为我们移除了 Top-K，必须靠奖励告诉它“尽量跳过 6 层”
            LENGTH_PENALTY_COEF = 0.5 
            length_penalty = LENGTH_PENALTY_COEF * ((num_skipped - K_TO_SKIP) ** 2) # 偏离 6 层越远，惩罚越大
            
            # 【修改点 3】：使用对数级奖励，缩小极端灾难组合带来的方差
            # 比如 KL=0.005 -> Reward ≈ 5.29; KL=0.5 -> Reward ≈ 0.69
            # 减去 length_penalty 引导模型在保证低 KL 的同时，将跳层数收敛到 6
            rewards = -torch.log(kl_divs + 1e-5) - length_penalty
            
            batch_mean_reward = rewards.mean().item()
            if running_baseline == 0.0:
                running_baseline = batch_mean_reward
            else:
                running_baseline = 0.9 * running_baseline + 0.1 * batch_mean_reward
                
            advantages = (rewards - running_baseline).to(device) # [B]
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                # 计算策略的对数概率
                log_probs_action = M_skip * torch.log(skip_probs + 1e-9) + (1 - M_skip) * torch.log(1 - skip_probs + 1e-9)
                sum_log_probs = log_probs_action.sum(dim=1) 
                
                entropy = -(skip_probs * torch.log(skip_probs + 1e-9) + (1 - skip_probs) * torch.log(1 - skip_probs + 1e-9))
                mean_entropy = entropy.sum(dim=1).mean() 
                
                ENTROPY_COEF = 0.05 
                
                rl_loss = -(advantages * sum_log_probs).mean() - ENTROPY_COEF * mean_entropy
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
                    'KL': f'{kl_divs.mean().item():.4f}', 
                    'Adv': f'{advantages.mean().item():.2f}',
                    'Skips': f'{num_skipped.float().mean().item():.1f}', # 观察当前平均跳过了几层
                    'Ent': f'{mean_entropy.item():.4f}' 
                })
        
        if rank == 0:
            torch.save(router.module.state_dict(), f'router_rl_epoch_{epoch+1}.pth')
            print(f"Epoch {epoch+1} Saved. Avg KL: {kl_divs.mean().item():.4f}")
    
    cleanup_ddp()

if __name__ == '__main__':
    main()