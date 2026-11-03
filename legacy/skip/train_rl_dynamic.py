import os
import argparse
import datetime
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM

from dataset_router_dynamic import DynamicRouterDataset, pad_dynamic_collate_fn
from model_router_v4 import DynamicConditionalRouter

# ==============================================================================
# --- 超参数配置 ---
# ==============================================================================
ZARR_BASE_NAME = '/home/liushenyu/test/opt13b_greedy_routing_labels_v3_part_'
NUM_PARTS = 4
MODEL_PATH = "/home/liushenyu/opt-13b/"
SFT_WEIGHT_PATH = "dynamic_router_sft_epoch_20.pth"

BATCH_SIZE = 2             # 单卡 Batch Size
GRAD_ACCUMULATION_STEPS = 8 # 等效 Batch = 2 * 8 * 4卡 = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10            # 【最强大脑决策】：扩增至 10 Epoch，配合学习率退火让策略充分平稳探索
NUM_OPT_LAYERS = 40
D_MODEL = 5120
MAX_SKIP_LAYERS = 16
NUM_SAMPLES = 4            # 每条样本探索 4 组路径

# ==============================================================================
# --- 软跳层 Hook (支持 Batch 内独立掩码) ---
# ==============================================================================
class BatchSkipHook:
    def __init__(self, skip_mask):
        self.skip_mask = skip_mask.view(-1, 1, 1)

    def __call__(self, module, args, output):
        hidden_states_in = args[0]
        if isinstance(output, tuple):
            hidden_states_out = output[0]
            mixed_out = self.skip_mask * hidden_states_in + (1.0 - self.skip_mask) * hidden_states_out
            return (mixed_out,) + output[1:]
        else:
            mixed_out = self.skip_mask * hidden_states_in + (1.0 - self.skip_mask) * output
            return mixed_out

class EarlyExitException(Exception):
    pass

def extract_layer1_embeddings(base_model, input_ids, attention_mask=None):
    saved_state = []
    def hook_fn(module, args, output):
        saved_state.append(output[0] if isinstance(output, tuple) else output)
        raise EarlyExitException()
    handle = base_model.model.decoder.layers[1].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except EarlyExitException:
        pass
    finally:
        handle.remove()
    return saved_state[0]

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
        print(f"--- [RL Phase 2.0] Multi-Sample RL on {world_size} GPUs (10 Epochs + Float32 Router) ---")

    # 1. 主干模型使用 FP16 运行（节约显存）
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    opt_model.eval()
    for param in opt_model.parameters():
        param.requires_grad = False
    transformer_layers = opt_model.model.decoder.layers

    # 2. Router 必须保持 Float32，确保 RL 梯度的数值稳定性
    router = DynamicConditionalRouter(
        embedding_dim=D_MODEL, hidden_dim=512, num_layers=NUM_OPT_LAYERS, max_skip=MAX_SKIP_LAYERS
    ).to(device).float()

    if rank == 0:
        print(f"Loading SFT initial weights from {SFT_WEIGHT_PATH}...")
    state_dict = torch.load(SFT_WEIGHT_PATH, map_location=device)
    router.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    router = DDP(router, device_ids=[rank], find_unused_parameters=False)

    # 3. 聚合数据
    datasets = [DynamicRouterDataset(f"{ZARR_BASE_NAME}{i}", max_skip_steps=MAX_SKIP_LAYERS, num_layers=NUM_OPT_LAYERS) 
                for i in range(NUM_PARTS) if os.path.exists(f"{ZARR_BASE_NAME}{i}")]
    full_dataset = ConcatDataset(datasets)
    train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        pin_memory=True, collate_fn=pad_dynamic_collate_fn, drop_last=True
    )

    optimizer = optim.AdamW(router.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # 增加余弦退火调度器，保证探索到收敛的平稳过渡
    total_steps = (len(train_loader) // GRAD_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 用于监控权重是否真正发生物理更新
    prev_param_copy = copy.deepcopy(router.module.classifier[0].weight.data) if rank == 0 else None

    # 4. RL 训练循环
    for epoch in range(NUM_EPOCHS):
        router.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [RL]", disable=(rank != 0))

        for step_idx, (input_ids, padding_mask, target_k, _) in enumerate(pbar):
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            attention_mask = (~padding_mask).long()
            target_k = target_k.to(device)
            B = input_ids.size(0)

            # (1) 提取特征 (OPT 保持 FP16)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                embeddings = extract_layer1_embeddings(opt_model, input_ids, attention_mask=attention_mask)

            # (2) Router 前向传播强制使用 Float32，杜绝任何 FP16 下溢或 NaN
            logits = router(embeddings.float(), target_k, src_padding_mask=padding_mask).float()
            raw_skip_probs = torch.sigmoid(-logits) # [B, 40]
            
            protect_mask = torch.ones(NUM_OPT_LAYERS, device=device, dtype=torch.float32)
            protect_mask[:2] = 0.0
            skip_probs = raw_skip_probs * protect_mask # [B, 40]

            # (3) 动作采样
            skip_probs_expanded = skip_probs.unsqueeze(1).expand(B, NUM_SAMPLES, NUM_OPT_LAYERS) # [B, N, 40]
            M_skip = torch.bernoulli(skip_probs_expanded).detach()                                # [B, N, 40]
            num_skipped = M_skip.sum(dim=-1)                                                     # [B, N]

            # (4) OPT 计算真实 KL
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    base_out = opt_model(input_ids, attention_mask=attention_mask)
                    base_probs = F.softmax(base_out.logits[:, -1, :], dim=-1)

            kl_divs = torch.zeros(B, NUM_SAMPLES, device=device)
            for n in range(NUM_SAMPLES):
                handles = []
                for layer_idx in range(NUM_OPT_LAYERS):
                    layer_mask = M_skip[:, n, layer_idx]
                    h = transformer_layers[layer_idx].register_forward_hook(BatchSkipHook(layer_mask))
                    handles.append(h)

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        skip_out = opt_model(input_ids, attention_mask=attention_mask)
                        skip_log_probs = F.log_softmax(skip_out.logits[:, -1, :], dim=-1)

                for h in handles:
                    h.remove()

                kl_divs[:, n] = F.kl_div(skip_log_probs.float(), base_probs.float(), reduction='none').sum(dim=-1)

            # (5) 复合奖励与 Advantage 计算 (在 Float32 下安全计算)
            baseline_kl = kl_divs.mean(dim=1, keepdim=True)
            adv_perf = (torch.log1p(baseline_kl) - torch.log1p(kl_divs))

            target_k_expanded = target_k.unsqueeze(1).expand(B, NUM_SAMPLES)
            length_penalty = 0.05 * ((num_skipped - target_k_expanded.float()) ** 2)

            advantages = adv_perf - length_penalty
            # Advantage 标准化：防止极端梯度
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            # (6) 策略梯度损失（数值绝对安全设计）
            p_clamped = torch.clamp(skip_probs_expanded, min=1e-6, max=1.0 - 1e-6)
            action_log_probs = M_skip * torch.log(p_clamped) + (1.0 - M_skip) * torch.log(1.0 - p_clamped)
            sum_log_probs = action_log_probs.sum(dim=-1)

            pg_loss = -(advantages * sum_log_probs).mean()

            p_single_clamped = torch.clamp(skip_probs, min=1e-6, max=1.0 - 1e-6)
            entropy = -(p_single_clamped * torch.log(p_single_clamped) + (1.0 - p_single_clamped) * torch.log(1.0 - p_single_clamped))
            mean_entropy = entropy.sum(dim=-1).mean()

            batch_variance = skip_probs.var(dim=0).mean()

            # 总 Loss (直接反向传播，不需要 Scaler)
            total_loss = (pg_loss - 0.02 * mean_entropy - 0.2 * batch_variance) / GRAD_ACCUMULATION_STEPS
            total_loss.backward()

            # (7) 优化器单步更新
            if (step_idx + 1) % GRAD_ACCUMULATION_STEPS == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if rank == 0:
                pbar.set_postfix({
                    'KL': f'{baseline_kl.mean().item():.3f}',
                    'Skips': f'{num_skipped.float().mean().item():.1f}',
                    'Var': f'{batch_variance.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })

        # 每个 Epoch 结束时的健康检查
        if rank == 0:
            current_param = router.module.classifier[0].weight.data
            param_diff = (current_param - prev_param_copy).abs().sum().item()
            prev_param_copy = copy.deepcopy(current_param)
            
            torch.save(router.module.state_dict(), f'router_rl_dynamic_epoch_{epoch+1}.pth')
            print(f"\n Epoch {epoch+1} Completed | Avg KL: {baseline_kl.mean().item():.4f} | Batch Var: {batch_variance.item():.4f} | Param Shift: {param_diff:.6f}")
            if param_diff < 1e-6:
                print("⚠️ [警告]: 权重参数未发生位移，请检查梯度传递！")
            else:
                print("✅ [健康]: 权重参数正在稳步进化更新！")

    cleanup_ddp()

if __name__ == '__main__':
    main()