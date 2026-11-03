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

# ==============================================================================
# --- 模型定义：DynamicConditionalRouterTransformer ---
# ==============================================================================
class DynamicConditionalRouterTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 5120,       # OPT-13B Embedding 维度
        hidden_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        num_opt_layers: int = 40,
        max_skip: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_opt_layers = num_opt_layers
        self.max_skip = max_skip

        # 1. 输入投影层：5120 -> hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.proj_ln = nn.LayerNorm(hidden_dim)

        # 2. 条件 Token 嵌入 (索引 1..max_skip)
        self.k_embedding = nn.Embedding(max_skip + 1, hidden_dim)

        # 3. 2 层 Transformer Encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 4. 路由决策头
        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_opt_layers)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.k_embedding.weight, std=0.02)
        for p in self.router_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeddings: torch.Tensor, target_k: torch.Tensor, src_padding_mask: torch.Tensor = None):
        embeddings = embeddings.float()
        B, S, _ = embeddings.shape

        h_seq = self.proj_ln(self.in_proj(embeddings))

        clamped_k = torch.clamp(target_k.long(), min=1, max=self.max_skip)
        k_tokens = self.k_embedding(clamped_k).unsqueeze(1)
        h = torch.cat([k_tokens, h_seq], dim=1)

        if src_padding_mask is not None:
            cond_mask = torch.zeros((B, 1), dtype=torch.bool, device=embeddings.device)
            full_key_padding_mask = torch.cat([cond_mask, src_padding_mask.bool()], dim=1)
        else:
            full_key_padding_mask = None

        h_out = self.encoder(h, src_key_padding_mask=full_key_padding_mask)
        h_out = self.final_norm(h_out)

        cond_rep = h_out[:, 0, :]
        logits = self.router_head(cond_rep)
        return logits


# ==============================================================================
# --- 实验配置与工具函数 ---
# ==============================================================================
ZARR_BASE_NAME = '/home/liushenyu/test/opt13b_greedy_routing_labels_v3_part_'
MODEL_PATH = "/home/liushenyu/opt-13b/"
NUM_PARTS = 4

BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
GRAD_ACCUMULATION_STEPS = 8

NUM_OPT_LAYERS = 40
D_MODEL = 5120
MAX_SKIP_LAYERS = 16

def setup_ddp(local_rank):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def extract_raw_embeddings(base_model, input_ids, attention_mask=None):
    """
    仅提取 Token Embedding + Position Embedding，不经过任何 Transformer 块
    """
    decoder = base_model.model.decoder
    token_embeds = decoder.embed_tokens(input_ids)
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pos_embeds = decoder.embed_positions(attention_mask, past_key_values_length=0)
    
    embeddings = token_embeds + pos_embeds
    if hasattr(decoder, "project_in") and decoder.project_in is not None:
        embeddings = decoder.project_in(embeddings)
    return embeddings


# ==============================================================================
# --- 主训练流程 ---
# ==============================================================================
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
        print(f"--- [Router SFT] Starting on {world_size} GPUs ---")
        print(f"Input Mode: Raw Embedding (No Transformer Layers) | Router: Locked Float32")

    # 1. 挂载主干模型 (仅使用其 Embedding 表，其余层全部冻结)
    if rank == 0:
        print("Loading base model embedding tables...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to(device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    # 2. 初始化 Router Transformer
    model = DynamicConditionalRouterTransformer(
        input_dim=D_MODEL,
        hidden_dim=1024,
        num_heads=4,
        num_layers=2,
        dim_feedforward=2048,
        num_opt_layers=NUM_OPT_LAYERS,
        max_skip=MAX_SKIP_LAYERS,
        dropout=0.1
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 3. 聚合 Zarr 数据集
    datasets = []
    for i in range(NUM_PARTS):
        path = f"{ZARR_BASE_NAME}{i}"
        if os.path.exists(path):
            datasets.append(DynamicRouterDataset(path, max_skip_steps=MAX_SKIP_LAYERS, num_layers=NUM_OPT_LAYERS))
        else:
            if rank == 0:
                print(f"Warning: Dataset shard {path} not found.")

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
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=max(1, len(train_loader) // GRAD_ACCUMULATION_STEPS),
        epochs=NUM_EPOCHS
    )
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
            target_masks = target_masks.to(device).float()
            attention_mask = (~padding_mask).long()

            # 纯词表查表提取 Embedding，无任何 Decoder 层前向
            with torch.no_grad():
                embeddings = extract_raw_embeddings(base_model, input_ids, attention_mask=attention_mask)

            # Router 前向与 Loss 计算：Float32
            embeddings_f32 = embeddings.float()
            predicted_logits = model(embeddings_f32, target_k, src_padding_mask=padding_mask)

            loss = loss_fn(predicted_logits, target_masks) / GRAD_ACCUMULATION_STEPS
            loss.backward()
            train_loss += loss.item() * GRAD_ACCUMULATION_STEPS

            # 统计命中率
            with torch.no_grad():
                batch_hits = 0
                total_in_batch = 0
                for b in range(target_masks.size(0)):
                    k_val = target_k[b].item()
                    _, pred_skip_idx = torch.topk(predicted_logits[b], k=k_val, largest=True)
                    _, gt_skip_idx = torch.topk(target_masks[b], k=k_val, largest=True)

                    pred_set = set(pred_skip_idx.tolist())
                    gt_set = set(gt_skip_idx.tolist())
                    batch_hits += len(pred_set.intersection(gt_set))
                    total_in_batch += k_val

                train_hits += batch_hits
                train_total_targets += total_in_batch

            # 参数更新
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if rank == 0:
                curr_hit = train_hits / max(train_total_targets, 1)
                pbar.set_postfix({
                    'loss': f'{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}',
                    'hit_rate': f'{curr_hit:.3f}'
                })

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
                target_masks = target_masks.to(device).float()
                attention_mask = (~padding_mask).long()

                embeddings = extract_raw_embeddings(base_model, input_ids, attention_mask=attention_mask)
                embeddings_f32 = embeddings.float()
                predicted_logits = model(embeddings_f32, target_k, src_padding_mask=padding_mask)
                loss = loss_fn(predicted_logits, target_masks)

                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size

                batch_hits = 0
                total_in_batch = 0
                for b in range(target_masks.size(0)):
                    k_val = target_k[b].item()
                    _, pred_skip_idx = torch.topk(predicted_logits[b], k=k_val, largest=True)
                    _, gt_skip_idx = torch.topk(target_masks[b], k=k_val, largest=True)

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
            torch.save(model.module.state_dict(), f'transformer_router_raw_embed_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()