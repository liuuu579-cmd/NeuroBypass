import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ==============================================================================
# --- 1. Router 结构定义 (与训练一致) ---
# ==============================================================================
class DynamicConditionalRouterTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 5120,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        num_opt_layers: int = 40,
        max_skip: int = 16,
        dropout: float = 0.0
    ): 
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_opt_layers = num_opt_layers
        self.max_skip = max_skip

        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.proj_ln = nn.LayerNorm(hidden_dim)
        self.k_embedding = nn.Embedding(max_skip + 1, hidden_dim)

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

        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_opt_layers)
        )

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
        return self.router_head(cond_rep)


# ==============================================================================
# --- 2. 剪层 Hook 与 Embedding 提取 ---
# ==============================================================================
def get_skip_layer_hook():
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        return args[0]
    return hook

def extract_raw_embeddings(base_model, input_ids, attention_mask=None):
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
# --- 3. PIQA 零样本困惑度打分评估 ---
# ==============================================================================
def evaluate_piqa_sample(model, tokenizer, prompt, sol1, sol2, skip_layers, device):
    """
    计算选项 1 和 2 的归一化条件对数似然，并提取 prompt 末位与未剪枝模型的 KL
    """
    transformer_layers = model.model.decoder.layers
    handles = [transformer_layers[l].register_forward_hook(get_skip_layer_hook()) for l in skip_layers]

    results = {}
    try:
        with torch.no_grad():
            scores = []
            for sol in [sol1, sol2]:
                full_text = prompt + " " + sol
                enc = tokenizer(full_text, return_tensors='pt').to(device)
                input_ids = enc.input_ids
                
                # 仅针对 solution 部分计算 Loss
                prompt_len = len(tokenizer(prompt, return_tensors='pt').input_ids[0])
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(input_ids, labels=labels)
                neg_log_likelihood = -outputs.loss.item()
                scores.append(neg_log_likelihood)

            pred_choice = int(np.argmax(scores))

            # 提取 Prompt 末位的 Logits 用于评估特征保真度
            prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            prompt_logits = model(prompt_ids).logits[:, -1, :]
    finally:
        for h in handles:
            h.remove()

    return pred_choice, prompt_logits


# ==============================================================================
# --- 4. 主评测执行流程 ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/liushenyu/opt-13b/")
    parser.add_argument("--router_ckpt", type=str, default="transformer_router_raw_embed_epoch_20.pth")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--target_k", type=int, default=5, help="Router 的条件跳层数")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading Base Model: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    base_model.eval()

    print(f"Loading Router Model: {args.router_ckpt}...")
    router = DynamicConditionalRouterTransformer(
        input_dim=5120,
        hidden_dim=1024,
        num_heads=4,
        num_layers=2,
        dim_feedforward=2048,
        num_opt_layers=40,
        max_skip=16
    ).to(device)
    
    if os.path.exists(args.router_ckpt):
        router.load_state_dict(torch.load(args.router_ckpt, map_location=device))
        print("Router weights successfully loaded.")
    else:
        print(f"Warning: Router checkpoint {args.router_ckpt} not found! Using random initialization.")
    router.eval()

    # 预设静态 Baseline 的固定剪枝层 (根据实验测定)
    STATIC_GREEDY_LAYERS = [11, 10, 9, 18, 5, 37]  # Naive Static K=6
    SHORTGPT_LAYERS = [24, 25, 26, 27, 28, 29]      # ShortGPT 块影响度典型剪枝区间

    # 加载 PIQA 验证集
    print(f"Loading PIQA validation set (evaluating first {args.num_samples} samples)...")
    dataset = load_dataset("piqa", split="validation")
    eval_subset = [dataset[i] for i in range(min(args.num_samples, len(dataset)))]

    methods = [
        "Baseline (Full)",
        "Naive Static (K=6)",
        "ShortGPT (K=6)",
        f"Ours (Router K={args.target_k})"
    ]

    metrics = {m: {"correct": 0, "total": 0, "kl_divs": [], "skip_counts": []} for m in methods}

    for item in tqdm(eval_subset, desc="Evaluating Benchmarks"):
        prompt = item['goal']
        sol1 = item['sol1']
        sol2 = item['sol2']
        gt_label = item['label']

        # 1. 运行 Full Model 作为真实参考
        pred_full, logits_full = evaluate_piqa_sample(base_model, tokenizer, prompt, sol1, sol2, [], device)
        probs_full = F.softmax(logits_full.float(), dim=-1)
        
        metrics["Baseline (Full)"]["correct"] += int(pred_full == gt_label)
        metrics["Baseline (Full)"]["total"] += 1
        metrics["Baseline (Full)"]["skip_counts"].append(0)

        # 2. 运行各剪枝策略
        # 2.1 动态 Router 决策
        prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            raw_emb = extract_raw_embeddings(base_model, prompt_ids)
            k_tensor = torch.tensor([args.target_k], device=device)
            router_logits = router(raw_emb, k_tensor)  # [1, 40]
            
            # 保护 Layer 0 与 1 不被裁剪 (置为 -inf)
            router_logits[0, :2] = -float('inf')
            _, skip_indices = torch.topk(router_logits[0], k=args.target_k, largest=True)
            ours_skip_layers = skip_indices.tolist()

        eval_configs = [
            ("Naive Static (K=6)", STATIC_GREEDY_LAYERS),
            ("ShortGPT (K=6)", SHORTGPT_LAYERS),
            (f"Ours (Router K={args.target_k})", ours_skip_layers)
        ]

        for method_name, layers_to_skip in eval_configs:
            pred, logits_pruned = evaluate_piqa_sample(
                base_model, tokenizer, prompt, sol1, sol2, layers_to_skip, device
            )
            # 计算 KL 散度
            log_probs_pruned = F.log_softmax(logits_pruned.float(), dim=-1)
            kl = F.kl_div(log_probs_pruned, probs_full, reduction='sum').item()

            metrics[method_name]["correct"] += int(pred == gt_label)
            metrics[method_name]["total"] += 1
            metrics[method_name]["kl_divs"].append(kl)
            metrics[method_name]["skip_counts"].append(len(layers_to_skip))

    # ==============================================================================
    # --- 5. 打印对比实验表格 ---
    # ==============================================================================
    print("\n" + "="*80)
    print(f"{'Method':<28} | {'Avg Skips':<10} | {'PIQA Acc (%)':<14} | {'Mean KL Divergence':<18}")
    print("-" * 80)

    for m in methods:
        acc = (metrics[m]["correct"] / metrics[m]["total"]) * 100.0
        avg_skip = np.mean(metrics[m]["skip_counts"])
        kl_val = np.mean(metrics[m]["kl_divs"]) if len(metrics[m]["kl_divs"]) > 0 else 0.0
        print(f"{m:<28} | {avg_skip:<10.1f} | {acc:<14.2f} | {kl_val:<18.4f}")

    print("="*80 + "\n")

if __name__ == '__main__':
    main()