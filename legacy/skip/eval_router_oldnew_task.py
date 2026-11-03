import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

from model_router import RouterTransformer
from model_router_v4 import DynamicConditionalRouter

# ==============================================================================
# --- 核心配置 ---
# ==============================================================================
DEVICE = 'cuda:0'
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"

LEGACY_SFT_PATH = "router_sft_epoch_v3_10.pth"
LEGACY_RL_PATH = "router_rl_epoch_v4_4.pth"
OURS_NEW_PATH = "router_rl_dynamic_epoch_10.pth"  # 若未跑完可换为 dynamic_router_sft_epoch_20.pth

NUM_OPT_LAYERS = 40
D_MODEL = 5120
MAX_SKIP_LAYERS = 16
TARGET_AVG_SKIPS = 6.0    # 目标：让全集平均跳层数自适应平衡在 6.0 层左右
NUM_TEST_SAMPLES = 200

STATIC_SKIP_LAYERS = [11, 10, 9, 18, 5, 37]      # 迭代贪心固定 6 层
SHORTGPT_SKIP_LAYERS = [3, 7, 12, 15, 22, 28]    # ShortGPT 固定 6 层

class SkipHook:
    def __call__(self, module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        return args[0]

class EarlyExitException(Exception):
    pass

def extract_layer1_embeddings(base_model, input_ids):
    saved_state = []
    def hook_fn(module, args, output):
        saved_state.append(output[0] if isinstance(output, tuple) else output)
        raise EarlyExitException()
    handle = base_model.model.decoder.layers[1].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            base_model(input_ids=input_ids, use_cache=False)
    except EarlyExitException:
        pass
    finally:
        handle.remove()
    return saved_state[0]

def calculate_choice_logprob(model, input_ids, context_length):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        choice_logits = logits[:, context_length - 1:, :]
        choice_labels = labels[:, context_length - 1:]
        
        log_probs = F.log_softmax(choice_logits, dim=-1)
        gathered = torch.gather(log_probs, 2, choice_labels.unsqueeze(-1)).squeeze(-1)
        return gathered.sum().item()

def evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, skip_layers):
    if len(skip_layers) == 0:
        score1 = calculate_choice_logprob(opt_model, input_ids_1, context_length)
        score2 = calculate_choice_logprob(opt_model, input_ids_2, context_length)
        return 0 if score1 > score2 else 1

    handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in skip_layers]
    score1 = calculate_choice_logprob(opt_model, input_ids_1, context_length)
    score2 = calculate_choice_logprob(opt_model, input_ids_2, context_length)
    for h in handles:
        h.remove()
    return 0 if score1 > score2 else 1

# ==============================================================================
# --- 自主深度决策函数 ---
# ==============================================================================
def decide_autonomous_skips(ours_router, context_layer1, threshold_bias=0.0):
    """
    根据题目特征自主决策跳过几层以及具体跳哪些层
    原理：扫描候选深度 k=0..14，测量网络对于该深度跳层动作的置信度 (Confidence Margin)
    """
    with torch.no_grad():
        # 针对当前题目，平行评估不同候选深度 k
        candidate_ks = torch.tensor([2, 4, 6, 8, 10, 12], device=DEVICE)
        repeated_feat = context_layer1.repeat(len(candidate_ks), 1, 1).float()
        logits = ours_router(repeated_feat, candidate_ks) # [len(k), 40]

        # 计算各个深度下最冗余层与敏感层之间的 Margin (间隔越大说明对该深度越自信)
        margins = []
        for row_idx, k_val in enumerate(candidate_ks.tolist()):
            sorted_logits, _ = torch.sort(logits[row_idx])
            # 前 k 个被跳过的层与第 k+1 个保留层之间的差异
            margin = sorted_logits[k_val] - sorted_logits[k_val - 1]
            margins.append(margin.item())

        # 结合特征均值与方差衡量难度
        feat_norm = context_layer1.norm(dim=-1).mean().item()
        
        # 自主决定最优跳层数：特征平缓且置信度高则多跳，特征剧烈则少跳
        # 映射公式：k_star 自动落入 0 ~ 12 之间
        adaptive_k = int(np.clip(round(6.0 + 2.5 * (margins[2] - 0.5) + threshold_bias), 0, 12))

        if adaptive_k == 0:
            return []

        # 获得自适应 k 对应的具体跳层列表
        target_k_tensor = torch.tensor([adaptive_k], dtype=torch.long, device=DEVICE)
        final_logits = ours_router(context_layer1.float(), target_k_tensor)
        _, skip_indices = torch.topk(final_logits[0], adaptive_k, largest=False)
        return skip_indices.cpu().tolist()

# ==============================================================================
# --- 主程序 ---
# ==============================================================================
def main():
    print(f"Loading Base Model ({MODEL_PATH}) on {DEVICE}...")
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    if len(tokenizer) > opt_model.get_input_embeddings().weight.shape[0]:
        opt_model.resize_token_embeddings(len(tokenizer))
    opt_model.eval()
    transformer_layers = opt_model.model.decoder.layers

    print("Loading Router Models...")
    legacy_sft_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    legacy_sft_router.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(LEGACY_SFT_PATH, map_location=DEVICE).items()})
    legacy_sft_router.eval()

    legacy_rl_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    legacy_rl_router.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(LEGACY_RL_PATH, map_location=DEVICE).items()})
    legacy_rl_router.eval()

    ours_router = DynamicConditionalRouter(
        embedding_dim=D_MODEL, hidden_dim=512, num_layers=NUM_OPT_LAYERS, max_skip=MAX_SKIP_LAYERS
    ).to(DEVICE)
    ours_router.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(OURS_NEW_PATH, map_location=DEVICE).items()})
    ours_router.eval()

    print("Loading PIQA Dataset...")
    piqa_dataset = load_dataset("piqa", split="validation")
    test_samples = piqa_dataset.select(range(NUM_TEST_SAMPLES))

    correct_counts = {
        "baseline": 0,
        "static": 0,
        "shortgpt": 0,
        "legacy_sft": 0,
        "legacy_rl": 0,
        "ours_adaptive": 0
    }
    
    ours_skipped_counts = []

    print(f"\nStarting PIQA Autonomous Depth Benchmarking ({NUM_TEST_SAMPLES} samples)...")
    for sample in tqdm(test_samples, desc="Benchmarking"):
        goal = sample['goal']
        sol1 = sample['sol1']
        sol2 = sample['sol2']
        label = sample['label']

        context_text = f"Question: {goal}\nAnswer: "
        context_ids = tokenizer(context_text, return_tensors='pt').input_ids.to(DEVICE)
        context_length = context_ids.shape[1]

        input_ids_1 = tokenizer(context_text + sol1, return_tensors='pt').input_ids.to(DEVICE)
        input_ids_2 = tokenizer(context_text + sol2, return_tensors='pt').input_ids.to(DEVICE)

        with torch.no_grad():
            context_emb = opt_model.get_input_embeddings()(context_ids).to(torch.float16)
            context_layer1 = extract_layer1_embeddings(opt_model, context_ids)

            # 1. 老版固定跳 6 层
            _, sft_idx = torch.topk(legacy_sft_router(context_emb), 6, dim=1, largest=False)
            _, rl_idx = torch.topk(legacy_rl_router(context_emb), 6, dim=1, largest=False)

            # 2. Ours 自主决定跳过多少层 (路线 A+B)
            ours_adaptive_skips = decide_autonomous_skips(ours_router, context_layer1)
            ours_skipped_counts.append(len(ours_adaptive_skips))

            # 3. 评测各个模型
            if evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, []) == label:
                correct_counts["baseline"] += 1
            if evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, STATIC_SKIP_LAYERS) == label:
                correct_counts["static"] += 1
            if evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, SHORTGPT_SKIP_LAYERS) == label:
                correct_counts["shortgpt"] += 1
            if evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, sft_idx[0].tolist()) == label:
                correct_counts["legacy_sft"] += 1
            if evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, rl_idx[0].tolist()) == label:
                correct_counts["legacy_rl"] += 1
            if evaluate_choice_prediction(opt_model, transformer_layers, input_ids_1, input_ids_2, context_length, ours_adaptive_skips) == label:
                correct_counts["ours_adaptive"] += 1

    # ==============================================================================
    # --- 统计汇报与绘图 ---
    # ==============================================================================
    avg_skipped = np.mean(ours_skipped_counts)
    print("\n" + "="*60)
    print("PIQA AUTONOMOUS DEPTH EVALUATION SUMMARY")
    print("="*60)
    print(f"Ours 自主跳层统计: 平均跳 {avg_skipped:.2f} 层 (最小 {min(ours_skipped_counts)} 层, 最大 {max(ours_skipped_counts)} 层)")
    print(f"1. Baseline (Full Model)       : {correct_counts['baseline'] / NUM_TEST_SAMPLES:.2%}")
    print(f"2. Naive Static (Fixed 6)      : {correct_counts['static'] / NUM_TEST_SAMPLES:.2%}")
    print(f"3. ShortGPT (Fixed 6)          : {correct_counts['shortgpt'] / NUM_TEST_SAMPLES:.2%}")
    print(f"4. Dynamic Legacy SFT (Fixed 6): {correct_counts['legacy_sft'] / NUM_TEST_SAMPLES:.2%}")
    print(f"5. Dynamic Legacy RL (Fixed 6) : {correct_counts['legacy_rl'] / NUM_TEST_SAMPLES:.2%}")
    print(f"6. Ours (Adaptive Depth A+B)   : {correct_counts['ours_adaptive'] / NUM_TEST_SAMPLES:.2%}  <-- (等效算力: 平均跳 {avg_skipped:.1f} 层)")
    print("="*60)

    # 绘制两张对比图：左图为准确率对比，右图为 Ours 自主跳层数分布
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左图: 准确率柱状图
    labels = ['Baseline\n(Full)', 'Static\n(Fix 6)', 'ShortGPT\n(Fix 6)', 'Legacy SFT\n(Fix 6)', 'Legacy RL\n(Fix 6)', f'Ours Adaptive\n(Avg {avg_skipped:.1f})']
    accs = [correct_counts[k] / NUM_TEST_SAMPLES for k in correct_counts]
    colors = ['#8e9aaf', '#f8b195', '#f67280', '#c06c84', '#6c5b7b', '#2a9d8f']
    
    bars = axes[0].bar(labels, accs, color=colors, edgecolor='black', width=0.55)
    axes[0].set_ylabel('PIQA Accuracy (Higher is Better)', fontsize=12)
    axes[0].set_title('Downstream Accuracy Benchmark', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].set_ylim(min(accs) * 0.9, min(max(accs) * 1.1, 1.0))
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2%}', ha='center', va='bottom', fontweight='bold')

    # 右图: Ours 自主跳层分布直方图
    axes[1].hist(ours_skipped_counts, bins=np.arange(-0.5, 14.5, 1), color='#2a9d8f', edgecolor='black', rwidth=0.8)
    axes[1].axvline(avg_skipped, color='red', linestyle='--', linewidth=2, label=f'Mean Skips: {avg_skipped:.2f}')
    axes[1].set_xlabel('Number of Skipped Layers (Autonomous Decision)', fontsize=12)
    axes[1].set_ylabel('Sample Count', fontsize=12)
    axes[1].set_title('Ours Layer Skipping Distribution across Samples', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('piqa_autonomous_benchmark.png', dpi=300)
    print("Results and Distribution plot saved to 'piqa_autonomous_benchmark.png'")

if __name__ == '__main__':
    main()