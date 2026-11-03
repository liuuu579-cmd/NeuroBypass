import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入老版与新版 Router
from model_router import RouterTransformer
from model_router_v4 import DynamicConditionalRouter

# --- ShortGPT 核心计算函数 ---
def block_influence(input_hidden_state: torch.Tensor, output_hidden_state: torch.Tensor, angular=False):
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = '/home/liushenyu/test/c4_subset_1024mb.txt'

# 模型权重路径
LEGACY_SFT_PATH = "router_sft_epoch_v3_10.pth"
LEGACY_RL_PATH = "router_rl_epoch_v4_4.pth"
# 新版（路线 A+B）模型权重路径
OURS_NEW_PATH = "router_rl_dynamic_epoch_10.pth"  # 若尚未跑完 RL 10 epoch，可暂填 dynamic_router_sft_epoch_20.pth

DEVICE = 'cuda:0'
NUM_OPT_LAYERS = 40
D_MODEL = 5120
MAX_SKIP_LAYERS = 16
K_TO_SKIP = 6  # 统一在跳过 6 层基准下横向对比
NUM_TEST_SAMPLES = 200
NUM_CALIB_SAMPLES = 10  # ShortGPT 校准样本数

# 静态跳层组合基线 (迭代贪心)
STATIC_SKIP_LAYERS = [11, 10, 9, 18, 5, 37]

# 通用跳层 Hook
class SkipHook:
    def __call__(self, module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        return args[0]

# 用于高效截获 Layer 1 特征的早退 Hook
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

def main():
    print("Loading OPT-13B Model...")
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    if len(tokenizer) > opt_model.get_input_embeddings().weight.shape[0]:
        opt_model.resize_token_embeddings(len(tokenizer))
    opt_model.eval()
    transformer_layers = opt_model.model.decoder.layers

    print("Loading 3 Routers (Legacy SFT, Legacy RL, and Ours New Model)...")
    # 1. 老版 SFT Router
    legacy_sft_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    sft_state = {k.replace('module.', ''): v for k, v in torch.load(LEGACY_SFT_PATH, map_location=DEVICE).items()}
    legacy_sft_router.load_state_dict(sft_state)
    legacy_sft_router.eval()

    # 2. 老版 RL Router
    legacy_rl_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    rl_state = {k.replace('module.', ''): v for k, v in torch.load(LEGACY_RL_PATH, map_location=DEVICE).items()}
    legacy_rl_router.load_state_dict(rl_state)
    legacy_rl_router.eval()

    # 3. 新版动态自适应条件 Router (Ours: Route A+B)
    ours_router = DynamicConditionalRouter(
        embedding_dim=D_MODEL, hidden_dim=512, num_layers=NUM_OPT_LAYERS, max_skip=MAX_SKIP_LAYERS
    ).to(DEVICE)
    ours_state = {k.replace('module.', ''): v for k, v in torch.load(OURS_NEW_PATH, map_location=DEVICE).items()}
    ours_router.load_state_dict(ours_state)
    ours_router.eval()

    print("Preparing unseen data (Calibration for ShortGPT + Test)...")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(file_size - 5000000) 
        text = f.read()

    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    chunk_size = 256
    calib_inputs = []
    test_inputs = []

    for i in range(NUM_CALIB_SAMPLES):
        start = i * chunk_size
        calib_inputs.append(all_tokens[start:start+chunk_size].unsqueeze(0).to(DEVICE))

    for i in range(NUM_TEST_SAMPLES):
        start = (NUM_CALIB_SAMPLES + i) * chunk_size
        test_inputs.append(all_tokens[start:start+chunk_size].unsqueeze(0).to(DEVICE))

    # ==========================================
    # 1. 计算 ShortGPT 静态策略
    # ==========================================
    print("Computing ShortGPT Layer Importances (Block Influence)...")
    importances = [0.0] * NUM_OPT_LAYERS
    with torch.no_grad():
        for calib_input in tqdm(calib_inputs, desc="ShortGPT Calibration"):
            outputs = opt_model(calib_input, output_hidden_states=True)
            hiddens = outputs.hidden_states
            for i in range(NUM_OPT_LAYERS):
                bi_score = block_influence(hiddens[i], hiddens[i+1]).sum().cpu().item()
                importances[i] += bi_score
            del outputs

    shortgpt_skip_layers = np.argsort(importances)[:K_TO_SKIP].tolist()
    print(f"ShortGPT picked layers to skip: {shortgpt_skip_layers}")

    # ==========================================
    # 2. 5 种方法在独立测试集上循环评测
    # ==========================================
    results_static = []
    results_shortgpt = []
    results_legacy_sft = []
    results_legacy_rl = []
    results_ours = []

    target_k_tensor = torch.tensor([K_TO_SKIP], dtype=torch.long, device=DEVICE)

    print(f"\nEvaluating {NUM_TEST_SAMPLES} samples across 5 methods...")
    for i, input_ids in enumerate(tqdm(test_inputs, desc="Evaluating")):
        with torch.no_grad():
            # 基础表征 (供老版 Router)
            embeddings = opt_model.get_input_embeddings()(input_ids).to(torch.float16)

            # 1. 基准模型输出 (全层推理)
            base_out = opt_model(input_ids)
            base_probs = F.softmax(base_out.logits[:, -1, :], dim=-1)
            del base_out

            # 2. 策略 1: Naive Static Pruning
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in STATIC_SKIP_LAYERS]
            static_out = opt_model(input_ids)
            static_kl = F.kl_div(F.log_softmax(static_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_static.append(static_kl)
            for h in handles: h.remove()
            del static_out

            # 3. 策略 2: ShortGPT (Static BI)
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in shortgpt_skip_layers]
            sgpt_out = opt_model(input_ids)
            sgpt_kl = F.kl_div(F.log_softmax(sgpt_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_shortgpt.append(sgpt_kl)
            for h in handles: h.remove()
            del sgpt_out

            # 4. 策略 3: Legacy SFT Router
            legacy_sft_logits = legacy_sft_router(embeddings)
            _, legacy_sft_idx = torch.topk(legacy_sft_logits, K_TO_SKIP, dim=1, largest=False)
            legacy_sft_layers = legacy_sft_idx[0].tolist()
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in legacy_sft_layers]
            sft_out = opt_model(input_ids)
            sft_kl = F.kl_div(F.log_softmax(sft_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_legacy_sft.append(sft_kl)
            for h in handles: h.remove()
            del sft_out

            # 5. 策略 4: Legacy RL Router
            legacy_rl_logits = legacy_rl_router(embeddings)
            _, legacy_rl_idx = torch.topk(legacy_rl_logits, K_TO_SKIP, dim=1, largest=False)
            legacy_rl_layers = legacy_rl_idx[0].tolist()
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in legacy_rl_layers]
            rl_out = opt_model(input_ids)
            rl_kl = F.kl_div(F.log_softmax(rl_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_legacy_rl.append(rl_kl)
            for h in handles: h.remove()
            del rl_out

            # 6. 策略 5: Ours Dynamic Conditional Router (Route A+B)
            layer1_feat = extract_layer1_embeddings(opt_model, input_ids)
            ours_logits = ours_router(layer1_feat.float(), target_k_tensor)
            _, ours_idx = torch.topk(ours_logits, K_TO_SKIP, dim=1, largest=False)
            ours_layers = ours_idx[0].tolist()
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in ours_layers]
            ours_out = opt_model(input_ids)
            ours_kl = F.kl_div(F.log_softmax(ours_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_ours.append(ours_kl)
            for h in handles: h.remove()
            del ours_out

    # ==========================================
    # 3. 统计汇总与绘制 5 策略对比柱状图
    # ==========================================
    mean_static = np.mean(results_static)
    mean_shortgpt = np.mean(results_shortgpt)
    mean_legacy_sft = np.mean(results_legacy_sft)
    mean_legacy_rl = np.mean(results_legacy_rl)
    mean_ours = np.mean(results_ours)

    print("\n" + "="*60)
    print("FINAL 5-METHOD COMPARISON ON UNSEEN DATA")
    print("="*60)
    print(f"1. Naive Static (Manual Iterative) KL : {mean_static:.4f}")
    print(f"2. ShortGPT (Static BI, ACL 2025) KL  : {mean_shortgpt:.4f}")
    print(f"3. Dynamic Router (Legacy SFT) KL     : {mean_legacy_sft:.4f}")
    print(f"4. Dynamic Router (Legacy RL) KL      : {mean_legacy_rl:.4f}")
    print(f"5. Ours Dynamic Conditional Router KL : {mean_ours:.4f}")
    print("="*60)

    # 绘制高清柱状图
    labels = [
        'Naive Static\n(Manual)', 
        'ShortGPT\n(ACL 2025)', 
        'Dynamic Router\n(Legacy SFT)', 
        'Dynamic Router\n(Legacy RL)',
        'Ours\n(Adaptive A+B)'
    ]
    means = [mean_static, mean_shortgpt, mean_legacy_sft, mean_legacy_rl, mean_ours]
    colors = ['#f8b195', '#f67280', '#c06c84', '#6c5b7b', '#355c7d']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, means, color=colors, edgecolor='black', width=0.55)

    plt.ylabel('KL Divergence (Lower is Better)', fontsize=12)
    plt.title(f'Performance Comparison: 5 Routing & Pruning Strategies (Skipping {K_TO_SKIP} Layers on OPT-13B)', fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 0.02 * max(means), 
            f'{yval:.4f}', 
            ha='center', 
            va='bottom', 
            fontweight='bold',
            fontsize=10
        )

    plt.ylim(0, max(means) * 1.15)
    plt.tight_layout()
    plt.savefig('final_5_methods_comparison.png', dpi=300)
    print("Plot successfully saved to 'final_5_methods_comparison.png'")

if __name__ == "__main__":
    main()