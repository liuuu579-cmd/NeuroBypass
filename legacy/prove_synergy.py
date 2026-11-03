import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_TEST_SAMPLES = 5  # 只测 5 条样本求平均，以节省时间
TEST_SEQ_LEN = 128
MAX_SKIP_LAYERS = 35  # 最多尝试同时跳过多少层 (OPT-13B共40层)

# =====================================================================
# >>> Post-Hook 实现 <<<
# =====================================================================
def get_skip_layer_hook():
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        else:
            return args[0]
    return hook

def main():
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers
    transformer_layers = model.model.decoder.layers

    # --- 准备少量测试数据 ---
    print("Preparing text samples...")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read(50000) # 随便读一点
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    
    test_inputs = []
    for i in range(NUM_TEST_SAMPLES):
        start = i * TEST_SEQ_LEN
        end = start + TEST_SEQ_LEN
        test_inputs.append(all_tokens[start:end].unsqueeze(0).to(DEVICE))

    # --- 用于记录结果的数组 ---
    all_theoretical_kl = np.zeros((NUM_TEST_SAMPLES, MAX_SKIP_LAYERS))
    all_actual_kl = np.zeros((NUM_TEST_SAMPLES, MAX_SKIP_LAYERS))

    for sample_idx, input_ids in enumerate(test_inputs):
        print(f"\n--- Processing Sample {sample_idx + 1}/{NUM_TEST_SAMPLES} ---")
        
        # 1. 计算 Baseline
        with torch.no_grad():
            baseline_outputs = model(input_ids)
            baseline_logits = baseline_outputs.logits[:, -1, :]
            baseline_probs = F.softmax(baseline_logits, dim=-1)
        
        # 2. 计算每层单独跳过的 KL 散度
        single_layer_kl = []
        print("  Calculating single-layer KL (Leave-One-Out)...")
        for layer_idx in tqdm(range(NUM_LAYERS), leave=False):
            handle = transformer_layers[layer_idx].register_forward_hook(get_skip_layer_hook())
            with torch.no_grad():
                skip_outputs = model(input_ids)
                skip_logits = skip_outputs.logits[:, -1, :]
                skip_log_probs = F.log_softmax(skip_logits, dim=-1)
            handle.remove()
            
            kl = F.kl_div(skip_log_probs, baseline_probs, reduction='sum').item()
            single_layer_kl.append((layer_idx, max(0, kl))) # 防浮点数负值
            
            del skip_outputs, skip_logits, skip_log_probs
            
        # 3. 按 KL 从小到大排序，找出“最没用”的层
        single_layer_kl.sort(key=lambda x: x[1])
        least_important_layers = [x[0] for x in single_layer_kl]
        their_kl_scores = [x[1] for x in single_layer_kl]
        
        # 4. 累加跳层测试 (Cumulative Skip)
        print("  Testing synergistic accumulation...")
        cumulative_theoretical_kl = 0.0
        
        for k in tqdm(range(1, MAX_SKIP_LAYERS + 1), leave=False):
            # 这一轮我们要跳过的层集合
            current_skip_layers = least_important_layers[:k]
            
            # a. 计算理论上的线性累加 KL
            cumulative_theoretical_kl += their_kl_scores[k-1]
            all_theoretical_kl[sample_idx, k-1] = cumulative_theoretical_kl
            
            # b. 真实测试：同时给这 K 层挂上 Hook
            handles = []
            for layer_idx in current_skip_layers:
                h = transformer_layers[layer_idx].register_forward_hook(get_skip_layer_hook())
                handles.append(h)
                
            with torch.no_grad():
                multi_skip_outputs = model(input_ids)
                multi_skip_logits = multi_skip_outputs.logits[:, -1, :]
                multi_skip_log_probs = F.log_softmax(multi_skip_logits, dim=-1)
            
            # 必须移除所有 Hook！
            for h in handles:
                h.remove()
                
            actual_kl = F.kl_div(multi_skip_log_probs, baseline_probs, reduction='sum').item()
            all_actual_kl[sample_idx, k-1] = max(0, actual_kl)
            
            del multi_skip_outputs, multi_skip_logits, multi_skip_log_probs
        
        torch.cuda.empty_cache()

    # --- 5. 数据求平均并画图 ---
    print("\nPlotting results...")
    mean_theoretical = np.mean(all_theoretical_kl, axis=0)
    mean_actual = np.mean(all_actual_kl, axis=0)
    std_actual = np.std(all_actual_kl, axis=0) # 阴影方差
    
    x_axis = np.arange(1, MAX_SKIP_LAYERS + 1)
    
    plt.figure(figsize=(10, 6))
    
    # 理论线 (线性累加)
    plt.plot(x_axis, mean_theoretical, label='Theoretical Sum (Linear Independence)', 
             color='blue', linestyle='--', linewidth=2)
    
    # 真实线 (同时跳过)
    plt.plot(x_axis, mean_actual, label='Actual Error (Synergistic Effect)', 
             color='red', linewidth=2, marker='o', markersize=4)
    plt.fill_between(x_axis, mean_actual - std_actual, mean_actual + std_actual, color='red', alpha=0.2)
    
    plt.title('Proof of Synergistic Effect in Layer Skipping (OPT-13B)', fontsize=14)
    plt.xlabel('Number of Least Important Layers Skipped Simultaneously', fontsize=12)
    plt.ylabel('KL Divergence (Log Scale)', fontsize=12)
    
    # 因为指数爆炸太夸张，Y轴用对数刻度更好展示
    plt.yscale('log') 
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig('synergistic_effect_proof.png', dpi=300)
    print("Done! Saved as 'synergistic_effect_proof.png'.")

if __name__ == '__main__':
    main()