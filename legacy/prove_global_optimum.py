import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from itertools import combinations
import random

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 核心实验参数
K_TO_SKIP = 6           # 我们要跳过多少层
ENUM_POOL_SIZE = 12     # 从最不重要的多少层中进行穷举 (C_12^4 = 495次)

def get_skip_layer_hook():
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        else:
            return args[0]
    return hook

def eval_combination(model, transformer_layers, input_ids, baseline_probs, layers_to_skip):
    """评估跳过特定组合层的 KL 散度"""
    handles = []
    for layer_idx in layers_to_skip:
        h = transformer_layers[layer_idx].register_forward_hook(get_skip_layer_hook())
        handles.append(h)
        
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
    for h in handles:
        h.remove()
        
    kl = F.kl_div(log_probs, baseline_probs, reduction='sum').item()
    return max(0, kl)

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers
    transformer_layers = model.model.decoder.layers

    # --- 准备 1 条测试数据 (稍微长一点，更有代表性) ---
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read(10000)
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    input_ids = all_tokens[:256].unsqueeze(0).to(DEVICE)

    # 0. Baseline
    print("Calculating Baseline...")
    with torch.no_grad():
        baseline_outputs = model(input_ids)
        baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)

    # 1. 测单层 KL，构建候选池
    print("Step 1: Calculating Single-Layer KL...")
    single_kls = []
    for i in tqdm(range(NUM_LAYERS), leave=False):
        kl = eval_combination(model, transformer_layers, input_ids, baseline_probs, [i])
        single_kls.append((i, kl))
    
    single_kls.sort(key=lambda x: x[1])
    # Naive Bottom-K 组合
    naive_combination = [x[0] for x in single_kls[:K_TO_SKIP]]
    # 构建穷举候选池
    candidate_pool = [x[0] for x in single_kls[:ENUM_POOL_SIZE]]
    
    # 2. 策略 A: Random Baseline (测 10 次取平均)
    print("\nStep 2: Random Strategy...")
    random_kls = []
    for _ in range(10):
        rand_comb = random.sample(range(NUM_LAYERS), K_TO_SKIP)
        random_kls.append(eval_combination(model, transformer_layers, input_ids, baseline_probs, rand_comb))
    random_kl = np.mean(random_kls)
    
    # 3. 策略 B: Naive Bottom-K (你上一张图的策略)
    print("Step 3: Naive Bottom-K Strategy...")
    naive_kl = eval_combination(model, transformer_layers, input_ids, baseline_probs, naive_combination)
    
    # 4. 策略 C: Iterative Greedy
    print("Step 4: Iterative Greedy Strategy...")
    greedy_combination = []
    available_layers = list(range(NUM_LAYERS))
    for step in range(K_TO_SKIP):
        best_kl = float('inf')
        best_layer = -1
        for l in tqdm(available_layers, leave=False, desc=f"Greedy Step {step+1}"):
            test_comb = greedy_combination + [l]
            kl = eval_combination(model, transformer_layers, input_ids, baseline_probs, test_comb)
            if kl < best_kl:
                best_kl = kl
                best_layer = l
        greedy_combination.append(best_layer)
        available_layers.remove(best_layer)
    greedy_kl = best_kl
    
    # 5. 策略 D: Global Enumeration (在 Candidate Pool 中穷举)
    #print(f"Step 5: Global Enumeration in Pool (C_{ENUM_POOL_SIZE}^{K_TO_SKIP} = {len(list(combinations(candidate_pool, K_TO_SKIP)))} combinations)...")
    best_global_kl = float('inf')
    best_global_comb = None
    
    #for comb in tqdm(combinations(candidate_pool, K_TO_SKIP), total=len(list(combinations(candidate_pool, K_TO_SKIP)))):
    #    kl = eval_combination(model, transformer_layers, input_ids, baseline_probs, comb)
    #    if kl < best_global_kl:
    #        best_global_kl = kl
    #        best_global_comb = comb

    # --- 打印结果 ---
    print("\n" + "="*50)
    print(f"RESULTS FOR SKIPPING {K_TO_SKIP} LAYERS:")
    print(f"Random (Avg)   : KL = {random_kl:.4f}")
    print(f"Naive Bottom-K : KL = {naive_kl:.4f} (Layers: {naive_combination})")
    print(f"Iterative Greedy: KL = {greedy_kl:.4f} (Layers: {greedy_combination})")
    #print(f"Global Optimum : KL = {best_global_kl:.4f} (Layers: {best_global_comb})")
    print("="*50)

    # --- 画图 ---
    strategies = ['Random\n(Baseline)', 'Naive Bottom-K\n(Current)', 'Iterative Greedy', 'Global Optimum\n(Target for RL)']
    kl_values = [random_kl, naive_kl, greedy_kl, best_global_kl]
    colors = ['gray', 'red', 'orange', 'green']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, kl_values, color=colors)
    plt.yscale('log') # 使用对数坐标
    
    plt.title(f'Comparison of Routing Strategies (Skipping {K_TO_SKIP} Layers)', fontsize=14)
    plt.ylabel('KL Divergence (Log Scale) - Lower is Better', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱子上标出数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('routing_strategies_comparison.png', dpi=300)
    print("Saved bar chart to 'routing_strategies_comparison.png'")

if __name__ == '__main__':
    main()