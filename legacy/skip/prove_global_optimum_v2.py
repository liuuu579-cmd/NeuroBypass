import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from itertools import combinations
import random
import gc

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = '/home/liushenyu/test/c4_subset_1024mb.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 核心实验参数
K_TO_SKIP = 6       
ENUM_POOL_SIZE = 12     

def get_skip_layer_hook():
    """用于跳过某层的 Hook"""
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        else:
            return args[0]
    return hook

def eval_combination_kl(model, transformer_layers, input_ids, baseline_probs, layers_to_skip):
    """【最终评测用】只走前向传播，计算 KL 散度"""
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
    
def compute_cosine_importance(model, transformer_layers, input_ids, active_layers, dropped_layers=None):
    """【流派三：静态特征几何学】计算层输入与输出的余弦相似度距离"""
    if dropped_layers is None:
        dropped_layers = []
        
    forward_handles = []
    for layer_idx in dropped_layers:
        h = transformer_layers[layer_idx].register_forward_hook(get_skip_layer_hook())
        forward_handles.append(h)
        
    importances = {i: 0.0 for i in active_layers}
    sim_hook_handles = []

    def get_cosine_hook(layer_idx):
        def forward_hook(module, inputs, outputs):
            in_tensor = inputs[0].detach()
            out_tensor = outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()
            cos_sim = F.cosine_similarity(in_tensor, out_tensor, dim=-1)
            mean_sim = cos_sim.mean().item()
            importances[layer_idx] = 1.0 - mean_sim
        return forward_hook

    for i in active_layers:
        h = transformer_layers[i].register_forward_hook(get_cosine_hook(i))
        sim_hook_handles.append(h)

    with torch.no_grad():
        model(input_ids)
    
    for h in forward_handles + sim_hook_handles:
        h.remove()
        
    return importances

def iterative_feature_greedy(model, transformer_layers, input_ids, search_space, k_to_skip):
    """
    【极速动态贪心】迭代特征匹配 (Iterative Feature MSE Greedy)
    每剪掉一层，利用 1 次前向传播动态更新剩余层的 H_in 和 H_out 的 MSE 差异。
    避免连续剪枝造成的坍塌，时间复杂度仅为 O(K)。
    """
    dropped_layers = []
    available_layers = list(search_space)
    
    for step in tqdm(range(k_to_skip), desc="Iterative Feature Greedy"):
        forward_handles = []
        measure_handles = []
        layer_scores = {}
        
        # 1. 屏蔽掉已经决定剪掉的层
        for layer_idx in dropped_layers:
            h = transformer_layers[layer_idx].register_forward_hook(get_skip_layer_hook())
            forward_handles.append(h)
            
        # 2. 给存活的层挂载监控 Hook
        def get_measure_hook(layer_idx):
            def hook(module, inputs, outputs):
                in_tensor = inputs[0].detach()
                out_tensor = outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()
                # 计算特征之间的 MSE (均方误差)
                mse_loss = F.mse_loss(in_tensor, out_tensor).item()
                layer_scores[layer_idx] = mse_loss
            return hook

        for i in available_layers:
            h = transformer_layers[i].register_forward_hook(get_measure_hook(i))
            measure_handles.append(h)
            
        # 3. 只需要 1 次前向传播！
        with torch.no_grad():
            model(input_ids)
            
        # 4. 找出当前状态下，产生特征偏移最小（最摸鱼）的层
        best_layer = min(layer_scores, key=layer_scores.get)
        
        # 5. 加入剪枝列表
        dropped_layers.append(best_layer)
        available_layers.remove(best_layer)
        
        for h in forward_handles + measure_handles:
            h.remove()
            
    return dropped_layers

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    
    NUM_LAYERS = model.config.num_hidden_layers
    transformer_layers = model.model.decoder.layers

    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read(10000)
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    input_ids = all_tokens[:256].unsqueeze(0).to(DEVICE)

    print("Calculating Baseline Probs for final KL eval...")
    model.eval()
    with torch.no_grad():
        baseline_outputs = model(input_ids)
        baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)

    for param in model.parameters():
        param.requires_grad = False # 全部走前向，不需要梯度了

    PROTECT_START = 2
    PROTECT_END = 2
    search_space = list(range(PROTECT_START, NUM_LAYERS - PROTECT_END))

    # ================= 搜索阶段 =================
    
    print("\nStep 1: Calculating Initial Cosine Distance Importance (Forward Only)...")
    initial_cosine_scores = compute_cosine_importance(
        model, transformer_layers, input_ids, active_layers=search_space
    )
    
    sorted_layers_by_cos = sorted(initial_cosine_scores.items(), key=lambda x: x[1])
    naive_combination = [x[0] for x in sorted_layers_by_cos[:K_TO_SKIP]]
    candidate_pool = [x[0] for x in sorted_layers_by_cos[:ENUM_POOL_SIZE]]
    
    print("\nStep 2: Iterative Greedy Strategy using Cosine Distance...")
    greedy_combination = []
    available_layers = list(search_space) 
    for step in tqdm(range(K_TO_SKIP), desc="Iterative Cosine Pruning"):
        current_scores = compute_cosine_importance(
            model, transformer_layers, input_ids, available_layers, dropped_layers=greedy_combination
        )
        best_layer = min(current_scores, key=current_scores.get)
        greedy_combination.append(best_layer)
        available_layers.remove(best_layer)

    print("\nStep 3: Running NEW Iterative Feature MSE Greedy (O(K) Speed)...")
    # ������ 新大招：调用我们的极速动态特征贪心算法 ������
    feature_greedy_comb = iterative_feature_greedy(
        model, transformer_layers, input_ids, search_space, K_TO_SKIP
    )

    # ================= 评测阶段 =================
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    print("\nStarting Final KL Divergence Evaluation (Forward Only)...")
    
    print("Evaluating Random Strategy...")
    random_kls = []
    for _ in range(10):
        rand_comb = random.sample(range(NUM_LAYERS), K_TO_SKIP)
        random_kls.append(eval_combination_kl(model, transformer_layers, input_ids, baseline_probs, rand_comb))
    random_kl = np.mean(random_kls)
    
    print("Evaluating Cosine Naive Bottom-K Strategy...")
    naive_kl = eval_combination_kl(model, transformer_layers, input_ids, baseline_probs, naive_combination)
    
    print("Evaluating Cosine Iterative Greedy Strategy...")
    greedy_kl = eval_combination_kl(model, transformer_layers, input_ids, baseline_probs, greedy_combination)
    
    print("Evaluating Feature MSE Greedy Strategy...")
    feature_greedy_kl = eval_combination_kl(model, transformer_layers, input_ids, baseline_probs, feature_greedy_comb)
    
    total_combs = len(list(combinations(candidate_pool, K_TO_SKIP)))
    print(f"Evaluating Global Enumeration in Cosine Pool (C_{ENUM_POOL_SIZE}^{K_TO_SKIP} = {total_combs} combinations)...")
    best_global_kl = float('inf')
    best_global_comb = None
    
    for comb in tqdm(combinations(candidate_pool, K_TO_SKIP), total=total_combs):
        kl = eval_combination_kl(model, transformer_layers, input_ids, baseline_probs, comb)
        if kl < best_global_kl:
            best_global_kl = kl
            best_global_comb = comb

    # --- 打印结果 ---
    print("\n" + "="*70)
    print(f"RESULTS FOR SKIPPING {K_TO_SKIP} LAYERS:")
    print(f"Random (Avg)             : KL = {random_kl:.4f}")
    print(f"Cosine Naive Bottom-K    : KL = {naive_kl:.4f} (Layers: {naive_combination})")
    print(f"Cosine Iterative Greedy  : KL = {greedy_kl:.4f} (Layers: {greedy_combination})")
    print(f"Feature MSE Greedy (NEW) : KL = {feature_greedy_kl:.4f} (Layers: {feature_greedy_comb})")
    print(f"KL Global Enum (in Pool) : KL = {best_global_kl:.4f} (Layers: {list(best_global_comb)})")
    print("="*70)

    # --- 画图 ---
    strategies = [
        'Random\n(Baseline)', 
        'Cosine\nNaive Bottom-K', 
        'Cosine\nIterative Greedy', 
        'Feature MSE\nIterative Greedy', 
        'KL Global Enum\n(in Pool)'
    ]
    kl_values = [random_kl, naive_kl, greedy_kl, feature_greedy_kl, best_global_kl]
    colors = ['gray', 'red', 'orange', 'blue', 'green']
    
    plt.figure(figsize=(12, 6)) # 稍微加宽一点适应5个柱子
    bars = plt.bar(strategies, kl_values, color=colors)
    plt.yscale('log')
    
    plt.title(f'Comparison of Routing Strategies (Skipping {K_TO_SKIP} Layers)', fontsize=14)
    plt.ylabel('KL Divergence (Log Scale) - Lower is Better', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('routing_strategies_comparison_5bars.png', dpi=300)
    print("Saved bar chart to 'routing_strategies_comparison_5bars.png'")

if __name__ == '__main__':
    main()