import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_router import RouterTransformer

# --- ShortGPT 的核心计算函数 ---
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

SFT_WEIGHT_PATH = "router_sft_epoch_v3_10.pth"
RL_WEIGHT_PATH = "router_rl_epoch_v4_4.pth"

DEVICE = 'cuda:0'
NUM_OPT_LAYERS = 40
D_MODEL = 5120
K_TO_SKIP = 6
NUM_TEST_SAMPLES = 200
NUM_CALIB_SAMPLES = 10  # ShortGPT 需要少量校准数据来计算层重要性

# 你的 Baseline 静态跳层组合
STATIC_SKIP_LAYERS = [11, 10, 9, 18, 5, 37] #11, 10, 9, 18, 5, 37  迭代贪心    #6, 2, 10, 11, 12, 13  候选池全局最优

class SkipHook:
    def __call__(self, module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        return args[0]

def main():
    print("Loading OPT-13B Model...")
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    if len(tokenizer) > opt_model.get_input_embeddings().weight.shape[0]:
        opt_model.resize_token_embeddings(len(tokenizer))
    opt_model.eval()
    transformer_layers = opt_model.model.decoder.layers

    print("Loading Routers...")
    sft_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    sft_state = {k.replace('module.', ''): v for k, v in torch.load(SFT_WEIGHT_PATH, map_location=DEVICE).items()}
    sft_router.load_state_dict(sft_state)
    sft_router.eval()

    rl_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    rl_state = {k.replace('module.', ''): v for k, v in torch.load(RL_WEIGHT_PATH, map_location=DEVICE).items()}
    rl_router.load_state_dict(rl_state)
    rl_router.eval()

    print("Preparing unseen data (Calibration for ShortGPT + Test)...")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(file_size - 5000000) 
        text = f.read()
    
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    
    # 划分校准集 (ShortGPT用) 和 测试集 (所有人用)
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
    # 1. 计算 ShortGPT 策略 (预先计算静态跳层)
    # ==========================================
    print("Computing ShortGPT Layer Importances (Block Influence)...")
    importances = [0.0] * NUM_OPT_LAYERS
    with torch.no_grad():
        for calib_input in tqdm(calib_inputs, desc="ShortGPT Calibration"):
            # 获取所有隐藏层状态
            outputs = opt_model(calib_input, output_hidden_states=True)
            hiddens = outputs.hidden_states
            
            # hiddens[0] 是 embedding, hiddens[1] 是第0层输出... hiddens[i+1] 是第i层输出
            for i in range(NUM_OPT_LAYERS):
                in_hidden = hiddens[i]
                out_hidden = hiddens[i+1]
                bi_score = block_influence(in_hidden, out_hidden).sum().cpu().item()
                importances[i] += bi_score
            del outputs

    # 获取重要性最低的 K_TO_SKIP 层
    shortgpt_skip_layers = np.argsort(importances)[:K_TO_SKIP].tolist()
    print(f"ShortGPT calculated importances: {importances}")
    print(f"ShortGPT picked layers to skip: {shortgpt_skip_layers}")
    # ==========================================

    results_static = []
    results_shortgpt = []
    results_sft = []
    results_rl = []

    print(f"\nEvaluating {NUM_TEST_SAMPLES} samples...")
    for i, input_ids in enumerate(tqdm(test_inputs)):
        with torch.no_grad():
            embeddings = opt_model.get_input_embeddings()(input_ids).to(torch.float16)
            
            # 1. 获取 Baseline (不跳层) 真实输出
            base_out = opt_model(input_ids)
            base_probs = F.softmax(base_out.logits[:, -1, :], dim=-1)
            del base_out

            # 2. 评估 Static Pruning (Naive Baseline)
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in STATIC_SKIP_LAYERS]
            static_out = opt_model(input_ids)
            static_kl = F.kl_div(F.log_softmax(static_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_static.append(static_kl)
            for h in handles: h.remove()
            del static_out

            # 3. 评估 ShortGPT Pruning (Static but calculated via BI)
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in shortgpt_skip_layers]
            sgpt_out = opt_model(input_ids)
            sgpt_kl = F.kl_div(F.log_softmax(sgpt_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_shortgpt.append(sgpt_kl)
            for h in handles: h.remove()
            del sgpt_out

            # 4. 评估 SFT Router (Dynamic)
            sft_logits = sft_router(embeddings)
            _, sft_skip_idx = torch.topk(sft_logits, K_TO_SKIP, dim=1, largest=False)
            sft_layers = sft_skip_idx[0].tolist()
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in sft_layers]
            sft_out = opt_model(input_ids)
            sft_kl = F.kl_div(F.log_softmax(sft_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_sft.append(sft_kl)
            for h in handles: h.remove()
            del sft_out

            # 5. 评估 RL Router (Dynamic)
            rl_logits = rl_router(embeddings)
            _, rl_skip_idx = torch.topk(rl_logits, K_TO_SKIP, dim=1, largest=False) 
            rl_layers = rl_skip_idx[0].tolist()
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in rl_layers]
            rl_out = opt_model(input_ids)
            rl_kl = F.kl_div(F.log_softmax(rl_out.logits[:, -1, :], dim=-1), base_probs, reduction='sum').item()
            results_rl.append(rl_kl)
            for h in handles: h.remove()
            del rl_out

    # --- 统计与画图 ---
    mean_static = np.mean(results_static)
    mean_shortgpt = np.mean(results_shortgpt)
    mean_sft = np.mean(results_sft)
    mean_rl = np.mean(results_rl)

    print("\n" + "="*50)
    print("FINAL EVALUATION ON UNSEEN DATA")
    print("="*50)
    print(f"Static Pruning (Naive) KL   : {mean_static:.4f}")
    print(f"ShortGPT (Static BI) KL     : {mean_shortgpt:.4f}")
    print(f"Dynamic SFT Router KL       : {mean_sft:.4f}")
    print(f"Dynamic RL Router KL        : {mean_rl:.4f}")
    print("="*50)

    # 画柱状图
    labels = ['Naive Static\n(Manual)', 'ShortGPT\n(Static BI)', 'Dynamic Router\n(SFT)', 'Dynamic Router\n(RL)']
    means = [mean_static, mean_shortgpt, mean_sft, mean_rl]
    colors = ['#ff9999', '#ffcc99', '#66b3ff', '#99ff99']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, color=colors, edgecolor='black')
    
    plt.ylabel('KL Divergence (Lower is Better)', fontsize=12)
    plt.title(f'Performance Comparison (Skipping {K_TO_SKIP} Layers on OPT-13B)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_evaluation_results.png', dpi=300)
    print("Plot saved to 'final_evaluation_results.png'")

if __name__ == "__main__":
    main()