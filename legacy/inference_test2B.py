# inference_multilabel.py (Final Corrected Visualization)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from predictor_model2A import PredictorTransformer
from utils2 import AttentionStateCatcher 

# --- 配置 ---
OPT_MODEL_PATH = "/home/liushenyu/opt-13b/"
PREDICTOR_MODEL_PATH = "predictor_multilabel_k256_epoch_20.pth" # 确保路径正确
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# OPT-13B Specifics
NUM_LAYERS_OPT = 40 
D_MODEL = 5120
HEAD_DIM = 128
SEQ_LEN = 2048

# 预测器模型的超参数
NUM_PREDICT_LAYERS = 5
PREDICTOR_NHEAD = 16
PREDICTOR_ENCODER_LAYERS = 8
PREDICTOR_D_HID = 4096

# 预取策略配置
# 您的console输出Top-12，所以这里我们设置一个能产生k=12的百分比
# 假设序列长度为108，则 12/108 ≈ 0.111
TOP_K_PERCENT = 0.12 # 您可以根据实际情况调整

# --- load_models 和 get_ground_truth 函数保持不变 ---
def load_models():
    # ... (代码同前，无需修改) ...
    print("Loading OPT-13B model...")
    opt_model = AutoModelForCausalLM.from_pretrained(
        OPT_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map='auto'
    )
    opt_model.eval()
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_PATH, use_fast=False)

    print(f"Loading Predictor model (trained for {NUM_PREDICT_LAYERS} layers)...")
    predictor = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_PREDICT_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()
    
    return opt_model, tokenizer, predictor


def get_ground_truth(opt_model, input_ids):
    # ... (代码同前，无需修改) ...
    catcher = AttentionStateCatcher(opt_model)
    catcher.register_hooks()
    with torch.no_grad():
        opt_model(input_ids)
    
    captured_data = catcher.get_captured_data()
    catcher.remove_hooks()
    
    embeddings = captured_data['embeddings'].unsqueeze(0).to(torch.float32).to(DEVICE)
    keys = captured_data['keys'].unsqueeze(0).to(torch.float32).to(DEVICE)
    queries = captured_data['queries'].unsqueeze(0).to(torch.float32).to(DEVICE)
    
    q_last = queries[0, :, -1, :, :] 
    k_all = keys[0, :, :, :, :]      
    
    attn_scores = torch.einsum('lhd,lshd->lhs', q_last, k_all) / np.sqrt(HEAD_DIM)
    ground_truth_attn_probs = F.softmax(attn_scores, dim=-1)
    
    return embeddings, ground_truth_attn_probs


# <<< CRITICAL AND FINAL MODIFICATION HERE >>>
def visualize_attention_comparison(true_attn, predicted_probs, predicted_top_k_indices, layer_to_show, current_seq_len):
    """
    修正后的可视化函数：
    使用 palette 参数来确保颜色与索引的精确对应。
    """
    true_attn_avg = true_attn.mean(dim=0).cpu().numpy()
    predicted_probs_np = predicted_probs.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(f"Layer {layer_to_show}: Predicted Prefetch vs. Ground Truth (Top {TOP_K_PERCENT*100:.1f}%)", fontsize=16)
    
    x_axis = np.arange(current_seq_len)
    
    # --- 1. 创建颜色映射 ---
    predicted_indices_set = set(predicted_top_k_indices.cpu().numpy())
    
    # 为上图创建颜色列表
    top_colors = ['red' if i in predicted_indices_set else 'skyblue' for i in range(current_seq_len)]
    
    # 为下图创建颜色列表
    bottom_colors = ['red' if i in predicted_indices_set else 'lightgreen' for i in range(current_seq_len)]

    # --- 2. 使用 palette 参数绘图 ---

    # 上图：显示我们的预测
    sns.barplot(x=x_axis, y=predicted_probs_np, ax=axes[0], palette=top_colors)
    axes[0].set_title("Predicted Prefetch Probability")
    axes[0].set_ylabel("Prefetch Probability")

    # 下图：显示基准真相
    sns.barplot(x=x_axis, y=true_attn_avg, ax=axes[1], palette=bottom_colors)
    axes[1].set_title("Ground Truth Attention (Averaged over all heads)")
    axes[1].set_xlabel("Token Position in Sequence")
    axes[1].set_ylabel("Attention Probability")

    # 移除旧的、不可靠的上色循环
    # for i, patch in enumerate... (REMOVED)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"prefetch_comparison_layer_{layer_to_show}.png")
    print(f"Saved visualization to prefetch_comparison_layer_{layer_to_show}.png")
    plt.close()

def main():
    # ... (main函数的前半部分无需修改) ...
    opt_model, tokenizer, predictor = load_models()

    text = (
        "The rich get richer and the poor get poorer eh? Or is it the rich think different and play by a different set of rules? Do the rich take responsibility and action? Poor people believe 'Life happens to me.' Rich people are committed to be rich. Poor people WANT to be rich. Rich people think big. Poor people think small. Rich people focus on opportunities. Poor people focus on obstacles. Rich people are willing to promote themselves and their value. Poor people think negatively about selling and promotion. Poor people are closed to new"
    )
    
    print("\n--- Input Text ---")
    print(text)
    
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(DEVICE)
    if input_ids.shape[1] > SEQ_LEN:
        input_ids = input_ids[:, :SEQ_LEN]
    
    current_seq_len = input_ids.shape[1]
    print(f"\nTokenized sequence length: {current_seq_len}")

    print("\nStep 1: Getting ground truth from full OPT-13B pass...")
    embeddings, ground_truth_attn_probs = get_ground_truth(opt_model, input_ids)
    
    print("\nStep 2: Using predictor to get prefetch probability prediction...")
    with torch.no_grad():
        predicted_logits = predictor(embeddings)
    predicted_prefetch_probs = torch.sigmoid(predicted_logits).squeeze(0)

    print(f"\nStep 3 & 4: Simulating prefetching, Evaluating and Visualizing PER LAYER (Top {TOP_K_PERCENT*100:.1f}%)...")
    k = int(current_seq_len * TOP_K_PERCENT)
    if k == 0: k=1
    
    all_overlaps = []
    layers_to_visualize = [0, 1, 2, 3, 4]

    for layer_idx in range(NUM_LAYERS_OPT):
        if layer_idx < NUM_PREDICT_LAYERS:
            prediction_source_idx = layer_idx
        else:
            prediction_source_idx = NUM_PREDICT_LAYERS - 1

        predicted_probs_for_layer = predicted_prefetch_probs[prediction_source_idx]
        _, predicted_top_k_indices = torch.topk(predicted_probs_for_layer, k=k, dim=-1)
        
        true_avg_probs_for_layer = ground_truth_attn_probs[layer_idx].mean(dim=0)
        _, true_top_k_indices = torch.topk(true_avg_probs_for_layer, k=k, dim=-1)
        
        overlap = len(set(predicted_top_k_indices.cpu().numpy()) & set(true_top_k_indices.cpu().numpy()))
        overlap_ratio = overlap / k
        all_overlaps.append(overlap_ratio)
        
        if layer_idx in layers_to_visualize:
            print(f"\n--- Details for Layer {layer_idx} ---")
            print(f"(Using prediction from predictor's layer {prediction_source_idx})")
            
            pred_indices_sorted, _ = torch.sort(predicted_top_k_indices)
            print(f"Predicted Top-{k} indices: {pred_indices_sorted.cpu().numpy()}")
            print(f"Overlap Ratio with Ground Truth: {overlap_ratio:.2%}")

            # 调用最终修正后的可视化函数
            visualize_attention_comparison(
                ground_truth_attn_probs[layer_idx], 
                predicted_probs_for_layer,
                predicted_top_k_indices,
                layer_idx,                          
                current_seq_len
            )

    avg_overlap_ratio = np.mean(all_overlaps)
    print(f"\n\n--- Final Evaluation Metric ---")
    print(f"Average Per-Layer Top-{k} Overlap Ratio across all {NUM_LAYERS_OPT} layers: {avg_overlap_ratio:.2%}")

if __name__ == '__main__':
    main()