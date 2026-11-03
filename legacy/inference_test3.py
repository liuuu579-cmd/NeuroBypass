# inference_test.py (已更新以适配 MLP 模型)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# <<< 1. 修改导入: 导入 PredictorMLP >>>
from predictor_model2 import PredictorMLP 
from utils2 import AttentionStateCatcher 

# --- 配置 ---
OPT_MODEL_PATH = "/home/liushenyu/opt-13b/"
# <<< 2. 关键: 修改要加载的权重文件名，确保与 MLP 训练保存的文件名一致 >>>
PREDICTOR_MODEL_PATH = "predictor_mlp_epoch_10.pth" # 假设这是你训练好的MLP模型
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# OPT-13B Specifics
NUM_LAYERS = 40
D_MODEL = 5120
HEAD_DIM = 128
SEQ_LEN = 2048

# <<< 3. 删除/注释掉不再使用的 Transformer 超参数 >>>
# PREDICTOR_NHEAD = 16
# PREDICTOR_ENCODER_LAYERS = 8
# PREDICTOR_D_HID = 4096

# 预取策略配置
TOP_K_PERCENT = 0.3

def load_models():
    """加载OPT-13B和我们训练的MLP预测器模型"""
    print("Loading OPT-13B model...")
    opt_model = AutoModelForCausalLM.from_pretrained(
        OPT_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map='auto'
    )
    opt_model.eval()
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_PATH, use_fast=False)

    print("Loading MLP Predictor model...")
    # <<< 4. 关键修改: 初始化 PredictorMLP >>>
    predictor = PredictorMLP(
        input_dim=D_MODEL,
        num_opt_layers=NUM_LAYERS,
        # 确保这里的结构与训练时一致
        #hidden_dims=[2048, 1024],
        hidden_dims=[4096, 2048, 1024, 512], 
        dropout=0.2 # dropout在eval模式下会自动关闭，但保持一致性是个好习惯
    ).to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()
    
    return opt_model, tokenizer, predictor

# --- get_ground_truth 函数 (无变化) ---
def get_ground_truth(opt_model, input_ids):
    """
    获取OPT模型在完整KV缓存下的内部状态和注意力分布。
    """
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

# --- visualize_attention_comparison 函数 (无变化) ---
def visualize_attention_comparison(true_attn, predicted_attn, top_k_indices, layer_to_show, current_seq_len):
    # (此函数代码保持不变，此处省略以保持简洁)
    true_attn_avg = true_attn.mean(dim=0).cpu().numpy()
    predicted_attn = predicted_attn.cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(f"Layer {layer_to_show}: Predicted Attention vs. Ground Truth (Top {TOP_K_PERCENT*100}%)", fontsize=16)
    x_axis = np.arange(current_seq_len)
    sns.barplot(x=x_axis, y=predicted_attn, ax=axes[0], color='skyblue')
    axes[0].set_title("Predicted Attention Distribution (used for selecting Top-K)")
    axes[0].set_ylabel("Attention Probability")
    top_k_set = set(top_k_indices.cpu().numpy())
    for i, patch in enumerate(axes[0].patches):
        if i in top_k_set:
            patch.set_facecolor('red')
    sns.barplot(x=x_axis, y=true_attn_avg, ax=axes[1], color='lightgreen')
    axes[1].set_title("Ground Truth Attention (Averaged over all heads)")
    axes[1].set_xlabel("Token Position in Sequence")
    axes[1].set_ylabel("Attention Probability")
    for i, patch in enumerate(axes[1].patches):
        if i in top_k_set:
            patch.set_facecolor('red')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"attention_comparison_layer_{layer_to_show}.png")
    print(f"Saved visualization to attention_comparison_layer_{layer_to_show}.png")
    plt.close()

def main():
    opt_model, tokenizer, predictor = load_models()

    # (输入文本部分无变化)
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

    # 1. 获取基准真相 (无变化)
    print("\nStep 1: Getting ground truth from full OPT-13B pass...")
    embeddings, ground_truth_attn_probs = get_ground_truth(opt_model, input_ids)

    # 2. 使用我们的预测器进行预测
    print("\nStep 2: Using predictor to get attention score prediction...")
    with torch.no_grad():
        # <<< 5. 关键修改: MLP模型调用不需要 padding_mask >>>
        predicted_scores = predictor(embeddings) 
    
    # 将分数转换为概率 (无变化)
    predicted_attn_probs = F.softmax(predicted_scores, dim=-1).squeeze(0)

    # 3. & 4. 逐层模拟预取、评估并可视化 (逻辑完全不变)
    print(f"\nStep 3 & 4: Simulating prefetching, Evaluating and Visualizing PER LAYER (Top {TOP_K_PERCENT*100}%)...")
    k = int(current_seq_len * TOP_K_PERCENT)
    if k == 0: k=1
    
    all_overlaps = []
    layers_to_visualize = [0, 10, 20, 39]

    # --- 主循环：遍历每一层 (无变化) ---
    for layer_idx in range(NUM_LAYERS):
        predicted_probs_for_layer = predicted_attn_probs[layer_idx]
        _, predicted_top_k_indices = torch.topk(predicted_probs_for_layer, k=k, dim=-1)
        
        true_avg_probs_for_layer = ground_truth_attn_probs[layer_idx].mean(dim=0)
        _, true_top_k_indices = torch.topk(true_avg_probs_for_layer, k=k, dim=-1)
        
        overlap = len(set(predicted_top_k_indices.cpu().numpy()) & set(true_top_k_indices.cpu().numpy()))
        overlap_ratio = overlap / k
        all_overlaps.append(overlap_ratio)
        
        if layer_idx in layers_to_visualize:
            print(f"\n--- Details for Layer {layer_idx} ---")
            
            pred_indices_sorted, _ = torch.sort(predicted_top_k_indices)
            print(f"Layer {layer_idx} - Predicted Top-{k} indices: {pred_indices_sorted.cpu().numpy()}")
            # 使用 convert_ids_to_tokens 可能会有兼容性问题，如果报错可以换成 decode
            # tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0][pred_indices_sorted]]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0][pred_indices_sorted])
            print(f"Layer {layer_idx} - Corresponding tokens: {tokens}")
            print(f"Layer {layer_idx} - Overlap Ratio with Ground Truth: {overlap_ratio:.2%}")

            visualize_attention_comparison(
                ground_truth_attn_probs[layer_idx],
                predicted_attn_probs[layer_idx],
                predicted_top_k_indices,
                layer_idx,
                current_seq_len
            )

    # --- 最终评估 (无变化) ---
    avg_overlap_ratio = np.mean(all_overlaps)
    print(f"\n\n--- Final Evaluation Metric ---")
    print(f"Average Per-Layer Top-{k} Overlap Ratio across all {NUM_LAYERS} layers: {avg_overlap_ratio:.2%}")

if __name__ == '__main__':
    main()