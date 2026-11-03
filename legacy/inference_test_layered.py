# inference_test_layered.py 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # 导入os

# <<< 1. 导入新的分层模型 >>>
from predictor_model_layered import SingleLayerPredictorMLP
from utils import AttentionStateCatcher 

# --- 配置 ---
OPT_MODEL_PATH = "/home/liushenyu/opt-13b/"
# <<< 2. 关键: 指向包含5个模型状态的权重文件 >>>
PREDICTOR_MODEL_PATH = "layered_predictors_epoch_20.pth" # 使用你训练好的最佳模型
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- OPT-13B 和 预测器超参数 (必须与训练时保持一致) ---
NUM_LAYERS = 40
D_MODEL = 5120
HEAD_DIM = 128
SEQ_LEN = 2048
NUM_PREDICTOR_LAYERS = 5 # 我们训练了5个模型
PREDICTOR_HIDDEN_DIMS = [4096, 2048]
PREDICTOR_DROPOUT = 0.2

# --- 预取策略配置 ---
# 注意: 推理时的Top K%可以和训练时的k值不同。这里我们用30%做评估
TOP_K_PERCENT = 0.3 

def load_models():
    """加载OPT-13B和我们训练的5个分层MLP预测器模型"""
    print("Loading OPT-13B model...")
    opt_model = AutoModelForCausalLM.from_pretrained(
        OPT_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map='auto'
    )
    opt_model.eval()
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_PATH, use_fast=False)

    print(f"Loading {NUM_PREDICTOR_LAYERS} Layered MLP Predictor models...")
    # <<< 3. 关键修改: 初始化5个模型并加载各自的权重 >>>
    
    # 创建一个模型列表
    predictors = [
        SingleLayerPredictorMLP(
            input_dim=D_MODEL,
            output_dim=SEQ_LEN, # 每个模型输出整个序列的长度
            hidden_dims=PREDICTOR_HIDDEN_DIMS,
            dropout=PREDICTOR_DROPOUT
        ).to(DEVICE) for _ in range(NUM_PREDICTOR_LAYERS)
    ]
    
    # 加载包含所有模型状态的字典
    state_dicts = torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE)
    
    # 为每个模型加载对应的状态
    for i, model in enumerate(predictors):
        model_key = f'model_layer_{i}'
        if model_key in state_dicts:
            model.load_state_dict(state_dicts[model_key])
            model.eval()
        else:
            raise KeyError(f"Weight for '{model_key}' not found in the checkpoint file.")
            
    return opt_model, tokenizer, predictors

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
    
    # 注意：我们的模型输入需要最后一个token的embedding
    # 训练时我们使用了 embeddings[:, -1, :]，这里也应该保持一致
    # 但为了评估，我们需要完整的 embedding 序列来做 mask
    full_embeddings = captured_data['embeddings'].unsqueeze(0).to(torch.float32).to(DEVICE)
    query_embedding = full_embeddings[:, -1, :] # Shape: [1, D_MODEL]

    keys = captured_data['keys'].unsqueeze(0).to(torch.float32).to(DEVICE)
    queries = captured_data['queries'].unsqueeze(0).to(torch.float32).to(DEVICE)
    
    q_last = queries[0, :, -1, :, :]
    k_all = keys[0, :, :, :, :]
    
    attn_scores = torch.einsum('lhd,lshd->lhs', q_last, k_all) / np.sqrt(HEAD_DIM)
    ground_truth_attn_probs = F.softmax(attn_scores, dim=-1)
    
    return full_embeddings, query_embedding, ground_truth_attn_probs

# --- visualize_attention_comparison 函数 (无变化) ---
def visualize_attention_comparison(true_attn, predicted_attn, top_k_indices, layer_to_show, current_seq_len, is_reused=False):
    # (此函数代码保持不变，此处省略以保持简洁)
    true_attn_avg = true_attn.mean(dim=0).cpu().numpy()
    predicted_attn = predicted_attn.cpu().numpy()
    
    title_suffix = " (Reused from Layer 4)" if is_reused else ""
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(f"Layer {layer_to_show}: Predicted Attention vs. Ground Truth{title_suffix}", fontsize=16)
    
    x_axis = np.arange(current_seq_len)
    sns.barplot(x=x_axis, y=predicted_attn, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Predicted Scores (Top {TOP_K_PERCENT*100:.1f}%)")
    axes[0].set_ylabel("Score")
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
    
    save_path = f"attention_comparison_layer_{layer_to_show}.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    opt_model, tokenizer, predictors = load_models()

    text = (
        "Project Summary: A Study on Dynamic KV Cache Prefetching Strategies in LLM Inference. The core idea of this project is to solve the performance bottleneck caused by the huge Key-Value (KV) cache in long-text generation inference of large language models (LLMs). Existing methods, such as Infini-Attention, rely on heuristic or simplified 'rehearsal' calculations to infer and prefetch important KV cache entries. Our core innovation is to propose a learning-based prediction scheme that directly predicts the attention scores for"
    )
    
    print("\n--- Input Text ---")
    print(text)
    
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(DEVICE)
    current_seq_len = input_ids.shape[1]
    
    if current_seq_len > SEQ_LEN:
        input_ids = input_ids[:, :SEQ_LEN]
        current_seq_len = SEQ_LEN
    
    print(f"\nTokenized sequence length: {current_seq_len}")

    # 1. 获取基准真相
    print("\nStep 1: Getting ground truth from full OPT-13B pass...")
    full_embeddings, query_embedding, ground_truth_attn_probs = get_ground_truth(opt_model, input_ids)

    # 2. 使用我们的预测器进行预测
    print("\nStep 2: Using layered predictors to get attention score predictions...")
    predicted_scores_all_layers = []
    with torch.no_grad():
        # <<< 4. 关键修改: 循环调用前5个模型，并复用第5个的结果 >>>
        reused_scores = None
        for i in range(NUM_LAYERS):
            if i < NUM_PREDICTOR_LAYERS:
                # 对前5层，使用各自的模型进行预测
                scores = predictors[i](query_embedding)
                predicted_scores_all_layers.append(scores)
                if i == NUM_PREDICTOR_LAYERS - 1:
                    reused_scores = scores # 保存最后一层的预测结果
            else:
                # 对后续层，复用最后一层的预测结果
                predicted_scores_all_layers.append(reused_scores)
    
    # 将列表合并成一个 tensor
    predicted_scores = torch.cat(predicted_scores_all_layers, dim=0) # Shape: [40, 2048]

    # 3. & 4. 逐层模拟预取、评估并可视化
    print(f"\nStep 3 & 4: Simulating prefetching, Evaluating and Visualizing PER LAYER (Top {TOP_K_PERCENT*100:.1f}%)...")
    k = int(current_seq_len * TOP_K_PERCENT)
    if k == 0: k=1
    
    all_overlaps = []
    layers_to_visualize = [0, 1, 2, 3, 4] # 可视化第0, 4(最后一个独立预测), 10, 39层

    for layer_idx in range(NUM_LAYERS):
        # <<< 5. 关键修改: 从合并后的 tensor 中获取当前层的预测 >>>
        # 注意: predicted_scores 没有 softmax，因为它直接学习排序，这没问题
        predicted_scores_for_layer = predicted_scores[layer_idx, :current_seq_len]
        _, predicted_top_k_indices = torch.topk(predicted_scores_for_layer, k=k, dim=-1)
        
        # Ground truth 部分保持不变
        true_avg_probs_for_layer = ground_truth_attn_probs[layer_idx, :, :current_seq_len].mean(dim=0)
        _, true_top_k_indices = torch.topk(true_avg_probs_for_layer, k=k, dim=-1)
        
        overlap = len(set(predicted_top_k_indices.cpu().numpy()) & set(true_top_k_indices.cpu().numpy()))
        overlap_ratio = overlap / k
        all_overlaps.append(overlap_ratio)
        
        if layer_idx in layers_to_visualize:
            print(f"\n--- Details for Layer {layer_idx} ---")
            is_reused = layer_idx >= NUM_PREDICTOR_LAYERS
            if is_reused:
                print("(Prediction reused from Layer 4)")

            pred_indices_sorted, _ = torch.sort(predicted_top_k_indices)
            print(f"Layer {layer_idx} - Predicted Top-{k} indices: {pred_indices_sorted.cpu().numpy()}")
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0][pred_indices_sorted])
            print(f"Layer {layer_idx} - Corresponding tokens: {tokens}")
            print(f"Layer {layer_idx} - Overlap Ratio with Ground Truth: {overlap_ratio:.2%}")

            visualize_attention_comparison(
                ground_truth_attn_probs[layer_idx, :, :current_seq_len],
                predicted_scores_for_layer, # 直接可视化分数即可
                predicted_top_k_indices,
                layer_idx,
                current_seq_len,
                is_reused
            )

    # --- 最终评估 ---
    avg_overlap_ratio = np.mean(all_overlaps)
    print(f"\n\n--- Final Evaluation Metric ---")
    print(f"Average Per-Layer Top-{k} Overlap Ratio across all {NUM_LAYERS} layers: {avg_overlap_ratio:.2%}")

if __name__ == '__main__':
    main()