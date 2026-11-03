# inference_multilabel.py (Modified for ALPHA and BUDGET_RATIO with new prefetch logic)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from predictor_model2A import PredictorTransformer
from dataset2 import TrajectoryDataset 


# --- 配置 ---
# (确保这里的路径和文件名与你使用 ALPHA 训练出的模型一致)
PREDICTOR_MODEL_PATH = "predictor_topk_0.3_iwl_epoch_20.pth" # <<< 示例: 使用你训练好的模型
#PREDICTOR_MODEL_PATH = "predictor_alpha_2_dyn_posw_epoch_20.pth" # <<< 示例: 使用你训练好的模型
ZARR_PATH = 'opt13b_trajectories_scores_multi_point'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 你可以选择任意一条训练数据来进行测试
SAMPLE_INDEX_TO_TEST = 102

# --- 新的预取策略配置 (必须与训练代码匹配!) ---
# <<< MODIFICATION 1: Add BUDGET_RATIO to match training script >>>
ALPHA = 2.0  # 这个 ALPHA 值应该和你训练时使用的值相匹配
BUDGET_RATIO = 0.3 # 这个 BUDGET_RATIO 值也必须和你训练时使用的值相匹配
# <<< END MODIFICATION 1 >>>

# 模型结构超参数 (必须与训练时一致)
D_MODEL = 5120
NUM_PREDICT_LAYERS = 5


PREDICTOR_NHEAD = 4
PREDICTOR_ENCODER_LAYERS = 1
PREDICTOR_D_HID = 2048


'''
PREDICTOR_NHEAD = 16
PREDICTOR_ENCODER_LAYERS = 8
PREDICTOR_D_HID = 4096
'''
NUM_LAYERS_OPT = 40

# --- visualize_attention_comparison 函数修改 ---
# <<< MODIFICATION 2: Update visualization title >>>
def visualize_attention_comparison(true_attn_probs, predicted_prefetch_mask, ground_truth_mask, layer_to_show, current_seq_len):
    """
    修改后的可视化函数，用于比较 Ground Truth 注意力概率和预测的二值预取掩码。
    """
    predicted_mask_np = predicted_prefetch_mask.cpu().numpy()
    ground_truth_mask_np = ground_truth_mask.cpu().numpy() # 使用传入的真实掩码
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    # 更新标题以包含所有相关参数
    fig.suptitle(f"Layer {layer_to_show}: Prefetch Comparison (ALPHA={ALPHA}, BUDGET={BUDGET_RATIO*100}%)", fontsize=16)
    
    x_axis = np.arange(current_seq_len)
    
    top_colors = ['red' if val == 1 else 'skyblue' for val in predicted_mask_np]
    bottom_colors = ['red' if val == 1 else 'lightgreen' for val in ground_truth_mask_np]

    # 上图：显示我们的预测 (二值掩码)
    sns.barplot(x=x_axis, y=predicted_mask_np, ax=axes[0], palette=top_colors)
    axes[0].set_title(f"Predicted Prefetch Mask (1 = Prefetch, Actual count: {int(predicted_mask_np.sum())} tokens)")
    axes[0].set_ylabel("Prediction")
    axes[0].set_yticks([0, 1])

    # 下图：显示基准真相 (二值掩码)
    sns.barplot(x=x_axis, y=ground_truth_mask_np, ax=axes[1], palette=bottom_colors)
    axes[1].set_title(f"Ground Truth Prefetch Mask (1 = Prefetch, Actual count: {int(ground_truth_mask_np.sum())} tokens)")
    axes[1].set_xlabel("Token Position in Sequence")
    axes[1].set_ylabel("Ground Truth")
    axes[1].set_yticks([0, 1])


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"prefetch_comparison_alpha_{ALPHA}_budget_{BUDGET_RATIO}_layer_{layer_to_show}.png")
    print(f"Saved visualization to prefetch_comparison_alpha_{ALPHA}_budget_{BUDGET_RATIO}_layer_{layer_to_show}.png")
    plt.close()
# <<< END MODIFICATION 2 >>>

def main():
    # --- 模型加载 ---
    print(f"Loading Predictor model from {PREDICTOR_MODEL_PATH}...")
    predictor = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_PREDICT_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()
    print("Predictor model loaded successfully.")

    # --- Step 1: 从训练数据集中加载一条样本 ---
    print(f"\nLoading training sample from '{ZARR_PATH}' at index {SAMPLE_INDEX_TO_TEST}...")
    dataset = TrajectoryDataset(ZARR_PATH)
    
    if SAMPLE_INDEX_TO_TEST >= len(dataset):
        raise IndexError(f"SAMPLE_INDEX_TO_TEST ({SAMPLE_INDEX_TO_TEST}) is out of bounds for the dataset of size {len(dataset)}.")

    sample_embeddings, sample_true_scores = dataset[SAMPLE_INDEX_TO_TEST]
    
    non_zero_mask = sample_embeddings.abs().sum(dim=-1) > 0
    current_seq_len = non_zero_mask.sum().item()
    print(f"Sample loaded. Effective sequence length: {current_seq_len}")

    embeddings = sample_embeddings[:current_seq_len].unsqueeze(0).to(torch.float32).to(DEVICE) 
    true_scores = sample_true_scores[:, :current_seq_len].to(DEVICE)

    # --- Step 2: 生成 Ground Truth 预取掩码 (使用Alpha + Budget) ---
    # <<< MODIFICATION 3: Replicate the EXACT ground truth generation from training >>>
    print(f"Generating ground truth prefetch mask using ALPHA={ALPHA} and BUDGET_RATIO={BUDGET_RATIO}...")
    with torch.no_grad():
        # 1. 应用 ALPHA 规则
        max_scores, _ = torch.max(true_scores, dim=-1, keepdim=True)
        thresholds = max_scores - ALPHA
        ground_truth_mask_alpha = (true_scores >= thresholds).float() # Shape: (L, S)

        # 2. 计算预算并检查是否超出
        # k_cap for each layer. Shape: (L)
        k_cap = int(current_seq_len * BUDGET_RATIO)
        num_selected_alpha = ground_truth_mask_alpha.sum(dim=-1) # Shape: (L)
        
        # 3. 如果超出预算，则用 topk 强制约束
        ground_truth_mask = ground_truth_mask_alpha.clone()
        exceeded_layers_mask = num_selected_alpha > k_cap
        
        if torch.any(exceeded_layers_mask):
            # 获取需要修正的层的索引
            exceeded_indices = torch.where(exceeded_layers_mask)[0]
            print(f"Ground Truth: Layers {exceeded_indices.cpu().numpy()} exceeded budget ({k_cap}). Applying top-k constraint.")
            
            # 对这些层应用 top-k
            scores_to_cap = true_scores[exceeded_layers_mask] # Shape: (num_exceeded, S)
            _, topk_indices = torch.topk(scores_to_cap, k=k_cap, dim=-1)
            
            # 创建新的掩码并替换
            capped_mask = torch.zeros_like(scores_to_cap, device=DEVICE)
            capped_mask.scatter_(-1, topk_indices, 1.0)
            ground_truth_mask[exceeded_layers_mask] = capped_mask

    # <<< END MODIFICATION 3 >>>

    print("\nStep 3: Using predictor to get prefetch probability prediction...")
    with torch.no_grad():
        predicted_logits = predictor(embeddings).squeeze(0) # Shape: (L, S)
    predicted_prefetch_probs = torch.sigmoid(predicted_logits)

    # --- Step 4: 根据预测概率和预算生成预测掩码 ---
    # <<< MODIFICATION 4 (REPLACED): 根据参考函数逻辑，生成 Top-k + 最近Token 的掩码 >>>
    print(f"Generating predicted mask with new logic: Top-k (Budget-based) OR Recency Mask...")

    # 1. 基础的 Top-K 预取掩码 (纯粹基于模型预测分数的 Top-K)
    k_cap_pred = int(current_seq_len * BUDGET_RATIO)
    print(f"Applying Top-k constraint with k = {k_cap_pred} based on budget.")
    
    # topk_pred_indices 是模型认为分数最高的 k 个位置
    _, topk_pred_indices = torch.topk(predicted_prefetch_probs, k=k_cap_pred, dim=-1)
    base_prefetch_mask = torch.zeros_like(predicted_prefetch_probs, device=DEVICE)
    base_prefetch_mask.scatter_(-1, topk_pred_indices, 1.0)
    NUM_RECENT_TOKENS = 10
    # ==============================================================================
    # >>> [新增检查逻辑]：检查模型是否“自发”地学会了选择最近的 10 个 token <<<
    # ==============================================================================
    print(f"\n--- [DEBUG CHECK] Does the TRAINED MODEL naturally select the last {NUM_RECENT_TOKENS} tokens? ---")
    recent_start_idx = max(0, current_seq_len - NUM_RECENT_TOKENS)
    # 生成最近10个token的索引张量
    recent_indices_tensor = torch.arange(recent_start_idx, current_seq_len, device=DEVICE)
    
    for i in range(NUM_PREDICT_LAYERS):
        # 获取当前层模型Top-K选中的索引
        layer_selected_indices = topk_pred_indices[i]
        
        # 检查 recent_indices 是否都在 layer_selected_indices 中
        # 使用 torch.isin 判断 (需要 PyTorch >= 1.10)
        is_captured = torch.isin(recent_indices_tensor, layer_selected_indices)
        num_captured = is_captured.sum().item()
        
        status = "PERFECT" if num_captured == len(recent_indices_tensor) else "FAILED"
        print(f"Layer {i}: {status} - Model raw Top-K picked {num_captured}/{len(recent_indices_tensor)} recent tokens.")
        
        # 如果没选全，打印一下漏掉的token的预测概率是多少，方便分析
        if num_captured < len(recent_indices_tensor):
            missing_mask = ~is_captured
            missing_indices = recent_indices_tensor[missing_mask]
            missing_probs = predicted_prefetch_probs[i, missing_indices]
            print(f"    -> Missing indices: {missing_indices.cpu().numpy()}")
            print(f"    -> Probabilities of missing tokens: {missing_probs.detach().cpu().numpy()}")
            print(f"    -> (Note: The lowest prob in Top-K was: {torch.min(torch.topk(predicted_prefetch_probs[i], k=k_cap_pred).values).item():.4f})")
    print("----------------------------------------------------------------------------------\n")
    # ==============================================================================

    # 2. 强制保留最近 token 的掩码 (这是原本的补丁逻辑)
    print(f"Applying recency mask to always keep the last {NUM_RECENT_TOKENS} tokens.")
    recency_mask = torch.zeros_like(predicted_prefetch_probs, device=DEVICE)
    num_to_force_keep = min(current_seq_len, NUM_RECENT_TOKENS)
    if num_to_force_keep > 0:
        recency_mask[:, -num_to_force_keep:] = 1.0

    # 3. 使用逻辑 "或" 将两个掩码合并
    print("Combining the Top-k mask and Recency mask using logical OR.")
    predicted_mask = torch.logical_or(base_prefetch_mask.bool(), recency_mask.bool()).float()
    
    # 打印合并前后预取数量的变化，用于调试
    num_predicted_before = base_prefetch_mask.sum(dim=-1).cpu().numpy()
    num_predicted_after = predicted_mask.sum(dim=-1).cpu().numpy()
    print(f"Num predicted tokens per layer (Top-k only): {num_predicted_before.astype(int)}")
    print(f"Num predicted tokens per layer (Combined):   {num_predicted_after.astype(int)}")
    # <<< END MODIFICATION 4 >>>

    print(f"\nStep 5: Evaluating and Visualizing PER LAYER (ALPHA={ALPHA}, BUDGET={BUDGET_RATIO})...")
    
    # --- Step 6: 计算评估指标 ---
    # <<< MODIFICATION 5: Minor updates to evaluation logic, no structural change needed >>>
    all_metrics = []
    layers_to_visualize = [0, 1, 2, 3, 4] 

    for layer_idx in range(NUM_LAYERS_OPT):
        prediction_source_idx = min(layer_idx, NUM_PREDICT_LAYERS - 1)

        pred_mask_layer = predicted_mask[prediction_source_idx]
        true_mask_layer = ground_truth_mask[layer_idx]
        
        tp = (pred_mask_layer * true_mask_layer).sum().item()
        fp = (pred_mask_layer * (1 - true_mask_layer)).sum().item()
        fn = ((1 - pred_mask_layer) * true_mask_layer).sum().item()

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        num_predicted = pred_mask_layer.sum().item()
        num_true = true_mask_layer.sum().item()

        all_metrics.append({
            'Layer': layer_idx,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Num Predicted': int(num_predicted),
            'Num True': int(num_true)
        })

        if layer_idx in layers_to_visualize:
            print(f"\n--- Details for Layer {layer_idx} (using prediction from layer {prediction_source_idx}) ---")
            print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")
            print(f"Predicted to prefetch {int(num_predicted)} tokens. Ground truth had {int(num_true)}.")

            # 为了可视化，我们传递原始的注意力概率
            true_attn_probs_for_viz = F.softmax(true_scores[layer_idx], dim=-1)
            visualize_attention_comparison(
                true_attn_probs_for_viz, # 原始概率，但绘图函数现在主要使用掩码
                pred_mask_layer,         # 预测掩码
                true_mask_layer,         # <<< NEW >>> 传递真实的掩码用于绘图
                layer_idx,
                current_seq_len
            )
            
    # --- 最终评估 ---
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.mean()
    
    # 在计算平均值时，使用 ground truth 的 k_cap
    k_cap_gt = int(current_seq_len * BUDGET_RATIO)
    
    print("\n\n--- Final Evaluation Metrics ---")
    print(f"Averaged across all {NUM_LAYERS_OPT} layers:")
    print(f"  - Average Precision: {avg_metrics['Precision']:.2%}")
    print(f"  - Average Recall:    {avg_metrics['Recall']:.2%}")
    print(f"  - Average F1 Score:    {avg_metrics['F1 Score']:.2%}")
    print(f"  - Avg Num Predicted: {avg_metrics['Num Predicted']:.1f} (Budget for top-k part was {k_cap_pred})")
    print(f"  - Avg Num True:      {avg_metrics['Num True']:.1f}")

    print("\n--- Per-Layer Metrics ---")
    print(metrics_df.to_string(index=False, formatters={
        'Precision': '{:.2%}'.format,
        'Recall': '{:.2%}'.format,
        'F1 Score': '{:.2%}'.format
    }))
    # <<< END MODIFICATION 5 >>>

if __name__ == '__main__':
    main()