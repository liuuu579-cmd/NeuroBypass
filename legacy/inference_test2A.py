# inference_test2A_modified.py 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 导入我们的模型和工具
from predictor_model2A import PredictorTransformer
from dataset2 import TrajectoryDataset 

# --- 配置 ---
OPT_MODEL_PATH = "/home/liushenyu/opt-13b/"
PREDICTOR_MODEL_PATH = "predictor_5layers_epoch_10.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ZARR_PATH = 'opt13b_trajectories_scores_2t' 

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
TOP_K_PERCENT = 0.3

def load_predictor_and_tokenizer():
    """加载我们训练的预测器模型和分词器"""
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
    
    return tokenizer, predictor

def visualize_attention_comparison(true_attn, predicted_attn, top_k_indices, layer_to_show, current_seq_len):
    """可视化函数 (此函数无需修改)"""
    if true_attn.dim() == 2:
        true_attn_avg = true_attn.mean(dim=0).cpu().numpy()
    else:
        true_attn_avg = true_attn.cpu().numpy()
        
    predicted_attn = predicted_attn.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(f"Layer {layer_to_show}: Predicted Attention vs. Ground Truth (Top {TOP_K_PERCENT*100}%)", fontsize=16)
    
    x_axis = np.arange(current_seq_len)
    
    sns.barplot(x=x_axis, y=predicted_attn, ax=axes[0], color='skyblue')
    axes[0].set_title("Predicted Attention Distribution (used for selecting Top-K)")
    axes[0].set_ylabel("Attention Score/Prob")
    
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
    tokenizer, predictor = load_predictor_and_tokenizer()

    # --- 1. 从数据集中加载一个随机样本作为 Ground Truth ---
    print("\nStep 1: Loading data from dataset...")
    try:
        dataset = TrajectoryDataset(ZARR_PATH)
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset from {ZARR_PATH}. Make sure the path is correct.")
        print(f"Error: {e}")
        return

    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"Randomly selected sample #{sample_idx} for inference test.")
    
    embeddings_cpu, true_scores_cpu = dataset[sample_idx]
    
    embeddings = embeddings_cpu.unsqueeze(0).to(DEVICE)
    ground_truth_attn_probs = F.softmax(true_scores_cpu, dim=-1).to(DEVICE)
    current_seq_len = embeddings.shape[1]
    print(f"Sequence length of the sample: {current_seq_len}")
    
    # --- 2. 使用我们的预测器进行预测 ---
    print("\nStep 2: Using predictor to get attention score prediction...")
    # <<< 关键修正 >>>
    # 像训练时一样，创建padding mask并传递给模型
    # 同时使用 torch.autocast 来匹配训练时的混合精度环境
    with torch.no_grad(), torch.autocast("cuda"):
        # src_padding_mask的形状为 [Batch, SeqLen], True的地方代表是padding，需要被mask掉
        src_padding_mask = (embeddings.sum(dim=-1) == 0)
        
        # 将mask传递给predictor
        predicted_scores = predictor(embeddings, src_padding_mask=src_padding_mask)
    # <<< 修正结束 >>>
    
    predicted_attn_probs = F.softmax(predicted_scores, dim=-1).squeeze(0)

    # --- 3. & 4. 逐层模拟预取、评估并可视化 ---
    print(f"\nStep 3 & 4: Simulating prefetching, Evaluating and Visualizing PER LAYER (Top {TOP_K_PERCENT*100}%)...")
    k = int(current_seq_len * TOP_K_PERCENT)
    if k == 0: k=1
    
    all_overlaps = []
    layers_to_visualize = [0, 1, 2, 3, 4, 10, 39] 

    for layer_idx in range(NUM_LAYERS_OPT):
        if layer_idx < NUM_PREDICT_LAYERS:
            prediction_source_idx = layer_idx
        else:
            prediction_source_idx = NUM_PREDICT_LAYERS - 1

        predicted_probs_for_layer = predicted_attn_probs[prediction_source_idx]
        _, predicted_top_k_indices = torch.topk(predicted_probs_for_layer, k=k, dim=-1)
        
        true_probs_for_layer = ground_truth_attn_probs[layer_idx]
        _, true_top_k_indices = torch.topk(true_probs_for_layer, k=k, dim=-1)
        
        overlap = len(set(predicted_top_k_indices.cpu().numpy()) & set(true_top_k_indices.cpu().numpy()))
        overlap_ratio = overlap / k
        all_overlaps.append(overlap_ratio)
        
        if layer_idx in layers_to_visualize:
            print(f"\n--- Details for Layer {layer_idx} ---")
            print(f"(Using prediction from predictor's layer {prediction_source_idx})")
            
            pred_indices_sorted, _ = torch.sort(predicted_top_k_indices)
            print(f"Predicted Top-{k} indices: {pred_indices_sorted.cpu().numpy()}")
            print("(Token visualization skipped as we are loading embeddings directly)")
            print(f"Overlap Ratio with Ground Truth: {overlap_ratio:.2%}")

            visualize_attention_comparison(
                true_probs_for_layer,
                predicted_probs_for_layer,
                predicted_top_k_indices,            
                layer_idx,
                current_seq_len
            )

    avg_overlap_ratio = np.mean(all_overlaps)
    print(f"\n\n--- Final Evaluation Metric ---")
    print(f"Average Per-Layer Top-{k} Overlap Ratio across all {NUM_LAYERS_OPT} layers: {avg_overlap_ratio:.2%}")
    print(f"(This was calculated by using predictor layers 0-4 for OPT layers 0-4, and predictor layer 4 for all OPT layers 5-39)")

if __name__ == '__main__':
    main()