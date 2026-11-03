# inference_test.py (最终适配版，修复设备不匹配错误)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入我们的模型和工具
from predictor_model import PredictorTransformer
from utils import AttentionStateCatcher # utils.py 中是您提供的 AttentionStateCatcher

# --- 配置 ---
OPT_MODEL_PATH = "/home/liushenyu/opt-13b/"
PREDICTOR_MODEL_PATH = "predictor_epoch_10.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# OPT-13B Specifics
NUM_LAYERS = 40
D_MODEL = 5120
NUM_HEADS = 40
HEAD_DIM = 128
SEQ_LEN = 1024

# 预取策略配置
TOP_K_PERCENT = 0.1

def load_models():
    """加载OPT-13B和我们训练的预测器模型"""
    print("Loading OPT-13B model...")
    opt_model = AutoModelForCausalLM.from_pretrained(OPT_MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    opt_model.eval()
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)
    opt_model.resize_token_embeddings(len(tokenizer))

    print("Loading Predictor model...")
    predictor = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_LAYERS,
        opt_head_dim=HEAD_DIM
    ).to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()
    
    return opt_model, tokenizer, predictor

def get_ground_truth(opt_model, input_ids):
    """
    获取OPT模型在完整KV缓存下的内部状态和注意力分布。
    此函数适配了会squeeze掉batch维度的AttentionStateCatcher。
    """
    catcher = AttentionStateCatcher(opt_model)
    catcher.register_hooks()
    with torch.no_grad():
        # input_ids shape is [1, Seq_Len], which is a batch of 1
        opt_model(input_ids)
    
    # get_captured_data() 会移除batch维度
    captured_data = catcher.get_captured_data()
    catcher.remove_hooks() # 及时移除钩子
    
    # --- VVVVVVVV 关键适配代码 VVVVVVVV ---
    # `captured_data` 中的张量都没有batch维度, 我们需要手动加上
    # embeddings: [S, D] -> [1, S, D]
    # **并将其移动到正确的设备**
    embeddings = captured_data['embeddings'].unsqueeze(0).to(torch.float32).to(DEVICE)
    # keys: [L, S, H, D] -> [1, L, S, H, D]
    keys = captured_data['keys'].unsqueeze(0).to(torch.float32).to(DEVICE)
    # queries: [L, S, H, D] -> [1, L, S, H, D]
    queries = captured_data['queries'].unsqueeze(0).to(torch.float32).to(DEVICE)
    # --- ^^^^^^^^ 适配结束 ^^^^^^^^ ---
    
    # 现在所有张量都有了 batch=1 的维度，且都在DEVICE上，后续代码可以统一处理
    # q_last: [L, H, D]
    q_last = queries[0, :, -1, :, :]
    # k_all: [L, S, H, D]
    k_all = keys[0, :, :, :, :]
    
    # attn_scores: [L, H, S]
    attn_scores = torch.einsum('lhd,lshd->lhs', q_last, k_all) / np.sqrt(HEAD_DIM)
    # attn_probs: [L, H, S]
    ground_truth_attn_probs = F.softmax(attn_scores, dim=-1)
    
    # 返回带有 batch 维度的 embeddings 和 keys，以供后续使用
    return embeddings, keys, ground_truth_attn_probs

# visualize_attention_comparison 函数保持不变
# 优化了top_k_indices在cpu()后使用numpy()以兼容plt/sns
def visualize_attention_comparison(true_attn, predicted_attn, top_k_indices, layer_to_show):
    num_heads, seq_len = true_attn.shape
    true_attn_avg = true_attn.mean(dim=0).cpu().numpy()
    predicted_attn = predicted_attn.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(f"Layer {layer_to_show}: Predicted Attention vs. Ground Truth (Top {TOP_K_PERCENT*100}%)", fontsize=16)
    
    sns.barplot(x=np.arange(seq_len), y=predicted_attn, ax=axes[0], color='skyblue')
    axes[0].set_title("Predicted Attention Distribution (used for selecting Top-K)")
    axes[0].set_ylabel("Attention Score")
    for i in top_k_indices.cpu().numpy(): # <--- 修正：确保索引在CPU上
        if i < len(axes[0].patches):
            axes[0].patches[i].set_facecolor('red')

    sns.barplot(x=np.arange(seq_len), y=true_attn_avg, ax=axes[1], color='lightgreen')
    axes[1].set_title("Ground Truth Attention (Averaged over all heads)")
    axes[1].set_xlabel("Token Position in Sequence")
    axes[1].set_ylabel("Attention Probability")
    for i in top_k_indices.cpu().numpy(): # <--- 修正：确保索引在CPU上
        if i < len(axes[1].patches):
            axes[1].patches[i].set_facecolor('red')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"attention_comparison_layer_{layer_to_show}.png")
    print(f"Saved visualization to attention_comparison_layer_{layer_to_show}.png")
    plt.close()

# main 函数保持不变，因为之前的修改已经正确处理了 input_ids 到 DEVICE
def main():
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

    # 1. 获取基准真相 (Ground Truth)
    print("\nStep 1: Getting ground truth from full OPT-13B pass...")
    embeddings, full_keys, ground_truth_attn_probs = get_ground_truth(opt_model, input_ids)
    # embeddings: [1, S, D], full_keys: [1, L, S, H, D], ground_truth_attn_probs: [L, H, S]

    # 2. 使用我们的预测器进行预测
    print("\nStep 2: Using predictor to get attention prediction...")
    with torch.no_grad():
        # predictor 输入需要 batch 维度，我们的 embeddings 已经是 [1, S, D] 且在DEVICE上
        predicted_proxy = predictor(embeddings) # -> [1, L, D_head]
    
    # 3. 计算预测的注意力分布
    # full_keys[0]: [L, S, H, D] -> keys_avg_heads: [L, S, D_head]
    keys_avg_heads = full_keys[0].mean(dim=2)
    # predicted_proxy[0]: [L, D_head]
    # predicted_scores: [L, S]
    predicted_scores = torch.einsum('ld,lsd->ls', predicted_proxy[0], keys_avg_heads)
    predicted_attn_probs = F.softmax(predicted_scores, dim=-1)

    # 4. 模拟预取流程
    print(f"\nStep 3: Simulating prefetching based on prediction (Top {TOP_K_PERCENT*100}%)...")
    k = int(current_seq_len * TOP_K_PERCENT)
    if k == 0: k=1
    
    global_predicted_scores = predicted_attn_probs.mean(dim=0) # [S]
    top_k_scores, top_k_indices = torch.topk(global_predicted_scores, k=k, dim=-1)
    
    top_k_indices_sorted, _ = torch.sort(top_k_indices)
    print(f"Selected Top-{k} indices for prefetching: {top_k_indices_sorted.cpu().numpy()}")
    
    selected_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][top_k_indices_sorted])
    print(f"Corresponding tokens: {selected_tokens}")

    # 5. 计算指标并可视化
    print("\nStep 4: Evaluating and Visualizing...")
    total_overlap = 0
    for layer in range(NUM_LAYERS):
        true_top_k_indices = torch.topk(ground_truth_attn_probs[layer].mean(dim=0), k=k)[1]
        overlap = len(set(top_k_indices.cpu().numpy()) & set(true_top_k_indices.cpu().numpy()))
        total_overlap += overlap / k

    avg_overlap_ratio = total_overlap / NUM_LAYERS
    print(f"\n--- Evaluation Metric ---")
    print(f"Average Top-{k} Overlap Ratio across all layers: {avg_overlap_ratio:.2%}")
    print("(This measures how well our predicted important tokens match the actual important tokens)")

    layers_to_visualize = [0, 10, 20, 39]
    for layer_idx in layers_to_visualize:
        visualize_attention_comparison(
            ground_truth_attn_probs[layer_idx],
            predicted_attn_probs[layer_idx],
            top_k_indices,
            layer_idx
        )

if __name__ == '__main__':
    main()