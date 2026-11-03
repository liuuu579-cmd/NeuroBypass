import os
import torch
import numpy as np
import zarr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import gc
import traceback

# 导入你原有的工具类
from utils2 import AttentionStateCatcher 

# ==========================================
# 1. 核心配置区
# ==========================================
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
OUTPUT_ZARR_DIR = 'opt13b_real_iok_results.zarr' # 换了个名字避免冲突
OUTPUT_IMG_DIR = 'analysis_results_visual'

CHUNK_SIZE = 2048
TARGET_TRAJECTORIES = 100
PRIMARY_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- 论文核心假设参数 ---
BUDGET_RATIO = 0.40  # 预取容量限制: 20%
MIN_T = 64           # 前 64 个 Token 的统计噪音太大，跳过不算

# ==========================================
# 2. 高效的数据计算与收集
# ==========================================
def compute_layer_intersection_rates(queries, keys):
    """
    向量化计算整层序列中，相邻两步(t与t+1)的 Top-K 交集率。
    利用 GPU 矩阵乘法，瞬间完成原本需要嵌套循环的计算。
    """
    S, H, D = queries.shape
    
    # 转置为 batch 乘法格式: (H, S, D)
    q = queries.permute(1, 0, 2).float()
    k = keys.permute(1, 0, 2).float()
    
    # 核心: 一次性计算所有 step 的 Attention Scores
    # attn 形状: (H, S, S)
    attn = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)
    
    # 跨多头平均，模拟最终的 heavy-hitter 重要度: (S, S)
    attn_avg = attn.mean(dim=0)
    
    # 应用因果掩码 (下三角)
    mask = torch.tril(torch.ones(S, S, device=queries.device))
    attn_avg = attn_avg.masked_fill(mask == 0, float('-inf'))
    
    # Softmax 计算概率
    attn_probs = torch.softmax(attn_avg, dim=-1)
    
    rates = []
    # 逐 step 计算相邻两步的 Top-K 交集
    for t in range(MIN_T, S - 1):
        # 根据动态上下文长度计算 Budget (也可以固定大小，取决于论文写死还是动态)
        k_t = max(1, int(t * BUDGET_RATIO))
        k_t1 = max(1, int((t + 1) * BUDGET_RATIO))
        
        # 提取 top-K 索引
        _, top_t = torch.topk(attn_probs[t, :t+1], k_t)
        _, top_t1 = torch.topk(attn_probs[t+1, :t+2], k_t1)
        
        # 移至 CPU 进行集合运算
        arr_t = top_t.cpu().numpy()
        arr_t1 = top_t1.cpu().numpy()
        
        # 计算交集个数 / 最大容量限制
        intersect = np.intersect1d(arr_t, arr_t1, assume_unique=True)
        rate = len(intersect) / max(k_t, k_t1)
        rates.append(rate)
        
    return np.array(rates, dtype=np.float32)

def collect_data():
    print("正在加载模型和分词器 (float16)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    model.eval()
    
    config = AutoConfig.from_pretrained(MODEL_PATH)
    NUM_LAYERS = config.num_hidden_layers
    
    # 初始化 Zarr
    if not os.path.exists(OUTPUT_ZARR_DIR):
        os.makedirs(OUTPUT_ZARR_DIR)
    root = zarr.open_group(OUTPUT_ZARR_DIR, mode='a')
    
    seq_len_analyzed = CHUNK_SIZE - MIN_T - 1
    shape = (TARGET_TRAJECTORIES, NUM_LAYERS, seq_len_analyzed)
    
    if 'topk_iok' not in root:
        root.create_dataset('topk_iok', shape=shape, dtype='f4')
        start_index = 0
    else:
        # 简单的续传逻辑
        non_zero_rows = np.where(root['topk_iok'][:, 0, 0] != 0)[0]
        start_index = len(non_zero_rows)

    if start_index >= TARGET_TRAJECTORIES:
        print("数据已经收集完毕，直接跳过生成阶段。")
        return NUM_LAYERS

    catcher = AttentionStateCatcher(model)

    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    num_chunks = len(all_tokens) // CHUNK_SIZE

    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="计算真实 IoK 轨迹")
    
    chunk_idx = start_index
    trajectories_written = start_index
    
    while trajectories_written < TARGET_TRAJECTORIES and chunk_idx < num_chunks:
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
        if len(chunk_tokens) < CHUNK_SIZE:
            break
            
        input_ids = chunk_tokens.unsqueeze(0).to(PRIMARY_DEVICE)
        catcher.reset()
        catcher.register_hooks()

        try:
            with torch.no_grad():
                model(input_ids)
            
            catcher.remove_hooks()
            captured_data = catcher.get_captured_data()
            
            traj_iok_results = np.zeros((NUM_LAYERS, seq_len_analyzed), dtype=np.float32)

            for layer_idx in range(NUM_LAYERS):
                queries_gpu = captured_data['queries'][layer_idx].to(PRIMARY_DEVICE).squeeze(0)
                keys_gpu = captured_data['keys'][layer_idx].to(PRIMARY_DEVICE).squeeze(0)

                # 极速计算一整层的 IoK
                layer_iok = compute_layer_intersection_rates(queries_gpu, keys_gpu)
                traj_iok_results[layer_idx, :] = layer_iok
                
                del queries_gpu, keys_gpu
                torch.cuda.empty_cache()

            root['topk_iok'][trajectories_written] = traj_iok_results
            
            trajectories_written += 1
            pbar.update(1)

        except Exception as e:
            traceback.print_exc()
            catcher.remove_hooks()
        finally:
            chunk_idx += 1
            gc.collect()
            torch.cuda.empty_cache()

    pbar.close()
    return NUM_LAYERS

# ==========================================
# 3. 生成顶级学术图表
# ==========================================
def plot_real_data(num_layers):
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    root = zarr.open_group(OUTPUT_ZARR_DIR, mode='r')
    data = root['topk_iok'][:]
    
    print(f"数据加载完成，形状为: {data.shape} (轨迹数, 层数, 序列步数)")
    
    # 我们挑几层有代表性的画图，比如 Layer 0 (浅层散漫), Layer 4 (目标预测层), Layer 39 (深层)
    target_layers = [0, 4, num_layers - 1] 
    
    for layer in target_layers:
        print(f"正在绘制 Layer {layer} 的图表...")
        
        # 1. 取出第一条轨迹，画 Step-by-Step 趋势图 (展现真实局部特征)
        single_traj_data = data[0, layer, :]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        steps = np.arange(MIN_T, MIN_T + len(single_traj_data))
        
        # 画原始灰线
        ax.plot(steps, single_traj_data, color='gray', alpha=0.3, linewidth=1, label='Raw Step-by-Step IoK')
        # 画平滑蓝线
        smoothed_data = pd.Series(single_traj_data).rolling(window=16, min_periods=1).mean()
        ax.plot(steps, smoothed_data, color='#1f77b4', linewidth=2.5, label='Moving Average (Window=16)')

        ax.set_title(f'Temporal Locality of Top-K KV Cache (Layer {layer}, Budget={BUDGET_RATIO*100}%)\nReal OPT-13B Inference on C4', fontsize=14, pad=15)
        ax.set_xlabel('Decoding Step (Sequence Length)', fontsize=12)
        ax.set_ylabel('Top-K Intersection Rate (IoK)', fontsize=12)
        ax.set_ylim(0.4, 1.05)
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f'real_trend_layer_{layer}.png'), dpi=300)
        plt.close()

        # 2. 将所有轨迹拼接拉平，画宏观统计 CDF 图 (终极杀手锏)
        # 把 (100, 1984) 压平成 198,400 个点
        all_traj_data = data[:, layer, :].flatten()
        # 剔除可能的 0 值 (如果之前中断过导致有空数据)
        all_traj_data = all_traj_data[all_traj_data > 0]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.ecdfplot(data=all_traj_data, ax=ax, color='#ff7f0e', linewidth=3)
        
        percentile_10_val = np.percentile(all_traj_data, 10)
        ax.axvline(x=percentile_10_val, color='r', linestyle='--', alpha=0.6)
        ax.axhline(y=0.10, color='r', linestyle='--', alpha=0.6)
        ax.plot(percentile_10_val, 0.10, marker='o', color='red')
        ax.text(percentile_10_val - 0.02, 0.15, 
                f'90% of steps have\nIoK > {percentile_10_val:.2f}', 
                color='red', fontsize=11, ha='right')

        ax.set_title(f'Cumulative Distribution of Top-K Intersection (Layer {layer})', fontsize=14, pad=15)
        ax.set_xlabel('Top-K Intersection Rate (IoK)', fontsize=12)
        ax.set_ylabel('Cumulative Fraction of Decoding Steps', fontsize=12)
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f'real_cdf_layer_{layer}.png'), dpi=300)
        plt.close()


def plot_real_data_average_all_layers():
    """
    计算并绘制所有层的平均 Top-K 交集率 (System-Level Macro View)
    """
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    root = zarr.open_group(OUTPUT_ZARR_DIR, mode='r')
    
    # data 形状: (轨迹数, 层数, 序列步数)
    data = root['topk_iok'][:]
    print(f"原始数据加载完成，形状为: {data.shape}")
    
    # === 核心修改：在 Layer 维度 (axis=1) 上求平均 ===
    # avg_data 形状变为: (轨迹数, 序列步数)
    avg_data = np.mean(data, axis=1)
    print(f"已计算所有层的平均值，平均后数据形状为: {avg_data.shape}")
    
    # ---------------------------------------------------------
    # 1. 绘制方法二：单条轨迹的全局平均趋势图 (Moving Average Trend)
    # ---------------------------------------------------------
    single_traj_avg = avg_data[0, :] # 取第一条轨迹展示动态趋势
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    steps = np.arange(MIN_T, MIN_T + len(single_traj_avg))
    
    # 画原始散点折线（由于已经经过了 40 层平均，这里的毛刺会明显少很多）
    ax.plot(steps, single_traj_avg, color='gray', alpha=0.4, linewidth=1, label='Raw Step-by-Step Average IoK')
    
    # 计算并画滑动平均线
    smoothed_data = pd.Series(single_traj_avg).rolling(window=16, min_periods=1).mean()
    ax.plot(steps, smoothed_data, color='#1f77b4', linewidth=3, label='Moving Average (Window=16)')

    #ax.set_title(f'Global Temporal Locality of KV Cache (Average across all 40 Layers)\nBudget={BUDGET_RATIO*100}%, Real OPT-13B Inference', fontsize=14, pad=15)
    ax.set_xlabel('Decoding Step (Sequence Length)', fontsize=22)
    ax.set_ylabel('Top-K Intersection Rate (IoK)', fontsize=22)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right', fontsize=20)
    
    plt.tight_layout()
    trend_filename = os.path.join(OUTPUT_IMG_DIR, 'real_trend_average_all_layers.png')
    plt.savefig(trend_filename, dpi=300)
    plt.close()
    print(f"全局平均趋势图已保存至: {trend_filename}")

    # ---------------------------------------------------------
    # 2. 绘制方法三：所有轨迹打平后的全局 CDF 累积分布图 (定海神针)
    # ---------------------------------------------------------
    # 压平所有轨迹的数据，用于统计 CDF
    all_traj_avg_flat = avg_data.flatten()
    # 剔除异常的0值（如果有的话）
    all_traj_avg_flat = all_traj_avg_flat[all_traj_avg_flat > 0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.ecdfplot(data=all_traj_avg_flat, ax=ax, color='#ff7f0e', linewidth=3)
    
    # 计算 10% 分位数（即 90% 的情况都大于这个值）
    percentile_10_val = np.percentile(all_traj_avg_flat, 10)
    
    # 绘制参考线
    ax.axvline(x=percentile_10_val, color='r', linestyle='--', alpha=0.6)
    ax.axhline(y=0.10, color='r', linestyle='--', alpha=0.6)
    ax.plot(percentile_10_val, 0.10, marker='o', color='red')
    
    # 添加强有力的文本标注
    ax.text(percentile_10_val - 0.02, 0.15, 
            f'90% of steps maintain\nan average IoK > {percentile_10_val:.2f}', 
            color='red', fontsize=11, ha='right', weight='bold')

    ax.set_title('Cumulative Distribution of Global Top-K Intersection (All Layers Avg)', fontsize=13, pad=15)
    ax.set_xlabel('Global Average Top-K Intersection Rate (IoK)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction of Decoding Steps', fontsize=12)
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    cdf_filename = os.path.join(OUTPUT_IMG_DIR, 'real_cdf_average_all_layers.png')
    plt.savefig(cdf_filename, dpi=300)
    plt.close()
    print(f"全局平均 CDF 图已保存至: {cdf_filename}")

if __name__ == '__main__':
    # 第一步: 收集真实数据到 Zarr (支持断点续传)
    #num_layers = collect_data()
    
    # 第二步: 读取 Zarr 绘制顶会级别的漂亮图表
    plot_real_data_average_all_layers()
    print(f"全部完成！图片已保存在 {OUTPUT_IMG_DIR}/ 目录。")