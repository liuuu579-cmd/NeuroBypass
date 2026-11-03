import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
ZARR_PATH = 'opt13b_layer_skip_scores'  # 你刚才生成数据的目录
NUM_SAMPLES_TO_PLOT = 50  # 画多少条数据的热力图
SKIP_THRESHOLD = 0.05     # 设定一个 KL 散度阈值，低于这个值认为该层“可跳过”

def main():
    if not os.path.exists(ZARR_PATH):
        print(f"Error: Directory {ZARR_PATH} not found.")
        return

    print(f"Loading data from {ZARR_PATH}...")
    root = zarr.open_group(ZARR_PATH, mode='r')
    
    if 'skip_scores' not in root:
        print("Error: 'skip_scores' dataset not found in Zarr.")
        return

    # 读取全部或部分数据
    # Shape: [num_trajectories, num_layers]
    skip_scores = root['skip_scores'][:]
    total_samples, num_layers = skip_scores.shape
    
    print(f"Total samples: {total_samples}")
    print(f"Number of layers: {num_layers}")
    
    # 随机采样一些数据来画图
    indices = np.random.choice(total_samples, NUM_SAMPLES_TO_PLOT, replace=False)
    # 或者直接取前 N 条
    # indices = np.arange(NUM_SAMPLES_TO_PLOT)
    
    sampled_scores = skip_scores[indices]

    # --- 统计分析 ---
    # 计算平均每层的重要性
    avg_score_per_layer = np.mean(skip_scores, axis=0)
    
    # 计算在该阈值下，平均每条数据可以跳过多少层
    skippable_counts = np.sum(skip_scores < SKIP_THRESHOLD, axis=1)
    avg_skipped = np.mean(skippable_counts)
    
    print("\n--- Statistics ---")
    print(f"KL Divergence Threshold for Skipping: {SKIP_THRESHOLD}")
    print(f"Average layers skipped per sample: {avg_skipped:.2f} / {num_layers} ({avg_skipped/num_layers:.1%})")
    print(f"Max layers skipped in a single sample: {np.max(skippable_counts)}")
    print(f"Min layers skipped in a single sample: {np.min(skippable_counts)}")

    # --- 绘图 1: 热力图 ---
    plt.figure(figsize=(15, 10))
    
    # 使用 log1p (log(1+x)) 来处理数据，因为 KL 散度可能跨度很大（有的 0.0001，有的 10.0）
    # 这样能让暗部细节（接近 0 的部分）看得更清楚
    plot_data = np.log1p(sampled_scores)
    
    sns.heatmap(plot_data, cmap="viridis", cbar_kws={'label': 'Log(1 + KL_Divergence)'})
    
    plt.title(f'Layer Importance Heatmap (Top {NUM_SAMPLES_TO_PLOT} Random Samples)')
    plt.xlabel('Layer Index (0-31)')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig('layer_importance_heatmap.png')
    print("\nHeatmap saved to 'layer_importance_heatmap.png'")

    # --- 绘图 2:各层平均重要性柱状图 ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_layers), avg_score_per_layer, color='skyblue')
    plt.xlabel('Layer Index')
    plt.ylabel('Average KL Divergence')
    plt.title('Average Importance per Layer (across all samples)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('avg_layer_importance.png')
    print("Bar chart saved to 'avg_layer_importance.png'")
    
    # --- 绘图 3: 跳层数量分布直方图 ---
    plt.figure(figsize=(10, 6))
    plt.hist(skippable_counts, bins=range(num_layers + 2), align='left', rwidth=0.8, color='orange')
    plt.xlabel('Number of Skippable Layers')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Skippable Layers (Threshold < {SKIP_THRESHOLD})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('skip_count_distribution.png')
    print("Histogram saved to 'skip_count_distribution.png'")

    plt.show()

if __name__ == '__main__':
    main()