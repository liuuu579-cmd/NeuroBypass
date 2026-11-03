import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
# 确保这个路径与数据采集脚本的 OUTPUT_ZARR_DIR 一致
ZARR_PATH = 'opt13b_local_similarity_results' 
PARTITIONS = [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
DISTANCES = [1, 2]
OUTPUT_DIR = 'analysis_results' # 创建一个新目录来存放所有图片

# --- 主分析函数（已修改为循环处理所有层） ---
def analyze_all_layers_and_plot():
    """
    主分析和绘图函数。
    循环遍历所有层，计算每层所有轨迹的平均相似度，并生成报告和图表。
    """
    print(f"--- 开始分析局部注意力相似度 (所有层平均) ---")
    print(f"数据源: {ZARR_PATH}")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    try:
        zarr_root = zarr.open(ZARR_PATH, mode='r')
    except Exception as e:
        print(f"错误: 无法打开 Zarr 目录 '{ZARR_PATH}'.")
        print(f"具体错误: {e}")
        return

    # 1. 一次性加载所有数据
    # 形状: (num_trajectories, num_layers, num_partitions, num_distances)
    try:
        spearman_data = zarr_root['spearman_similarity'][:]
        cosine_data = zarr_root['cosine_similarity'][:]
    except KeyError as e:
        print(f"错误: 在Zarr数据中未找到数据集: {e}。请确保数据生成脚本已成功运行。")
        return

    print("数据加载完成。")
    print(f"数据维度 (轨迹数, 层数, 分区数, 距离数): {spearman_data.shape}")
    # --- 插入诊断代码 ---
    print("\n--- 开始诊断 Layer 0, Partition 0-512, Dist t+1 ---")
    layer_to_diag = 0
    partition_to_diag = 0
    dist_to_diag = 0 # 对应 t+1

    # 提取这一个数据点上所有轨迹的值
    spearman_values_for_one_point = spearman_data[:, layer_to_diag, partition_to_diag, dist_to_diag]
    
    print(f"共有 {len(spearman_values_for_one_point)} 个值。")
    print(f"平均值: {np.mean(spearman_values_for_one_point):.4f}") # 这应该就是您看到的 0.1237
    print(f"标准差: {np.std(spearman_values_for_one_point):.4f}")  # 标准差应该会很大
    print(f"最小值: {np.min(spearman_values_for_one_point):.4f}")  # 可能是负数
    print(f"最大值: {np.max(spearman_values_for_one_point):.4f}")  # 可能是正数
    
    # 绘制直方图来查看分布
    plt.figure(figsize=(10, 6))
    sns.histplot(spearman_values_for_one_point, kde=True, bins=30)
    plt.title(f"Distribution of Spearman Corr. for Layer {layer_to_diag}, Partition 0-512, Dist t+1")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.grid(True)
    diag_filename = "spearman_distribution_diagnosis.png"
    plt.savefig(diag_filename)
    plt.close()
    print(f"诊断直方图已保存至: {diag_filename}\n")
    # --- 诊断代码结束 ---
    num_trajectories, num_layers, _, _ = spearman_data.shape
    print(f"发现 {num_trajectories} 条轨迹, {num_layers} 层。将开始逐层分析...")

    # 2. 外层循环，遍历每一层
    for layer_index in range(num_layers):
        print(f"\n{'='*20} 分析 Layer {layer_index} {'='*20}")

        # 3. 提取当前层所有轨迹的数据
        # 形状变为: (num_trajectories, num_partitions, num_distances)
        spearman_all_traj_layer = spearman_data[:, layer_index, :, :]
        cosine_all_traj_layer = cosine_data[:, layer_index, :, :]

        # 4. 沿着轨迹维度(axis=0)计算平均值
        # 形状变为: (num_partitions, num_distances)，这正是后续函数需要的格式
        spearman_avg_layer_data = np.mean(spearman_all_traj_layer, axis=0)
        cosine_avg_layer_data = np.mean(cosine_all_traj_layer, axis=0)

        # --- 打印当前层的平均结果 ---
        print(f"--- 平均相似度结果 (Layer {layer_index}, Average over {num_trajectories} trajectories) ---")
        partition_labels = [f'{s}-{e}' for s, e in PARTITIONS]
        header = f"{'Partition':<12} | {'Dist':<5} | {'Spearman Sim.':<15} | {'Cosine Sim.':<15}"
        print(header)
        print("-" * len(header))
        for i_part, label in enumerate(partition_labels):
            for i_dist, dist in enumerate(DISTANCES):
                spearman_val = spearman_avg_layer_data[i_part, i_dist]
                cosine_val = cosine_avg_layer_data[i_part, i_dist]
                print(f"{label:<12} | t+{dist:<4} | {spearman_val:<15.4f} | {cosine_val:<15.4f}")

        # --- 为当前层绘制分组条形图 ---
        plot_grouped_bar_chart(
            spearman_avg_layer_data,
            'Spearman Rank Correlation',
            layer_index,
            os.path.join(OUTPUT_DIR, f'avg_spearman_similarity_layer_{layer_index}.png')
        )
        plot_grouped_bar_chart(
            cosine_avg_layer_data,
            'Cosine Similarity',
            layer_index,
            os.path.join(OUTPUT_DIR, f'avg_cosine_similarity_layer_{layer_index}.png')
        )

    print("\n--- 所有层分析完成 ---")


# --- 绘图函数（已修改以接收层索引并更新标题） ---
def plot_grouped_bar_chart(data, metric_name, layer_index, filename):
    """
    为给定的相似度数据绘制分组条形图。
    data: 一个 (num_partitions, num_distances) 的 numpy 数组
    layer_index: 当前层的索引，用于图表标题
    filename: 保存图表的文件名
    """
    sns.set_style("whitegrid")
    
    n_partitions = data.shape[0]
    n_distances = data.shape[1]
    
    partition_labels = [f'{s}-{e}' for s, e in PARTITIONS]
    
    x = np.arange(n_partitions)  # 分区的位置
    width = 0.35  # 条形的宽度

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制每个距离的条形
    for i in range(n_distances):
        dist = DISTANCES[i]
        offset = width * (i - (n_distances - 1) / 2)
        rects = ax.bar(x + offset, data[:, i], width, label=f'Similarity(t, t+{dist})')
        ax.bar_label(rects, padding=3, fmt='%.3f')

    ax.set_ylabel(metric_name)
    # 更新标题，明确指出是所有轨迹的平均值
    ax.set_title(f'Average Attention {metric_name} between Nearby Tokens\n(Average over all trajectories, Layer {layer_index})')
    ax.set_xticks(x)
    ax.set_xticklabels(partition_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    plt.savefig(filename)
    # print(f"已保存图表: {filename}") # 在循环中打印太多信息，可以注释掉
    plt.close(fig)


if __name__ == '__main__':
    analyze_all_layers_and_plot()