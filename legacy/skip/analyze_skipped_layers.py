import os
import zarr
import numpy as np
from collections import Counter

# 配置参数
NUM_GPUS = 8
ZARR_DIR_PREFIX = "/home/liushenyu/test/opt13b_greedy_routing_labels_part_"

def main():
    all_subsets = []
    individual_layer_counts = Counter()
    total_samples = 0

    print("开始读取和合并多卡 Zarr 数据...")

    for gpu_id in range(NUM_GPUS):
        zarr_path = f"{ZARR_DIR_PREFIX}{gpu_id}"
        
        if not os.path.exists(zarr_path):
            print(f"[警告] 找不到路径: {zarr_path}，跳过。")
            continue

        try:
            root = zarr.open_group(zarr_path, mode='r')
            masks = root['greedy_masks'][:]
            
            # 依靠原代码写入的 attrs 来判断哪些行是真实有效写入的
            attrs = root['embeddings'].attrs
            # 提取所有已被标记为 written 的索引
            written_indices = [
                int(k.split('_')[1]) for k in attrs.keys() 
                if k.startswith('row_') and k.endswith('_written') and attrs[k] == True
            ]
            
            if not written_indices:
                continue
                
            max_valid_idx = max(written_indices) + 1
            
            for i in range(max_valid_idx):
                mask = masks[i]
                
                # 原始代码中跳过的层被标记为 0.0，保留的层为 1.0
                # 使用 < 0.5 来避免浮点数精度问题
                skipped_layers = np.where(mask < 0.5)[0]
                
                # 确保确实跳过了层（正常应该是6层）
                if len(skipped_layers) > 0:
                    # 转为 tuple 以便用于字典或 Counter 的 key
                    subset_tuple = tuple(skipped_layers.tolist())
                    all_subsets.append(subset_tuple)
                    
                    # 统计单层的命中率
                    for layer in skipped_layers:
                        individual_layer_counts[layer] += 1
                        
            print(f"成功读取 GPU {gpu_id} 的数据，有效样本数: {max_valid_idx}")
            
        except Exception as e:
            print(f"[错误] 读取 GPU {gpu_id} 数据时出错: {e}")

    total_samples = len(all_subsets)
    if total_samples == 0:
        print("没有读取到任何有效数据，请检查 Zarr 文件夹。")
        return

    print(f"\n================ 统计分析完成 ================")
    print(f"总计有效数据条数: {total_samples}")

    # 1. 统计最常见的 6 层组合
    subset_counter = Counter(all_subsets)
    print("\n[1] 最常被跳过的【6层组合】TOP 20:")
    print("-" * 60)
    for subset, count in subset_counter.most_common(20):
        percentage = (count / total_samples) * 100
        print(f"层组合 {subset}: 出现 {count:4d} 次 ({percentage:5.2f}%)")

    # 2. 统计所有可能的组合种类数量
    unique_combinations = len(subset_counter)
    print(f"\n在这 {total_samples} 条数据中，一共出现了 {unique_combinations} 种不同的 6层组合。")

    # 3. 统计每一层被跳过的整体概率 (直方图可视化)
    print("\n[2] 每一层的独立被跳过频率分布:")
    print("-" * 60)
    
    # 获取模型的总层数 (OPT-13B 通常是 40)
    if individual_layer_counts:
        num_layers = max(individual_layer_counts.keys()) + 1
        
        for layer in range(num_layers):
            count = individual_layer_counts.get(layer, 0)
            percentage = (count / total_samples) * 100
            
            # 画一个简单的 ASCII 进度条来直观展示
            bar_length = int(percentage / 2)  # 每 2% 画一个方块
            bar = "█" * bar_length
            
            print(f"第 {layer:2d} 层: 出现 {count:5d} 次 | 概率: {percentage:5.2f}% | {bar}")

if __name__ == '__main__':
    main()