# verify_zarr.py
import zarr
import numpy as np

ZARR_PATH = 'opt13b_dynamic_trajectories'
CHECKPOINT = 512
TRAJECTORY_IDX = 0
LAYER_IDX = 20

try:
    zarr_root = zarr.open(ZARR_PATH, mode='r')
    dataset_name = f'attention_scores_at_{CHECKPOINT}'
    scores = zarr_root[dataset_name][TRAJECTORY_IDX, LAYER_IDX, :]

    print(f"--- 验证 Zarr 数据 ---")
    print(f"加载数据: {dataset_name}, Trajectory {TRAJECTORY_IDX}, Layer {LAYER_IDX}")
    print(f"数据形状: {scores.shape}")
    print(f"数据类型: {scores.dtype}")
    print(f"最小值: {np.min(scores)}")
    print(f"最大值: {np.max(scores)}")
    print(f"平均值: {np.mean(scores)}")
    
    if np.all(scores == 0):
        print("\n[结论] 数据验证失败：数组中的所有值都为 0。")
    else:
        print("\n[结论] 数据似乎正常。")

except Exception as e:
    print(f"加载数据时出错: {e}")