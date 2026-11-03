# dataset.py
import torch
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, zarr_path: str):
        """
        Args:
            zarr_path (str): Path to the root Zarr directory.
        """
        self.root = zarr.open(zarr_path, mode='r')
        
        # 验证数据是否存在
        required_keys = ['embeddings', 'queries', 'keys']
        for key in required_keys:
            if key not in self.root:
                raise ValueError(f"Zarr store at {zarr_path} is missing required dataset '{key}'")
        
        self.num_trajectories = self.root['embeddings'].shape[0]

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        # Zarr支持高效的切片读取，这里我们一次读取一个轨迹
        # Zarr返回的是Numpy数组，我们需要转换为Tensor
        # .astype('float32') 是因为torch.float16在CPU上计算支持不佳, 转为float32更安全
        embeddings = torch.from_numpy(self.root['embeddings'][idx].astype('float32'))
        queries = torch.from_numpy(self.root['queries'][idx].astype('float32'))
        keys = torch.from_numpy(self.root['keys'][idx].astype('float32'))
        
        return embeddings, queries, keys