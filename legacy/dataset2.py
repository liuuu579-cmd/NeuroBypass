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
        
        ### <<< 修改: 验证新的数据集 'attention_scores'
        required_keys = ['embeddings', 'attention_scores']
        for key in required_keys:
            if key not in self.root:
                raise ValueError(f"Zarr store at {zarr_path} is missing required dataset '{key}'")
        
        self.num_trajectories = self.root['embeddings'].shape[0]
    
    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        # Zarr返回的是Numpy数组，我们需要转换为Tensor
        # .astype('float32') 是因为torch.float16在CPU上计算支持不佳, 转为float32更安全
        embeddings = torch.from_numpy(self.root['embeddings'][idx].astype('float32'))
        
        ### <<< 修改: 加载 attention_scores 而不是 QK
        attention_scores = torch.from_numpy(self.root['attention_scores'][idx].astype('float32'))
        
        return embeddings, attention_scores