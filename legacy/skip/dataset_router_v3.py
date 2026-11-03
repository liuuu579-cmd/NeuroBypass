# dataset_router.py
import torch
import zarr
from torch.utils.data import Dataset

class RouterDataset(Dataset):
    def __init__(self, zarr_path: str):
        self.root = zarr.open(zarr_path, mode='r')
        
        # 【修改点 1】：把检查的 key 改为新特征
        required_keys = ['layer1_states', 'greedy_masks']
        for key in required_keys:
            if key not in self.root:
                raise ValueError(f"Zarr store at {zarr_path} is missing required dataset '{key}'")
        
        # 【修改点 2】：根据新特征获取样本总数
        self.num_trajectories = self.root['layer1_states'].shape[0]
    
    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        # 【修改点 3】：读取新的 Layer 1 隐藏状态！
        # 变量名保持叫 embeddings，这样外部代码完全无感
        embeddings = torch.from_numpy(self.root['layer1_states'][idx].astype('float32'))
        
        # [NUM_LAYERS] -> 全是 0 和 1
        mask = torch.from_numpy(self.root['greedy_masks'][idx].astype('float32')) 
        return embeddings, mask