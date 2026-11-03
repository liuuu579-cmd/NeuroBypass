import torch
from torch.utils.data import Dataset
import zarr
import numpy as np

class DynamicRouterDataset(Dataset):
    def __init__(self, zarr_path, max_skip_steps=16, num_layers=40, kl_threshold=0.5):
        self.root = zarr.open_group(zarr_path, mode='r')
        self.input_ids = self.root['input_ids']
        self.seq_lens = self.root['seq_lens']
        self.ordered_skips = self.root['ordered_skips']
        self.kl_divergences = self.root['kl_divergences']
        
        self.length = len(self.seq_lens)
        self.max_skip_steps = max_skip_steps
        self.num_layers = num_layers
        self.kl_threshold = kl_threshold

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 取出 Token IDs 及有效序列长度
        seq_len = int(self.seq_lens[idx])
        ids = torch.tensor(self.input_ids[idx][:seq_len], dtype=torch.long)
        
        skips_trajectory = self.ordered_skips[idx]     # shape: (16,)
        kl_trajectory = self.kl_divergences[idx]        # shape: (16,)
        
        # 2. 确定该样本在本次 step 中的 target_k (剪枝深度)
        # 策略：50% 概率由 KL 阈值自适应截断（学习按难度裁剪），50% 概率均匀随机采样（保证 Router 对任意 k 的泛化）
        if np.random.rand() > 0.5:
            # 找到超出容忍误差的临界点
            valid_indices = np.where(kl_trajectory <= self.kl_threshold)[0]
            if len(valid_indices) == 0:
                target_k = 1 # 极难样本，保底只剪 1 层
            else:
                target_k = int(valid_indices[-1] + 1)
        else:
            target_k = int(np.random.randint(1, self.max_skip_steps + 1))
            
        target_k = min(max(target_k, 1), self.max_skip_steps)
        
        # 3. 动态构建该剪枝深度下的 Ground Truth Binary Mask
        # 1.0 表示保留，0.0 表示跳过
        mask = torch.ones(self.num_layers, dtype=torch.float32)
        actual_skipped_layers = skips_trajectory[:target_k]
        mask[actual_skipped_layers] = 0.0
        
        return ids, target_k, mask

def pad_dynamic_collate_fn(batch):
    """自适应 Padding 规避大内存浪费"""
    input_ids_list, target_k_list, mask_list = zip(*batch)
    
    max_len = max([len(x) for x in input_ids_list])
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool) # True 为 padding
    
    for i, ids in enumerate(input_ids_list):
        padded_ids[i, :len(ids)] = ids
        padding_mask[i, :len(ids)] = False
        
    target_k = torch.tensor(target_k_list, dtype=torch.long)
    masks = torch.stack(mask_list, dim=0)
    
    return padded_ids, padding_mask, target_k, masks