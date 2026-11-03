# dataset_router.py
import torch
import zarr
from torch.utils.data import Dataset

class RouterDataset(Dataset):
    def __init__(self, zarr_path: str):
        self.root = zarr.open(zarr_path, mode='r')
        
        required_keys = ['embeddings', 'greedy_masks']
        for key in required_keys:
            if key not in self.root:
                raise ValueError(f"Zarr store at {zarr_path} is missing required dataset '{key}'")
        
        self.num_trajectories = self.root['embeddings'].shape[0]
    
    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        # [CHUNK_SIZE, D_MODEL]
        embeddings = torch.from_numpy(self.root['embeddings'][idx].astype('float32'))
        # [NUM_LAYERS] -> 全是 0 和 1
        mask = torch.from_numpy(self.root['greedy_masks'][idx].astype('float32')) 
        return embeddings, mask