import os
import torch
import zarr
import argparse
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, default=0, help="Logical worker ID (e.g., 0, 1, 2, 3)")
parser.add_argument("--world_size", type=int, default=4, help="Total number of active workers")
parser.add_argument("--gpu_id", type=int, default=0, help="The actual physical GPU ID to use (e.g., 4, 5, 6, 7)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
args = parser.parse_args()

MODEL_PATH = "/home/liushenyu/opt-13b/"
ZARR_BASE_NAME = 'opt13b_greedy_routing_labels_part_'
NUM_TOTAL_PARTS = 8  # 我们总共有 8 个 Zarr 文件夹
DEVICE = f'cuda:{args.gpu_id}'

def main():
    print(f"[Worker {args.worker_id} on GPU {args.gpu_id}] Loading OPT-13B to extract Layer 1 features...")
    
    # 1. 正常加载模型 (只加载一次！)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    
    # 【核心黑魔法】：物理删除多余的层，只保留前 2 层 (Layer 0 和 Layer 1)
    model.model.decoder.layers = model.model.decoder.layers[:2]
    model = model.to(DEVICE)
    model.eval()

    print(f"[Worker {args.worker_id}] Model trimmed. Starting tasks...")

    # 2. 遍历所有的 8 个文件夹，如果取余属于当前 worker，则处理
    for part_id in range(NUM_TOTAL_PARTS):
        if part_id % args.world_size != args.worker_id:
            continue  # 不是分配给我的任务，跳过
            
        zarr_dir = f"{ZARR_BASE_NAME}{part_id}"
        print(f"\n[Worker {args.worker_id}] ===> Processing {zarr_dir}")
        
        if not os.path.exists(zarr_dir):
            print(f"[Worker {args.worker_id}] Zarr directory {zarr_dir} not found. Skipping.")
            continue

        # 打开当前的 Zarr
        root = zarr.open_group(zarr_dir, mode='a')
        
        if 'embeddings' not in root:
            print(f"[Worker {args.worker_id}] No 'embeddings' found in {zarr_dir}. Skipping.")
            continue
            
        embeddings_ds = root['embeddings']
        total_samples = embeddings_ds.shape[0]
        chunk_size = embeddings_ds.shape[1]
        d_model = embeddings_ds.shape[2]
        
        # 创建新的 dataset 用于存储深层特征
        if 'hidden_states_layer1' not in root:
            root.create_dataset('hidden_states_layer1', 
                                shape=(total_samples, chunk_size, d_model),
                                chunks=(1, chunk_size, d_model), 
                                dtype='f2') # float16
        
        # 寻找断点续传的位置
        start_idx = 0
        while start_idx < total_samples and root['hidden_states_layer1'].attrs.get(f'row_{start_idx}_h1_written', False):
            start_idx += 1

        if start_idx >= total_samples:
            print(f"[Worker {args.worker_id}] {zarr_dir} already fully upgraded. Moving to next.")
            continue

        pbar = tqdm(total=total_samples, initial=start_idx, 
                    desc=f"Worker {args.worker_id} -> Part {part_id}", 
                    position=args.worker_id)

        # 批量处理当前文件夹的数据
        for i in range(start_idx, total_samples, args.batch_size):
            end_idx = min(i + args.batch_size, total_samples)
            
            # 从 Zarr 读取旧的 embeddings (shape: [B, S, D])
            batch_embeddings = torch.tensor(embeddings_ds[i:end_idx]).to(DEVICE)
            
            # 构造 attention_mask
            attention_mask = (batch_embeddings.sum(dim=-1) != 0).long()
            
            with torch.no_grad():
                outputs = model(
                    inputs_embeds=batch_embeddings, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                layer1_features = outputs.hidden_states[2]
                
            # 写入 Zarr
            root['hidden_states_layer1'][i:end_idx] = layer1_features.cpu().numpy()
            
            # 记录写入状态（防断电）
            for row in range(i, end_idx):
                root['hidden_states_layer1'].attrs[f'row_{row}_h1_written'] = True
                pbar.update(1)

        pbar.close()

    print(f"\n[Worker {args.worker_id}] All assigned parts completed successfully!")

if __name__ == '__main__':
    main()