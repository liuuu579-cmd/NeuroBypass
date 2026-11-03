# generate_trajectories2.py
import os
import torch
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils2 import AttentionStateCatcher
import gc

# 设置代理 (如果需要)
# os.environ['http_proxy'] = "http://192.168.143.85:7890"
# os.environ['https_proxy'] = "http://192.168.143.85:7890"
# os.environ['all_proxy'] = "socks5://192.168.143.85:7890"

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
OUTPUT_ZARR_DIR = 'opt13b_trajectories_scores_2t'
CHUNK_SIZE = 2048
TARGET_TRAJECTORIES = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_or_open_zarr(path, num_trajectories):
    """创建或打开Zarr数据存储，使用正确的形状。"""
    if not os.path.exists(path):
        os.makedirs(path)

    root = zarr.open_group(path, mode='a')
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    NUM_LAYERS = config.num_hidden_layers
    D_MODEL = config.hidden_size

    if 'embeddings' not in root:
        root.create_dataset('embeddings', shape=(num_trajectories, CHUNK_SIZE, D_MODEL), 
                            chunks=(1, CHUNK_SIZE, D_MODEL), dtype='f2')
    
    if 'attention_scores' not in root:
        root.create_dataset('attention_scores', shape=(num_trajectories, NUM_LAYERS, CHUNK_SIZE),
                            chunks=(1, NUM_LAYERS, CHUNK_SIZE), dtype='f2')

    if 'queries' in root: del root['queries']
    if 'keys' in root: del root['keys']
        
    return root

### <<< 修改开始: 修正此函数以处理4D张量
def calculate_target_scores(queries, keys):
    """
    根据 utils.py 提供的4D张量，从 Q 和 K 计算目标注意力分数。
    Args:
        queries (torch.Tensor): Shape (Num_Layers, Seq_Len, Num_Heads, Head_Dim)
        keys (torch.Tensor): Shape (Num_Layers, Seq_Len, Num_Heads, Head_Dim)
    Returns:
        torch.Tensor: Shape (Num_Layers, Seq_Len)
    """
    # 1. 只取最后一个 token 的 query
    # queries shape: (L, S, H, D_h) -> (40, 2048, 40, 128)
    # true_queries_last_token shape: (L, 1, H, D_h) -> (40, 1, 40, 128)
    true_queries_last_token = queries[:, -1:, :, :]

    # 2. 在 Head 维度 (dim=2) 上取平均
    # true_query_avg_heads shape: (L, 1, D_h) -> (40, 1, 128)
    true_query_avg_heads = true_queries_last_token.mean(dim=2)
    # keys_avg_heads shape: (L, S, D_h) -> (40, 2048, 128)
    keys_avg_heads = keys.mean(dim=2)
    
    # 3. 计算注意力分数 (einsum)
    # l=layer, n=query_seq (size 1), s=key_seq, d=head_dim
    # Expression: lnd,lsd->lns
    # (40, 1, 128) @ (40, 2048, 128) -> (40, 1, 2048)
    true_scores = torch.einsum('lnd,lsd->lns', true_query_avg_heads, keys_avg_heads)
    
    # 4. 移除多余的 query_seq 维度 (dim=1)
    # Shape: (Num_Layers, Seq_Len)
    return true_scores.squeeze(1)
### <<< 修改结束

def main():
    if DEVICE == 'cpu':
        print("Warning: Running on CPU. This will be extremely slow.")
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    print("Preparing Zarr data store...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, TARGET_TRAJECTORIES)
    
    start_index = 0
    # A more robust check for existing data
    if 'embeddings' in zarr_root:
        while start_index < TARGET_TRAJECTORIES and zarr_root['embeddings'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
    
    print(f"Found {start_index} existing trajectories. Starting from index {start_index}.")

    catcher = AttentionStateCatcher(model)
    
    all_tokens = []
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        print("Tokenizing input text...")
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    print(f"Text tokenized. Found {num_chunks} chunks of size {CHUNK_SIZE}.")

    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="Generating Trajectories")

    chunk_idx = start_index
    while start_index < TARGET_TRAJECTORIES and chunk_idx < num_chunks:
        
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
        input_ids = chunk_tokens.unsqueeze(0).to(DEVICE)

        catcher.reset()
        catcher.register_hooks()

        try:
            with torch.no_grad():
                model(input_ids)
            
            catcher.remove_hooks()
            captured_data = catcher.get_captured_data()
            
            if captured_data:
                queries = captured_data['queries'].to(DEVICE) # Move to GPU for calculation
                keys = captured_data['keys'].to(DEVICE)
                
                target_scores = calculate_target_scores(queries, keys)

                zarr_root['embeddings'][start_index] = captured_data['embeddings'].to(torch.float16).cpu().numpy()
                zarr_root['attention_scores'][start_index] = target_scores.to(torch.float16).cpu().numpy()
                
                # Mark the row as written to handle restarts correctly
                zarr_root['embeddings'].attrs[f'row_{start_index}_written'] = True
                
                start_index += 1
                pbar.update(1)

        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for better debugging
            catcher.remove_hooks()
        
        chunk_idx += 1
        
        # Explicitly delete tensors to free CUDA memory
        if 'captured_data' in locals() and captured_data:
            del captured_data
        if 'queries' in locals():
            del queries
        if 'keys' in locals():
            del keys
        if 'target_scores' in locals():
            del target_scores
            
        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    print("\nData generation finished.")
    print(f"Total trajectories saved: {start_index}")

if __name__ == '__main__':
    main()