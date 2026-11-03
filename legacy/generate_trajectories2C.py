# generate_trajectories2.py
import os
import torch
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from utils2 import AttentionStateCatcher
import gc
import traceback

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
OUTPUT_ZARR_DIR = 'opt13b_trajectories_scores_multi_point' # 建议使用新目录
CHUNK_SIZE = 2048 # CHUNK_SIZE 仍然代表我们一次从文本中读取的最大长度
TARGET_TRAJECTORIES = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 新增：多点采样配置 ---
START_LEN = 128
STEP = 16
# SAMPLING_POINTS 将包含 [128, 144, 160, ..., 2048]
SAMPLING_POINTS = list(range(START_LEN, CHUNK_SIZE + 1, STEP))

def create_or_open_zarr(path, num_trajectories):
    """创建或打开Zarr数据存储，使用正确的形状。"""
    if not os.path.exists(path):
        os.makedirs(path)

    root = zarr.open_group(path, mode='a')
    
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    NUM_LAYERS = config.num_hidden_layers
    D_MODEL = config.hidden_size

    # Zarr数据集的形状保持不变，因为我们会将数据填充到 CHUNK_SIZE
    if 'embeddings' not in root:
        root.create_dataset('embeddings', shape=(num_trajectories, CHUNK_SIZE, D_MODEL), 
                            chunks=(1, CHUNK_SIZE, D_MODEL), dtype='f2')
    
    if 'attention_scores' not in root:
        root.create_dataset('attention_scores', shape=(num_trajectories, NUM_LAYERS, CHUNK_SIZE),
                            chunks=(1, NUM_LAYERS, CHUNK_SIZE), dtype='f2')

    if 'queries' in root: del root['queries']
    if 'keys' in root: del root['keys']
        
    return root

def calculate_target_scores(queries, keys):
    """
    根据 utils.py 提供的4D张量，从 Q 和 K 计算目标注意力分数。
    此函数无需修改，因为它能自动处理不同长度的序列。
    Args:
        queries (torch.Tensor): Shape (Num_Layers, Seq_Len, Num_Heads, Head_Dim)
        keys (torch.Tensor): Shape (Num_Layers, Seq_Len, Num_Heads, Head_Dim)
    Returns:
        torch.Tensor: Shape (Num_Layers, Seq_Len)
    """
    true_queries_last_token = queries[:, -1:, :, :]
    true_query_avg_heads = true_queries_last_token.mean(dim=2)
    keys_avg_heads = keys.mean(dim=2)
    true_scores = torch.einsum('lnd,lsd->lns', true_query_avg_heads, keys_avg_heads)
    return true_scores.squeeze(1)

def main():
    if DEVICE == 'cpu':
        print("Warning: Running on CPU. This will be extremely slow.")
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    # 解决潜在的 vocab size 不匹配问题
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    print("Preparing Zarr data store...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, TARGET_TRAJECTORIES)
    
    start_index = 0
    if 'embeddings' in zarr_root:
        while start_index < TARGET_TRAJECTORIES and zarr_root['embeddings'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
    
    print(f"Found {start_index} existing trajectories. Starting from index {start_index}.")
    print(f"Multi-point sampling enabled: Start={START_LEN}, Step={STEP}, End={CHUNK_SIZE}.")
    print(f"Total sampling points per chunk: {len(SAMPLING_POINTS)}")


    catcher = AttentionStateCatcher(model)
    
    all_tokens = []
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        print("Tokenizing input text...")
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    print(f"Text tokenized. Found {num_chunks} chunks of size {CHUNK_SIZE}.")

    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="Generating Trajectories")

    chunk_idx = start_index // len(SAMPLING_POINTS) # 估算一个起始的 chunk_idx
    
    # 外层循环：遍历文本块
    while start_index < TARGET_TRAJECTORIES and chunk_idx < num_chunks:
        
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]

        # ==================== 修改核心：内层循环，模拟自回归过程 ====================
        try:
            # 内层循环：遍历当前文本块中的每一个采样点
            for current_len in SAMPLING_POINTS:
                # 如果已经采集够了，就提前结束所有循环
                if start_index >= TARGET_TRAJECTORIES:
                    break

                # 1. 准备当前长度的输入
                input_ids = chunk_tokens[:current_len].unsqueeze(0).to(DEVICE)

                # 检查 input_ids 中的 token 是否在 embedding 范围内
                # 这是一个预防措施，应对之前提到的 vocab size 问题
                max_token_id = input_ids.max().item()
                if max_token_id >= model.get_input_embeddings().weight.shape[0]:
                    print(f"Skipping point at len {current_len} in chunk {chunk_idx} due to out-of-range token_id: {max_token_id}")
                    continue

                # 2. 运行模型并捕获状态
                catcher.reset()
                catcher.register_hooks()
                with torch.no_grad():
                    model(input_ids)
                catcher.remove_hooks()
                
                captured_data = catcher.get_captured_data()
                
                if captured_data:
                    # 3. 计算注意力分数
                    # 此时 captured_data 中的张量序列长度都是 current_len
                    queries = captured_data['queries'].to(DEVICE)
                    keys = captured_data['keys'].to(DEVICE)
                    target_scores = calculate_target_scores(queries, keys) # Shape: (Layers, current_len)

                    # 4. 数据填充 (Padding) 以匹配Zarr的固定形状 (2048)
                    # a. 填充 Embeddings
                    embeddings_at_len = captured_data['embeddings'].squeeze(0) # Shape: (current_len, D_MODEL)
                    padded_embeddings = torch.zeros(CHUNK_SIZE, embeddings_at_len.shape[-1], dtype=torch.float16, device='cpu')
                    padded_embeddings[:current_len, :] = embeddings_at_len.to(torch.float16).cpu()

                    # b. 填充 Attention Scores
                    padded_scores = torch.zeros(target_scores.shape[0], CHUNK_SIZE, dtype=torch.float16, device='cpu')
                    padded_scores[:, :current_len] = target_scores.to(torch.float16).cpu()
                    
                    # 5. 存储填充后的数据
                    zarr_root['embeddings'][start_index] = padded_embeddings.numpy()
                    zarr_root['attention_scores'][start_index] = padded_scores.numpy()
                    
                    zarr_root['embeddings'].attrs[f'row_{start_index}_written'] = True
                    
                    # 6. 更新计数器和进度条
                    start_index += 1
                    pbar.update(1)

                # 7. 精细的内存管理：在每个采样点后清理，防止累积
                if 'captured_data' in locals() and captured_data: del captured_data
                if 'queries' in locals(): del queries
                if 'keys' in locals(): del keys
                if 'target_scores' in locals(): del target_scores
                if 'padded_embeddings' in locals(): del padded_embeddings
                if 'padded_scores' in locals(): del padded_scores
                gc.collect()
                torch.cuda.empty_cache()
            
            # 内层循环结束后，如果目标已达成，也跳出外层循环
            if start_index >= TARGET_TRAJECTORIES:
                break
        
        except Exception as e:
            print(f"\n---!!! An error occurred while processing chunk {chunk_idx}. Skipping this chunk. !!!---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            traceback.print_exc()
            # 确保钩子在出错时被移除，防止影响下一个chunk
            catcher.remove_hooks()
            # 强制清理CUDA内存，尝试恢复上下文
            gc.collect()
            torch.cuda.empty_cache()
        
        chunk_idx += 1
        # ============================ 修改结束 ============================

    pbar.close()
    print("\nData generation finished.")
    print(f"Total trajectories saved: {start_index}")

if __name__ == '__main__':
    main()