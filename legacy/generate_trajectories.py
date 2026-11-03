# generate_trajectories.py
import os
import torch
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import AttentionStateCatcher
import gc


# 设置代理 (如果需要)
os.environ['http_proxy'] = "http://192.168.143.85:7890"
os.environ['https_proxy'] = "http://192.168.143.85:7890"
os.environ['all_proxy'] = "socks5://192.168.143.85:7890"
# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/" # 通常和模型路径相同
INPUT_TEXT_FILE = 'c4_subset_50mb.txt'
OUTPUT_ZARR_DIR = 'opt13b_trajectories_2t'
CHUNK_SIZE = 2000  # 每个文本块的token数
TARGET_TRAJECTORIES = 600 # 目标生成10万个轨迹文件
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_or_open_zarr(path, num_trajectories):
    """创建或打开Zarr数据存储，使用正确的形状。"""
    if not os.path.exists(path):
        os.makedirs(path)

    root = zarr.open_group(path, mode='a')
    
    # 获取模型配置
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # OPT-13B specific dimensions
    NUM_LAYERS = config.num_hidden_layers # 40
    D_MODEL = config.hidden_size        # 5120
    NUM_HEADS = config.num_attention_heads # 40
    HEAD_DIM = D_MODEL // NUM_HEADS      # 128

    # --- ！！！检查并确认这里的形状！！！ ---
    if 'embeddings' not in root:
        # Shape: (Seq_Len, D_model)
        root.create_dataset('embeddings', shape=(num_trajectories, CHUNK_SIZE, D_MODEL), 
                            chunks=(1, CHUNK_SIZE, D_MODEL), dtype='f2')
    if 'queries' not in root:
        # Shape: (Num_Layers, Seq_Len, Num_Heads, Head_Dim)
        root.create_dataset('queries', shape=(num_trajectories, NUM_LAYERS, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), 
                            chunks=(1, 1, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), dtype='f2')
    if 'keys' not in root:
         # Shape: (Num_Layers, Seq_Len, Num_Heads, Head_Dim)
        root.create_dataset('keys', shape=(num_trajectories, NUM_LAYERS, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), 
                            chunks=(1, 1, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), dtype='f2')
    return root

def main():
    if DEVICE == 'cpu':
        print("Warning: Running on CPU. This will be extremely slow.")
    MODEL_ID = "facebook/opt-13b" # 使用Hugging Face的模型ID
    print("Loading model and tokenizer...")
    # 使用float16以节省内存
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    # --- ！！！在这里加入修复代码 ！！！---
    # 调整模型嵌入层大小以匹配分词器的词汇表大小
    # 这是解决词汇表不匹配问题的标准做法
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    print(f"Resized model token embeddings to match tokenizer vocabulary size: {new_vocab_size}")
    # --- 修复代码结束 ---
    model.eval()
    # --- ！！！在这里加入最终诊断代码 ！！！---
    print("-" * 50)
    print("--- DIAGNOSTICS ---")
    model_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = tokenizer.vocab_size
    print(f"Model's configured vocabulary size (from local config.json): {model_vocab_size}")
    print(f"Tokenizer's vocabulary size (from official Hub version):  {tokenizer_vocab_size}")

    if model_vocab_size != tokenizer_vocab_size:
        print("\n>>> MISMATCH DETECTED! This is the root cause of the error. <<<")
        print("The model expects a smaller vocabulary than the tokenizer provides.")
    print("-" * 50)
    # --- 诊断结束 ---
    print("Preparing Zarr data store...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, TARGET_TRAJECTORIES)
    
    
    start_index = 0
    # 查找下一个可写的索引
    while start_index < TARGET_TRAJECTORIES and zarr_root['embeddings'][start_index].sum() != 0:
        start_index += 1
    
    print(f"Found {start_index} existing trajectories. Starting from index {start_index}.")

    catcher = AttentionStateCatcher(model)
    
    # 读取文本文件并处理
    all_tokens = []
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        print("Tokenizing input text...")
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    print(f"Text tokenized. Found {num_chunks} chunks of size {CHUNK_SIZE}.")

    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="Generating Trajectories")

    chunk_idx = start_index # 从合适的chunk开始
    while start_index < TARGET_TRAJECTORIES and chunk_idx < num_chunks:
        
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
        input_ids = chunk_tokens.unsqueeze(0).to(DEVICE)

        catcher.reset()
        catcher.register_hooks()

        try:
            with torch.no_grad():
                # 运行模型一次，这将触发所有的hooks
                model(input_ids)
            
            # 移除钩子
            catcher.remove_hooks()
            
            # 获取捕获的数据
            captured_data = catcher.get_captured_data()
            
            if captured_data:
                # 存入Zarr数组
                # 将数据转换为float16以节省空间
                zarr_root['embeddings'][start_index] = captured_data['embeddings'].to(torch.float16).numpy()
                zarr_root['queries'][start_index] = captured_data['queries'].to(torch.float16).numpy()
                zarr_root['keys'][start_index] = captured_data['keys'].to(torch.float16).numpy()

                start_index += 1
                pbar.update(1)

        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
            catcher.remove_hooks() # 确保即使出错也移除钩子
        
        chunk_idx += 1
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    print("\nData generation finished.")
    print(f"Total trajectories saved: {start_index}")

if __name__ == '__main__':
    main()