# generate_trajectories_multi_gpu_fixed.py
import os
import torch
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils2 import AttentionStateCatcher
import gc

import torch.multiprocessing as mp
import torch.distributed as dist
from filelock import FileLock
import datetime # 引入datetime来增加超时时间

# --- 配置 ---
CONFIG = {
    "MODEL_PATH": "/home/liushenyu/opt-13b/",
    "TOKENIZER_PATH": "/home/liushenyu/opt-13b/",
    "INPUT_TEXT_FILE": 'c4_subset_1024mb.txt',
    "OUTPUT_ZARR_DIR": 'opt13b_trajectories_scores_2t_C',
    "CHUNK_SIZE": 2048,
    "TARGET_TRAJECTORIES": 5000,
    "MIN_SEQ_LEN": 128,
    "STEP_SIZE": 32
}

# --- 多卡设置辅助函数 (无变化) ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 增加超时设置，以防模型加载等操作过慢
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# --- 核心函数 (无变化) ---
def create_or_open_zarr(path, num_trajectories, model_path, chunk_size):
    if not os.path.exists(path):
        os.makedirs(path)
    root = zarr.open_group(path, mode='a')
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    NUM_LAYERS = config.num_hidden_layers
    D_MODEL = config.hidden_size
    shape_embeddings = (num_trajectories, chunk_size, D_MODEL)
    chunks_embeddings = (1, chunk_size, D_MODEL)
    shape_scores = (num_trajectories, NUM_LAYERS, chunk_size)
    chunks_scores = (1, NUM_LAYERS, chunk_size)
    if 'embeddings' not in root:
        root.create_dataset('embeddings', shape=shape_embeddings, chunks=chunks_embeddings, dtype='f2')
    if 'attention_scores' not in root:
        root.create_dataset('attention_scores', shape=shape_scores, chunks=chunks_scores, dtype='f2')
    return root

def calculate_target_scores(queries, keys):
    true_queries_last_token = queries[:, -1:, :, :]
    true_query_avg_heads = true_queries_last_token.mean(dim=2)
    keys_avg_heads = keys.mean(dim=2)
    true_scores = torch.einsum('lnd,lsd->lns', true_query_avg_heads, keys_avg_heads)
    return true_scores.squeeze(1)

# --- 主工作函数 (修改后) ---
# 新增了 chunk_list 作为参数
def worker_fn(rank, world_size, config, chunk_list):
    """
    为单个GPU执行数据生成任务的函数。
    """
    print(f"[{rank}] Worker started. Setting up process group...")
    setup(rank, world_size)
    
    DEVICE = f'cuda:{rank}'

    if rank == 0:
        print("[0] All workers started. Loading model on each device...")

    # 每个进程加载自己的模型副本
    model = AutoModelForCausalLM.from_pretrained(config["MODEL_PATH"], torch_dtype=torch.float16).to(DEVICE)
    # 不再需要加载 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(config["TOKENIZER_PATH"], use_fast=False)
    
    model.resize_token_embeddings(model.config.vocab_size + 1) # 假设 tokenizer 已经调整过
    model.eval()

    # --- 数据准备 (已在主进程完成) ---
    # 【REMOVED】删除了所有在 worker 中读取和分词的逻辑
    # 【REMOVED】删除了 dist.broadcast_object_list
    num_chunks = len(chunk_list)
    
    # Zarr 存储准备
    zarr_root = zarr.open_group(config["OUTPUT_ZARR_DIR"], mode='a')
    lock_path = os.path.join(config["OUTPUT_ZARR_DIR"], ".lock")
    counter_path = os.path.join(config["OUTPUT_ZARR_DIR"], "_trajectory_counter.txt")

    catcher = AttentionStateCatcher(model)
    
    pbar = None
    if rank == 0:
        # 读取初始计数
        if os.path.exists(counter_path):
            with open(counter_path, "r") as f:
                try:
                    initial_count = int(f.read())
                except ValueError:
                    initial_count = 0
        else:
            initial_count = 0
        pbar = tqdm(total=config["TARGET_TRAJECTORIES"], initial=initial_count, desc="Generating Trajectories")

    # 主循环 (逻辑不变)
    for chunk_idx in range(rank, num_chunks, world_size):
        chunk_tokens = chunk_list[chunk_idx]
        local_results = []
        
        current_len = config["MIN_SEQ_LEN"]
        while current_len <= config["CHUNK_SIZE"]:
            # 【检查是否已完成】在开始计算前检查，避免不必要的工作
            with FileLock(lock_path):
                with open(counter_path, "r") as f:
                    current_count = int(f.read())
            if current_count >= config["TARGET_TRAJECTORIES"]:
                break # 如果已达到目标，此 worker 不再处理新的 chunk

            input_ids = chunk_tokens[:current_len].unsqueeze(0).to(DEVICE)
            catcher.reset()
            catcher.register_hooks()
            
            try:
                with torch.no_grad():
                    model(input_ids)
                
                catcher.remove_hooks()
                captured_data = catcher.get_captured_data()
                
                if captured_data:
                    queries = captured_data['queries'].to(DEVICE, non_blocking=True)
                    keys = captured_data['keys'].to(DEVICE, non_blocking=True)
                    target_scores = calculate_target_scores(queries, keys)

                    embeddings = captured_data['embeddings']
                    padded_embeddings = torch.zeros(config["CHUNK_SIZE"], embeddings.shape[-1], dtype=torch.float16)
                    padded_embeddings[:current_len] = embeddings.to(torch.float16)
                    
                    padded_scores = torch.zeros(target_scores.shape[0], config["CHUNK_SIZE"], dtype=torch.float16)
                    padded_scores[:, :current_len] = target_scores.to(torch.float16)
                    
                    local_results.append({
                        "embeddings": padded_embeddings.cpu().numpy(),
                        "scores": padded_scores.cpu().numpy()
                    })

            except Exception as e:
                print(f"[{rank}] Error processing chunk {chunk_idx} at length {current_len}: {e}")
            
            if current_len == config["CHUNK_SIZE"]: break
            current_len = min(current_len + config["STEP_SIZE"], config["CHUNK_SIZE"])
            
            if 'captured_data' in locals(): del captured_data
            if 'queries' in locals(): del queries
            if 'keys' in locals(): del keys
            gc.collect()
            torch.cuda.empty_cache()

        # 同步写入 (逻辑不变)
        if local_results:
            with FileLock(lock_path):
                with open(counter_path, "r") as f:
                    start_index = int(f.read())
                
                if start_index >= config["TARGET_TRAJECTORIES"]:
                    pass # 其他进程可能已经完成了任务
                else:
                    num_to_write = len(local_results)
                    # 确保不会超出目标
                    num_to_write = min(num_to_write, config["TARGET_TRAJECTORIES"] - start_index)
                    if num_to_write > 0:
                        end_index = start_index + num_to_write
                        
                        embeddings_batch = np.stack([res["embeddings"] for res in local_results[:num_to_write]])
                        scores_batch = np.stack([res["scores"] for res in local_results[:num_to_write]])
                        
                        zarr_root['embeddings'][start_index:end_index] = embeddings_batch
                        zarr_root['attention_scores'][start_index:end_index] = scores_batch

                        with open(counter_path, "w") as f:
                            f.write(str(end_index))
                        
                        if rank == 0:
                            pbar.update(num_to_write)
        
        # 再次检查，如果已完成，可以提前退出循环
        with FileLock(lock_path):
            with open(counter_path, "r") as f:
                current_count = int(f.read())
        if current_count >= config["TARGET_TRAJECTORIES"]:
            break
    
    dist.barrier()
    if rank == 0:
        if pbar: pbar.close()
        print("\nData generation finished from all workers.")

    cleanup()

# --- 启动器 (修改后) ---
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPUs found. Exiting.")
        exit()

    # 【MOVED】1. 在启动多进程前，先完成所有预处理
    print("--- Pre-processing on main process ---")
    print("Loading tokenizer to prepare data...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["TOKENIZER_PATH"], use_fast=False)
    
    print(f"Reading and tokenizing '{CONFIG['INPUT_TEXT_FILE']}'...")
    with open(CONFIG["INPUT_TEXT_FILE"], 'r', encoding='utf-8') as f:
        text = f.read()
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    num_chunks = len(all_tokens) // CONFIG["CHUNK_SIZE"]
    chunk_list = [all_tokens[i * CONFIG["CHUNK_SIZE"] : (i + 1) * CONFIG["CHUNK_SIZE"]] for i in range(num_chunks)]
    del text, all_tokens # 释放内存
    gc.collect()
    print(f"Tokenization complete. Found {num_chunks} chunks.")

    # 【MOVED】2. 在主进程中创建 Zarr 结构和计数器
    print("Preparing Zarr data store...")
    trajectories_per_chunk = (CONFIG["CHUNK_SIZE"] - CONFIG["MIN_SEQ_LEN"]) // CONFIG["STEP_SIZE"] + 1
    ESTIMATED_TRAJECTORIES = num_chunks * trajectories_per_chunk
    print(f"Estimated total trajectories if all chunks are processed: {ESTIMATED_TRAJECTORIES}")
    create_or_open_zarr(CONFIG["OUTPUT_ZARR_DIR"], ESTIMATED_TRAJECTORIES, CONFIG["MODEL_PATH"], CONFIG["CHUNK_SIZE"])
    
    counter_path = os.path.join(CONFIG["OUTPUT_ZARR_DIR"], "_trajectory_counter.txt")
    if not os.path.exists(counter_path):
        with open(counter_path, "w") as f:
            f.write("0")
    print("--- Pre-processing finished ---\n")

    print(f"Found {world_size} GPUs. Spawning {world_size} processes...")
    
    # 【MODIFIED】3. 将预处理好的 chunk_list 作为参数传递
    mp.spawn(worker_fn,
             args=(world_size, CONFIG, chunk_list),
             nprocs=world_size,
             join=True)