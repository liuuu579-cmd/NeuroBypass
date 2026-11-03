import os
import torch
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
# 你不需要修改 utils2.py，保持原样即可
from utils2 import AttentionStateCatcher 
import gc
import traceback
from scipy.stats import spearmanr
from torch.nn.functional import cosine_similarity

# --- 配置 (无变化) ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
OUTPUT_ZARR_DIR = 'opt13b_local_similarity_results' 
CHUNK_SIZE = 2048
TARGET_TRAJECTORIES = 100
PRIMARY_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PARTITIONS = [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
SAMPLES_PER_PARTITION = 200
DISTANCES = [1, 2] 

# --- 函数 (无变化) ---
def create_or_open_zarr(path, num_trajectories):
    """创建或打开Zarr数据存储，用于存放预计算的相似度。"""
    if not os.path.exists(path):
        os.makedirs(path)
    root = zarr.open_group(path, mode='a')
    config = AutoConfig.from_pretrained(MODEL_PATH)
    NUM_LAYERS = config.num_hidden_layers
    shape = (num_trajectories, NUM_LAYERS, len(PARTITIONS), len(DISTANCES))
    chunks = (1, 1, len(PARTITIONS), len(DISTANCES))
    if 'spearman_similarity' not in root:
        root.create_dataset('spearman_similarity', shape=shape, chunks=chunks, dtype='f4')
    if 'cosine_similarity' not in root:
        root.create_dataset('cosine_similarity', shape=shape, chunks=chunks, dtype='f4')
    return root

def calculate_attention_scores(queries, keys, t):
    """计算单个时间点 t 的注意力分数 (softmax后)"""
    # 这里的queries/keys维度是 [S, H, D]，并且已经位于GPU上
    q_at_t = queries[t-1:t, :, :].to(torch.float32) 
    k_history = keys[:t, :, :].to(torch.float32)
    head_dim = queries.shape[-1]
    scores_raw = torch.einsum('shd,thd->hts', q_at_t, k_history)
    scores_scaled = scores_raw / (head_dim ** 0.5)
    scores_avg_head = scores_scaled.mean(dim=0)
    scores_softmax = torch.softmax(scores_avg_head, dim=-1)
    return scores_softmax.squeeze(0)

def calculate_similarity_for_points(queries_layer_gpu, keys_layer_gpu, t_base, t_comp):
    """计算两个时间点 t_base 和 t_comp 的注意力分布相似度。"""
    # 传入的张量已经是GPU上的
    scores_base = calculate_attention_scores(queries_layer_gpu, keys_layer_gpu, t_base)
    scores_comp = calculate_attention_scores(queries_layer_gpu, keys_layer_gpu, t_comp)
    scores_comp_truncated = scores_comp[:t_base]
    
    cos_sim = cosine_similarity(scores_base.unsqueeze(0), scores_comp_truncated.unsqueeze(0)).item()

    scores_base_cpu = scores_base.cpu().numpy()
    scores_comp_truncated_cpu = scores_comp_truncated.cpu().numpy()
    
    # 修复spearmanr在输入为常数时可能返回nan的问题
    if np.std(scores_base_cpu) < 1e-6 or np.std(scores_comp_truncated_cpu) < 1e-6:
        spearman_corr = 0.0
    else:
        spearman_corr, _ = spearmanr(scores_base_cpu, scores_comp_truncated_cpu)
        if np.isnan(spearman_corr): spearman_corr = 0.0

    return spearman_corr, cos_sim

def main():
    if not torch.cuda.is_available():
        print("警告: 正在CPU上运行，这将非常缓慢。")
    
    print("正在加载模型和分词器 (模型使用 float16 精度)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    model.eval()
    config = AutoConfig.from_pretrained(MODEL_PATH)
    NUM_LAYERS = config.num_hidden_layers
    print("模型加载完成。")

    print(f"准备Zarr数据存储于: {OUTPUT_ZARR_DIR}")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, TARGET_TRAJECTORIES)
    
    start_index = 0
    try:
        non_zero_rows = np.where(zarr_root['spearman_similarity'][:, 0, 0, 0] != 0)[0]
        if len(non_zero_rows) > 0:
            start_index = len(zarr_root['spearman_similarity'][:len(non_zero_rows)+1])
    except Exception:
        start_index = 0
           
    print(f"将从轨迹索引 {start_index} 开始。")
    
    # --- 核心修改点 1: 在循环外创建 Catcher ---
    catcher = AttentionStateCatcher(model)

    print("正在读取和分词输入文本...")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    num_chunks = len(all_tokens) // CHUNK_SIZE
    print(f"文本分词完成。找到 {num_chunks} 个大小为 {CHUNK_SIZE} 的数据块。")

    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="生成轨迹并计算相似度")
    
    chunk_idx = start_index
    trajectories_written = start_index
    while trajectories_written < TARGET_TRAJECTORIES and chunk_idx < num_chunks:
        
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
        input_ids = chunk_tokens.unsqueeze(0).to(PRIMARY_DEVICE)

        catcher.reset()
        catcher.register_hooks()

        try:
            # --- 核心修改点 2: 模型只运行一次，捕获所有层的数据到CPU ---
            with torch.no_grad():
                model(input_ids)
            
            catcher.remove_hooks()
            captured_data = catcher.get_captured_data()
            
            if not captured_data or 'queries' not in captured_data or 'keys' not in captured_data:
                chunk_idx += 1
                continue
            
            traj_spearman_results = np.zeros((NUM_LAYERS, len(PARTITIONS), len(DISTANCES)))
            traj_cosine_results = np.zeros((NUM_LAYERS, len(PARTITIONS), len(DISTANCES)))

            # --- 核心修改点 3: 逐层处理计算，避免高频内存往返 ---
            # captured_data['queries'] 是一个包含40个CPU张量的列表/字典
            for layer_idx in tqdm(range(NUM_LAYERS), desc=f"Trajectory {trajectories_written} Layers", leave=False):
                # a. 从CPU数据中获取当前层，并将其移动到GPU
                queries_layer_gpu = captured_data['queries'][layer_idx].to(PRIMARY_DEVICE).squeeze(0)
                keys_layer_gpu = captured_data['keys'][layer_idx].to(PRIMARY_DEVICE).squeeze(0)

                # b. 在GPU上执行该层的所有计算
                for i_part, (start, end) in enumerate(PARTITIONS):
                    sample_start = max(1, start)
                    sample_end = end - max(DISTANCES)
                    if sample_end <= sample_start: continue
                    
                    # 修复 RuntimeWarning, 同时确保采样数量不超过分区大小
                    num_available_samples = sample_end - sample_start
                    num_to_sample = min(SAMPLES_PER_PARTITION, num_available_samples)

                    base_points = np.random.choice(range(sample_start, sample_end), num_to_sample, replace=False)

                    for i_dist, dist in enumerate(DISTANCES):
                        partition_dist_spearman = []
                        partition_dist_cosine = []
                        
                        for t_base in base_points:
                            t_comp = t_base + dist
                            # 将GPU上的张量传递给计算函数
                            spearman, cosine = calculate_similarity_for_points(
                                queries_layer_gpu, keys_layer_gpu, t_base, t_comp
                            )
                            partition_dist_spearman.append(spearman)
                            partition_dist_cosine.append(cosine)
                        
                        # 修复 RuntimeWarning
                        if partition_dist_spearman:
                            traj_spearman_results[layer_idx, i_part, i_dist] = np.nanmean(partition_dist_spearman)
                            traj_cosine_results[layer_idx, i_part, i_dist] = np.nanmean(partition_dist_cosine)
                
                # c. 清理当前层的GPU显存
                del queries_layer_gpu, keys_layer_gpu
                gc.collect()
                torch.cuda.empty_cache()

            # 将整个 trajectory 的结果写入 Zarr
            zarr_root['spearman_similarity'][trajectories_written] = traj_spearman_results
            zarr_root['cosine_similarity'][trajectories_written] = traj_cosine_results
            
            trajectories_written += 1
            pbar.update(1)

        except Exception as e:
            traceback.print_exc()
            catcher.remove_hooks() # 确保钩子被移除
        
        finally:
            chunk_idx += 1
            if 'captured_data' in locals(): del captured_data
            gc.collect()
            torch.cuda.empty_cache()

    pbar.close()
    print(f"\n数据生成完成。总共处理了 {trajectories_written} 条轨迹。")

if __name__ == '__main__':
    main()