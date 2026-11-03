import os
import torch
import torch.nn.functional as F
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import gc
import traceback

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
OUTPUT_ZARR_DIR = 'opt13b_greedy_routing_labels' # 更改了输出目录名
CHUNK_SIZE = 2048 
TARGET_TRAJECTORIES = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 核心跳层配置 ---
NUM_SKIP_LAYERS = 6  # 目标跳过层数 (约15%加速)

# --- 多点采样配置 ---
START_LEN = 128
STEP = 16
SAMPLING_POINTS = list(range(START_LEN, CHUNK_SIZE + 1, STEP))

def create_or_open_zarr(path, num_trajectories, num_layers, d_model):
    if not os.path.exists(path):
        os.makedirs(path)

    root = zarr.open_group(path, mode='a')
    
    if 'embeddings' not in root:
        root.create_dataset('embeddings', shape=(num_trajectories, CHUNK_SIZE, d_model), 
                            chunks=(1, CHUNK_SIZE, d_model), dtype='f2')
    
    # 名字改为了 greedy_masks, 存 0/1 二值掩码
    if 'greedy_masks' not in root:
        root.create_dataset('greedy_masks', shape=(num_trajectories, num_layers),
                            chunks=(1, num_layers), dtype='f4')
        
    return root

# =====================================================================
# >>> Post-Hook 实现 <<<
# =====================================================================
def get_skip_layer_hook():
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        else:
            return args[0]
    return hook

def main():
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    config = model.config
    NUM_LAYERS = config.num_hidden_layers
    D_MODEL = config.hidden_size

    transformer_layers = model.model.decoder.layers

    print("Preparing Zarr data store...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, TARGET_TRAJECTORIES, NUM_LAYERS, D_MODEL)
    
    start_index = 0
    if 'embeddings' in zarr_root:
        while start_index < TARGET_TRAJECTORIES and zarr_root['embeddings'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
    
    print(f"Found {start_index} existing trajectories. Generating Greedy Labels (K={NUM_SKIP_LAYERS})...")
    
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="Generating Greedy Routing Labels")

    chunk_idx = start_index // len(SAMPLING_POINTS)
    
    while start_index < TARGET_TRAJECTORIES and chunk_idx < num_chunks:
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]

        try:
            for current_len in SAMPLING_POINTS:
                if start_index >= TARGET_TRAJECTORIES:
                    break

                input_ids = chunk_tokens[:current_len].unsqueeze(0).to(DEVICE)
                
                max_token_id = input_ids.max().item()
                if max_token_id >= model.get_input_embeddings().weight.shape[0]:
                    continue

                # 1. 提取 Embedding
                with torch.no_grad():
                    input_embeddings = model.get_input_embeddings()(input_ids)
                
                # 2. Baseline 前向传播 (获取基准 Target 分布)
                with torch.no_grad():
                    baseline_outputs = model(input_ids)
                    baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)

                # ==============================================================
                # 3. 核心：迭代式贪心算法 (Iterative Greedy)
                # ==============================================================
                skipped_layers = []
                available_layers = list(range(NUM_LAYERS))
                
                # 寻找最佳的 K 层组合
                for step in range(NUM_SKIP_LAYERS):
                    best_kl = float('inf')
                    best_layer = -1
                    
                    for layer_idx in available_layers:
                        # 当前尝试的组合 = 之前已确认跳过的层 + 本次尝试的层
                        test_skip_set = skipped_layers + [layer_idx]
                        
                        # 批量挂载 Hook
                        handles = []
                        for l in test_skip_set:
                            handles.append(transformer_layers[l].register_forward_hook(get_skip_layer_hook()))
                        
                        # 跑一次前向算 KL
                        with torch.no_grad():
                            skip_outputs = model(input_ids)
                            skip_log_probs = F.log_softmax(skip_outputs.logits[:, -1, :], dim=-1)
                            
                        # 立刻移除所有 Hook！
                        for h in handles:
                            h.remove()
                            
                        kl_div = F.kl_div(skip_log_probs, baseline_probs, reduction='sum').item()
                        
                        if kl_div < best_kl:
                            best_kl = kl_div
                            best_layer = layer_idx
                            
                        del skip_outputs, skip_log_probs
                        
                    # 确认本轮最佳选择
                    skipped_layers.append(best_layer)
                    available_layers.remove(best_layer)
                    
                # 4. 生成二值掩码标签 (1=Keep, 0=Skip)
                binary_mask = np.ones(NUM_LAYERS, dtype=np.float32)
                binary_mask[skipped_layers] = 0.0  # 把找出的 6 个层标记为跳过 (0)
                
                # ==============================================================
                # 5. 存储
                embeddings_at_len = input_embeddings.squeeze(0)
                padded_embeddings = torch.zeros(CHUNK_SIZE, D_MODEL, dtype=torch.float16, device='cpu')
                padded_embeddings[:current_len, :] = embeddings_at_len.to(torch.float16).cpu()
                
                zarr_root['embeddings'][start_index] = padded_embeddings.numpy()
                zarr_root['greedy_masks'][start_index] = binary_mask  # 存入掩码
                zarr_root['embeddings'].attrs[f'row_{start_index}_written'] = True
                
                start_index += 1
                pbar.update(1)

                del input_ids, input_embeddings, baseline_outputs, baseline_probs
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError in chunk {chunk_idx}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
        
        chunk_idx += 1

    pbar.close()
    print("\nData generation finished.")

if __name__ == '__main__':
    main()