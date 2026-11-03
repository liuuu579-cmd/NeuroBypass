import os
import torch
import torch.nn.functional as F
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import gc

# =========================================================================
# --- 参数与环境配置 ---
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--world_size", type=int, default=4)
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

worker_id = args.worker_id
world_size = args.world_size
gpu_id = args.gpu_id

DEVICE = f'cuda:{gpu_id}'
torch.cuda.set_device(gpu_id)

MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'

OUTPUT_ZARR_DIR = f'opt13b_greedy_routing_labels_v3_part_{worker_id}'
CHUNK_SIZE = 2048 

TOTAL_TARGET = 2000
LOCAL_TARGET = TOTAL_TARGET // world_size 
if worker_id < (TOTAL_TARGET % world_size): 
    LOCAL_TARGET += 1

NUM_SKIP_LAYERS = 16  
START_LEN = 128
STEP = 16
SAMPLING_POINTS = list(range(START_LEN, CHUNK_SIZE + 1, STEP))

# =========================================================================
# --- Zarr 极简存储方案重构 ---
# =========================================================================
def create_or_open_zarr(path, num_trajectories, max_seq_len, num_skip_steps):
    if not os.path.exists(path):
        os.makedirs(path)
    root = zarr.open_group(path, mode='a')
    
    # 仅保存 Token IDs，极大地缩减存储 (dtype=i4)
    if 'input_ids' not in root:
        root.create_dataset('input_ids', shape=(num_trajectories, max_seq_len), 
                            chunks=(1, max_seq_len), dtype='i4', fill_value=0)
    # 记录该序列的真实有效长度
    if 'seq_lens' not in root:
        root.create_dataset('seq_lens', shape=(num_trajectories,), 
                            chunks=(100,), dtype='i4')
    # 记录 1 到 16 步每次贪心剪掉的层索引，完美保留所有中间状态
    if 'ordered_skips' not in root:
        root.create_dataset('ordered_skips', shape=(num_trajectories, num_skip_steps),
                            chunks=(1, num_skip_steps), dtype='i2')
    # 【新增】记录 1 到 16 步对应的 KL 散度，用于后续定义"句子难度"和"动态阈值"
    if 'kl_divergences' not in root:
        root.create_dataset('kl_divergences', shape=(num_trajectories, num_skip_steps),
                            chunks=(1, num_skip_steps), dtype='f4')
    return root

def get_skip_layer_hook():
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        else:
            return args[0]
    return hook

def main():
    print(f"[Worker {worker_id} -> GPU {gpu_id}] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers
    transformer_layers = model.model.decoder.layers

    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, LOCAL_TARGET, CHUNK_SIZE, NUM_SKIP_LAYERS)
    
    start_index = 0
    if 'input_ids' in zarr_root:
        while start_index < LOCAL_TARGET and zarr_root['input_ids'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
            
    if start_index >= LOCAL_TARGET:
        print(f"[Worker {worker_id}] Task completed. Exiting.")
        return
    
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    pbar = tqdm(total=LOCAL_TARGET, initial=start_index, desc=f"Worker {worker_id}", position=worker_id)

    for chunk_idx in range(num_chunks):
        if chunk_idx % world_size != worker_id:
            continue 
            
        if start_index >= LOCAL_TARGET:
            break
            
        chunk_tokens = all_tokens[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]

        try:
            for current_len in SAMPLING_POINTS:
                if start_index >= LOCAL_TARGET:
                    break

                input_ids = chunk_tokens[:current_len].unsqueeze(0).to(DEVICE)
                if input_ids.max().item() >= model.get_input_embeddings().weight.shape[0]:
                    continue

                # 提取 Baseline 的标准答案分布
                with torch.no_grad():
                    baseline_outputs = model(input_ids)
                    baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)

                skipped_layers = []
                # Router 吃的是 Layer 1，保护 0 和 1 不被剪
                available_layers = list(range(2, NUM_LAYERS)) 
                
                # 记录这16次迭代每一步剪掉的具体层号
                ordered_step_skips = [] 
                # 【新增】记录这16次迭代每一步的最优 KL 散度
                ordered_step_kls = []

                for step in range(NUM_SKIP_LAYERS):
                    best_kl = float('inf')
                    best_layer = -1
                    
                    for layer_idx in available_layers:
                        test_skip_set = skipped_layers + [layer_idx]
                        handles = [transformer_layers[l].register_forward_hook(get_skip_layer_hook()) for l in test_skip_set]
                        
                        with torch.no_grad():
                            skip_outputs = model(input_ids)
                            skip_log_probs = F.log_softmax(skip_outputs.logits[:, -1, :], dim=-1)
                            
                        for h in handles: h.remove()
                            
                        kl_div = F.kl_div(skip_log_probs, baseline_probs, reduction='sum').item()
                        
                        if kl_div < best_kl:
                            best_kl = kl_div
                            best_layer = layer_idx
                            
                        del skip_outputs, skip_log_probs
                        
                    skipped_layers.append(best_layer)
                    available_layers.remove(best_layer)
                    ordered_step_skips.append(best_layer) # 核心：记录状态路径
                    ordered_step_kls.append(best_kl) # 【核心新增】：记录状态路径的代价
                
                # 写入 Zarr，数据量极小
                padded_ids = np.zeros(CHUNK_SIZE, dtype=np.int32)
                padded_ids[:current_len] = input_ids.cpu().numpy()[0]
                
                zarr_root['input_ids'][start_index] = padded_ids
                zarr_root['seq_lens'][start_index] = current_len
                zarr_root['ordered_skips'][start_index] = np.array(ordered_step_skips, dtype=np.int16)
                # 【新增】写入 KL 数组
                zarr_root['kl_divergences'][start_index] = np.array(ordered_step_kls, dtype=np.float32)
                zarr_root['input_ids'].attrs[f'row_{start_index}_written'] = True
                start_index += 1
                pbar.update(1)

                del input_ids, baseline_outputs, baseline_probs
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n[Worker {worker_id}] OOM Error, skipping.")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            pass 

    pbar.close()

if __name__ == '__main__':
    main()