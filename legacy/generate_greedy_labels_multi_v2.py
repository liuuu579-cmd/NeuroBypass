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
# --- 精准匹配你的 Bash 脚本参数 ---
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, default=0, help="当前进程的 ID (0~world_size-1)")
parser.add_argument("--world_size", type=int, default=4, help="总进程数 (比如 4 或 8)")
parser.add_argument("--gpu_id", type=int, default=0, help="使用的物理 GPU ID")
args = parser.parse_args()

worker_id = args.worker_id
world_size = args.world_size
gpu_id = args.gpu_id

DEVICE = f'cuda:{gpu_id}'
torch.cuda.set_device(gpu_id) # 强制当前进程只使用自己对应的 GPU！

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'

# 根据 worker_id 建立独立的 Zarr 文件夹，绝不冲突
OUTPUT_ZARR_DIR = f'opt13b_greedy_routing_labels_v2_part_{worker_id}' 
CHUNK_SIZE = 2048 

# 分配每个 worker 的任务量
TOTAL_TARGET = 2000
LOCAL_TARGET = TOTAL_TARGET // world_size 
if worker_id < (TOTAL_TARGET % world_size): 
    LOCAL_TARGET += 1

# 【核心实验设定】：跳过 16 层，迫使模型产生“输入依赖”的动态决策
NUM_SKIP_LAYERS = 16  

START_LEN = 128
STEP = 16
SAMPLING_POINTS = list(range(START_LEN, CHUNK_SIZE + 1, STEP))

def create_or_open_zarr(path, num_trajectories, num_layers, d_model):
    if not os.path.exists(path):
        os.makedirs(path)
    root = zarr.open_group(path, mode='a')
    
    # 特征保存为 layer1_states
    if 'layer1_states' not in root:
        root.create_dataset('layer1_states', shape=(num_trajectories, CHUNK_SIZE, d_model), 
                            chunks=(1, CHUNK_SIZE, d_model), dtype='f2')
    if 'greedy_masks' not in root:
        root.create_dataset('greedy_masks', shape=(num_trajectories, num_layers),
                            chunks=(1, num_layers), dtype='f4')
    return root

def get_skip_layer_hook():
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        else:
            return args[0]
    return hook

def main():
    print(f"[Worker {worker_id} -> GPU {gpu_id}] Loading model and tokenizer on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers
    D_MODEL = model.config.hidden_size
    transformer_layers = model.model.decoder.layers

    print(f"[Worker {worker_id}] Preparing local Zarr data store ({LOCAL_TARGET} samples)...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, LOCAL_TARGET, NUM_LAYERS, D_MODEL)
    
    start_index = 0
    if 'layer1_states' in zarr_root:
        while start_index < LOCAL_TARGET and zarr_root['layer1_states'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
            
    if start_index >= LOCAL_TARGET:
        print(f"[Worker {worker_id}] Already finished all {LOCAL_TARGET} trajectories. Exiting.")
        return
    
    print(f"[Worker {worker_id}] Resuming from index {start_index}.")
    
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    
    pbar = tqdm(total=LOCAL_TARGET, initial=start_index, 
                desc=f"Worker {worker_id} Progress", position=worker_id)

    for chunk_idx in range(num_chunks):
        # 任务分发：只处理属于当前 worker 的文本块
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

                # 一次性跑出 Baseline KL 和 Layer 1 的特征
                with torch.no_grad():
                    baseline_outputs = model(input_ids, output_hidden_states=True)
                    baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)
                    
                    # hidden_states[2] 就是 Layer 1 结束后的输出
                    layer1_features = baseline_outputs.hidden_states[2].squeeze(0).cpu()

                skipped_layers = []
                
                # 【重要物理逻辑】Router 吃的是 Layer 1，所以 Layer 0 和 Layer 1 不能剪！
                available_layers = list(range(2, NUM_LAYERS)) 
                
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
                    
                binary_mask = np.ones(NUM_LAYERS, dtype=np.float32)
                binary_mask[skipped_layers] = 0.0  
                
                padded_features = torch.zeros(CHUNK_SIZE, D_MODEL, dtype=torch.float16, device='cpu')
                padded_features[:current_len, :] = layer1_features.to(torch.float16)
                
                zarr_root['layer1_states'][start_index] = padded_features.numpy()
                zarr_root['greedy_masks'][start_index] = binary_mask
                zarr_root['layer1_states'].attrs[f'row_{start_index}_written'] = True
                
                start_index += 1
                pbar.update(1)

                del input_ids, baseline_outputs, baseline_probs, layer1_features
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n[Worker {worker_id}] OOM Error, skipping this chunk.")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            pass 

    pbar.close()
    print(f"\n[Worker {worker_id}] Finished successfully!")

if __name__ == '__main__':
    main()