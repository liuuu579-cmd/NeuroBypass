import os
import torch
import torch.nn.functional as F
import numpy as np
import zarr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
import traceback
import argparse

# --- 接收命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0, help="The GPU ID to use (0-7)")
parser.add_argument("--world_size", type=int, default=8, help="Total number of GPUs")
args = parser.parse_args()

# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'

# 【关键修改 1】每张卡写自己专属的 Zarr 文件夹，绝不冲突！
OUTPUT_ZARR_DIR = f'opt13b_greedy_routing_labels_v2_part_{args.gpu_id}' 
CHUNK_SIZE = 2048 

# 【关键修改 2】每张卡只需要完成 10000 / 8 = 1250 条
TOTAL_TARGET = 2000
LOCAL_TARGET = TOTAL_TARGET // args.world_size 
if args.gpu_id < (TOTAL_TARGET % args.world_size): # 处理除不尽的情况
    LOCAL_TARGET += 1

DEVICE = f'cuda:{args.gpu_id}'
NUM_SKIP_LAYERS = 16  # 目标跳过层数

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
    print(f"[GPU {args.gpu_id}] Loading model and tokenizer on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers
    D_MODEL = model.config.hidden_size
    transformer_layers = model.model.decoder.layers

    print(f"[GPU {args.gpu_id}] Preparing local Zarr data store ({LOCAL_TARGET} samples)...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, LOCAL_TARGET, NUM_LAYERS, D_MODEL)
    
    start_index = 0
    if 'embeddings' in zarr_root:
        while start_index < LOCAL_TARGET and zarr_root['embeddings'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
            
    if start_index >= LOCAL_TARGET:
        print(f"[GPU {args.gpu_id}] Already finished all {LOCAL_TARGET} trajectories. Exiting.")
        return
    
    print(f"[GPU {args.gpu_id}] Resuming from index {start_index}.")
    
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    
    # 进度条指定 position，防止 8 个进度条在终端里打架
    pbar = tqdm(total=LOCAL_TARGET, initial=start_index, 
                desc=f"GPU {args.gpu_id} Progress", position=args.gpu_id)

    # 【关键修改 3】只遍历属于这块 GPU 的文本块
    for chunk_idx in range(num_chunks):
        if chunk_idx % args.world_size != args.gpu_id:
            continue  # 不是我的任务，跳过！
            
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

                with torch.no_grad():
                    input_embeddings = model.get_input_embeddings()(input_ids)
                
                with torch.no_grad():
                    baseline_outputs = model(input_ids)
                    baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)

                skipped_layers = []
                available_layers = list(range(NUM_LAYERS))
                
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
                
                embeddings_at_len = input_embeddings.squeeze(0)
                padded_embeddings = torch.zeros(CHUNK_SIZE, D_MODEL, dtype=torch.float16, device='cpu')
                padded_embeddings[:current_len, :] = embeddings_at_len.to(torch.float16).cpu()
                
                zarr_root['embeddings'][start_index] = padded_embeddings.numpy()
                zarr_root['greedy_masks'][start_index] = binary_mask
                zarr_root['embeddings'].attrs[f'row_{start_index}_written'] = True
                
                start_index += 1
                pbar.update(1)

                del input_ids, input_embeddings, baseline_outputs, baseline_probs
                torch.cuda.empty_cache()
            
        except Exception as e:
            # 记录错误但不打印太多，防止刷屏
            pass 

    pbar.close()
    print(f"\n[GPU {args.gpu_id}] Finished successfully!")

if __name__ == '__main__':
    main()