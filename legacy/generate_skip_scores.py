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
OUTPUT_ZARR_DIR = 'opt13b_layer_skip_scores'
CHUNK_SIZE = 2048 
TARGET_TRAJECTORIES = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    if 'skip_scores' not in root:
        root.create_dataset('skip_scores', shape=(num_trajectories, num_layers),
                            chunks=(1, num_layers), dtype='f4')
        
    return root

# =====================================================================
# >>> 修复核心：使用 Post-Hook 实现完美跳层 <<<
# 逻辑：层算完了 -> 拦截输出 -> 把输出强行改成输入 -> 下一层接收到的就是输入
# =====================================================================
def get_skip_layer_hook():
    def hook(module, args, output):
        # args 是一个 tuple，args[0] 是 Layer 的 Input (Hidden States)
        # output 是 Layer 的 Output (通常也是 tuple)
        
        # 我们执行 Identity Mapping: Output = Input
        if isinstance(output, tuple):
            # 保留 output[1:] (比如 cache)，只把 hidden_states (output[0]) 替换为 input
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

    # OPT 的层列表
    transformer_layers = model.model.decoder.layers

    print("Preparing Zarr data store...")
    zarr_root = create_or_open_zarr(OUTPUT_ZARR_DIR, TARGET_TRAJECTORIES, NUM_LAYERS, D_MODEL)
    
    start_index = 0
    if 'embeddings' in zarr_root:
        while start_index < TARGET_TRAJECTORIES and zarr_root['embeddings'].attrs.get(f'row_{start_index}_written', False):
            start_index += 1
    
    print(f"Found {start_index} existing trajectories.")
    
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    num_chunks = len(all_tokens) // CHUNK_SIZE
    pbar = tqdm(total=TARGET_TRAJECTORIES, initial=start_index, desc="Generating Scores")

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
                
                # 2. Baseline 前向传播 (不跳层)
                with torch.no_grad():
                    baseline_outputs = model(input_ids)
                    baseline_logits = baseline_outputs.logits[:, -1, :] 
                    baseline_probs = F.softmax(baseline_logits, dim=-1)

                layer_kl_scores = np.zeros(NUM_LAYERS, dtype=np.float32)

                # 3. Leave-One-Out 测试
                for layer_idx in range(NUM_LAYERS):
                    # 【改动】使用 register_forward_hook (不需要 pre, 不需要 kwargs)
                    handle = transformer_layers[layer_idx].register_forward_hook(get_skip_layer_hook())
                    
                    try:
                        with torch.no_grad():
                            skip_outputs = model(input_ids)
                            skip_logits = skip_outputs.logits[:, -1, :]
                            skip_log_probs = F.log_softmax(skip_logits, dim=-1)
                            
                        # 计算 KL 散度
                        kl_div = F.kl_div(skip_log_probs, baseline_probs, reduction='sum').item()
                        layer_kl_scores[layer_idx] = kl_div
                        
                    finally:
                        # 确保无论如何 hook 都要被移除
                        handle.remove()

                # 4. 存储
                embeddings_at_len = input_embeddings.squeeze(0)
                padded_embeddings = torch.zeros(CHUNK_SIZE, D_MODEL, dtype=torch.float16, device='cpu')
                padded_embeddings[:current_len, :] = embeddings_at_len.to(torch.float16).cpu()
                
                zarr_root['embeddings'][start_index] = padded_embeddings.numpy()
                zarr_root['skip_scores'][start_index] = layer_kl_scores
                zarr_root['embeddings'].attrs[f'row_{start_index}_written'] = True
                
                start_index += 1
                pbar.update(1)

                del input_ids, input_embeddings, baseline_outputs, baseline_logits, baseline_probs
                if 'skip_outputs' in locals(): del skip_outputs
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