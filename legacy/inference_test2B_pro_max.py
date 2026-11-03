# online_evaluation.py (Final fix for dtype mismatch)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from predictor_model2A import PredictorTransformer
from utils2 import AttentionStateCatcher 

# --- 核心配置 ---
OPT_MODEL_PATH = "/home/liushenyu/opt-13b/"
PREDICTOR_MODEL_PATH = "predictor_topk_0.3_iwl_epoch_20.pth"
INPUT_TEXT_FILE = 'c4_subset_1024mb.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 评估范围和策略配置 ---
EVAL_START_INDEX = 10000
NUM_SAMPLES_TO_EVAL = 1000
ALPHA = 5.0
BUDGET_RATIO = 0.3 

# --- 数据生成参数 ---
CHUNK_SIZE = 2048
START_LEN = 128
STEP = 16
SAMPLING_POINTS = list(range(START_LEN, CHUNK_SIZE + 1, STEP))
SAMPLES_PER_CHUNK = len(SAMPLING_POINTS)

# --- 模型结构超参数 ---
NUM_PREDICT_LAYERS = 5
PREDICTOR_NHEAD = 16
PREDICTOR_ENCODER_LAYERS = 8
PREDICTOR_D_HID = 4096
D_MODEL = 5120
NUM_LAYERS_OPT = 40

def get_ground_truth_scores_from_qk(queries, keys):
    true_queries_last_token = queries[:, -1:, :, :]
    true_query_avg_heads = true_queries_last_token.mean(dim=2)
    keys_avg_heads = keys.mean(dim=2)
    true_scores = torch.einsum('lnd,lsd->lns', true_query_avg_heads, keys_avg_heads)
    return true_scores.squeeze(1)

def main():
    if DEVICE == 'cpu':
        print("Warning: Running on CPU. This will be extremely slow.")

    print("Loading OPT-13B model...")
    opt_model = AutoModelForCausalLM.from_pretrained(
        OPT_MODEL_PATH, 
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    opt_model.eval()

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_PATH, use_fast=False)

    print(f"Loading Predictor model from {PREDICTOR_MODEL_PATH}...")
    predictor = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_PREDICT_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()
    print("All models loaded successfully.")

    print(f"Reading and tokenizing text from {INPUT_TEXT_FILE}...")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    num_chunks = len(all_tokens) // CHUNK_SIZE
    print(f"Text tokenized into {num_chunks} chunks of size {CHUNK_SIZE}.")

    results = {
        f'layer_{i}': {
            'spearman_corr': [], 'precision': [], 'recall': [], 'f1_score': []
        } for i in range(NUM_PREDICT_LAYERS)
    }
    catcher = AttentionStateCatcher(opt_model)

    print(f"\nStarting online evaluation for {NUM_SAMPLES_TO_EVAL} samples, starting from index {EVAL_START_INDEX}...")
    
    pbar = tqdm(range(NUM_SAMPLES_TO_EVAL), desc="Evaluating Samples")
    for i in pbar:
        current_sample_index = EVAL_START_INDEX + i
        
        chunk_idx = current_sample_index // SAMPLES_PER_CHUNK
        point_idx_in_chunk = current_sample_index % SAMPLES_PER_CHUNK
        
        if chunk_idx >= num_chunks:
            print(f"\nWarning: Not enough text chunks to reach sample {current_sample_index}. Stopping early.")
            break

        try:
            current_len = SAMPLING_POINTS[point_idx_in_chunk]
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_tokens = all_tokens[chunk_start : chunk_start + CHUNK_SIZE]
            input_ids = chunk_tokens[:current_len].unsqueeze(0).to(DEVICE)

            catcher.reset()
            catcher.register_hooks()
            with torch.no_grad():
                opt_model(input_ids)
            catcher.remove_hooks()
            
            captured_data = catcher.get_captured_data()
            if not captured_data: continue

            embeddings = captured_data['embeddings'].unsqueeze(0).to(torch.float32).to(DEVICE)
            queries = captured_data['queries'].to(DEVICE)
            keys = captured_data['keys'].to(DEVICE)
            
            true_scores = get_ground_truth_scores_from_qk(queries, keys)

            with torch.no_grad():
                predicted_logits = predictor(embeddings).squeeze(0)

            with torch.no_grad():
                max_scores, _ = torch.max(true_scores, dim=-1, keepdim=True)
                thresholds = max_scores - ALPHA
                ground_truth_mask_alpha = (true_scores >= thresholds).float()
                
                k_cap = int(current_len * BUDGET_RATIO)
                if k_cap == 0: k_cap = 1 
                
                num_selected_alpha = ground_truth_mask_alpha.sum(dim=-1)
                ground_truth_mask = ground_truth_mask_alpha.clone()
                exceeded_layers_mask = num_selected_alpha > k_cap

                if torch.any(exceeded_layers_mask):
                    scores_to_cap = true_scores[exceeded_layers_mask]
                    _, topk_indices = torch.topk(scores_to_cap, k=k_cap, dim=-1)
                    capped_mask = torch.zeros_like(scores_to_cap, device=DEVICE)
                    capped_mask.scatter_(-1, topk_indices, 1.0)
                    
                    # <<< FIX: Convert the source mask's dtype to match the destination's dtype before assignment >>>
                    ground_truth_mask[exceeded_layers_mask] = capped_mask.to(ground_truth_mask.dtype)

            predicted_mask = torch.zeros_like(predicted_logits, device=DEVICE)
            _, topk_pred_indices = torch.topk(predicted_logits, k=k_cap, dim=-1)
            predicted_mask.scatter_(-1, topk_pred_indices, 1.0)

            for layer_idx in range(NUM_PREDICT_LAYERS):
                pred_logits_layer = predicted_logits[layer_idx].cpu().numpy()
                true_scores_layer = true_scores[layer_idx].cpu().numpy()
                corr, _ = spearmanr(pred_logits_layer, true_scores_layer)
                results[f'layer_{layer_idx}']['spearman_corr'].append(corr if not np.isnan(corr) else 0.0)
                pred_mask_layer = predicted_mask[layer_idx]
                true_mask_layer = ground_truth_mask[layer_idx] 
                tp = (pred_mask_layer * true_mask_layer).sum().item()
                fp = (pred_mask_layer * (1 - true_mask_layer)).sum().item()
                fn = ((1 - pred_mask_layer) * true_mask_layer).sum().item()
                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2 * (precision * recall) / (precision + recall + epsilon)
                results[f'layer_{layer_idx}']['precision'].append(precision)
                results[f'layer_{layer_idx}']['recall'].append(recall)
                results[f'layer_{layer_idx}']['f1_score'].append(f1)
            
            del captured_data, embeddings, queries, keys, true_scores, predicted_logits
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing sample index {current_sample_index}: {e}")
            catcher.remove_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue
            
    pbar.close()
    print("\n\n--- Final Evaluation Report ---")
    print(f"Averaged over {len(results['layer_0']['spearman_corr'])} successfully processed samples.")
    
    summary_data = []
    for i in range(NUM_PREDICT_LAYERS):
        layer_name = f'layer_{i}'
        avg_spearman = np.mean(results[layer_name]['spearman_corr'])
        avg_precision = np.mean(results[layer_name]['precision'])
        avg_recall = np.mean(results[layer_name]['recall'])
        avg_f1 = np.mean(results[layer_name]['f1_score'])
        
        summary_data.append({
            'Layer': i,
            'Avg Logit Spearman Correlation': avg_spearman,
            'Avg Precision': avg_precision,
            'Avg Recall': avg_recall,
            'Avg F1 Score': avg_f1
        })
        
    summary_df = pd.DataFrame(summary_data)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()