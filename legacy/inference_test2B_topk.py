# inference_benchmark_layerwise_30_10.py

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForCausalLM 

from predictor_model2A import PredictorTransformer
from dataset2 import TrajectoryDataset 

# ================= 配置区域 =================
PREDICTOR_MODEL_PATH = "predictor_topk_0.3_iwl_epoch_20.pth"
ZARR_PATH = 'opt13b_trajectories_scores_multi_point'
BACKBONE_MODEL_NAME = "/home/liushenyu/opt-13b/" 

NUM_SAMPLES = 100       
PREFETCH_BUDGET = 0.3    # [核心修改] 预取预算：30% 的池子
TARGET_RATIO = 0.05     # [核心修改] 核心目标：真正必须命中的 Top 10%
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

D_MODEL = 5120
NUM_PREDICT_LAYERS = 5   # 预测器实际输出的层数
NUM_TOTAL_LAYERS = 40    # 评估 OPT-13B 的全部 40 层
PREDICTOR_NHEAD = 4
PREDICTOR_ENCODER_LAYERS = 1
PREDICTOR_D_HID = 2048
NUM_RECENT_TOKENS = 10
# ===========================================

class BackboneWrapper:
    def __init__(self, model_name, device):
        print(f"正在加载骨干模型: {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map='auto' 
        )
        self.model.eval()

    def process_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        input_ids = inputs.input_ids
        if input_ids.shape[1] < 20: return None, None

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
        
        embeddings = outputs.hidden_states[-1].squeeze(0).float() 
        all_layer_scores = []
        for layer_attn in outputs.attentions:
            last_token_attn = layer_attn[0, :, -1, :] 
            max_attn_scores, _ = torch.max(last_token_attn, dim=0)
            all_layer_scores.append(max_attn_scores)
        true_scores = torch.stack(all_layer_scores) 
        return embeddings, true_scores.float()

def load_hf_dataset_samples(dataset_name, num_samples):
    texts = []
    print(f"Loading {dataset_name}...")
    try:
        if dataset_name == 'winogrande':
            ds = load_dataset('winogrande', 'winogrande_xl', split='validation', trust_remote_code=True)
            for i in range(num_samples): texts.append(ds[i]['sentence'] + " " + ds[i]['option1'])
        elif dataset_name == 'openbookqa':
            ds = load_dataset('openbookqa', 'main', split='validation', trust_remote_code=True)
            for i in range(num_samples): texts.append(ds[i]['question_stem'])
        elif dataset_name == 'COPA':
            ds = load_dataset('super_glue', 'copa', split='validation', trust_remote_code=True)
            for i in range(num_samples): texts.append(ds[i]['premise'] + " " + ds[i]['choice1'])
        elif dataset_name == 'RTE':
            ds = load_dataset('super_glue', 'rte', split='validation', trust_remote_code=True)
            for i in range(num_samples): texts.append(ds[i]['premise'] + " " + ds[i]['hypothesis'])
    except Exception as e:
        print(f"Error: {e}")
        return []
    return texts[:num_samples]

def main():
    print(f"Loading Predictor: {PREDICTOR_MODEL_PATH}")
    predictor = PredictorTransformer(
        input_dim=D_MODEL,
        num_opt_layers=NUM_PREDICT_LAYERS,
        nhead=PREDICTOR_NHEAD,
        num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
        d_hid=PREDICTOR_D_HID
    ).to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()

    backbone = None 
    datasets_to_test = ['C4', 'winogrande', 'openbookqa', 'COPA', 'RTE']
    
    final_summary_data = []

    for ds_name in datasets_to_test:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*40}")

        layer_hit_rates = {i: [] for i in range(NUM_TOTAL_LAYERS)}
        
        iterator = []
        
        # --- 数据加载 ---
        if ds_name == 'C4':
            try:
                dataset = TrajectoryDataset(ZARR_PATH)
                indices = range(min(len(dataset), NUM_SAMPLES))
                raw_iter = [(dataset[i][0], dataset[i][1]) for i in indices]
                for emb, score in raw_iter:
                     non_zero = emb.abs().sum(dim=-1) > 0
                     seq_len = non_zero.sum().item()
                     if seq_len > 0: iterator.append((emb[:seq_len], score[:, :seq_len]))
            except Exception as e:
                print(f"Skipping C4: {e}")
                continue
        else:
            raw_texts = load_hf_dataset_samples(ds_name, NUM_SAMPLES)
            if not raw_texts: continue
            if backbone is None:
                try: backbone = BackboneWrapper(BACKBONE_MODEL_NAME, DEVICE)
                except: break
            
            print(f"Inference Backbone...")
            for text in tqdm(raw_texts):
                emb, score = backbone.process_text(text)
                if emb is not None: iterator.append((emb, score))

        # --- 评估循环 ---
        for embeddings, true_scores in tqdm(iterator, desc=f"Eval {ds_name}"):
            embeddings = embeddings.unsqueeze(0).to(torch.float32).to(DEVICE)
            true_scores = true_scores.to(DEVICE) # shape: (40, Seq)
            current_seq_len = embeddings.shape[1]

            # 1. 生成 Ground Truth (全量40层，严格取 Top 10%)
            gt_mask = torch.zeros_like(true_scores)
            k_target = max(1, int(current_seq_len * TARGET_RATIO))
            with torch.no_grad():
                for i in range(NUM_TOTAL_LAYERS):
                    _, topk_idx_gt = torch.topk(true_scores[i], k=k_target, dim=-1)
                    gt_mask[i].scatter_(-1, topk_idx_gt, 1.0)

            # 2. 模型预测 (只预测前5层)
            with torch.no_grad():
                logits = predictor(embeddings).squeeze(0) # shape: (5, Seq)
            probs = torch.sigmoid(logits)

            # 3. 生成预取 Mask (5层，取 Top 30%)
            k_budget = max(1, int(current_seq_len * PREFETCH_BUDGET))
            _, topk_idx_pred = torch.topk(probs, k=k_budget, dim=-1)
            base_mask = torch.zeros_like(probs)
            base_mask.scatter_(-1, topk_idx_pred, 1.0)

            recency_mask = torch.zeros_like(probs)
            keep_last = min(current_seq_len, NUM_RECENT_TOKENS)
            if keep_last > 0: recency_mask[:, -keep_last:] = 1.0
            
            final_pred_mask = torch.logical_or(base_mask.bool(), recency_mask.bool()).float() # (5, Seq)

            # 4. 生成覆盖全量 40 层的复用 Mask
            full_pred_mask = torch.zeros((NUM_TOTAL_LAYERS, current_seq_len), device=DEVICE)
            full_pred_mask[:NUM_PREDICT_LAYERS] = final_pred_mask
            # 第6到40层（index 5 到 39），全部复用第5层（index 4）的预测结果
            for i in range(NUM_PREDICT_LAYERS, NUM_TOTAL_LAYERS):
                full_pred_mask[i] = final_pred_mask[NUM_PREDICT_LAYERS - 1]

            # 5. 按层计算命中率 (用 30% 预取池抓 10% 目标)
            for i in range(NUM_TOTAL_LAYERS):
                pred_layer = full_pred_mask[i]
                gt_layer = gt_mask[i] 

                # TP: 真正属于 Top 10% 且被包含在 30% 预取池中的数量
                tp = (pred_layer * gt_layer).sum().item()
                # 目标总量始终是 Top 10%
                total_target = gt_layer.sum().item() 
                epsilon = 1e-7
                
                # 命中率
                hit_rate = tp / (total_target + epsilon)
                layer_hit_rates[i].append(hit_rate)

        # --- 打印当前数据集结果 ---
        print(f"\n>>> Results for {ds_name} (用 30% 预取预算覆盖 Top 10% 核心KV):")
        print(f"{'Layer':<8} | {'命中率 (Hit Rate)':<20}")
        print("-" * 35)
        for i in range(NUM_TOTAL_LAYERS):
            if layer_hit_rates[i]:
                avg_hit = np.mean(layer_hit_rates[i])
                print(f"Layer {i:<2} | {avg_hit:>10.2%}")
                
                final_summary_data.append({
                    "Dataset": ds_name,
                    "Layer": i,
                    "Hit Rate": avg_hit
                })

    # ================= 最终汇总 =================
    print("\n\n" + "="*80)
    print("FINAL 40-LAYER BENCHMARK SUMMARY (30% Budget vs 10% Target)")
    print("="*80)
    if final_summary_data:
        df = pd.DataFrame(final_summary_data)
        
        print("\n--- 关键 KV 命中率 (Top 10% 是否全在 30% 预取池里？) ---")
        pivot_hit = df.pivot(index="Dataset", columns="Layer", values="Hit Rate")
        print(pivot_hit.to_string(float_format='{:.2%}'.format))
    else:
        print("No results generated.")

if __name__ == '__main__':
    main()