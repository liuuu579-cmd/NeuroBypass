# inference_benchmark_layerwise.py

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
ALPHA = 2.0             
BUDGET_RATIO = 0.3      
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

D_MODEL = 5120
NUM_PREDICT_LAYERS = 5   # 我们只关注这 5 层的表现
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
    
    # 用于存储所有结果的列表，方便最后打印
    final_summary_data = []

    for ds_name in datasets_to_test:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*40}")

        # 针对每一层存储指标： key=层号(0-4), value=list of scores
        layer_recalls = {i: [] for i in range(NUM_PREDICT_LAYERS)}
        
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
            true_scores = true_scores.to(DEVICE) # (40, Seq)
            current_seq_len = embeddings.shape[1]

            # 1. 生成 GT (基于全层计算 Budget)
            with torch.no_grad():
                max_scores, _ = torch.max(true_scores, dim=-1, keepdim=True)
                gt_mask = (true_scores >= (max_scores - ALPHA)).float()
                k_cap = max(1, int(current_seq_len * BUDGET_RATIO))
                
                num_selected = gt_mask.sum(dim=-1)
                exceeded = num_selected > k_cap
                if torch.any(exceeded):
                    scores_cap = true_scores[exceeded]
                    _, topk_idx = torch.topk(scores_cap, k=k_cap, dim=-1)
                    capped = torch.zeros_like(scores_cap)
                    capped.scatter_(-1, topk_idx, 1.0)
                    gt_mask[exceeded] = capped

            # 2. 模型预测 (5层)
            with torch.no_grad():
                logits = predictor(embeddings).squeeze(0) # (5, Seq)
            probs = torch.sigmoid(logits)

            # 3. 生成预测 Mask (5层)
            k_cap_pred = max(1, int(current_seq_len * BUDGET_RATIO))
            _, topk_idx_pred = torch.topk(probs, k=k_cap_pred, dim=-1)
            base_mask = torch.zeros_like(probs)
            base_mask.scatter_(-1, topk_idx_pred, 1.0)

            recency_mask = torch.zeros_like(probs)
            keep_last = min(current_seq_len, NUM_RECENT_TOKENS)
            if keep_last > 0: recency_mask[:, -keep_last:] = 1.0
            
            # (5, Seq)
            final_pred_mask = torch.logical_or(base_mask.bool(), recency_mask.bool()).float()

            # 4. 按层计算指标 (只看前5层)
            for i in range(NUM_PREDICT_LAYERS):
                # 提取第 i 层的 mask
                pred_layer = final_pred_mask[i]
                gt_layer = gt_mask[i] # 对应 GT 的第 i 层

                tp = (pred_layer * gt_layer).sum().item()
                total_true = gt_layer.sum().item()
                epsilon = 1e-7
                
                # Hit Rate (Recall) = 命中了多少个应该被保留的 token
                recall = tp / (total_true + epsilon)
                layer_recalls[i].append(recall)

        # --- 打印当前数据集结果 ---
        print(f"\n>>> Results for {ds_name} (Hit Rate / Recall):")
        for i in range(NUM_PREDICT_LAYERS):
            if layer_recalls[i]:
                avg_rec = np.mean(layer_recalls[i])
                print(f"  Layer {i}: {avg_rec:.2%}")
                
                # 收集数据用于最后汇总
                final_summary_data.append({
                    "Dataset": ds_name,
                    "Layer": f"Layer {i}",
                    "Hit Rate": avg_rec
                })
            else:
                print(f"  Layer {i}: N/A")

    # ================= 最终汇总表格 =================
    print("\n\n" + "="*60)
    print("FINAL LAYER-WISE BENCHMARK SUMMARY (Hit Rate)")
    print("="*60)
    if final_summary_data:
        # 使用 Pivot Table 格式化输出: 行是数据集，列是层
        df = pd.DataFrame(final_summary_data)
        pivot_df = df.pivot(index="Dataset", columns="Layer", values="Hit Rate")
        # 调整列顺序
        cols = [f"Layer {i}" for i in range(NUM_PREDICT_LAYERS)]
        pivot_df = pivot_df[cols]
        
        print(pivot_df.to_string(formatters={c: '{:.2%}'.format for c in cols}))
        print("\n(数值为预取命中率/Recall: 真正需要的KV对中有多少被成功预取)")
    else:
        print("No results generated.")

if __name__ == '__main__':
    main()