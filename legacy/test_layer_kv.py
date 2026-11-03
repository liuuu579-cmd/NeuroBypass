import torch
import torch.nn as nn
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import types
import math

# ==========================================
# 配置区域
# ==========================================
MODEL_PATH = "/home/liushenyu/opt-13b" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 策略参数
KEEP_RATIO = 0.3    # 保留 30%
TEST_SAMPLES = 100  # 测试样本数
SEQ_LEN = 1024      

# ==========================================
# 全局状态
# ==========================================
class GlobalKVState:
    def __init__(self):
        self.cutoff_layer = -1
        self.saved_indices = None
        self.ratio = KEEP_RATIO
        
kv_state = GlobalKVState()

# ==========================================
# 自定义 Attention Forward (类型安全修复版)
# ==========================================
def custom_opt_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor = None,
    past_key_value=None,
    attention_mask=None,
    layer_head_mask=None,
    output_attentions=False,
    **kwargs, 
):
    # --- Helper: Local Shape ---
    def _local_shape(tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # ---------------------------

    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # 1. Projection
    query_states = self.q_proj(hidden_states) * self.scaling

    # KV Handling
    if is_cross_attention and past_key_value is not None:
        key_states, value_states = past_key_value[0], past_key_value[1]
    elif is_cross_attention:
        key_states = _local_shape(self.k_proj(key_value_states), -1, bsz)
        value_states = _local_shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        if isinstance(past_key_value, tuple) and isinstance(past_key_value[0], tuple):
             past_key_value = past_key_value[0]
        key_states = _local_shape(self.k_proj(hidden_states), -1, bsz)
        value_states = _local_shape(self.v_proj(hidden_states), -1, bsz)
        try:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        except: pass
    else:
        key_states = _local_shape(self.k_proj(hidden_states), -1, bsz)
        value_states = _local_shape(self.v_proj(hidden_states), -1, bsz)

    # Reshape
    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = _local_shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)
    src_len = key_states.size(1)

    # 2. Attention Scores
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    # Causal Mask Handling
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            try:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            except:
                # 手动创建 mask, 确保 dtype 一致
                causal_mask = torch.full((tgt_len, src_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                attn_weights = attn_weights + causal_mask.unsqueeze(0)
        else:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        causal_mask = torch.full((tgt_len, src_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask.unsqueeze(0)
        
    attn_weights = torch.max(
        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    )

    # 3. Softmax
    if attn_weights.dtype == torch.float16:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # ==========================================================
    #  Strategy: Top-K + Sink Protection + Diagonal Protection
    # ==========================================================
    current_layer = self.layer_idx
    k_val = max(1, int(src_len * kv_state.ratio))

    if current_layer <= kv_state.cutoff_layer:
        # TopK
        top_values, top_indices = torch.topk(attn_weights, k_val, dim=-1)
        
        # Mask Construction
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(-1, top_indices, 1.0)

        # Sink Protection (Index 0)
        mask[..., 0] = 1.0 
        
        # Diagonal Protection
        if tgt_len > 1 and tgt_len == src_len:
            diag_mask = torch.eye(tgt_len, src_len, device=mask.device, dtype=mask.dtype).unsqueeze(0)
            mask = torch.max(mask, diag_mask)

        # Apply
        attn_weights = attn_weights * mask
        
        # Renormalize (add small float ensures div safety but promotes to float32)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
        
        # 【关键修复】强制转回 fp16，防止后续 bmm 报错
        attn_weights = attn_weights.to(value_states.dtype)

        # Save
        if current_layer == kv_state.cutoff_layer:
            kv_state.saved_indices = top_indices.detach()

    else:
        # Reuse
        if kv_state.saved_indices is not None:
            saved_idx = kv_state.saved_indices
            if saved_idx.shape[0] == attn_weights.shape[0]:
                curr_idx = saved_idx
                if curr_idx.shape[-1] > k_val: curr_idx = curr_idx[..., :k_val]
                
                if curr_idx.shape[-1] <= attn_weights.shape[-1]:
                    mask = torch.zeros_like(attn_weights)
                    mask.scatter_(-1, curr_idx, 1.0)
                    
                    mask[..., 0] = 1.0
                    if tgt_len > 1 and tgt_len == src_len:
                         diag_mask = torch.eye(tgt_len, src_len, device=mask.device, dtype=mask.dtype).unsqueeze(0)
                         mask = torch.max(mask, diag_mask)

                    attn_weights = attn_weights * mask
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
                    
                    # 【关键修复】强制转回 fp16
                    attn_weights = attn_weights.to(value_states.dtype)
    # ==========================================================

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    
    # 双重保险：在 bmm 前确保类型一致
    attn_probs = attn_probs.to(value_states.dtype) 
    
    attn_output = torch.bmm(attn_probs, value_states)

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, None

# ==========================================
# 3. 评估函数 (Target-Only Loss)
# ==========================================
def evaluate_winogrande_clean(model, tokenizer, dataset):
    model.eval()
    correct = 0
    total = 0
    total_time = 0
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    for i, sample in enumerate(dataset):
        if i >= TEST_SAMPLES: break
        
        sent = sample['sentence']
        opt1 = sample['option1']
        opt2 = sample['option2']
        label = int(sample['answer']) 
        
        text_a = sent.replace("_", opt1)
        text_b = sent.replace("_", opt2)

        context_part = sent.split("_")[0]
        
        # --- Option 1 ---
        kv_state.saved_indices = None 
        t_start = time.time()
        
        enc_a = tokenizer(text_a, return_tensors='pt').to(DEVICE)
        ids_a = enc_a.input_ids
        
        enc_ctx = tokenizer(context_part, add_special_tokens=True)
        len_ctx = len(enc_ctx.input_ids) 
        if len_ctx >= ids_a.shape[1]: len_ctx = 0 
        
        labels_a = ids_a.clone()
        labels_a[:, :len_ctx] = -100 

        with torch.no_grad():
            out_a = model(ids_a, use_cache=False)
            logits_a = out_a.logits[..., :-1, :].contiguous()
            shift_labels_a = labels_a[..., 1:].contiguous()
            loss_a = loss_fct(logits_a.view(-1, logits_a.size(-1)), shift_labels_a.view(-1))
            # 安全求均值：如果所有标签都被 mask，则 loss 为 inf (避免除以0)
            valid_elements = shift_labels_a.view(-1) != -100
            if valid_elements.sum() > 0:
                loss_a = loss_a[valid_elements].mean().item()
            else:
                loss_a = float('inf')

        # --- Option 2 ---
        kv_state.saved_indices = None
        enc_b = tokenizer(text_b, return_tensors='pt').to(DEVICE)
        ids_b = enc_b.input_ids
        
        labels_b = ids_b.clone()
        labels_b[:, :len_ctx] = -100 

        with torch.no_grad():
            out_b = model(ids_b, use_cache=False)
            logits_b = out_b.logits[..., :-1, :].contiguous()
            shift_labels_b = labels_b[..., 1:].contiguous()
            loss_b = loss_fct(logits_b.view(-1, logits_b.size(-1)), shift_labels_b.view(-1))
            valid_elements = shift_labels_b.view(-1) != -100
            if valid_elements.sum() > 0:
                loss_b = loss_b[valid_elements].mean().item()
            else:
                loss_b = float('inf')

        t_end = time.time()
        total_time += (t_end - t_start)

        prediction = 1 if loss_a < loss_b else 2
        if prediction == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    avg_latency = (total_time / total) * 1000 if total > 0 else 0
    return acc, avg_latency

# ==========================================
# 主程序
# ==========================================
def main():
    print(f"Loading: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    
    try: layers = model.model.decoder.layers
    except: layers = model.model.layers
    num_layers = len(layers)

    print("Injecting Attention Logic (Fixed Type Safety)...")
    for idx, layer in enumerate(layers):
        layer.self_attn.layer_idx = idx 
        layer.self_attn.forward = types.MethodType(custom_opt_attention_forward, layer.self_attn)

    print("Loading Winogrande...")
    data = load_dataset("winogrande", "winogrande_xl", split="validation")

    print(f"\nTest Start - Ratio: {KEEP_RATIO*100}%")
    print("="*70)
    print(f"{'Cutoff':<10} | {'Acc':<10} | {'Lat(ms)':<10}")
    print("-" * 70)

    steps = [0, 5, 10, 20, 30, num_layers-1]
    
    for i in steps:
        kv_state.cutoff_layer = i
        acc, latency = evaluate_winogrande_clean(model, tokenizer, data)
        print(f"{i:<10} | {acc:.4f}     | {latency:.2f}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()