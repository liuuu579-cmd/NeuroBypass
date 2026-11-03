import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_router_dynamic import DynamicRouterDataset
from model_router_v4 import DynamicConditionalRouter

# ==============================================================================
# --- 评测配置 ---
# ==============================================================================
DEVICE = "cuda:0"
MODEL_PATH = "/home/liushenyu/opt-13b/"
CHECKPOINT_PATH = "router_rl_dynamic_epoch_2.pth"
TEST_ZARR_PATH = "/home/liushenyu/test/opt13b_greedy_routing_labels_v3_part_0"

NUM_OPT_LAYERS = 40
D_MODEL = 5120
MAX_SKIP_LAYERS = 16
NUM_EVAL_SAMPLES = 50  # 评测样本数，50 条足够精确反映分布并保证速度

# Hook 类：用于高效提取 Layer 1 特征
class EarlyExitException(Exception):
    pass

def extract_layer1_embeddings(base_model, input_ids):
    saved_hidden_state = []
    def hook_fn(module, args, output):
        saved_hidden_state.append(output[0] if isinstance(output, tuple) else output)
        raise EarlyExitException()

    handle = base_model.model.decoder.layers[1].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            base_model(input_ids=input_ids, use_cache=False)
    except EarlyExitException:
        pass
    finally:
        handle.remove()
    return saved_hidden_state[0]

# 跳层 Hook
def get_skip_layer_hook():
    def hook(module, args, output):
        return (args[0],) + output[1:] if isinstance(output, tuple) else args[0]
    return hook

def main():
    print(f"Loading Base Model from {MODEL_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE)
    base_model.eval()

    print(f"Loading Router Model from {CHECKPOINT_PATH}...")
    router = DynamicConditionalRouter(
        embedding_dim=D_MODEL, 
        hidden_dim=512, 
        num_layers=NUM_OPT_LAYERS, 
        max_skip=MAX_SKIP_LAYERS
    ).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    router.load_state_dict(cleaned_state_dict)
    router.half()
    router.eval()
    print(f"[OK] 成功载入权重: {CHECKPOINT_PATH}")

    dataset = DynamicRouterDataset(TEST_ZARR_PATH, max_skip_steps=MAX_SKIP_LAYERS, num_layers=NUM_OPT_LAYERS)
    eval_indices = np.linspace(0, len(dataset) - 1, NUM_EVAL_SAMPLES, dtype=int)

    # 重点测试两档深度：常规剪枝 (k=6) 与 深度剪枝 (k=16)
    test_k_values = [6, 16]

    for k_val in test_k_values:
        print(f"\n" + "="*60)
        print(f"              评估剪枝深度: N = {k_val}")
        print("="*60)

        predicted_masks = []
        router_kl_list = []
        greedy_gt_kl_list = []
        all_predicted_skipped_layers = []

        for sample_count, idx in enumerate(eval_indices):
            # 获取样本 Token 和贪心真实标签
            ids, _, _ = dataset[idx]
            input_ids = ids.unsqueeze(0).to(DEVICE)
            skips_gt = dataset.ordered_skips[idx][:k_val]
            
            # 1. 计算 Full Model Baseline 输出
            with torch.no_grad():
                baseline_outputs = base_model(input_ids)
                baseline_probs = F.softmax(baseline_outputs.logits[:, -1, :], dim=-1)

            # 2. 提取 Layer 1 特征并让 Router 预测跳层策略
            layer1_feat = extract_layer1_embeddings(base_model, input_ids)
            target_k_tensor = torch.tensor([k_val], dtype=torch.long, device=DEVICE)
            
            with torch.no_grad():
                pred_logits = router(layer1_feat, target_k_tensor)
                # 取出预测跳过的 k 个层（logits 最小的 k 层）
                _, pred_skips = torch.topk(pred_logits[0], k=k_val, largest=False)
                pred_skips = sorted(pred_skips.cpu().tolist())

            predicted_masks.append(tuple(pred_skips))
            all_predicted_skipped_layers.extend(pred_skips)

            # 3. 测量 Router 策略的真实模型 KL
            handles = [base_model.model.decoder.layers[l].register_forward_hook(get_skip_layer_hook()) for l in pred_skips]
            with torch.no_grad():
                router_out = base_model(input_ids)
                router_log_probs = F.log_softmax(router_out.logits[:, -1, :], dim=-1)
                router_kl = F.kl_div(router_log_probs, baseline_probs, reduction='sum').item()
            for h in handles: h.remove()
            router_kl_list.append(router_kl)

            # 4. 测量真实贪心标签的真实模型 KL
            handles_gt = [base_model.model.decoder.layers[l].register_forward_hook(get_skip_layer_hook()) for l in skips_gt]
            with torch.no_grad():
                gt_out = base_model(input_ids)
                gt_log_probs = F.log_softmax(gt_out.logits[:, -1, :], dim=-1)
                gt_kl = F.kl_div(gt_log_probs, baseline_probs, reduction='sum').item()
            for h in handles_gt: h.remove()
            greedy_gt_kl_list.append(gt_kl)

            if (sample_count + 1) % 10 == 0:
                print(f"进度: {sample_count + 1}/{NUM_EVAL_SAMPLES} | 当前样本 Router KL: {router_kl:.4f} | GT KL: {gt_kl:.4f}")

        # ==============================================================================
        # --- 结果统计分析 ---
        # ==============================================================================
        unique_masks = len(set(predicted_masks))
        top_patterns = Counter(predicted_masks).most_common(3)
        layer_freq = Counter(all_predicted_skipped_layers).most_common(10)

        print("\n--- 结果汇报 ---")
        print(f"1. 掩码多样性:")
        print(f"   - 总测试样本: {NUM_EVAL_SAMPLES} 条")
        print(f"   - 独立出现的不同剪枝组合数: {unique_masks} 种 (占比: {unique_masks/NUM_EVAL_SAMPLES*100:.1f}%)")
        print(f"   - 出现频次最高的组合前 3 名:")
        for pattern, count in top_patterns:
            print(f"     * {list(pattern)} -> 出现 {count} 次 ({count/NUM_EVAL_SAMPLES*100:.1f}%)")

        print(f"\n2. 层选择偏好 (最常被跳过的 Layer Top 10):")
        freq_str = ", ".join([f"Layer {l}: {c}次" for l, c in layer_freq])
        print(f"   {freq_str}")

        print(f"\n3. 性能表现 (KL 散度 - 越低越好):")
        print(f"   - Router 平均 KL 散度: {np.mean(router_kl_list):.4f} (中位数: {np.median(router_kl_list):.4f})")
        print(f"   - 贪心上限 平均 KL 散度: {np.mean(greedy_gt_kl_list):.4f} (中位数: {np.median(greedy_gt_kl_list):.4f})")
        print("="*60)

if __name__ == '__main__':
    main()