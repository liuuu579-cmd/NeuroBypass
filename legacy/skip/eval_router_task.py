import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset # 新增依赖
from model_router import RouterTransformer

# --- ShortGPT 的核心计算函数 ---
def block_influence(input_hidden_state: torch.Tensor, output_hidden_state: torch.Tensor, angular=False):
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim
def calculate_choice_logprob(model, input_ids, context_length):
    """
    计算特定选项的对数似然得分 (Log-Likelihood)
    只对 Choice 部分的 token 计算交叉熵/对数概率
    """
    with torch.no_grad():
        outputs = model(input_ids)
        # Shift logits and labels for next token prediction
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        # 定位到 choice 部分的 logits 和 labels
        # context_length - 1 是因为我们要用 context 的最后一个 token 来预测 choice 的第一个 token
        choice_logits = logits[:, context_length - 1:, :]
        choice_labels = labels[:, context_length - 1:]
        
        log_probs = F.log_softmax(choice_logits, dim=-1)
        # 提取 target token 对应的 log prob
        gathered_log_probs = torch.gather(log_probs, 2, choice_labels.unsqueeze(-1)).squeeze(-1)
        
        # 通常采用 sum 或者 mean，sum 是标准做法
        return gathered_log_probs.sum().item()
# --- 配置 ---
MODEL_PATH = "/home/liushenyu/opt-13b/"
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = '/home/liushenyu/test/c4_subset_1024mb.txt'

SFT_WEIGHT_PATH = "router_sft_epoch_v3_10.pth"
RL_WEIGHT_PATH = "router_rl_epoch_v4_4.pth"

DEVICE = 'cuda:0'
NUM_OPT_LAYERS = 40
D_MODEL = 5120
K_TO_SKIP = 6
NUM_TEST_SAMPLES = 200
NUM_CALIB_SAMPLES = 10  # ShortGPT 需要少量校准数据来计算层重要性

# 你的 Baseline 静态跳层组合
STATIC_SKIP_LAYERS = [11, 10, 9, 18, 5, 37] #11, 10, 9, 18, 5, 37  迭代贪心    #6, 2, 10, 11, 12, 13  候选池全局最优

class SkipHook:
    def __call__(self, module, args, output):
        if isinstance(output, tuple):
            return (args[0],) + output[1:]
        return args[0]

def main():
    print("Loading OPT-13B Model...")
    opt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    if len(tokenizer) > opt_model.get_input_embeddings().weight.shape[0]:
        opt_model.resize_token_embeddings(len(tokenizer))
    opt_model.eval()
    transformer_layers = opt_model.model.decoder.layers

    print("Loading Routers...")
    sft_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    sft_state = {k.replace('module.', ''): v for k, v in torch.load(SFT_WEIGHT_PATH, map_location=DEVICE).items()}
    sft_router.load_state_dict(sft_state)
    sft_router.eval()

    rl_router = RouterTransformer(input_dim=D_MODEL, num_opt_layers=NUM_OPT_LAYERS, nhead=4, num_encoder_layers=2, d_hid=2048).to(DEVICE)
    rl_state = {k.replace('module.', ''): v for k, v in torch.load(RL_WEIGHT_PATH, map_location=DEVICE).items()}
    rl_router.load_state_dict(rl_state)
    rl_router.eval()

    print("Loading PIQA Dataset...")
    piqa_dataset = load_dataset("piqa", split="validation")
    # 为了测试效率，这里依然取部分样本。真实评估时建议用全部(约1800条)
    test_samples = piqa_dataset.select(range(NUM_TEST_SAMPLES)) 
    
    # 记录各策略预测正确的数量
    correct_counts = {
        "baseline": 0,
        "static": 0,
        "shortgpt": 0,
        "sft": 0,
        "rl": 0
    }

    # ==========================================
    # 2. 计算 ShortGPT 策略 (如果需要，可复用之前的逻辑，或用 PIQA 的前几个样本做 Calibration)
    # 为了简化，这里假设你已经算出了 shortgpt_skip_layers
    # ==========================================
    shortgpt_skip_layers = [3, 7, 12, 15, 22, 28] # 替换为你真实算出来的值
    STATIC_SKIP_LAYERS = [11, 10, 9, 18, 5, 37]

    print(f"\nEvaluating PIQA {NUM_TEST_SAMPLES} samples...")
    for sample in tqdm(test_samples):
        goal = sample['goal']
        sol1 = sample['sol1']
        sol2 = sample['sol2']
        label = sample['label'] # 0 for sol1, 1 for sol2
        
        # 构造 prompt (Context)
        # PIQA 通常格式: "Question: {goal}\nAnswer:"
        context_text = f"Question: {goal}\nAnswer: "
        context_ids = tokenizer(context_text, return_tensors='pt').input_ids.to(DEVICE)
        context_length = context_ids.shape[1]
        
        # 构造两个选项的完整输入
        text1 = context_text + sol1
        text2 = context_text + sol2
        input_ids_1 = tokenizer(text1, return_tensors='pt').input_ids.to(DEVICE)
        input_ids_2 = tokenizer(text2, return_tensors='pt').input_ids.to(DEVICE)
        
        embeddings_1 = opt_model.get_input_embeddings()(input_ids_1).to(torch.float16)
        embeddings_2 = opt_model.get_input_embeddings()(input_ids_2).to(torch.float16)
        
        # 定义一个评估当前策略的辅助函数
        def evaluate_strategy(skip_layers_1, skip_layers_2=None):
            # 允许对两个选项应用不同的动态层 (因为序列变了)
            if skip_layers_2 is None: skip_layers_2 = skip_layers_1
            
            # Choice 1
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in skip_layers_1]
            score1 = calculate_choice_logprob(opt_model, input_ids_1, context_length)
            for h in handles: h.remove()
            
            # Choice 2
            handles = [transformer_layers[l].register_forward_hook(SkipHook()) for l in skip_layers_2]
            score2 = calculate_choice_logprob(opt_model, input_ids_2, context_length)
            for h in handles: h.remove()
            
            return 0 if score1 > score2 else 1

        with torch.no_grad():
            # 1. Baseline (不跳层)
            pred_base = evaluate_strategy([])
            if pred_base == label: correct_counts["baseline"] += 1

            # 2. Static Pruning
            pred_static = evaluate_strategy(STATIC_SKIP_LAYERS)
            if pred_static == label: correct_counts["static"] += 1

            # 3. ShortGPT Pruning
            pred_sgpt = evaluate_strategy(shortgpt_skip_layers)
            if pred_sgpt == label: correct_counts["shortgpt"] += 1

            # 4. SFT Router (Dynamic) - 为每个选项动态计算要跳过的层
            _, sft_idx_1 = torch.topk(sft_router(embeddings_1), K_TO_SKIP, dim=1, largest=False)
            _, sft_idx_2 = torch.topk(sft_router(embeddings_2), K_TO_SKIP, dim=1, largest=False)
            pred_sft = evaluate_strategy(sft_idx_1[0].tolist(), sft_idx_2[0].tolist())
            if pred_sft == label: correct_counts["sft"] += 1

            # 5. RL Router (Dynamic)
            _, rl_idx_1 = torch.topk(rl_router(embeddings_1), K_TO_SKIP, dim=1, largest=False)
            _, rl_idx_2 = torch.topk(rl_router(embeddings_2), K_TO_SKIP, dim=1, largest=False)
            pred_rl = evaluate_strategy(rl_idx_1[0].tolist(), rl_idx_2[0].tolist())
            if pred_rl == label: correct_counts["rl"] += 1

    # --- 统计与画图 ---
    acc_baseline = correct_counts["baseline"] / NUM_TEST_SAMPLES
    acc_static = correct_counts["static"] / NUM_TEST_SAMPLES
    acc_shortgpt = correct_counts["shortgpt"] / NUM_TEST_SAMPLES
    acc_sft = correct_counts["sft"] / NUM_TEST_SAMPLES
    acc_rl = correct_counts["rl"] / NUM_TEST_SAMPLES

    print("\n" + "="*50)
    print(f"FINAL PIQA ZERO-SHOT ACCURACY ({NUM_TEST_SAMPLES} samples)")
    print("="*50)
    print(f"Baseline (No Pruning)  : {acc_baseline:.2%}")
    print(f"Static Pruning (Naive) : {acc_static:.2%}")
    print(f"ShortGPT (Static BI)   : {acc_shortgpt:.2%}")
    print(f"Dynamic SFT Router     : {acc_sft:.2%}")
    print(f"Dynamic RL Router      : {acc_rl:.2%}")
    print("="*50)

    # 画柱状图
    labels = ['Baseline\n(Full Model)', 'Naive Static\n(Manual)', 'ShortGPT\n(Static BI)', 'Dynamic\n(SFT)', 'Dynamic\n(RL)']
    accuracies = [acc_baseline, acc_static, acc_shortgpt, acc_sft, acc_rl]
    colors = ['#cccccc', '#ff9999', '#ffcc99', '#66b3ff', '#99ff99']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=colors, edgecolor='black')
    
    plt.ylabel('PIQA Accuracy (Higher is Better)', fontsize=12)
    plt.title(f'PIQA Performance (Skipping {K_TO_SKIP} Layers on OPT-13B)', fontsize=14)
    # 设置 Y 轴范围稍微比最大最小准确率宽一点，为了美观
    plt.ylim(min(accuracies) * 0.9, min(max(accuracies) * 1.1, 1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('piqa_evaluation_results.png', dpi=300)
    print("Plot saved to 'piqa_evaluation_results.png'")

if __name__ == "__main__":
    main()