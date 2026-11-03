import lm_eval
import json
import gc          # 导入 gc 模块
import torch       # 确保导入了 torch
from my_router_model import UniversalPrunedOPT

def main():
    # 基础配置
    MODEL_PATH = "/home/liushenyu/opt-13b/"
    TASKS = ["winogrande", "race"] # 为了测试速度，先把 MMLU 拿掉。等测通了再把 "mmlu" 加进列表
    K_TO_SKIP = 6
    
    # 定义你要对比的所有策略
    strategies_config = {
        "Baseline": {
            "strategy": "baseline"
        },
        "Static_Naive": {
            "strategy": "static",
            "skip_layers": [11, 10, 9, 18, 5, 37]
        },
        "Static_ShortGPT": {
            "strategy": "static",
            "skip_layers": [3, 7, 12, 15, 22, 28] 
        },
        "Dynamic_SFT": {
            "strategy": "sft",
            "router_weight_path": "router_sft_epoch_v3_10.pth" 
        },
        "Dynamic_RL": {
            "strategy": "rl",
            "router_weight_path": "router_rl_epoch_v4_4.pth" 
        }
    }

    all_results = {}

    for name, config in strategies_config.items():
        print(f"\n{'='*50}\nEvaluating Strategy: {name}\n{'='*50}")
        
        # 实例化对应的模型
        model = UniversalPrunedOPT(
            pretrained=MODEL_PATH,
            strategy=config.get("strategy"),
            router_weight_path=config.get("router_weight_path"),
            skip_layers=config.get("skip_layers"),
            k_to_skip=K_TO_SKIP,
            batch_size=1
        )
        
        # 运行评测
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=TASKS,
            num_fewshot=0,
            device="cuda:0"
        )
        
        # 打印当前策略表格
        print(f"\nResults for {name}:")
        print(lm_eval.utils.make_table(results))
        
        # 提取核心数据保存 (只取 acc 分数)
        all_results[name] = {task: results["results"][task]["acc,none"] for task in TASKS}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    # 汇总输出最终的精简对比字典
    print("\n\n" + "="*50)
    print("FINAL COMPREHENSIVE RESULTS")
    print("="*50)
    print(json.dumps(all_results, indent=4))

if __name__ == "__main__":
    main()