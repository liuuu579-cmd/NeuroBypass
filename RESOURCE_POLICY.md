# GPU 与 CPU 配额入口

所有后续实验统一通过 **`tools/run_with_resources_v2.py`** 启动。现有源程序保持原样，后续算法和评测变化只新增独立版本。

资源规则为 `CPU 核心数 = floor(服务器 CPU 总数 × 本任务 GPU 数 / 服务器 GPU 总数)`。当前服务器是 64 个物理 CPU 核心、6 块 RTX 6000 Ada：

| GPU | CPU 核心上限（整个实验进程树合计） |
| ---: | ---: |
| 1 | 10 |
| 2 | 21 |
| 3 | 32 |
| 4 | 42 |
| 5 | 53 |
| 6 | 64 |

```bash
cd /home/liushenyu/test/NeuroBypass

# 查看计划。
python tools/run_with_resources_v2.py --gpus 1,3 --dry-run

# 启动新程序，限制在两卡与 21 核以内。
python tools/run_with_resources_v2.py \
  --gpus 1,3 --run-dir runs/new_experiment_001 -- \
  /home/liushenyu/anaconda3/envs/traindata/bin/python your_new_program.py
```

`--gpus` 使用物理 nvidia-smi 编号，子程序看到的逻辑编号从 0 开始。入口限制 CPU affinity 和常见线程池，并为经该入口启动的同用户任务分配不重叠的 GPU/CPU。它检查忙卡，但不控制其他人的任务。启动两卡可见性后，仍需程序显式使用两卡或启动两个 worker。

运行目录必须全新，里面记录 `allocation.json`、`stdout.log` 和 `status.json`。两卡真实探针和四项针对性测试已通过；本次验证没有启动正式训练。

详细实验顺序、比较协议和扩大数据的条件见 [下一阶段推进方案](analysis/NEXT_STEPS_AND_RESOURCES_20261103.zh-CN.md)。先用 **2 GPU / 21 CPU** 重评已有模型，再做动态收益与输入消融，最后按分组独立切分重新生成训练数据。
