#!/bin/bash
# 限制当前脚本及子进程只对 0, 1, 2, 3 号 GPU 可见
# 这是一层额外的安全锁，确保绝对不会占用别人的 4 号和 5 号 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "开始启动 4 个 Worker，限制使用前 32 个 CPU 核心..."

# 使用 taskset 严格限制每个进程使用的 CPU 核心
# taskset -c 0-7 表示该进程只能在 0 到 7 号这 8 个核心上运行

# 终端 1 (Worker 0 -> GPU 0, CPU 0-7)
taskset -c 0-7 python generate_greedy_labels_multi_v3.py --worker_id 0 --world_size 4 --gpu_id 0 &

# 终端 2 (Worker 1 -> GPU 1, CPU 8-15)
taskset -c 8-15 python generate_greedy_labels_multi_v3.py --worker_id 1 --world_size 4 --gpu_id 1 &

# 终端 3 (Worker 2 -> GPU 2, CPU 16-23)
taskset -c 16-23 python generate_greedy_labels_multi_v3.py --worker_id 2 --world_size 4 --gpu_id 2 &

# 终端 4 (Worker 3 -> GPU 3, CPU 24-31)
taskset -c 24-31 python generate_greedy_labels_multi_v3.py --worker_id 3 --world_size 4 --gpu_id 3 &

echo "所有 4 个 Worker 已成功在后台启动！"
echo "正在等待数据生成完成 (期间请勿直接关闭当前进程)..."
wait
echo "太棒了！所有进程均已运行完毕。"