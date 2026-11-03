#!/bin/bash

# 使用 torchrun 启动

# 确保环境变量依然设置
export NCCL_SOCKET_IFNAME=ens6f0
export GLOO_SOCKET_IFNAME=ens6f0

# ======================= 关键调试步骤 =======================
# 将 CUDA 内核启动设置为同步阻塞模式。
# 这会牺牲性能，但能让错误在发生的确切位置被捕获，而不是异步报告。
export CUDA_LAUNCH_BLOCKING=1
# ==========================================================

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train2D.py