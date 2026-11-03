#!/bin/bash
# 开启 8 个后台进程，每个进程分派到一个独立的 GPU



# 终端 1
python generate_greedy_labels_multi_v3.py --worker_id 0 --world_size 4 --gpu_id 0 &

# 终端 2
python generate_greedy_labels_multi_v3.py --worker_id 1 --world_size 4 --gpu_id 1 &

# 终端 3
python generate_greedy_labels_multi_v3.py --worker_id 2 --world_size 4 --gpu_id 2 &

# 终端 4
python generate_greedy_labels_multi_v3.py --worker_id 3 --world_size 4 --gpu_id 3 &

echo "All 8 workers started. Waiting for completion..."
wait
echo "Done! All data generated."