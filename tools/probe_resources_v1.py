"""Verify actual CPU affinity and visible GPUs without loading model weights."""
import json
import os

import torch


def main():
    expected_cpus = int(os.environ['NEUROBYPASS_CPU_COUNT'])
    expected_gpus = int(os.environ['NEUROBYPASS_GPU_COUNT'])
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    cpu_ids = sorted(os.sched_getaffinity(0))
    assert len(cpu_ids) == expected_cpus
    assert torch.cuda.device_count() == expected_gpus
    details = []
    for i in range(expected_gpus):
        with torch.cuda.device(i):
            tensor = torch.arange(8, device=f'cuda:{i}')
            assert tensor.sum().item() == 28
            details.append({'logical_index': i, 'name': torch.cuda.get_device_name(i)})
    print(json.dumps({'cpu_count': len(cpu_ids), 'cpu_ids': cpu_ids,
                      'visible_gpu_count': torch.cuda.device_count(),
                      'visible_gpu_uuids': os.environ['CUDA_VISIBLE_DEVICES'],
                      'torch_threads': torch.get_num_threads(), 'gpu_checks': details}, indent=2))


if __name__ == '__main__':
    main()
