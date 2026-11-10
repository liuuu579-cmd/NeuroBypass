"""Small feasibility pilot; transfer is included and low-memory startup is real."""
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from horizontal_quality_v3 import pin_threads
from prefix_downstream_correction_core_v1 import policies
from prompt_cost_core_v1 import cached_logits
from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import fixed_route
from selective_residency_core_v1 import SelectiveBank, request, route
from stage1_benchmark_v2 import sha256, stamp, write_json


def worker(rank, config, messages):
    try:
        import psutil
        from transformers import AutoModelForCausalLM
        cpus = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))[rank*10:(rank+1)*10]
        pin_threads(cpus); torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        # Initialize allocator tracking before loading any model weights.
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        model.requires_grad_(False)
        if any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('Full model was placed on GPU during startup')
        bank = SelectiveBank(model, f'cuda:{rank}')
        policy = config['policies'][rank]
        router = DepthRouter().to(f'cuda:{rank}').eval().requires_grad_(False)
        router.load_state_dict(torch.load(policy['checkpoint']['path'], map_location='cpu', weights_only=True)['state_dict'])
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter()-started
        data = np.load(config['private_inputs'], allow_pickle=False)
        length = config['lengths'][rank]
        records, first_logits = [], []
        for repetition in range(4):
            ids = torch.tensor(data[repetition, :length], dtype=torch.long, device=f'cuda:{rank}')[None]
            row, logits = request(model, bank, router, ids, policy, 32, selective=True,
                                  reset_peak=repetition != 0)
            records.append({'deployment': 'selective_cpu_bank', 'repetition': repetition,
                'cold': repetition == 0, **row})
            first_logits.append(logits)
            messages.put({'kind': 'progress', 'rank': rank, 'phase': 'selective', 'repetition': repetition})
        low_phase_peak = max(r['peak_allocated_bytes'] for r in records)
        # This is an exact appended-prefix observation check, not a chat quality
        # test. The context grows but the router still reads the first128tokens.
        grown = torch.tensor(data[0, :256], dtype=torch.long, device=f'cuda:{rank}')[None]
        with torch.inference_mode():
            before = route(model, router, grown[:, :128], policy)
            after = route(model, router, grown, policy)
        if before != after:
            raise ValueError('Deterministic first128-token route changed unexpectedly')
        selective_checks = []
        # Probe cache numerics while only required weights are resident.
        bank.activate(before)
        with torch.inference_mode():
            ids = grown[:, :160]
            actual = cached_logits(model, ids, before)
            with fixed_route(model, before):
                reference = model(ids[:, :-1], use_cache=False).logits[:, 127:].float()
            import torch.nn.functional as F
            p, q = F.log_softmax(reference, -1), F.log_softmax(actual, -1)
            error = (p.exp()*(p-q)).sum(-1).abs()
            check = {'mean_absolute_position_kl': error.mean().item(),
                     'max_absolute_position_kl': error.max().item()}
            if check['mean_absolute_position_kl'] > 1e-4 or check['max_absolute_position_kl'] > 1e-3:
                raise ValueError('Selective cache failed existing FP16 numeric tolerances')
            selective_checks.append(check)
            del actual, reference, p, q, error, ids
        # Fully resident controls start only AFTER all selective startup and
        # request peaks have been captured. They use identical CPU masters.
        bank.activate([])
        for repetition in range(4):
            ids = torch.tensor(data[repetition, :length], dtype=torch.long, device=f'cuda:{rank}')[None]
            row, logits = request(model, bank, router, ids, policy, 32, selective=False)
            own = records[repetition]
            if row['skipped_layers'] != own['skipped_layers'] or row['generated_tokens_sha256'] != own['generated_tokens_sha256']:
                raise ValueError('Selective and resident routes/generation differ')
            error = (logits-first_logits[repetition]).abs().max().item()
            if error > 1e-4:
                raise ValueError('Selective and resident prefill logits differ')
            records.append({'deployment': 'all_weights_resident_dynamic', 'repetition': repetition,
                'cold': False, 'selective_max_logit_error': error, **row})
        for repetition in range(4):
            ids = torch.tensor(data[repetition, :length], dtype=torch.long, device=f'cuda:{rank}')[None]
            row, _ = request(model, bank, None, ids, policy, 32, selective=False)
            records.append({'deployment': 'full_model', 'repetition': repetition, 'cold': False, **row})
        result = {'rank': rank, 'policy': policy, 'length': length, 'records': records,
            'model_cpu_load_and_shared_gpu_setup_seconds': setup_seconds,
            'cpu_master_decoder_weight_bytes': sum(bank.block_bytes),
            'final_process_rss_bytes': psutil.Process().memory_info().rss,
            'selective_startup_and_first_four_requests_peak_allocated_bytes': low_phase_peak,
            'full_gpu_load_before_selective_measurements': False,
            'append_same_first128_route_unchanged': True, 'cache_probes': selective_checks,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': 10,
            'thread_cpu_ids': {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}}
        write_json(Path(config['output']) / f'worker_{rank:03d}.json', result)
        messages.put({'kind': 'done', 'rank': rank})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def main():
    if os.environ.get('NEUROBYPASS_GPU_COUNT') != '4':
        raise ValueError('Use the four-GPU resource launcher')
    root, output = Path(__file__).resolve().parents[1], Path(os.environ['NEUROBYPASS_RUN_DIR'])
    _, all_policies = policies(root)
    wanted = [(10, 579), (12, 579), (10, 580), (12, 580)]
    selected = [next(p for p in all_policies if p['name'] == 'm06' and (p['k'], p['seed']) == pair) for pair in wanted]
    source = root / 'runs/stage8_cost_main_20261108_01'
    old = json.loads((source / 'protocol.json').read_text())
    if sha256(old['private_inputs']) != old['input_sha256']:
        raise ValueError('Timing inputs changed')
    sources = [Path(__file__), root / 'tools/selective_residency_core_v1.py',
        root / 'tools/prompt_cost_core_v1.py', root / 'tools/prompt_fixed_core_v1.py',
        root / 'tools/prompt_depth_core_v1.py']
    config = {'model': '/home/liushenyu/opt-13b', 'output': str(output), 'policies': selected,
        'private_inputs': old['private_inputs'], 'private_input_sha256': old['input_sha256'],
        'lengths': [128, 512, 1024, 512], 'new_tokens': 32, 'repeats': 4,
        'source_sha256': {str(p): sha256(p) for p in sources},
        'stage': 'Engineering feasibility pilot, not final throughput/latency benchmark',
        'timing': 'Each independent request includes route, evict/load and full prefill. CPU master-to-GPU copies include deepcopy overhead. EOS ignored.',
        'memory': 'CPU-only load then selective GPU residency; capture startup/request peaks before any full-GPU control. PyTorch allocated/reserved; not NVML process peak.',
        'limits': ['Four requests per card; not enough for stable throughput/P95 conclusions.',
            'Independent training-derived timing prompts, not real multi-turn conversation.',
            'All weights kept in CPU memory; no disk streaming, pinned-memory optimization, batching or concurrency.',
            'Static physical pruning retains its own memory/latency advantages and is not rebenchmarked here.']}
    tracked = {str(p): sha256(p) for p in root.rglob('*') if p.is_file() and not p.is_relative_to(output)
        and '.git' not in p.parts and p.suffix in {'.py', '.json', '.jsonl', '.md', '.csv', '.png'}}
    for p in root.parent.glob('*.py'):
        tracked[str(p)] = sha256(p)
    write_json(output / 'preservation_before.json', tracked)
    write_json(output / 'protocol.json', config)
    context = mp.get_context('spawn'); messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, config, messages)) for rank in range(4)]
    done = set()
    try:
        for p in processes:
            p.start()
        pin_threads(list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))[40:42])
        while len(done) < 4:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Selective-residency worker exited early')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            if item['kind'] == 'done':
                done.add(item['rank'])
            else:
                stamp('selective_residency_progress', **{k: v for k, v in item.items() if k != 'kind'})
        for p in processes:
            p.join(timeout=30)
            if p.exitcode != 0:
                raise RuntimeError('Selective-residency worker failed')
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate(); p.join(timeout=10)
            if p.is_alive():
                p.kill(); p.join(timeout=5)
    for p, expected in tracked.items():
        if sha256(p) != expected:
            raise ValueError('Existing evidence changed: ' + p)
    write_json(output / 'preservation_after.json', {'preserved_files': len(tracked)})
    stamp('selective_residency_pilot_complete', workers=4, requests=48)


if __name__ == '__main__':
    main()
