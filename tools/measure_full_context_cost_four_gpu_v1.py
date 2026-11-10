"""Full-context deployment rerun: route, evict/load, prefill, decode and memory.

CPU-only startup precedes selective measurement; full-GPU controls come later.
Static physical removal is included as a steady-state control. No source or
checkpoint mutation. Independent requests rebuild full-context caches.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_full_context_four_gpu_v1 import policies
from full_context_core_v1 import load_router, route_prompt, visible_context
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, resources, setup_worker, thread_record,
    verify_preserved, write)
from prompt_cost_core_v1 import Layout, parameter_bytes, trial
from prompt_fixed_core_v1 import FixedRouteSession, bind_cache, continuation_logits
from selective_residency_core_v1 import SelectiveBank
from stage1_benchmark_v2 import sha256, stamp


class ProcessMemorySampler:
    """100ms sampled nvidia-smi process allocation; not an exact device peak."""
    def __init__(self):
        self.stop_event = threading.Event(); self.peak_mib = 0; self.samples = 0; self.failures = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                value = sum(int(line.split(',')[1].strip()) for line in result.stdout.splitlines()
                    if int(line.split(',')[0].strip()) == os.getpid())
                self.peak_mib = max(self.peak_mib, value); self.samples += 1
            except (ValueError, subprocess.SubprocessError): self.failures += 1
            self.stop_event.wait(.1)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop_event.set(); self.thread.join(timeout=6)

    def record(self):
        return {'sampled_process_peak_mib': self.peak_mib, 'samples': self.samples,
            'failures': self.failures, 'interval_seconds': .1}


def request(model, bank, router, ids, policy, new_tokens, selective, reset_peak=True):
    if ids.size(1)+new_tokens > model.config.max_position_embeddings:
        raise ValueError('Base-model and router context reservation differ')
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    if reset_peak: torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    skips = sorted(route_prompt(model, router, ids, policy['k'], policy['prefix'])) if router is not None else []
    torch.cuda.synchronize(); routed = time.perf_counter()
    movement = bank.activate(skips) if selective else {'loaded_layers': [], 'evicted_layers': [],
        'host_to_device_weight_bytes': 0, 'resident_decoder_layers': len(bank.active)}
    switched = time.perf_counter()
    session, generated = FixedRouteSession(model, skips), []
    with torch.inference_mode():
        logits = session.forward(ids)
        token = logits[:, -1].argmax(-1, keepdim=True)
        first_logits = logits[:, -1].float().cpu()
        generated.append(token); del logits
        torch.cuda.synchronize(); first = time.perf_counter()
        for _ in range(1, new_tokens):
            logits = session.forward(token)
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token); del logits
        torch.cuda.synchronize(); end = time.perf_counter()
    length = ids.size(1)+new_tokens-1
    expected = 2*(40-len(skips))*length*5120*2
    if session.cache_bytes() != expected: raise ValueError('Incorrect route-specific cache size')
    try: bind_cache(session.cache, tuple(skips)+(999,))
    except ValueError: pass
    else: raise ValueError('Cache accepted a changed route')
    row = {**movement, 'skipped_layers': skips, 'prompt_tokens': ids.size(1),
        'router_observation_tokens': ids.size(1) if router is not None else 0,
        'new_tokens': new_tokens, 'cached_tokens': length, 'kv_bytes': expected,
        'route_seconds': routed-start, 'weight_switch_seconds': switched-routed,
        'prefill_seconds': first-switched, 'ttft_seconds': first-start,
        'decode_seconds': end-first, 'total_seconds': end-start,
        'decode_tokens_per_second': (new_tokens-1)/(end-first),
        'end_to_end_tokens_per_second': new_tokens/(end-start),
        'model_cuda_parameter_bytes': parameter_bytes(model),
        'router_cuda_parameter_bytes': parameter_bytes(router) if router is not None else 0,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'generated_tokens_sha256': hashlib.sha256(torch.cat(generated, 1).cpu().numpy().tobytes()).hexdigest(),
        'cache_layer_lengths_verified': True, 'cache_route_change_rejected': True}
    del session, token, generated
    return row, first_logits


def worker(rank, config, inbox, messages):
    setup_worker(rank)
    import psutil
    from transformers import AutoModelForCausalLM
    out, device = Path(config['output']), f'cuda:{rank}'
    policy = config['policies'][rank]
    data = Records(config['data_run'])
    candidates = sorted((r for r in data.rows if r['split'] == 'train'),
        key=lambda r: (-r['prompt_length'], r['sample']))[:config['repeats']]
    inputs, lengths = [], []
    for record in candidates:
        array, p = data[record['sample']]
        ids = torch.tensor(array[:min(p, config['lengths'][rank])], device='cpu', dtype=torch.long)[None]
        ids, removed = visible_context(ids, reserve_tokens=config['new_tokens'])
        inputs.append(ids); lengths.append({'sample': record['sample'], 'prompt_tokens': ids.size(1), 'removed': removed})
    if len(inputs) != config['repeats']: raise ValueError('Insufficient timing inputs')
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    with ProcessMemorySampler() as startup_sampler:
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True).eval().requires_grad_(False)
        if any(p.device.type != 'cpu' for p in model.parameters()): raise ValueError('Full model reached GPU before selective startup')
        bank = SelectiveBank(model, device)
        router = load_router(policy['checkpoint']['path'], device)
        torch.cuda.synchronize(); setup_seconds = time.perf_counter()-started
        rows, references = [], []
        for repeat, cpu_ids in enumerate(inputs):
            ids = cpu_ids.to(device)
            with torch.inference_mode():
                row, first = request(model, bank, router, ids, policy, config['new_tokens'], True, reset_peak=repeat != 0)
            references.append(first); rows.append({'deployment': 'selective_cpu_bank', 'repeat': repeat, 'cold': repeat == 0, **row})
            if row['router_observation_tokens'] != ids.size(1): raise ValueError('Selective observation truncated')
            if repeat == 0:
                with torch.inference_mode():
                    reference = continuation_logits(model, torch.cat([ids, ids[:, -1:]], 1), ids.size(1), row['skipped_layers'])[:, 0].cpu()
                if (first-reference).abs().max().item() > 1e-4: raise ValueError('Cache prefill and serial forward differ')
            del ids
        selective_peak = max(row['peak_allocated_bytes'] for row in rows)
    # All fully resident work begins only after selective startup peaks end.
    bank.activate([])
    for deployment, selected_router in [('resident_dynamic', router), ('full', None)]:
        for repeat, cpu_ids in enumerate(inputs):
            ids = cpu_ids.to(device)
            with ProcessMemorySampler() as sampler, torch.inference_mode():
                row, first = request(model, bank, selected_router, ids, policy, config['new_tokens'], False)
            if selected_router is not None:
                own = rows[repeat]
                if row['skipped_layers'] != own['skipped_layers'] or row['generated_tokens_sha256'] != own['generated_tokens_sha256']:
                    raise ValueError('Resident and selective generation differ')
                if (first-references[repeat]).abs().max().item() > 1e-4: raise ValueError('Resident logits differ')
            rows.append({'deployment': deployment, 'repeat': repeat, 'cold': False, 'process_memory': sampler.record(), **row})
            del ids
    # Steady-state physical static control excludes layout preparation, as in
    # stage8. It is not a claim about static cold-start or disk-load latency.
    layout = Layout(model)
    static = {'name': 'static_joint_kl', 'kind': 'static', 'seed': None, 'k': policy['k'],
        'skips': config['global_path'][:policy['k']], 'deployment': 'physical'}
    layout.activate(static)
    for repeat, cpu_ids in enumerate(inputs):
        ids = cpu_ids.to(device)
        with ProcessMemorySampler() as sampler, torch.inference_mode():
            row = trial(model, ids, static, None, config['new_tokens'])
        rows.append({'deployment': 'static_physical', 'repeat': repeat, 'cold': False,
            'prompt_tokens': ids.size(1), 'router_observation_tokens': 0, 'process_memory': sampler.record(), **row})
        del ids
    write(out / f'worker_{rank:03d}.json', {'rank': rank, 'policy': policy, 'inputs': lengths,
        'records': rows, 'cpu_model_load_and_shared_setup_seconds': setup_seconds,
        'selective_startup_peak_allocated_bytes': selective_peak,
        'selective_startup_process_memory': startup_sampler.record(),
        'cpu_master_decoder_weight_bytes': sum(bank.block_bytes), 'process_rss_bytes': psutil.Process().memory_info().rss,
        'full_model_on_gpu_before_selective_startup': False, **thread_record()})
    messages.put({'kind': 'done', 'rank': rank})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--probe', action='store_true'); args = p.parse_args(); resources()
    root, out, training = Path(__file__).resolve().parents[1], Path(os.environ['NEUROBYPASS_RUN_DIR']), args.training_run.resolve()
    tracked = preserve(root, out)
    trained, paths, all_policies = policies(training, [10, 12], [6])
    seed2 = 580 if 580 in trained['seeds'] else 579
    pairs = [(10, 579), (12, 579), (10, seed2), (12, seed2)]
    selected = [next(p for p in all_policies if p['kind'] == 'router' and (p['k'], p['seed']) == pair) for pair in pairs]
    config = {'output': str(out), 'model': trained['model'], 'training_run': str(training),
        'data_run': trained['data_run'], 'policies': selected, 'global_path': paths['static_joint_kl'],
        'lengths': [128, 512, 1024, 2016], 'repeats': 2 if args.probe else 8,
        'new_tokens': 32 if args.probe else 128, 'probe': args.probe,
        'observation': 'Complete same visible prompt for router and base; explicit left truncation reserves generated tokens.',
        'limits': ['Independent requests, not a real multi-turn task; every new route rebuilds the full-context KV.',
            'Static physical preparation excluded; selective request copies included.',
            '100ms nvidia-smi sampled process peak is a lower bound on true process peak.',
            'Timing inputs are training-derived natural text; no quality conclusion from cost samples.'],
        'source_sha256': {str(q): sha256(q) for q in [Path(__file__), root/'tools/full_context_core_v1.py', root/'tools/selective_residency_core_v1.py']}}
    write(out / 'protocol.json', config)
    with Workers(worker, config) as workers: workers.receive(4, 'done')
    records = []
    for rank in range(4):
        worker_row = json.loads((out / f'worker_{rank:03d}.json').read_text())
        assert worker_row['full_model_on_gpu_before_selective_startup'] is False
        assert len(worker_row['records']) == config['repeats']*4
        records.extend(worker_row['records'])
    write(out / 'verification.json', {'passed': True, 'requests': len(records),
        'selective_resident_routes_logits_generation_checked': True, 'full_observation_checked': True})
    verify_preserved(out, tracked)
    stamp('full_context_cost_complete', requests=len(records))


if __name__ == '__main__': main()
