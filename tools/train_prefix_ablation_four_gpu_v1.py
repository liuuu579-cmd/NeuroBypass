"""Retrain m=2/4/8 with conditional joint-greedy labels, unchanged stage6 fit.

Run only through run_with_resources_v2.py after stage9 has finished. All labels,
checkpoints and evaluations use exclusive new paths. No downstream task data is
loaded here. The m=0/6/K controls are read from audited stage6 evidence.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizontal_quality_v3 import pin_threads
from prefix_ablation_core_v1 import (BUDGETS, NEW_PREFIX_COUNTS, aggregate,
    arm_name, fork_prefix, read_rows, reuse_steps)
from prompt_depth_core_v1 import validate_steps
from prompt_depth_candidates_serial_v2 import prefix_candidate_costs
from prompt_fixed_core_v1 import continuation_logits, continuation_metrics
from stage1_benchmark_v2 import sha256, stamp, write_json
from train_prompt_depth_four_gpu_v4 import fit, search


def source_paths(source, seeds):
    paths = [source / name for name in ['protocol.json', 'static_paths.json',
        'fixed_global_policy.json', 'verification.json', 'summary.json']]
    for pattern in ['pure_labels_*.jsonl', 'hybrid_labels_*.jsonl', 'pure_branches_*.jsonl',
            'hybrid_branches_*.jsonl', 'static_quality_*.jsonl']:
        paths.extend(sorted(source.glob(pattern)))
    for arm in ['pure', 'hybrid']:
        for seed in seeds:
            for suffix in ['history.json', 'quality.jsonl', 'routes.json', 'checkpoints.json']:
                paths.append(source / f'{arm}_seed{seed}_{suffix}')
            for info in json.loads((source / f'{arm}_seed{seed}_checkpoints.json').read_text()).values():
                paths.append(Path(info['path']))
    return sorted(set(paths))


def generate(rank, model, data, config, messages):
    source, output = Path(config['source_run']), Path(config['output'])
    old = defaultdict(list)
    for pattern in ['pure_labels_*.jsonl', 'hybrid_labels_*.jsonl',
            'pure_branches_*.jsonl', 'hybrid_branches_*.jsonl']:
        for path in sorted(source.glob(pattern)):
            for record in read_rows(path):
                old[record['sample']].append(record)
    samples = sorted(config['train_ids'] + config['validation_ids'])[rank::4]
    train = set(config['train_ids'])
    handles, branches = {}, {}
    for m in NEW_PREFIX_COUNTS:
        arm = arm_name(m)
        handles[m] = (output / f'{arm}_labels_{rank:03d}.jsonl').open('x')
        branches[m] = (output / f'{arm}_branches_{rank:03d}.jsonl').open('x')
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for offset, sample in enumerate(samples):
                ids = torch.tensor(data[sample], device=next(model.parameters()).device, dtype=torch.long)[None]
                teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
                cached = list(old[sample])
                for m in NEW_PREFIX_COUNTS:
                    prefix = config['global_path'][:m]
                    record = search(model, ids, teacher, prefix, 24, reuse_steps(cached, sample))
                    record.update({'sample': sample, 'split': 'train' if sample in train else 'validation',
                        'arm': arm_name(m), 'm': m})
                    handles[m].write(json.dumps(record) + '\n'); handles[m].flush()
                    cached.append(record)
                    if sample in train:
                        initial, alternative = fork_prefix(record, sample, config['label_seed'], prefix)
                        branch = search(model, ids, teacher, initial, 24, reuse_steps(cached, sample))
                        branch.update({'sample': sample, 'split': 'train', 'arm': arm_name(m), 'm': m,
                            'alternative_rank': alternative, 'expert_prefix_length': len(initial)})
                        branches[m].write(json.dumps(branch) + '\n'); branches[m].flush()
                        cached.append(branch)
                messages.put({'kind': 'progress', 'phase': 'labels', 'rank': rank,
                    'completed_documents': offset + 1, 'total_documents': len(samples),
                    'elapsed_seconds': time.monotonic() - started})
    finally:
        for stream in [*handles.values(), *branches.values()]:
            stream.close()
    return {'kind': 'labels_done', 'rank': rank, 'documents': len(samples)}


def worker(rank, config, inbox, messages):
    try:
        allowed = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
        worker_cpus = allowed[rank * 10:(rank + 1) * 10]
        pin_threads(worker_cpus)
        torch.set_num_threads(10); torch.set_num_interop_threads(1)
        pin_threads(worker_cpus); torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config['base_model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
        data = np.load(config['data_file'], allow_pickle=False)
        sample = config['train_ids'][rank]
        ids = torch.tensor(data[sample], device=f'cuda:{rank}', dtype=torch.long)[None]
        candidate_probes = []
        with torch.inference_mode():
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
            from prompt_fixed_candidate_batch_v1 import batched_candidate_costs
            for m in NEW_PREFIX_COUNTS:
                prefix = config['global_path'][:m]
                candidates = [i for i in range(2, 40) if i not in prefix][:2]
                costs = prefix_candidate_costs(model, ids, prefix, candidates, 'kl', teacher)
                reference = batched_candidate_costs(model, ids, 128, prefix, candidates, teacher, 1)
                error = max(abs(a[1] - b[1]) for a, b in zip(costs, reference))
                if error > 1e-6:
                    raise ValueError('Shared-prefix candidate scoring mismatch')
                candidate_probes.append({'m': m, 'prefix': prefix, 'costs': costs,
                    'reference': reference, 'max_abs_error': error, 'tolerance': 1e-6})
            # Replays establish compatibility before reusing old development endpoints.
            sample = config['validation_ids'][rank % len(config['validation_ids'])]
            ids = torch.tensor(data[sample], device=f'cuda:{rank}', dtype=torch.long)[None]
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
            endpoint_replays = []
            for arm in ['pure', 'hybrid']:
                for seed in config['seeds']:
                    records = read_rows(Path(config['source_run']) / f'{arm}_seed{seed}_quality.jsonl')
                    for k in BUDGETS:
                        ref = next(r for r in records if r['sample'] == sample and r['k'] == k and r['policy'] == 'matched')
                        result = continuation_metrics(continuation_logits(model, ids, 128, ref['skipped_layers']), teacher, ids, 128)
                        errors = {name: abs(result[name] - ref[name]) for name in ['continuation_kl', 'continuation_nll']}
                        if max(errors.values()) > 1e-6:
                            raise ValueError('Stage6 endpoint replay mismatch')
                        endpoint_replays.append({'arm': arm, 'seed': seed, 'k': k,
                            'sample': sample, 'errors': errors, 'tolerance': 1e-6})
        pin_threads(worker_cpus)
        write_json(Path(config['output']) / f'worker_probe_{rank:03d}.json', {
            'rank': rank, 'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads(),
            'thread_cpu_ids': {t.name: sorted(os.sched_getaffinity(int(t.name))) for t in Path('/proc/self/task').iterdir()},
            'candidate_probes': candidate_probes, 'endpoint_replays': endpoint_replays})
        messages.put({'kind': 'ready', 'rank': rank})
        while True:
            task = inbox.get()
            if task is None:
                break
            if task['kind'] == 'generate':
                messages.put(generate(rank, model, data, config, messages))
            elif task['kind'] == 'fit':
                local = {**config, 'fixed_prefix': config['global_path'][:task['m']]}
                outcome = fit(rank, model, data, local, task, messages)
                messages.put({**outcome, 'm': task['m']})
            else:
                raise ValueError('Unknown task')
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def receive(messages, processes):
    while True:
        try:
            item = messages.get(timeout=5)
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                raise RuntimeError('A worker exited before completion')
            continue
        if item['kind'] == 'error':
            raise RuntimeError(item['traceback'])
        if item['kind'] == 'progress':
            stamp('prefix_ablation_progress', **{k: v for k, v in item.items() if k != 'kind'})
            continue
        return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    if (int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 4
            or int(os.environ.get('NEUROBYPASS_CPU_COUNT', '0')) != 42):
        p.error('Use run_with_resources_v2.py with four GPUs / 42 CPU cores')
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    source = root / 'runs/stage6_depth_train_main_20261107_01'
    old = json.loads((source / 'protocol.json').read_text())
    audit = json.loads((source / 'verification.json').read_text())
    if audit['probe'] or audit['old_test_used'] or not audit['verified_quality_rows']:
        raise ValueError('Source is not an audited main experiment')
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    config = {key: old[key] for key in ['base_model', 'data_file', 'data_sha256', 'train_ids',
        'validation_ids', 'selection_ids', 'static_calibration_ids', 'label_seed']}
    config.update({'source_run': str(source), 'output': str(output), 'probe': args.probe,
        'prefix_counts_to_train': list(NEW_PREFIX_COUNTS), 'budgets': list(BUDGETS),
        'global_path': json.loads((source / 'static_paths.json').read_text())['static_joint_kl'],
        'epochs': 1 if args.probe else 24, 'seeds': [579] if args.probe else [579, 580, 581]})
    if args.probe:
        config.update({'train_ids': old['train_ids'][:4], 'validation_ids': old['validation_ids'][:4],
            'selection_ids': [s for s in old['selection_ids'] if s in old['validation_ids'][:4]][:2]})
        if not config['selection_ids']:
            raise ValueError('Probe requires an original selection document')
    if sha256(config['data_file']) != config['data_sha256']:
        raise ValueError('Training inputs changed')
    sources = [Path(__file__), root / 'tools/prefix_ablation_core_v1.py',
        root / 'tools/train_prompt_depth_four_gpu_v4.py', root / 'tools/prompt_depth_core_v1.py',
        root / 'tools/prompt_depth_candidates_serial_v2.py', root / 'tools/train_prompt_fixed_v1.py',
        root / 'tools/prompt_fixed_core_v1.py', root / 'tools/horizontal_quality_v3.py']
    config.update({'source_sha256': {str(path): sha256(path) for path in sources},
        'source_evidence_sha256': {str(path): sha256(path) for path in source_paths(source, config['seeds'])},
        'max_k': 24, 'new_router_max_k': 24, 'old_test_used': False, 'weights_frozen': True,
        'input': 'First128 prompt token+position embeddings only; one route per prompt and continuation.',
        'labels': 'Fresh conditional greedy trajectories for each m; joint continuation KL over32 targets, search every candidate on the complete selected set. Reuse only identical document/input/entire selected set.',
        'training': 'Reuse unchanged stage6 fit: fresh width256/max_k24 router, same seeds, document order, batch16,24epochs, AdamW2e-4, per-document mean softCE+0.25hardCE.',
        'training_depth': 'Keep expert/branch supervision through24 for compatibility with existing m0/m6 endpoints. Only K10/12 checkpoints and evaluation are needed.',
        'branch': 'Same low-cost alternative rule as stage6 hybrid; fork depths m+1,m+5,m+9,m+13, alternate ranks2..5; never change global prefix.',
        'checkpoint_selection': 'Original fixed C4 selection32, lowest per-K KL at epoch1,4,...24. No benchmark score chooses m or checkpoints.',
        'endpoints': 'Audited stage6 m0/m6 and static mK reused; GPU replay before reuse. Static endpoint is one path, not three independent models.',
        'limits': ['C4 validation100 already used for development; other68 are not an independent test.',
            'Intermediate m has fewer supervised states by construction; report state counts and equal document/update weighting.',
            'This stage is development quality and training, not downstream benchmark or timing.',
            'No restoration of OPT weights and no complete ShortGPT/Shortened paper reproduction.']})
    write_json(output / 'protocol.json', config)
    tracked = {str(path): sha256(path) for path in root.rglob('*') if path.is_file()
        and not path.is_relative_to(output) and '.git' not in path.parts
        and path.suffix in {'.py', '.md', '.json', '.jsonl', '.csv', '.png'}}
    for path in root.parent.glob('*.py'):
        tracked[str(path)] = sha256(path)
    write_json(output / 'preservation_before.json', tracked)
    context = mp.get_context('spawn')
    messages = context.Queue(); inboxes = [context.Queue() for _ in range(4)]
    processes = [context.Process(target=worker, args=(rank, config, inboxes[rank], messages)) for rank in range(4)]
    outcomes = []
    try:
        for process in processes:
            process.start()
        allowed = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
        pin_threads(allowed[40:42])
        for _ in range(4):
            if receive(messages, processes)['kind'] != 'ready':
                raise ValueError('Unexpected startup message')
        for inbox in inboxes:
            inbox.put({'kind': 'generate'})
        for _ in range(4):
            if receive(messages, processes)['kind'] != 'labels_done':
                raise ValueError('Unexpected label phase message')
        for m in NEW_PREFIX_COUNTS:
            labels = [r for path in sorted(output.glob(f'{arm_name(m)}_labels_*.jsonl')) for r in read_rows(path)]
            branches = [r for path in sorted(output.glob(f'{arm_name(m)}_branches_*.jsonl')) for r in read_rows(path)]
            if sorted(r['sample'] for r in labels) != sorted(config['train_ids'] + config['validation_ids']):
                raise ValueError('Incomplete expert labels')
            if sorted(r['sample'] for r in branches) != config['train_ids']:
                raise ValueError('Incomplete training branches')
            for row in labels:
                validate_steps(row, config['global_path'][:m], 24)
            for row in branches:
                validate_steps(row, row['initial_prefix'], 24)
        jobs = iter([{'kind': 'fit', 'arm': arm_name(m), 'm': m, 'seed': seed}
            for seed in config['seeds'] for m in NEW_PREFIX_COUNTS])
        pending = set()
        for rank in range(4):
            task = next(jobs, None)
            if task is not None:
                inboxes[rank].put(task); pending.add(rank)
        while pending:
            item = receive(messages, processes)
            if item['kind'] != 'fit_done' or item['rank'] not in pending:
                raise ValueError('Unexpected fit completion')
            outcomes.append(item); rank = item['rank']
            task = next(jobs, None)
            if task is None:
                pending.remove(rank)
            else:
                inboxes[rank].put(task)
        for inbox in inboxes:
            inbox.put(None)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Worker failed at shutdown')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
            if process.is_alive():
                process.kill(); process.join(timeout=5)
    for path, digest in {**tracked, **config['source_evidence_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Existing evidence changed: ' + path)
    summary = aggregate(output, config)
    write_json(output / 'training_outcomes.json', outcomes)
    write_json(output / 'summary.json', {**summary, 'probe': args.probe,
        'elapsed_seconds': time.monotonic() - started})
    write_json(output / 'preservation_after.json', {'preserved_files': len(tracked)})
    stamp('prefix_ablation_complete', elapsed_seconds=time.monotonic() - started,
        quality_rows=summary['quality_rows_including_reused_controls'])


if __name__ == '__main__':
    main()
