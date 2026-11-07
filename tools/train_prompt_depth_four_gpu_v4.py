"""Four-GPU deeper experiment v4: exact batch-one shared-prefix candidate tails.

All files are new. Four independent frozen OPT replicas share document search;
then three seed rounds train pure/hybrid routers in parallel. No old test rows
are used, and all inference routes depend only on prompt embeddings.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_depth_cache_v1 import audit_route
from diagnose_prompt_router_v2 import derangement, paired_summary
from prompt_depth_core_v1 import DepthRouter, document_loss, fork_prefix, pack_record, validate_steps
from prompt_depth_candidates_serial_v2 import prefix_candidate_costs
from prompt_fixed_core_v1 import best_candidate, continuation_logits, continuation_metrics, raw_prompt_embeddings
from stage1_benchmark_v2 import sha256, stamp, write_json


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def search(model, ids, teacher, initial, maximum, old=None):
    selected, steps = list(initial), []
    lookup = {tuple(sorted(s['previous_skips'])): s for s in (old or {}).get('steps', [])}
    for number in range(len(selected) + 1, maximum + 1):
        candidates = [i for i in range(2, 40) if i not in selected]
        cached = lookup.get(tuple(sorted(selected)))
        costs = cached['candidate_costs'] if cached else prefix_candidate_costs(
            model, ids, selected, candidates, 'kl', teacher)
        chosen, value = best_candidate(costs)
        steps.append({'step': number, 'previous_skips': selected.copy(),
            'selected_layer': chosen, 'joint_continuation_kl': value,
            'candidate_costs': costs, 'reused_old_entire_state': cached is not None})
        selected.append(chosen)
    record = {'initial_prefix': list(initial), 'ordered_skips': selected, 'steps': steps}
    validate_steps(record, initial, maximum)
    return record


def generate(rank, model, data, config, messages):
    output, device = Path(config['output']), next(model.parameters()).device
    old = {r['sample']: r for p in sorted(Path(config['main_run']).glob('labels_*.jsonl')) for r in lines(p)}
    handles = {arm: (output / f'{arm}_labels_{rank:03d}.jsonl').open('x') for arm in ['pure', 'hybrid']}
    branches = {arm: (output / f'{arm}_branches_{rank:03d}.jsonl').open('x') for arm in ['pure', 'hybrid']}
    samples = sorted(config['train_ids'] + config['validation_ids'])[rank::4]
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for offset, sample in enumerate(samples):
                ids = torch.tensor(data[sample], device=device, dtype=torch.long)[None]
                teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
                for arm in ['pure', 'hybrid']:
                    initial = [] if arm == 'pure' else config['fixed_prefix']
                    record = search(model, ids, teacher, initial, 24, old[sample] if arm == 'pure' else None)
                    record.update({'sample': sample, 'split': old[sample]['split'], 'arm': arm})
                    handles[arm].write(json.dumps(record) + '\n'); handles[arm].flush()
                    if sample in set(config['train_ids']):
                        prefix, alternative = fork_prefix(record, arm, sample, config['label_seed'], config['fixed_prefix'])
                        branch = search(model, ids, teacher, prefix, 24, record)
                        branch.update({'sample': sample, 'split': 'train', 'arm': arm,
                            'alternative_rank': alternative, 'expert_prefix_length': len(prefix)})
                        branches[arm].write(json.dumps(branch) + '\n'); branches[arm].flush()
                messages.put({'kind': 'progress', 'phase': 'labels', 'rank': rank,
                    'completed_documents': offset + 1, 'total_documents': len(samples),
                    'elapsed_seconds': time.monotonic() - started})
    finally:
        for handle in list(handles.values()) + list(branches.values()): handle.close()
    return {'kind': 'labels_done', 'rank': rank, 'documents': len(samples)}


def static_costs(model, data, config, task):
    total = np.zeros(len(task['candidates']), dtype=np.float64)
    samples = config['static_calibration_ids']
    with torch.inference_mode():
        for sample in samples:
            ids = torch.tensor(data[sample], device=next(model.parameters()).device, dtype=torch.long)[None]
            if task['objective'] == 'kl':
                teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
                costs = prefix_candidate_costs(model, ids, task['selected'], task['candidates'], 'kl', teacher)
            else:
                costs = prefix_candidate_costs(model, ids, task['selected'], task['candidates'], 'nll')
            total += np.asarray([value for _, value in costs]) / len(samples)
    return [[i, float(v)] for i, v in zip(task['candidates'], total)]


def padded_training(records, branches, samples):
    packed = [pack_record(records[s], branches[s]) for s in samples]
    slots = max(row[0].shape[0] for row in packed)
    states = torch.zeros(len(samples), slots, 40)
    targets = torch.zeros_like(states)
    valid = torch.ones(len(samples), slots, 40, dtype=torch.bool)
    chosen = torch.zeros(len(samples), slots, dtype=torch.long)
    present = torch.zeros(len(samples), slots)
    for i, row in enumerate(packed):
        n = row[0].shape[0]
        states[i, :n], targets[i, :n] = torch.from_numpy(row[0]), torch.from_numpy(row[1])
        valid[i, :n], chosen[i, :n] = torch.from_numpy(row[2]), torch.from_numpy(row[3])
        present[i, :n] = 1
    return states, targets, valid, chosen, present


def fit(rank, model, data, config, task, messages):
    output, arm, seed = Path(config['output']), task['arm'], task['seed']
    records = {r['sample']: r for p in sorted(output.glob(f'{arm}_labels_*.jsonl')) for r in lines(p)}
    branches = {r['sample']: r for p in sorted(output.glob(f'{arm}_branches_*.jsonl')) for r in lines(p)}
    samples, device = config['train_ids'], next(model.parameters()).device
    tensors = padded_training(records, branches, samples)
    embeddings = torch.empty(len(samples), 128, 5120, dtype=torch.float16)
    with torch.inference_mode():
        for offset in range(0, len(samples), 8):
            group = samples[offset:offset + 8]
            ids = torch.tensor(data[group, :128], device=device, dtype=torch.long)
            embeddings[offset:offset + len(group)].copy_(raw_prompt_embeddings(model, ids).cpu())
    torch.manual_seed(seed)
    router = DepthRouter().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    updates = math.ceil(len(samples) / 16)
    total = updates * config['epochs']
    def schedule(n):
        warm = max(1, total // 20)
        return (n + 1) / warm if n < warm else .1 + .9 * .5 * (1 + math.cos(math.pi * (n - warm) / max(1, total - warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    prefix = [] if arm == 'pure' else config['fixed_prefix']
    best, checkpoints, history = {}, {}, []
    validation_teachers = {}
    with torch.inference_mode():
        for sample in config['selection_ids']:
            ids = torch.tensor(data[sample], device=device, dtype=torch.long)[None]
            validation_teachers[sample] = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
    for epoch in range(1, config['epochs'] + 1):
        router.train()
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + epoch * 10)).tolist()
        loss_sum = 0.
        for offset in range(0, len(order), 16):
            group = order[offset:offset + 16]
            states, targets, valid, chosen, present = [t[group].to(device) for t in tensors]
            optimizer.zero_grad(set_to_none=True)
            scores = router(embeddings[group].to(device), states, states.sum(-1).long() + 1)
            loss = document_loss(scores, targets, valid, chosen, present)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
            optimizer.step(); scheduler.step(); loss_sum += loss.item()
        row = {'epoch': epoch, 'mean_loss': loss_sum / updates}
        if epoch == 1 or epoch % 4 == 0 or epoch == config['epochs']:
            router.eval()
            metrics = {k: [] for k in config['budgets']}
            with torch.inference_mode():
                for sample in config['selection_ids']:
                    ids = torch.tensor(data[sample], device=device, dtype=torch.long)[None]
                    path = router.route(raw_prompt_embeddings(model, ids[:, :128]), 24, prefix)[0].tolist()
                    for k in config['budgets']:
                        value = continuation_metrics(continuation_logits(model, ids, 128, path[:k]), validation_teachers[sample], ids, 128)
                        metrics[k].append(value['continuation_kl'])
            row['selection_kl_by_k'] = {str(k): float(np.mean(values)) for k, values in metrics.items()}
            improved = [k for k in config['budgets'] if row['selection_kl_by_k'][str(k)] < best.get(k, float('inf'))]
            if improved:
                path = output / f'private_{arm}_seed{seed}_epoch{epoch:03d}.pth'
                with path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'arm': arm, 'seed': seed,
                        'epoch': epoch, 'max_k': 24, 'validation': row}, stream)
                for k in improved:
                    best[k] = row['selection_kl_by_k'][str(k)]
                    checkpoints[str(k)] = {'path': str(path), 'sha256': sha256(path), 'epoch': epoch,
                        'selection_continuation_kl': best[k]}
        history.append(row)
        messages.put({'kind': 'progress', 'phase': 'fit', 'rank': rank, 'arm': arm, 'seed': seed, **row})
    write_json(output / f'{arm}_seed{seed}_history.json', history)
    write_json(output / f'{arm}_seed{seed}_checkpoints.json', checkpoints)
    routes = {}
    with torch.inference_mode():
        for k in config['budgets']:
            router.load_state_dict(torch.load(checkpoints[str(k)]['path'], map_location='cpu', weights_only=True)['state_dict'])
            router.eval()
            for sample in config['validation_ids']:
                ids = torch.tensor(data[sample, :128], device=device, dtype=torch.long)[None]
                routes[sample, k] = router.route(raw_prompt_embeddings(model, ids), k, prefix)[0].tolist()
    donors = derangement(config['validation_ids'], 579)
    write_json(output / f'{arm}_seed{seed}_routes.json', [{'sample': s, 'k': k, 'ordered_skips': path} for (s, k), path in routes.items()])
    with (output / f'{arm}_seed{seed}_quality.jsonl').open('x') as stream, torch.inference_mode():
        for sample in config['validation_ids']:
            ids = torch.tensor(data[sample], device=device, dtype=torch.long)[None]
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
            for k in config['budgets']:
                for policy, path in [('matched', routes[sample, k]), ('shuffled', routes[donors[sample], k])]:
                    metric = continuation_metrics(continuation_logits(model, ids, 128, path), teacher, ids, 128)
                    row = {'sample': sample, 'split': 'validation', 'arm': arm, 'seed': seed, 'k': k,
                        'policy': policy, 'skipped_layers': sorted(path), **metric}
                    stream.write(json.dumps(row) + '\n')
            stream.flush()
    return {'kind': 'fit_done', 'rank': rank, 'arm': arm, 'seed': seed,
        'training_documents': len(samples), 'max_k': 24, 'checkpoints': checkpoints,
        'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads()}


def evaluate_static(rank, model, data, config, task):
    output, policies = Path(config['output']), task['paths']
    pure = {r['sample']: r for p in sorted(output.glob('pure_labels_*.jsonl')) for r in lines(p)}
    hybrid = {r['sample']: r for p in sorted(output.glob('hybrid_labels_*.jsonl')) for r in lines(p)}
    with (output / f'static_quality_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
        for sample in config['validation_ids'][rank::4]:
            ids = torch.tensor(data[sample], device=next(model.parameters()).device, dtype=torch.long)[None]
            teacher_logits = continuation_logits(model, ids, 128, [])
            teacher = F.log_softmax(teacher_logits, -1)
            choices = [(0, 'full', [])]
            for k in config['budgets']:
                choices += [(k, name, path[:k]) for name, path in policies.items()]
                choices += [(k, 'offline_pure_greedy', pure[sample]['ordered_skips'][:k]),
                    (k, 'offline_hybrid_greedy', hybrid[sample]['ordered_skips'][:k])]
            for k, policy, path in choices:
                logits = teacher_logits if not path else continuation_logits(model, ids, 128, path)
                row = {'sample': sample, 'split': 'validation', 'arm': 'static', 'seed': None,
                    'k': k, 'policy': policy, 'skipped_layers': sorted(path),
                    **continuation_metrics(logits, teacher, ids, 128)}
                stream.write(json.dumps(row) + '\n')
            stream.flush()
    return {'kind': 'static_evaluation_done', 'rank': rank}


def worker(rank, config, inbox, messages):
    try:
        # Ten disjoint worker cores plus two controller cores: total union42.
        allowed = [int(i) for i in os.environ['NEUROBYPASS_CPU_IDS'].split(',')]
        os.sched_setaffinity(0, allowed[rank * 10:(rank + 1) * 10])
        torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config['base_model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval()
        model.requires_grad_(False)
        data = np.load(config['data_file'], allow_pickle=False)
        sample = config['train_ids'][rank]
        ids = torch.tensor(data[sample], device=f'cuda:{rank}', dtype=torch.long)[None]
        with torch.inference_mode():
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), -1)
            probe = []
            for selected in [config['fixed_prefix'], list(range(2, 14))]:
                candidates = [i for i in range(2, 40) if i not in selected][:2]
                for objective in ['kl', 'nll']:
                    shared = prefix_candidate_costs(model, ids, selected, candidates, objective, teacher)
                    from prompt_fixed_candidate_batch_v1 import batched_candidate_costs
                    from depth_paper_pilot_v2 import candidate_costs
                    serial = (batched_candidate_costs(model, ids, 128, selected, candidates, teacher, 1)
                        if objective == 'kl' else candidate_costs(model, ids, selected, candidates, 'nll', batch_size=1))
                    error = max(abs(a[1] - b[1]) for a, b in zip(shared, serial))
                    if error > 1e-6:
                        write_json(Path(config['output']) / f'failed_candidate_probe_{rank:03d}.json', {'rank': rank, 'selected': selected, 'objective': objective, 'shared': shared, 'serial': serial, 'error': error})
                        raise ValueError('Shared prefix differs from full serial route')
                    probe.append({'selected': selected, 'objective': objective, 'shared_prefix': shared, 'serial': serial,
                        'max_abs_error': error, 'tolerance': 1e-6})
        write_json(Path(config['output']) / f'worker_probe_{rank:03d}.json', {'rank': rank,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': 10, 'candidate_probes': probe})
        messages.put({'kind': 'worker_ready', 'rank': rank})
        while True:
            task = inbox.get()
            if task is None: break
            kind = task['kind']
            if kind == 'set_fixed_prefix':
                config['fixed_prefix'] = task['prefix']
                messages.put({'kind': 'fixed_prefix_set', 'rank': rank})
            elif kind == 'generate': messages.put(generate(rank, model, data, config, messages))
            elif kind == 'static_cost':
                messages.put({'kind': 'static_cost_done', 'rank': rank, 'objective': task['objective'],
                    'step': len(task['selected']) + 1, 'costs': static_costs(model, data, config, task)})
            elif kind == 'bi':
                from depth_paper_pilot_v2 import collect_bi
                with torch.inference_mode():
                    result = collect_bi(model, data, task['samples'], next(model.parameters()).device)
                messages.put({'kind': 'bi_done', 'rank': rank, **result})
            elif kind == 'fit': messages.put(fit(rank, model, data, config, task, messages))
            elif kind == 'static_evaluate': messages.put(evaluate_static(rank, model, data, config, task))
            elif kind == 'cache_audit':
                with torch.inference_mode():
                    probes = [audit_route(model, ids, task['path'][:[6, 10, 12, 24][rank]])]
                write_json(Path(config['output']) / f'cache_probe_{rank:03d}.json', probes)
                messages.put({'kind': 'cache_audit_done', 'rank': rank})
            else: raise ValueError('Unknown worker task')
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def receive(messages, processes, count, expected):
    found = []
    while len(found) < count:
        try: row = messages.get(timeout=5)
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                raise RuntimeError('Workers exited before expected completion')
            continue
        if row['kind'] == 'progress':
            stamp('prompt_depth_progress', **{k: v for k, v in row.items() if k != 'kind'}); continue
        if row['kind'] == 'error': raise RuntimeError(row['traceback'])
        if row['kind'] != expected: raise ValueError('Unexpected worker phase: ' + row['kind'])
        found.append(row)
    return found


def aggregate(output, config):
    rows = [r for p in sorted(output.glob('*quality*.jsonl')) for r in lines(p)]
    quality, comparisons = [], []
    populations = {'validation_all100': config['validation_ids'], 'selection32': config['selection_ids'],
        'other_validation': [s for s in config['validation_ids'] if s not in set(config['selection_ids'])]}
    for name, samples in populations.items():
        if not samples: continue
        groups = {}
        for row in rows:
            if row['sample'] in set(samples):
                groups.setdefault((row['arm'], row['seed'], row['k'], row['policy']), []).append(row)
        for (arm, seed, k, policy), group in sorted(groups.items(), key=lambda v: str(v[0])):
            nll = float(np.mean([r['continuation_nll'] for r in group]))
            quality.append({'population': name, 'arm': arm, 'seed': seed, 'k': k, 'policy': policy,
                'documents': len(group), 'continuation_kl': float(np.mean([r['continuation_kl'] for r in group])),
                'continuation_nll': nll, 'perplexity': math.exp(nll),
                'unique_masks': len({tuple(r['skipped_layers']) for r in group})})
        keyed = {(r['arm'], r['seed'], r['k'], r['policy'], r['sample']): r for r in rows}
        for arm in ['pure', 'hybrid']:
            for seed in config['seeds']:
                for k in config['budgets']:
                    for ref in ['shuffled', 'static_joint_kl', 'sleb_nll_opt']:
                        for metric in ['continuation_kl', 'continuation_nll']:
                            differences = [keyed[arm, seed, k, 'matched', s][metric] -
                                keyed[(arm, seed, k, 'shuffled', s) if ref == 'shuffled' else ('static', None, k, ref, s)][metric] for s in samples]
                            comparisons.append({'population': name, 'arm': arm, 'seed': seed, 'k': k,
                                'reference': ref, 'metric': metric, **paired_summary(differences)})
    return {'quality': quality, 'paired_comparisons': comparisons, 'quality_rows': len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--main-run', type=Path, default=Path('runs/stage3_prompt_fixed_main_20261104_01'))
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 4 or int(os.environ.get('NEUROBYPASS_CPU_COUNT', '0')) != 42:
        parser.error('Use the four-GPU/42-core resource launcher')
    started, root = time.monotonic(), Path(__file__).resolve().parents[1]
    output, main_run = Path(os.environ['NEUROBYPASS_RUN_DIR']), args.main_run.resolve()
    experiment = json.loads((main_run / 'experiment.json').read_text())
    manifest = json.loads((Path(experiment['configuration']['data_run']) / 'data_manifest.json').read_text())
    train = sorted(r['sample'] for r in manifest['records'] if r['split'] == 'train')
    validation = sorted(r['sample'] for r in manifest['records'] if r['split'] == 'validation')
    selection = json.loads((main_run / 'router_training_protocol.json').read_text())['checkpoint_validation_documents']
    fixed_file = root / 'runs/stage3_small_prompt_ablation_20261104_01/small_constant_routes.json'
    fixed = json.loads(fixed_file.read_text())[0]['ordered_skips']
    if fixed != [4, 8, 2, 10, 16, 14] or sha256(experiment['data_file']) != experiment['data_sha256']:
        raise ValueError('Fixed training-derived prefix/data provenance mismatch')
    if args.probe:
        train, validation, selection = train[:4], validation[:4], selection[:2]
    # Larger fixed train-only calibration; never rank static layers using validation.
    calibration = sorted(np.random.default_rng(579).choice(train, min(128, len(train)), replace=False).tolist())
    config = {'output': str(output), 'main_run': str(main_run), 'base_model': '/home/liushenyu/opt-13b',
        'data_file': experiment['data_file'], 'data_sha256': experiment['data_sha256'], 'probe': args.probe,
        'train_ids': train, 'validation_ids': validation, 'selection_ids': selection,
        'static_calibration_ids': calibration, 'fixed_prefix': fixed, 'fixed_prefix_source': str(fixed_file),
        'fixed_prefix_source_sha256': sha256(fixed_file), 'budgets': [6, 8, 10, 12, 16, 20, 24],
        'epochs': 1 if args.probe else 24, 'seeds': [579] if args.probe else [579, 580, 581], 'label_seed': 579}
    tracked = {str(p): sha256(p) for p in root.rglob('*') if p.is_file() and not p.is_relative_to(output)
        and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv', '.png'} and '.git' not in p.parts}
    for p in root.parent.glob('*.py'): tracked[str(p)] = sha256(p)
    write_json(output / 'preservation_before.json', tracked)
    write_json(output / 'launch_protocol.json', {**config, 'source_sha256': {p.name: sha256(p) for p in
        [Path(__file__), root / 'tools/prompt_depth_core_v1.py', root / 'tools/prompt_depth_candidates_serial_v2.py']}, 'old_test_used': False,
        'weights_frozen': True, 'new_router_max_k': 24, 'input': '128 prompt token+position embeddings only',
        'labels': '32 actual continuation tokens joint KL, each already-skipped set plus candidate',
        'old_labels_reuse': 'Only identical sample, input and entire skipped-set state',
        'branches': 'One deterministic low-cost alternative per training document; fork lengths5/9/13/17 pure or7/11/15/19 hybrid, greedily complete to24.',
        'training': 'Fresh initialization; per-document mean soft CE+0.25 hardCE; AdamW2e-4, batch16,24epochs; three fixed seeds.',
        'checkpoint_selection': 'Per-budget lowest continuation KL on original selection32 at epochs1,4,...24; full development curves retained.',
        'limitations': ['Validation100 already used for development; other68 are not an independent test.',
            'Offline experts see actual continuations and are not deployable prompt-only bounds.',
            'Variable branch lengths and fewer hybrid states are documented; comparisons are descriptive.',
            'Fixed6 prefix comes from prior800train learning; static128 calibration is stronger than prior16 but not a full paper recipe.',
            'No recovery training or formal downstream/timing benchmark in this experiment.',
            'All new candidate costs use batch-one real bypass tails and shared unchanged prefixes; local exact probes do not prove every possible route.']})
    allowed = [int(i) for i in os.environ['NEUROBYPASS_CPU_IDS'].split(',')]
    context = mp.get_context('spawn')
    messages, inboxes = context.Queue(), [context.Queue() for _ in range(4)]
    processes = [context.Process(target=worker, args=(r, config, inboxes[r], messages)) for r in range(4)]
    outcomes, paths = [], {}
    try:
        for process in processes: process.start()
        os.sched_setaffinity(0, allowed[40:42])
        receive(messages, processes, 4, 'worker_ready')
        # Two candidates shards per objective; all four GPUs work every static step.
        static, sleb = [], []
        with (output / 'static_search_steps.jsonl').open('x') as stream:
            for step in range(1, 25):
                for rank in range(4):
                    objective = 'kl' if rank < 2 else 'nll'
                    selected = static if objective == 'kl' else sleb
                    candidates = [i for i in range(2, 40) if i not in selected]
                    inboxes[rank].put({'kind': 'static_cost', 'objective': objective,
                        'selected': selected.copy(), 'candidates': candidates[rank % 2::2]})
                replies = receive(messages, processes, 4, 'static_cost_done')
                for objective, selected in [('kl', static), ('nll', sleb)]:
                    costs = sorted(c for row in replies if row['objective'] == objective for c in row['costs'])
                    layer, value = best_candidate(costs)
                    stream.write(json.dumps({'objective': objective, 'step': step, 'previous_skips': selected.copy(),
                        'selected_layer': layer, 'candidate_costs': costs, 'joint_cost': value}) + '\n'); stream.flush()
                    selected.append(layer)
                stamp('prompt_depth_static_step', step=step, joint_kl_layer=static[-1], sleb_layer=sleb[-1])
        # Recompute BI on the same expanded calibration, four disjoint shards.
        for rank in range(4):
            inboxes[rank].put({'kind': 'bi', 'samples': calibration[rank::4]})
        bi_parts = receive(messages, processes, 4, 'bi_done')
        tokens = sum(row['tokens_per_layer'] for row in bi_parts)
        scores = [[i, sum(row['scores'][i][1] * row['tokens_per_layer'] for row in bi_parts) / tokens] for i in range(40)]
        write_json(output / 'shortgpt_calibration.json', {'scores': scores, 'tokens_per_layer': tokens})
        paths = {'static_joint_kl': static, 'sleb_nll_opt': sleb,
            'shortgpt_bi_opt': [i for i, _ in sorted(scores, key=lambda row: (row[1], row[0])) if i >= 2]}
        for seed in [579, 580, 581]:
            paths[f'random_static_seed{seed}'] = np.random.default_rng(seed).permutation(range(2, 40)).tolist()
        nll_first = next(r for r in lines(output / 'static_search_steps.jsonl') if r['objective'] == 'nll')
        ranked = [i for i, _ in sorted(nll_first['candidate_costs'], key=lambda row: (row[1], row[0]))]
        paths['shortened_ppl_opt'] = ranked
        paths['shortened_ppl_plus'] = [i for i in ranked if i not in [0, 1, 2, 3, 38, 39]]
        write_json(output / 'static_paths.json', paths)
        # Choose the global prefix directly from train-only static calibration,
        # avoiding the old constant-router checkpoint's validation selection.
        fixed = static[:6]
        global_file = output / 'fixed_global_policy.json'
        write_json(global_file, {'ordered_skips': fixed, 'calibration_ids': calibration,
            'split': 'train', 'objective': 'joint continuation KL', 'validation_used': False,
            'source': 'First six steps of this run train-only static joint greedy path'})
        config.update({'fixed_prefix': fixed, 'fixed_prefix_source': str(global_file),
            'fixed_prefix_source_sha256': sha256(global_file)})
        protocol = json.loads((output / 'launch_protocol.json').read_text())
        protocol.update(config)
        protocol['fixed_prefix_design'] = 'Current-run static joint-KL first6, selected only on fixed128 training calibration records.'
        protocol['limitations'] = [text.replace('Fixed6 prefix comes from prior800train learning; static128 calibration is stronger than prior16 but not a full paper recipe.',
            'Fixed6 prefix uses current-run train-only128 calibration, with no validation-based choice; not a full paper recipe.') for text in protocol['limitations']]
        write_json(output / 'protocol.json', protocol)
        for inbox in inboxes: inbox.put({'kind': 'set_fixed_prefix', 'prefix': fixed})
        receive(messages, processes, 4, 'fixed_prefix_set')
        for inbox in inboxes: inbox.put({'kind': 'generate'})
        receive(messages, processes, 4, 'labels_done')
        for arm in ['pure', 'hybrid']:
            records = [r for p in sorted(output.glob(f'{arm}_labels_*.jsonl')) for r in lines(p)]
            branches = [r for p in sorted(output.glob(f'{arm}_branches_*.jsonl')) for r in lines(p)]
            if sorted(r['sample'] for r in records) != sorted(train + validation) or sorted(r['sample'] for r in branches) != train:
                raise ValueError('Incomplete/duplicate label shards')
            for record in records: validate_steps(record, [] if arm == 'pure' else fixed)
            for record in branches: validate_steps(record, record['initial_prefix'])
        stamp('prompt_depth_labels_complete', documents=len(train) + len(validation))
        for rank in range(4): inboxes[rank].put({'kind': 'static_evaluate', 'paths': paths})
        receive(messages, processes, 4, 'static_evaluation_done')
        jobs = [{'kind': 'fit', 'arm': arm, 'seed': seed} for seed in config['seeds'] for arm in ['pure', 'hybrid']]
        for offset in range(0, len(jobs), 4):
            batch = jobs[offset:offset + 4]
            for rank, task in enumerate(batch): inboxes[rank].put(task)
            outcomes += receive(messages, processes, len(batch), 'fit_done')
        # Numerical audits last because FP32 conversion changes in-memory dtype.
        for rank in range(4): inboxes[rank].put({'kind': 'cache_audit', 'path': sleb})
        receive(messages, processes, 4, 'cache_audit_done')
        for inbox in inboxes: inbox.put(None)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0: raise RuntimeError('Worker did not exit cleanly')
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join(timeout=10)
            if process.is_alive(): process.kill(); process.join(timeout=5)
    write_json(output / 'training_outcomes.json', outcomes)
    write_json(output / 'summary.json', {**aggregate(output, config), 'probe': args.probe,
        'elapsed_seconds': time.monotonic() - started, 'old_test_used': False})
    if any(sha256(path) != digest for path, digest in tracked.items()):
        raise ValueError('A preexisting source/evidence file changed')
    write_json(output / 'preservation_after.json', {'verified_existing_files': len(tracked),
        'data_sha256': sha256(config['data_file'])})
    stamp('prompt_depth_complete', elapsed_seconds=time.monotonic() - started)


if __name__ == '__main__': main()
