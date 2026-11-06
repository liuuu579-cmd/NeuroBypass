"""Stage5: three matched CE versus gap-weighted pairwise ranking comparisons.

Uses frozen stage4 expert+branch labels only; never generates validation labels
or reads the old test records for training/evaluation. All outputs are new files.
"""
import argparse
import hashlib
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

from diagnose_prompt_router_v2 import derangement, paired_summary
from prompt_fixed_core_v1 import (SequentialGreedyRouter, continuation_logits,
    continuation_metrics, raw_prompt_embeddings)
from stage1_benchmark_v2 import sha256, stamp, write_json
from train_augmented_router_v2 import pack_documents, read_lines


def classification_loss(scores, targets, valid, chosen):
    logits = (-scores).masked_fill(~valid, -1e9)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean() + .25 * F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), chosen.reshape(-1))


def gap_pairwise_loss(scores, costs, valid):
    """For c_i<c_j, prefer s_i<s_j; each state has total pair weight one.

    L_state = sum_ij (c_j-c_i)_+ * softplus(s_i-s_j)
              / sum_ij (c_j-c_i)_+ . Protected/selected layers have zero weight.
    Tied true costs have no preference. No new temperature/margin is tuned.
    """
    gaps = (costs.unsqueeze(-2) - costs.unsqueeze(-1)).clamp_min(0)
    pairs = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    weights = gaps * pairs
    weight_sum = weights.sum(dim=(-2, -1))
    differences = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    losses = (weights * F.softplus(differences)).sum(dim=(-2, -1)) / weight_sum.clamp_min(1e-12)
    return losses.mean()


def packed_costs(records, branches):
    result = np.zeros((len(records), 9, 40), dtype=np.float32)
    for index, record in enumerate(records):
        steps = record['steps'] + branches[record['sample']]['steps']
        for offset, step in enumerate(steps):
            for layer, cost in step['candidate_costs']:
                result[index, offset, layer] = cost
    return torch.from_numpy(result)


def summarize(rows, validation_ids, selection_ids, seeds):
    keyed = {(r['seed'], r['arm'], r['sample'], r['policy']): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError('Duplicate quality row')
    for seed in seeds:
        for arm in ['classification', 'pairwise']:
            for sample in validation_ids:
                donors = [keyed[seed, arm, sample, f'shuffled_{i}'] for i in range(4)]
                keyed[seed, arm, sample, 'shuffled_mean4'] = {
                    metric: float(np.mean([r[metric] for r in donors]))
                    for metric in ['continuation_kl', 'continuation_nll']}
    populations = {'validation_all100': validation_ids,
        'validation_selection32': selection_ids,
        'validation_other68': [s for s in validation_ids if s not in set(selection_ids)]}
    policies = ['router', 'shuffled_mean4', 'static_learned', 'static_old', 'full', 'oracle_greedy']
    quality, comparisons = [], []
    for population, samples in populations.items():
        if not samples:
            continue
        for seed_group, group in [(str(s), [s]) for s in seeds] + [('mean_seeds', seeds)]:
            values = {(arm, s, policy, metric): float(np.mean([
                keyed[seed, arm, s, policy][metric] for seed in group]))
                for arm in ['classification', 'pairwise'] for s in samples
                for policy in policies for metric in ['continuation_kl', 'continuation_nll']}
            for arm in ['classification', 'pairwise']:
                for policy in policies:
                    kl = float(np.mean([values[arm, s, policy, 'continuation_kl'] for s in samples]))
                    nll = float(np.mean([values[arm, s, policy, 'continuation_nll'] for s in samples]))
                    quality.append({'population': population, 'seed_group': seed_group,
                        'arm': arm, 'policy': policy, 'documents': len(samples), 'seeds': len(group),
                        'continuation_kl': kl, 'continuation_nll': nll, 'perplexity': math.exp(nll)})
            definitions = [('pairwise', 'classification', 'router'),
                ('classification', 'classification', 'shuffled_mean4'),
                ('pairwise', 'pairwise', 'shuffled_mean4'),
                ('classification', 'classification', 'static_learned'),
                ('pairwise', 'pairwise', 'static_learned')]
            for arm, reference_arm, policy in definitions:
                for metric in ['continuation_kl', 'continuation_nll']:
                    delta = [values[arm, s, 'router', metric] - values[reference_arm, s, policy, metric]
                             for s in samples]
                    comparisons.append({'population': population, 'seed_group': seed_group,
                        'arm': arm, 'reference_arm': reference_arm, 'reference_policy': policy,
                        'metric': metric, **paired_summary(delta)})
    return {'quality': quality, 'paired_comparisons': comparisons,
        'seed_aggregation': 'Average seed/donor costs within each target document before bootstrapping documents. Three seeds are not 300 independent documents.'}


def fit_seed(model, embeddings, data, records, packed, task, rank, seed, messages):
    arm = ['classification', 'pairwise'][rank]
    device = next(model.parameters()).device
    states, targets, valid, chosen, costs = packed
    torch.manual_seed(seed)
    router = SequentialGreedyRouter().to(device)
    initial_hash = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in router.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    updates_per_epoch = math.ceil(len(task['train_ids']) / 16)
    updates_total = task['epochs'] * updates_per_epoch
    def schedule(update):
        warmup = max(1, updates_total // 20)
        if update < warmup:
            return (update + 1) / warmup
        fraction = (update - warmup) / max(1, updates_total - warmup)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(task['output'])
    best, best_path, history = float('inf'), None, []
    for epoch in range(1, task['epochs'] + 1):
        router.train()
        order = torch.randperm(len(task['train_ids']), generator=torch.Generator().manual_seed(seed + epoch * 10)).tolist()
        loss_sum, norm_sum = 0., 0.
        for offset in range(0, len(order), 16):
            indices = order[offset:offset + 16]
            selected = states[indices].to(device)
            available = valid[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = router(embeddings[indices].to(device), selected, selected.sum(-1).long() + 1)
            if arm == 'classification':
                loss = classification_loss(scores, targets[indices].to(device), available, chosen[indices].to(device))
            else:
                loss = gap_pairwise_loss(scores, costs[indices].to(device), available)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            norm_sum += norm.item()
        row = {'epoch': epoch, 'mean_loss': loss_sum / updates_per_epoch,
            'mean_unclipped_gradient_norm': norm_sum / updates_per_epoch,
            'gradient_updates': epoch * updates_per_epoch, 'lr': scheduler.get_last_lr()[0]}
        if epoch == 1 or epoch % 4 == 0 or epoch == task['epochs']:
            router.eval()
            metrics = []
            with torch.inference_mode():
                for sample in task['selection_ids']:
                    ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                    path = router.route(raw_prompt_embeddings(model, ids[:, :128]))[0].tolist()
                    teacher = F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
                    metrics.append(continuation_metrics(continuation_logits(model, ids, 128, path), teacher, ids, 128))
            row['continuation_kl'] = float(np.mean([r['continuation_kl'] for r in metrics]))
            row['continuation_nll'] = float(np.mean([r['continuation_nll'] for r in metrics]))
            if row['continuation_kl'] < best:
                best = row['continuation_kl']
                best_path = output / f'private_{arm}_seed{seed}_epoch{epoch:03d}.pth'
                with best_path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'epoch': epoch,
                        'seed': seed, 'arm': arm, 'validation': row}, stream)
        history.append(row)
        messages.put({'kind': 'progress', 'arm': arm, 'seed': seed, **row})
    write_json(output / f'{arm}_seed{seed}_history.json', history)
    router.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)['state_dict'])
    router.eval()
    with torch.inference_mode():
        routes = {}
        for sample in task['validation_ids']:
            ids = torch.tensor(data[sample, :128], dtype=torch.long, device=device)[None]
            routes[sample] = router.route(raw_prompt_embeddings(model, ids))[0].tolist()
        # Identical donor maps for every seed and both arms.
        donors = [derangement(task['validation_ids'], 579 + 100 + repeat) for repeat in range(4)]
        write_json(output / f'{arm}_seed{seed}_routes.json', [{'sample': s, 'ordered_skips': path} for s, path in routes.items()])
        with (output / f'{arm}_seed{seed}_quality.jsonl').open('x') as stream:
            for sample in task['validation_ids']:
                ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                teacher_logits = continuation_logits(model, ids, 128, [])
                teacher = F.log_softmax(teacher_logits, dim=-1)
                policies = [('full', []), ('router', routes[sample]), ('static_old', task['static_old']),
                    ('static_learned', task['static_learned']), ('oracle_greedy', records[sample]['ordered_skips'])]
                policies += [(f'shuffled_{repeat}', routes[donors[repeat][sample]]) for repeat in range(4)]
                for policy, path in policies:
                    logits = teacher_logits if not path else continuation_logits(model, ids, 128, path)
                    row = {'seed': seed, 'arm': arm, 'sample': sample, 'split': 'validation',
                        'policy': policy, 'skipped_layers': sorted(path),
                        **continuation_metrics(logits, teacher, ids, 128)}
                    stream.write(json.dumps(row) + '\n')
                stream.flush()
    return {'seed': seed, 'arm': arm, 'parameter_count': sum(p.numel() for p in router.parameters()),
        'initial_parameter_sha256': initial_hash, 'gradient_updates': task['epochs'] * updates_per_epoch,
        'best_checkpoint': str(best_path), 'best_checkpoint_sha256': sha256(best_path),
        'best_selection_kl': best, 'unique_validation_masks': len({tuple(sorted(path)) for path in routes.values()})}


def worker(rank, task, messages):
    try:
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(task['base_model'], torch_dtype=torch.float16,
            local_files_only=True, attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False)
        data = np.load(task['data_file'], allow_pickle=False)
        records = {r['sample']: r for p in sorted(Path(task['main_run']).glob('labels_*.jsonl')) for r in read_lines(p)}
        branches = {r['sample']: r for p in sorted(Path(task['branch_run']).glob('branch_labels_*.jsonl')) for r in read_lines(p)}
        training_records = [records[s] for s in task['train_ids']]
        states, targets, valid, chosen = pack_documents(training_records, branches, 'augmented')
        packed = (states, targets, valid, chosen, packed_costs(training_records, branches))
        with torch.inference_mode():
            embeddings = torch.empty((len(training_records), 128, 5120), dtype=torch.float16)
            for offset in range(0, len(training_records), 8):
                samples = task['train_ids'][offset:offset + 8]
                ids = torch.tensor(data[samples, :128], dtype=torch.long, device=device)
                embeddings[offset:offset + len(samples)].copy_(raw_prompt_embeddings(model, ids).cpu())
        results = []
        for seed in task['seeds']:
            result = fit_seed(model, embeddings, data, records, packed, task, rank, seed, messages)
            results.append(result)
            messages.put({'kind': 'seed_complete', **result})
        messages.put({'kind': 'done', 'rank': rank, 'runs': results,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads()})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-run', type=Path, required=True)
    parser.add_argument('--base-model', default='/home/liushenyu/opt-13b')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 2:
        parser.error('Use the two-GPU resource launcher')
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    branch_run = args.branch_run.resolve()
    previous = json.loads((branch_run / 'protocol.json').read_text())
    verified = json.loads((branch_run / 'verification.json').read_text())
    if verified['probe'] or verified['summary_sha256'] != sha256(branch_run / 'summary.json'):
        raise ValueError('Use audited production branch data')
    labels = [r for p in sorted(Path(previous['main_run']).glob('labels_*.jsonl')) for r in read_lines(p)]
    if Counter(r['split'] for r in labels) != {'train': 800, 'validation': 100}:
        raise ValueError('Unexpected original label population')
    if sha256(previous['data_file']) != previous['data_sha256']:
        raise ValueError('Input token data changed')
    branch_hashes = {str(p): sha256(p) for p in sorted(branch_run.glob('branch_labels_*.jsonl'))}
    original_branch_hashes = {p: digest for p, digest in json.loads((branch_run / 'preservation_before.json').read_text()).items()
                             if Path(p).name.startswith('labels_') and Path(p).parent == Path(previous['main_run'])}
    for path, digest in original_branch_hashes.items():
        if sha256(path) != digest:
            raise ValueError('Expert labels changed since the branch experiment')
    train_ids, validation_ids, selection_ids = previous['train_ids'], previous['validation_ids'], previous['selection_ids']
    if args.probe:
        train_ids, validation_ids, selection_ids = train_ids[:4], validation_ids[:4], selection_ids[:2]
    task = {'output': str(output), 'main_run': previous['main_run'], 'branch_run': str(branch_run),
        'base_model': args.base_model, 'data_file': previous['data_file'], 'data_sha256': previous['data_sha256'],
        'train_ids': train_ids, 'validation_ids': validation_ids, 'selection_ids': selection_ids,
        'static_old': previous['static_old'], 'static_learned': previous['static_learned'],
        'seeds': [579] if args.probe else [579, 580, 581], 'epochs': 1 if args.probe else 24}
    tracked = {str(p.resolve()): sha256(p) for p in root.rglob('*') if p.is_file()
        and not p.is_relative_to(output) and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv'} and '.git' not in p.parts}
    for path in root.parent.glob('*.py'):
        tracked[str(path.resolve())] = sha256(path)
    write_json(output / 'preservation_before.json', tracked)
    write_json(output / 'protocol.json', {**task, 'probe': args.probe, 'test_used': False,
        'base_model_frozen': True, 'source_sha256': sha256(__file__), 'branch_label_sha256': branch_hashes,
        'classification_loss': 'Original soft target CE + 0.25 hard greedy-choice CE.',
        'pairwise_loss': 'Per-state sum[(c_j-c_i)_+ softplus(s_i-s_j)] / sum[(c_j-c_i)_+], only valid pairs.',
        'pairwise_scale': 'Raw router scores, no additional temperature/margin; equal costs have zero pair weight.',
        'matched_training': 'Width256; identical per-seed initialization, document order, batch16, nine states/document, 50 updates/epoch, 24 epochs.',
        'optimizer': 'AdamW lr2e-4 weight_decay0.01, warmup5pct cosine schedule, clip1.',
        'checkpoint_selection': 'Existing fixed32 validation IDs, epochs1,4,8,12,16,20,24 by real continuation KL.',
        'donor_seed': 579, 'seed_summary': 'Average metrics within target document over seeds; then paired document bootstrap.',
        'limitations': ['This is reused development validation, not an independent test.',
            'Three seeds are a small replication; document bootstrap of their mean does not estimate all training-seed uncertainty.',
            'Objective scale changes while optimizer is fixed; no per-objective hyperparameter tuning is performed.',
            'Pairwise weights include all valid unequal candidate pairs, so very damaging candidates receive more weight.',
            'C4 natural continuation and previous data/near-duplicate limitations apply.']})
    context = mp.get_context('spawn')
    messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, task, messages)) for rank in range(2)]
    results = []
    try:
        for process in processes:
            process.start()
        while len(results) < 2:
            try:
                message = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Workers exited without complete results')
                continue
            if message['kind'] == 'error':
                raise RuntimeError(message['traceback'])
            if message['kind'] == 'done':
                results.append(message)
            else:
                stamp('ranking_router_progress', **{k: v for k, v in message.items() if k != 'kind'})
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Worker did not exit cleanly')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    runs = [r for result in results for r in result['runs']]
    for seed in task['seeds']:
        pair = [r for r in runs if r['seed'] == seed]
        if len(pair) != 2 or len({r['initial_parameter_sha256'] for r in pair}) != 1 or len({r['gradient_updates'] for r in pair}) != 1:
            raise ValueError('Matched per-seed controls failed')
    rows = [r for p in sorted(output.glob('*_quality.jsonl')) for r in read_lines(p)]
    tables = summarize(rows, validation_ids, selection_ids, task['seeds'])
    write_json(output / 'summary.json', {**tables, 'runs': runs, 'seeds': task['seeds'],
        'elapsed_seconds': time.monotonic() - started, 'test_used': False, 'probe': args.probe})
    write_json(output / 'workers.json', [{k: r[k] for k in ['rank', 'cpu_ids', 'threads']} for r in results])
    if any(sha256(path) != digest for path, digest in tracked.items()):
        raise ValueError('Preexisting evidence/source changed')
    write_json(output / 'preservation_after.json', {'verified_existing_files': len(tracked),
        'data_sha256': sha256(task['data_file']), 'branch_label_sha256': branch_hashes})
    stamp('ranking_router_complete', elapsed_seconds=time.monotonic() - started, runs=runs)


if __name__ == '__main__':
    main()
