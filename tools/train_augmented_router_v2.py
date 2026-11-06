"""Matched conditional-state augmentation experiment, with immutable old evidence.

Run through the two-GPU resource launcher. Each training document contributes its
six expert states and three late states. The control repeats expert states4-6;
the treatment uses a perturbed prefix and then conditional greedy steps4-6.
Both arms encode the same prompts once per batch and take identical updates.
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
from prompt_fixed_core_v1 import (SequentialGreedyRouter, best_candidate,
    continuation_logits, continuation_metrics, raw_prompt_embeddings)
from stage1_benchmark_v2 import sha256, stamp, write_json
from train_prompt_fixed_v1 import targets_from_record


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def branch_prefix(record, own_path, seed):
    """Use an off-expert own prefix; otherwise force a low-cost third choice."""
    expert = record['ordered_skips']
    if set(own_path[:3]) != set(expert[:3]):
        return own_path[:3], {'kind': 'own_prefix', 'alternative_rank': None}
    costs = sorted(record['steps'][2]['candidate_costs'], key=lambda item: (item[1], item[0]))
    rank = 1 + ((record['sample'] + seed) % 4)  # zero-based ranks1-4 = top2-5
    prefix = expert[:2] + [costs[rank][0]]
    if set(prefix) == set(expert[:3]):
        raise ValueError('The forced branch must differ from the expert prefix')
    return prefix, {'kind': 'expert_low_cost_fork', 'alternative_rank': rank + 1}


def pack_documents(records, branches, arm):
    arrays = []
    for record in records:
        base = targets_from_record(record)
        extra = targets_from_record({'steps': branches[record['sample']]['steps']}) if arm == 'augmented' else tuple(x[3:] for x in base)
        arrays.append(tuple(np.concatenate([a, b], axis=0) for a, b in zip(base, extra)))
    return tuple(torch.from_numpy(np.stack([row[i] for row in arrays])) for i in range(4))


def build_tables(rows, validation_ids, selection_ids, seed):
    """Aggregate actual evaluations, pairing at target-document level."""
    keyed = {(row['arm'], row['sample'], row['policy']): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError('Duplicate evaluation records')
    for arm in ['baseline', 'augmented']:
        for sample in validation_ids:
            donors = [keyed[arm, sample, f'shuffled_{i}'] for i in range(4)]
            keyed[arm, sample, 'shuffled_mean4'] = {
                metric: float(np.mean([r[metric] for r in donors]))
                for metric in ['continuation_kl', 'continuation_nll']}
    populations = {'validation_all100': validation_ids, 'validation_selection32': selection_ids,
        'validation_other68': [s for s in validation_ids if s not in set(selection_ids)]}
    quality, comparisons = [], []
    policies = ['full', 'static_old', 'static_learned', 'oracle_greedy', 'router', 'shuffled_mean4']
    for population, samples in populations.items():
        if not samples:
            continue
        for arm in ['baseline', 'augmented']:
            for policy in policies:
                group = [keyed[arm, sample, policy] for sample in samples]
                nll = float(np.mean([r['continuation_nll'] for r in group]))
                quality.append({'population': population, 'arm': arm, 'policy': policy,
                    'samples': len(samples), 'continuation_kl': float(np.mean([r['continuation_kl'] for r in group])),
                    'continuation_nll': nll, 'perplexity': math.exp(nll),
                    'unique_masks': (len({tuple(r['skipped_layers']) for r in group})
                                     if policy != 'shuffled_mean4' else None)})
            for reference in ['static_old', 'static_learned', 'oracle_greedy', 'shuffled_mean4']:
                for metric in ['continuation_kl', 'continuation_nll']:
                    comparisons.append({'population': population, 'arm': arm,
                        'reference_arm': arm, 'reference_policy': reference, 'metric': metric,
                        **paired_summary([keyed[arm, s, 'router'][metric] - keyed[arm, s, reference][metric]
                                          for s in samples], seed)})
        for metric in ['continuation_kl', 'continuation_nll']:
            comparisons.append({'population': population, 'arm': 'augmented',
                'reference_arm': 'baseline', 'reference_policy': 'router', 'metric': metric,
                **paired_summary([keyed['augmented', s, 'router'][metric] -
                    keyed['baseline', s, 'router'][metric] for s in samples], seed)})
    return {'quality': quality, 'paired_comparisons': comparisons}


def generate(model, router, data, task, records, output, rank, messages):
    all_rows = []
    with (output / f'branch_labels_{rank:03d}.jsonl').open('x') as handle, torch.inference_mode():
        for offset, sample in enumerate(task['train_ids'][rank::2]):
            record = records[sample]
            ids = torch.tensor(data[sample], dtype=torch.long, device=next(model.parameters()).device)[None]
            own_path = router.route(raw_prompt_embeddings(model, ids[:, :128]))[0].tolist()
            prefix, metadata = branch_prefix(record, own_path, task['seed'])
            initial = prefix.copy()
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
            # Reuse only when the entire selected set is identical for this document.
            expert_lookup = {tuple(sorted(step['previous_skips'])): step['candidate_costs'] for step in record['steps']}
            steps, forwards = [], 0
            for number in range(4, 7):
                key = tuple(sorted(prefix))
                reused = key in expert_lookup
                if reused:
                    costs = expert_lookup[key]
                else:
                    costs = []
                    for layer in range(2, 40):
                        if layer in prefix:
                            continue
                        logits = continuation_logits(model, ids, 128, prefix + [layer])
                        log_probs = F.log_softmax(logits, dim=-1)
                        cost = (teacher.exp() * (teacher - log_probs)).sum(-1).mean().item()
                        costs.append([layer, cost])
                        forwards += 1
                selected, cost = best_candidate(costs)
                steps.append({'step': number, 'previous_skips': prefix.copy(),
                    'selected_layer': selected, 'joint_continuation_kl': cost,
                    'candidate_costs': costs, 'reused_expert_costs': reused})
                prefix.append(selected)
            row = {'sample': sample, 'split': 'train', 'source': metadata['kind'],
                'alternative_rank': metadata['alternative_rank'], 'old_router_path': own_path,
                'initial_prefix': initial, 'ordered_skips': prefix, 'steps': steps,
                'new_candidate_forwards': forwards}
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            all_rows.append(row)
            if (offset + 1) % 10 == 0 or task['probe']:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'branch_labels',
                    'documents': offset + 1, 'total': len(task['train_ids'][rank::2]),
                    'new_candidate_forwards': sum(r['new_candidate_forwards'] for r in all_rows)})
    return {'kind': 'generated', 'rank': rank, 'documents': len(all_rows),
        'new_candidate_forwards': sum(r['new_candidate_forwards'] for r in all_rows)}


def fit(model, data, task, records, output, rank, messages):
    arm = ['baseline', 'augmented'][rank]
    device = next(model.parameters()).device
    training_records = [records[sample] for sample in task['train_ids']]
    branches = {row['sample']: row for p in sorted(output.glob('branch_labels_*.jsonl')) for row in read_lines(p)}
    states, targets, valid, chosen = pack_documents(training_records, branches, arm)
    with torch.inference_mode():
        embeddings = torch.empty((len(training_records), 128, 5120), dtype=torch.float16)
        for offset in range(0, len(training_records), 8):
            samples = task['train_ids'][offset:offset + 8]
            ids = torch.tensor(data[samples, :128], dtype=torch.long, device=device)
            embeddings[offset:offset + len(samples)].copy_(raw_prompt_embeddings(model, ids).cpu())
    torch.manual_seed(task['seed'])
    router = SequentialGreedyRouter().to(device)
    initial_hash = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in router.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    updates_per_epoch = math.ceil(len(training_records) / 16)
    total_updates = task['epochs'] * updates_per_epoch
    def schedule(update):
        warmup = max(1, total_updates // 20)
        if update < warmup:
            return (update + 1) / warmup
        fraction = (update - warmup) / max(1, total_updates - warmup)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best, best_path, history = float('inf'), None, []
    gradient_updates = 0
    for epoch in range(1, task['epochs'] + 1):
        router.train()
        order = torch.randperm(len(training_records), generator=torch.Generator().manual_seed(task['seed'] + epoch * 10)).tolist()
        total_loss = 0.
        for offset in range(0, len(order), 16):
            indices = order[offset:offset + 16]
            emb = embeddings[indices].to(device)
            selected = states[indices].to(device)
            target = targets[indices].to(device)
            available = valid[indices].to(device)
            label = chosen[indices].to(device)
            steps = selected.sum(-1).long() + 1
            optimizer.zero_grad(set_to_none=True)
            scores = router(emb, selected, steps)
            logits = (-scores).masked_fill(~available, -1e9)
            loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean() + .25 * F.cross_entropy(logits.reshape(-1, 40), label.reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            gradient_updates += 1
            total_loss += loss.item()
        row = {'epoch': epoch, 'mean_loss': total_loss / updates_per_epoch,
            'gradient_updates': gradient_updates, 'lr': scheduler.get_last_lr()[0]}
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
                best_path = output / f'private_{arm}_epoch_{epoch:03d}.pth'
                with best_path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'epoch': epoch,
                                'arm': arm, 'validation': row}, stream)
        history.append(row)
        messages.put({'kind': 'progress', 'phase': 'fit', 'arm': arm, **row})
    write_json(output / f'{arm}_history.json', history)
    router.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)['state_dict'])
    router.eval()
    with torch.inference_mode():
        routes = {}
        for sample in task['validation_ids']:
            ids = torch.tensor(data[sample, :128], dtype=torch.long, device=device)[None]
            routes[sample] = router.route(raw_prompt_embeddings(model, ids))[0].tolist()
        donors = [derangement(task['validation_ids'], task['seed'] + 100 + repeat) for repeat in range(4)]
        write_json(output / f'{arm}_routes.json', [{'sample': s, 'ordered_skips': path} for s, path in routes.items()])
        write_json(output / f'{arm}_donors.json', donors)
        with (output / f'{arm}_quality.jsonl').open('x') as stream:
            for sample in task['validation_ids']:
                ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                teacher_logits = continuation_logits(model, ids, 128, [])
                teacher = F.log_softmax(teacher_logits, dim=-1)
                policies = [('full', []), ('router', routes[sample]), ('static_old', task['static_old']),
                    ('static_learned', task['static_learned']), ('oracle_greedy', records[sample]['ordered_skips'])]
                policies += [(f'shuffled_{i}', routes[donors[i][sample]]) for i in range(4)]
                for policy, path in policies:
                    logits = teacher_logits if not path else continuation_logits(model, ids, 128, path)
                    row = {'arm': arm, 'sample': sample, 'split': 'validation', 'policy': policy,
                        'skipped_layers': sorted(path), **continuation_metrics(logits, teacher, ids, 128)}
                    stream.write(json.dumps(row) + '\n')
                stream.flush()
    return {'kind': 'fitted', 'rank': rank, 'arm': arm, 'epochs': task['epochs'],
        'parameter_count': sum(p.numel() for p in router.parameters()),
        'initial_parameter_sha256': initial_hash, 'best_checkpoint': str(best_path),
        'best_checkpoint_sha256': sha256(best_path), 'best_selection_kl': best,
        'training_documents': len(training_records), 'state_presentations_per_epoch': len(training_records) * 9,
        'gradient_updates': gradient_updates, 'width': 256,
        'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads()}


def worker(rank, task, inbox, messages):
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
        records = {row['sample']: row for path in sorted(Path(task['main_run']).glob('labels_*.jsonl')) for row in read_lines(path)}
        router = SequentialGreedyRouter().to(device).eval()
        router.load_state_dict(torch.load(task['old_checkpoint']['path'], map_location='cpu', weights_only=True)['state_dict'])
        if task['probe']:
            from prompt_fixed_experiment_v2 import cache_probe
            with torch.inference_mode():
                ids = torch.tensor(data[task['train_ids'][rank]], dtype=torch.long, device=device)[None]
                checks = [cache_probe(model, ids, 128, path) for path in [[], task['static_old']]]
                write_json(Path(task['output']) / f'probe_cache_{rank:03d}.json', checks)
        messages.put(generate(model, router, data, task, records, Path(task['output']), rank, messages))
        del router
        torch.cuda.empty_cache()
        if inbox.get() != 'fit':
            raise ValueError('Expected fit signal after both label shards completed')
        messages.put(fit(model, data, task, records, Path(task['output']), rank, messages))
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def receive(messages, processes):
    while True:
        try:
            message = messages.get(timeout=5)
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError('A worker exited with failure')
            if all(p.exitcode is not None for p in processes):
                raise RuntimeError('Workers exited without completing the protocol')
            continue
        if message['kind'] == 'error':
            raise RuntimeError(message['traceback'])
        if message['kind'] == 'progress':
            stamp('branch_router_progress', **{k: v for k, v in message.items() if k != 'kind'})
            continue
        return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--ablation-run', type=Path, required=True)
    parser.add_argument('--base-model', default='/home/liushenyu/opt-13b')
    parser.add_argument('--seed', type=int, default=579)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 2:
        parser.error('Use the two-GPU resource launcher')
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    main_run, ablation_run = args.main_run.resolve(), args.ablation_run.resolve()
    labels = [row for path in sorted(main_run.glob('labels_*.jsonl')) for row in read_lines(path)]
    if Counter(row['split'] for row in labels) != {'train': 800, 'validation': 100}:
        raise ValueError('Unexpected input labels')
    experiment = json.loads((main_run / 'experiment.json').read_text())
    old_checkpoint = json.loads((main_run / 'router_selected_checkpoint.json').read_text())
    old_protocol = json.loads((main_run / 'router_training_protocol.json').read_text())
    static_old = json.loads((main_run / 'static_policy.json').read_text())['ordered_skips']
    static_routes = json.loads((ablation_run / 'small_constant_routes.json').read_text())
    if len({tuple(row['ordered_skips']) for row in static_routes}) != 1:
        raise ValueError('Learned static reference must be constant')
    if sha256(experiment['data_file']) != experiment['data_sha256'] or sha256(old_checkpoint['path']) != old_checkpoint['sha256']:
        raise ValueError('Old data/checkpoint hash mismatch')
    train_ids = sorted(row['sample'] for row in labels if row['split'] == 'train')
    validation_ids = sorted(row['sample'] for row in labels if row['split'] == 'validation')
    selection_ids = old_protocol['checkpoint_validation_documents']
    if args.probe:
        train_ids, validation_ids, selection_ids = train_ids[:4], validation_ids[:4], selection_ids[:2]
    configuration = {'main_run': str(main_run), 'ablation_run': str(ablation_run),
        'base_model': args.base_model, 'seed': args.seed, 'probe': args.probe,
        'output': str(output), 'data_file': experiment['data_file'], 'data_sha256': experiment['data_sha256'],
        'old_checkpoint': old_checkpoint, 'train_ids': train_ids, 'validation_ids': validation_ids,
        'selection_ids': selection_ids, 'epochs': 1 if args.probe else 24,
        'static_old': static_old, 'static_learned': static_routes[0]['ordered_skips']}
    tracked = {str(p.resolve()): sha256(p) for p in root.rglob('*') if p.is_file()
        and not p.is_relative_to(output) and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv'} and '.git' not in p.parts}
    for p in root.parent.glob('*.py'):
        tracked[str(p.resolve())] = sha256(p)
    write_json(output / 'preservation_before.json', tracked)
    write_json(output / 'protocol.json', {**configuration, 'source_sha256': sha256(__file__),
        'base_model_frozen': True, 'test_used': False,
        'branch_design': 'Old router prefix3 if it differs from expert set3, otherwise expert prefix2 plus seeded rank2-5 third candidate; greedy complete steps4-6.',
        'matched_training': 'Same width256, initialization, batch16, prompt order, 50 updates per epoch, 24 epochs and nine state slots per document.',
        'control_slots': 'Six expert states plus repeated expert steps4-6.',
        'augmented_slots': 'Six expert states plus conditional greedy branch steps4-6.',
        'loss': 'Same per-state soft CE + 0.25 hard CE, with original candidate-temperature rule.',
        'optimizer': 'AdamW lr2e-4 weight_decay0.01, warmup5pct cosine, clip1.',
        'selection': 'Original fixed32 validation IDs at epochs1,4,...24; no validation cost used as a training label.',
        'limitations': ['Development validation has already been inspected; other68 are only excluded from checkpoint selection.',
            'One seed and one frozen behavior router; this is not online iterative DAgger.',
            'Branch oracle labels see real training continuations; prompt-only inference does not.',
            'Repeated late-state control differs from historical six-state training; use the matched control for causal comparison.',
            'C4 natural continuation and earlier data/near-duplicate limitations apply.']})
    context = mp.get_context('spawn')
    messages, inboxes = context.Queue(), [context.Queue(), context.Queue()]
    processes = [context.Process(target=worker, args=(rank, configuration, inboxes[rank], messages)) for rank in range(2)]
    fitted = []
    try:
        for process in processes:
            process.start()
        generated = [receive(messages, processes) for _ in range(2)]
        if any(row['kind'] != 'generated' for row in generated):
            raise ValueError('Incomplete generation protocol')
        branches = [row for path in sorted(output.glob('branch_labels_*.jsonl')) for row in read_lines(path)]
        if sorted(row['sample'] for row in branches) != train_ids:
            raise ValueError('Missing/duplicate training branch documents')
        write_json(output / 'generation_summary.json', {'documents': len(branches),
            'states': len(branches) * 3, 'sources': dict(Counter(row['source'] for row in branches)),
            'candidate_scores': sum(len(step['candidate_costs']) for row in branches for step in row['steps']),
            'new_candidate_forwards': sum(row['new_candidate_forwards'] for row in branches),
            'reused_expert_states': sum(step['reused_expert_costs'] for row in branches for step in row['steps'])})
        stamp('branch_labels_complete', documents=len(branches))
        for inbox in inboxes:
            inbox.put('fit')
        fitted = [receive(messages, processes) for _ in range(2)]
        if any(row['kind'] != 'fitted' for row in fitted):
            raise ValueError('Incomplete fit protocol')
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
    if len({row['initial_parameter_sha256'] for row in fitted}) != 1 or len({row['gradient_updates'] for row in fitted}) != 1:
        raise ValueError('Matched initialization/update control failed')
    quality_rows = [row for arm in ['baseline', 'augmented'] for row in read_lines(output / f'{arm}_quality.jsonl')]
    tables = build_tables(quality_rows, validation_ids, selection_ids, args.seed)
    write_json(output / 'summary.json', {**tables, 'arms': fitted,
        'generation': json.loads((output / 'generation_summary.json').read_text()),
        'elapsed_seconds': time.monotonic() - started, 'test_used': False, 'probe': args.probe})
    if any(sha256(path) != digest for path, digest in tracked.items()):
        raise ValueError('A preexisting source/evidence file changed')
    write_json(output / 'preservation_after.json', {'verified_existing_files': len(tracked),
        'old_checkpoint_sha256': sha256(old_checkpoint['path']), 'data_sha256': sha256(experiment['data_file'])})
    stamp('branch_router_complete', elapsed_seconds=time.monotonic() - started, arms=fitted)


if __name__ == '__main__':
    main()
