"""Frozen stage-3 router diagnosis on training/validation data only.

Adds evidence without changing source, checkpoints or previous experiment output.
Use the resource launcher with two GPUs. Candidate search is an offline oracle
that sees the continuation; those costs are never an inference observation.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import (SequentialGreedyRouter, continuation_logits,
    continuation_metrics, raw_prompt_embeddings)
from stage1_benchmark_v2 import sha256, stamp, write_json
from train_prompt_fixed_v1 import targets_from_record


def derangement(samples, seed):
    order = np.random.default_rng(seed).permutation(samples).tolist()
    if len(order) < 2:
        raise ValueError('Need at least two records for a derangement')
    return {sample: order[(i + 1) % len(order)] for i, sample in enumerate(order)}


def route_context(router, context, k=6):
    selected = context.new_zeros((context.size(0), 40))
    path = []
    for step in range(1, k + 1):
        scores = router.score(context, selected, torch.full(
            (context.size(0),), step, dtype=torch.long, device=context.device))
        unavailable = selected.bool().clone()
        unavailable[:, :2] = True
        index = scores.masked_fill(unavailable, float('inf')).argmin(-1)
        selected.scatter_(1, index[:, None], 1)
        path.append(index)
    return torch.stack(path, dim=1)


def choice_statistics(scores, candidate_costs, target=None):
    """Score an actual available choice, resolving ties by layer index."""
    costs = {int(i): float(cost) for i, cost in candidate_costs}
    order = sorted(costs, key=lambda layer: (costs[layer], layer))
    choice = min(costs, key=lambda layer: (float(scores[layer]), layer))
    result = {'choice': choice, 'candidate_rank': order.index(choice) + 1,
              'regret': costs[choice] - costs[order[0]],
              'chosen_joint_kl': costs[choice], 'best_joint_kl': costs[order[0]],
              'top1': int(choice == order[0]), 'top5': int(choice in order[:5])}
    if target is not None:
        indices = np.array(sorted(costs))
        logits = -np.asarray(scores, np.float64)[indices]
        log_probs = logits - np.logaddexp.reduce(logits)
        result['soft_cross_entropy'] = float(-(np.asarray(target)[indices] * log_probs).sum())
    return result


def paired_summary(differences, seed=579):
    values = np.asarray(differences, np.float64)
    indices = np.random.default_rng(seed).integers(0, len(values), (10000, len(values)))
    return {'samples': len(values), 'mean_delta': float(values.mean()),
            'bootstrap_95pct': np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
            'fraction_negative': float((values < 0).mean())}


def aggregate_choices(rows, key='policies'):
    groups = defaultdict(list)
    for row in rows:
        for policy, metrics in row[key].items():
            groups[(row['split'], row['step'], policy)].append(metrics)
    output = []
    for (split, step, policy), group in sorted(groups.items()):
        metrics = {name: float(np.mean([r[name] for r in group]))
                   for name in group[0] if name != 'choice'}
        output.append({'split': split, 'step': step, 'policy': policy,
                       'samples': len(group), **metrics})
    return output


def worker(rank, configuration, result_queue):
    try:
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.set_num_interop_threads(1)
        torch.manual_seed(configuration['seed'])
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(configuration['base_model'],
            torch_dtype=torch.float16, local_files_only=True,
            attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False)
        router = SequentialGreedyRouter().to(device).eval()
        router.load_state_dict(torch.load(configuration['checkpoint'],
            map_location='cpu', weights_only=True)['state_dict'])
        records = configuration['labels']
        data = np.load(configuration['data_file'], allow_pickle=False)
        by_sample = {r['sample']: r for r in records}
        train_ids = sorted(r['sample'] for r in records if r['split'] == 'train')
        validation_ids = sorted(r['sample'] for r in records if r['split'] == 'validation')
        contexts = {}
        with torch.inference_mode():
            all_ids = train_ids + validation_ids
            for offset in range(0, len(all_ids), 8):
                samples = all_ids[offset:offset + 8]
                ids = torch.tensor(data[samples, :128], dtype=torch.long, device=device)
                encoded = router.encode(raw_prompt_embeddings(model, ids))
                for sample, context in zip(samples, encoded):
                    contexts[sample] = context[None]
            mean_context = torch.cat([contexts[s] for s in train_ids]).mean(0, keepdim=True)
            # Learn this no-prompt control only from training label distributions.
            marginal = np.mean([targets_from_record(by_sample[s])[1] for s in train_ids], axis=0)
            marginal_scores = -np.log(np.maximum(marginal, 1e-30))
            global_path = []
            for step in range(6):
                valid = [i for i in range(2, 40) if i not in global_path]
                global_path.append(min(valid, key=lambda i: (marginal_scores[step, i], i)))
            mean_path = route_context(router, mean_context)[0].tolist()
            routes = {s: route_context(router, contexts[s])[0].tolist() for s in all_ids}
            expert_states = [Counter(tuple(sorted(by_sample[s]['steps'][t]['previous_skips']))
                                     for s in train_ids) for t in range(6)]
            donor_maps = {}
            for split, samples in [('train', train_ids), ('validation', validation_ids)]:
                donor_maps[split] = [derangement(samples, configuration['seed'] + 100 + repeat)
                                     for repeat in range(4)]
            teacher_rows, route_rows = [], []
            for sample in sorted(all_ids)[rank::2]:
                record = by_sample[sample]
                states, targets, _, _ = targets_from_record(record)
                state_tensor = torch.tensor(states, device=device)[None]
                steps = torch.arange(1, 7, device=device)[None]
                donor = donor_maps[record['split']][0][sample]
                score_sets = {}
                for name, context in [('matched', contexts[sample]), ('shuffled', contexts[donor]),
                                      ('mean_train_context', mean_context)]:
                    score_sets[name] = router.score(context, state_tensor, steps)[0].cpu().numpy()
                score_sets['train_label_prior'] = marginal_scores
                path = routes[sample]
                first_divergence = next((i + 1 for i, (a, b) in enumerate(
                    zip(path, record['ordered_skips'])) if a != b), None)
                route_rows.append({'sample': sample, 'split': record['split'],
                    'router_path': path, 'expert_path': record['ordered_skips'],
                    'first_ordered_divergence': first_divergence,
                    'final_set_jaccard': len(set(path) & set(record['ordered_skips'])) /
                                         len(set(path) | set(record['ordered_skips'])),
                    'router_prefix_training_counts': [expert_states[t][tuple(sorted(path[:t]))]
                                                       for t in range(6)],
                    'expert_prefix_training_counts': [expert_states[t][tuple(sorted(
                        record['ordered_skips'][:t]))] for t in range(6)]})
                for t, step in enumerate(record['steps']):
                    target = targets[t]
                    entropy = float(-(target * np.log(np.maximum(target, 1e-30))).sum())
                    teacher_rows.append({'sample': sample, 'split': record['split'],
                        'step': t + 1, 'previous_skips': step['previous_skips'],
                        'target_entropy': entropy, 'target_effective_candidates': math.exp(entropy),
                        'target_top1_mass': float(target.max()),
                        'target_top5_mass': float(np.sort(target)[-5:].sum()),
                        'policies': {name: choice_statistics(scores[t], step['candidate_costs'], target)
                                     for name, scores in score_sets.items()}})
            result_queue.put({'kind': 'progress', 'rank': rank, 'phase': 'teacher_states_complete',
                              'documents': len(route_rows)})
            quality_rows = []
            eval_ids = sorted(configuration['quality_train_ids'] + validation_ids)[rank::2]
            for offset, sample in enumerate(eval_ids):
                record = by_sample[sample]
                ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                teacher_logits = continuation_logits(model, ids, 128, [])
                teacher = F.log_softmax(teacher_logits, dim=-1)
                policies = [('full', []), ('router', routes[sample]),
                    ('mean_train_context', mean_path), ('train_label_prior', global_path),
                    ('static16', configuration['static']), ('oracle_greedy', record['ordered_skips'])]
                policies += [(f'shuffled_{repeat}', routes[donor_maps[record['split']][repeat][sample]])
                             for repeat in range(4)]
                for policy, path in policies:
                    logits = teacher_logits if not path else continuation_logits(model, ids, 128, path)
                    quality_rows.append({'sample': sample, 'split': record['split'],
                        'policy': policy, 'skipped_layers': sorted(path),
                        **continuation_metrics(logits, teacher, ids, 128)})
                if (offset + 1) % 10 == 0:
                    result_queue.put({'kind': 'progress', 'rank': rank,
                        'phase': 'quality', 'documents': offset + 1, 'total': len(eval_ids)})
            off_policy_rows, repairs = [], []
            actual_forwards = 0
            for sample in [s for s in sorted(all_ids)[rank::2] if s in set(configuration['off_policy_ids'])]:
                record = by_sample[sample]
                path = routes[sample]
                ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                teacher = F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
                final_costs = None
                for t in range(6):
                    selected = path[:t]
                    reuse = set(selected) == set(record['steps'][t]['previous_skips'])
                    if reuse:
                        costs = record['steps'][t]['candidate_costs']
                    else:
                        costs = []
                        for index in range(2, 40):
                            if index in selected:
                                continue
                            logits = continuation_logits(model, ids, 128, selected + [index])
                            log_probs = F.log_softmax(logits, dim=-1)
                            cost = (teacher.exp() * (teacher - log_probs)).sum(-1).mean().item()
                            costs.append([index, cost])
                            actual_forwards += 1
                    state = torch.zeros((1, 40), device=device)
                    state[0, selected] = 1
                    step_tensor = torch.tensor([t + 1], device=device)
                    donor = donor_maps['validation'][0][sample]
                    scores = {name: router.score(context, state, step_tensor)[0].cpu().numpy()
                        for name, context in [('matched', contexts[sample]),
                            ('shuffled', contexts[donor]), ('mean_train_context', mean_context)]}
                    scores['train_label_prior'] = marginal_scores[t]
                    comparisons = {name: choice_statistics(score, costs)
                                   for name, score in scores.items()}
                    if comparisons['matched']['choice'] != path[t]:
                        raise ValueError('Off-policy path failed replay')
                    off_policy_rows.append({'sample': sample, 'split': 'validation',
                        'step': t + 1, 'previous_skips': selected, 'reused_expert_costs': reuse,
                        'candidate_costs': costs, 'policies': comparisons})
                    final_costs = costs
                best_last = min(final_costs, key=lambda item: (item[1], item[0]))[0]
                repaired_path = path[:5] + [best_last]
                repaired_metrics = continuation_metrics(continuation_logits(model, ids, 128,
                    repaired_path), teacher, ids, 128)
                teacher_last = next(row for row in teacher_rows if row['sample'] == sample
                                    and row['step'] == 6)['policies']['matched']
                expert_prefix_repair = record['ordered_skips'][:5] + [teacher_last['choice']]
                expert_prefix_metrics = continuation_metrics(continuation_logits(model, ids, 128,
                    expert_prefix_repair), teacher, ids, 128)
                repairs.append({'sample': sample, 'router_prefix5_oracle_last_path': repaired_path,
                    'router_prefix5_oracle_last': repaired_metrics,
                    'oracle_prefix5_router_last_path': expert_prefix_repair,
                    'oracle_prefix5_router_last': expert_prefix_metrics})
                result_queue.put({'kind': 'progress', 'rank': rank, 'phase': 'own_path_scan',
                                  'sample': sample, 'candidate_forwards_so_far': actual_forwards})
            result_queue.put({'kind': 'done', 'rank': rank, 'teacher_rows': teacher_rows,
                'route_rows': route_rows, 'quality_rows': quality_rows,
                'off_policy_rows': off_policy_rows, 'repairs': repairs,
                'actual_candidate_forwards': actual_forwards,
                'mean_context_path': mean_path, 'train_label_prior_path': global_path,
                'worker': {'pid': os.getpid(), 'rank': rank,
                    'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads(),
                    'gpu_name': torch.cuda.get_device_name(rank)}})
    except BaseException:
        result_queue.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def summarize(results, configuration):
    teacher_rows = [r for result in results for r in result['teacher_rows']]
    route_rows = [r for result in results for r in result['route_rows']]
    quality_rows = [r for result in results for r in result['quality_rows']]
    off_rows = [r for result in results for r in result['off_policy_rows']]
    repairs = [r for result in results for r in result['repairs']]
    quality, quality_pairs = [], []
    for split in ['train', 'validation']:
        selected = [r for r in quality_rows if r['split'] == split]
        samples = sorted({r['sample'] for r in selected})
        by_key = {(r['sample'], r['policy']): r for r in selected}
        for sample in samples:
            donors = [by_key[sample, f'shuffled_{i}'] for i in range(4)]
            by_key[sample, 'shuffled_mean4'] = {'sample': sample, 'split': split,
                'policy': 'shuffled_mean4', **{metric: float(np.mean([r[metric] for r in donors]))
                for metric in ['continuation_kl', 'continuation_nll']}}
        policies = ['full', 'router', 'mean_train_context', 'train_label_prior',
                    'static16', 'oracle_greedy', 'shuffled_mean4']
        for policy in policies:
            group = [by_key[sample, policy] for sample in samples]
            quality.append({'split': split, 'policy': policy, 'samples': len(samples),
                'continuation_kl': float(np.mean([r['continuation_kl'] for r in group])),
                'continuation_nll': float(np.mean([r['continuation_nll'] for r in group])),
                'perplexity': math.exp(float(np.mean([r['continuation_nll'] for r in group])))})
        for reference in ['mean_train_context', 'train_label_prior', 'static16',
                          'oracle_greedy', 'shuffled_mean4']:
            for metric in ['continuation_kl', 'continuation_nll']:
                quality_pairs.append({'split': split, 'reference': reference, 'metric': metric,
                    **paired_summary([by_key[s, 'router'][metric] - by_key[s, reference][metric]
                                      for s in samples])})
    path_summary = []
    for split in ['train', 'validation']:
        rows = [r for r in route_rows if r['split'] == split]
        for step in range(6):
            path_summary.append({'split': split, 'step': step + 1, 'samples': len(rows),
                'own_prefix_unseen_in_expert_training': float(np.mean([
                    r['router_prefix_training_counts'][step] == 0 for r in rows])),
                'expert_prefix_unseen_in_expert_training': float(np.mean([
                    r['expert_prefix_training_counts'][step] == 0 for r in rows])),
                'prefix_set_matches_own_expert': float(np.mean([
                    set(r['router_path'][:step]) == set(r['expert_path'][:step]) for r in rows])),
                'ordered_path_not_yet_diverged': float(np.mean([
                    r['first_ordered_divergence'] is None or r['first_ordered_divergence'] >= step + 1
                    for r in rows]))})
    labels = []
    for split in ['train', 'validation']:
        for step in range(1, 7):
            rows = [r for r in teacher_rows if r['split'] == split and r['step'] == step]
            labels.append({'split': split, 'step': step, **{name: float(np.mean([r[name] for r in rows]))
                for name in ['target_entropy', 'target_effective_candidates',
                             'target_top1_mass', 'target_top5_mass']}})
    off_ids = set(configuration['off_policy_ids'])
    repair_summary = []
    for policy in ['router_prefix5_oracle_last', 'oracle_prefix5_router_last']:
        router = {r['sample']: r for r in quality_rows if r['policy'] == 'router' and r['sample'] in off_ids}
        rows = [r[policy] for r in repairs]
        repair_summary.append({'policy': policy, 'samples': len(rows),
            'continuation_kl': float(np.mean([r['continuation_kl'] for r in rows])),
            'continuation_nll': float(np.mean([r['continuation_nll'] for r in rows])),
            'paired_kl_difference_from_router': paired_summary([
                r[policy]['continuation_kl'] - router[r['sample']]['continuation_kl'] for r in repairs])})
    for policy in ['router', 'oracle_greedy', 'static16', 'mean_train_context']:
        rows = [r for r in quality_rows if r['sample'] in off_ids and r['policy'] == policy]
        repair_summary.append({'policy': policy, 'samples': len(rows),
            'continuation_kl': float(np.mean([r['continuation_kl'] for r in rows])),
            'continuation_nll': float(np.mean([r['continuation_nll'] for r in rows]))})
    return {'teacher_state_metrics': aggregate_choices(teacher_rows),
        'teacher_state_metrics_off_policy_subset': aggregate_choices([
            r for r in teacher_rows if r['sample'] in off_ids]),
        'own_state_metrics': aggregate_choices(off_rows), 'quality': quality,
        'quality_paired_comparisons': quality_pairs, 'path_coverage': path_summary,
        'target_diagnostics': labels, 'oracle_repair_diagnostics': repair_summary,
        'mean_context_path': results[0]['mean_context_path'],
        'train_label_prior_path': results[0]['train_label_prior_path'],
        'actual_new_candidate_forwards': sum(r['actual_candidate_forwards'] for r in results),
        'counts': {'teacher_documents': len(route_rows), 'teacher_states': len(teacher_rows),
                   'quality_rows': len(quality_rows), 'off_policy_states': len(off_rows)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--base-model', default='/home/liushenyu/opt-13b')
    parser.add_argument('--seed', type=int, default=579)
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 2:
        parser.error('Use the two-GPU resource launcher')
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    experiment = json.loads((args.main_run / 'experiment.json').read_text())
    selected = json.loads((args.main_run / 'router_selected_checkpoint.json').read_text())
    checkpoint_protocol = json.loads((args.main_run / 'router_training_protocol.json').read_text())
    records = [json.loads(line) for path in sorted(args.main_run.glob('labels_*.jsonl'))
               for line in path.read_text().splitlines()]
    if Counter(r['split'] for r in records) != Counter({'train': 800, 'validation': 100}):
        raise ValueError('Only existing train/validation labels are allowed')
    train_ids = sorted(r['sample'] for r in records if r['split'] == 'train')
    validation_ids = sorted(r['sample'] for r in records if r['split'] == 'validation')
    reserved = set(checkpoint_protocol['checkpoint_validation_documents'])
    off_pool = [s for s in validation_ids if s not in reserved]
    off_ids = sorted(np.random.default_rng(args.seed).choice(off_pool, 16, replace=False).tolist())
    train_quality = sorted(np.random.default_rng(args.seed + 1).choice(train_ids, 64, replace=False).tolist())
    if sha256(selected['path']) != selected['sha256']:
        raise ValueError('Checkpoint hash changed')
    if sha256(experiment['data_file']) != experiment['data_sha256']:
        raise ValueError('Input token data hash changed')
    preserved = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
        if p.is_file() and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv'}
        and not p.is_relative_to(output) and '.git' not in p.parts}
    configuration = {'seed': args.seed, 'base_model': args.base_model,
        'main_run': str(args.main_run.resolve()), 'checkpoint': selected['path'],
        'checkpoint_sha256': selected['sha256'], 'data_file': experiment['data_file'],
        'data_sha256': experiment['data_sha256'], 'labels': records,
        'static': json.loads((args.main_run / 'static_policy.json').read_text())['ordered_skips'],
        'quality_train_ids': train_quality, 'off_policy_ids': off_ids}
    public = {k: v for k, v in configuration.items() if k != 'labels'}
    public.update({'source_sha256': sha256(__file__), 'test_used': False,
        'updated_parameters': False, 'quality_validation_documents': validation_ids,
        'matching_controls': 'Four derangements within each split; average costs per target document.',
        'off_policy_selection': '16 seeded validation IDs excluding the 32 checkpoint-selection IDs.',
        'oracle_warning': 'Greedy costs/repairs see the continuation; diagnostic upper bounds only.',
        'mean_control': 'Mean encoded context over the 800 training prompts; no validation fitting.',
        'prior_control': 'Per-step mean soft label distribution over training trajectories; not a strong static search.',
        'limitations': ['Validation is reused for development, not a new held-out test.',
            'State novelty and late-step regret do not alone identify a causal effect of exposure bias.',
            'C4 natural continuation and the previous data limitations still apply.',
            'Bootstrap intervals are descriptive; document dependence and multiple comparisons are not corrected.']})
    write_json(output / 'protocol.json', public)
    write_json(output / 'preservation_before.json', preserved)
    context = mp.get_context('spawn')
    result_queue = context.Queue()
    processes = [context.Process(target=worker, args=(rank, configuration, result_queue)) for rank in range(2)]
    results = []
    try:
        for process in processes:
            process.start()
        while len(results) < 2:
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                for process in processes:
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(f'Worker {process.pid} exited {process.exitcode}')
                if all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Workers exited before sending complete results')
                continue
            if result['kind'] == 'error':
                raise RuntimeError(result['traceback'])
            if result['kind'] == 'progress':
                stamp('router_diagnosis_progress', **{k: v for k, v in result.items() if k != 'kind'})
                continue
            if result['kind'] != 'done':
                raise ValueError('Unexpected worker message')
            results.append(result)
            for name in ['teacher_rows', 'route_rows', 'quality_rows', 'off_policy_rows', 'repairs']:
                with (output / f'{name}_{result["rank"]:03d}.jsonl').open('x') as handle:
                    for row in result[name]:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
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
    summary = summarize(results, configuration)
    summary['elapsed_seconds'] = time.monotonic() - started
    if summary['counts'] != {'teacher_documents': 900, 'teacher_states': 5400,
                             'quality_rows': 1640, 'off_policy_states': 96}:
        raise ValueError('Incomplete diagnosis counts')
    write_json(output / 'summary.json', summary)
    write_json(output / 'workers.json', [r['worker'] for r in results])
    if any(sha256(root / name) != digest for name, digest in preserved.items()):
        raise ValueError('A preexisting evidence/source file changed')
    write_json(output / 'preservation_after.json', {'verified_existing_files': len(preserved),
        'preservation_before_sha256': sha256(output / 'preservation_before.json'),
        'checkpoint_sha256': sha256(selected['path']), 'test_used': False,
        'source_sha256': sha256(__file__)})
    stamp('router_diagnosis_complete', counts=summary['counts'], elapsed_seconds=summary['elapsed_seconds'])


if __name__ == '__main__':
    main()
