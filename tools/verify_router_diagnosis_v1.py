"""Read-only arithmetic/split audit; append one new verification record."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from diagnose_prompt_router_v2 import choice_statistics, summarize
from stage1_benchmark_v2 import sha256, write_json


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(a, b, tolerance=2e-6):
    if not math.isclose(a, b, abs_tol=tolerance, rel_tol=1e-6):
        raise ValueError(f'Arithmetic mismatch: {a} != {b}')


def verify(root, run):
    status = json.loads((run / 'status.json').read_text())
    if status != {**status, 'state': 'completed', 'exit_code': 0}:
        raise ValueError('Run has not successfully completed')
    protocol = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    labels = {r['sample']: r for p in sorted(Path(protocol['main_run']).glob('labels_*.jsonl'))
              for r in read_lines(p)}
    results = []
    for worker in json.loads((run / 'workers.json').read_text()):
        rank = worker['rank']
        item = {key: read_lines(run / f'{key}_{rank:03d}.jsonl')
            for key in ['teacher_rows', 'route_rows', 'quality_rows', 'off_policy_rows', 'repairs']}
        item.update({key: summary[key] for key in ['mean_context_path', 'train_label_prior_path']})
        item['actual_candidate_forwards'] = sum(len(row['candidate_costs'])
            for row in item['off_policy_rows'] if not row['reused_expert_costs'])
        if not set(worker['cpu_ids']) <= set(range(21)) or worker['threads'] > 10:
            raise ValueError('Worker exceeds the assigned CPU budget')
        results.append(item)
    routes = {r['sample']: r for result in results for r in result['route_rows']}
    teacher = [r for result in results for r in result['teacher_rows']]
    off_policy = [r for result in results for r in result['off_policy_rows']]
    quality = [r for result in results for r in result['quality_rows']]
    repairs = [r for result in results for r in result['repairs']]
    if len(routes) != 900 or Counter(r['split'] for r in routes.values()) != {'train': 800, 'validation': 100}:
        raise ValueError('Unexpected diagnostic population')
    if Counter((r['sample'], r['step']) for r in teacher) != Counter(
            (sample, step) for sample in labels for step in range(1, 7)):
        raise ValueError('Missing/duplicated teacher states')
    for row in teacher:
        label = labels[row['sample']]['steps'][row['step'] - 1]
        if row['previous_skips'] != label['previous_skips']:
            raise ValueError('Teacher conditioning changed')
        for policy in row['policies'].values():
            scores = np.full(40, 1.)
            scores[policy['choice']] = 0.
            expected = choice_statistics(scores, label['candidate_costs'])
            for name in ['choice', 'candidate_rank', 'regret', 'chosen_joint_kl', 'best_joint_kl', 'top1', 'top5']:
                close(policy[name], expected[name])
    if Counter((r['sample'], r['step']) for r in off_policy) != Counter(
            (sample, step) for sample in protocol['off_policy_ids'] for step in range(1, 7)):
        raise ValueError('Missing/duplicated own-route states')
    checked_candidates = 0
    for row in off_policy:
        path = routes[row['sample']]['router_path']
        if path[:row['step'] - 1] != row['previous_skips']:
            raise ValueError('Own path conditioning changed')
        if row['policies']['matched']['choice'] != path[row['step'] - 1]:
            raise ValueError('Own path choice mismatch')
        available = set(range(2, 40)) - set(row['previous_skips'])
        if {i for i, _ in row['candidate_costs']} != available:
            raise ValueError('Incomplete off-policy candidate set')
        if len(row['candidate_costs']) != len(available):
            raise ValueError('Duplicate candidate')
        checked_candidates += len(available)
        if row['reused_expert_costs']:
            if row['candidate_costs'] != labels[row['sample']]['steps'][row['step'] - 1]['candidate_costs']:
                raise ValueError('Reused expert scores changed')
        for policy in row['policies'].values():
            scores = np.full(40, 1.)
            scores[policy['choice']] = 0.
            expected = choice_statistics(scores, row['candidate_costs'])
            for name, value in expected.items():
                close(policy[name], value)
    by_quality = {(r['sample'], r['policy']): r for r in quality}
    if len(by_quality) != 1640:
        raise ValueError('Quality records duplicated or missing')
    if set(s for s, _ in by_quality) != set(protocol['quality_train_ids'] + protocol['quality_validation_documents']):
        raise ValueError('Unexpected quality split')
    for row in quality:
        if labels[row['sample']]['split'] != row['split'] or row['split'] == 'test':
            raise ValueError('Split contamination')
        if row['policy'] == 'full':
            close(row['continuation_kl'], 0)
        else:
            if len(row['skipped_layers']) != 6 or len(set(row['skipped_layers'])) != 6:
                raise ValueError('Invalid skip budget')
        if row['policy'] == 'oracle_greedy':
            close(row['continuation_kl'], labels[row['sample']]['continuation_kl'])
    off_final = {r['sample']: r for r in off_policy if r['step'] == 6}
    teacher_final = {r['sample']: r for r in teacher if r['step'] == 6}
    for row in repairs:
        sample = row['sample']
        best = min(cost for _, cost in off_final[sample]['candidate_costs'])
        close(row['router_prefix5_oracle_last']['continuation_kl'], best)
        close(row['oracle_prefix5_router_last']['continuation_kl'],
              teacher_final[sample]['policies']['matched']['chosen_joint_kl'])
        close(by_quality[sample, 'router']['continuation_kl'],
              off_final[sample]['policies']['matched']['chosen_joint_kl'])
    recomputed = summarize(results, protocol)
    for key, value in recomputed.items():
        if json.dumps(value, sort_keys=True) != json.dumps(summary[key], sort_keys=True):
            raise ValueError('Summary does not reproduce: ' + key)
    preserved = json.loads((run / 'preservation_before.json').read_text())
    for name, digest in preserved.items():
        if sha256(root / name) != digest:
            raise ValueError('Existing evidence changed: ' + name)
    if sha256(protocol['checkpoint']) != protocol['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    if sha256(protocol['data_file']) != protocol['data_sha256']:
        raise ValueError('Private input data changed')
    return {'verified_teacher_states': len(teacher), 'verified_own_path_states': len(off_policy),
        'verified_own_path_candidates': checked_candidates, 'verified_quality_rows': len(quality),
        'preserved_existing_files': len(preserved), 'summary_sha256': sha256(run / 'summary.json'),
        'test_used': False, 'checkpoint_unchanged': True, 'data_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.run.resolve())
    write_json(args.run / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
