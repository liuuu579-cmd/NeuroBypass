"""Read-only provenance and arithmetic audit for stage5 loss comparison."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from diagnose_prompt_router_v2 import derangement
from stage1_benchmark_v2 import sha256, write_json
from train_augmented_router_v2 import read_lines
from train_router_ranking_v1 import summarize
from verify_augmented_router_v1 import verify as verify_branch


def verify(root, run):
    status = json.loads((run / 'status.json').read_text())
    if status.get('state') != 'completed' or status.get('exit_code') != 0:
        raise ValueError('Run has not successfully completed')
    protocol = json.loads((run / 'protocol.json').read_text())
    data = json.loads((run / 'summary.json').read_text())
    branch_run = Path(protocol['branch_run'])
    branch_audit = verify_branch(root, branch_run)
    if protocol['test_used'] or data['test_used'] or branch_audit['test_used']:
        raise ValueError('Old test split used in development')
    if not protocol['probe'] and (protocol['seeds'] != [579, 580, 581] or protocol['epochs'] != 24):
        raise ValueError('Production seed/training budget changed')
    old_protocol = json.loads((branch_run / 'protocol.json').read_text())
    if not protocol['probe']:
        for field in ['train_ids', 'validation_ids', 'selection_ids', 'static_old', 'static_learned']:
            if protocol[field] != old_protocol[field]:
                raise ValueError('Input protocol changed: ' + field)
    for path, digest in protocol['branch_label_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Branch label snapshot changed')
    if sha256(protocol['data_file']) != protocol['data_sha256']:
        raise ValueError('Private input token data changed')
    labels = {r['sample']: r for p in Path(protocol['main_run']).glob('labels_*.jsonl') for r in read_lines(p)}
    runs = {(r['seed'], r['arm']): r for r in data['runs']}
    expected_keys = {(seed, arm) for seed in protocol['seeds'] for arm in ['classification', 'pairwise']}
    if set(runs) != expected_keys or len(data['runs']) != len(expected_keys):
        raise ValueError('Missing/duplicate training run')
    quality = []
    for seed in protocol['seeds']:
        pair = [runs[seed, arm] for arm in ['classification', 'pairwise']]
        if len({r['initial_parameter_sha256'] for r in pair}) != 1 or len({r['parameter_count'] for r in pair}) != 1:
            raise ValueError('Matched initialization/capacity failed')
        for arm in ['classification', 'pairwise']:
            record = runs[seed, arm]
            expected_updates = math.ceil(len(protocol['train_ids']) / 16) * protocol['epochs']
            if record['gradient_updates'] != expected_updates:
                raise ValueError('Training update budget changed')
            history = json.loads((run / f'{arm}_seed{seed}_history.json').read_text())
            if [r['epoch'] for r in history] != list(range(1, protocol['epochs'] + 1)):
                raise ValueError('Incomplete training history')
            expected_validation_epochs = [e for e in range(1, protocol['epochs'] + 1)
                if e == 1 or e % 4 == 0 or e == protocol['epochs']]
            scored = [r for r in history if 'continuation_kl' in r]
            if [r['epoch'] for r in scored] != expected_validation_epochs:
                raise ValueError('Checkpoint selection schedule changed')
            if any(not math.isfinite(r['mean_loss']) or not math.isfinite(r['mean_unclipped_gradient_norm']) for r in history):
                raise ValueError('Nonfinite training history')
            best = min(scored, key=lambda r: r['continuation_kl'])
            if best['continuation_kl'] != record['best_selection_kl']:
                raise ValueError('Incorrect checkpoint score')
            if not record['best_checkpoint'].endswith(f"epoch{best['epoch']:03d}.pth"):
                raise ValueError('Wrong checkpoint epoch')
            if sha256(record['best_checkpoint']) != record['best_checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            routes = json.loads((run / f'{arm}_seed{seed}_routes.json').read_text())
            route_map = {r['sample']: r['ordered_skips'] for r in routes}
            if len(routes) != len(route_map) or sorted(route_map) != protocol['validation_ids']:
                raise ValueError('Invalid validation decision population')
            donors = [derangement(protocol['validation_ids'], protocol['donor_seed'] + 100 + repeat) for repeat in range(4)]
            rows = read_lines(run / f'{arm}_seed{seed}_quality.jsonl')
            if len(rows) != len(protocol['validation_ids']) * 9:
                raise ValueError('Missing quality rows')
            for row in rows:
                sample, policy = row['sample'], row['policy']
                if sample not in protocol['validation_ids'] or row['split'] != 'validation' or row['seed'] != seed or row['arm'] != arm:
                    raise ValueError('Quality record population mismatch')
                if policy == 'full':
                    expected = []
                    if row['continuation_kl'] != 0:
                        raise ValueError('Full-model KL not zero')
                elif policy == 'router':
                    expected = route_map[sample]
                elif policy in ['static_old', 'static_learned']:
                    expected = protocol[policy]
                elif policy == 'oracle_greedy':
                    expected = labels[sample]['ordered_skips']
                    if not math.isclose(row['continuation_kl'], labels[sample]['continuation_kl'], abs_tol=2e-6, rel_tol=1e-6):
                        raise ValueError('Oracle quality changed')
                else:
                    repeat = int(policy.removeprefix('shuffled_'))
                    expected = route_map[donors[repeat][sample]]
                if row['skipped_layers'] != sorted(expected):
                    raise ValueError('Evaluation route mismatch')
                if expected and (len(expected) != 6 or len(set(expected)) != 6 or not set(expected) <= set(range(2, 40))):
                    raise ValueError('Invalid route budget')
                if row['continuation_tokens'] != 32:
                    raise ValueError('Evaluation workload changed')
            quality.extend(rows)
    for worker in json.loads((run / 'workers.json').read_text()):
        if not set(worker['cpu_ids']) <= set(range(21)) or worker['threads'] > 10:
            raise ValueError('Worker CPU budget exceeded')
    tables = summarize(quality, protocol['validation_ids'], protocol['selection_ids'], protocol['seeds'])
    for field, value in tables.items():
        if json.dumps(value, sort_keys=True) != json.dumps(data[field], sort_keys=True):
            raise ValueError('Summary does not reproduce: ' + field)
    tracked = json.loads((run / 'preservation_before.json').read_text())
    for path, digest in tracked.items():
        if sha256(path) != digest:
            raise ValueError('Preexisting file changed: ' + path)
    reproduction = None
    if not protocol['probe']:
        previous = {r['sample']: r for r in read_lines(branch_run / 'augmented_quality.jsonl') if r['policy'] == 'router'}
        current = {r['sample']: r for r in quality if r['seed'] == 579 and r['arm'] == 'classification' and r['policy'] == 'router'}
        reproduction = {'same_routes': sum(previous[s]['skipped_layers'] == current[s]['skipped_layers'] for s in previous),
            'documents': len(previous), 'max_kl_abs_difference': max(abs(previous[s]['continuation_kl'] - current[s]['continuation_kl']) for s in previous)}
    return {'verified_quality_rows': len(quality), 'verified_training_runs': len(runs),
        'verified_epochs': len(runs) * protocol['epochs'], 'training_documents': len(protocol['train_ids']),
        'matched_per_seed_initialization_capacity_updates': True,
        'reused_branch_labels_unchanged': True, 'preserved_existing_files': len(tracked),
        'historical_ce_seed579_reproduction': reproduction, 'test_used': False,
        'probe': protocol['probe'], 'summary_sha256': sha256(run / 'summary.json')}


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
