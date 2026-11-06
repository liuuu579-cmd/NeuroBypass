"""Read-only audit of branch-label provenance and matched training results."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256, write_json
from train_augmented_router_v2 import branch_prefix, build_tables, pack_documents, read_lines


def close(a, b, tolerance=2e-6):
    if not math.isclose(a, b, abs_tol=tolerance, rel_tol=1e-6):
        raise ValueError(f'Numerical mismatch: {a} != {b}')


def verify(root, run):
    status = json.loads((run / 'status.json').read_text())
    if status.get('state') != 'completed' or status.get('exit_code') != 0:
        raise ValueError('Run not successfully complete')
    protocol = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    old_run = Path(protocol['main_run'])
    old_records = {row['sample']: row for path in old_run.glob('labels_*.jsonl') for row in read_lines(path)}
    old_protocol = json.loads((old_run / 'router_training_protocol.json').read_text())
    if not protocol['probe']:
        if protocol['selection_ids'] != old_protocol['checkpoint_validation_documents']:
            raise ValueError('Checkpoint selection population changed')
        if len(protocol['train_ids']) != 800 or len(protocol['validation_ids']) != 100 or protocol['epochs'] != 24:
            raise ValueError('Main training budget changed')
    if protocol['test_used'] or summary['test_used']:
        raise ValueError('Test used during development')
    branches = [row for path in sorted(run.glob('branch_labels_*.jsonl')) for row in read_lines(path)]
    if sorted(row['sample'] for row in branches) != protocol['train_ids']:
        raise ValueError('Missing or duplicate training branch record')
    candidate_count, new_forwards, reused, unique_added = 0, 0, 0, 0
    for record in branches:
        sample = record['sample']
        old = old_records[sample]
        if record['split'] != 'train' or old['split'] != 'train':
            raise ValueError('Non-training branch labels')
        prefix, metadata = branch_prefix(old, record['old_router_path'], protocol['seed'])
        if record['initial_prefix'] != prefix or record['source'] != metadata['kind']:
            raise ValueError('Branch selection rule changed')
        if record['alternative_rank'] != metadata['alternative_rank']:
            raise ValueError('Incorrect alternate-candidate rank')
        if set(prefix) == set(old['ordered_skips'][:3]):
            raise ValueError('No effective initial branch')
        lookup = {tuple(sorted(step['previous_skips'])): step for step in old['steps']}
        if [step['step'] for step in record['steps']] != [4, 5, 6]:
            raise ValueError('Incomplete late conditional trajectory')
        forwards = 0
        for step in record['steps']:
            if step['previous_skips'] != prefix:
                raise ValueError('Candidate labels do not follow the joint greedy trajectory')
            costs = step['candidate_costs']
            expected = set(range(2, 40)) - set(prefix)
            if len(costs) != len(expected) or {layer for layer, _ in costs} != expected:
                raise ValueError('Incomplete or duplicate candidate set')
            if any(not math.isfinite(cost) or cost < -1e-6 for _, cost in costs):
                raise ValueError('Nonfinite/negative candidate KL')
            best_layer, best_cost = min(costs, key=lambda item: (item[1], item[0]))
            if step['selected_layer'] != best_layer:
                raise ValueError('Greedy choice is not the lowest joint candidate KL')
            close(step['joint_continuation_kl'], best_cost)
            key = tuple(sorted(prefix))
            if step['reused_expert_costs'] != (key in lookup):
                raise ValueError('Invalid expert-score reuse')
            if key in lookup:
                if costs != lookup[key]['candidate_costs']:
                    raise ValueError('Reused candidate costs changed')
                reused += 1
            else:
                unique_added += 1
                forwards += len(costs)
            candidate_count += len(costs)
            prefix = prefix + [best_layer]
        if record['ordered_skips'] != prefix or len(prefix) != len(set(prefix)) or not set(prefix) <= set(range(2, 40)):
            raise ValueError('Invalid full branch route')
        if forwards != record['new_candidate_forwards']:
            raise ValueError('Incorrect candidate-forward count')
        new_forwards += forwards
    expected_generation = {'documents': len(branches), 'states': len(branches) * 3,
        'sources': dict(Counter(row['source'] for row in branches)), 'candidate_scores': candidate_count,
        'new_candidate_forwards': new_forwards, 'reused_expert_states': reused}
    if summary['generation'] != expected_generation or json.loads((run / 'generation_summary.json').read_text()) != expected_generation:
        raise ValueError('Generation summary does not reproduce')
    records = [old_records[s] for s in protocol['train_ids']]
    branch_map = {row['sample']: row for row in branches}
    baseline = pack_documents(records, branch_map, 'baseline')
    augmented = pack_documents(records, branch_map, 'augmented')
    for left, right in zip(baseline, augmented):
        np.testing.assert_array_equal(left[:, :6].numpy(), right[:, :6].numpy())
    np.testing.assert_array_equal(baseline[0].sum(-1).numpy(), augmented[0].sum(-1).numpy())
    for state, target, available, chosen in [baseline, augmented]:
        if tuple(state.shape) != (len(branches), 9, 40):
            raise ValueError('Training slot budget changed')
        np.testing.assert_allclose(target.sum(-1).numpy(), 1., atol=2e-6)
        if available[:, :, :2].any() or (available & state.bool()).any():
            raise ValueError('Protected/selected layers are available for skipping')
    arms = {row['arm']: row for row in summary['arms']}
    if set(arms) != {'baseline', 'augmented'}:
        raise ValueError('Missing training arm')
    if len({r['initial_parameter_sha256'] for r in arms.values()}) != 1:
        raise ValueError('Different initial weights')
    if len({r['parameter_count'] for r in arms.values()}) != 1:
        raise ValueError('Different parameter counts')
    quality = []
    for arm, row in arms.items():
        if row['gradient_updates'] != math.ceil(len(branches) / 16) * protocol['epochs']:
            raise ValueError('Gradient-update budget changed')
        if row['state_presentations_per_epoch'] != len(branches) * 9:
            raise ValueError('State-presentation budget changed')
        history = json.loads((run / f'{arm}_history.json').read_text())
        if [r['epoch'] for r in history] != list(range(1, protocol['epochs'] + 1)):
            raise ValueError('Incomplete training history')
        validations = [r for r in history if 'continuation_kl' in r]
        best = min(validations, key=lambda r: r['continuation_kl'])
        close(best['continuation_kl'], row['best_selection_kl'])
        if not row['best_checkpoint'].endswith(f"epoch_{best['epoch']:03d}.pth"):
            raise ValueError('Selected wrong epoch')
        if sha256(row['best_checkpoint']) != row['best_checkpoint_sha256']:
            raise ValueError('Checkpoint hash changed')
        if not set(row['cpu_ids']) <= set(range(21)) or row['threads'] > 10:
            raise ValueError('CPU budget exceeded')
        routes = json.loads((run / f'{arm}_routes.json').read_text())
        route_map = {r['sample']: r['ordered_skips'] for r in routes}
        if len(routes) != len(route_map) or sorted(route_map) != protocol['validation_ids']:
            raise ValueError('Missing validation decisions')
        donors = json.loads((run / f'{arm}_donors.json').read_text())
        for mapping in donors:
            if sorted(map(int, mapping.keys())) != protocol['validation_ids'] or sorted(mapping.values()) != protocol['validation_ids']:
                raise ValueError('Invalid shuffled donor population')
            if any(int(s) == donor for s, donor in mapping.items()):
                raise ValueError('Shuffled control includes a self-donor')
        rows = read_lines(run / f'{arm}_quality.jsonl')
        if len(rows) != len(protocol['validation_ids']) * 9:
            raise ValueError('Incomplete evaluation rows')
        for record in rows:
            sample, policy = record['sample'], record['policy']
            if record['split'] != 'validation' or sample not in protocol['validation_ids']:
                raise ValueError('Evaluation split contamination')
            if policy == 'full':
                expected = []
                close(record['continuation_kl'], 0)
            elif policy == 'router':
                expected = route_map[sample]
            elif policy == 'oracle_greedy':
                expected = old_records[sample]['ordered_skips']
                close(record['continuation_kl'], old_records[sample]['continuation_kl'])
            elif policy in ['static_old', 'static_learned']:
                expected = protocol[policy]
            else:
                index = int(policy.removeprefix('shuffled_'))
                expected = route_map[donors[index][str(sample)]]
            if record['skipped_layers'] != sorted(expected):
                raise ValueError('Quality record used a different route')
            if expected and (len(expected) != 6 or len(set(expected)) != 6 or not set(expected) <= set(range(2, 40))):
                raise ValueError('Invalid route budget')
        quality.extend(rows)
    tables = build_tables(quality, protocol['validation_ids'], protocol['selection_ids'], protocol['seed'])
    for key, value in tables.items():
        if json.dumps(value, sort_keys=True) != json.dumps(summary[key], sort_keys=True):
            raise ValueError('Summary does not reproduce: ' + key)
    tracked = json.loads((run / 'preservation_before.json').read_text())
    for path, digest in tracked.items():
        if sha256(path) != digest:
            raise ValueError('Preexisting evidence changed: ' + path)
    if sha256(protocol['data_file']) != protocol['data_sha256'] or sha256(protocol['old_checkpoint']['path']) != protocol['old_checkpoint']['sha256']:
        raise ValueError('Old input/checkpoint changed')
    return {'training_documents': len(branches), 'verified_branch_states': len(branches) * 3,
        'verified_branch_candidate_scores': candidate_count, 'new_candidate_forwards': new_forwards,
        'unique_added_states': unique_added, 'verified_quality_rows': len(quality),
        'equal_initialization_and_capacity': True, 'equal_gradient_updates': True,
        'equal_late_step_weighting': True, 'preserved_existing_files': len(tracked),
        'test_used': False, 'probe': protocol['probe'], 'summary_sha256': sha256(run / 'summary.json')}


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
