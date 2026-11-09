"""Independent CPU audit of prefix ablation labels, checkpoints and controls."""
import argparse
import json
import math
from pathlib import Path

from prefix_ablation_core_v1 import (BUDGETS, NEW_PREFIX_COUNTS, aggregate, arm_name,
    combined_quality, fork_prefix, read_rows)
from prompt_depth_core_v1 import validate_steps
from stage1_benchmark_v2 import sha256, write_json


def verify(run):
    run = Path(run).resolve()
    config = json.loads((run / 'protocol.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'completed' and status['exit_code'] == 0
    assert config['old_test_used'] is False and config['weights_frozen'] is True
    assert config['max_k'] == 24 and config['budgets'] == list(BUDGETS)
    assert config['prefix_counts_to_train'] == list(NEW_PREFIX_COUNTS)
    source = Path(config['source_run'])
    old = json.loads((source / 'protocol.json').read_text())
    paths = json.loads((source / 'static_paths.json').read_text())
    assert config['global_path'] == paths['static_joint_kl']
    train, validation, selection = config['train_ids'], config['validation_ids'], config['selection_ids']
    assert len(train) == len(set(train)) and len(validation) == len(set(validation))
    assert not set(train) & set(validation) and set(selection) <= set(validation)
    if config['probe']:
        assert train == old['train_ids'][:4] and validation == old['validation_ids'][:4]
        assert config['seeds'] == [579] and config['epochs'] == 1
    else:
        assert train == old['train_ids'] and len(train) == 800
        assert validation == old['validation_ids'] and len(validation) == 100
        assert selection == old['selection_ids'] and len(selection) == 32
        assert config['seeds'] == old['seeds'] == [579, 580, 581]
        assert config['epochs'] == old['epochs'] == 24
    assert config['static_calibration_ids'] == old['static_calibration_ids']
    assert set(config['static_calibration_ids']) <= set(old['train_ids'])
    assert sha256(config['data_file']) == config['data_sha256'] == old['data_sha256']
    for group in ['source_sha256', 'source_evidence_sha256']:
        for path, digest in config[group].items():
            assert sha256(path) == digest, path
    tracked = json.loads((run / 'preservation_before.json').read_text())
    for path, digest in tracked.items():
        assert sha256(path) == digest, path
    assert json.loads((run / 'preservation_after.json').read_text())['preserved_files'] == len(tracked)
    alloc = json.loads((run / 'allocation.json').read_text())
    assert alloc['cpu_count'] == len(set(alloc['cpu_ids'])) == 42
    assert len(set(alloc['gpu_indices'])) == 4
    candidate_probes = endpoint_replays = 0
    for rank in range(4):
        probe = json.loads((run / f'worker_probe_{rank:03d}.json').read_text())
        assert probe['cpu_ids'] == alloc['cpu_ids'][rank * 10:(rank + 1) * 10]
        assert probe['threads'] == 10
        assert all(set(cpus) <= set(probe['cpu_ids']) for cpus in probe['thread_cpu_ids'].values())
        assert len(probe['candidate_probes']) == 3
        for item in probe['candidate_probes']:
            assert item['prefix'] == config['global_path'][:item['m']]
            assert item['max_abs_error'] <= item['tolerance'] == 1e-6
            assert item['max_abs_error'] == max(abs(a[1] - b[1]) for a, b in zip(item['costs'], item['reference']))
            candidate_probes += 1
        assert len(probe['endpoint_replays']) == 2 * 2 * len(config['seeds'])
        for item in probe['endpoint_replays']:
            assert item['sample'] in validation and item['k'] in BUDGETS
            assert max(item['errors'].values()) <= item['tolerance'] == 1e-6
            endpoint_replays += 1
    counts, states, scores, reused_states = [], 0, 0, 0
    for m in NEW_PREFIX_COUNTS:
        arm, prefix = arm_name(m), config['global_path'][:m]
        records = [r for path in sorted(run.glob(f'{arm}_labels_*.jsonl')) for r in read_rows(path)]
        branches = [r for path in sorted(run.glob(f'{arm}_branches_*.jsonl')) for r in read_rows(path)]
        assert sorted(r['sample'] for r in records) == sorted(train + validation)
        assert sorted(r['sample'] for r in branches) == train
        experts = {r['sample']: r for r in records}
        main_states = branch_states = 0
        for row in records + branches:
            assert row['arm'] == arm and row['m'] == m
            assert row['split'] == ('train' if row['sample'] in train else 'validation')
            assert row['initial_prefix'][:m] == prefix
            validate_steps(row, row['initial_prefix'], 24)
            for step in row['steps']:
                assert all(math.isfinite(v) and v >= -1e-5 for _, v in step['candidate_costs'])
                states += 1; scores += len(step['candidate_costs'])
                reused_states += int(step['reused_old_entire_state'])
        for row in records:
            assert row['initial_prefix'] == prefix and len(row['steps']) == 24 - m
            main_states += len(row['steps'])
        for row in branches:
            expected, alternative = fork_prefix(experts[row['sample']], row['sample'], config['label_seed'], prefix)
            assert row['initial_prefix'] == expected
            assert row['expert_prefix_length'] == len(expected) and row['alternative_rank'] == alternative
            branch_states += len(row['steps'])
        counts.append({'m': m, 'expert_records': len(records), 'training_branches': len(branches),
            'expert_states': main_states, 'branch_states': branch_states})
        for seed in config['seeds']:
            history = json.loads((run / f'{arm}_seed{seed}_history.json').read_text())
            checkpoints = json.loads((run / f'{arm}_seed{seed}_checkpoints.json').read_text())
            assert len(history) == config['epochs'] and [r['epoch'] for r in history] == list(range(1, config['epochs'] + 1))
            expected_epochs = [e for e in range(1, config['epochs'] + 1) if e == 1 or e % 4 == 0 or e == config['epochs']]
            assert [r['epoch'] for r in history if 'selection_kl_by_k' in r] == expected_epochs
            assert all(math.isfinite(r['mean_loss']) for r in history)
            for k in BUDGETS:
                entry = checkpoints[str(k)]
                best = min((r['selection_kl_by_k'][str(k)], r['epoch']) for r in history if 'selection_kl_by_k' in r)
                assert (entry['selection_continuation_kl'], entry['epoch']) == best
                assert Path(entry['path']).parent == run and sha256(entry['path']) == entry['sha256']
            routes = json.loads((run / f'{arm}_seed{seed}_routes.json').read_text())
            assert len(routes) == len(validation) * len(BUDGETS)
            assert {(r['sample'], r['k']) for r in routes} == {(s, k) for s in validation for k in BUDGETS}
            for row in routes:
                assert row['ordered_skips'][:m] == prefix
                assert len(row['ordered_skips']) == len(set(row['ordered_skips'])) == row['k']
                assert all(2 <= i < 40 for i in row['ordered_skips'])
    rows = combined_quality(run, config)
    seen = set()
    for row in rows:
        key = row['m'], row['seed'], row['k'], row['policy'], row['sample']
        assert key not in seen; seen.add(key)
        assert row['sample'] in validation and row['split'] == 'validation'
        assert len(row['skipped_layers']) == len(set(row['skipped_layers'])) == row['k']
        assert all(2 <= i < 40 for i in row['skipped_layers'])
        assert set(config['global_path'][:row['m']]) <= set(row['skipped_layers'])
        assert all(math.isfinite(row[name]) for name in ['continuation_kl', 'continuation_nll', 'first_next_token_kl'])
        if row['m'] == row['k']:
            assert row['seed'] is None and row['policy'] == 'static_joint_kl'
    expected_keys = set()
    for sample in validation:
        for k in BUDGETS:
            expected_keys.add((k, None, k, 'static_joint_kl', sample))
            for m in [0, 2, 4, 6, 8]:
                for seed in config['seeds']:
                    for policy in ['matched', 'shuffled']:
                        expected_keys.add((m, seed, k, policy, sample))
    assert seen == expected_keys
    outcomes = json.loads((run / 'training_outcomes.json').read_text())
    assert {(r['m'], r['seed']) for r in outcomes} == {(m, s) for m in NEW_PREFIX_COUNTS for s in config['seeds']}
    assert len(outcomes) == len(NEW_PREFIX_COUNTS) * len(config['seeds'])
    summary = json.loads((run / 'summary.json').read_text())
    for name, value in aggregate(run, config).items():
        assert summary[name] == value, name
    result = {'status': 'passed', 'probe': config['probe'], 'preserved_files': len(tracked),
        'verified_label_states': states, 'verified_candidate_scores': scores,
        'reused_identical_document_states': reused_states, 'states_by_m': counts,
        'verified_training_runs': len(outcomes), 'candidate_probes': candidate_probes,
        'endpoint_replays': endpoint_replays, 'quality_rows_including_reused_controls': len(rows),
        'cpu_budget': 42, 'old_test_used': False,
        'limits': 'Audits recorded search scores and training outputs; not an independent replay of every forward or router update.'}
    write_json(run / 'verification.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.run)), flush=True)


if __name__ == '__main__':
    main()
