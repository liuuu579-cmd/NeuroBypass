"""CPU-only independent checks before accepting a deeper-router run."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from prompt_depth_core_v1 import validate_steps
from stage1_benchmark_v2 import sha256, write_json
from train_prompt_depth_four_gpu_v4 import aggregate, lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    if (run / 'verification.json').exists(): raise FileExistsError('Verification output must be new')
    config = json.loads((run / 'protocol.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['exit_code'] == 0 and status['state'] == 'completed'
    train, validation = config['train_ids'], config['validation_ids']
    assert not set(train) & set(validation)
    assert set(config['static_calibration_ids']) <= set(train)
    assert set(config['selection_ids']) <= set(validation)
    assert config['old_test_used'] is False and config['new_router_max_k'] == 24
    assert sha256(config['data_file']) == config['data_sha256']
    assert sha256(config['fixed_prefix_source']) == config['fixed_prefix_source_sha256']
    global_policy = json.loads(Path(config['fixed_prefix_source']).read_text())
    assert global_policy['validation_used'] is False
    assert set(global_policy['calibration_ids']) <= set(train)
    assert global_policy['ordered_skips'] == config['fixed_prefix']
    root = Path(__file__).resolve().parents[1]
    for name, digest in config['source_sha256'].items(): assert sha256(root / 'tools' / name) == digest
    snapshot = json.loads((run / 'preservation_before.json').read_text())
    for name, digest in snapshot.items(): assert sha256(name) == digest, name
    sets, states, candidates = [], 0, 0
    for arm in ['pure', 'hybrid']:
        prefix = [] if arm == 'pure' else config['fixed_prefix']
        records = [r for path in sorted(run.glob(f'{arm}_labels_*.jsonl')) for r in lines(path)]
        branches = [r for path in sorted(run.glob(f'{arm}_branches_*.jsonl')) for r in lines(path)]
        assert sorted(r['sample'] for r in records) == sorted(train + validation)
        assert sorted(r['sample'] for r in branches) == train
        for row in records:
            validate_steps(row, prefix)
            assert row['split'] == ('train' if row['sample'] in train else 'validation')
            states += len(row['steps']); candidates += sum(len(s['candidate_costs']) for s in row['steps'])
        for row in branches:
            validate_steps(row, row['initial_prefix'])
            assert row['split'] == 'train'
            if arm == 'hybrid': assert row['initial_prefix'][:6] == config['fixed_prefix']
            states += len(row['steps']); candidates += sum(len(s['candidate_costs']) for s in row['steps'])
    cpu_sets = [set(json.loads(path.read_text())['cpu_ids']) for path in sorted(run.glob('worker_probe_*.json'))]
    assert len(cpu_sets) == 4 and all(len(s) == 10 for s in cpu_sets)
    assert len(set.union(*cpu_sets)) == 40
    allocation = json.loads((run / 'allocation.json').read_text())
    assert allocation['cpu_count'] == 42 and len(allocation['gpu_indices']) == 4
    assert set.union(*cpu_sets) <= set(allocation['cpu_ids'])
    static_steps = lines(run / 'static_search_steps.jsonl')
    assert len(static_steps) == 48
    paths = json.loads((run / 'static_paths.json').read_text())
    for objective, policy in [('kl', 'static_joint_kl'), ('nll', 'sleb_nll_opt')]:
        ordered = [r for r in static_steps if r['objective'] == objective]
        record = {'steps': ordered, 'ordered_skips': paths[policy]}
        validate_steps(record)
    for arm in ['pure', 'hybrid']:
        for seed in config['seeds']:
            history = json.loads((run / f'{arm}_seed{seed}_history.json').read_text())
            selected = json.loads((run / f'{arm}_seed{seed}_checkpoints.json').read_text())
            assert len(history) == config['epochs']
            for k in config['budgets']:
                entry = selected[str(k)]
                scores = [r['selection_kl_by_k'][str(k)] for r in history if 'selection_kl_by_k' in r]
                assert entry['selection_continuation_kl'] == min(scores)
                assert sha256(entry['path']) == entry['sha256']
            routes = json.loads((run / f'{arm}_seed{seed}_routes.json').read_text())
            assert len(routes) == len(validation) * len(config['budgets'])
            for row in routes:
                path = row['ordered_skips']
                assert len(path) == row['k'] == len(set(path)) and all(2 <= i < 40 for i in path)
                if arm == 'hybrid': assert path[:6] == config['fixed_prefix']
    rows = [r for path in sorted(run.glob('*quality*.jsonl')) for r in lines(path)]
    keys = [(r['sample'], r['arm'], r['seed'], r['k'], r['policy']) for r in rows]
    assert len(keys) == len(set(keys))
    for row in rows:
        assert row['sample'] in validation and row['split'] == 'validation'
        assert len(row['skipped_layers']) == row['k'] and len(set(row['skipped_layers'])) == row['k']
        assert all(math.isfinite(row[m]) for m in ['continuation_kl', 'continuation_nll'])
    summary = json.loads((run / 'summary.json').read_text())
    computed = aggregate(run, config)
    for key, value in computed.items(): assert summary[key] == value, key
    caches = [json.loads(path.read_text()) for path in sorted(run.glob('cache_probe_*.json'))]
    assert len(caches) == 4 and all(len(items) == 1 for items in caches)
    write_json(run / 'verification.json', {'probe': config['probe'], 'source_preserved_files': len(snapshot),
        'verified_label_states': states, 'verified_candidate_scores': candidates,
        'verified_quality_rows': len(rows), 'training_documents': len(train),
        'validation_documents': len(validation), 'seeds': config['seeds'], 'budgets': config['budgets'],
        'summary_sha256': sha256(run / 'summary.json'), 'old_test_used': False,
        'cpu_worker_union': 40, 'cpu_task_budget': 42,
        'limits': 'Checks recorded choices, scores, provenance and aggregates; does not independently rerun every model forward.'})
    print(json.dumps({'verified': str(run), 'quality_rows': len(rows), 'label_states': states}), flush=True)


if __name__ == '__main__': main()
