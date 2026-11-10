"""Audit corrected prefixes, reused evidence, predictions and score reduction."""
import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from horizontal_data_v1 import DATA, HARNESS
from prefix_downstream_correction_core_v1 import (combined_rows, iter_rows, key,
    partition, policies, validate_policy)
from stage1_benchmark_v2 import sha256, write_json


def verify(run):
    root = Path(__file__).resolve().parents[1]
    run = Path(run).resolve()
    config = json.loads((run / 'protocol.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'completed' and status['exit_code'] == 0
    path, selected = policies(root)
    assert config['global_path'] == path and config['all_policies'] == selected
    original = json.loads((Path(config['reuse_run']) / 'protocol.json').read_text())
    reused, rerun = partition(original['policies'], selected)
    assert reused == config['reused_policies'] and rerun == config['policies']
    assert config['probe'] == original['probe'] and config['budgets'] == [10, 12]
    for policy in selected:
        validate_policy(policy, path)
    for group in ['source_sha256', 'provenance_sha256']:
        for name, digest in config[group].items():
            assert sha256(name) == digest, name
    manifest = json.loads((run / 'data_manifest.json').read_text())
    assert sha256(run / 'data_manifest.json') == config['data_manifest_sha256']
    assert sha256(config['private_inputs']) == manifest['private_input_sha256']
    for task, item in manifest['tasks'].items():
        name = task.replace('_conditional', '')
        assert sha256(DATA[name] / f"{name}-{item['split']}.arrow") == item['sha256']
    assert subprocess.check_output(['git', '-C', str(HARNESS), 'rev-parse', 'HEAD'], text=True).strip() == config['harness']['commit']
    for name, digest in config['harness']['task_files'].items():
        assert sha256(HARNESS / 'lm_eval/tasks' / name) == digest
    snapshot = json.loads((run / 'preservation_before.json').read_text())
    for name, digest in snapshot.items():
        assert sha256(name) == digest, name
    assert json.loads((run / 'preservation_after.json').read_text())['preserved_files'] == len(snapshot)
    allocation = json.loads((run / 'allocation.json').read_text())
    assert len(set(allocation['gpu_indices'])) == 4
    assert allocation['cpu_count'] == len(set(allocation['cpu_ids'])) == 42
    inputs = [json.loads(line) for line in Path(config['private_inputs']).open()]
    records = {(r['task'], r['sample']): r for r in inputs}
    assert len(records) == len(inputs)
    expected_counts = {task: item.get('evaluated_records', item.get('evaluated_windows')) for task, item in manifest['tasks'].items()}
    assert dict(Counter(r['task'] for r in inputs)) == expected_counts
    if config['probe']:
        assert all(v == 4 for v in expected_counts.values())
    else:
        assert expected_counts == {'piqa': 1838, 'hellaswag': 10042, 'winogrande': 1267, 'wikitext_conditional': 746}
    policy_map = {key(p): p for p in selected}
    seen, totals, new_rows = set(), defaultdict(lambda: defaultdict(float)), 0
    masks = defaultdict(set)
    new_keys = {key(p) for p in rerun}
    for row in iter_rows(run):
        assert key(row) in new_keys
        new_rows += 1
    assert new_rows == len(inputs) * 6
    for row in combined_rows(run, config):
        pkey = key(row)
        assert pkey in policy_map
        policy = policy_map[pkey]
        record = records[row['task'], row['sample']]
        uid = (row['task'], row['sample'], pkey)
        assert uid not in seen
        seen.add(uid)
        assert row['document_sha256'] == record['document_sha256']
        assert row['document'] == record.get('document')
        for field in ['target_lengths', 'normalization_lengths', 'truncated_context_tokens']:
            assert row[field] == record[field]
        assert row['router_observation_tokens'] == len(record['route_ids']) <= 128
        mask = row['skipped_layers']
        assert mask == sorted(set(mask)) and len(mask) == policy['k']
        assert all(2 <= layer < 40 for layer in mask)
        if policy['kind'] == 'static':
            assert mask == sorted(policy['skips'])
        else:
            # Validate declared m as well as membership in the actual executed mask.
            assert len(policy['prefix']) == policy['m']
            assert policy['prefix'] == path[:policy['m']]
            assert set(policy['prefix']) <= set(mask)
        scores = np.asarray(row['scores'], dtype=float)
        assert len(scores) == len(record['sequence_ids']) and np.isfinite(scores).all()
        assert (scores <= 0).all()
        group = row['task'], *pkey
        total = totals[group]
        total['records'] += 1
        masks[group].add(tuple(mask))
        if record['gold'] is not None:
            prediction = int(scores.argmax())
            assert row['gold'] == record['gold'] and row['prediction'] == prediction
            assert row['correct'] == int(prediction == record['gold'])
            total['correct'] += int(prediction == record['gold'])
            if record['task'] != 'winogrande':
                normalized = scores / np.asarray(record['normalization_lengths'], dtype=float)
                assert row['prediction_norm'] == int(normalized.argmax())
                assert row['correct_norm'] == int(normalized.argmax() == record['gold'])
                total['correct_norm'] += int(normalized.argmax() == record['gold'])
        else:
            assert row['nll_sum'] == -scores[0]
            assert row['scored_tokens'] == record['target_lengths'][0]
            total['nll_sum'] += -scores[0]
            total['tokens'] += record['target_lengths'][0]
    assert len(seen) == len(inputs) * 27
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['quality_rows'] == len(seen) and summary['new_quality_rows'] == new_rows
    assert summary['reused_quality_rows'] == len(inputs) * 21
    assert summary['policies'] == 27 and summary['input_records'] == len(inputs)
    summary_keys = set()
    for row in summary['quality']:
        group = row['task'], row['policy'], row['seed'], row['k']
        assert group not in summary_keys
        summary_keys.add(group)
        total = totals[group]
        assert row['records'] == total['records'] == expected_counts[row['task']]
        assert row['unique_masks'] == len(masks[group])
        if row['task'] == 'wikitext_conditional':
            assert row['scored_tokens'] == total['tokens']
            for name, value in [('nll_sum', total['nll_sum']), ('nll_per_token', total['nll_sum'] / total['tokens']),
                                ('conditional_ppl', math.exp(total['nll_sum'] / total['tokens']))]:
                assert math.isclose(row[name], value, rel_tol=1e-12, abs_tol=1e-12)
        else:
            assert math.isclose(row['accuracy'], total['correct'] / total['records'], abs_tol=1e-12)
            if row['task'] != 'winogrande':
                assert math.isclose(row['accuracy_norm'], total['correct_norm'] / total['records'], abs_tol=1e-12)
    assert summary_keys == set(totals) and len(summary_keys) == 108
    # Bind each recorded replay to its exact old row and recompute its error.
    replay_lookup = {}
    for rank in range(4):
        local = inputs[rank::4]
        first = {task: next(r for r in local if r['task'] == task) for task in expected_counts}
        replay_lookup[rank] = {(task, r['sample'], key(p)) for task, r in first.items() for p in config['replay_policies']}
    needed = set().union(*replay_lookup.values())
    old_rows = {(r['task'], r['sample'], key(r)): r for r in iter_rows(config['reuse_run'])
                if (r['task'], r['sample'], key(r)) in needed}
    reference_count, replay_count, replay_error = 0, 0, 0
    for rank in range(4):
        probe = json.loads((run / f'worker_probe_{rank:03d}.json').read_text())
        assert probe['cpu_ids'] == allocation['cpu_ids'][rank * 10:(rank + 1) * 10]
        assert probe['threads'] == 10
        assert all(set(cpus) <= set(probe['cpu_ids']) for cpus in probe['thread_cpu_ids'].values())
        assert len(probe['likelihood_probes']) == 8
        assert {(p['task'], tuple(p['skips'])) for p in probe['likelihood_probes']} == {
            (task, tuple(mask)) for task in expected_counts for mask in [[], sorted(path[:8])]} 
        for item in probe['likelihood_probes']:
            assert abs(item['score'] - item['reference_score']) == item['absolute_error'] <= item['tolerance'] == 2e-4
            reference_count += 1
        got = set()
        for item in probe['unchanged_control_replays']:
            uid = item['task'], item['sample'], key(item)
            assert uid not in got
            got.add(uid)
            old = old_rows[uid]
            assert item['skipped_layers'] == old['skipped_layers']
            assert len(item['scores']) == len(old['scores'])
            error = max(abs(a - b) for a, b in zip(item['scores'], old['scores']))
            assert error == item['max_absolute_error'] <= item['tolerance'] == 2e-4
            replay_error = max(replay_error, error)
            replay_count += 1
        assert got == replay_lookup[rank]
    assert replay_count == 80 and reference_count == 32
    result = {'status': 'passed', 'probe': config['probe'], 'verified_rows': len(seen),
        'new_quality_rows': new_rows, 'reused_quality_rows': len(inputs) * 21,
        'policies': 27, 'records_by_task': expected_counts, 'cpu_budget': 42,
        'source_preserved_files': len(snapshot), 'likelihood_reference_probes': reference_count,
        'unchanged_control_gpu_replays': replay_count, 'replay_max_absolute_error': replay_error,
        'fixed_m08_prefix': path[:8], 'independent_prediction_and_nll_reduction': True,
        'limits': ['Full m08 is recomputed; unchanged controls have hash validation and sampled GPU replay.',
                   'Existing downstream development tasks are reused; this is not an independent final test.']}
    write_json(run / 'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().run)), flush=True)
