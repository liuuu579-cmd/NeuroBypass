"""CPU audit for the stage11 fixed-prefix downstream follow-up."""
import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from horizontal_data_v1 import DATA, HARNESS
from horizontal_quality_v3 import summary_tables
from stage1_benchmark_v2 import sha256, write_json


def verify(run):
    run = Path(run).resolve()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((run / 'protocol.json').read_text())
    manifest = json.loads((run / 'data_manifest.json').read_text())
    assert sha256(run / 'data_manifest.json') == config['data_manifest_sha256']
    assert sha256(run / 'private_benchmark_inputs.jsonl') == manifest['private_input_sha256']
    for name, task in manifest['tasks'].items():
        source = DATA['wikitext' if name == 'wikitext_conditional' else name] / f"{name.replace('_conditional', '')}-{task['split']}.arrow"
        assert sha256(source) == task['sha256']
    for name, digest in config['source_sha256'].items():
        assert sha256(root / 'tools' / name) == digest
    assert subprocess.check_output(['git', '-C', str(HARNESS), 'rev-parse', 'HEAD'], text=True).strip() == config['harness']['commit']
    for name, digest in config['harness']['task_files'].items():
        assert sha256(HARNESS / 'lm_eval/tasks' / name) == digest
    stage6 = Path('runs/stage6_depth_train_main_20261107_01')
    stage10 = Path('runs/stage10_prefix_main_20261109_01')
    for base, group in [(stage6, 'stage6_sha256'), (stage10, 'stage10_sha256')]:
        for name, digest in config[group].items():
            assert sha256(base / name) == digest
    snapshot = json.loads((run / 'preservation_before.json').read_text())
    for name, digest in snapshot.items():
        assert sha256(name) == digest, name
    assert json.loads((run / 'preservation_after.json').read_text())['preserved_files'] == len(snapshot)
    allocation = json.loads((run / 'allocation.json').read_text())
    assert allocation['cpu_count'] == len(set(allocation['cpu_ids'])) == 42
    assert len(set(allocation['gpu_indices'])) == 4
    cores, probes = set(), 0
    for rank in range(4):
        probe = json.loads((run / f'worker_probe_{rank:03d}.json').read_text())
        assert len(probe['cpu_ids']) == probe['threads'] == 10
        assert not cores.intersection(probe['cpu_ids'])
        cores.update(probe['cpu_ids'])
        assert all(set(cpus) <= set(probe['cpu_ids']) for cpus in probe['thread_cpu_ids'].values())
        for item in probe['likelihood_probes']:
            assert abs(item['score'] - item['reference_score']) <= item['tolerance'] == 2e-4
            probes += 1
    assert cores <= set(allocation['cpu_ids'])
    policies = {(p['name'], p['seed'], p['k']): p for p in config['policies']}
    assert len(policies) == 27 and config['budgets'] == [10, 12]
    static = json.loads((stage6 / 'static_paths.json').read_text())
    fixed = json.loads((stage6 / 'fixed_global_policy.json').read_text())['ordered_skips']
    inputs = [json.loads(line) for line in (run / 'private_benchmark_inputs.jsonl').open()]
    original = {(r['task'], r['sample']): r for r in inputs}
    assert len(original) == len(inputs)
    seen = set()
    rows = []
    for path in sorted(run.glob('quality_*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            key = row['policy'], row['seed'], row['k']
            assert key in policies
            policy = policies[key]
            record = original[row['task'], row['sample']]
            unique = (row['task'], row['sample']) + key
            assert unique not in seen
            seen.add(unique); rows.append(row)
            assert row['document_sha256'] == record['document_sha256']
            assert row['document'] == record.get('document')
            assert row['router_observation_tokens'] == len(record['route_ids']) <= 128
            for name in ['target_lengths', 'normalization_lengths', 'truncated_context_tokens']:
                assert row[name] == record[name]
            mask = row['skipped_layers']
            assert mask == sorted(set(mask)) and len(mask) == policy['k']
            assert all(2 <= layer < 40 for layer in mask)
            if policy['kind'] == 'static':
                assert mask == sorted(policy['skips'])
            else:
                assert set(policy['prefix']) <= set(mask)
                assert policy['name'] in {'m00', 'm04', 'm06', 'm08'}
            scores = np.asarray(row['scores'])
            assert len(scores) == len(record['sequence_ids']) and np.isfinite(scores).all() and (scores <= 0).all()
    assert len(seen) == len(inputs) * len(policies)
    summary = json.loads((run / 'summary.json').read_text())
    expected = summary_tables(rows)
    assert summary['quality'] == expected
    assert summary['quality_rows'] == len(rows) and summary['input_records'] == len(inputs)
    result = {'status': 'passed', 'probe': config['probe'], 'verified_rows': len(rows),
        'records_by_task': dict(Counter(r['task'] for r in inputs)), 'policies': len(policies),
        'likelihood_reference_probes': probes, 'source_preserved_files': len(snapshot),
        'worker_cpu_union': sorted(cores), 'cpu_budget': 42,
        'limits': ['Checks recorded routes and score aggregation; not an independent GPU replay of all decisions.',
            'This follow-up reuses checkpoints selected before this task dataset.']}
    write_json(run / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run)), flush=True)


if __name__ == '__main__':
    main()
