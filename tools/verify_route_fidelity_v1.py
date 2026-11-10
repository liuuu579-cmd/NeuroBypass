"""CPU audit for the frozen route-fidelity crossover."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from stage1_benchmark_v2 import sha256, write_json


def verify(run):
    run = Path(run).resolve()
    config = json.loads((run / 'protocol.json').read_text())
    allocation = json.loads((run / 'allocation.json').read_text())
    assert json.loads((run / 'status.json').read_text())['state'] == 'completed'
    assert allocation['cpu_count'] == 42 and len(set(allocation['gpu_indices'])) == 4
    for path, digest in config['source_sha256'].items():
        assert sha256(path) == digest, path
    for path, digest in config['provenance_sha256'].items():
        assert sha256(path) == digest, path
    for rank in range(4):
        probe = json.loads((run / f'worker_probe_{rank:03d}.json').read_text())
        assert probe['threads'] == 10
        assert probe['cpu_ids'] == allocation['cpu_ids'][rank*10:(rank+1)*10]
        assert all(set(v) <= set(probe['cpu_ids']) for v in probe['thread_cpu_ids'].values())
        for item in probe['likelihood_probes']:
            assert item['absolute_error'] <= item['tolerance'] == 2e-4
    inputs = [json.loads(line) for line in (run / 'private_inputs.jsonl').open()]
    assignments = json.loads((run / 'assignments.json').read_text())
    assert sha256(run / 'private_inputs.jsonl') == config['private_input_sha256']
    assert sha256(run / 'assignments.json') == config['assignments_sha256']
    assert len({(r['task'], r['sample']) for r in inputs}) == len(inputs)
    assert len(assignments) == len(inputs) * 3
    for a in assignments:
        assert a['sample'] != a['donor_sample']
        if a['task'] == 'wikitext_conditional':
            assert a['recipient_document'] != a['donor_document']
    expected = len(inputs) * 30
    rows = []
    for path in sorted(run.glob('diagnostics_*.jsonl')):
        rows.extend(json.loads(line) for line in path.open())
    assert len(rows) == expected, (len(rows), expected)
    input_keys = {(r['task'], r['sample']) for r in inputs}
    seen = set()
    for row in rows:
        key = (row['task'], row['sample'], row['seed'], row['k'], row['condition'])
        assert key not in seen; seen.add(key)
        assert (row['task'], row['sample']) in input_keys
        assert row['skipped_layers'] == sorted(set(row['skipped_layers']))
        assert len(row['skipped_layers']) == row['k']
        assert all(2 <= i < 40 for i in row['skipped_layers'])
        assert math.isfinite(row['teacher_kl_sum']) and row['teacher_kl_sum'] >= 0
        assert row['target_tokens'] > 0
    assert len(seen) == expected
    snapshot = json.loads((run / 'preservation_before.json').read_text())
    for path, digest in snapshot.items():
        assert sha256(path) == digest, path
    assert json.loads((run / 'preservation_after.json').read_text())['preserved_files'] == len(snapshot)
    result = {'status': 'passed', 'records': len(inputs), 'diagnostic_rows': len(rows),
        'conditions': 5, 'policies': 6, 'cpu_budget': 42,
        'source_preserved_files': len(snapshot),
        'likelihood_reference_probes': sum(len(json.loads((run / f'worker_probe_{r:03d}.json').read_text())['likelihood_probes']) for r in range(4)),
        'limits': ['Frozen development crossover; not an independent final test.',
                   'Future-target minimum KL is diagnostic only and is not deployable.']}
    write_json(run / 'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().run)), flush=True)
