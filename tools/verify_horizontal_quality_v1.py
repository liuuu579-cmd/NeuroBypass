"""CPU audit of frozen inputs, checkpoints, routes, scores and reported metrics."""
import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from horizontal_data_v1 import HARNESS
from stage1_benchmark_v2 import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((run / 'protocol.json').read_text())
    manifest = json.loads((run / 'data_manifest.json').read_text())
    assert sha256(run / 'data_manifest.json') == config['data_manifest_sha256']
    assert sha256(run / 'private_benchmark_inputs.jsonl') == manifest['private_input_sha256']
    for task in manifest['tasks'].values(): assert sha256(task['path']) == task['sha256']
    for name, digest in config['source_sha256'].items(): assert sha256(root / 'tools' / name) == digest
    assert subprocess.check_output(['git', '-C', str(HARNESS), 'rev-parse', 'HEAD'], text=True).strip() == config['harness']['commit']
    for name, digest in config['harness']['task_files'].items():
        assert sha256(HARNESS / 'lm_eval/tasks' / name) == digest
    snapshot = json.loads((run / 'preservation_before.json').read_text())
    for name, digest in snapshot.items(): assert sha256(name) == digest, name
    assert json.loads((run / 'preservation_after.json').read_text())['preserved_files'] == len(snapshot)
    alloc = json.loads((run / 'allocation.json').read_text())
    assert alloc['cpu_count'] == 42 and len(set(alloc['cpu_ids'])) == 42
    assert len(alloc['gpu_indices']) == 4
    cores, nprobes = set(), 0
    for rank in range(4):
        probe = json.loads((run / f'worker_probe_{rank:03d}.json').read_text())
        assert len(probe['cpu_ids']) == probe['threads'] == 10
        assert not cores.intersection(probe['cpu_ids'])
        cores.update(probe['cpu_ids'])
        assert {r['task'] for r in probe['likelihood_probes']} == set(manifest['tasks'])
        for record in probe['likelihood_probes']:
            assert abs(record['score']-record['reference_score']) <= record['tolerance'] == 2e-4
            nprobes += 1
    assert cores <= set(alloc['cpu_ids'])
    policies = {(p['name'], p['seed'], p['k']): p for p in config['policies']}
    assert len(policies) == 37 and config['budgets'] == [10, 12, 16]
    paths = json.loads((Path(config['training_run']) / 'static_paths.json').read_text())
    assert config['fixed_prefix'] == paths['static_joint_kl'][:6]
    for policy in policies.values():
        if policy['kind'] == 'router':
            assert sha256(policy['checkpoint']['path']) == policy['checkpoint']['sha256']
            assert policy['prefix'] == (config['fixed_prefix'] if policy['name'] == 'hybrid' else [])
        elif policy['name'] != 'full': assert policy['skips'] == paths[policy['name']][:policy['k']]
    inputs = [json.loads(s) for s in (run / 'private_benchmark_inputs.jsonl').read_text().splitlines()]
    original = {(r['task'], r['sample']): r for r in inputs}
    assert len(original) == len(inputs)
    seen, aggregate = set(), defaultdict(lambda: {'n': 0, 'correct': 0, 'correct_norm': 0, 'nll': 0., 'tokens': 0, 'masks': set()})
    for path in sorted(run.glob('quality_*.jsonl')):
        rank = int(path.stem.split('_')[-1])
        for line in path.open():
            row = json.loads(line)
            key = row['policy'], row['seed'], row['k']
            policy = policies[key]
            index = row['task'], row['sample']
            record = original[index]
            unique = index + key
            assert unique not in seen
            seen.add(unique)
            assert row['document_sha256'] == record['document_sha256']
            assert row['document'] == record.get('document')
            assert row['router_observation_tokens'] == len(record['route_ids']) <= 128
            for name in ['target_lengths', 'normalization_lengths', 'truncated_context_tokens']:
                assert row[name] == record[name]
            mask = row['skipped_layers']
            assert mask == sorted(set(mask)) and len(mask) == policy['k']
            assert all(2 <= layer < 40 for layer in mask)
            if policy['kind'] == 'static': assert mask == sorted(policy['skips'])
            else: assert set(policy['prefix']) <= set(mask)
            scores = np.asarray(row['scores'])
            assert len(scores) == len(record['sequence_ids']) and np.isfinite(scores).all()
            assert (scores <= 0).all()
            group = aggregate[(row['task'],) + key]
            group['n'] += 1; group['masks'].add(tuple(mask))
            if record['gold'] is not None:
                correct = int(scores.argmax() == record['gold'])
                assert row['gold'] == record['gold'] and row['prediction'] == int(scores.argmax())
                assert row['correct'] == correct
                group['correct'] += correct
                if row['task'] != 'winogrande':
                    pred = int((scores / np.asarray(record['normalization_lengths'])).argmax())
                    assert row['prediction_norm'] == pred
                    assert row['correct_norm'] == int(pred == record['gold'])
                    group['correct_norm'] += row['correct_norm']
            else:
                assert row['nll_sum'] == -scores[0] and row['scored_tokens'] == record['target_lengths'][0]
                group['nll'] -= scores[0]; group['tokens'] += row['scored_tokens']
    expected_counts = Counter(r['task'] for r in inputs)
    assert len(seen) == len(inputs) * len(policies)
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['quality_rows'] == len(seen) and summary['input_records'] == len(inputs)
    assert len(summary['quality']) == len(aggregate)
    for row in summary['quality']:
        group = aggregate[row['task'], row['policy'], row['seed'], row['k']]
        assert group['n'] == row['records'] == expected_counts[row['task']]
        assert row['unique_masks'] == len(group['masks'])
        if row['task'] == 'wikitext_conditional':
            assert abs(row['nll_sum'] - group['nll']) < 1e-6
            assert row['scored_tokens'] == group['tokens']
            assert abs(row['conditional_ppl'] - math.exp(group['nll']/group['tokens'])) < 1e-8
        else:
            assert row['accuracy'] == group['correct']/group['n']
            if row['task'] != 'winogrande': assert row['accuracy_norm'] == group['correct_norm']/group['n']
    write_json(run / 'verification.json', {'status': 'passed', 'probe': config['probe'],
        'verified_rows': len(seen), 'records_by_task': dict(expected_counts),
        'policies': len(policies), 'likelihood_reference_probes': nprobes,
        'source_preserved_files': len(snapshot), 'worker_cpu_union': sorted(cores),
        'cpu_budget': 42, 'limits': ['CPU audit checks recorded routes, not an independent GPU replay of all router decisions.',
            'Answer-blind input construction audited separately by counterfactual template/tokenization tests.']})
    print(json.dumps({'status': 'passed', 'rows': len(seen), 'probe': config['probe']}), flush=True)


if __name__ == '__main__': main()
