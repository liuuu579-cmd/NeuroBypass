"""Audit equal-capacity training controls and independently reproduce summaries."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from diagnose_prompt_router_v2 import paired_summary
from stage1_benchmark_v2 import sha256, write_json


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Arithmetic mismatch: {a} != {b}')


def verify(root, run):
    status = json.loads((run / 'status.json').read_text())
    if status.get('state') != 'completed' or status.get('exit_code') != 0:
        raise ValueError('Run not successfully complete')
    protocol = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    diagnosis = Path(protocol['diagnosis_reference'])
    diagnosis_protocol = json.loads((diagnosis / 'protocol.json').read_text())
    selected = {r['arm']: r for r in summary['selected']}
    if set(selected) != {'small_prompt', 'small_constant'}:
        raise ValueError('Missing comparison arms')
    if len({r['initial_parameter_sha256'] for r in selected.values()}) != 1:
        raise ValueError('Different initialization')
    if len({r['parameter_count'] for r in selected.values()}) != 1:
        raise ValueError('Different parameter counts')
    by_key = {}
    for arm, result in selected.items():
        history = json.loads((run / f'{arm}_training_history.json').read_text())
        if [r['epoch'] for r in history] != list(range(1, 25)):
            raise ValueError('Incomplete training')
        validation = [r for r in history if 'continuation_kl' in r]
        if [r['epoch'] for r in validation] != [1, 4, 8, 12, 16, 20, 24]:
            raise ValueError('Checkpoint selection schedule changed')
        best = min(validation, key=lambda r: r['continuation_kl'])
        close(best['continuation_kl'], result['selection_kl'])
        if not result['best_checkpoint'].endswith(f"epoch_{best['epoch']:03d}.pth"):
            raise ValueError('Selected wrong epoch')
        if sha256(result['best_checkpoint']) != result['best_checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        routes = json.loads((run / f'{arm}_routes.json').read_text())
        route_map = {r['sample']: r['ordered_skips'] for r in routes}
        if len(routes) != 164 or len(route_map) != 164:
            raise ValueError('Incomplete routes')
        if arm == 'small_constant' and len({tuple(r['ordered_skips']) for r in routes}) != 1:
            raise ValueError('Constant-input model depends on input')
        rows = read_lines(run / f'{arm}_quality.jsonl')
        if len(rows) != (564 if arm == 'small_prompt' else 164):
            raise ValueError('Incomplete quality records')
        for row in rows:
            key = (arm, row['sample'], row['policy'])
            if key in by_key:
                raise ValueError('Duplicate quality record')
            by_key[key] = row
            if row['policy'] == 'matched' and row['skipped_layers'] != sorted(route_map[row['sample']]):
                raise ValueError('Wrong matched route')
            route = row['skipped_layers']
            if len(route) != 6 or len(set(route)) != 6 or not set(route) <= set(range(2, 40)):
                raise ValueError('Invalid route')
        worker = result['worker']
        if not set(worker['cpu_ids']) <= set(range(21)) or worker['threads'] > 10:
            raise ValueError('CPU budget exceeded')
    for path in diagnosis.glob('quality_rows_*.jsonl'):
        for row in read_lines(path):
            if row['policy'] in ['router', 'static16', 'oracle_greedy', 'full']:
                by_key['reference_' + row['policy'], row['sample'], 'matched'] = row
    for sample in diagnosis_protocol['quality_validation_documents']:
        rows = [by_key['small_prompt', sample, f'shuffled_{i}'] for i in range(4)]
        by_key['small_prompt_shuffled', sample, 'matched'] = {
            name: float(np.mean([r[name] for r in rows]))
            for name in ['continuation_kl', 'continuation_nll']}
    populations = {'train_diagnostic64': protocol['quality_train_ids'],
        'validation_all100': diagnosis_protocol['quality_validation_documents'],
        'validation_selection32': protocol['selection_ids'],
        'validation_other68': [s for s in diagnosis_protocol['quality_validation_documents']
                               if s not in set(protocol['selection_ids'])]}
    if len(populations['validation_other68']) != 68:
        raise ValueError('Validation partition changed')
    for row in summary['quality']:
        samples = populations[row['population']]
        group = [by_key[row['policy'], sample, 'matched'] for sample in samples]
        if row['samples'] != len(samples):
            raise ValueError('Incorrect population size')
        for metric in ['continuation_kl', 'continuation_nll']:
            close(row[metric], float(np.mean([r[metric] for r in group])))
        close(row['perplexity'], math.exp(row['continuation_nll']))
    for row in summary['paired_comparisons']:
        samples = populations[row['population']]
        expected = paired_summary([by_key['small_prompt', s, 'matched'][row['metric']] -
            by_key[row['reference'], s, 'matched'][row['metric']] for s in samples])
        for name in ['mean_delta', 'fraction_negative']:
            close(row[name], expected[name])
        np.testing.assert_allclose(row['bootstrap_95pct'], expected['bootstrap_95pct'], rtol=0, atol=1e-12)
    preserved = json.loads((run / 'preservation_before.json').read_text())
    for name, digest in preserved.items():
        if sha256(root / name) != digest:
            raise ValueError('Preexisting source/evidence changed: ' + name)
    if protocol['test_used'] or summary['test_used']:
        raise ValueError('Pilot used test data')
    return {'verified_training_epochs': 48, 'verified_quality_rows': 728,
        'equal_initialization_and_capacity': True, 'constant_input_invariance': True,
        'preserved_existing_files': len(preserved), 'test_used': False,
        'summary_sha256': sha256(run / 'summary.json')}


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
