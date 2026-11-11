"""Summarize audited Stage 13 runs without publishing question-level records."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def quality_rows(run):
    for path in sorted(run.glob('quality_*.jsonl')):
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)


def interval(deltas, weights, groups, seed):
    _, inverse = np.unique(groups, return_inverse=True)
    size = int(inverse.max()) + 1
    numerator = np.bincount(inverse, weights=deltas)
    denominator = np.bincount(inverse, weights=weights)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(20):
        sample = rng.integers(size, size=(100, size))
        draws.extend((numerator[sample].sum(1) / denominator[sample].sum(1)).tolist())
    return np.quantile(draws, [.025, .975]).tolist()


def horizontal_comparisons(run):
    groups = defaultdict(dict)
    for row in quality_rows(run):
        if row['policy'] not in {'m06', 'static_joint_kl'}:
            continue
        task = row['task']
        key = (row['sample'], row['policy'], row['seed'])
        group = groups[task, row['k']]
        if key in group or row['condition'] != 'matched':
            raise ValueError('Invalid horizontal coverage')
        if task == 'wikitext_conditional':
            value, count, unit = row['nll_sum'], row['scored_tokens'], row['document']
        else:
            value = 100 * row['correct' if task == 'winogrande' else 'correct_norm']
            count, unit = 1, row['sample']
        group[key] = (value, count, unit)
    summary = load(run/'summary.json')['quality']
    result = []
    for index, ((task, k), group) in enumerate(sorted(groups.items())):
        wiki = task == 'wikitext_conditional'
        count = next(r['records'] for r in summary if r['task'] == task and r['policy'] == 'full')
        keys = sorted({key[0] for key in group})
        if len(keys) != count or len(group) != 4 * count:
            raise ValueError('Missing matched questions or model seeds')
        dynamic, static, weights, units = [], [], [], []
        for sample in keys:
            base = group[sample, 'static_joint_kl', None]
            models = [group[sample, 'm06', seed] for seed in [579, 580, 581]]
            if any(r[1:] != base[1:] for r in models):
                raise ValueError('Paired target coverage differs')
            dynamic.append(sum(r[0] for r in models)/3)
            static.append(base[0]); weights.append(base[1]); units.append(base[2])
        a, b = sum(dynamic)/sum(weights), sum(static)/sum(weights)
        expected = [r for r in summary if r['task'] == task and r['k'] == k and r['policy'] == 'm06']
        expected_dynamic = np.mean([r['nll_per_token'] if wiki else 100*r.get('accuracy_norm', r['accuracy']) for r in expected])
        if not math.isclose(a, expected_dynamic, abs_tol=1e-10):
            raise ValueError('Paired values disagree with audited aggregate')
        delta = np.asarray(dynamic)-static
        result.append({'task': task, 'k': k, 'questions_or_windows': count,
            'bootstrap_units': len(set(units)), 'dynamic': math.exp(a) if wiki else a,
            'static_joint_kl': math.exp(b) if wiki else b, 'paired_difference': a-b,
            'difference_unit': 'nll_per_token' if wiki else 'percentage_points',
            'ci95': interval(delta, weights, units, 13000+index)})
    if len(result) != 12:
        raise ValueError('Expected four tasks and three budgets')
    return result


def c4_comparisons(run, selection):
    rows = defaultdict(dict)
    for row in quality_rows(run):
        if row['task'] != 'c4_development' or row['k'] not in [10, 12] or row['policy'] != 'm06':
            continue
        key = (row['seed'], row['condition'])
        group = rows[row['sample'], row['k'], row['router_observation_tokens']]
        if key in group: raise ValueError('Duplicate C4 row')
        group[key] = row['teacher_kl_sum']/row['target_tokens']
    result = []
    for k in [10, 12]:
        for subset in ['all', 'over128', 'over128_outside_checkpoint_selection']:
            records = [(s, r) for (s, kk, p), r in sorted(rows.items()) if kk == k and (
                subset == 'all' or (p > 128 and (subset == 'over128' or s not in selection)))]
            seeds = [579, 580, 581]
            matched = np.asarray([np.mean([r[s, 'matched'] for s in seeds]) for _, r in records])
            shuffled = np.asarray([np.mean([r[s, f'shuffled_{n}'] for s in seeds for n in [1701, 1702, 1703]]) for _, r in records])
            diff = shuffled-matched
            result.append({'k': k, 'population': subset, 'documents': len(records),
                'matched': float(matched.mean()), 'shuffled': float(shuffled.mean()),
                'matched_benefit': float(diff.mean()), 'ci95': interval(diff, np.ones(len(diff)), [s for s, _ in records], 13200+k)})
    return result


def cost_comparisons(run):
    result = []
    for file in sorted(run.glob('worker_*.json')):
        worker = load(file); deployments = defaultdict(list)
        for row in worker['records']:
            if not row['cold']:
                deployments[row['deployment']].append(row)
        summary = {}
        for name, rows in deployments.items():
            summary[name] = {'requests': len(rows), 'prompt_tokens': sorted({r['prompt_tokens'] for r in rows}),
                'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in rows)/2**30,
                'mean_ttft_seconds': float(np.mean([r['ttft_seconds'] for r in rows])),
                'mean_end_to_end_tokens_per_second': float(np.mean([r['end_to_end_tokens_per_second'] for r in rows]))}
        cold = next(r for r in worker['records'] if r['cold'])
        result.append({'rank': worker['rank'], 'k': worker['policy']['k'], 'seed': worker['policy']['seed'],
            'cold_selective_ttft_seconds_excluding_cpu_model_setup': cold['ttft_seconds'],
            'cpu_model_load_and_shared_setup_seconds': worker['cpu_model_load_and_shared_setup_seconds'],
            'sampled_selective_process_memory': worker['selective_startup_process_memory'], 'deployments': summary})
    return result


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = {k: Path(v) for k, v in load(a.runs).items()}
    for run in runs.values():
        assert load(run/'status.json')['state'] == 'completed' and load(run/'independent_audit.json')['passed']
    selected = set(load(runs['training']/'protocol.json')['selection_ids'])
    local = [p for k in ['development', 'horizontal'] for p in sorted(runs[k].glob('routes_*.json'))]
    result = {'horizontal_m06_vs_static': horizontal_comparisons(runs['horizontal']),
        'c4_m06_vs_shuffled': c4_comparisons(runs['development'], selected),
        'cost': cost_comparisons(runs['cost']),
        'method': 'Average seeds within input first; 2000 paired bootstrap replicates over questions or C4 documents, over articles for token-weighted Wiki NLL. Unadjusted exploratory intervals. C4 permutations can cross input lengths.',
        'local_only_route_registries': [{'run': p.parent.name, 'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in local]}
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__': main()
