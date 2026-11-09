"""Fixed-prefix ablation helpers; existing programs and evidence stay read-only."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnose_prompt_router_v2 import paired_summary
from prompt_fixed_core_v1 import best_candidate


NEW_PREFIX_COUNTS = (2, 4, 8)
BUDGETS = (10, 12)


def arm_name(m):
    if m not in NEW_PREFIX_COUNTS:
        raise ValueError('Unsupported new prefix count')
    return f'm{m:02d}'


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def fork_prefix(record, sample, seed, fixed):
    """Same low-cost fork rule as stage6, with depths relative to fixed m."""
    m = len(fixed)
    if record['initial_prefix'] != list(fixed):
        raise ValueError('Expert route does not start from its fixed prefix')
    depths = [m + 1, m + 5, m + 9, m + 13]
    depth = depths[(sample + seed) % len(depths)]
    state = next(row for row in record['steps'] if row['step'] == depth)
    ranked = sorted(state['candidate_costs'], key=lambda row: (row[1], row[0]))
    alternative = 1 + ((sample * 7 + seed) % 4)
    prefix = record['ordered_skips'][:depth - 1] + [ranked[alternative][0]]
    if prefix[:m] != list(fixed) or len(prefix) != len(set(prefix)):
        raise ValueError('Fork changed the fixed prefix or duplicated a layer')
    if set(prefix) == set(record['ordered_skips'][:depth]):
        raise ValueError('Fork must leave the expert state')
    return prefix, alternative + 1


def reuse_steps(records, sample):
    """Cache keys are entire skipped sets; every record must be this document."""
    cache = {}
    for record in records:
        if record['sample'] != sample:
            raise ValueError('Cannot reuse a candidate state from another document')
        for step in record['steps']:
            key = tuple(sorted(step['previous_skips']))
            if key in cache and sorted(cache[key]['candidate_costs']) != sorted(step['candidate_costs']):
                raise ValueError('Identical document/state has inconsistent candidate costs')
            cache[key] = step
    return {'steps': list(cache.values())}


def combined_quality(run, config):
    """Reuse audited stage6 endpoints without treating static masks as 3 seeds."""
    rows = []
    for path in sorted(Path(run).glob('m*_seed*_quality.jsonl')):
        for row in read_rows(path):
            m = int(row['arm'][1:])
            rows.append({**row, 'm': m, 'origin': 'new_matched_training'})
    source = Path(config['source_run'])
    for arm, m in [('pure', 0), ('hybrid', 6)]:
        for seed in config['seeds']:
            for row in read_rows(source / f'{arm}_seed{seed}_quality.jsonl'):
                if row['sample'] in config['validation_ids'] and row['k'] in BUDGETS:
                    rows.append({**row, 'm': m, 'origin': 'audited_stage6_endpoint'})
    for path in sorted(source.glob('static_quality_*.jsonl')):
        for row in read_rows(path):
            if (row['sample'] in config['validation_ids'] and row['k'] in BUDGETS
                    and row['policy'] == 'static_joint_kl'):
                rows.append({**row, 'm': row['k'], 'origin': 'audited_stage6_static'})
    return rows


def aggregate(run, config):
    rows = combined_quality(run, config)
    populations = {
        'validation_all100': config['validation_ids'],
        'selection32': config['selection_ids'],
        'other_validation': [s for s in config['validation_ids'] if s not in config['selection_ids']],
    }
    quality, comparisons = [], []
    for population, samples in populations.items():
        if not samples:
            continue
        groups = defaultdict(list)
        for row in rows:
            if row['sample'] in samples:
                groups[row['m'], row['seed'], row['k'], row['policy']].append(row)
        for (m, seed, k, policy), values in sorted(groups.items(), key=lambda item: str(item[0])):
            nll = float(np.mean([r['continuation_nll'] for r in values]))
            quality.append({
                'population': population, 'm': m, 'seed': seed, 'k': k, 'policy': policy,
                'documents': len(values), 'continuation_kl': float(np.mean([r['continuation_kl'] for r in values])),
                'continuation_nll': nll, 'conditional_ppl': math.exp(nll),
                'unique_masks': len({tuple(r['skipped_layers']) for r in values}),
                'origin': values[0]['origin'],
            })
        keyed = {(r['m'], r['seed'], r['k'], r['policy'], r['sample']): r for r in rows}
        for m in NEW_PREFIX_COUNTS:
            for k in BUDGETS:
                for metric in ['continuation_kl', 'continuation_nll']:
                    for reference in ['shuffled', 'm6', 'static_joint_kl']:
                        # Average model seeds within each document before paired bootstrap.
                        differences = []
                        for sample in samples:
                            per_seed = []
                            for seed in config['seeds']:
                                a = keyed[m, seed, k, 'matched', sample][metric]
                                ref = ((m, seed, k, 'shuffled', sample) if reference == 'shuffled'
                                    else (6, seed, k, 'matched', sample) if reference == 'm6'
                                    else (k, None, k, 'static_joint_kl', sample))
                                per_seed.append(a - keyed[ref][metric])
                            differences.append(float(np.mean(per_seed)))
                        comparisons.append({'population': population, 'm': m, 'k': k,
                            'reference': reference, 'metric': metric, 'models': len(config['seeds']),
                            **paired_summary(differences)})
    return {'quality': quality, 'paired_comparisons': comparisons,
        'quality_rows_including_reused_controls': len(rows)}
