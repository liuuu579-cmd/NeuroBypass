"""Read-only diagnostics of recorded m6 masks and non-selection C4 fidelity."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'runs/stage6_depth_train_main_20261107_01'
    downstream = root / 'runs/stage11_prefix_downstream_main_20261109_01'
    protocol = json.loads((source / 'protocol.json').read_text())
    documents = sorted(set(protocol['validation_ids']) - set(protocol['selection_ids']))
    assert len(documents) == 68
    diversity = defaultdict(Counter)
    tracked = [source / 'protocol.json', root / 'tools/prompt_cost_core_v1.py']
    for path in sorted(downstream.glob('quality_*.jsonl')):
        tracked.append(path)
        for line in path.open():
            row = json.loads(line)
            if row['policy'] == 'm06':
                diversity[row['task'], row['seed'], row['k']][tuple(row['skipped_layers'])] += 1
    mask_stats = []
    for (task, seed, k), counts in sorted(diversity.items()):
        n = sum(counts.values())
        mask_stats.append({'task': task, 'seed': seed, 'k': k, 'records': n,
            'unique_masks': len(counts), 'dominant_mask_fraction': max(counts.values()) / n,
            'mask_entropy_bits': -sum((c/n)*math.log2(c/n) for c in counts.values()),
            'same_mask_probability_independent_draws': sum(c*c for c in counts.values()) / (n*n)})
    values = {}
    for seed in [579, 580, 581]:
        path = source / f'hybrid_seed{seed}_quality.jsonl'
        tracked.append(path)
        for line in path.open():
            row = json.loads(line)
            if row['sample'] in documents and row['k'] in [10, 12] and row['policy'] in ['matched', 'shuffled']:
                values[row['sample'], seed, row['k'], row['policy']] = row['continuation_kl']
    estimates, rng = [], np.random.default_rng(579)
    for k in [10, 12]:
        a = np.asarray([np.mean([values[s, seed, k, 'matched'] for seed in [579, 580, 581]]) for s in documents])
        b = np.asarray([np.mean([values[s, seed, k, 'shuffled'] for seed in [579, 580, 581]]) for s in documents])
        delta = b-a
        boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        estimates.append({'k': k, 'documents': len(documents), 'matched_kl': float(a.mean()),
            'shuffled_kl': float(b.mean()), 'shuffled_minus_matched_kl': float(delta.mean()),
            'ci95_low': float(lo), 'ci95_high': float(hi), 'bootstrap_replicates': 5000,
            'unit': 'document after averaging three training seeds'})
    write_json(args.output, {'status': 'completed', 'source_sha256': {str(p): sha256(p) for p in tracked},
        'mask_diversity': mask_stats, 'c4_nonselection_development': estimates,
        'multi_turn_observation': 'Current choose() slices ids[:,:128]. Unchanged first128 tokens imply identical deterministic routes for the same checkpoint/K.',
        'limits': ['Reanalysis of previously used development data, not independent confirmation.',
                   'C4 shuffled routes are the original one-permutation control.',
                   'Mask diversity alone does not establish beneficial input-path matching.']})
    print(json.dumps({'output': str(args.output), 'c4': estimates, 'mask_groups': len(mask_stats)}), flush=True)


if __name__ == '__main__':
    main()
