"""Read-only audit of v3 greedy trajectories; writes only aggregate statistics."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import zarr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    shards, ids_all, lengths_all, skips_all, kls_all, done_all = [], [], [], [], [], []
    for path in sorted(args.source.glob('opt13b_greedy_routing_labels_v3_part_*')):
        g = zarr.open_group(str(path), mode='r')
        ids, lengths = g['input_ids'][:], g['seq_lens'][:]
        skips, kls = g['ordered_skips'][:], g['kl_divergences'][:]
        done = np.array([bool(g['input_ids'].attrs.get(f'row_{i}_written', False))
                         for i in range(len(lengths))])
        shards.append({'name': path.name, 'allocated_rows': len(lengths),
                       'completed_rows': int(done.sum())})
        ids_all.append(ids); lengths_all.append(lengths)
        skips_all.append(skips); kls_all.append(kls); done_all.append(done)
    if not shards:
        raise FileNotFoundError('No v3 shards found')
    ids, lengths = np.concatenate(ids_all), np.concatenate(lengths_all)
    skips, kls, done = np.concatenate(skips_all), np.concatenate(kls_all), np.concatenate(done_all)
    invalid_ids = (ids < 0).any(axis=1)
    invalid_lengths = (lengths < 128) | (lengths > ids.shape[1])
    invalid_skips = np.array([len(set(row.tolist())) != len(row) or
                              np.any(row < 2) or np.any(row >= 40) for row in skips])
    valid = done & ~invalid_lengths & ~invalid_skips & ~invalid_ids & np.isfinite(kls).all(axis=1)
    group_ids = [hashlib.sha256(row[:128].tobytes()).hexdigest() for row in ids]
    exact_keys = [(group_ids[i], int(lengths[i])) for i in range(len(ids))]
    sizes = Counter(group_ids)
    # Reproduce the random_split seed and permutation used by both SFT trainers.
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(ids), generator=generator).tolist()
    n_train = len(ids) - int(0.1 * len(ids))
    train, val = perm[:n_train], perm[n_train:]
    train_groups = {group_ids[i] for i in train}
    val_overlap = sum(group_ids[i] in train_groups for i in val)
    ks = [1, 2, 4, 6, 8, 12, 16]
    labels = {}
    for k in ks:
        sets = [tuple(sorted(row[:k].tolist())) for row in skips[valid]]
        counts = Counter(sets)
        freq = np.bincount(skips[valid, :k].ravel(), minlength=40)
        static = set(np.argsort(-freq, kind='stable')[:k].tolist())
        labels[str(k)] = {
            'unique_masks': len(counts), 'most_common_mask': list(counts.most_common(1)[0][0]),
            'most_common_mask_fraction': counts.most_common(1)[0][1] / len(sets),
            'frequency_static_mask': sorted(static),
            'frequency_static_label_overlap': float(np.mean([len(static.intersection(s)) / k for s in sets])),
            'mean_greedy_kl': float(kls[valid, k-1].mean()),
            'median_greedy_kl': float(np.median(kls[valid, k-1])),
            'p90_greedy_kl': float(np.quantile(kls[valid, k-1], .9)),
        }
    threshold = .5
    max_valid_k = [max(np.where(row <= threshold)[0] + 1, default=0) for row in kls[valid]]
    data = {
        'scope': 'Aggregate audit of stored data, not a new pruning quality benchmark.',
        'shards': shards, 'allocated_rows': len(ids), 'valid_complete_rows': int(valid.sum()),
        'invalid_lengths': int(invalid_lengths.sum()), 'invalid_skip_trajectories': int(invalid_skips.sum()),
        'nonfinite_kl_rows': int((~np.isfinite(kls).all(axis=1)).sum()),
        'negative_kl_values': int((kls < 0).sum()), 'minimum_stored_kl': float(kls.min()),
        'unique_128_token_prefix_groups': len(sizes),
        'prefix_group_sizes': sorted(sizes.values()),
        'duplicate_same_prefix_same_length_rows': len(ids) - len(set(exact_keys)),
        'random_split_seed42': {'train_rows': len(train), 'val_rows': len(val),
                                'val_rows_with_train_prefix_group': val_overlap,
                                'overlap_fraction': val_overlap / max(1, len(val)),
                                'train_groups': len(train_groups),
                                'val_groups': len({group_ids[i] for i in val})},
        'rows_with_decreasing_kl_step': int((np.diff(kls[valid], axis=1) < -1e-6).any(axis=1).sum()),
        'max_k_with_kl_at_most_0_5_histogram': dict(sorted(Counter(int(k) for k in max_valid_k).items())),
        'labels_by_k': labels,
        'limitations': ['Prefix groups identify overlapping source chunks, not independently identified documents.',
                        'Label overlap is not task accuracy or KL quality.',
                        'Stored KL is from FP16 last-token distributions.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
