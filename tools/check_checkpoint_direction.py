"""CPU-only semantic check using existing checkpoints and read-only v3 data.

No OPT decoder blocks are loaded or executed. Embedding tensors are read from
the existing PyTorch shard with mmap; results are diagnostic, not held-out scores.
"""
import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr


def load_class(path):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'DynamicConditionalRouterTransformer')
    scope = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[node.name]


def choose_skips(logits, k, largest=False):
    scores = logits.clone()
    scores[..., :2] = -torch.inf if largest else torch.inf
    return scores.topk(k, largest=largest).indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cls = load_class(args.snapshot / 'legacy/skip/model_router_transformer.py')
    router = cls(dropout=0.).eval()
    state = torch.load(args.source / 'skip/transformer_router_raw_embed_epoch_20.pth',
                       weights_only=True, mmap=True, map_location='cpu')
    router.load_state_dict(state)
    weight_map = json.loads((args.base_model / 'pytorch_model.bin.index.json').read_text())['weight_map']
    embedding_shards = {}
    weights = {}
    for name in ['decoder.embed_tokens.weight', 'decoder.embed_positions.weight']:
        filename = weight_map[name]
        if filename not in embedding_shards:
            embedding_shards[filename] = torch.load(args.base_model / filename,
                                                  weights_only=True, mmap=True, map_location='cpu')
        weights[name] = embedding_shards[filename][name]
    samples, groups = [], set()
    for path in sorted(args.source.glob('opt13b_greedy_routing_labels_v3_part_*')):
        g = zarr.open_group(str(path), mode='r')
        for i in range(len(g['seq_lens'])):
            if not g['input_ids'].attrs.get(f'row_{i}_written', False):
                continue
            ids = g['input_ids'][i][:int(g['seq_lens'][i])]
            group = hashlib.sha256(ids[:128].tobytes()).hexdigest()
            if group not in groups:
                groups.add(group)
                samples.append((path.name, i, torch.tensor(ids, dtype=torch.long),
                                g['ordered_skips'][i].tolist()))
    results = {}
    for k in [6, 16]:
        good, bad, logged_bad, bces, output_masks = [], [], [], [], []
        with torch.inference_mode():
            for shard, row, ids, order in samples:
                pos = torch.arange(2, len(ids) + 2)
                embeddings = (F.embedding(ids, weights['decoder.embed_tokens.weight']) +
                              F.embedding(pos, weights['decoder.embed_positions.weight'])).unsqueeze(0)
                logits = router(embeddings, torch.tensor([k]))[0]
                target = torch.ones(40)
                target[order[:k]] = 0.
                correct = set(choose_skips(logits, k).tolist())
                reversed_set = set(choose_skips(logits, k, largest=True).tolist())
                expected = set(order[:k])
                good.append(len(correct & expected) / k)
                bad.append(len(reversed_set & expected) / k)
                logged_bad.append(len(set(logits.topk(k).indices.tolist()) &
                                      set(target.topk(k).indices.tolist())) / k)
                bces.append(F.binary_cross_entropy_with_logits(logits, target).item())
                output_masks.append(tuple(sorted(correct)))
        counts = Counter(output_masks)
        results[str(k)] = {'sample_count': len(samples),
                           'correct_bottom_k_label_overlap': sum(good) / len(good),
                           'legacy_top_k_label_overlap_with_actual_skip_targets': sum(bad) / len(bad),
                           'legacy_logged_hit_metric': sum(logged_bad) / len(logged_bad),
                           'mean_bce': sum(bces) / len(bces),
                           'unique_corrected_masks': len(counts),
                           'most_common_corrected_mask': list(counts.most_common(1)[0][0]),
                           'most_common_mask_fraction': counts.most_common(1)[0][1] / len(samples)}
    # Even an ideal classifier is decoded backwards by the latest evaluator.
    ideal_target = torch.ones(40)
    ideal_target[[2, 5, 7, 9, 11, 14]] = 0.
    ideal_logits = (ideal_target * 2 - 1) * 12
    ideal_skips = set(torch.where(ideal_target == 0)[0].tolist())
    ideal_test = {'bottom_k_correct_hits': len(set(choose_skips(ideal_logits, 6).tolist()) & ideal_skips),
                  'legacy_top_k_correct_hits': len(set(choose_skips(ideal_logits, 6, True).tolist()) & ideal_skips)}
    assert ideal_test == {'bottom_k_correct_hits': 6, 'legacy_top_k_correct_hits': 0}
    result = {'checkpoint': 'skip/transformer_router_raw_embed_epoch_20.pth',
              'scope': 'One completed row per distinct 128-token prefix group; these are training-source diagnostics.',
              'ideal_classifier_semantic_test': ideal_test, 'by_k': results,
              'limitations': ['No full OPT forward pass, downstream benchmark, or generalization claim.',
                              'Labels are greedy targets, not global optima.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
