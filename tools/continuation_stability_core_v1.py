"""Prompt-fixed complete-mask preference stability across four continuation windows."""
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from full_context_data_v1 import Records
from full_context_experiment_v1 import read_lines, write
from headroom_ruler_core_v1 import paired


SEED = 270579
WINDOW = 32
WINDOWS = 4
TOLERANCE = .01


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<i4').tobytes()).hexdigest()


def prepare(out, previous):
    from transformers import AutoTokenizer
    data = Records(previous['data_run']); manifest = data.manifest
    tokenizer = AutoTokenizer.from_pretrained(previous['model'], use_fast=False, local_files_only=True)
    eligible = {}; excluded = Counter()
    with Path(manifest['corpus']).open('rb') as stream:
        for sample in previous['train_ids']:
            row = data.rows[sample]; p = row['prompt_length']
            if p + 128 > 2048:
                excluded['context_limit'] += 1; continue
            stream.seek(row['source_start_byte']); raw = stream.read(row['source_end_byte'] - row['source_start_byte'])
            assert hashlib.sha256(raw).hexdigest() == row['source_raw_sha256']
            text = raw.decode('utf8').strip()
            assert hashlib.sha256(' '.join(text.split()).encode()).hexdigest() == row['record_sha256']
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2047)
            old, _ = data[sample]
            assert np.array_equal(np.asarray(tokens[:p+32]), old)
            if len(tokens) < p + 128:
                excluded['insufficient_same_document_continuation'] += 1; continue
            eligible[sample] = tokens[:p+128]
    order = list(eligible); np.random.default_rng(SEED).shuffle(order)
    selected = []; seen_record = set(); seen_lead = set()
    for sample in order:
        row = data.rows[sample]
        if row['record_sha256'] in seen_record or row['record_lead_sha256'] in seen_lead:
            continue
        selected.append(sample); seen_record.add(row['record_sha256']); seen_lead.add(row['record_lead_sha256'])
        if len(selected) == 128:
            break
    assert len(selected) == 128
    path = out / 'private_inputs.jsonl'
    with path.open('x') as target:
        for sample in selected:
            p = data.rows[sample]['prompt_length']; tokens = eligible[sample]
            target.write(json.dumps({'sample': sample, 'prompt_length': p, 'ids': tokens,
                'original_sequence_sha256': data.rows[sample]['sequence_sha256'], 'prompt_sha256': digest(tokens[:p]),
                'extended_sequence_sha256': digest(tokens)}, allow_nan=False) + '\n')
    result = {'training_documents_considered': 800, 'eligible_documents': len(eligible), 'selected_documents': 128,
        'selection_seed': SEED, 'sampling': 'Uniform deterministic permutation of eligible original TRAIN IDs, first128 unique record/lead groups; no KL used.',
        'excluded': dict(excluded), 'selected_ids': selected,
        'prompt_length_histogram': dict(Counter(data.rows[i]['prompt_length'] for i in selected)),
        'minimum_prompt_tokens': min(data.rows[i]['prompt_length'] for i in selected),
        'maximum_prompt_tokens': max(data.rows[i]['prompt_length'] for i in selected),
        'selection_bias': 'Conditional on at least128 same-document continuation tokens with original prompt unchanged; excludes many short documents and all2016-token prompts.'}
    write(out / 'sampling.json', result)
    return path, result


def token_metrics(logits, teacher_log_probs, targets):
    log_probs = torch.log_softmax(logits.float(), -1)
    assert log_probs.shape == teacher_log_probs.shape and log_probs.shape[:2] == targets.shape
    kl = (teacher_log_probs.exp() * (teacher_log_probs - log_probs)).sum(-1)
    nll = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
    assert torch.isfinite(kl).all() and torch.isfinite(nll).all()
    return kl[0].cpu(), nll[0].cpu()


def estimate(values):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if not len(a):
        return {'mean': None, 'ci95': None, 'documents': 0}
    result = paired(a, None, SEED)
    return {'mean': result['difference'], 'ci95': result['ci95'], 'documents': len(a)}


def row_correlation(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    divisor = np.sqrt((a*a).sum(-1) * (b*b).sum(-1))
    return np.divide((a*b).sum(-1), divisor, out=np.full(divisor.shape, np.nan), where=divisor > 0)


def stability(costs, static_index=1):
    costs = np.asarray(costs, dtype=np.float64); assert costs.ndim == 3 and costs.shape[1:] == (4, 8)
    n, _, c = costs.shape; best = costs.argmin(-1); ranks = rankdata(costs, axis=-1)
    minimum = costs.min(-1); pairs = []
    route_pairs = list(combinations(range(c), 2))
    for a, b in combinations(range(4), 2):
        near_a = costs[:, a] <= minimum[:, a, None] + TOLERANCE
        near_b = costs[:, b] <= minimum[:, b, None] + TOLERANCE
        near_overlap = (near_a & near_b).sum(1) / (near_a | near_b).sum(1)
        observed = float(np.mean(best[:, a] == best[:, b]))
        marginal_a = np.bincount(best[:, a], minlength=c) / n; marginal_b = np.bincount(best[:, b], minlength=c) / n
        expected = float(marginal_a @ marginal_b)
        gap_a = np.stack([costs[:, a, i] - costs[:, a, j] for i, j in route_pairs], -1)
        gap_b = np.stack([costs[:, b, i] - costs[:, b, j] for i, j in route_pairs], -1)
        meaningful = (np.abs(gap_a) >= TOLERANCE) & (np.abs(gap_b) >= TOLERANCE)
        matches = (np.sign(gap_a) == np.sign(gap_b)) & meaningful
        doc_agreement = np.divide(matches.sum(1), meaningful.sum(1), out=np.full(n, np.nan), where=meaningful.sum(1) > 0)
        # Center each route across documents, removing fixed mean-mask preference.
        delta_a = costs[:, a] - costs[:, a, static_index, None]
        delta_b = costs[:, b] - costs[:, b, static_index, None]
        delta_a -= delta_a.mean(0); delta_b -= delta_b.mean(0)
        denominator = float(np.sqrt((delta_a**2).sum() * (delta_b**2).sum()))
        pairs.append({'windows': [a, b], 'winner_agreement': observed, 'chance_agreement': expected,
            'cohen_kappa': (observed - expected) / (1 - expected) if expected < 1 else None,
            'within_document_route_spearman': estimate(row_correlation(ranks[:, a], ranks[:, b])),
            'near_optimal_set_jaccard_tau001': estimate(near_overlap),
            'material_pairwise_order_agreement_tau001': estimate(doc_agreement),
            'material_route_pairs': int(meaningful.sum()),
            'document_centered_relative_gap_pearson': float((delta_a * delta_b).sum() / denominator) if denominator > 0 else None,
            'first_winner_regret_on_second': estimate(costs[np.arange(n), b, best[:, a]] - minimum[:, b]),
            'first_winner_within_tau001_on_second': float(np.mean(costs[np.arange(n), b, best[:, a]] <= minimum[:, b] + TOLERANCE))})
    margins = np.sort(costs, axis=-1)[:, :, 1] - minimum
    return {'window_pairs': pairs, 'winner_histograms': [dict(Counter(best[:, w].tolist())) for w in range(4)],
        'best_second_gap_quantiles': np.quantile(margins, [.25, .5, .75]).tolist(),
        'fraction_winner_margin_below_tau001': float(np.mean(margins < TOLERANCE)),
        'tolerance_nats_per_token': TOLERANCE}


def frozen_routes(root, ids):
    requested = set(ids); result = {name: {} for name in ('embedding_gate', 'hidden2_gate')}
    evidence = {'embedding_gate': root / 'runs/stage25_crossfit_20261113_02_main/private_crossfit_rows.jsonl',
        'hidden2_gate': root / 'runs/stage26_semantic_prefix_20261113_01_main/private_quality_rows.jsonl'}
    for name, path in evidence.items():
        for row in read_lines(path):
            stem = 'neural_gate_s' if name == 'embedding_gate' else 'hidden2_gate_s'
            if row['sample'] in requested and row['cohort'] == 'outer' and row['policy'].startswith(stem):
                seed = int(row['policy'].removeprefix(stem)); sample = row['sample']
                assert seed not in result[name].setdefault(sample, {})
                result[name][sample][seed] = row['route_index']
        assert set(result[name]) == requested
        assert all(set(value) == {579, 580, 581} for value in result[name].values())
    return result


def summarize(root, ids, costs, nll):
    n = len(ids); base = costs[:, :, 1]; base_nll = nll[:, :, 1]
    first_best = costs[:, 0].argmin(1); whole_best = costs.mean(1).argmin(1)
    routes = frozen_routes(root, ids)
    policies = {'strong_static': np.ones((n, 4, 1), dtype=int),
        'first32_future_oracle': np.broadcast_to(first_best[:, None, None], (n, 4, 1)),
        'full128_future_oracle': np.broadcast_to(whole_best[:, None, None], (n, 4, 1))}
    leave_out = np.stack([(costs.sum(1) - costs[:, w]).argmin(1) for w in range(4)], 1)
    policies['other96_future_oracle'] = leave_out[:, :, None]
    for name in routes:
        selected = np.asarray([[routes[name][sample][seed] for seed in (579, 580, 581)] for sample in ids])
        policies[name + '_mean3'] = np.broadcast_to(selected[:, None, :], (n, 4, 3))
    rows = []
    for name, paths in policies.items():
        own_kl = np.take_along_axis(costs, paths, axis=2).mean(2)
        own_nll = np.take_along_axis(nll, paths, axis=2).mean(2)
        for window in (0, 1, 2, 3, 'later96', 'all128'):
            columns = [window] if isinstance(window, int) else ([1, 2, 3] if window == 'later96' else [0, 1, 2, 3])
            for metric, own, reference in (('kl', own_kl, base), ('nll', own_nll, base_nll)):
                value = own[:, columns].mean(1); static = reference[:, columns].mean(1)
                rows.append({'policy': name, 'window': window, 'metric': metric, 'mean': float(value.mean()),
                    'static_mean': float(static.mean()), 'improvement': estimate(static - value)})
    per_route = [{'window': w, 'route': route, 'kl': float(costs[:, w, route].mean()),
        'nll': float(nll[:, w, route].mean())} for w in range(4) for route in range(8)]
    return {'documents': n, 'stability': stability(costs), 'policy_transfer': rows, 'per_route_window_means': per_route,
        'policy_limits': 'Frozen embedding/hidden2 decisions precede this experiment and use original full prompts only; mean3 averages independent-seed scores. All oracle policies read actual future and are diagnostics only. other96 changes the diagnostic decision for each evaluated window and is not a deployment policy.',
        'statistical_limits': 'Reviewed training-document diagnostic with historically selected bank; eligibility-conditioned sample. Windows are ordered contiguous text with growing teacher-forced context, not IID continuations. Bootstrap resamples documents only, ignores candidate selection and router training uncertainty; no multiple-comparison correction.'}


def cpu_checks():
    costs = np.broadcast_to(np.arange(8, dtype=float)[None, None, :], (3, 4, 8)).copy()
    stable = stability(costs)
    assert all(row['winner_agreement'] == 1 for row in stable['window_pairs'])
    assert all(abs(row['within_document_route_spearman']['mean'] - 1) < 1e-12 for row in stable['window_pairs'])
    costs[:, 1] = costs[:, 1, ::-1]
    changed = stability(costs)['window_pairs'][0]
    assert changed['winner_agreement'] == 0 and abs(changed['within_document_route_spearman']['mean'] + 1) < 1e-12
    assert changed['first_winner_regret_on_second']['mean'] == 7
    logits = torch.tensor([[[2., 0.], [0., 2.]]]); teacher = logits.log_softmax(-1)
    kl, nll = token_metrics(logits, teacher, torch.tensor([[0, 1]]))
    assert torch.equal(kl, torch.zeros(2)) and torch.allclose(nll, -teacher[0, :, 0].new_tensor([teacher[0, 0, 0], teacher[0, 1, 1]]))
    return {'passed': True, 'identical_window_ranks': True, 'reversed_preferences_detected': True,
        'teacher_kl_zero_and_target_nll_alignment': True}
