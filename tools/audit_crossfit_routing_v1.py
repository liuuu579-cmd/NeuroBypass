"""Independent reconstruction of Stage25 splits, fitted choices and joint-cost metrics."""
import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load_choice
from full_context_experiment_v1 import write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def routes(scores, baseline, margin):
    best = np.argmin(scores, axis=1)
    result = np.full(len(scores), baseline, dtype=np.int64)
    for i, candidate in enumerate(best):
        if scores[i, baseline] - scores[i, candidate] > margin:
            result[i] = candidate
    return result


def gate(scores, costs, baseline, scale):
    choices = [(float(costs[:, baseline].mean()), 0, 0, {'static': True, 'margin_units': 0.})]
    for serial, margin in enumerate((0., .05, .1, .2, .4), 1):
        chosen = routes(scores, baseline, margin * scale)
        choices.append((float(np.mean([row[c] for row, c in zip(costs, chosen)])),
            int(np.count_nonzero(chosen != baseline)), serial, {'static': False, 'margin_units': margin}))
    return min(choices, key=lambda item: item[:3])[3]


def apply_gate(scores, rule, baseline, scale):
    return np.full(len(scores), baseline) if rule['static'] else routes(scores, baseline, rule['margin_units'] * scale)


def audit(root, out):
    cfg = load(out / 'protocol.json'); assert not cfg['probe']
    cpus = load(out / 'allocation.json')['cpu_ids']
    os.sched_setaffinity(0, cpus[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert len(cpus) == 42
    for name, digest in cfg['source_sha256'].items():
        assert sha256(root / 'tools' / name) == digest
    for name, digest in cfg['evidence_sha256'].items():
        assert sha256(name) == digest
    preserved = load(out / 'preservation_before.json')
    for name, digest in preserved.items():
        assert sha256(name) == digest, name
    old = Path(cfg['old_run']); previous = load(old / 'protocol.json')
    ids = cfg['train800_ids']; assert ids == previous['train_ids'] and len(ids) == 800
    assert not set(ids) & set(previous['validation_ids'])
    manifest = load(Path(cfg['data_run']) / 'data_manifest.json')['records']
    costs_by_id = {row['sample']: row for rank in range(6) for row in lines(old / f'private_labels_{rank:03d}.jsonl') if row['sample'] in ids}
    assert len(costs_by_id) == 800
    for i, row in costs_by_id.items():
        assert row['sequence_sha256'] == manifest[i]['sequence_sha256']
    bank = load(old / 'candidate_bank.json')
    assert bank['k'] == 20 and len(bank['routes']) == 8
    assert all(len(set(r['skips'])) == 20 for r in bank['routes'])
    assert len({tuple(r['skips']) for r in bank['routes']}) == 8
    lookup = {i: n for n, i in enumerate(ids)}
    costs = np.asarray([costs_by_id[i]['kl'] for i in ids], dtype=np.float64)
    nll = np.asarray([costs_by_id[i]['nll'] for i in ids], dtype=np.float64)
    features = torch.load(out / 'private_f0_features.pth', weights_only=True).numpy().astype(np.float64)
    assert features.shape == (800, 258) and np.isfinite(features).all()
    stored_choices = {(r['fold'], r['policy']): r for r in load(out / 'policy_choices.json')}
    expected = {}; all_outer = []; replay_checks = []; fitting_checks = []
    embeddings = torch.load(old / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    maximum_replay_error = 0.; feature_error = 0.
    projection = torch.randn(5120, 128, generator=torch.Generator().manual_seed(250579)) / np.sqrt(5120)
    for fold in cfg['folds']:
        f = fold['fold']; parts = [fold[key] for key in ('fit_ids', 'selection_ids', 'outer_ids')]
        assert list(map(len, parts)) == [520, 80, 200]
        assert set.union(*(set(v) for v in parts)) == set(ids)
        assert all(not set(parts[a]) & set(parts[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        for field in ('record_sha256', 'record_lead_sha256'):
            groups = [{manifest[i][field] for i in part} for part in parts]
            assert all(not groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        all_outer += parts[2]; rows = [[lookup[i] for i in part] for part in parts]
        joined = sum(rows, []); fit, selection, outer = rows
        target = costs[fit] - costs[fit, :1]
        baseline = int(np.argmin(np.mean(costs[fit], axis=0)))
        scale = max(float(np.std(target[:, 1:], ddof=1)), .03)
        predictions = {'fold_static': np.full(800, baseline), 'future_known_best': np.argmin(costs[joined], axis=1)}
        for seed in (579, 580, 581):
            name = f'f{f}_s{seed}'; receipt = load(out / f'{name}_fit.json')
            history = [row for row in load(out / f'{name}_history.json') if 'selection_kl' in row]
            best = min(history, key=lambda row: row['selection_kl'])
            assert best['epoch'] == receipt['best']['epoch']
            path = Path(receipt['best']['path']); assert sha256(path) == receipt['best']['sha256']
            saved = torch.load(path, weights_only=True, map_location='cpu')
            assert saved['epoch'] == best['epoch'] and saved['fold'] == f and saved['seed'] == seed
            prior32 = torch.tensor(costs[fit], dtype=torch.float32)
            prior32 = prior32 - prior32[:, :1]
            assert torch.equal(saved['state_dict']['prior'], prior32.mean(0))
            assert torch.equal(saved['state_dict']['scale'], prior32[:, 1:].std().clamp_min(.03))
            assert receipt['updates'] == 24 * math.ceil(520 / 16)
            scores_path = out / f'private_{name}_scores.pth'
            assert sha256(scores_path) == receipt['score_sha256']
            bundle = torch.load(scores_path, weights_only=True)
            assert bundle['ids'] == sum(parts, [])
            scores = bundle['scores'].numpy().astype(np.float64)
            predictions[f'neural_raw_s{seed}'] = np.argmin(scores, axis=1)
            selection_metric = costs[selection, np.argmin(scores[520:600], axis=1)].mean()
            assert abs(selection_metric - best['selection_kl']) < 1e-10
            rule = gate(scores[520:600], costs[selection], baseline, scale)
            entry = stored_choices[f, f'neural_gate_s{seed}']
            assert entry['baseline_index'] == baseline and abs(entry['scale'] - scale) < 1e-12
            assert all(entry['best'][key] == value for key, value in rule.items())
            predictions[f'neural_gate_s{seed}'] = apply_gate(scores, rule, baseline, scale)
            model = load_choice(path, 'cpu')
            with torch.inference_mode():
                for sample in parts[2][:2]:
                    assert embeddings[sample].shape[1] == manifest[sample]['prompt_length']
                    replay = model(embeddings[sample]).numpy()[0]
                    reference = scores[sum(parts, []).index(sample)]
                    error = float(np.max(np.abs(replay - reference))); maximum_replay_error = max(maximum_replay_error, error)
                    assert np.allclose(replay, reference, atol=3e-5, rtol=1e-4)
                    assert replay.argmin() == reference.argmin()
            replay_checks.append({'fold': f, 'seed': seed, 'documents': 2, 'selected_epoch': best['epoch']})
        for family in ('ridge_content', 'ridge_length', 'ridge_shuffled'):
            x = features if family != 'ridge_length' else features[:, -2:]
            center = x[fit].mean(0); spread = np.maximum(x[fit].std(0), 1e-6)
            train = (x[fit] - center) / spread
            y = target.copy()
            if family == 'ridge_shuffled':
                y = y[np.random.default_rng(250900 + f).permutation(520)]
            prior = y.mean(0); grid = []
            for penalty in (1., 10., 100., 1000.):
                # Cholesky reconstruction is independent of the training solve.
                chol = np.linalg.cholesky(train.T @ train + penalty * np.eye(train.shape[1]))
                weights = np.linalg.solve(chol.T, np.linalg.solve(chol, train.T @ (y - prior)))
                score = prior + ((x[joined] - center) / spread) @ weights
                rule = gate(score[520:600], costs[selection], baseline, scale)
                chosen = apply_gate(score, rule, baseline, scale)
                value = float(costs[selection, chosen[520:600]].mean())
                count = int(np.count_nonzero(chosen[520:600] != baseline))
                grid.append((value, count, penalty, rule, chosen))
            best = min(grid, key=lambda item: item[:3]); entry = stored_choices[f, family]
            assert entry['penalty'] == best[2]
            assert all(entry['best'][key] == value for key, value in best[3].items())
            predictions[family] = best[4]
        for policy, chosen in predictions.items():
            for position, sample in enumerate(sum(parts, [])):
                cohort = 'fit' if position < 520 else ('selection' if position < 600 else 'outer')
                route = int(chosen[position]); row = lookup[sample]
                expected[f, policy, cohort, sample] = {'route_index': route, 'baseline_index': baseline,
                    'kl': float(costs[row, route]), 'nll': float(nll[row, route]),
                    'static_kl': float(costs[row, baseline]), 'static_nll': float(nll[row, baseline])}
        for sample in parts[2][:2]:
            with torch.inference_mode():
                projected = embeddings[sample][0].float() @ projection
                value = torch.cat([projected.mean(0), projected.std(0, unbiased=False), torch.tensor([
                    np.log(len(projected)), np.log(len(projected)) ** 2], dtype=torch.float32)]).numpy()
            error = float(np.max(np.abs(value - features[lookup[sample]]))); feature_error = max(feature_error, error)
            assert np.allclose(value, features[lookup[sample]], atol=3e-5, rtol=1e-4)
        worker = load(out / f'worker_{f:03d}.json')
        assert worker['cpu_ids'] == cpus[f*10:(f+1)*10] and worker['threads'] == 10
        assert all(set(v) <= set(worker['cpu_ids']) for v in worker['thread_cpu_ids'].values())
        assert sha256(out / f'private_f{f}_features.pth') == worker['feature_sha256']
        fitting_checks.append({'fold': f, 'baseline_index': baseline, 'scale': scale})
    assert len(all_outer) == 800 and Counter(all_outer) == Counter(ids)
    actual = lines(out / 'private_crossfit_rows.jsonl')
    assert len(actual) == len(expected) == 35200
    for row in actual:
        replay = expected.pop((row['fold'], row['policy'], row['cohort'], row['sample']))
        assert row.items() >= replay.items()
    assert not expected
    summary = load(out / 'summary.json')
    assert summary['evaluated_outer_documents'] == 800 and not summary['engineering_only']
    by_key = {(row['fold'], row['policy'], row['cohort']): row for row in summary['per_fold']}
    groups = {}
    for row in actual:
        groups.setdefault((row['fold'], row['policy'], row['cohort']), []).append(row)
    for key, group in groups.items():
        result = by_key.pop(key); own = np.asarray([r['kl'] for r in group]); static = np.asarray([r['static_kl'] for r in group])
        assert abs(result['kl'] - own.mean()) < 1e-12
        assert abs(result['nll'] - np.mean([r['nll'] for r in group])) < 1e-12
        assert abs(result['difference'] - (static - own).mean()) < 1e-12
        assert result['beneficial'] == np.count_nonzero(static > own)
        assert result['harmful'] == np.count_nonzero(static < own)
        assert result['switches'] == sum(r['route_index'] != r['baseline_index'] for r in group)
    assert not by_key
    outer_rows = [r for r in actual if r['cohort'] == 'outer']
    for result in summary['out_of_fold800']:
        family = result['policy']; metric = result['metric']; differences = []; measured = []; static = []
        policies = [family] if not family.endswith('_mean3') else [family.replace('_mean3', f'_s{seed}') for seed in (579, 580, 581)]
        for sample in ids:
            group = [row for row in outer_rows if row['sample'] == sample and row['policy'] in policies]
            assert len(group) == len(policies)
            own = np.mean([row[metric] for row in group]); base = group[0]['static_' + metric]
            measured.append(own); static.append(base); differences.append(base - own)
        assert abs(result['own_mean'] - np.mean(measured)) < 1e-12
        assert abs(result['static_mean'] - np.mean(static)) < 1e-12
        assert abs(result['difference'] - np.mean(differences)) < 1e-12
        draws = np.random.default_rng(250579).integers(0, 800, size=(2000, 800))
        interval = np.quantile(np.asarray(differences)[draws].mean(1), [.025, .975])
        assert np.allclose(interval, result['ci95'], atol=1e-12, rtol=0)
    return {'passed': True, 'auditor': str(Path(__file__).relative_to(root)), 'auditor_sha256': sha256(__file__),
        'preserved_files': len(preserved), 'joint_cost_rows_reconstructed': len(actual), 'outer_documents_once_each': 800,
        'historical_dev100_excluded': True, 'record_and_lead_groups_disjoint': True,
        'priors_scales_static_and_ridge_fitted_only_on520': True, 'gates_epochs_and_penalties_selected_only_on80': True,
        'checkpoint_replays': replay_checks, 'maximum_cpu_gpu_score_error': maximum_replay_error,
        'maximum_cpu_gpu_full_prompt_feature_error': feature_error, 'fold_fit_checks': fitting_checks,
        'all_summary_metrics_and_bootstrap_reconstructed': True, 'resource_threads_verified': '4GPU/42CPU',
        'candidate_bank_caveat': cfg['limits']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(root, args.run.resolve()); write(args.run / 'independent_audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
