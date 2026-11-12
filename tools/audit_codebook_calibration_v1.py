"""Independent Stage21 grid replay; never uses other68 to choose policies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def digest(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); out = args.run.resolve(); root = Path(__file__).resolve().parents[1]
    cfg, summary, fits = [read(out / name) for name in ('protocol.json', 'summary.json', 'policy_fits.json')]
    assert read(out / 'verification.json')['passed']
    for name, expected in cfg['source_sha256'].items():
        assert digest(root / 'tools' / name) == expected
    for path, expected in cfg['evidence_sha256'].items():
        assert digest(path) == expected
    for checkpoint in cfg['checkpoints']:
        assert digest(checkpoint['path']) == checkpoint['sha256']
    previous = Path(cfg['previous_run']); original = read(previous / 'protocol.json')
    for field in ('train_ids', 'validation_ids', 'selection_ids'):
        assert cfg[field] == original[field]
    assert len(cfg['train_ids']) == 800 and len(cfg['validation_ids']) == 100 and len(cfg['selection_ids']) == 32
    assert not set(cfg['train_ids']) & set(cfg['validation_ids'])
    index = {i: j for j, i in enumerate(cfg['ids'])}; assert len(index) == 900
    train = np.array([index[i] for i in cfg['train_ids']]); select = np.array([index[i] for i in cfg['selection_ids']])
    labels = {r['sample']: r for rank in range(6) for r in lines(previous / f'private_labels_{rank:03d}.jsonl')}
    costs = np.array([labels[i]['kl'] for i in cfg['ids']]); nll = np.array([labels[i]['nll'] for i in cfg['ids']])
    baseline = int(costs[train].mean(0).argmin()); assert baseline == summary['baseline_index'] == 1
    scale = float((costs[train] - costs[train, baseline, None]).std())
    assert scale == summary['scale_from_train800']
    predictions = {}
    cpu_plan = read(out / 'allocation.json')['cpu_ids']; assert len(cpu_plan) == 64
    for rank in range(6):
        p = out / f'private_predictions_{rank:03d}.pth'; receipt = read(out / f'prediction_worker_{rank:03d}.json')
        assert receipt['sha256'] == digest(p) and receipt['full_prompt_lengths_verified']
        assert receipt['cpu_ids'] == cpu_plan[rank * 10:(rank + 1) * 10]
        assert all(set(a) <= set(receipt['cpu_ids']) for a in receipt['thread_cpu_ids'].values())
        predictions.update({k: v.numpy().astype(np.float64) for k, v in torch.load(p, weights_only=True).items()})
    assert len(predictions) == 15
    for family in ('old256_prompt', 'huber_prompt', 'rank_prompt'):
        predictions[f'ensemble_{family}'] = np.mean([predictions[f'{family}_s{s}'] for s in (579, 580, 581)], axis=0)
    predictions['ensemble_all9_prompt'] = np.mean([predictions[f'{f}_s{s}']
        for f in ('old256_prompt', 'huber_prompt', 'rank_prompt') for s in (579, 580, 581)], axis=0)
    assert set(fits) == set(predictions) and len(fits) == 19
    all_routes = {}; grid_count = 0
    for name, predicted in predictions.items():
        mean_p = predicted[train].mean(0); mean_y = costs[train].mean(0)
        fitted = fits[name]
        assert np.array_equal(mean_p, fitted['train_prediction_mean'])
        assert np.array_equal(mean_y, fitted['train_cost_mean'])
        assert fitted['scale'] == scale and fitted['baseline'] == baseline
        assert len(fitted['grid']) == 37
        independently_chosen = []
        for serial, candidate in enumerate(fitted['grid']):
            if candidate['family'] == 'static':
                decision = np.full(900, baseline)
            else:
                score = predicted if candidate['family'] == 'raw_gate' else mean_y + candidate['alpha'] * (predicted - mean_p)
                decision = []
                for row in score:
                    best = int(np.argmin(row))
                    decision.append(best if row[baseline] - row[best] > candidate['margin_units'] * scale else baseline)
                decision = np.array(decision)
            quality = float(costs[select, decision[select]].mean())
            switches = int(sum(decision[select] != baseline))
            assert quality == candidate['selection_kl'] and switches == candidate['selection_switches']
            independently_chosen.append((quality, switches, serial, decision))
            grid_count += 1
        best = min(independently_chosen, key=lambda r: r[:3])
        assert fitted['chosen'] == fitted['grid'][best[2]]
        all_routes[name] = best[3]
    singles = [n for n in predictions if 'prompt' in n and not n.startswith('ensemble_')]
    selected = min(singles, key=lambda n: (fits[n]['chosen']['selection_kl'], fits[n]['chosen']['selection_switches'], n))
    policy = read(out / 'selected_policy.json')
    assert selected == summary['selected_model'] == policy['model'] and policy['members'] == [selected]
    assert policy['fit'] == fits[selected]
    groups = {'train800': train, 'selection32': select,
        'other68': np.array([index[i] for i in cfg['validation_ids'] if i not in cfg['selection_ids']]),
        'all_validation100': np.array([index[i] for i in cfg['validation_ids']])}
    for row in summary['per_model']:
        ix = groups[row['cohort']]
        routes = predictions[row['model']].argmin(1) if row['variant'] == 'raw' else all_routes[row['model']]
        own = costs[ix, routes[ix]]
        assert abs(float(own.mean()) - row['kl']) < 1e-12
        assert abs(float(nll[ix, routes[ix]].mean()) - row['nll']) < 1e-12
        assert abs(float((costs[ix, baseline] - own).mean()) - row['difference']) < 1e-12
    max_kl, max_nll, raw_rows = 0., 0., 0
    for rank in range(6):
        for row in lines(previous / f'private_quality_{rank:03d}.jsonl'):
            if row['seed'] is None:
                continue
            name = f"{row['policy']}_s{row['seed']}"; ix = index[row['source_sample']]
            route = int(predictions[name][ix].argmin()); assert route == row['route_index']
            max_kl = max(max_kl, abs(row['kl'] - costs[ix, route]))
            max_nll = max(max_nll, abs(row['nll'] - nll[ix, route])); raw_rows += 1
    assert raw_rows == 1500 and max_kl < 1e-6 and max_nll < 1e-5
    replay = [r for rank in range(6) for r in lines(out / f'private_replay_{rank:03d}.jsonl')]
    assert len(replay) == len({r['sample'] for r in replay}) == 100
    assert {r['sample'] for r in replay} == set(cfg['validation_ids'])
    actual_kl, actual_nll = 0., 0.
    for row in replay:
        ix = index[row['sample']]; route = all_routes[selected][ix]; assert row['route'] == route
        actual_kl = max(actual_kl, abs(row['kl'] - costs[ix, route]))
        actual_nll = max(actual_nll, abs(row['nll'] - nll[ix, route]))
    assert actual_kl < 1e-6 and actual_nll < 1e-5
    preservation = read(out / 'preservation_before.json')
    for path, expected in preservation.items():
        assert digest(path) == expected
    receipt = {'passed': True, 'source_sha256': digest(__file__), 'models': len(fits),
        'independently_recomputed_grid_points': grid_count, 'original32_only_selection_reproduced': True,
        'train800_validation100_disjoint': True, 'raw_route_rows_exact': raw_rows,
        'raw_kl_max_error': max_kl, 'raw_nll_max_error': max_nll,
        'actual_selected_route_documents': 100, 'actual_kl_max_error': actual_kl, 'actual_nll_max_error': actual_nll,
        'old_files_preserved': len(preservation), 'checkpoints_and_evidence_unchanged': True,
        'six_gpu_64_cpu_worker_thread_affinity_verified': True}
    with (out / 'independent_audit.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
