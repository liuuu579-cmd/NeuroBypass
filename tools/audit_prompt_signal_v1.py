"""Independent audit of complete-prompt cost alignment and selection-only policy choice."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).open()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    out = parser.parse_args().run.resolve(); root = Path(__file__).resolve().parents[1]
    cfg = read(out / 'protocol.json'); old = Path(cfg['old_run']); source_cfg = read(old / 'protocol.json')
    assert read(out / 'verification.json')['passed']
    for name, expected in cfg['source_sha256'].items():
        assert digest(root / 'tools' / name) == expected
    for path, expected in cfg['evidence_sha256'].items():
        assert digest(path) == expected
    for field in ('train800_ids', 'selection32_ids', 'validation100_ids'):
        key = {'train800_ids': 'train_ids', 'selection32_ids': 'selection_ids', 'validation100_ids': 'validation_ids'}[field]
        assert cfg[field] == source_cfg[key]
    expected64 = np.random.default_rng(230579).permutation(cfg['train800_ids'])[:64].tolist()
    assert expected64 == cfg['training64_ids']
    assert set(cfg['training64_ids']) <= set(cfg['train800_ids'])
    assert not set(cfg['train800_ids']) & set(cfg['validation100_ids'])
    old_labels = {r['sample']: r for rank in range(6) for r in lines(old / f'private_labels_{rank:03d}.jsonl')}
    manifests = read(Path(cfg['data_run']) / 'data_manifest.json')['records']
    plan = read(out / 'allocation.json'); assert plan['cpu_count'] == 21 and len(plan['gpu_indices']) == 2
    all_rows = []; projection_errors = []; future_replay_errors = []
    for rank in range(2):
        path = out / f'private_prompt_costs_{rank:03d}.jsonl'; receipt = read(out / f'worker_{rank:03d}.json')
        rows = lines(path); assert digest(path) == receipt['labels_sha256']
        assert [r['sample'] for r in rows] == cfg['ids'][rank::2]
        assert receipt['documents'] == len(rows) and receipt['base_model_frozen']
        assert receipt['cpu_ids'] == plan['cpu_ids'][rank * 10:(rank + 1) * 10]
        assert receipt['threads'] == 10 and receipt['no_future_tokens_in_prompt_scoring']
        assert all(set(v) <= set(receipt['cpu_ids']) for v in receipt['thread_cpu_ids'].values())
        for check in receipt['checks']:
            assert check['dense_vs_itself_prompt_kl_zero']
            projection_errors.append(check['prompt_chunk_replay_error'])
            future_replay_errors.append(check['continuation_kl_replay_error'])
        for row in rows:
            manifest = manifests[row['sample']]
            assert row['sequence_sha256'] == old_labels[row['sample']]['sequence_sha256'] == manifest['sequence_sha256']
            assert row['prompt_length'] == manifest['prompt_length']
            assert row['scored_tokens'] == row['prompt_length'] - 1
            assert len(row['prompt_kl']) == len(row['prompt_nll']) == 8
            assert np.isfinite(row['prompt_kl']).all() and min(row['prompt_kl']) >= -1e-5
        all_rows += rows
    assert len(all_rows) == len({r['sample'] for r in all_rows}) == 164
    assert max(projection_errors) < 2e-4 and max(future_replay_errors) < 1e-6
    by_id = {r['sample']: r for r in all_rows}; ids = cfg['ids']; ix = {s: i for i, s in enumerate(ids)}
    prompt = np.array([by_id[s]['prompt_kl'] for s in ids])
    future = np.array([old_labels[s]['kl'] for s in ids]); nll = np.array([old_labels[s]['nll'] for s in ids])
    training800 = np.array([old_labels[s]['kl'] for s in cfg['train800_ids']])
    baseline = int(training800.mean(0).argmin()); assert baseline == cfg['baseline'] == 1
    train = np.array([ix[i] for i in cfg['training64_ids']]); selection = np.array([ix[i] for i in cfg['selection32_ids']])
    prior = (training800 - training800[:, baseline:baseline + 1]).mean(0)
    prompt_mean = (prompt[train] - prompt[train, baseline, None]).mean(0)
    scale = (training800 - training800[:, baseline:baseline + 1]).std()
    fit = read(out / 'policy_fits.json')
    assert np.array_equal(prior, fit['prior']) and np.array_equal(prompt_mean, fit['train_prompt_relative_mean'])
    assert float(scale) == fit['scale'] and len(fit['grid']) == 16
    def decisions(score, margin):
        chosen = []
        for row in score:
            best = int(np.argmin(row))
            chosen.append(best if row[baseline] - row[best] > margin else baseline)
        return np.array(chosen)
    center = prompt - prompt[:, baseline:baseline + 1] - prompt_mean
    recomputed = []
    for serial, (a, m) in enumerate((a, m) for a in (0., .25, .5, 1.) for m in (0., .1, .2, .4)):
        routes = decisions(prior + a * center, m * scale)
        value = float(future[selection, routes[selection]].mean()); count = int((routes[selection] != baseline).sum())
        entry = {'alpha': a, 'margin_units': m, 'selection_kl': value, 'selection_switches': count, 'serial': serial}
        assert fit['grid'][serial] == entry
        recomputed.append(entry)
    chosen = min(recomputed, key=lambda r: (r['selection_kl'], r['selection_switches'], r['serial']))
    assert chosen == fit['chosen']
    selected = decisions(prior + chosen['alpha'] * center, chosen['margin_units'] * scale)
    direct = decisions(prompt, 0.)
    cohorts = {'train64_diagnostic': train, 'selection32': selection,
        'other68': np.array([ix[i] for i in cfg['validation100_ids'] if i not in cfg['selection32_ids']]),
        'all_validation100': np.array([ix[i] for i in cfg['validation100_ids']])}
    summary = read(out / 'summary.json')
    assert summary['scored_prompt_tokens'] == sum(r['scored_tokens'] for r in all_rows)
    assert summary['max_prompt_length'] == max(r['prompt_length'] for r in all_rows)
    for row in summary['results']:
        group = cohorts[row['cohort']]
        routes = {'train_aligned_prompt_gate': selected, 'exact_prompt_argmin': direct,
            'strong_static': np.full(len(ids), baseline)}[row['policy']]
        own = future[group, routes[group]]; delta = future[group, baseline] - own
        assert abs(row['kl'] - float(own.mean())) < 1e-12
        assert abs(row['nll'] - float(nll[group, routes[group]].mean())) < 1e-12
        assert abs(row['difference'] - float(delta.mean())) < 1e-12
        assert row['beneficial_switches'] == int((delta > 0).sum())
        assert row['harmful_switches'] == int((delta < 0).sum())
    for path, expected in read(out / 'preservation_before.json').items():
        assert digest(path) == expected
    receipt = {'passed': True, 'source_sha256': digest(__file__), 'documents': 164,
        'full_prompt_scoring_lengths_match_source_manifest': True,
        'training64_selection_identity_reproduced_without_costs': True,
        'sixteen_policy_grid_and_original32_choice_independently_reproduced': True,
        'all_continuation_aggregates_reproduced': True, 'training_validation_disjoint': True,
        'max_prompt_projection_chunk_error': max(projection_errors),
        'max_existing_continuation_kl_replay_error': max(future_replay_errors),
        'two_gpu_21_cpu_thread_affinity_verified': True, 'old_files_preserved': True}
    with (out / 'independent_audit.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
