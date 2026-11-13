"""Independent Stage26 metric reconstruction and GPU feature/checkpoint replays."""
import argparse
import inspect
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load_choice
from audit_crossfit_routing_v1 import apply_gate, gate
from crossfit_routing_core_v1 import grouped_folds, neural_fit
from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers, read_lines, resources, setup_worker, thread_record, write
from horizontal_quality_v3 import pin_threads
from semantic_prefix_core_v1 import fit_representation, hidden_prompt, load_prefix
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(Path(path).read_text())


def paths(out, cfg, arm, fold, seed):
    directory = Path(cfg['reference_run']) if arm == 'embedding' else out
    name = f'f{fold}_s{seed}' if arm == 'embedding' else f'{arm}_f{fold}_s{seed}'
    return directory / f'{name}_fit.json', directory / f'{name}_history.json', directory / f'private_{name}_scores.pth'


def worker(rank, job, inbox, messages):
    setup_worker(rank); out = Path(job['main_run']); audit_out = Path(job['output']); cfg = load(out / 'protocol.json')
    device = f'cuda:{rank}'; data = Records(cfg['data_run']); fold = cfg['folds'][rank]
    original = torch.load(Path(cfg['old_run']) / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    hidden = {}
    for j in range(4):
        hidden.update(torch.load(out / f'private_hidden2_{j:03d}.pth', weights_only=True, map_location='cpu'))
    assert set(hidden) == set(cfg['ids'])
    assert all(value.shape == (1, data.rows[i]['prompt_length'], 5120) for i, value in hidden.items())
    selected = sorted(fold['outer_ids'], key=lambda i: (data.rows[i]['prompt_length'], i))
    samples = [selected[0], selected[len(selected) // 2], selected[-1]]
    prefix = load_prefix(cfg['model'], device); feature_error = 0.; replay_error = 0.
    with torch.inference_mode():
        for sample in samples:
            raw, p = data[sample]
            prompt = torch.tensor(raw[:p], device=device, dtype=torch.long)[None]
            actual, embedding = hidden_prompt(prefix, prompt)
            feature_error = max(feature_error, float((actual.cpu() - hidden[sample]).abs().max()))
            assert torch.equal(actual.cpu(), hidden[sample])
            assert torch.equal(embedding.cpu(), original[sample])
    del prefix; torch.cuda.empty_cache()
    checked = []
    for seed in cfg['seeds']:
        for arm in ('embedding', 'hidden2', 'constant'):
            fit_path, _, score_path = paths(out, cfg, arm, rank, seed)
            receipt = load(fit_path); bundle = torch.load(score_path, weights_only=True)
            router = load_choice(receipt['best']['path'], device)
            assert router.observe == (arm != 'constant')
            with torch.inference_mode():
                for sample in samples:
                    value = hidden[sample] if arm == 'hidden2' else original[sample]
                    actual = router(value.to(device)).cpu()
                    expected = bundle['scores'][bundle['ids'].index(sample):bundle['ids'].index(sample) + 1]
                    error = float((actual - expected).abs().max()); replay_error = max(replay_error, error)
                    assert torch.equal(actual, expected), (rank, seed, arm, error)
                if arm == 'constant':
                    zeros = torch.zeros(1, 1, 5120, dtype=torch.float16, device=device)
                    noise = torch.randn(1, 17, 5120, dtype=torch.float16, device=device)
                    assert torch.equal(router(zeros), router(noise))
                    assert torch.equal(bundle['scores'], bundle['scores'][:1].expand_as(bundle['scores']))
            checked.append({'arm': arm, 'seed': seed, 'documents': len(samples), 'epoch': receipt['best']['epoch']})
            del router
    write(audit_out / f'worker_{rank:03d}.json', {'passed': True, 'fold': rank,
        'full_prompt_lengths': [data.rows[i]['prompt_length'] for i in samples], 'maximum_feature_error': feature_error,
        'maximum_router_score_error': replay_error, 'constant_ignores_content_and_length': True,
        'checkpoints': checked, **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def reconstruct(root, out, audit_out):
    cfg = load(out / 'protocol.json'); assert not cfg['probe'] and cfg['epochs'] == 24 and cfg['seeds'] == [579, 580, 581]
    for name, digest in cfg['source_sha256'].items():
        assert sha256(root / 'tools' / name) == digest
    for path, digest in cfg['evidence_sha256'].items():
        assert sha256(path) == digest
    preserved = load(out / 'preservation_before.json')
    for path, digest in preserved.items():
        assert sha256(path) == digest, path
    original_source = inspect.getsource(neural_fit)
    expected_source = original_source.replace('def neural_fit(out, cfg, fold, embeddings, labels, seed, device, messages, rank):',
        'def fit_representation(out, cfg, fold, embeddings, labels, seed, device, messages, rank, arm, observe=True):')
    expected_source = expected_source.replace("name = f'f{fold[\"fold\"]}_s{seed}'", "name = f'{arm}_f{fold[\"fold\"]}_s{seed}'")
    expected_source = expected_source.replace('width=128, observe=True', 'width=128, observe=observe').replace("'observe': True", "'observe': observe")
    expected_source = expected_source.replace("'fold': fold['fold'], 'seed': seed,", "'fold': fold['fold'], 'arm': arm, 'seed': seed,")
    expected_source = expected_source.replace("'phase': 'neural_fit', 'fold':", "'phase': 'neural_fit', 'arm': arm, 'fold':")
    assert expected_source.strip() == inspect.getsource(fit_representation).strip()
    old = Path(cfg['old_run']); previous = load(old / 'protocol.json'); ids = cfg['ids']
    assert ids == previous['train_ids'] and len(ids) == 800 and not set(ids) & set(previous['validation_ids'])
    manifest = load(Path(cfg['data_run']) / 'data_manifest.json')['records']
    assert cfg['folds'] == grouped_folds(ids, manifest) == load(Path(cfg['reference_run']) / 'protocol.json')['folds']
    bank = load(old / 'candidate_bank.json')
    assert bank['k'] == 20 and len(bank['routes']) == 8
    assert all(len(set(r['skips'])) == 20 and not set(r['skips']) & {0, 1} for r in bank['routes'])
    labels = {r['sample']: r for j in range(6) for r in read_lines(old / f'private_labels_{j:03d}.jsonl') if r['sample'] in ids}
    assert len(labels) == 800
    for sample in ids:
        assert labels[sample]['sequence_sha256'] == manifest[sample]['sequence_sha256'] == cfg['sequence_sha256'][str(sample)]
    choices = {(r['fold'], r['arm'], r['seed']): r for r in load(out / 'policy_choices.json')}
    reconstructed = {}; outer_seen = []; initialization = []; full_prompt_tokens = 0
    for rank in range(4):
        path = out / f'private_hidden2_{rank:03d}.pth'; receipt = load(out / f'features_worker_{rank:03d}.json')
        assert sha256(path) == receipt['sha256']
        shard = torch.load(path, weights_only=True, map_location='cpu')
        assert set(shard) == set(ids[rank::4]) and len(shard) == receipt['documents']
        for sample, value in shard.items():
            assert value.shape == (1, manifest[sample]['prompt_length'], 5120) and value.dtype == torch.float16
            full_prompt_tokens += value.shape[1]
        del shard
        probe = load(Path(cfg['probe_run']) / f'prefix_reference_{rank:03d}.json')
        assert probe['passed'] and probe['tokens'] == (64, 129, 513, 2016)[rank] and probe['maximum_error'] == 0
    for fold in cfg['folds']:
        f = fold['fold']; parts = [fold[key] for key in ('fit_ids', 'selection_ids', 'outer_ids')]
        assert [len(v) for v in parts] == [520, 80, 200]
        for field in ('record_sha256', 'record_lead_sha256'):
            keys = [{manifest[i][field] for i in part} for part in parts]
            assert all(not keys[a] & keys[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        joined = sum(parts, []); costs = np.asarray([labels[i]['kl'] for i in joined]); nll = np.asarray([labels[i]['nll'] for i in joined])
        outer_seen += parts[2]; baseline = int(costs[:520].mean(0).argmin())
        relative = costs[:520] - costs[:520, :1]; scale = max(float(relative[:, 1:].std(ddof=1)), .03)
        predictions = {'fold_static': np.full(800, baseline), 'future_known_best': costs.argmin(1)}
        for seed in cfg['seeds']:
            hashes = []
            for arm in ('embedding', 'hidden2', 'constant'):
                fit_path, history_path, score_path = paths(out, cfg, arm, f, seed)
                receipt = load(fit_path); hashes.append(receipt['initial_sha256'])
                assert receipt['updates'] == 792 and receipt['fit_documents'] == 520 and receipt['selection_documents'] == 80
                assert sha256(receipt['best']['path']) == receipt['best']['sha256'] and sha256(score_path) == receipt['score_sha256']
                checkpoint = torch.load(receipt['best']['path'], weights_only=True, map_location='cpu')
                target32 = torch.tensor(costs[:520], dtype=torch.float32); target32 -= target32[:, :1].clone()
                assert torch.equal(checkpoint['state_dict']['prior'], target32.mean(0))
                assert torch.equal(checkpoint['state_dict']['scale'], target32[:, 1:].std().clamp_min(.03))
                assert checkpoint['observe'] == (arm != 'constant')
                history = [r for r in load(history_path) if 'selection_kl' in r]
                assert [r['epoch'] for r in history] == [0, 1, 4, 8, 12, 16, 20, 24]
                best = min(history, key=lambda row: row['selection_kl'])
                assert best['epoch'] == receipt['best']['epoch'] == checkpoint['epoch']
                bundle = torch.load(score_path, weights_only=True); assert bundle['ids'] == joined
                scores = bundle['scores'].numpy().astype(np.float64)
                assert abs(costs[np.arange(520, 600), scores[520:600].argmin(1)].mean() - best['selection_kl']) < 1e-12
                rule = gate(scores[520:600], costs[520:600], baseline, scale)
                selected = choices.pop((f, arm, seed))
                assert selected['fit_receipt_sha256'] == sha256(fit_path)
                assert selected['baseline_index'] == baseline and abs(selected['scale'] - scale) < 1e-12
                assert all(selected['gate'][key] == value for key, value in rule.items())
                predictions[f'{arm}_raw_s{seed}'] = scores.argmin(1)
                predictions[f'{arm}_gate_s{seed}'] = apply_gate(scores, rule, baseline, scale)
            assert len(set(hashes)) == 1
            initialization.append({'fold': f, 'seed': seed, 'initial_sha256': hashes[0]})
        for policy, routes in predictions.items():
            for position, (sample, route) in enumerate(zip(joined, routes)):
                cohort = 'fit' if position < 520 else ('selection' if position < 600 else 'outer')
                reconstructed[f, policy, cohort, sample] = (int(route), baseline, float(costs[position, route]),
                    float(nll[position, route]), float(costs[position, baseline]), float(nll[position, baseline]))
    assert not choices and len(outer_seen) == 800 and Counter(outer_seen) == Counter(ids)
    actual = read_lines(out / 'private_quality_rows.jsonl'); assert len(actual) == len(reconstructed) == 64000
    groups = defaultdict(list); outer = defaultdict(list)
    for row in actual:
        key = row['fold'], row['policy'], row['cohort'], row['sample']
        expected = reconstructed.pop(key)
        assert tuple(row[k] for k in ('route_index', 'baseline_index', 'kl', 'nll', 'static_kl', 'static_nll')) == expected
        groups[row['fold'], row['cohort'], row['policy']].append(row)
        if row['cohort'] == 'outer':
            outer[row['policy'], row['sample']].append(row)
    assert not reconstructed
    summary = load(out / 'summary.json'); assert summary['outer_documents'] == 800 and not summary['engineering_only']
    for row in summary['per_fold']:
        group = groups.pop((row['fold'], row['cohort'], row['policy']))
        for metric in ('kl', 'nll'):
            assert abs(row[metric] - np.mean([r[metric] for r in group])) < 1e-12
        assert abs(row['improvement'] - np.mean([r['static_kl'] - r['kl'] for r in group])) < 1e-12
        assert row['switches'] == sum(r['route_index'] != r['baseline_index'] for r in group)
        assert row['beneficial'] == sum(r['kl'] < r['static_kl'] for r in group)
        assert row['harmful'] == sum(r['kl'] > r['static_kl'] for r in group)
        assert row['route_histogram'] == {str(k): v for k, v in Counter(r['route_index'] for r in group).items()}
    assert not groups
    data = {}; draws = np.random.default_rng(260579).integers(0, 800, (2000, 800))
    def verify_difference(difference, row):
        a = np.asarray(difference); assert abs(row['difference'] - a.mean()) < 1e-12
        assert np.allclose(row['ci95'], np.quantile(a[draws].mean(1), [.025, .975]), atol=1e-12, rtol=0)
    for row in summary['outer_summary']:
        policy, metric = row['policy'], row['metric']; base = policy.removesuffix('_mean3')
        members = [f'{base}_s{s}' for s in cfg['seeds']] if policy.endswith('_mean3') else [policy]
        own = []; static = []
        for sample in sorted(ids):
            group = sum((outer[m, sample] for m in members), []); assert len(group) == len(members)
            own.append(np.mean([r[metric] for r in group])); static.append(group[0]['static_' + metric])
        own = np.asarray(own); static = np.asarray(static); data[policy, metric] = own
        assert abs(row['mean'] - own.mean()) < 1e-12 and abs(row['static_mean'] - static.mean()) < 1e-12
        verify_difference(static - own, row)
    for row in summary['comparisons']:
        verify_difference(data[row['baseline'], row['metric']] - data[row['policy'], row['metric']], row)
    allocation = load(out / 'allocation.json'); assert allocation['gpu_indices'] == [0, 1, 2, 3] and allocation['cpu_count'] == 42
    for f in range(4):
        for directory, filename in ((out, f'worker_{f:03d}.json'), (out, f'features_worker_{f:03d}.json'), (audit_out, f'worker_{f:03d}.json')):
            receipt = load(directory / filename); assert receipt['threads'] == 10
            assert receipt['cpu_ids'] == allocation['cpu_ids'][f*10:(f+1)*10]
            assert all(set(v) <= set(receipt['cpu_ids']) for v in receipt['thread_cpu_ids'].values())
    return {'passed': True, 'auditor_sha256': sha256(__file__), 'preserved_files': len(preserved),
        'private_quality_rows_reconstructed': len(actual), 'all800_documents_one_outer_fold_each': True,
        'full_prompt_tokens_verified': full_prompt_tokens, 'historical_dev100_excluded': True,
        'original_fit_unchanged_except_observation_and_names': True, 'matched_updates_per_model': 792,
        'matched_initializations': initialization, 'inner80_only_gate_and_checkpoint_reconstructed': True,
        'all_metrics_and_bootstrap_reconstructed': True, 'gpu_feature_replay_documents': 12,
        'gpu_router_replay_evaluations': 108, 'maximum_gpu_feature_error': 0., 'maximum_gpu_score_error': 0.,
        'constant_content_and_length_invariance_verified': True, 'resources_verified': '4GPU/42CPU',
        'limits': cfg['limits']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = args.run.resolve(); audit_out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    cpus = resources(); pin_threads(cpus[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert load(out / 'verification.json')['passed']
    job = {'main_run': str(out), 'output': str(audit_out)}
    with Workers(worker, job) as workers:
        workers.receive(4, 'done')
    result = reconstruct(root, out, audit_out)
    write(audit_out / 'independent_audit.json', result); write(out / 'independent_audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
