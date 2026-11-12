"""Independent audit of Stage24 identities, target means, checkpoints and quality."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root, out):
    cfg, verification, summary = [read(out / name) for name in ('protocol.json', 'verification.json', 'summary.json')]
    assert verification['passed']
    for name, expected in cfg['source_sha256'].items():
        assert digest(root / 'tools' / name) == expected
    for path, expected in cfg['evidence_sha256'].items():
        assert digest(path) == expected
    old = Path(cfg['old_run']); earlier = Path(cfg['prompt_run']); previous = read(old / 'protocol.json')
    assert set(cfg['train_ids']) <= set(previous['train_ids'])
    assert set(cfg['selection_ids']) <= set(previous['selection_ids'])
    assert set(cfg['validation_ids']) <= set(previous['validation_ids'])
    assert not set(cfg['train_ids']) & set(cfg['validation_ids'])
    if not cfg['probe']:
        for field in ('train_ids', 'selection_ids', 'validation_ids'):
            assert cfg[field] == previous[field]
        assert cfg['epochs'] == 24 and len(cfg['train_ids']) == 800
    bank = read(out / 'candidate_bank.json')
    assert bank == read(old / 'candidate_bank.json') and bank['k'] == 20 and len(bank['routes']) == 8
    assert all(len(set(r['skips'])) == 20 for r in bank['routes'])
    old_labels = {r['sample']: r for rank in range(6) for r in lines(old / f'private_labels_{rank:03d}.jsonl')}
    reused = {r['sample']: r for rank in range(2) for r in lines(earlier / f'private_prompt_costs_{rank:03d}.jsonl')}
    data = read(Path(cfg['data_run']) / 'data_manifest.json')['records']
    plan = read(out / 'allocation.json'); assert plan['cpu_count'] == 42 and len(plan['gpu_indices']) == 4
    prompts = {}; new_count = 0; reused_count = 0
    for rank in range(4):
        path = out / f'private_prompt_costs_{rank:03d}.jsonl'; receipt = read(out / f'labels_worker_{rank:03d}.json')
        assert digest(path) == receipt['labels_sha256'] and receipt['passed']
        assert receipt['cpu_ids'] == plan['cpu_ids'][rank * 10:(rank + 1) * 10]
        assert receipt['threads'] == 10
        assert all(set(cpus) <= set(receipt['cpu_ids']) for cpus in receipt['thread_cpu_ids'].values())
        rows = lines(path); assert [r['sample'] for r in rows] == (cfg['validation_ids'] + cfg['train_ids'])[rank::4]
        assert receipt['same_chunk_reused_kl_nll_replay_error'] < 1e-6
        assert receipt['full_prompt_embeddings_replay_exact']
        for row in rows:
            sample = row['sample']; assert sample not in prompts
            assert row['prompt_length'] == data[sample]['prompt_length']
            assert row['scored_tokens'] == row['prompt_length'] - 1
            assert row['sequence_sha256'] == data[sample]['sequence_sha256'] == old_labels[sample]['sequence_sha256']
            assert len(row['prompt_kl']) == len(row['prompt_nll']) == 8
            assert np.isfinite(row['prompt_kl']).all() and min(row['prompt_kl']) >= -1e-5
            if row['reused']:
                assert row['prompt_kl'] == reused[sample]['prompt_kl'] and row['prompt_nll'] == reused[sample]['prompt_nll']
                reused_count += 1
            else:
                assert sample not in reused; new_count += 1
            prompts[sample] = row
    assert set(prompts) == set(cfg['train_ids'] + cfg['validation_ids'])
    if not cfg['probe']:
        assert len(prompts) == 900 and reused_count == 164 and new_count == 736
    # Float32 training targets derived independently; tolerate reduction rounding only.
    future = np.array([old_labels[s]['kl'] for s in cfg['train_ids']], dtype=np.float32)
    full = np.array([prompts[s]['prompt_kl'] for s in cfg['train_ids']], dtype=np.float32)
    f, p = future - future[:, :1], full - full[:, :1]
    prior = f.mean(0); aligned = prior + p - p.mean(0)
    assert np.allclose((.5 * (f + aligned)).mean(0), prior, atol=1e-6)
    assert np.allclose(aligned.mean(0), prior, atol=1e-6)
    quality = [r for rank in range(4) for r in lines(out / f'private_quality_{rank:03d}.jsonl')]
    index = {(r['source_sample'], r['policy'], r['arm'], r['seed']): r for r in quality}
    assert len(index) == len(quality) == 19 * len(cfg['validation_ids'])
    maximum_kl_error = 0.
    for rank in range(4):
        receipt = read(out / f'quality_worker_{rank:03d}.json')
        assert receipt['quality_sha256'] == digest(out / f'private_quality_{rank:03d}.jsonl')
        assert receipt['documents'] == len(cfg['validation_ids'][rank::4])
        assert receipt['cpu_ids'] == plan['cpu_ids'][rank * 10:(rank + 1) * 10]
        assert all(set(v) <= set(receipt['cpu_ids']) for v in receipt['thread_cpu_ids'].values())
    for row in quality:
        sample, route = row['source_sample'], row['route_index']
        assert sample in cfg['validation_ids'] and row['k'] == 20 and 0 <= route < 8
        error = abs(row['kl'] - old_labels[sample]['kl'][route]); maximum_kl_error = max(maximum_kl_error, error)
        assert error < 1e-6
    seeds = (579, 580, 581)
    for kind in ('mixed', 'prompt_full'):
        for seed in seeds:
            receipts = []
            for arm in ('prompt', 'constant'):
                receipt = read(out / f'{kind}_{arm}_s{seed}_fit.json'); receipts.append(receipt)
                assert receipt['updates'] == cfg['epochs'] * ((len(cfg['train_ids']) + 15) // 16)
                assert receipt['train_documents'] == len(cfg['train_ids'])
                assert digest(receipt['best']['path']) == receipt['best']['sha256']
                assert receipt['lambda'] == (.5 if kind == 'mixed' else 1.)
                assert np.allclose(receipt['common_prior'], prior, atol=1e-6)
                assert np.allclose(receipt['target_mean'], prior, atol=1e-6)
                assert receipt['parameters'] == 921992
                history = read(out / f'{kind}_{arm}_s{seed}_history.json')
                assert [h['epoch'] for h in history] == list(range(cfg['epochs'] + 1))
                candidates = [h for h in history if 'selection_kl' in h]
                expected_epochs = [epoch for epoch in range(cfg['epochs'] + 1) if epoch in (0, 1) or epoch % 4 == 0 or epoch == cfg['epochs']]
                assert [h['epoch'] for h in candidates] == expected_epochs
                best = min(candidates, key=lambda r: r['selection_kl'])
                assert best['epoch'] == receipt['best']['epoch'] and best['selection_kl'] == receipt['best']['selection_kl']
                actual = np.mean([index[s, kind, arm, seed]['kl'] for s in cfg['selection_ids']])
                assert abs(actual - best['selection_kl']) < 1e-6
                if not cfg['probe']:
                    old_receipt = read(old / f'huber_{arm}_s{seed}_fit.json')
                    assert receipt['initial_sha256'] == old_receipt['initial_sha256']
                    assert receipt['updates'] == old_receipt['updates']
            assert receipts[0]['initial_sha256'] == receipts[1]['initial_sha256']
            assert receipts[0]['common_prior'] == receipts[1]['common_prior']
            assert receipts[0]['common_scale'] == receipts[1]['common_scale']
    old_quality = {(r['source_sample'], r['policy'], r['seed']): r for rank in range(6) for r in lines(old / f'private_quality_{rank:03d}.jsonl')}
    for sample in cfg['validation_ids']:
        for arm in ('prompt', 'constant'):
            for seed in seeds:
                a, b = index[sample, 'future32', arm, seed], old_quality[sample, f'huber_{arm}', seed]
                assert a['route_index'] == b['route_index'] and abs(a['kl'] - b['kl']) < 1e-6
    for row in summary['comparisons']:
        group = [s for s in cfg['validation_ids'] if row['cohort'] == 'all_validation100' or
            ((s in cfg['selection_ids']) == (row['cohort'] == 'selection32'))]
        own = np.array([np.mean([index[s, row['policy'], row['arm'], seed][row['metric']] for seed in seeds]) for s in group])
        if row['baseline'] == 'strong_static':
            control = np.array([index[s, 'strong_static', 'none', None][row['metric']] for s in group])
        else:
            control = np.array([np.mean([index[s, 'future32', 'prompt', seed][row['metric']] for seed in seeds]) for s in group])
        assert abs(float(own.mean()) - row['own_mean']) < 1e-12
        assert abs(float(control.mean()) - row['baseline_mean']) < 1e-12
        assert abs(float((control - own).mean()) - row['difference']) < 1e-12
    for reference in cfg['reference_checkpoints'].values():
        assert digest(reference['path']) == reference['sha256']
    preservation = read(out / 'preservation_before.json')
    for path, expected in preservation.items():
        assert digest(path) == expected
    receipt = {'passed': True, 'probe': cfg['probe'], 'auditor_sha256': digest(__file__),
        'training_documents': len(cfg['train_ids']), 'validation_documents': len(cfg['validation_ids']),
        'complete_prompt_label_documents': len(prompts), 'reused_prompt_documents': reused_count,
        'new_prompt_documents': new_count, 'initialization_updates_and_normalization_matched': True,
        'supervision_means_match_future_training_prior': True, 'epochs_selected_only_on_original_selection_ids': True,
        'future32_control_routes_reproduced': True, 'quality_rows': len(quality), 'max_quality_kl_error': maximum_kl_error,
        'four_gpu_42_cpu_worker_affinity_verified': True, 'old_files_preserved': len(preservation)}
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    out = parser.parse_args().run.resolve(); root = Path(__file__).resolve().parents[1]
    result = audit(root, out)
    with (out / 'independent_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
