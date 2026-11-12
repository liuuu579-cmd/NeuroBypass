"""Independent read-only Stage20 control, replay, and checkpoint-selection audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    cfg = load(run / 'protocol.json')
    assert not cfg['probe']
    assert load(run / 'verification.json')['passed']
    for name, digest in cfg['source_sha256'].items():
        assert sha256(root / 'tools' / name) == digest
    for path, digest in cfg['evidence_sha256'].items():
        assert sha256(path) == digest
    assert sha256(run / 'private_embeddings.pth') == cfg['embedding_sha256']
    assert sha256(run / 'candidate_bank.json') == cfg['bank_sha256']
    assert sha256(run / 'private_inputs.jsonl') == cfg['input_sha256']
    rows, labels = [], {}
    for rank in range(6):
        path = run / f'private_labels_{rank:03d}.jsonl'
        assert sha256(path) == load(run / f'labels_{rank:03d}.json')['labels_sha256']
        for r in read_lines(path):
            assert r['sample'] not in labels
            labels[r['sample']] = r
        path = run / f'private_quality_{rank:03d}.jsonl'
        assert sha256(path) == load(run / f'worker_{rank:03d}.json')['quality_sha256']
        rows += read_lines(path)
    assert len(labels) == 900 and len(rows) == 1700
    assert len(cfg['train_ids']) == 800 and len(cfg['validation_ids']) == 100
    assert not set(cfg['train_ids']) & set(cfg['validation_ids'])
    assert sum(not r['reused'] for r in labels.values()) == 544
    means = np.array([labels[i]['kl'] for i in cfg['train_ids']]).mean(0)
    controls = load(run / 'static_controls.json')
    assert controls['best_route'] == int(means.argmin())
    assert np.array_equal(means, np.array(controls['training_mean_kl']))
    previous = Path(cfg['previous_run'])
    old = {(r['sample'], r['seed']): r for i in range(3)
        for r in read_lines(previous / f'private_quality_{i:03d}.jsonl')
        if r['bank_name'] == 'k20_b08' and r['policy'] == 'prompt' and r['task'] == 'c4_development'}
    replay = []
    for r in rows:
        error = abs(r['kl'] - labels[r['source_sample']]['kl'][r['route_index']])
        assert error < 1e-6
        if r['policy'] == 'old256_prompt':
            reference = old[r['sample'], r['seed']]
            assert reference['route_index'] == r['route_index'] and reference['skips'] == r['skips']
            replay.append(abs(r['kl'] - reference['kl']))
    assert len(replay) == 300 and max(replay) < 1e-6
    checkpoints = []
    for objective in ('huber', 'rank'):
        for seed in cfg['seeds']:
            for arm in ('prompt', 'constant'):
                name = f'{objective}_{arm}_s{seed}'
                record = load(run / f'{name}_fit.json')
                history = load(run / f'{name}_history.json')
                selected = min((r for r in history if 'selection_kl' in r), key=lambda r: (r['selection_kl'], r['epoch']))
                assert selected['epoch'] == record['best']['epoch']
                assert selected['selection_kl'] == record['best']['selection_kl']
                assert sha256(record['best']['path']) == record['best']['sha256']
                used = [r for r in rows if r['policy'] == f'{objective}_{arm}' and r['seed'] == seed and r['source_sample'] in cfg['selection_ids']]
                assert len(used) == 32
                assert abs(np.mean([r['kl'] for r in used]) - selected['selection_kl']) < 1e-6
                checkpoints.append({'objective': objective, 'arm': arm, 'seed': seed, 'epoch': selected['epoch'], 'passed': True})
    allocation = load(run / 'allocation.json')
    assert len(allocation['gpu_indices']) == 6 and allocation['cpu_count'] == 64
    for path, digest in load(run / 'preservation_before.json').items():
        assert sha256(path) == digest
    receipt = {'passed': True, 'original_training_documents': 800, 'validation_documents': 100,
        'new_joint_label_documents': 544, 'existing_joint_label_documents': 356,
        'strongest_static_recomputed_training_only': True,
        'old256_router_replay_rows': len(replay), 'old256_replay_max_abs_error': max(replay),
        'checkpoints': checkpoints, 'gpu_count': 6, 'cpu_count': 64,
        'preserved_old_files_rechecked': True, 'audit_source_sha256': sha256(Path(__file__))}
    write(run / 'independent_audit.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
