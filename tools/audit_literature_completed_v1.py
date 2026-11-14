"""Independent CPU verification of completed Stage29 evidence and aggregates.

No old artifact writes; no GPU, training, inference or model selection.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def legal(mask, k):
    assert mask == sorted(set(mask))
    assert len(mask) == k and all(2 <= i < 40 for i in mask)


def independent_bootstrap(values, clusters):
    a = np.array(values, dtype=float)
    groups = defaultdict(list)
    for x, cluster in zip(a, clusters):
        groups[str(cluster)].append(float(x))
    totals = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    draws = np.random.default_rng(290579).integers(0, len(totals), (2000, len(totals)))
    boot = totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return float(a.mean()), np.quantile(boot, [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.run.resolve()
    cfg = load(out/'protocol.json')
    summary = load(out/'summary.json')
    assert load(out/'status.json')['exit_code'] == 0
    manifest = load(Path(cfg['data_run'])/'data_manifest.json')
    assert sha(Path(cfg['data_run'])/'data_manifest.json') == cfg['data_manifest_sha256']
    assert len(cfg['calibration_ids']) == 128
    assert all(manifest['records'][i]['split'] == 'train' for i in cfg['calibration_ids'])
    checked_hashes = 0
    for path, digest in cfg['source_sha256'].items():
        assert sha(path) == digest
        checked_hashes += 1
    for policy in cfg['policies']:
        if 'checkpoint' in policy:
            assert sha(policy['checkpoint']['path']) == policy['checkpoint']['sha256']
            checked_hashes += 1
    preserved = load(out/'preservation_before.json')
    for path, digest in preserved.items():
        assert sha(path) == digest, path
    assert len(preserved) == load(out/'preservation_after.json')['preserved_files']
    for name, expected in load(out/'evidence_hashes.json').items():
        assert sha(out/name) == expected['sha256']
        assert (out/name).stat().st_size == expected['bytes']
    inputs = {(r['task'], r['sample']): r for r in rows(out/'private_inputs.jsonl')}
    aggregate = {}
    k20 = defaultdict(lambda: defaultdict(list))
    unique_rows = total_rows = 0
    max_full_error = 0.
    full = {}
    for rank, k in enumerate(cfg['budgets']):
        policies = {(p['name'], p['seed'], p['k']): p for p in load(out/f'policies_{rank:03d}.json')}
        seen = set()
        for r in rows(out/f'private_quality_{rank:03d}.jsonl'):
            total_rows += 1
            ik = r['task'], r['sample']
            pk = r['policy'], r['seed'], r['k']
            uid = ik+pk
            assert uid not in seen
            seen.add(uid)
            policy = policies[pk]
            record = inputs[ik]
            assert r['document_sha256'] == record['document_sha256']
            assert r['router_tokens'] == len(record['route_ids'])
            assert r['target_tokens'] == sum(record['target_lengths'])
            legal(r['skips'], r['k'])
            if policy['kind'] == 'static':
                assert r['skips'] == sorted(policy['skips'])
            elif policy['kind'] == 'router':
                assert set(policy['prefix']).issubset(r['skips'])
            else:
                assert r['skips'] in [sorted(v['skips']) for v in policy['bank']['routes']]
            scores = r['scores']
            assert len(scores) == len(record['sequence_ids'])
            assert all(math.isfinite(v) for v in scores)
            assert math.isfinite(r['teacher_kl_sum']) and r['teacher_kl_sum'] >= -1e-3
            if record['gold'] is None:
                assert r['nll_sum'] == -scores[0]
                assert r['scored_tokens'] == record['target_lengths'][0]
            else:
                prediction = max(range(len(scores)), key=lambda i: scores[i])
                assert r['prediction'] == prediction
                assert r['correct'] == int(prediction == record['gold'])
                if record['task'] != 'winogrande':
                    prediction = max(range(len(scores)), key=lambda i: scores[i]/record['normalization_lengths'][i])
                    assert r['prediction_norm'] == prediction
                    assert r['correct_norm'] == int(prediction == record['gold'])
            if r['policy'] == 'full' and ik in full:
                error = max(abs(x-y) for x, y in zip(scores, full[ik]))
                assert error < 1e-5
                max_full_error = max(max_full_error, error)
                continue
            if r['policy'] == 'full':
                full[ik] = scores
            unique_rows += 1
            key = (r['task'],)+pk
            if key not in aggregate:
                aggregate[key] = {'count': 0, 'tokens': 0, 'kl': 0., 'nll': 0.,
                    'correct': 0, 'correct_norm': 0, 'masks': set()}
            a = aggregate[key]
            a['count'] += 1
            a['tokens'] += r['target_tokens']
            a['kl'] += r['teacher_kl_sum']
            a['nll'] += r.get('nll_sum', 0.)
            a['correct'] += r.get('correct', 0)
            a['correct_norm'] += r.get('correct_norm', 0)
            a['masks'].add(tuple(r['skips']))
            if r['k'] == 20 and r['policy'] in ('codebook8_train256_prompt', 'evopress_whole_opt', 'static_refined_bank_train800'):
                field = 'correct_norm' if r['task'] in ('piqa', 'hellaswag') else 'correct'
                value = r['nll_sum']/r['scored_tokens'] if r['task'] == 'wikitext_conditional' else r[field]
                k20[r['task'], r['policy']][ik].append(value)
        assert len(seen) == len(inputs)*len(policies)
        assert sha(out/f'private_quality_{rank:03d}.jsonl') == load(out/f'quality_receipt_{rank:03d}.json')['private_quality_sha256']
        print(json.dumps({'event': 'quality_audited', 'rank': rank, 'rows': len(seen)}), flush=True)
    assert unique_rows == summary['verified_unique_rows']
    assert len(summary['quality']) == len(aggregate)
    for r in summary['quality']:
        a = aggregate[tuple(r[x] for x in ('task', 'policy', 'seed', 'k'))]
        assert r['records'] == a['count']
        assert r['unique_masks'] == len(a['masks'])
        assert abs(r['teacher_kl_per_token']-a['kl']/a['tokens']) < 1e-10
        if 'conditional_ppl' in r:
            assert abs(r['nll_per_token']-a['nll']/a['tokens']) < 1e-10
            assert abs(r['conditional_ppl']-math.exp(a['nll']/a['tokens'])) < 1e-7
        else:
            assert abs(r['accuracy']-a['correct']/a['count']) < 1e-12
            if 'accuracy_norm' in r:
                assert abs(r['accuracy_norm']-a['correct_norm']/a['count']) < 1e-12
    paired_checked = []
    for task in ('hellaswag', 'piqa', 'winogrande', 'wikitext_conditional'):
        ids = sorted(ik for ik in inputs if ik[0] == task)
        dyn = k20[task, 'codebook8_train256_prompt']
        for base in ('evopress_whole_opt', 'static_refined_bank_train800'):
            values = [float(np.mean(dyn[ik]))-float(np.mean(k20[task, base][ik])) for ik in ids]
            clusters = [inputs[ik]['document'] for ik in ids] if task == 'wikitext_conditional' else list(range(len(ids)))
            difference, ci = independent_bootstrap(values, clusters)
            original = next(r for r in summary['paired_comparisons'] if (r['task'], r['k'], r['dynamic'], r['baseline']) ==
                (task, 20, 'codebook8_train256_prompt', base))['difference_dynamic_minus_baseline']
            assert abs(difference-original['difference']) < 1e-12
            assert max(abs(x-y) for x, y in zip(ci, original['ci95'])) < 1e-12
            paired_checked.append({'task': task, 'baseline': base, 'difference': difference, 'ci95': ci})
    search_audits = []
    for rank, k in enumerate(cfg['budgets']):
        traces = list(rows(out/f'search_trace_{rank:03d}.jsonl'))
        for seed in cfg['search_seeds']:
            current = [r for r in traces if r['seed'] == seed]
            assert len(current) == 1+2*cfg['search']['generations']
            parent = None
            for r in current:
                for mask in r['candidates']:
                    legal(mask, k)
                assert len(r['candidates']) == len(set(tuple(v) for v in r['candidates']))
                order = sorted(range(len(r['candidates'])), key=lambda i: (r['fitnesses'][i], tuple(r['candidates'][i])))
                n = 2 if r['stage'] == 'screen' else 1
                assert r['survivor_indices'] == order[:n]
                assert all(math.isfinite(v) for v in r['fitnesses'])
                if r['stage'] == 'initial':
                    assert r['generation'] == 0 and len(r['candidates']) == 64
                    parent = r['candidates'][order[0]]
                elif r['stage'] == 'screen':
                    assert len(r['candidates']) == 32 and parent not in r['candidates']
                    for mask in r['candidates']:
                        assert 1 <= len(set(parent)-set(mask)) <= 3
                    survivors = [r['candidates'][i] for i in order[:2]]
                else:
                    assert r['candidates'] == survivors+[parent]
                    parent = r['candidates'][order[0]]
            result = load(out/f'evopress_k{k}_s{seed}.json')
            assert result['skips'] == parent and len(result['history']) == 64
            search_audits.append({'k': k, 'seed': seed, 'selection_stages': len(current),
                'unique_searched_masks': result['unique_searched_masks'], 'search_seconds': result['search_seconds']})
    result = {'passed': True, 'input_records': len(inputs), 'total_rows_including_full_replays': total_rows,
        'unique_quality_rows': unique_rows, 'aggregate_groups_recomputed': len(aggregate),
        'cross_gpu_full_max_error': max_full_error, 'preserved_files_rehashed': len(preserved),
        'source_and_checkpoint_hash_checks': checked_hashes, 'calibration_train_only': True,
        'paired_k20_comparisons_recomputed': paired_checked, 'search_traces_audited': search_audits,
        'limits': 'CPU audit of stored evidence; no new GPU model replay. Per-seed means and original unadjusted bootstrap intervals; no across-seed population inference.'}
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('paired_k20_comparisons_recomputed', 'search_traces_audited')}), flush=True)


if __name__ == '__main__':
    main()
