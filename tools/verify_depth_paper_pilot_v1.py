"""Audit the completed depth pilot without changing any prior result."""
import argparse
import json
import math
import random
from pathlib import Path

from depth_paper_pilot_v2 import importance_order, tables
from diagnose_prompt_router_v2 import derangement
from stage1_benchmark_v2 import sha256, write_json


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def check_search(steps, ordered, samples, objective, maximum):
    if len(steps) != maximum or len(ordered) != maximum:
        raise ValueError('Incomplete search')
    for number, row in enumerate(steps, 1):
        prior = ordered[:number - 1]
        if row['step'] != number or row['previous_skips'] != prior or row['source_samples'] != samples or row['objective'] != objective:
            raise ValueError('Search state/provenance mismatch')
        pairs = row['candidate_costs']
        if len(pairs) != len({i for i, cost in pairs}) or {i for i, cost in pairs} != set(range(2, 40)) - set(prior):
            raise ValueError('Invalid candidate population')
        if any(not math.isfinite(cost) for i, cost in pairs):
            raise ValueError('Nonfinite candidate cost')
        layer, cost = min(pairs, key=lambda pair: (pair[1], pair[0]))
        if row['selected_layer'] != layer or ordered[number - 1] != layer or row['joint_cost'] != cost:
            raise ValueError('Not the greedy joint minimum')
    return sum(len(row['candidate_costs']) * len(samples) for row in steps)


def verify(root, run):
    status = json.loads((run / 'status.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    if status.get('state') != 'completed' or status.get('exit_code') != 0:
        raise ValueError('Incomplete run')
    if protocol['old_test_used'] or summary['old_test_used'] or summary['new_router_trained']:
        raise ValueError('Wrong experiment scope')
    data_run = Path(protocol['data_file']).parent
    manifest = json.loads((data_run / 'data_manifest.json').read_text())
    count = 2 if protocol['probe'] else 16
    for split, field in [('calibration', 'calibration_ids'), ('pilot', 'pilot_ids')]:
        if protocol[field] != [r['sample'] for r in manifest['records'] if r['split'] == split][:count]:
            raise ValueError('Split changed')
    if sha256(protocol['data_file']) != protocol['data_sha256'] or sha256(protocol['router_checkpoint']) != protocol['router_sha256']:
        raise ValueError('Old private inputs changed')
    if sha256(root / 'tools/depth_paper_pilot_v2.py') != protocol['source_sha256']:
        raise ValueError('Run source hash mismatch')
    maximum = max(protocol['budgets'])
    if not protocol['probe'] and protocol['budgets'] != [6, 8, 10, 12, 16, 20, 24]:
        raise ValueError('Budget grid changed')
    static = json.loads((run / 'static_paths.json').read_text())
    scores = 0
    for objective, policy in [('nll', 'sleb_nll_opt'), ('kl', 'static_joint_kl')]:
        path = json.loads((run / f'calibration_{objective}_path.json').read_text())
        steps = lines(run / f'calibration_{objective}_steps.jsonl')
        if path['steps'] != steps or static[policy] != path['ordered_skips']:
            raise ValueError('Calibration records disagree')
        scores += check_search(steps, path['ordered_skips'], protocol['calibration_ids'], objective, maximum)
    first = lines(run / 'calibration_nll_steps.jsonl')[0]['candidate_costs']
    if static['shortened_ppl_opt'] != importance_order(first) or static['shortened_ppl_plus'] != importance_order(first, (0, 1, 2, 3, 38, 39)):
        raise ValueError('One-shot PPL ranking changed')
    bi = json.loads((run / 'shortgpt_calibration.json').read_text())
    if static['shortgpt_bi_opt'] != importance_order(bi['scores']) or bi['tokens_per_layer'] != len(protocol['calibration_ids']) * 160:
        raise ValueError('BI calibration changed')
    for seed in [579, 580, 581]:
        path = list(range(2, 40))
        random.Random(seed).shuffle(path)
        if path != static[f'random_static_seed{seed}']:
            raise ValueError('Random reference changed')
    oracles = {}
    for path in sorted(run.glob('oracle_paths_*.jsonl')):
        for row in lines(path):
            sample = row['sample']
            if sample in oracles or sample not in protocol['pilot_ids'] or row['split'] != 'pilot':
                raise ValueError('Wrong oracle population')
            steps = lines(run / f'oracle_sample{sample:04d}_steps.jsonl')
            if steps != row['steps']:
                raise ValueError('Oracle records disagree')
            scores += check_search(steps, row['ordered_skips'], [sample], 'kl', maximum)
            oracles[sample] = row['ordered_skips']
    if sorted(oracles) != protocol['pilot_ids']:
        raise ValueError('Missing oracle')
    donors = derangement(protocol['pilot_ids'], 579)
    quality = [r for p in sorted(run.glob('quality_*.jsonl')) for r in lines(p)]
    expected_keys = {(s, 0, 'full') for s in protocol['pilot_ids']}
    for sample in protocol['pilot_ids']:
        for k in protocol['budgets']:
            expected_keys |= {(sample, k, policy) for policy in list(static) + ['oracle_joint_kl', 'oracle_shuffled']}
        if 6 in protocol['budgets']:
            expected_keys |= {(sample, 6, 'router_trained_k6'), (sample, 6, 'learned_static_k6')}
    keys = {(r['sample'], r['k'], r['policy']) for r in quality}
    if keys != expected_keys or len(keys) != len(quality):
        raise ValueError('Incomplete or duplicate quality rows')
    for row in quality:
        sample, k, policy = row['sample'], row['k'], row['policy']
        actual = row['skipped_layers']
        if row['split'] != 'pilot' or row['continuation_tokens'] != 32:
            raise ValueError('Quality workload changed')
        if len(actual) != k or len(set(actual)) != k or not set(actual) <= set(range(2, 40)):
            raise ValueError('Invalid skip budget')
        if policy == 'router_trained_k6':
            if k != 6:
                raise ValueError('K6 router extrapolated')
            continue
        if policy == 'full':
            expected = []
            if row['continuation_kl'] != 0:
                raise ValueError('Full model KL nonzero')
        elif policy == 'learned_static_k6':
            expected = [2, 4, 8, 10, 14, 16]
        elif policy == 'oracle_joint_kl':
            expected = oracles[sample][:k]
        elif policy == 'oracle_shuffled':
            expected = oracles[donors[sample]][:k]
        else:
            expected = static[policy][:k]
        if actual != sorted(expected):
            raise ValueError('Evaluation mask changed')
    computed = tables(quality)
    for field in computed:
        if json.dumps(computed[field], sort_keys=True) != json.dumps(summary[field], sort_keys=True):
            raise ValueError('Summary arithmetic mismatch')
    cache = json.loads((run / 'cache_probes.json').read_text())
    for k, item in zip([6, maximum], cache):
        half = item['fp16']
        if half['precision'] != 'torch.float16' or half['route'] != sorted(static['sleb_nll_opt'][:k]):
            raise ValueError('Cache audit precision or route mismatch')
        if not half['passes_original_distribution_tolerance']:
            full = item['fp32']
            if not full or not full['passes_original_distribution_tolerance'] or full['precision'] != 'torch.float32':
                raise ValueError('Unresolved cache difference')
    if len(cache) != 2:
        raise ValueError('Missing cache probe')
    for worker in json.loads((run / 'workers.json').read_text()):
        if not set(worker['cpu_ids']) <= set(range(21)) or worker['threads'] > 10:
            raise ValueError('Resource budget exceeded')
    preserved = json.loads((run / 'preservation_before.json').read_text())
    for path, digest in preserved.items():
        if sha256(path) != digest:
            raise ValueError('Old source/evidence changed: ' + path)
    return {'verified_quality_rows': len(quality), 'verified_scalar_candidate_scores': scores,
        'calibration_documents': len(protocol['calibration_ids']), 'pilot_documents': len(protocol['pilot_ids']),
        'budgets': protocol['budgets'], 'source_preserved_files': len(preserved),
        'old_test_used': False, 'new_router_trained': False, 'probe': protocol['probe'],
        'fp16_cache_outcomes': [r['outcome'] for r in cache], 'summary_sha256': sha256(run / 'summary.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    result = verify(root, run)
    write_json(run / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
