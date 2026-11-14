"""Stage30: small frozen Stage7 versus EvoPress comparison, exact old behavior.

No fitting, no baseline search, no performance timing. Preserve the original
Stage7 question templates, observation boundary and per-K model checkpoints.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import ThreeWorkers, setup
from full_context_experiment_v1 import preserve, read_lines, thread_record, verify_preserved, write
from headroom_ruler_core_v1 import paired
from horizontal_quality_v3 import score_probe, score_sequence, summary_tables
from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import raw_prompt_embeddings
from prompt_permutation_core_v1 import score_fields
from stage1_benchmark_v2 import sha256, stamp

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'runs/stage7_horizontal_main_20261108_02'
EVO = ROOT/'runs/stage29_literature_20261113_01_main'
KS = [12, 16]


def load(path):
    return json.loads(Path(path).read_text())


def prepare(out, probe):
    source = read_lines(OLD/'private_benchmark_inputs.jsonl')
    newer = {(r['task'], r['sample']): r for r in read_lines(EVO/'private_inputs.jsonl')}
    rng = np.random.default_rng(300579)
    selected = []
    for task in ('piqa', 'hellaswag', 'winogrande'):
        population = [r for r in source if r['task'] == task]
        positions = sorted(rng.choice(len(population), 4 if probe else 64, replace=False).tolist())
        selected.extend(population[i] for i in positions)
    wiki = defaultdict(list)
    for r in source:
        if r['task'] == 'wikitext_conditional':
            wiki[r['document']].append(r)
    documents = sorted(rng.choice(sorted(wiki), 4 if probe else 32, replace=False).tolist())
    selected.extend(wiki[d][int(rng.integers(len(wiki[d])))] for d in documents)
    selected.sort(key=lambda r: (r['task'], r['sample']))
    for r in selected:
        other = newer[r['task'], r['sample']]
        for key in ('route_ids', 'sequence_ids', 'target_lengths', 'normalization_lengths', 'gold', 'document_sha256'):
            assert r[key] == other[key], (r['task'], r['sample'], key)
        assert 1 <= len(r['route_ids']) <= 128
    with (out/'private_inputs.jsonl').open('x') as stream:
        for r in selected:
            stream.write(json.dumps(r)+'\n')
    write(out/'data_manifest.json', {'seed': 300579, 'label_blind_sampling': True,
        'task_counts': {t: sum(r['task'] == t for r in selected) for t in sorted({r['task'] for r in selected})},
        'wiki_sampling': 'Uniform32 articles then one random window per article; paired article sampling, no contiguous-window overcount.',
        'private_inputs_sha256': sha256(out/'private_inputs.jsonl'),
        'stage7_input_sha256': sha256(OLD/'private_benchmark_inputs.jsonl'),
        'stage29_input_sha256': sha256(EVO/'private_inputs.jsonl'),
        'exact_shared_inputs': True,
        'router_tokens': {t: {'min': min(len(r['route_ids']) for r in selected if r['task'] == t),
            'max': max(len(r['route_ids']) for r in selected if r['task'] == t)} for t in sorted({r['task'] for r in selected})},
        'scope': 'Random subset of previously reviewed task populations, exploratory; no new final test, no training or policy selection.'})
    return selected


def make_policies():
    old = load(OLD/'protocol.json')
    names = ('pure', 'hybrid', 'static_joint_kl')
    policies = [p for p in old['policies'] if p['name'] == 'full' or (p['k'] in KS and p['name'] in names)]
    for p in policies:
        if p['kind'] == 'router':
            assert sha256(p['checkpoint']['path']) == p['checkpoint']['sha256']
    for k in KS:
        for seed in (579, 580, 581):
            path = EVO/f'evopress_k{k}_s{seed}.json'
            r = load(path)
            policies.append({'name': 'evopress_whole_opt', 'kind': 'static', 'k': k,
                'seed': seed, 'skips': r['skips'], 'source': str(path), 'source_sha256': sha256(path)})
    assert len(policies) == 21
    return old, policies


def worker(rank, cfg, inbox, messages):
    setup(rank)
    from transformers import AutoModelForCausalLM
    device = f'cuda:{rank}'
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    routers = {}
    for policy in cfg['policies']:
        if policy['kind'] == 'router':
            path = policy['checkpoint']['path']
            if path not in routers:
                router = DepthRouter().to(device).eval()
                router.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
                routers[path] = router.requires_grad_(False)
    out = Path(cfg['output'])
    all_inputs = read_lines(out/'private_inputs.jsonl')
    inputs = all_inputs[rank::3]
    probes = []
    with torch.inference_mode():
        for task in sorted({r['task'] for r in inputs}):
            r = next(r for r in inputs if r['task'] == task)
            for mask in ([], cfg['fixed_prefix']):
                probes.append({'task': task, **score_probe(model, r, mask)})
    write(out/f'worker_probe_{rank:03d}.json', {'passed': True, 'likelihood_probes': probes, **thread_record()})
    with torch.inference_mode(), (out/f'private_quality_{rank:03d}.jsonl').open('x') as stream:
        for offset, r in enumerate(inputs):
            observation = torch.tensor(r['route_ids'], dtype=torch.long, device=device)[None]
            emb = raw_prompt_embeddings(model, observation)
            assert emb.shape[1] == len(r['route_ids'])
            encoded = {}
            cache = {}
            for p in cfg['policies']:
                if p['kind'] == 'static':
                    route = p['skips']
                else:
                    path = p['checkpoint']['path']
                    if path not in encoded:
                        encoded[path] = routers[path].encode(emb)
                    route = routers[path].route_encoded(encoded[path], p['k'], p['prefix'])[0].tolist()
                    assert route[:len(p['prefix'])] == p['prefix']
                mask = tuple(sorted(route))
                assert len(mask) == len(set(mask)) == p['k']
                assert all(2 <= i < 40 for i in mask)
                if mask not in cache:
                    cache[mask] = [score_sequence(model, ids, n, mask) for ids, n in zip(r['sequence_ids'], r['target_lengths'])]
                scores = cache[mask]
                row = {'task': r['task'], 'sample': r['sample'], 'document': r.get('document'),
                    'document_sha256': r['document_sha256'], 'policy': p['name'], 'seed': p['seed'], 'k': p['k'],
                    'skipped_layers': list(mask), 'router_observation_tokens': len(r['route_ids']),
                    'target_lengths': r['target_lengths'], 'normalization_lengths': r['normalization_lengths'],
                    'truncated_context_tokens': r['truncated_context_tokens'], 'scores': scores, **score_fields(r, scores)}
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            if offset % 8 == 0 or offset+1 == len(inputs):
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'stage7_evopress',
                    'completed': offset+1, 'total': len(inputs)})
    write(out/f'worker_receipt_{rank:03d}.json', {'passed': True,
        'quality_sha256': sha256(out/f'private_quality_{rank:03d}.jsonl'), **thread_record()})
    messages.put({'kind': 'done', 'rank': rank})


def audit_replays(out, result, cfg):
    # Historical replay is independent of the new GPU scoring pass. Check both
    # old router outputs and newly rerun fixed EvoPress configurations.
    target = {(r['task'], r['sample'], r['policy'], r['seed'], r['k']): r for r in result}
    matched = set()
    errors = {'stage7': 0., 'stage29_evopress': 0.}
    counts = defaultdict(int)
    for label, directory, pattern in [('stage7', OLD, 'quality_*.jsonl'), ('stage29_evopress', EVO, 'private_quality_*.jsonl')]:
        for path in sorted(directory.glob(pattern)):
            for r in read_lines(path):
                if (label == 'stage7' and r['policy'] == 'evopress_whole_opt') or (label != 'stage7' and r['policy'] != 'evopress_whole_opt'):
                    continue
                key = r['task'], r['sample'], r['policy'], r['seed'], r['k']
                if key not in target:
                    continue
                new = target[key]
                assert key not in matched
                assert r['document_sha256'] == new['document_sha256']
                assert r.get('skips', r.get('skipped_layers')) == new['skipped_layers']
                assert r.get('router_tokens', r.get('router_observation_tokens')) == new['router_observation_tokens']
                error = max(abs(a-b) for a, b in zip(r['scores'], new['scores']))
                errors[label] = max(error, errors[label])
                assert error < 2e-4, (label, key, error)
                for field in ('prediction', 'prediction_norm', 'correct', 'correct_norm'):
                    assert r.get(field) == new.get(field), (label, key, field)
                matched.add(key)
                counts[label] += 1
    assert matched == set(target), (len(matched), len(target))
    write(out/'historical_replay.json', {'passed': True, 'rows': dict(counts), 'max_abs_score_error': errors,
        'tolerance': 2e-4, 'all_routes_observations_predictions_match': True})


def summarize(out, cfg, inputs):
    result = [r for rank in range(3) for r in read_lines(out/f'private_quality_{rank:03d}.jsonl')]
    assert len(result) == len(inputs)*len(cfg['policies'])
    registry = {(r['task'], r['sample']): r for r in inputs}
    policies = {(p['name'], p['seed'], p['k']): p for p in cfg['policies']}
    seen = set()
    group = defaultdict(list)
    for r in result:
        key = r['task'], r['sample'], r['policy'], r['seed'], r['k']
        assert key not in seen
        seen.add(key)
        p = policies[r['policy'], r['seed'], r['k']]
        record = registry[r['task'], r['sample']]
        assert r['router_observation_tokens'] == len(record['route_ids'])
        assert r['target_lengths'] == record['target_lengths']
        if p['kind'] == 'static':
            assert r['skipped_layers'] == sorted(p['skips'])
        else:
            assert set(p['prefix']).issubset(r['skipped_layers'])
        for field, value in score_fields(record, r['scores']).items():
            assert r[field] == value
        group[r['task'], r['policy'], r['k']].append(r)
    audit_replays(out, result, cfg)
    quality = summary_tables(result)
    means = []
    for key, items in sorted(group.items()):
        t, name, k = key
        q = [v for v in quality if (v['task'], v['policy'], v['k']) == key]
        row = {'task': t, 'policy': name, 'k': k, 'seeds': len(q), 'records_per_seed': q[0]['records']}
        if t == 'wikitext_conditional':
            nll = float(np.mean([v['nll_per_token'] for v in q]))
            row.update({'nll_per_token': nll, 'conditional_ppl_exp_mean_nll': math.exp(nll),
                'conditional_ppl_mean_per_seed': float(np.mean([v['conditional_ppl'] for v in q]))})
        else:
            row['accuracy'] = float(np.mean([v['accuracy'] for v in q]))
            if t != 'winogrande':
                row['accuracy_norm'] = float(np.mean([v['accuracy_norm'] for v in q]))
        means.append(row)
    comparisons = []
    for task in sorted({r['task'] for r in inputs}):
        identities = sorted(key for key in registry if key[0] == task)
        for k in KS:
            for dyn in ('pure', 'hybrid'):
                for baseline in ('evopress_whole_opt', 'static_joint_kl'):
                    per = {}
                    for name in (dyn, baseline):
                        per[name] = defaultdict(list)
                        for r in group[task, name, k]:
                            v = r['nll_sum']/r['scored_tokens'] if task == 'wikitext_conditional' else r['correct'] if task == 'winogrande' else r['correct_norm']
                            per[name][r['task'], r['sample']].append(v)
                        assert sorted(per[name]) == identities
                    differences = [float(np.mean(per[dyn][key])-np.mean(per[baseline][key])) for key in identities]
                    clusters = [registry[key]['document'] for key in identities] if task == 'wikitext_conditional' else None
                    comparisons.append({'task': task, 'k': k, 'dynamic': dyn, 'baseline': baseline,
                        'metric': 'nll_per_token_window_average' if task == 'wikitext_conditional' else 'accuracy',
                        'dynamic_minus_baseline': paired(differences, clusters=clusters, seed=300579)})
    write(out/'summary.json', {'quality': quality, 'means': means, 'paired_comparisons': comparisons,
        'quality_rows': len(result), 'input_records': len(inputs), 'probe': cfg['probe'],
        'statistics': 'Mean3 independent models, not ensemble; Wiki main PPL exp(mean seed token-weighted NLL) matches Stage7 table. Paired2000 bootstrap, unadjusted exploratory intervals.'})
    write(out/'verification.json', {'passed': True, 'all_quality_rows': len(result), 'unique_complete_coverage': True,
        'historical_router_routes_and_predictions_replayed': True, 'evopress_scores_replayed': True,
        'no_new_training_or_search': True, 'shared_exact_inputs': True, 'old_128_observation_preserved': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    setup()
    out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    tracked = preserve(ROOT, out)
    old, policies = make_policies()
    inputs = prepare(out, args.probe)
    sources = [Path(__file__), ROOT/'tools/prompt_depth_core_v1.py', ROOT/'tools/prompt_fixed_core_v1.py',
        ROOT/'tools/horizontal_quality_v3.py', ROOT/'tools/horizontal_data_v1.py']
    for name, digest in old['source_sha256'].items():
        assert sha256(ROOT/'tools'/name) == digest
    cfg = {'output': str(out), 'probe': args.probe, 'model': old['model'], 'budgets': KS,
        'policies': policies, 'fixed_prefix': old['fixed_prefix'],
        'source_sha256': {str(p): sha256(p) for p in sources},
        'stage7_protocol_sha256': sha256(OLD/'protocol.json'), 'evopress_protocol_sha256': sha256(EVO/'protocol.json'),
        'private_inputs_sha256': sha256(out/'private_inputs.jsonl'),
        'selection': 'Exact frozen Stage7 pure/hybrid perK checkpoints,3seeds. Same-K frozen Stage29 EvoPress masks,3seeds. No new model selection.',
        'observation': old['observation'], 'scoring': old['scoring'],
        'resources': '3 idle GPUs0/1/2,32CPU; workers10 each/controller2. GPU3 excluded because busy.',
        'limitations': ['Stage7 is the historical128-observation model, not full-context retraining, not codebook routing.',
            'Both old and new source task populations have been reviewed. Small subset cannot establish noninferiority or general superiority.',
            'EvoPress64generation OPT adaptation with variable-length TRAIN calibration; old router was trained on128-token prompt records. Not equal calibration/training cost.',
            'Only overlapping K12/16, no new K10 search. No recovery, timing or peak-memory claims.']}
    write(out/'protocol.json', cfg)
    stamp('stage30_begin', probe=args.probe, input_records=len(inputs), policies=len(policies))
    with ThreeWorkers(worker, cfg) as workers:
        workers.receive('done')
    summarize(out, cfg, inputs)
    verify_preserved(out, tracked)
    stamp('stage30_complete', probe=args.probe, inputs=len(inputs))


if __name__ == '__main__':
    main()
