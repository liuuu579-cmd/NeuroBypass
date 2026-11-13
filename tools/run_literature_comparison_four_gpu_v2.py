"""Stage29: frozen routers versus stronger literature whole-layer baselines.

Four independent OPT replicas each own one K. Exclusive output paths only.
EvoPress searches original training calibration; task labels never select masks,
router checkpoints, search budgets or seeds. A probe is an engineering pilot.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import load_choice
from evaluate_full_context_four_gpu_v1 import encode_full_question, logs, policies
from full_context_core_v1 import load_router, prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, read_lines, resources,
    setup_worker, thread_record, verify_preserved, write)
from headroom_ruler_core_v1 import paired
from horizontal_data_v1 import DATA, read_dataset, wiki_windows
from horizontal_quality_v3 import pin_threads, score_probe
from literature_evolution_core_v1 import OFFICIAL, Fitness, algorithm_checks, evolve, legal, log_probs
from prompt_fixed_core_v1 import fixed_route
from prompt_permutation_core_v1 import score_fields
from stage1_benchmark_v2 import sha256, stamp

KS = [12, 16, 20, 24]
ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT/'runs/stage13_full_context_20261110_02_training'
BANK_RUN = ROOT/'runs/stage20_codebook_refine_20261112_02_main'


def load(path):
    return json.loads(Path(path).read_text())


def prepare(out, model, probe):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=True)
    records = []
    counts = {}
    hashes = {}
    rng = np.random.default_rng(290579)
    for task in ('piqa', 'hellaswag', 'winogrande'):
        data = read_dataset(task, 'validation')
        indices = sorted(rng.choice(len(data), 8, replace=False).tolist()) if probe else range(len(data))
        records += [encode_full_question(task, data[i], i, tokenizer) for i in indices]
        counts[task] = len(indices)
        hashes[task] = sha256(DATA[task]/f'{task}-validation.arrow')
    wiki = wiki_windows(read_dataset('wikitext', 'test'), tokenizer)
    if probe:
        seen = set()
        subset = []
        for r in wiki:
            if r['document'] not in seen:
                seen.add(r['document']); subset.append(r)
            if len(subset) == 8:
                break
        wiki = subset
    records += wiki
    counts['wikitext_conditional'] = len(wiki)
    hashes['wikitext_conditional'] = sha256(DATA['wikitext']/'wikitext-test.arrow')
    records.sort(key=lambda r: (r['task'], r['sample']))
    with (out/'private_inputs.jsonl').open('x') as stream:
        for row in records:
            stream.write(json.dumps(row)+'\n')
    write(out/'data_manifest.json', {'task_records': counts, 'dataset_sha256': hashes,
        'private_inputs_sha256': sha256(out/'private_inputs.jsonl'),
        'observation': 'Complete actual answer-blind prompt; Wino preblank only; Wiki fixed128 context and up to384 future scored tokens.',
        'evaluation_scope': 'Previously reviewed benchmark populations, not a new untouched final test. Wiki conditional PPL is not standard rolling PPL.'})
    return records


def frozen_policies(probe):
    cfg, paths, all_policies = policies(TRAINING, KS, m_values=[6])
    result = [p for p in all_policies if p['seed'] is None or not probe or p['seed'] == 579]
    bank = load(BANK_RUN/'candidate_bank.json')
    control = load(BANK_RUN/'static_controls.json')
    result.append({'name': 'static_refined_bank_train800', 'kind': 'static', 'seed': None, 'k': 20,
        'skips': bank['routes'][control['best_route']]['skips'], 'selection': 'TRAIN800 mean KL only'})
    old = load(BANK_RUN/'protocol.json')['old_checkpoints']
    for seed in ([579] if probe else [579, 580, 581]):
        cp = old[str(seed)]
        if sha256(cp['path']) != cp['sha256']:
            raise ValueError('Frozen codebook checkpoint changed')
        result.append({'name': 'codebook8_train256_prompt', 'kind': 'codebook', 'seed': seed, 'k': 20,
            'checkpoint': cp, 'bank': bank})
    return cfg, paths, result


def gpu_checks(rank, model, data, cfg, fitness):
    k = KS[rank]
    sample = max(cfg['calibration_ids'], key=lambda s: len(fitness.raw[s]))
    raw = fitness.raw[sample]
    length = min(513, len(raw))
    ids = torch.tensor(raw[:length], device=f'cuda:{rank}', dtype=torch.long)[None]
    mask = cfg['global_path'][:k]
    captured = {'attention': [0]*40, 'ffn': [0]*40}
    handles = []
    for i, layer in enumerate(model.model.decoder.layers):
        for field, module in [('attention', layer.self_attn), ('ffn', layer.fc1)]:
            def capture(mod, args, i=i, field=field):
                captured[field][i] += 1
            handles.append(module.register_forward_pre_hook(capture))
    with torch.inference_mode():
        try:
            student = log_probs(model, ids, mask)
        finally:
            for h in handles:
                h.remove()
        for values in captured.values():
            assert values == [int(i not in mask) for i in range(40)]
        teacher = fitness.teachers[sample][:, :length-1].to(f'cuda:{rank}')
        manual = (teacher.exp()*(teacher-student)).sum(-1).double().mean().item()
        reference = F.kl_div(student, teacher, log_target=True, reduction='sum').double().item()/(length-1)
        actual = fitness.evaluate(mask, [(sample, length)])
        assert abs(manual-reference) < 1e-5 and abs(manual-actual) < 1e-7
        # Independent normal CausalLM and full LM-head shape for both sides.
        with fixed_route(model, mask):
            normal = F.log_softmax(model(ids[:, :-1], use_cache=False).logits.float(), -1)
        assert torch.equal(normal, student)
    return {'passed': True, 'k': k, 'tokens': length, 'actual_component_calls': captured,
        'manual_kl': manual, 'independent_kl_div': reference, 'fitness_kl': actual,
        'normal_causal_lm_max_logprob_error': float((normal-student).abs().max()), **thread_record()}


def score_worker(rank, model, cfg, selected, inputs, messages):
    out = Path(cfg['output'])
    k = KS[rank]
    local_policies = [p for p in cfg['policies'] if p['k'] in (0, k)] + [
        {'name': 'evopress_whole_opt', 'kind': 'static', 'k': k, 'seed': r['seed'], 'skips': r['skips']}
        for r in selected]
    routers = {}
    for p in local_policies:
        if p['kind'] == 'router':
            routers[p['checkpoint']['path']] = load_router(p['checkpoint']['path'], f'cuda:{rank}')
        elif p['kind'] == 'codebook':
            routers[p['checkpoint']['path']] = load_choice(p['checkpoint']['path'], f'cuda:{rank}')
    write(out/f'policies_{rank:03d}.json', local_policies)
    start = time.monotonic()
    with (out/f'private_quality_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
        for offset, r in enumerate(inputs):
            teacher = [logs(model, ids, n, []) for ids, n in zip(r['sequence_ids'], r['target_lengths'])]
            cache = {(): {'scores': [score for _, score in teacher], 'teacher_kl_sum': 0.,
                'target_tokens': sum(r['target_lengths'])}}
            observed = torch.tensor(r['route_ids'], device=f'cuda:{rank}', dtype=torch.long)[None]
            emb = prompt_embeddings(model, observed)
            assert emb.shape[1] == len(r['route_ids'])
            encoded = {}
            for p in local_policies:
                if p['kind'] == 'static':
                    mask = p['skips']
                else:
                    router = routers[p['checkpoint']['path']]
                    if p['kind'] == 'router':
                        path = p['checkpoint']['path']
                        if path not in encoded:
                            encoded[path] = router.encode(emb)
                        route = router.route_encoded(encoded[path], k, p['prefix'])[0].tolist()
                        assert route[:p['m']] == p['prefix']
                        mask = route
                    else:
                        index = int(router(emb).argmin(-1).item())
                        mask = p['bank']['routes'][index]['skips']
                key = legal(mask, p['k']) if p['k'] else ()
                if key not in cache:
                    scores = []
                    kl = 0.
                    for ids, n, (lp, _) in zip(r['sequence_ids'], r['target_lengths'], teacher):
                        lq, score = logs(model, ids, n, key)
                        kl += (lp.exp()*(lp-lq)).sum(-1).double().sum().item()
                        scores.append(score)
                    cache[key] = {'scores': scores, 'teacher_kl_sum': kl, 'target_tokens': sum(r['target_lengths'])}
                row = {'task': r['task'], 'sample': r['sample'], 'document': r.get('document'),
                    'document_sha256': r['document_sha256'], 'policy': p['name'], 'seed': p['seed'],
                    'k': p['k'], 'skips': list(key), 'router_tokens': len(r['route_ids']),
                    **cache[key], **score_fields(r, cache[key]['scores'])}
                stream.write(json.dumps(row)+'\n')
            stream.flush()
            if offset % 32 == 0 or offset+1 == len(inputs):
                messages.put({'kind': 'progress', 'phase': 'horizontal_quality', 'rank': rank, 'k': k,
                    'completed': offset+1, 'total': len(inputs), 'seconds': time.monotonic()-start})
    write(out/f'quality_receipt_{rank:03d}.json', {'k': k, 'seconds': time.monotonic()-start,
        'private_quality_sha256': sha256(out/f'private_quality_{rank:03d}.jsonl'), **thread_record()})


def worker(rank, cfg, inbox, messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
    out = Path(cfg['output'])
    data = Records(cfg['data_run'])
    fitness = Fitness(model, data, cfg['calibration_ids'])
    write(out/f'gpu_verification_{rank:03d}.json', gpu_checks(rank, model, data, cfg, fitness))
    messages.put({'kind': 'progress', 'phase': 'calibration_ready', 'rank': rank, 'k': KS[rank],
        'teacher_seconds': fitness.teacher_seconds})
    results = []
    with (out/f'search_trace_{rank:03d}.jsonl').open('x') as trace:
        def log(row):
            trace.write(json.dumps(row)+'\n'); trace.flush()
        for seed in cfg['search_seeds']:
            before = fitness.stats()
            def progress(row):
                messages.put({'kind': 'progress', 'phase': 'evolution', 'rank': rank,
                    'k': KS[rank], 'seed': seed, **{x: y for x, y in row.items() if x != 'skips'}})
            result = evolve(KS[rank], seed, fitness, cfg['search'], log, progress)
            result['fitness_stats_before'] = before
            result['fitness_stats_after'] = fitness.stats()
            results.append(result)
            write(out/f'evopress_k{KS[rank]}_s{seed}.json', result)
    write(out/f'search_receipt_{rank:03d}.json', {'k': KS[rank], 'statistics': fitness.stats(), **thread_record()})
    del fitness, data
    gc.collect(); torch.cuda.empty_cache()
    score_worker(rank, model, cfg, results, read_lines(out/'private_inputs.jsonl'), messages)
    messages.put({'kind': 'done', 'rank': rank})


def summarize(out, cfg, inputs):
    records = {(r['task'], r['sample']): r for r in inputs}
    grouped = defaultdict(list)
    verified = 0
    full_reference = {}
    for rank in range(4):
        plist = load(out/f'policies_{rank:03d}.json')
        by_policy = {(p['name'], p['seed'], p['k']): p for p in plist}
        seen = set()
        for row in read_lines(out/f'private_quality_{rank:03d}.jsonl'):
            key = row['task'], row['sample']
            r = records[key]
            pk = row['policy'], row['seed'], row['k']
            uid = key+pk
            assert uid not in seen
            seen.add(uid)
            p = by_policy[pk]
            legal(row['skips'], row['k'])
            if p['kind'] == 'static':
                assert row['skips'] == sorted(p['skips'])
            elif p['kind'] == 'codebook':
                assert row['skips'] in [sorted(x['skips']) for x in p['bank']['routes']]
            else:
                assert set(p['prefix']).issubset(row['skips'])
            assert row['router_tokens'] == len(r['route_ids'])
            assert row['target_tokens'] == sum(r['target_lengths'])
            assert math.isfinite(row['teacher_kl_sum']) and row['teacher_kl_sum'] >= -1e-3
            for name, value in score_fields(r, row['scores']).items():
                assert row[name] == value
            if row['policy'] == 'full':
                if key in full_reference:
                    assert max(abs(a-b) for a, b in zip(row['scores'], full_reference[key]['scores'])) < 1e-5
                    continue
                full_reference[key] = row
            grouped[(row['task'],) + pk].append(row)
            verified += 1
        assert len(seen) == len(inputs)*len(plist)
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = dict(zip(('task', 'policy', 'seed', 'k'), key))
        row['records'] = len(values)
        row['unique_masks'] = len({tuple(v['skips']) for v in values})
        row['teacher_kl_per_token'] = sum(v['teacher_kl_sum'] for v in values)/sum(v['target_tokens'] for v in values)
        if values[0].get('correct') is not None:
            row['accuracy'] = float(np.mean([v['correct'] for v in values]))
            if 'correct_norm' in values[0]:
                row['accuracy_norm'] = float(np.mean([v['correct_norm'] for v in values]))
        else:
            row['nll_per_token'] = sum(v['nll_sum'] for v in values)/sum(v['scored_tokens'] for v in values)
            row['conditional_ppl'] = math.exp(row['nll_per_token'])
        summaries.append(row)
    comparisons = []
    # Average model replicates on the same input before paired resampling.
    for task in sorted({r['task'] for r in inputs}):
        expected = sorted(key for key in records if key[0] == task)
        metric = 'correct_norm' if task in ('piqa', 'hellaswag') else 'correct' if task == 'winogrande' else 'nll_sum'
        for k in KS:
            dyn_names = ['m06'] + (['codebook8_train256_prompt'] if k == 20 else [])
            base_names = ['evopress_whole_opt', 'sleb_nll_opt', 'shortgpt_bi_opt', 'shortened_ppl_opt',
                'shortened_ppl_plus', 'static_joint_kl'] + (['static_refined_bank_train800'] if k == 20 else [])
            for dyn in dyn_names:
                for base in base_names:
                    values = {}
                    for name in (dyn, base):
                        per_input = defaultdict(list)
                        for (t, p, seed, budget), rows in grouped.items():
                            if (t, p, budget) != (task, name, k):
                                continue
                            for r in rows:
                                per_input[t, r['sample']].append(r)
                        assert sorted(per_input) == expected
                        values[name] = [float(np.mean([v[metric]/v['scored_tokens'] if metric == 'nll_sum' else v[metric]
                            for v in per_input[key]])) for key in expected]
                    delta = np.asarray(values[dyn])-np.asarray(values[base])
                    clusters = [records[key].get('document', key[1]) for key in expected] if metric == 'nll_sum' else None
                    comparisons.append({'task': task, 'k': k, 'dynamic': dyn, 'baseline': base,
                        'metric': 'document_window_mean_nll' if metric == 'nll_sum' else metric,
                        'difference_dynamic_minus_baseline': paired(delta, clusters=clusters, seed=290579),
                        'direction': 'lower_is_better' if metric == 'nll_sum' else 'higher_is_better'})
    write(out/'summary.json', {'quality': summaries, 'paired_comparisons': comparisons,
        'verified_unique_rows': verified, 'input_records': len(inputs), 'probe': cfg['probe'],
        'statistics': 'Mean across independent model/search seeds per input, not an ensemble. Paired2000 bootstrap; Wiki clusters articles. Unadjusted intervals.'})
    write(out/'verification.json', {'passed': True, 'exact_row_coverage': True, 'row_scores_recomputed': True,
        'cross_gpu_full_model_agreement': True, 'full_router_observation': True,
        'no_downstream_policy_selection': True, 'training_source_and_checkpoint_hashes_checked': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    resources()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    pin_threads(resources()[40:42])
    tracked = preserve(ROOT, out)
    write(out/'algorithm_verification.json', algorithm_checks())
    trained, paths, selected = frozen_policies(args.probe)
    calibration = trained['calibration_ids']
    if args.probe:
        data = Records(trained['data_run'])
        # Representative variable lengths; TRAIN only, no quality based choice.
        calibration = sorted(set(min(calibration, key=lambda s: (abs(data.rows[s]['prompt_length']-n), s))
            for n in (64, 128, 256, 512, 1024, 2016)))
    data = Records(trained['data_run'])
    available = sum(len(data[s][0]) for s in calibration)
    final_budget = min(4096 if args.probe else 32768, available)
    inputs = prepare(out, trained['model'], args.probe)
    search = {'initially_generated': 8 if args.probe else 64, 'offspring': 4 if args.probe else 32,
        'generations': 2 if args.probe else 64, 'initial_tokens': min(1024 if args.probe else 2048, available),
        'stage_tokens': [min(1024 if args.probe else 2048, available), final_budget],
        'population_size': 1, 'survivors': [2, 1], 'maximum_mutations': 3,
        'fitness': 'Token weighted full shifted-sequence teacher KL, as upstream, not continuation-only KL.'}
    sources = [Path(__file__), ROOT/'tools/literature_evolution_core_v1.py',
        ROOT/'tools/evaluate_full_context_four_gpu_v1.py', ROOT/'tools/full_context_core_v1.py',
        ROOT/'tools/prompt_fixed_core_v1.py', ROOT/'tools/adaptive_choice_core_v3.py']
    cfg = {'output': str(out), 'probe': args.probe, 'model': trained['model'], 'data_run': trained['data_run'],
        'budgets': KS, 'calibration_ids': calibration, 'calibration_input_tokens': available,
        'search_seeds': [579] if args.probe else [579, 580, 581], 'search': search,
        'policies': selected, 'global_path': paths['static_joint_kl'], 'official_evopress': OFFICIAL,
        'source_sha256': {str(p): sha256(p) for p in sources},
        'data_manifest_sha256': sha256(Path(trained['data_run'])/'data_manifest.json'),
        'static_paths_sha256': sha256(TRAINING/'static_paths.json'),
        'bank_sha256': sha256(BANK_RUN/'candidate_bank.json'),
        'policy_freeze': 'm6 established Stage13 checkpoints at each K; K20 bank8 established Stage19 seeds; no new router fitting or downstream selection.',
        'common_setting': 'OPT13B FP16 SDPA; exact whole-layer K; protect original layers0/1; complete prompt; one fixed shared route for all options; frozen backbone, no recovery.',
        'adaptation_limits': ['Same128 TRAIN calibration identities as existing baselines; dataset and variable lengths differ from original paper.',
            'EvoPress whole-block option with native all-next-token KL. Our static KL uses32 continuation targets; report objective difference.',
            'Main64 generations is a declared bounded search, below paper script roughly224-267 generations at these K. Not an official table reproduction.',
            'Static refined bank uses additional TRAIN search and data. Do not claim equal offline search cost.',
            'Existing SLEB/ShortGPT/Shortened are frozen OPT adaptations; recovery recipes are outside this table.',
            'BlockPruner uses separate attention/FFN units and requires a separate component-budget comparison; not relabeled as whole-layer SLEB.'],
        'main_expansion_rule': 'Pilot engineering verification only; full run configuration is fixed here, not selected by pilot accuracy.',
        'test_documents_used_for_search': 0, 'resources': '4GPU0-3 /42CPU,10 cores per worker+2 controller'}
    write(out/'protocol.json', cfg)
    stamp('stage29_begin', probe=args.probe, inputs=len(inputs), search=search)
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'done')
    summarize(out, cfg, inputs)
    verify_preserved(out, tracked)
    stamp('stage29_complete', probe=args.probe)


if __name__ == '__main__':
    main()
