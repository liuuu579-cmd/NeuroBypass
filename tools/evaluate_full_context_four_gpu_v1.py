"""New checkpoints: full-prompt quality and frozen within-task crossover.

Development: all variable-length C4 development documents and fixed task
subsets, teacher KL and three permutations. Horizontal: complete old task
populations, regenerated answer-blind prompt observations, all m arms/seeds.
"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from full_context_core_v1 import OBSERVATION_VERSION, load_router, prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, read_lines, resources, setup_worker,
    thread_record, verify_preserved, write)
from horizontal_data_v1 import DATA, encode_pair, question, read_dataset, wiki_windows
from horizontal_quality_v3 import score_probe
from prompt_fixed_core_v1 import fixed_route
from prompt_permutation_core_v1 import assignments, score_fields
from stage1_benchmark_v2 import sha256, stamp


def encode_full_question(task, row, index, tokenizer):
    prompt, pairs, choices, gold = question(task, row)
    observed = tokenizer.encode(prompt, add_special_tokens=True) or [tokenizer.bos_token_id]
    sequences = [encode_pair(tokenizer, context, continuation) for context, continuation in pairs]
    # Existing benchmark inputs fit, so retain their exact likelihood protocol.
    # A future long-option adapter needs a common retained context definition.
    if len(observed) > 2048 or any(removed for _, _, removed in sequences):
        raise ValueError('Benchmark needs explicit common-context truncation')
    return {'task': task, 'sample': index, 'route_ids': observed,
        'sequence_ids': [s for s, _, _ in sequences], 'target_lengths': [n for _, n, _ in sequences],
        'truncated_context_tokens': [n for _, _, n in sequences],
        'normalization_lengths': [len(text) for text in choices], 'gold': gold,
        'document_sha256': hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}


def policies(training, budgets, m_values=None):
    config = json.loads((training / 'protocol.json').read_text())
    if config['observation_version'] != OBSERVATION_VERSION:
        raise ValueError('Wrong training observation contract')
    paths = json.loads((training / 'static_paths.json').read_text())
    result = [{'name': 'full', 'kind': 'static', 'k': 0, 'seed': None, 'skips': []}]
    for k in budgets:
        for name in ['static_joint_kl', 'sleb_nll_opt', 'shortgpt_bi_opt',
                'shortened_ppl_opt', 'shortened_ppl_plus', 'random_static_seed579']:
            result.append({'name': name, 'kind': 'static', 'k': k, 'seed': None, 'skips': paths[name][:k]})
        for m in config['arms'] if m_values is None else m_values:
            if k < m: continue
            for seed in config['seeds']:
                cp = json.loads((training / f'm{m:02d}_seed{seed}_checkpoints.json').read_text())[str(k)]
                if sha256(cp['path']) != cp['sha256']: raise ValueError('Checkpoint digest mismatch')
                result.append({'name': f'm{m:02d}', 'm': m, 'kind': 'router', 'seed': seed, 'k': k,
                    'checkpoint': cp, 'prefix': paths['static_joint_kl'][:m]})
    return config, paths, result


def prepare(output, training, mode, probe):
    from transformers import AutoTokenizer
    cfg = json.loads((training / 'protocol.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], use_fast=False, local_files_only=True)
    inputs, provenance = [], {}
    rng = np.random.default_rng(579)
    for task in ['piqa', 'hellaswag', 'winogrande']:
        dataset = read_dataset(task, 'validation')
        selected = range(len(dataset)) if mode == 'horizontal' else sorted(rng.choice(len(dataset), 4 if probe else 64, replace=False).tolist())
        inputs += [encode_full_question(task, dataset[i], i, tokenizer) for i in selected]
        provenance[task] = sha256(DATA[task] / f'{task}-validation.arrow')
    windows = wiki_windows(read_dataset('wikitext', 'test'), tokenizer)
    if mode != 'horizontal':
        if probe:
            seen, subset = set(), []
            for row in windows:
                if row['document'] not in seen:
                    subset.append(row); seen.add(row['document'])
                if len(subset) == 4: break
            windows = subset
        else:
            windows = [windows[i] for i in sorted(rng.choice(len(windows), 64, replace=False).tolist())]
    inputs += windows
    provenance['wikitext_conditional'] = sha256(DATA['wikitext'] / 'wikitext-test.arrow')
    if mode == 'development':
        data = Records(cfg['data_run'])
        samples = cfg['validation_ids']
        if probe: samples = samples[:4]
        for sample in samples:
            ids, p = data[sample]
            inputs.append({'task': 'c4_development', 'sample': sample, 'document': sample,
                'document_sha256': data.rows[sample]['record_sha256'], 'route_ids': ids[:p].tolist(),
                'sequence_ids': [ids.tolist()], 'target_lengths': [32], 'normalization_lengths': [32],
                'truncated_context_tokens': [0], 'gold': None})
        provenance['c4_manifest'] = sha256(Path(cfg['data_run']) / 'data_manifest.json')
    inputs.sort(key=lambda r: (r['task'], r['sample']))
    with (output / 'private_inputs.jsonl').open('x') as stream:
        for row in inputs: stream.write(json.dumps(row)+'\n')
    write(output / 'data_manifest.json', {'provenance_sha256': provenance,
        'private_input_sha256': sha256(output / 'private_inputs.jsonl'),
        'task_records': {t: sum(r['task'] == t for r in inputs) for t in sorted({r['task'] for r in inputs})},
        'router_tokens': {t: {'min': min(len(r['route_ids']) for r in inputs if r['task'] == t),
            'max': max(len(r['route_ids']) for r in inputs if r['task'] == t)} for t in sorted({r['task'] for r in inputs})},
        'answer_blind': 'MC common question prefix, Wino preblank only; Wiki existing128-token context, NOT384 future targets; C4 entire variable-length prompt.',
        'test_policy': 'C4 reserved test excluded. MC/Wiki are previously reviewed benchmark populations, not a new untouched final set.'})
    return inputs


def logs(model, ids, n, mask):
    x = torch.tensor(ids, device=next(model.parameters()).device, dtype=torch.long)[None]
    with fixed_route(model, mask):
        hidden = model.model(x[:, :-1], attention_mask=torch.ones_like(x[:, :-1]), use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)[:, -n:].float()
    lp = F.log_softmax(logits, -1)
    score = lp.gather(-1, x[:, -n:, None]).double().sum().item()
    return lp, score


def worker(rank, config, inbox, messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
    routers = {p['checkpoint']['path']: None for p in config['policies'] if p['kind'] == 'router'}
    for path in routers: routers[path] = load_router(path, f'cuda:{rank}')
    out = Path(config['output']); inputs = read_lines(out / 'private_inputs.jsonl')
    local, registry = inputs[rank::4], {}
    with torch.inference_mode():
        # Full registry on every rank; donors may belong to another shard.
        for record in inputs if config['mode'] == 'development' else local:
            ids = torch.tensor(record['route_ids'], device=f'cuda:{rank}', dtype=torch.long)[None]
            emb = prompt_embeddings(model, ids); encoded = {}
            for policy in config['policies']:
                if policy['kind'] != 'router': continue
                path = policy['checkpoint']['path']
                if path not in encoded: encoded[path] = routers[path].encode(emb)
                ordered = routers[path].route_encoded(encoded[path], policy['k'], policy['prefix'])[0].tolist()
                if ordered[:policy['m']] != policy['prefix'] or len(set(ordered)) != policy['k'] or min(ordered) < 2:
                    raise ValueError('Inference route contract failed')
                registry[record['task'], record['sample'], policy['name'], policy['seed'], policy['k']] = sorted(ordered)
    write(out / f'routes_{rank:03d}.json', [{'task': t, 'sample': s, 'policy': name,
        'seed': seed, 'k': k, 'skipped_layers': route} for (t, s, name, seed, k), route in sorted(registry.items())])
    donors = {(r['task'], r['sample'], r['permutation_seed']): r['donor_sample']
        for r in json.loads((out / 'assignments.json').read_text())}
    probes = []
    with torch.inference_mode():
        for task in sorted({r['task'] for r in local}):
            record = next(r for r in local if r['task'] == task)
            for mask in [[], config['global_path'][:10]]:
                probes.append({'task': task, **score_probe(model, record, mask)})
    write(out / f'worker_probe_{rank:03d}.json', {**thread_record(), 'likelihood_probes': probes})
    with (out / f'quality_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
        for offset, record in enumerate(local):
            teacher = [logs(model, ids, n, []) for ids, n in zip(record['sequence_ids'], record['target_lengths'])]
            cached = {}
            for policy in config['policies']:
                base = (record['task'], record['sample'], policy['name'], policy['seed'], policy['k'])
                route = policy['skips'] if policy['kind'] == 'static' else registry[base]
                conditions = [('matched', record['sample'], route)]
                if config['mode'] == 'development' and policy['kind'] == 'router':
                    for seed in [1701, 1702, 1703]:
                        donor = donors[record['task'], record['sample'], seed]
                        conditions.append((f'shuffled_{seed}', donor,
                            registry[record['task'], donor, policy['name'], policy['seed'], policy['k']]))
                for condition, donor, mask in conditions:
                    key = tuple(sorted(mask))
                    if key not in cached:
                        scores, kl = [], 0.
                        for ids, n, (lp, _) in zip(record['sequence_ids'], record['target_lengths'], teacher):
                            lq, score = logs(model, ids, n, mask)
                            kl += (lp.exp()*(lp-lq)).sum(-1).double().sum().item(); scores.append(score)
                        cached[key] = {'scores': scores, 'teacher_kl_sum': kl, 'target_tokens': sum(record['target_lengths'])}
                    row = {'task': record['task'], 'sample': record['sample'], 'document': record.get('document'),
                        'document_sha256': record['document_sha256'], 'policy': policy['name'], 'seed': policy['seed'],
                        'm': policy.get('m'), 'k': policy['k'], 'condition': condition, 'donor_sample': donor,
                        'skipped_layers': list(key), 'router_observation_tokens': len(record['route_ids']),
                        **cached[key], **score_fields(record, cached[key]['scores'])}
                    stream.write(json.dumps(row)+'\n')
            stream.flush()
            if offset % 8 == 0 or offset+1 == len(local):
                messages.put({'kind': 'progress', 'phase': config['mode'], 'rank': rank,
                    'completed': offset+1, 'total': len(local), 'task': record['task']})
    messages.put({'kind': 'done', 'rank': rank})


def audit_and_summary(out, config, inputs):
    records = {(r['task'], r['sample']): r for r in inputs}
    policy_by = {(p['name'], p['seed'], p['k']): p for p in config['policies']}
    registry = {}
    for rank in range(4):
        routes = json.loads((out / f'routes_{rank:03d}.json').read_text())
        current = {(r['task'], r['sample'], r['policy'], r['seed'], r['k']): r['skipped_layers'] for r in routes}
        if len(current) != len(routes): raise ValueError('Duplicate route registry')
        if config['mode'] == 'development' and rank > 0 and current != registry:
            raise ValueError('Workers disagree on full route registry')
        registry.update(current)
    donors = {(r['task'], r['sample'], f"shuffled_{r['permutation_seed']}"): r['donor_sample']
        for r in json.loads((out / 'assignments.json').read_text())}
    rows = [r for p in out.glob('quality_*.jsonl') for r in read_lines(p)]
    seen, groups = set(), defaultdict(list)
    for row in rows:
        uid = tuple(row[k] for k in ['task', 'sample', 'policy', 'seed', 'k', 'condition'])
        if uid in seen: raise ValueError('Duplicate quality row')
        seen.add(uid)
        record = records[row['task'], row['sample']]
        policy = policy_by[row['policy'], row['seed'], row['k']]
        donor = row['sample'] if row['condition'] == 'matched' else donors[row['task'], row['sample'], row['condition']]
        expected = policy['skips'] if policy['kind'] == 'static' else registry[row['task'], donor, row['policy'], row['seed'], row['k']]
        assert row['donor_sample'] == donor and row['skipped_layers'] == sorted(expected)
        assert len(expected) == len(set(expected)) == row['k']
        assert row['router_observation_tokens'] == len(record['route_ids'])
        assert row['target_tokens'] == sum(record['target_lengths'])
        assert math.isfinite(row['teacher_kl_sum']) and row['teacher_kl_sum'] > -1e-4
        assert all(math.isfinite(v) for v in row['scores'])
        for key, value in score_fields(record, row['scores']).items(): assert row[key] == value
        groups[row['task'], row['policy'], row['seed'], row['k'], row['condition']].append(row)
    expected = len(inputs)*sum(4 if config['mode'] == 'development' and p['kind'] == 'router' else 1 for p in config['policies'])
    if len(rows) != expected: raise ValueError('Incomplete quality coverage')
    summaries = []
    for key, values in sorted(groups.items(), key=lambda v: str(v[0])):
        if len(values) != sum(r['task'] == key[0] for r in inputs): raise ValueError('Incomplete task/policy group')
        row = dict(zip(['task', 'policy', 'seed', 'k', 'condition'], key)); row['records'] = len(values)
        row['teacher_kl_per_token'] = sum(v['teacher_kl_sum'] for v in values)/sum(v['target_tokens'] for v in values)
        row['unique_masks'] = len({tuple(v['skipped_layers']) for v in values})
        if values[0].get('correct') is not None:
            row['accuracy'] = float(np.mean([v['correct'] for v in values]))
            if 'correct_norm' in values[0]: row['accuracy_norm'] = float(np.mean([v['correct_norm'] for v in values]))
        else:
            row['nll_per_token'] = sum(v['nll_sum'] for v in values)/sum(v['scored_tokens'] for v in values)
            row['conditional_ppl'] = math.exp(row['nll_per_token'])
        summaries.append(row)
    write(out / 'summary.json', {'quality': summaries, 'quality_rows': len(rows), 'input_records': len(inputs)})
    write(out / 'verification.json', {'passed': True, 'rows': len(rows), 'route_registry_entries': len(registry),
        'full_observation_lengths_checked': True, 'route_prefixes_checked': True, 'gold_scores_replayed': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--mode', choices=['development', 'horizontal'], required=True)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args(); resources()
    root, out, training = Path(__file__).resolve().parents[1], Path(os.environ['NEUROBYPASS_RUN_DIR']), args.training_run.resolve()
    tracked = preserve(root, out)
    trained = json.loads((training / 'protocol.json').read_text())
    budgets = [10, 12] if args.probe else ([6, 8, 10, 12, 16, 20, 24] if args.mode == 'development' else [10, 12, 16])
    _, paths, selected = policies(training, budgets)
    inputs = prepare(out, training, args.mode, args.probe)
    write(out / 'assignments.json', assignments(inputs) if args.mode == 'development' else [])
    config = {'output': str(out), 'training_run': str(training), 'model': trained['model'], 'mode': args.mode,
        'probe': args.probe, 'policies': selected, 'global_path': paths['static_joint_kl'],
        'observation_version': OBSERVATION_VERSION, 'training_protocol_sha256': sha256(training / 'protocol.json'),
        'private_input_sha256': sha256(out / 'private_inputs.jsonl'),
        'source_sha256': {str(q): sha256(q) for q in [Path(__file__), root/'tools/full_context_core_v1.py']},
        'development_limits': 'C4 prompts vary in length; task-wise permutations can cross lengths. Report length strata separately; no causal content-only attribution.',
        'baseline_scope': 'Frozen OPT adaptations without recovery; not full official recipe replications.'}
    write(out / 'protocol.json', config)
    with Workers(worker, config) as workers: workers.receive(4, 'done')
    audit_and_summary(out, config, inputs)
    verify_preserved(out, tracked)
    stamp('full_context_evaluation_complete', mode=args.mode, inputs=len(inputs))


if __name__ == '__main__': main()
