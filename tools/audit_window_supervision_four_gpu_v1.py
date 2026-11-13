"""Independent Stage28 provenance, target, selection, arithmetic and GPU replay."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gc
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load_choice
from deep_swap_core_v1 import StaticSwapRouter
from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers, read_lines, resources, setup_worker, thread_record, write
from horizontal_quality_v3 import pin_threads
from prompt_fixed_core_v1 import fixed_route
from stage1_benchmark_v2 import sha256


ARMS = ('first32_embedding', 'mean128_embedding', 'first32_constant', 'mean128_constant')


def load(path):
    return json.loads(Path(path).read_text())


def digest(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i4').tobytes()).hexdigest()


def get_tokens(out, cfg):
    values = {}
    for directory in cfg['cache_runs'] + [str(out)]:
        directory = Path(directory)
        pattern = 'private_new_token_metrics_*.pth' if list(directory.glob('private_new_token_metrics_*.pth')) else 'private_token_metrics_*.pth'
        for path in sorted(directory.glob(pattern)):
            for i, value in torch.load(path, weights_only=True).items():
                if i in values:
                    assert all(torch.equal(values[i][m], value[m]) for m in ('kl', 'nll'))
                values[i] = value
    return {i: values[i] for i in cfg['ids']}


def worker(rank, job, inbox, messages):
    from transformers import AutoModelForCausalLM
    setup_worker(rank); device = f'cuda:{rank}'; out = Path(job['main_run']); audit_out = Path(job['output'])
    cfg = load(out / 'protocol.json'); inputs = {r['sample']: r for r in read_lines(cfg['input_file'])}
    sample = max(cfg['ids'][rank::4], key=lambda i: inputs[i]['prompt_length'])
    row = inputs[sample]; p = row['prompt_length']; ids = torch.tensor(row['ids'], dtype=torch.long, device=device)[None]
    expected = get_tokens(out, cfg)[sample]
    bank = load(Path(cfg['old_run']) / 'candidate_bank.json')
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    def probs(skips):
        with fixed_route(model, skips):
            hidden = model.model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False).last_hidden_state
            return model.lm_head(hidden[:, p-1:p+127]).float().log_softmax(-1)
    maximum = 0.
    with torch.inference_mode():
        teacher = probs([])
        for route in bank['routes']:
            student = probs(route['skips'])
            kl = (teacher.exp() * (teacher - student)).sum(-1)[0].cpu()
            nll = -student.gather(-1, ids[:, p:, None]).squeeze(-1)[0].cpu()
            maximum = max(maximum, float((kl - expected['kl'][:, route['index']]).abs().max()),
                float((nll - expected['nll'][:, route['index']]).abs().max()))
            assert torch.equal(kl, expected['kl'][:, route['index']])
            assert torch.equal(nll, expected['nll'][:, route['index']])
    # Verify cached observations directly against original OPT, full prompt only.
    embeddings = torch.load(Path(cfg['old_run']) / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    with torch.inference_mode():
        observed = model.model.decoder(input_ids=ids[:, :p], attention_mask=torch.ones_like(ids[:, :p]),
            use_cache=False, output_hidden_states=True).hidden_states[0].cpu()
    assert torch.equal(observed, embeddings[sample])
    del model, teacher, student, observed; gc.collect(); torch.cuda.empty_cache()
    fold = cfg['folds'][rank]; replay_ids = fold['selection_ids'][:1] + fold['outer_ids'][:2]
    score_error = 0.; replay_count = 0
    for seed in cfg['seeds']:
        for arm in ARMS:
            name = f'{arm}_f{rank}_s{seed}'; receipt = load(out / f'{name}_fit.json')
            router = load_choice(receipt['best']['path'], device)
            saved = torch.load(out / f'private_{name}_scores.pth', weights_only=True)
            with torch.inference_mode():
                for sample_id in replay_ids:
                    prediction = router(embeddings[sample_id].to(device)).cpu()[0]
                    reference = saved['scores'][saved['ids'].index(sample_id)]
                    error = float((prediction - reference).abs().max()); score_error = max(score_error, error)
                    assert error < 1e-6; replay_count += 1
            del router
    write(audit_out / f'worker_{rank:03d}.json', {'passed': True, 'joint_mask_replays': 8,
        'token_metric_maximum_error': maximum, 'router_score_replays': replay_count,
        'router_score_maximum_error': score_error, 'full_prompt_embedding_exact': True, **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def reconstruct_folds(ids, rows):
    # Build components by graph traversal rather than the training union-find.
    by_key = {}
    for i in ids:
        for field in ('record_sha256', 'record_lead_sha256'):
            by_key.setdefault((field, rows[i][field]), []).append(i)
    groups = []; visited = set()
    for i in ids:
        if i in visited:
            continue
        pending = [i]; component = set()
        while pending:
            j = pending.pop()
            if j in component:
                continue
            component.add(j)
            for field in ('record_sha256', 'record_lead_sha256'):
                pending += [v for v in by_key[field, rows[j][field]] if v not in component]
        visited |= component; groups.append([v for v in ids if v in component])
    np.random.default_rng(280579).shuffle(groups); groups.sort(key=lambda g: -len(g))
    buckets = [[] for _ in range(4)]
    for g in groups:
        minimum = min(len(b) for b in buckets); index = next(j for j, b in enumerate(buckets) if len(b) == minimum)
        buckets[index] += g
    result = []
    for f, bucket in enumerate(buckets):
        outer = sorted(bucket); remaining = [g for g in groups if not set(g) & set(outer)]
        np.random.default_rng(280580+f).shuffle(remaining)
        selection = []; fit = []
        for g in remaining:
            if len(selection) < 35:
                selection += g
            else:
                fit += g
        result.append({'fold': f, 'fit_ids': sorted(fit), 'selection_ids': sorted(selection), 'outer_ids': outer})
    return result


def select_gate(scores, costs, baseline, scale):
    choices = [(float(costs[:, baseline].mean()), 0, 0, np.full(len(scores), baseline), True, 0.)]
    for serial, margin in enumerate((0., .05, .1, .2, .4), 1):
        chosen = scores.argmin(1)
        routes = np.where(scores[:, baseline] - scores[np.arange(len(scores)), chosen] > margin * scale, chosen, baseline)
        value = float(np.mean([costs[i, r] for i, r in enumerate(routes)]))
        choices.append((value, int(np.sum(routes != baseline)), serial, routes, False, margin))
    return min(choices, key=lambda x: x[:3])


def check_estimate(delta, row):
    assert abs(float(delta.mean()) - row['difference']) < 1e-12
    draws = np.random.default_rng(280579).integers(0, len(delta), (2000, len(delta)))
    assert np.allclose(np.quantile(delta[draws].mean(1), [.025, .975]), row['ci95'], atol=1e-12, rtol=0)


def audit(root, out, audit_out):
    from transformers import AutoTokenizer
    cfg = load(out / 'protocol.json'); assert not cfg['probe'] and len(cfg['ids']) == 351
    assert cfg['epochs'] == 24 and cfg['seeds'] == [579, 580, 581] and cfg['arms'] == list(ARMS)
    for name, expected in cfg['source_sha256'].items():
        assert sha256(root / 'tools' / name) == expected
    for path, expected in cfg['evidence_sha256'].items():
        assert sha256(path) == expected
    preserved = load(out / 'preservation_before.json')
    for path, expected in preserved.items():
        assert sha256(path) == expected, path
    previous = load(Path(cfg['old_run']) / 'protocol.json'); data = Records(cfg['data_run'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], use_fast=False, local_files_only=True)
    eligible = {}; excluded = Counter()
    with Path(data.manifest['corpus']).open('rb') as source:
        for i in previous['train_ids']:
            row = data.rows[i]; p = row['prompt_length']
            if p > 1920:
                excluded['context_limit'] += 1; continue
            source.seek(row['source_start_byte']); raw = source.read(row['source_end_byte']-row['source_start_byte'])
            assert hashlib.sha256(raw).hexdigest() == row['source_raw_sha256']
            assert hashlib.sha256(' '.join(raw.decode().strip().split()).encode()).hexdigest() == row['record_sha256']
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(raw.decode().strip(), add_special_tokens=False, truncation=True, max_length=2047)
            original, _ = data[i]; assert tokens[:p+32] == original.tolist()
            if len(tokens) < p+128:
                excluded['insufficient_same_document_continuation'] += 1; continue
            eligible[i] = tokens[:p+128]
    sampling = load(out / 'sampling.json'); assert list(eligible) == sampling['eligible_ids'] == cfg['ids']
    assert len(eligible) == 351 and sampling['excluded'] == dict(excluded)
    assert not set(eligible) & set(previous['validation_ids'])
    assert sha256(cfg['input_file']) == cfg['input_file_sha256']
    inputs = read_lines(cfg['input_file']); assert [r['sample'] for r in inputs] == cfg['ids']
    for row in inputs:
        i = row['sample']; p = row['prompt_length']; assert row['ids'] == eligible[i] and p == data.rows[i]['prompt_length']
        assert digest(row['ids']) == row['extended_sequence_sha256'] and digest(row['ids'][:p]) == row['prompt_sha256']
        assert digest(row['ids'][:p+32]) == row['original_sequence_sha256'] == data.rows[i]['sequence_sha256']
    assert reconstruct_folds(cfg['ids'], data.rows) == cfg['folds']
    token = get_tokens(out, cfg); labels = {}; allocation = load(out / 'allocation.json')
    assert allocation['gpu_indices'] == [0, 1, 2, 3] and allocation['cpu_count'] == 42
    metric_values = 0
    for rank in range(4):
        receipt = load(out / f'labels_worker_{rank:03d}.json')
        assert sha256(out / f'private_labels_{rank:03d}.jsonl') == receipt['labels_sha256']
        assert sha256(out / f'private_new_token_metrics_{rank:03d}.pth') == receipt['tokens_sha256']
        assert receipt['maximum_cache_replay_error'] == 0
        rows = read_lines(out / f'private_labels_{rank:03d}.jsonl')
        assert [r['sample'] for r in rows] == cfg['ids'][rank::4]
        for row in rows:
            i = row['sample']; assert i not in labels; labels[i] = row
            assert digest(eligible[i]) == row['extended_sequence_sha256']
            for metric in ('kl', 'nll'):
                values = token[i][metric]; assert values.shape == (128, 8) and torch.isfinite(values).all()
                assert values.reshape(4, 32, 8).mean(1).double().tolist() == row[metric]
                assert np.max(np.abs(values.double().numpy().reshape(4, 32, 8).mean(1) - row[metric])) < 3e-6
                metric_values += values.numel()
        for directory in (out, audit_out):
            evidence = load(directory / f'worker_{rank:03d}.json')
            assert evidence['cpu_ids'] == allocation['cpu_ids'][rank*10:(rank+1)*10] and evidence['threads'] == 10
            assert all(set(v) <= set(evidence['cpu_ids']) for v in evidence['thread_cpu_ids'].values())
    private = read_lines(out / 'private_quality_rows.jsonl')
    index = {(r['fold'], r['sample'], r['policy']): r for r in private}; assert len(index) == len(private)
    choices = {(r['fold'], r['seed'], r['arm']): r for r in load(out / 'policy_choices.json')}
    parameter_checks = []; evaluated = 0
    for fold in cfg['folds']:
        f = fold['fold']; ids = fold['fit_ids']+fold['selection_ids']+fold['outer_ids']; nf = len(fold['fit_ids']); ns = len(fold['selection_ids'])
        costs = np.asarray([labels[i]['kl'] for i in ids]); mean = costs.mean(1)
        base = int(mean[:nf].mean(0).argmin()); relative = mean[:nf] - mean[:nf, :1]
        gate_scale = max(float(relative[:, 1:].std(ddof=1)), .03)
        for seed in cfg['seeds']:
            parameter_digests = []
            for arm in ARMS:
                target_costs = costs[:nf, 0] if arm.startswith('first32') else mean[:nf]
                target = torch.tensor(target_costs, dtype=torch.float32); target = target - target[:, :1]
                scale = target[:, 1:].std().clamp_min(.03)
                torch.manual_seed(seed); initial = StaticSwapRouter(8, width=128, observe=arm.endswith('_embedding'))
                parameter_digests.append(hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in initial.parameters())).hexdigest())
                initial.prior.copy_(target.mean(0)); initial.scale.copy_(scale)
                state_digest = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in initial.state_dict().values())).hexdigest()
                name = f'{arm}_f{f}_s{seed}'; receipt = load(out / f'{name}_fit.json')
                assert receipt['initial_sha256'] == state_digest
                assert receipt['prior'] == target.mean(0).tolist() and receipt['scale'] == float(scale)
                assert receipt['updates'] == 24 * math.ceil(nf / 16)
                history = load(out / f'{name}_history.json'); assert [r['epoch'] for r in history] == list(range(25))
                selected = min((r for r in history if 'selection_kl' in r), key=lambda r: (r['selection_kl'], r['epoch']))
                assert selected['epoch'] == receipt['best']['epoch'] and selected['selection_kl'] == receipt['best']['selection_kl']
                assert sha256(receipt['best']['path']) == receipt['best']['sha256']
                checkpoint = torch.load(receipt['best']['path'], weights_only=True, map_location='cpu')
                assert torch.equal(checkpoint['state_dict']['prior'], target.mean(0)) and torch.equal(checkpoint['state_dict']['scale'], scale)
                score_file = out / f'private_{name}_scores.pth'; assert sha256(score_file) == receipt['score_sha256']
                saved = torch.load(score_file, weights_only=True); assert saved['ids'] == ids
                scores = saved['scores'].numpy().astype(np.float64); selected_routes = scores[nf:nf+ns].argmin(1)
                selected_kl = float(mean[nf:nf+ns][np.arange(ns), selected_routes].mean())
                assert abs(selected_kl - receipt['best']['selection_kl']) < 1e-12
                if arm.endswith('_constant'):
                    assert np.max(np.abs(scores-scores[:1])) == 0
                gate = select_gate(scores[nf:nf+ns], mean[nf:nf+ns], base, gate_scale)
                chosen = choices[f, seed, arm]
                assert chosen['baseline'] == base and chosen['scale'] == gate_scale
                assert chosen['gate']['static'] == gate[4] and chosen['gate']['margin_units'] == gate[5]
                assert chosen['gate']['selection_kl'] == gate[0] and chosen['gate']['switches'] == gate[1]
                raw = scores.argmin(1)
                routed = np.full(len(ids), base) if gate[4] else np.where(scores[:, base] - scores[np.arange(len(ids)), raw] > gate[5]*gate_scale, raw, base)
                for rule, routes in (('raw', raw), ('gate', routed)):
                    policy = f'{arm}_{rule}_s{seed}'
                    for j, (i, route) in enumerate(zip(ids, routes)):
                        row = index[f, i, policy]; assert row['route'] == int(route) and row['baseline'] == base
                        assert row['cohort'] == ('fit' if j < nf else ('selection' if j < nf+ns else 'outer'))
                        assert row['kl'] == labels[i]['kl'][0][:0] + costs[j, :, route].tolist()
                        assert row['nll'] == np.asarray(labels[i]['nll'])[:, route].tolist(); evaluated += 1
                del initial, checkpoint
            assert len(set(parameter_digests)) == 1
            parameter_checks.append({'fold': f, 'seed': seed, 'four_arm_learnable_initial_sha256': parameter_digests[0]})
        for policy, routes in (('fold_static', np.full(len(ids), base)), ('historical_static1', np.ones(len(ids), dtype=int)), ('future_known_best128', mean.argmin(1))):
            for j, (i, route) in enumerate(zip(ids, routes)):
                row = index[f, i, policy]; assert row['route'] == int(route) and row['kl'] == costs[j, :, route].tolist()
    outer = {(r['sample'], r['policy']): r for r in private if r['cohort'] == 'outer'}
    summary = load(out / 'summary.json'); assert summary['outer_documents'] == 351 and not summary['engineering_only']
    ids = sorted(cfg['ids']); policies = sorted({r['policy'] for r in summary['outer_summary']})
    arrays = {}
    for policy in policies:
        members = [policy] if not policy.endswith('_mean3') else [policy.removesuffix('_mean3')+f'_s{s}' for s in cfg['seeds']]
        arrays[policy] = {m: np.asarray([[outer[i, p][m] for p in members] for i in ids]).mean(1) for m in ('kl', 'nll')}
        histogram = next(r for r in summary['route_histograms'] if r['policy'] == policy)
        routes = [outer[i, p]['route'] for i in ids for p in members]
        assert histogram['route_histogram'] == {str(k): v for k, v in Counter(routes).items()}
        assert histogram['switch_fraction_vs_fold_static'] == float(np.mean([outer[i, p]['route'] != outer[i, p]['baseline'] for i in ids for p in members]))
    columns = {'first32': [0], 'later96': [1, 2, 3], 'all128': [0, 1, 2, 3]}
    for row in summary['outer_summary']:
        own = arrays[row['policy']][row['metric']][:, columns[row['window']]].mean(1)
        baseline = arrays['fold_static'][row['metric']][:, columns[row['window']]].mean(1)
        assert abs(row['mean'] - own.mean()) < 1e-12 and abs(row['static_mean'] - baseline.mean()) < 1e-12
        check_estimate(baseline-own, row)
    for row in summary['comparisons']:
        difference = (arrays[row['baseline']][row['metric']] - arrays[row['policy']][row['metric']])[:, columns[row['window']]].mean(1)
        check_estimate(difference, row)
    write(out / 'independent_audit.json', {'passed': True, 'independent_source_recovery': True, 'eligible_documents': 351,
        'folds_reconstructed_without_training_split_helper': True, 'per_token_metric_values': metric_values,
        'quality_rows_checked': len(private), 'neural_route_rows_checked': evaluated,
        'learnable_initialization_checks': parameter_checks, 'common_inner128_selection_reconstructed': True,
        'fit_only_target_prior_and_scale_verified': True, 'no_input_scores_identical_across_documents': True,
        'joint_mask_gpu_replays': 32, 'router_gpu_replays': 144, 'full_prompt_embedding_gpu_checks': 4,
        'summary_and_bootstrap_reconstructed': True, 'old_files_preserved': len(preserved),
        'audit_source_sha256': sha256(Path(__file__)), 'limits': cfg['limits']})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True, type=Path); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    pin_threads(resources()[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cfg = {'main_run': str(args.run), 'output': str(out)}
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'done')
    audit(root, args.run, out)
    print(json.dumps({'independent_audit': 'passed', 'run': str(args.run)}), flush=True)


if __name__ == '__main__':
    main()
