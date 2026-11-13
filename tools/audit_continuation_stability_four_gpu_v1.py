"""Independent Stage27 sampling, arithmetic, ranking and real OPT replays."""
import argparse
import hashlib
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers, read_lines, resources, setup_worker, thread_record, write
from horizontal_quality_v3 import pin_threads
from prompt_fixed_core_v1 import fixed_route
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(Path(path).read_text())


def digest(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i4').tobytes()).hexdigest()


def check_estimate(values, expected):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    assert len(a) == expected['documents']
    if not len(a):
        assert expected['mean'] is None and expected['ci95'] is None; return
    assert abs(a.mean() - expected['mean']) < 1e-12
    draws = np.random.default_rng(270579).integers(0, len(a), (2000, len(a)))
    assert np.allclose(np.quantile(a[draws].mean(1), [.025, .975]), expected['ci95'], atol=1e-12, rtol=0)


def worker(rank, job, inbox, messages):
    from transformers import AutoModelForCausalLM
    setup_worker(rank); out = Path(job['main_run']); audit_out = Path(job['output']); cfg = load(out / 'protocol.json')
    inputs = {r['sample']: r for r in read_lines(cfg['input_file'])}
    sample = max(cfg['ids'][rank::4], key=lambda i: inputs[i]['prompt_length'])
    row = inputs[sample]; p = row['prompt_length']; device = f'cuda:{rank}'
    ids = torch.tensor(row['ids'], dtype=torch.long, device=device)[None]
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    bank = load(Path(cfg['old_run']) / 'candidate_bank.json')
    expected = torch.load(out / f'private_token_metrics_{rank:03d}.pth', weights_only=True)[sample]
    def log_probs(skips):
        with fixed_route(model, skips):
            hidden = model.model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False).last_hidden_state
            logits = model.lm_head(hidden[:, p-1:p+127]).float()
        return logits.log_softmax(-1)
    maximum = 0.
    with torch.inference_mode():
        teacher = log_probs([])
        for route in bank['routes']:
            student = log_probs(route['skips'])
            kl = (teacher.exp() * (teacher - student)).sum(-1)[0].cpu()
            nll = -student.gather(-1, ids[:, p:, None]).squeeze(-1)[0].cpu()
            maximum = max(maximum, float((kl - expected['kl'][:, route['index']]).abs().max()),
                float((nll - expected['nll'][:, route['index']]).abs().max()))
            assert torch.equal(kl, expected['kl'][:, route['index']])
            assert torch.equal(nll, expected['nll'][:, route['index']])
    write(audit_out / f'worker_{rank:03d}.json', {'passed': True, 'documents': 1, 'prompt_tokens': p,
        'continuation_tokens': 128, 'complete_masks': 8, 'maximum_token_metric_error': maximum, **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def audit(root, out, audit_out):
    from transformers import AutoTokenizer
    cfg = load(out / 'protocol.json'); assert not cfg['probe'] and len(cfg['ids']) == 128
    assert cfg['windows'] == [[0, 32], [32, 64], [64, 96], [96, 128]] and cfg['near_tie_tolerance'] == .01
    for name, expected in cfg['source_sha256'].items():
        assert sha256(root / 'tools' / name) == expected
    for path, expected in cfg['evidence_sha256'].items():
        assert sha256(path) == expected
    assert sha256(cfg['input_file']) == cfg['input_file_sha256']
    preserved = load(out / 'preservation_before.json')
    for path, expected in preserved.items():
        assert sha256(path) == expected, path
    previous = load(Path(cfg['old_run']) / 'protocol.json'); records = Records(cfg['data_run'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], use_fast=False, local_files_only=True)
    eligible = {}; excluded = Counter()
    with Path(records.manifest['corpus']).open('rb') as stream:
        for sample in previous['train_ids']:
            row = records.rows[sample]; p = row['prompt_length']
            if p > 1920:
                excluded['context_limit'] += 1; continue
            stream.seek(row['source_start_byte']); text_bytes = stream.read(row['source_end_byte'] - row['source_start_byte'])
            assert hashlib.sha256(text_bytes).hexdigest() == row['source_raw_sha256']
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(text_bytes.decode('utf8').strip(), add_special_tokens=False,
                truncation=True, max_length=2047)
            old, _ = records[sample]; assert tokens[:p+32] == old.tolist()
            if len(tokens) < p + 128:
                excluded['insufficient_same_document_continuation'] += 1; continue
            eligible[sample] = tokens[:p+128]
    permutation = list(eligible); np.random.default_rng(270579).shuffle(permutation)
    selected = []; full = set(); lead = set()
    for sample in permutation:
        row = records.rows[sample]
        if row['record_sha256'] in full or row['record_lead_sha256'] in lead:
            continue
        selected.append(sample); full.add(row['record_sha256']); lead.add(row['record_lead_sha256'])
        if len(selected) == 128:
            break
    sampling = load(out / 'sampling.json')
    assert cfg['ids'] == sampling['selected_ids'] == selected
    assert sampling['eligible_documents'] == len(eligible) and sampling['excluded'] == dict(excluded)
    assert not set(selected) & set(previous['validation_ids'])
    inputs = read_lines(cfg['input_file']); assert [r['sample'] for r in inputs] == selected
    for row in inputs:
        sample = row['sample']; p = records.rows[sample]['prompt_length']
        assert row['ids'] == eligible[sample] and row['prompt_length'] == p
        assert digest(row['ids']) == row['extended_sequence_sha256'] and digest(row['ids'][:p]) == row['prompt_sha256']
        assert digest(row['ids'][:p+32]) == records.rows[sample]['sequence_sha256']
    rows = {}; metric_values = 0; maximum_mean_error = 0.
    allocation = load(out / 'allocation.json'); assert allocation['gpu_indices'] == [0, 1, 2, 3] and allocation['cpu_count'] == 42
    for rank in range(4):
        receipt = load(out / f'worker_{rank:03d}.json')
        label_file = out / f'private_labels_{rank:03d}.jsonl'; metric_file = out / f'private_token_metrics_{rank:03d}.pth'
        assert sha256(label_file) == receipt['label_sha256'] and sha256(metric_file) == receipt['token_metrics_sha256']
        group = read_lines(label_file); assert [r['sample'] for r in group] == selected[rank::4]
        token = torch.load(metric_file, weights_only=True); assert set(token) == set(selected[rank::4])
        for row in group:
            sample = row['sample']; assert sample not in rows
            for metric in ('kl', 'nll'):
                values = token[sample][metric]; assert values.shape == (128, 8) and torch.isfinite(values).all()
                reconstructed = values.reshape(4, 32, 8).mean(1).double().numpy()
                independent = values.double().numpy().reshape(4, 32, 8).mean(1)
                assert np.array_equal(reconstructed, row[metric])
                error = float(np.max(np.abs(independent - reconstructed))); maximum_mean_error = max(maximum_mean_error, error)
                assert error < 3e-6
                metric_values += values.numel()
            assert row['extended_sequence_sha256'] == next(r for r in inputs if r['sample'] == sample)['extended_sequence_sha256']
            rows[sample] = row
        assert receipt['maximum_direct_old32_kl_replay_error'] < 1e-6
        assert receipt['maximum_repeated_long_kl_error'] == receipt['maximum_altered_future_prefix_logprob_error'] == 0
        for directory in (out, audit_out):
            worker_receipt = load(directory / f'worker_{rank:03d}.json')
            assert worker_receipt['cpu_ids'] == allocation['cpu_ids'][rank*10:(rank+1)*10] and worker_receipt['threads'] == 10
            assert all(set(v) <= set(worker_receipt['cpu_ids']) for v in worker_receipt['thread_cpu_ids'].values())
    costs = np.asarray([rows[i]['kl'] for i in selected]); nll = np.asarray([rows[i]['nll'] for i in selected])
    summary = load(out / 'summary.json'); assert summary['documents'] == 128 and not summary['engineering_only']
    best = costs.argmin(-1); minimum = costs.min(-1); gap = np.sort(costs, axis=-1)[:, :, 1] - minimum
    assert np.allclose(summary['stability']['best_second_gap_quantiles'], np.quantile(gap, [.25, .5, .75]), atol=1e-12, rtol=0)
    assert summary['stability']['fraction_winner_margin_below_tau001'] == float(np.mean(gap < .01))
    for expected in summary['stability']['window_pairs']:
        a, b = expected['windows']; winner_match = float(np.mean(best[:, a] == best[:, b]))
        chance = sum(np.mean(best[:, a] == r) * np.mean(best[:, b] == r) for r in range(8))
        assert abs(expected['winner_agreement'] - winner_match) < 1e-12 and abs(expected['chance_agreement'] - chance) < 1e-12
        if chance < 1:
            assert abs(expected['cohen_kappa'] - (winner_match - chance) / (1 - chance)) < 1e-12
        correlations = []; overlaps = []; material_agreement = []
        for document in range(128):
            correlations.append(spearmanr(costs[document, a], costs[document, b]).statistic)
            near_a = set(np.flatnonzero(costs[document, a] <= minimum[document, a] + .01))
            near_b = set(np.flatnonzero(costs[document, b] <= minimum[document, b] + .01))
            overlaps.append(len(near_a & near_b) / len(near_a | near_b))
            agreement = []
            for i, j in combinations(range(8), 2):
                da = costs[document, a, i] - costs[document, a, j]; db = costs[document, b, i] - costs[document, b, j]
                if abs(da) >= .01 and abs(db) >= .01:
                    agreement.append(np.sign(da) == np.sign(db))
            material_agreement.append(np.mean(agreement) if agreement else np.nan)
        check_estimate(correlations, expected['within_document_route_spearman'])
        check_estimate(overlaps, expected['near_optimal_set_jaccard_tau001'])
        check_estimate(material_agreement, expected['material_pairwise_order_agreement_tau001'])
        regret = costs[np.arange(128), b, best[:, a]] - minimum[:, b]
        check_estimate(regret, expected['first_winner_regret_on_second'])
        assert expected['first_winner_within_tau001_on_second'] == float(np.mean(regret <= .01))
        da = costs[:, a] - costs[:, a, 1, None]; db = costs[:, b] - costs[:, b, 1, None]
        da -= da.mean(0); db -= db.mean(0)
        denominator = np.sqrt((da*da).sum() * (db*db).sum())
        assert abs(expected['document_centered_relative_gap_pearson'] - float((da*db).sum() / denominator)) < 1e-12
    policies = {'strong_static': np.ones((128, 4, 1), dtype=int)}
    for name, route in (('first32_future_oracle', costs[:, 0].argmin(1)), ('full128_future_oracle', costs.mean(1).argmin(1))):
        policies[name] = np.broadcast_to(route[:, None, None], (128, 4, 1))
    policies['other96_future_oracle'] = np.stack([np.mean(costs[:, [j for j in range(4) if j != w]], axis=1).argmin(1) for w in range(4)], 1)[:, :, None]
    for family, source, prefix in (('embedding_gate_mean3', root / 'runs/stage25_crossfit_20261113_02_main/private_crossfit_rows.jsonl', 'neural_gate_s'),
        ('hidden2_gate_mean3', root / 'runs/stage26_semantic_prefix_20261113_01_main/private_quality_rows.jsonl', 'hidden2_gate_s')):
        decision = {}
        for row in read_lines(source):
            if row['sample'] in selected and row['cohort'] == 'outer' and row['policy'].startswith(prefix):
                key = row['sample'], int(row['policy'].removeprefix(prefix)); assert key not in decision
                decision[key] = row['route_index']
        assert len(decision) == 128 * 3
        routes = np.asarray([[decision[i, seed] for seed in (579, 580, 581)] for i in selected])
        policies[family] = np.broadcast_to(routes[:, None, :], (128, 4, 3))
    for expected in summary['policy_transfer']:
        metric = costs if expected['metric'] == 'kl' else nll; window = expected['window']
        windows = [window] if isinstance(window, int) else ([1, 2, 3] if window == 'later96' else [0, 1, 2, 3])
        values = []
        for document in range(128):
            values.append(np.mean([metric[document, w, r] for w in windows for r in policies[expected['policy']][document, w]]))
        reference = metric[:, windows, 1].mean(1)
        assert abs(expected['mean'] - np.mean(values)) < 1e-12 and abs(expected['static_mean'] - reference.mean()) < 1e-12
        check_estimate(reference - values, expected['improvement'])
    for expected in summary['per_route_window_means']:
        w, r = expected['window'], expected['route']
        assert abs(expected['kl'] - costs[:, w, r].mean()) < 1e-12
        assert abs(expected['nll'] - nll[:, w, r].mean()) < 1e-12
    return {'passed': True, 'auditor_sha256': sha256(__file__), 'preserved_files': len(preserved),
        'eligible_documents_independently_recovered': len(eligible), 'selected_documents': 128,
        'same_document_extended_tokens_and_original_full_prompts_verified': True,
        'original_dev100_excluded': True, 'selection_reproduced_without_labels': True,
        'per_token_metric_values_verified': metric_values, 'maximum_fp32_vs_fp64_window_mean_error': maximum_mean_error,
        'all_ranking_and_transfer_metrics_reconstructed': True, 'frozen_prior_outer_routes_verified': True,
        'independent_opt_replay_documents': 4, 'independent_opt_complete_mask_replays': 32,
        'independent_opt_token_metrics_exact': True, 'old_short_replay_and_long_future_causality_checks_passed': True,
        'resources': '4GPU/42CPU', 'limits': cfg['limits']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = args.run.resolve(); audit_out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    cpus = resources(); pin_threads(cpus[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert load(out / 'verification.json')['passed']
    with Workers(worker, {'main_run': str(out), 'output': str(audit_out)}) as workers:
        workers.receive(4, 'done')
    result = audit(root, out, audit_out); write(out / 'independent_audit.json', result); write(audit_out / 'independent_audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
