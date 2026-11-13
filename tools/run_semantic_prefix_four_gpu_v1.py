"""Stage26 frozen two-layer OPT features, paired router and no-input controls."""
import argparse
import gc
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load
from crossfit_routing_core_v1 import gate_fit, gated, grouped_folds
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, read_lines, resources, setup_worker,
    thread_record, verify_preserved, write)
from headroom_ruler_core_v1 import paired
from horizontal_quality_v3 import pin_threads
from semantic_prefix_core_v1 import fit_representation, hidden_prompt, load_prefix
from stage1_benchmark_v2 import sha256, stamp


def worker(rank, cfg, inbox, messages):
    setup_worker(rank); device = f'cuda:{rank}'; out = Path(cfg['output'])
    old = Path(cfg['old_run']); data = Records(cfg['data_run'])
    original = torch.load(old / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    embeddings = {i: original[i] for i in cfg['ids']}; del original
    prefix = load_prefix(cfg['model'], device)
    if cfg['probe']:
        from transformers import AutoModel
        sample = max(cfg['reference_train_ids'], key=lambda i: data.rows[i]['prompt_length'])
        raw, p = data[sample]; length = min(p, (64, 129, 513, 2016)[rank])
        prompt = torch.tensor(raw[:length], device=device, dtype=torch.long)[None]
        with torch.inference_mode():
            measured, _ = hidden_prompt(prefix, prompt)
            model = AutoModel.from_pretrained(cfg['model'], local_files_only=True,
                torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
            full = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=False,
                output_hidden_states=True, return_dict=True).hidden_states[2]
            maximum = float((measured - full).abs().max())
            assert torch.equal(measured, full), maximum
        write(out / f'prefix_reference_{rank:03d}.json', {'passed': True, 'tokens': length,
            'matches_full_opt_hidden_states2_exactly': True, 'maximum_error': maximum,
            'final_layer_norm_not_applied': prefix.final_layer_norm is None})
        del model, full, measured; gc.collect(); torch.cuda.empty_cache()
    hidden_values = {}; lengths = []
    with torch.inference_mode():
        for offset, sample in enumerate(cfg['ids'][rank::4]):
            raw, p = data[sample]
            assert data.rows[sample]['sequence_sha256'] == cfg['sequence_sha256'][str(sample)]
            prompt = torch.tensor(raw[:p], device=device, dtype=torch.long)[None]
            hidden, embedding = hidden_prompt(prefix, prompt)
            assert torch.equal(embedding.cpu(), embeddings[sample])
            assert hidden.shape == embeddings[sample].shape and hidden.shape[1] == p
            assert torch.isfinite(hidden).all()
            hidden_values[sample] = hidden.cpu(); lengths.append(p)
            if offset % 25 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'semantic_features',
                    'documents': offset + 1, 'total': len(cfg['ids'][rank::4])})
    path = out / f'private_hidden2_{rank:03d}.pth'
    with path.open('xb') as stream:
        torch.save(hidden_values, stream)
    write(out / f'features_worker_{rank:03d}.json', {'passed': True, 'documents': len(hidden_values),
        'minimum_tokens': min(lengths), 'maximum_tokens': max(lengths), 'all_prompt_tokens': True,
        'embedding_matches_old_exactly': True, 'sha256': sha256(path), **thread_record()})
    del prefix, hidden_values; gc.collect(); torch.cuda.empty_cache()
    messages.put({'kind': 'features_done', 'rank': rank}); assert inbox.get()['kind'] == 'fit'
    hidden_values = {}
    for j in range(4):
        hidden_values.update(torch.load(out / f'private_hidden2_{j:03d}.pth', weights_only=True, map_location='cpu'))
    assert set(hidden_values) == set(cfg['ids'])
    labels = {r['sample']: r for j in range(6) for r in read_lines(old / f'private_labels_{j:03d}.jsonl') if r['sample'] in hidden_values}
    fold = cfg['folds'][rank]
    for seed in cfg['seeds']:
        for arm in ('hidden2', 'constant') + (('embedding',) if cfg['probe'] else ()):
            feature = hidden_values if arm == 'hidden2' else embeddings
            fit_representation(out, cfg, fold, feature, labels, seed, device, messages, rank,
                arm=arm, observe=arm != 'constant')
    write(out / f'worker_{rank:03d}.json', {'passed': True, 'fold': rank, 'seeds': cfg['seeds'], **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def score_record(out, cfg, arm, fold, seed):
    if arm == 'embedding' and not cfg['probe']:
        directory = Path(cfg['reference_run']); name = f'f{fold}_s{seed}'
    else:
        directory = out; name = f'{arm}_f{fold}_s{seed}'
    receipt = load(directory / f'{name}_fit.json')
    path = directory / f'private_{name}_scores.pth'; assert sha256(path) == receipt['score_sha256']
    assert sha256(receipt['best']['path']) == receipt['best']['sha256']
    return receipt, torch.load(path, weights_only=True), str(directory / f'{name}_fit.json')


def finalize(out, cfg):
    old = Path(cfg['old_run'])
    labels = {r['sample']: r for j in range(6) for r in read_lines(old / f'private_labels_{j:03d}.jsonl') if r['sample'] in cfg['ids']}
    scored = []; choices = []; checks = []
    for fold in cfg['folds']:
        f = fold['fold']; ids = fold['fit_ids'] + fold['selection_ids'] + fold['outer_ids']
        nfit, nsel = len(fold['fit_ids']), len(fold['selection_ids'])
        costs = np.asarray([labels[i]['kl'] for i in ids]); nll = np.asarray([labels[i]['nll'] for i in ids])
        baseline = int(costs[:nfit].mean(0).argmin()); relative = costs[:nfit] - costs[:nfit, :1]
        scale = max(float(relative[:, 1:].std(ddof=1)), .03)
        predictions = [('fold_static', np.full(len(ids), baseline)), ('future_known_best', costs.argmin(1))]
        for seed in cfg['seeds']:
            initials = []
            for arm in ('embedding', 'hidden2', 'constant'):
                receipt, saved, source = score_record(out, cfg, arm, f, seed)
                assert saved['ids'] == ids
                score = saved['scores'].numpy().astype(np.float64)
                initials.append(receipt['initial_sha256'])
                rule = gate_fit(score[nfit:nfit+nsel], costs[nfit:nfit+nsel], baseline, scale)
                predictions += [(f'{arm}_raw_s{seed}', score.argmin(1)),
                    (f'{arm}_gate_s{seed}', gated(score, rule, baseline, scale))]
                choices.append({'fold': f, 'arm': arm, 'seed': seed, 'baseline_index': baseline,
                    'scale': scale, 'gate': rule, 'selected_epoch': receipt['best']['epoch'],
                    'fit_receipt': source, 'fit_receipt_sha256': sha256(source)})
            assert len(set(initials)) == 1
            checks.append({'fold': f, 'seed': seed, 'three_arms_initial_sha256': initials[0],
                'identical_initial_parameters': True})
        for policy, routes in predictions:
            for position, (sample, route) in enumerate(zip(ids, routes)):
                cohort = 'fit' if position < nfit else ('selection' if position < nfit + nsel else 'outer')
                scored.append({'fold': f, 'sample': sample, 'cohort': cohort, 'policy': policy,
                    'route_index': int(route), 'baseline_index': baseline, 'kl': float(costs[position, route]),
                    'nll': float(nll[position, route]), 'static_kl': float(costs[position, baseline]),
                    'static_nll': float(nll[position, baseline])})
    with (out / 'private_quality_rows.jsonl').open('x') as stream:
        for row in scored:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    grouped = defaultdict(list)
    for row in scored:
        grouped[row['fold'], row['cohort'], row['policy']].append(row)
    aggregates = []
    for (fold, cohort, policy), group in grouped.items():
        aggregates.append({'fold': fold, 'cohort': cohort, 'policy': policy, 'documents': len(group),
            'kl': float(np.mean([r['kl'] for r in group])), 'nll': float(np.mean([r['nll'] for r in group])),
            'improvement': float(np.mean([r['static_kl'] - r['kl'] for r in group])),
            'beneficial': sum(r['kl'] < r['static_kl'] for r in group),
            'harmful': sum(r['kl'] > r['static_kl'] for r in group),
            'switches': sum(r['route_index'] != r['baseline_index'] for r in group),
            'route_histogram': dict(Counter(r['route_index'] for r in group))})
    outer = [r for r in scored if r['cohort'] == 'outer']; per_document = {}; summary = []
    policies = ['fold_static', 'future_known_best'] + [f'{arm}_{rule}_mean3' for arm in ('embedding', 'hidden2', 'constant') for rule in ('raw', 'gate')]
    for policy in policies:
        base = policy.removesuffix('_mean3'); members = [f'{base}_s{seed}' for seed in cfg['seeds']] if policy.endswith('_mean3') else [policy]
        records = {}
        for sample in sorted({r['sample'] for r in outer}):
            group = [r for r in outer if r['sample'] == sample and r['policy'] in members]
            assert len(group) == len(members)
            records[sample] = {metric: float(np.mean([r[metric] for r in group])) for metric in ('kl', 'nll', 'static_kl', 'static_nll')}
        per_document[policy] = records
        for metric in ('kl', 'nll'):
            own = np.asarray([r[metric] for r in records.values()]); static = np.asarray([r['static_' + metric] for r in records.values()])
            summary.append({'policy': policy, 'metric': metric, 'mean': float(own.mean()),
                'static_mean': float(static.mean()), **paired(static - own, None, 260579)})
    comparisons = []
    for rule in ('raw', 'gate'):
        own = per_document[f'hidden2_{rule}_mean3']
        for baseline in (f'embedding_{rule}_mean3', f'constant_{rule}_mean3', 'fold_static'):
            for metric in ('kl', 'nll'):
                differences = [per_document[baseline][sample][metric] - own[sample][metric] for sample in own]
                comparisons.append({'policy': f'hidden2_{rule}_mean3', 'baseline': baseline,
                    'metric': metric, **paired(differences, None, 260579)})
    result = {'engineering_only': cfg['probe'], 'outer_documents': len({r['sample'] for r in outer}),
        'outer_summary': summary, 'comparisons': comparisons, 'per_fold': aggregates, 'limits': cfg['limits']}
    write(out / 'summary.json', result); write(out / 'policy_choices.json', choices); write(out / 'paired_initialization.json', checks)
    return result


def report(root, out, cfg, summary):
    text = ['# 阶段26：前两层语义隐藏状态的小试', '',
        '冻结OPT-13B，使用所有候选共同保留的第0、1层处理全文提示词后的隐藏状态，作为路由器输入；取第1层残差输出，不施加模型最终LayerNorm，不包含任何续写token。K20/40、8个完整组合库、完整组合联合未来32-token KL监督、回答内固定路径不变。', '',
        '沿用Stage25四折，每折520训练、80选模、200复核，三个种子。优化器、初始参数、数据顺序、24轮更新预算及检查点/门槛选择规则保持一致。embedding组复用Stage25准确对应的检查点；hidden2组重新训练；constant组同参数同更新预算但不读取内容或长度。mean3是三个独立种子的质量平均，不是三模型集成推理。', '',
        '这些外折结果已经在Stage25查看，因此本轮属于旧训练文档上的开发诊断，不是新的独立测试。原100开发文档不参与本轮。候选库历史上用过800条中的256条，仍有候选筛选依赖。没有根据本轮复核结果调整配置。', '',
        '| 外折800文档策略 | KL，越低越好 | 相对折内静态改善 | 探索性95%区间 |', '|---|---:|---:|---|']
    for row in summary['outer_summary']:
        if row['metric'] == 'kl':
            lo, hi = row['ci95']; text.append(f"| {row['policy']} | {row['mean']:.6f} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    text += ['', '| 语义路由比较 | 基线 | KL改善 | 探索性95%区间 |', '|---|---|---:|---|']
    for row in summary['comparisons']:
        if row['metric'] == 'kl':
            lo, hi = row['ci95']; text.append(f"| {row['policy']} | {row['baseline']} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    text += ['', '正改善表示语义路由更好。区间只反映文档重采样，未覆盖训练和折间依赖，未做多重比较校正，不能将本轮自适应开发结果当作最终显著性证据。future_known_best读取真实未来KL，仅表示同候选库的诊断空间。', '',
        '工程检查分别在64、129、513、2016个提示词token上验证两层提取结果与完整原OPT的hidden_states[2]逐元素相同；全文embedding与旧缓存逐元素相同。训练和选模只用每折指定子集。仅测试路由质量，没有测量前缀KV复用、部署吞吐、端到端加速、峰值显存或恢复训练效果。', '',
        '使用GPU 0—3与42个CPU核心，worker各10核、控制器2核。保全旧源码/结果/检查点；公开源码、协议、审计和聚合结果，参数、隐藏状态、token和逐文档输出留在本地。', '',
        f"[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [配对初始化](../{out.relative_to(root)}/paired_initialization.json) · [独立审计](../{out.relative_to(root)}/independent_audit.json)"]
    with (root / 'analysis' / f'{out.name}_results.zh-CN.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true'); parser.add_argument('--probe-run', type=Path); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    cpus = resources(); pin_threads(cpus[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    preserved = preserve(root, out)
    old = root / 'runs/stage20_codebook_refine_20261112_02_main'; previous = load(old / 'protocol.json')
    reference = root / 'runs/stage25_crossfit_20261113_02_main'; ref = load(reference / 'protocol.json')
    data = Records(previous['data_run']); ids = previous['train_ids']; folds = grouped_folds(ids, data.rows)
    assert folds == ref['folds']
    bank = load(old / 'candidate_bank.json'); assert bank['k'] == 20 and len(bank['routes']) == 8
    assert all(not set(route['skips']) & {0, 1} for route in bank['routes'])
    if args.probe:
        folds = [{**f, 'fit_ids': f['fit_ids'][:16], 'selection_ids': f['selection_ids'][:8], 'outer_ids': f['outer_ids'][:8]} for f in folds]
        ids = sorted({i for f in folds for key in ('fit_ids', 'selection_ids', 'outer_ids') for i in f[key]})
    else:
        assert args.probe_run and load(args.probe_run / 'verification.json')['passed']
        for rank in range(4):
            assert load(args.probe_run / f'prefix_reference_{rank:03d}.json')['matches_full_opt_hidden_states2_exactly']
    evidence = [old / 'protocol.json', old / 'candidate_bank.json', old / 'private_embeddings.pth',
        reference / 'protocol.json', reference / 'independent_audit.json', Path(previous['data_run']) / 'data_manifest.json',
        Path(previous['model']) / 'config.json', Path(previous['model']) / 'pytorch_model.bin.index.json',
        Path(previous['model']) / 'pytorch_model-00001-of-00003.bin']
    evidence += [old / f'private_labels_{rank:03d}.jsonl' for rank in range(6)]
    for f in range(4):
        for seed in (579, 580, 581):
            path = reference / f'f{f}_s{seed}_fit.json'; evidence += [path, reference / f'private_f{f}_s{seed}_scores.pth', Path(load(path)['best']['path'])]
    sources = ['semantic_prefix_core_v1.py', 'run_semantic_prefix_four_gpu_v1.py', 'launch_semantic_prefix_v1.py']
    cfg = {'output': str(out), 'old_run': str(old), 'reference_run': str(reference), 'model': previous['model'],
        'data_run': previous['data_run'], 'probe': args.probe, 'probe_run': str(args.probe_run) if args.probe_run else None,
        'ids': ids, 'reference_train_ids': previous['train_ids'], 'folds': folds, 'epochs': 1 if args.probe else 24,
        'seeds': [579] if args.probe else [579, 580, 581], 'arms': ['embedding', 'hidden2', 'constant'],
        'sequence_sha256': {str(i): data.rows[i]['sequence_sha256'] for i in ids},
        'evidence_sha256': {str(path): sha256(path) for path in evidence},
        'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'primary_comparison': 'hidden2_gate_mean3 vs embedding_gate_mean3; also fold_static and matched no-input constant. No outer-driven configuration changes.',
        'feature': 'Raw hidden state immediately after frozen decoder layers0/1 on ALL prompt tokens, before layer2 and without final LayerNorm. Prompt-only; never future tokens.',
        'selection': 'Earliest minimum inner80 KL at epochs0/1/4/...24; existing margin grid0/.05/.1/.2/.4 plus alwaysstatic on inner80 only.',
        'resources': '4GPU/42CPU; worker10/controller2',
        'limits': 'Repeatedly reviewed train800 development diagnostic, not independent generalization. Fixed bank historically selected from256 of these800. Document bootstrap ignores training/fold dependence. No speed/memory claim.'}
    assert not set(ids) & set(previous['validation_ids'])
    write(out / 'protocol.json', cfg)
    stamp('semantic_prefix_start', probe=args.probe, documents=len(ids))
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'features_done')
        values = {}
        for rank in range(4):
            path = out / f'private_hidden2_{rank:03d}.pth'
            assert sha256(path) == load(out / f'features_worker_{rank:03d}.json')['sha256']
            shard = torch.load(path, weights_only=True, map_location='cpu')
            assert not set(shard) & set(values); values.update(shard)
        assert set(values) == set(ids)
        assert all(value.shape == (1, data.rows[sample]['prompt_length'], 5120) for sample, value in values.items())
        del values; gc.collect()
        workers.send_all({'kind': 'fit'}); workers.receive(4, 'done')
    summary = finalize(out, cfg)
    for path, digest in cfg['evidence_sha256'].items():
        assert sha256(path) == digest
    verify_preserved(out, preserved)
    write(out / 'verification.json', {'passed': True, 'probe': args.probe, 'full_prompt_verified': True,
        'protected_prefix_0_1_shared_by_all_masks': True, 'no_future_observation': True,
        'three_arm_initial_parameters_match': True, 'old_evidence_unchanged': True,
        'historical_dev100_excluded': True, 'development_only_disclosed': True})
    if not args.probe:
        report(root, out, cfg, summary)
    stamp('semantic_prefix_complete', probe=args.probe, run=str(out))


if __name__ == '__main__':
    main()
