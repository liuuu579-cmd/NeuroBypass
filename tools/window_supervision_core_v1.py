"""Stage28: matched first-window versus four-window joint-mask supervision."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from continuation_stability_core_v1 import digest
from crossfit_routing_core_v1 import gate_fit, gated
from full_context_data_v1 import Records
from full_context_experiment_v1 import read_lines, write
from headroom_ruler_core_v1 import paired
from stage1_benchmark_v2 import sha256


ARMS = ('first32_embedding', 'mean128_embedding', 'first32_constant', 'mean128_constant')
PREFIX = 'stage28_window_supervision_20261113_01'


def load(path):
    return json.loads(Path(path).read_text())


def prepare_all(out, previous):
    from transformers import AutoTokenizer
    data = Records(previous['data_run'])
    tokenizer = AutoTokenizer.from_pretrained(previous['model'], use_fast=False, local_files_only=True)
    eligible = []; excluded = Counter()
    with Path(data.manifest['corpus']).open('rb') as source, (out / 'private_inputs.jsonl').open('x') as target:
        for sample in previous['train_ids']:
            row = data.rows[sample]; p = row['prompt_length']
            if p + 128 > 2048:
                excluded['context_limit'] += 1; continue
            source.seek(row['source_start_byte'])
            raw = source.read(row['source_end_byte'] - row['source_start_byte'])
            assert hashlib.sha256(raw).hexdigest() == row['source_raw_sha256']
            text = raw.decode('utf8').strip()
            assert hashlib.sha256(' '.join(text.split()).encode()).hexdigest() == row['record_sha256']
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2047)
            original, _ = data[sample]; assert tokens[:p+32] == original.tolist()
            if len(tokens) < p + 128:
                excluded['insufficient_same_document_continuation'] += 1; continue
            tokens = tokens[:p+128]; eligible.append(sample)
            target.write(json.dumps({'sample': sample, 'prompt_length': p, 'ids': tokens,
                'original_sequence_sha256': digest(tokens[:p+32]), 'prompt_sha256': digest(tokens[:p]),
                'extended_sequence_sha256': digest(tokens)}, allow_nan=False) + '\n')
    assert len(eligible) == 351 and not set(eligible) & set(previous['validation_ids'])
    sampling = {'considered': len(previous['train_ids']), 'eligible': len(eligible), 'eligible_ids': eligible,
        'excluded': dict(excluded), 'prompt_histogram': dict(Counter(data.rows[i]['prompt_length'] for i in eligible)),
        'rule': 'All original TRAIN documents with unchanged full prompt and128 following same-document tokens within2048; no KL filtering.',
        'limits': 'Eligibility favors longer texts; original100 development documents excluded.'}
    write(out / 'sampling.json', sampling)
    return out / 'private_inputs.jsonl', sampling


def make_folds(ids, records):
    parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i in ids:
        for field in ('record_sha256', 'record_lead_sha256'):
            key = (field, records[i][field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    groups = list(groups.values()); np.random.default_rng(280579).shuffle(groups)
    groups.sort(key=lambda g: -len(g)); buckets = [[] for _ in range(4)]
    for group in groups:
        buckets[min(range(4), key=lambda j: (len(buckets[j]), j))].extend(group)
    folds = []
    for f in range(4):
        outer = sorted(buckets[f]); rest = [g for g in groups if not set(g) & set(outer)]
        np.random.default_rng(280580 + f).shuffle(rest)
        selection = []; fit = []
        for group in rest:
            (selection if len(selection) < 35 else fit).extend(group)
        folds.append({'fold': f, 'fit_ids': sorted(fit), 'selection_ids': sorted(selection), 'outer_ids': outer})
    assert sorted(i for f in folds for i in f['outer_ids']) == sorted(ids)
    for f in folds:
        for field in ('record_sha256', 'record_lead_sha256'):
            sets = [{records[i][field] for i in f[key]} for key in ('fit_ids', 'selection_ids', 'outer_ids')]
            assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    return folds


def labels_for_fit(rows, fold, arm):
    # Existing fitter reads fit labels for gradients/prior/scale, selection labels
    # for checkpoint choice, and never reads outer labels. Share128 selection.
    fit = set(fold['fit_ids']); labels = {}
    for i, row in rows.items():
        windows = np.asarray(row['kl'])
        cost = windows[0] if arm.startswith('first32') and i in fit else windows.mean(0)
        labels[i] = {'kl': cost.tolist()}
    return labels


def finalize(out, cfg):
    rows = {r['sample']: r for rank in range(4) for r in read_lines(out / f'private_labels_{rank:03d}.jsonl')}
    records = []; choices = []; matched = []
    for fold in cfg['folds']:
        f = fold['fold']; ids = fold['fit_ids'] + fold['selection_ids'] + fold['outer_ids']
        nf, ns = len(fold['fit_ids']), len(fold['selection_ids'])
        costs = np.asarray([rows[i]['kl'] for i in ids]); nll = np.asarray([rows[i]['nll'] for i in ids])
        mean = costs.mean(1); baseline = int(mean[:nf].mean(0).argmin())
        relative = mean[:nf] - mean[:nf, :1]; scale = max(float(relative[:, 1:].std(ddof=1)), .03)
        policies = {'fold_static': np.full(len(ids), baseline), 'historical_static1': np.ones(len(ids), dtype=int),
            'future_known_best128': mean.argmin(1)}
        for seed in cfg['seeds']:
            parameter_hashes = []
            for arm in ARMS:
                name = f'{arm}_f{f}_s{seed}'; receipt = load(out / f'{name}_fit.json')
                path = out / f'private_{name}_scores.pth'
                assert sha256(path) == receipt['score_sha256'] and sha256(receipt['best']['path']) == receipt['best']['sha256']
                saved = torch.load(path, weights_only=True); assert saved['ids'] == ids
                scores = saved['scores'].numpy().astype(np.float64)
                rule = gate_fit(scores[nf:nf+ns], mean[nf:nf+ns], baseline, scale)
                policies[f'{arm}_raw_s{seed}'] = scores.argmin(1)
                policies[f'{arm}_gate_s{seed}'] = gated(scores, rule, baseline, scale)
                choices.append({'fold': f, 'arm': arm, 'seed': seed, 'baseline': baseline, 'scale': scale,
                    'gate': rule, 'selected_epoch': receipt['best']['epoch'], 'score_sha256': sha256(path)})
                parameter_hashes.append(receipt['initial_sha256'])
            # Full initial-state hashes differ between targets due to prior/scale
            # buffers. Identical learnable initialization is audited separately.
            assert parameter_hashes[0] == parameter_hashes[2] and parameter_hashes[1] == parameter_hashes[3]
            matched.append({'fold': f, 'seed': seed, 'same_target_embedding_constant_initial_state': True,
                'first32_state': parameter_hashes[0], 'mean128_state': parameter_hashes[1]})
        for policy, routes in policies.items():
            for j, (sample, route) in enumerate(zip(ids, routes)):
                records.append({'fold': f, 'sample': sample, 'policy': policy, 'route': int(route), 'baseline': baseline,
                    'cohort': 'fit' if j < nf else ('selection' if j < nf+ns else 'outer'),
                    'kl': costs[j, :, route].tolist(), 'nll': nll[j, :, route].tolist()})
    with (out / 'private_quality_rows.jsonl').open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    outer = [r for r in records if r['cohort'] == 'outer']; ids = sorted({r['sample'] for r in outer})
    indexed = {(r['sample'], r['policy']): r for r in outer}
    policies = ['fold_static', 'historical_static1', 'future_known_best128'] + [f'{arm}_{rule}_mean3' for arm in ARMS for rule in ('raw', 'gate')]
    values = {}; summary = []; histograms = []
    for policy in policies:
        members = [policy] if not policy.endswith('_mean3') else [policy.removesuffix('_mean3') + f'_s{s}' for s in cfg['seeds']]
        values[policy] = {metric: np.asarray([[indexed[i, p][metric] for p in members] for i in ids]).mean(1) for metric in ('kl', 'nll')}
        routes = [indexed[i, p]['route'] for i in ids for p in members]
        histograms.append({'policy': policy, 'decisions': len(routes), 'route_histogram': dict(Counter(routes)),
            'switch_fraction_vs_fold_static': float(np.mean([indexed[i, p]['route'] != indexed[i, p]['baseline'] for i in ids for p in members]))})
        for metric in ('kl', 'nll'):
            for window, columns in (('first32', [0]), ('later96', [1, 2, 3]), ('all128', [0, 1, 2, 3])):
                own = values[policy][metric][:, columns].mean(1)
                baseline_values = values['fold_static'][metric][:, columns].mean(1)
                summary.append({'policy': policy, 'metric': metric, 'window': window, 'mean': float(own.mean()),
                    'static_mean': float(baseline_values.mean()), **paired(baseline_values - own, None, 280579)})
    comparisons = []
    for rule in ('raw', 'gate'):
        policy = f'mean128_embedding_{rule}_mean3'
        for baseline_policy in (f'first32_embedding_{rule}_mean3', f'mean128_constant_{rule}_mean3', 'fold_static', 'historical_static1'):
            for metric in ('kl', 'nll'):
                for window, columns in (('first32', [0]), ('later96', [1, 2, 3]), ('all128', [0, 1, 2, 3])):
                    delta = (values[baseline_policy][metric] - values[policy][metric])[:, columns].mean(1)
                    comparisons.append({'policy': policy, 'baseline': baseline_policy, 'metric': metric,
                        'window': window, **paired(delta, None, 280579)})
    result = {'engineering_only': cfg['probe'], 'outer_documents': len(ids), 'outer_summary': summary,
        'comparisons': comparisons, 'route_histograms': histograms, 'limits': cfg['limits']}
    write(out / 'summary.json', result); write(out / 'policy_choices.json', choices); write(out / 'paired_initialization.json', matched)
    return result


def report(root, out, cfg, summary):
    lines = ['# 阶段28：首窗与多窗口联合KL监督配对小试', '',
        '原训练集800条中，351条在保持全文提示词不变时还有128个同文档续写token且总长不超过2048；全部纳入，原100条开发文档不参与。固定K20/40及原8个完整剪层组合。每个组合的标签来自实际联合前向，不相加单层代价；回答内固定选层。', '',
        '四折按正文及开头哈希分组，每折约228条拟合、35条内层选模、88条外折复核，三个种子579/580/581。四组为first32/mean128监督各配全文embedding及无输入constant。每组24轮、相同初始化参数、数据顺序、batch与优化器。first32取同一长序列首窗，mean128取四个32-token窗等权均值，消除短/长序列数值形状混杂。监督范围与提示词观察范围是两个不同概念：两组路由器都只读取完整提示词，不读取未来token。', '',
        '训练拟合标签决定各组先验与缩放；所有检查点均用共同内层128-token KL选择，epoch0/1/4/...24，严格改善才保存。共同静态参考只由拟合子集的128-token KL均值选择；门槛也统一在内层128-token KL上选。训练接口的fit/selection标签按互斥ID构造，不混用。无输入组参数量与更新预算相同，不含内容或长度。mean3表示三个独立种子的质量均值，不是集成推理。', '',
        '| 外折策略 | 全128 KL，越低越好 | 相对折内静态改善 | 探索性95%区间 |', '|---|---:|---:|---|']
    for row in summary['outer_summary']:
        if row['metric'] == 'kl' and row['window'] == 'all128':
            lo, hi = row['ci95']; lines.append(f"| {row['policy']} | {row['mean']:.6f} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '| 多窗口embedding比较 | 参考 | 全128 KL改善 | 探索性95%区间 |', '|---|---|---:|---|']
    for row in summary['comparisons']:
        if row['metric'] == 'kl' and row['window'] == 'all128':
            lo, hi = row['ci95']; lines.append(f"| {row['policy']} | {row['baseline']} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '正改善表示多窗口组更好。历史静态索引1和future_known_best128单列：前者已有候选筛选依赖，后者读取真实未来，仅为同库收益空间诊断。主要对比预先固定为mean128_embedding_gate与first32_embedding_gate，同时必须检查constant及折内静态。', '',
        '这是反复查看过的旧训练文档上的开发诊断，候选库曾用其中部分数据筛选，不是独立测试。351条有续写长度选择偏倚；区间仅按外折文档重采样，忽略训练与折间依赖，未做多重比较校正。不能把更低训练标签损失直接解释为更好动态决策。没有新增恢复微调、任务准确率或加速测量。', '',
        'GPU0—3 /42CPU，四worker各10核、控制器2核。所有旧源文件、报告和结果验哈希保全；新增参数、缓存、语料和逐文档记录仅本地保存，公开程序、协议、审计与聚合结果。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [独立审计](../{out.relative_to(root)}/independent_audit.json)']
    with (root / 'analysis' / f'{out.name}_results.zh-CN.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
