"""Stage25 train800-only crossfit; conditional fixed bank, nested model selection."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load
from crossfit_routing_core_v1 import SEEDS, RIDGES, choose, cpu_checks, gate_fit, gated, grouped_folds, neural_fit, ridge_scores
from full_context_experiment_v1 import Workers, preserve, read_lines, resources, setup_worker, thread_record, verify_preserved, write
from headroom_ruler_core_v1 import paired
from horizontal_quality_v3 import pin_threads
from stage1_benchmark_v2 import sha256, stamp


def worker(rank, cfg, inbox, messages):
    setup_worker(rank); out = Path(cfg['output']); device = f'cuda:{rank}'
    fold = cfg['folds'][rank]
    old = Path(cfg['old_run'])
    all_embeddings = torch.load(old / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    embeddings = {i: all_embeddings[i] for i in cfg['train800_ids']}; del all_embeddings
    records = load(Path(cfg['data_run']) / 'data_manifest.json')['records']
    for i, value in embeddings.items():
        assert value.shape[1] == records[i]['prompt_length']
    labels = {r['sample']: r for j in range(6) for r in read_lines(old / f'private_labels_{j:03d}.jsonl') if r['sample'] in embeddings}
    for seed in SEEDS:
        neural_fit(out, cfg, fold, embeddings, labels, seed, device, messages, rank)
    # Fixed random feature extraction has no label fit and consumes ALL prompt tokens.
    generator = torch.Generator(device='cpu').manual_seed(250579)
    projection = (torch.randn(5120, 128, generator=generator) / np.sqrt(5120)).to(device)
    features = []
    with torch.inference_mode():
        for i in cfg['train800_ids']:
            x = embeddings[i][0].to(device).float() @ projection
            features.append(torch.cat([x.mean(0), x.std(0, unbiased=False), x.new_tensor([np.log(len(x)), np.log(len(x)) ** 2])]).cpu())
    with (out / f'private_f{rank}_features.pth').open('xb') as stream:
        torch.save(torch.stack(features), stream)
    write(out / f'worker_{rank:03d}.json', {'passed': True, 'fold': rank,
        'full_prompt_length_verified': True, 'feature_sha256': sha256(out / f'private_f{rank}_features.pth'), **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def finalize(out, cfg):
    ids = cfg['train800_ids']; lookup = {s: i for i, s in enumerate(ids)}
    old = Path(cfg['old_run'])
    labels = {r['sample']: r for j in range(6) for r in read_lines(old / f'private_labels_{j:03d}.jsonl') if r['sample'] in lookup}
    costs = np.array([labels[s]['kl'] for s in ids]); nll = np.array([labels[s]['nll'] for s in ids])
    evaluations = []; choices = []; scored = []; features_reference = None
    for fold in cfg['folds']:
        f = fold['fold']; fit_rows = np.array([lookup[s] for s in fold['fit_ids']]); sel_rows = np.array([lookup[s] for s in fold['selection_ids']]); outer_rows = np.array([lookup[s] for s in fold['outer_ids']])
        all_rows = np.concatenate([fit_rows, sel_rows, outer_rows]); fitn, seln = len(fit_rows), len(sel_rows)
        relative = costs[fit_rows] - costs[fit_rows, :1]
        baseline = int(costs[fit_rows].mean(0).argmin()); scale = max(float(relative[:, 1:].std(ddof=1)), .03)
        bounds = costs[outer_rows].min(1)
        if features_reference is None:
            features_reference = torch.load(out / f'private_f{f}_features.pth', weights_only=True).numpy().astype(np.float64)
        else:
            other_features = torch.load(out / f'private_f{f}_features.pth', weights_only=True).numpy()
            assert np.allclose(features_reference, other_features, atol=1e-6)
        predictions = []
        for seed in SEEDS:
            name = f'f{f}_s{seed}'; receipt = load(out / f'{name}_fit.json')
            path = out / f'private_{name}_scores.pth'; assert sha256(path) == receipt['score_sha256']
            saved = torch.load(path, weights_only=True)
            assert saved['ids'] == fold['fit_ids'] + fold['selection_ids'] + fold['outer_ids']
            score = saved['scores'].numpy().astype(np.float64)
            selection_score = score[fitn:fitn + seln]
            fitted_gate = gate_fit(selection_score, costs[sel_rows], baseline, scale)
            predictions.append((f'neural_raw_s{seed}', score.argmin(1)))
            predictions.append((f'neural_gate_s{seed}', gated(score, fitted_gate, baseline, scale)))
            choices.append({'fold': f, 'policy': f'neural_gate_s{seed}', 'best': fitted_gate,
                'baseline_index': baseline, 'scale': scale, 'selected_epoch': receipt['best']['epoch']})
            history = [r for r in load(out / f'{name}_history.json') if 'selection_kl' in r]
            best = min(history, key=lambda r: r['selection_kl']); assert best['epoch'] == receipt['best']['epoch']
            assert abs(costs[sel_rows, score[fitn:fitn + seln].argmin(1)].mean() - best['selection_kl']) < 1e-7
        for family in ('ridge_content', 'ridge_length', 'ridge_shuffled'):
            features = features_reference if family != 'ridge_length' else features_reference[:, -2:]
            permutation = np.random.default_rng(250900 + f).permutation(fitn) if family == 'ridge_shuffled' else None
            grid = []
            for penalty in RIDGES:
                score = ridge_scores(features, fit_rows, sel_rows, outer_rows, costs, penalty, permutation)
                gate = gate_fit(score[fitn:fitn + seln], costs[sel_rows], baseline, scale)
                grid.append({'penalty': penalty, 'gate': gate, 'score': score})
            best = min(grid, key=lambda r: (r['gate']['selection_kl'], r['gate']['switches'], r['penalty']))
            predictions.append((family, gated(best['score'], best['gate'], baseline, scale)))
            choices.append({'fold': f, 'policy': family, 'penalty': best['penalty'], 'baseline_index': baseline,
                'scale': scale, 'best': best['gate'], 'grid': [{'penalty': g['penalty'], 'gate': g['gate']} for g in grid]})
        predictions += [('fold_static', np.full(len(all_rows), baseline)), ('future_known_best', costs[all_rows].argmin(1))]
        for name, routes in predictions:
            for cohort, local, global_rows in [('fit', np.arange(fitn), fit_rows), ('selection', np.arange(fitn, fitn + seln), sel_rows),
                ('outer', np.arange(fitn + seln, len(all_rows)), outer_rows)]:
                chosen = routes[local]; own = costs[global_rows, chosen]; static = costs[global_rows, baseline]
                evaluations.append({'fold': f, 'policy': name, 'cohort': cohort, 'documents': len(global_rows),
                    'kl': float(own.mean()), 'nll': float(nll[global_rows, chosen].mean()), 'static_kl': float(static.mean()),
                    'difference': float((static - own).mean()), 'baseline_index': baseline,
                    'switches': int((chosen != baseline).sum()), 'beneficial': int((static > own).sum()), 'harmful': int((static < own).sum()),
                    'unique_routes': len(set(chosen.tolist()))})
                for sample_row, route in zip(global_rows, chosen):
                    scored.append({'sample': ids[sample_row], 'fold': f, 'policy': name, 'cohort': cohort,
                        'route_index': int(route), 'baseline_index': baseline, 'kl': float(costs[sample_row, route]),
                        'nll': float(nll[sample_row, route]), 'static_kl': float(costs[sample_row, baseline]),
                        'static_nll': float(nll[sample_row, baseline])})
    summary_rows = []
    outer = [r for r in scored if r['cohort'] == 'outer']
    for name in ['neural_raw_mean3', 'neural_gate_mean3', 'ridge_content', 'ridge_length', 'ridge_shuffled', 'fold_static', 'future_known_best']:
        records = []
        for sample in sorted({r['sample'] for r in outer}):
            group = [r for r in outer if r['sample'] == sample and
                (r['policy'] in [f"{'neural_raw' if name == 'neural_raw_mean3' else 'neural_gate'}_s{s}" for s in SEEDS]
                    if name in ('neural_raw_mean3', 'neural_gate_mean3') else r['policy'] == name)]
            assert len(group) == (3 if name in ('neural_raw_mean3', 'neural_gate_mean3') else 1)
            records.append({'kl': float(np.mean([r['kl'] for r in group])), 'nll': float(np.mean([r['nll'] for r in group])),
                'static_kl': group[0]['static_kl'], 'static_nll': group[0]['static_nll']})
        for metric in ('kl', 'nll'):
            own = np.array([r[metric] for r in records]); static = np.array([r['static_' + metric] for r in records])
            summary_rows.append({'policy': name, 'metric': metric, 'own_mean': float(own.mean()), 'static_mean': float(static.mean()),
                **paired(static - own, None, 250579),
                'limits': 'Exploratory document bootstrap ignores fold dependence and training uncertainty; fixed candidate bank was historically chosen on these training documents.'})
    # Error-mode summary: in-fit wins do not establish out-of-fold learnability.
    with (out / 'private_crossfit_rows.jsonl').open('x') as stream:
        for row in scored:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write(out / 'policy_choices.json', choices)
    summary = {'engineering_only': bool(cfg['probe']), 'evaluated_outer_documents': len({r['sample'] for r in outer}), 'out_of_fold800': summary_rows, 'per_fold': evaluations, 'limits': cfg['limits']}
    write(out / 'summary.json', summary)
    return summary


def make_report(root, out, summary):
    destination = root / 'analysis' / f'{out.name}_results.zh-CN.md'
    text = ['# 阶段25：原800训练文档内部的交叉验证诊断', '',
        '只使用原800训练文档，不使用原100条开发文档选择配置或报告本轮指标。按文档正文/开头哈希连通分组，四折；每折约520条梯度训练、80条内部选模、200条外折复核。K20/40、8个完整组合、全文embedding与回答内固定路径不变。', '',
        '候选库历史上用这800条中的256条构建，因此这是固定既有候选库条件下的可学习性诊断，非完全无泄漏的最终测试。每折静态组合仅按该折梯度训练子集的均值联合KL选择，不用外折标签。', '',
        '神经路由器保持Stage20未来32-token完整组合KL监督、模型结构、优化器、24轮和检查点评测频率；三个种子。检查点由内80条选定；切换门槛及始终静态选项也只由内80条选定。简单Ridge模型使用固定随机投影后全文特征均值/标准差及长度，另列只看长度和打乱拟合标签的对照。所有特征标准化、先验和回归仅在本折梯度训练数据拟合。', '',
        '| 外折合并800文档策略 | KL | 折内训练选静态KL | 静态减策略 | 探索性95%区间 |',
        '|---|---:|---:|---:|---|']
    for r in summary['out_of_fold800']:
        if r['metric'] == 'kl':
            lo, hi = r['ci95']; text.append(f"| {r['policy']} | {r['own_mean']:.5f} | {r['static_mean']:.5f} | {r['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    text += ['', 'mean3表示每文档三个独立种子的指标平均，不是运行三模型集成选层。future_known_best读取未来真实KL，仅表示同库诊断收益空间，不是线上路由器。逐折的拟合、选择和外折KL/NLL、切换有益/有害次数、逐种子结果见汇总。', '',
        '文档bootstrap区间未覆盖训练和折间依赖，不作多重比较校正。若训练内收益明显而外折没有，则优先考虑数据量、标签波动或输入表征不足；不能据一个负诊断证明动态剪枝不可行。', '',
        '4GPU/42CPU，worker各10核、控制器2核。不重算大模型标签、不训练OPT、不做吞吐/显存或独立泛化声明。旧文件验哈希保全；公开自写源码和聚合信息，权重、embedding、逐文档记录不上传。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [选模规则](../{out.relative_to(root)}/policy_choices.json) · [校验](../{out.relative_to(root)}/verification.json)']
    with destination.open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    return destination


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true'); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    cpus = resources(); pin_threads(cpus[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    preserved = preserve(root, out)
    old = root / 'runs/stage20_codebook_refine_20261112_02_main'; previous = load(old / 'protocol.json')
    ids = previous['train_ids']; records = load(Path(previous['data_run']) / 'data_manifest.json')['records']
    assert sha256(old / 'private_embeddings.pth') == previous['embedding_sha256']
    folds = grouped_folds(ids, records)
    if args.probe:
        folds = [{**f, 'fit_ids': f['fit_ids'][:16], 'selection_ids': f['selection_ids'][:8], 'outer_ids': f['outer_ids'][:8]} for f in folds]
        ids = sorted({s for f in folds for key in ('fit_ids', 'selection_ids', 'outer_ids') for s in f[key]})
    sources = ['crossfit_routing_core_v1.py', 'run_crossfit_routing_four_gpu_v2.py', 'launch_crossfit_routing_v2.py']
    evidence = [old / 'protocol.json', old / 'candidate_bank.json', old / 'private_embeddings.pth', Path(previous['data_run']) / 'data_manifest.json']
    evidence += [old / f'private_labels_{i:03d}.jsonl' for i in range(6)]
    cfg = {'output': str(out), 'old_run': str(old), 'data_run': previous['data_run'], 'train800_ids': ids,
        'folds': folds, 'epochs': 1 if args.probe else 24, 'probe': args.probe,
        'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'evidence_sha256': {str(p): sha256(p) for p in evidence}, 'resources': '4GPU/42CPU, worker10/controller2',
        'split_seed': 250579, 'selection_seeds': [250580 + i for i in range(4)], 'neural_seeds': SEEDS,
        'selection': 'Only each fold inner80 chooses epoch, ridge penalty or gate. Outer200 costs excluded from fitting/normalization/selection. Historical dev100 not used.',
        'limits': 'Conditional fixed bank chosen historically from256 of original train800; not fully independent candidate search CV. Exploratory bootstrap ignores fold/model-fit dependence. Same reviewed training source, not external generalization. Full prompt embeddings include position. No speed claim.'}
    assert set(ids) <= set(previous['train_ids']) and not set(ids) & set(previous['validation_ids'])
    write(out / 'protocol.json', cfg); write(out / 'cpu_checks.json', cpu_checks())
    stamp('crossfit_start', probe=args.probe, folds=[{k: len(f[k]) for k in ('fit_ids', 'selection_ids', 'outer_ids')} for f in folds])
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'done')
    summary = finalize(out, cfg)
    for path, expected in cfg['evidence_sha256'].items():
        assert sha256(path) == expected
    verify_preserved(out, preserved)
    write(out / 'verification.json', {'passed': True, 'probe': args.probe, 'folds': 4, 'seeds_per_fold': 3,
        'full_prompt_lengths_verified': True, 'historical_dev100_unused': True, 'fold_group_disjoint': True,
        'old_evidence_unchanged': True, 'fixed_library_selection_caveat_disclosed': True})
    if not args.probe:
        make_report(root, out, summary)
    stamp('crossfit_complete', probe=args.probe, run=str(out))


if __name__ == '__main__':
    main()
