"""Append aggregate interpretation of the completed, audited crossfit diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np

from full_context_experiment_v1 import read_lines, write


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = args.run.resolve()
    summary = load(out / 'summary.json'); audit = load(out / 'independent_audit.json'); assert audit['passed']
    cfg = load(out / 'protocol.json'); ids = cfg['train800_ids']; old = Path(cfg['old_run'])
    labels = {r['sample']: r for rank in range(6) for r in read_lines(old / f'private_labels_{rank:03d}.jsonl') if r['sample'] in ids}
    costs = np.asarray([labels[i]['kl'] for i in ids]); best800 = int(costs.mean(0).argmin())
    history_reference = float(costs[:, best800].mean())
    folds = []; cohort_rows = []; seed_rows = []
    for cohort in ('fit', 'selection', 'outer'):
        for family in ('neural_raw', 'neural_gate', 'ridge_content', 'ridge_length', 'ridge_shuffled', 'fold_static', 'future_known_best'):
            rows = [r for r in summary['per_fold'] if r['cohort'] == cohort and
                (r['policy'].startswith(family + '_s') if family.startswith('neural_') else r['policy'] == family)]
            cohort_rows.append({'cohort': cohort, 'policy': family, 'kl': float(np.mean([r['kl'] for r in rows])),
                'improvement': float(np.mean([r['difference'] for r in rows])),
                'mean_switch_fraction': float(np.mean([r['switches'] / r['documents'] for r in rows])),
                'beneficial_switch_fraction': sum(r['beneficial'] for r in rows) / max(1, sum(r['switches'] for r in rows)),
                'interpretation': 'Fit/selection are diagnostics; only outer is excluded from fitting and model selection.'})
    for seed in (579, 580, 581):
        rows = [r for r in summary['per_fold'] if r['cohort'] == 'outer' and r['policy'] == f'neural_gate_s{seed}']
        seed_rows.append({'seed': seed, 'kl': float(np.mean([r['kl'] for r in rows])),
            'improvement_over_fold_static': float(np.mean([r['difference'] for r in rows]))})
    for fold in cfg['folds']:
        f = fold['fold']; rows = [r for r in summary['per_fold'] if r['cohort'] == 'outer' and r['fold'] == f and r['policy'].startswith('neural_gate_s')]
        folds.append({'fold': f, 'static_index': rows[0]['baseline_index'],
            'neural_gate_mean3_improvement': float(np.mean([r['difference'] for r in rows]))})
    index = {(r['policy'], r['metric']): r for r in summary['out_of_fold800']}
    dynamic = index['neural_gate_mean3', 'kl']; headroom = index['future_known_best', 'kl']
    bank = load(old / 'candidate_bank.json')
    common_kept = sorted(set(range(40)) - set.union(*(set(r['skips']) for r in bank['routes'])))
    diagnostic = {'cohort_aggregates': cohort_rows, 'seed_outcomes': seed_rows, 'fold_outcomes': folds,
        'historical800_best_static_reference': {'index': best800, 'kl': history_reference,
            'dynamic_minus_historical_static_kl': dynamic['own_mean'] - history_reference,
            'caveat': 'This static path used all800 labels historically, including current outer documents. Context only; not an independent crossfit baseline.'},
        'captured_fraction_of_conditional_headroom': dynamic['difference'] / headroom['difference'],
        'common_kept_layers': common_kept,
        'conclusion': 'Small positive average versus fold-fitted static, uncertainty crosses zero, seed and fold instability. No stable dynamic advantage established; no post-outer tuning performed.'}
    write(out / 'diagnostic_interpretation.json', diagnostic)
    text = ['# 阶段25结论：有组合收益空间，但尚无稳定的输入选层优势', '',
        '本轮已完成四折、每折三个种子的训练集内部交叉验证，并通过独立程序复核。每折520条梯度训练、80条选模、200条外折评估，外折合并恰好覆盖原800条训练文档一次。原100条开发文档未用于本轮训练、选模或指标。每次回答仍固定一个K20完整组合，标签为组合联合KL，保留层间协同。', '',
        '| 外折800文档 | KL，越低越好 | 相对折内静态改善 |', '|---|---:|---:|']
    labels_public = {'fold_static': '各折训练子集选出的固定静态', 'neural_raw_mean3': '神经路由器，三种子均值',
        'neural_gate_mean3': '神经路由器，加内部选定的切换门槛', 'ridge_content': '全文简单特征回归',
        'ridge_length': '仅长度回归', 'ridge_shuffled': '打乱训练标签对照', 'future_known_best': '已知未来真实KL的组合最优，仅诊断'}
    for policy in labels_public:
        row = index[policy, 'kl']; text.append(f"| {labels_public[policy]} | {row['own_mean']:.6f} | {row['difference']:+.6f} |")
    lo, hi = dynamic['ci95']
    text += ['', f"神经路由器加门槛的平均改善为{dynamic['difference']:.6f}，探索性95%区间为[{lo:+.6f}, {hi:+.6f}]，跨过零。三个种子的改善分别为" + '、'.join(f"{r['improvement_over_fold_static']:+.6f}" for r in seed_rows) + '，方向不完全一致。当前不能写成稳定优于静态。', '',
        '## 收益为什么没有转化为稳定优势', '', '| 神经路由器加门槛 | 相对各折静态的平均KL改善 |', '|---|---:|']
    for cohort, name in (('fit', '梯度训练子集'), ('selection', '内部选模子集，有选择偏差'), ('outer', '外折评估')):
        row = next(r for r in cohort_rows if r['cohort'] == cohort and r['policy'] == 'neural_gate')
        text.append(f"| {name} | {row['improvement']:+.6f} |")
    text += ['', f"同库未来已知最优有{headroom['difference']:.6f}的平均KL改善空间，目前动态策略只取得约{100 * diagnostic['captured_fraction_of_conditional_headroom']:.2f}%。这说明候选组合之间存在真实的样本差异；它不保证这些差异能从当前输入表征中预测出来。训练内到外折的收益衰减与过拟合、监督噪声或表征不足相容，尚不能确定唯一原因。", '',
        '简单全文特征回归也只有很小的外折改善，区间跨零；不能据其点估计选择为新的获胜方案。仅长度和打乱标签对照同样未给出可靠优势。后续不能继续查看这800条的外折结果来调门槛，并仍把它们称作未见评估。', '',
        '## 静态基线的解释', '',
        '四折的静态索引依次为' + '、'.join(str(r['static_index']) for r in folds) + '，均只按各自520条训练数据选择。不同折允许选择不同静态，但每一折中的所有评估文档使用同一个静态组合。', '',
        f"作为历史背景，原800条一起选出的强静态是索引{best800}，在这800条上的KL为{history_reference:.6f}；本轮门槛动态为{dynamic['own_mean']:.6f}。因此也不能用本轮微小的正均值宣称已经超过历史强静态。历史静态选择用过全部800条，包含本轮外折文档，所以此项只是背景参考，不替代折内拟合的主基线。", '',
        '## 下一步建议', '',
        '优先做一个改变输入表征的小试：在确认前两层为所有候选共同保留层后，将完整提示词经过这两层的隐藏状态提供给路由器。候选库、K、完整组合监督和回答内固定路径保持一致，先比较它与原embedding路由器、无输入常量路由器及强静态。原embedding已训练过两层小型编码器，因此这项小试的重点是是否获得预训练大模型的语义表征，不能把当前结果解释为完全没有上下文建模。', '',
        '需要预先固定新的训练/选模/复核协议：本轮外折已经查看，后续复用只能视为开发诊断，最终结论需另备未查看文档，并在训练文档上构建候选库。没有必要先做全模型恢复训练或扩大门槛网格；它们不能直接回答输入选层信号是否足够。前两层的计算可计入正常prefill并尝试复用，但是否无额外开销仍需实现和实测。', '',
        '## 证据与限制', '',
        f"独立审计重建了{audit['joint_cost_rows_reconstructed']}条策略/文档/分组指标，复算回归、检查点选择、门槛和bootstrap；12个检查点共24次CPU复推，最大CPU/GPU路由分数差为{audit['maximum_cpu_gpu_score_error']:.3g}。{audit['preserved_files']}个既有文件验哈希保全。使用GPU 0—3和42个CPU核心，未使用另外两张卡。", '',
        '候选库曾在这800条中的256条上筛选，因此本轮是固定候选库条件下的可学习性诊断，不是完全独立的候选搜索交叉验证。文档bootstrap不包含折间与模型训练的不确定性，未做多重比较校正。本轮没有新增吞吐量、显存、下游准确率或模型恢复结果。权重、embedding、原始语料和逐文档路由记录仅保存在本地。', '',
        f"[完整结果](stage25_crossfit_20261113_02_main_results.zh-CN.md) · [聚合诊断](../{out.relative_to(root)}/diagnostic_interpretation.json) · [独立审计](../{out.relative_to(root)}/independent_audit.json)"]
    with (root / 'analysis/STAGE25_CROSSFIT_CONCLUSION_20261113.zh-CN.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    print(json.dumps(diagnostic, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
