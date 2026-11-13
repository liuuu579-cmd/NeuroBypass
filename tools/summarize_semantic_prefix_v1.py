"""Append a factual interpretation of the completed Stage26 pilot."""
import json
from pathlib import Path

import numpy as np

from full_context_experiment_v1 import write


def load(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]; out = root / 'runs/stage26_semantic_prefix_20261113_01_main'
    summary = load(out / 'summary.json'); audit = load(out / 'independent_audit.json'); assert audit['passed']
    index = {(r['policy'], r['metric']): r for r in summary['outer_summary']}
    comparisons = {(r['policy'], r['baseline'], r['metric']): r for r in summary['comparisons']}
    seeds = []; cohort = []; fold_rows = []
    for seed in (579, 580, 581):
        row = {'seed': seed}
        for arm in ('embedding', 'hidden2', 'constant'):
            group = [r for r in summary['per_fold'] if r['cohort'] == 'outer' and r['policy'] == f'{arm}_gate_s{seed}']
            row[arm + '_kl'] = float(np.mean([r['kl'] for r in group]))
        row['hidden2_improvement_over_constant'] = row['constant_kl'] - row['hidden2_kl']
        seeds.append(row)
    for name in ('fit', 'selection', 'outer'):
        for arm in ('embedding', 'hidden2', 'constant'):
            group = [r for r in summary['per_fold'] if r['cohort'] == name and r['policy'].startswith(arm + '_gate_s')]
            cohort.append({'cohort': name, 'arm': arm, 'kl': float(np.mean([r['kl'] for r in group])),
                'improvement_over_fold_static': float(np.mean([r['improvement'] for r in group])),
                'beneficial': sum(r['beneficial'] for r in group), 'harmful': sum(r['harmful'] for r in group),
                'switches': sum(r['switches'] for r in group)})
    for fold in range(4):
        record = {'fold': fold}
        for arm in ('embedding', 'hidden2', 'constant'):
            group = [r for r in summary['per_fold'] if r['cohort'] == 'outer' and r['fold'] == fold and r['policy'].startswith(arm + '_gate_s')]
            record[arm + '_kl'] = float(np.mean([r['kl'] for r in group]))
        fold_rows.append(record)
    constants = [r for r in summary['per_fold'] if r['policy'].startswith('constant_gate_s')]
    assert all(r['route_histogram'] == {'1': r['documents']} for r in constants)
    semantic = index['hidden2_gate_mean3', 'kl']; constant = index['constant_gate_mean3', 'kl']
    primary = comparisons['hidden2_gate_mean3', 'embedding_gate_mean3', 'kl']
    content_effect = comparisons['hidden2_gate_mean3', 'constant_gate_mean3', 'kl']
    result = {'seed_results': seeds, 'fold_results': fold_rows, 'cohort_results': cohort,
        'constant_all_cohorts_all_seeds_same_route_index': 1, 'primary_comparison': primary,
        'semantic_vs_no_input': content_effect,
        'interpretation': 'No stable improvement from frozen two-layer features over embedding or matched no-input control. No-input control selects one fixed mask for every document; small gain over fold-static cannot establish prompt-conditioned benefit.',
        'no_additional_training_or_outer_tuning': True, 'limits': summary['limits']}
    write(out / 'diagnostic_interpretation.json', result)
    text = ['# 阶段26结论：两层语义表征尚未带来稳定的动态收益', '',
        '已完成四折、每折三个种子的小试，并通过独立复核。OPT冻结，完整提示词经过所有候选都保留的第0、1层，取第1层输出作为路由输入；候选仍是K20/40的8个完整组合，监督仍是完整组合未来32-token联合KL。每个回答固定一个路径。', '',
        '| 外折合并800条旧训练文档 | KL | NLL |', '|---|---:|---:|']
    names = [('fold_static', '每折训练520条选静态'), ('embedding_gate_mean3', '原embedding路由＋门槛'),
        ('hidden2_gate_mean3', '两层语义路由＋门槛'), ('constant_gate_mean3', '无输入常量路由对照')]
    for policy, label in names:
        text.append(f"| {label} | {index[policy, 'kl']['mean']:.6f} | {index[policy, 'nll']['mean']:.6f} |")
    lo, hi = primary['ci95']; clo, chi = content_effect['ci95']
    text += ['', f"相对原embedding路由，语义路由KL平均改善{primary['difference']:.6f}，探索性95%区间[{lo:+.6f}, {hi:+.6f}]。相对无输入常量路由只改善{content_effect['difference']:.6f}，区间[{clo:+.6f}, {chi:+.6f}]。两者均跨零，不能认为改输入已经带来可靠优势。NLL点估计中常量对照还略优于语义路由。", '',
        '## 常量对照澄清了什么', '',
        '所有折、所有种子、所有文档上，常量路由最终都固定选择候选索引1。它没有读取提示词的内容或长度。因此，语义路由与它之间的差异才有助于判断输入适应是否有效；仅超过每折520条均值KL选出的静态，不能单独证明动态选层的价值。', '',
        '四折的均值KL静态有一折选择索引0，其余选择索引1；常量路由的Huber回归、内部检查点选择也能在该折选到索引1。常量对照恢复同一个强固定组合，已经取得与语义路由几乎相同的平均质量。不能将这部分固定组合或训练目标差异当成逐文档动态收益。', '',
        '| 随机种子 | 语义路由KL | 常量对照KL | 常量减语义，正为语义更好 |', '|---|---:|---:|---:|']
    for row in seeds:
        text.append(f"| {row['seed']} | {row['hidden2_kl']:.6f} | {row['constant_kl']:.6f} | {row['hidden2_improvement_over_constant']:+.6f} |")
    text += ['', '## 当前判断与下一步', '',
        '这次只更换输入表示的尝试，没有解决稳定选层的问题。它不证明语义输入无效，也不证明动态剪枝不可行；只能说明在固定8路径、当前数据量、相同路由器与更新预算下，前两层隐藏状态没有建立可重复的优势。没有再根据这轮结果增加深度、调门槛或挑选最好种子。', '',
        '如果继续路线1，下一步更值得先检验监督标签的可预测性：对预先指定的一小批训练文档，使用相同提示词但不重叠的多个后续片段，分别计算全部8个完整组合的联合KL，量化最佳组合及收益排序是否稳定。若单段32-token标签的排名随续写窗口强烈变化，应先改进标签估计；若排名稳定，再考虑更有针对性的表示或路由目标。这只是下一步建议，本轮没有启动该实验。', '',
        '## 执行与解释边界', '',
        '每折520训练、80选模、200复核，24轮、每模型792次更新，三个种子。embedding组复用Stage25精确匹配的检查点；两层语义和常量组重新训练。三组对应种子的初始参数逐位相同，优化器、数据顺序、检查点和门槛网格一致。mean3是每文档独立种子质量平均，不是多模型集成部署。', '',
        f"工程检查在64、129、513、2016 token上与完整OPT的hidden_states[2]逐位一致。独立审计重建{audit['private_quality_rows_reconstructed']}条文档/策略/群体指标，验证全文共{audit['full_prompt_tokens_verified']}个提示词token，复推12条长短提示词特征及108次路由输出，误差均为零；{audit['preserved_files']}个旧文件验哈希保全。", '',
        '本轮仍是已查看过的原800训练文档上的开发诊断，不是独立泛化测试；原100开发文档没有参与。候选库历史上用过其中256条，bootstrap没有包含训练/折间依赖且未作多重比较校正。没有测吞吐、显存或前缀KV复用收益，没有训练恢复适配器。使用四张GPU与42个CPU核心，全部源文件和历史结果保持不变。', '',
        f"[完整结果](stage26_semantic_prefix_20261113_01_main_results.zh-CN.md) · [聚合诊断](../{out.relative_to(root)}/diagnostic_interpretation.json) · [独立审计](../{out.relative_to(root)}/independent_audit.json)"]
    with (root / 'analysis/STAGE26_SEMANTIC_PREFIX_CONCLUSION_20261113.zh-CN.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
