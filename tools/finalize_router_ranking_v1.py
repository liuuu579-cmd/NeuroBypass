"""Create a fresh stage5 report, CSV tables and public-file manifest after audit.

This program does not train, tune, alter prior outputs or publish remotely.
The existing verifier must have succeeded; every new output is exclusive-create.
"""
import argparse
import csv
import json
from pathlib import Path

from stage1_benchmark_v2 import sha256, write_json


ARMS = {'classification': '分类 CE', 'pairwise': 'KL 差加权排序'}
POLICIES = {'router': '真实提示词路由', 'shuffled_mean4': '交换提示词方案 4 次平均',
    'static_learned': '同 800 文档固定输入静态', 'static_old': '旧 16 文档校准静态',
    'full': '完整 OPT', 'oracle_greedy': '真实续写贪心 oracle'}
POPULATIONS = {'validation_all100': '全部 100 个开发验证文档',
    'validation_selection32': '用于选择检查点的 32 个文档',
    'validation_other68': '不用于选择检查点的其他 68 个开发文档'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    audit = json.loads((run / 'verification.json').read_text())
    if protocol['probe'] or audit['probe'] or audit['summary_sha256'] != sha256(run / 'summary.json'):
        raise ValueError('Require audited production results')
    if summary['test_used'] or audit['test_used']:
        raise ValueError('Development protocol unexpectedly used the old test split')

    def quality(pop, arm, policy='router', group='mean_seeds'):
        return next(r for r in summary['quality'] if r['population'] == pop
            and r['seed_group'] == group and r['arm'] == arm and r['policy'] == policy)

    def comparison(pop, arm, reference_arm, policy, metric='continuation_kl', group='mean_seeds'):
        return next(r for r in summary['paired_comparisons'] if r['population'] == pop
            and r['seed_group'] == group and r['arm'] == arm and r['reference_arm'] == reference_arm
            and r['reference_policy'] == policy and r['metric'] == metric)

    delta = comparison('validation_all100', 'pairwise', 'classification', 'router')
    lo, hi = delta['bootstrap_95pct']
    if hi < 0:
        verdict = '排序组的三种子均值 KL 更低，当前开发文档 bootstrap 区间完全小于零。'
    elif lo > 0:
        verdict = '排序组的三种子均值 KL 更高，当前开发文档 bootstrap 区间完全大于零。'
    else:
        verdict = '本次未建立排序组相对分类组的明确 KL 改善：三种子均值的开发文档 bootstrap 区间跨零。'
    all_pop = 'validation_all100'
    lines = ['# Stage5：分类与 KL 差加权排序的三种子对照', '', verdict, '',
        f"100 个开发文档上，分类组 KL={quality(all_pop, 'classification')['continuation_kl']:.6f}，"
        f"排序组 KL={quality(all_pop, 'pairwise')['continuation_kl']:.6f}；排序减分类 Δ={delta['mean_delta']:.6f}，"
        f'95% 文档 bootstrap 区间 [{lo:.6f}, {hi:.6f}]。', '',
        '这里的均值是三个独立训练种子的评价成本先按目标文档平均，不是线上把三个 Router 集成，也不是把三个种子当成 300 个独立文档。所有验证文档都已用于开发；结果不能作为独立泛化证据。', '',
        '## 固定实验条件', '',
        '- 冻结 OPT-13B；Router 只看 128-token 提示词 embedding，每轮贪心选 6/40 层跳过，保护层 0、1。选定方案用于本轮 prefill 和全部续写。新一轮换方案则重建整个上下文的 KV cache。',
        '- 复用 stage4 的 800 个训练文档，每文档原 6 个专家状态加 3 个后段分叉状态；候选成本始终针对完整的“已跳层集合 + 当前候选层”，保留层间协同。不新增训练标签或验证标签。',
        '- 每个种子 579、580、581 的两组共享初始化、width256 的 2,586,665 参数网络、样本顺序、batch16、24 epochs、每 epoch 50 更新，共 1200 更新。AdamW lr2e-4、weight_decay0.01、5% warmup + cosine、梯度裁剪 1。',
        '- 分类目标：原 soft-target CE + 0.25 hard greedy-choice CE。排序目标：每状态对全部有效候选对计算 `Σ(c_j-c_i)_+ softplus(s_i-s_j) / Σ(c_j-c_i)_+`，然后平均状态。低成本应获得更低 Router 分数；保护/已选层及等成本对权重为零。',
        '- 两目标未单独调参；梯度和损失尺度可能不同。排序覆盖全部不等成本候选对，损害最大的候选对权重更大，未限制为最终最优候选附近的排序。训练损失的绝对数值不能跨目标直接比较。',
        '- 原固定 32 个文档在 epochs 1、4、8、12、16、20、24 的真实续写 KL 用于选检查点；其余 68 个只排除检查点选择，仍属于重复使用的开发验证集。旧 100 个测试文档未参与本轮。', '',
        '## 续写质量', '',
        'KL、NLL 越低越好；PPL 为 exp(平均 NLL)。指标基于 C4 自然文本的 32-token 真实续写，不是自由生成、聊天或下游任务得分。', '']
    for pop, name in POPULATIONS.items():
        lines += [f'### {name}', '', '| 训练目标 | 方案 | KL | NLL | PPL |',
                  '| --- | --- | ---: | ---: | ---: |']
        for arm in ARMS:
            for policy in ['router', 'shuffled_mean4']:
                row = quality(pop, arm, policy)
                lines.append(f"| {ARMS[arm]} | {POLICIES[policy]} | {row['continuation_kl']:.6f} | {row['continuation_nll']:.6f} | {row['perplexity']:.4f} |")
        for policy in ['static_learned', 'static_old', 'full', 'oracle_greedy']:
            row = quality(pop, 'classification', policy)
            lines.append(f"| 共用对照 | {POLICIES[policy]} | {row['continuation_kl']:.6f} | {row['continuation_nll']:.6f} | {row['perplexity']:.4f} |")
        lines.append('')

    lines += ['## 每个种子的结果', '',
        '所有检查点均由事先指定的 32 个文档选择；不按下面的 100/68 文档结果重新挑选种子或检查点。', '',
        '| 种子 | 目标 | 选择 epoch | 选择 32 KL | 全部 100 KL | 其他 68 KL | 全部 100 NLL | 全部 100 PPL | 100 文档不同掩码数 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    checkpoints = []
    for seed in protocol['seeds']:
        for arm in ARMS:
            meta = next(r for r in summary['runs'] if r['seed'] == seed and r['arm'] == arm)
            history = json.loads((run / f'{arm}_seed{seed}_history.json').read_text())
            best = min((r for r in history if 'continuation_kl' in r), key=lambda r: r['continuation_kl'])
            row = quality(all_pop, arm, group=str(seed))
            other = quality('validation_other68', arm, group=str(seed))
            lines.append(f"| {seed} | {ARMS[arm]} | {best['epoch']} | {meta['best_selection_kl']:.6f} | {row['continuation_kl']:.6f} | {other['continuation_kl']:.6f} | {row['continuation_nll']:.6f} | {row['perplexity']:.4f} | {meta['unique_validation_masks']} |")
            checkpoints.append(f"- {arm} / seed{seed}：`{Path(meta['best_checkpoint']).name}`，SHA256=`{meta['best_checkpoint_sha256']}`。")
    lines += ['', '## 文档配对比较', '',
        '差值均为本行 Router 减对照，负值更好。每个目标文档先平均三种子成本，交换对照还先平均 4 个共用的固定 donor 映射；然后对目标文档进行 10000 次配对 bootstrap。下表区间不校正多重比较、文档依赖或全部训练种子不确定性。', '']
    definitions = [('排序减分类', 'pairwise', 'classification', 'router'),
        ('分类减交换', 'classification', 'classification', 'shuffled_mean4'),
        ('排序减交换', 'pairwise', 'pairwise', 'shuffled_mean4'),
        ('分类减学习静态', 'classification', 'classification', 'static_learned'),
        ('排序减学习静态', 'pairwise', 'pairwise', 'static_learned')]
    for pop, name in POPULATIONS.items():
        lines += [f'### {name}', '', '| 比较 | 指标 | 差值 | 95% 区间 | 文档差值小于零比例 |',
                  '| --- | --- | ---: | --- | ---: |']
        for label, arm, reference_arm, policy in definitions:
            for metric in ['continuation_kl', 'continuation_nll']:
                row = comparison(pop, arm, reference_arm, policy, metric)
                left, right = row['bootstrap_95pct']
                lines.append(f"| {label} | {metric} | {row['mean_delta']:.6f} | [{left:.6f}, {right:.6f}] | {row['fraction_negative']:.1%} |")
        lines.append('')

    lines += ['### 种子敏感性：排序减分类', '',
        '| 种子 | 全部 100 KL 差 | 全部 100 NLL 差 | 其他 68 KL 差 | 其他 68 NLL 差 |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for seed in protocol['seeds']:
        fields = [comparison(pop, 'pairwise', 'classification', 'router', metric, str(seed))['mean_delta']
            for pop in ['validation_all100', 'validation_other68'] for metric in ['continuation_kl', 'continuation_nll']]
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(f'{value:.6f}' for value in fields) + ' |')

    reproduction = audit['historical_ce_seed579_reproduction']
    lines += ['', '## 来源与保全审计', '',
        f"6 项损失和汇总测试、微型运行审计及正式审计均通过。正式审计覆盖 {audit['verified_training_runs']} 次训练、{audit['verified_epochs']} 个 epoch、{audit['verified_quality_rows']} 条质量记录；初始化、参数量、更新次数按种子匹配，分叉和专家标签保持不变。", '',
        f"seed579 分类组对上一阶段分叉分类组：{reproduction['same_routes']}/{reproduction['documents']} 个文档的跳层集合相同，最大逐文档 KL 绝对差={reproduction['max_kl_abs_difference']:.9g}。这项复现核验用于确认原目标控制链。", '',
        f"正式运行前记录的 {audit['preserved_existing_files']} 个既有源文件/研究记录哈希均保持不变，私有输入 token 数组及选中检查点哈希也已核验。耗时 {summary['elapsed_seconds'] / 60:.2f} 分钟。", '',
        '资源入口分配物理 GPU 1、3，整个进程树 CPU0–20 共 21 核，每个 worker 10 个 torch 线程。仅新增独立程序、协议和结果文件；没有修改旧源文件。私有权重、输入 token 数组、原始语料和生成 token 序列不发布。', '',
        '检查点记录：', '', *checkpoints, '',
        f"运行目录：`{run.relative_to(root)}`；训练程序：`tools/train_router_ranking_v1.py`；详细质量与配对表见 `analysis/stage5_ranking_quality_20261106.csv` 和 `analysis/stage5_ranking_paired_20261106.csv`。", '',
        '## 解释边界与后续决策', '',
        '本轮只回答固定标签、容量和预算下换训练目标的效果。三种子不是完整的种子分布；文档 bootstrap 不能弥补重复使用验证集，也不能证明新文档泛化。真实续写贪心 oracle 看到了后缀，不是可部署方案，不保证全局最优，其相对差距也未必能仅由提示词预测。', '',
        '是否学会使用输入，应检查自身路由与共用置换方案的配对差异；路由掩码多样性本身不代表输入匹配有效。是否优于静态，应同时检查同数据学习静态对照，而不是只比旧 16 条校准静态。KL 改善也不能自动视为 NLL、任务质量或生成收益改善。', '',
        '保留本次全部种子和不确定结果，不继续按这个开发集搜索损失超参数。若三种子和其他 68 文档仍不能一致支持输入匹配及静态优势，优先检查“同一提示词、不同真实续写”的条件贪心标签稳定性，并增加按来源分组的新训练文档；若观察到一致优势，先冻结方案再做新独立评估。恢复微调应作为后续独立变量，静态和动态均须获得匹配预算。', '',
        '本轮未新增端到端延迟或显存测量，不能据质量结果宣称加速。固定跳过 6 层的推理实现、每轮路由和换轮重建缓存开销仍需单独计时。', '']

    report_path = root / 'analysis/STAGE5_RANKING_RESULTS_20261106.zh-CN.md'
    quality_path = root / 'analysis/stage5_ranking_quality_20261106.csv'
    paired_path = root / 'analysis/stage5_ranking_paired_20261106.csv'
    manifest = root / 'analysis/stage5_ranking_upload_20261106.json'
    for path in [report_path, quality_path, paired_path, manifest]:
        if path.exists():
            raise FileExistsError('Refusing to replace existing evidence: ' + str(path))
    with report_path.open('x') as handle:
        handle.write('\n'.join(lines))
    for path, rows in [(quality_path, summary['quality']), (paired_path, summary['paired_comparisons'])]:
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    files = [str(p.relative_to(root)) for p in [report_path, quality_path, paired_path]]
    files += ['tools/train_router_ranking_v1.py', 'tools/test_router_ranking_v1.py',
        'tools/verify_router_ranking_v1.py', 'tools/finalize_router_ranking_v1.py',
        'analysis/STAGE5_RANKING_PROTOCOL_20261106.zh-CN.md']
    for directory in [root / 'runs/stage5_ranking_probe_20261106_01', run]:
        files += [str(p.relative_to(root)) for p in sorted(directory.iterdir())
            if p.is_file() and not p.name.startswith('private_') and p.suffix in {'.json', '.jsonl', '.log'}]
    files.append(str(manifest.relative_to(root)))
    write_json(manifest, files)
    print(json.dumps({'report': str(report_path), 'manifest': str(manifest),
        'public_files': len(files), 'summary_sha256': audit['summary_sha256']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
