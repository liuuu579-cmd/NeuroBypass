"""Finite completion tail: audit, new report and append-only GitHub publication."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from stage1_benchmark_v2 import sha256, write_json
from verify_augmented_router_v1 import verify


def report(root, run, verification):
    data = json.loads((run / 'summary.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    if protocol['probe']:
        raise ValueError('A four-document probe is not a main result')
    def comparison(arm, reference_arm, policy, metric='continuation_kl'):
        return next(r for r in data['paired_comparisons'] if r['population'] == 'validation_all100'
            and r['arm'] == arm and r['reference_arm'] == reference_arm
            and r['reference_policy'] == policy and r['metric'] == metric)
    delta = comparison('augmented', 'baseline', 'router')
    shuffled = comparison('augmented', 'augmented', 'shuffled_mean4')
    static = comparison('augmented', 'augmented', 'static_learned')
    improves = delta['bootstrap_95pct'][1] < 0
    matches = shuffled['bootstrap_95pct'][1] < 0 and static['bootstrap_95pct'][1] < 0
    conclusion = ('增加分叉状态后，验证KL相对相同步数和后期权重的对照有稳定改善。' if improves else
                  '本轮未建立相对相同步数和后期权重对照的稳定验证KL改善。')
    conclusion += ('同时优于交换方案与已有学习静态对照，支持继续做独立记录和多种子验证。' if matches else
                   '本轮尚未同时建立相对交换方案和已有学习静态对照的稳定优势。')
    lines = ['# 阶段4：联合贪心分叉状态的匹配对照实验', '', conclusion, '',
        '本报告对应2026-11-05启动的开发实验，OPT-13B权重冻结，只训练每轮提示词固定选层的Router。输入128 token，质量目标为后续32 token的teacher KL，每轮跳过6/40层，保护层0、1。', '',
        '## 新增标签与比较设计', '',
        f"使用原800条训练记录，每条新增第4至6步三个状态，共{verification['verified_branch_states']}个状态。若旧Router自身的前三层集合不同于该记录的贪心集合，则使用它；否则保留正确前两层，从第三步联合候选的第2至5名中按固定种子选一层，再逐步枚举剩余候选并选联合KL最低的一层。", '',
        '每一步评分均针对“已有剪层集合+候选层”的整个方案，保留层协同信息。离线标签可观察训练记录的真实续写；线上Router仍只看提示词embedding。重合状态仅在同一文档、整个已剪集合相同的条件下复用旧评分。', '',
        f"完成候选评分{verification['verified_branch_candidate_scores']}次，其中{verification['new_candidate_forwards']}次是新增模型前向；相对旧轨迹实际新增{verification['unique_added_states']}个不同条件状态。来源计数：`{json.dumps(data['generation']['sources'], ensure_ascii=False)}`。", '',
        '| 控制项 | 基线组 | 分叉组 |', '| --- | --- | --- |',
        '| 每条训练记录的状态槽位 | 原6步+重复原第4至6步 | 原6步+分叉第4至6步 |',
        '| Router | 宽256、2层编码器 | 相同 |',
        '| 初始化/样本顺序 | seed579 | 相同 |',
        '| batch/更新次数 | batch16、50步/epoch、24epoch | 相同 |',
        '| 目标 | 软CE+0.25硬CE | 相同 |',
        '| 静态基座 | 冻结OPT | 相同 |', '',
        '这样匹配编码次数、梯度更新和后期步骤权重，避免把增加训练量误当作新增状态覆盖的收益。当前基线重复了后期状态，与阶段3六状态训练存在差异，主要因果比较应看本次两组。', '',
        '## 验证质量', '',
        '100条验证记录已经用于开发诊断，不是新独立测试。原32条用于在epoch1、4、8、12、16、20、24选择checkpoint，另外68条只排除checkpoint选择，不能称为未接触的测试集。旧100条测试记录没有参与本轮训练或追加评估。', '',
        '| 组别 | 方案 | 验证100条 KL | 其他68条 KL | 验证100条 PPL |',
        '| --- | --- | ---: | ---: | ---: |']
    names = {'full': '完整OPT', 'static_old': '旧16条校准静态', 'static_learned': '800条训练的固定输入静态',
        'oracle_greedy': '逐样本贪心搜索', 'router': '真实提示词Router', 'shuffled_mean4': '交换方案4次平均'}
    for arm in ['baseline', 'augmented']:
        for policy in names:
            row = next(r for r in data['quality'] if r['population'] == 'validation_all100' and r['arm'] == arm and r['policy'] == policy)
            other = next(r for r in data['quality'] if r['population'] == 'validation_other68' and r['arm'] == arm and r['policy'] == policy)
            lines.append(f"| {'基线' if arm == 'baseline' else '分叉'} | {names[policy]} | {row['continuation_kl']:.6f} | {other['continuation_kl']:.6f} | {row['perplexity']:.4f} |")
    lines += ['', '## 配对比较', '',
        '差值为本行Router减去对照，负值更好。对每个目标文档先平均四次交换结果，再按100条文档做10000次配对bootstrap；区间为描述性结果，未校正文档依赖和多重比较。', '',
        '| 比较 | 指标 | 差值 | 95%区间 | 更优比例 |', '| --- | --- | ---: | --- | ---: |']
    for label, reference_arm, reference_policy in [('分叉减基线', 'baseline', 'router'),
            ('分叉减交换', 'augmented', 'shuffled_mean4'), ('分叉减学习静态', 'augmented', 'static_learned')]:
        for metric in ['continuation_kl', 'continuation_nll']:
            row = comparison('augmented', reference_arm, reference_policy, metric)
            lo, hi = row['bootstrap_95pct']
            lines.append(f"| {label} | {metric} | {row['mean_delta']:.6f} | [{lo:.6f}, {hi:.6f}] | {row['fraction_negative']:.1%} |")
    lines += ['', '## checkpoint与核验', '']
    for arm in data['arms']:
        path = Path(arm['best_checkpoint'])
        lines.append(f"- {arm['arm']}：`{path.name}`，验证选择KL={arm['best_selection_kl']:.6f}，SHA256=`{arm['best_checkpoint_sha256']}`。")
    lines += ['', f"验收了{verification['verified_quality_rows']}条质量记录，确认初始化、参数量、梯度更新和后期权重一致。保护了{verification['preserved_existing_files']}个既有文件；旧checkpoint与输入token数组哈希保持不变。运行耗时{data['elapsed_seconds'] / 60:.1f}分钟。", '',
        '全过程通过统一入口使用物理GPU1、3，CPU0至20合计21核，每worker10线程。模型权重、原始文本、token数组及生成序列不上传。代码只新增独立版本，旧源文件和报告未修改。', '',
        '本轮没有新增速度或显存测量；不能据Router质量变化推断更快推理。缓存策略仍是同一轮固定方案，新一轮换方案时重建整个上下文的KV缓存。', '',
        '## 下一步', '',
        ('先冻结本次方案，在新独立记录和至少多个随机种子上确认相对交换与同数据静态的收益，再考虑双方预算一致的恢复微调。' if improves and matches else
         '先保留本次负面或不确定结果。下一项控制实验应保持状态数据不变，比较原分类目标与直接约束候选联合KL差的排序目标；若仍无输入匹配收益，再扩大独立训练文档，并检查同提示词不同续写的标签稳定性。'), '',
        '离线贪心看到了真实后缀且不保证全局最优，它的差距不全是仅根据提示词可预测的信息。一种子、固定K、C4自然续写和既有数据近重复限制仍适用，不能直接外推聊天或完整下游任务。', '',
        f"运行目录：`{run.relative_to(root)}`。生产程序为`tools/train_augmented_router_v2.py`；同名v1是未执行的早期草稿，不属于结果来源。", '']
    path = root / 'analysis/STAGE4_BRANCH_RESULTS_20261105.zh-CN.md'
    with path.open('x') as handle:
        handle.write('\n'.join(lines))
    csv_path = root / 'analysis/stage4_branch_results_20261105.csv'
    with csv_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['population', 'arm', 'policy', 'samples',
            'continuation_kl', 'continuation_nll', 'perplexity', 'unique_masks'])
        writer.writeheader()
        writer.writerows(data['quality'])
    return path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    verification = verify(root, run)
    write_json(run / 'verification.json', verification)
    report_path, csv_path = report(root, run, verification)
    files = [str(p.relative_to(root)) for p in [report_path, csv_path]]
    files += [str(p.relative_to(root)) for p in sorted(run.iterdir())
              if p.is_file() and not p.name.startswith('private_') and p.suffix in {'.json', '.jsonl', '.log'}]
    manifest = root / 'analysis/stage4_branch_completion_upload_20261105.json'
    files.append(str(manifest.relative_to(root)))
    write_json(manifest, files)
    if args.publish:
        result = subprocess.run([sys.executable, str(root / 'tools/publish_additions_v2.py'),
            '--root', str(root), '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Report matched conditional greedy branch-state training and validation'],
            cwd=root, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError('Publication failed: ' + result.stderr[-3000:])
        receipt = json.loads(result.stdout)
        write_json(root / 'analysis/stage4_branch_completion_publication_20261105.json', receipt)
        print(json.dumps(receipt), flush=True)
    print(json.dumps({'state': 'finalized', 'report': str(report_path), 'verification': verification}), flush=True)


if __name__ == '__main__':
    main()
