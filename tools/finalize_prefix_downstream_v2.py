"""Publish an append-only corrected stage11 table with paired uncertainty."""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from finalize_horizontal_quality_v2 import paired_interval, save_csv
from prefix_downstream_correction_core_v1 import SEEDS, aggregate_seed_metric, combined_rows
from stage1_benchmark_v2 import write_json
from verify_prefix_downstream_v2 import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    audit = verify(run)
    if audit['probe']:
        print(json.dumps({'state': 'corrected_probe_audited', **audit}), flush=True)
        return
    config = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    grouped = {(r['task'], r['policy'], r['seed'], r['k']): r for r in summary['quality']}
    tasks = ['piqa', 'hellaswag', 'winogrande', 'wikitext_conditional']
    def metric(task, name, k):
        seeds = [None] if name in ['full', 'static_joint_kl'] else SEEDS
        rows = [grouped[task, name, seed, 0 if name == 'full' else k] for seed in seeds]
        field = 'nll_per_token' if task == 'wikitext_conditional' else 'accuracy' if task == 'winogrande' else 'accuracy_norm'
        return aggregate_seed_metric([r[field] for r in rows], wiki=task == 'wikitext_conditional')
    arrays = defaultdict(dict)
    for row in combined_rows(run, config):
        value = row['nll_sum'] if row['task'] == 'wikitext_conditional' else row['correct'] if row['task'] == 'winogrande' else row['correct_norm']
        arrays[row['task'], row['policy'], row['seed'], row['k']][row['sample']] = (value, row.get('scored_tokens'), row['document'])
    ordered = {key: np.asarray([values[i] for i in sorted(values)], dtype=object) for key, values in arrays.items()}
    comparisons, rng = [], np.random.default_rng(579)
    for task in tasks:
        for k in [10, 12]:
            for arm, baseline in [('m04', 'static_joint_kl'), ('m06', 'static_joint_kl'), ('m08', 'static_joint_kl'), ('m08', 'm06'), ('m08', 'm04')]:
                a = np.mean([ordered[task, arm, seed, k][:, 0].astype(float) for seed in SEEDS], axis=0)
                seeds = [None] if baseline == 'static_joint_kl' else SEEDS
                b = np.mean([ordered[task, baseline, seed, k][:, 0].astype(float) for seed in seeds], axis=0)
                meta = ordered[task, baseline, seeds[0], k]
                wiki = task == 'wikitext_conditional'
                estimate, lo, hi = paired_interval(a, b, meta[:, 1].astype(float) if wiki else None, meta[:, 2], rng)
                comparisons.append({'task': task, 'k': k, 'arm': arm, 'baseline': baseline, 'seed': 'mean3',
                    'metric': 'nll_per_token_difference' if wiki else 'accuracy_difference',
                    'difference': estimate, 'ci95_low': lo, 'ci95_high': hi,
                    'direction': 'negative favors arm' if wiki else 'positive favors arm',
                    'bootstrap_unit': 'article' if wiki else 'question', 'bootstrap_replicates': 2000})
    quality_csv = root / 'analysis/stage11_prefix_downstream_corrected_quality_20261110.csv'
    paired_csv = root / 'analysis/stage11_prefix_downstream_corrected_paired_20261110.csv'
    save_csv(quality_csv, summary['quality']); save_csv(paired_csv, comparisons)
    text = ['# 阶段11固定前缀下游评测：更正结果', '',
        '本报告替代旧阶段11报告中的m=8列，并统一更正跨种子WikiText PPL汇总。旧程序把仅有6层的fixed_global_policy.json切片到8，实际仍得到6层，造成m=8训练/推理前缀不一致。旧验收只验证前缀包含关系，漏查len(prefix)==m。旧文件保留作为审计证据，旧m=8列不能用于比较。', '',
        '本次只补跑K=10/12、3个种子的6个m=8策略，固定前缀为[4,9,2,12,17,7,24,37]；checkpoint保持不变。完整模型、静态联合KL和m=0/4/6共21个策略逐条复用，核对哈希并抽样GPU重放。路由每题只观察共同提示词并固定路径，所有候选共享路径。', '',
        '选择题为百分比；PIQA/HellaSwag使用acc_norm，WinoGrande使用acc。Wiki为条件PPL，先平均三个种子的token加权NLL再取exp；不再平均各自PPL。', '',
        '| 任务 | K | 完整模型 | 静态KL | m=0 | m=4 | m=6 | m=8（更正） |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for task in tasks:
        for k in [10, 12]:
            text.append('| ' + ' | '.join([task, str(k), *[f'{metric(task, name, k):.3f}' for name in
                ['full', 'static_joint_kl', 'm00', 'm04', 'm06', 'm08']]]) + ' |')
    text += ['', '## m=8的配对差异', '',
        '下列选择题差异以百分点表示，正值有利于m=8；Wiki以每token NLL差表示，负值有利于m=8。先在同一题上平均三个训练种子，再按题或Wiki文章进行2000次配对bootstrap；区间未做多重比较校正，属于探索性分析。', '',
        '| 任务 | K | 对照 | 差异 | 95%区间 |', '| --- | ---: | --- | ---: | --- |']
    for row in comparisons:
        if row['arm'] != 'm08':
            continue
        scale = 1 if row['task'] == 'wikitext_conditional' else 100
        text.append(f"| {row['task']} | {row['k']} | {row['baseline']} | {scale*row['difference']:+.4f} | [{scale*row['ci95_low']:+.4f}, {scale*row['ci95_high']:+.4f}] |")
    text += ['', '## 结论边界与下一步', '',
        '本轮是既有下游开发任务的更正复核，不是新的独立测试；不依据下游分数重选checkpoint，基础模型仍冻结，未做恢复训练。m控制固定前缀长度，K控制总剪层数，两者必须分别报告。不能把质量接近静态贪心解释为已经证明输入适应收益；阶段9置换实验尚未支持该机制。', '',
        '吞吐量和显存已有阶段8实测；其m=6配置不受本次m=8推理配置错误影响。只有在确定新的部署工作点后，才需要补测其成本，不能把阶段8数值直接作为m=8测量值。', '',
        f"本次验收{audit['new_quality_rows']}条新记录，复用{audit['reused_quality_rows']}条未受影响记录；合计{audit['verified_rows']}条、27个策略。{audit['unchanged_control_gpu_replays']}条旧对照GPU重放最大误差{audit['replay_max_absolute_error']:.8g}，{audit['likelihood_reference_probes']}个标准CausalLM似然探针通过；{audit['source_preserved_files']}个既有文件保持哈希不变。使用4卡/42核。", '',
        f"更正运行：`{run.relative_to(root)}`。原运行：`{Path(config['reuse_run']).relative_to(root)}`。逐种子结果见quality CSV；完整配对比较见paired CSV。", '']
    report = root / 'analysis/STAGE11_PREFIX_DOWNSTREAM_CORRECTED_RESULTS_20261110.zh-CN.md'
    with report.open('x') as stream:
        stream.write('\n'.join(text))
    if args.publish:
        names = ['protocol.json', 'data_manifest.json', 'summary.json', 'verification.json',
                 'preservation_after.json', 'allocation.json', 'status.json']
        files = [str((run / name).relative_to(root)) for name in names]
        files += [str(p.relative_to(root)) for p in sorted(run.glob('worker_probe_*.json'))]
        files += [str(p.relative_to(root)) for p in [report, quality_csv, paired_csv]]
        manifest = root / 'analysis/stage11_prefix_downstream_corrected_completed_upload_20261110.json'
        write_json(manifest, files)
        subprocess.run([sys.executable, '-B', str(root / 'tools/publish_additions_v7.py'), '--root', str(root),
            '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Correct m8 downstream prefixes and seed-level PPL aggregation',
            '--receipt', str(root / 'analysis/stage11_prefix_downstream_corrected_publication_20261110.json')], check=True, cwd=root)
    print(json.dumps({'state': 'corrected_results_finalized', 'report': str(report)}), flush=True)


if __name__ == '__main__':
    main()
