"""Write immutable tables, a scientific plot and a Chinese depth-pilot report."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256


NAMES = {'shortgpt_bi_opt': 'ShortGPT-BI OPT适配', 'sleb_nll_opt': 'SLEB-NLL OPT适配',
    'shortened_ppl_opt': 'Shortened-PPL OPT适配', 'shortened_ppl_plus': 'Shortened-PPL+首尾保护',
    'static_joint_kl': '静态联合KL贪心', 'oracle_joint_kl': '真实续写逐样本贪心参考',
    'oracle_shuffled': '贪心参考方案置换', 'router_trained_k6': '当前已训练Router，仅K6',
    'learned_static_k6': '已有800文档学习静态，仅K6', 'full': '完整OPT'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    audit = json.loads((run / 'verification.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    if audit['probe'] or audit['summary_sha256'] != sha256(run / 'summary.json'):
        raise ValueError('Require audited 16-document pilot')
    quality = summary['quality']
    rows = [json.loads(line) for p in run.glob('quality_*.jsonl') for line in p.read_text().splitlines()]
    keyed = {(r['sample'], r['k'], r['policy']): r for r in rows}
    for k in protocol['budgets']:
        nll = float(np.mean([keyed[s, k, f'random_static_seed{seed}']['continuation_nll']
            for s in protocol['pilot_ids'] for seed in [579, 580, 581]]))
        kl = float(np.mean([keyed[s, k, f'random_static_seed{seed}']['continuation_kl']
            for s in protocol['pilot_ids'] for seed in [579, 580, 581]]))
        quality.append({'k': k, 'depth_fraction': k / 40, 'policy': 'random_static_mean3',
            'documents': len(protocol['pilot_ids']), 'continuation_kl': kl, 'continuation_nll': nll,
            'perplexity': math.exp(nll), 'unique_masks': None})
    NAMES['random_static_mean3'] = '随机静态三种子成本平均'
    full = next(r for r in quality if r['policy'] == 'full')
    for row in quality:
        row['ppl_ratio_to_full'] = row['perplexity'] / full['perplexity']
    all_rows = sorted(quality, key=lambda r: (r['k'], r['policy']))
    report = root / 'analysis/STAGE6_DEPTH_PILOT_RESULTS_20261106.zh-CN.md'
    csv_path = root / 'analysis/stage6_depth_pilot_quality_20261106.csv'
    comparison_path = root / 'analysis/stage6_depth_pilot_paired_20261106.csv'
    figure = root / 'analysis/stage6_depth_pilot_curves_20261106.png'
    for path in [report, csv_path, comparison_path, figure]:
        if path.exists():
            raise FileExistsError('Refusing to replace old result: ' + str(path))
    for path, records in [(csv_path, all_rows), (comparison_path, summary['paired_comparisons'])]:
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'shortgpt_bi_opt': ('ShortGPT BI (OPT adaptation)', '-o'),
        'sleb_nll_opt': ('SLEB NLL (OPT adaptation)', '-s'),
        'shortened_ppl_opt': ('Shortened PPL (OPT adaptation)', '-^'),
        'shortened_ppl_plus': ('Shortened PPL+ (protected edges)', '-v'),
        'static_joint_kl': ('Static joint-KL greedy', '-D'),
        'oracle_joint_kl': ('Offline greedy (sees true continuation)', '--o'),
        'oracle_shuffled': ('Shuffled offline-greedy routes', '--s'),
        'random_static_mean3': ('Random static (3-seed mean costs)', ':x')}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for policy, (label, style) in styles.items():
        group = sorted((r for r in quality if r['policy'] == policy), key=lambda r: r['k'])
        axes[0].plot([r['depth_fraction'] * 100 for r in group], [r['continuation_kl'] for r in group], style, label=label, markersize=4)
        axes[1].plot([r['depth_fraction'] * 100 for r in group], [r['ppl_ratio_to_full'] for r in group], style, label=label, markersize=4)
    for policy, label, marker in [('router_trained_k6', 'Trained router (K=6 only)', '*'),
            ('learned_static_k6', 'Learned static (K=6 only)', 'P')]:
        row = next(r for r in quality if r['policy'] == policy)
        axes[0].scatter([15], [row['continuation_kl']], marker=marker, s=100, color='black', label=label, zorder=5)
        axes[1].scatter([15], [row['ppl_ratio_to_full']], marker=marker, s=100, color='black', zorder=5)
    for ax in axes:
        ax.set_xlabel('Transformer layers skipped (%)')
        ax.set_yscale('log')
        ax.grid(True, alpha=.25, which='both')
    axes[0].set_ylabel('Teacher KL on 32-token continuation (lower is better)')
    axes[1].set_ylabel('Conditional PPL / full-model PPL (lower is better)')
    axes[1].axhline(1, color='grey', linewidth=1)
    axes[0].legend(fontsize=7, loc='best')
    fig.suptitle('OPT-13B depth pilot: 16 calibration + 16 reused pilot records; no recovery training', fontsize=11)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    text = ['# 阶段6：论文基线与6至24层剪枝先导结果', '',
        '本轮已实际新增并运行 ShortGPT-BI、SLEB-NLL、Shortened-PPL/PPL+ 的OPT适配，对15%至60%的剪层深度进行了扫描。结果是16条旧pilot上的诊断，尚非完整下游横向评测。', '',
        '未训练新的大K Router。只有K=6是当前已训练Router的结果；K>6的逐样本曲线是看到了真实续写的离线贪心参考，不能部署，也不保证全局最优或仅凭提示词可学到。', '',
        f"完整OPT在同一16条记录上的条件续写PPL={full['perplexity']:.4f}。校准仅使用另外16条旧calibration；旧100条test未用于模型评估。OPT权重冻结，输入128 token，后续32 token质量计分。", '',
        '## 方法与适配差异', '',
        '- ShortGPT：按所有校准token的层输入/输出余弦差均值一次性排序。',
        '- SLEB：每步在已跳集合上逐候选测全部159个next-token NLL，选择均值较小的候选并重新评分。',
        '- Shortened-PPL：使用SLEB第一步相同的单层删除NLL成本，但此后不重评分，直接取一次性排名。PPL+额外保护0、1、2、3、38、39。',
        '- 除PPL+外保护层0、1。SLEB上游默认首尾屏障、原论文数据和模型配置与本适配不同；这里不是论文默认配方的完全复现。PPL+保护范围也不同，单独标注，不能把差异全部归因于排名算法。',
        '- 静态联合KL使用相同16条校准文档，目标与离线逐样本搜索均为32-token续写KL；每步成本针对整个已跳集合和新候选。随机路径固定三个种子，不按结果搜索最优随机集合。',
        '- 论文、官方源码固定版本与后续BlockPruner/SkipDecode等对照安排见[补充协议](STAGE6_PAPERS_AND_DEPTH_SWEEP_20261106.zh-CN.md)。本轮没有执行这些后续方法、Taylor+或恢复训练。', '',
        '## 各预算的条件续写质量', '',
        'KL和PPL越低越好。下面PPL不是标准全量WikiText或C4语言建模PPL；不同历史数据和目标下的数值不得直接相除。随机三种子先平均NLL再取exp，不是平均三个PPL。', '',
        '| 跳层K | 深度占比 | 方法 | KL | PPL | PPL/完整模型 |',
        '| ---: | ---: | --- | ---: | ---: | ---: |']
    main_policies = list(styles) + ['router_trained_k6', 'learned_static_k6']
    for row in all_rows:
        if row['policy'] not in main_policies:
            continue
        text.append(f"| {row['k']} | {row['depth_fraction']:.0%} | {NAMES[row['policy']]} | {row['continuation_kl']:.6f} | {row['perplexity']:.4f} | {row['ppl_ratio_to_full']:.3f} |")
    text += ['', '![深度与质量曲线](stage6_depth_pilot_curves_20261106.png)', '',
        '## 输入匹配诊断', '',
        '差值为真实续写逐样本贪心参考减去置换方案，负值更好。重采样单位为16条pilot文档，10000次配对bootstrap；这些样本此前已使用，区间仅描述本诊断，不是新测试或完整训练种子不确定性。', '',
        '| K | KL差 | 95%文档区间 | NLL差 | 95%文档区间 |',
        '| ---: | ---: | --- | ---: | --- |']
    for k in protocol['budgets']:
        pair = [next(r for r in summary['paired_comparisons'] if r['k'] == k and r['reference'] == 'oracle_shuffled' and r['metric'] == metric)
                for metric in ['continuation_kl', 'continuation_nll']]
        a, b = pair
        text.append(f"| {k} | {a['mean_delta']:.6f} | [{a['bootstrap_95pct'][0]:.6f}, {a['bootstrap_95pct'][1]:.6f}] | {b['mean_delta']:.6f} | [{b['bootstrap_95pct'][0]:.6f}, {b['bootstrap_95pct'][1]:.6f}] |")
    text += ['', '## 缓存与来源核验', '',
        '初次2条校准/2条pilot的v1微型运行在K=24缓存分布检查处失败，失败目录完整保留。独立复核发现FP16串行缓存与并行重算平均分布差约2.40e-4、最大位置差约1.42e-3；同一路由FP32重算降至约1.50e-7与4.34e-7，所有保留/跳层KV长度正确。没有放宽原容差。', '',
        '生产先导来源为新增 `tools/depth_paper_pilot_v2.py`，对当前运行另外记录FP16分布差；若超过原容差，必须以FP32确认，不能把FP32通过写成FP16等价通过。FP32仅用于缓存逻辑复核，不用于下面FP16质量曲线。', '']
    cache = json.loads((run / 'cache_probes.json').read_text())
    for item in cache:
        half = item['fp16']
        line = f"- K={len(half['route'])}：FP16平均分布差={half['mean_absolute_position_kl']:.9g}，最大位置差={half['max_absolute_position_kl']:.9g}，结果=`{item['outcome']}`。"
        if item['fp32']:
            line += f" FP32平均差={item['fp32']['mean_absolute_position_kl']:.9g}。"
        text.append(line)
    text += ['',
        f"验收{audit['verified_quality_rows']}条质量记录、{audit['verified_scalar_candidate_scores']}次标量候选评分，核对了候选集合、完整联合状态、逐步最小值、固定路径、置换、来源split、预算与汇总。{audit['source_preserved_files']}个运行前既有文件哈希保持一致。私有token数组和旧Router检查点哈希未变。", '',
        f"本次耗时{summary['elapsed_seconds'] / 60:.2f}分钟，通过统一入口使用物理GPU1、3与整个进程树21核，每worker10个torch线程。离线候选batch仅提高搜索吞吐，未用于测部署速度，不声称本轮新增了真实加速或权重显存压缩。运行目录：`{run.relative_to(root)}`。", '',
        '## 下一步', '',
        '后续在更大K上训练纯动态、纯静态及“固定6层全局集合+动态余层”的匹配对照。扩展step embedding及第7–24步联合标签；深层分叉覆盖也同步扩展。全局集合只用训练/校准数据确定，不能用测试统计。最优K由预先声明的质量预算和开发集选择，再在冻结下游评测中报告整条曲线。', '',
        '16条先导样本不能证明存在对所有输入都无用的层，也不能证明学习式动态方案必然在更深剪枝胜出。保留强静态与重新评分的论文方法，是区分全局冗余、组合效应和输入匹配收益所必需的。', '']
    with report.open('x') as stream:
        stream.write('\n'.join(text))
    print(json.dumps({'report': str(report), 'figure': str(figure), 'quality_csv': str(csv_path),
        'paired_csv': str(comparison_path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
