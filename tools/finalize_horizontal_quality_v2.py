"""Audit, summarize and publish a finite frozen downstream quality experiment."""
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import write_json


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def paired_interval(a, b, tokens, documents, rng):
    difference = a - b
    if tokens is None:
        # Exact empirical bootstrap distribution compressed by distinct outcomes.
        values, counts = np.unique(difference, return_counts=True)
        draws = rng.multinomial(len(difference), counts/counts.sum(), size=2000)
        samples = draws @ values / len(difference)
        estimate = difference.mean()
    else:
        _, inverse = np.unique(documents, return_inverse=True)
        n = inverse.max() + 1
        differences, lengths = np.zeros(n), np.zeros(n)
        np.add.at(differences, inverse, difference)
        np.add.at(lengths, inverse, tokens)
        draws = rng.multinomial(n, np.full(n, 1/n), size=2000)
        samples = (draws @ differences) / (draws @ lengths)
        estimate = difference.sum()/tokens.sum()
    lo, hi = np.quantile(samples, [.025, .975])
    return float(estimate), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    config = json.loads((run / 'protocol.json').read_text())
    if config['probe']: raise ValueError('Small train/validation probe is not a benchmark result')
    subprocess.run([sys.executable, '-B', str(root / 'tools/verify_horizontal_quality_v2.py'), '--run', str(run)], check=True, cwd=root)
    summary = json.loads((run / 'summary.json').read_text())
    audit = json.loads((run / 'verification.json').read_text())
    arrays = defaultdict(dict)
    for path in sorted(run.glob('quality_*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            key = row['task'], row['policy'], row['seed'], row['k']
            value = row['nll_sum'] if row['task'] == 'wikitext_conditional' else row['correct'] if row['task'] == 'winogrande' else row['correct_norm']
            arrays[key][row['sample']] = (value, row.get('scored_tokens'), row['document'])
    ordered = {}
    for key, values in arrays.items():
        ordered[key] = np.asarray([values[i] for i in sorted(values)], dtype=object)
    comparisons, rng = [], np.random.default_rng(579)
    tasks = ['piqa', 'hellaswag', 'winogrande', 'wikitext_conditional']
    baselines = ['static_joint_kl', 'sleb_nll_opt', 'shortgpt_bi_opt', 'shortened_ppl_opt', 'shortened_ppl_plus', 'random_static_seed579']
    for task in tasks:
        for k in config['budgets']:
            for arm in ['pure', 'hybrid']:
                seed_arrays = [ordered[task, arm, seed, k][:, 0].astype(float) for seed in [579, 580, 581]]
                for seed, a in list(zip([579, 580, 581], seed_arrays)) + [('mean3', np.mean(seed_arrays, axis=0))]:
                    for baseline in baselines:
                        rows = ordered[task, baseline, None, k]
                        wiki = task == 'wikitext_conditional'
                        estimate, lo, hi = paired_interval(a, rows[:, 0].astype(float),
                            rows[:, 1].astype(float) if wiki else None, rows[:, 2], rng)
                        comparisons.append({'task': task, 'k': k, 'arm': arm, 'seed': seed, 'baseline': baseline,
                            'metric': 'nll_per_token_difference' if wiki else 'accuracy_difference',
                            'difference': estimate, 'ci95_low': lo, 'ci95_high': hi,
                            'direction': 'negative favors router' if wiki else 'positive favors router',
                            'bootstrap_unit': 'article' if wiki else 'question', 'bootstrap_replicates': 2000})
    quality_csv = root / 'analysis/stage7_horizontal_quality_20261108.csv'
    paired_csv = root / 'analysis/stage7_horizontal_paired_20261108.csv'
    save_csv(quality_csv, summary['quality']); save_csv(paired_csv, comparisons)
    text = ['# 冻结路由的下游横向质量评测', '',
        '已完成OPT-13B质量比较。本轮固定m=6，不依据下游结果重选检查点或超参数；吞吐量和显存测试尚未执行。', '',
        'PIQA/HellaSwag表内为acc_norm百分比，WinoGrande为acc百分比；Wiki为文章内128条件token、最多384目标token的条件PPL。它不是标准rolling PPL。动态列是三个训练种子的均值，Wiki先平均NLL再取exp。', '',
        '| 任务 | K | 完整模型 | 静态联合KL | SLEB适配 | ShortGPT适配 | Shortened适配 | Shortened+ | 纯动态 | 固定6+动态 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    by_key = {(r['task'], r['policy'], r['seed'], r['k']): r for r in summary['quality']}
    def metric(task, policy, k):
        seeds = [579, 580, 581] if policy in ['pure', 'hybrid'] else [None]
        rows = [by_key[task, policy, seed, 0 if policy == 'full' else k] for seed in seeds]
        if task == 'wikitext_conditional': return f"{math.exp(np.mean([r['nll_per_token'] for r in rows])):.3f}"
        name = 'accuracy' if task == 'winogrande' else 'accuracy_norm'
        return f"{100*np.mean([r[name] for r in rows]):.2f}"
    for task in tasks:
        for k in config['budgets']:
            names = ['full', *baselines[:-1], 'pure', 'hybrid']
            text.append('| ' + ' | '.join([task, str(k), *[metric(task, policy, k) for policy in names]]) + ' |')
    text += ['', '## 配对比较与解释', '',
        '逐种子结果、未归一化准确率、随机静态、路由种类数均保留在完整CSV。配对CSV使用2000次bootstrap：选择题按题，Wiki按文章。mean3对同一题的三个模型先平均，仍以题/文章为抽样单位。区间未校正多重比较，用作探索性不确定性描述，不能直接等同于所有任务显著领先。', '',
        '应直接读取每项结果判断胜负，不从此前C4上的接近静态贪心推断必然超过其它方法。静态联合贪心保留为强对照；这些比较本身不能证明路由收益来自输入适应，进一步因果解释仍需输入置换消融。', '',
        '所有文献基线均为当前OPT设置下的冻结权重适配，不等同完整官方复现；本地缓存版本与当前harness的数据标识不同。路由只用共同提示词，WinoGrande仅看到空格前前缀；选择题短提示词长度也不同于训练的128token。结论限定于记录的实现、模型与数据。', '',
        f"共验收{audit['verified_rows']}条质量记录，{audit['policies']}个策略，{audit['likelihood_reference_probes']}个标准CausalLM似然对照探针。{audit['source_preserved_files']}个既有文件哈希保持一致。运行耗时{summary['elapsed_seconds']/3600:.2f}小时，使用4卡/42核。", '',
        f"来源目录：`{run.relative_to(root)}`。本轮不包含固定层数m的重新训练消融，也未测吞吐量或峰值显存。后续成本比较需先审阅质量、固定工作点，再公平计入路由、权重常驻与KV cache。", '']
    report = root / 'analysis/STAGE7_HORIZONTAL_RESULTS_20261108.zh-CN.md'
    with report.open('x') as stream: stream.write('\n'.join(text))
    if args.publish:
        files = [str(p.relative_to(root)) for p in run.iterdir() if p.is_file()
            and p.suffix in {'.json', '.jsonl', '.log'} and not p.name.startswith('private_')]
        files += [str(p.relative_to(root)) for p in [report, quality_csv, paired_csv]]
        files += [str(p.relative_to(root)) for p in root.glob('tools/*horizontal*.py')]
        files += ['analysis/STAGE7_HORIZONTAL_AND_FIXED_PREFIX_PROTOCOL_20261108.zh-CN.md',
            'analysis/stage7_horizontal_input_audit_20261108.json']
        manifest = root / 'analysis/stage7_horizontal_completed_upload_20261108.json'
        files.append(str(manifest.relative_to(root)))
        write_json(manifest, sorted(set(files)))
        subprocess.run([sys.executable, '-B', str(root / 'tools/publish_additions_v7.py'), '--root', str(root),
            '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Complete frozen four-GPU downstream quality evaluation',
            '--receipt', str(root / 'analysis/stage7_horizontal_completed_publication_20261108.json')], check=True, cwd=root)
    print(json.dumps({'report': str(report), 'paired_comparisons': len(comparisons)}), flush=True)


if __name__ == '__main__': main()
