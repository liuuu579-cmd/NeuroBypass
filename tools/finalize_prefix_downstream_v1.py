"""Finalize and append-only publish stage11 downstream fixed-prefix results."""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import write_json
from verify_prefix_downstream_v1 import verify


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    audit = verify(run)
    if audit['probe']:
        print(json.dumps({'state': 'probe_audited', **audit}), flush=True)
        return
    summary = json.loads((run / 'summary.json').read_text())
    quality_csv = root / 'analysis/stage11_prefix_downstream_quality_20261109.csv'
    save_csv(quality_csv, summary['quality'])
    rows = [r for r in summary['quality'] if r['task'] != 'wikitext_conditional']
    text = ['# 固定层数消融的下游复核', '',
        '本轮冻结阶段10已经选择的m=4/8路由器，连同阶段6的m=0纯动态、m=6混合和静态联合KL，统一复测PIQA、HellaSwag、WinoGrande与WikiText条件PPL。路由只观察共同提示词；每题只选一次路径并复用给全部候选。', '',
        '这仍是阶段7任务上的开发复核，不是新的独立测试；本轮不依据下游分数重新选择m或checkpoint。基础OPT权重冻结，未做恢复训练。', '',
        '| 任务 | K | 完整模型 | 静态KL | m=0 | m=4 | m=6 | m=8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    grouped = {(r['task'], r['policy'], r['seed'], r['k']): r for r in summary['quality']}
    for task in ['piqa', 'hellaswag', 'winogrande', 'wikitext_conditional']:
        for k in [10, 12]:
            def metric(name, seed=None):
                row = grouped[task, name, seed, 0 if name == 'full' else k]
                if task == 'wikitext_conditional': return math.exp(row['nll_per_token'])
                return 100 * (row['accuracy'] if task == 'winogrande' else row['accuracy_norm'])
            vals = [metric('full'), metric('static_joint_kl'),
                *[float(np.mean([metric(f'm{m:02d}', seed) for seed in [579, 580, 581]])) for m in [0, 4, 6, 8]]]
            text.append('| ' + ' | '.join([task, str(k), *[f'{v:.3f}' for v in vals]]) + ' |')
    text += ['', '选择题为百分比，Wiki为条件PPL，均按阶段7定义计算。路由器各训练种子在同一题上先求均值；完整模型和静态路径是单一策略。', '',
        '阶段9显示输入与路径匹配在这些任务上的收益接近于零，因此本表重点判断m=8在实际任务上是否保留质量，而不是把C4开发集上的微小差异当作机制证明。最终部署配置应同时考虑质量、真实端到端吞吐量、峰值显存和路由器开销。', '',
        f"验收{audit['verified_rows']}条记录、{audit['policies']}个策略、{audit['likelihood_reference_probes']}个似然探针；{audit['source_preserved_files']}个既有文件哈希保持一致。运行目录：`{run.relative_to(root)}`。", '']
    report = root / 'analysis/STAGE11_PREFIX_DOWNSTREAM_RESULTS_20261109.zh-CN.md'
    with report.open('x') as stream:
        stream.write('\n'.join(text))
    if args.publish:
        names = ['protocol.json', 'data_manifest.json', 'summary.json', 'verification.json',
            'preservation_after.json', 'allocation.json', 'status.json']
        files = [str((run / name).relative_to(root)) for name in names]
        files += [str(path.relative_to(root)) for path in sorted(run.glob('worker_probe_*.json'))]
        files += [str(path.relative_to(root)) for path in [report, quality_csv]]
        files += [str(path.relative_to(root)) for path in sorted(root.glob('tools/*prefix_downstream*.py'))]
        files += ['analysis/STAGE11_PREFIX_DOWNSTREAM_PROTOCOL_20261109.zh-CN.md']
        manifest = root / 'analysis/stage11_prefix_downstream_completed_upload_20261109.json'
        write_json(manifest, sorted(set(files)))
        subprocess.run([sys.executable, '-B', str(root / 'tools/publish_additions_v7.py'),
            '--root', str(root), '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Evaluate fixed-prefix ablation on downstream tasks',
            '--receipt', str(root / 'analysis/stage11_prefix_downstream_publication_20261109.json')], cwd=root, check=True)
    print(json.dumps({'report': str(report), 'state': 'finalized'}), flush=True)


if __name__ == '__main__':
    main()
