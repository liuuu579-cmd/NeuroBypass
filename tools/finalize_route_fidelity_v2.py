"""Versioned finalization after a prior report-name collision."""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--run', type=Path, required=True); p.add_argument('--publish', action='store_true')
    args = p.parse_args(); root, run = args.root.resolve(), args.run.resolve()
    # The previous finalizer completed the independent CPU audit before its
    # report-name collision. Reuse that immutable audit rather than attempting
    # to overwrite verification.json.
    audit = json.loads((run / 'verification.json').read_text())
    if audit.get('status') != 'passed':
        raise ValueError('Existing route-fidelity audit did not pass')
    rows = [json.loads(line) for path in sorted(run.glob('diagnostics_*.jsonl')) for line in path.open()]
    groups = defaultdict(list)
    for row in rows:
        metric = row.get('correct_norm', row.get('correct'))
        if metric is None: metric = row['teacher_kl_sum'] / row['target_tokens']
        groups[row['task'], row['condition'], row['seed'], row['k']].append(float(metric))
    summary = []
    for (task, condition, seed, k), values in sorted(groups.items()):
        summary.append({'task': task, 'condition': condition, 'seed': seed, 'k': k, 'records': len(values),
            'metric_mean': float(np.mean(values)), 'metric': 'accuracy_or_norm_accuracy' if task not in ['c4_development', 'wikitext_conditional'] else 'teacher_kl_per_token'})
    write_json(run / 'diagnostic_summary_v2.json', {'groups': summary, 'interpretation': 'Teacher KL/token for C4/Wiki; accuracy for MC. Matched, three shuffled donors and static joint KL.', 'seed_note': 'Model seeds are aggregated per record and are not independent samples.'})
    by = {(r['task'], r['condition'], r['seed'], r['k']): r['metric_mean'] for r in summary}
    text = ['# 输入与路径匹配诊断（正式开发复核）', '',
        '本轮冻结m=6路由器与checkpoint，比较输入自身路径、三次任务内置换路径和静态联合KL。路由只看输入前缀；候选答案、gold和未来目标token不进入路由。C4使用68条未参与checkpoint选择的开发文档，PIQA、HellaSwag、WinoGrande和WikiText各预先固定64条输入。这是开发诊断，不是新的独立测试。', '',
        'C4与WikiText使用完整词表teacher KL/token；选择题报告acc_norm（WinoGrande为acc）。未来目标可见的最小KL仅作离线上界诊断，不能解释为可部署路由。', '',
        '| 任务 | K | matched | shuffled均值 | static | 指标 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for task in sorted({r['task'] for r in summary}):
        for k in [10, 12]:
            matched = np.mean([by[task, 'matched', seed, k] for seed in [579, 580, 581]])
            shuffled = np.mean([by[task, f'shuffled_{seed}', 579, k] for seed in [1701, 1702, 1703]])
            static = by[task, 'static_joint_kl', 579, k]
            kind = 'KL/token（越低越好）' if task in ['c4_development', 'wikitext_conditional'] else '准确率（越高越好）'
            text.append(f'| {task} | {k} | {matched:.6f} | {shuffled:.6f} | {static:.6f} | {kind} |')
    text += ['', 'matched相对shuffled若稳定更好，才支持输入与路径匹配；同时降低teacher KL比单看任务准确率更有说服力。当前结果只用于决定是否值得进入最终独立测试。路径多样性见`stage12_route_adaptivity_reanalysis_20261110.json`。', '', f"验收{audit['diagnostic_rows']}条记录、{audit['records']}个输入，{audit['source_preserved_files']}个既有文件哈希不变。运行目录：`{run.relative_to(root)}`。", '']
    report = root / 'analysis/STAGE12_ROUTE_FIDELITY_RESULTS_V2_20261110.zh-CN.md'
    with report.open('x') as stream: stream.write('\n'.join(text))
    if args.publish:
        names = ['protocol.json', 'assignments.json', 'diagnostic_summary_v2.json', 'verification.json', 'preservation_after.json', 'allocation.json', 'status.json']
        files = [str((run / n).relative_to(root)) for n in names]
        files += [str((run / f'worker_probe_{r:03d}.json').relative_to(root)) for r in range(4)]
        files += [str(p.relative_to(root)) for p in [report, root / 'analysis/stage12_route_adaptivity_reanalysis_20261110.json']]
        manifest = root / 'analysis/stage12_route_fidelity_v2_completed_upload_20261110.json'
        write_json(manifest, files)
        subprocess.run([sys.executable, '-B', str(root / 'tools/publish_additions_v7.py'), '--root', str(root), '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest), '--message', 'Complete prompt path fidelity development diagnostic', '--receipt', str(root / 'analysis/stage12_route_fidelity_v2_publication_20261110.json')], check=True, cwd=root)
    print(json.dumps({'state': 'finalized', 'report': str(report), 'audit': audit}), flush=True)


if __name__ == '__main__': main()
