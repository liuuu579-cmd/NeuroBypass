"""Report and append-only publish the development fixed-prefix ablation."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from finalize_horizontal_quality_v2 import save_csv
from stage1_benchmark_v2 import write_json
from verify_prefix_ablation_v1 import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    audit = verify(run)
    if audit['probe']:
        print(json.dumps({'state': 'probe_audited', **audit}), flush=True)
        return
    config = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    quality = root / 'analysis/stage10_prefix_quality_20261109.csv'
    paired = root / 'analysis/stage10_prefix_paired_20261109.csv'
    save_csv(quality, summary['quality']); save_csv(paired, summary['paired_comparisons'])
    text = ['# 固定层数消融：相同总预算下的静态与动态分工', '',
        '总剪层数K固定为10/12，比较m=0/2/4/6/8/K。m=2/4/8重新生成条件联合贪心专家/分叉标签并分别训练3个种子；m=0/6及m=K沿用已验收的阶段6端点，复用前在GPU重放评分。', '',
        '固定层来自同一条训练集128条校准样本的静态联合KL路径的前m步。所有路由只看128个提示词embedding，并在整段提示词/续写内沿用同一路径。基础OPT权重冻结。', '',
        '这是已使用的C4开发数据：800条训练、100条验证；其中原32条按每个K选择检查点，其余68条也不是独立测试。表内为100条开发验证的KL及128+32条件PPL，不是下游任务成绩或标准rolling PPL。', '',
        '| K | 固定m | 动态K-m | KL均值 | KL种子范围 | 条件PPL |',
        '| ---: | ---: | ---: | ---: | --- | ---: |']
    for k in config['budgets']:
        for m in [0, 2, 4, 6, 8, k]:
            rows = [r for r in summary['quality'] if r['population'] == 'validation_all100'
                and r['k'] == k and r['m'] == m and r['policy'] in ['matched', 'static_joint_kl']]
            kl = [r['continuation_kl'] for r in rows]
            span = f'{min(kl):.6f}–{max(kl):.6f}' if m != k else '单一静态路径'
            ppl = math.exp(float(np.mean([r['continuation_nll'] for r in rows])))
            text.append(f'| {k} | {m} | {k-m} | {np.mean(kl):.6f} | {span} | {ppl:.4f} |')
    text += ['', '## 训练与解释范围', '',
        '沿用未修改的阶段6训练函数、width256/max_k24结构、种子579/580/581、相同文档顺序、batch16、24epochs、AdamW2e-4及逐文档平均softCE+0.25hardCE。保留到24步的专家与分叉监督，以匹配已有m=0/6检查点；本轮只选取和评价K=10/12。', '',
        '各m的可训练条件状态数不同，这是固定前缀变长后的结构差异，具体状态数保留在验收文件。分叉位于m+1/m+5/m+9/m+13步，使用与旧混合组相同的低成本第2至5候选扰动，并从整个已剪集合继续贪心。仅相同文档、输入和整个已剪集合可复用旧候选分数。', '',
        '配对表先在每条文档内平均三个模型，再按文档bootstrap；静态端点只有一条路径，不重复计为三个独立种子。区间未校正多重比较。m=6是默认超参数，本实验用于观察分工和平台区间，不要求6在每个设置最优。', '',
        '本阶段未使用PIQA、HellaSwag、WinoGrande、WikiText评分选择m；后续若在这些已审阅任务上追加评测，应标注为消融。未执行OPT恢复训练，也未新增吞吐量或显存测量。', '',
        f"验收{audit['verified_training_runs']}次新训练、{audit['verified_label_states']}个条件状态、{audit['verified_candidate_scores']}个候选分数、{audit['quality_rows_including_reused_controls']}条含复用端点的质量记录；{audit['preserved_files']}个既有文件哈希未变。4GPU/42CPU核心，耗时{summary['elapsed_seconds']/3600:.2f}小时。", '',
        f"运行目录：`{run.relative_to(root)}`。权重、语料和token数组只保留本地。", '']
    report = root / 'analysis/STAGE10_PREFIX_RESULTS_20261109.zh-CN.md'
    with report.open('x') as stream:
        stream.write('\n'.join(text))
    if args.publish:
        # Compact allowlist avoids uploading weights, tokens and bulky per-record logs.
        names = ['protocol.json', 'summary.json', 'verification.json', 'training_outcomes.json', 'status.json', 'allocation.json']
        files = [str((run / name).relative_to(root)) for name in names]
        files += [str(path.relative_to(root)) for path in sorted(run.glob('m*_seed*_history.json'))]
        files += [str(path.relative_to(root)) for path in sorted(run.glob('m*_seed*_checkpoints.json'))]
        files += [str(path.relative_to(root)) for path in sorted(run.glob('worker_probe_*.json'))]
        files += [str(path.relative_to(root)) for path in [report, quality, paired]]
        files += [str(path.relative_to(root)) for path in sorted(root.glob('tools/*prefix_ablation*.py'))]
        files += ['analysis/STAGE10_PREFIX_PROTOCOL_20261109.zh-CN.md', 'analysis/BASELINE_OFFICIAL_AUDIT_20261109.zh-CN.md']
        manifest = root / 'analysis/stage10_prefix_completed_upload_20261109.json'
        write_json(manifest, sorted(set(files)))
        subprocess.run([sys.executable, '-B', str(root / 'tools/publish_additions_v7.py'),
            '--root', str(root), '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Report matched fixed-prefix retraining development ablation',
            '--receipt', str(root / 'analysis/stage10_prefix_completed_publication_20261109.json')], cwd=root, check=True)
    print(json.dumps({'report': str(report), 'state': 'finalized'}), flush=True)


if __name__ == '__main__':
    main()
