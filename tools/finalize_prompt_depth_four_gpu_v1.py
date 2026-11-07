"""Finalize a completed deeper experiment, then publish immutable additions."""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    config = json.loads((run / 'protocol.json').read_text())
    if config['probe']: raise ValueError('A small probe is not a final training result')
    subprocess.run([sys.executable, '-B', str(root / 'tools/verify_prompt_depth_four_gpu_v4.py'),
        '--run', str(run)], cwd=root, check=True)
    audit = json.loads((run / 'verification.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    report = root / 'analysis/STAGE6_FOUR_GPU_DEPTH_RESULTS_20261107.zh-CN.md'
    csv_path = root / 'analysis/stage6_four_gpu_depth_quality_20261107.csv'
    comparisons = root / 'analysis/stage6_four_gpu_depth_paired_20261107.csv'
    for path, rows in [(csv_path, summary['quality']), (comparisons, summary['paired_comparisons'])]:
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    text = ['# 四卡深层剪枝训练：静态、纯动态与混合对照', '',
        '本轮新增程序支持max_k=24，实际生成深层联合标签并训练了纯动态/混合两组、三个种子。OPT权重冻结；每轮由128个提示词embedding决定固定跳层路径，用于整个提示词和续写。', '',
        f"训练记录{len(config['train_ids'])}条，开发验证{len(config['validation_ids'])}条；旧100条test未用于本轮评估。静态校准固定为训练集中的{len(config['static_calibration_ids'])}条。", '',
        f"混合组固定前缀为 `{config['fixed_prefix']}`，直接来自本轮训练校准静态联合KL的前6步，没有使用验证集选择全局层。", '',
        '## 主结果', '',
        '以下PPL为128+32自然续写条件PPL，不能冒充标准WikiText/全量C4 PPL。动态组列出三个训练种子的均值与KL范围；PPL先跨文档和种子平均NLL再取exp。验证集已反复用于开发，结果不是独立测试。', '',
        '| K | 方法 | KL均值 | KL种子范围 | 条件PPL |',
        '| ---: | --- | ---: | --- | ---: |']
    all_rows = [r for r in summary['quality'] if r['population'] == 'validation_all100']
    for k in config['budgets']:
        for arm in ['pure', 'hybrid']:
            group = [r for r in all_rows if r['k'] == k and r['arm'] == arm and r['policy'] == 'matched']
            values = [r['continuation_kl'] for r in group]
            ppl = math.exp(float(np.mean([r['continuation_nll'] for r in group])))
            text.append(f"| {k} | {'纯动态' if arm == 'pure' else '固定6层+动态余层'} | {np.mean(values):.6f} | {min(values):.6f}–{max(values):.6f} | {ppl:.4f} |")
        for policy, name in [('static_joint_kl', '静态联合KL贪心'), ('sleb_nll_opt', 'SLEB-NLL OPT适配'),
                ('shortgpt_bi_opt', 'ShortGPT-BI OPT适配'), ('shortened_ppl_opt', 'Shortened-PPL OPT适配'),
                ('shortened_ppl_plus', 'Shortened-PPL+首尾保护'),
                ('offline_pure_greedy', '看到真实续写的离线逐样本贪心')]:
            row = next(r for r in all_rows if r['k'] == k and r['policy'] == policy)
            text.append(f"| {k} | {name} | {row['continuation_kl']:.6f} | — | {row['perplexity']:.4f} |")
    text += ['', '## 协议与解释范围', '',
        '- 每步评分整个已跳集合加候选层，保留所有候选代价；只在同文档、同输入、整个集合相同时复用旧标签。',
        '- 纯动态覆盖24个主路径状态，混合覆盖固定6层后的18个主路径状态。每条训练文档另有一个浅层或深层低成本分叉，贪心完成到24层。',
        '- 使用原分类目标soft CE+0.25 hard CE，逐文档等权重；不继续未获益的pairwise损失搜索。',
        '- 两组均全新初始化width256 Router，batch16、24epochs、种子579/580/581；按每个K在原固定32条开发记录上的KL独立选检查点。完整100条及其余68条都属于已使用的开发数据。',
        '- SLEB按全部159个next-token NLL逐步重评分；Shortened-PPL按第一次排名取前K，plus额外保护0/1/2/3/38/39；ShortGPT按160个位置输入/输出余弦差一次排名。其余方法保护0/1。数据/保护层配置属于OPT适配，不宣称完整复现默认论文。',
        '- 对每个种子分别做同输入与置换输入方案配对，保留文档bootstrap结果；不能把三个种子乘以100条当作300个独立测试样本。',
        '- 离线逐样本贪心见到了真实续写，不是可部署路由器、全局最优保证或可由提示词学到的上界。',
        '- K=20/24保留为压力点；不能单靠最深K或KL替代下游质量预算。正式任务评测、恢复训练和部署计时仍未在本轮执行。', '',
        '## 数值与资源验收', '',
        '初次候选异构batch在深层出现较大的FP16偏差，因此本轮正式标签全部改为batch1真实跳层，并只复用未改变的前缀activation。候选探针对照完整逐条推理，保持1e-6标量差容差。早期失败的v1/v2及运行目录完整保留。', '']
    for path in sorted(run.glob('cache_probe_*.json')):
        for item in json.loads(path.read_text()):
            half = item['fp16']
            text.append(f"- K={len(half['route'])}：FP16缓存/重算平均绝对位置KL={half['mean_absolute_position_kl']:.7g}，最大={half['max_absolute_position_kl']:.7g}；{item['outcome']}。")
    text += ['', '缓存检查对应少量固定静态路径，不能证明所有动态路径稳定。FP16失败时单独保留，并用FP32确认结构；不放宽缓存原容差，也不把FP32通过当作FP16通过。', '',
        f"本轮GPU0/1/2/3，共42核预算；四worker各10个互不重叠核心，父控制器余2核。验收{audit['verified_label_states']}个标签状态、{audit['verified_candidate_scores']}个候选分数、{audit['verified_quality_rows']}条质量记录。{audit['source_preserved_files']}个既有文件哈希未变。耗时{summary['elapsed_seconds']/3600:.2f}小时。", '',
        f"来源：`{run.relative_to(root)}`。原始质量表见`{csv_path.name}`；配对结果见`{comparisons.name}`。未上传权重、token数组或语料。", '']
    with report.open('x') as stream: stream.write('\n'.join(text))
    if args.publish:
        files = [str(path.relative_to(root)) for path in run.iterdir() if path.is_file()
            and path.suffix in {'.json', '.jsonl', '.log'} and not path.name.startswith('private_')]
        files += [str(path.relative_to(root)) for path in [report, csv_path, comparisons]]
        files += [str(path.relative_to(root)) for path in root.glob('tools/*prompt_depth*.py')]
        files += [str(path.relative_to(root)) for path in root.glob('tools/run_depth_four_gpu*.py')]
        files.append('analysis/STAGE6_FOUR_GPU_TRAINING_PROTOCOL_20261107.zh-CN.md')
        manifest = root / 'analysis/stage6_four_gpu_completed_upload_20261107.json'
        files.append(str(manifest.relative_to(root)))
        write_json(manifest, sorted(set(files)))
        subprocess.run([sys.executable, '-B', str(root / 'tools/publish_additions_v7.py'), '--root', str(root),
            '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Complete four-GPU deep pruning routers with static and hybrid controls',
            '--receipt', str(root / 'analysis/stage6_four_gpu_completed_publication_20261107.json')], check=True, cwd=root)
    print(json.dumps({'report': str(report), 'summary_sha256': sha256(run / 'summary.json')}), flush=True)


if __name__ == '__main__': main()
