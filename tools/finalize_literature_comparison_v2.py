"""Aggregate-only publication material for Stage29; never publish private rows."""
import argparse
import json
from pathlib import Path

from full_context_experiment_v1 import write
from stage1_benchmark_v2 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--allowlist', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.run.resolve()
    cfg = json.loads((out/'protocol.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert json.loads((out/'verification.json').read_text())['passed']
    assert json.loads((out/'preservation_after.json').read_text())['preserved_files'] > 0
    scope = '工程小试' if cfg['probe'] else '完整横评'
    lines = [f'# 阶段29：EvoPress 与冻结动态路由的{scope}', '',
        '比较 OPT-13B 在剪去12/16/20/24个完整解码层（30%/40%/50%/60%）后的质量。',
        '所有基座权重冻结，无恢复微调；保护原始层0/1，每道题所有选项共用一次完整提示词路由。', '',
        'EvoPress 使用整层模式、原生全序列 next-token 教师KL、多阶段候选筛选和精英保留。',
        '校准来自原TRAIN文档，未用任务分数选择静态组合或动态检查点。主实验为64初始候选、32子代、64代、三搜索种子；',
        '筛选预算为2048/32768输入token（末档不超过可用TRAIN token）。这一预算低于论文脚本，结果只代表本次适配与搜索预算。',
        '既有静态联合KL以32个续写位置为目标，EvoPress为全序列目标；强静态bank还使用额外TRAIN搜索，不能据此声称等搜索成本。',
        'SLEB、ShortGPT、Shortened也为冻结OPT适配版本，没有复现各论文全部训练配方。', '',
        '动态主参照固定为Stage13 m=6，各K和种子均使用先前选定检查点；K20另列Stage19 bank8路由。没有新训练或按横评分数挑种子。',
        '结果按独立种子平均，不是集成投票。保留静态联合KL与K20强静态，以区分文献比较和输入适配收益。', '',
        '前三项选择题指标为准确率（PIQA/HellaSwag使用acc_norm），Wiki为token加权条件PPL，越低越好。',
        '这些是已使用过的任务人口，并非新保留终测；Wiki不是标准rolling PPL。小试的8题/任务仅检查工程流程。', '',
        '| 任务 | K | 方法 | 种子数 | 指标 |', '|---|---:|---|---:|---:|']
    grouped = {}
    for row in summary['quality']:
        grouped.setdefault((row['task'], row['k'], row['policy']), []).append(row)
    for (task, k, name), rows in sorted(grouped.items()):
        field = 'conditional_ppl' if task == 'wikitext_conditional' else 'accuracy' if task == 'winogrande' else 'accuracy_norm'
        value = sum(r[field] for r in rows)/len(rows)
        shown = f'{value:.4f}' if field == 'conditional_ppl' else f'{100*value:.3f}%'
        lines.append(f'| {task} | {k} | {name} | {len(rows)} | {shown} |')
    lines += ['', '配对差异、未校正95%区间、逐种子指标和质量审计见同运行目录的summary.json/verification.json。',
        '先在同题上平均独立种子，再按题重采样；Wiki按文章聚类。Wiki配对项使用窗口平均NLL，与token加权PPL另列。',
        '不能因若干均值领先就宣称全面优于所有静态方法；搜索收敛、等预算恢复与其他骨干仍需分别验证。', '',
        '参考：[EvoPress论文](https://arxiv.org/abs/2410.14649)，[官方实现](https://github.com/IST-DASLab/EvoPress)。',
        '官方代码审阅固定到 a66a6974e91d27ede9043f07f885709732a9a9b1；本仓库仅发布独立适配程序，不上传第三方副本。', '',
        f'运行目录：`{out.relative_to(root)}`。']
    with args.report.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    files = ['tools/literature_evolution_core_v1.py', 'tools/run_literature_comparison_four_gpu_v2.py',
        'tools/finalize_literature_comparison_v2.py', 'tools/launch_literature_comparison_v2.py',
        'analysis/STAGE29_LITERATURE_COMPARISON_PROTOCOL_20261113.zh-CN.md', str(args.report.resolve().relative_to(root))]
    for name in ('protocol.json', 'data_manifest.json', 'algorithm_verification.json', 'summary.json',
            'verification.json', 'preservation_after.json'):
        files.append(str((out/name).relative_to(root)))
    for pattern in ('gpu_verification_*.json', 'evopress_k*_s*.json', 'search_receipt_*.json', 'quality_receipt_*.json'):
        files += [str(p.relative_to(root)) for p in sorted(out.glob(pattern))]
    # Do not publish per-document scores/routes, tokens, embeddings, weights or
    # raw corpus. Search traces stay local, aggregates have exact file hashes.
    private = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
        for pattern in ('private_*', 'search_trace_*.jsonl') for p in sorted(out.glob(pattern))}
    receipt = out/'private_evidence_hashes.json'
    write(receipt, private)
    files.append(str(receipt.relative_to(root)))
    # Publisher disallows names starting private_, so keep the hashes in a
    # separately named public manifest; the private receipt itself stays local.
    files.remove(str(receipt.relative_to(root)))
    public = out/'evidence_hashes.json'
    write(public, private)
    files.append(str(public.relative_to(root)))
    write(args.allowlist, files)
    print(json.dumps({'report': str(args.report), 'public_files': len(files)}), flush=True)


if __name__ == '__main__':
    main()
