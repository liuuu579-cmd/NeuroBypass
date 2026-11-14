"""Stage30 small-comparison report and safe aggregate publication manifest."""
import argparse
import json
from pathlib import Path

from full_context_experiment_v1 import write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--allowlist', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.run.resolve()
    cfg = load(out/'protocol.json')
    summary = load(out/'summary.json')
    assert load(out/'verification.json')['passed']
    replay = load(out/'historical_replay.json')
    assert replay['passed']
    preserved = load(out/'preservation_after.json')['preserved_files']
    means = {(r['task'], r['policy'], r['k']): r for r in summary['means']}
    names = {'pure': '原纯动态', 'hybrid': '原固定6层+动态', 'static_joint_kl': '原静态联合KL',
        'evopress_whole_opt': 'EvoPress适配', 'full': '完整模型'}
    lines = ['# 阶段30：阶段7原模型与EvoPress小规模横评', '',
        f'工程检查：{cfg["probe"]}。共{summary["input_records"]}个输入、{summary["quality_rows"]}条评分，',
        f'已核对旧路径、观察范围和预测与历史记录一致；原有{preserved}个文件哈希保持不变。', '',
        '这里的模型就是阶段7原版纯动态和固定6层+动态，均逐步生成层索引，不是后来从8套方案中选一套的版本。',
        '复用原有三个种子与逐K检查点；EvoPress复用阶段29三个64代搜索组合。没有新训练、搜索或根据下游结果选模型。', '',
        '每项选择题64题，Wiki32篇文章各一随机窗口，抽样种子300579；工程检查另用每任务4条。',
        'PIQA/Hella使用acc_norm，Wino使用acc；Wiki为条件PPL，按阶段7口径先平均各种子NLL再exp。', '',
        '| 任务 | K | 原纯动态 | 原固定6层+动态 | EvoPress适配 | 原静态联合KL | 完整模型 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task in ('piqa', 'hellaswag', 'winogrande', 'wikitext_conditional'):
        field = 'conditional_ppl_exp_mean_nll' if task == 'wikitext_conditional' else 'accuracy' if task == 'winogrande' else 'accuracy_norm'
        for k in cfg['budgets']:
            values = []
            for name in ('pure', 'hybrid', 'evopress_whole_opt', 'static_joint_kl', 'full'):
                row = means[task, name, 0 if name == 'full' else k]
                values.append(f'{row[field]:.3f}' if task == 'wikitext_conditional' else f'{100*row[field]:.3f}%')
            lines.append('| '+task+' | '+str(k)+' | '+' | '.join(values)+' |')
    lines += ['', '## 原固定6层+动态相对EvoPress的配对差异', '',
        '差异为动态减EvoPress；选择题单位为百分点，正值有利动态；Wiki单位为每token NLL，负值有利动态。',
        '先在同题平均三种子，再按题或Wiki文章进行2000次bootstrap。未作多重比较校正。', '',
        '| 任务 | K | 差异 | 95%区间 |', '|---|---:|---:|---|']
    for r in summary['paired_comparisons']:
        if r['dynamic'] != 'hybrid' or r['baseline'] != 'evopress_whole_opt':
            continue
        scale = 1 if r['task'] == 'wikitext_conditional' else 100
        v = r['dynamic_minus_baseline']
        lines.append(f'| {r["task"]} | {r["k"]} | {scale*v["difference"]:+.4f} | '
            f'[{scale*v["ci95"][0]:+.4f}, {scale*v["ci95"][1]:+.4f}] |')
    lines += ['', '## 核对与适用范围', '',
        '逐条历史重放误差：`'+json.dumps(replay['max_abs_score_error'])+'`；验收阈值2e-4，所有路由和选择题预测一致。',
        '旧模型的128-token观察边界原样保留。阶段7选择题最长106 token，故本次选择题没有因128上限而丢弃可见提示词；',
        'Wiki仍为128条件token+最多384目标token，不是标准rolling PPL。双方使用完全相同的编码输入、评分和题目。',
        '但模型校准/训练预算不同：EvoPress原生全序列KL、变长TRAIN校准、64代；旧路由器在旧128-token训练记录上训练。',
        '这是已多次审阅任务人口中的小样本方向复核，不证明全面领先、独立泛化或统计非劣效。',
        '主表三种子平均不是集成。未测吞吐、首token延迟或峰值显存，本结果不含推理加速结论。', '',
        f'运行目录：`{out.relative_to(root)}`。GPU0/1/2、32CPU，仅新增输出，权重/token/逐题记录留在本地。']
    with args.report.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    files = ['tools/compare_stage7_evopress_three_gpu_v1.py', 'tools/finalize_stage7_evopress_v1.py',
        'tools/launch_stage7_evopress_v1.py', 'analysis/STAGE30_STAGE7_VS_EVOPRESS_PROTOCOL_20261114.zh-CN.md',
        str(args.report.resolve().relative_to(root))]
    for name in ('protocol.json', 'data_manifest.json', 'summary.json', 'verification.json',
            'historical_replay.json', 'preservation_after.json'):
        files.append(str((out/name).relative_to(root)))
    for pattern in ('worker_probe_*.json', 'worker_receipt_*.json'):
        files += [str(path.relative_to(root)) for path in sorted(out.glob(pattern))]
    manifest = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(out.glob('private_*'))}
    write(out/'evidence_hashes.json', manifest)
    files.append(str((out/'evidence_hashes.json').relative_to(root)))
    write(args.allowlist, files)
    print(json.dumps({'report': str(args.report), 'public_files': len(files)}), flush=True)


if __name__ == '__main__':
    main()
