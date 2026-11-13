"""Wait for Stage28 finite training, audit independently, report and publish additions."""
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PREFIX = 'stage28_window_supervision_20261113_01'


def read(path):
    return json.loads(path.read_text())


def conclusion(root, run):
    summary = read(run / 'summary.json')
    own = 'mean128_embedding_gate_mean3'
    def score(policy):
        return next(r['mean'] for r in summary['outer_summary'] if r['policy'] == policy and r['metric'] == 'kl' and r['window'] == 'all128')
    def comparison(baseline):
        return next(r for r in summary['comparisons'] if r['policy'] == own and r['baseline'] == baseline and r['metric'] == 'kl' and r['window'] == 'all128')
    comparisons = [comparison(b) for b in ('first32_embedding_gate_mean3', 'mean128_constant_gate_mean3', 'fold_static')]
    target_better = comparisons[0]['difference'] > 0
    dynamic_better = comparisons[1]['difference'] > 0 and comparisons[2]['difference'] > 0
    supported = all(r['ci95'][0] > 0 for r in comparisons)
    text = ['# 阶段28结论：多窗口联合KL监督', '',
        '本轮已完成配对训练与独立审计。351条旧训练文档，四折外折复核，三个种子；first32/mean128监督各训练全文embedding与无输入constant，共48次正式拟合。固定K20/40和8个完整组合；训练标签来自同一128-token长前向，完整提示词输入不变。', '',
        '| 策略 | 全128-token KL，越低越好 |', '|---|---:|']
    for policy in ('fold_static', 'historical_static1', 'first32_embedding_gate_mean3', own,
        'first32_constant_gate_mean3', 'mean128_constant_gate_mean3', 'future_known_best128'):
        text.append(f'| {policy} | {score(policy):.9f} |')
    text += ['', '| 多窗口动态相对参考的KL改善 | 均值 | 探索性95%区间 |', '|---|---:|---|']
    for row in comparisons:
        lo, hi = row['ci95']; text.append(f"| {row['baseline']} | {row['difference']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
    text += ['',
        ('多窗口监督在主要均值对比上优于首窗监督。' if target_better else '多窗口监督在主要均值对比上没有优于首窗监督。'),
        ('多窗口动态的全128 KL均值同时低于对应无输入常量与折内静态。' if dynamic_better else '多窗口动态没有同时在全128 KL均值上优于对应无输入常量与折内静态。'),
        ('三个预先指定比较的探索性文档bootstrap区间均为正，但仍需独立数据验证。' if supported else '预先指定的三个比较未全部得到正向区间支持，本轮不足以建立稳定动态收益。'), '',
        '检查点与门槛统一由每折内层128-token KL选择；折内静态仅由拟合子集选择。各目标可有不同拟合先验/缩放，但四组可学习参数初始化、顺序、优化器及更新预算配对。mean3为独立种子的质量均值，不是集成。独立审计恢复原文与分组，复核目标/检查点/门槛、逐token算术、路由与bootstrap，并在真实OPT上重放全部8组合和全文embedding。', '',
        '此为已多次查看的旧训练文档上的开发小试；候选库历史上用过其中部分文档，351条又受长续写可用性筛选，不能宣称独立泛化或最终统计显著性。future_known_best128读取未来，仅表示候选库诊断空间。保持回答内固定路径；本轮不声称任务准确率、吞吐、峰值显存或恢复微调改善。', '',
        '下一步判断：若多窗口只改善相对首窗而未胜过常量/静态，应保留其作为更合适的训练目标，但不要继续以单纯延长监督窗口作为动态创新证据。若三个参考均出现有意义改善，再锁定方案进行新的独立文档与任务级验证。', '',
        f'[完整结果]({PREFIX}_main_results.zh-CN.md) · [独立审计](../runs/{PREFIX}_main/independent_audit.json)']
    path = root / 'analysis/STAGE28_WINDOW_SUPERVISION_CONCLUSION_20261113.zh-CN.md'
    with path.open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    return path


def main():
    root = Path(__file__).resolve().parents[1]; cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cpus[40:42])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    completion = root / 'analysis' / (PREFIX + '_completion.json')
    started = time.monotonic()
    while not completion.exists():
        if time.monotonic()-started > 10800:
            raise TimeoutError('Finite Stage28 training did not complete within3hours')
        time.sleep(5)
    assert read(completion)['state'] == 'completed', 'Training failed; existing evidence preserved'
    run = root / 'runs' / (PREFIX + '_main'); probe = root / 'runs' / (PREFIX + '_probe')
    audit = root / 'runs' / (PREFIX + '_audit'); log = root / 'analysis' / (PREFIX + '_audit.log')
    assert not audit.exists()
    ast.parse((root / 'tools/audit_window_supervision_four_gpu_v1.py').read_text())
    command = [sys.executable, '-B', '-u', str(root / 'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3',
        '--run-dir', str(audit), '--', sys.executable, '-B', '-u', str(root / 'tools/audit_window_supervision_four_gpu_v1.py'), '--run', str(run)]
    with log.open('xb') as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    assert result.returncode == 0, 'Independent audit failed; do not publish unverified conclusions'
    assert read(run / 'independent_audit.json')['passed']
    conclusion_path = conclusion(root, run)
    sys.path.insert(0, str(root / 'tools'))
    from run_static_swap_pipeline_v1 import publish
    from full_context_experiment_v1 import write
    files = [root / 'tools' / name for name in ('window_supervision_core_v1.py', 'run_window_supervision_four_gpu_v1.py',
        'launch_window_supervision_v1.py', 'audit_window_supervision_four_gpu_v1.py', 'finish_window_supervision_v1.py', 'finish_window_supervision_v2.py')]
    files += [conclusion_path, completion, root / 'analysis' / (PREFIX + '_main_results.zh-CN.md')]
    files += [run / name for name in ('protocol.json', 'sampling.json', 'summary.json', 'policy_choices.json',
        'paired_initialization.json', 'verification.json', 'independent_audit.json', 'preservation_after.json', 'allocation.json', 'status.json')]
    for rank in range(4):
        files += [run / f'labels_worker_{rank:03d}.json', run / f'worker_{rank:03d}.json', audit / f'worker_{rank:03d}.json']
        for seed in (579, 580, 581):
            for arm in ('first32_embedding', 'mean128_embedding', 'first32_constant', 'mean128_constant'):
                files += [run / f'{arm}_f{rank}_s{seed}_fit.json', run / f'{arm}_f{rank}_s{seed}_history.json']
    files += [probe / name for name in ('protocol.json', 'sampling.json', 'verification.json', 'preservation_after.json', 'status.json')]
    files += [audit / name for name in ('allocation.json', 'status.json')]
    assert len(files) == len(set(files))
    assert all(p.is_file() and p.stat().st_size < 1024*1024 and not p.name.startswith('private_') for p in files)
    receipt = publish(root, files, PREFIX + '_audited',
        'Compare matched first32 and four-window joint KL supervision with no-input controls and independent four-fold audit', dict(os.environ))
    write(run / 'publication_verified.json', receipt)
    print(json.dumps({'finished': True, 'publication': receipt}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
