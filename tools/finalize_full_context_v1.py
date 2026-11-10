"""Read-only run audits, paired development statistics and append-only report."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def audit_run(run, kind):
    if json.loads((run/'status.json').read_text())['state'] != 'completed':
        raise ValueError('Run did not complete: '+str(run))
    allocation = json.loads((run/'allocation.json').read_text())
    assert allocation['cpu_count'] == len(set(allocation['cpu_ids'])) == 42
    assert len(set(allocation['gpu_indices'])) == 4
    before = json.loads((run/'preservation_before.json').read_text())
    for name, digest in before.items():
        if sha256(name) != digest: raise ValueError('Preexisting evidence changed: '+name)
    assert json.loads((run/'preservation_after.json').read_text())['preserved_files'] == len(before)
    config = json.loads((run/'protocol.json').read_text())
    for name, digest in config['source_sha256'].items():
        assert sha256(name) == digest
    verification = json.loads((run/('training_verification.json' if kind == 'train' else 'verification.json')).read_text())
    assert verification['passed'] is True
    for rank in range(4):
        p = run / (f'worker_{rank:03d}.json' if kind == 'cost' else f'worker_probe_{rank:03d}.json')
        row = json.loads(p.read_text())
        allowed = allocation['cpu_ids'][rank*10:(rank+1)*10]
        assert row['cpu_ids'] == allowed and row['threads'] == 10
        assert all(set(cpus) <= set(allowed) for cpus in row['thread_cpu_ids'].values())
        if kind == 'train':
            for probe in row['candidate_probes']:
                assert probe['max_abs_error'] <= probe['tolerance'] == 1e-6
        elif kind == 'quality':
            for probe in row['likelihood_probes']: assert probe['absolute_error'] <= probe['tolerance']
    if kind == 'train':
        assert json.loads((run/'label_verification.json').read_text())['passed']
        assert verification['test_documents_used'] == 0
        for cp in verification['checkpoints']: assert sha256(cp['path']) == cp['sha256']
    return {'run': str(run), 'kind': kind, 'passed': True, 'preserved_files': len(before)}


def paired(run):
    config = json.loads((run/'protocol.json').read_text())
    inputs = read_lines(run/'private_inputs.jsonl')
    rows = [r for p in run.glob('quality_*.jsonl') for r in read_lines(p)]
    by = {(r['task'], r['sample'], r['policy'], r['seed'], r['k'], r['condition']): r for r in rows}
    rng, result = np.random.default_rng(579), []
    arms = sorted({(p['name'], p['k']) for p in config['policies'] if p['kind'] == 'router'})
    for task in sorted({r['task'] for r in inputs}):
        populations = [('all', [r for r in inputs if r['task'] == task])]
        if task == 'c4_development':
            for lo, hi in [(1, 128), (129, 511), (512, 1023), (1024, 2048)]:
                subset = [r for r in inputs if r['task'] == task and lo <= len(r['route_ids']) <= hi]
                if subset: populations.append((f'length_{lo}_{hi}', subset))
        for population, local in populations:
            documents = [r['document'] if task == 'wikitext_conditional' else r['sample'] for r in local]
            _, inverse = np.unique(documents, return_inverse=True)
            units = int(inverse.max())+1
            draws = rng.multinomial(units, np.full(units, 1/units), size=2000)
            for arm, k in arms:
                seeds = sorted(p['seed'] for p in config['policies'] if p['name'] == arm and p['k'] == k)
                def field(record, seed, condition, metric):
                    row = by[task, record['sample'], arm, seed, k, condition]
                    if metric == 'kl': return row['teacher_kl_sum']/row['target_tokens']
                    return row['correct'] if task == 'winogrande' else row['correct_norm']
                metrics = ['kl'] if local[0]['gold'] is None else ['kl', 'accuracy']
                for metric in metrics:
                    matched = np.array([np.mean([field(r, s, 'matched', metric) for s in seeds]) for r in local])
                    shuffled = np.array([np.mean([field(r, s, f'shuffled_{p}', metric)
                        for s in seeds for p in [1701, 1702, 1703]]) for r in local])
                    difference = shuffled-matched if metric == 'kl' else matched-shuffled
                    totals = np.bincount(inverse, weights=difference, minlength=units)
                    counts = np.bincount(inverse, minlength=units)
                    bootstrap = (draws@totals)/(draws@counts)
                    result.append({'task': task, 'population': population, 'policy': arm, 'k': k,
                        'metric': metric, 'documents': units, 'inputs': len(local), 'seeds': seeds,
                        'matched': float(matched.mean()), 'shuffled_mean3': float(shuffled.mean()),
                        'matched_benefit': float(difference.mean()),
                        'ci95': np.quantile(bootstrap, [.025, .975]).tolist()})
    return {'comparisons': result, 'method': 'Average seeds/permutations within input first; paired input bootstrap, Wiki article clusters. Positive benefit favors matched.',
        'limits': 'Exploratory development comparisons, no multiplicity correction. C4 donor permutations cross length strata, so content-only causality is not established.'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path); p.add_argument('--kind', choices=['train', 'quality', 'cost'])
    p.add_argument('--all-runs', type=Path); p.add_argument('--report', type=Path); p.add_argument('--upload-list', type=Path)
    args = p.parse_args()
    if args.run:
        result = audit_run(args.run.resolve(), args.kind)
        write(args.run/'independent_audit.json', result)
        print(json.dumps(result)); return
    runs = json.loads(args.all_runs.read_text()); root = Path(__file__).resolve().parents[1]
    audits = [audit_run(Path(runs[name]), kind) for name, kind in
        [('training', 'train'), ('development', 'quality'), ('horizontal', 'quality'), ('cost', 'cost')]]
    train, dev, horizontal, cost = [Path(runs[k]) for k in ['training', 'development', 'horizontal', 'cost']]
    paired_results = paired(dev); write(dev/'paired_summary.json', paired_results)
    sections = ['# 完整提示词路由修复与重训重测', '',
        '旧推理入口仅观察前128 token。新版本读取基座实际可见的完整提示词，超长时显式保留最近上下文，路由与prefill使用完全相同的token。答案和未来续写不进入预测器。', '',
        '旧训练本来只有128-token提示词；旧短题与Wiki128-token条件上下文结果保留其原协议含义。长提示词成本测试存在截断，须以本轮新部署结果为准。新旧训练分布和静态校准均有变化，因此差异不能仅归因于修复一行截断。', '',
        '新训练保留原800/100文档划分；预留100测试文档未使用。m=0/4/6/8，各三个种子，逐步条件联合KL贪心与替代分支，冻结基座，无恢复微调。全部运行经4卡/42核资源与旧文件哈希审计。', '',
        '## 开发集输入匹配诊断', '', '| 任务 | m | K | matched KL | shuffled KL | 改善及95%CI |', '|---|---:|---:|---:|---:|---:|']
    for row in paired_results['comparisons']:
        if row['population'] == 'all' and row['policy'] == 'm06' and row['k'] in [10, 12] and row['metric'] == 'kl':
            sections.append(f"| {row['task']} | 6 | {row['k']} | {row['matched']:.6f} | {row['shuffled_mean3']:.6f} | {row['matched_benefit']:.6f} [{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}] |")
    sections += ['', '先在输入内平均模型种子和三次置换，再按问题/文档bootstrap；Wiki按文章聚类。完整任务准确率、长度分组及全部m/K见paired_summary。仍属开发诊断；跨长度供体不能证明仅由内容匹配产生收益。', '',
        '## 完整横向评测', '', '| 任务 | 方法 | seed | K | 质量指标 |', '|---|---|---:|---:|---:|']
    quality = json.loads((horizontal/'summary.json').read_text())['quality']
    for row in quality:
        if row['condition'] != 'matched': continue
        metric = row.get('accuracy_norm', row.get('accuracy', row.get('conditional_ppl')))
        sections.append(f"| {row['task']} | {row['policy']} | {row['seed']} | {row['k']} | {metric:.6f} |")
    sections += ['', 'PIQA/HellaSwag报告acc_norm，WinoGrande为acc，Wiki为128-token条件窗口PPL。论文方法是无恢复的OPT适配版，不代表完整官方配方。', '',
        '## 部署成本', '', '| GPU任务 | 部署 | 实际输入长度 | 热TTFT秒 | 热端到端token/s | allocated峰值GiB |', '|---:|---|---:|---:|---:|---:|']
    for path in sorted(cost.glob('worker_*.json')):
        worker = json.loads(path.read_text())
        for deployment in ['selective_cpu_bank', 'resident_dynamic', 'full', 'static_physical']:
            group = [r for r in worker['records'] if r['deployment'] == deployment and not r['cold']]
            sections.append(f"| {worker['rank']} | {deployment} | {min(r['prompt_tokens'] for r in group)}–{max(r['prompt_tokens'] for r in group)} | {np.mean([r['ttft_seconds'] for r in group]):.4f} | {np.mean([r['end_to_end_tokens_per_second'] for r in group]):.3f} | {max(r['peak_allocated_bytes'] for r in group)/2**30:.3f} |")
    sections += ['', '按需加载包含换层搬运与完整prefill；静态物理删层的布局准备不计入稳态请求时间。冷启动、100ms采样进程显存峰值与CPU权重银行内存见worker证据。采样峰值是下界；这里是独立请求，未测真实多轮对话或并发服务。', '',
        '恢复微调暂未进行，先用本轮冻结权重结果重新判断需要；动态与静态方案应使用相同恢复预算。', '', '## 证据', '']
    for name, run in runs.items(): sections.append(f'- {name}: `{run}`')
    with args.report.open('x') as stream: stream.write('\n'.join(sections)+'\n')
    write(root/'analysis/stage13_full_context_final_audit_20261110.json', {'passed': True, 'audits': audits})
    files = [str(args.report.relative_to(root)), 'analysis/stage13_full_context_final_audit_20261110.json']
    for run in [train, dev, horizontal, cost]:
        for path in sorted(run.glob('*.json')):
            if path.name.startswith('private_') or path.name in ['preservation_before.json', 'allocation.json']: continue
            files.append(str(path.relative_to(root)))
    # Token arrays and private per-example inputs/checkpoints are never public.
    write(args.upload_list, sorted(set(files)))


if __name__ == '__main__': main()
