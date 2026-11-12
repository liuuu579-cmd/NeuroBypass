"""Read-only Stage19 allocation diagnosis; future-informed bounds are NOT policies."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def reference_bound(costs, budgets, reference_k, slack):
    """Future-informed mean-K bound under one actual mean-KL constraint.

    LP can randomize at one boundary document. Also give a feasible deterministic
    allocation by picking the lowest-KL supported choice in a fractional row.
    Neither allocation is observable at inference or a runtime measurement.
    """
    costs = np.asarray(costs, dtype=float)
    n, m = costs.shape
    ref = budgets.index(reference_k)
    limit = float(costs[:, ref].mean() + slack)
    equality = csr_matrix((np.ones(n * m), (np.repeat(np.arange(n), m), np.arange(n * m))), shape=(n, n * m))
    solution = linprog(-np.tile(budgets, n) / n, A_ub=costs.reshape(1, -1) / n, b_ub=[limit],
        A_eq=equality, b_eq=np.ones(n), bounds=(0, 1), method='highs', options={'threads': 1})
    assert solution.success, solution.message
    probs = solution.x.reshape(n, m)
    chosen = []
    fractional = 0
    for i, row in enumerate(probs):
        support = np.flatnonzero(row > 1e-7)
        assert len(support)
        fractional += int(len(support) > 1)
        chosen.append(int(min(support, key=lambda j: (costs[i, j], -budgets[j]))))
    kl = float(costs[np.arange(n), chosen].mean())
    assert kl <= limit + 1e-7
    return {'reference_k': reference_k, 'kl_allowance': slack, 'reference_mean_kl': float(costs[:, ref].mean()),
        'future_informed_fractional_mean_k_upper_bound': float(-solution.fun), 'fractional_rows': fractional,
        'future_informed_feasible_mean_k': float(np.mean(np.array(budgets)[chosen])),
        'future_informed_feasible_mean_kl': kl, 'future_informed_k_counts': dict(sorted(Counter(budgets[i] for i in chosen).items())),
        'future_targets_used': True, 'runtime_measured': False}


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs/stage19_parallel_20261112_03_budget'
    prefix = 'stage20_budget_allocation_diagnosis_20261112_01'
    cfg = load(run / 'protocol.json')
    bank = load(run / 'budget_bank.json')
    paths = [run / f'private_quality_{i:03d}.jsonl' for i in range(3)]
    for rank, path in enumerate(paths):
        assert sha256(path) == load(run / f'quality_worker_{rank:03d}.json')['quality_sha256']
    rows = [r for p in paths for r in read_lines(p)]
    inputs = read_lines(run / 'private_inputs.jsonl')
    index = {(r['sample'], r['policy'], r['seed']): r for r in rows}
    selection = set(cfg['selection_ids'])
    bounds = []
    for cohort in ('selection32', 'other68', 'mc96'):
        ids = [r['sample'] for r in inputs if (r.get('gold') is not None if cohort == 'mc96' else
            r['task'] == 'c4_development' and ((r['source_sample'] in selection) == (cohort == 'selection32')))]
        costs = [[index[sample, f'static_k{k:02d}', None]['kl'] for k in bank['budgets']] for sample in ids]
        for slack in (0., .02):
            bounds.append({'cohort': cohort, 'documents': len(ids), **reference_bound(costs, bank['budgets'], 12, slack)})
    actual = [r for r in rows if r['policy'] == 'prompt_ref12_slack020' and r['seed'] == 580]
    groups = []
    for task in sorted({r['task'] for r in actual}):
        items = [r for r in actual if r['task'] == task]
        groups.append({'task': task, 'records': len(items), 'k_counts': dict(sorted(Counter(r['k'] for r in items).items())),
            'mean_k': float(np.mean([r['k'] for r in items])), 'mean_prompt_tokens': float(np.mean([r['prompt_tokens'] for r in items]))})
    by_budget = []
    for k in sorted({r['k'] for r in actual}):
        items = [r for r in actual if r['k'] == k and 'correct_norm' in r]
        if items:
            by_budget.append({'k': k, 'questions': len(items), 'mean_prompt_tokens': float(np.mean([r['prompt_tokens'] for r in items])),
                'dynamic_correct': sum(r['correct_norm'] for r in items),
                'static12_correct_on_same_questions': sum(index[r['sample'], 'static_k12', None]['correct_norm'] for r in items)})
    result = {'passed': True, 'source_run': str(run), 'actual_seed': 580,
        'policy_formula': 'argmin_K(predicted continuation KL(K|full prompt) + lambda*(40-K)/40)',
        'actual_lambda': 3., 'actual_group_allocations': groups, 'mc_group_diagnostics': by_budget,
        'future_informed_quality_budget_bounds': bounds,
        'source_sha256': sha256(Path(__file__)),
        'limits': 'Bounds use actual future continuation KL and are offline diagnostics only, not deployable routing, accuracy guarantees, or measured speedups. Fixed nested mask path only. Existing reviewed development populations.'}
    destination = root / 'analysis' / (prefix + '.json')
    write(destination, result)
    report = root / 'analysis' / (prefix + '.zh-CN.md')
    lines = ['# 路线5预算分配诊断', '',
        '当前预测器读取完整提示词embedding，预测每个固定K组合的续写KL，再最小化预测KL加执行层比例的代价。实际动态种子580采用lambda=3；K候选为0/6/8/10/12/16/20/24。该规则未直接使用题目正确率或实测每请求耗时，也不是手写的“难题少剪、简单题多剪”。', '',
        '## 实际分配', '', '| 群体 | 数量 | K分布 | 平均K |', '|---|---:|---|---:|']
    for r in groups:
        lines.append(f"| {r['task']} | {r['records']} | {r['k_counts']} | {r['mean_k']:.3f} |")
    lines += ['', '## 看到真实续写后的离线预算空间', '',
        '以下用真实未来续写KL分配预算，属于不可部署的诊断参考。每个群体约束平均KL不超过静态K12加给定容差，再尽量增加平均跳层数。线性规划可能在边界文档混合两个K；确定性可行值将该文档换成支持集内KL更低的K，仍满足约束。没有把这些数值当作线上收益或加速。', '',
        '| 群体 | KL容差 | 静态K | 理想可行平均K | 理想平均KL | 分数解平均K上界 |', '|---|---:|---:|---:|---:|---:|']
    for r in bounds:
        lines.append(f"| {r['cohort']} | {r['kl_allowance']:.2f} | 12 | {r['future_informed_feasible_mean_k']:.3f} | {r['future_informed_feasible_mean_kl']:.5f} | {r['future_informed_fractional_mean_k_upper_bound']:.3f} |")
    lines += ['', '可优先改进三处：以实际耗时而非执行层比例建模代价，并计入全文路由与按需搬运；训练预测相对静态的每个K边际损害并使训练目标更贴合预算决策；只在选模32条上加密系数搜索或拟合质量约束策略。若优化目标改为任务准确率，必须补充训练来源的任务监督，不能用待测96题答案训练或按题回选K。', '',
        '预算决策和路径组合是两个维度；当前只用单条嵌套静态路径，所以收益空间结论也仅针对这条路径。本诊断不启动路线5重训，不改变已完成实验。', '', f'[聚合数据]({destination.name})']
    with report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(report), 'bounds': bounds}, ensure_ascii=False))


if __name__ == '__main__':
    main()
