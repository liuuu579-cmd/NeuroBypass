"""Stage22: baseline-relative calibration and heuristic disagreement abstention.

Pure CPU replay of saved Stage21 scores. No new checkpoint and no online quality
or speed claim. Model disagreement and in-sample residuals are uncertainty
proxies, not calibrated confidence bounds.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from full_context_experiment_v1 import preserve, read_lines, verify_preserved, write
from headroom_ruler_core_v1 import paired
from stage1_benchmark_v2 import sha256


SHRINKS = (0., .25, .5, .75, 1.)
PENALTIES = (0., .25, .5, 1., 1.5, 2.)
MARGINS = (0., .05, .1, .2, .4, .8)


def load(path):
    return json.loads(Path(path).read_text())


def fit_relative(member_scores, train_costs, baseline):
    relative = member_scores - member_scores[:, :, baseline:baseline + 1]
    mean = relative.mean(0)
    target = train_costs - train_costs[:, baseline:baseline + 1]
    x_mean, y_mean = mean.mean(0), target.mean(0)
    x, y = mean - x_mean, target - y_mean
    slope = np.clip((x * y).sum(0) / np.maximum((x * x).sum(0), 1e-12), 0., 2.)
    scale = float(target.std())
    assert scale > 0 and slope[baseline] == 0
    return {'baseline': baseline, 'train_prediction_mean': x_mean.tolist(),
        'train_cost_mean': y_mean.tolist(), 'nonnegative_clipped_ols': slope.tolist(),
        'scale': scale}


def score_relative(member_scores, fit, train_member_scores, train_costs, shrink, penalty):
    baseline = fit['baseline']
    center = np.asarray(fit['train_prediction_mean']); prior = np.asarray(fit['train_cost_mean'])
    slope = shrink * np.asarray(fit['nonnegative_clipped_ols'])
    train_rel = train_member_scores - train_member_scores[:, :, baseline:baseline + 1]
    target_rel = train_costs - train_costs[:, baseline:baseline + 1]
    residual = target_rel - (prior + slope * (train_rel.mean(0) - center))
    sigma = residual.std(0, ddof=1)
    relative = member_scores - member_scores[:, :, baseline:baseline + 1]
    predicted = prior + slope * (relative.mean(0) - center)
    disagreement = relative.std(0, ddof=1) * np.abs(slope)
    proxy = np.sqrt(sigma ** 2 + disagreement ** 2)
    score = predicted + penalty * proxy
    assert np.all(score[:, baseline] == 0)
    return score


def decide(score, baseline, margin):
    assert margin >= 0 and np.isfinite(score).all()
    best = score.argmin(1)
    gain = score[:, baseline] - score[np.arange(len(score)), best]
    return np.where(gain > margin, best, baseline).astype(np.int64)


def select_policy(train_scores, train_costs, selection_scores, selection_costs, baseline):
    """Accept no other68 data: train statistics plus original32 choice only."""
    fit = fit_relative(train_scores, train_costs, baseline)
    grid = [{'shrink': 0., 'penalty': 0., 'margin_units': 0., 'static': True}]
    grid += [{'shrink': s, 'penalty': p, 'margin_units': m, 'static': False}
        for s in SHRINKS for p in PENALTIES for m in MARGINS]
    for serial, policy in enumerate(grid):
        if policy['static']:
            routes = np.full(len(selection_costs), baseline)
        else:
            score = score_relative(selection_scores, fit, train_scores, train_costs,
                policy['shrink'], policy['penalty'])
            routes = decide(score, baseline, policy['margin_units'] * fit['scale'])
        policy.update({'serial': serial, 'selection_kl': float(selection_costs[np.arange(len(routes)), routes].mean()),
            'selection_switches': int((routes != baseline).sum())})
    chosen = min(grid, key=lambda p: (p['selection_kl'], p['selection_switches'], p['serial']))
    return {'fit': fit, 'grid': grid, 'chosen': chosen}


def decisions(member_scores, train_scores, train_costs, fitted):
    policy, fit = fitted['chosen'], fitted['fit']
    if policy['static']:
        return np.full(member_scores.shape[1], fit['baseline'], dtype=np.int64)
    score = score_relative(member_scores, fit, train_scores, train_costs, policy['shrink'], policy['penalty'])
    return decide(score, fit['baseline'], policy['margin_units'] * fit['scale'])


def cpu_checks():
    # A common per-document cost or predicted-score offset must not alter a mask.
    rng = np.random.default_rng(220579)
    scores = rng.normal(size=(3, 12, 4))
    costs = rng.uniform(.5, 1.5, size=(12, 4)); costs[:, 1] -= .4
    fitted = select_policy(scores[:, :8], costs[:8], scores[:, 8:], costs[8:], 1)
    shifted_scores = scores + rng.normal(size=(3, 12, 1))
    shifted_costs = costs + rng.normal(size=(12, 1))
    shifted = select_policy(shifted_scores[:, :8], shifted_costs[:8], shifted_scores[:, 8:], shifted_costs[8:], 1)
    assert np.array_equal(decisions(scores, scores[:, :8], costs[:8], fitted),
        decisions(shifted_scores, shifted_scores[:, :8], shifted_costs[:8], shifted))
    assert decide(np.array([[0., 0., 1.], [-.125, 0., 1.], [-.25, 0., 1.]]), 1, .125).tolist() == [1, 1, 0]
    return {'passed': True, 'common_cost_and_score_offsets_do_not_change_routes': True,
        'ties_and_exact_margin_keep_static': True, 'selection_function_has_no_other68_argument': True}


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert len(os.sched_getaffinity(0)) <= 2, 'CPU-only diagnostic: run taskset with at most2 cores'
    root = Path(__file__).resolve().parents[1]
    previous = root / 'runs/stage21_codebook_calibrate_20261112_03'
    out = root / 'runs/stage22_route1_uncertainty_diagnostic_20261112_02'
    out.mkdir(exist_ok=False)
    preserved = preserve(root, out)
    cfg = load(previous / 'protocol.json'); old = Path(cfg['previous_run'])
    ids = cfg['ids']; lookup = {sample: i for i, sample in enumerate(ids)}
    train = np.array([lookup[i] for i in cfg['train_ids']]); selection = np.array([lookup[i] for i in cfg['selection_ids']])
    validation = np.array([lookup[i] for i in cfg['validation_ids']])
    other = np.array([lookup[i] for i in cfg['validation_ids'] if i not in set(cfg['selection_ids'])])
    assert len(train) == 800 and len(selection) == 32 and len(other) == 68
    assert not set(train) & set(validation) and len(set(ids)) == 900
    evidence_paths = [previous / 'protocol.json', old / 'static_controls.json']
    evidence_paths += [old / f'private_labels_{rank:03d}.jsonl' for rank in range(6)]
    evidence_paths += [previous / f'private_predictions_{rank:03d}.pth' for rank in range(6)]
    evidence = {str(p): sha256(p) for p in evidence_paths}
    for path in evidence_paths:
        if str(path) in cfg['evidence_sha256']:
            assert evidence[str(path)] == cfg['evidence_sha256'][str(path)]
    labels = {r['sample']: r for rank in range(6) for r in read_lines(old / f'private_labels_{rank:03d}.jsonl')}
    costs = np.array([labels[i]['kl'] for i in ids]); nll = np.array([labels[i]['nll'] for i in ids])
    baseline = int(costs[train].mean(0).argmin()); assert baseline == load(old / 'static_controls.json')['best_route'] == 1
    scores = {}
    for rank in range(6):
        path = previous / f'private_predictions_{rank:03d}.pth'
        assert sha256(path) == load(previous / f'prediction_worker_{rank:03d}.json')['sha256']
        scores.update({k: v.numpy().astype(np.float64) for k, v in torch.load(path, weights_only=True).items()})
    members = [f'{family}_prompt_s{seed}' for family in ('old256', 'huber', 'rank') for seed in (579, 580, 581)]
    full_scores = np.stack([scores[name] for name in members])
    fitted = select_policy(full_scores[:, train], costs[train], full_scores[:, selection], costs[selection], baseline)
    chosen_routes = decisions(full_scores, full_scores[:, train], costs[train], fitted)
    # Control: making all prompts identical produces a fixed mask, not dynamic gain.
    constant_scores = np.broadcast_to(full_scores[:, train].mean(1, keepdims=True), full_scores.shape)
    constant_fit = select_policy(constant_scores[:, train], costs[train], constant_scores[:, selection], costs[selection], baseline)
    constant_routes = decisions(constant_scores, constant_scores[:, train], costs[train], constant_fit)
    assert np.all(constant_routes == baseline)
    # Compare to the frozen Stage21 raw ensemble; this must replay the existing table.
    raw_routes = full_scores.mean(0).argmin(1)
    cohorts = {'train800': train, 'selection32': selection, 'other68': other, 'all_validation100': validation}
    evaluated = []
    for name, routes in [('relative_uncertainty_gate', chosen_routes), ('raw_ensemble', raw_routes),
                         ('constant_observation', constant_routes), ('strong_static', np.full(900, baseline))]:
        for cohort, ix in cohorts.items():
            own = costs[ix, routes[ix]]; static = costs[ix, baseline]
            switched = routes[ix] != baseline; delta = static - own
            evaluated.append({'policy': name, 'cohort': cohort, 'kl': float(own.mean()),
                'nll': float(nll[ix, routes[ix]].mean()), 'static_kl': float(static.mean()),
                'static_nll': float(nll[ix, baseline].mean()), 'switches': int(switched.sum()),
                'unique_routes': len(set(routes[ix].tolist())),
                'beneficial_switches': int((delta > 0).sum()), 'harmful_switches': int((delta < 0).sum()),
                **paired(delta, None, 220579)})
    prior = load(previous / 'summary.json')
    for row in evaluated:
        if row['policy'] == 'raw_ensemble':
            old_row = next(r for r in prior['per_model'] if r['model'] == 'ensemble_all9_prompt'
                and r['variant'] == 'raw' and r['cohort'] == row['cohort'])
            assert abs(row['kl'] - old_row['kl']) < 1e-12
    write(out / 'protocol.json', {'source': str(__file__), 'source_sha256': sha256(__file__),
        'previous_run': str(previous), 'evidence_sha256': evidence, 'member_names': members,
        'train800_ids': cfg['train_ids'], 'selection32_ids': cfg['selection_ids'], 'validation100_ids': cfg['validation_ids'],
        'baseline_index': baseline, 'resources': 'CPU-only; affinity2/torch1; no GPU allocation or inference',
        'calibration': 'All predicted and target costs relative to training-best static1. Per-route OLS clipped0..2 times shrink, plus training relative-cost mean. Uncertainty proxy=sqrt(train in-sample residual variance + scaled relative model-disagreement variance).',
        'selection': '181 fixed settings; actual original32 complete-mask KL, then fewer switches, then fixed grid order. Other68 not an argument to selection.',
        'limits': 'Reviewed development. In-sample calibration residuals are optimistic. Ensemble disagreement is not a calibrated confidence bound. Nine routers cost extra inference; no deployment speed claim. Different data-size and loss members are correlated; no epistemic coverage interpretation.'})
    write(out / 'policy_fits.json', fitted)
    write(out / 'summary.json', {'evaluations': evaluated, 'chosen': fitted['chosen'], 'grid_count': len(fitted['grid'])})
    write(out / 'private_selected_routes.json', {str(sample): int(chosen_routes[lookup[sample]]) for sample in cfg['validation_ids']})
    write(out / 'cpu_checks.json', cpu_checks())
    for path, expected in evidence.items():
        assert sha256(path) == expected
    verify_preserved(out, preserved)
    write(out / 'verification.json', {'passed': True, 'source_and_evidence_verified': True,
        'raw_stage21_ensemble_reproduced': True, 'constant_control_is_strong_static': True,
        'train800_and_validation100_disjoint': True, 'relative_to_static_baseline': True,
        'selection32_only': True, 'old_files_preserved': len(preserved),
        'cpu_ids': sorted(os.sched_getaffinity(0)), 'torch_threads': torch.get_num_threads(), 'gpu_used': False})
    report = root / 'analysis' / f'{out.name}_results.zh-CN.md'
    lines = ['# 阶段22：相对强静态收益的门控诊断', '',
        '仅用既有预测和完整组合联合KL离线计算，不更新权重。K20/40、原8组合、完整提示词路由不变；9个旧提示词路由器预测均值与分歧作为诊断。所有代价先减去同文档的强静态index1代价，以免将文档难度误当作换层收益。', '',
        '逐组合校准斜率及残差只用800训练；收缩系数、残差/分歧惩罚和切换门槛的181个固定配置，仅原32条选择。68条只在选完之后复核。训练残差是样本内残差，分歧是启发式指标，均不提供置信保证；全部100条仍是已查看的开发文档。', '',
        '| 群体 | 策略 | KL | 强静态KL | 静态减动态 | 配对95%区间 | 有益/有害切换 |',
        '|---|---|---:|---:|---:|---|---|']
    for r in evaluated:
        if r['policy'] in ('relative_uncertainty_gate', 'raw_ensemble') and r['cohort'] != 'train800':
            lo, hi = r['ci95']; lines.append(f"| {r['cohort']} | {r['policy']} | {r['kl']:.5f} | {r['static_kl']:.5f} | {r['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {r['beneficial_switches']}/{r['harmful_switches']} |")
    result = next(r for r in evaluated if r['policy'] == 'relative_uncertainty_gate' and r['cohort'] == 'other68')
    conclusion = ('复核68条仍未超过强静态；停止继续堆叠分数校准和门槛，下一步检查监督代价是否能由提示词预测。'
        if result['difference'] <= 0 else '复核68条出现点估计改善，仍需结合配对区间和开发选择偏差判断，不能直接宣称稳定优势。')
    lines += ['', f"原32条选定配置：`{fitted['chosen']}`。", '', conclusion, '',
        '常量观察对照始终回到强静态。旧Stage21原始集成KL完全复现。CPU限制2核、torch1线程、无GPU计算。旧文件验哈希保全；不上传逐文档路由、原始标签、embedding和权重。9模型集成有额外推理开销，本轮无速度或显存声明。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [配置](../{out.relative_to(root)}/policy_fits.json) · [校验](../{out.relative_to(root)}/verification.json)']
    with report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'run': str(out), 'report': str(report), 'other68': result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
