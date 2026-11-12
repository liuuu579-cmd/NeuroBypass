"""Stage21 v2: calibrated routing; explicit FP32 historical-label NLL tolerance."""
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load, load_choice, score_joint, teachers_for
from codebook_calibration_core_v1 import ALPHAS, MARGINS, apply_policy, cpu_checks, fit_policy
from codebook_refinement_core_v2 import SEEDS, SixWorkers, setup
from full_context_core_v1 import prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import preserve, read_lines, thread_record, verify_preserved, write
from headroom_ruler_core_v1 import paired
from stage1_benchmark_v2 import sha256, stamp


def worker(rank, cfg, inbox, messages):
    setup(rank)
    out, old = Path(cfg['output']), Path(cfg['previous_run'])
    device = f'cuda:{rank}'
    embeddings = torch.load(old / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    data = Records(cfg['data_run'])
    for sample in cfg['ids']:
        assert embeddings[sample].shape[1] == data.rows[sample]['prompt_length']
    predictions = {}
    for item in cfg['checkpoints'][rank::6]:
        assert sha256(item['path']) == item['sha256']
        router = load_choice(item['path'], device)
        rows = []
        with torch.inference_mode():
            for offset, sample in enumerate(cfg['ids']):
                rows.append(router(embeddings[sample].to(device)).cpu())
                if offset % 150 == 0:
                    messages.put({'kind': 'progress', 'rank': rank, 'phase': 'scores',
                        'model': item['name'], 'completed': offset + 1, 'total': len(cfg['ids'])})
        predictions[item['name']] = torch.cat(rows)
        assert torch.isfinite(predictions[item['name']]).all()
        del router
    path = out / f'private_predictions_{rank:03d}.pth'
    with path.open('xb') as stream:
        torch.save(predictions, stream)
    write(out / f'prediction_worker_{rank:03d}.json', {'passed': True, 'sha256': sha256(path),
        'models': list(predictions), 'full_prompt_lengths_verified': True, **thread_record()})
    del embeddings, predictions
    gc.collect(); torch.cuda.empty_cache()
    messages.put({'kind': 'scores_done', 'rank': rank})
    assert inbox.get()['kind'] == 'replay'
    # Independently run real OPT for the selected policy on the complete validation set.
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    policy = load(out / 'selected_policy.json')
    members = policy['members']
    routers = [load_choice(next(c['path'] for c in cfg['checkpoints'] if c['name'] == name), device) for name in members]
    bank = load(old / 'candidate_bank.json')
    labels = {r['sample']: r for i in range(6) for r in read_lines(old / f'private_labels_{i:03d}.jsonl')}
    table_routes = load(out / 'private_selected_routes.json')
    error = 0.; count = 0
    with torch.inference_mode(), (out / f'private_replay_{rank:03d}.jsonl').open('x') as stream:
        for offset, record in enumerate(read_lines(old / 'private_inputs.jsonl')[rank::6]):
            sample = record['source_sample']
            embedding = prompt_embeddings(model, torch.tensor(record['route_ids'], device=device)[None])
            assert embedding.shape[1] == len(record['route_ids'])
            predicted = torch.stack([r(embedding).cpu() for r in routers]).mean(0).numpy()
            route = int(apply_policy(predicted, policy['fit'])[0])
            assert route == table_routes[str(sample)]
            teachers = teachers_for(model, record)
            value = score_joint(model, record, bank['routes'][route]['skips'], teachers)
            differences = [abs(value[metric] - labels[sample][metric][route]) for metric in ('kl', 'nll')]
            assert differences[0] < 1e-6 and differences[1] < 1e-5
            error = max(error, *differences)
            stream.write(json.dumps({'sample': sample, 'route': route, **value}, allow_nan=False) + '\n')
            count += 1
            if offset % 4 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'actual_selected_policy_replay', 'completed': count})
    write(out / f'replay_worker_{rank:03d}.json', {'passed': True, 'documents': count,
        'max_abs_kl_nll_error': error, 'actual_prompt_route_equals_cached_route': True,
        'base_model_frozen': not any(p.requires_grad for p in model.parameters()), **thread_record()})
    messages.put({'kind': 'done', 'rank': rank})
    assert inbox.get() is None


def summarize(out, cfg):
    old = Path(cfg['previous_run'])
    rows = {r['sample']: r for rank in range(6) for r in read_lines(old / f'private_labels_{rank:03d}.jsonl')}
    ids = cfg['ids']; lookup = {sample: j for j, sample in enumerate(ids)}
    train = np.array([lookup[i] for i in cfg['train_ids']]); selection = np.array([lookup[i] for i in cfg['selection_ids']])
    costs = np.array([rows[i]['kl'] for i in ids], dtype=np.float64)
    nll = np.array([rows[i]['nll'] for i in ids], dtype=np.float64)
    baseline = int(costs[train].mean(0).argmin())
    assert baseline == load(old / 'static_controls.json')['best_route'] == 1
    scale = float((costs[train] - costs[train, baseline, None]).std())
    predictions = {}
    for rank in range(6):
        path = out / f'private_predictions_{rank:03d}.pth'
        assert sha256(path) == load(out / f'prediction_worker_{rank:03d}.json')['sha256']
        predictions.update({k: v.numpy().astype(np.float64) for k, v in torch.load(path, weights_only=True).items()})
    # Reproduce every saved raw validation decision before applying any calibration.
    replay_count = 0
    for rank in range(6):
        for row in read_lines(old / f'private_quality_{rank:03d}.jsonl'):
            if row['seed'] is None:
                continue
            name = f"{row['policy']}_s{row['seed']}"
            route = int(predictions[name][lookup[row['source_sample']]].argmin())
            assert route == row['route_index'], (name, row['source_sample'], route, row['route_index'])
            for metric, values in (('kl', costs), ('nll', nll)):
                assert abs(row[metric] - values[lookup[row['source_sample']], route]) < (1e-6 if metric == 'kl' else 1e-5)
            replay_count += 1
    assert replay_count == 1500
    members = {name: [name] for name in predictions}
    for family in ('old256_prompt', 'huber_prompt', 'rank_prompt'):
        name = f'ensemble_{family}'
        group = [f'{family}_s{s}' for s in SEEDS]
        predictions[name] = np.mean([predictions[k] for k in group], axis=0); members[name] = group
    group = [f'{f}_s{s}' for f in ('old256_prompt', 'huber_prompt', 'rank_prompt') for s in SEEDS]
    predictions['ensemble_all9_prompt'] = np.mean([predictions[k] for k in group], axis=0)
    members['ensemble_all9_prompt'] = group
    cohorts = {'train800': train, 'selection32': selection,
        'other68': np.array([lookup[i] for i in cfg['validation_ids'] if i not in set(cfg['selection_ids'])]),
        'all_validation100': np.array([lookup[i] for i in cfg['validation_ids']])}
    fits = {}; summaries = []; routes_by_model = {}; diagnostics = []
    for name, scores in predictions.items():
        fitted = fit_policy(scores, costs, train, selection, baseline, scale)
        fits[name] = fitted
        calibrated_routes = apply_policy(scores, fitted)
        routes_by_model[name] = calibrated_routes
        # Raw argmin is the previous experiment's behavior, including default argmin ties.
        raw_routes = scores.argmin(1)
        for cohort, ix in cohorts.items():
            for variant, routes in (('raw', raw_routes), ('calibrated_gated', calibrated_routes)):
                own = costs[ix, routes[ix]]; static = costs[ix, baseline]
                summaries.append({'model': name, 'variant': variant, 'cohort': cohort,
                    'kl': float(own.mean()), 'nll': float(nll[ix, routes[ix]].mean()),
                    'static_kl': float(static.mean()), 'static_nll': float(nll[ix, baseline].mean()),
                    'switches': int((routes[ix] != baseline).sum()), 'unique_routes': len(set(routes[ix].tolist())),
                    **paired(static - own, None, 210579)})
        relative = costs - costs[:, baseline, None]
        predicted = scores - scores[:, baseline, None]
        diagnostics.append({'model': name, 'train_relative_mean_prediction_error': (predicted[train].mean(0) - relative[train].mean(0)).tolist(),
            'train_relative_mse': float(np.mean((predicted[train] - relative[train]) ** 2))})
    singles = [name for name in predictions if 'prompt' in name and not name.startswith('ensemble_')]
    # One deployment candidate, picked using ONLY original32; never pick using other68.
    chosen = min(singles, key=lambda name: (fits[name]['chosen']['selection_kl'], fits[name]['chosen']['selection_switches'], name))
    write(out / 'policy_fits.json', fits)
    write(out / 'selected_policy.json', {'model': chosen, 'members': members[chosen], 'fit': fits[chosen],
        'selection_rule': 'Lowest original32 actual KL, then fewer switches, then model name; single saved prompt router only.'})
    write(out / 'private_selected_routes.json', {str(i): int(routes_by_model[chosen][lookup[i]]) for i in cfg['validation_ids']})
    aggregates = []
    for family in ('old256_prompt', 'huber_prompt', 'rank_prompt', 'huber_constant', 'rank_constant'):
        for cohort, ix in cohorts.items():
            routes = [routes_by_model[f'{family}_s{s}'] for s in SEEDS]
            own = np.mean([costs[ix, r[ix]] for r in routes], axis=0)
            aggregates.append({'family': family, 'cohort': cohort, 'kl': float(own.mean()),
                'static_kl': float(costs[ix, baseline].mean()), **paired(costs[ix, baseline] - own, None, 210579)})
    bounds = [{'cohort': name, 'static_kl': float(costs[ix, baseline].mean()),
        'future_known_best_kl': float(costs[ix].min(1).mean()),
        'static_to_future_known_gap': float((costs[ix, baseline] - costs[ix].min(1)).mean())} for name, ix in cohorts.items()]
    summary = {'selected_model': chosen, 'raw_replay_rows': replay_count, 'baseline_index': baseline,
        'scale_from_train800': scale, 'per_model': summaries, 'three_seed_mean': aggregates,
        'headroom_diagnostic': bounds, 'prediction_diagnostics': diagnostics,
        'interpretation': 'All100 previously reviewed development. Grids and model chosen only on original32; other68 are a separate reviewed cohort. Bootstrap exploratory; no multiplicity or complete seed uncertainty. Ensembles are diagnostic with extra routing compute. No new throughput claim.'}
    write(out / 'summary.json', summary)
    return summary


def make_report(root, out, cfg, summary):
    report = root / 'analysis' / f'{out.name}_results.zh-CN.md'
    lines = ['# 阶段21：组合预测均值校准与保守切换', '',
        '路线1继续；路线5不重训。K20/40、原8个完整组合、完整提示词embedding、回答内固定路径。复用旧256训练与新800训练的9个提示词检查点，以及6个800训练常量对照；本轮不更新模型权重。', '',
        '静态基准为800条训练均值选出的index1。校准公式：训练真实组合KL均值 + alpha × (预测组合分数 − 训练预测均值)。alpha为0/0.25/0.5/0.75/1；切换门槛为训练相对KL标准差的0/0.05/0.1/0.2/0.4/0.8倍；另列原始分数加门槛及始终静态。预测收益严格超过门槛才离开静态，同分保留静态。监督使用完整组合真实联合KL，未加总逐层损失。', '',
        '均值与尺度只看原800训练；每个模型的校准参数以及最终单模型只用原32条选择集确定，另外68条不参与。本轮继续使用已复核过的100条开发文档，不称独立最终测试。9模型及多个配置的选择具有开发选择偏差。', '',
        '| 群体 | 校准后模型组（三种子平均） | 动态KL | 强静态KL | 静态减动态 | 配对95%区间 |',
        '|---|---|---:|---:|---:|---|']
    for r in summary['three_seed_mean']:
        if r['cohort'] not in ('other68', 'all_validation100'):
            continue
        lo, hi = r['ci95']
        lines.append(f"| {r['cohort']} | {r['family']} | {r['kl']:.5f} | {r['static_kl']:.5f} | {r['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    chosen = summary['selected_model']
    lines += ['', f'原32条选出的单模型：`{chosen}`。此后用真实OPT在全部100条上重新生成完整提示词embedding、计算路由并重放所选完整组合KL/NLL；校验缓存路由一致、KL误差小于1e-6、NLL误差小于1e-5（历史标签含FP32舍入）。', '',
        '| 单模型群体 | KL | 强静态KL | 离开静态次数 | 配对95%区间（静态减动态） |', '|---|---:|---:|---:|---|']
    for r in summary['per_model']:
        if r['model'] == chosen and r['variant'] == 'calibrated_gated' and r['cohort'] != 'train800':
            lo, hi = r['ci95']; lines.append(f"| {r['cohort']} | {r['kl']:.5f} | {r['static_kl']:.5f} | {r['switches']} | [{lo:+.5f}, {hi:+.5f}] |")
    other = next(r for r in summary['per_model'] if r['model'] == chosen and r['variant'] == 'calibrated_gated' and r['cohort'] == 'other68')
    if other['ci95'][0] > 0:
        conclusion = '32条选出的单模型在其他68条出现正向改善，文档配对区间为正；仍需种子稳定性和后续任务复核。'
    elif other['difference'] > 0:
        conclusion = '32条选出的单模型在其他68条点估计改善，但配对区间包含0，暂不能确认稳定超越强静态。'
    else:
        conclusion = '32条选出的单模型在其他68条仍未超过强静态，本轮校准和门槛不足以建立动态优势。'
    lines += ['', conclusion, '',
        '候选库逐文档未来最优只用于诊断：它看到了真实续写，不能作为线上路由性能。三种子及九模型平均分数集成为额外计算诊断，不作吞吐加速主张。完整单模型、原始与校准、选择网格、预测偏差、NLL均在汇总与拟合记录中。', '',
        '使用6GPU/64CPU，worker各10核、控制器4核；所有旧源码和研究记录验哈希保全，检查点输入另验哈希，公开聚合结果和自写源码，不上传权重、token、embedding或逐文档路由。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [校准配置](../{out.relative_to(root)}/policy_fits.json) · [校验](../{out.relative_to(root)}/verification.json)']
    with report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    return report


def main():
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    setup(); tracked = preserve(root, out)
    previous = root / 'runs/stage20_codebook_refine_20261112_02_main'
    old = load(previous / 'protocol.json')
    checkpoints = []
    for family in ('huber', 'rank'):
        for arm in ('prompt', 'constant'):
            for seed in SEEDS:
                receipt = load(previous / f'{family}_{arm}_s{seed}_fit.json')
                checkpoints.append({'name': f'{family}_{arm}_s{seed}', **receipt['best']})
    for seed, receipt in old['old_checkpoints'].items():
        checkpoints.append({'name': f'old256_prompt_s{seed}', **receipt})
    assert sha256(previous / 'private_embeddings.pth') == old['embedding_sha256']
    evidence = {str(previous / name): sha256(previous / name) for name in
        ['protocol.json', 'private_embeddings.pth', 'candidate_bank.json', 'static_controls.json', 'private_inputs.jsonl'] +
        [f'private_{kind}_{rank:03d}.jsonl' for kind in ('labels', 'quality') for rank in range(6)]}
    sources = ['codebook_calibration_core_v1.py', 'run_codebook_calibration_six_gpu_v2.py']
    cfg = {'previous_run': str(previous), 'output': str(out), 'model': old['model'], 'data_run': old['data_run'],
        'train_ids': old['train_ids'], 'validation_ids': old['validation_ids'], 'selection_ids': old['selection_ids'],
        'ids': old['train_ids'] + old['validation_ids'], 'checkpoints': checkpoints,
        'alpha_grid': ALPHAS, 'margin_grid_units': MARGINS,
        'resources': '6GPU/64CPU; each worker10/controller4',
        'source_sha256': {name: sha256(root / 'tools' / name) for name in sources}, 'evidence_sha256': evidence,
        'selection_policy': 'Train800 only for mean correction and KL scale; original32 only for gate/alpha and single-model selection; other68 never used to select. No weights updated. Complete-mask joint KL only.',
        'limits': 'Repeatedly reviewed development; ensemble diagnostics cost extra routing compute; no throughput/generalization claim.'}
    assert len(set(cfg['ids'])) == 900 and len(cfg['selection_ids']) == 32
    assert set(cfg['selection_ids']) <= set(cfg['validation_ids'])
    write(out / 'protocol.json', cfg); write(out / 'cpu_checks.json', cpu_checks())
    stamp('stage21_start', checkpoints=len(checkpoints), documents=900)
    with SixWorkers(worker, cfg) as workers:
        workers.receive('scores_done')
        summary = summarize(out, cfg)
        stamp('stage21_policy_selected', model=summary['selected_model'])
        workers.send({'kind': 'replay'}); workers.receive('done')
    assert sum(load(out / f'replay_worker_{rank:03d}.json')['documents'] for rank in range(6)) == 100
    for path, digest in evidence.items():
        assert sha256(path) == digest
    for cp in checkpoints:
        assert sha256(cp['path']) == cp['sha256']
    verify_preserved(out, tracked)
    write(out / 'verification.json', {'passed': True, 'raw_validation_rows_reproduced': summary['raw_replay_rows'],
        'real_model_selected_policy_validation_documents': 100, 'full_prompt_and_complete_mask': True,
        'train_validation_disjoint': True, 'original32_only_selection': True, 'old_checkpoints_unchanged': True,
        'raw_replay_kl_tolerance': 1e-6, 'raw_replay_nll_tolerance': 1e-5,
        'historical_nll_precision': 'Old complete-mask tables contain FP32-rounded reductions; stage20 actual quality uses double reduction. Stage21 v1 detected max existing discrepancy1.0841e-6; NLL tolerance1e-5, KL remains1e-6.'})
    report = make_report(root, out, cfg, summary)
    from run_static_swap_pipeline_v1 import publish
    files = [report] + [root / 'tools' / name for name in sources]
    files += [p for p in out.glob('*.json') if not p.name.startswith('private_') and p.name not in
        ('preservation_before.json', 'allocation.json')]
    receipt = publish(root, files, out.name, 'Test training-only score calibration and conservative K20 routing against strong static', dict(os.environ))
    write(out / 'publication_verified.json', receipt)
    stamp('stage21_complete', report=str(report))


if __name__ == '__main__':
    main()
