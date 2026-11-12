"""Stage20: fixed K20/bank8, train800 versus saved train256; two losses, three seeds."""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import load, load_choice, score_joint, teachers_for
from codebook_refinement_core_v2 import SEEDS, SixWorkers, cpu_checks, fit, setup
from full_context_core_v1 import prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import preserve, read_lines, thread_record, verify_preserved, write
from headroom_ruler_core_v1 import paired
from prompt_fixed_core_v1 import continuation_logits
from stage1_benchmark_v2 import sha256, stamp


def worker(rank, cfg, inbox, messages):
    setup(rank)
    out = Path(cfg['output'])
    device = f'cuda:{rank}'
    objective = ('huber', 'rank')[rank // 3]
    seed = SEEDS[rank % 3]
    bank = load(out / 'candidate_bank.json')
    old = load(out / 'private_reused_labels.json')
    data = Records(cfg['data_run'])
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    samples = sorted(cfg['train_ids'] + cfg['validation_ids'])[rank::6]
    maximum_replay = 0.
    with torch.inference_mode(), (out / f'private_labels_{rank:03d}.jsonl').open('x') as stream:
        for offset, sample in enumerate(samples):
            if str(sample) in old:
                row = old[str(sample)]
            else:
                raw, p = data[sample]
                ids = torch.tensor(raw, device=device, dtype=torch.long)[None]
                teacher = F.log_softmax(continuation_logits(model, ids, p, []), -1)
                kl, nll = [], []
                for route in bank['routes']:
                    lp = F.log_softmax(continuation_logits(model, ids, p, route['skips']), -1)
                    kl.append(float((teacher.exp() * (teacher - lp)).sum(-1).double().mean()))
                    nll.append(float(-lp.gather(-1, ids[:, p:, None]).double().mean()))
                row = {'sample': sample, 'kl': kl, 'nll': nll, 'reused': False,
                    'sequence_sha256': data.rows[sample]['sequence_sha256']}
            assert row['sequence_sha256'] == data.rows[sample]['sequence_sha256']
            assert len(row['kl']) == 8 and np.isfinite(row['kl']).all()
            if offset == 0:
                raw, p = data[sample]
                r = {'sequence_ids': [raw.tolist()], 'target_lengths': [32]}
                teacher_list = teachers_for(model, r)
                for i in (0, 7):
                    direct = score_joint(model, r, bank['routes'][i]['skips'], teacher_list)
                    error = abs(direct['kl'] - row['kl'][i])
                    assert error < 1e-6, (sample, i, error)
                    maximum_replay = max(maximum_replay, error)
                embedding = prompt_embeddings(model, torch.tensor(raw[:p], device=device, dtype=torch.long)[None]).cpu()
                cached = torch.load(out / 'private_embeddings.pth', weights_only=True, map_location='cpu')[sample]
                assert torch.equal(embedding, cached) and embedding.shape[1] == p
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            if offset % 8 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'labels', 'completed': offset + 1, 'total': len(samples)})
    write(out / f'labels_{rank:03d}.json', {'passed': True, 'documents': len(samples),
        'labels_sha256': sha256(out / f'private_labels_{rank:03d}.jsonl'),
        'joint_label_replay_max_abs_error': maximum_replay, 'embedding_replay_exact': True, **thread_record()})
    messages.put({'kind': 'labels_done', 'rank': rank})
    assert inbox.get()['kind'] == 'fit'
    labels = {r['sample']: r for i in range(6) for r in read_lines(out / f'private_labels_{i:03d}.jsonl')}
    costs = {i: r['kl'] for i, r in labels.items()}
    embeddings = torch.load(out / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    for observe in (True, False):
        fit(out, costs, embeddings, cfg, seed, observe, objective, device, messages, rank)
    del embeddings
    gc.collect()
    messages.put({'kind': 'fit_done', 'rank': rank})
    assert inbox.get()['kind'] == 'quality'
    controls = load(out / 'static_controls.json')
    routers = {}
    for obj in ('huber', 'rank'):
        for s in SEEDS:
            for arm in ('prompt', 'constant'):
                record = load(out / f'{obj}_{arm}_s{s}_fit.json')
                assert sha256(record['best']['path']) == record['best']['sha256']
                routers[f'{obj}_{arm}', s] = load_choice(record['best']['path'], device)
    for s, cp in cfg['old_checkpoints'].items():
        assert sha256(cp['path']) == cp['sha256']
        routers['old256_prompt', int(s)] = load_choice(cp['path'], device)
    inputs = read_lines(out / 'private_inputs.jsonl')
    replay = []
    with torch.inference_mode(), (out / f'private_quality_{rank:03d}.jsonl').open('x') as stream:
        for offset, r in enumerate(inputs[rank::6]):
            embedding = prompt_embeddings(model, torch.tensor(r['route_ids'], device=device)[None])
            assert embedding.shape[1] == len(r['route_ids'])
            proposals = [('static_old256', None, 0), ('static_train800', None, controls['best_route'])]
            proposals += [(name, s, int(router(embedding).argmin(-1))) for (name, s), router in routers.items()]
            teachers = teachers_for(model, r)
            cache = {}
            for policy, s, index in proposals:
                route = bank['routes'][index]
                if index not in cache:
                    cache[index] = score_joint(model, r, route['skips'], teachers)
                    error = abs(cache[index]['kl'] - costs[r['source_sample']][index])
                    assert error < 1e-6, (policy, error)
                    replay.append(error)
                stream.write(json.dumps({'sample': r['sample'], 'source_sample': r['source_sample'],
                    'task': r['task'], 'policy': policy, 'seed': s, 'route_index': index,
                    'k': 20, 'skips': route['skips'], **cache[index]}, allow_nan=False) + '\n')
            stream.flush()
            if offset % 8 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'quality', 'completed': offset + 1, 'total': len(inputs[rank::6])})
    write(out / f'worker_{rank:03d}.json', {'passed': True,
        'quality_sha256': sha256(out / f'private_quality_{rank:03d}.jsonl'),
        'quality_label_replay_max_abs_error': max(replay), 'base_model_frozen': not any(p.requires_grad for p in model.parameters()), **thread_record()})
    messages.put({'kind': 'done', 'rank': rank})
    assert inbox.get() is None


def finalize(out, cfg):
    inputs = read_lines(out / 'private_inputs.jsonl')
    rows = [r for i in range(6) for r in read_lines(out / f'private_quality_{i:03d}.jsonl')]
    assert len(rows) == len(inputs) * 17
    index = {(r['sample'], r['policy'], r['seed']): r for r in rows}
    assert len(index) == len(rows)
    selection = set(cfg['selection_ids'])
    comparisons, per_seed = [], []
    for cohort in ('all_validation', 'selection32', 'other68'):
        ids = [r['sample'] for r in inputs if cohort == 'all_validation' or
            (r['source_sample'] in selection) == (cohort == 'selection32')]
        if not ids:
            continue
        for policy in ('old256_prompt', 'huber_prompt', 'huber_constant', 'rank_prompt', 'rank_constant'):
            for baseline in ('static_old256', 'static_train800'):
                for metric in ('kl', 'nll'):
                    own = np.array([np.mean([index[i, policy, s][metric] for s in SEEDS]) for i in ids])
                    static = np.array([index[i, baseline, None][metric] for i in ids])
                    comparisons.append({'cohort': cohort, 'policy': policy, 'baseline': baseline, 'metric': metric,
                        'dynamic_mean': float(own.mean()), 'static_mean': float(static.mean()), **paired(static - own, None, 200579)})
            for s in SEEDS:
                own = [index[i, policy, s] for i in ids]
                per_seed.append({'cohort': cohort, 'policy': policy, 'seed': s,
                    'kl': float(np.mean([r['kl'] for r in own])), 'nll': float(np.mean([r['nll'] for r in own])),
                    'unique_masks': len({r['route_index'] for r in own})})
    checks = []
    for objective in ('huber', 'rank'):
        for seed in SEEDS:
            a, b = [load(out / f'{objective}_{arm}_s{seed}_fit.json') for arm in ('prompt', 'constant')]
            assert a['initial_sha256'] == b['initial_sha256'] and a['updates'] == b['updates'] and a['parameters'] == b['parameters']
            checks.append({'objective': objective, 'seed': seed, 'passed': True})
    for seed in SEEDS:
        a, b = [load(out / f'{objective}_prompt_s{seed}_fit.json') for objective in ('huber', 'rank')]
        assert a['initial_sha256'] == b['initial_sha256'] and a['updates'] == b['updates']
    # Check old trained routers reproduce the exact prior Stage19 validation rows.
    previous = {r['sample']: r for i in range(3) for r in read_lines(Path(cfg['previous_run']) / f'private_quality_{i:03d}.jsonl')
        if r['bank_name'] == 'k20_b08' and r['policy'] == 'fixed_best' and r['task'] == 'c4_development'}
    for r in inputs:
        assert abs(index[r['sample'], 'static_old256', None]['kl'] - previous[r['sample']]['kl']) < 1e-6
    summary = {'comparisons': comparisons, 'per_seed': per_seed, 'paired_training_checks': checks,
        'quality_rows': len(rows), 'limits': cfg['limits']}
    write(out / 'summary.json', summary)
    return summary


def publish_result(root, out, cfg, summary):
    from run_static_swap_pipeline_v1 import publish
    report = root / 'analysis' / f'{out.name}_results.zh-CN.md'
    lines = ['# 阶段20：K20、8组合的验证集强化', '',
        '保持Stage19的8个完整组合不变，将训练文档由256扩展到原800条；保留原32条开发身份选检查点，其他68条复核，全部100条仍为反复使用的开发集。基础OPT冻结，全文embedding，回答内固定路径。', '',
        '两种训练目标：原smoothL1联合KL残差回归；回归加0.5倍代价差加权的组合成对排序损失。三个种子579/580/581，各有同初始化、同更新预算的常量观察对照。8组合库不在800条上重建；另按800条训练均值重新选同库最强静态，避免沿用较弱旧静态。', '',
        '扩数据实验的24轮更新数高于旧256条实验，不称作计算预算匹配的纯数据规模消融。组合监督全部是真实联合KL，不将逐层损失相加。', '',
        '| 群体 | 路由策略 | 动态KL | 800条最强静态KL | 静态减动态 | 配对95%区间 |',
        '|---|---|---:|---:|---:|---|']
    for r in summary['comparisons']:
        if r['metric'] == 'kl' and r['baseline'] == 'static_train800':
            lo, hi = r['ci95']
            lines.append(f"| {r['cohort']} | {r['policy']} | {r['dynamic_mean']:.5f} | {r['static_mean']:.5f} | {r['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', '旧256条检查点也在相同100条输入上实际重放。逐种子KL/NLL、路径数量、旧静态与新静态的全部比较保存在汇总中。区间按文档配对bootstrap，未校正多重比较，不含完整种子不确定性。', '',
        '本轮只强化验证集质量，不增加外部泛化任务，不作新的吞吐或显存声明。使用6张GPU、64个CPU核心，worker各10核、控制器4核。旧文件全部保留并验哈希；权重、token和逐文档私有记录不上传。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [验收](../{out.relative_to(root)}/verification.json)']
    with report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    files = [report] + [root / 'tools' / name for name in cfg['source_sha256']]
    files += [p for p in out.glob('*.json') if not p.name.startswith('private_') and p.name not in ('preservation_before.json', 'allocation.json')]
    receipt = publish(root, files, out.name, 'Refine K20 codebook routing with training800, ranking loss and stronger static controls', dict(os.environ))
    write(out / 'publication_verified.json', receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    setup()
    tracked = preserve(root, out)
    previous = root / 'runs/stage19_parallel_20261112_03_codebook'
    old_cfg = load(previous / 'protocol.json')
    full_cfg = load(root / 'runs/stage17_static_swap_20261111_01/protocol.json')
    bank = load(previous / 'k20_b08_bank.json')
    assert bank['k'] == 20 and len(bank['routes']) == 8
    write(out / 'candidate_bank.json', bank)
    sources = ['codebook_refinement_core_v2.py', 'run_codebook_refinement_six_gpu_v2.py', 'launch_codebook_refinement_v2.py']
    cfg = {'model': old_cfg['model'], 'data_run': old_cfg['data_run'], 'output': str(out), 'previous_run': str(previous),
        'train_ids': full_cfg['train_ids'], 'validation_ids': old_cfg['validation_ids'], 'selection_ids': old_cfg['selection_ids'],
        'old_train_ids': old_cfg['train_ids'], 'seeds': SEEDS, 'epochs': 24, 'probe': args.probe,
        'bank_sha256': sha256(out / 'candidate_bank.json'), 'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'resources': '6GPU/64CPU, workers10 each/controller4',
        'training': 'Full prompt embedding+positions, K20 fixed bank8. SmoothL1 versus SmoothL1+0.5 cost-gap-weighted complete-mask pairwise logistic ranking. AdamW2e-4/batch16/24epochs/5%warmup/cosine/clip1.',
        'selection': 'Earliest minimum actual joint KL on original32 IDs at epochs0/1/4/...24; no validation rows in gradient updates. Other68 reported separately, all previously reviewed development.',
        'limits': 'No generalization or acceleration claim. Expanded training uses more updates than Stage19; not compute-matched data scaling. Fixed library from original256 training only. New training-mean strongest fixed mask chosen from these8, not the full85 with unavailable800-document labels.'}
    if args.probe:
        # Include genuinely new labels as well as old rows in the engineering run.
        old_set = set(old_cfg['train_ids'])
        fresh = [i for i in cfg['train_ids'] if i not in old_set][:6]
        cfg['train_ids'] = old_cfg['train_ids'][:6] + fresh
        cfg['selection_ids'] = cfg['selection_ids'][:6]
        cfg['validation_ids'] = cfg['selection_ids']
        cfg['epochs'] = 1
    assert not set(cfg['train_ids']) & set(cfg['validation_ids'])
    keep = set(cfg['train_ids'] + cfg['validation_ids'])
    embeddings, evidence = {}, {}
    old_embeddings = root / 'runs/stage17_static_swap_20261111_01'
    for rank in range(4):
        path = old_embeddings / f'private_embeddings_{rank:03d}.pth'
        digest = sha256(path)
        assert digest == load(old_embeddings / f'label_result_{rank:03d}.json')['embedding_sha256']
        values = torch.load(path, weights_only=True, map_location='cpu')
        embeddings.update({i: v for i, v in values.items() if i in keep})
        evidence[str(path)] = digest
        del values
    assert set(embeddings) == keep
    with (out / 'private_embeddings.pth').open('xb') as stream:
        torch.save(embeddings, stream)
    del embeddings
    gc.collect()
    source_run = root / 'runs/stage18_deep_swap_pilot_20261112_01_main_k20'
    reused = {}
    for rank in range(4):
        path = source_run / f'private_labels_{rank:03d}.jsonl'
        digest = sha256(path)
        assert digest == load(source_run / f'label_result_{rank:03d}.json')['labels_sha256']
        evidence[str(path)] = digest
        for r in read_lines(path):
            if r['sample'] in keep:
                idx = [route['source_route'] for route in bank['routes']]
                reused[str(r['sample'])] = {'sample': r['sample'], 'kl': [r['kl'][j] for j in idx],
                    'nll': [r['nll'][j] for j in idx], 'sequence_sha256': r['sequence_sha256'], 'reused': True}
    write(out / 'private_reused_labels.json', reused)
    cfg['evidence_sha256'] = evidence
    cfg['embedding_sha256'] = sha256(out / 'private_embeddings.pth')
    cfg['old_checkpoints'] = {str(s): load(previous / f'k20_b08_prompt_s{s}_fit.json')['best'] for s in SEEDS}
    inputs = [r for r in read_lines(previous / 'private_inputs.jsonl') if r['task'] == 'c4_development' and r['source_sample'] in cfg['validation_ids']]
    with (out / 'private_inputs.jsonl').open('x') as stream:
        for r in inputs:
            stream.write(json.dumps(r) + '\n')
    cfg['input_sha256'] = sha256(out / 'private_inputs.jsonl')
    write(out / 'protocol.json', cfg)
    write(out / 'cpu_checks.json', cpu_checks())
    with SixWorkers(worker, cfg) as workers:
        workers.receive('labels_done')
        labels = {r['sample']: r for i in range(6) for r in read_lines(out / f'private_labels_{i:03d}.jsonl')}
        assert set(labels) == keep
        means = np.array([labels[i]['kl'] for i in cfg['train_ids']]).mean(0)
        write(out / 'static_controls.json', {'best_route': int(means.argmin()), 'training_mean_kl': means.tolist(),
            'train_documents': len(cfg['train_ids']), 'training_only': True})
        workers.send({'kind': 'fit'})
        workers.receive('fit_done')
        workers.send({'kind': 'quality'})
        workers.receive('done')
    summary = finalize(out, cfg)
    verify_preserved(out, tracked)
    write(out / 'verification.json', {'passed': True, 'quality_rows': summary['quality_rows'],
        'training_validation_disjoint': True, 'same_complete_bank_and_full_prompt': True,
        'paired_initialization_updates': True, 'new_static_chosen_on_training800': not args.probe,
        'joint_labels_and_old_baseline_replayed': True})
    write(out / 'gpu_work_complete.json', {'state': 'completed'})
    stamp('stage20_gpu_complete', quality_rows=summary['quality_rows'])
    if not args.probe:
        publish_result(root, out, cfg, summary)


if __name__ == '__main__':
    main()
