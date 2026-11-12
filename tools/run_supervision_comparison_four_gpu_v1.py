"""Stage24 four-GPU label completion, paired supervision training and real quality replay."""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import load, load_choice, score_joint, teachers_for
from full_context_core_v1 import prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers, preserve, read_lines, resources, setup_worker, thread_record, verify_preserved, write
from headroom_ruler_core_v1 import paired
from prompt_signal_diagnostic_v2 import hidden_prompt, prompt_cost
from stage1_benchmark_v2 import sha256, stamp
from supervision_comparison_core_v1 import JOBS, SEEDS, cpu_checks, fit, targets


def dense_model(cfg, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)


def worker(rank, cfg, inbox, messages):
    setup_worker(rank); device = f'cuda:{rank}'; out = Path(cfg['output'])
    old = Path(cfg['old_run']); earlier = Path(cfg['prompt_run'])
    bank = load(out / 'candidate_bank.json'); data = Records(cfg['data_run'])
    reused = {r['sample']: r for i in range(2) for r in read_lines(earlier / f'private_prompt_costs_{i:03d}.jsonl')}
    model = dense_model(cfg, device)
    inputs = {r['source_sample']: r for r in read_lines(old / 'private_inputs.jsonl')}
    future = {r['sample']: r for i in range(6) for r in read_lines(old / f'private_labels_{i:03d}.jsonl')}
    samples = (cfg['validation_ids'] + cfg['train_ids'])[rank::4]
    replay_error = 0.; reused_count = 0
    with torch.inference_mode(), (out / f'private_prompt_costs_{rank:03d}.jsonl').open('x') as stream:
        for offset, sample in enumerate(samples):
            raw, p = data[sample]
            assert data.rows[sample]['sequence_sha256'] == future[sample]['sequence_sha256']
            if sample in reused:
                row = {**reused[sample], 'reused': True}; reused_count += 1
                assert row['scored_tokens'] == p - 1 and row['prompt_length'] == p
                assert row['sequence_sha256'] == data.rows[sample]['sequence_sha256']
            else:
                row = None
            if row is None or offset == 0:
                prompt = torch.tensor(raw[:p], device=device, dtype=torch.long)[None]
                teacher_hidden = hidden_prompt(model, prompt, [])
                teacher = torch.cat([F.log_softmax(model.lm_head(teacher_hidden[:, i:i + 64]).float(), -1)
                    for i in range(0, p - 1, 64)], dim=1)
                assert prompt_cost(model, teacher, teacher_hidden, prompt)['kl'] == 0
                values = []
                for route in bank['routes']:
                    hidden = hidden_prompt(model, prompt, route['skips'])
                    value = prompt_cost(model, teacher, hidden, prompt)
                    assert np.isfinite([value['kl'], value['nll']]).all()
                    values.append(value)
                measured = {'sample': sample, 'prompt_length': p, 'scored_tokens': p - 1,
                    'sequence_sha256': data.rows[sample]['sequence_sha256'], 'reused': False,
                    'prompt_kl': [v['kl'] for v in values], 'prompt_nll': [v['nll'] for v in values]}
                if row is not None:
                    replay_error = max(replay_error, max(abs(a - b) for key in ('prompt_kl', 'prompt_nll')
                        for a, b in zip(row[key], measured[key])))
                    assert replay_error < 1e-6
                else:
                    row = measured
                if offset == 0:
                    cached = torch.load(old / 'private_embeddings.pth', weights_only=True, map_location='cpu')[sample]
                    actual = prompt_embeddings(model, prompt).cpu()
                    assert torch.equal(actual, cached) and actual.shape[1] == p
                    direct = score_joint(model, inputs[sample], bank['routes'][1]['skips'], teachers_for(model, inputs[sample]))
                    assert abs(direct['kl'] - future[sample]['kl'][1]) < 1e-6
                del teacher, teacher_hidden, hidden
            stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
            if offset % 12 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'labels',
                    'documents': offset + 1, 'total': len(samples), 'reused': reused_count})
    write(out / f'labels_worker_{rank:03d}.json', {'passed': True, 'documents': len(samples),
        'reused_documents': reused_count, 'new_documents': len(samples) - reused_count,
        'same_chunk_reused_kl_nll_replay_error': replay_error,
        'full_prompt_embeddings_replay_exact': True, 'existing_continuation_kl_replayed': True,
        'labels_sha256': sha256(out / f'private_prompt_costs_{rank:03d}.jsonl'), **thread_record()})
    del model; gc.collect(); torch.cuda.empty_cache()
    messages.put({'kind': 'labels_done', 'rank': rank}); assert inbox.get()['kind'] == 'fit'
    prompt_rows = {r['sample']: r for i in range(4) for r in read_lines(out / f'private_prompt_costs_{i:03d}.jsonl')}
    embeddings = torch.load(old / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    for kind, seed in JOBS[rank::4]:
        for observe in (True, False):
            fit(out, cfg, future, prompt_rows, embeddings, kind, seed, observe, device, messages, rank)
    del embeddings; gc.collect(); torch.cuda.empty_cache()
    messages.put({'kind': 'fit_done', 'rank': rank}); assert inbox.get()['kind'] == 'quality'
    model = dense_model(cfg, device)
    routers = {}
    for kind in ('mixed', 'prompt_full'):
        for seed in SEEDS:
            for arm in ('prompt', 'constant'):
                receipt = load(out / f'{kind}_{arm}_s{seed}_fit.json')
                assert sha256(receipt['best']['path']) == receipt['best']['sha256']
                routers[kind, arm, seed] = load_choice(receipt['best']['path'], device)
    for seed in SEEDS:
        for arm in ('prompt', 'constant'):
            receipt = cfg['reference_checkpoints'][f'{arm}_s{seed}']
            assert sha256(receipt['path']) == receipt['sha256']
            routers['future32', arm, seed] = load_choice(receipt['path'], device)
    error = 0.; documents = 0
    with torch.inference_mode(), (out / f'private_quality_{rank:03d}.jsonl').open('x') as stream:
        for offset, sample in enumerate(cfg['validation_ids'][rank::4]):
            record = inputs[sample]
            embedding = prompt_embeddings(model, torch.tensor(record['route_ids'], device=device)[None])
            assert embedding.shape[1] == len(record['route_ids'])
            proposals = [('strong_static', 'none', None, 1)]
            proposals += [(kind, arm, seed, int(router(embedding).argmin(-1))) for (kind, arm, seed), router in routers.items()]
            teachers = teachers_for(model, record); cache = {}
            for kind, arm, seed, route in proposals:
                if route not in cache:
                    cache[route] = score_joint(model, record, bank['routes'][route]['skips'], teachers)
                    error = max(error, abs(cache[route]['kl'] - future[sample]['kl'][route]))
                    assert error < 1e-6
                stream.write(json.dumps({'source_sample': sample, 'policy': kind, 'arm': arm, 'seed': seed,
                    'route_index': route, 'k': 20, **cache[route]}, allow_nan=False) + '\n')
            stream.flush(); documents += 1
            if offset % 6 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'quality', 'documents': documents})
    write(out / f'quality_worker_{rank:03d}.json', {'passed': True, 'documents': documents,
        'max_continuation_kl_replay_error': error, 'full_prompt_lengths': True, 'base_model_frozen': True,
        'quality_sha256': sha256(out / f'private_quality_{rank:03d}.jsonl'), **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def finalize(out, cfg):
    rows = [r for i in range(4) for r in read_lines(out / f'private_quality_{i:03d}.jsonl')]
    assert len(rows) == len(cfg['validation_ids']) * 19
    index = {(r['source_sample'], r['policy'], r['arm'], r['seed']): r for r in rows}
    assert len(index) == len(rows)
    comparisons = []; per_seed = []; training_checks = []
    for kind in ('mixed', 'prompt_full'):
        for seed in SEEDS:
            records = [load(out / f'{kind}_{arm}_s{seed}_fit.json') for arm in ('prompt', 'constant')]
            assert records[0]['initial_sha256'] == records[1]['initial_sha256']
            assert records[0]['updates'] == records[1]['updates']
            if not cfg['probe']:
                assert records[0]['updates'] == 1200
            for record in records:
                values = [r for r in load(out / f"{kind}_{record['arm']}_s{seed}_history.json") if 'selection_kl' in r]
                best = min(values, key=lambda r: r['selection_kl'])
                assert best['epoch'] == record['best']['epoch']
                replay = np.mean([index[s, kind, record['arm'], seed]['kl'] for s in cfg['selection_ids']])
                assert abs(replay - record['best']['selection_kl']) < 1e-6
            training_checks.append({'kind': kind, 'seed': seed, 'paired_initialization_and_updates': True})
    old = Path(cfg['old_run'])
    old_quality = {(r['source_sample'], r['policy'], r['seed']): r for i in range(6)
        for r in read_lines(old / f'private_quality_{i:03d}.jsonl')}
    for sample in cfg['validation_ids']:
        for arm in ('prompt', 'constant'):
            for seed in SEEDS:
                a = index[sample, 'future32', arm, seed]; b = old_quality[sample, f'huber_{arm}', seed]
                assert a['route_index'] == b['route_index'] and abs(a['kl'] - b['kl']) < 1e-6
    for cohort in ('selection32', 'other68', 'all_validation100'):
        ids = [s for s in cfg['validation_ids'] if cohort == 'all_validation100' or
            ((s in cfg['selection_ids']) == (cohort == 'selection32'))]
        if not ids:
            continue
        for kind in ('future32', 'mixed', 'prompt_full'):
            for arm in ('prompt', 'constant'):
                for seed in SEEDS:
                    own = [index[s, kind, arm, seed] for s in ids]
                    per_seed.append({'cohort': cohort, 'policy': kind, 'arm': arm, 'seed': seed,
                        'kl': float(np.mean([r['kl'] for r in own])), 'nll': float(np.mean([r['nll'] for r in own])),
                        'switches': sum(r['route_index'] != 1 for r in own), 'unique_routes': len({r['route_index'] for r in own})})
                for metric in ('kl', 'nll'):
                    own = np.array([np.mean([index[s, kind, arm, seed][metric] for seed in SEEDS]) for s in ids])
                    for baseline in ('strong_static', 'future32_prompt'):
                        control = np.array([index[s, 'strong_static', 'none', None][metric] if baseline == 'strong_static'
                            else np.mean([index[s, 'future32', 'prompt', seed][metric] for seed in SEEDS]) for s in ids])
                        comparisons.append({'cohort': cohort, 'policy': kind, 'arm': arm, 'metric': metric,
                            'baseline': baseline, 'own_mean': float(own.mean()), 'baseline_mean': float(control.mean()),
                            **paired(control - own, None, 240579)})
    summary = {'comparisons': comparisons, 'per_seed': per_seed, 'training_checks': training_checks,
        'quality_rows': len(rows), 'reused_future32_control_rows': len(cfg['validation_ids']) * 6,
        'limits': cfg['limits']}
    write(out / 'summary.json', summary)
    return summary


def report(root, out, summary):
    path = root / 'analysis' / f'{out.name}_results.zh-CN.md'
    text = ['# 阶段24：全文提示词、续写与混合监督对照', '',
        'K20/40，原8个完整组合库不变，全文提示词embedding输入，回答内固定路径，OPT冻结。原800训练文档、原32条选检查点、其余68条单列复核；100条仍为反复使用的开发文档。', '',
        '未来32-token监督对照复用Stage20 Huber三种子及常量对照，已逐条重放路由与KL。新增全文提示词lambda1和混合lambda0.5两组，各三种子及常量对照。提示词标签覆盖P-1个已观察下一词位置，不读取续写；完整组合实际联合KL保留协同。', '',
        'F和P分别相对index0；A=训练均值F+P-训练均值P，T=(1-lambda)F+lambdaA。所有组共享F的训练均值先验与训练标准差、初始化、smoothL1、AdamW2e-4、batch16、4096token微批、24轮1200更新、调度器与检查点评测频率。仅训练目标改变；强静态仍是800训练选定的index1。', '',
        '| 群体 | 监督 | 观察 | KL | 强静态KL | 静态减动态 | 配对95%区间 |',
        '|---|---|---|---:|---:|---:|---|']
    for r in summary['comparisons']:
        if r['metric'] == 'kl' and r['baseline'] == 'strong_static' and r['cohort'] != 'selection32':
            lo, hi = r['ci95']; text.append(f"| {r['cohort']} | {r['policy']} | {r['arm']} | {r['own_mean']:.5f} | {r['baseline_mean']:.5f} | {r['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    text += ['', '全部NLL、逐种子、选择集、原续写监督对照的配对比较见汇总。区间按同文档先平均3种子再bootstrap，未校正多重比较，不含完整种子不确定性。没有按68条选择lambda、种子或门槛，没有新增校准/集成。', '',
        '4张GPU/42CPU；worker各10核、控制器2核。训练前验证新旧全文提示词标签重放、embedding长度及值一致；实际模型重放每条开发路径KL。旧文件与输入检查点验哈希保全。源码、协议、审计与汇总上传，权重、token、embedding和逐文档标签留本地。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [验收](../{out.relative_to(root)}/verification.json)']
    with path.open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true'); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    cpus = resources(); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    from horizontal_quality_v3 import pin_threads
    pin_threads(cpus[40:42]); preserved = preserve(root, out)
    old = root / 'runs/stage20_codebook_refine_20261112_02_main'; earlier = root / 'runs/stage23_prompt_signal_20261112_02_main'
    previous = load(old / 'protocol.json')
    assert load(old / 'independent_audit.json')['passed'] and load(earlier / 'independent_audit.json')['passed']
    for rank in range(2):
        assert sha256(earlier / f'private_prompt_costs_{rank:03d}.jsonl') == load(earlier / f'worker_{rank:03d}.json')['labels_sha256']
    assert sha256(old / 'private_embeddings.pth') == previous['embedding_sha256']
    bank = load(old / 'candidate_bank.json'); assert bank['k'] == 20 and len(bank['routes']) == 8
    write(out / 'candidate_bank.json', bank)
    references = {}; initial = {}
    for seed in SEEDS:
        for arm in ('prompt', 'constant'):
            receipt = load(old / f'huber_{arm}_s{seed}_fit.json')
            assert receipt['updates'] == 1200 and receipt['train_documents'] == 800
            assert sha256(receipt['best']['path']) == receipt['best']['sha256']
            references[f'{arm}_s{seed}'] = receipt['best']; initial[f'{arm}_s{seed}'] = receipt['initial_sha256']
    sources = ['supervision_comparison_core_v1.py', 'run_supervision_comparison_four_gpu_v1.py', 'launch_supervision_comparison_v1.py']
    evidence = [old / 'protocol.json', old / 'private_embeddings.pth', old / 'candidate_bank.json', earlier / 'protocol.json', earlier / 'independent_audit.json']
    evidence += [earlier / f'private_prompt_costs_{i:03d}.jsonl' for i in range(2)]
    evidence += [old / f'private_{kind}_{i:03d}.jsonl' for kind in ('labels', 'quality') for i in range(6)]
    cfg = {'output': str(out), 'old_run': str(old), 'prompt_run': str(earlier), 'model': previous['model'],
        'data_run': previous['data_run'], 'train_ids': previous['train_ids'], 'selection_ids': previous['selection_ids'],
        'validation_ids': previous['validation_ids'], 'epochs': 24, 'probe': args.probe,
        'reference_checkpoints': references, 'reference_initial_sha256': initial,
        'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'evidence_sha256': {str(p): sha256(p) for p in evidence},
        'resources': '4GPU/42CPU; workers10 each/controller2',
        'targets': 'Relative index0: F=future32 cost, P=fullprompt cost; A=mean_trainF+P-mean_trainP. T=(1-lambda)F+lambdaA; lambda0/0.5/1 fixed. All use same F prior and scale, matched initialization and updates.',
        'selection': 'Original32 actual continuation joint KL; earliest minimum epoch0/1/4/...24. Other68 only report. No lambda, seed or threshold selection on68.',
        'limits': 'Reviewed development; no throughput/memory/generalization claim. Constant observer paired controls. Complete-mask costs, no sums of single-layer costs.'}
    if args.probe:
        reuse_ids = set(r['sample'] for i in range(2) for r in read_lines(earlier / f'private_prompt_costs_{i:03d}.jsonl'))
        cfg['train_ids'] = [i for i in previous['train_ids'] if i in reuse_ids][:8] + [i for i in previous['train_ids'] if i not in reuse_ids][:8]
        cfg['validation_ids'] = previous['selection_ids'][:4] + [i for i in previous['validation_ids'] if i not in previous['selection_ids']][:4]
        cfg['selection_ids'] = previous['selection_ids'][:4]; cfg['epochs'] = 1
    assert not set(cfg['train_ids']) & set(cfg['validation_ids'])
    write(out / 'protocol.json', cfg); write(out / 'cpu_checks.json', cpu_checks())
    stamp('supervision_start', probe=args.probe, train=len(cfg['train_ids']), validation=len(cfg['validation_ids']))
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'labels_done')
        workers.send_all({'kind': 'fit'}); workers.receive(4, 'fit_done')
        workers.send_all({'kind': 'quality'}); workers.receive(4, 'done')
    summary = finalize(out, cfg)
    for path, digest in cfg['evidence_sha256'].items():
        assert sha256(path) == digest
    for receipt in references.values():
        assert sha256(receipt['path']) == receipt['sha256']
    verify_preserved(out, preserved)
    write(out / 'verification.json', {'passed': True, 'probe': args.probe, 'quality_rows': summary['quality_rows'],
        'identical_future_initialization': not args.probe, 'paired_arms_and_updates': True,
        'reference_controls_routes_and_kl_reproduced': True, 'checkpoints_and_old_evidence_unchanged': True,
        'only_original32_selects_epochs': True, 'full_prompt_labels_and_embeddings': True})
    if not args.probe:
        report(root, out, summary)
    stamp('supervision_complete', probe=args.probe, run=str(out))


if __name__ == '__main__':
    main()
