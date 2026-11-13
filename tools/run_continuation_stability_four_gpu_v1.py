"""Stage27: natural continuation-window label stability; no router training."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import load
from continuation_stability_core_v1 import cpu_checks, digest, prepare, summarize, token_metrics
from full_context_experiment_v1 import Workers, preserve, read_lines, resources, setup_worker, thread_record, verify_preserved, write
from horizontal_quality_v3 import pin_threads
from prompt_fixed_core_v1 import continuation_logits
from stage1_benchmark_v2 import sha256, stamp


def worker(rank, cfg, inbox, messages):
    from transformers import AutoModelForCausalLM
    setup_worker(rank); device = f'cuda:{rank}'; out = Path(cfg['output'])
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    bank = load(Path(cfg['old_run']) / 'candidate_bank.json')
    inputs = {r['sample']: r for r in read_lines(cfg['input_file'])}
    old_labels = {r['sample']: r for j in range(6) for r in read_lines(Path(cfg['old_run']) / f'private_labels_{j:03d}.jsonl')}
    saved = {}; errors = []; short_error = 0.; repeat_error = 0.; future_leak_error = 0.
    with torch.inference_mode(), (out / f'private_labels_{rank:03d}.jsonl').open('x') as stream:
        for offset, sample in enumerate(cfg['ids'][rank::4]):
            row = inputs[sample]; p = row['prompt_length']; ids = torch.tensor(row['ids'], dtype=torch.long, device=device)[None]
            assert len(row['ids']) == p + 128 and digest(row['ids']) == row['extended_sequence_sha256']
            assert digest(row['ids'][:p]) == row['prompt_sha256']
            assert digest(row['ids'][:p+32]) == row['original_sequence_sha256'] == old_labels[sample]['sequence_sha256']
            teacher = continuation_logits(model, ids, p, []).log_softmax(-1)
            assert teacher.shape[1] == 128
            kls = []; nlls = []
            for route in bank['routes']:
                logits = continuation_logits(model, ids, p, route['skips'])
                kl, nll = token_metrics(logits, teacher, ids[:, p:])
                kls.append(kl); nlls.append(nll)
            kl_tokens = torch.stack(kls, 1); nll_tokens = torch.stack(nlls, 1)
            costs = kl_tokens.reshape(4, 32, 8).mean(1).double().numpy()
            losses = nll_tokens.reshape(4, 32, 8).mean(1).double().numpy()
            errors.append(np.abs(costs[0] - np.asarray(old_labels[sample]['kl'])))
            if offset == 0:
                short = ids[:, :p+32]; short_teacher = continuation_logits(model, short, p, []).log_softmax(-1)
                for route in bank['routes']:
                    kl, nll = token_metrics(continuation_logits(model, short, p, route['skips']), short_teacher, short[:, p:])
                    short_error = max(short_error, abs(float(kl.mean()) - old_labels[sample]['kl'][route['index']]))
                    assert abs(float(kl.mean()) - old_labels[sample]['kl'][route['index']]) < 1e-6
                    assert abs(float(nll.mean()) - old_labels[sample]['nll'][route['index']]) < 1e-5
                again, _ = token_metrics(continuation_logits(model, ids, p, bank['routes'][1]['skips']), teacher, ids[:, p:])
                repeat_error = float((again - kl_tokens[:, 1]).abs().max()); assert repeat_error == 0
                altered = ids.clone(); altered[:, p+32:] = altered[:, p+32:].flip(1)
                altered_teacher = continuation_logits(model, altered, p, []).log_softmax(-1)
                future_leak_error = float((altered_teacher[:, :32] - teacher[:, :32]).abs().max())
                assert future_leak_error == 0
                alt, _ = token_metrics(continuation_logits(model, altered, p, bank['routes'][1]['skips']), altered_teacher, altered[:, p:])
                assert torch.equal(alt[:32], kl_tokens[:32, 1])
            measured = {'sample': sample, 'prompt_length': p, 'prompt_sha256': row['prompt_sha256'],
                'extended_sequence_sha256': row['extended_sequence_sha256'], 'kl': costs.tolist(), 'nll': losses.tolist()}
            stream.write(json.dumps(measured, allow_nan=False) + '\n'); stream.flush()
            saved[sample] = {'kl': kl_tokens, 'nll': nll_tokens}
            if offset % 4 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'window_labels', 'documents': offset + 1,
                    'total': len(cfg['ids'][rank::4]), 'maximum_old_first32_difference': float(np.max(errors))})
    path = out / f'private_token_metrics_{rank:03d}.pth'
    with path.open('xb') as stream:
        torch.save(saved, stream)
    write(out / f'worker_{rank:03d}.json', {'passed': True, 'documents': len(saved),
        'label_sha256': sha256(out / f'private_labels_{rank:03d}.jsonl'), 'token_metrics_sha256': sha256(path),
        'maximum_direct_old32_kl_replay_error': short_error, 'maximum_repeated_long_kl_error': repeat_error,
        'maximum_altered_future_prefix_logprob_error': future_leak_error,
        'maximum_long_vs_old_first32_kl_difference': float(np.max(errors)),
        'mean_long_vs_old_first32_kl_difference': float(np.mean(errors)), **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def report(root, out, summary):
    text = ['# 阶段27：固定提示词的续写窗口标签稳定性', '',
        '128条原训练文档，保留原始全文提示词和K20的8个完整剪层组合。读取同一文档中紧随提示词的128个真实token，分为四个互不重叠的32-token窗口。每个组合从头到尾固定，分别计算教师强制下每个位置的真实联合KL与NLL。不更新模型或路由器，不重新构建候选库。', '',
        '后续窗口的模型上下文包含此前真实续写，因此这是随续写位置推进的偏好稳定性诊断；不是同一条件分布下独立采样的标签噪声估计。窗口1—3的内容未提供给路由器，旧路由决策在本轮前已固定。', '',
        '| 窗口对，从0编号 | 最优组合一致率 | 文档内8组合排名Spearman | 首窗最优在次窗的KL后悔值 |', '|---|---:|---:|---:|']
    for row in summary['stability']['window_pairs']:
        text.append(f"| {row['windows']} | {100 * row['winner_agreement']:.2f}% | {row['within_document_route_spearman']['mean']:.4f} | {row['first_winner_regret_on_second']['mean']:.6f} |")
    text += ['', '| 策略 | 评分范围 | KL | 相对强静态改善 | 探索性95%区间 |', '|---|---|---:|---:|---|']
    for row in summary['policy_transfer']:
        if row['metric'] == 'kl' and row['window'] in (0, 'later96', 'all128'):
            lo, hi = row['improvement']['ci95']
            text.append(f"| {row['policy']} | {row['window']} | {row['mean']:.6f} | {row['improvement']['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    text += ['', 'strong_static固定使用历史强静态索引1。embedding/hidden2沿用原四折外折文档对应的三个独立种子决策；mean3为质量均值，不是三模型集成。first32/full128/other96_future_oracle均读取真实未来，只用于诊断，不是线上可用的策略。other96对每个被评估窗口分别用其余三个窗口选组合，也不代表回答内允许换层。', '',
        '抽样不读取KL，但要求原提示词之后有128个同文档token，因此偏向有较长正文的文档；排除2016-token提示词与后续正文不足的文档。原100条开发文档不参与。候选库曾用原训练文档筛选，本轮是开发诊断，不作独立泛化声明。区间按文档重采样，不覆盖模型训练、候选筛选与折间依赖，未作多重比较校正。', '',
        '每worker首次文档重放旧32-token联合KL，重复新长序列，并修改后96个token检查因果前缀不变。长序列首32与旧短序列的小幅差异单列报告；不把数值重算差异当作标签随机性。', '',
        '使用4GPU/42CPU，worker各10核、控制器2核；旧文件验哈希保全，权重、语料、token与逐文档结果保留本地。未新增吞吐、显存或恢复训练结果。', '',
        f"[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [抽样](../{out.relative_to(root)}/sampling.json) · [独立审计](../{out.relative_to(root)}/independent_audit.json)"]
    with (root / 'analysis' / f'{out.name}_results.zh-CN.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true'); parser.add_argument('--probe-run', type=Path); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    cpus = resources(); pin_threads(cpus[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    preserved = preserve(root, out); old = root / 'runs/stage20_codebook_refine_20261112_02_main'; previous = load(old / 'protocol.json')
    if args.probe:
        input_file, sampling = prepare(out, previous)
    else:
        assert args.probe_run and load(args.probe_run / 'verification.json')['passed']
        old_cfg = load(args.probe_run / 'protocol.json'); input_file = Path(old_cfg['input_file']); sampling = load(args.probe_run / 'sampling.json')
        assert sha256(input_file) == old_cfg['input_file_sha256']; write(out / 'sampling.json', sampling)
    ids = sampling['selected_ids'][:4] if args.probe else sampling['selected_ids']
    assert not set(ids) & set(previous['validation_ids']) and set(ids) <= set(previous['train_ids'])
    bank = load(old / 'candidate_bank.json'); assert bank['k'] == 20 and len(bank['routes']) == 8
    assert all(len(set(r['skips'])) == 20 for r in bank['routes'])
    sources = ('continuation_stability_core_v1.py', 'run_continuation_stability_four_gpu_v1.py', 'launch_continuation_stability_v1.py')
    data = Path(previous['data_run']); model = Path(previous['model'])
    manifest = load(data / 'data_manifest.json')
    evidence = [old / 'protocol.json', old / 'candidate_bank.json', data / 'data_manifest.json', Path(manifest['corpus']),
        model / 'config.json', model / 'pytorch_model.bin.index.json']
    evidence += [model / f'pytorch_model-{i:05d}-of-00003.bin' for i in range(1, 4)]
    evidence += [old / f'private_labels_{i:03d}.jsonl' for i in range(6)]
    evidence += [root / 'runs/stage25_crossfit_20261113_02_main/private_crossfit_rows.jsonl',
        root / 'runs/stage26_semantic_prefix_20261113_01_main/private_quality_rows.jsonl']
    cfg = {'output': str(out), 'old_run': str(old), 'model': previous['model'], 'data_run': previous['data_run'],
        'probe': args.probe, 'probe_run': str(args.probe_run) if args.probe_run else None, 'ids': ids,
        'input_file': str(input_file), 'input_file_sha256': sha256(input_file), 'sampling_seed': 270579,
        'windows': [[0, 32], [32, 64], [64, 96], [96, 128]], 'strong_static_index': 1,
        'near_tie_tolerance': .01, 'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'evidence_sha256': {str(path): sha256(path) for path in evidence},
        'protocol': 'Select128 eligible original TRAIN documents without KL; preserve original full prompt. Frozen K20 bank8, four contiguous nonoverlapping32-token windows. No fitting or configuration search.',
        'primary_diagnostics': 'W0-vs-later winner/rank agreement and W0 oracle transfer relative to strong static1; frozen prior embedding/hidden2 gate policies on128-token score. Near-tie/material-order tolerance .01 fixed before labels.',
        'limits': 'Reviewed training data and historically selected bank; eligibility-conditioned sample. Later teacher-forced contexts contain earlier continuation tokens. Not IID repeated continuations or a direct label-noise estimate. Future-known diagnostics are not deployment policies. No speed claims.'}
    assert cfg['evidence_sha256'][manifest['corpus']] == manifest['corpus_sha256']
    write(out / 'protocol.json', cfg); write(out / 'cpu_checks.json', cpu_checks())
    stamp('continuation_stability_start', probe=args.probe, documents=len(ids))
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'done')
    labels = {r['sample']: r for rank in range(4) for r in read_lines(out / f'private_labels_{rank:03d}.jsonl')}
    assert set(labels) == set(ids)
    costs = np.asarray([labels[i]['kl'] for i in ids]); nll = np.asarray([labels[i]['nll'] for i in ids])
    summary = summarize(root, ids, costs, nll); summary['engineering_only'] = args.probe
    write(out / 'summary.json', summary)
    assert sha256(input_file) == cfg['input_file_sha256']
    for path, digest in cfg['evidence_sha256'].items():
        assert sha256(path) == digest
    verify_preserved(out, preserved)
    write(out / 'verification.json', {'passed': True, 'probe': args.probe, 'documents': len(ids),
        'original_full_prompt_unchanged': True, 'no_router_or_opt_training': True,
        'complete_joint_masks_scored': True, 'historical_dev100_excluded': True, 'old_evidence_unchanged': True})
    if not args.probe:
        report(root, out, summary)
    stamp('continuation_stability_complete', probe=args.probe, run=str(out))


if __name__ == '__main__':
    main()
