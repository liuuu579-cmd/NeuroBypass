"""Stage23: does exact full-prompt joint KL select useful continuation masks?

An OFFLINE diagnostic, not a deployment router. Uses one dense and eight student
prefills per input, no future tokens in prompt scoring. No checkpoints updated.
"""
import gc
import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import load, score_joint, teachers_for
from full_context_data_v1 import Records
from full_context_experiment_v1 import preserve, read_lines, thread_record, verify_preserved, write
from headroom_ruler_core_v1 import paired
from horizontal_quality_v3 import pin_threads
from prompt_fixed_core_v1 import fixed_route
from stage1_benchmark_v2 import sha256, stamp


ALPHAS = (0., .25, .5, 1.)
MARGINS = (0., .1, .2, .4)


def setup(rank=None):
    assert os.environ['NEUROBYPASS_GPU_COUNT'] == '2' and os.environ['NEUROBYPASS_CPU_COUNT'] == '21'
    cpus = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
    chosen = cpus[20:] if rank is None else cpus[rank * 10:(rank + 1) * 10]
    pin_threads(chosen); torch.set_num_threads(len(chosen)); torch.set_num_interop_threads(1)
    if rank is not None:
        torch.cuda.set_device(rank)


def hidden_prompt(model, prompt, skips):
    # Predict only tokens already present in the prompt: token1 through tokenP-1.
    with fixed_route(model, skips):
        return model.model(prompt[:, :-1], attention_mask=torch.ones_like(prompt[:, :-1]),
            use_cache=False).last_hidden_state


def prompt_cost(model, teacher_log_probs, student_hidden, prompt, chunk=64):
    length = student_hidden.shape[1]
    assert length == prompt.shape[1] - 1 == teacher_log_probs.shape[1]
    kl_sum, nll_sum = 0., 0.
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        student = F.log_softmax(model.lm_head(student_hidden[:, start:end]).float(), -1)
        teacher = teacher_log_probs[:, start:end]
        kl_sum += float((teacher.exp() * (teacher - student)).sum(-1).double().sum())
        nll_sum += float(-student.gather(-1, prompt[:, start + 1:end + 1, None]).double().sum())
    return {'kl': kl_sum / length, 'nll': nll_sum / length, 'scored_tokens': length}


def worker(rank, cfg, messages):
    try:
        setup(rank)
        out = Path(cfg['output']); device = f'cuda:{rank}'
        data = Records(cfg['data_run']); bank = load(cfg['bank_path'])
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        old = Path(cfg['old_run'])
        labels = {r['sample']: r for i in range(6) for r in read_lines(old / f'private_labels_{i:03d}.jsonl')}
        inputs = {r['source_sample']: r for r in read_lines(old / 'private_inputs.jsonl')}
        checks = []; lengths = []
        with torch.inference_mode(), (out / f'private_prompt_costs_{rank:03d}.jsonl').open('x') as stream:
            for offset, sample in enumerate(cfg['ids'][rank::2]):
                raw, p = data[sample]
                assert data.rows[sample]['sequence_sha256'] == labels[sample]['sequence_sha256']
                prompt = torch.tensor(raw[:p], dtype=torch.long, device=device)[None]
                teacher_hidden = hidden_prompt(model, prompt, [])
                teacher = torch.cat([F.log_softmax(model.lm_head(teacher_hidden[:, i:i + 64]).float(), -1)
                    for i in range(0, p - 1, 64)], dim=1)
                assert teacher.shape[1] == p - 1
                if offset == 0:
                    assert prompt_cost(model, teacher, teacher_hidden, prompt)['kl'] == 0
                    # Our first row is a dev document, enabling exact old continuation replay.
                    record = inputs[sample]; dense = teachers_for(model, record)
                    replay = score_joint(model, record, bank['routes'][cfg['baseline']]['skips'], dense)
                    assert abs(replay['kl'] - labels[sample]['kl'][cfg['baseline']]) < 1e-6
                    checks.append({'continuation_kl_replay_error': abs(replay['kl'] - labels[sample]['kl'][cfg['baseline']]),
                        'dense_vs_itself_prompt_kl_zero': True})
                    del dense
                values = []
                for route in bank['routes']:
                    student_hidden = hidden_prompt(model, prompt, route['skips'])
                    result = prompt_cost(model, teacher, student_hidden, prompt)
                    assert result['kl'] >= -1e-5 and np.isfinite([result['kl'], result['nll']]).all()
                    if offset == 0 and route['index'] == cfg['baseline']:
                        alternate = prompt_cost(model, teacher, student_hidden, prompt, chunk=31)
                        projection_error = abs(result['kl'] - alternate['kl'])
                        assert projection_error < 2e-4, ('fp16_projection_chunk_error', projection_error)
                        checks[-1]['prompt_chunk_replay_error'] = abs(result['kl'] - alternate['kl'])
                    values.append(result)
                row = {'sample': sample, 'prompt_length': p, 'scored_tokens': p - 1,
                    'sequence_sha256': data.rows[sample]['sequence_sha256'],
                    'prompt_kl': [v['kl'] for v in values], 'prompt_nll': [v['nll'] for v in values]}
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                lengths.append(p)
                del teacher, teacher_hidden, student_hidden
                messages.put({'kind': 'progress', 'rank': rank, 'documents': offset + 1,
                    'total': len(cfg['ids'][rank::2]), 'prompt_length': p})
        receipt = {'passed': True, 'rank': rank, 'documents': len(lengths), 'min_prompt_length': min(lengths),
            'max_prompt_length': max(lengths), 'over128': sum(p > 128 for p in lengths),
            'labels_sha256': sha256(out / f'private_prompt_costs_{rank:03d}.jsonl'),
            'checks': checks, 'no_future_tokens_in_prompt_scoring': True, 'base_model_frozen': True,
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(), **thread_record()}
        write(out / f'worker_{rank:03d}.json', receipt)
        messages.put({'kind': 'done', 'rank': rank})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def decide(scores, baseline, margin=0.):
    best = scores.argmin(1)
    gain = scores[:, baseline] - scores[np.arange(len(scores)), best]
    return np.where(gain > margin, best, baseline).astype(np.int64)


def choose(train_prompt, future_train800, selection_prompt, selection_future, baseline):
    prior = (future_train800 - future_train800[:, baseline:baseline + 1]).mean(0)
    train_relative = train_prompt - train_prompt[:, baseline:baseline + 1]
    centered = (selection_prompt - selection_prompt[:, baseline:baseline + 1]) - train_relative.mean(0)
    scale = float((future_train800 - future_train800[:, baseline:baseline + 1]).std())
    choices = []
    for alpha in ALPHAS:
        for margin in MARGINS:
            routes = decide(prior + alpha * centered, baseline, margin * scale)
            choices.append({'alpha': alpha, 'margin_units': margin,
                'selection_kl': float(selection_future[np.arange(len(routes)), routes].mean()),
                'selection_switches': int((routes != baseline).sum()), 'serial': len(choices)})
    chosen = min(choices, key=lambda r: (r['selection_kl'], r['selection_switches'], r['serial']))
    return {'prior': prior.tolist(), 'train_prompt_relative_mean': train_relative.mean(0).tolist(),
        'scale': scale, 'grid': choices, 'chosen': chosen}


def main():
    setup(); root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    preservation = preserve(root, out)
    old = root / 'runs/stage20_codebook_refine_20261112_02_main'
    previous = load(old / 'protocol.json')
    # Freeze the 64 training identities independently of costs and validation.
    training64 = np.random.default_rng(230579).permutation(previous['train_ids'])[:64].tolist()
    ids = previous['validation_ids'] + training64
    assert len(set(ids)) == 164
    baseline = load(old / 'static_controls.json')['best_route']; assert baseline == 1
    sources = ['prompt_signal_diagnostic_v2.py']
    cfg = {'output': str(out), 'old_run': str(old), 'model': previous['model'], 'data_run': previous['data_run'],
        'bank_path': str(old / 'candidate_bank.json'), 'baseline': baseline, 'ids': ids,
        'training64_ids': training64, 'train800_ids': previous['train_ids'],
        'selection32_ids': previous['selection_ids'], 'validation100_ids': previous['validation_ids'],
        'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'evidence_sha256': {str(old / name): sha256(old / name) for name in
            ['protocol.json', 'candidate_bank.json', 'static_controls.json', 'private_inputs.jsonl'] +
            [f'private_labels_{i:03d}.jsonl' for i in range(6)]},
        'grid': {'alpha': ALPHAS, 'margin_units': MARGINS},
        'resources': '2GPU/21CPU; workers10 each/controller1',
        'scoring': 'Exact joint KL on all observed prompt positions1..P-1; no future continuation token in scoring. Existing frozen continuation32-token labels only for train means and development outcome.',
        'selection': 'Prompt train mean from random original64 training docs, continuation prior/scale from original800.16 predefined alpha/margin settings chosen only on original32; other68 never used for selection.',
        'limits': 'Offline teacher+8 student forward passes; not a deployable router, speedup, learned prompt-only upper bound, or guarantee that its distilled router would match this diagnostic. Reviewed development; prompts/continuations vary in length.'}
    write(out / 'protocol.json', cfg)
    context = mp.get_context('spawn'); messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, cfg, messages)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        done = set()
        while len(done) < 2:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (0, None) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Prompt diagnostic worker ended before receipt')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            if item['kind'] == 'done':
                assert item['rank'] not in done; done.add(item['rank'])
            else:
                stamp('prompt_signal_progress', **{k: v for k, v in item.items() if k != 'kind'})
        for process in processes:
            process.join(timeout=30)
        assert all(p.exitcode == 0 for p in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
    rows = {r['sample']: r for rank in range(2) for r in read_lines(out / f'private_prompt_costs_{rank:03d}.jsonl')}
    assert set(rows) == set(ids)
    old_labels = {r['sample']: r for rank in range(6) for r in read_lines(old / f'private_labels_{rank:03d}.jsonl')}
    prompt = np.array([rows[s]['prompt_kl'] for s in ids]); costs = np.array([old_labels[s]['kl'] for s in ids])
    nll = np.array([old_labels[s]['nll'] for s in ids]); lookup = {s: i for i, s in enumerate(ids)}
    train = np.array([lookup[i] for i in training64]); selection = np.array([lookup[i] for i in previous['selection_ids']])
    future800 = np.array([old_labels[i]['kl'] for i in previous['train_ids']])
    assert int(future800.mean(0).argmin()) == baseline
    fitted = choose(prompt[train], future800, prompt[selection], costs[selection], baseline)
    centered = prompt - prompt[:, baseline:baseline + 1] - np.array(fitted['train_prompt_relative_mean'])
    chosen = fitted['chosen']; score = np.array(fitted['prior']) + chosen['alpha'] * centered
    calibrated = decide(score, baseline, chosen['margin_units'] * fitted['scale'])
    direct = decide(prompt, baseline)
    cohorts = {'train64_diagnostic': train, 'selection32': selection,
        'other68': np.array([lookup[i] for i in previous['validation_ids'] if i not in previous['selection_ids']]),
        'all_validation100': np.array([lookup[i] for i in previous['validation_ids']])}
    results = []
    for name, routes in [('exact_prompt_argmin', direct), ('train_aligned_prompt_gate', calibrated),
                         ('strong_static', np.full(len(ids), baseline))]:
        for cohort, ix in cohorts.items():
            delta = costs[ix, baseline] - costs[ix, routes[ix]]
            results.append({'policy': name, 'cohort': cohort, 'kl': float(costs[ix, routes[ix]].mean()),
                'nll': float(nll[ix, routes[ix]].mean()), 'static_kl': float(costs[ix, baseline].mean()),
                'switches': int((routes[ix] != baseline).sum()), 'beneficial_switches': int((delta > 0).sum()),
                'harmful_switches': int((delta < 0).sum()), **paired(delta, None, 230579)})
    write(out / 'policy_fits.json', fitted)
    write(out / 'summary.json', {'results': results, 'scored_prompt_tokens': sum(r['scored_tokens'] for r in rows.values()),
        'max_prompt_length': max(r['prompt_length'] for r in rows.values()), 'future_labels_reused': True,
        'no_checkpoint_updated': True, 'prompt_cost_only_diagnostic': True})
    write(out / 'private_selected_routes.json', {str(s): int(calibrated[lookup[s]]) for s in previous['validation_ids']})
    for path, digest in cfg['evidence_sha256'].items():
        assert sha256(path) == digest
    verify_preserved(out, preservation)
    write(out / 'verification.json', {'passed': True, 'documents': 164, 'prompt_train64_only_for_alignment': True,
        'original32_only_for_gate_selection': True, 'no_future_tokens_in_prompt_scoring': True,
        'projection_chunk_absolute_tolerance': 2e-4, 'projection_precision': 'FP16 matmul can depend slightly on row chunk shape; actual difference recorded per worker, dense identity uses identical64-token chunks.',
        'each_document_scored_all_prompt_positions': all(r['scored_tokens'] == r['prompt_length'] - 1 for r in rows.values()),
        'full_complete_mask_joint_costs': True, 'no_model_training': True, 'old_files_preserved': len(preservation)})
    report = root / 'analysis' / f'{out.name}_results.zh-CN.md'
    lines = ['# 阶段23：提示词全文联合KL能否迁移到续写选层', '',
        '保持K20/40与原8个完整组合。随机固定64条原训练文档，加原100条开发文档，每条离线计算完整模型和8条学生路径的提示词联合KL；评分覆盖已观察提示词的所有P-1个下一词位置，不读取任何续写token。监督与指标均为完整组合实际代价，保留层间协同。', '',
        '这不是线上路由器，也不是所有提示词路由方法的收益上限：离线计算每条输入9次模型前向。它只检验提示词自身KL这个候选监督信号是否能选择后续32-token质量较好的组合。尚未训练新的预测器，也不宣称速度收益。', '',
        '强静态index1仍由原800条训练续写KL确定。提示词代价与续写代价的均值对齐只用随机64条训练提示词和原800续写训练标签；16个固定alpha/门槛配置只用原32条选择，68条只复核。直接提示词argmin作为预先指定的另一条规则分列，不用68条选赢家。', '',
        '| 群体 | 规则 | 续写KL | 强静态KL | 静态减规则 | 配对95%区间 | 有益/有害切换 |',
        '|---|---|---:|---:|---:|---|---|']
    for row in results:
        if row['cohort'] != 'train64_diagnostic' and row['policy'] != 'strong_static':
            lo, hi = row['ci95']; lines.append(f"| {row['cohort']} | {row['policy']} | {row['kl']:.5f} | {row['static_kl']:.5f} | {row['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {row['beneficial_switches']}/{row['harmful_switches']} |")
    lines += ['', f"原32条选定配置：`{chosen}`。", '',
        '若对齐规则在68条仍差于强静态，不能据此断言提示词不含可预测信号，只能说明这条提示词KL规则未建立迁移收益；若改善，再考虑将该完整组合代价蒸馏到单路由器并重新评测。所有文档为已使用开发文档，配对区间为探索性结果，不校正多重比较。', '',
        '使用2GPU/21CPU，每worker10核、控制器1核。权重冻结；稠密自比KL为0、不同词表投影分块的KL误差小于2e-4（FP16矩阵乘法按行数可能产生舍入差异，实际误差另列），并重放旧续写KL；旧文件验哈希保全。模型权重、token、逐文档标签与路由留本地，仅公开自写源码、协议、校验和聚合结果。', '',
        f'[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [校验](../{out.relative_to(root)}/verification.json)']
    with report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    stamp('prompt_signal_complete', run=str(out), chosen=chosen)


if __name__ == '__main__':
    main()
