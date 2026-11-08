"""Four-GPU frozen horizontal quality evaluation; no new training or timing.

Each question routes ONCE from its common answer-blind prompt and reuses the
route for every answer. All option likelihoods use batch-one genuine bypass.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizontal_data_v1 import DATA, HARNESS, encode_question, policies, read_dataset, wiki_windows
from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import fixed_route, raw_prompt_embeddings
from stage1_benchmark_v2 import sha256, stamp, write_json


def pin_threads(cpus):
    for thread in Path('/proc/self/task').iterdir():
        try: os.sched_setaffinity(int(thread.name), cpus)
        except ProcessLookupError: pass


def score_sequence(model, ids, target_length, skips):
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    with fixed_route(model, skips):
        hidden = model.model(x[:, :-1], attention_mask=torch.ones_like(x[:, :-1]), use_cache=False).last_hidden_state
        # Match normal CausalLM GEMM dimensions before slicing: FP16 matmul
        # kernels can change scores when only target positions reach the head.
        logits = model.lm_head(hidden)[:, -target_length:].float()
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, -target_length:].reshape(-1), reduction='none')
    if not torch.isfinite(losses).all(): raise ValueError('Nonfinite answer score')
    return -losses.double().sum().item()


def prepare(root, output, probe):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/liushenyu/opt-13b', use_fast=False, local_files_only=True)
    inputs, manifest = [], {}
    for task, count in [('piqa', 1838), ('hellaswag', 10042), ('winogrande', 1267)]:
        split = 'train' if probe else 'validation'
        data = read_dataset(task, split)
        if not probe and len(data) != count: raise ValueError('Unexpected benchmark length')
        path = DATA[task] / f'{task}-{split}.arrow'
        selected = range(4) if probe else range(len(data))
        rows = [encode_question(task, data[index], index, tokenizer) for index in selected]
        inputs.extend(rows)
        manifest[task] = {'path': str(path), 'sha256': sha256(path), 'split': split,
            'available_records': len(data), 'evaluated_records': len(rows),
            'router_tokens_min': min(len(r['route_ids']) for r in rows),
            'router_tokens_max': max(len(r['route_ids']) for r in rows),
            'truncated_context_tokens': sum(sum(r['truncated_context_tokens']) for r in rows)}
    split = 'validation' if probe else 'test'
    wiki = read_dataset('wikitext', split)
    windows = wiki_windows(wiki, tokenizer)
    if probe: windows = windows[:4]
    inputs.extend(windows)
    path = DATA['wikitext'] / f'wikitext-{split}.arrow'
    manifest['wikitext_conditional'] = {'path': str(path), 'sha256': sha256(path), 'split': split,
        'text_lines': len(wiki), 'evaluated_windows': len(windows),
        'evaluated_documents': len({r['document'] for r in windows}),
        'scored_tokens': sum(r['target_lengths'][0] for r in windows)}
    private = output / 'private_benchmark_inputs.jsonl'
    with private.open('x') as stream:
        for row in inputs: stream.write(json.dumps(row) + '\n')
    write_json(output / 'data_manifest.json', {'tasks': {name: {key: value for key, value in task.items() if key != 'path'} for name, task in manifest.items()}, 'private_input_sha256': sha256(private),
        'tokenizer': 'OPT local slow tokenizer; explicit add_special_tokens=True for MC and BOS for each Wiki window',
        'wiki': 'Article boundaries at single-equals headings; within each article local512token windows with128context+up to384targets, stride384; first127 article text tokens unscored, scored targets do not overlap; token-weighted conditional PPL.',
        'mc': 'Zero-shot task templates from pinned harness; context trailing whitespace moves to target, boundary tokenization asserted; all target tokens scored; left context truncated only if total exceeds2049.',
        'normalization': 'PIQA/HellaSwag acc_norm divides option loglikelihood by choice CHARACTER length as pinned harness; WinoGrande reports acc only.'})
    return inputs


def score_probe(model, record, skips):
    ids, n = record['sequence_ids'][0], record['target_lengths'][0]
    score = score_sequence(model, ids, n, skips)
    x = torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device)[None]
    with fixed_route(model, skips):
        # Independent reference exercises the normal CausalLM forward/head path.
        logits = model(x[:, :-1], attention_mask=torch.ones_like(x[:, :-1]), use_cache=False).logits[:, -n:].float()
    reference = F.log_softmax(logits, -1).gather(-1, x[:, -n:, None]).double().sum().item()
    difference = abs(score - reference)
    if difference > 2e-4: raise ValueError('Likelihood scorer differs from reference causal LM')
    return {'skips': sorted(skips), 'score': score, 'reference_score': reference,
        'absolute_error': difference, 'tolerance': 2e-4}


def worker(rank, config, messages):
    try:
        cpu = [int(i) for i in os.environ['NEUROBYPASS_CPU_IDS'].split(',')]
        pin_threads(cpu[rank * 10:(rank + 1) * 10])
        torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval()
        model.requires_grad_(False)
        routers = {}
        for policy in config['policies']:
            if policy['kind'] == 'router':
                path = policy['checkpoint']['path']
                if path not in routers:
                    router = DepthRouter().to(f'cuda:{rank}').eval()
                    router.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
                    router.requires_grad_(False); routers[path] = router
        output = Path(config['output'])
        inputs = [json.loads(line) for line in Path(config['private_inputs']).read_text().splitlines()][rank::4]
        with torch.inference_mode():
            first = inputs[0]
            probes = []
            for task in sorted({r['task'] for r in inputs}):
                record = next(r for r in inputs if r['task'] == task)
                for mask in [[], config['fixed_prefix']]:
                    probes.append({'task': task, **score_probe(model, record, mask)})
        write_json(output / f'worker_probe_{rank:03d}.json', {'rank': rank,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': 10,
            'thread_cpu_ids': {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}, 'likelihood_probes': probes})
        started = time.monotonic()
        with (output / f'quality_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
            for offset, record in enumerate(inputs):
                route_x = torch.tensor(record['route_ids'], dtype=torch.long, device=f'cuda:{rank}')[None]
                emb = raw_prompt_embeddings(model, route_x)
                encoded = {}
                route_scores = {}
                for policy in config['policies']:
                    if policy['kind'] == 'static':
                        route = policy['skips']
                    else:
                        path = policy['checkpoint']['path']; router = routers[path]
                        if path not in encoded: encoded[path] = router.encode(emb)
                        route = router.route_encoded(encoded[path], policy['k'], policy['prefix'])[0].tolist()
                    key = tuple(sorted(route))
                    if key not in route_scores:
                        route_scores[key] = [score_sequence(model, ids, length, route)
                            for ids, length in zip(record['sequence_ids'], record['target_lengths'])]
                    scores = route_scores[key]
                    row = {'task': record['task'], 'sample': record['sample'], 'document': record.get('document'),
                        'document_sha256': record['document_sha256'], 'policy': policy['name'], 'seed': policy['seed'],
                        'k': policy['k'], 'skipped_layers': list(key), 'router_observation_tokens': len(record['route_ids']),
                        'target_lengths': record['target_lengths'], 'scores': scores,
                        'normalization_lengths': record['normalization_lengths'],
                        'truncated_context_tokens': record['truncated_context_tokens']}
                    if record['gold'] is not None:
                        row.update({'gold': record['gold'], 'prediction': int(np.argmax(scores)),
                            'correct': int(np.argmax(scores) == record['gold'])})
                        if record['task'] != 'winogrande':
                            normalized = np.asarray(scores) / np.asarray(record['normalization_lengths'])
                            row.update({'prediction_norm': int(normalized.argmax()),
                                'correct_norm': int(normalized.argmax() == record['gold'])})
                    else:
                        row.update({'nll_sum': -scores[0], 'scored_tokens': record['target_lengths'][0]})
                    stream.write(json.dumps(row) + '\n')
                stream.flush()
                if offset % 20 == 0 or offset + 1 == len(inputs):
                    messages.put({'kind': 'progress', 'rank': rank, 'completed_records': offset + 1,
                        'total_records': len(inputs), 'task': record['task'], 'elapsed_seconds': time.monotonic() - started})
        messages.put({'kind': 'done', 'rank': rank, 'records': len(inputs)})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def summary_tables(rows):
    groups, quality = defaultdict(list), []
    for row in rows: groups[row['task'], row['policy'], row['seed'], row['k']].append(row)
    for (task, policy, seed, k), values in sorted(groups.items(), key=lambda r: str(r[0])):
        row = {'task': task, 'policy': policy, 'seed': seed, 'k': k, 'records': len(values),
            'unique_masks': len({tuple(v['skipped_layers']) for v in values})}
        if task == 'wikitext_conditional':
            nll, tokens = sum(v['nll_sum'] for v in values), sum(v['scored_tokens'] for v in values)
            row.update({'nll_sum': nll, 'scored_tokens': tokens, 'nll_per_token': nll / tokens, 'conditional_ppl': math.exp(nll / tokens)})
        else:
            row['accuracy'] = float(np.mean([v['correct'] for v in values]))
            if task != 'winogrande': row['accuracy_norm'] = float(np.mean([v['correct_norm'] for v in values]))
        quality.append(row)
    return quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 4: p.error('Use four-GPU resource launcher')
    started, root = time.monotonic(), Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    training = root / 'runs/stage6_depth_train_main_20261107_01'
    budgets = [10, 12, 16]
    selected = policies(root, training, budgets)
    fixed = json.loads((training / 'fixed_global_policy.json').read_text())['ordered_skips']
    config = {'output': str(output), 'model': '/home/liushenyu/opt-13b', 'training_run': str(training),
        'budgets': budgets, 'policies': selected, 'fixed_prefix': fixed, 'probe': args.probe,
        'private_inputs': str(output / 'private_benchmark_inputs.jsonl')}
    tracked = {str(path): sha256(path) for path in root.rglob('*') if path.is_file()
        and not path.is_relative_to(output) and '.git' not in path.parts
        and path.suffix in {'.py', '.json', '.jsonl', '.md', '.csv', '.png'}}
    for path in root.parent.glob('*.py'): tracked[str(path)] = sha256(path)
    write_json(output / 'preservation_before.json', tracked)
    inputs = prepare(root, output, args.probe)
    task_paths = ['piqa/piqa.yaml', 'hellaswag/hellaswag.yaml', 'hellaswag/utils.py',
        'winogrande/default.yaml', 'winogrande/preprocess_winogrande.py']
    write_json(output / 'protocol.json', {**config, 'source_sha256': {path.name: sha256(path) for path in
        [Path(__file__), root / 'tools/horizontal_data_v1.py', root / 'tools/prompt_depth_core_v1.py', root / 'tools/prompt_fixed_core_v1.py']},
        'harness': {'root': str(HARNESS), 'commit': '95d580638385578c1c07fa554cf16ad7f5b5f460',
            'task_files': {name: sha256(HARNESS / 'lm_eval/tasks' / name) for name in task_paths}},
        'data_manifest_sha256': sha256(output / 'data_manifest.json'),
        'observation': 'MC common question prefix only: BOS+first127 text tokens, variable short length without padding. WinoGrande uses only text before blank, no option or scored suffix. Wiki observes first128tokens of localwindow.',
        'route': 'Exactly one path per question/policy and all options reuse it; no correct label, answer option or future Wiki target reaches router.',
        'scoring': 'FP16 OPT SDPA, batch1, use_cache=False, genuine layer bypass; full-sequence LM head GEMM then target slicing and FP64 loss reduction; teacher forcing. Explicit BOS, max input2048, char normalized acc as pinned task recipe.',
        'selection': 'Existing per-K C4 development-selected checkpoints, all3seeds; no checkpoint/hyperparameter/static path selection using benchmark.',
        'cost_policy': 'No throughput/peak-memory timing in this quality stage. User requests cost comparison after quality results justify it.',
        'limitations': ['Custom adapter audited against pinned task recipes; local old dataset revisions differ from latest task dataset identifiers.',
            'WikiText conditional128+up-to384window PPL is not standard rolling/full corpus PPL.',
            'WinoGrande routing only sees pre-blank common prefix, intentionally less information than filled scoring contexts.',
            'Baseline methods are frozen OPT adaptations without recovery, not complete official paper replications.',
            'Three seeds are repeated models on the same questions, not three times as many independent benchmark samples.']})
    context = mp.get_context('spawn')
    messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, config, messages)) for rank in range(4)]
    done = []
    cpu = [int(i) for i in os.environ['NEUROBYPASS_CPU_IDS'].split(',')]
    try:
        for process in processes: process.start()
        pin_threads(cpu[40:42])
        while len(done) < 4:
            try: row = messages.get(timeout=5)
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in processes) or all(process.exitcode is not None for process in processes):
                    raise RuntimeError('Worker exited without completing benchmark')
                continue
            if row['kind'] == 'error': raise RuntimeError(row['traceback'])
            if row['kind'] == 'progress': stamp('horizontal_quality_progress', **{k: v for k, v in row.items() if k != 'kind'})
            elif row['kind'] == 'done': done.append(row)
            else: raise ValueError('Unexpected worker message')
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0: raise RuntimeError('Worker exit failure')
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join(timeout=10)
            if process.is_alive(): process.kill(); process.join(timeout=5)
    rows = [json.loads(line) for path in sorted(output.glob('quality_*.jsonl')) for line in path.read_text().splitlines()]
    if len(rows) != len(inputs) * len(selected): raise ValueError('Incomplete quality output')
    write_json(output / 'summary.json', {'quality': summary_tables(rows), 'quality_rows': len(rows),
        'input_records': len(inputs), 'policies': len(selected), 'probe': args.probe,
        'elapsed_seconds': time.monotonic() - started})
    if any(sha256(path) != digest for path, digest in tracked.items()): raise ValueError('Existing evidence changed')
    write_json(output / 'preservation_after.json', {'preserved_files': len(tracked)})
    stamp('horizontal_quality_complete', elapsed_seconds=time.monotonic() - started, rows=len(rows))


if __name__ == '__main__': main()
