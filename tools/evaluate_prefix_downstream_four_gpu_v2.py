"""Correct only stage11 m=8 inference; preserve and replay unchanged controls."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from horizontal_quality_v3 import pin_threads, score_probe, score_sequence, summary_tables
from prefix_downstream_correction_core_v1 import (STAGE6, STAGE10, combined_rows,
    iter_rows, key, partition, policies)
from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import raw_prompt_embeddings
from stage1_benchmark_v2 import sha256, stamp, write_json


def worker(rank, config, messages):
    try:
        cpu = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
        pin_threads(cpu[rank * 10:(rank + 1) * 10])
        torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval()
        model.requires_grad_(False)
        inputs = [json.loads(line) for line in Path(config['private_inputs']).open()][rank::4]
        first = {task: next(r for r in inputs if r['task'] == task) for task in sorted({r['task'] for r in inputs})}
        replay_keys = {key(p) for p in config['replay_policies']}
        replay_inputs = {(r['task'], r['sample']) for r in first.values()}
        originals = {(r['task'], r['sample'], key(r)): r for r in iter_rows(config['reuse_run'])
            if key(r) in replay_keys and (r['task'], r['sample']) in replay_inputs}
        routers = {}
        for policy in config['policies'] + config['replay_policies']:
            if policy['kind'] == 'router':
                path = policy['checkpoint']['path']
                if path not in routers:
                    router = DepthRouter().to(f'cuda:{rank}').eval()
                    router.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
                    router.requires_grad_(False); routers[path] = router

        def route(policy, emb, encoded):
            if policy['kind'] == 'static':
                return sorted(policy['skips'])
            path = policy['checkpoint']['path']
            if path not in encoded:
                encoded[path] = routers[path].encode(emb)
            ordered = routers[path].route_encoded(encoded[path], policy['k'], policy['prefix'])[0].tolist()
            if ordered[:policy['m']] != policy['prefix'] or len(set(ordered)) != policy['k']:
                raise ValueError('Executed route violates the fixed-prefix contract')
            return sorted(ordered)

        output = Path(config['output'])
        probes, replays = [], []
        with torch.inference_mode():
            for task, record in first.items():
                for mask in [[], config['global_path'][:8]]:
                    probes.append({'task': task, **score_probe(model, record, mask)})
                x = torch.tensor(record['route_ids'], dtype=torch.long, device=f'cuda:{rank}')[None]
                emb, encoded = raw_prompt_embeddings(model, x), {}
                for policy in config['replay_policies']:
                    mask = route(policy, emb, encoded)
                    original = originals[task, record['sample'], key(policy)]
                    if mask != original['skipped_layers']:
                        raise ValueError('Unchanged control route failed GPU replay')
                    scores = [score_sequence(model, ids, n, mask) for ids, n in zip(record['sequence_ids'], record['target_lengths'])]
                    error = max(abs(a - b) for a, b in zip(scores, original['scores']))
                    if error > 2e-4:
                        raise ValueError('Unchanged control likelihood failed GPU replay')
                    replays.append({'task': task, 'sample': record['sample'], 'policy': policy['name'],
                        'seed': policy['seed'], 'k': policy['k'], 'skipped_layers': mask, 'scores': scores,
                        'max_absolute_error': error, 'tolerance': 2e-4})
        write_json(output / f'worker_probe_{rank:03d}.json', {'rank': rank,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': 10,
            'thread_cpu_ids': {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()},
            'likelihood_probes': probes, 'unchanged_control_replays': replays})
        started = time.monotonic()
        with (output / f'quality_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
            for offset, record in enumerate(inputs):
                x = torch.tensor(record['route_ids'], dtype=torch.long, device=f'cuda:{rank}')[None]
                emb, encoded, cached = raw_prompt_embeddings(model, x), {}, {}
                for policy in config['policies']:
                    mask = route(policy, emb, encoded)
                    if tuple(mask) not in cached:
                        cached[tuple(mask)] = [score_sequence(model, ids, n, mask)
                            for ids, n in zip(record['sequence_ids'], record['target_lengths'])]
                    scores = cached[tuple(mask)]
                    row = {'task': record['task'], 'sample': record['sample'], 'document': record.get('document'),
                        'document_sha256': record['document_sha256'], 'policy': policy['name'], 'seed': policy['seed'],
                        'k': policy['k'], 'skipped_layers': mask, 'router_observation_tokens': len(record['route_ids']),
                        'target_lengths': record['target_lengths'], 'scores': scores,
                        'normalization_lengths': record['normalization_lengths'],
                        'truncated_context_tokens': record['truncated_context_tokens']}
                    if record['gold'] is not None:
                        prediction = int(np.argmax(scores))
                        row.update(gold=record['gold'], prediction=prediction, correct=int(prediction == record['gold']))
                        if record['task'] != 'winogrande':
                            prediction = int((np.asarray(scores) / np.asarray(record['normalization_lengths'])).argmax())
                            row.update(prediction_norm=prediction, correct_norm=int(prediction == record['gold']))
                    else:
                        row.update(nll_sum=-scores[0], scored_tokens=record['target_lengths'][0])
                    stream.write(json.dumps(row) + '\n')
                stream.flush()
                if offset % 20 == 0 or offset + 1 == len(inputs):
                    messages.put({'kind': 'progress', 'rank': rank, 'completed_records': offset + 1,
                        'total_records': len(inputs), 'task': record['task'], 'elapsed_seconds': time.monotonic() - started})
        messages.put({'kind': 'done', 'rank': rank})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def run(args):
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 4:
        raise ValueError('Use four-GPU resource launcher')
    started = time.monotonic()
    root, source = Path(__file__).resolve().parents[1], args.reuse_run.resolve()
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    original = json.loads((source / 'protocol.json').read_text())
    if original['probe'] != args.probe or json.loads((source / 'status.json').read_text())['state'] != 'completed':
        raise ValueError('Original run is incomplete or wrong split')
    global_path, selected = policies(root)
    reused, rerun = partition(original['policies'], selected)
    replay_keys = {('full', None, 0), ('static_joint_kl', None, 10), ('m00', 579, 10), ('m04', 580, 12), ('m06', 581, 12)}
    replay_policies = [p for p in reused if key(p) in replay_keys]
    private = source / 'private_benchmark_inputs.jsonl'
    manifest = json.loads((source / 'data_manifest.json').read_text())
    if sha256(private) != manifest['private_input_sha256']:
        raise ValueError('Original private input changed')
    write_json(output / 'data_manifest.json', manifest)
    tracked = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()
        and not p.is_relative_to(output) and '.git' not in p.parts
        and p.suffix in {'.py', '.json', '.jsonl', '.md', '.csv', '.png'}}
    for p in root.parent.glob('*.py'):
        tracked[str(p)] = sha256(p)
    write_json(output / 'preservation_before.json', tracked)
    sources = [Path(__file__), *[root / 'tools' / name for name in [
        'prefix_downstream_correction_core_v1.py', 'verify_prefix_downstream_v2.py',
        'finalize_prefix_downstream_v2.py', 'test_prefix_downstream_correction_v1.py',
        'horizontal_quality_v3.py', 'horizontal_data_v1.py', 'prompt_depth_core_v1.py', 'prompt_fixed_core_v1.py']]]
    provenance = [source / name for name in ['protocol.json', 'summary.json', 'verification.json',
        'data_manifest.json', 'private_benchmark_inputs.jsonl', 'status.json']]
    provenance += sorted(source.glob('quality_*.jsonl'))
    provenance += [root / STAGE6 / 'static_paths.json', root / STAGE6 / 'fixed_global_policy.json',
                   root / STAGE10 / 'protocol.json', root / STAGE10 / 'verification.json']
    config = {'output': str(output), 'model': original['model'], 'budgets': [10, 12],
        'policies': rerun, 'reused_policies': reused, 'all_policies': selected,
        'replay_policies': replay_policies, 'global_path': global_path, 'probe': args.probe,
        'reuse_run': str(source), 'private_inputs': str(private),
        'harness': original['harness'], 'data_manifest_sha256': sha256(output / 'data_manifest.json'),
        'source_sha256': {str(p): sha256(p) for p in sources},
        'provenance_sha256': {str(p): sha256(p) for p in provenance},
        'correction': 'Only m08 is rerun with the same frozen checkpoints and training prefix global_path[:8]. Old m08 is excluded.',
        'route': original['route'], 'scoring': original['scoring'], 'selection': original['selection'],
        'seed_aggregation': 'Question-wise mean accuracy; Wiki exp(mean token-weighted NLL across seeds).',
        'limitations': original['limitations'] + ['Unchanged control replay is sampled, not a full GPU rescore.']}
    write_json(output / 'protocol.json', config)
    context = mp.get_context('spawn'); messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, config, messages)) for rank in range(4)]
    done = set()
    try:
        for process in processes:
            process.start()
        pin_threads(list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))[40:42])
        while len(done) < 4:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker exited before completing correction')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            if item['kind'] == 'done':
                done.add(item['rank'])
            else:
                stamp('prefix_correction_progress', **{k: v for k, v in item.items() if k != 'kind'})
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Correction worker failed')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
            if process.is_alive():
                process.kill(); process.join(timeout=5)
    rows = list(combined_rows(output, config))
    n = sum(1 for _ in private.open())
    if len(rows) != n * 27:
        raise ValueError('Incomplete corrected comparison')
    write_json(output / 'summary.json', {'quality': summary_tables(rows), 'quality_rows': len(rows),
        'new_quality_rows': n * 6, 'reused_quality_rows': n * 21, 'input_records': n,
        'policies': 27, 'probe': args.probe, 'elapsed_seconds': time.monotonic() - started})
    for p, digest in tracked.items():
        if sha256(p) != digest:
            raise ValueError('Existing evidence changed: ' + p)
    write_json(output / 'preservation_after.json', {'preserved_files': len(tracked)})
    stamp('prefix_correction_complete', new_rows=n * 6, reused_rows=n * 21)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-run', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    run(parser.parse_args())
