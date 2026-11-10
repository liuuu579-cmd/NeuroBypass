"""Frozen prompt/path crossover: teacher KL and task scores on development data."""
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
import torch.nn.functional as F

from horizontal_quality_v3 import pin_threads, score_probe
from prefix_downstream_correction_core_v1 import policies
from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import fixed_route, raw_prompt_embeddings
from prompt_permutation_core_v1 import assignments, score_fields
from stage1_benchmark_v2 import sha256, stamp, write_json


def prepare(root, output, probe):
    old = root / 'runs/stage11_prefix_downstream_main_20261109_01'
    manifest = json.loads((old / 'data_manifest.json').read_text())
    private = old / 'private_benchmark_inputs.jsonl'
    if sha256(private) != manifest['private_input_sha256']:
        raise ValueError('Benchmark inputs changed')
    records = [json.loads(line) for line in private.open()]
    selected = []
    rng = np.random.default_rng(579)
    for task in sorted({r['task'] for r in records}):
        group = [r for r in records if r['task'] == task]
        indices = sorted(rng.choice(len(group), 4 if probe else 64, replace=False).tolist())
        selected.extend(group[i] for i in indices)
    source = root / 'runs/stage6_depth_train_main_20261107_01'
    config = json.loads((source / 'protocol.json').read_text())
    data_manifest = root / 'runs/stage3_prompt_fixed_pilot_20261104_01/data_manifest.json'
    c4_manifest = json.loads(data_manifest.read_text())
    if sha256(config['data_file']) != config['data_sha256']:
        raise ValueError('C4 inputs changed')
    data = np.load(config['data_file'], allow_pickle=False)
    c4_ids = sorted(set(config['validation_ids']) - set(config['selection_ids']))
    if probe:
        c4_ids = c4_ids[:4]
    for sample in c4_ids:
        ids = data[sample].tolist()
        selected.append({'task': 'c4_development', 'sample': sample, 'document': sample,
            'document_sha256': c4_manifest['records'][sample]['record_sha256'],
            'route_ids': ids[:128], 'sequence_ids': [ids], 'target_lengths': [32],
            'normalization_lengths': [32], 'truncated_context_tokens': [0], 'gold': None})
    selected.sort(key=lambda r: (r['task'], r['sample']))
    with (output / 'private_inputs.jsonl').open('x') as stream:
        for record in selected:
            stream.write(json.dumps(record) + '\n')
    return selected, [old / 'protocol.json', old / 'data_manifest.json', private,
        source / 'protocol.json', Path(config['data_file']), data_manifest]


def logs(model, ids, n, mask):
    x = torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device)[None]
    with fixed_route(model, mask):
        hidden = model.model(x[:, :-1], attention_mask=torch.ones_like(x[:, :-1]), use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)[:, -n:].float()
    lp = F.log_softmax(logits, -1)
    score = lp.gather(-1, x[:, -n:, None]).double().sum().item()
    return lp, score


def worker(rank, config, messages):
    try:
        cpu = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))[rank*10:(rank+1)*10]
        pin_threads(cpu); torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
        routers = {}
        for policy in config['policies']:
            path = policy['checkpoint']['path']
            if path not in routers:
                router = DepthRouter().to(f'cuda:{rank}').eval().requires_grad_(False)
                router.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
                routers[path] = router
        inputs = [json.loads(line) for line in (Path(config['output']) / 'private_inputs.jsonl').open()]
        registry = {}
        with torch.inference_mode():
            for record in inputs:
                ids = torch.tensor(record['route_ids'], dtype=torch.long, device=f'cuda:{rank}')[None]
                emb, encoded = raw_prompt_embeddings(model, ids), {}
                for policy in config['policies']:
                    path = policy['checkpoint']['path']
                    if path not in encoded:
                        encoded[path] = routers[path].encode(emb)
                    ordered = routers[path].route_encoded(encoded[path], policy['k'], policy['prefix'])[0].tolist()
                    if ordered[:6] != config['global_path'][:6]:
                        raise ValueError('Route prefix differs from training')
                    registry[record['task'], record['sample'], policy['seed'], policy['k']] = sorted(ordered)
        route_rows = [{'task': task, 'sample': sample, 'seed': seed, 'k': k, 'skipped_layers': mask}
            for (task, sample, seed, k), mask in sorted(registry.items())]
        write_json(Path(config['output']) / f'routes_{rank:03d}.json', route_rows)
        donors = {(r['task'], r['sample'], r['permutation_seed']): r['donor_sample']
            for r in json.loads((Path(config['output']) / 'assignments.json').read_text())}
        local = inputs[rank::4]
        # Every worker needs the complete route registry because a shuffled donor
        # can belong to another rank's input shard; records remain read-only.
        all_inputs = inputs
        probes = []
        with torch.inference_mode():
            for task in sorted({r['task'] for r in local}):
                record = next(r for r in local if r['task'] == task)
                for mask in [[], config['global_path'][:10]]:
                    probes.append({'task': task, **score_probe(model, record, mask)})
        write_json(Path(config['output']) / f'worker_probe_{rank:03d}.json', {'rank': rank,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': 10,
            'thread_cpu_ids': {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()},
            'likelihood_probes': probes})
        with (Path(config['output']) / f'diagnostics_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
            for index, record in enumerate(local):
                teacher = [logs(model, ids, n, []) for ids, n in zip(record['sequence_ids'], record['target_lengths'])]
                cache = {}
                for policy in config['policies']:
                    masks = [('matched', record['sample'], registry[record['task'], record['sample'], policy['seed'], policy['k']])]
                    for seed in [1701, 1702, 1703]:
                        donor = donors[record['task'], record['sample'], seed]
                        masks.append((f'shuffled_{seed}', donor, registry[record['task'], donor, policy['seed'], policy['k']]))
                    masks.append(('static_joint_kl', None, config['global_path'][:policy['k']]))
                    for condition, donor, mask in masks:
                        mask = tuple(sorted(mask))
                        if mask not in cache:
                            scores, kl = [], 0
                            for ids, n, (lp, full_score) in zip(record['sequence_ids'], record['target_lengths'], teacher):
                                lq, score = logs(model, ids, n, mask)
                                kl += (lp.exp() * (lp-lq)).sum(-1).double().sum().item()
                                scores.append(score)
                            cache[mask] = {'scores': scores, 'teacher_kl_sum': kl,
                                'target_tokens': sum(record['target_lengths']),
                                'teacher_scores': [item[1] for item in teacher]}
                        row = {'task': record['task'], 'sample': record['sample'], 'document': record.get('document'),
                            'document_sha256': record['document_sha256'], 'seed': policy['seed'], 'k': policy['k'],
                            'condition': condition, 'donor_sample': donor, 'skipped_layers': list(mask),
                            **cache[mask], **score_fields(record, cache[mask]['scores'])}
                        stream.write(json.dumps(row) + '\n')
                stream.flush()
                if index % 8 == 0 or index+1 == len(local):
                    messages.put({'kind': 'progress', 'rank': rank, 'completed': index+1, 'total': len(local)})
        messages.put({'kind': 'done', 'rank': rank})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if os.environ.get('NEUROBYPASS_GPU_COUNT') != '4':
        raise ValueError('Use four-GPU resource launcher')
    root, output = Path(__file__).resolve().parents[1], Path(os.environ['NEUROBYPASS_RUN_DIR'])
    records, provenance = prepare(root, output, args.probe)
    path, selected = policies(root)
    selected = [p for p in selected if p['name'] == 'm06']
    write_json(output / 'assignments.json', assignments(records))
    config = {'output': str(output), 'model': '/home/liushenyu/opt-13b', 'policies': selected,
        'global_path': path, 'probe': args.probe, 'input_records': len(records),
        'private_input_sha256': sha256(output / 'private_inputs.jsonl'),
        'assignments_sha256': sha256(output / 'assignments.json'),
        'source_sha256': {str(p): sha256(p) for p in [Path(__file__),
            root / 'tools/prompt_permutation_core_v1.py', root / 'tools/horizontal_quality_v3.py',
            root / 'tools/prefix_downstream_correction_core_v1.py', root / 'tools/prompt_fixed_core_v1.py']},
        'provenance_sha256': {str(p): sha256(p) for p in provenance},
        'selection': 'Fixed random seed579 subset; no outcome-based sample/checkpoint/mask selection.',
        'metric': 'Teacher full-vocabulary forward KL on all target tokens; all MC options, no gold-dependent path selection.',
        'limits': ['Development diagnosis on previously reviewed tasks, not an independent final test.',
            'C4 uses68 non-checkpoint-selection development documents; benchmark subsets64 each.',
            'Candidate minimum KL sees future targets and is not deployable or globally optimal.']}
    write_json(output / 'protocol.json', config)
    tracked = {str(p): sha256(p) for p in root.rglob('*') if p.is_file() and not p.is_relative_to(output)
        and '.git' not in p.parts and p.suffix in {'.py', '.json', '.jsonl', '.md', '.csv', '.png'}}
    for p in root.parent.glob('*.py'):
        tracked[str(p)] = sha256(p)
    write_json(output / 'preservation_before.json', tracked)
    context = mp.get_context('spawn'); messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, config, messages)) for rank in range(4)]
    done = set()
    try:
        for p in processes:
            p.start()
        pin_threads(list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))[40:42])
        while len(done) < 4:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Route-fidelity worker exited early')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            if item['kind'] == 'done':
                done.add(item['rank'])
            else:
                stamp('route_fidelity_progress', **{k: v for k, v in item.items() if k != 'kind'})
        for p in processes:
            p.join(timeout=30)
            if p.exitcode != 0:
                raise RuntimeError('Route-fidelity worker failed')
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate(); p.join(timeout=10)
            if p.is_alive():
                p.kill(); p.join(timeout=5)
    count = sum(1 for p in output.glob('diagnostics_*.jsonl') for _ in p.open())
    if count != len(records)*30:
        raise ValueError('Incomplete fidelity crossover')
    for p, expected in tracked.items():
        if sha256(p) != expected:
            raise ValueError('Existing evidence changed: ' + p)
    write_json(output / 'preservation_after.json', {'preserved_files': len(tracked)})
    stamp('route_fidelity_complete', records=len(records), rows=count)


if __name__ == '__main__':
    main()
