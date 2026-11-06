"""Pilot paper-derived depth baselines and joint search through 24 skipped layers.

Run through the two-GPU resource launcher. No training, old-test evaluation,
third-party code execution, original-file changes or speed claims are made.
Candidate batching is an offline search device only.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_prompt_router_v2 import derangement, paired_summary
from prompt_fixed_candidate_batch_v1 import candidate_routes
from prompt_fixed_core_v1 import (SequentialGreedyRouter, best_candidate,
    continuation_logits, continuation_metrics, fixed_route, raw_prompt_embeddings)
from prompt_fixed_experiment_v2 import cache_probe
from stage1_benchmark_v2 import sha256, stamp, write_json


def block_influence(before, after):
    return (1 - F.cosine_similarity(before.float(), after.float(), dim=-1)).sum().item(), before.shape[0] * before.shape[1]


def importance_order(costs, protected=(0, 1)):
    return [int(layer) for layer, cost in sorted(costs, key=lambda row: (row[1], row[0]))
            if layer not in set(protected)]


def candidate_costs(model, ids, selected, candidates, objective, teacher=None, batch_size=8):
    """Return the cost of each entire selected-set-plus-candidate combination."""
    costs = []
    for offset in range(0, len(candidates), batch_size):
        group = candidates[offset:offset + batch_size]
        batch = ids[:, :-1].expand(len(group), -1).contiguous()
        with candidate_routes(model, selected, group):
            hidden = model.model(batch, attention_mask=torch.ones_like(batch), use_cache=False).last_hidden_state
            logits = model.lm_head(hidden if objective == 'nll' else hidden[:, 127:]).float()
        if objective == 'nll':
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].expand(len(group), -1).reshape(-1), reduction='none').reshape(len(group), -1).mean(-1)
        elif objective == 'kl':
            logs = F.log_softmax(logits, dim=-1)
            losses = (teacher.exp() * (teacher - logs)).sum(-1).mean(-1)
        else:
            raise ValueError('Unknown search objective')
        if not torch.isfinite(losses).all():
            raise ValueError('Nonfinite candidate loss')
        costs.extend([[layer, loss.item()] for layer, loss in zip(group, losses)])
    return costs


def collect_bi(model, data, samples, device):
    sums, counts = [0.] * 40, [0] * 40
    handles = []
    def hook(index):
        def capture(module, inputs, output):
            total, count = block_influence(inputs[0], output[0])
            sums[index] += total
            counts[index] += count
        return capture
    try:
        for index, layer in enumerate(model.model.decoder.layers):
            handles.append(layer.register_forward_hook(hook(index)))
        for sample in samples:
            ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
            model.model(ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(count != len(samples) * 160 for count in counts):
        raise ValueError('BI observation count mismatch')
    scores = [[i, sums[i] / counts[i]] for i in range(40)]
    return {'scores': scores, 'tokens_per_layer': counts[0], 'ordered_skips': importance_order(scores)}


def greedy(model, data, samples, maximum, objective, output_path, messages, rank):
    device = next(model.parameters()).device
    ids_list = [torch.tensor(data[s], dtype=torch.long, device=device)[None] for s in samples]
    teachers = [F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
                for ids in ids_list] if objective == 'kl' else [None] * len(ids_list)
    selected, steps = [], []
    with output_path.open('x') as stream:
        for step in range(1, maximum + 1):
            candidates = [i for i in range(2, 40) if i not in selected]
            totals = defaultdict(float)
            for ids, teacher in zip(ids_list, teachers):
                for layer, cost in candidate_costs(model, ids, selected, candidates, objective, teacher):
                    totals[layer] += cost / len(ids_list)
            costs = [[layer, totals[layer]] for layer in candidates]
            layer, cost = best_candidate(costs)
            row = {'step': step, 'previous_skips': selected.copy(), 'selected_layer': layer,
                'objective': objective, 'joint_cost': cost, 'candidate_costs': costs,
                'source_samples': samples}
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            selected.append(layer)
            steps.append(row)
            if step == 1 or step % 4 == 0 or step == maximum:
                messages.put({'kind': 'progress', 'rank': rank, 'objective': objective,
                    'documents': len(samples), 'step': step, 'max_k': maximum, 'cost': cost})
    return {'ordered_skips': selected, 'steps': steps}


def tables(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['k'], row['policy']].append(row)
    quality = []
    for (k, policy), group in sorted(groups.items()):
        nll = float(np.mean([r['continuation_nll'] for r in group]))
        quality.append({'k': k, 'depth_fraction': k / 40, 'policy': policy, 'documents': len(group),
            'continuation_kl': float(np.mean([r['continuation_kl'] for r in group])),
            'continuation_nll': nll, 'perplexity': math.exp(nll),
            'unique_masks': len({tuple(r['skipped_layers']) for r in group})})
    keyed = {(r['sample'], r['k'], r['policy']): r for r in rows}
    comparisons = []
    for k, policy in sorted(groups):
        if policy != 'oracle_joint_kl':
            continue
        samples = [r['sample'] for r in groups[k, policy]]
        for reference in ['static_joint_kl', 'sleb_nll_opt', 'shortgpt_bi_opt', 'shortened_ppl_opt', 'oracle_shuffled']:
            for metric in ['continuation_kl', 'continuation_nll']:
                comparisons.append({'k': k, 'policy': policy, 'reference': reference, 'metric': metric,
                    **paired_summary([keyed[s, k, policy][metric] - keyed[s, k, reference][metric] for s in samples])})
    return {'quality': quality, 'paired_comparisons': comparisons}


def worker(rank, config, inbox, messages):
    try:
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config['base_model'], torch_dtype=torch.float16,
            local_files_only=True, attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False)
        data = np.load(config['data_file'], allow_pickle=False)
        output = Path(config['output'])
        with torch.inference_mode():
            # Check heterogeneous candidate batching against real serial bypasses.
            ids = torch.tensor(data[config['calibration_ids'][0]], dtype=torch.long, device=device)[None]
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
            probe = {}
            for objective in ['nll', 'kl']:
                batch = candidate_costs(model, ids, [2, 5], [3, 4], objective, teacher, 2)
                serial = []
                for layer in [3, 4]:
                    if objective == 'kl':
                        logs = F.log_softmax(continuation_logits(model, ids, 128, [2, 5, layer]), dim=-1)
                        value = (teacher.exp() * (teacher - logs)).sum(-1).mean().item()
                    else:
                        with fixed_route(model, [2, 5, layer]):
                            hidden = model.model(ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False).last_hidden_state
                            logits = model.lm_head(hidden).float()
                        value = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)).item()
                    serial.append([layer, value])
                error = max(abs(a[1] - b[1]) for a, b in zip(batch, serial))
                if error > .002:
                    raise ValueError('Offline candidate batching differs materially from serial bypass')
                probe[objective] = {'batched': batch, 'serial': serial, 'max_abs_error': error, 'tolerance': .002}
            write_json(output / f'candidate_probe_{rank:03d}.json', probe)
            if rank == 0:
                bi = collect_bi(model, data, config['calibration_ids'], device)
                write_json(output / 'shortgpt_calibration.json', bi)
            objective = ['nll', 'kl'][rank]
            path = greedy(model, data, config['calibration_ids'], max(config['budgets']), objective,
                output / f'calibration_{objective}_steps.jsonl', messages, rank)
            write_json(output / f'calibration_{objective}_path.json', path)
            messages.put({'kind': 'calibration_done', 'rank': rank})
            task = inbox.get()
            if task['kind'] != 'oracles':
                raise ValueError('Unexpected worker phase')
            with (output / f'oracle_paths_{rank:03d}.jsonl').open('x') as stream:
                for offset, sample in enumerate(task['samples']):
                    result = greedy(model, data, [sample], max(config['budgets']), 'kl',
                        output / f'oracle_sample{sample:04d}_steps.jsonl', messages, rank)
                    stream.write(json.dumps({'sample': sample, 'split': 'pilot', **result}) + '\n')
                    stream.flush()
                    messages.put({'kind': 'progress', 'rank': rank, 'phase': 'oracle',
                        'finished_documents': offset + 1, 'total': len(task['samples'])})
            messages.put({'kind': 'oracles_done', 'rank': rank})
            task = inbox.get()
            if task['kind'] != 'evaluate':
                raise ValueError('Unexpected worker phase')
            router = SequentialGreedyRouter().to(device).eval()
            router.load_state_dict(torch.load(config['router_checkpoint'], map_location='cpu', weights_only=True)['state_dict'])
            with (output / f'quality_{rank:03d}.jsonl').open('x') as stream:
                for sample in task['samples']:
                    ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                    teacher_logits = continuation_logits(model, ids, 128, [])
                    teacher = F.log_softmax(teacher_logits, dim=-1)
                    policies = [(0, 'full', [])]
                    for k in config['budgets']:
                        for name, ordered in task['static_paths'].items():
                            policies.append((k, name, ordered[:k]))
                        policies += [(k, 'oracle_joint_kl', task['oracles'][sample][:k]),
                            (k, 'oracle_shuffled', task['oracles'][task['donors'][sample]][:k])]
                    if 6 in config['budgets']:
                        policies += [(6, 'router_trained_k6', router.route(raw_prompt_embeddings(model, ids[:, :128]), 6)[0].tolist()),
                            (6, 'learned_static_k6', [4, 8, 2, 10, 16, 14])]
                    for k, name, skips in policies:
                        logits = teacher_logits if k == 0 else continuation_logits(model, ids, 128, skips)
                        row = {'sample': sample, 'split': 'pilot', 'k': k, 'policy': name,
                            'skipped_layers': sorted(skips), **continuation_metrics(logits, teacher, ids, 128)}
                        if not all(math.isfinite(row[m]) for m in ['continuation_kl', 'continuation_nll']):
                            raise ValueError('Nonfinite evaluation metrics')
                        stream.write(json.dumps(row) + '\n')
                    stream.flush()
            if rank == 0:
                sample = config['pilot_ids'][0]
                ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                probes = [cache_probe(model, ids, 128, task['static_paths']['sleb_nll_opt'][:k])
                    for k in [6, max(config['budgets'])]]
                write_json(output / 'cache_probes.json', probes)
            messages.put({'kind': 'evaluation_done', 'rank': rank, 'cpu_ids': sorted(os.sched_getaffinity(0)),
                'threads': torch.get_num_threads()})
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def receive(messages, processes, expected):
    found = []
    while len(found) < 2:
        try:
            item = messages.get(timeout=5)
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                raise RuntimeError('Workers exited without complete results')
            continue
        if item['kind'] == 'error':
            raise RuntimeError(item['traceback'])
        if item['kind'] == expected:
            found.append(item)
        elif item['kind'] == 'progress':
            stamp('depth_pilot_progress', **{k: v for k, v in item.items() if k != 'kind'})
        else:
            raise ValueError('Unexpected message ' + str(item))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', default='/home/liushenyu/opt-13b')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 2:
        parser.error('Use the two-GPU resource launcher')
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    data_run = root / 'runs/stage3_prompt_fixed_pilot_20261104_01'
    manifest = json.loads((data_run / 'data_manifest.json').read_text())
    data_file = data_run / 'private_prompt_continuation_ids.npy'
    if sha256(data_file) != manifest['data_sha256']:
        raise ValueError('Input data changed')
    calibration = [r['sample'] for r in manifest['records'] if r['split'] == 'calibration']
    pilot = [r['sample'] for r in manifest['records'] if r['split'] == 'pilot']
    metadata = json.loads((root / 'runs/stage4_branch_main_20261105_01/summary.json').read_text())
    chosen = next(r for r in metadata['arms'] if r['arm'] == 'augmented')
    if sha256(chosen['best_checkpoint']) != chosen['best_checkpoint_sha256']:
        raise ValueError('Frozen K6 router changed')
    if args.probe:
        calibration, pilot = calibration[:2], pilot[:2]
    config = {'output': str(output), 'base_model': args.base_model, 'data_file': str(data_file),
        'data_sha256': manifest['data_sha256'], 'calibration_ids': calibration, 'pilot_ids': pilot,
        'budgets': [6, 12, 24] if args.probe else [6, 8, 10, 12, 16, 20, 24],
        'router_checkpoint': chosen['best_checkpoint'], 'router_sha256': chosen['best_checkpoint_sha256']}
    tracked = {str(p.resolve()): sha256(p) for p in root.rglob('*') if p.is_file()
        and not p.is_relative_to(output) and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv'} and '.git' not in p.parts}
    for path in root.parent.glob('*.py'):
        tracked[str(path.resolve())] = sha256(path)
    write_json(output / 'preservation_before.json', tracked)
    write_json(output / 'protocol.json', {**config, 'probe': args.probe, 'old_test_used': False,
        'trained_new_router': False, 'protected_layers': [0, 1], 'source_sha256': sha256(__file__),
        'static_calibration': '16 existing calibration documents, all159 next-token NLL for SLEB/Shortened; all160 hidden-state positions for BI; continuation32 KL for matched static greedy.',
        'oracle': 'Offline search sees each pilot true continuation; not deployable, not globally optimal, not a learned-router result or prompt-predictable bound.',
        'adaptations': ['OPT implementation with common protected0,1 and C4 calibration, not native paper data/model recipes.',
            'shortened_ppl_plus additionally preserves0,1,2,3,38,39 following the official plus heuristic.',
            'SLEB original default protects first and last surviving blocks; this common-mask adaptation instead protects fixed0,1.',
            'No Taylor+, recovery fine-tuning, BlockPruner or SkipDecode execution in this pilot.',
            'Candidate batching is offline search acceleration only; no deployment speed/memory measurement.'],
        'sources': {'ShortGPT': 'https://arxiv.org/html/2403.03853v2',
            'ShortenedLLaMA': 'https://github.com/Nota-NetsPresso/shortened-llm/tree/4031a9212aa0dbeec8f2ad504af9346c758b0147',
            'SLEB': 'https://github.com/jiwonsong-dev/SLEB/tree/d07129af60520e751087b8abb04a268a3c7ec861'}})
    context = mp.get_context('spawn')
    messages = context.Queue()
    inboxes = [context.Queue() for _ in range(2)]
    processes = [context.Process(target=worker, args=(r, config, inboxes[r], messages)) for r in range(2)]
    try:
        for process in processes:
            process.start()
        receive(messages, processes, 'calibration_done')
        for rank in range(2):
            inboxes[rank].put({'kind': 'oracles', 'samples': pilot[rank::2]})
        receive(messages, processes, 'oracles_done')
        nll = json.loads((output / 'calibration_nll_path.json').read_text())
        kl = json.loads((output / 'calibration_kl_path.json').read_text())
        bi = json.loads((output / 'shortgpt_calibration.json').read_text())
        static_paths = {'sleb_nll_opt': nll['ordered_skips'], 'static_joint_kl': kl['ordered_skips'],
            'shortgpt_bi_opt': bi['ordered_skips'], 'shortened_ppl_opt': importance_order(nll['steps'][0]['candidate_costs']),
            'shortened_ppl_plus': importance_order(nll['steps'][0]['candidate_costs'], (0, 1, 2, 3, 38, 39))}
        for seed in [579, 580, 581]:
            path = list(range(2, 40))
            random.Random(seed).shuffle(path)
            static_paths[f'random_static_seed{seed}'] = path
        write_json(output / 'static_paths.json', static_paths)
        oracles = {r['sample']: r['ordered_skips'] for p in output.glob('oracle_paths_*.jsonl')
            for r in map(json.loads, p.read_text().splitlines())}
        donors = derangement(pilot, 579)
        for rank in range(2):
            inboxes[rank].put({'kind': 'evaluate', 'samples': pilot[rank::2], 'oracles': oracles,
                'static_paths': static_paths, 'donors': donors})
        workers = receive(messages, processes, 'evaluation_done')
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Worker did not exit cleanly')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    rows = [json.loads(line) for p in output.glob('quality_*.jsonl') for line in p.read_text().splitlines()]
    write_json(output / 'summary.json', {**tables(rows), 'elapsed_seconds': time.monotonic() - started,
        'probe': args.probe, 'old_test_used': False, 'new_router_trained': False})
    write_json(output / 'workers.json', workers)
    for path, digest in tracked.items():
        if sha256(path) != digest:
            raise ValueError('Preexisting file changed: ' + path)
    if sha256(data_file) != manifest['data_sha256']:
        raise ValueError('Private data changed')
    write_json(output / 'preservation_after.json', {'verified_existing_files': len(tracked),
        'data_sha256': sha256(data_file), 'router_sha256': sha256(config['router_checkpoint'])})
    stamp('depth_pilot_complete', elapsed_seconds=time.monotonic() - started, quality_rows=len(rows))


if __name__ == '__main__':
    main()
