"""New full-visible-prompt labels, static controls and routers; no old labels reused.

Four frozen OPT replicas, original document splits, complete conditional joint
greedy sets and alternative branches. m0/m4/m6/m8, three seeds, budgets to24.
"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from check_full_context_v1 import checks
from full_context_core_v1 import (OBSERVATION_VERSION, FullContextRouter, candidate_costs,
    load_router, prompt_embeddings, prompt_only, route_prompt)
from full_context_data_v1 import Records, prepare
from full_context_experiment_v1 import (Workers, preserve, read_lines, resources, setup_worker,
    thread_record, verify_preserved, write)
from prompt_depth_core_v1 import document_loss, validate_steps
from prompt_fixed_core_v1 import best_candidate, continuation_logits, continuation_metrics, fixed_route
from stage1_benchmark_v2 import sha256, stamp
from train_prompt_depth_four_gpu_v4 import padded_training


def tensor_record(data, sample, device):
    array, p = data[sample]
    return torch.tensor(array, device=device, dtype=torch.long)[None], p


def search(model, ids, p, teacher, initial, maximum, cache):
    selected, steps = list(initial), []
    for number in range(len(selected)+1, maximum+1):
        key = tuple(sorted(selected))
        if key not in cache:
            cache[key] = candidate_costs(model, ids, p, selected,
                [i for i in range(2, 40) if i not in selected], teacher)
        costs = cache[key]
        chosen, value = best_candidate(costs)
        steps.append({'step': number, 'previous_skips': selected.copy(),
            'selected_layer': chosen, 'joint_continuation_kl': value, 'candidate_costs': costs})
        selected.append(chosen)
    record = {'initial_prefix': list(initial), 'ordered_skips': selected, 'steps': steps}
    validate_steps(record, initial, maximum)
    return record


def generate(rank, model, data, config, messages):
    out, device = Path(config['output']), next(model.parameters()).device
    arms = config['arms']
    handles = {(m, kind): (out / f'm{m:02d}_{kind}_{rank:03d}.jsonl').open('x')
        for m in arms for kind in ['labels', 'branches']}
    samples = sorted(config['train_ids']+config['validation_ids'])[rank::4]
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for offset, sample in enumerate(samples):
                ids, p = tensor_record(data, sample, device)
                teacher = F.log_softmax(continuation_logits(model, ids, p, []), -1)
                cache = {}
                for m in arms:
                    initial = config['global_path'][:m]
                    record = search(model, ids, p, teacher, initial, config['max_k'], cache)
                    record.update({'sample': sample, 'split': data.rows[sample]['split'], 'm': m,
                        'prompt_length': p, 'sequence_sha256': data.rows[sample]['sequence_sha256']})
                    handles[m, 'labels'].write(json.dumps(record)+'\n'); handles[m, 'labels'].flush()
                    if sample in set(config['train_ids']):
                        positions = [5, 9, 13, 17] if m == 0 else [m+1, m+5, m+9, m+13]
                        positions = [v for v in positions if v < config['max_k']]
                        length = positions[(sample+579) % len(positions)]
                        state = next(r for r in record['steps'] if r['step'] == length)
                        costs = sorted(state['candidate_costs'], key=lambda row: (row[1], row[0]))
                        alternative = 1 + ((sample*7+579) % 4)
                        prefix = record['ordered_skips'][:length-1]+[costs[alternative][0]]
                        if set(prefix) == set(record['ordered_skips'][:length]) or prefix[:m] != initial:
                            raise ValueError('Invalid conditional alternative branch')
                        branch = search(model, ids, p, teacher, prefix, config['max_k'], cache)
                        branch.update({'sample': sample, 'split': 'train', 'm': m, 'prompt_length': p,
                            'sequence_sha256': data.rows[sample]['sequence_sha256'],
                            'alternative_rank': alternative+1, 'expert_prefix_length': length})
                        handles[m, 'branches'].write(json.dumps(branch)+'\n'); handles[m, 'branches'].flush()
                messages.put({'kind': 'progress', 'phase': 'conditional_labels', 'rank': rank,
                    'completed': offset+1, 'total': len(samples), 'prompt_tokens': p,
                    'elapsed_seconds': time.monotonic()-started})
    finally:
        for stream in handles.values():
            stream.close()


def fit(rank, model, data, config, task, messages, embeddings):
    out, m, seed = Path(config['output']), task['m'], task['seed']
    device, samples = next(model.parameters()).device, config['train_ids']
    records = {r['sample']: r for path in out.glob(f'm{m:02d}_labels_*.jsonl') for r in read_lines(path)}
    branches = {r['sample']: r for path in out.glob(f'm{m:02d}_branches_*.jsonl') for r in read_lines(path)}
    tensors = padded_training(records, branches, samples)
    if not embeddings:
        # Detached CPU tensors are normal tensors so autograd may save inputs.
        for sample in samples:
            ids, p = tensor_record(data, sample, device)
            with torch.no_grad():
                embeddings[sample] = prompt_embeddings(model, prompt_only(ids, p)).detach().cpu()
    torch.manual_seed(seed)
    router = FullContextRouter(max_k=config['max_k']).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    updates = math.ceil(len(samples)/16); total = updates*config['epochs']
    def schedule(n):
        warm = max(1, total//20)
        if n < warm: return (n+1)/warm
        return .1+.9*.5*(1+math.cos(math.pi*(n-warm)/max(1, total-warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    budgets = [k for k in config['budgets'] if k >= m]
    best, checkpoints, history, teachers = {}, {}, [], {}
    with torch.inference_mode():
        for sample in config['selection_ids']:
            ids, p = tensor_record(data, sample, device)
            teachers[sample] = F.log_softmax(continuation_logits(model, ids, p, []), -1)
    for epoch in range(1, config['epochs']+1):
        router.train(); loss_sum = 0.
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed+epoch*10)).tolist()
        for offset in range(0, len(order), 16):
            group = order[offset:offset+16]
            optimizer.zero_grad(set_to_none=True)
            # Same effective document batch16; token-budget microbatches avoid
            # quadratic full-context encoder attention exhausting GPU memory.
            pending = sorted(group, key=lambda i: embeddings[samples[i]].shape[1])
            while pending:
                sub = [pending.pop(0)]
                while pending and (len(sub)+1)*embeddings[samples[pending[0]]].shape[1] <= 4096:
                    sub.append(pending.pop(0))
                length = max(embeddings[samples[i]].shape[1] for i in sub)
                emb = torch.zeros(len(sub), length, 5120, dtype=torch.float16, device=device)
                valid_tokens = torch.zeros(len(sub), length, dtype=torch.bool, device=device)
                for index, i in enumerate(sub):
                    value = embeddings[samples[i]]
                    emb[index, :value.shape[1]] = value[0].to(device)
                    valid_tokens[index, :value.shape[1]] = True
                states, targets, valid, chosen, present = [t[sub].to(device) for t in tensors]
                scores = router(emb, states, states.sum(-1).long()+1, valid_tokens)
                loss = document_loss(scores, targets, valid, chosen, present)*len(sub)/len(group)
                if not torch.isfinite(loss): raise ValueError('Nonfinite router loss')
                loss.backward(); loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
            optimizer.step(); scheduler.step()
        row = {'epoch': epoch, 'mean_loss': loss_sum/updates}
        if epoch == 1 or epoch % 4 == 0 or epoch == config['epochs']:
            router.eval(); metrics = {k: [] for k in budgets}
            with torch.inference_mode():
                for sample in config['selection_ids']:
                    ids, p = tensor_record(data, sample, device)
                    path = route_prompt(model, router, prompt_only(ids, p), config['max_k'], config['global_path'][:m])
                    for k in budgets:
                        value = continuation_metrics(continuation_logits(model, ids, p, path[:k]), teachers[sample], ids, p)
                        metrics[k].append(value['continuation_kl'])
            row['selection_kl_by_k'] = {str(k): float(np.mean(v)) for k, v in metrics.items()}
            improved = [k for k in budgets if row['selection_kl_by_k'][str(k)] < best.get(k, float('inf'))]
            if improved:
                path = out / f'private_m{m:02d}_seed{seed}_epoch{epoch:03d}.pth'
                with path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'm': m, 'seed': seed, 'epoch': epoch,
                        'max_k': config['max_k'], 'validation': row, 'global_path': config['global_path'],
                        'observation_version': OBSERVATION_VERSION, 'data_manifest_sha256': config['data_manifest_sha256']}, stream)
                for k in improved:
                    best[k] = row['selection_kl_by_k'][str(k)]
                    checkpoints[str(k)] = {'path': str(path), 'sha256': sha256(path), 'epoch': epoch,
                        'selection_continuation_kl': best[k]}
        history.append(row)
        messages.put({'kind': 'progress', 'phase': 'fit', 'rank': rank, 'm': m, 'seed': seed, **row})
    write(out / f'm{m:02d}_seed{seed}_history.json', history)
    write(out / f'm{m:02d}_seed{seed}_checkpoints.json', checkpoints)
    del router, optimizer, teachers, tensors
    gc.collect(); torch.cuda.empty_cache()


def gpu_probe(rank, model, data, config):
    sample = max(config['train_ids'], key=lambda s: data.rows[s]['prompt_length'])
    full, available = tensor_record(data, sample, next(model.parameters()).device)
    p = min([64, 129, 513, 2016][rank], available)
    ids = full[:, :p+32]
    result = []
    selected = [4, 9, 2, 12, 17, 7]
    candidates = [3, 20, 39]
    with torch.inference_mode():
        teacher = F.log_softmax(continuation_logits(model, ids, p, []), -1)
        for objective in ['kl', 'nll']:
            shared = candidate_costs(model, ids, p, selected, candidates, teacher, objective)
            serial = []
            for candidate in candidates:
                with fixed_route(model, selected+[candidate]):
                    # Independent normal CausalLM forward and full LM head.
                    logits = model(ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False).logits.float()
                if objective == 'kl':
                    lp = F.log_softmax(logits[:, p-1:], -1)
                    loss = (teacher.exp()*(teacher-lp)).sum(-1).mean().item()
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)).item()
                serial.append([candidate, loss])
            error = max(abs(a[1]-b[1]) for a, b in zip(shared, serial))
            if error > 5e-5: raise ValueError('Variable-length candidate scorer differs from full serial forward')
            result.append({'objective': objective, 'prompt_length': p, 'shared': shared, 'serial': serial,
                'max_abs_error': error, 'tolerance': 5e-5})
        emb = prompt_embeddings(model, prompt_only(ids, p))
        if emb.shape[1] != p: raise ValueError('GPU observation was truncated')
    return {'candidate_probes': result, 'observed_tokens': p, **thread_record()}


def worker(rank, config, inbox, messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
    data, embeddings = Records(config['data_run']), {}
    write(Path(config['output']) / f'worker_probe_{rank:03d}.json', gpu_probe(rank, model, data, config))
    messages.put({'kind': 'ready', 'rank': rank})
    while True:
        task = inbox.get()
        if task is None: break
        kind = task['kind']
        if kind == 'static':
            total = np.zeros(len(task['candidates']), dtype=np.float64)
            with torch.inference_mode():
                for sample in config['calibration_ids']:
                    ids, p = tensor_record(data, sample, next(model.parameters()).device)
                    teacher = F.log_softmax(continuation_logits(model, ids, p, []), -1) if task['objective'] == 'kl' else None
                    costs = candidate_costs(model, ids, p, task['selected'], task['candidates'], teacher, task['objective'])
                    total += np.asarray([v for _, v in costs])/len(config['calibration_ids'])
            messages.put({'kind': 'static_done', 'rank': rank, 'objective': task['objective'],
                'costs': [[i, float(v)] for i, v in zip(task['candidates'], total)]})
        elif kind == 'bi':
            scores, count = np.zeros(40), 0
            with torch.inference_mode():
                for sample in config['calibration_ids'][rank::4]:
                    ids, p = tensor_record(data, sample, next(model.parameters()).device)
                    hidden = model.model(ids[:, :-1], use_cache=False, output_hidden_states=True).hidden_states
                    for index in range(40):
                        # For the last block, decoder hidden_states[-1] includes
                        # final norm. Capture the block output directly instead.
                        if index == 39:
                            captured = {}
                            handle = model.model.decoder.layers[39].register_forward_hook(
                                lambda mod, args, value: captured.update(output=value[0]))
                            try: model.model(ids[:, :-1], use_cache=False)
                            finally: handle.remove()
                            after = captured['output']
                        else: after = hidden[index+1]
                        value = 1-F.cosine_similarity(hidden[index].float(), after.float(), dim=-1)
                        scores[index] += value.double().sum().item()
                    count += ids.size(1)-1
            messages.put({'kind': 'bi_done', 'rank': rank, 'sums': scores.tolist(), 'tokens': count})
        elif kind == 'prefix':
            config['global_path'] = task['path']
            messages.put({'kind': 'prefix_done', 'rank': rank})
        elif kind == 'labels':
            generate(rank, model, data, config, messages)
            messages.put({'kind': 'labels_done', 'rank': rank})
        elif kind == 'fit':
            fit(rank, model, data, config, task, messages, embeddings)
            messages.put({'kind': 'fit_done', 'rank': rank, 'm': task['m'], 'seed': task['seed']})
        else: raise ValueError('Unknown worker task')


def audit_labels(data, config, out):
    for m in config['arms']:
        for kind in ['labels', 'branches']:
            rows = [r for p in out.glob(f'm{m:02d}_{kind}_*.jsonl') for r in read_lines(p)]
            expected = config['train_ids'] + ([] if kind == 'branches' else config['validation_ids'])
            if len(rows) != len(expected) or {r['sample'] for r in rows} != set(expected):
                raise ValueError('Incomplete or duplicate labels')
            for row in rows:
                source = data.rows[row['sample']]
                assert row['split'] == source['split']
                assert row['prompt_length'] == source['prompt_length']
                assert row['sequence_sha256'] == source['sequence_sha256']
                assert row['initial_prefix'][:m] == config['global_path'][:m]
                validate_steps(row, row['initial_prefix'], config['max_k'])
    write(out / 'label_verification.json', {'passed': True, 'arms': config['arms'],
        'documents': len(config['train_ids'])+len(config['validation_ids']),
        'conditional_joint_candidates_complete': True, 'no_old_labels_reused': True, 'test_documents_used': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--data-run', type=Path)
    args = parser.parse_args(); resources()
    root, out = Path(__file__).resolve().parents[1], Path(os.environ['NEUROBYPASS_RUN_DIR'])
    tracked = preserve(root, out)
    write(out / 'cpu_regression.json', checks())
    if args.data_run is None:
        prepare(root, out, '/home/liushenyu/opt-13b'); data_run = out
    else: data_run = args.data_run.resolve()
    data = Records(data_run)
    old = json.loads((root / 'runs/stage6_depth_train_main_20261107_01/protocol.json').read_text())
    train, validation, selection = old['train_ids'], old['validation_ids'], old['selection_ids']
    if args.probe:
        def representative(samples):
            chosen = []
            for length in [64, 128, 512, 2016]:
                sample = min((s for s in samples if s not in chosen), key=lambda s: (abs(data.rows[s]['prompt_length']-length), s))
                chosen.append(sample)
            return sorted(chosen)
        train, validation = representative(train), representative(validation)
        selection = validation
    calibration = sorted(np.random.default_rng(579).choice(train, min(128, len(train)), replace=False).tolist())
    config = {'model': '/home/liushenyu/opt-13b', 'output': str(out), 'data_run': str(data_run),
        'data_manifest_sha256': sha256(data_run / 'data_manifest.json'), 'probe': args.probe,
        'train_ids': train, 'validation_ids': validation, 'selection_ids': selection,
        'calibration_ids': calibration, 'arms': [0, 4, 6, 8], 'seeds': [579] if args.probe else [579, 580, 581],
        'budgets': [10, 12] if args.probe else [6, 8, 10, 12, 16, 20, 24],
        'max_k': 12 if args.probe else 24, 'epochs': 1 if args.probe else 24,
        'observation_version': OBSERVATION_VERSION, 'observation': 'All actual prompt tokens; no128cap; token+position embeddings, right padding masked.',
        'labels': 'Full-model KL on32 future targets; condition every next-layer candidate on the entire selected set; no old labels reused.',
        'branches': 'One deterministic low-cost alternative per training document per m, greedily completed to maxK.',
        'training': 'Fresh parameters, effective document batch16; token-budget4096 microbatches, masked attention and pooling; soft CE+.25hard CE; AdamW2e-4.',
        'selection': 'Same32 original development document identities, variable prompt boundaries; perK best at epochs1/4/...24; no test.',
        'source_sha256': {str(p): sha256(p) for p in sorted((root / 'tools').glob('*full_context*_v1.py'))},
        'recovery_training': False, 'frozen_base': True}
    write(out / 'launch_protocol.json', config)
    paths = {'static_joint_kl': [], 'sleb_nll_opt': []}
    started = time.monotonic()
    with Workers(worker, config) as workers:
        workers.receive(4, 'ready')
        with (out / 'static_search_steps.jsonl').open('x') as stream:
            for step in range(1, config['max_k']+1):
                for rank in range(4):
                    objective = 'kl' if rank < 2 else 'nll'
                    selected = paths['static_joint_kl' if objective == 'kl' else 'sleb_nll_opt']
                    candidates = [i for i in range(2, 40) if i not in selected]
                    workers.inboxes[rank].put({'kind': 'static', 'objective': objective,
                        'selected': selected.copy(), 'candidates': candidates[rank%2::2]})
                replies = workers.receive(4, 'static_done')
                for objective, name in [('kl', 'static_joint_kl'), ('nll', 'sleb_nll_opt')]:
                    costs = sorted(c for row in replies if row['objective'] == objective for c in row['costs'])
                    layer, value = best_candidate(costs)
                    row = {'objective': objective, 'step': step, 'previous_skips': paths[name].copy(),
                        'selected_layer': layer, 'joint_cost': value, 'candidate_costs': costs}
                    stream.write(json.dumps(row)+'\n'); stream.flush(); paths[name].append(layer)
                stamp('full_context_static_step', step=step, elapsed_seconds=time.monotonic()-started)
        workers.send_all({'kind': 'bi'})
        parts = workers.receive(4, 'bi_done')
        tokens = sum(r['tokens'] for r in parts)
        bi = np.sum([r['sums'] for r in parts], axis=0)/tokens
        write(out / 'shortgpt_calibration.json', {'scores': bi.tolist(), 'tokens': tokens})
        paths['shortgpt_bi_opt'] = sorted(range(2, 40), key=lambda i: (bi[i], i))
        first = next(r for r in read_lines(out / 'static_search_steps.jsonl') if r['objective'] == 'nll')
        paths['shortened_ppl_opt'] = [i for i, _ in sorted(first['candidate_costs'], key=lambda row: (row[1], row[0]))]
        paths['shortened_ppl_plus'] = [i for i in paths['shortened_ppl_opt'] if i not in [0, 1, 2, 3, 38, 39]]
        paths['random_static_seed579'] = np.random.default_rng(579).permutation(range(2, 40)).tolist()
        write(out / 'static_paths.json', paths)
        config['global_path'] = paths['static_joint_kl']
        write(out / 'fixed_global_policy.json', {'ordered_skips': config['global_path'],
            'calibration_ids': calibration, 'split': 'train', 'validation_used': False})
        write(out / 'protocol.json', config)
        workers.send_all({'kind': 'prefix', 'path': config['global_path']}); workers.receive(4, 'prefix_done')
        workers.send_all({'kind': 'labels'}); workers.receive(4, 'labels_done')
        audit_labels(data, config, out)
        # All four arms occupy the four GPUs, then advance to the next seed.
        for seed in config['seeds']:
            for rank, m in enumerate(config['arms']):
                workers.inboxes[rank].put({'kind': 'fit', 'm': m, 'seed': seed})
            workers.receive(4, 'fit_done')
    checkpoints = []
    for m in config['arms']:
        for seed in config['seeds']:
            selected = json.loads((out / f'm{m:02d}_seed{seed}_checkpoints.json').read_text())
            for k in [v for v in config['budgets'] if v >= m]:
                if sha256(selected[str(k)]['path']) != selected[str(k)]['sha256']:
                    raise ValueError('Selected checkpoint digest mismatch')
                checkpoints.append({'m': m, 'seed': seed, 'k': k, **selected[str(k)]})
    verify_preserved(out, tracked)
    write(out / 'training_verification.json', {'passed': True, 'checkpoints': checkpoints,
        'test_documents_used': 0, 'elapsed_seconds': time.monotonic()-started})
    stamp('full_context_training_complete', probe=args.probe, checkpoints=len(checkpoints))


if __name__ == '__main__': main()
