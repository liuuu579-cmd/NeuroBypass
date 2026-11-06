"""Diagnose cached-versus-recomputed distributions without hiding FP16 failures.

FP32 confirmation converts only retained blocks and shared decoder modules;
skipped blocks are placed on CPU to fit the GPU. Original files are read-only.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import cached_continuation_logits, continuation_logits
from stage1_benchmark_v2 import sha256, write_json


def distributions(model, ids, route):
    expected = continuation_logits(model, ids, 128, route)
    actual, session = cached_continuation_logits(model, ids, 128, route)
    reference, observed = F.log_softmax(expected, dim=-1), F.log_softmax(actual, dim=-1)
    kl = (reference.exp() * (reference - observed)).sum(-1)
    expected_bytes = 2 * (40 - len(route)) * 159 * 5120 * next(model.model.decoder.embed_tokens.parameters()).element_size()
    if session.cache_bytes() != expected_bytes:
        raise ValueError('Cache storage mismatch')
    return {'route': sorted(route), 'precision': str(next(model.model.decoder.embed_tokens.parameters()).dtype),
        'mean_absolute_position_kl': kl.abs().mean().item(), 'max_absolute_position_kl': kl.abs().max().item(),
        'argmax_agreement': (expected.argmax(-1) == actual.argmax(-1)).float().mean().item(),
        'mean_logit_abs_error': (expected - actual).abs().mean().item(),
        'max_logit_abs_error': (expected - actual).abs().max().item(),
        'cache_bytes': session.cache_bytes(), 'verified_all_layer_cache_lengths': True,
        'passes_original_distribution_tolerance': kl.abs().mean().item() <= 1e-4 and kl.abs().max().item() <= 1e-3}


def retained_precision(model, route, dtype, device):
    """Move skipped weights off GPU; keep every retained original layer index."""
    decoder = model.model.decoder
    for index in route:
        decoder.layers[index].to(device='cpu')
    torch.cuda.empty_cache()
    for name, module in decoder.named_children():
        if name != 'layers':
            module.to(device=device, dtype=dtype)
    model.lm_head.to(device=device, dtype=dtype)
    for index, layer in enumerate(decoder.layers):
        if index not in route:
            layer.to(device=device, dtype=dtype)
    torch.cuda.empty_cache()


def audit_route(model, ids, route):
    half = distributions(model, ids, route)
    if half['passes_original_distribution_tolerance']:
        return {'fp16': half, 'fp32': None, 'outcome': 'fp16_cache_consistent'}
    retained_precision(model, route, torch.float32, ids.device)
    full = distributions(model, ids, route)
    result = {'fp16': half, 'fp32': full,
        'outcome': 'fp16_numerical_difference_fp32_consistent' if full['passes_original_distribution_tolerance'] else 'unresolved_cache_difference'}
    if not full['passes_original_distribution_tolerance']:
        raise ValueError('FP32 cache difference remains: ' + json.dumps(result))
    return result


def worker(rank, config, messages):
    try:
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained('/home/liushenyu/opt-13b', torch_dtype=torch.float16,
            local_files_only=True, attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False)
        data = np.load(config['data_file'], allow_pickle=False)
        ids = torch.tensor(data[config['pilot_ids'][0]], dtype=torch.long, device=device)[None]
        k = [6, 24][rank]
        with torch.inference_mode():
            result = audit_route(model, ids, config['path'][:k])
        write_json(Path(os.environ['NEUROBYPASS_RUN_DIR']) / f'cache_k{k:02d}.json', result)
        messages.put({'kind': 'done', 'k': k, 'result': result,
            'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads()})
    except BaseException:
        messages.put({'kind': 'error', 'traceback': traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-run', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 2:
        parser.error('Use the two-GPU resource launcher')
    config = json.loads((args.failed_run / 'protocol.json').read_text())
    config['path'] = json.loads((args.failed_run / 'static_paths.json').read_text())['sleb_nll_opt']
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    write_json(output / 'protocol.json', {'failed_run': str(args.failed_run.resolve()),
        'data_sha256': sha256(config['data_file']), 'source_sha256': sha256(__file__),
        'purpose': 'Diagnose numerical versus structural cache disagreement; original tolerances unchanged.'})
    context = mp.get_context('spawn')
    messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, config, messages)) for rank in range(2)]
    results = []
    try:
        for process in processes:
            process.start()
        while len(results) < 2:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Workers exited without complete diagnostics')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            results.append(item)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Diagnostic worker failed')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    write_json(output / 'summary.json', {'diagnostics': results})
    print(json.dumps({'diagnostics': results}), flush=True)


if __name__ == '__main__':
    main()
