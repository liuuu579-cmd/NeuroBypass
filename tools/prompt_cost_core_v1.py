"""Cached generation timing and runtime-only physical static layer removal."""
import gc
import hashlib
import time

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from prompt_fixed_core_v1 import bind_cache, fixed_route, raw_prompt_embeddings


def parameter_bytes(module):
    return sum(p.numel()*p.element_size() for p in module.parameters() if p.device.type == 'cuda')


def cache_bytes(cache):
    return sum(t.numel()*t.element_size() for layer in cache.layers
        for t in [layer.keys, layer.values] if t is not None)


class Layout:
    """Keep removed blocks on CPU solely to restore the next benchmark policy.

    They are absent from the static decoder ModuleList and CUDA allocation.
    Moving layouts is preparation, excluded from steady-state request timing.
    Original source and checkpoint files are never written.
    """
    def __init__(self, model):
        self.model = model
        self.blocks = list(model.model.decoder.layers)
        self.device = next(model.parameters()).device

    def activate(self, policy):
        physical = set(policy['skips']) if policy['deployment'] == 'physical' else set()
        for index, layer in enumerate(self.blocks):
            layer.to('cpu' if index in physical else self.device)
        kept = [layer for index, layer in enumerate(self.blocks) if index not in physical]
        self.model.model.decoder.layers = torch.nn.ModuleList(kept)
        for index, layer in enumerate(kept): layer.self_attn.layer_idx = index
        self.model.config.num_hidden_layers = len(kept)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def choose(model, ids, policy, router):
    if router is None: return policy['skips']
    observation = ids[:, :128]
    return router.route(raw_prompt_embeddings(model, observation), policy['k'], policy['prefix'])[0].tolist()


def trial(model, ids, policy, router, new_tokens):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    start = time.perf_counter()
    skips = choose(model, ids, policy, router)
    torch.cuda.synchronize()
    routed = time.perf_counter()
    actual_skips = [] if policy['deployment'] == 'physical' else skips
    cache = DynamicCache(); bind_cache(cache, actual_skips)
    generated = []
    with fixed_route(model, actual_skips):
        out = model(ids, attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token); del out
        torch.cuda.synchronize()
        first = time.perf_counter()
        for step in range(1, new_tokens):
            out = model(token, attention_mask=torch.ones((1, ids.size(1)+step), dtype=torch.long, device=ids.device),
                past_key_values=cache, use_cache=True)
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token); del out
        torch.cuda.synchronize()
        end = time.perf_counter()
    length = ids.size(1) + new_tokens - 1
    for index in range(len(model.model.decoder.layers)):
        assert cache.get_seq_length(index) == (0 if index in actual_skips else length)
    expected = 2*(40-policy['k'])*length*5120*2
    assert cache_bytes(cache) == expected
    # Provenance guard must reject switching a populated cache to a new path.
    try:
        bind_cache(cache, tuple(actual_skips) + (999,))
    except ValueError: pass
    else: raise ValueError('Cache provenance failed to reject a route change')
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    digest = hashlib.sha256(torch.cat(generated, dim=1).cpu().numpy().tobytes()).hexdigest()
    return {'skipped_layers': sorted(skips), 'route_seconds': routed-start,
        'prefill_seconds': first-routed, 'ttft_seconds': first-start,
        'decode_seconds': end-first, 'total_seconds': end-start,
        'decode_tokens_per_second': (new_tokens-1)/(end-first),
        'end_to_end_tokens_per_second': new_tokens/(end-start),
        'new_tokens': new_tokens, 'cached_tokens': length, 'kv_bytes': expected,
        'resident_allocated_bytes': resident, 'peak_allocated_bytes': allocated,
        'peak_reserved_bytes': reserved, 'model_cuda_parameter_bytes': parameter_bytes(model),
        'router_cuda_parameter_bytes': parameter_bytes(router) if router is not None else 0,
        'generated_tokens_sha256': digest, 'cache_layer_lengths_verified': True,
        'cache_route_change_rejected': True}


def cached_logits(model, ids, skips, prompt_length=128):
    pieces, cache = [], DynamicCache()
    with fixed_route(model, skips):
        first = model(ids[:, :prompt_length], attention_mask=torch.ones_like(ids[:, :prompt_length]), past_key_values=cache, use_cache=True)
        pieces.append(first.logits[:, -1:]); del first
        for position in range(prompt_length, ids.size(1)-1):
            out = model(ids[:, position:position+1], attention_mask=torch.ones_like(ids[:, :position+1]), past_key_values=cache, use_cache=True)
            pieces.append(out.logits); del out
    return torch.cat(pieces, 1).float()


def cache_probe(model, layout, ids, policy, router):
    layout.activate({**policy, 'deployment': 'bypass'})
    route = choose(model, ids[:, :128], policy, router)
    actual = cached_logits(model, ids, route)
    with fixed_route(model, route): reference = model(ids[:, :-1], use_cache=False).logits[:, 127:].float()
    lp, lq = F.log_softmax(reference, -1), F.log_softmax(actual, -1)
    error = (lp.exp()*(lp-lq)).sum(-1).abs()
    result = {'policy': policy['name'], 'seed': policy['seed'], 'k': policy['k'],
        'skipped_layers': sorted(route), 'mean_absolute_position_kl': error.mean().item(),
        'max_absolute_position_kl': error.max().item(),
        'mean_kl_tolerance': 1e-4, 'max_kl_tolerance': 1e-3,
        'cache_within_tolerance': error.mean().item() <= 1e-4 and error.max().item() <= 1e-3}
    del reference, lp, lq, error
    if policy['kind'] == 'static' and route:
        layout.activate({**policy, 'deployment': 'physical'})
        physical = cached_logits(model, ids, [])
        result['physical_vs_bypass_max_logit_error'] = (physical-actual).abs().max().item()
        result['physical_equivalent'] = result['physical_vs_bypass_max_logit_error'] <= 1e-4
        del physical
    return result
