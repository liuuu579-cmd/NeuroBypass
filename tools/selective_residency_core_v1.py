"""CPU weight bank and evict-before-load dynamic decoder residency."""
import copy
import gc
import hashlib
import time

import torch

from prompt_cost_core_v1 import cache_bytes, parameter_bytes
from prompt_fixed_core_v1 import FixedRouteSession, raw_prompt_embeddings


class SelectiveBank:
    def __init__(self, model, device):
        if any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('Selective deployment must start with all weights on CPU')
        self.model, self.device = model, torch.device(device)
        self.masters = list(model.model.decoder.layers)
        self.active = set()
        self.block_bytes = [sum(p.numel() * p.element_size() for p in m.parameters()) for m in self.masters]
        decoder = model.model.decoder
        for name in ['embed_tokens', 'embed_positions', 'project_in', 'project_out', 'final_layer_norm']:
            module = getattr(decoder, name, None)
            if module is not None:
                module.to(self.device)
        model.lm_head.to(self.device)
        model.tie_weights()
        if model.lm_head.weight.data_ptr() != decoder.embed_tokens.weight.data_ptr():
            raise ValueError('OPT shared input/output embedding lost its alias')

    def activate(self, skips):
        skips = set(map(int, skips))
        if any(i < 0 or i >= len(self.masters) for i in skips):
            raise ValueError('Invalid layer selection')
        keep = set(range(len(self.masters))) - skips
        removed, added = sorted(self.active - keep), sorted(keep - self.active)
        # No GPU reference to the evicted block remains. CPU masters were never
        # mutated, so there is no device-to-host weight writeback.
        for index in removed:
            self.model.model.decoder.layers[index] = self.masters[index]
        self.active.difference_update(removed)
        if removed:
            gc.collect()
            torch.cuda.empty_cache()
        for index in added:
            self.model.model.decoder.layers[index] = copy.deepcopy(self.masters[index]).to(self.device)
            self.active.add(index)
        torch.cuda.synchronize(self.device)
        if self.active != keep:
            raise ValueError('Residency differs from required route')
        for index, block in enumerate(self.model.model.decoder.layers):
            if any((p.device.type == 'cuda') != (index in keep) for p in block.parameters()):
                raise ValueError('Unrequested GPU block or missing requested block')
        return {'loaded_layers': added, 'evicted_layers': removed,
                'host_to_device_weight_bytes': sum(self.block_bytes[i] for i in added),
                'device_to_host_weight_bytes': 0, 'resident_decoder_layers': len(keep)}


def route(model, router, ids, policy):
    with torch.inference_mode():
        return sorted(router.route(raw_prompt_embeddings(model, ids[:, :128]),
            policy['k'], policy['prefix'])[0].tolist())


def request(model, bank, router, ids, policy, new_tokens, selective, reset_peak=True):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    skips = route(model, router, ids, policy) if router is not None else []
    torch.cuda.synchronize()
    routed = time.perf_counter()
    movement = bank.activate(skips) if selective else {'loaded_layers': [], 'evicted_layers': [],
        'host_to_device_weight_bytes': 0, 'device_to_host_weight_bytes': 0,
        'resident_decoder_layers': len(bank.active)}
    switched = time.perf_counter()
    session = FixedRouteSession(model, skips)
    generated = []
    with torch.inference_mode():
        logits = session.forward(ids)
        token = logits[:, -1].argmax(-1, keepdim=True)
        first_logits = logits[:, -1].float().cpu()
        generated.append(token); del logits
        torch.cuda.synchronize()
        first = time.perf_counter()
        for step in range(1, new_tokens):
            logits = session.forward(token)
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token); del logits
        torch.cuda.synchronize()
        end = time.perf_counter()
    cached_tokens = ids.size(1) + new_tokens - 1
    kv = session.cache_bytes()
    expected = 2 * (40-len(skips)) * cached_tokens * 5120 * 2
    if kv != expected:
        raise ValueError('KV tensor storage differs from active layer count')
    result = {**movement, 'skipped_layers': skips, 'route_seconds': routed-start,
        'weight_switch_seconds': switched-routed, 'prefill_seconds': first-switched,
        'ttft_seconds': first-start, 'decode_seconds': end-first, 'total_seconds': end-start,
        'decode_tokens_per_second': (new_tokens-1)/(end-first),
        'end_to_end_tokens_per_second': new_tokens/(end-start), 'prompt_tokens': ids.size(1),
        'new_tokens': new_tokens, 'kv_bytes': kv, 'cached_tokens': cached_tokens,
        'model_cuda_parameter_bytes': parameter_bytes(model),
        'router_cuda_parameter_bytes': parameter_bytes(router) if router is not None else 0,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'generated_tokens_sha256': hashlib.sha256(torch.cat(generated, 1).cpu().numpy().tobytes()).hexdigest()}
    # No cache or generated GPU tensor is retained across requests. A new path
    # always starts with a fresh full-context prefill in this initial pilot.
    del session, token, generated
    return result, first_logits
