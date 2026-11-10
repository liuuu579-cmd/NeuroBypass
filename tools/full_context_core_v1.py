"""Prompt-only, variable-length routing and conditional candidate scoring.

This version never truncates an observation behind the caller's back. Call
visible_context BEFORE both routing and generation when a context is too long.
Existing bounded-observation programs and checkpoints remain immutable.
"""
import torch
import torch.nn.functional as F

from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import fixed_route

OBSERVATION_VERSION = 'full-visible-prompt-v1'


def visible_context(prompt_ids, context_limit=2048, reserve_tokens=0):
    """Explicit left truncation shared by the base model and router.

    prompt_ids must contain context only, never answer options or targets.
    The caller receives the exact tensor to use for both routing and prefill.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError('Need one nonempty prompt')
    budget = context_limit - reserve_tokens
    if not 0 <= reserve_tokens < context_limit or budget < 1:
        raise ValueError('Invalid generation reservation')
    removed = max(0, prompt_ids.shape[1] - budget)
    return prompt_ids[:, removed:], removed


def prompt_only(sequence_ids, prompt_length):
    """An explicit boundary prevents teacher-forced targets reaching the router."""
    if sequence_ids.ndim != 2 or not 1 <= prompt_length <= sequence_ids.shape[1]:
        raise ValueError('Invalid prompt boundary')
    return sequence_ids[:, :prompt_length]


def prompt_embeddings(model, prompt_ids, valid=None):
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1:
        raise ValueError('Empty observation')
    limit = model.config.max_position_embeddings
    if prompt_ids.shape[1] > limit:
        raise ValueError('Truncate base-model context explicitly before routing')
    if valid is None:
        valid = torch.ones_like(prompt_ids, dtype=torch.bool)
    if valid.shape != prompt_ids.shape or valid.dtype != torch.bool:
        raise ValueError('Invalid attention mask')
    if not valid.any(1).all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError('Only nonempty right-padded observations are supported')
    decoder = model.model.decoder
    value = decoder.embed_tokens(prompt_ids)
    if decoder.project_in is not None:
        value = decoder.project_in(value)
    value = value + decoder.embed_positions(valid.long(), past_key_values_length=0)
    return value.masked_fill(~valid[..., None], 0)


class FullContextRouter(DepthRouter):
    """Parameter-compatible encoder with padding-aware attention AND pooling."""
    observation_version = OBSERVATION_VERSION

    def encode(self, embeddings, valid=None):
        if valid is None:
            valid = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if valid.shape != embeddings.shape[:2] or valid.dtype != torch.bool or not valid.any(1).all():
            raise ValueError('Invalid router padding mask')
        padding = None if valid.all() else ~valid
        hidden = self.encoder(self.projection(embeddings.float()), src_key_padding_mask=padding)
        weights = self.pool(hidden).masked_fill(~valid[..., None], -torch.inf).softmax(dim=1)
        return (hidden * weights).sum(dim=1)

    def forward(self, embeddings, selected, steps, valid=None):
        return self.score(self.encode(embeddings, valid), selected, steps)

    def route(self, embeddings, k=24, prefix=(), valid=None):
        return self.route_encoded(self.encode(embeddings, valid), k, prefix)


def route_prompt(model, router, prompt_ids, k, prefix=()):
    return router.route(prompt_embeddings(model, prompt_ids), k, prefix)[0].tolist()


def load_router(checkpoint, device):
    record = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if record.get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('Checkpoint was not trained under the full-context contract')
    router = FullContextRouter(max_k=record['max_k']).to(device).eval().requires_grad_(False)
    router.load_state_dict(record['state_dict'])
    return router


def candidate_costs(model, ids, prompt_length, selected, candidates, teacher=None, objective='kl'):
    """Exact joint masks; reuse only the activations BEFORE a new skipped layer.

    KL covers continuation positions only, independent of prompt length. The
    NLL control retains its historical full-sequence likelihood objective.
    """
    if not 1 <= prompt_length < ids.size(1) <= model.config.max_position_embeddings:
        raise ValueError('Invalid prompt/continuation boundary or context limit')
    if (len(candidates) != len(set(candidates)) or set(candidates) & set(selected)
            or any(i < 2 or i >= len(model.model.decoder.layers) for i in candidates)):
        raise ValueError('Invalid joint candidate set')
    if objective == 'kl' and (teacher is None or teacher.shape[1] != ids.size(1) - prompt_length):
        raise ValueError('Teacher continuation is misaligned')
    decoder, captured = model.model.decoder, {}
    def capture(module, args, kwargs):
        captured.update(kwargs)
    handle = decoder.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with fixed_route(model, selected):
            baseline = model.model(ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
                use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    if captured.get('past_key_value') is not None or captured.get('use_cache'):
        raise ValueError('Candidate search must not use a cache')
    costs = []
    for candidate in candidates:
        hidden = baseline.hidden_states[candidate]
        for index in range(candidate + 1, len(decoder.layers)):
            if index not in selected:
                hidden = decoder.layers[index](hidden, **captured)[0]
        if decoder.final_layer_norm is not None:
            hidden = decoder.final_layer_norm(hidden)
        if decoder.project_out is not None:
            hidden = decoder.project_out(hidden)
        logits = model.lm_head(hidden[:, prompt_length-1:] if objective == 'kl' else hidden).float()
        if objective == 'kl':
            log_probs = F.log_softmax(logits, -1)
            loss = (teacher.exp() * (teacher - log_probs)).sum(-1).mean()
        elif objective == 'nll':
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        else:
            raise ValueError('Unknown candidate objective')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite candidate cost')
        costs.append([int(candidate), loss.item()])
    return costs
