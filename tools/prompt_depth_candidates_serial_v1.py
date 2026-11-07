"""Exact batch-one candidate tails with shared unchanged prefix activations.

This offline search optimization never batches heterogeneous routes and makes
no deployment acceleration claim. The installed OPT path supplies all masks.
"""
import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import fixed_route


def prefix_candidate_costs(model, ids, selected, candidates, objective='kl', teacher=None):
    if len(candidates) != len(set(candidates)) or set(candidates) & set(selected):
        raise ValueError('Invalid candidates')
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
        raise ValueError('Offline shared prefix cannot use a KV cache')
    costs = []
    for candidate in candidates:
        # Original decoder hidden_states[i] is the input to original layer i.
        hidden = baseline.hidden_states[candidate]
        for index in range(candidate + 1, len(decoder.layers)):
            if index not in selected:
                hidden = decoder.layers[index](hidden, **captured)[0]
        if decoder.final_layer_norm is not None:
            hidden = decoder.final_layer_norm(hidden)
        if decoder.project_out is not None:
            hidden = decoder.project_out(hidden)
        logits = model.lm_head(hidden if objective == 'nll' else hidden[:, 127:]).float()
        if objective == 'kl':
            log_probs = F.log_softmax(logits, -1)
            loss = (teacher.exp() * (teacher - log_probs)).sum(-1).mean()
        elif objective == 'nll':
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        else:
            raise ValueError('Unknown objective')
        if not torch.isfinite(loss): raise ValueError('Nonfinite candidate cost')
        costs.append([candidate, loss.item()])
    return costs
