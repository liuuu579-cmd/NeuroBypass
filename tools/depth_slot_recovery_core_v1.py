"""Equal-capacity recovery pilot: q/v LoRA indexed by executed depth.

An adapter bank is shared across all routes. This is a depth-slot parameter
tying variant, not conventional independently parameterized original-layer LoRA.
The original OPT parameters and all on-disk artifacts remain read-only.
"""
import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from prompt_fixed_core_v1 import fixed_route, valid_route


class Slot(nn.Module):
    def __init__(self, hidden, rank, generator):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, hidden, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(hidden, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5), generator=generator)

    def forward(self, value):
        return F.linear(F.linear(value.float(), self.a), self.b)


class DepthSlots(nn.Module):
    def __init__(self, hidden, kept, rank=8, alpha=16, seed=579):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.slots = nn.ModuleList([nn.ModuleDict({
            name: Slot(hidden, rank, generator) for name in ('q_proj', 'v_proj')
        }) for _ in range(kept)])
        self.hidden, self.kept, self.rank, self.alpha = hidden, kept, rank, alpha
        self.enabled = False
        self.mapping = {}
        self.checkpoint_blocks = True
        self.block_calls = {}

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Projection(nn.Module):
    def __init__(self, base, bank, layer, name):
        super().__init__()
        self.base = base
        # The bank is owned once outside OPT; avoid repeated registration/cycles.
        object.__setattr__(self, 'bank', bank)
        self.layer, self.name = layer, name

    def forward(self, value):
        output = self.base(value)
        if not self.bank.enabled:
            return output
        slot = self.bank.mapping[self.layer]
        update = self.bank.slots[slot][self.name](value)
        return output + (update * (self.bank.alpha/self.bank.rank)).to(output.dtype)


def attach(model, bank):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze the base before attaching adapters')
    if bank.hidden != model.config.hidden_size:
        raise ValueError('Adapter hidden dimension mismatch')
    for index, block in enumerate(model.model.decoder.layers):
        for name in ('q_proj', 'v_proj'):
            projection = getattr(block.self_attn, name)
            if isinstance(projection, Projection):
                raise ValueError('Adapters already attached')
            setattr(block.self_attn, name, Projection(projection, bank, index, name))
        original = block.forward

        def forward(*args, _original=original, _index=index, **kwargs):
            bank.block_calls[_index] = bank.block_calls.get(_index, 0)+1
            if bank.enabled and bank.checkpoint_blocks and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False,
                    preserve_rng_state=False, **kwargs)
            return _original(*args, **kwargs)
        block.forward = forward


@contextlib.contextmanager
def routed(model, bank, skips, adapters=True):
    route = valid_route(model, skips)
    kept = [i for i in range(len(model.model.decoder.layers)) if i not in route]
    if adapters and len(kept) != bank.kept:
        raise ValueError('Adapter bank requires the preregistered retained depth')
    old = bank.enabled, bank.mapping
    bank.enabled = adapters
    bank.mapping = dict(zip(kept, range(len(kept)))) if adapters else {}
    try:
        with fixed_route(model, route):
            # Keep this context active through checkpoint recomputation/backward.
            yield
    finally:
        bank.enabled, bank.mapping = old


def log_probs(model, ids, prompt_length):
    if not 1 <= prompt_length < ids.size(1) <= model.config.max_position_embeddings:
        raise ValueError('Invalid prompt/target boundary')
    hidden = model.model(ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
        use_cache=False).last_hidden_state
    logits = model.lm_head(hidden[:, prompt_length-1:]).float()
    return F.log_softmax(logits, -1)


def objective(student, teacher, targets):
    if student.shape != teacher.shape or student.shape[:2] != targets.shape:
        raise ValueError('Teacher/student/target misalignment')
    ce = F.nll_loss(student.flatten(0, 1), targets.flatten())
    kl = (teacher.exp()*(teacher-student)).sum(-1).mean()
    return ce+kl, ce, kl


def save_adapter(path, bank, metadata):
    with path.open('xb') as stream:
        torch.save({'state_dict': {n: p.detach().cpu().clone() for n,p in bank.state_dict().items()},
            'metadata': metadata}, stream)
