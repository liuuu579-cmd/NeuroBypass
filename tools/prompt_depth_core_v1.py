"""Prompt-fixed deeper routers and audited conditional greedy records.

The original six-step implementation remains unchanged. All budgets here are
externally specified; the router does not predict an autonomous skip count.
"""
import numpy as np
import torch

from prompt_fixed_core_v1 import SequentialGreedyRouter, best_candidate
from train_prompt_fixed_v1 import targets_from_record


class DepthRouter(SequentialGreedyRouter):
    def __init__(self, input_dim=5120, width=256, max_k=24, dropout=.1):
        super().__init__(input_dim=input_dim, width=width, max_k=max_k, dropout=dropout)

    def route(self, embeddings, k=24, prefix=()):
        return self.route_encoded(self.encode(embeddings), k, prefix)

    def route_encoded(self, context, k=24, prefix=()):
        prefix = [int(i) for i in prefix]
        if not 1 <= k <= self.max_k or len(prefix) > k:
            raise ValueError('Unsupported total budget or prefix length')
        if len(prefix) != len(set(prefix)) or any(i < 2 or i >= self.layers for i in prefix):
            raise ValueError('Invalid fixed prefix')
        selected = context.new_zeros((context.size(0), self.layers))
        path = [torch.full((context.size(0),), i, device=context.device, dtype=torch.long) for i in prefix]
        if prefix:
            selected[:, prefix] = 1
        for step in range(len(prefix) + 1, k + 1):
            scores = self.score(context, selected, torch.full((context.size(0),), step,
                device=context.device, dtype=torch.long))
            unavailable = selected.bool().clone()
            unavailable[:, :2] = True
            choice = scores.masked_fill(unavailable, float('inf')).argmin(-1)
            selected.scatter_(1, choice[:, None], 1)
            path.append(choice)
        return torch.stack(path, dim=1)


def validate_steps(record, initial=(), maximum=24):
    selected = list(initial)
    for step in record['steps']:
        if step['step'] != len(selected) + 1 or step['previous_skips'] != selected:
            raise ValueError('Conditional state/prefix mismatch')
        costs = step['candidate_costs']
        if {int(i) for i, _ in costs} != set(range(2, 40)) - set(selected):
            raise ValueError('Incomplete joint candidate set')
        layer, _ = best_candidate(costs)
        if layer != step['selected_layer']:
            raise ValueError('Greedy choice is not the recorded joint minimum')
        selected.append(layer)
    if selected != record['ordered_skips'] or len(selected) != maximum:
        raise ValueError('Final route mismatch')
    return True


def fork_prefix(record, arm, sample, seed, fixed=()):
    # Different training records cover forks in shallow and deep states.
    positions = [5, 9, 13, 17] if arm == 'pure' else [7, 11, 15, 19]
    length = positions[(sample + seed) % len(positions)]
    state = next(row for row in record['steps'] if row['step'] == length)
    costs = sorted(state['candidate_costs'], key=lambda row: (row[1], row[0]))
    alternative = 1 + ((sample * 7 + seed) % 4)
    prefix = record['ordered_skips'][:length - 1] + [costs[alternative][0]]
    if set(prefix) == set(record['ordered_skips'][:length]):
        raise ValueError('Fork must differ from the expert set')
    if arm == 'hybrid' and prefix[:len(fixed)] != list(fixed):
        raise ValueError('A hybrid fork changed its fixed global prefix')
    return prefix, alternative + 1


def pack_record(record, branch):
    a, b = targets_from_record(record), targets_from_record(branch)
    return tuple(np.concatenate([x, y], axis=0) for x, y in zip(a, b))


def document_loss(scores, targets, valid, chosen, present):
    """Equal document weights, regardless of variable branch length/padding."""
    import torch.nn.functional as F
    logits = (-scores).masked_fill(~valid, -1e9)
    soft = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
    hard = F.cross_entropy(logits.reshape(-1, 40), chosen.reshape(-1), reduction='none').reshape_as(soft)
    per_state = soft + .25 * hard
    return ((per_state * present).sum(-1) / present.sum(-1).clamp_min(1)).mean()
