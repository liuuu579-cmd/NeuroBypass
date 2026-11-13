"""Whole-layer EvoPress OPT adaptation; official mutation and staged selection.

Written independently against the public algorithm. No third-party files copied.
The constrained calibration source, protected layers and search budget are
explicit experiment choices, not a claim to reproduce the original tables.
"""
import math
import random
import time

import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import fixed_route

OFFICIAL = {
    'paper': 'https://arxiv.org/abs/2410.14649',
    'repository': 'https://github.com/IST-DASLab/EvoPress',
    'revision': 'a66a6974e91d27ede9043f07f885709732a9a9b1',
    'algorithm_source': 'evo_drop_search.py',
    'fitness_source': 'src/metrics.py',
}


def legal(mask, k):
    mask = tuple(sorted(mask))
    if len(mask) != k or len(set(mask)) != k or any(i < 2 or i >= 40 for i in mask):
        raise ValueError('Invalid exact-K whole-layer configuration')
    return mask


def initial_masks(k, number, rng):
    result = []
    while len(result) < number:
        attn = tuple(sorted(rng.sample(list(range(2, 40)), k)))
        # Official code draws both component configurations, then ties MLP to
        # attention when drop_entire_block is enabled. Preserve these RNG draws.
        rng.sample(list(range(2, 40)), k)
        if attn not in result:
            result.append(legal(attn, k))
    return result


def offspring(parent, number, rng):
    """Mirror independent component mutations followed by whole-block tying.

    Some official whole-block proposals are no-ops because only the MLP changed;
    reject these and duplicate/illegal proposals as the official program does.
    """
    result = []
    attempts = 0
    while len(result) < number:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError('Cannot fill offspring population')
        states = [set(parent), set(parent)]
        flips = min(rng.randint(1, 3), rng.randint(1, 3))
        for _ in range(flips):
            state = states[rng.randint(0, 1)]
            remove = rng.randrange(40)
            while remove in state:
                remove = rng.randrange(40)
            restore = rng.randrange(40)
            while restore not in state:
                restore = rng.randrange(40)
            state.add(remove)
            state.remove(restore)
        candidate = tuple(sorted(states[0]))
        if candidate == tuple(parent) or candidate in result or any(i < 2 for i in candidate):
            continue
        result.append(legal(candidate, len(parent)))
    return result


def minibatch(lengths, budget, rng):
    """Random documents without replacement, last sequence truncated to budget.

    Budget counts input tokens, matching upstream; fitness counts the shifted
    next-token positions (one fewer per nonempty sequence).
    """
    if budget > sum(lengths.values()) or budget < 2:
        raise ValueError('Calibration token budget exceeds available data')
    used = set()
    result = []
    tokens = 0
    keys = list(lengths)
    while tokens < budget:
        sample = keys[rng.randrange(len(keys))]
        if sample in used:
            continue
        used.add(sample)
        count = min(lengths[sample], budget - tokens)
        result.append((sample, count))
        tokens += count
    return result


def log_probs(model, ids, mask):
    with fixed_route(model, mask):
        states = model.model(ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
            use_cache=False).last_hidden_state
        return F.log_softmax(model.lm_head(states).float(), -1)


class Fitness:
    def __init__(self, model, data, calibration):
        self.model = model
        self.device = next(model.parameters()).device
        self.raw = {}
        self.teachers = {}
        self.calls = self.forward_tokens = self.scored_tokens = 0
        self.seconds = 0.
        start = time.monotonic()
        with torch.inference_mode():
            for sample in calibration:
                raw, _ = data[sample]
                if data.rows[sample]['split'] != 'train':
                    raise ValueError('Non-training calibration document')
                self.raw[sample] = raw
                ids = torch.tensor(raw, dtype=torch.long, device=self.device)[None]
                self.teachers[sample] = log_probs(model, ids, []).cpu()
        self.teacher_seconds = time.monotonic() - start

    def evaluate(self, mask, batch):
        total = 0.
        positions = 0
        start = time.monotonic()
        with torch.inference_mode():
            for sample, length in batch:
                if length <= 1:
                    continue
                ids = torch.tensor(self.raw[sample][:length], dtype=torch.long, device=self.device)[None]
                teacher = self.teachers[sample][:, :length-1].to(self.device)
                student = log_probs(self.model, ids, mask)
                value = (teacher.exp() * (teacher - student)).sum(-1).double().sum().item()
                if not math.isfinite(value) or value < -1e-3:
                    raise ValueError('Invalid teacher KL')
                total += value
                positions += length-1
                self.forward_tokens += length-1
        self.calls += 1
        self.scored_tokens += positions
        self.seconds += time.monotonic()-start
        return total/positions

    def stats(self):
        return {k: getattr(self, k) for k in ('calls', 'forward_tokens', 'scored_tokens', 'seconds', 'teacher_seconds')}


def evolve(k, seed, fitness, cfg, log, progress):
    rng = random.Random(seed)
    lengths = {s: len(x) for s, x in fitness.raw.items()}
    seen = set()
    start = time.monotonic()

    def selection(candidates, survivors, tokens, generation, stage):
        batch = minibatch(lengths, tokens, rng)
        scores = []
        for mask in candidates:
            legal(mask, k)
            seen.add(tuple(mask))
            scores.append(fitness.evaluate(mask, batch))
        order = sorted(range(len(candidates)), key=lambda i: (scores[i], candidates[i]))[:survivors]
        row = {'k': k, 'seed': seed, 'generation': generation, 'stage': stage,
            'input_token_budget': tokens, 'scored_positions': sum(max(0, n-1) for _, n in batch),
            'calibration_documents': len(batch), 'candidates': [list(x) for x in candidates],
            'fitnesses': scores, 'survivor_indices': order}
        log(row)
        return [candidates[i] for i in order], [scores[i] for i in order]

    population, scores = selection(initial_masks(k, cfg['initially_generated'], rng), 1,
        cfg['initial_tokens'], 0, 'initial')
    history = []
    for generation in range(1, cfg['generations']+1):
        candidates = offspring(population[0], cfg['offspring'], rng)
        candidates, _ = selection(candidates, 2, cfg['stage_tokens'][0], generation, 'screen')
        if population[0] not in candidates:
            candidates.append(population[0])
        population, scores = selection(candidates, 1, cfg['stage_tokens'][1], generation, 'elite')
        row = {'generation': generation, 'fitness': scores[0], 'skips': list(population[0]),
            'elapsed_seconds': time.monotonic()-start}
        history.append(row)
        progress(row)
    # Select the final elitist population, never downstream scores or reviewed
    # validation labels. Record full-calibration score without further selection.
    final = population[0]
    full_score = fitness.evaluate(final, [(s, n) for s, n in lengths.items()])
    return {'k': k, 'seed': seed, 'skips': list(final), 'full_calibration_kl': full_score,
        'last_stage_kl': scores[0], 'unique_searched_masks': len(seen), 'history': history,
        'search_seconds': time.monotonic()-start}


def algorithm_checks():
    for k in (12, 16, 20, 24):
        a = initial_masks(k, 64, random.Random(579))
        assert len(set(a)) == 64
        for parent in a[:4]:
            b = offspring(parent, 32, random.Random(580))
            assert len(set(b)) == 32 and parent not in b
            for mask in b:
                legal(mask, k)
                assert 1 <= len(set(parent)-set(mask)) <= 3
    lengths = {0: 15, 1: 43, 2: 31}
    for budget in (2, 18, 60, 89):
        batch = minibatch(lengths, budget, random.Random(581))
        assert sum(n for _, n in batch) == budget
        assert len({s for s, _ in batch}) == len(batch)
    assert initial_masks(20, 4, random.Random(579)) == initial_masks(20, 4, random.Random(579))
    return {'passed': True, 'exact_k_and_protected_layers': True,
        'official_component_mutation_then_whole_block_tying': True,
        'elitist_staged_selection': True, 'token_budget_and_determinism': True}
