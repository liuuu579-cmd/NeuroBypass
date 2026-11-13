"""Four-fold conditional routing learnability diagnostic; no historical dev selection."""
import gc
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import cpu_fit_predictions, load_choice
from deep_swap_core_v1 import StaticSwapRouter
from full_context_experiment_v1 import write
from stage1_benchmark_v2 import sha256


SEEDS = (579, 580, 581)
MARGINS = (0., .05, .1, .2, .4)
RIDGES = (1., 10., 100., 1000.)


def grouped_folds(ids, records):
    # Connected components for either repeated full text or repeated lead text.
    parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i in ids:
        for field in ('record_sha256', 'record_lead_sha256'):
            key = (field, records[i][field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    ordered = list(groups.values())
    np.random.default_rng(250579).shuffle(ordered)
    ordered.sort(key=lambda g: -len(g))  # stable randomized tie order
    buckets = [[] for _ in range(4)]
    for g in ordered:
        bucket = min(range(4), key=lambda b: (len(buckets[b]), b)); buckets[bucket].extend(g)
    folds = []
    for fold in range(4):
        outer = sorted(buckets[fold]); available = [g for g in ordered if not set(g) & set(outer)]
        np.random.default_rng(250580 + fold).shuffle(available)
        selection = []; fit = []
        for g in available:
            (selection if len(selection) < 80 else fit).extend(g)
        folds.append({'fold': fold, 'fit_ids': sorted(fit), 'selection_ids': sorted(selection), 'outer_ids': outer})
    assert sorted(s for f in folds for s in f['outer_ids']) == sorted(ids)
    for f in folds:
        groups_of = [{records[i][field] for i in f[part]} for part in ('fit_ids', 'selection_ids', 'outer_ids')
            for field in ()]  # below keeps both grouping keys explicit
        for field in ('record_sha256', 'record_lead_sha256'):
            sets = [{records[i][field] for i in f[part]} for part in ('fit_ids', 'selection_ids', 'outer_ids')]
            assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    return folds


def choose(scores, baseline, margin=0.):
    candidate = np.asarray(scores).argmin(1)
    improvement = scores[:, baseline] - scores[np.arange(len(scores)), candidate]
    return np.where(improvement > margin, candidate, baseline).astype(np.int64)


def gate_fit(scores, costs, baseline, scale):
    records = [{'static': True, 'margin_units': 0., 'selection_kl': float(costs[:, baseline].mean()), 'switches': 0, 'serial': 0}]
    for margin in MARGINS:
        routes = choose(scores, baseline, margin * scale)
        records.append({'static': False, 'margin_units': margin,
            'selection_kl': float(costs[np.arange(len(costs)), routes].mean()),
            'switches': int((routes != baseline).sum()), 'serial': len(records)})
    return min(records, key=lambda r: (r['selection_kl'], r['switches'], r['serial']))


def gated(scores, fitted, baseline, scale):
    return np.full(len(scores), baseline, dtype=np.int64) if fitted['static'] else choose(scores, baseline, fitted['margin_units'] * scale)


def neural_fit(out, cfg, fold, embeddings, labels, seed, device, messages, rank):
    name = f'f{fold["fold"]}_s{seed}'
    fit_ids, selection = fold['fit_ids'], fold['selection_ids']
    costs = torch.tensor([labels[i]['kl'] for i in fit_ids], dtype=torch.float32)
    target = costs - costs[:, :1]; scale = target[:, 1:].std().clamp_min(.03).to(device)
    torch.manual_seed(seed)
    router = StaticSwapRouter(8, width=128, observe=True).to(device)
    router.prior.copy_(target.mean(0).to(device)); router.scale.copy_(scale)
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in router.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    total = cfg['epochs'] * math.ceil(len(fit_ids) / 16)
    def schedule(n):
        warm = max(1, total // 20)
        return (n + 1) / warm if n < warm else .1 + .9 * .5 * (1 + math.cos(math.pi * (n - warm) / max(1, total - warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    selection_cost = np.array([labels[i]['kl'] for i in selection])
    history = []; best = math.inf; checkpoint = None
    for epoch in range(cfg['epochs'] + 1):
        losses = []
        if epoch:
            router.train()
            order = torch.randperm(len(fit_ids), generator=torch.Generator().manual_seed(seed + 10 * epoch)).tolist()
            for offset in range(0, len(order), 16):
                batch = order[offset:offset + 16]
                pending = sorted(batch, key=lambda i: embeddings[fit_ids[i]].shape[1])
                optimizer.zero_grad(set_to_none=True); loss_sum = 0.
                while pending:
                    sub = [pending.pop(0)]
                    while pending and (len(sub) + 1) * embeddings[fit_ids[pending[0]]].shape[1] <= 4096:
                        sub.append(pending.pop(0))
                    length = max(embeddings[fit_ids[i]].shape[1] for i in sub)
                    x = torch.zeros(len(sub), length, 5120, dtype=torch.float16, device=device)
                    valid = torch.zeros(len(sub), length, dtype=torch.bool, device=device)
                    for j, i in enumerate(sub):
                        value = embeddings[fit_ids[i]][0]; x[j, :len(value)] = value.to(device); valid[j, :len(value)] = True
                    prediction = router(x, valid)
                    loss = F.smooth_l1_loss(prediction[:, 1:] / scale, target[sub, 1:].to(device) / scale) * len(sub) / len(batch)
                    assert torch.isfinite(loss); loss.backward(); loss_sum += float(loss.detach())
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(router.parameters(), 1.))
                optimizer.step(); scheduler.step(); losses.append(loss_sum)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else 0.}
        if epoch in (0, 1) or epoch % 4 == 0 or epoch == cfg['epochs']:
            router.eval()
            selected = cpu_fit_predictions(router, embeddings, selection, device).argmin(-1).numpy()
            value = float(selection_cost[np.arange(len(selected)), selected].mean())
            row['selection_kl'] = value
            if value < best:
                path = out / f'private_{name}_e{epoch:03d}.pth'
                with path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'candidates': 8, 'width': 128, 'observe': True,
                        'seed': seed, 'epoch': epoch, 'fold': fold['fold']}, stream)
                checkpoint = {'path': str(path), 'sha256': sha256(path), 'epoch': epoch, 'selection_kl': value}; best = value
        history.append(row)
        messages.put({'kind': 'progress', 'rank': rank, 'phase': 'neural_fit', 'fold': fold['fold'], 'seed': seed, **row})
    saved = load_choice(checkpoint['path'], device)
    all_ids = fit_ids + selection + fold['outer_ids']
    score = cpu_fit_predictions(saved, embeddings, all_ids, device).numpy()
    with (out / f'private_{name}_scores.pth').open('xb') as stream:
        torch.save({'ids': all_ids, 'scores': torch.from_numpy(score)}, stream)
    write(out / f'{name}_history.json', history)
    write(out / f'{name}_fit.json', {'best': checkpoint, 'fold': fold['fold'], 'seed': seed,
        'fit_documents': len(fit_ids), 'selection_documents': len(selection), 'outer_documents': len(fold['outer_ids']),
        'updates': total, 'initial_sha256': initial, 'prior': router.prior.detach().cpu().tolist(), 'scale': float(scale),
        'score_sha256': sha256(out / f'private_{name}_scores.pth')})
    del router, saved, optimizer
    gc.collect(); torch.cuda.empty_cache()


def ridge_scores(features, fit_rows, selection_rows, outer_rows, costs, penalty, permutation=None):
    x = np.asarray(features, dtype=np.float64)
    mean, std = x[fit_rows].mean(0), x[fit_rows].std(0)
    std = np.maximum(std, 1e-6)
    train = (x[fit_rows] - mean) / std
    target = costs[fit_rows] - costs[fit_rows, :1]
    if permutation is not None:
        target = target[permutation]
    prior = target.mean(0); y = target - prior
    weights = np.linalg.solve(train.T @ train + penalty * np.eye(train.shape[1]), train.T @ y)
    all_rows = list(fit_rows) + list(selection_rows) + list(outer_rows)
    predictions = prior + ((x[all_rows] - mean) / std) @ weights
    return predictions


def cpu_checks():
    score = np.array([[0., 0., 1.], [-.125, 0., 1.], [-.25, 0., 1.]])
    assert choose(score, 1, .125).tolist() == [1, 1, 0]
    features = np.arange(24, dtype=float).reshape(8, 3); costs = np.arange(32, dtype=float).reshape(8, 4) / 20
    a = ridge_scores(features, [0, 1, 2, 3], [4, 5], [6, 7], costs, 1.)
    altered = costs.copy(); altered[6:] = -999
    b = ridge_scores(features, [0, 1, 2, 3], [4, 5], [6, 7], altered, 1.)
    assert np.array_equal(a, b)
    altered[4:6] = 10000
    c = ridge_scores(features, [0, 1, 2, 3], [4, 5], [6, 7], altered, 1.)
    assert np.array_equal(a, c)
    return {'passed': True, 'strict_tie_keep_static': True, 'outer_labels_do_not_affect_ridge': True,
        'selection_labels_do_not_affect_ridge_fit': True}
