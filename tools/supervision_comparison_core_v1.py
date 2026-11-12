"""Stage24: change supervision, keep initialization, optimizer and update budget."""
import gc
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import cpu_fit_predictions, load_choice
from deep_swap_core_v1 import StaticSwapRouter
from full_context_experiment_v1 import thread_record, write
from stage1_benchmark_v2 import sha256


SEEDS = (579, 580, 581)
JOBS = [(kind, seed) for kind in ('mixed', 'prompt_full') for seed in SEEDS]


def targets(future, prompt, weight):
    assert weight in (0., .5, 1.)
    future = torch.as_tensor(future, dtype=torch.float32)
    prompt = torch.as_tensor(prompt, dtype=torch.float32)
    f = future - future[:, :1]
    p = prompt - prompt[:, :1]
    prior = f.mean(0)
    aligned = prior[None] + p - p.mean(0)[None]
    target = (1. - weight) * f + weight * aligned
    assert torch.all(target[:, 0] == 0)
    assert torch.allclose(target.mean(0), prior, atol=1e-6, rtol=0)
    # Same prior AND normalization for all supervision arms, matching Stage20.
    scale = f[:, 1:].std().clamp_min(.03)
    return target, prior, scale


def fit(out, cfg, future, prompt_cost, embeddings, kind, seed, observe, device, messages, rank):
    train, selection = cfg['train_ids'], cfg['selection_ids']
    arm = 'prompt' if observe else 'constant'; name = f'{kind}_{arm}_s{seed}'
    values = np.array([future[i]['kl'] for i in train])
    prompts = np.array([prompt_cost[i]['prompt_kl'] for i in train])
    target, prior, scale = targets(values, prompts, .5 if kind == 'mixed' else 1.)
    scale = scale.to(device)
    torch.manual_seed(seed)
    router = StaticSwapRouter(8, width=128, observe=observe).to(device)
    router.prior.copy_(prior.to(device)); router.scale.copy_(scale)
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in router.state_dict().values())).hexdigest()
    if not cfg['probe']:
        assert initial == cfg['reference_initial_sha256'][f'{arm}_s{seed}']
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    total = cfg['epochs'] * math.ceil(len(train) / 16)
    def schedule(n):
        warm = max(1, total // 20)
        return (n + 1) / warm if n < warm else .1 + .9 * .5 * (1 + math.cos(math.pi * (n - warm) / max(1, total - warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    # Every arm chooses checkpoints on the same REAL CONTINUATION KL on original32.
    selection_cost = np.array([future[i]['kl'] for i in selection])
    best = math.inf; checkpoint = None; history = []
    for epoch in range(cfg['epochs'] + 1):
        losses = []
        if epoch:
            router.train()
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + 10 * epoch)).tolist()
            for offset in range(0, len(order), 16):
                batch = order[offset:offset + 16]
                pending = sorted(batch, key=lambda i: embeddings[train[i]].shape[1])
                optimizer.zero_grad(set_to_none=True); loss_sum = 0.
                while pending:
                    sub = [pending.pop(0)]
                    while pending and (len(sub) + 1) * embeddings[train[pending[0]]].shape[1] <= 4096:
                        sub.append(pending.pop(0))
                    length = max(embeddings[train[i]].shape[1] for i in sub) if observe else 1
                    x = torch.zeros(len(sub), length, 5120, dtype=torch.float16, device=device)
                    valid = torch.zeros(len(sub), length, dtype=torch.bool, device=device)
                    if observe:
                        for j, i in enumerate(sub):
                            value = embeddings[train[i]][0]
                            x[j, :len(value)] = value.to(device); valid[j, :len(value)] = True
                    else:
                        valid[:] = True
                    prediction = router(x, valid)
                    loss = F.smooth_l1_loss(prediction[:, 1:] / scale, target[sub, 1:].to(device) / scale) * len(sub) / len(batch)
                    assert torch.isfinite(loss)
                    loss.backward(); loss_sum += float(loss.detach())
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(router.parameters(), 1.))
                optimizer.step(); scheduler.step(); losses.append(loss_sum)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else 0.}
        if epoch in (0, 1) or epoch % 4 == 0 or epoch == cfg['epochs']:
            router.eval()
            routes = cpu_fit_predictions(router, embeddings, selection, device).argmin(-1).numpy()
            value = float(selection_cost[np.arange(len(routes)), routes].mean())
            row.update({'selection_kl': value, 'selection_unique_routes': len(set(routes.tolist()))})
            if value < best:
                path = out / f'private_{name}_e{epoch:03d}.pth'
                with path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'candidates': 8, 'width': 128,
                        'observe': observe, 'seed': seed, 'epoch': epoch,
                        'bank_sha256': sha256(out / 'candidate_bank.json')}, stream)
                best = value; checkpoint = {'path': str(path), 'sha256': sha256(path), 'epoch': epoch, 'selection_kl': value}
        history.append(row)
        messages.put({'kind': 'progress', 'rank': rank, 'phase': 'fit', 'kind_name': kind,
            'seed': seed, 'arm': arm, **row})
    saved = load_choice(checkpoint['path'], device)
    routes = cpu_fit_predictions(saved, embeddings, train, device).argmin(-1).numpy()
    write(out / f'{name}_history.json', history)
    write(out / f'{name}_fit.json', {'kind': kind, 'lambda': .5 if kind == 'mixed' else 1., 'seed': seed,
        'arm': arm, 'best': checkpoint, 'initial_sha256': initial,
        'parameters': sum(p.numel() for p in router.parameters()), 'updates': total, 'train_documents': len(train),
        'train_actual_continuation_kl': float(values[np.arange(len(routes)), routes].mean()),
        'target_mean': target.mean(0).tolist(), 'common_prior': prior.tolist(), 'common_scale': float(scale),
        'selection_metric': 'Original32 actual continuation joint KL, never prompt target or other68', **thread_record()})
    del saved, router, optimizer
    gc.collect(); torch.cuda.empty_cache()


def cpu_checks():
    f = torch.tensor([[1., .8, 1.3], [.5, .6, .9], [1.4, 1.2, 1.6]])
    p = torch.tensor([[2., 2.1, 2.2], [.7, .3, .9], [1., 1.3, .8]])
    ts = [targets(f, p, weight) for weight in (0., .5, 1.)]
    assert torch.equal(ts[0][0], f - f[:, :1])
    for target, prior, scale in ts:
        assert torch.allclose(target.mean(0), ts[0][1], atol=1e-6)
        assert torch.equal(prior, ts[0][1]) and torch.equal(scale, ts[0][2])
    assert torch.allclose(ts[1][0], .5 * (ts[0][0] + ts[2][0]))
    assert torch.allclose(targets(f + 3, p + 7, .5)[0], ts[1][0], atol=1e-6)
    return {'passed': True, 'all_target_means_equal_future_training_prior': True,
        'identical_scale_and_prior': True, 'half_mix_exact': True, 'reference_index0_is_zero': True,
        'per_document_common_offsets_cancel': True}
