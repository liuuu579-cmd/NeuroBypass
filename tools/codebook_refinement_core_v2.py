"""Six-GPU Stage20 training primitives; immutable complete-mask evidence."""
import gc
import hashlib
import math
import multiprocessing as mp
import os
import queue
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import cpu_fit_predictions, load_choice
from deep_swap_core_v1 import StaticSwapRouter
from full_context_experiment_v1 import thread_record, write
from horizontal_quality_v3 import pin_threads
from stage1_benchmark_v2 import sha256, stamp

SEEDS = [579, 580, 581]


def setup(rank=None):
    assert os.environ['NEUROBYPASS_GPU_COUNT'] == '6'
    assert os.environ['NEUROBYPASS_CPU_COUNT'] == '64'
    cpus = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
    assert len(cpus) == 64
    chosen = cpus[60:] if rank is None else cpus[rank * 10:(rank + 1) * 10]
    pin_threads(chosen)
    torch.set_num_threads(len(chosen))
    torch.set_num_interop_threads(1)
    if rank is not None:
        torch.cuda.set_device(rank)


def guarded(target, rank, cfg, inbox, messages):
    try:
        target(rank, cfg, inbox, messages)
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


class SixWorkers:
    def __init__(self, target, cfg):
        ctx = mp.get_context('spawn')
        self.messages = ctx.Queue()
        self.inboxes = [ctx.Queue() for _ in range(6)]
        self.processes = [ctx.Process(target=guarded, args=(target, r, cfg, self.inboxes[r], self.messages)) for r in range(6)]

    def __enter__(self):
        for p in self.processes:
            p.start()
        return self

    def send(self, value):
        for q in self.inboxes:
            q.put(value)

    def receive(self, kind):
        result = []
        while len(result) < 6:
            try:
                r = self.messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in self.processes) or all(p.exitcode is not None for p in self.processes):
                    raise RuntimeError('Worker exited before phase completion')
                continue
            if r['kind'] == 'error':
                raise RuntimeError(r['traceback'])
            if r['kind'] == 'progress':
                stamp('stage20_progress', **{k: v for k, v in r.items() if k != 'kind'})
                continue
            assert r['kind'] == kind, r
            result.append(r)
        assert len({r['rank'] for r in result}) == 6
        return result

    def __exit__(self, typ, exc, tb):
        if typ is None:
            self.send(None)
            for p in self.processes:
                p.join(timeout=30)
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        if typ is None:
            assert all(p.exitcode == 0 for p in self.processes)


def ranking_loss(predicted, target, scale):
    """Cost-gap-weighted ranking of COMPLETE masks, not individual layers."""
    a, b = torch.triu_indices(target.shape[-1], target.shape[-1], offset=1, device=target.device)
    gap = (target[:, b] - target[:, a]) / scale
    estimated = (predicted[:, b] - predicted[:, a]) / scale
    weight = gap.abs().clamp(max=5.)
    values = F.softplus(-gap.sign() * estimated) * weight
    return (values.sum(-1) / weight.sum(-1).clamp_min(1e-8)).mean()


def loss_value(predicted, target, scale, objective):
    regression = F.smooth_l1_loss(predicted[:, 1:] / scale, target[:, 1:] / scale)
    if objective == 'huber':
        return regression
    assert objective == 'rank'
    return regression + .5 * ranking_loss(predicted, target, scale)


def fit(out, costs_by_id, embeddings, cfg, seed, observe, objective, device, messages, rank):
    arm = 'prompt' if observe else 'constant'
    name = f'{objective}_{arm}_s{seed}'
    train = cfg['train_ids']
    selection = cfg['selection_ids']
    costs = torch.tensor([costs_by_id[i] for i in train], dtype=torch.float32)
    target = costs - costs[:, :1]
    scale = target[:, 1:].std().clamp_min(.03).to(device)
    torch.manual_seed(seed)
    router = StaticSwapRouter(8, width=128, observe=observe).to(device)
    router.prior.copy_(target.mean(0).to(device))
    router.scale.copy_(scale)
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in router.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
    per_epoch = math.ceil(len(train) / 16)
    total = cfg['epochs'] * per_epoch

    def schedule(n):
        warm = max(1, total // 20)
        return (n + 1) / warm if n < warm else .1 + .9 * .5 * (1 + math.cos(math.pi * (n - warm) / max(1, total - warm)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    true_selection = np.array([costs_by_id[i] for i in selection])
    history = []
    best = math.inf
    checkpoint = None
    for epoch in range(cfg['epochs'] + 1):
        losses = []
        if epoch:
            router.train()
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + 10 * epoch)).tolist()
            for offset in range(0, len(order), 16):
                batch = order[offset:offset + 16]
                pending = sorted(batch, key=lambda i: embeddings[train[i]].shape[1])
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
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
                            x[j, :len(value)] = value.to(device)
                            valid[j, :len(value)] = True
                    else:
                        valid[:] = True
                    predicted = router(x, valid)
                    loss = loss_value(predicted, target[sub].to(device), scale, objective) * len(sub) / len(batch)
                    assert torch.isfinite(loss)
                    loss.backward()
                    loss_sum += float(loss.detach())
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(router.parameters(), 1.))
                optimizer.step()
                scheduler.step()
                losses.append(loss_sum)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else 0.}
        if epoch in (0, 1) or epoch % 4 == 0 or epoch == cfg['epochs']:
            router.eval()
            scores = cpu_fit_predictions(router, embeddings, selection, device)
            indices = scores.argmin(-1).numpy()
            value = float(true_selection[np.arange(len(indices)), indices].mean())
            row['selection_kl'] = value
            row['selection_unique_routes'] = len(set(indices.tolist()))
            if value < best:
                path = out / f'private_{name}_e{epoch:03d}.pth'
                with path.open('xb') as stream:
                    torch.save({'state_dict': router.state_dict(), 'candidates': 8, 'width': 128,
                        'observe': observe, 'seed': seed, 'epoch': epoch,
                        'bank_sha256': sha256(out / 'candidate_bank.json')}, stream)
                best = value
                checkpoint = {'path': str(path), 'sha256': sha256(path), 'epoch': epoch, 'selection_kl': value}
        history.append(row)
        messages.put({'kind': 'progress', 'rank': rank, 'phase': 'fit', 'objective': objective, 'seed': seed, 'arm': arm, **row})
    saved = load_choice(checkpoint['path'], device)
    chosen = cpu_fit_predictions(saved, embeddings, train, device).argmin(-1).numpy()
    receipt = {'objective': objective, 'seed': seed, 'arm': arm, 'best': checkpoint,
        'initial_sha256': initial, 'parameters': sum(p.numel() for p in router.parameters()),
        'updates': total, 'train_documents': len(train),
        'train_selected_kl': float(costs[np.arange(len(chosen)), chosen].mean()),
        'future_best_train_kl': float(costs.min(-1).values.mean()), **thread_record()}
    write(out / f'{name}_history.json', history)
    write(out / f'{name}_fit.json', receipt)
    del router, saved, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def cpu_checks():
    target = torch.tensor([[0., -.4, .3], [0., .2, -.1]])
    good = target.clone().requires_grad_()
    bad = (-target).requires_grad_()
    assert ranking_loss(good, target, .2) < ranking_loss(bad, target, .2)
    assert ranking_loss(good, torch.zeros_like(target), .2) == 0
    value = loss_value(good, target, .2, 'rank')
    value.backward()
    assert torch.isfinite(good.grad).all()
    rowwise = sum(loss_value(good[i:i + 1], target[i:i + 1], .2, 'rank') for i in range(2)) / 2
    assert torch.allclose(value, rowwise)
    return {'passed': True, 'lower_complete_mask_cost_preferred': True, 'ties_have_zero_ranking_weight': True,
        'finite_gradients': True, 'weighted_microbatch_equivalence': True}
