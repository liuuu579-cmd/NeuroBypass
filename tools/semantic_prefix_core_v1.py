"""Full-prompt two-layer pretrained features and exactly matched Stage25 router fits."""
import gc
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import cpu_fit_predictions, load_choice
from deep_swap_core_v1 import StaticSwapRouter
from full_context_experiment_v1 import write
from stage1_benchmark_v2 import sha256


def load_prefix(directory, device):
    from accelerate import init_empty_weights
    from transformers import OPTConfig
    from transformers.models.opt.modeling_opt import OPTDecoder
    import json
    directory = Path(directory)
    config = OPTConfig.from_pretrained(directory, local_files_only=True)
    assert config.hidden_size == 5120 and config.num_hidden_layers == 40
    config.num_hidden_layers = 2
    config._remove_final_layer_norm = True
    config._attn_implementation = 'sdpa'
    with init_empty_weights():
        prefix = OPTDecoder(config)
    mapping = json.loads((directory / 'pytorch_model.bin.index.json').read_text())['weight_map']
    required = {'decoder.' + key for key in prefix.state_dict()}
    tensors = {}
    for shard in sorted({mapping[key] for key in required}):
        values = torch.load(directory / shard, weights_only=True, map_location='cpu')
        tensors.update({key.removeprefix('decoder.'): values[key] for key in required if mapping[key] == shard})
        del values
    prefix.load_state_dict(tensors, strict=True, assign=True)
    del tensors
    prefix = prefix.to(device).eval().requires_grad_(False)
    assert len(prefix.layers) == 2 and prefix.final_layer_norm is None
    assert prefix.project_in is None and prefix.project_out is None
    return prefix


def hidden_prompt(prefix, prompt):
    assert prompt.ndim == 2 and prompt.shape[0] == 1 and 1 <= prompt.shape[1] <= 2048
    result = prefix(input_ids=prompt, attention_mask=torch.ones_like(prompt),
        use_cache=False, output_hidden_states=True, return_dict=True)
    assert len(result.hidden_states) == 3
    assert torch.equal(result.last_hidden_state, result.hidden_states[2])
    return result.last_hidden_state, result.hidden_states[0]


def fit_representation(out, cfg, fold, embeddings, labels, seed, device, messages, rank, arm, observe=True):
    name = f'{arm}_f{fold["fold"]}_s{seed}'
    fit_ids, selection = fold['fit_ids'], fold['selection_ids']
    costs = torch.tensor([labels[i]['kl'] for i in fit_ids], dtype=torch.float32)
    target = costs - costs[:, :1]; scale = target[:, 1:].std().clamp_min(.03).to(device)
    torch.manual_seed(seed)
    router = StaticSwapRouter(8, width=128, observe=observe).to(device)
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
                    torch.save({'state_dict': router.state_dict(), 'candidates': 8, 'width': 128, 'observe': observe,
                        'seed': seed, 'epoch': epoch, 'fold': fold['fold']}, stream)
                checkpoint = {'path': str(path), 'sha256': sha256(path), 'epoch': epoch, 'selection_kl': value}; best = value
        history.append(row)
        messages.put({'kind': 'progress', 'rank': rank, 'phase': 'neural_fit', 'arm': arm, 'fold': fold['fold'], 'arm': arm, 'seed': seed, **row})
    saved = load_choice(checkpoint['path'], device)
    all_ids = fit_ids + selection + fold['outer_ids']
    score = cpu_fit_predictions(saved, embeddings, all_ids, device).numpy()
    with (out / f'private_{name}_scores.pth').open('xb') as stream:
        torch.save({'ids': all_ids, 'scores': torch.from_numpy(score)}, stream)
    write(out / f'{name}_history.json', history)
    write(out / f'{name}_fit.json', {'best': checkpoint, 'fold': fold['fold'], 'arm': arm, 'seed': seed,
        'fit_documents': len(fit_ids), 'selection_documents': len(selection), 'outer_documents': len(fold['outer_ids']),
        'updates': total, 'initial_sha256': initial, 'prior': router.prior.detach().cpu().tolist(), 'scale': float(scale),
        'score_sha256': sha256(out / f'private_{name}_scores.pth')})
    del router, saved, optimizer
    gc.collect(); torch.cuda.empty_cache()

