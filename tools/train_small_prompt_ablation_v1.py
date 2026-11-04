"""Capacity-controlled prompt/no-prompt pilot, using train/validation only.

Two independent GPU workers fit equal-capacity routers, with common initialization,
document order and budget. The constant-input control trains on 800 documents,
making a stronger learned static control than the old 16-document calibration.
All source/checkpoints/results are appended to exclusive new paths.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_prompt_router_v2 import derangement, paired_summary
from prompt_fixed_core_v1 import (SequentialGreedyRouter, continuation_logits,
    continuation_metrics, raw_prompt_embeddings)
from stage1_benchmark_v2 import sha256, stamp, write_json
from train_prompt_fixed_v1 import targets_from_record


class ControlledRouter(SequentialGreedyRouter):
    def __init__(self, observe_prompt, width=64):
        super().__init__(width=width, dropout=.1)
        self.observe_prompt = observe_prompt

    def encode(self, embeddings):
        # Both arms have identical parameter count/architecture. This arm has
        # no prompt-dependent input at training, checkpoint selection or inference.
        return super().encode(embeddings if self.observe_prompt else torch.zeros_like(embeddings))


def select_validation(model, router, data, samples):
    records = []
    router.eval()
    with torch.inference_mode():
        for sample in samples:
            ids = torch.tensor(data[sample], dtype=torch.long, device=next(model.parameters()).device)[None]
            route = router.route(raw_prompt_embeddings(model, ids[:, :128]))[0].tolist()
            teacher = F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
            records.append(continuation_metrics(continuation_logits(model, ids, 128, route), teacher, ids, 128))
    return {name: float(np.mean([r[name] for r in records]))
            for name in ['continuation_kl', 'continuation_nll']}


def worker(rank, configuration, result_queue):
    try:
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        torch.manual_seed(configuration['seed'])
        np.random.seed(configuration['seed'])
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(configuration['base_model'],
            torch_dtype=torch.float16, local_files_only=True,
            attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False)
        data = np.load(configuration['data_file'], allow_pickle=False)
        records = configuration['labels']
        by_sample = {r['sample']: r for r in records}
        train_ids = sorted(r['sample'] for r in records if r['split'] == 'train')
        validation_ids = sorted(r['sample'] for r in records if r['split'] == 'validation')
        with torch.inference_mode():
            embeddings = torch.empty((800, 128, 5120), dtype=torch.float16)
            for offset in range(0, len(train_ids), 8):
                samples = train_ids[offset:offset + 8]
                ids = torch.tensor(data[samples, :128], dtype=torch.long, device=device)
                embeddings[offset:offset + len(samples)].copy_(raw_prompt_embeddings(model, ids).cpu())
        packed = [targets_from_record(by_sample[s]) for s in train_ids]
        states = torch.from_numpy(np.stack([p[0] for p in packed]))
        targets = torch.from_numpy(np.stack([p[1] for p in packed]))
        valid = torch.from_numpy(np.stack([p[2] for p in packed]))
        chosen = torch.from_numpy(np.stack([p[3] for p in packed]))
        arm = 'small_prompt' if rank == 0 else 'small_constant'
        # Reset after loading the frozen model for common initialization.
        torch.manual_seed(configuration['seed'])
        router = ControlledRouter(observe_prompt=(rank == 0)).to(device)
        initial_hash = __import__('hashlib').sha256(b''.join(
            t.detach().cpu().numpy().tobytes() for t in router.state_dict().values())).hexdigest()
        optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=.01)
        total_updates = configuration['epochs'] * 50
        def schedule(update):
            warmup = max(1, total_updates // 20)
            if update < warmup:
                return (update + 1) / warmup
            fraction = (update - warmup) / max(1, total_updates - warmup)
            return .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        output = Path(configuration['output'])
        best = float('inf')
        best_path = None
        history = []
        for epoch in range(1, configuration['epochs'] + 1):
            router.train()
            order = torch.randperm(800, generator=torch.Generator().manual_seed(
                configuration['seed'] + epoch * 10)).tolist()
            loss_sum = 0.
            for offset in range(0, 800, 16):
                indices = order[offset:offset + 16]
                batch_embeddings = embeddings[indices].to(device)
                batch_states = states[indices].to(device)
                batch_targets = targets[indices].to(device)
                batch_valid = valid[indices].to(device)
                batch_chosen = chosen[indices].to(device)
                steps = torch.arange(1, 7, device=device).expand(len(indices), -1)
                optimizer.zero_grad(set_to_none=True)
                prediction = router(batch_embeddings, batch_states, steps)
                logits = (-prediction).masked_fill(~batch_valid, -1e9)
                soft_loss = -(batch_targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                hard_loss = F.cross_entropy(logits.reshape(-1, 40), batch_chosen.reshape(-1))
                loss = soft_loss + .25 * hard_loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item()
            row = {'epoch': epoch, 'mean_loss': loss_sum / 50, 'lr': scheduler.get_last_lr()[0]}
            if epoch == 1 or epoch % 4 == 0:
                score = select_validation(model, router, data, configuration['selection_ids'])
                row.update(score)
                if score['continuation_kl'] < best:
                    best = score['continuation_kl']
                    best_path = output / f'private_{arm}_epoch_{epoch:03d}.pth'
                    with best_path.open('xb') as handle:
                        torch.save({'state_dict': router.state_dict(), 'epoch': epoch,
                            'width': 64, 'observe_prompt': rank == 0, 'validation': score}, handle)
            history.append(row)
            result_queue.put({'kind': 'progress', 'arm': arm, **row})
        router.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)['state_dict'])
        router.eval()
        donors = [derangement(validation_ids, configuration['seed'] + 100 + repeat) for repeat in range(4)]
        quality_train = configuration['quality_train_ids']
        populations = [('train_diagnostic', quality_train), ('validation', validation_ids)]
        routes = {}
        with torch.inference_mode():
            for split, samples in populations:
                for sample in samples:
                    ids = torch.tensor(data[sample, :128], dtype=torch.long, device=device)[None]
                    routes[sample] = router.route(raw_prompt_embeddings(model, ids))[0].tolist()
            if rank == 1 and len({tuple(path) for path in routes.values()}) != 1:
                raise ValueError('Constant input control produced different routes')
            quality_rows = []
            for split, samples in populations:
                for sample in samples:
                    ids = torch.tensor(data[sample], dtype=torch.long, device=device)[None]
                    teacher = F.log_softmax(continuation_logits(model, ids, 128, []), dim=-1)
                    policies = [('matched', routes[sample])]
                    if split == 'validation' and rank == 0:
                        policies += [(f'shuffled_{i}', routes[donors[i][sample]]) for i in range(4)]
                    for policy, path in policies:
                        logits = continuation_logits(model, ids, 128, path)
                        quality_rows.append({'sample': sample, 'split': split, 'arm': arm,
                            'policy': policy, 'skipped_layers': sorted(path),
                            **continuation_metrics(logits, teacher, ids, 128)})
        result_queue.put({'kind': 'done', 'arm': arm, 'history': history,
            'initial_parameter_sha256': initial_hash,
            'parameter_count': sum(p.numel() for p in router.parameters()),
            'best_checkpoint': str(best_path), 'best_checkpoint_sha256': sha256(best_path),
            'selection_kl': best, 'quality_rows': quality_rows,
            'routes': [{'sample': s, 'ordered_skips': p} for s, p in sorted(routes.items())],
            'worker': {'rank': rank, 'pid': os.getpid(), 'cpu_ids': sorted(os.sched_getaffinity(0)),
                       'threads': torch.get_num_threads()}})
    except BaseException:
        result_queue.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--diagnosis-run', type=Path, required=True)
    parser.add_argument('--base-model', default='/home/liushenyu/opt-13b')
    parser.add_argument('--seed', type=int, default=579)
    args = parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 2:
        parser.error('Use the two-GPU resource launcher')
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    started = time.monotonic()
    experiment = json.loads((args.main_run / 'experiment.json').read_text())
    old_protocol = json.loads((args.main_run / 'router_training_protocol.json').read_text())
    diagnosis_protocol = json.loads((args.diagnosis_run / 'protocol.json').read_text())
    records = [json.loads(line) for p in sorted(args.main_run.glob('labels_*.jsonl'))
               for line in p.read_text().splitlines()]
    if len(records) != 900 or any(r['split'] not in ['train', 'validation'] for r in records):
        raise ValueError('Unexpected labels')
    if sha256(experiment['data_file']) != experiment['data_sha256']:
        raise ValueError('Private input data changed')
    configuration = {'output': str(output), 'seed': args.seed, 'base_model': args.base_model,
        'data_file': experiment['data_file'], 'labels': records, 'epochs': 24,
        'selection_ids': old_protocol['checkpoint_validation_documents'],
        'quality_train_ids': diagnosis_protocol['quality_train_ids']}
    public = {k: v for k, v in configuration.items() if k != 'labels'}
    public.update({'source_sha256': sha256(__file__), 'test_used': False,
        'design': 'Two width64, dropout0.1 routers; real vs zero prompt embeddings; same initialization, batches, optimizer and budget.',
        'base_model_frozen': True, 'recovery_finetuning_performed': False,
        'label_objective': 'Existing joint conditional greedy continuation KL targets; soft CE + 0.25 hard CE.',
        'optimizer': 'AdamW lr2e-4, weight_decay0.01, clip1, warmup5pct cosine schedule',
        'batch_size': 16, 'updates_per_epoch': 50, 'checkpoint_selection': 'Existing 32 validation IDs, epochs1,4,...24.',
        'width256_reference': str(args.main_run), 'diagnosis_reference': str(args.diagnosis_run),
        'limitations': ['One seed only; development pilot, not a new held-out test.',
            'All validation has already been inspected during diagnosis; the remaining68 are only excluded from checkpoint selection.',
            'The old width256 reference used two-GPU DDP with different sharding; a width effect alone is not isolated.',
            'A constant-prompt trained router is a learned static control, not an optimized globally optimal static mask.',
            'C4 continuation and earlier data limitations apply.']})
    preserved = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
        if p.is_file() and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv'}
        and not p.is_relative_to(output) and '.git' not in p.parts}
    write_json(output / 'protocol.json', public)
    write_json(output / 'preservation_before.json', preserved)
    context = mp.get_context('spawn')
    messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, configuration, messages)) for rank in range(2)]
    results = []
    try:
        for process in processes:
            process.start()
        while len(results) < 2:
            try:
                message = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('A worker exited without complete results')
                continue
            if message['kind'] == 'error':
                raise RuntimeError(message['traceback'])
            if message['kind'] == 'progress':
                stamp('small_ablation_training', **{k: v for k, v in message.items() if k != 'kind'})
                continue
            results.append(message)
            arm = message['arm']
            write_json(output / f'{arm}_training_history.json', message['history'])
            write_json(output / f'{arm}_routes.json', message['routes'])
            with (output / f'{arm}_quality.jsonl').open('x') as handle:
                for row in message['quality_rows']:
                    handle.write(json.dumps(row) + '\n')
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Worker did not exit cleanly')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    if len({r['initial_parameter_sha256'] for r in results}) != 1:
        raise ValueError('Paired arms had different initial weights')
    by_arm = {r['arm']: r for r in results}
    by_key = {(r['arm'], row['sample'], row['policy']): row for r in results for row in r['quality_rows']}
    reference_rows = [json.loads(line) for p in sorted(args.diagnosis_run.glob('quality_rows_*.jsonl'))
                      for line in p.read_text().splitlines()]
    for row in reference_rows:
        if row['policy'] in ['router', 'static16', 'oracle_greedy', 'full']:
            by_key['reference_' + row['policy'], row['sample'], 'matched'] = row
    for sample in diagnosis_protocol['quality_validation_documents']:
        donor_rows = [by_key['small_prompt', sample, f'shuffled_{i}'] for i in range(4)]
        by_key['small_prompt_shuffled', sample, 'matched'] = {
            name: float(np.mean([r[name] for r in donor_rows]))
            for name in ['continuation_kl', 'continuation_nll']}
    populations = {'train_diagnostic64': configuration['quality_train_ids'],
        'validation_all100': diagnosis_protocol['quality_validation_documents'],
        'validation_selection32': configuration['selection_ids'],
        'validation_other68': [s for s in diagnosis_protocol['quality_validation_documents']
                               if s not in set(configuration['selection_ids'])]}
    quality = []
    comparisons = []
    for population, samples in populations.items():
        arms = ['small_prompt', 'small_constant', 'reference_router',
                'reference_static16', 'reference_oracle_greedy', 'reference_full']
        if population != 'train_diagnostic64':
            arms.append('small_prompt_shuffled')
        for arm in arms:
            group = [by_key[arm, sample, 'matched'] for sample in samples]
            mean_nll = float(np.mean([r['continuation_nll'] for r in group]))
            quality.append({'population': population, 'policy': arm, 'samples': len(samples),
                'continuation_kl': float(np.mean([r['continuation_kl'] for r in group])),
                'continuation_nll': mean_nll, 'perplexity': math.exp(mean_nll)})
        for reference in [a for a in arms if a not in ['small_prompt', 'reference_full']]:
            for metric in ['continuation_kl', 'continuation_nll']:
                comparisons.append({'population': population, 'reference': reference, 'metric': metric,
                    **paired_summary([by_key['small_prompt', s, 'matched'][metric] -
                                     by_key[reference, s, 'matched'][metric] for s in samples])})
    selected = [{k: r[k] for k in ['arm', 'initial_parameter_sha256', 'parameter_count',
        'best_checkpoint', 'best_checkpoint_sha256', 'selection_kl', 'worker']} for r in results]
    summary = {'quality': quality, 'paired_comparisons': comparisons, 'selected': selected,
               'elapsed_seconds': time.monotonic() - started, 'test_used': False}
    write_json(output / 'summary.json', summary)
    if any(sha256(root / name) != digest for name, digest in preserved.items()):
        raise ValueError('Preexisting source/evidence changed')
    write_json(output / 'preservation_after.json', {'verified_existing_files': len(preserved),
        'source_sha256': sha256(__file__), 'test_used': False,
        'data_sha256': sha256(experiment['data_file'])})
    stamp('small_ablation_complete', selected=selected, elapsed_seconds=summary['elapsed_seconds'])


if __name__ == '__main__':
    main()
