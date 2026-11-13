"""Finite matched supervision probe/main; reuse verified Stage27 joint masks."""
import os
# The resource launcher sets BLAS threads to10. Override before NumPy imports;
# Torch compute threads are then explicitly10 inside each pinned worker.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from continuation_stability_core_v1 import digest, token_metrics
from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers, preserve, read_lines, resources, setup_worker, thread_record, verify_preserved, write
from horizontal_quality_v3 import pin_threads
from prompt_fixed_core_v1 import continuation_logits
from semantic_prefix_core_v1 import fit_representation
from stage1_benchmark_v2 import sha256, stamp
from window_supervision_core_v1 import ARMS, finalize, labels_for_fit, load, make_folds, prepare_all, report


def worker(rank, cfg, inbox, messages):
    from transformers import AutoModelForCausalLM
    setup_worker(rank); device = f'cuda:{rank}'; out = Path(cfg['output'])
    bank = load(Path(cfg['old_run']) / 'candidate_bank.json')
    inputs = {r['sample']: r for r in read_lines(cfg['input_file'])}
    cached = {}; cache_sources = {}
    for directory in cfg['cache_runs']:
        directory = Path(directory)
        for j in range(4):
            for row in read_lines(directory / f'private_labels_{j:03d}.jsonl'):
                sample = row['sample']
                if sample in cfg['ids']:
                    assert row['extended_sequence_sha256'] == inputs[sample]['extended_sequence_sha256']
                    if sample in cached:
                        assert cached[sample]['kl'] == row['kl'] and cached[sample]['nll'] == row['nll']
                    cached[sample] = row; cache_sources[sample] = str(directory)
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    reused = 0; computed = 0; maximum = 0.; sources = {}; saved_tokens = {}
    with torch.inference_mode(), (out / f'private_labels_{rank:03d}.jsonl').open('x') as stream:
        for offset, sample in enumerate(cfg['ids'][rank::4]):
            row = inputs[sample]; p = row['prompt_length']
            assert digest(row['ids']) == row['extended_sequence_sha256']
            assert digest(row['ids'][:p]) == row['prompt_sha256']
            if sample not in cached or offset == 0:
                ids = torch.tensor(row['ids'], dtype=torch.long, device=device)[None]
                teacher = continuation_logits(model, ids, p, []).log_softmax(-1)
                kl = []; nll = []
                for route in bank['routes']:
                    a, b = token_metrics(continuation_logits(model, ids, p, route['skips']), teacher, ids[:, p:])
                    kl.append(a); nll.append(b)
                kt = torch.stack(kl, 1); nt = torch.stack(nll, 1)
                result = {k: row[k] for k in ('sample', 'prompt_length', 'prompt_sha256', 'extended_sequence_sha256')}
                result.update(kl=kt.reshape(4, 32, 8).mean(1).double().tolist(), nll=nt.reshape(4, 32, 8).mean(1).double().tolist())
                if sample in cached:
                    error = max(float(np.max(np.abs(np.asarray(result[k]) - cached[sample][k]))) for k in ('kl', 'nll'))
                    maximum = max(maximum, error); assert error == 0
                saved_tokens[sample] = {'kl': kt, 'nll': nt}; computed += 1
            else:
                result = cached[sample]; reused += 1; sources[cache_sources[sample]] = sources.get(cache_sources[sample], 0) + 1
            stream.write(__import__('json').dumps(result, allow_nan=False) + '\n'); stream.flush()
            if offset % 8 == 0:
                messages.put({'kind': 'progress', 'rank': rank, 'phase': 'window_labels',
                    'documents': offset+1, 'total': len(cfg['ids'][rank::4]), 'computed': computed, 'reused': reused})
    with (out / f'private_new_token_metrics_{rank:03d}.pth').open('xb') as stream:
        torch.save(saved_tokens, stream)
    write(out / f'labels_worker_{rank:03d}.json', {'passed': True, 'computed': computed, 'reused': reused,
        'reuse_sources': sources, 'maximum_cache_replay_error': maximum,
        'labels_sha256': sha256(out / f'private_labels_{rank:03d}.jsonl'),
        'tokens_sha256': sha256(out / f'private_new_token_metrics_{rank:03d}.pth'), **thread_record()})
    del model, teacher, saved_tokens; gc.collect(); torch.cuda.empty_cache()
    messages.put({'kind': 'labels_done', 'rank': rank}); assert inbox.get()['kind'] == 'fit'
    values = torch.load(Path(cfg['old_run']) / 'private_embeddings.pth', weights_only=True, map_location='cpu')
    embeddings = {i: values[i] for i in cfg['ids']}; del values
    assert all(value.shape == (1, inputs[i]['prompt_length'], 5120) for i, value in embeddings.items())
    labels = {r['sample']: r for j in range(4) for r in read_lines(out / f'private_labels_{j:03d}.jsonl')}
    fold = cfg['folds'][rank]
    for seed in cfg['seeds']:
        for arm in ARMS:
            fit_representation(out, cfg, fold, embeddings, labels_for_fit(labels, fold, arm), seed,
                device, messages, rank, arm, observe=arm.endswith('_embedding'))
    write(out / f'worker_{rank:03d}.json', {'passed': True, 'seeds': cfg['seeds'], 'fold': rank, **thread_record()})
    messages.put({'kind': 'done', 'rank': rank}); assert inbox.get() is None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true'); parser.add_argument('--probe-run', type=Path); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    pin_threads(resources()[40:42]); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    preserved = preserve(root, out); old = root / 'runs/stage20_codebook_refine_20261112_02_main'
    stage27 = root / 'runs/stage27_continuation_stability_20261113_01_main'; previous = load(old / 'protocol.json')
    if args.probe:
        input_file, sampling = prepare_all(out, previous)
    else:
        assert args.probe_run and load(args.probe_run / 'verification.json')['passed']
        probe_cfg = load(args.probe_run / 'protocol.json'); input_file = Path(probe_cfg['input_file'])
        assert sha256(input_file) == probe_cfg['input_file_sha256']
        sampling = load(args.probe_run / 'sampling.json'); write(out / 'sampling.json', sampling)
    data = Records(previous['data_run']); ids = sampling['eligible_ids']; folds = make_folds(ids, data.rows)
    if args.probe:
        cached = set(load(stage27 / 'protocol.json')['ids'])
        folds = [{**f, **{key: [i for i in f[key] if i in cached][:count]
            for key, count in (('fit_ids', 16), ('selection_ids', 8), ('outer_ids', 8))}} for f in folds]
        assert all(len(f['fit_ids']) == 16 and len(f['selection_ids']) >= 4 and len(f['outer_ids']) == 8 for f in folds)
        ids = sorted({i for f in folds for key in ('fit_ids', 'selection_ids', 'outer_ids') for i in f[key]})
    bank = load(old / 'candidate_bank.json'); assert bank['k'] == 20 and len(bank['routes']) == 8
    assert all(len(set(r['skips'])) == 20 for r in bank['routes'])
    cache_runs = [str(stage27)] + ([str(args.probe_run)] if not args.probe else [])
    sources = ['window_supervision_core_v1.py', 'run_window_supervision_four_gpu_v1.py', 'launch_window_supervision_v1.py',
        'semantic_prefix_core_v1.py', 'deep_swap_core_v1.py', 'crossfit_routing_core_v1.py', 'prompt_fixed_core_v1.py',
        'continuation_stability_core_v1.py', 'full_context_experiment_v1.py', 'run_with_resources_v2.py', 'adaptive_choice_core_v3.py']
    evidence = [old / 'protocol.json', old / 'candidate_bank.json', old / 'private_embeddings.pth',
        Path(previous['data_run']) / 'data_manifest.json']
    for directory in cache_runs:
        evidence += [Path(directory) / 'protocol.json', Path(directory) / 'verification.json']
        evidence += [Path(directory) / f'private_labels_{j:03d}.jsonl' for j in range(4)]
    inherited = load(stage27 / 'protocol.json')['evidence_sha256']
    evidence += [Path(p) for p in inherited if '/opt-13b/' in p or p.endswith('c4_subset_1024mb.txt')]
    cfg = {'output': str(out), 'old_run': str(old), 'model': previous['model'], 'data_run': previous['data_run'],
        'probe': args.probe, 'ids': ids, 'folds': folds, 'seeds': [579] if args.probe else [579, 580, 581],
        'epochs': 1 if args.probe else 24, 'arms': list(ARMS), 'cache_runs': cache_runs,
        'input_file': str(input_file), 'input_file_sha256': sha256(input_file),
        'source_sha256': {name: sha256(root / 'tools' / name) for name in sources},
        'evidence_sha256': {str(p): sha256(p) for p in evidence},
        'fold_rule': 'record/lead connected components; four balanced outer folds seed280579; inner35 seed280580+fold.',
        'fit': 'Same24 epochs, seeds, architecture, initialization parameters, ordering and optimizer as existing fitter; train target first32 vs mean4x32 from SAME long-sequence forward. Target-dependent fit-only prior/scale.',
        'observation': 'All original prompt embeddings and positions; no continuation input; one complete K20 mask per response.',
        'selection': 'COMMON128-token inner KL for all checkpoints (earliest minimum epochs0/1/4/...24) and gates; static baseline and gate scale from fit128 only.',
        'primary_comparison': 'mean128_embedding_gate_mean3 vs first32_embedding_gate_mean3 on outer all128 KL; also matched mean128_constant_gate_mean3 and fold_static.',
        'limits': 'Development diagnostic on previously reviewed training data, historically selected bank; eligibility-conditioned351. Document bootstrap ignores training/fold dependence and multiplicity. Original dev100 excluded. No speed claims.'}
    for path, expected in inherited.items():
        if path in cfg['evidence_sha256']:
            assert cfg['evidence_sha256'][path] == expected
    assert not set(ids) & set(previous['validation_ids'])
    write(out / 'protocol.json', cfg); stamp('window_supervision_start', probe=args.probe, documents=len(ids))
    with Workers(worker, cfg) as workers:
        workers.receive(4, 'labels_done'); workers.send_all({'kind': 'fit'}); workers.receive(4, 'done')
    summary = finalize(out, cfg)
    for path, expected in cfg['evidence_sha256'].items():
        assert sha256(path) == expected
    verify_preserved(out, preserved)
    write(out / 'verification.json', {'passed': True, 'probe': args.probe, 'full_prompt': True,
        'common128_selection': True, 'no_future_observation': True, 'joint_complete_masks': True, 'old_evidence_preserved': True})
    if not args.probe:
        report(root, out, cfg, summary)
    stamp('window_supervision_complete', run=str(out), probe=args.probe)


if __name__ == '__main__':
    main()
