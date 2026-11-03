"""Two-GPU calibration and held-out-block evaluation; all outputs are new files.

Run inside run_with_resources_v2.py. No training or original-source writes occur.
The calibration objective matches stored labels: mean last-token KL(full||pruned).
Tests additionally measure every valid prediction position and next-token NLL.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from check_checkpoint_direction import choose_skips, load_class
from evaluate_router_corrected import bypass_layers, raw_embeddings


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def stamp(event, **values):
    print(json.dumps({'event': event, 'time': datetime.now().astimezone().isoformat(), **values}), flush=True)


def prepare_data(args, output):
    import zarr
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, local_files_only=True)
    # Explicitly compare stored v3 prefixes; older feature-only stores lack token provenance.
    known = set()
    for path in sorted(args.source.glob('opt13b_greedy_routing_labels_v3_part_*')):
        group = zarr.open_group(str(path), mode='r')
        for row in range(len(group['seq_lens'])):
            if group['input_ids'].attrs.get(f'row_{row}_written', False):
                known.add(hashlib.sha256(np.asarray(group['input_ids'][row][:128], dtype='<i4').tobytes()).hexdigest())
    size = args.corpus.stat().st_size
    arrays, provenance, used = {}, [], set()
    for split, fraction, count in [('calibration', .5, args.calibration_samples),
                                    ('test', .75, args.test_samples)]:
        with args.corpus.open('rb') as stream:
            stream.seek(int(size * fraction))
            stream.readline()  # Align to an actual UTF-8 line boundary.
            start = stream.tell()
            raw = stream.read(2 * 1024 * 1024)
            # Avoid a partial trailing UTF-8 character or incomplete last line.
            raw = raw.rsplit(b'\n', 1)[0]
        text = raw.decode('utf-8', errors='strict')
        tokens = tokenizer.encode(text, add_special_tokens=False)
        width = args.sequence_length - 1
        if len(tokens) < count * width:
            raise ValueError('Corpus region is too short')
        starts = np.linspace(0, len(tokens) - width, count, dtype=int).tolist()
        rows = []
        for i, token_offset in enumerate(starts):
            ids = np.asarray([tokenizer.bos_token_id] + tokens[token_offset:token_offset+width], dtype=np.int32)
            prefix = hashlib.sha256(ids[:128].astype('<i4').tobytes()).hexdigest()
            digest = hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest()
            if prefix in known or prefix in used:
                raise ValueError('Duplicate or known latest-training prefix in evaluation data')
            used.add(prefix)
            rows.append(ids)
            provenance.append({'split': split, 'sample': i, 'source_region_start_byte': start,
                               'source_region_end_byte': start + len(raw),
                               'source_region_sha256': hashlib.sha256(raw).hexdigest(),
                               'token_offset_in_region': token_offset,
                               'content_token_count': width, 'input_tokens_with_bos': len(ids),
                               'token_ids_sha256': digest})
        if any(b-a < width for a, b in zip(starts, starts[1:])):
            raise ValueError('Selected token blocks overlap')
        arrays[split] = np.stack(rows)
    data_file = output / 'private_eval_token_ids.npz'
    with data_file.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    info = {'corpus': str(args.corpus), 'corpus_bytes': size,
            'corpus_mtime_ns': args.corpus.stat().st_mtime_ns,
            'data_file_sha256': sha256(data_file), 'sequence_length': args.sequence_length,
            'calibration_samples': args.calibration_samples, 'test_samples': args.test_samples,
            'latest_training_prefixes_checked': len(known), 'latest_training_prefix_matches': 0,
            'split_method': 'Nonoverlapping token blocks from separate middle and later corpus byte regions.',
            'limitations': ['Block independence, not verified document independence.',
                            'Legacy feature-only training data cannot be exhaustively checked for overlap.',
                            'The same local C4 corpus was used historically; this is not a new external dataset.'],
            'samples': provenance}
    write_json(output / 'data_manifest.json', info)
    return data_file


def model_logits(model, ids, indices):
    with bypass_layers(model, indices):
        return model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits


def quality_metrics(logits, teacher_log_probs, ids):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    pointwise = (teacher_log_probs.exp() * (teacher_log_probs - log_probs)).sum(dim=-1)
    nll = F.nll_loss(log_probs[:, :-1].reshape(-1, log_probs.size(-1)), ids[:, 1:].reshape(-1), reduction='mean')
    return {'last_token_kl': pointwise[:, -1].mean().item(),
            'all_token_mean_kl': pointwise.mean().item(),
            'prediction_position_mean_kl': pointwise[:, :-1].mean().item(),
            'next_token_nll': nll.item(), 'next_token_count': ids.numel()-ids.size(0)}


def legacy_class(snapshot):
    path = snapshot / 'legacy/skip/model_router.py'
    spec = importlib.util.spec_from_file_location('stage1_legacy_router', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RouterTransformer


def worker(rank, arguments, data_file, inbox, results):
    try:
        args = argparse.Namespace(**arguments)
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16,
                    local_files_only=True, attn_implementation='sdpa').to(device).eval()
        model.config.use_cache = False
        model.requires_grad_(False)
        router = load_class(args.snapshot / 'legacy/skip/model_router_transformer.py')(dropout=0.).to(device).eval()
        state = torch.load(args.router_checkpoint, weights_only=True, mmap=True, map_location='cpu')
        router.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()})
        del state
        legacy = legacy_class(args.snapshot)(input_dim=5120, num_opt_layers=40,
                    nhead=4, num_encoder_layers=2, d_hid=2048).to(device).eval()
        state = torch.load(args.legacy_checkpoint, weights_only=True, mmap=True, map_location='cpu')
        legacy.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()})
        del state
        data = np.load(data_file)
        calibration = [torch.tensor(row, dtype=torch.long, device=device).unsqueeze(0) for row in data['calibration']]
        calibration_teacher = []
        with torch.inference_mode():
            for ids in calibration:
                logits = model_logits(model, ids, [])[:, -1].float()
                calibration_teacher.append(F.softmax(logits, dim=-1))
        results.put({'kind':'ready','rank':rank,'pid':os.getpid(),
                     'cpu_ids':sorted(os.sched_getaffinity(0)), 'torch_threads':torch.get_num_threads(),
                     'gpu_name':torch.cuda.get_device_name(rank)})
        while True:
            task = inbox.get()
            if task is None:
                return
            if task['kind'] == 'calibration':
                values = []
                with torch.inference_mode():
                    for index in task['candidates']:
                        indices = task['selected'] + [index]
                        total = 0.
                        for ids, probs in zip(calibration, calibration_teacher):
                            logits = model_logits(model, ids, indices)[:, -1].float()
                            total += F.kl_div(F.log_softmax(logits, dim=-1), probs, reduction='sum').item()
                        values.append([index, total/len(calibration)])
                results.put({'kind':'calibration','rank':rank,'step':task['step'],'values':values})
            elif task['kind'] == 'test':
                for sample in task['samples']:
                    ids = torch.tensor(data['test'][sample], dtype=torch.long, device=device).unsqueeze(0)
                    with torch.inference_mode():
                        teacher_logits = model_logits(model, ids, [])
                        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
                        policies = [('full',0,[])]
                        embeddings = raw_embeddings(model, ids, torch.ones_like(ids))
                        for k in args.k:
                            keep_logits = router(embeddings, torch.tensor([k], device=device))[0]
                            policies.extend([('router_corrected', k, choose_skips(keep_logits,k).tolist()),
                                             ('router_reversed_diagnostic', k, choose_skips(keep_logits,k,True).tolist()),
                                             ('calibrated_static_greedy', k, task['static_order'][:k])])
                            for replicate in range(args.random_replicates):
                                rng = random.Random(args.seed + 100000*sample + 100*k + replicate)
                                policies.append((f'random_{replicate}', k, rng.sample(range(2,40), k)))
                        # Historical evaluation used raw token embeddings. Also explicitly
                        # test Layer-1 features, whose use was described by the v3 dataset.
                        if 6 in args.k:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                raw = model.get_input_embeddings()(ids)
                                scores = legacy(raw)[0]
                            policies.append(('legacy_rl_historical_tokens',6,choose_skips(scores,6).tolist()))
                            saved = []
                            class Captured(Exception):
                                pass
                            def capture(module, inputs, output):
                                saved.append(output[0] if isinstance(output,tuple) else output)
                                raise Captured()
                            handle = model.model.decoder.layers[1].register_forward_hook(capture)
                            try:
                                model(ids, use_cache=False)
                            except Captured:
                                pass
                            finally:
                                handle.remove()
                            if not saved:
                                raise RuntimeError('Layer-1 capture failed')
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                scores = legacy(saved[0])[0]
                            policies.append(('legacy_rl_layer1_features',6,choose_skips(scores,6).tolist()))
                        rows = []
                        for policy,k,indices in policies:
                            if k and (len(set(indices)) != k or min(indices)<2 or max(indices)>39):
                                raise ValueError('Invalid pruning mask')
                            logits = teacher_logits if k==0 else model_logits(model,ids,indices)
                            row = {'sample':sample,'rank':rank,'policy':policy,'k':k,
                                   'skipped_layers':sorted(indices), **quality_metrics(logits,teacher_log_probs,ids)}
                            if not all(math.isfinite(row[name]) for name in ['last_token_kl','all_token_mean_kl','next_token_nll']):
                                raise ValueError('Nonfinite metric')
                            rows.append(row)
                        results.put({'kind':'test_sample','rank':rank,'sample':sample,'rows':rows})
                        del teacher_logits,teacher_log_probs,embeddings
                results.put({'kind':'test_done','rank':rank})
            else:
                raise ValueError('Unknown task')
    except BaseException:
        results.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


def get_result(results, processes):
    while True:
        try:
            result = results.get(timeout=5)
            if result['kind'] == 'error':
                raise RuntimeError(result['traceback'])
            return result
        except queue.Empty:
            for process in processes:
                if process.exitcode is not None:
                    raise RuntimeError(f'Worker {process.pid} exited unexpectedly ({process.exitcode})')


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['policy'],row['k'])].append(row)
    summaries = []
    for (policy,k),subset in sorted(groups.items()):
        value = {'policy':policy,'k':k,'samples':len(subset)}
        for metric in ['last_token_kl','all_token_mean_kl','prediction_position_mean_kl','next_token_nll']:
            a = np.asarray([r[metric] for r in subset])
            value[metric] = {'mean':float(a.mean()),'median':float(np.median(a)),
                             'p90':float(np.quantile(a,.9)),
                             'standard_error':float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0.}
        value['perplexity'] = math.exp(value['next_token_nll']['mean'])
        masks = Counter(tuple(r['skipped_layers']) for r in subset)
        value['unique_masks'] = len(masks)
        value['most_common_mask_fraction'] = masks.most_common(1)[0][1]/len(subset)
        value['most_common_mask'] = list(masks.most_common(1)[0][0])
        value['layer_skip_frequency'] = {str(i):sum(i in r['skipped_layers'] for r in subset)/len(subset) for i in range(40)}
        summaries.append(value)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/home/liushenyu/test'))
    p.add_argument('--snapshot', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--base-model', type=Path, default=Path('/home/liushenyu/opt-13b'))
    p.add_argument('--corpus', type=Path, default=Path('/home/liushenyu/test/c4_subset_1024mb.txt'))
    p.add_argument('--router-checkpoint', type=Path, default=Path('/home/liushenyu/test/skip/transformer_router_raw_embed_epoch_20.pth'))
    p.add_argument('--legacy-checkpoint', type=Path, default=Path('/home/liushenyu/test/skip/router_rl_epoch_v4_4.pth'))
    p.add_argument('--calibration-samples',type=int,default=8)
    p.add_argument('--test-samples',type=int,default=256)
    p.add_argument('--sequence-length',type=int,default=256)
    p.add_argument('--k',type=int,nargs='+',default=[6,16])
    p.add_argument('--random-replicates',type=int,default=3)
    p.add_argument('--seed',type=int,default=579)
    args = p.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT','0'))!=2:
        p.error('Run through the resource launcher with exactly two GPUs')
    if args.sequence_length<128 or args.sequence_length>2048 or any(k<1 or k>16 for k in args.k):
        p.error('Sequence length must be 128..2048 and K 1..16')
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    for path in [args.router_checkpoint,args.legacy_checkpoint,args.corpus]:
        if not path.is_file():raise FileNotFoundError(path)
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    metadata={'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'program_sha256':sha256(__file__),'router_sha256':sha256(args.router_checkpoint),
              'legacy_router_sha256':sha256(args.legacy_checkpoint),
              'model_config_sha256':sha256(args.base_model/'config.json'),
              'torch_version':torch.__version__,'cpu_ids':sorted(os.sched_getaffinity(0)),
              'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'metrics_scope':'C4 blocks, not PIQA, no measured speed claim.'}
    write_json(output/'experiment.json',metadata)
    stamp('preparing_data')
    data_file=prepare_data(args,output)
    context=mp.get_context('spawn')
    results=context.Queue()
    queues=[context.Queue() for _ in range(2)]
    processes=[context.Process(target=worker,args=(rank,vars(args),data_file,queues[rank],results)) for rank in range(2)]
    rows=[]
    started=time.monotonic()
    try:
        for process in processes:process.start()
        ready=[get_result(results,processes) for _ in processes]
        if any(r['kind']!='ready' for r in ready):raise RuntimeError('Worker readiness protocol error')
        write_json(output/'workers.json',ready)
        stamp('workers_ready',workers=ready)
        selected=[]
        with (output/'static_calibration_steps.jsonl').open('x') as calibration_log:
            for step in range(max(args.k)):
                available=[i for i in range(2,40) if i not in selected]
                for rank in range(2):
                    queues[rank].put({'kind':'calibration','step':step+1,'selected':selected,
                                      'candidates':available[rank::2]})
                responses=[get_result(results,processes) for _ in processes]
                if any(r['kind']!='calibration' or r['step']!=step+1 for r in responses):
                    raise RuntimeError('Calibration protocol error')
                values=sorted([v for response in responses for v in response['values']])
                index,cost=min(values,key=lambda v:(v[1],v[0]))
                selected.append(index)
                result={'step':step+1,'layer':index,'mean_calibration_last_token_kl':cost,
                        'ordered_skips':selected.copy(),'candidate_costs':values,
                        'elapsed_seconds':time.monotonic()-started}
                calibration_log.write(json.dumps(result)+'\n');calibration_log.flush()
                stamp('static_calibration_step',**{k:v for k,v in result.items() if k!='candidate_costs'})
        write_json(output/'static_policy.json',{'ordered_skips':selected,'calibration_samples':args.calibration_samples,
                    'objective':'Mean last-token KL(full||pruned), iterative greedy across layers 2..39.',
                    'limitations':'Calibration-set greedy result, not a global optimum.'})
        for rank in range(2):
            queues[rank].put({'kind':'test','static_order':selected,'samples':list(range(rank,args.test_samples,2))})
        done=0;completed=0
        with (output/'sample_metrics.jsonl').open('x') as stream:
            while done<2:
                result=get_result(results,processes)
                if result['kind']=='test_done':done+=1;continue
                if result['kind']!='test_sample':raise RuntimeError('Test protocol error')
                for row in result['rows']:
                    stream.write(json.dumps(row)+'\n');rows.append(row)
                stream.flush();completed+=1
                stamp('test_progress',completed=completed,total=args.test_samples,
                      elapsed_seconds=time.monotonic()-started)
        write_json(output/'summary.json',{'metadata':metadata,'policies':summarize(rows),
                    'elapsed_seconds':time.monotonic()-started,'sample_count':args.test_samples,
                    'limitations':['No document-level independence guarantee.',
                                    'Calibration greedy does not imply global optimality.',
                                    'Legacy RL tested under both historical raw tokens and Layer-1 features.',
                                    'Quality metrics only; not a throughput benchmark.']})
        stamp('benchmark_completed',samples=args.test_samples,elapsed_seconds=time.monotonic()-started)
    finally:
        for inbox in queues:inbox.put(None)
        for process in processes:
            if process.pid is not None:process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=10)
            if process.is_alive():process.kill();process.join()


if __name__=='__main__':
    main()
