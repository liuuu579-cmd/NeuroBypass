"""Two-GPU prompt-fixed pilot, joint greedy labels, routing training and evaluation.

Invoke through run_with_resources_v2.py; each stage writes exclusively new paths.
Natural-text continuations are a first pilot, not a dialogue benchmark.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import socket
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import (FixedRouteSession,SequentialGreedyRouter,best_candidate,
    cached_continuation_logits,continuation_logits,continuation_metrics,raw_prompt_embeddings)
from prompt_fixed_data_v1 import load_records,prepare_records
from stage1_benchmark_v2 import sha256,stamp,write_json


def receive(results,processes):
    while True:
        try:
            record=results.get(timeout=5)
            if record['kind']=='error':raise RuntimeError(record['traceback'])
            return record
        except queue.Empty:
            for process in processes:
                if process.exitcode is not None:raise RuntimeError(f'Worker {process.pid} exited {process.exitcode}')


def cache_probe(model,ids,prompt_length,route):
    expected=continuation_logits(model,ids,prompt_length,route)
    actual,session=cached_continuation_logits(model,ids,prompt_length,route)
    a=F.log_softmax(actual,dim=-1);b=F.log_softmax(expected,dim=-1)
    kl=(b.exp()*(b-a)).sum(-1).abs().mean().item()
    error=(actual-expected).abs()
    position_kl=(b.exp()*(b-a)).sum(-1).abs()
    # Measured OPT FP16 serial/parallel rounding can move low-probability
    # logits by >0.125 while probability distributions remain very close.
    # Check both mean and worst-position distribution error, retaining raw errors.
    if kl>1e-4 or position_kl.max().item()>1e-3:raise ValueError('Cached decoding differs materially from prefix recomputation')
    expected_bytes=2*(len(model.model.decoder.layers)-len(route))*(ids.size(1)-1)*model.config.hidden_size*2
    if session.cache_bytes()!=expected_bytes:raise ValueError('Unexpected cache storage')
    return {'route':sorted(route),'cached_vs_recomputed_mean_kl':kl,'maximum_logit_abs_error':error.max().item(),
            'mean_logit_abs_error':error.mean().item(),'maximum_position_kl':position_kl.max().item(),
            'argmax_agreement':(actual.argmax(-1)==expected.argmax(-1)).float().mean().item(),
            'tolerance_basis':'Measured SDPA FP16 serial-versus-parallel error; mean KL <=1e-4 and max-position KL <=1e-3',
            'cache_bytes':session.cache_bytes(),
            'retained_cache_length':session.length,'all_layer_lengths_verified':True}


def worker(rank,arguments,data_file,inbox,results):
    try:
        args=argparse.Namespace(**arguments)
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']));torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed);torch.cuda.set_device(rank)
        device=torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.float16,
              local_files_only=True,attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False);data=np.load(data_file,allow_pickle=False)
        router=None
        results.put({'kind':'ready','rank':rank,'pid':os.getpid(),'cpu_ids':sorted(os.sched_getaffinity(0)),
                     'threads':torch.get_num_threads(),'gpu':torch.cuda.get_device_name(rank)})
        while True:
            task=inbox.get()
            if task is None:return
            kind=task['kind']
            if kind=='fit':
                from train_prompt_fixed_v1 import train_router
                outcome=train_router(rank,args,model,data,task,results)
                results.put({'kind':'fit_done','rank':rank,**outcome});continue
            with torch.inference_mode():
                if kind=='probe':
                    ids=torch.tensor(data[task['sample']],dtype=torch.long,device=device).unsqueeze(0)
                    probes=[cache_probe(model,ids,args.prompt_length,route) for route in [[],[2,5,9,11,14,18]]]
                    results.put({'kind':'probe_done','rank':rank,'sample':task['sample'],'probes':probes})
                elif kind=='label':
                    sample=task['sample'];started=time.monotonic()
                    ids=torch.tensor(data[sample],dtype=torch.long,device=device).unsqueeze(0)
                    teacher=F.log_softmax(continuation_logits(model,ids,args.prompt_length,[]),dim=-1)
                    selected=[];steps=[]
                    for step in range(1,args.k+1):
                        costs=[]
                        for index in range(2,model.config.num_hidden_layers):
                            if index in selected:continue
                            logits=continuation_logits(model,ids,args.prompt_length,selected+[index])
                            cost=(teacher.exp()*(teacher-F.log_softmax(logits,dim=-1))).sum(-1).mean().item()
                            costs.append([index,cost])
                        index,cost=best_candidate(costs)
                        steps.append({'step':step,'previous_skips':selected.copy(),'selected_layer':index,
                                      'joint_continuation_kl':cost,'candidate_costs':costs})
                        selected.append(index)
                    final=continuation_logits(model,ids,args.prompt_length,selected)
                    record={'sample':sample,'split':task['split'],'ordered_skips':selected,'steps':steps,
                            'keep_mask':[int(i not in selected) for i in range(model.config.num_hidden_layers)],
                            **continuation_metrics(final,teacher,ids,args.prompt_length),
                            'elapsed_seconds':time.monotonic()-started}
                    if task.get('cache_probe'):record['cache_probe']=cache_probe(model,ids,args.prompt_length,selected)
                    results.put({'kind':'label_done','rank':rank,'record':record})
                elif kind=='static':
                    ids_list=[torch.tensor(data[s],dtype=torch.long,device=device).unsqueeze(0) for s in task['samples']]
                    teachers=[F.log_softmax(continuation_logits(model,ids,args.prompt_length,[]),dim=-1) for ids in ids_list]
                    costs=[]
                    for index in task['candidates']:
                        values=[]
                        for ids,teacher in zip(ids_list,teachers):
                            logits=continuation_logits(model,ids,args.prompt_length,task['selected']+[index])
                            values.append((teacher.exp()*(teacher-F.log_softmax(logits,dim=-1))).sum(-1).mean().item())
                        costs.append([index,sum(values)/len(values)])
                    results.put({'kind':'static_done','rank':rank,'step':task['step'],'costs':costs})
                elif kind=='route':
                    router=SequentialGreedyRouter().to(device).eval()
                    checkpoint=torch.load(task['checkpoint'],map_location='cpu',weights_only=True)
                    router.load_state_dict(checkpoint['state_dict'])
                    records=[]
                    for sample in task['samples']:
                        ids=torch.tensor(data[sample,:args.prompt_length],dtype=torch.long,device=device).unsqueeze(0)
                        torch.cuda.synchronize();started=time.perf_counter()
                        path=router.route(raw_prompt_embeddings(model,ids),args.k)[0].tolist()
                        torch.cuda.synchronize()
                        records.append({'sample':sample,'ordered_skips':path,'router_seconds':time.perf_counter()-started})
                    results.put({'kind':'route_done','rank':rank,'records':records})
                elif kind=='evaluate':
                    sample=task['sample'];ids=torch.tensor(data[sample],dtype=torch.long,device=device).unsqueeze(0)
                    teacher_logits=continuation_logits(model,ids,args.prompt_length,[])
                    teacher=F.log_softmax(teacher_logits,dim=-1);rows=[]
                    for policy,route in task['policies']:
                        logits=teacher_logits if not route else continuation_logits(model,ids,args.prompt_length,route)
                        rows.append({'sample':sample,'policy':policy,'k':len(route),'skipped_layers':sorted(route),
                                     **continuation_metrics(logits,teacher,ids,args.prompt_length)})
                    results.put({'kind':'evaluation_done','rank':rank,'rows':rows})
                elif kind=='benchmark':
                    sample=task['sample'];prompt=torch.tensor(data[sample,:args.prompt_length],dtype=torch.long,device=device).unsqueeze(0)
                    timing=[];generated=[]
                    # One warmup and three measured repetitions, fixed output length.
                    for policy,preselected in task['policies']:
                        for repetition in range(-1,3):
                            torch.cuda.synchronize();start=time.perf_counter()
                            route=router.route(raw_prompt_embeddings(model,prompt),args.k)[0].tolist() if policy=='router' else preselected
                            torch.cuda.synchronize();routed=time.perf_counter()
                            session=FixedRouteSession(model,route)
                            logits=session.forward(prompt)[:,-1:];token=logits.argmax(-1)
                            torch.cuda.synchronize();prefilled=time.perf_counter()
                            tokens=[token.item()]
                            for _ in range(task['new_tokens']-1):
                                token=session.forward(token)[:,-1:].argmax(-1);tokens.append(token.item())
                            torch.cuda.synchronize();finished=time.perf_counter()
                            if repetition>=0:
                                timing.append({'sample':sample,'policy':policy,'repetition':repetition,
                                               'new_tokens':task['new_tokens'],'route_seconds':routed-start,
                                               'prefill_seconds':prefilled-routed,'decode_seconds':finished-prefilled,
                                               'total_seconds':finished-start,'cache_bytes':session.cache_bytes(),
                                               'skipped_layers':sorted(route),'eos_count':tokens.count(model.config.eos_token_id)})
                                generated.append({'sample':sample,'policy':policy,'repetition':repetition,'token_ids':tokens})
                    results.put({'kind':'benchmark_done','rank':rank,'rows':timing,'private_generations':generated})
                else:raise ValueError('Unknown worker task '+kind)
    except BaseException:results.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['policy']].append(row)
    output=[]
    for name,group in sorted(groups.items()):
        result={'policy':name,'samples':len(group),'k':group[0]['k']}
        for metric in ['continuation_kl','first_next_token_kl','continuation_nll']:
            result[metric]=float(np.mean([r[metric] for r in group]))
        result['perplexity']=math.exp(result['continuation_nll'])
        result['unique_masks']=len({tuple(r['skipped_layers']) for r in group})
        output.append(result)
    return output


def paired(rows,policy,reference,seed=579):
    selected={r['sample']:r for r in rows if r['policy']==policy};control={r['sample']:r for r in rows if r['policy']==reference}
    if set(selected)!=set(control):raise ValueError('Unpaired samples')
    ids=sorted(selected);rng=np.random.default_rng(seed);indices=rng.integers(0,len(ids),(10000,len(ids)));result=[]
    for metric in ['continuation_kl','continuation_nll']:
        differences=np.array([selected[s][metric]-control[s][metric] for s in ids])
        result.append({'policy':policy,'reference':reference,'metric':metric,'samples':len(ids),
                       'mean_delta':float(differences.mean()),'bootstrap_95pct':np.quantile(differences[indices].mean(1),[.025,.975]).tolist(),
                       'fraction_better':float((differences<0).mean())})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pilot','main'],required=True)
    parser.add_argument('--base-model',type=Path,default=Path('/home/liushenyu/opt-13b'))
    parser.add_argument('--data-run',type=Path)
    parser.add_argument('--static-run',type=Path)
    parser.add_argument('--prompt-length',type=int,default=128)
    parser.add_argument('--continuation-length',type=int,default=32)
    parser.add_argument('--k',type=int,default=6)
    parser.add_argument('--seed',type=int,default=579)
    parser.add_argument('--epochs',type=int,default=24)
    args=parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT','0'))!=2:parser.error('Use the two-GPU resource launcher')
    if args.k!=6 or args.prompt_length!=128 or args.continuation_length!=32:parser.error('Initial protocol is fixed at prompt128/continuation32/K6')
    root=Path(__file__).resolve().parents[1];output=Path(os.environ['NEUROBYPASS_RUN_DIR'])
    if args.data_run:manifest,data_file=load_records(args.data_run)
    else:manifest,data_file=prepare_records(root,output,args.base_model,args.prompt_length,args.continuation_length,args.seed)
    experiment={'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'data_file':str(data_file),'data_sha256':manifest['data_sha256'],'split_counts':manifest['split_counts'],
                'source_sha256':{p.name:sha256(p) for p in [Path(__file__),root/'tools/prompt_fixed_core_v1.py',root/'tools/prompt_fixed_data_v1.py']},
                'cpu_ids':sorted(os.sched_getaffinity(0)),'gpu_visibility':os.environ['CUDA_VISIBLE_DEVICES'],
                'label_objective':'Teacher KL averaged over 32 continuation predictions; same jointly selected route on prompt and continuation.',
                'router_observation':'Prompt token and position embeddings only; no continuation is passed to router.',
                'limitations':manifest['limitations']+['Natural continuation is not a chat or downstream task benchmark.']}
    write_json(output/'experiment.json',experiment)
    splits={name:[r['sample'] for r in manifest['records'] if r['split']==name] for name in manifest['split_counts']}
    context=mp.get_context('spawn');results=context.Queue();inboxes=[context.Queue() for _ in range(2)]
    processes=[context.Process(target=worker,args=(rank,vars(args),data_file,inboxes[rank],results)) for rank in range(2)]
    started=time.monotonic()
    try:
        for process in processes:process.start()
        ready=[receive(results,processes) for _ in processes]
        if any(r['kind']!='ready' for r in ready):raise ValueError('Worker readiness mismatch')
        write_json(output/'workers.json',ready);stamp('prompt_fixed_workers_ready',workers=ready)
        for rank in range(2):inboxes[rank].put({'kind':'probe','sample':splits['pilot'][rank]})
        probes=[receive(results,processes) for _ in range(2)]
        if any(r['kind']!='probe_done' for r in probes):raise ValueError('Cache probe mismatch')
        write_json(output/'cache_probes.json',probes);stamp('cache_probes_passed')
        static=[]
        if args.static_run:
            policy=json.loads((args.static_run/'static_policy.json').read_text())
            if policy['data_sha256']!=manifest['data_sha256']:raise ValueError('Static policy uses different data')
            static=policy['ordered_skips'];write_json(output/'static_policy.json',policy)
        else:
            with (output/'static_search_steps.jsonl').open('x') as handle:
                for step in range(1,args.k+1):
                    candidates=[i for i in range(2,40) if i not in static]
                    for rank in range(2):inboxes[rank].put({'kind':'static','samples':splits['calibration'],'step':step,
                                                         'selected':static.copy(),'candidates':candidates[rank::2]})
                    replies=[receive(results,processes) for _ in range(2)]
                    if any(r['kind']!='static_done' or r['step']!=step for r in replies):raise ValueError('Static protocol mismatch')
                    costs=sorted([c for reply in replies for c in reply['costs']]);index,cost=best_candidate(costs)
                    previous=static.copy();static.append(index)
                    handle.write(json.dumps({'step':step,'previous_skips':previous,'selected_layer':index,
                                              'ordered_skips':static.copy(),'mean_joint_continuation_kl':cost,'candidate_costs':costs})+'\n');handle.flush()
                    stamp('prompt_fixed_static_step',step=step,layer=index,kl=cost)
            write_json(output/'static_policy.json',{'ordered_skips':static,'calibration_samples':splits['calibration'],
                       'data_sha256':manifest['data_sha256'],'objective':experiment['label_objective']})
        selected=splits['pilot'] if args.mode=='pilot' else splits['train']+splits['validation']
        labels=[]
        for i,sample in enumerate(selected):inboxes[i%2].put({'kind':'label','sample':sample,
                     'split':manifest['records'][sample]['split'],'cache_probe':args.mode=='pilot' and i<2})
        # Rotate evidence into small shards for review/publication. Private token
        # arrays and checkpoints are never part of these label records.
        for offset in range(0,len(selected),50):
            with (output/f'labels_{offset//50:03d}.jsonl').open('x') as handle:
                for _ in range(min(50,len(selected)-offset)):
                    reply=receive(results,processes)
                    if reply['kind']!='label_done':raise ValueError('Label protocol mismatch')
                    labels.append(reply['record']);handle.write(json.dumps(reply['record'])+'\n');handle.flush()
                    if len(labels)%10==0 or len(labels)==len(selected):
                        stamp('prompt_fixed_labels',completed=len(labels),total=len(selected),last_sample=reply['record']['sample'],
                              elapsed_seconds=time.monotonic()-started)
        write_json(output/'label_summary.json',{'samples':len(labels),'split_counts':dict((s,sum(r['split']==s for r in labels)) for s in ['train','validation','pilot']),
                   'mean_seconds_per_sample_per_gpu':float(np.mean([r['elapsed_seconds'] for r in labels])),
                   'mean_final_continuation_kl':float(np.mean([r['continuation_kl'] for r in labels])),
                   'unique_masks':len({tuple(sorted(r['ordered_skips'])) for r in labels})})
        if args.mode=='pilot':
            evaluations=splits['pilot'];mask_map={r['sample']:r['ordered_skips'] for r in labels}
            for i,sample in enumerate(evaluations):inboxes[i%2].put({'kind':'evaluate','sample':sample,
                                        'policies':[('full',[]),('static',static),('per_input_greedy',mask_map[sample])]})
        else:
            # The checkpoint selection uses a fixed validation subset only.
            with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
            fit={'kind':'fit','output':str(output),'manifest':manifest,'static':static,'epochs':args.epochs,
                 'init_method':f'tcp://127.0.0.1:{port}','validation_ids':splits['validation'][:32]}
            for inbox in inboxes:inbox.put(fit)
            outcomes=[]
            while len(outcomes)<2:
                reply=receive(results,processes)
                if reply['kind']=='fit_done':outcomes.append(reply)
                elif reply['kind']=='fit_progress':stamp('router_fit_progress',**{k:v for k,v in reply.items() if k!='kind'})
                else:raise ValueError('Training protocol mismatch')
            checkpoint=next(r['checkpoint'] for r in outcomes if r['rank']==0)
            write_json(output/'training_outcomes.json',outcomes)
            for rank in range(2):inboxes[rank].put({'kind':'route','checkpoint':checkpoint,'samples':splits['test'][rank::2]+([splits['train'][0]] if rank==0 else [])})
            routed=[receive(results,processes) for _ in range(2)]
            if any(r['kind']!='route_done' for r in routed):raise ValueError('Router protocol mismatch')
            route_records=[r for reply in routed for r in reply['records']]
            write_json(output/'test_router_decisions.json',route_records)
            mask_map={r['sample']:r['ordered_skips'] for r in route_records};evaluations=splits['test']
            order=evaluations.copy();random.Random(args.seed+2).shuffle(order)
            donors={sample:order[(i+1)%len(order)] for i,sample in enumerate(order)}
            write_json(output/'shuffled_donors.json',donors)
            for i,sample in enumerate(evaluations):inboxes[i%2].put({'kind':'evaluate','sample':sample,'policies':[
                ('full',[]),('static',static),('router',mask_map[sample]),('shuffled_router',mask_map[donors[sample]]),
                ('fixed_input_router',mask_map[splits['train'][0]])]})
        rows=[]
        with (output/'sample_metrics.jsonl').open('x') as handle:
            for _ in evaluations:
                reply=receive(results,processes)
                if reply['kind']!='evaluation_done':raise ValueError('Evaluation protocol mismatch')
                for row in reply['rows']:rows.append(row);handle.write(json.dumps(row)+'\n')
                handle.flush()
        comparisons=[]
        if args.mode=='pilot':comparisons+=paired(rows,'per_input_greedy','static')
        else:
            for reference in ['static','shuffled_router','fixed_input_router']:comparisons+=paired(rows,'router',reference)
        # Measure free autoregressive rollout; EOS does not terminate the fixed
        # workload. GPU timing includes synchronization and Python dispatch.
        timing_ids=evaluations[:(4 if args.mode=='pilot' else 8)]
        for i,sample in enumerate(timing_ids):
            policies=[('full',[]),('static',static)]
            if args.mode=='main':policies.append(('router',mask_map[sample]))
            inboxes[i%2].put({'kind':'benchmark','sample':sample,'policies':policies,'new_tokens':32 if args.mode=='pilot' else 64})
        timing=[]
        with (output/'private_generated_token_ids.jsonl').open('x') as private,(output/'generation_timings.jsonl').open('x') as public:
            for _ in timing_ids:
                reply=receive(results,processes)
                if reply['kind']!='benchmark_done':raise ValueError('Benchmark protocol mismatch')
                for row in reply['rows']:timing.append(row);public.write(json.dumps(row)+'\n')
                for row in reply['private_generations']:private.write(json.dumps(row)+'\n')
                public.flush();private.flush()
        timing_summary=[]
        for policy in sorted({r['policy'] for r in timing}):
            group=[r for r in timing if r['policy']==policy]
            timing_summary.append({'policy':policy,'measured_runs':len(group),
                                  **{name:float(np.median([r[name] for r in group])) for name in ['route_seconds','prefill_seconds','decode_seconds','total_seconds','cache_bytes']}})
        write_json(output/'summary.json',{'mode':args.mode,'policies':summarize(rows),'paired_comparisons':comparisons,
                   'generation_timing_medians':timing_summary,'elapsed_seconds':time.monotonic()-started,
                   'scope':'Prompt-fixed K6 natural-text continuation; fixed-length free generation timing, EOS ignored for workload equality.',
                   'limitations':experiment['limitations']+['Timing workloads share a server and use only a few prompts; this is not a serving throughput benchmark.']})
        stamp('prompt_fixed_completed',mode=args.mode,elapsed_seconds=time.monotonic()-started)
    finally:
        for inbox in inboxes:inbox.put(None)
        for process in processes:
            if process.pid is not None:process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=10)
            if process.is_alive():process.kill();process.join()


if __name__=='__main__':main()
