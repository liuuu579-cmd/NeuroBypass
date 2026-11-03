"""Per-input greedy quality references and shuffled-input controls on 32 blocks.

Searches are expensive diagnostics using the full teacher at decision time. They
are not a deployable router, global optimum, or an unbiased quality upper bound.
All input/source files are read-only; new evidence is written under the launcher.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_router_corrected import bypass_layers
from stage1_benchmark_v2 import model_logits, quality_metrics, sha256, stamp, summarize, write_json


def choose_best_candidate(costs):
    """Return a deterministic minimum without silently accepting invalid scores."""
    if not costs or any(not math.isfinite(float(value)) for _,value in costs):
        raise ValueError('Empty or nonfinite candidate costs')
    if len({int(index) for index,_ in costs}) != len(costs):
        raise ValueError('Duplicate candidate layers')
    return min(costs, key=lambda item: (item[1],item[0]))


def shuffled_donors(samples, seed):
    """Random cyclic permutation gives a derangement with each mask used once."""
    if len(set(samples)) != len(samples) or len(samples)<2:
        raise ValueError('At least two distinct samples required')
    order=list(samples)
    random.Random(seed).shuffle(order)
    return {sample:order[(i+1)%len(order)] for i,sample in enumerate(order)}


def last_only_logits(model,ids,indices):
    """Avoid materializing vocabulary logits for all prefix positions in search."""
    with bypass_layers(model,indices):
        state=model.model(ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        return model.lm_head(state[:,-1:]).float()


def evaluate_candidate(model,ids,indices,teacher_log_probs,objective):
    if objective=='last_token':
        logits=last_only_logits(model,ids,indices)
        reference=teacher_log_probs[:,-1:]
    elif objective=='all_tokens':
        logits=model_logits(model,ids,indices).float()
        reference=teacher_log_probs
    else:
        raise ValueError('Unknown objective')
    log_probs=F.log_softmax(logits,dim=-1)
    return (reference.exp()*(reference-log_probs)).sum(dim=-1).mean().item()


def worker(rank,arguments,tasks,results):
    try:
        args=argparse.Namespace(**arguments)
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(rank)
        device=torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.float16,
                        local_files_only=True,attn_implementation='sdpa').to(device).eval()
        model.config.use_cache=False
        model.requires_grad_(False)
        data=np.load(args.stage1_run/'private_eval_token_ids.npz')['test']
        results.put({'kind':'ready','rank':rank,'pid':os.getpid(),
                     'cpu_ids':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),
                     'gpu_name':torch.cuda.get_device_name(rank)})
        while True:
            task=tasks.get()
            if task is None:return
            sample=task['sample']
            ids=torch.tensor(data[sample],dtype=torch.long,device=device).unsqueeze(0)
            with torch.inference_mode():
                full=model_logits(model,ids,[])
                teacher_log_probs=F.log_softmax(full.float(),dim=-1)
                if task['kind']=='search':
                    selected=[]
                    started=time.monotonic()
                    for step in range(1,max(args.k)+1):
                        costs=[]
                        for index in range(2,model.config.num_hidden_layers):
                            if index in selected:continue
                            value=evaluate_candidate(model,ids,selected+[index],teacher_log_probs,task['objective'])
                            costs.append((index,value))
                        index,cost=choose_best_candidate(costs)
                        selected.append(index)
                        sorted_costs=sorted(costs,key=lambda row:(row[1],row[0]))
                        record={'kind':'search_step','sample':sample,'rank':rank,
                                'objective':task['objective'],'step':step,'selected_layer':index,
                                'objective_kl':cost,'second_best_gap':sorted_costs[1][1]-cost,
                                'ordered_skips':selected.copy(),'candidate_costs':costs,
                                'search_elapsed_seconds':time.monotonic()-started}
                        if step in args.k:
                            pruned=model_logits(model,ids,selected)
                            metrics=quality_metrics(pruned,teacher_log_probs,ids)
                            metric_name='last_token_kl' if task['objective']=='last_token' else 'all_token_mean_kl'
                            # Last-only lm_head is mathematically equivalent, with some
                            # FP16 GEMM rounding difference from the full matrix call.
                            if abs(metrics[metric_name]-cost)>max(.01,.02*abs(cost)):
                                raise RuntimeError('Search and full-logit metric disagree materially')
                            record['quality_metrics']=metrics
                        results.put(record)
                    results.put({'kind':'search_done','sample':sample,'rank':rank,
                                 'objective':task['objective'],'ordered_skips':selected,
                                 'elapsed_seconds':time.monotonic()-started})
                elif task['kind']=='transfer':
                    rows=[]
                    for policy,k,indices,extra in task['policies']:
                        if len(set(indices))!=k or any(index<2 or index>=40 for index in indices):
                            raise ValueError('Invalid skip mask')
                        logits=full if not indices else model_logits(model,ids,indices)
                        rows.append({'sample':sample,'rank':rank,'policy':policy,'k':k,
                                     'skipped_layers':sorted(indices),**extra,
                                     **quality_metrics(logits,teacher_log_probs,ids)})
                    results.put({'kind':'transfer_done','sample':sample,'rank':rank,'rows':rows})
                else:raise ValueError('Unknown task')
    except BaseException:
        results.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


def wait_result(results,processes):
    while True:
        try:
            result=results.get(timeout=5)
            if result['kind']=='error':raise RuntimeError(result['traceback'])
            return result
        except queue.Empty:
            for process in processes:
                if process.exitcode is not None:raise RuntimeError(f'Worker {process.pid} exited {process.exitcode}')


def compare_paired(rows,reference_policy,k,metric,selected_policy,seed):
    first={r['sample']:r for r in rows if r['policy']==selected_policy and r['k']==k}
    reference={r['sample']:r for r in rows if r['policy']==reference_policy and r['k']==k}
    if set(first)!=set(reference):raise ValueError('Comparison has mismatched samples')
    samples=sorted(first)
    diff=np.array([first[i][metric]-reference[i][metric] for i in samples])
    a=np.mean([first[i][metric] for i in samples]);b=np.mean([reference[i][metric] for i in samples])
    rng=np.random.default_rng(seed)
    bootstrap=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(axis=1)
    return {'policy':selected_policy,'reference':reference_policy,'k':k,'metric':metric,
            'samples':len(samples),'mean_delta':float(diff.mean()),
            'relative_mean_reduction':float((b-a)/b) if b>0 else None,
            'paired_bootstrap_95pct':np.quantile(bootstrap,[.025,.975]).tolist(),
            'fraction_strictly_better':float((diff<0).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1-run',type=Path,default=Path('/home/liushenyu/test/NeuroBypass/runs/stage1_c4_20261103_02'))
    p.add_argument('--base-model',type=Path,default=Path('/home/liushenyu/opt-13b'))
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--objectives',nargs='+',choices=['last_token','all_tokens'],default=['last_token','all_tokens'])
    p.add_argument('--k',type=int,nargs='+',default=[6,16])
    p.add_argument('--seed',type=int,default=579)
    args=p.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT','0'))!=2:p.error('Use two GPUs through the resource launcher')
    if args.samples<2 or len(set(args.objectives))!=len(args.objectives) or any(k<1 or k>16 for k in args.k):
        p.error('Need >=2 samples, distinct objectives, and K in 1..16')
    output=Path(os.environ['NEUROBYPASS_RUN_DIR'])
    data_file=args.stage1_run/'private_eval_token_ids.npz'
    stage1_manifest=json.loads((args.stage1_run/'data_manifest.json').read_text())
    if sha256(data_file)!=stage1_manifest['data_file_sha256']:raise ValueError('Input dataset hash mismatch')
    count=len(np.load(data_file)['test'])
    if args.samples>count:p.error('Requested more samples than stage-1 test set')
    samples=sorted(random.Random(args.seed).sample(range(count),args.samples))
    donors=shuffled_donors(samples,args.seed+1)
    static=json.loads((args.stage1_run/'static_policy.json').read_text())['ordered_skips']
    old_rows=[json.loads(line) for line in (args.stage1_run/'sample_metrics.jsonl').read_text().splitlines()]
    baseline=[row for row in old_rows if row['sample'] in samples]
    metadata={'configuration':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
              'sample_ids':samples,'shuffled_mask_donors':donors,'seed':args.seed,
              'program_sha256':sha256(__file__),'input_data_sha256':sha256(data_file),
              'static_policy_sha256':sha256(args.stage1_run/'static_policy.json'),
              'model_config_sha256':sha256(args.base_model/'config.json'),
              'cpu_ids':sorted(os.sched_getaffinity(0)),'gpu_visibility':os.environ['CUDA_VISIBLE_DEVICES'],
              'torch_version':torch.__version__,
              'scope':'Per-input full-teacher greedy search on evaluation prompts; expensive reference, not deployable policy.',
              'limitations':['Search is scored on the same prompt; no generalization or downstream-task claim.',
                             'Greedy is not global optimality or a guaranteed upper bound.',
                             'Input blocks, not independently verified documents.']}
    write_json(output/'experiment.json',metadata)
    context=mp.get_context('spawn');results=context.Queue();inboxes=[context.Queue() for _ in range(2)]
    processes=[context.Process(target=worker,args=(rank,vars(args),inboxes[rank],results)) for rank in range(2)]
    paths={};rows=[];started=time.monotonic()
    try:
        for process in processes:process.start()
        ready=[wait_result(results,processes) for _ in processes]
        if any(r['kind']!='ready' for r in ready):raise RuntimeError('Worker startup protocol error')
        write_json(output/'workers.json',ready);stamp('stage2_workers_ready',workers=ready)
        jobs=[{'kind':'search','sample':sample,'objective':objective} for objective in args.objectives for sample in samples]
        job_by_rank={rank:[] for rank in range(2)}
        for i,job in enumerate(jobs):job_by_rank[i%2].append(job)
        positions={0:0,1:0}
        for rank in range(2):inboxes[rank].put(job_by_rank[rank][0]);positions[rank]=1
        completed=0
        with (output/'greedy_search_steps.jsonl').open('x') as steps,(output/'greedy_paths.jsonl').open('x') as path_log:
            while completed<len(jobs):
                record=wait_result(results,processes)
                if record['kind']=='search_step':
                    steps.write(json.dumps(record)+'\n');steps.flush()
                    if record['step'] in args.k:
                        rows.append({'sample':record['sample'],'rank':record['rank'],
                                     'policy':'matched_greedy_'+record['objective'],'k':record['step'],
                                     'skipped_layers':sorted(record['ordered_skips']),**record['quality_metrics']})
                    if record['step'] in [6,12,16]:
                        stamp('stage2_search_progress',sample=record['sample'],objective=record['objective'],
                              step=record['step'],objective_kl=record['objective_kl'],
                              completed_searches=completed,total_searches=len(jobs),
                              elapsed_seconds=time.monotonic()-started)
                elif record['kind']=='search_done':
                    paths[(record['objective'],record['sample'])]=record['ordered_skips']
                    path_log.write(json.dumps(record)+'\n');path_log.flush();completed+=1
                    rank=record['rank']
                    if positions[rank]<len(job_by_rank[rank]):
                        inboxes[rank].put(job_by_rank[rank][positions[rank]]);positions[rank]+=1
                else:raise RuntimeError('Search protocol error')
        # Retain every selected mask exactly once under a mismatched input.
        # These are paired quality comparisons at fixed K, not random new masks.
        for i,sample in enumerate(samples):
            policies=[('full',0,[],{})]
            for k in args.k:
                policies.append(('calibrated_static_greedy',k,static[:k],{}))
                for objective in args.objectives:
                    donor=donors[sample]
                    policies.append(('shuffled_greedy_'+objective,k,paths[(objective,donor)][:k],{'donor_sample':donor}))
            inboxes[i%2].put({'kind':'transfer','sample':sample,'policies':policies})
        with (output/'sample_metrics.jsonl').open('x') as metrics:
            for row in rows:metrics.write(json.dumps(row)+'\n')
            metrics.flush()
            for _ in samples:
                result=wait_result(results,processes)
                if result['kind']!='transfer_done':raise RuntimeError('Transfer protocol error')
                for row in result['rows']:rows.append(row);metrics.write(json.dumps(row)+'\n')
                metrics.flush()
        # Fresh baseline computations must match stage 1 on identical inputs.
        old={(row['sample'],row['policy'],row['k']):row for row in baseline}
        for row in rows:
            key=(row['sample'],row['policy'],row['k'])
            if key in old:
                for metric in ['last_token_kl','all_token_mean_kl','next_token_nll']:
                    if abs(row[metric]-old[key][metric])>1e-5:
                        raise RuntimeError('Repeated baseline differs from stage 1')
        comparisons=[]
        for objective in args.objectives:
            for k in args.k:
                for metric in ['last_token_kl','all_token_mean_kl','next_token_nll']:
                    for reference in ['calibrated_static_greedy','shuffled_greedy_'+objective]:
                        comparisons.append(compare_paired(rows,reference,k,metric,'matched_greedy_'+objective,args.seed))
        write_json(output/'summary.json',{'metadata':metadata,'sample_count':len(samples),
                   'policies':summarize(rows),'paired_comparisons':comparisons,
                   'elapsed_seconds':time.monotonic()-started,
                   'matched_stage1_baseline_verified':True})
        stamp('stage2_completed',samples=len(samples),objectives=args.objectives,elapsed_seconds=time.monotonic()-started)
    finally:
        for inbox in inboxes:inbox.put(None)
        for process in processes:
            if process.pid is not None:process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=10)
            if process.is_alive():process.kill();process.join()


if __name__=='__main__':main()
