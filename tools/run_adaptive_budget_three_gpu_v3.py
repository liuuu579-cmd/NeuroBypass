"""Route5: prompt-conditioned K, frozen quality/cost operating points and actual full-dataset wall time."""
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_choice_core_v3 import (PREFIX,SEEDS,ThreeWorkers,choose_cost,cpu_checks,cpu_fit_predictions,
    fit_choice,load,load_choice,preservation,read_old_evidence,score_joint,setup,source_manifest,
    summary_rows,teachers_for)
from full_context_core_v1 import prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import read_lines,write,verify_preserved,thread_record
from prompt_fixed_core_v1 import continuation_logits,cached_continuation_logits
from stage1_benchmark_v2 import sha256,stamp


def calibrate(out,cfg,bank,seed,arm,embeddings,costs,device):
    fit=load(out/f'budget_{arm}_s{seed}_fit.json');router=load_choice(fit['best']['path'],device)
    samples=cfg['selection_ids'];scores=cpu_fit_predictions(router,embeddings,samples,device)
    true=np.array([costs[i] for i in samples]);work=np.array([40-r['k'] for r in bank['routes']]);grid=[]
    for penalty in cfg['penalty_grid']:
        index=choose_cost(scores,bank,penalty).numpy()
        grid.append({'penalty':penalty,'selection_kl':float(true[np.arange(len(index)),index].mean()),
            'executed_layers':float(work[index].mean()),'unique_k':sorted({bank['routes'][i]['k'] for i in index})})
    ops=[]
    for target in cfg['target_k']:
        baseline=bank['budgets'].index(target);baseline_kl=float(true[:,baseline].mean())
        for tolerance in cfg['kl_tolerances']:
            eligible=[r for r in grid if r['selection_kl']<=baseline_kl+tolerance]
            # The fixed reference is always available; no forced dynamic failure.
            fallback={'penalty':None,'selection_kl':baseline_kl,'executed_layers':40-target,
                'unique_k':[target],'fixed_index':baseline}
            options=eligible+[fallback];best=min(options,key=lambda r:(r['executed_layers'],r['selection_kl'],r['penalty'] is not None))
            name=f'{arm}_ref{target:02d}_slack{int(round(tolerance*1000)):03d}'
            op={'name':name,'seed':seed,'arm':arm,'target_k':target,'kl_tolerance':tolerance,
                'selection_static_kl':baseline_kl,'selection_policy':best}
            if best['penalty'] is None:op.update({'kind':'static','index':baseline,'penalty':None})
            else:op.update({'kind':'router','penalty':best['penalty']})
            ops.append(op)
    result={'seed':seed,'arm':arm,'checkpoint':fit['best'],'selection_ids':samples,'grid':grid,'operating_points':ops}
    write(out/f'budget_{arm}_s{seed}_calibration.json',result);del router;return ops


def label_worker(rank,model,data,cfg,messages):
    out=Path(cfg['output']);bank=load(out/'budget_bank.json');device=f'cuda:{rank}';rows=[];errors=[]
    samples=sorted(cfg['train_ids']+cfg['validation_ids'])[rank::3]
    with torch.inference_mode(),(out/f'private_labels_{rank:03d}.jsonl').open('x') as stream:
        for offset,sample in enumerate(samples):
            raw,p=data[sample];ids=torch.tensor(raw,device=device)[None]
            teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1);costs=[]
            for route in bank['routes']:
                if route['k']==0:costs.append(0.);continue
                lp=F.log_softmax(continuation_logits(model,ids,p,route['skips']),-1)
                costs.append(float((teacher.exp()*(teacher-lp)).sum(-1).mean()))
            assert all(np.isfinite(costs));row={'sample':sample,'prompt_tokens':p,'sequence_sha256':data.rows[sample]['sequence_sha256'],'kl':costs}
            stream.write(json.dumps(row)+'\n');stream.flush()
            if offset==0:
                idx=bank['budgets'].index(12);cached,_=cached_continuation_logits(model,ids[:,:p+4],p,bank['routes'][idx]['skips'])
                direct=continuation_logits(model,ids[:,:p+4],p,bank['routes'][idx]['skips'])
                a,b=F.log_softmax(direct,-1),F.log_softmax(cached,-1)
                error=float((a.exp()*(a-b)).sum(-1).abs().mean());assert error<1e-4;errors.append(error)
            if offset%8==0:messages.put({'kind':'progress','phase':'labels','lane':'budget','rank':rank,'completed':offset+1,'total':len(samples)})
    write(out/f'label_{rank:03d}.json',{'passed':True,'documents':len(samples),'labels_sha256':sha256(out/f'private_labels_{rank:03d}.jsonl'),
        'cache_kl_max':max(errors),'whole_joint_masks':True,**thread_record()})


def worker(rank,cfg,inbox,messages):
    setup(rank);out=Path(cfg['output']);device=f'cuda:{rank}';seed=SEEDS[rank]
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float16,
        attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    data=Records(cfg['data_run']);bank=load(out/'budget_bank.json')
    label_worker(rank,model,data,cfg,messages);messages.put({'kind':'labels_done','rank':rank});assert inbox.get()['kind']=='fit'
    records={r['sample']:r for i in range(3) for r in read_lines(out/f'private_labels_{i:03d}.jsonl')}
    costs={i:r['kl'] for i,r in records.items()};embeddings=torch.load(out/'private_embeddings.pth',weights_only=True,map_location='cpu')
    for observe in (True,False):
        fit_choice(out,'budget',bank,costs,embeddings,cfg,seed,observe,device,messages,rank)
        calibrate(out,cfg,bank,seed,'prompt' if observe else 'constant',embeddings,costs,device)
    del embeddings;gc.collect();messages.put({'kind':'fit_done','rank':rank});assert inbox.get()['kind']=='quality'
    routers={};ops=[]
    for s in SEEDS:
        for arm in ('prompt','constant'):
            fit=load(out/f'budget_{arm}_s{s}_fit.json');routers[arm,s]=load_choice(fit['best']['path'],device)
            ops+=load(out/f'budget_{arm}_s{s}_calibration.json')['operating_points']
    inputs=read_lines(out/'private_inputs.jsonl');replay=[]
    with torch.inference_mode(),(out/f'private_quality_{rank:03d}.jsonl').open('x') as stream:
        for offset,r in enumerate(inputs[rank::3]):
            ids=torch.tensor(r['route_ids'],device=device)[None];emb=prompt_embeddings(model,ids);assert emb.shape[1]==len(r['route_ids'])
            scores={key:router(emb) for key,router in routers.items()};teachers=teachers_for(model,r)
            cache={i:score_joint(model,r,route['skips'],teachers) for i,route in enumerate(bank['routes'])}
            proposals=[(f'static_k{route["k"]:02d}',None,i) for i,route in enumerate(bank['routes'])]
            for op in ops:
                index=op['index'] if op['kind']=='static' else int(choose_cost(scores[op['arm'],op['seed']],bank,op['penalty'])[0])
                proposals.append((op['name'],op['seed'],index))
            for policy,s,index in proposals:
                route=bank['routes'][index]
                if r['task']=='c4_development':
                    error=abs(cache[index]['kl']-costs[r['source_sample']][index]);assert error<1e-6;replay.append(error)
                stream.write(json.dumps({'sample':r['sample'],'source_sample':r.get('source_sample'),'task':r['task'],
                    'policy':policy,'seed':s,'bank_name':'budget','index':index,'k':route['k'],
                    'prompt_tokens':len(r['route_ids']),'skips':route['skips'],**cache[index]},allow_nan=False)+'\n')
            stream.flush()
            if offset%8==0:messages.put({'kind':'progress','rank':rank,'phase':'quality','lane':'budget','completed':offset+1,'total':len(inputs[rank::3])})
    write(out/f'quality_worker_{rank:03d}.json',{'passed':True,'replay_max_abs_error':max(replay),'quality_sha256':sha256(out/f'private_quality_{rank:03d}.jsonl'),**thread_record()})
    messages.put({'kind':'quality_done','rank':rank});assert inbox.get()['kind']=='timing'
    # Warm CPU checkpoint bank; prior quality-phase GPU peak is NOT called a
    # selective cold-start peak. Deployment-phase peaks are reset per trial.
    del routers;model.to('cpu');gc.collect();torch.cuda.empty_cache()
    from budget_runtime_v3 import run_dataset,runtime_checks
    engineering=next(r for r in inputs if r['task']=='c4_development' and len(r['route_ids'])<=256)
    runtime,checked=runtime_checks(model,bank,device,engineering);write(out/f'runtime_checks_{rank:03d}.json',checked)
    if cfg['probe']:
        runtime.empty();write(out/f'worker_{rank:03d}.json',{'passed':True,'probe':True,'runtime_checks_passed':True,**thread_record()})
        messages.put({'kind':'done','rank':rank});assert inbox.get() is None;return
    fit=load(out/f'budget_prompt_s{seed}_fit.json');router=load_choice(fit['best']['path'],device)
    own=load(out/f'budget_prompt_s{seed}_calibration.json')['operating_points']
    fixed=[{'name':f'static_k{route["k"]:02d}','kind':'static','index':i,'seed':None} for i,route in enumerate(bank['routes'])]
    selected=fixed+own
    # Exact same ordered full260-record evaluation population per policy/repeat.
    # Use balanced cyclic orders across seeds/repeats; never choose fastest run.
    result=[]
    for repeat in range(3):
        order=selected[(rank+repeat)%len(selected):]+selected[:(rank+repeat)%len(selected)]
        for op in order:
            deployment='physical_static' if op['kind']=='static' else 'resident'
            item=run_dataset(runtime,router if op['kind']=='router' else None,bank,op,inputs,deployment,repeat)
            result.append(item);messages.put({'kind':'progress','rank':rank,'phase':'dataset_timing','policy':op['name'],
                'repeat':repeat,'seconds':item['dataset_wall_seconds'],'mean_k':item['mean_k']})
    mc_inputs=[r for r in inputs if r.get('gold') is not None]
    assert len(mc_inputs)==96
    for repeat in range(3):
        order=selected[(rank+repeat)%len(selected):]+selected[:(rank+repeat)%len(selected)]
        for op in order:
            deployment='physical_static' if op['kind']=='static' else 'resident'
            item=run_dataset(runtime,router if op['kind']=='router' else None,bank,op,mc_inputs,deployment,repeat)
            result.append(item);messages.put({'kind':'progress','rank':rank,'phase':'mc_dataset_timing','policy':op['name'],
                'repeat':repeat,'seconds':item['dataset_wall_seconds'],'mean_k':item['mean_k']})
    # Generation is a separate equal-output-budget check, not substituted for
    # likelihood-based dataset evaluation or a multi-turn conversational test.
    generation_inputs=[r for r in inputs if r['task']=='c4_prior_fresh'][::2]
    short_ops=[next(o for o in fixed if o['name']=='static_k12')]+[o for o in own if o['target_k']==12]
    for repeat in range(2):
        for op in short_ops:
            deploy='physical_static' if op['kind']=='static' else 'resident'
            result.append(run_dataset(runtime,router if op['kind']=='router' else None,bank,op,generation_inputs,deploy,repeat,64))
        for op in [o for o in own if o['target_k']==12 and o['kind']=='router']:
            result.append(run_dataset(runtime,router,bank,op,generation_inputs,'selective',repeat,64))
    # Selective dataset scoring includes weight movement for the primaryK12
    # strict/slack operating points, preserving order and route recomputation.
    for repeat in range(2):
        for op in [o for o in own if o['target_k']==12 and o['kind']=='router']:
            result.append(run_dataset(runtime,router,bank,op,inputs,'selective',repeat))
            result.append(run_dataset(runtime,router,bank,op,mc_inputs,'selective',repeat))
    with (out/f'private_timing_{rank:03d}.json').open('x') as stream:json.dump(result,stream,allow_nan=False)
    compact=[{k:v for k,v in r.items() if k!='requests'} for r in result]
    write(out/f'timing_{rank:03d}.json',{'rank':rank,'seed':seed,'trials':compact,
        'private_timing_sha256':sha256(out/f'private_timing_{rank:03d}.json'),**thread_record()})
    runtime.empty();write(out/f'worker_{rank:03d}.json',{'passed':True,'runtime_checks_passed':True,**thread_record()})
    messages.put({'kind':'done','rank':rank});assert inbox.get() is None


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(os.environ['NEUROBYPASS_RUN_DIR']);setup();tracked=preservation(root,out)
    source=load(root/'runs/stage18_deep_swap_pilot_20261112_01_main_k16/protocol.json')
    training=load(root/'runs/stage13_full_context_20261110_02_training/protocol.json')
    budgets=[0,6,8,10,12,16,20,24]
    bank={'budgets':budgets,'routes':[{'index':i,'k':k,'skips':sorted(training['global_path'][:k])} for i,k in enumerate(budgets)],
        'origin':'One global joint-greedy nested path from original training calibration; changing K is the only adaptive choice in this pilot.'}
    write(out/'budget_bank.json',bank)
    cfg={'lane':'budget','output':str(out),'model':source['model'],'data_run':source['data_run'],
        'train_ids':source['train_ids'],'validation_ids':source['validation_ids'],'selection_ids':source['selection_ids'],
        'epochs':24,'seeds':SEEDS,'penalty_grid':[0.,.25,.5,.75,1.,1.5,2.,3.,4.,6.,8.,12.,16.,24.],
        'target_k':[10,12,16],'kl_tolerances':[0.,.02],'source_sha256':source_manifest(root),
        'resources':'3GPU/32CPU, worker10each/controller2',
        'training':'Predict full joint KL for eachK from full visible prompt; paired prompt/constant cost-residual regressors.',
        'selection':'Original32 IDs choose checkpoint by mean penalty-grid decision regret, then choose per-reference-K operating point with smallest executed-layer mean subject to mean selection KL <= reference+.00/.02. Static reference is eligible fallback; all policy choices frozen before task evaluation.',
        'timing':'Exact same ordered260-record pretokenized dataset per policy; three whole-dataset repeats on each seed GPU. Full/static K6/8/10/12/16/20/24 physically removed; dynamic resident; primaryK12 selective includes evict/load. Fresh KV per input; cached teacher-forced option scoring, not free generation. Separate16-request64-token generation check.',
        'quality_tradeoff':'Report accuracy differences0/1/2 percentage points and full static speed-quality frontier; no presumption any decline is acceptable. Quality KL/NLL also retained.',
        'limits':'Previously reviewed exploratory task populations, not untouched final test; CPU checkpoint load/tokenization excluded equally, GPU layout preparation reported separately and included in secondary total. Single-request batches, not optimized concurrent-serving benchmark. Selective peaks are phase peaks, not process-lifetime cold-start peaks.'}
    cfg['probe']=args.probe
    if args.probe:
        cfg['train_ids']=cfg['train_ids'][::32];cfg['selection_ids']=cfg['selection_ids'][:4]
        cfg['validation_ids']=cfg['selection_ids'];cfg['epochs']=1
    labels,embeddings,evidence=read_old_evidence(root/'runs/stage17_static_swap_20261111_01');keep=set(cfg['train_ids']+cfg['validation_ids'])
    embeddings={i:v for i,v in embeddings.items() if i in keep}
    with (out/'private_embeddings.pth').open('xb') as stream:torch.save(embeddings,stream)
    cfg['embedding_sha256']=sha256(out/'private_embeddings.pth');cfg['embedding_source_sha256']=evidence
    del embeddings,labels;gc.collect()
    inputs=read_lines(root/'runs/stage18_deep_swap_pilot_20261112_01_main_k16/private_quality_inputs.jsonl')
    if args.probe:inputs=[r for r in inputs if r['task']=='c4_development' and r['source_sample'] in cfg['selection_ids']]
    with (out/'private_inputs.jsonl').open('x') as stream:
        for r in inputs:stream.write(json.dumps(r)+'\n')
    cfg['quality_input_sha256']=sha256(out/'private_inputs.jsonl');write(out/'protocol.json',cfg);write(out/'cpu_checks.json',cpu_checks())
    with ThreeWorkers(worker,cfg) as workers:
        workers.receive('labels_done');workers.send({'kind':'fit'});workers.receive('fit_done')
        for seed in SEEDS:
            a,b=[load(out/f'budget_{arm}_s{seed}_fit.json') for arm in ('prompt','constant')]
            assert a['initial_sha256']==b['initial_sha256'] and a['updates']==b['updates']
        workers.send({'kind':'quality'});workers.receive('quality_done')
        stamp('budget_quality_complete_waiting_for_isolated_timing')
        sibling=root/'runs'/f'{PREFIX}_codebook';started=time.monotonic()
        while not args.probe and not (sibling/'gpu_work_complete.json').exists():
            if (sibling/'status.json').exists() and load(sibling/'status.json')['state']=='failed':break
            if time.monotonic()-started>7200:raise RuntimeError('Sibling GPU-work completion not confirmed within2h; no contaminated timing claim')
            time.sleep(5)
        workers.send({'kind':'timing'});workers.receive('done')
    from stage19_finalize_v3 import finalize_budget,publish_lane
    if not args.probe:finalize_budget(out,cfg,inputs)
    verify_preserved(out,tracked)
    write(out/'verification.json',{'passed':True,'same_dataset_order_and_token_budget':True,
        'prompt_only_variable_k':True,'static_frontier_and_physical_removal':True,
        'routing_switching_and_full_prefill_counted':True,'paired_seed_initialization_and_updates':True,
        'timing_waited_for_sibling_gpu_completion':True,'runtime_physical_and_resident_checks':True})
    write(out/'gpu_work_complete.json',{'state':'completed','lane':'budget'})
    if not args.probe:publish_lane(root,out,cfg,'budget')


if __name__=='__main__':main()
