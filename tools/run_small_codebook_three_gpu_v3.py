"""Route1: training-only complementary libraries of 4/8 full masks at K16/20/24."""
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from adaptive_choice_core_v3 import (PREFIX,SEEDS,ThreeWorkers,complementary_indices,cpu_checks,
    fit_choice,load,load_choice,preservation,read_old_evidence,resource_cpus,score_joint,
    setup,source_manifest,summary_rows,teachers_for)
from full_context_core_v1 import prompt_embeddings
from full_context_experiment_v1 import read_lines,write,verify_preserved,thread_record
from stage1_benchmark_v2 import sha256,stamp


def worker(rank,cfg,inbox,messages):
    setup(rank);out=Path(cfg['output']);seed=SEEDS[rank];device=f'cuda:{rank}'
    embeddings=torch.load(out/'private_embeddings.pth',weights_only=True,map_location='cpu')
    tables=load(out/'private_costs.json');banks={name:load(out/f'{name}_bank.json') for name in cfg['names']}
    for name,bank in banks.items():
        costs={int(i):v for i,v in tables[name].items()}
        for observe in (True,False):fit_choice(out,name,bank,costs,embeddings,cfg,seed,observe,device,messages,rank)
    del embeddings;gc.collect();messages.put({'kind':'fit_done','rank':rank});assert inbox.get()['kind']=='quality'
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float16,
        attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    routers={}
    for name in cfg['names']:
        for s in SEEDS:
            for arm in ('prompt','constant'):
                record=load(out/f'{name}_{arm}_s{s}_fit.json');cp=record['best'];assert sha256(cp['path'])==cp['sha256']
                routers[name,arm,s]=load_choice(cp['path'],device)
    inputs=read_lines(out/'private_inputs.jsonl');source_tables=load(out/'private_source_costs.json');replay_errors=[]
    with torch.inference_mode(),(out/f'private_quality_{rank:03d}.jsonl').open('x') as stream:
        for offset,r in enumerate(inputs[rank::3]):
            ids=torch.tensor(r['route_ids'],device=device)[None];emb=prompt_embeddings(model,ids)
            assert emb.shape[1]==len(r['route_ids']);teachers=teachers_for(model,r);cache={}
            for name,bank in banks.items():
                proposals=[('fixed_best',None,0)]
                for s in SEEDS:
                    for arm in ('prompt','constant'):
                        pred=routers[name,arm,s](emb);proposals.append((arm,s,int(pred.argmin(-1))))
                for policy,s,index in proposals:
                    route=bank['routes'][index];mask=tuple(route['skips'])
                    if mask not in cache:cache[mask]=score_joint(model,r,mask,teachers)
                    values=cache[mask]
                    if r['task']=='c4_development':
                        expected=source_tables[str(bank['k'])][str(r['source_sample'])][route['source_route']]
                        error=abs(values['kl']-expected);replay_errors.append(error);assert error<1e-6,(name,error)
                    row={'sample':r['sample'],'source_sample':r.get('source_sample'),'task':r['task'],
                        'policy':policy,'seed':s,'bank_name':name,'route_index':index,'k':bank['k'],
                        'prompt_tokens':len(r['route_ids']),'skips':list(mask),**values}
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush()
            if offset%8==0:messages.put({'kind':'progress','rank':rank,'phase':'quality','lane':'codebook','completed':offset+1,'total':len(inputs[rank::3])})
    write(out/f'worker_{rank:03d}.json',{'passed':True,'quality_sha256':sha256(out/f'private_quality_{rank:03d}.jsonl'),
        'full_prompt_observed':True,'c4_label_replay_max_abs_error':max(replay_errors),**thread_record()})
    messages.put({'kind':'done','rank':rank});assert inbox.get() is None


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(os.environ['NEUROBYPASS_RUN_DIR']);setup()
    tracked=preservation(root,out);source=load(root/'runs/stage18_deep_swap_pilot_20261112_01_main_k16/protocol.json')
    cfg={'lane':'codebook','output':str(out),'model':source['model'],'data_run':source['data_run'],
        'train_ids':source['train_ids'],'validation_ids':source['validation_ids'],'selection_ids':source['selection_ids'],
        'epochs':24,'seeds':SEEDS,'names':[],'resources':'3 GPUs /32 CPU cores, workers10 each/controller2',
        'source_sha256':source_manifest(root),'bank_selection':'Training256 mean KL fixed best first; greedily add masks minimizing training mean per-document min joint KL; nested4/8. No validation in library construction.',
        'selection':'Original32 development identities choose checkpoints, epochs0/1/4/...24; other68 previously reviewed development reported separately.',
        'limits':'Fixed-K path learning only, no recovery or deployment speed claim. Prior/constant/static controls share exact same candidate library.'}
    cfg['probe']=args.probe
    if args.probe:
        cfg['train_ids']=cfg['train_ids'][::32];cfg['selection_ids']=cfg['selection_ids'][:4]
        cfg['validation_ids']=cfg['selection_ids'];cfg['epochs']=1
    labels16,embeddings,provenance=read_old_evidence(root/'runs/stage17_static_swap_20261111_01')
    tables={};source_tables={};bank_digest={}
    for k in (16,20,24):
        run=root/'runs'/('stage17_static_swap_20261111_01' if k==16 else f'stage18_deep_swap_pilot_20261112_01_main_k{k}')
        if k==16:labels=labels16
        else:
            labels,extra,more=read_old_evidence(run);provenance.update(more);del extra
        source_bank=load(run/'candidate_bank.json');bank_digest[str(run/'candidate_bank.json')]=sha256(run/'candidate_bank.json')
        train_cost=np.array([labels[i]['kl'] for i in cfg['train_ids']]);chosen=complementary_indices(train_cost,8)
        assert chosen[0]==int(train_cost.mean(0).argmin())
        source_tables[str(k)]={str(i):labels[i]['kl'] for i in cfg['train_ids']+cfg['validation_ids']}
        for size in (4,8):
            name=f'k{k:02d}_b{size:02d}';cfg['names'].append(name)
            routes=[{'index':j,'source_route':i,'k':k,'skips':source_bank['routes'][i]['skips']} for j,i in enumerate(chosen[:size])]
            bank={'k':k,'routes':routes,'source_bank':str(run/'candidate_bank.json'),'source_bank_sha256':sha256(run/'candidate_bank.json'),
                'training_documents':len(cfg['train_ids']),'train_best_index':0,'coverage_greedy':True}
            write(out/f'{name}_bank.json',bank)
            tables[name]={str(i):[labels[i]['kl'][j] for j in chosen[:size]] for i in cfg['train_ids']+cfg['validation_ids']}
    write(out/'private_costs.json',tables);write(out/'private_source_costs.json',source_tables)
    keep=set(cfg['train_ids']+cfg['validation_ids']);embeddings={i:v for i,v in embeddings.items() if i in keep}
    with (out/'private_embeddings.pth').open('xb') as stream:torch.save(embeddings,stream)
    cfg['evidence_sha256']={**provenance,**bank_digest};cfg['embedding_sha256']=sha256(out/'private_embeddings.pth')
    del embeddings,labels16;gc.collect()
    inputs=read_lines(root/'runs/stage18_deep_swap_pilot_20261112_01_main_k16/private_quality_inputs.jsonl')
    if args.probe:inputs=[r for r in inputs if r['task']=='c4_development' and r['source_sample'] in cfg['selection_ids']]
    with (out/'private_inputs.jsonl').open('x') as stream:
        for r in inputs:stream.write(json.dumps(r)+'\n')
    cfg['quality_input_sha256']=sha256(out/'private_inputs.jsonl');write(out/'protocol.json',cfg);write(out/'cpu_checks.json',cpu_checks())
    with ThreeWorkers(worker,cfg) as workers:
        workers.receive('fit_done');workers.send({'kind':'quality'});workers.receive('done')
    rows=[r for rank in range(3) for r in read_lines(out/f'private_quality_{rank:03d}.jsonl')]
    assert len(rows)==len(inputs)*len(cfg['names'])*7
    for name in cfg['names']:
        for seed in SEEDS:
            a,b=[load(out/f'{name}_{arm}_s{seed}_fit.json') for arm in ('prompt','constant')]
            assert a['initial_sha256']==b['initial_sha256'] and a['updates']==b['updates'] and a['parameters']==b['parameters']
    from stage19_finalize_v3 import finalize_codebook
    write(out/'summary.json',finalize_codebook(rows,inputs,cfg,out));verify_preserved(out,tracked)
    write(out/'verification.json',{'passed':True,'quality_rows':len(rows),'paired_initialization_budget':True,
        'training_only_complete_mask_library':True,'source_sha256_verified':True,'constant_k_and_prompt_boundaries':True})
    write(out/'gpu_work_complete.json',{'state':'completed','lane':'codebook'})
    stamp('codebook_gpu_complete',quality_rows=len(rows))
    from stage19_finalize_v3 import publish_lane
    if not args.probe:publish_lane(root,out,cfg,'codebook')


if __name__=='__main__':main()
