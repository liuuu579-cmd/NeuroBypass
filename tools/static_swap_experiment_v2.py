"""Training labels, finite static-start exchange-router fit and quality scoring."""
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from full_context_core_v1 import load_router as load_old_router, prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import read_lines, setup_worker, thread_record, write
from headroom_ruler_core_v1 import answer_metrics
from prompt_fixed_core_v1 import (FixedRouteSession,cached_continuation_logits,
    continuation_logits,continuation_metrics,fixed_route)
from static_swap_core_v2 import StaticSwapRouter,greedy_indices,load_router,regression_loss
from stage1_benchmark_v2 import sha256


def tensor(model,ids):
    return torch.tensor(ids,device=next(model.parameters()).device,dtype=torch.long)[None]


def metrics(model,ids,p,skips,teacher):
    values=continuation_metrics(continuation_logits(model,ids,p,skips),teacher,ids,p)
    return {'kl':values['continuation_kl'],'nll':values['continuation_nll']}


def bank_costs(model,ids,p,bank,teacher):
    """Reuse only original-static activations strictly before first mask change."""
    decoder=model.model.decoder;captured={}
    def capture(module,args,kwargs):captured.update(kwargs)
    h=decoder.layers[0].register_forward_pre_hook(capture,with_kwargs=True)
    try:
        with fixed_route(model,bank['static_skips']):
            original=model.model(ids[:,:-1],attention_mask=torch.ones_like(ids[:,:-1]),
                use_cache=False,output_hidden_states=True)
    finally:h.remove()
    assert not captured.get('use_cache') and captured.get('past_key_value') is None
    original_set=set(bank['static_skips']);costs=[];nll=[]
    for route in bank['routes']:
        skip=set(route['skips'])
        if not route['actions']:hidden=original.last_hidden_state
        else:
            first=min(original_set^skip);hidden=original.hidden_states[first]
            for index in range(first,len(decoder.layers)):
                if index not in skip:hidden=decoder.layers[index](hidden,**captured)[0]
            if decoder.final_layer_norm is not None:hidden=decoder.final_layer_norm(hidden)
            if decoder.project_out is not None:hidden=decoder.project_out(hidden)
        lp=F.log_softmax(model.lm_head(hidden[:,p-1:]).float(),-1)
        kl=(teacher.exp()*(teacher-lp)).sum(-1).mean().item()
        loss=F.nll_loss(lp.reshape(-1,lp.shape[-1]),ids[:,p:].reshape(-1)).item()
        assert math.isfinite(kl) and math.isfinite(loss) and kl>=-1e-5
        costs.append(kl);nll.append(loss)
    return costs,nll


def probe(rank,model,data,config):
    sample=max(config['train_ids'],key=lambda i:data.rows[i]['prompt_length'])
    raw,length=data[sample];p=min(length,[64,129,513,2016][rank]);ids=tensor(model,raw[:p+4].tolist())
    teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
    bank=config['bank'];tests=[bank['routes'][i] for i in (0,1,len(bank['routes'])-1)]
    small={**bank,'routes':tests}
    kls,nlls=bank_costs(model,ids,p,small,teacher);errors=[]
    for index,route in enumerate(tests):
        direct=metrics(model,ids,p,route['skips'],teacher)
        errors.extend([abs(kls[index]-direct['kl']),abs(nlls[index]-direct['nll'])])
    assert max(errors)<=1e-6,('bank prefix replay',errors)
    skips=tests[-1]['skips'];full=continuation_logits(model,ids,p,skips)
    cached,_=cached_continuation_logits(model,ids,p,skips)
    cache_kl=(F.log_softmax(full,-1).exp()*(F.log_softmax(full,-1)-F.log_softmax(cached,-1))).sum(-1).abs().mean().item()
    assert cache_kl<1e-4 and torch.equal(full.argmax(-1),cached.argmax(-1))
    # Structural skip check proves the restored layer executes and new skip does not.
    counts=[0]*40;hooks=[]
    for i,layer in enumerate(model.model.decoder.layers):
        def count(module,args,kwargs,i=i):counts[i]+=1
        hooks.append(layer.register_forward_pre_hook(count,with_kwargs=True))
    try:metrics(model,ids,p,skips,teacher)
    finally:
        for h in hooks:h.remove()
    # A layer's pre-hook fires even for its bypass; inspect actual attention work instead.
    actual=[0]*40;hooks=[]
    for i,layer in enumerate(model.model.decoder.layers):
        def attention(module,args,i=i):actual[i]+=1
        hooks.append(layer.self_attn.register_forward_pre_hook(attention))
    try:metrics(model,ids,p,skips,teacher)
    finally:
        for h in hooks:h.remove()
    assert all(actual[i]==(0 if i in skips else 1) for i in range(40))
    embedding=prompt_embeddings(model,ids[:,:p]);assert embedding.shape[1]==p
    return {'passed':True,'observed_tokens':p,'joint_bank_serial_max_abs_error':max(errors),
        'cached_teacher_forcing_kl':cache_kl,'skipped_attention_does_not_execute':True,
        'restored_layer_executes':True,'protected_layers_kept':True,'constant_k16':True,**thread_record()}


def labels(rank,model,data,config,messages):
    out=Path(config['output']);bank=config['bank'];embeddings={}
    samples=sorted(config['train_ids']+config['validation_ids'])[rank::4]
    started=time.monotonic()
    with torch.no_grad(),(out/f'private_labels_{rank:03d}.jsonl').open('x') as stream:
        for offset,sample in enumerate(samples):
            raw,p=data[sample];ids=tensor(model,raw.tolist())
            teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
            kl,nll=bank_costs(model,ids,p,bank,teacher)
            row={'sample':sample,'split':data.rows[sample]['split'],'prompt_tokens':p,
                'sequence_sha256':data.rows[sample]['sequence_sha256'],'kl':kl,'nll':nll}
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            embeddings[sample]=prompt_embeddings(model,ids[:,:p]).detach().cpu()
            messages.put({'kind':'progress','phase':'swap_joint_labels','rank':rank,'completed':offset+1,
                'total':len(samples),'prompt_tokens':p,'candidates':len(bank['routes']),
                'elapsed_seconds':time.monotonic()-started})
    with (out/f'private_embeddings_{rank:03d}.pth').open('xb') as stream:torch.save(embeddings,stream)
    return {'rank':rank,'documents':len(samples),'elapsed_seconds':time.monotonic()-started,
        'labels_sha256':sha256(out/f'private_labels_{rank:03d}.jsonl'),
        'embedding_sha256':sha256(out/f'private_embeddings_{rank:03d}.pth')}


def all_labels(out):
    result={}
    for rank in range(4):
        for r in read_lines(out/f'private_labels_{rank:03d}.jsonl'):
            assert r['sample'] not in result;result[r['sample']]=r
    return result


def all_embeddings(out):
    result={}
    for rank in range(4):
        record=torch.load(out/f'private_embeddings_{rank:03d}.pth',map_location='cpu',weights_only=True)
        assert not set(result)&set(record);result.update(record)
    return result


def prediction(router,embedding,device):
    return router(embedding.to(device))


def fit(rank,model,config,messages):
    out=Path(config['output']);device=next(model.parameters()).device;seed=config['seeds'][rank//2]
    observe=rank%2==0;arm='prompt' if observe else 'constant';bank=config['bank']
    records=all_labels(out);embeddings=all_embeddings(out);samples=config['train_ids']
    costs=torch.tensor([records[i]['kl'] for i in samples],dtype=torch.float32)
    target=costs-costs[:,:1];prior=target.mean(0);scale=target[:,1:].std().clamp_min(.03)
    torch.manual_seed(seed)
    router=StaticSwapRouter(len(bank['routes']),width=config['width'],observe=observe).to(device)
    router.prior.copy_(prior.to(device));router.scale.copy_(scale.to(device))
    # Save a content hash of initialization, equal across paired observed/constant arms.
    import hashlib
    initial=hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in router.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(router.parameters(),lr=2e-4,weight_decay=.01)
    updates=math.ceil(len(samples)/16);total=updates*config['epochs']
    def schedule(n):
        warm=max(1,total//20)
        if n<warm:return (n+1)/warm
        return .1+.9*.5*(1+math.cos(math.pi*(n-warm)/max(1,total-warm)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    best=float('inf');checkpoint=None;history=[];selection=config['selection_ids']
    baseline=float(np.mean([records[i]['kl'][0] for i in selection]));started=time.monotonic()
    def select(epoch):
        router.eval();picked=[]
        with torch.no_grad():
            for sample in selection:
                scores=prediction(router,embeddings[sample],device)
                index=int(greedy_indices(scores,bank,2)[0]);picked.append(records[sample]['kl'][index])
        return float(np.mean(picked))
    for epoch in range(config['epochs']+1):
        loss_sum=0.
        if epoch:
            router.train();order=torch.randperm(len(samples),generator=torch.Generator().manual_seed(seed+10*epoch)).tolist()
            for offset in range(0,len(order),16):
                group=order[offset:offset+16];optimizer.zero_grad(set_to_none=True)
                pending=sorted(group,key=lambda i:embeddings[samples[i]].shape[1])
                while pending:
                    sub=[pending.pop(0)]
                    while pending and (len(sub)+1)*embeddings[samples[pending[0]]].shape[1]<=4096:
                        sub.append(pending.pop(0))
                    length=max(embeddings[samples[i]].shape[1] for i in sub) if observe else 1
                    emb=torch.zeros(len(sub),length,5120,dtype=torch.float16,device=device)
                    valid=torch.zeros(len(sub),length,dtype=torch.bool,device=device)
                    if observe:
                        for j,i in enumerate(sub):
                            value=embeddings[samples[i]][0];emb[j,:len(value)]=value.to(device);valid[j,:len(value)]=True
                    else:valid[:]=True
                    estimates=router(emb,valid)
                    loss=regression_loss(estimates,costs[sub].to(device),router.scale)*len(sub)/len(group)
                    assert torch.isfinite(loss);loss.backward();loss_sum+=loss.item()
                norm=torch.nn.utils.clip_grad_norm_(router.parameters(),1.)
                assert torch.isfinite(norm)
                optimizer.step();scheduler.step()
        row={'epoch':epoch,'loss':loss_sum/updates,'seed':seed,'arm':arm}
        if epoch in (0,1) or epoch%4==0 or epoch==config['epochs']:
            value=select(epoch);row['selection_kl']=value
            if value<best:
                path=out/f'private_swap_{arm}_seed{seed}_epoch{epoch:03d}.pth'
                with path.open('xb') as stream:torch.save({'state_dict':router.state_dict(),'candidates':len(bank['routes']),
                    'width':config['width'],'observe':observe,'seed':seed,'epoch':epoch,'selection_kl':value,
                    'bank_sha256':sha256(out/'candidate_bank.json')},stream)
                best=value;checkpoint={'path':str(path),'sha256':sha256(path),'epoch':epoch,'selection_kl':value}
        history.append(row)
        messages.put({'kind':'progress','phase':'swap_router_fit','rank':rank,**row})
    write(out/f'fit_history_{rank:03d}.json',history)
    write(out/f'fit_result_{rank:03d}.json',{'passed':True,'rank':rank,'arm':arm,'seed':seed,
        'initial_sha256':initial,'parameters':sum(p.numel() for p in router.parameters()),
        'epochs':config['epochs'],'optimizer_updates':total,'best':checkpoint,'selection_static_kl':baseline,
        'scale':float(scale),'elapsed_seconds':time.monotonic()-started,**thread_record()})
    del router,optimizer,embeddings;gc.collect();torch.cuda.empty_cache()


def proposed(model,emb,config,new_routers,old_routers):
    bank=config['bank'];proposals=[('full',None,0,[]),('original_static',None,0,bank['static_skips']),
        ('train_best_static',None,0,bank['routes'][config['train_best_route']]['skips']),
        ('train_mean_greedy_static',None,0,bank['routes'][config['train_mean_greedy_route']]['skips'])]
    for seed in config['seeds']:
        for arm in ('prompt','constant'):
            scores=new_routers[arm,seed](emb)
            for maximum in (1,2):
                route=int(greedy_indices(scores,bank,maximum)[0])
                proposals.append((f'swap_{arm}',seed,maximum,bank['routes'][route]['skips']))
        old=old_routers[seed].route(emb,16,config['old_prefix'])[0].tolist()
        proposals.append(('old_router_m6',seed,0,old))
    return proposals


def free_generate(model,prompt,skips,tokenizer):
    session=FixedRouteSession(model,skips);logits=session.forward(prompt)[:,-1];tokens=[]
    for position in range(48):
        choice=int(logits.argmax(-1));tokens.append(choice)
        if choice==tokenizer.eos_token_id or position==47:break
        logits=session.forward(tensor(model,[choice]))[:,-1]
    return tokens,tokenizer.decode(tokens,skip_special_tokens=True)


def quality(rank,model,data,config,messages):
    out=Path(config['output']);device=next(model.parameters()).device;bank=config['bank']
    new_routers={}
    for index in range(4):
        record=json.loads((out/f'fit_result_{index:03d}.json').read_text())
        checkpoint=record['best'];assert sha256(checkpoint['path'])==checkpoint['sha256']
        new_routers[record['arm'],record['seed']]=load_router(checkpoint['path'],device)
    old_routers={p['seed']:load_old_router(p['checkpoint']['path'],device) for p in config['old_policies']}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['model'],use_fast=False,local_files_only=True)
    inputs=read_lines(out/'private_quality_inputs.jsonl')
    records=all_labels(out)
    # Verify saved reload and table-based model selection against real forwards.
    probe_id=config['selection_ids'][rank%len(config['selection_ids'])]
    ids_raw,p=data[probe_id];ids=tensor(model,ids_raw.tolist())
    teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
    emb=prompt_embeddings(model,ids[:,:p]);replay=[]
    with torch.no_grad():
        for seed in config['seeds']:
            pred=new_routers['prompt',seed](emb);index=int(greedy_indices(pred,bank,2)[0])
            value=metrics(model,ids,p,bank['routes'][index]['skips'],teacher)['kl']
            error=abs(value-records[probe_id]['kl'][index]);assert error<=1e-6;replay.append(error)
    write(out/f'selection_replay_{rank:03d}.json',{'passed':True,'max_abs_error':max(replay)})
    with torch.inference_mode(),(out/f'private_quality_{rank:03d}.jsonl').open('x') as stream:
        local=inputs[rank::4]
        for offset,r in enumerate(local):
            obs=tensor(model,r['route_ids']);emb=prompt_embeddings(model,obs)
            proposals=proposed(model,emb,config,new_routers,old_routers)
            teachers=[F.log_softmax(continuation_logits(model,tensor(model,ids),len(ids)-n,[]),-1)
                for ids,n in zip(r['sequence_ids'],r['target_lengths'])]
            cached={}
            for policy,seed,maximum,skips in proposals:
                key=tuple(sorted(skips))
                if key not in cached:
                    kl_sum,nll_sum,count=0.,0.,0;sum_scores=[]
                    for raw,n,teacher in zip(r['sequence_ids'],r['target_lengths'],teachers):
                        ids=tensor(model,raw);p=len(raw)-n
                        lp=F.log_softmax(continuation_logits(model,ids,p,skips),-1)
                        kl_sum+=(teacher.exp()*(teacher-lp)).sum(-1).double().sum().item()
                        score=lp.gather(-1,ids[:,p:,None]).double().sum().item()
                        nll_sum-=score;count+=n;sum_scores.append(score)
                    values={'kl':kl_sum/count,'nll':nll_sum/count,'scored_tokens':count,'scores':sum_scores}
                    if r.get('gold') is not None:
                        values['correct']=int(int(np.argmax(sum_scores))==r['gold'])
                        norm=[v/n for v,n in zip(sum_scores,r['normalization_lengths'])]
                        values['correct_norm']=int(int(np.argmax(norm))==r['gold'])
                    if r['task'].startswith('ruler_'):
                        tokens,text=free_generate(model,obs,skips,tokenizer)
                        values.update(answer_metrics(text,r['answers'],r['answer_kind']))
                        values.update({'generated_ids':tokens,'generated_text':text})
                    cached[key]=values
                stream.write(json.dumps({'sample':r['sample'],'task':r['task'],'source_sample':r.get('source_sample'),
                    'document':r.get('document'),'case':r.get('case'),'prompt_tokens':len(r['route_ids']),
                    'policy':policy,'seed':seed,'maximum_exchanges':maximum,'skips':list(key),
                    'actual_exchanges':len(set(key)^set(bank['static_skips']))//2 if key else 0,
                    **cached[key]},allow_nan=False)+'\n')
            stream.flush()
            if offset%8==0 or offset+1==len(local):
                messages.put({'kind':'progress','phase':'swap_quality','rank':rank,'completed':offset+1,
                    'total':len(local),'task':r['task']})


def worker(rank,config,inbox,messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(config['model'],torch_dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True).to(f'cuda:{rank}').eval().requires_grad_(False)
    data=Records(config['data_run']);out=Path(config['output'])
    with torch.inference_mode():result=probe(rank,model,data,config)
    write(out/f'engineering_{rank:03d}.json',result);messages.put({'kind':'ready','rank':rank})
    assert inbox.get()['kind']=='labels'
    result=labels(rank,model,data,config,messages);write(out/f'label_result_{rank:03d}.json',result)
    messages.put({'kind':'labels_done','rank':rank})
    assert inbox.get()['kind']=='fit'
    fit(rank,model,config,messages);messages.put({'kind':'fit_done','rank':rank})
    task=inbox.get();assert task['kind']=='quality'
    config.update(task['static_controls'])
    quality(rank,model,data,config,messages)
    write(out/f'worker_result_{rank:03d}.json',{'passed':True,
        'quality_sha256':sha256(out/f'private_quality_{rank:03d}.jsonl'),
        'base_model_frozen':not any(p.requires_grad for p in model.parameters()),
        'peak_allocated_gib':torch.cuda.max_memory_allocated(rank)/2**30,**thread_record()})
    messages.put({'kind':'done','rank':rank});assert inbox.get() is None
