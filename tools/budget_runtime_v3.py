"""Dataset wall-time and generation probes with resident, physical-static and selective variable-K execution.

Already tokenized inputs; CPU checkpoint read is shared setup, reported separately.
Every input creates a fresh full-context KV cache. No oracle targets enter routing.
"""
import gc
import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from full_context_core_v1 import prompt_embeddings
from prompt_fixed_core_v1 import FixedRouteSession,continuation_logits
from selective_residency_core_v1 import SelectiveBank
from adaptive_choice_core_v3 import choose_cost


class Runtime:
    def __init__(self,model,device):
        assert all(p.device.type=='cpu' for p in model.parameters())
        self.model=model;self.device=torch.device(device);self.bank=SelectiveBank(model,device)
        self.original_count=len(self.bank.masters);self.original_layers=None;self.current=None
    def restore_indices(self):
        if self.original_layers is not None:
            self.model.model.decoder.layers=torch.nn.ModuleList(self.original_layers);self.original_layers=None
        for i,layer in enumerate(self.model.model.decoder.layers):layer.self_attn.layer_idx=i
        self.model.config.num_hidden_layers=self.original_count
    def empty(self):
        self.restore_indices();self.bank.activate(list(range(self.original_count)));self.current=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(self.device)
    def activate(self,skips,physical):
        key=(tuple(sorted(skips)),physical)
        if self.current==key:return {'loaded_layers':[],'evicted_layers':[],'host_to_device_weight_bytes':0}
        self.restore_indices();movement=self.bank.activate(skips)
        if physical:
            self.original_layers=list(self.model.model.decoder.layers)
            keep=[layer for i,layer in enumerate(self.original_layers) if i not in set(skips)]
            self.model.model.decoder.layers=torch.nn.ModuleList(keep)
            for i,layer in enumerate(keep):layer.self_attn.layer_idx=i
            self.model.config.num_hidden_layers=len(keep)
        self.current=key;return movement


def selected_route(model,router,bank,op,prompt):
    if op['kind']=='static':return op['index']
    assert router is not None
    with torch.inference_mode():
        embedding=prompt_embeddings(model,prompt)
        assert embedding.shape[1]==prompt.shape[1]
        return int(choose_cost(router(embedding),bank,op['penalty'])[0])


def cached_score(model,r,actual_skips,device,check=False):
    scores=[];tokens=0;peak_cache=0;precision_checks=[]
    with torch.inference_mode():
        for raw,n in zip(r['sequence_ids'],r['target_lengths']):
            ids=torch.tensor(raw,dtype=torch.long,device=device)[None];p=len(raw)-n
            session=FixedRouteSession(model,actual_skips)
            logits=session.forward(ids[:,:p])[:,-1:].float()
            first=F.log_softmax(logits,-1).gather(-1,ids[:,p:p+1,None]).double().sum()
            score=first
            if n>1:
                suffix=F.log_softmax(session.forward(ids[:,p:-1]).float(),-1)
                score=score+suffix.gather(-1,ids[:,p+1:,None]).double().sum()
            scores.append(float(score));tokens+=n
            expected=2*(len(model.model.decoder.layers)-len(actual_skips))*(len(raw)-1)*5120*2
            assert session.cache_bytes()==expected;peak_cache=max(peak_cache,expected)
            if check:
                reference=F.log_softmax(continuation_logits(model,ids,p,actual_skips),-1)
                ref=float(reference.gather(-1,ids[:,p:,None]).double().sum())
                cached_lp=torch.cat([F.log_softmax(logits,-1),suffix],1) if n>1 else F.log_softmax(logits,-1)
                token_kl=(reference.exp()*(reference-cached_lp)).sum(-1).abs()
                mean_kl=float(token_kl.mean());max_kl=float(token_kl.max())
                # Match the established cache-distribution contract. Different
                # FP16 prefill/continuation kernel shapes need not give an
                # identical scalar target NLL. Report those differences too.
                assert mean_kl<=1e-4 and max_kl<=1e-3,(mean_kl,max_kl)
                precision_checks.append({'mean_kl':mean_kl,'max_position_kl':max_kl,
                    'target_nll_abs_error':abs(ref-float(score))/n,
                    'argmax_agreement':float((reference.argmax(-1)==cached_lp.argmax(-1)).float().mean())})
            del session,ids,logits,score,first
    row={'nll':-sum(scores)/tokens,'scored_tokens':tokens,'scores':scores,'peak_kv_bytes':peak_cache}
    if r.get('gold') is not None:row['correct_norm']=int(np.argmax([v/n for v,n in zip(scores,r['normalization_lengths'])])==r['gold'])
    if check:row['cached_serial_precision']=precision_checks
    return row


def generation(model,prompt,skips,count):
    session=FixedRouteSession(model,skips);generated=[]
    with torch.inference_mode():
        logits=session.forward(prompt)[:,-1:]
        for i in range(count):
            token=logits.argmax(-1);generated.append(int(token))
            if i+1<count:logits=session.forward(token)[:,-1:]
    digest=hashlib.sha256(np.asarray(generated,dtype='<i8').tobytes()).hexdigest()
    kv=session.cache_bytes();del session,logits
    return digest,kv


def run_dataset(runtime,router,bank,op,records,deployment,repeat,generate_tokens=0):
    assert deployment in ('resident','physical_static','selective')
    if deployment=='physical_static':assert op['kind']=='static'
    runtime.empty();model=runtime.model;device=runtime.device
    torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);prep_start=time.perf_counter()
    if deployment=='resident':runtime.activate([],False)
    elif deployment=='physical_static':runtime.activate(bank['routes'][op['index']]['skips'],True)
    torch.cuda.synchronize(device);prepare=time.perf_counter()-prep_start
    # CPU models/routers are already loaded. Request wall time includes H2D
    # inputs, routing, activation/eviction, full prefill, scoring/decode, cleanup.
    rows=[];torch.cuda.synchronize(device);start=time.perf_counter()
    with torch.inference_mode():
        for r in records:
            prompt=torch.tensor(r['route_ids'],device=device,dtype=torch.long)[None]
            t0=time.perf_counter();index=selected_route(model,router,bank,op,prompt)
            torch.cuda.synchronize(device);routed=time.perf_counter();route=bank['routes'][index]
            movement={'loaded_layers':[],'evicted_layers':[],'host_to_device_weight_bytes':0}
            if deployment=='selective':movement=runtime.activate(route['skips'],True)
            torch.cuda.synchronize(device);activated=time.perf_counter()
            actual=[] if deployment in ('physical_static','selective') else route['skips']
            if generate_tokens:
                digest,kv=generation(model,prompt,actual,generate_tokens)
                values={'generated_tokens_sha256':digest,'new_tokens':generate_tokens,'peak_kv_bytes':kv}
            else:values=cached_score(model,r,actual,device)
            torch.cuda.synchronize(device);ended=time.perf_counter()
            rows.append({'sample':r['sample'],'task':r['task'],'index':index,'k':route['k'],
                'prompt_tokens':len(r['route_ids']),'route_seconds':routed-t0,'switch_seconds':activated-routed,
                'compute_seconds':ended-activated,'request_seconds':ended-t0,**movement,**values})
            del prompt
    torch.cuda.synchronize(device);elapsed=time.perf_counter()-start
    return {'deployment':deployment,'policy':op['name'],'seed':op.get('seed'),'repeat':repeat,
        'generation_tokens_per_request':generate_tokens,'records':len(rows),'dataset_wall_seconds':elapsed,
        'gpu_layout_setup_seconds':prepare,'dataset_including_gpu_setup_seconds':prepare+elapsed,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'mean_k':float(np.mean([r['k'] for r in rows])),'unique_k':sorted({r['k'] for r in rows}),
        'h2d_weight_bytes':sum(r['host_to_device_weight_bytes'] for r in rows),
        'route_seconds_sum':sum(r['route_seconds'] for r in rows),'switch_seconds_sum':sum(r['switch_seconds'] for r in rows),
        'accuracy_norm':float(np.mean([r['correct_norm'] for r in rows if 'correct_norm' in r])) if any('correct_norm' in r for r in rows) else None,
        'requests':rows}


def runtime_checks(model,bank,device,record):
    runtime=Runtime(model,device);prompt=torch.tensor(record['route_ids'],device=device)[None]
    indices=[0,bank['budgets'].index(12),len(bank['routes'])-1];checks=[]
    for index in indices:
        skips=bank['routes'][index]['skips'];runtime.activate([],False)
        with torch.inference_mode():
            serial=cached_score(model,record,skips,device,True)
            resident_digest,_=generation(model,prompt,skips,8)
        runtime.activate(skips,True)
        with torch.inference_mode():
            physical=cached_score(model,record,[],device,True)
            physical_digest,_=generation(model,prompt,[],8)
        error=abs(serial['nll']-physical['nll']);assert error<1e-6 and resident_digest==physical_digest
        checks.append({'k':len(skips),'physical_resident_nll_abs_error':error,'generation_hash_equal':True,
            'resident_cached_serial_precision':serial['cached_serial_precision'],
            'physical_cached_serial_precision':physical['cached_serial_precision']})
    runtime.empty();del prompt
    return runtime,{'passed':True,'checks':checks,'new_cache_for_every_input':True,
        'full_prompt_router_contract':True,'real_physical_static_and_selective_layers':True}
