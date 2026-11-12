"""Shared three-GPU pilot primitives: immutable evidence, full-mask codebooks and prompt-only cost prediction."""
import gc
import hashlib
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

from deep_swap_core_v1 import StaticSwapRouter
from full_context_experiment_v1 import read_lines,write,verify_preserved,thread_record
from horizontal_quality_v3 import pin_threads
from stage1_benchmark_v2 import sha256,stamp

SEEDS=[579,580,581]
PREFIX='stage19_parallel_20261112_03'

def load(path):return json.loads(Path(path).read_text())


def resource_cpus():
    assert os.environ['NEUROBYPASS_GPU_COUNT']=='3' and os.environ['NEUROBYPASS_CPU_COUNT']=='32'
    cpus=list(map(int,os.environ['NEUROBYPASS_CPU_IDS'].split(',')));assert len(cpus)==32
    return cpus


def setup(rank=None):
    cpus=resource_cpus();selected=cpus[30:32] if rank is None else cpus[rank*10:(rank+1)*10]
    pin_threads(selected);torch.set_num_threads(len(selected));torch.set_num_interop_threads(1)
    if rank is not None:torch.cuda.set_device(rank)


def preservation(root,out):
    # Both sibling runs are active concurrently. Snapshot every pre-existing
    # research artifact while excluding only the two NEW stage19 run namespaces.
    ignored=[out]+[root/'runs'/f'{PREFIX}_{lane}{suffix}' for lane in ('codebook','budget') for suffix in ('','_probe')]
    tracked={str(p):sha256(p) for p in root.rglob('*') if p.is_file()
        and not any(p.is_relative_to(q) for q in ignored) and '.git' not in p.parts
        and not (p.parent==root/'analysis' and p.name.startswith(PREFIX))
        and p.suffix in {'.py','.md','.json','.jsonl','.csv','.png'}}
    tracked.update({str(p):sha256(p) for p in root.parent.glob('*.py')})
    write(out/'preservation_before.json',tracked);return tracked


def source_manifest(root):
    names=['adaptive_choice_core_v3.py','run_small_codebook_three_gpu_v3.py',
        'run_adaptive_budget_three_gpu_v3.py','budget_runtime_v3.py','stage19_finalize_v3.py','launch_stage19_parallel_v4.py']
    return {name:sha256(root/'tools'/name) for name in names}


def guarded(target,rank,cfg,inbox,messages):
    try:target(rank,cfg,inbox,messages)
    except BaseException:messages.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


class ThreeWorkers:
    def __init__(self,target,cfg):
        ctx=mp.get_context('spawn');self.messages=ctx.Queue();self.inboxes=[ctx.Queue() for _ in range(3)]
        self.processes=[ctx.Process(target=guarded,args=(target,r,cfg,self.inboxes[r],self.messages)) for r in range(3)]
    def __enter__(self):
        for p in self.processes:p.start()
        return self
    def send(self,value):
        for q in self.inboxes:q.put(value)
    def receive(self,kind):
        result=[]
        while len(result)<3:
            try:r=self.messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in self.processes) or all(p.exitcode is not None for p in self.processes):raise RuntimeError('Workers exited early')
                continue
            if r['kind']=='error':raise RuntimeError(r['traceback'])
            if r['kind']=='progress':stamp('stage19_progress',**{k:v for k,v in r.items() if k!='kind'});continue
            assert r['kind']==kind,r;result.append(r)
        assert len({r['rank'] for r in result})==3
        return result
    def __exit__(self,typ,exc,tb):
        if typ is None:
            self.send(None)
            for p in self.processes:p.join(timeout=30)
        for p in self.processes:
            if p.is_alive():p.terminate();p.join(timeout=10)
            if p.is_alive():p.kill();p.join(timeout=5)
        if typ is None:assert all(p.exitcode==0 for p in self.processes)


def read_old_evidence(run):
    run=Path(run);labels={};embeddings={};manifest={}
    for rank in range(4):
        receipt=load(run/f'label_result_{rank:03d}.json')
        for kind,key in [('labels','labels_sha256'),('embeddings','embedding_sha256')]:
            path=run/f'private_{kind}_{rank:03d}.{ "jsonl" if kind=="labels" else "pth"}'
            assert sha256(path)==receipt[key];manifest[str(path)]=receipt[key]
        labels.update({r['sample']:r for r in read_lines(run/f'private_labels_{rank:03d}.jsonl')})
        embeddings.update(torch.load(run/f'private_embeddings_{rank:03d}.pth',map_location='cpu',weights_only=True))
    return labels,embeddings,manifest


def complementary_indices(costs,size):
    """Greedy facility coverage over complete joint masks; training rows ONLY."""
    costs=np.asarray(costs,dtype=np.float64);chosen=[int(costs.mean(0).argmin())]
    while len(chosen)<size:
        current=costs[:,chosen].min(1)
        values=np.minimum(current[:,None],costs).mean(0);values[chosen]=np.inf
        chosen.append(int(values.argmin()))
    return chosen


def choose_cost(scores,bank,penalty=0.):
    work=scores.new_tensor([(40-r['k'])/40 for r in bank['routes']])
    return (scores+penalty*work[None]).argmin(-1)


def cpu_fit_predictions(router,embeddings,samples,device):
    with torch.inference_mode():return torch.cat([router(embeddings[i].to(device)).detach().cpu() for i in samples])


def load_choice(path,device):
    record=torch.load(path,map_location='cpu',weights_only=True)
    model=StaticSwapRouter(record['candidates'],width=record['width'],observe=record['observe']).to(device)
    model.load_state_dict(record['state_dict']);model.eval().requires_grad_(False)
    return model


def fit_choice(out,name,bank,costs_by_id,embeddings,cfg,seed,observe,device,messages,rank):
    """Paired residual regressors; targets are full-mask KL, never layer sums."""
    train=cfg['train_ids'];selection=cfg['selection_ids'];arm='prompt' if observe else 'constant'
    train_cost=torch.tensor([costs_by_id[i] for i in train],dtype=torch.float32)
    target=train_cost-train_cost[:,:1];scale=target[:,1:].std().clamp_min(.03)
    torch.manual_seed(seed);router=StaticSwapRouter(len(bank['routes']),width=128,observe=observe).to(device)
    router.prior.copy_(target.mean(0).to(device));router.scale.copy_(scale.to(device))
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in router.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(router.parameters(),lr=2e-4,weight_decay=.01)
    per_epoch=math.ceil(len(train)/16);total=per_epoch*cfg['epochs']
    def schedule(n):
        warm=max(1,total//20)
        return (n+1)/warm if n<warm else .1+.9*.5*(1+math.cos(math.pi*(n-warm)/max(1,total-warm)))
    sched=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule);history=[];best=math.inf;checkpoint=None
    key=f'{name}_{arm}_s{seed}';started=time.monotonic()
    selection_cost=np.array([costs_by_id[i] for i in selection])
    for epoch in range(cfg['epochs']+1):
        losses=[]
        if epoch:
            router.train();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+10*epoch)).tolist()
            for offset in range(0,len(order),16):
                batch=order[offset:offset+16];pending=sorted(batch,key=lambda i:embeddings[train[i]].shape[1]);optimizer.zero_grad(set_to_none=True)
                batch_loss=0.
                while pending:
                    sub=[pending.pop(0)]
                    while pending and (len(sub)+1)*embeddings[train[pending[0]]].shape[1]<=4096:sub.append(pending.pop(0))
                    length=max(embeddings[train[i]].shape[1] for i in sub) if observe else 1
                    x=torch.zeros(len(sub),length,5120,dtype=torch.float16,device=device);valid=torch.zeros(len(sub),length,dtype=torch.bool,device=device)
                    if observe:
                        for j,i in enumerate(sub):
                            val=embeddings[train[i]][0];x[j,:len(val)]=val.to(device);valid[j,:len(val)]=True
                    else:valid[:]=True
                    pred=router(x,valid);loss=F.smooth_l1_loss(pred[:,1:]/scale.to(device),target[sub,1:].to(device)/scale.to(device))*len(sub)/len(batch)
                    assert torch.isfinite(loss);loss.backward();batch_loss+=float(loss.detach())
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(router.parameters(),1.))
                optimizer.step();sched.step();losses.append(batch_loss)
        row={'epoch':epoch,'loss':float(np.mean(losses)) if losses else 0.}
        if epoch in (0,1) or epoch%4==0 or epoch==cfg['epochs']:
            router.eval();scores=cpu_fit_predictions(router,embeddings,selection,device)
            if cfg['lane']=='budget':
                # One checkpoint serves a predeclared range of cost penalties;
                # choose by regret relative to the same bank at each penalty.
                penalties=cfg['penalty_grid'];work=np.array([(40-r['k'])/40 for r in bank['routes']]);regrets=[]
                for penalty in penalties:
                    indices=choose_cost(scores,bank,penalty).numpy()
                    objective=selection_cost+penalty*work[None]
                    regrets.append(float(np.mean(objective[np.arange(len(indices)),indices]-objective.min(1))))
                value=float(np.mean(regrets));row['selection_penalized_regret']=value
            else:
                indices=scores.argmin(-1).numpy();value=float(selection_cost[np.arange(len(indices)),indices].mean());row['selection_kl']=value
            if value<best:
                path=out/f'private_{key}_e{epoch:03d}.pth'
                with path.open('xb') as stream:torch.save({'state_dict':router.state_dict(),'candidates':len(bank['routes']),
                    'width':128,'observe':observe,'seed':seed,'epoch':epoch,'bank_sha256':sha256(out/f'{name}_bank.json')},stream)
                best=value;checkpoint={'path':str(path),'sha256':sha256(path),'epoch':epoch,'selection_objective':value}
        history.append(row);messages.put({'kind':'progress','rank':rank,'phase':'fit','lane':cfg['lane'],'name':name,'arm':arm,'seed':seed,**row})
    write(out/f'{key}_history.json',history)
    saved=load_choice(checkpoint['path'],device)
    chosen=cpu_fit_predictions(saved,embeddings,train,device).argmin(-1).numpy()
    train_selected=float(train_cost[np.arange(len(chosen)),chosen].mean())
    result={'arm':arm,'name':name,'seed':seed,'initial_sha256':initial,'best':checkpoint,
        'parameters':sum(p.numel() for p in router.parameters()),'updates':total,'epochs':cfg['epochs'],
        'train_selected_kl':train_selected,'train_bank_best_kl':float(train_cost.min(-1).values.mean()),
        'seconds':time.monotonic()-started,'scale':float(scale)}
    write(out/f'{key}_fit.json',result)
    del router,saved,optimizer;gc.collect();torch.cuda.empty_cache();return result


def score_joint(model,record,skips,teachers=None):
    """Real complete-mask forward; shared route across every answer option."""
    from prompt_fixed_core_v1 import continuation_logits
    device=next(model.parameters()).device;scores=[];kl_sum=0.;count=0
    with torch.inference_mode():
        for option,(raw,n) in enumerate(zip(record['sequence_ids'],record['target_lengths'])):
            ids=torch.tensor(raw,dtype=torch.long,device=device)[None]
            lp=F.log_softmax(continuation_logits(model,ids,len(raw)-n,skips),-1)
            score=float(lp.gather(-1,ids[:,-n:,None]).double().sum());scores.append(score);count+=n
            if teachers is not None:
                teacher=teachers[option].to(device);kl_sum+=float((teacher.exp()*(teacher-lp)).sum(-1).double().sum())
    values={'kl':kl_sum/count if teachers is not None else None,'nll':-sum(scores)/count,'scored_tokens':count,'scores':scores}
    if record.get('gold') is not None:
        values['correct']=int(np.argmax(scores)==record['gold'])
        values['correct_norm']=int(np.argmax([v/n for v,n in zip(scores,record['normalization_lengths'])])==record['gold'])
    return values


def teachers_for(model,r):
    from prompt_fixed_core_v1 import continuation_logits
    device=next(model.parameters()).device;result=[]
    with torch.inference_mode():
        for raw,n in zip(r['sequence_ids'],r['target_lengths']):
            ids=torch.tensor(raw,device=device)[None]
            result.append(F.log_softmax(continuation_logits(model,ids,len(raw)-n,[]),-1).cpu())
    return result


def summary_rows(rows,inputs):
    from collections import defaultdict
    from headroom_ruler_core_v1 import paired
    grouped=defaultdict(list)
    for r in rows:grouped[(r['task'],r['policy'],r.get('seed'),r.get('bank_name'))].append(r)
    result=[]
    for key,items in sorted(grouped.items(),key=lambda z:str(z[0])):
        r=dict(zip(('task','policy','seed','bank_name'),key));r['records']=len(items)
        for metric in ('kl','nll','correct_norm','k'):
            if metric in items[0] and items[0][metric] is not None:r[metric]=float(np.mean([v[metric] for v in items]))
        r['unique_masks']=len({tuple(v['skips']) for v in items});result.append(r)
    return result


def cpu_checks():
    cost=np.array([[1.,.5,1.5],[1.,1.5,.5]])
    assert complementary_indices(cost,3)==[0,1,2]
    bank={'routes':[{'k':0},{'k':20}]}
    scores=torch.tensor([[0.,.1],[0.,.8]])
    assert choose_cost(scores,bank,0).tolist()==[0,0]
    assert choose_cost(scores,bank,1).tolist()==[1,0]
    a=StaticSwapRouter(3,input_dim=12,width=16,dropout=0).eval();x=torch.randn(1,201,12)
    assert a(x).shape==(1,3)
    return {'passed':True,'complete_mask_complementarity_greedy':True,'variable_k_penalty':True,'prompt_only_choices':True}
