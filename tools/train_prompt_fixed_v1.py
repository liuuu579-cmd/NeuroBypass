"""DDP supervised fitting of a prompt-only, set-conditioned greedy router.

Frozen OPT supplies prompt embeddings and continuation quality validation only.
All model checkpoints use exclusive new filenames. Test continuations are never
used in training, temperature selection or checkpoint selection.
"""
import datetime
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from prompt_fixed_core_v1 import (SequentialGreedyRouter,continuation_logits,
    continuation_metrics,raw_prompt_embeddings)
from stage1_benchmark_v2 import sha256,write_json


def targets_from_record(record,layers=40):
    states=[];probabilities=[];valids=[];chosen=[]
    for step in record['steps']:
        state=np.zeros(layers,np.float32);state[step['previous_skips']]=1
        valid=np.zeros(layers,np.bool_);costs=np.full(layers,np.inf,np.float64)
        for layer,cost in step['candidate_costs']:valid[layer]=True;costs[layer]=cost
        if set(np.where(valid)[0])!=set(range(2,layers))-set(step['previous_skips']):raise ValueError('Invalid candidate set')
        values=costs[valid];temperature=max(.005,.15*(np.quantile(values,.90)-values.min()))
        weights=np.exp(-(values-values.min())/temperature);weights/=weights.sum()
        target=np.zeros(layers,np.float32);target[valid]=weights
        states.append(state);probabilities.append(target);valids.append(valid);chosen.append(step['selected_layer'])
    return np.stack(states),np.stack(probabilities),np.stack(valids),np.asarray(chosen,np.int64)


def validation(model,router,data,sample_ids,prompt_length,static,rank,world_size):
    totals=torch.zeros(6,dtype=torch.float64,device=next(router.parameters()).device)
    masks=[]
    with torch.inference_mode():
        for sample in sample_ids[rank::world_size]:
            ids=torch.tensor(data[sample],dtype=torch.long,device=totals.device).unsqueeze(0)
            # Prompt slice is the only router input. The held-out continuation
            # is used solely for quality evaluation of the resulting route.
            path=router.route(raw_prompt_embeddings(model,ids[:,:prompt_length]),6)[0].tolist()
            teacher=F.log_softmax(continuation_logits(model,ids,prompt_length,[]),dim=-1)
            metrics=continuation_metrics(continuation_logits(model,ids,prompt_length,path),teacher,ids,prompt_length)
            reference=continuation_metrics(continuation_logits(model,ids,prompt_length,static),teacher,ids,prompt_length)
            totals+=totals.new_tensor([metrics['continuation_kl'],metrics['continuation_nll'],reference['continuation_kl'],reference['continuation_nll'],len(set(path)),1])
            masks.append(tuple(sorted(path)))
    dist.all_reduce(totals)
    return {'validation_samples':int(totals[5].item()),'continuation_kl':(totals[0]/totals[5]).item(),
            'continuation_nll':(totals[1]/totals[5]).item(),'static_continuation_kl':(totals[2]/totals[5]).item(),
            'static_continuation_nll':(totals[3]/totals[5]).item()}


def train_router(rank,args,model,data,task,results):
    output=Path(task['output']);device=next(model.parameters()).device
    dist.init_process_group('nccl',init_method=task['init_method'],rank=rank,world_size=2,timeout=datetime.timedelta(minutes=30))
    try:
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        records=[json.loads(line) for path in sorted(output.glob('labels_*.jsonl')) for line in path.read_text().splitlines()]
        by_sample={r['sample']:r for r in records}
        train_ids=sorted(r['sample'] for r in task['manifest']['records'] if r['split']=='train')
        if len(train_ids)!=800 or any(sample not in by_sample for sample in train_ids):raise ValueError('Missing training labels')
        local=train_ids[rank::2]
        # Fixed OPT embeddings are computed once and retained only in memory.
        # No continuation embeddings are extracted or passed to the router.
        embeddings=torch.empty((len(local),args.prompt_length,model.config.hidden_size),dtype=torch.float16)
        with torch.inference_mode():
            for offset in range(0,len(local),8):
                samples=local[offset:offset+8]
                ids=torch.tensor(data[samples,:args.prompt_length],dtype=torch.long,device=device)
                embeddings[offset:offset+len(samples)].copy_(raw_prompt_embeddings(model,ids).cpu())
        packed=[targets_from_record(by_sample[s]) for s in local]
        states=torch.from_numpy(np.stack([p[0] for p in packed]));targets=torch.from_numpy(np.stack([p[1] for p in packed]))
        valid=torch.from_numpy(np.stack([p[2] for p in packed]));chosen=torch.from_numpy(np.stack([p[3] for p in packed]))
        router=SequentialGreedyRouter().to(device)
        ddp=DistributedDataParallel(router,device_ids=[rank],broadcast_buffers=False)
        optimizer=torch.optim.AdamW(ddp.parameters(),lr=2e-4,weight_decay=.01)
        batch_size=8;updates_per_epoch=math.ceil(len(local)/batch_size);updates_total=updates_per_epoch*task['epochs']
        def schedule(update):
            warmup=max(1,updates_total//20)
            if update<warmup:return (update+1)/warmup
            fraction=(update-warmup)/max(1,updates_total-warmup)
            return .1+.9*.5*(1+math.cos(math.pi*fraction))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        initial=validation(model,router,data,task['validation_ids'],args.prompt_length,task['static'],rank,2)
        best=float('inf');best_path=None;history=[]
        if rank==0:
            write_json(output/'router_training_protocol.json',{'architecture':'Prompt Transformer width256, two encoder layers; next-layer head conditioned on selected-layer set and step.',
                       'training_documents':800,'validation_label_documents':100,'checkpoint_validation_documents':task['validation_ids'],
                       'base_model_frozen':True,'recovery_finetuning_performed':False,
                       'input':'128 prompt embeddings only','label':'Joint greedy continuation-KL candidate costs at six conditional states',
                       'loss':'Soft candidate ranking cross entropy + 0.25 hard greedy choice cross entropy',
                       'temperature':'max(0.005, 0.15 * (candidate KL 90th percentile - best candidate KL)) per state',
                       'epochs':task['epochs'],'batch_size_per_gpu':8,'world_size':2,'optimizer':'AdamW lr2e-4 weight_decay0.01',
                       'checkpoint_selection':'Lowest continuation KL on fixed 32-document validation subset at epochs1,4,8,...,24',
                       'initial_random_router_validation':initial,'source_sha256':sha256(__file__)})
        for epoch in range(1,task['epochs']+1):
            ddp.train();generator=torch.Generator().manual_seed(args.seed+epoch*10+rank)
            order=torch.randperm(len(local),generator=generator).tolist();loss_sum=0.;batches=0
            for offset in range(0,len(order),batch_size):
                indices=order[offset:offset+batch_size]
                batch_embeddings=embeddings[indices].to(device);batch_states=states[indices].to(device)
                batch_targets=targets[indices].to(device);batch_valid=valid[indices].to(device)
                batch_chosen=chosen[indices].to(device)
                steps=torch.arange(1,7,device=device).expand(len(indices),-1)
                optimizer.zero_grad(set_to_none=True)
                predicted=ddp(batch_embeddings,batch_states,steps)
                selection_logits=(-predicted).masked_fill(~batch_valid,-1e9)
                log_probs=F.log_softmax(selection_logits,dim=-1)
                soft_loss=-(batch_targets*log_probs).sum(-1).mean()
                hard_loss=F.cross_entropy(selection_logits.reshape(-1,40),batch_chosen.reshape(-1))
                loss=soft_loss+.25*hard_loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite routing loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.)
                optimizer.step();scheduler.step();loss_sum+=loss.item();batches+=1
            losses=torch.tensor([loss_sum,batches],dtype=torch.float64,device=device);dist.all_reduce(losses)
            record={'epoch':epoch,'mean_loss':(losses[0]/losses[1]).item(),'lr':scheduler.get_last_lr()[0]}
            if epoch==1 or epoch%4==0 or epoch==task['epochs']:
                router.eval();score=validation(model,router,data,task['validation_ids'],args.prompt_length,task['static'],rank,2)
                record.update(score)
                if rank==0 and score['continuation_kl']<best:
                    best=score['continuation_kl'];best_path=output/f'private_router_epoch_{epoch:03d}.pth'
                    with best_path.open('xb') as stream:torch.save({'state_dict':router.state_dict(),'epoch':epoch,'validation':score},stream)
            history.append(record)
            if rank==0:results.put({'kind':'fit_progress',**record})
        if rank==0:
            write_json(output/'router_training_history.json',history)
            write_json(output/'router_selected_checkpoint.json',{'path':str(best_path),'sha256':sha256(best_path),
                       'best_validation_continuation_kl':best,'selection_uses_test':False})
        dist.barrier()
        selected_path=[str(best_path) if rank==0 else None];dist.broadcast_object_list(selected_path,src=0)
        return {'checkpoint':selected_path[0],'training_documents_per_rank':len(local),'epochs':task['epochs']}
    finally:dist.destroy_process_group()
