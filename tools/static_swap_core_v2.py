"""Static-start prompt-conditioned joint-mask costs and at most two exchanges.

One exchange restores an originally skipped layer and skips an originally kept
layer. Every reachable complete mask is scored independently, never by adding
single-layer damage. All source files pre-dating this module stay immutable.
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def build_bank(training, actions_count=16):
    """Shortlist atomic exchanges from TRAIN expert-mask disagreements only."""
    config=json.loads((training/'protocol.json').read_text())
    static=sorted(config['global_path'][:16]);base=set(static);votes=Counter();sources=[];seen=set()
    for m in (0,6):
        for path in sorted(training.glob(f'm{m:02d}_labels_*.jsonl')):
            sources.append(path)
            for row in read_lines(path):
                if row['split']!='train':continue
                assert row['sample'] in config['train_ids'] and (m,row['sample']) not in seen
                seen.add((m,row['sample']))
                target=set(row['ordered_skips'][:16]);restore=base-target;skip=target-base
                assert len(restore)==len(skip)
                if restore:
                    for a in restore:
                        for b in skip:votes[a,b]+=1/(len(restore)*len(skip))
    assert len(seen)==1600
    actions=[list(pair) for pair,_ in sorted(votes.items(),key=lambda v:(-v[1],v[0]))[:actions_count]]
    routes=[{'index':0,'actions':[],'skips':static}]
    for index,(a,b) in enumerate(actions):
        assert a in base and b not in base and min(a,b)>=2
        routes.append({'index':len(routes),'actions':[index],'skips':sorted((base-{a})|{b})})
    for i,j in combinations(range(len(actions)),2):
        a,b=actions[i];c,d=actions[j]
        if a==c or b==d:continue
        routes.append({'index':len(routes),'actions':[i,j],'skips':sorted((base-{a,c})|{b,d})})
    # Different pairings can produce the SAME final mask. Keep one cost/output
    # per mask while retaining every valid greedy transition into that mask.
    unique=[];by_mask={}
    for route in routes:
        key=tuple(route['skips'])
        if key not in by_mask:
            by_mask[key]=len(unique)
            unique.append({**route,'index':len(unique),'action_paths':[route['actions']]})
        elif route['actions'] not in unique[by_mask[key]]['action_paths']:
            unique[by_mask[key]]['action_paths'].append(route['actions'])
    routes=unique
    assert len({tuple(r['skips']) for r in routes})==len(routes)
    assert all(len(r['skips'])==16 and len(set(r['skips'])^base)==2*len(r['actions']) for r in routes)
    return {'k':16,'static_skips':static,'actions':actions,'routes':routes,
        'shortlist_votes':[votes[tuple(a)] for a in actions],
        'shortlist_policy':'Top16 exchange pairs by normalized cooccurrence of static-vs-m0/m6 expert disagreement on original800 TRAIN documents only. Not an independent-layer damage estimate.',
        'source_sha256':{str(p):sha256(p) for p in sources},'training_records_used':len(seen),
        'candidate_policy':'Static + all16 atomic exchanges + all compatible two-exchange compositions. Full joint costs for EVERY reachable mask cover on-policy and off-policy second states.'}


def greedy_indices(costs,bank,maximum=2):
    """Greedy exchange, include stop at every step; static wins exact ties."""
    if maximum not in (1,2) or costs.shape[-1]!=len(bank['routes']):raise ValueError('Invalid bank/budget')
    shape=costs.shape[:-1];flat=costs.reshape(-1,costs.shape[-1]);chosen=[]
    for row in flat:
        current=0
        for _ in range(maximum):
            selected=set(bank['routes'][current]['actions'])
            valid=[current]+[r['index'] for r in bank['routes']
                if len(r['actions'])==len(selected)+1 and any(selected<set(path) for path in r.get('action_paths',[r['actions']]))]
            next_index=min(valid,key=lambda i:(float(row[i]),0 if i==current else 1,i))
            if next_index==current:break
            current=next_index
        chosen.append(current)
    return torch.tensor(chosen,device=costs.device,dtype=torch.long).reshape(shape)


class StaticSwapRouter(nn.Module):
    """Full prompt encoder + training-mean joint-cost prior + zero-init residual."""
    def __init__(self,candidates,input_dim=5120,width=128,dropout=.1,observe=True):
        super().__init__();self.observe=observe
        self.projection=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width),nn.GELU())
        block=nn.TransformerEncoderLayer(width,4,width*2,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,2,enable_nested_tensor=False)
        self.pool=nn.Linear(width,1)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,candidates-1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('prior',torch.zeros(candidates))
        self.register_buffer('scale',torch.tensor(.1))

    def encode(self,embeddings,valid=None):
        if not self.observe:
            # Identical architecture/parameter count, no prompt length/position features.
            embeddings=embeddings.new_zeros((embeddings.shape[0],1,embeddings.shape[-1]))
            valid=torch.ones(embeddings.shape[:2],device=embeddings.device,dtype=torch.bool)
        if valid is None:valid=torch.ones(embeddings.shape[:2],device=embeddings.device,dtype=torch.bool)
        assert valid.shape==embeddings.shape[:2] and valid.any(1).all()
        hidden=self.encoder(self.projection(embeddings.float()),src_key_padding_mask=None if valid.all() else ~valid)
        weights=self.pool(hidden).masked_fill(~valid[...,None],-torch.inf).softmax(1)
        return (hidden*weights).sum(1)

    def forward(self,embeddings,valid=None):
        context=self.encode(embeddings,valid)
        residual=self.head(context)
        predicted=self.prior[None,1:]+self.scale*residual
        return torch.cat([predicted.new_zeros((len(context),1)),predicted],-1)


def load_router(path,device):
    record=torch.load(path,map_location='cpu',weights_only=True)
    model=StaticSwapRouter(candidates=record['candidates'],width=record['width'],observe=record['observe']).to(device)
    model.load_state_dict(record['state_dict']);model.eval().requires_grad_(False)
    return model


def regression_loss(predicted,costs,scale):
    # All targets are complete-mask costs relative to the same original static.
    target=costs-costs[:,:1]
    return F.smooth_l1_loss(predicted[:,1:]/scale,target[:,1:]/scale)


def checks():
    bank={'routes':[{'index':0,'actions':[]},{'index':1,'actions':[0]},
        {'index':2,'actions':[1]},{'index':3,'actions':[0,1]}]}
    values=torch.tensor([[0.,-.1,-.05,-.2],[0.,.1,.2,-.4],[0.,0.,0.,0.]])
    assert greedy_indices(values,bank,1).tolist()==[1,0,0]
    assert greedy_indices(values,bank,2).tolist()==[3,0,0]
    torch.manual_seed(579)
    router=StaticSwapRouter(4,input_dim=12,width=16,dropout=0).eval()
    x=torch.randn(1,193,12);prior=torch.tensor([0.,-.1,.02,-.2]);router.prior.copy_(prior)
    assert torch.equal(router(x),prior[None])
    # Test the encoder independently of zero-initialized output residual.
    suffix=x.clone();suffix[:,128:]+=5
    assert not torch.allclose(router.encode(x),router.encode(suffix))
    short=x[:,:71];pad=torch.cat([short,100*torch.randn(1,122,12)],1)
    valid=torch.arange(193)[None]<71
    error=(router.encode(short)-router.encode(pad,valid)).abs().max().item()
    assert error<2e-5
    const=StaticSwapRouter(4,input_dim=12,width=16,dropout=0,observe=False).eval()
    assert torch.equal(const.encode(x),const.encode(x[:,:17]*9))
    costs=torch.tensor([[.7,.6,.8,.5]])
    assert regression_loss(prior[None],costs,.1).isfinite()
    return {'passed':True,'stop_and_tie_semantics':True,'two_step_greedy_checked':True,
        'no_uphill_two_step_shortcut':True,'zero_residual_equals_training_prior':True,
        'suffix_after128_reaches_encoder':True,'padding_encoder_max_abs_error':error,
        'constant_control_has_no_content_or_length':True}
