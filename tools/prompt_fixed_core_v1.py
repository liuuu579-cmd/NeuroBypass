"""OPT prompt-fixed routes, cache provenance, continuation scores and greedy router.

Original source and model files remain read-only. Beam search, heterogeneous
batched routes and switching an existing cache to a different route are outside
this initial implementation.
"""
import contextlib
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache


def valid_route(model,skips):
    route=tuple(sorted(int(i) for i in skips))
    if len(route)!=len(set(route)) or any(i<2 or i>=len(model.model.decoder.layers) for i in route):
        raise ValueError('Routes must contain distinct unprotected layer indices')
    return route


def bind_cache(cache,route):
    previous=getattr(cache,'_neurobypass_route',None)
    if previous is None:
        if cache.get_seq_length()!=0:raise ValueError('Cannot adopt a populated cache with unknown routing provenance')
        cache._neurobypass_route=tuple(route)
    elif previous!=tuple(route):
        raise ValueError('Route changed: rebuild the entire context into a new cache')


@contextlib.contextmanager
def fixed_route(model,skips):
    route=valid_route(model,skips)
    saved=[]
    def bypass(self,hidden_states,*args,**kwargs):
        if kwargs.get('output_attentions',False):raise ValueError('Attention-map outputs unsupported')
        return (hidden_states,)
    try:
        for index in route:
            layer=model.model.decoder.layers[index]
            saved.append((layer,'forward' in layer.__dict__,layer.__dict__.get('forward')))
            layer.forward=MethodType(bypass,layer)
        yield
    finally:
        for layer,existed,forward in saved:
            if existed:layer.forward=forward
            else:del layer.forward


class FixedRouteSession:
    """One prompt and answer under one route; new turns create new sessions."""
    def __init__(self,model,skips):
        self.model=model;self.route=valid_route(model,skips)
        self.cache=DynamicCache();bind_cache(self.cache,self.route)
        self.length=0

    def forward(self,ids):
        if ids.ndim!=2 or ids.size(0)!=1 or ids.size(1)<1:raise ValueError('Initial session supports one nonempty sequence')
        bind_cache(self.cache,self.route)
        if self.length!=self.cache.get_seq_length():raise ValueError('Cache length was changed externally')
        with fixed_route(self.model,self.route):
            output=self.model(ids,attention_mask=torch.ones((1,self.length+ids.size(1)),dtype=torch.long,device=ids.device),
                              past_key_values=self.cache,use_cache=True)
        self.length+=ids.size(1)
        if output.past_key_values is not self.cache:raise ValueError('Cache identity changed')
        for i in range(len(self.model.model.decoder.layers)):
            expected=0 if i in self.route else self.length
            if self.cache.get_seq_length(i)!=expected:raise ValueError(f'Unexpected cache length at layer {i}')
        return output.logits

    def cache_bytes(self):
        return sum(t.numel()*t.element_size() for layer in self.cache.layers
                   for t in [layer.keys,layer.values] if t is not None)


def continuation_logits(model,ids,prompt_length,skips):
    """Teacher forcing with fixed route; score only the continuation positions."""
    if not 1<=prompt_length<ids.size(1):raise ValueError('Need a nonempty prompt and continuation')
    with fixed_route(model,skips):
        states=model.model(ids[:,:-1],attention_mask=torch.ones_like(ids[:,:-1]),use_cache=False).last_hidden_state
        return model.lm_head(states[:,prompt_length-1:]).float()


def continuation_metrics(logits,teacher_log_probs,ids,prompt_length):
    log_probs=F.log_softmax(logits.float(),dim=-1)
    kl=(teacher_log_probs.exp()*(teacher_log_probs-log_probs)).sum(-1)
    targets=ids[:,prompt_length:]
    nll=F.nll_loss(log_probs.reshape(-1,log_probs.size(-1)),targets.reshape(-1)).item()
    return {'continuation_kl':kl.mean().item(),'first_next_token_kl':kl[:,0].mean().item(),
            'continuation_nll':nll,'continuation_tokens':targets.numel()}


def cached_continuation_logits(model,ids,prompt_length,skips):
    session=FixedRouteSession(model,skips)
    pieces=[session.forward(ids[:,:prompt_length])[:,-1:]]
    for position in range(prompt_length,ids.size(1)-1):
        pieces.append(session.forward(ids[:,position:position+1]))
    return torch.cat(pieces,dim=1).float(),session


def raw_prompt_embeddings(model,ids):
    decoder=model.model.decoder
    embeddings=decoder.embed_tokens(ids)
    if decoder.project_in is not None:embeddings=decoder.project_in(embeddings)
    return embeddings+decoder.embed_positions(torch.ones_like(ids),past_key_values_length=0)


def best_candidate(costs):
    if not costs or any(not torch.isfinite(torch.tensor(float(v))) for _,v in costs):raise ValueError('Invalid candidate costs')
    if len({i for i,_ in costs})!=len(costs):raise ValueError('Duplicate candidate index')
    return min(costs,key=lambda item:(item[1],item[0]))


class SequentialGreedyRouter(nn.Module):
    """Encode prompt once; condition every next-layer score on the current set.

    Lower scores mean lower predicted damage and are selected for skipping.
    Saved keep masks retain the historic convention 1=keep, 0=skip.
    """
    def __init__(self,input_dim=5120,width=256,layers=40,max_k=6,dropout=.1):
        super().__init__();self.layers=layers;self.max_k=max_k
        self.projection=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width),nn.GELU())
        block=nn.TransformerEncoderLayer(width,4,width*2,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,2,enable_nested_tensor=False)
        self.pool=nn.Linear(width,1)
        self.set_projection=nn.Linear(layers,width,bias=False)
        self.step_embedding=nn.Embedding(max_k+1,width)
        self.head=nn.Sequential(nn.LayerNorm(width*3),nn.Linear(width*3,width),nn.GELU(),nn.Linear(width,layers))

    def encode(self,embeddings):
        hidden=self.encoder(self.projection(embeddings.float()))
        weights=self.pool(hidden).softmax(dim=1)
        return (hidden*weights).sum(dim=1)

    def score(self,context,selected,steps):
        if selected.ndim==2:
            return self.head(torch.cat([context,self.set_projection(selected.float()),self.step_embedding(steps)],dim=-1))
        return self.head(torch.cat([context[:,None,:].expand(-1,selected.size(1),-1),
                                   self.set_projection(selected.float()),self.step_embedding(steps)],dim=-1))

    def forward(self,embeddings,selected,steps):
        return self.score(self.encode(embeddings),selected,steps)

    def route(self,embeddings,k=6):
        if not 1<=k<=self.max_k:raise ValueError('Unsupported skip budget')
        context=self.encode(embeddings);selected=context.new_zeros((context.size(0),self.layers));path=[]
        for step in range(1,k+1):
            score=self.score(context,selected,torch.full((context.size(0),),step,dtype=torch.long,device=context.device))
            unavailable=selected.bool().clone();unavailable[:,:2]=True
            index=score.masked_fill(unavailable,float('inf')).argmin(-1)
            selected.scatter_(1,index[:,None],1);path.append(index)
        return torch.stack(path,dim=1)
