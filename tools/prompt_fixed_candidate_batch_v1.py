"""Batch independent joint candidate evaluations for offline label search.

For different candidate routes in one batch, some skipped rows are still computed
and then discarded at that layer. This is solely an offline throughput device,
never used as evidence of inference acceleration or cached generation execution.
"""
import contextlib
from types import MethodType

import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import fixed_route


@contextlib.contextmanager
def candidate_routes(model,selected,candidates):
    if len(candidates)!=len(set(candidates)) or set(selected)&set(candidates):raise ValueError('Invalid candidate routes')
    saved=[]
    def make_forward(original,index):
        def forward(self,hidden_states,*args,**kwargs):
            if kwargs.get('use_cache',False) or kwargs.get('past_key_value') is not None:raise ValueError('Candidate batching forbids caches')
            if kwargs.get('output_attentions',False):raise ValueError('Attention maps unsupported')
            if len(candidates)==1:return (hidden_states,)
            output=original(hidden_states,*args,**kwargs)
            mask=torch.tensor([candidate==index for candidate in candidates],dtype=torch.bool,device=hidden_states.device)[:,None,None]
            return (torch.where(mask,hidden_states,output[0]),)
        return forward
    with fixed_route(model,selected):
        try:
            for index in candidates:
                if index<2 or index>=len(model.model.decoder.layers):raise ValueError('Protected or invalid candidate')
                layer=model.model.decoder.layers[index]
                saved.append((layer,'forward' in layer.__dict__,layer.__dict__.get('forward')))
                layer.forward=MethodType(make_forward(layer.forward,index),layer)
            yield
        finally:
            for layer,existed,forward in saved:
                if existed:layer.forward=forward
                else:del layer.forward


def batched_candidate_costs(model,ids,prompt_length,selected,candidates,teacher_log_probs,batch_size=8):
    costs=[]
    for offset in range(0,len(candidates),batch_size):
        group=candidates[offset:offset+batch_size]
        batch=ids[:,:-1].expand(len(group),-1).contiguous()
        with candidate_routes(model,selected,group):
            states=model.model(batch,attention_mask=torch.ones_like(batch),use_cache=False).last_hidden_state
            logits=model.lm_head(states[:,prompt_length-1:]).float()
        log_probs=F.log_softmax(logits,dim=-1)
        values=(teacher_log_probs.exp()*(teacher_log_probs-log_probs)).sum(-1).mean(-1)
        costs.extend([index,value.item()] for index,value in zip(group,values))
    return costs
