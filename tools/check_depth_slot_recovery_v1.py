"""Small OPT checks of recovery checkpoint gradients and routed KV cache."""
import argparse
from pathlib import Path

import torch
from transformers import OPTConfig, OPTForCausalLM

from depth_slot_recovery_core_v1 import DepthSlots, attach, log_probs, objective, routed
from full_context_experiment_v1 import write
from prompt_fixed_core_v1 import FixedRouteSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(579)
    config = OPTConfig(hidden_size=32,ffn_dim=64,num_hidden_layers=40,
        num_attention_heads=4,vocab_size=128,max_position_embeddings=256,
        word_embed_proj_dim=32,dropout=0,attention_dropout=0)
    model = OPTForCausalLM(config).eval().requires_grad_(False)
    bank = DepthSlots(32,24,rank=2,alpha=4).to('cpu')
    attach(model,bank)
    x = torch.randint(3,128,(1,27)); p = 19
    skips = list(range(2,18))
    with torch.no_grad(),routed(model,bank,[],adapters=False): teacher = log_probs(model,x,p)
    # Nonzero adapters make checkpoint and cache checks sensitive to the mapping.
    with torch.no_grad():
        for slot in bank.slots:
            for projection in slot.values(): projection.b.normal_(0,.002)
    gradients = []
    for enabled in (False,True):
        bank.checkpoint_blocks = enabled; bank.zero_grad(set_to_none=True)
        with routed(model,bank,skips):
            lp = log_probs(model,x,p)
            objective(lp,teacher,x[:,p:])[0].backward()
        gradients.append({n:v.grad.clone() for n,v in bank.named_parameters()})
    error = max((gradients[0][n]-gradients[1][n]).abs().max().item() for n in gradients[0])
    if error > 2e-6: raise ValueError('Checkpointed adapter gradients differ')
    with torch.no_grad(),routed(model,bank,skips):
        bulk = model(x[:,:-1],use_cache=False).logits[:,p-1:].float()
        session = FixedRouteSession(model,skips)
        pieces = [session.forward(x[:,:p])[:,-1:]]
        pieces += [session.forward(x[:,i:i+1]) for i in range(p,x.size(1)-1)]
        cached = torch.cat(pieces,1).float()
    cache_error = (bulk-cached).abs().max().item()
    if cache_error > 2e-5: raise ValueError('Recovered cache continuation differs from bulk')
    try:
        with torch.no_grad(),routed(model,bank,list(range(3,19))):
            session.route = tuple(range(3,19)); session.forward(x[:,:1])
    except ValueError:
        route_change_rejected = True
    else:
        raise ValueError('Recovered cache accepted a changed route')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Base freeze failed')
    write(args.output,{'passed':True,'checkpoint_gradient_max_abs_error':error,
        'cached_continuation_logits_max_abs_error':cache_error,
        'populated_cache_route_change_rejected':route_change_rejected,
        'base_frozen':True,'kept':24,'skipped':16,'test_model':'random tiny40-layer OPT, CPU'})


if __name__=='__main__': main()
