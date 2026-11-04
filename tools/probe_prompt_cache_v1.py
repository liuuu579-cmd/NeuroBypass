"""Quantify FP16 cached/full numerical differences without relaxing assertions."""
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import cached_continuation_logits,continuation_logits,FixedRouteSession,fixed_route
from stage1_benchmark_v2 import write_json


def worker(rank,queue):
    try:
        torch.set_num_threads(10);torch.set_num_interop_threads(1);torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained('/home/liushenyu/opt-13b',torch_dtype=torch.float16,
              local_files_only=True,attn_implementation='sdpa').to(f'cuda:{rank}').eval();model.requires_grad_(False)
        run=Path('/home/liushenyu/test/NeuroBypass/runs/stage3_prompt_fixed_pilot_20261104_01')
        manifest=json.loads((run/'data_manifest.json').read_text())
        samples=[r['sample'] for r in manifest['records'] if r['split']=='pilot'][:2]
        data=np.load(run/'private_prompt_continuation_ids.npy');records=[]
        with torch.inference_mode():
            for sample in samples:
                route=[] if rank==0 else [2,5,9,11,14,18]
                ids=torch.tensor(data[sample],dtype=torch.long,device=f'cuda:{rank}').unsqueeze(0)
                expected=continuation_logits(model,ids,128,route)
                actual,session=cached_continuation_logits(model,ids,128,route)
                a=F.log_softmax(actual,dim=-1);b=F.log_softmax(expected,dim=-1)
                pointwise=(b.exp()*(b-a)).sum(-1)
                # Full forward using the same sequence length but a larger
                # lm_head call isolates head GEMM grouping from decoder error.
                with fixed_route(model,route):full=model(ids[:,:-1],use_cache=False).logits[:,127:].float()
                fullb=F.log_softmax(full,dim=-1)
                headkl=(b.exp()*(b-fullb)).sum(-1)
                # Independently recompute all prefixes serially: this has the
                # same semantics as full parallel teacher forcing.
                recomputed=[]
                with fixed_route(model,route):
                    for position in range(128,160):
                        states=model.model(ids[:,:position],use_cache=False).last_hidden_state[:,-1:]
                        recomputed.append(model.lm_head(states).float())
                recomputed=torch.cat(recomputed,dim=1);c=F.log_softmax(recomputed,dim=-1)
                serialkl=(c.exp()*(c-a)).sum(-1)
                records.append({'rank':rank,'sample':sample,'route':route,'mean_kl':pointwise.mean().item(),
                                'max_position_kl':pointwise.max().item(),'logit_abs_max':(actual-expected).abs().max().item(),
                                'logit_abs_mean':(actual-expected).abs().mean().item(),
                                'argmax_agreement':(actual.argmax(-1)==expected.argmax(-1)).float().mean().item(),
                                'head_gemm_mean_kl':headkl.mean().item(),
                                'cached_vs_serial_prefix_mean_kl':serialkl.mean().item(),
                                'cached_vs_serial_logit_abs_max':(actual-recomputed).abs().max().item(),
                                'per_position_kl':pointwise.flatten().tolist(),'cache_bytes':session.cache_bytes()})
        queue.put({'kind':'done','records':records})
    except BaseException:queue.put({'kind':'error','traceback':traceback.format_exc()})


if __name__=='__main__':
    context=mp.get_context('spawn');queue=context.Queue()
    processes=[context.Process(target=worker,args=(rank,queue)) for rank in range(2)]
    for process in processes:process.start()
    outcomes=[queue.get(timeout=240) for _ in processes]
    for process in processes:process.join(timeout=10)
    write_json(Path(os.environ['NEUROBYPASS_RUN_DIR'])/'cache_numerics.json',outcomes)
    print(json.dumps(outcomes),flush=True)
    if any(o['kind']=='error' for o in outcomes):raise RuntimeError('Diagnostic worker failed')
