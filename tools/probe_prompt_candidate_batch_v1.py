"""Check candidate batching against individual full-route evaluations."""
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prompt_fixed_candidate_batch_v1 import batched_candidate_costs
from prompt_fixed_core_v1 import continuation_logits
from stage1_benchmark_v2 import write_json


def worker(rank,results):
    try:
        torch.set_num_threads(10);torch.set_num_interop_threads(1);torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained('/home/liushenyu/opt-13b',torch_dtype=torch.float16,
              local_files_only=True,attn_implementation='sdpa').to(f'cuda:{rank}').eval();model.requires_grad_(False)
        run=Path('/home/liushenyu/test/NeuroBypass/runs/stage3_prompt_fixed_pilot_20261104_01')
        manifest=json.loads((run/'data_manifest.json').read_text())
        sample=[r['sample'] for r in manifest['records'] if r['split']=='pilot'][rank]
        data=np.load(run/'private_prompt_continuation_ids.npy')
        ids=torch.tensor(data[sample],dtype=torch.long,device=f'cuda:{rank}').unsqueeze(0)
        records=[]
        with torch.inference_mode():
            teacher=F.log_softmax(continuation_logits(model,ids,128,[]),dim=-1)
            for selected in [[],[2,5,9]]:
                candidates=[i for i in range(2,40) if i not in selected]
                batched_candidate_costs(model,ids,128,selected,candidates[:8],teacher,8)
                torch.cuda.synchronize();started=time.perf_counter()
                batch=batched_candidate_costs(model,ids,128,selected,candidates,teacher,8)
                torch.cuda.synchronize();batch_time=time.perf_counter()-started
                started=time.perf_counter();individual=[]
                for index in candidates:
                    logp=F.log_softmax(continuation_logits(model,ids,128,selected+[index]),dim=-1)
                    individual.append([index,(teacher.exp()*(teacher-logp)).sum(-1).mean().item()])
                torch.cuda.synchronize();individual_time=time.perf_counter()-started
                errors=[abs(a[1]-b[1]) for a,b in zip(batch,individual)]
                chosen=min(batch,key=lambda pair:(pair[1],pair[0]))[0]
                best=min(individual,key=lambda pair:(pair[1],pair[0]))
                regret=dict(individual)[chosen]-best[1]
                if max(errors)>.002 or regret>.001:raise ValueError('Batched candidate scores materially differ')
                records.append({'rank':rank,'sample':sample,'selected':selected,'max_absolute_cost_error':max(errors),
                                'chosen_layer':chosen,'individual_best_layer':best[0],'individual_choice_regret':regret,
                                'batch_seconds':batch_time,'individual_seconds':individual_time,
                                'batch_scores':batch,'individual_scores':individual})
        results.put({'kind':'done','records':records})
    except BaseException:results.put({'kind':'error','traceback':traceback.format_exc()})


if __name__=='__main__':
    context=mp.get_context('spawn');results=context.Queue()
    processes=[context.Process(target=worker,args=(rank,results)) for rank in range(2)]
    for process in processes:process.start()
    records=[results.get(timeout=240) for _ in processes]
    for process in processes:process.join(timeout=10)
    write_json(Path(os.environ['NEUROBYPASS_RUN_DIR'])/'candidate_batch_probe.json',records)
    print(json.dumps(records),flush=True)
    if any(r['kind']=='error' for r in records):raise RuntimeError('Candidate batch probe failed')
