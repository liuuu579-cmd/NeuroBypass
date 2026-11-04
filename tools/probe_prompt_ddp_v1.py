"""Two-GPU backward/optimizer synchronization check before long label generation."""
import datetime
import json
import multiprocessing as mp
import os
import socket
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from prompt_fixed_core_v1 import SequentialGreedyRouter
from stage1_benchmark_v2 import write_json


def worker(rank,address,results):
    try:
        torch.set_num_threads(10);torch.set_num_interop_threads(1);torch.cuda.set_device(rank)
        dist.init_process_group('nccl',init_method=address,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=60))
        torch.manual_seed(579)
        router=SequentialGreedyRouter(input_dim=16,width=16,layers=8,max_k=3,dropout=0.).to(f'cuda:{rank}')
        ddp=DistributedDataParallel(router,device_ids=[rank]);optimizer=torch.optim.AdamW(ddp.parameters(),lr=.001)
        for _ in range(2):
            inputs=torch.randn(2,7,16,device=f'cuda:{rank}')+rank
            selected=torch.zeros(2,3,8,device=inputs.device);selected[:,1:,2]=1
            steps=torch.arange(1,4,device=inputs.device).expand(2,-1)
            optimizer.zero_grad(set_to_none=True);score=ddp(inputs,selected,steps)
            loss=score.square().mean();loss.backward();optimizer.step()
        flattened=torch.cat([p.detach().flatten() for p in router.parameters()])
        reference=flattened.clone();dist.broadcast(reference,src=0)
        maximum=(reference-flattened).abs().max().item()
        if maximum!=0:raise ValueError('DDP replicas diverged')
        results.put({'kind':'done','rank':rank,'maximum_parameter_difference':maximum,'cpu_ids':sorted(os.sched_getaffinity(0))})
        dist.destroy_process_group()
    except BaseException:results.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


if __name__=='__main__':
    with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    context=mp.get_context('spawn');results=context.Queue()
    processes=[context.Process(target=worker,args=(rank,f'tcp://127.0.0.1:{port}',results)) for rank in range(2)]
    for process in processes:process.start()
    records=[results.get(timeout=120) for _ in processes]
    for process in processes:process.join(timeout=10)
    write_json(Path(os.environ['NEUROBYPASS_RUN_DIR'])/'ddp_probe.json',records)
    print(json.dumps(records),flush=True)
    if any(r['kind']=='error' for r in records):raise RuntimeError('DDP preflight failed')
