"""Launch isolated3-GPU/32-CPU engineering probes, then both authorized research pilots concurrently."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PREFIX='stage19_parallel_20261112_03'


def main():
    root=Path(__file__).resolve().parents[1];python=sys.executable
    os.sched_setaffinity(0,[max(os.sched_getaffinity(0))])
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    processes=[];states=[]
    for probe in (True,False):
        current=[]
        for lane,gpus,program in [('codebook','0,1,2','run_small_codebook_three_gpu_v3.py'),('budget','3,4,5','run_adaptive_budget_three_gpu_v3.py')]:
            suffix='_probe' if probe else '';run=root/'runs'/f'{PREFIX}_{lane}{suffix}'
            log=root/'analysis'/f'{PREFIX}_{lane}{suffix}.log'
            assert not run.exists() and not log.exists()
            cmd=[python,'-B','-u',str(root/'tools/run_with_resources_v2.py'),'--gpus',gpus,'--run-dir',str(run),'--',python,'-B','-u',str(root/'tools'/program)]
            if probe:cmd.append('--probe')
            with log.open('xb') as stream:
                proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda:os.sched_setaffinity(0,range(os.cpu_count())))
            item={'lane':lane,'probe':probe,'pid':proc.pid,'run':str(run),'log':str(log),'gpus':gpus,'cpu_count':32}
            current.append((proc,item));print(json.dumps({'event':'started',**item}),flush=True)
            # Registry allocation is atomic; both children independently reserve
            # disjoint32-core CPU sets before their3 workers start.
            time.sleep(1)
        for proc,item in current:
            code=proc.wait();states.append({**item,'exit_code':code});print(json.dumps({'event':'finished',**states[-1]}),flush=True)
        if probe and any(item['exit_code']!=0 for item in states):break
    result={'state':'completed' if len(states)==4 and all(r['exit_code']==0 for r in states) else 'failed',
        'runs':states,'prefix':PREFIX}
    path=root/'analysis'/f'{PREFIX}_completion.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
    return 0 if result['state']=='completed' else 1


if __name__=='__main__':sys.exit(main())
