"""Probe then main route-fidelity crossover with CPU audit and publication."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--probe-run', type=Path, required=True); p.add_argument('--main-run', type=Path, required=True); p.add_argument('--completion', type=Path, required=True)
    a = p.parse_args(); root=a.root.resolve(); py=sys.executable; cpus=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,cpus[40:42])
    if any(x.exists() for x in [a.probe_run,a.main_run,a.completion]): raise FileExistsError('New Stage12 paths required')
    state='failed'; error=None
    try:
        for probe, run in [(True,a.probe_run.resolve()),(False,a.main_run.resolve())]:
            cmd=[py,'-B',str(root/'tools/run_with_resources_v2.py'),'--gpus','0,1,2,3','--run-dir',str(run),'--',py,'-B','-u',str(root/'tools/evaluate_route_fidelity_four_gpu_v2.py')]
            if probe: cmd.append('--probe')
            subprocess.run(cmd,cwd=root,check=True,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
            cmd=[py,'-B','-u',str(root/'tools/finalize_route_fidelity_v1.py'),'--root',str(root),'--run',str(run)]
            if not probe:cmd.append('--publish')
            subprocess.run(cmd,cwd=root,check=True)
        state='finalized'
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    result={'state':state,'error':error,'main_run':str(a.main_run.resolve()),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with a.completion.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True);return 0 if state=='finalized' else 1


if __name__=='__main__':sys.exit(main())
