"""Durable six-GPU engineering gate followed by the authorized K20 refinement."""
import json
import os
import subprocess
import sys
from pathlib import Path

PREFIX = 'stage20_codebook_refine_20261112_02'


def main():
    root = Path(__file__).resolve().parents[1]
    os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    states = []
    for probe in (True, False):
        name = PREFIX + ('_probe' if probe else '_main')
        run = root / 'runs' / name
        log = root / 'analysis' / (name + '.log')
        assert not run.exists() and not log.exists()
        cmd = [sys.executable, '-B', '-u', str(root / 'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3,4,5',
            '--run-dir', str(run), '--', sys.executable, '-B', '-u', str(root / 'tools/run_codebook_refinement_six_gpu_v2.py')]
        if probe:
            cmd.append('--probe')
        with log.open('xb') as stream:
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, range(os.cpu_count())))
        item = {'probe': probe, 'pid': proc.pid, 'run': str(run), 'log': str(log)}
        print(json.dumps({'event': 'started', **item}), flush=True)
        code = proc.wait()
        states.append({**item, 'exit_code': code})
        print(json.dumps({'event': 'finished', **states[-1]}), flush=True)
        if code:
            break
    result = {'state': 'completed' if len(states) == 2 and all(r['exit_code'] == 0 for r in states) else 'failed', 'runs': states}
    with (root / 'analysis' / (PREFIX + '_completion.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
    return 0 if result['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
