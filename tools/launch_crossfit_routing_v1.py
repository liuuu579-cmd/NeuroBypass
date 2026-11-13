"""Stage25 four-GPU probe then formal train-only crossfit; immutable outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path


PREFIX = 'stage25_crossfit_20261113_01'


def main():
    root = Path(__file__).resolve().parents[1]; allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [allowed[41]])
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    states = []
    for probe in (True, False):
        name = PREFIX + ('_probe' if probe else '_main'); run = root / 'runs' / name
        log = root / 'analysis' / (name + '.log'); assert not run.exists() and not log.exists()
        cmd = [sys.executable, '-B', '-u', str(root / 'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3',
            '--run-dir', str(run), '--', sys.executable, '-B', '-u', str(root / 'tools/run_crossfit_routing_four_gpu_v1.py')]
        if probe:
            cmd.append('--probe')
        with log.open('xb') as stream:
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, allowed))
        item = {'probe': probe, 'run': str(run), 'log': str(log), 'pid': process.pid}
        print(json.dumps({'event': 'started', **item}), flush=True)
        item['exit_code'] = process.wait(); states.append(item)
        print(json.dumps({'event': 'finished', **item}), flush=True)
        if item['exit_code']:
            break
    result = {'state': 'completed' if len(states) == 2 and all(r['exit_code'] == 0 for r in states) else 'failed', 'runs': states}
    with (root / 'analysis' / (PREFIX + '_completion.json')).open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return 0 if result['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
