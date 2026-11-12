"""Durable engineering probe then four-GPU supervision comparison; append-only outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path


PREFIX = 'stage24_supervision_20261112_01'


def main():
    root = Path(__file__).resolve().parents[1]
    allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, [allowed[-1]])
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    states = []
    for probe in (True, False):
        name = PREFIX + ('_probe' if probe else '_main'); run = root / 'runs' / name
        log = root / 'analysis' / (name + '.log')
        assert not run.exists() and not log.exists()
        cmd = [sys.executable, '-B', '-u', str(root / 'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3',
            '--run-dir', str(run), '--', sys.executable, '-B', '-u', str(root / 'tools/run_supervision_comparison_four_gpu_v1.py')]
        if probe:
            cmd.append('--probe')
        with log.open('xb') as stream:
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, allowed))
        state = {'probe': probe, 'pid': process.pid, 'run': str(run), 'log': str(log)}
        print(json.dumps({'event': 'started', **state}), flush=True)
        state['exit_code'] = process.wait(); states.append(state)
        print(json.dumps({'event': 'finished', **state}), flush=True)
        if state['exit_code']:
            break
    result = {'state': 'completed' if len(states) == 2 and all(s['exit_code'] == 0 for s in states) else 'failed', 'runs': states}
    if result['state'] == 'completed':
        # Sources and aggregate evidence are public; all private data/checkpoints stay local.
        from run_static_swap_pipeline_v1 import publish
        from full_context_experiment_v1 import write
        run = root / 'runs' / (PREFIX + '_main')
        sources = ['supervision_comparison_core_v1.py', 'run_supervision_comparison_four_gpu_v1.py', 'launch_supervision_comparison_v1.py']
        files = [root / 'tools' / name for name in sources] + [root / 'analysis' / (run.name + '_results.zh-CN.md')]
        files += [p for p in run.glob('*.json') if not p.name.startswith('private_') and p.name not in ('preservation_before.json', 'allocation.json')]
        try:
            receipt = publish(root, files, PREFIX + '_main', 'Compare matched full-prompt, future32 and mixed K20 routing supervision on four GPUs', env)
            write(run / 'publication_verified.json', receipt); result['publication'] = receipt
        except Exception as error:
            result['publication_error'] = str(error)
    with (root / 'analysis' / (PREFIX + '_completion.json')).open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return 0 if result['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
