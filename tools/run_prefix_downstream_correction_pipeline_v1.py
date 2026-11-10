"""Finite correction pipeline: four-GPU probe, audit, main, audit, publication."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def event(name, **fields):
    print(json.dumps({'event': name, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--probe-run', type=Path, required=True)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--completion', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    python = sys.executable
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 42:
        raise ValueError('Pipeline requires 42 available CPU cores')
    if any(p.exists() for p in [args.probe_run, args.main_run, args.completion]):
        raise FileExistsError('Correction requires exclusively new output paths')
    os.sched_setaffinity(0, available[40:42])
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    state, error = 'failed', None
    try:
        subprocess.run([python, '-B', str(root / 'tools/test_prefix_downstream_correction_v1.py'), '-v'], check=True, cwd=root)
        for probe, run, source in [
                (True, args.probe_run.resolve(), root / 'runs/stage11_prefix_downstream_probe_20261109_01'),
                (False, args.main_run.resolve(), root / 'runs/stage11_prefix_downstream_main_20261109_01')]:
            command = [python, '-B', str(root / 'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3',
                '--run-dir', str(run), '--', python, '-B', '-u', str(root / 'tools/evaluate_prefix_downstream_four_gpu_v2.py'),
                '--reuse-run', str(source)]
            if probe:
                command.append('--probe')
            event('starting_corrected_probe' if probe else 'starting_corrected_main', run=str(run))
            subprocess.run(command, check=True, cwd=root, preexec_fn=lambda: os.sched_setaffinity(0, available))
            allocation = json.loads((run / 'allocation.json').read_text())
            os.sched_setaffinity(0, allocation['cpu_ids'][40:42])
            command = [python, '-B', '-u', str(root / 'tools/finalize_prefix_downstream_v2.py'),
                '--root', str(root), '--run', str(run)]
            if not probe:
                command.append('--publish')
            subprocess.run(command, check=True, cwd=root)
            event('corrected_probe_passed' if probe else 'corrected_main_published', run=str(run))
        state = 'finalized'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        event('correction_pipeline_failed', error=error)
    result = {'state': state, 'error': error, 'main_run': str(args.main_run.resolve()),
              'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with args.completion.open('x') as stream:
        json.dump(result, stream, indent=2)
    event('correction_pipeline_complete', **result)
    return 0 if state == 'finalized' else 1


if __name__ == '__main__':
    sys.exit(main())
