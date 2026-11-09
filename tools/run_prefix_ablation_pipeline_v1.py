"""Wait for stage9, then finite GPU probe, audit, main training and publication."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def proc_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def event(name, **fields):
    print(json.dumps({'event': name, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--predecessor-run', type=Path, required=True)
    p.add_argument('--predecessor-pid', type=int, required=True)
    p.add_argument('--probe-run', type=Path, required=True)
    p.add_argument('--main-run', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    python = '/home/liushenyu/anaconda3/envs/traindata/bin/python'
    original_cpus = sorted(os.sched_getaffinity(0))
    predecessor = args.predecessor_run.resolve()
    allocation = json.loads((predecessor / 'allocation.json').read_text())
    os.sched_setaffinity(0, allocation['cpu_ids'][40:42])
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    identity = proc_start(args.predecessor_pid)
    event('waiting_for_stage9', predecessor_run=str(predecessor), predecessor_pid=args.predecessor_pid,
        predecessor_start=identity, main_run=str(args.main_run))
    state, code = 'failed', 1
    try:
        if args.probe_run.exists() or args.main_run.exists():
            raise FileExistsError('Both queued run directories must be new')
        if identity is not None:
            while proc_start(args.predecessor_pid) == identity:
                time.sleep(20)
        status = json.loads((predecessor / 'status.json').read_text())
        verification = json.loads((predecessor / 'verification.json').read_text())
        if status.get('state') != 'completed' or verification.get('status') != 'passed' or verification.get('probe'):
            raise ValueError('Stage9 did not finish and pass audit; queued GPU work stopped')
        event('stage9_audited', rows=verification['verified_rows'])
        for is_probe, run in [(True, args.probe_run.resolve()), (False, args.main_run.resolve())]:
            # Resource launcher must see original affinity to reserve its proportional42.
            os.sched_setaffinity(0, original_cpus)
            command = [python, '-B', str(root / 'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3',
                '--run-dir', str(run), '--', python, '-B', '-u', str(root / 'tools/train_prefix_ablation_four_gpu_v1.py')]
            if is_probe:
                command.append('--probe')
            event('starting_probe' if is_probe else 'starting_main', run=str(run))
            code = subprocess.run(command, cwd=root).returncode
            if code:
                raise RuntimeError(f'GPU job failed with exit code {code}')
            os.sched_setaffinity(0, [allocation['cpu_ids'][40]])
            command = [python, '-B', '-u', str(root / 'tools/finalize_prefix_ablation_v1.py'),
                '--root', str(root), '--run', str(run)]
            if not is_probe:
                command.append('--publish')
            code = subprocess.run(command, cwd=root).returncode
            if code:
                raise RuntimeError(f'Audit/report/publication failed with exit code {code}')
            event('probe_passed' if is_probe else 'main_finalized', run=str(run))
        state, code = 'finalized', 0
    except Exception as error:
        event('pipeline_failed', error_type=type(error).__name__, error=str(error))
    record = {'state': state, 'exit_code': code or 1 if state != 'finalized' else 0,
        'main_run': str(args.main_run.resolve()), 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    completion = root / 'analysis/stage10_prefix_pipeline_completion_20261109.json'
    with completion.open('x') as stream:
        json.dump(record, stream, indent=2)
    event('pipeline_complete', **record)
    return record['exit_code']


if __name__ == '__main__':
    sys.exit(main())
