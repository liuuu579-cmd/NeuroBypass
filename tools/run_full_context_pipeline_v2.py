"""Finite probe gates -> complete retraining -> quality/crossover/cost -> publish.

Adopts a completed training probe, never overwrites any existing run. All GPU
steps use the proportional resource launcher. Failure writes a concrete receipt
and prevents dependent steps from running.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from finalize_full_context_v1 import audit_run
from full_context_experiment_v1 import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training-probe', type=Path, required=True)
    p.add_argument('--data-run', type=Path, required=True)
    p.add_argument('--prefix', default='stage13_full_context_20261110_01')
    a = p.parse_args(); root = a.root.resolve(); py = sys.executable
    cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cpus[40:42])
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    completion = root/'analysis'/f'{a.prefix}_completion.json'
    runs = {name: root/'runs'/f'{a.prefix}_{name}' for name in
        ['probe_development', 'probe_cost', 'training', 'development', 'horizontal', 'cost']}
    if completion.exists() or any(r.exists() for r in runs.values()):
        raise FileExistsError('Pipeline requires new run and completion paths')
    events = root/'analysis'/f'{a.prefix}_events.jsonl'
    phase, state, error = 'training_probe_audit', 'failed', None
    def event(kind, **values):
        row = {'event': kind, 'phase': phase, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **values}
        with events.open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    def gpu(script, run, args):
        event('starting', run=str(run), program=script)
        command = [py, '-B', '-u', str(root/'tools/run_with_resources_v2.py'), '--gpus', '0,1,2,3',
            '--run-dir', str(run), '--', py, '-B', '-u', str(root/'tools'/script), *args]
        subprocess.run(command, cwd=root, env=env, check=True,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    def audit(run, kind):
        result = audit_run(run, kind)
        receipt = run/'independent_audit.json'
        if receipt.exists():
            if json.loads(receipt.read_text()) != result: raise ValueError('Conflicting audit receipt')
        else: write(receipt, result)
        event('audited', run=str(run), run_kind=kind)
    try:
        audit(a.training_probe.resolve(), 'train')
        phase = 'probe_development'
        gpu('evaluate_full_context_four_gpu_v1.py', runs[phase],
            ['--training-run', str(a.training_probe.resolve()), '--mode', 'development', '--probe'])
        audit(runs[phase], 'quality')
        phase = 'probe_cost'
        gpu('measure_full_context_cost_four_gpu_v2.py', runs[phase],
            ['--training-run', str(a.training_probe.resolve()), '--probe'])
        audit(runs[phase], 'cost')
        phase = 'training'
        gpu('train_full_context_four_gpu_v2.py', runs[phase], ['--data-run', str(a.data_run.resolve())])
        audit(runs[phase], 'train')
        for mode in ['development', 'horizontal']:
            phase = mode
            gpu('evaluate_full_context_four_gpu_v1.py', runs[phase],
                ['--training-run', str(runs['training']), '--mode', mode])
            audit(runs[phase], 'quality')
        phase = 'cost'
        gpu('measure_full_context_cost_four_gpu_v2.py', runs[phase], ['--training-run', str(runs['training'])])
        audit(runs[phase], 'cost')
        phase = 'final_report'
        run_file = root/'analysis'/f'{a.prefix}_runs.json'
        write(run_file, {k: str(v) for k, v in runs.items() if not k.startswith('probe_')})
        report = root/'analysis/STAGE13_FULL_CONTEXT_RETRAIN_RESULTS_20261110.zh-CN.md'
        files = root/'analysis'/f'{a.prefix}_completed_upload.json'
        subprocess.run([py, '-B', str(root/'tools/finalize_full_context_v1.py'), '--all-runs', str(run_file),
            '--report', str(report), '--upload-list', str(files)], cwd=root, env=env, check=True)
        phase = 'publication'
        subprocess.run([py, '-B', str(root/'tools/publish_additions_v7.py'), '--root', str(root),
            '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(files), '--message',
            'Report full-context retraining and audited quality and residency reruns', '--receipt',
            str(root/'analysis'/f'{a.prefix}_results_publication.json')], cwd=root, env=env, check=True)
        state = 'completed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        event('failed', error=error)
    write(completion, {'state': state, 'phase': phase, 'error': error,
        'runs': {k: str(v) for k, v in runs.items()},
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return 0 if state == 'completed' else 1


if __name__ == '__main__': sys.exit(main())
