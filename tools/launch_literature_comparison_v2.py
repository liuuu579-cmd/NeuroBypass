"""Run the engineering probe then the predeclared Stage29 comparison.

Continue only after successful engineering/audit checks, irrespective of pilot
quality. No old outputs are reused or overwritten. Publish aggregates on finish.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PREFIX = 'stage29_literature_20261113_01'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    inherited = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [inherited[-1]])
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    stages = []
    for probe in (True, False):
        tag = PREFIX + ('_probe' if probe else '_main')
        run = root/'runs'/tag
        logfile = root/'analysis'/f'{tag}.log'
        assert not run.exists() and not logfile.exists()
        command = [python, '-B', '-u', str(root/'tools/run_with_resources_v2.py'),
            '--gpus', '0,1,2,3', '--run-dir', str(run), '--', python, '-B', '-u',
            str(root/'tools/run_literature_comparison_four_gpu_v2.py')]
        if probe:
            command.append('--probe')
        with logfile.open('xb') as stream:
            proc = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, inherited))
        state = {'probe': probe, 'pid': proc.pid, 'run': str(run), 'log': str(logfile),
            'gpu_indices': [0, 1, 2, 3], 'cpu_count': 42}
        write(root/'analysis'/f'{tag}_launch.json', state)
        print(json.dumps({'event': 'stage29_started', **state}), flush=True)
        code = proc.wait()
        stages.append({**state, 'exit_code': code})
        if code:
            write(root/'analysis'/f'{PREFIX}_completion.json', {'state': 'failed', 'stages': stages})
            return code
        assert json.loads((run/'verification.json').read_text())['passed']
        report = root/'analysis'/f'STAGE29_{"PROBE" if probe else "RESULTS"}_20261113.zh-CN.md'
        allowlist = root/'analysis'/f'{tag}_public_files.json'
        subprocess.run([python, '-B', str(root/'tools/finalize_literature_comparison_v2.py'),
            '--run', str(run), '--report', str(report), '--allowlist', str(allowlist)],
            cwd=root, env=env, check=True)
        # Publish main results only; the probe cannot support method claims.
        if not probe:
            subprocess.run([python, '-B', '-u', str(root/'tools/publish_text_additions_v8.py'),
                '--root', str(root), '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(allowlist),
                '--message', 'Add frozen literature comparison with whole-layer EvoPress OPT adaptation',
                '--receipt', str(root/'analysis'/f'{PREFIX}_publication.json')], cwd=root, env=env, check=True)
    write(root/'analysis'/f'{PREFIX}_completion.json', {'state': 'completed', 'stages': stages})
    return 0


if __name__ == '__main__':
    sys.exit(main())
