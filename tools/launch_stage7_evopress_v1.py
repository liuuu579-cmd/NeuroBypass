"""Stage30 engineering probe then224-input frozen comparison and publication."""
import json
import os
import subprocess
import sys
from pathlib import Path

PREFIX = 'stage30_stage7_evopress_20261114_01'


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
        tag = PREFIX+('_probe' if probe else '_main')
        out = root/'runs'/tag
        log = root/'analysis'/f'{tag}.log'
        assert not out.exists() and not log.exists()
        cmd = [python, '-B', '-u', str(root/'tools/run_with_resources_v2.py'), '--gpus', '0,1,2',
            '--run-dir', str(out), '--', python, '-B', '-u', str(root/'tools/compare_stage7_evopress_three_gpu_v1.py')]
        if probe:
            cmd.append('--probe')
        with log.open('xb') as stream:
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, inherited))
        state = {'probe': probe, 'pid': process.pid, 'run': str(out), 'log': str(log), 'gpus': [0, 1, 2], 'cpus': 32}
        write(root/'analysis'/f'{tag}_launch.json', state)
        print(json.dumps({'event': 'stage30_started', **state}), flush=True)
        code = process.wait()
        stages.append({**state, 'exit_code': code})
        if code:
            write(root/'analysis'/f'{PREFIX}_completion.json', {'state': 'failed', 'stages': stages})
            return code
        report = root/'analysis'/f'STAGE30_{"PROBE" if probe else "RESULTS"}_20261114.zh-CN.md'
        allowlist = root/'analysis'/f'{tag}_public_files.json'
        subprocess.run([python, '-B', str(root/'tools/finalize_stage7_evopress_v1.py'), '--run', str(out),
            '--report', str(report), '--allowlist', str(allowlist)], cwd=root, env=env, check=True)
        if not probe:
            receipt = root/'analysis'/f'{PREFIX}_publication.json'
            subprocess.run([python, '-B', '-u', str(root/'tools/publish_text_additions_v9.py'), '--root', str(root),
                '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(allowlist),
                '--message', 'Compare original Stage7 pure and hybrid routers with frozen EvoPress on a small shared subset',
                '--receipt', str(receipt)], cwd=root, env=env, check=True)
    write(root/'analysis'/f'{PREFIX}_completion.json', {'state': 'completed', 'stages': stages})
    return 0


if __name__ == '__main__':
    sys.exit(main())
