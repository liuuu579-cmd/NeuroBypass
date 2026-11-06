"""Finite local controller for resource-limited experiment then audit/publication.

No recurring task, notifications, or unrelated process management. A nonzero run
or finalization exit writes a new failure record, with all original evidence kept.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--ablation-run', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    python = '/home/liushenyu/anaconda3/envs/traindata/bin/python'
    command = [sys.executable, str(root / 'tools/run_with_resources_v2.py'), '--gpus', '1,3',
        '--run-dir', str(run), '--', python, '-u', str(root / 'tools/train_augmented_router_v2.py'),
        '--main-run', str(args.main_run.resolve()), '--ablation-run', str(args.ablation_run.resolve())]
    code = subprocess.run(command, cwd=root).returncode
    if code == 0:
        code = subprocess.run(['taskset', '-c', '0', python, '-B', '-u',
            str(root / 'tools/finalize_augmented_router_v1.py'), '--root', str(root),
            '--run', str(run), '--publish'], cwd=root).returncode
    record = {'state': 'finalized' if code == 0 else 'failed', 'exit_code': code,
        'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'run': str(run)}
    write_new(root / 'analysis/stage4_branch_pipeline_completion_20261105.json', record)
    print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
