"""One finite four-GPU training job followed by CPU audit/report/publication."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    python = '/home/liushenyu/anaconda3/envs/traindata/bin/python'
    code = subprocess.run([python, '-B', str(root / 'tools/run_with_resources_v2.py'),
        '--gpus', '0,1,2,3', '--run-dir', str(run), '--', python, '-B', '-u',
        str(root / 'tools/train_prompt_depth_four_gpu_v4.py')], cwd=root).returncode
    if code == 0:
        code = subprocess.run(['taskset', '-c', '0', python, '-B', '-u',
            str(root / 'tools/finalize_prompt_depth_four_gpu_v1.py'), '--root', str(root),
            '--run', str(run), '--publish'], cwd=root).returncode
    record = {'state': 'finalized' if code == 0 else 'failed', 'exit_code': code,
        'run': str(run), 'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (root / 'analysis/stage6_four_gpu_pipeline_completion_20261107.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__': sys.exit(main())
