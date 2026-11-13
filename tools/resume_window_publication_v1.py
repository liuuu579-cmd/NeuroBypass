"""Resume the audited Stage28 allowlist with bounded immutable text-tree batches."""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]; cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cpus[40:42])
    prefix = 'stage28_window_supervision_20261113_01'; run = root / 'runs' / (prefix + '_main')
    assert json.loads((run / 'independent_audit.json').read_text())['passed']
    assert not (run / 'publication_verified.json').exists()
    paths = json.loads((root / 'analysis' / (prefix + '_audited_upload.json')).read_text())
    paths += ['tools/publish_stage28_git_v1.py', 'tools/publish_text_additions_v8.py', 'tools/resume_window_publication_v1.py']
    assert len(paths) == len(set(paths))
    manifest = root / 'analysis' / (prefix + '_batched_upload.json')
    with manifest.open('x') as stream:
        json.dump(paths, stream, indent=2); stream.write('\n')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    for attempt in range(1, 4):
        receipt = root / 'analysis' / (prefix + f'_batched_publication_attempt{attempt}.json')
        result = subprocess.run([sys.executable, '-B', '-u', str(root / 'tools/publish_text_additions_v8.py'),
            '--root', str(root), '--repo', 'liuuu579-cmd/NeuroBypass', '--files', str(manifest),
            '--message', 'Compare matched first32 and four-window joint KL supervision with no-input controls and independent audit',
            '--receipt', str(receipt)], cwd=root, env=env)
        if result.returncode == 0:
            value = json.loads(receipt.read_text())
            with (run / 'publication_verified.json').open('x') as stream:
                json.dump(value, stream, indent=2); stream.write('\n')
            print(json.dumps({'publication_verified': True, **value}), flush=True)
            return
    raise RuntimeError('Publication not verified; retain computation and all failed publication evidence')


if __name__ == '__main__':
    main()
