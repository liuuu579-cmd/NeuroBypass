"""Append only own Stage27 sources, protocols and aggregate audited results."""
import os
from pathlib import Path

from full_context_experiment_v1 import write
from run_static_swap_pipeline_v1 import publish


def main():
    root = Path(__file__).resolve().parents[1]; prefix = 'stage27_continuation_stability_20261113_01'
    run = root / 'runs' / (prefix + '_main'); probe = root / 'runs' / (prefix + '_probe'); audit = root / 'runs' / (prefix + '_audit')
    cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cpus[-2:])
    files = [root / 'tools' / name for name in ('continuation_stability_core_v1.py', 'run_continuation_stability_four_gpu_v1.py',
        'launch_continuation_stability_v1.py', 'audit_continuation_stability_four_gpu_v1.py',
        'summarize_continuation_stability_v1.py', 'publish_continuation_stability_v1.py')]
    files += [root / 'analysis' / name for name in ('STAGE27_CONTINUATION_STABILITY_CONCLUSION_20261113.zh-CN.md',
        prefix + '_main_results.zh-CN.md', prefix + '_completion.json')]
    files += [run / name for name in ('protocol.json', 'sampling.json', 'cpu_checks.json', 'summary.json', 'verification.json',
        'independent_audit.json', 'original_window_numerical_comparison.json', 'diagnostic_interpretation.json',
        'preservation_after.json', 'allocation.json', 'status.json')]
    files += [run / f'worker_{rank:03d}.json' for rank in range(4)]
    files += [probe / name for name in ('protocol.json', 'sampling.json', 'cpu_checks.json', 'verification.json', 'preservation_after.json', 'status.json')]
    files += [probe / f'worker_{rank:03d}.json' for rank in range(4)]
    files += [audit / name for name in ('allocation.json', 'status.json')]
    files += [audit / f'worker_{rank:03d}.json' for rank in range(4)]
    assert len(files) == len(set(files))
    assert all(path.is_file() and path.stat().st_size < 1024 * 1024 and not path.name.startswith('private_') for path in files)
    receipt = publish(root, files, prefix + '_audited',
        'Diagnose fixed-mask preference stability across natural continuation windows with numerical and oracle-transfer controls', dict(os.environ))
    write(run / 'publication_verified.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
