"""Publish only append-only sources, aggregate Stage25 results and audited receipts."""
import os
from pathlib import Path

from full_context_experiment_v1 import write
from run_static_swap_pipeline_v1 import publish


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs/stage25_crossfit_20261113_02_main'
    probe = root / 'runs/stage25_crossfit_20261113_02_probe'
    allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, allowed[-2:])
    sources = ['crossfit_routing_core_v1.py', 'run_crossfit_routing_four_gpu_v1.py',
        'run_crossfit_routing_four_gpu_v2.py', 'launch_crossfit_routing_v1.py', 'launch_crossfit_routing_v2.py',
        'audit_crossfit_routing_v1.py', 'summarize_crossfit_routing_v1.py', 'publish_crossfit_routing_v1.py']
    files = [root / 'tools' / name for name in sources]
    files += [root / 'analysis' / name for name in ('STAGE25_CROSSFIT_CONCLUSION_20261113.zh-CN.md',
        'stage25_crossfit_20261113_02_main_results.zh-CN.md', 'stage25_crossfit_20261113_02_completion.json')]
    files += [run / name for name in ('protocol.json', 'summary.json', 'policy_choices.json', 'verification.json',
        'independent_audit.json', 'diagnostic_interpretation.json', 'preservation_after.json', 'allocation.json', 'cpu_checks.json', 'status.json')]
    files += [run / f'worker_{rank:03d}.json' for rank in range(4)]
    files += [run / f'f{rank}_s{seed}_{kind}.json' for rank in range(4) for seed in (579, 580, 581) for kind in ('fit', 'history')]
    files += [probe / name for name in ('protocol.json', 'summary.json', 'verification.json', 'preservation_after.json', 'cpu_checks.json', 'status.json')]
    assert len(files) == len(set(files))
    assert all(path.is_file() and path.stat().st_size < 1024 * 1024 and not path.name.startswith('private_') for path in files)
    receipt = publish(root, files, 'stage25_crossfit_20261113_02_audited',
        'Report train-only grouped crossfit: small uncertain routing gains and fixed-bank learnability limits', dict(os.environ))
    write(run / 'publication_verified.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
