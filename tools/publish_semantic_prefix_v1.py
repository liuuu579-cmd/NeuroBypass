"""Append audited Stage26 source, protocol and aggregate results to GitHub."""
import os
from pathlib import Path

from full_context_experiment_v1 import write
from run_static_swap_pipeline_v1 import publish


def main():
    root = Path(__file__).resolve().parents[1]; prefix = 'stage26_semantic_prefix_20261113_01'
    run = root / 'runs' / (prefix + '_main'); probe = root / 'runs' / (prefix + '_probe'); audit = root / 'runs' / (prefix + '_audit')
    allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, allowed[-2:])
    files = [root / 'tools' / name for name in ('semantic_prefix_core_v1.py', 'run_semantic_prefix_four_gpu_v1.py',
        'launch_semantic_prefix_v1.py', 'audit_semantic_prefix_four_gpu_v1.py', 'summarize_semantic_prefix_v1.py', 'publish_semantic_prefix_v1.py')]
    files += [root / 'analysis' / name for name in ('STAGE26_SEMANTIC_PREFIX_CONCLUSION_20261113.zh-CN.md',
        prefix + '_main_results.zh-CN.md', prefix + '_completion.json')]
    files += [run / name for name in ('protocol.json', 'summary.json', 'policy_choices.json', 'paired_initialization.json',
        'verification.json', 'independent_audit.json', 'diagnostic_interpretation.json', 'preservation_after.json', 'allocation.json', 'status.json')]
    files += [run / f'{kind}_{rank:03d}.json' for kind in ('features_worker', 'worker') for rank in range(4)]
    files += [run / f'{arm}_f{rank}_s{seed}_{kind}.json' for arm in ('hidden2', 'constant') for rank in range(4)
        for seed in (579, 580, 581) for kind in ('fit', 'history')]
    files += [probe / name for name in ('protocol.json', 'verification.json', 'preservation_after.json', 'status.json')]
    files += [probe / f'prefix_reference_{rank:03d}.json' for rank in range(4)]
    files += [audit / name for name in ('allocation.json', 'status.json')]
    files += [audit / f'worker_{rank:03d}.json' for rank in range(4)]
    assert len(files) == len(set(files))
    assert all(path.is_file() and path.stat().st_size < 1024 * 1024 and not path.name.startswith('private_') for path in files)
    receipt = publish(root, files, prefix + '_audited',
        'Report frozen two-layer semantic routing pilot against matched embedding and no-input controls', dict(os.environ))
    write(run / 'publication_verified.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
