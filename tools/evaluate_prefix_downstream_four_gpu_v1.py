"""Four-GPU downstream check for the fixed-prefix ablation endpoints.

This is a new read-only evaluator. It reuses the audited horizontal scorer and
benchmark preparation, but builds its policy list from the stage10 checkpoints.
No source result or checkpoint is changed.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

from horizontal_data_v1 import DATA, HARNESS, read_dataset, encode_question, wiki_windows
from horizontal_quality_v3 import pin_threads, prepare, score_probe, summary_tables, worker
from stage1_benchmark_v2 import sha256, stamp, write_json


def policies(root, stage6, stage10, budgets):
    static = json.loads((stage6 / 'static_paths.json').read_text())
    fixed_path = json.loads((stage6 / 'fixed_global_policy.json').read_text())['ordered_skips']
    result = [{'name': 'full', 'kind': 'static', 'k': 0, 'skips': [], 'seed': None}]
    arms = [(0, stage6, 'pure'), (4, stage10, 'm04'), (6, stage6, 'hybrid'), (8, stage10, 'm08')]
    for k in budgets:
        result.append({'name': 'static_joint_kl', 'kind': 'static', 'k': k,
            'skips': static['static_joint_kl'][:k], 'seed': None})
        for m, run, arm in arms:
            for seed in [579, 580, 581]:
                checkpoints = json.loads((run / f'{arm}_seed{seed}_checkpoints.json').read_text())
                checkpoint = checkpoints[str(k)]
                if sha256(checkpoint['path']) != checkpoint['sha256']:
                    raise ValueError(f'Checkpoint digest mismatch: {checkpoint["path"]}')
                result.append({'name': f'm{m:02d}', 'kind': 'router', 'seed': seed,
                    'k': k, 'checkpoint': checkpoint, 'prefix': fixed_path[:m], 'm': m})
    return result


def run(args):
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 4:
        raise ValueError('Use four-GPU resource launcher')
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    stage6 = root / 'runs/stage6_depth_train_main_20261107_01'
    stage10 = root / 'runs/stage10_prefix_main_20261109_01'
    budgets = [10, 12]
    selected = policies(root, stage6, stage10, budgets)
    fixed = json.loads((stage6 / 'fixed_global_policy.json').read_text())['ordered_skips']
    config = {'output': str(output), 'model': '/home/liushenyu/opt-13b',
        'training_run': str(stage10), 'budgets': budgets, 'policies': selected,
        'fixed_prefix': fixed, 'probe': args.probe,
        'private_inputs': str(output / 'private_benchmark_inputs.jsonl')}
    tracked = {str(path): sha256(path) for path in root.rglob('*') if path.is_file()
        and not path.is_relative_to(output) and '.git' not in path.parts
        and path.suffix in {'.py', '.json', '.jsonl', '.md', '.csv', '.png'}}
    for path in root.parent.glob('*.py'):
        tracked[str(path)] = sha256(path)
    write_json(output / 'preservation_before.json', tracked)
    inputs = prepare(root, output, args.probe)
    task_files = ['piqa/piqa.yaml', 'hellaswag/hellaswag.yaml', 'hellaswag/utils.py',
        'winogrande/default.yaml', 'winogrande/preprocess_winogrande.py']
    write_json(output / 'protocol.json', {**config,
        'source_sha256': {path.name: sha256(path) for path in
            [Path(__file__), root / 'tools/horizontal_quality_v3.py', root / 'tools/horizontal_data_v1.py',
             root / 'tools/prompt_depth_core_v1.py', root / 'tools/prompt_fixed_core_v1.py']},
        'stage6_sha256': {name: sha256(stage6 / name) for name in ['static_paths.json', 'fixed_global_policy.json']},
        'stage10_sha256': {name: sha256(stage10 / name) for name in ['protocol.json', 'verification.json', 'summary.json']},
        'harness': {'root': str(HARNESS), 'commit': '95d580638385578c1c07fa554cf16ad7f5b5f460',
            'task_files': {name: sha256(HARNESS / 'lm_eval/tasks' / name) for name in task_files}},
        'data_manifest_sha256': sha256(output / 'data_manifest.json'),
        'route': 'One prompt-only route per policy and question; all answer options reuse it.',
        'scoring': 'Same batch-one FP16 SDPA genuine bypass scorer as stage7; full LM-head GEMM and FP64 reduction.',
        'selection': 'No downstream score selects a checkpoint or m. Existing stage6/stage10 checkpoints are frozen.',
        'comparison': 'm=4,6,8 routers plus m=0 pure dynamic, static joint-KL, and full model at K=10/12.',
        'limitations': ['This is a downstream development follow-up after stage7; it is not a new independent benchmark.',
            'm=4/8 use stage10 training endpoints; m=0/6 use audited stage6 endpoints.',
            'No restoration training or deployment timing is included.']})
    context = mp.get_context('spawn')
    messages = context.Queue()
    processes = [context.Process(target=worker, args=(rank, config, messages)) for rank in range(4)]
    done = []
    cpus = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
    try:
        for process in processes:
            process.start()
        pin_threads(cpus[40:42])
        while len(done) < 4:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker exited before downstream evaluation completed')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            if item['kind'] == 'progress':
                stamp('prefix_downstream_progress', **{k: v for k, v in item.items() if k != 'kind'})
            elif item['kind'] == 'done':
                done.append(item)
            else:
                raise ValueError('Unexpected worker message: ' + item['kind'])
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError('Downstream worker exit failure')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
            if process.is_alive():
                process.kill(); process.join(timeout=5)
    rows = [json.loads(line) for path in sorted(output.glob('quality_*.jsonl')) for line in path.read_text().splitlines()]
    if len(rows) != len(inputs) * len(selected):
        raise ValueError(f'Incomplete downstream output: {len(rows)}')
    write_json(output / 'summary.json', {'quality': summary_tables(rows), 'quality_rows': len(rows),
        'input_records': len(inputs), 'policies': len(selected), 'probe': args.probe,
        'elapsed_seconds': time.monotonic() - started})
    if any(sha256(path) != digest for path, digest in tracked.items()):
        raise ValueError('Existing evidence changed')
    write_json(output / 'preservation_after.json', {'preserved_files': len(tracked)})
    stamp('prefix_downstream_complete', rows=len(rows), elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    global started
    started = time.monotonic()
    run(args)


if __name__ == '__main__':
    main()
