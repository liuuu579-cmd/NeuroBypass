"""Append-only experiment I/O, resource checks, and finite four-worker driver."""
import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path

import torch

from horizontal_quality_v3 import pin_threads
from stage1_benchmark_v2 import sha256, stamp


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_lines(path):
    return [json.loads(line) for line in Path(path).open()]


def preserve(root, output):
    tracked = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()
        and not p.is_relative_to(output) and '.git' not in p.parts
        and p.suffix in {'.py', '.md', '.json', '.jsonl', '.csv', '.png'}}
    for p in root.parent.glob('*.py'):
        tracked[str(p)] = sha256(p)
    write(output / 'preservation_before.json', tracked)
    return tracked


def verify_preserved(output, tracked):
    for path, digest in tracked.items():
        if sha256(path) != digest:
            raise ValueError('Existing file changed: ' + path)
    write(output / 'preservation_after.json', {'preserved_files': len(tracked)})


def resources():
    if os.environ.get('NEUROBYPASS_GPU_COUNT') != '4' or os.environ.get('NEUROBYPASS_CPU_COUNT') != '42':
        raise ValueError('Use run_with_resources_v2.py with four GPUs /42 CPU cores')
    return list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))


def setup_worker(rank):
    allowed = resources()[rank*10:(rank+1)*10]
    pin_threads(allowed)
    torch.set_num_threads(10); torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    return allowed


def thread_record():
    return {'cpu_ids': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads(),
        'thread_cpu_ids': {p.name: sorted(os.sched_getaffinity(int(p.name)))
            for p in Path('/proc/self/task').iterdir()}}


def guarded_worker(target, rank, config, inbox, messages):
    try:
        target(rank, config, inbox, messages)
    except BaseException:
        messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


class Workers:
    def __init__(self, target, config):
        context = mp.get_context('spawn')
        self.messages = context.Queue()
        self.inboxes = [context.Queue() for _ in range(4)]
        self.processes = [context.Process(target=guarded_worker,
            args=(target, rank, config, self.inboxes[rank], self.messages)) for rank in range(4)]

    def __enter__(self):
        for process in self.processes:
            process.start()
        pin_threads(resources()[40:42])
        return self

    def receive(self, count, expected):
        rows = []
        while len(rows) < count:
            try:
                row = self.messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in self.processes) or all(p.exitcode is not None for p in self.processes):
                    raise RuntimeError('Workers exited before completion')
                continue
            if row['kind'] == 'error':
                raise RuntimeError(row['traceback'])
            if row['kind'] == 'progress':
                stamp('full_context_progress', **{k: v for k, v in row.items() if k != 'kind'})
                continue
            if row['kind'] != expected:
                raise ValueError('Unexpected worker phase: ' + row['kind'])
            rows.append(row)
        if len({r['rank'] for r in rows}) != count:
            raise ValueError('Duplicate worker completion')
        return rows

    def send_all(self, task):
        for inbox in self.inboxes:
            inbox.put(task)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            for inbox in self.inboxes:
                inbox.put(None)
            for process in self.processes:
                process.join(timeout=30)
        for process in self.processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
            if process.is_alive():
                process.kill(); process.join(timeout=5)
        if exc_type is None and any(p.exitcode != 0 for p in self.processes):
            raise RuntimeError('Unclean worker shutdown')
