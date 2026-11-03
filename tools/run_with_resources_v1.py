"""Linux resource launcher: floor(machine CPUs * selected GPUs / machine GPUs).

CPU affinity and GPU visibility are inherited by subprocesses. A per-user local
registry avoids overlapping allocations between jobs using this launcher.
Existing jobs outside this launcher are detected on GPUs, not reserved on CPUs.
"""
import argparse
import contextlib
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path


def cpu_budget(total_cpus, selected_gpus, total_gpus):
    if not 1 <= selected_gpus <= total_gpus:
        raise ValueError('GPU count is outside the machine inventory')
    count = total_cpus * selected_gpus // total_gpus
    if count < 1:
        raise ValueError('Proportional CPU budget is below one core')
    return count


def process_start(pid):
    try:
        # The command name in /proc/PID/stat may contain spaces and parentheses.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def record_alive(record):
    if record.get('launcher_start') == process_start(record['launcher_pid']):
        return True
    pgid = record.get('child_pgid')
    if pgid:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            pass
        except PermissionError:
            return True
    return False


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        index, identity, name, memory, utilization = [s.strip() for s in line.split(',')]
        rows.append({'index': int(index), 'uuid': identity, 'name': name,
                     'memory_used_mib': int(memory), 'utilization_percent': int(utilization)})
    if not rows:
        raise RuntimeError('No NVIDIA GPUs found')
    return rows


def choose_resources(selected, inventory, allowed_cpus, total_cpus, records,
                     max_used_mib=1024, max_utilization=10):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Specify one or more distinct physical GPU indices')
    by_index = {g['index']: g for g in inventory}
    if any(i not in by_index for i in selected):
        raise ValueError('A selected GPU is absent from nvidia-smi')
    held_gpus = {i for r in records for i in r['gpu_indices']}
    if held_gpus.intersection(selected):
        raise RuntimeError(f'GPUs already reserved by this launcher: {sorted(held_gpus.intersection(selected))}')
    busy = [i for i in selected if by_index[i]['memory_used_mib'] > max_used_mib
            or by_index[i]['utilization_percent'] > max_utilization]
    if busy:
        raise RuntimeError(f'Selected GPUs appear busy: {busy}; select idle GPUs or inspect their tasks')
    count = cpu_budget(total_cpus, len(selected), len(inventory))
    held_cpus = {i for r in records for i in r['cpu_ids']}
    free = sorted(set(allowed_cpus) - held_cpus)
    if len(free) < count:
        raise RuntimeError(f'Need {count} CPUs, only {len(free)} available in inherited affinity and registry')
    return {'machine_cpu_count': total_cpus, 'machine_gpu_count': len(inventory),
            'gpu_indices': selected, 'gpu_uuids': [by_index[i]['uuid'] for i in selected],
            'cpu_count': count, 'cpu_ids': free[:count],
            'threads_per_worker': max(1, count // len(selected)),
            'gpu_snapshot': [by_index[i] for i in selected]}


class Registry:
    def __init__(self, directory):
        self.directory = directory

    @contextlib.contextmanager
    def locked(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or self.directory.stat().st_uid != os.getuid():
            raise RuntimeError('Resource registry must be an owned real directory')
        with (self.directory / 'lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / 'jobs.json'
            records = json.loads(path.read_text()) if path.exists() else []
            records = [r for r in records if record_alive(r)]
            yield records
            temporary = self.directory / ('jobs.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('x') as stream:
                json.dump(records, stream, indent=2)
            os.replace(temporary, path)


def worker_environment(plan, run_dir):
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': ','.join(plan['gpu_uuids']),
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTHONDONTWRITEBYTECODE': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
                'NEUROBYPASS_CPU_COUNT': str(plan['cpu_count']),
                'NEUROBYPASS_CPU_IDS': ','.join(map(str, plan['cpu_ids'])),
                'NEUROBYPASS_GPU_COUNT': str(len(plan['gpu_indices'])),
                'NEUROBYPASS_RUN_DIR': str(run_dir)})
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS']:
        env[name] = str(plan['threads_per_worker'])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True, help='Physical nvidia-smi indices, e.g. 1,3')
    parser.add_argument('--run-dir', type=Path, help='Must be a new directory')
    parser.add_argument('--dry-run', action='store_true', help='Show allocation without reserving or running')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        selected = [int(i) for i in args.gpus.split(',')]
    except ValueError:
        parser.error('--gpus must contain comma-separated integers')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.dry_run and (not command or args.run_dir is None):
        parser.error('--run-dir and a command after -- are required')
    registry = Registry(Path(tempfile.gettempdir()) / f'neurobypass-resources-{os.getuid()}')
    identity = uuid.uuid4().hex
    run_dir = args.run_dir.resolve() if args.run_dir else None
    plan = None
    child = None
    previous_handlers = {}
    status = {'state': 'starting'}
    try:
        with registry.locked() as records:
            plan = choose_resources(selected, gpu_inventory(), os.sched_getaffinity(0),
                                    os.cpu_count(), records)
            plan.update({'allocation_id': identity, 'command': command,
                         'working_directory': os.getcwd(),
                         'created_at': datetime.now().astimezone().isoformat(),
                         'registry_scope': 'same user, same host, jobs using this launcher'})
            if args.dry_run:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
                return
            # Exclusive directory creation prevents accidentally overwriting an old run.
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / 'allocation.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
            records.append({'allocation_id': identity, 'launcher_pid': os.getpid(),
                            'launcher_start': process_start(os.getpid()),
                            'gpu_indices': selected, 'cpu_ids': plan['cpu_ids'],
                            'run_dir': str(run_dir), 'child_pgid': None})
        os.sched_setaffinity(0, plan['cpu_ids'])
        print(json.dumps({'event': 'allocated', **plan}, ensure_ascii=False), flush=True)
        def forward_signal(signum, frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signum)
        for signum in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        with (run_dir / 'stdout.log').open('xb') as log:
            child = subprocess.Popen(command, env=worker_environment(plan, run_dir),
                                     start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            with registry.locked() as records:
                next(r for r in records if r['allocation_id'] == identity)['child_pgid'] = child.pid
            while True:
                chunk = child.stdout.read1(65536)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = child.wait()
        status = {'state': 'completed' if code == 0 else 'failed', 'exit_code': code,
                  'finished_at': datetime.now().astimezone().isoformat()}
        print(json.dumps(status), flush=True)
        return code if code >= 0 else 128 - code
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if not args.dry_run:
            with registry.locked() as records:
                records[:] = [r for r in records if r['allocation_id'] != identity]
            if run_dir and (run_dir / 'allocation.json').exists():
                (run_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
