"""Test resource isolation and preservation/error behavior of the v2 launcher."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_with_resources_v2 as launcher


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.inventory = [{'index': i, 'uuid': f'GPU-test-{i}', 'name': 'test',
                           'memory_used_mib': 0, 'utilization_percent': 0} for i in range(6)]

    def test_proportions_and_concurrent_plans(self):
        self.assertEqual([launcher.cpu_budget(64, n, 6) for n in range(1, 7)], [10, 21, 32, 42, 53, 64])
        records = []
        for selected in [[0, 1], [2, 3], [4, 5]]:
            plan = launcher.choose_resources(selected, self.inventory, range(64), 64, records)
            self.assertEqual(plan['cpu_count'], 21)
            for prior in records:
                self.assertTrue(set(plan['cpu_ids']).isdisjoint(prior['cpu_ids']))
            records.append(plan)

    def test_reject_duplicate_reserved_busy_and_insufficient(self):
        with self.assertRaises(ValueError):
            launcher.choose_resources([1, 1], self.inventory, range(64), 64, [])
        with self.assertRaises(RuntimeError):
            launcher.choose_resources([1], self.inventory, range(64), 64,
                                      [{'gpu_indices': [1], 'cpu_ids': list(range(10))}])
        self.inventory[3]['memory_used_mib'] = 12000
        with self.assertRaises(RuntimeError):
            launcher.choose_resources([3], self.inventory, range(64), 64, [])
        with self.assertRaises(RuntimeError):
            launcher.choose_resources([0, 1], self.inventory, range(10), 64, [])

    def test_affinity_and_environment_inherited_by_grandchild(self):
        cpus = sorted(os.sched_getaffinity(0))[:2]
        plan = {'gpu_uuids': ['GPU-test-1', 'GPU-test-3'], 'gpu_indices': [1, 3],
                'cpu_count': len(cpus), 'cpu_ids': cpus, 'threads_per_worker': 1}
        probe = "import os,json;print(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'gpus':os.environ['CUDA_VISIBLE_DEVICES'],'threads':os.environ['OMP_NUM_THREADS']}))"
        outer = "import os,subprocess,sys;os.sched_setaffinity(0," + repr(cpus) + ");subprocess.run([sys.executable,'-c'," + repr(probe) + "],check=True)"
        result = subprocess.run([sys.executable, '-c', outer], check=True, capture_output=True,
                                text=True, env=launcher.worker_environment(plan, Path('/tmp/probe')))
        self.assertEqual(json.loads(result.stdout),
                         {'cpus': cpus, 'gpus': 'GPU-test-1,GPU-test-3', 'threads': '1'})

    def test_existing_run_not_modified_and_missing_command_releases_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing'
            existing.mkdir()
            (existing / 'allocation.json').write_text('{"original":true}')
            (existing / 'status.json').write_text('{"state":"original"}')
            before = {p.name: p.read_bytes() for p in existing.iterdir()}
            registry = launcher.Registry(root / 'registry')
            with patch.object(launcher, 'Registry', return_value=registry), \
                 patch.object(launcher, 'gpu_inventory', return_value=self.inventory), \
                 patch.object(launcher.os, 'cpu_count', return_value=64), \
                 patch.object(launcher.os, 'sched_getaffinity', return_value=set(range(64))), \
                 patch.object(launcher.os, 'sched_setaffinity'), \
                 patch.object(sys, 'argv', ['launcher', '--gpus', '1,3', '--run-dir', str(existing), '--', 'anything']):
                with self.assertRaises(FileExistsError):
                    launcher.main()
            self.assertEqual(before, {p.name: p.read_bytes() for p in existing.iterdir()})
            missing = root / 'missing_command_run'
            with patch.object(launcher, 'Registry', return_value=registry), \
                 patch.object(launcher, 'gpu_inventory', return_value=self.inventory), \
                 patch.object(launcher.os, 'cpu_count', return_value=64), \
                 patch.object(launcher.os, 'sched_getaffinity', return_value=set(range(64))), \
                 patch.object(launcher.os, 'sched_setaffinity'), \
                 patch.object(sys, 'argv', ['launcher', '--gpus', '1,3', '--run-dir', str(missing), '--', '/definitely/missing/program']), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    launcher.main()
            self.assertEqual(json.loads((registry.directory / 'jobs.json').read_text()), [])
            self.assertEqual(json.loads((missing / 'status.json').read_text())['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
