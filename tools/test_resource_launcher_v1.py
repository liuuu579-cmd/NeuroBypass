"""Focused tests for proportional budgets, conflict checks and inherited limits."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run_with_resources_v1 import choose_resources, cpu_budget, worker_environment


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.inventory = [{'index': i, 'uuid': f'GPU-test-{i}', 'name': 'test',
                           'memory_used_mib': 0, 'utilization_percent': 0} for i in range(6)]

    def test_proportions(self):
        self.assertEqual([cpu_budget(64, n, 6) for n in range(1, 7)], [10, 21, 32, 42, 53, 64])

    def test_simultaneous_jobs_have_disjoint_cpus(self):
        records = []
        for selected in [[0, 1], [2, 3], [4, 5]]:
            plan = choose_resources(selected, self.inventory, range(64), 64, records)
            self.assertEqual(plan['cpu_count'], 21)
            for prior in records:
                self.assertTrue(set(plan['cpu_ids']).isdisjoint(prior['cpu_ids']))
            records.append(plan)

    def test_reject_duplicate_reserved_busy_and_insufficient(self):
        with self.assertRaises(ValueError):
            choose_resources([1, 1], self.inventory, range(64), 64, [])
        held = [{'gpu_indices': [1], 'cpu_ids': list(range(10))}]
        with self.assertRaises(RuntimeError):
            choose_resources([1], self.inventory, range(64), 64, held)
        self.inventory[3]['memory_used_mib'] = 12000
        with self.assertRaises(RuntimeError):
            choose_resources([3], self.inventory, range(64), 64, [])
        with self.assertRaises(RuntimeError):
            choose_resources([0, 1], self.inventory, range(10), 64, [])

    def test_affinity_and_environment_inherited_by_grandchild(self):
        cpus = sorted(os.sched_getaffinity(0))[:2]
        plan = {'gpu_uuids': ['GPU-test-1', 'GPU-test-3'], 'gpu_indices': [1, 3],
                'cpu_count': len(cpus), 'cpu_ids': cpus, 'threads_per_worker': 1}
        probe = "import os,json;print(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'gpus':os.environ['CUDA_VISIBLE_DEVICES'],'threads':os.environ['OMP_NUM_THREADS']}))"
        outer = "import os,subprocess,sys;os.sched_setaffinity(0," + repr(cpus) + ");subprocess.run([sys.executable,'-c'," + repr(probe) + "],check=True)"
        result = subprocess.run([sys.executable, '-c', outer], check=True, capture_output=True,
                                text=True, env=worker_environment(plan, Path('/tmp/probe')))
        data = json.loads(result.stdout)
        self.assertEqual(data, {'cpus': cpus, 'gpus': 'GPU-test-1,GPU-test-3', 'threads': '1'})


if __name__ == '__main__':
    unittest.main()
