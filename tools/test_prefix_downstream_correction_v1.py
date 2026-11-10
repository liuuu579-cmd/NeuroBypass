"""Regression checks for the actual truncated-prefix and PPL aggregation bugs."""
import copy
import json
import math
import unittest
from pathlib import Path

from prefix_downstream_correction_core_v1 import (aggregate_seed_metric, partition,
    policies, validate_policy)


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.path, cls.corrected = policies(cls.root)
        cls.original = json.loads((cls.root / 'runs/stage11_prefix_downstream_main_20261109_01/protocol.json').read_text())['policies']

    def test_actual_legacy_m8_is_rejected(self):
        affected = [p for p in self.original if p['name'] == 'm08']
        self.assertEqual(len(affected), 6)
        for policy in affected:
            with self.assertRaisesRegex(ValueError, 'Fixed-prefix mismatch'):
                validate_policy(policy, self.path)

    def test_training_prefixes_and_reuse_contract(self):
        reused, rerun = partition(self.original, self.corrected)
        self.assertEqual((len(reused), len(rerun)), (21, 6))
        for policy in self.corrected:
            validate_policy(policy, self.path)
        self.assertTrue(all(p['prefix'] == [4, 9, 2, 12, 17, 7, 24, 37] for p in rerun))

    def test_equal_length_wrong_prefix_is_rejected(self):
        policy = copy.deepcopy(next(p for p in self.corrected if p['name'] == 'm08'))
        policy['prefix'][-1] = 39
        with self.assertRaises(ValueError):
            validate_policy(policy, self.path)

    def test_changed_reused_checkpoint_is_rejected(self):
        modified = copy.deepcopy(self.corrected)
        next(p for p in modified if p['name'] == 'm04')['checkpoint']['epoch'] += 1
        with self.assertRaisesRegex(ValueError, 'reuse a changed policy'):
            partition(self.original, modified)

    def test_ppl_is_exp_of_mean_nll(self):
        values = [math.log(10), math.log(40)]
        self.assertAlmostEqual(aggregate_seed_metric(values, wiki=True), 20)
        self.assertNotAlmostEqual(aggregate_seed_metric(values, wiki=True), 25)


if __name__ == '__main__':
    unittest.main()
