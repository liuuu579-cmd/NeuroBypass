"""Check candidate availability, KL regret arithmetic and matched controls."""
import unittest

import numpy as np

from diagnose_prompt_router_v2 import choice_statistics, derangement, paired_summary


class DiagnosisTests(unittest.TestCase):
    def test_only_available_layers_and_real_cost_regret(self):
        scores = np.full(40, 9.)
        scores[0] = -100.  # A protected layer must never become a choice.
        scores[7] = -3.
        result = choice_statistics(scores, [[3, .01], [7, .04], [8, .02]])
        self.assertEqual(result['choice'], 7)
        self.assertEqual(result['candidate_rank'], 3)
        self.assertAlmostEqual(result['regret'], .03)
        self.assertEqual(result['top1'], 0)

    def test_cost_ties_are_resolved_by_layer_index(self):
        scores = np.zeros(40)
        result = choice_statistics(scores, [[7, .02], [3, .02]])
        self.assertEqual(result['choice'], 3)
        self.assertEqual(result['regret'], 0)
        self.assertEqual(result['top1'], 1)

    def test_derangements_keep_the_same_control_population(self):
        samples = [4, 12, 38, 65, 89]
        donors = derangement(samples, 579)
        self.assertEqual(set(donors), set(samples))
        self.assertEqual(set(donors.values()), set(samples))
        self.assertTrue(all(target != source for target, source in donors.items()))
        self.assertEqual(donors, derangement(samples, 579))

    def test_bootstrap_preserves_paired_difference_direction(self):
        result = paired_summary([-.1, -.1, -.1])
        np.testing.assert_allclose(result['bootstrap_95pct'], [-.1, -.1])
        self.assertEqual(result['fraction_negative'], 1.)


if __name__ == '__main__':
    unittest.main()
