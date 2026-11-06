"""Small arithmetic checks for the new paper-baseline pilot."""
import math
import unittest

import torch

from depth_paper_pilot_v1 import block_influence, importance_order, tables


class PaperPilotArithmetic(unittest.TestCase):
    def test_bi_identity_is_zero_and_counts_tokens(self):
        before = torch.tensor([[[3., 4.], [0., 2.]]])
        total, count = block_influence(before, before * 5)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, 0., places=6)

    def test_bi_orthogonal_and_reversed_vectors(self):
        before = torch.tensor([[[1., 0.], [1., 0.]]])
        after = torch.tensor([[[0., 1.], [-1., 0.]]])
        total, count = block_influence(before, after)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total / count, 1.5)

    def test_shortened_plus_protection_and_deterministic_ties(self):
        costs = [[39, -.3], [0, -.1], [2, .1], [5, .2], [4, .2], [38, 0.]]
        self.assertEqual(importance_order(costs), [39, 38, 2, 4, 5])
        self.assertEqual(importance_order(costs, (0, 1, 2, 3, 38, 39)), [4, 5])

    def test_ppl_uses_mean_nll_not_mean_document_ppl(self):
        rows = [{'sample': 1, 'k': 6, 'policy': 'static', 'continuation_kl': .1,
                 'continuation_nll': 1., 'skipped_layers': [2, 3, 4, 5, 6, 7]},
                {'sample': 2, 'k': 6, 'policy': 'static', 'continuation_kl': .3,
                 'continuation_nll': 3., 'skipped_layers': [2, 3, 4, 5, 6, 7]}]
        row = tables(rows)['quality'][0]
        self.assertEqual(row['documents'], 2)
        self.assertEqual(row['unique_masks'], 1)
        self.assertAlmostEqual(row['continuation_kl'], .2)
        self.assertAlmostEqual(row['perplexity'], math.exp(2))


if __name__ == '__main__':
    unittest.main()
