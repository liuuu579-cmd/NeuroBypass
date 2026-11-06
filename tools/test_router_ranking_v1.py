"""Validate KL-gap weighting, score direction, masks and document pairing."""
import math
import unittest

import numpy as np
import torch

from train_router_ranking_v1 import gap_pairwise_loss, summarize


class RankingTests(unittest.TestCase):
    def test_matches_independent_pair_cost_arithmetic(self):
        scores = torch.tensor([.3, -.2, .8], dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([.01, .04, .10], dtype=torch.float64)
        valid = torch.ones(3, dtype=torch.bool)
        expected = (.03 * np.logaddexp(0, .5) + .09 * np.logaddexp(0, -.5)
                    + .06 * np.logaddexp(0, -1.)) / .18
        self.assertAlmostEqual(gap_pairwise_loss(scores, costs, valid).item(), expected)

    def test_gradient_prefers_skipping_lower_kl_candidate(self):
        scores = torch.tensor([0., 0.], requires_grad=True)
        costs = torch.tensor([.01, .09])
        loss = gap_pairwise_loss(scores, costs, torch.ones(2, dtype=torch.bool))
        loss.backward()
        self.assertGreater(scores.grad[0].item(), 0.)
        self.assertLess(scores.grad[1].item(), 0.)
        updated = scores.detach() - .1 * scores.grad
        self.assertEqual(updated.argmin().item(), 0)
        self.assertLess(gap_pairwise_loss(updated, costs, torch.ones(2, dtype=torch.bool)).item(), loss.item())

    def test_protected_candidates_have_no_loss_or_gradient(self):
        scores = torch.tensor([200., -200., .1, -.2], requires_grad=True)
        costs = torch.tensor([999., -999., .01, .02])
        valid = torch.tensor([False, False, True, True])
        loss = gap_pairwise_loss(scores, costs, valid)
        self.assertAlmostEqual(loss.item(), torch.nn.functional.softplus(torch.tensor(.3)).item(), places=6)
        loss.backward()
        self.assertEqual(scores.grad[0].item(), 0.)
        self.assertEqual(scores.grad[1].item(), 0.)

    def test_equal_costs_have_no_preference(self):
        scores = torch.tensor([.1, .2, .3], requires_grad=True)
        loss = gap_pairwise_loss(scores, torch.ones(3), torch.ones(3, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_cost_offset_and_positive_scale_do_not_change_state_loss(self):
        scores = torch.tensor([.3, -.2, .8], dtype=torch.float64)
        costs = torch.tensor([.01, .04, .10], dtype=torch.float64)
        valid = torch.ones(3, dtype=torch.bool)
        self.assertAlmostEqual(gap_pairwise_loss(scores, costs, valid).item(),
                               gap_pairwise_loss(scores, costs * 20 + 4., valid).item())

    def test_seed_and_donor_repeats_do_not_multiply_document_count(self):
        rows = []
        for seed in [579, 580, 581]:
            for arm in ['classification', 'pairwise']:
                for sample in [4, 9]:
                    for policy in ['full', 'static_old', 'static_learned', 'oracle_greedy', 'router'] + [f'shuffled_{i}' for i in range(4)]:
                        value = .2 if policy.startswith('shuffled') else .1
                        rows.append({'seed': seed, 'arm': arm, 'sample': sample, 'policy': policy,
                            'continuation_kl': value, 'continuation_nll': 2., 'skipped_layers': [2, 4, 6, 8, 10, 12]})
        result = summarize(rows, [4, 9], [4], [579, 580, 581])
        row = next(r for r in result['paired_comparisons'] if r['population'] == 'validation_all100'
            and r['seed_group'] == 'mean_seeds' and r['arm'] == 'pairwise'
            and r['reference_policy'] == 'shuffled_mean4' and r['metric'] == 'continuation_kl')
        self.assertEqual(row['samples'], 2)
        self.assertAlmostEqual(row['mean_delta'], -.1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
