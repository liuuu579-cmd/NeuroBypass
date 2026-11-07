"""CPU checks of protected/fixed routes, deep forks and padded document loss."""
import unittest

import torch

from prompt_depth_core_v1 import DepthRouter, document_loss, fork_prefix, validate_steps


def synthetic_record(prefix=(), maximum=24):
    selected, rows = list(prefix), []
    for step in range(len(selected) + 1, maximum + 1):
        costs = [[i, float(i)] for i in range(2, 40) if i not in selected]
        layer = costs[0][0]
        rows.append({'step': step, 'previous_skips': selected.copy(),
            'candidate_costs': costs, 'selected_layer': layer})
        selected.append(layer)
    return {'steps': rows, 'ordered_skips': selected}


class Checks(unittest.TestCase):
    def test_budget_and_fixed_prefix(self):
        torch.manual_seed(1)
        router = DepthRouter(input_dim=8, width=8, dropout=0).eval()
        path = router.route(torch.randn(2, 3, 8), 24, [4, 8, 2, 10, 16, 14])
        self.assertEqual(path[:, :6].tolist(), [[4, 8, 2, 10, 16, 14]] * 2)
        for row in path.tolist():
            self.assertEqual(len(set(row)), 24)
            self.assertTrue(all(2 <= i < 40 for i in row))
        with self.assertRaises(ValueError):
            router.route(torch.randn(1, 3, 8), 25)

    def test_global_prefix_cannot_change(self):
        fixed = [4, 8, 2, 10, 16, 14]
        record = synthetic_record(fixed)
        for sample in range(8):
            prefix, rank = fork_prefix(record, 'hybrid', sample, 579, fixed)
            self.assertEqual(prefix[:6], fixed)
            self.assertNotEqual(set(prefix), set(record['ordered_skips'][:len(prefix)]))
            self.assertIn(rank, [2, 3, 4, 5])

    def test_entire_state_and_candidate_set(self):
        record = synthetic_record()
        self.assertTrue(validate_steps(record))
        record['steps'][12]['candidate_costs'].pop()
        with self.assertRaises(ValueError):
            validate_steps(record)

    def test_padding_does_not_reweight_documents(self):
        scores = torch.randn(2, 3, 40, requires_grad=True)
        target = torch.softmax(torch.randn(2, 3, 40), -1)
        valid = torch.ones(2, 3, 40, dtype=torch.bool)
        chosen = torch.zeros(2, 3, dtype=torch.long)
        present = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
        loss = document_loss(scores, target, valid, chosen, present)
        parts = [document_loss(scores[:1, :1], target[:1, :1], valid[:1, :1], chosen[:1, :1], torch.ones(1, 1)),
            document_loss(scores[1:], target[1:], valid[1:], chosen[1:], torch.ones(1, 3))]
        self.assertTrue(torch.allclose(loss, sum(parts) / 2))
        loss.backward()
        self.assertEqual(scores.grad[0, 1:].abs().sum().item(), 0.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
