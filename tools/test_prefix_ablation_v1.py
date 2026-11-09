"""CPU checks for document/state reuse, protected forks and exact-budget routes."""
import copy
import unittest

import torch

from prefix_ablation_core_v1 import NEW_PREFIX_COUNTS, fork_prefix, reuse_steps
from prompt_depth_core_v1 import DepthRouter, document_loss, validate_steps
from train_prompt_depth_four_gpu_v4 import padded_training


def synthetic(prefix=(), sample=0):
    selected = list(prefix)
    steps = []
    for number in range(len(selected) + 1, 25):
        costs = [[i, .001 * ((i * 13 + number) % 41)] for i in range(2, 40) if i not in selected]
        chosen, value = min(costs, key=lambda item: (item[1], item[0]))
        steps.append({'step': number, 'previous_skips': selected.copy(), 'candidate_costs': costs,
            'selected_layer': chosen, 'joint_continuation_kl': value, 'reused_old_entire_state': False})
        selected.append(chosen)
    return {'sample': sample, 'split': 'train', 'initial_prefix': list(prefix),
        'ordered_skips': selected, 'steps': steps}


class PrefixAblationTests(unittest.TestCase):
    def test_branch_keeps_every_fixed_layer_and_changes_expert_state(self):
        global_path = [4, 9, 2, 12, 17, 7, 24, 37]
        for m in NEW_PREFIX_COUNTS:
            fixed = global_path[:m]
            for sample in range(8):
                record = synthetic(fixed, sample)
                prefix, rank = fork_prefix(record, sample, 579, fixed)
                self.assertEqual(prefix[:m], fixed)
                self.assertNotEqual(set(prefix), set(record['ordered_skips'][:len(prefix)]))
                self.assertIn(rank, [2, 3, 4, 5])
                self.assertIn(len(prefix), [m + 1, m + 5, m + 9, m + 13])
                self.assertTrue(validate_steps(synthetic(prefix, sample), prefix, 24))

    def test_reuse_requires_same_document_and_entire_state(self):
        record = synthetic([4, 9], 12)
        duplicate = copy.deepcopy(record)
        duplicate['steps'][0]['previous_skips'].reverse()
        cache = reuse_steps([record, duplicate], 12)
        self.assertEqual(len(cache['steps']), len(record['steps']))
        with self.assertRaises(ValueError):
            reuse_steps([record], 13)
        duplicate['steps'][0]['candidate_costs'][0][1] += .1
        with self.assertRaises(ValueError):
            reuse_steps([record, duplicate], 12)
        distinct = synthetic([4, 2], 12)
        cache = reuse_steps([record, distinct], 12)
        states = {tuple(sorted(s['previous_skips'])) for s in cache['steps']}
        self.assertIn((2, 4), states)
        self.assertIn((4, 9), states)

    def test_variable_branch_padding_has_no_loss_or_gradient_effect(self):
        samples = [0, 1, 2, 3]
        records = {s: synthetic([4, 9, 2, 12], s) for s in samples}
        branches = {}
        for sample in samples:
            prefix, _ = fork_prefix(records[sample], sample, 579, [4, 9, 2, 12])
            branches[sample] = synthetic(prefix, sample)
        states, targets, valid, chosen, present = padded_training(records, branches, samples)
        torch.manual_seed(579)
        scores = torch.randn_like(states, requires_grad=True)
        loss = document_loss(scores, targets, valid, chosen, present)
        extra = ~present.bool()
        changed = scores.detach().clone()
        changed[extra] = 1000
        self.assertEqual(loss.item(), document_loss(changed, targets, valid, chosen, present).item())
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(scores.grad[extra].abs().sum().item(), 0)

    def test_router_exact_k_and_prefix_under_adversarial_head(self):
        router = DepthRouter(input_dim=8, width=8, dropout=0).eval()
        with torch.no_grad():
            router.head[-1].weight.zero_()
            router.head[-1].bias.copy_(torch.arange(40, dtype=torch.float))
        embeddings = torch.randn(2, 5, 8)
        for m in NEW_PREFIX_COUNTS:
            prefix = [4, 9, 2, 12, 17, 7, 24, 37][:m]
            for k in [10, 12]:
                for path in router.route(embeddings, k, prefix).tolist():
                    self.assertEqual(path[:m], prefix)
                    self.assertEqual(len(set(path)), k)
                    self.assertTrue(all(2 <= i < 40 for i in path))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
