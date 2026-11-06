"""Meaningful checks for branching, conditioned targets and paired statistics."""
import unittest

import numpy as np
import torch

from train_augmented_router_v2 import branch_prefix, build_tables, pack_documents


def record(sample=0):
    selected, steps = [], []
    for step in range(6):
        chosen = 2 + step
        costs = [[layer, float(layer - chosen + 1) / 100]
                 for layer in range(2, 40) if layer not in selected]
        steps.append({'step': step + 1, 'previous_skips': selected.copy(),
                      'selected_layer': chosen, 'candidate_costs': costs})
        selected.append(chosen)
    return {'sample': sample, 'split': 'train', 'ordered_skips': selected, 'steps': steps}


class AugmentationTests(unittest.TestCase):
    def test_same_set_in_different_order_forces_a_distinct_branch(self):
        item = record()
        prefix, metadata = branch_prefix(item, [4, 2, 3, 5, 6, 7], 579)
        self.assertNotEqual(set(prefix), set(item['ordered_skips'][:3]))
        self.assertEqual(prefix[:2], [2, 3])
        self.assertIn(metadata['alternative_rank'], [2, 3, 4, 5])
        self.assertEqual(len(prefix), len(set(prefix)))

    def test_already_distinct_own_prefix_is_used(self):
        prefix, metadata = branch_prefix(record(), [2, 6, 9, 4, 10, 12], 579)
        self.assertEqual(prefix, [2, 6, 9])
        self.assertEqual(metadata['kind'], 'own_prefix')

    def test_control_and_augmentation_match_state_slots_and_protect_layers(self):
        item = record()
        branch = {'steps': []}
        selected = [2, 3, 9]
        for number, choice in zip(range(4, 7), [4, 5, 6]):
            branch['steps'].append({'step': number, 'previous_skips': selected.copy(),
                'selected_layer': choice, 'candidate_costs': [[i, .01 * i] for i in range(2, 40) if i not in selected]})
            selected.append(choice)
        control = pack_documents([item], {0: branch}, 'baseline')
        treatment = pack_documents([item], {0: branch}, 'augmented')
        self.assertEqual(control[0].shape, (1, 9, 40))
        for left, right in zip(control, treatment):
            self.assertTrue(torch.equal(left[:, :6], right[:, :6]))
        self.assertTrue(torch.equal(control[0].sum(-1), treatment[0].sum(-1)))
        self.assertFalse(torch.equal(control[0][:, 6:], treatment[0][:, 6:]))
        for packed in [control, treatment]:
            state, target, available, chosen = packed
            np.testing.assert_allclose(target.sum(-1).numpy(), 1., atol=1e-6)
            self.assertFalse(available[:, :, :2].any())
            self.assertFalse((available & state.bool()).any())
            self.assertEqual(chosen.shape, (1, 9))

    def test_shuffle_average_remains_document_paired(self):
        rows = []
        for arm in ['baseline', 'augmented']:
            for sample in [4, 9]:
                for policy in ['full', 'static_old', 'static_learned', 'oracle_greedy', 'router'] + [f'shuffled_{i}' for i in range(4)]:
                    value = .2 if policy.startswith('shuffled') else .1
                    rows.append({'arm': arm, 'sample': sample, 'policy': policy,
                        'continuation_kl': value, 'continuation_nll': 2., 'skipped_layers': [2, 4, 6, 8, 10, 12]})
        result = build_tables(rows, [4, 9], [4], 579)
        comparison = next(r for r in result['paired_comparisons'] if r['population'] == 'validation_all100'
            and r['arm'] == 'augmented' and r['reference_policy'] == 'shuffled_mean4' and r['metric'] == 'continuation_kl')
        self.assertEqual(comparison['samples'], 2)
        self.assertAlmostEqual(comparison['mean_delta'], -.1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
