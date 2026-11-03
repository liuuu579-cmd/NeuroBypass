"""Check greedy search objective, deterministic choices, and matched-mask controls."""
import unittest

import torch
import torch.nn.functional as F
from transformers import OPTConfig,OPTForCausalLM

from stage2_greedy_reference_v1 import (choose_best_candidate,shuffled_donors,last_only_logits,
                                       evaluate_candidate,model_logits,quality_metrics,compare_paired)


class GreedyReferenceTests(unittest.TestCase):
    def test_selection_and_permutation(self):
        self.assertEqual(choose_best_candidate([(4,.2),(3,.1),(2,.1)]),(2,.1))
        with self.assertRaises(ValueError):choose_best_candidate([(2,float('nan'))])
        donors=shuffled_donors(list(range(32)),579)
        self.assertEqual(set(donors),set(donors.values()))
        self.assertTrue(all(sample!=donor for sample,donor in donors.items()))
        self.assertEqual(donors,shuffled_donors(list(range(32)),579))

    def test_last_only_and_full_search_objectives(self):
        torch.set_num_threads(2);torch.manual_seed(579)
        model=OPTForCausalLM(OPTConfig(vocab_size=31,hidden_size=16,word_embed_proj_dim=16,
                ffn_dim=32,num_hidden_layers=4,num_attention_heads=2,max_position_embeddings=32,
                dropout=0.,attention_dropout=0.)).eval()
        ids=torch.tensor([[2,5,6,7,8]])
        with torch.inference_mode():
            full=model_logits(model,ids,[]);teacher=F.log_softmax(full.float(),dim=-1)
            for selected in [[],[2],[2,3]]:
                logits=model_logits(model,ids,selected)
                self.assertTrue(torch.allclose(last_only_logits(model,ids,selected)[:,0],logits[:,-1],atol=1e-6))
                metrics=quality_metrics(logits,teacher,ids)
                self.assertAlmostEqual(evaluate_candidate(model,ids,selected,teacher,'last_token'),metrics['last_token_kl'],places=6)
                self.assertAlmostEqual(evaluate_candidate(model,ids,selected,teacher,'all_tokens'),metrics['all_token_mean_kl'],places=6)
            # Brute-force all allowed single-layer choices agrees with the chosen step.
            costs=[(i,evaluate_candidate(model,ids,[i],teacher,'last_token')) for i in [2,3]]
            chosen,_=choose_best_candidate(costs)
            self.assertEqual(chosen,min([2,3],key=lambda i:quality_metrics(model_logits(model,ids,[i]),teacher,ids)['last_token_kl']))

    def test_pairing_keeps_sample_identity(self):
        rows=[{'sample':i,'policy':policy,'k':6,'metric':value}
              for i in [3,9] for policy,value in [('matched',1.),('static',2.)]]
        result=compare_paired(rows,'static',6,'metric','matched',579)
        self.assertEqual(result['mean_delta'],-1.)
        self.assertEqual(result['paired_bootstrap_95pct'],[-1.,-1.])
        self.assertEqual(result['fraction_strictly_better'],1.)


if __name__=='__main__':unittest.main()
