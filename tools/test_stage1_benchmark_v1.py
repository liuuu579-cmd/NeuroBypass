"""Verify sequence metrics and grouping on a small random OPT before real runs."""
import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers import OPTConfig, OPTForCausalLM

from stage1_benchmark_v1 import model_logits, quality_metrics, summarize


class BenchmarkTests(unittest.TestCase):
    def test_metrics_and_true_bypass(self):
        torch.set_num_threads(2)
        torch.manual_seed(579)
        model=OPTForCausalLM(OPTConfig(vocab_size=31,hidden_size=16,word_embed_proj_dim=16,
                ffn_dim=32,num_hidden_layers=4,num_attention_heads=2,max_position_embeddings=32,
                dropout=0.,attention_dropout=0.)).eval()
        ids=torch.tensor([[2,5,6,7,8]])
        calls=[]
        handle=model.model.decoder.layers[2].fc1.register_forward_hook(lambda *a:calls.append(1))
        with torch.inference_mode():
            teacher=model_logits(model,ids,[])
            self.assertEqual(len(calls),1)
            teacher_lp=F.log_softmax(teacher.float(),dim=-1)
            full=quality_metrics(teacher,teacher_lp,ids)
            self.assertAlmostEqual(full['all_token_mean_kl'],0.,places=6)
            self.assertAlmostEqual(full['last_token_kl'],0.,places=6)
            direct_loss=model(ids,labels=ids,use_cache=False).loss.item()
            self.assertAlmostEqual(full['next_token_nll'],direct_loss,places=6)
            calls.clear()
            skipped=model_logits(model,ids,[2])
            self.assertEqual(calls,[])
            metrics=quality_metrics(skipped,teacher_lp,ids)
            self.assertGreaterEqual(metrics['all_token_mean_kl'],-1e-6)
            model_logits(model,ids,[])
            self.assertEqual(len(calls),1)
            self.assertEqual(full['next_token_count'],4)
        handle.remove()

    def test_summary_uses_each_sample_once(self):
        rows=[]
        for i,nll in enumerate([1.,3.]):
            rows.append({'sample':i,'policy':'fixed','k':1,'skipped_layers':[2],
                         'last_token_kl':nll,'all_token_mean_kl':nll,
                         'prediction_position_mean_kl':nll,'next_token_nll':nll})
        result=summarize(rows)[0]
        self.assertEqual(result['samples'],2)
        self.assertEqual(result['next_token_nll']['mean'],2.)
        self.assertAlmostEqual(result['perplexity'],math.exp(2.))
        self.assertEqual(result['unique_masks'],1)
        self.assertEqual(result['layer_skip_frequency']['2'],1.)
        self.assertEqual(result['layer_skip_frequency']['1'],0.)
        self.assertTrue(set(range(0,256,2)).isdisjoint(range(1,256,2)))
        self.assertEqual(len(set(range(0,256,2))|set(range(1,256,2))),256)


if __name__=='__main__':
    unittest.main()
