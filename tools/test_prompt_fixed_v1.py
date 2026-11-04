"""Meaningful preflight checks for cached fixed-route decoding and joint search."""
import unittest

import torch
from transformers import OPTConfig,OPTForCausalLM

from prompt_fixed_core_v1 import (FixedRouteSession,SequentialGreedyRouter,best_candidate,
    bind_cache,cached_continuation_logits,continuation_logits,fixed_route)


class PromptFixedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(579)
        self.model=OPTForCausalLM(OPTConfig(vocab_size=53,hidden_size=16,word_embed_proj_dim=16,
              ffn_dim=32,num_hidden_layers=6,num_attention_heads=2,dropout=0.,attention_dropout=0.,
              max_position_embeddings=40,_attn_implementation='eager')).eval()
        self.ids=torch.tensor([[2,5,8,11,13,17,21,22,27]])

    def test_cached_logits_and_layer_calls(self):
        with torch.inference_mode():
            for route in [[],[2],[2,4],[3,4,5]]:
                counts=[0]*6;hooks=[]
                for index,layer in enumerate(self.model.model.decoder.layers):
                    hooks.append(layer.self_attn.register_forward_pre_hook(lambda module,args,i=index:counts.__setitem__(i,counts[i]+1)))
                try:
                    expected=continuation_logits(self.model,self.ids,5,route)
                    actual,session=cached_continuation_logits(self.model,self.ids,5,route)
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                    for i in range(6):self.assertEqual(counts[i],0 if i in route else 5)
                    self.assertEqual(session.cache_bytes(),2*(6-len(route))*8*16*4)
                finally:
                    for hook in hooks:hook.remove()

    def test_route_change_and_exception_restore(self):
        session=FixedRouteSession(self.model,[2])
        with torch.inference_mode():session.forward(self.ids[:,:5])
        with self.assertRaises(ValueError):bind_cache(session.cache,[3])
        with self.assertRaises(ValueError):FixedRouteSession(self.model,[0])
        original=self.model.model.decoder.layers[2].forward
        with self.assertRaises(RuntimeError):
            with fixed_route(self.model,[2]):raise RuntimeError('test')
        self.assertEqual(self.model.model.decoder.layers[2].forward,original)

    def test_joint_greedy_reconsiders_candidates(self):
        # First-layer ranking would pick 2 then 3, but after skipping 2 the
        # jointly evaluated combination 2+4 is better than 2+3.
        costs={():[(2,.01),(3,.02),(4,.03)],(2,):[(3,.8),(4,.04)]}
        selected=[]
        for _ in range(2):selected.append(best_candidate(costs[tuple(selected)])[0])
        self.assertEqual(selected,[2,4])

    def test_router_is_exact_k_and_conditions_on_selected_set(self):
        router=SequentialGreedyRouter(input_dim=16,width=16,layers=6,max_k=3,dropout=0.).eval()
        embeddings=torch.randn(2,5,16)
        with torch.inference_mode():
            path=router.route(embeddings,3)
            self.assertEqual(path.shape,(2,3))
            for row in path.tolist():self.assertEqual(len(set(row)),3);self.assertTrue(all(i>=2 for i in row))
            context=router.encode(embeddings)
            first=torch.zeros(2,6);second=first.clone();second[:,2]=1
            a=router.score(context,first,torch.ones(2,dtype=torch.long))
            b=router.score(context,second,torch.ones(2,dtype=torch.long))
            self.assertFalse(torch.equal(a,b))


if __name__=='__main__':unittest.main()
