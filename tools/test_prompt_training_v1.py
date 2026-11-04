"""Check joint-cost supervision, prompt-only routing and trainable parameters."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from prompt_fixed_core_v1 import SequentialGreedyRouter
from train_prompt_fixed_v1 import targets_from_record


class PromptTrainingTests(unittest.TestCase):
    def test_conditional_targets_change_after_first_skip(self):
        record={'steps':[{'previous_skips':[],'selected_layer':2,'candidate_costs':[[2,.01],[3,.02],[4,.03],[5,.04]]},
                         {'previous_skips':[2],'selected_layer':4,'candidate_costs':[[3,.8],[4,.04],[5,.2]]}]}
        states,targets,valid,chosen=targets_from_record(record,6)
        np.testing.assert_allclose(targets.sum(-1),1,atol=1e-6)
        self.assertEqual(chosen.tolist(),[2,4]);self.assertTrue(states[1,2]==1)
        self.assertFalse(valid[1,2]);self.assertEqual(targets[1].argmax(),4)

    def test_training_backward_and_exact_k(self):
        torch.set_num_threads(1);torch.manual_seed(579)
        router=SequentialGreedyRouter(input_dim=16,width=16,layers=8,max_k=3,dropout=0.)
        embeddings=torch.randn(2,7,16);states=torch.zeros(2,3,8)
        states[:,1,2]=1;states[:,2,[2,3]]=1
        valid=~states.bool();valid[:,:,:2]=False
        targets=valid.float()/valid.sum(-1,keepdim=True)
        steps=torch.arange(1,4).expand(2,-1)
        optimizer=torch.optim.AdamW(router.parameters(),lr=.001)
        previous=router.projection[0].weight.detach().clone()
        score=router(embeddings,states,steps)
        logp=F.log_softmax((-score).masked_fill(~valid,-1e9),dim=-1)
        loss=-(targets*logp).sum(-1).mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters()))
        optimizer.step();self.assertFalse(torch.equal(previous,router.projection[0].weight))
        router.eval()
        with torch.inference_mode():
            first=router.route(embeddings,3)
            # Inference has no continuation argument; its result is determined
            # by the same prompt embeddings and selected-set history.
            second=router.route(embeddings.clone(),3)
            self.assertTrue(torch.equal(first,second))


if __name__=='__main__':unittest.main()
