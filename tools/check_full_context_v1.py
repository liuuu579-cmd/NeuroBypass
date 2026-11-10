"""CPU regression checks: observation suffixes, padding and target boundaries."""
import json
from types import SimpleNamespace

import torch

from full_context_core_v1 import FullContextRouter, prompt_embeddings, prompt_only, visible_context
from prompt_depth_core_v1 import DepthRouter


def checks():
    torch.manual_seed(579)
    ids = torch.arange(300)[None]
    seen, removed = visible_context(ids, context_limit=256, reserve_tokens=32)
    assert removed == 76 and torch.equal(seen, ids[:, -224:])
    assert seen[0, -1] == 299
    sequence = torch.arange(545)[None]
    observation = prompt_only(sequence, 513)
    changed = sequence.clone(); changed[:, 513:] = -99
    assert torch.equal(observation, prompt_only(changed, 513))
    assert observation.shape[1] == 513
    router = FullContextRouter(input_dim=12, width=16, dropout=0).eval()
    legacy = DepthRouter(input_dim=12, width=16, dropout=0).eval()
    legacy.load_state_dict(router.state_dict())
    x = torch.randn(1, 257, 12)
    passed = []
    def capture(module, args):
        passed.append(args[0].detach().clone())
    handle = router.projection.register_forward_pre_hook(capture)
    a = router.encode(x)
    suffix = x.clone(); suffix[:, 128:] += 3
    b = router.encode(suffix)
    handle.remove()
    assert passed[0].shape[1] == 257 and torch.equal(passed[1][:, 128:], suffix[:, 128:])
    assert not torch.allclose(a, b)
    # No requirement that every changed suffix changes the discrete argmin mask.
    short = x[:, :73]
    padded = torch.cat([short, torch.randn(1, 184, 12)*100], 1)
    valid = torch.arange(257)[None] < 73
    error = (router.encode(short)-router.encode(padded, valid)).abs().max().item()
    assert error < 2e-5, error
    assert torch.equal(router.route(short, 12), router.route(padded, 12, valid=valid))
    compatible = (router.encode(x[:, :128])-legacy.encode(x[:, :128])).abs().max().item()
    assert compatible == 0
    class Positions(torch.nn.Module):
        def forward(self, mask, past_key_values_length=0):
            return ((mask.cumsum(-1)-1).clamp_min(0)*mask)[..., None].expand(-1, -1, 12).float()
    decoder = SimpleNamespace(embed_tokens=torch.nn.Embedding(1000, 12), project_in=None, embed_positions=Positions())
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048), model=SimpleNamespace(decoder=decoder))
    full = prompt_embeddings(model, sequence[:, :513])
    assert full.shape == (1, 513, 12)
    assert torch.equal(full[:, -1], decoder.embed_tokens(sequence[:, 512])+512)
    for bad in [torch.ones(1, 2049, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)]:
        try: prompt_embeddings(model, bad)
        except ValueError: pass
        else: raise AssertionError('Invalid observation accepted')
    route = router.route(x, 12, [4, 9, 2, 12, 17, 7])[0].tolist()
    assert route[:6] == [4, 9, 2, 12, 17, 7] and len(set(route)) == 12 and min(route) >= 2
    return {'passed': True, 'observed_tokens': 513, 'suffix_after128_reaches_encoder': True,
        'target_changes_leave_observation_unchanged': True, 'padding_max_abs_error': error,
        'legacy_unpadded_encoder_max_abs_error': compatible, 'left_truncation_retains_latest_token': True,
        'context_overflow_rejected': True, 'fixed_prefix_and_protected_layers_checked': True}


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(json.dumps(checks(), indent=2))
