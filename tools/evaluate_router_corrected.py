"""Evaluate the existing raw-embedding router with correct keep/skip semantics.

Input: local JSONL with a `prompt` string per row. No original file is written.
Uses exact-K selection, float32 KL, no KV cache, and real decoder bypass.
This is a last-token KL diagnostic; it does not measure generation speed or PIQA.
"""
import argparse
import contextlib
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from check_checkpoint_direction import choose_skips, load_class


@contextlib.contextmanager
def bypass_layers(model, indices):
    """Temporarily replace selected forwards, restoring them even after failure."""
    layers = model.model.decoder.layers
    saved = []
    def identity_forward(self, hidden_states, *args, **kwargs):
        if kwargs.get('use_cache', False) or kwargs.get('output_attentions', False):
            raise ValueError('Diagnostic bypass requires use_cache=False and no attention outputs')
        return (hidden_states,)
    try:
        for idx in sorted(set(indices)):
            layer = layers[idx]
            saved.append((layer, layer.__dict__.get('forward'), 'forward' in layer.__dict__))
            layer.forward = MethodType(identity_forward, layer)
        yield
    finally:
        for layer, original, existed in saved:
            if existed:
                layer.forward = original
            else:
                del layer.forward


def last_logits(model, ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    output = model(ids, attention_mask=attention_mask, use_cache=False)
    # Works for right padding as well as unpadded inputs.
    positions = torch.arange(ids.size(1), device=ids.device).expand_as(ids)
    last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError('An input has no valid tokens')
    return output.logits[torch.arange(ids.size(0), device=ids.device), last].float()


def kl_from_logits(pruned, teacher):
    return F.kl_div(F.log_softmax(pruned.float(), dim=-1),
                    F.softmax(teacher.float(), dim=-1), reduction='none').sum(dim=-1)


def raw_embeddings(model, ids, attention_mask):
    decoder = model.model.decoder
    x = decoder.embed_tokens(ids)
    if decoder.project_in is not None:
        x = decoder.project_in(x)
    return x + decoder.embed_positions(attention_mask, past_key_values_length=0)


def self_test():
    from transformers import OPTConfig, OPTForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = OPTForCausalLM(OPTConfig(vocab_size=31, hidden_size=16, word_embed_proj_dim=16,
                                    ffn_dim=32, num_hidden_layers=4, num_attention_heads=2,
                                    dropout=0., attention_dropout=0., max_position_embeddings=32,
                                    _attn_implementation='eager')).eval()
    ids = torch.tensor([[2, 5, 6, 7, 8]])
    layers, count = model.model.decoder.layers, [0]
    def counting_hook(module, args, output):
        count[0] += 1
    def identity_hook(module, args, output):
        return (args[0],) + output[1:]
    handle_count = layers[2].fc1.register_forward_hook(counting_hook)
    with torch.inference_mode():
        handle_post = layers[2].register_forward_hook(identity_hook)
        hooked = last_logits(model, ids)
        handle_post.remove()
        assert count[0] == 1, 'Post-hook should still execute the layer'
        count[0] = 0
        with bypass_layers(model, [2]):
            bypassed = last_logits(model, ids)
        assert count[0] == 0, 'True bypass must avoid computing the layer'
        assert torch.allclose(hooked, bypassed, atol=1e-6)
        last_logits(model, ids)
        assert count[0] == 1, 'Layer forward must be restored'
        padded = torch.tensor([[2, 5, 6, 1, 1]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        assert torch.allclose(last_logits(model, padded, mask), last_logits(model, padded[:, :3]), atol=1e-6)
    handle_count.remove()
    logits = torch.tensor([100., 100., -8., 8., -7., 7.])
    assert set(choose_skips(logits, 2).tolist()) == {2, 4}
    assert torch.allclose(kl_from_logits(logits, logits), torch.tensor(0.), atol=1e-6)
    print('PASS: skip direction, exact K, protected layers, last valid token, KL identity, bypass equivalence, avoided compute, restoration')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--snapshot', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--base-model', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--prompts', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--k', type=int, nargs='+', default=[6, 16])
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if any(v is None for v in [args.base_model, args.checkpoint, args.prompts, args.output]):
        p.error('--base-model, --checkpoint, --prompts and --output are required')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(k < 1 or k > 16 for k in args.k):
        p.error('Each k must be in 1..16')
    records = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if not records or any(not isinstance(r.get('prompt'), str) or not r['prompt'].strip() for r in records):
        p.error('Expected nonempty JSONL prompt strings')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, local_files_only=True)
    dtype = torch.float16 if str(args.device).startswith('cuda') else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype,
                                               local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    router = load_class(args.snapshot / 'legacy/skip/model_router_transformer.py')(dropout=0.).to(args.device).eval()
    router.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=args.device))
    rows = []
    with torch.inference_mode():
        for i, record in enumerate(records):
            ids = tokenizer(record['prompt'], return_tensors='pt').input_ids.to(args.device)
            if ids.shape[1] > model.config.max_position_embeddings:
                raise ValueError('Prompt exceeds model context length')
            attention_mask = torch.ones_like(ids)
            teacher = last_logits(model, ids, attention_mask)
            embeddings = raw_embeddings(model, ids, attention_mask)
            for k in args.k:
                logits = router(embeddings, torch.tensor([k], device=args.device))[0]
                corrected = choose_skips(logits, k).tolist()
                reversed_set = choose_skips(logits, k, largest=True).tolist()
                for policy, indices in [('corrected_bottom_k', corrected), ('legacy_top_k', reversed_set)]:
                    with bypass_layers(model, indices):
                        pruned = last_logits(model, ids, attention_mask)
                    rows.append({'sample': i, 'tokens': ids.shape[1], 'k': k, 'policy': policy,
                                 'skipped_layers': sorted(indices),
                                 'kl': kl_from_logits(pruned, teacher).item()})
            print(f'Completed prompt {i+1}/{len(records)}', flush=True)
    summary = {}
    for k in args.k:
        for policy in ['corrected_bottom_k', 'legacy_top_k']:
            subset = [r for r in rows if r['k'] == k and r['policy'] == policy]
            counts = Counter(tuple(r['skipped_layers']) for r in subset)
            summary[f'{policy}_k{k}'] = {'samples': len(subset),
                                       'mean_kl': sum(r['kl'] for r in subset) / len(subset),
                                       'unique_masks': len(counts),
                                       'most_common_mask_fraction': counts.most_common(1)[0][1]/len(subset)}
    result = {'scope': 'Last-token KL diagnostic on provided prompts; not downstream accuracy or speed.',
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'prompt_file_sha256': hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
              'torch_version': torch.__version__, 'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
