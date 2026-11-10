"""Record both serial-head conventions before accepting a candidate scorer."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from full_context_core_v1 import candidate_costs
from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers, setup_worker, thread_record, write
from prompt_fixed_core_v1 import continuation_logits, fixed_route


def worker(rank, config, inbox, messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('/home/liushenyu/opt-13b', local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
    data = Records(config['data'])
    sample = max((r['sample'] for r in data.rows if r['split'] == 'train'), key=lambda s: data.rows[s]['prompt_length'])
    array, available = data[sample]; p = min([64, 129, 513, 2016][rank], available)
    ids = torch.tensor(array[:p+32], device=f'cuda:{rank}', dtype=torch.long)[None]
    selected, candidates = [4, 9, 2, 12, 17, 7], [3, 20, 39]
    rows = []
    with torch.inference_mode():
        teacher = F.log_softmax(continuation_logits(model, ids, p, []), -1)
        for objective in ['kl', 'nll']:
            shared = dict(candidate_costs(model, ids, p, selected, candidates, teacher, objective))
            for candidate in candidates:
                lp_sliced = F.log_softmax(continuation_logits(model, ids, p, selected+[candidate]), -1)
                sliced = (teacher.exp()*(teacher-lp_sliced)).sum(-1).mean().item()
                with fixed_route(model, selected+[candidate]):
                    full = model(ids[:, :-1], use_cache=False).logits.float()
                lp_full = F.log_softmax(full[:, p-1:], -1)
                full_kl = (teacher.exp()*(teacher-lp_full)).sum(-1).mean().item()
                nll = F.cross_entropy(full.reshape(-1, full.size(-1)), ids[:, 1:].reshape(-1)).item()
                rows.append({'objective': objective, 'candidate': candidate, 'shared': shared[candidate],
                    'sliced_head_serial_kl': sliced, 'full_head_serial_kl': full_kl, 'full_head_serial_nll': nll,
                    'shared_minus_serial': shared[candidate]-(sliced if objective == 'kl' else nll),
                    'head_convention_kl_difference': full_kl-sliced})
    write(Path(config['output']) / f'diagnostic_{rank:03d}.json',
        {'rank': rank, 'prompt_length': p, 'rows': rows, **thread_record()})
    messages.put({'kind': 'done', 'rank': rank})


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    config = {'output': os.environ['NEUROBYPASS_RUN_DIR'],
        'data': str(root / 'runs/stage13_full_context_probe_20261110_01')}
    with Workers(worker, config) as workers:
        workers.receive(4, 'done')
