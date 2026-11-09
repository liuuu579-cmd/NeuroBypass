"""Four-card, within-card interleaved steady-state cached inference benchmark."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from horizontal_data_v1 import policies
from horizontal_quality_v3 import pin_threads
from prompt_cost_core_v1 import Layout, cache_probe, trial
from prompt_depth_core_v1 import DepthRouter
from stage1_benchmark_v2 import sha256, stamp, write_json


def worker(rank, config, messages):
    try:
        cpu = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
        pin_threads(cpu[rank*10:(rank+1)*10]); torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained('/home/liushenyu/opt-13b', local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval()
        model.requires_grad_(False); layout = Layout(model)
        routers = {}
        for policy in config['policies']:
            if policy['kind'] == 'router' and policy['checkpoint']['path'] not in routers:
                router = DepthRouter().eval().requires_grad_(False)
                router.load_state_dict(torch.load(policy['checkpoint']['path'], map_location='cpu', weights_only=True)['state_dict'])
                routers[policy['checkpoint']['path']] = router
        inputs = np.load(config['private_inputs'], allow_pickle=False)
        output = Path(config['output'])
        probes = []
        with torch.inference_mode():
            for policy in config['policies']:
                if policy['deployment'] == 'physical': continue
                router = routers.get(policy.get('checkpoint', {}).get('path'))
                if router is not None: router.to(f'cuda:{rank}')
                ids = torch.tensor(inputs[0, :160], device=f'cuda:{rank}', dtype=torch.long)[None]
                probes.append(cache_probe(model, layout, ids, policy, router))
                if router is not None: router.to('cpu')
                del ids
        write_json(output / f'cache_probe_{rank:03d}.json', {'rank': rank, 'probes': probes,
            'cpu_ids': sorted(os.sched_getaffinity(0)),
            'thread_cpu_ids': {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}})
        if any(not p['cache_within_tolerance'] or not p.get('physical_equivalent', True) for p in probes):
            raise ValueError('Cache or physical-deletion probe failed; retained unchanged tolerances')
        length = [128, 512, 1024, 512][rank]
        rounds = config['warmups'] + config['repeats']
        rng = random.Random(579)
        with (output / f'timing_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
            for repetition in range(rounds):
                order = list(config['policies']); rng.shuffle(order)
                for order_index, policy in enumerate(order):
                    layout.activate(policy)
                    router = routers.get(policy.get('checkpoint', {}).get('path'))
                    if router is not None: router.to(f'cuda:{rank}')
                    ids = torch.tensor(inputs[repetition, :length], device=f'cuda:{rank}', dtype=torch.long)[None]
                    result = trial(model, ids, policy, router, config['new_tokens'])
                    row = {'rank': rank, 'physical_gpu': config['gpu_indices'][rank], 'length': length,
                        'replicate': int(rank == 3), 'round': repetition, 'warmup': repetition < config['warmups'],
                        'order': order_index, 'policy': policy['name'], 'kind': policy['kind'],
                        'seed': policy['seed'], 'k': policy['k'], 'deployment': policy['deployment'], **result}
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    if router is not None: router.to('cpu')
                    del ids
                messages.put({'kind': 'progress', 'rank': rank, 'round': repetition+1, 'rounds': rounds, 'length': length})
        messages.put({'kind': 'done', 'rank': rank})
    except BaseException: messages.put({'kind': 'error', 'rank': rank, 'traceback': traceback.format_exc()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT', '0')) != 4: p.error('Use four-GPU resource launcher')
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    training = root / 'runs/stage6_depth_train_main_20261107_01'
    selected = [r for r in policies(root, training, [10, 12]) if r['name'] in ['full', 'hybrid', 'static_joint_kl', 'sleb_nll_opt']]
    selected = [{**r, 'deployment': 'bypass'} for r in selected]
    selected += [{**r, 'deployment': 'physical'} for r in selected if r['name'] in ['static_joint_kl', 'sleb_nll_opt']]
    source = root / 'runs/stage3_prompt_fixed_pilot_20261104_01/private_prompt_continuation_ids.npy'
    training_ids = json.loads((training / 'protocol.json').read_text())['train_ids']
    original = np.load(source, allow_pickle=False)
    # Timing-only composed training text; same content and length across policies.
    stream = original[training_ids].reshape(-1)
    inputs = np.stack([stream[i*1024:(i+1)*1024].copy() for i in range(25)])
    inputs[:, 0] = original[training_ids[0], 0]
    private = output / 'private_cost_inputs.npy'
    with private.open('xb') as handle: np.save(handle, inputs, allow_pickle=False)
    allocation = json.loads((output / 'allocation.json').read_text())
    config = {'output': str(output), 'private_inputs': str(private), 'input_sha256': sha256(private),
        'source_input_sha256': sha256(source), 'input_construction': '25 nonoverlapping composed1024token blocks from frozen training records only; per-request BOS explicitly set; timing content is not a quality dataset.',
        'policies': selected, 'gpu_indices': allocation['gpu_indices'], 'probe': args.probe,
        'warmups': 0 if args.probe else 5, 'repeats': 1 if args.probe else 20,
        'new_tokens': 4 if args.probe else 128, 'source_sha256': {p.name: sha256(p) for p in
            [Path(__file__), root/'tools/prompt_cost_core_v1.py', root/'tools/prompt_fixed_core_v1.py', root/'tools/prompt_depth_core_v1.py']},
        'quality_source': 'runs/stage7_horizontal_main_20261108_02/summary.json',
        'lengths_by_rank': [128,512,1024,512], 'precision': 'FP16 SDPA batch1',
        'timing': 'Within-card round-interleaved randomized policies; same prompt per round; CUDA sync at start/router/prefill/end; EOS ignored; dynamic includes router; first token belongs to prefill; decode throughput uses127later tokens.',
        'memory': 'PyTorch peak allocated/reserved after per-request allocator cleanup; actual KV tensor bytes, model CUDA parameter bytes, router CUDA parameter bytes. CUDA context and CPU shadow copies excluded; not whole-process NVML peak.',
        'physical': 'Remove blocks from runtime decoder and CUDA; renumber cache indices; CPU shadow only permits next policy restoration. Layout/weight transfer before timing is excluded as deployment setup.',
        'cache': 'Each request starts fresh; path fixed throughout. Changing populated cache path rejected. New turn must prefill entire accumulated context; prompt-length timings include full rebuild but do not separately time old-cache disposal.',
        'limitations': ['No multi-request dynamic batching or serving scheduler benchmark.', 'Timing uses normal OPT CausalLM full-sequence prefill head.', 'K10/12 selected after quality review; K16 remains quality stress point.', 'Rank3 repeats512token setting; speed ratios computed within each physical card, not across cards.']}
    write_json(output/'protocol.json',config)
    tracked = {str(path): sha256(path) for path in root.rglob('*') if path.is_file()
        and not path.is_relative_to(output) and '.git' not in path.parts
        and path.suffix in {'.py','.md','.json','.jsonl','.csv','.png'}}
    for path in root.parent.glob('*.py'): tracked[str(path)] = sha256(path)
    write_json(output/'preservation_before.json',tracked)
    context = mp.get_context('spawn'); messages = context.Queue()
    processes = [context.Process(target=worker,args=(rank,config,messages)) for rank in range(4)]
    completed = set(); started=time.monotonic()
    try:
        for process in processes: process.start()
        cpu = list(map(int,os.environ['NEUROBYPASS_CPU_IDS'].split(','))); pin_threads(cpu[40:42])
        while len(completed)<4:
            try: message=messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker exited before reporting completion')
                continue
            if message['kind']=='error': raise RuntimeError(message['traceback'])
            if message['kind']=='done': completed.add(message['rank'])
            else: stamp('cost_round_complete',**{k:v for k,v in message.items() if k!='kind'})
        for process in processes:
            process.join(timeout=30)
            if process.exitcode!=0: raise RuntimeError('Cost worker exited abnormally')
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join(timeout=10)
            if process.is_alive(): process.kill(); process.join(timeout=5)
    for name,digest in tracked.items():
        if sha256(name)!=digest: raise ValueError('Existing evidence changed: '+name)
    write_json(output/'preservation_after.json',{'preserved_files':len(tracked)})
    write_json(output/'completion.json',{'state':'completed','elapsed_seconds':time.monotonic()-started,
        'expected_rows':4*(config['warmups']+config['repeats'])*len(selected),'policies':len(selected),'probe':args.probe})
    stamp('cost_measurement_complete',elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':main()
