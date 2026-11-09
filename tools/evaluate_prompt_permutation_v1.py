"""Three frozen path permutations, matched scores reused from stage7."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import torch

from horizontal_data_v1 import policies, read_dataset, wiki_windows
from horizontal_quality_v3 import pin_threads, score_probe, score_sequence
from prompt_depth_core_v1 import DepthRouter
from prompt_fixed_core_v1 import raw_prompt_embeddings
from prompt_permutation_core_v1 import (PERMUTATION_SEEDS, assignments, load_reference,
    policy_key, record_key, score_fields)
from stage1_benchmark_v2 import sha256, stamp, write_json


def worker(rank, config, messages):
    try:
        cpus = list(map(int, os.environ['NEUROBYPASS_CPU_IDS'].split(',')))
        pin_threads(cpus[rank*10:(rank+1)*10]); torch.set_num_threads(10); torch.set_num_interop_threads(1)
        torch.cuda.set_device(rank)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained('/home/liushenyu/opt-13b', local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
        output = Path(config['output'])
        inputs = [json.loads(s) for s in Path(config['private_inputs']).read_text().splitlines()]
        if config['probe']:
            registry, routers = {}, {}
            with torch.inference_mode():
                for p in config['policies']:
                    path = p['checkpoint']['path']
                    if path not in routers:
                        router = DepthRouter().to(f'cuda:{rank}').eval().requires_grad_(False)
                        router.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
                        routers[path] = router
                for record in inputs:
                    ids = torch.tensor(record['route_ids'], device=f'cuda:{rank}', dtype=torch.long)[None]
                    emb = raw_prompt_embeddings(model, ids)
                    for p in config['policies']:
                        route = routers[p['checkpoint']['path']].route(emb, p['k'], p['prefix'])[0].tolist()
                        registry[record_key(record)+policy_key(p)] = {'skipped_layers': sorted(route)}
        else:
            registry = load_reference(config['source'], config['policies'])
        donor = {(r['task'],r['sample'],r['permutation_seed']): r['donor_sample']
            for r in json.loads((output/'assignments.json').read_text())}
        assigned = inputs[rank::4]
        probes, replays = [], []
        with torch.inference_mode():
            for task in sorted({r['task'] for r in assigned}):
                record = next(r for r in assigned if r['task']==task)
                probes.append({'task': task, **score_probe(model, record, [])})
                # Recompute every selected policy on one record per task/card,
                # establishing exact scoring compatibility before reusing old rows.
                for p in config['policies']:
                    ref = registry[record_key(record)+policy_key(p)]
                    scores = [score_sequence(model, ids, n, ref['skipped_layers'])
                        for ids,n in zip(record['sequence_ids'],record['target_lengths'])]
                    error = None if config['probe'] else max(abs(a-b) for a,b in zip(scores,ref['scores']))
                    if error is not None and error>2e-4: raise ValueError('Stage7 matched-score replay mismatch')
                    replays.append({'task':task,'sample':record['sample'],'policy':p['name'],
                        'seed':p['seed'],'k':p['k'],'max_score_error':error,'tolerance':2e-4})
        write_json(output/f'worker_probe_{rank:03d}.json',{'rank':rank,'cpu_ids':sorted(os.sched_getaffinity(0)),
            'thread_cpu_ids':{t.name:sorted(os.sched_getaffinity(int(t.name))) for t in Path('/proc/self/task').iterdir()},
            'likelihood_probes':probes,'matched_score_replays':replays})
        started=time.monotonic()
        with (output/f'permutation_{rank:03d}.jsonl').open('x') as stream, torch.inference_mode():
            for offset,record in enumerate(assigned):
                cache={}
                for p in config['policies']:
                    ref=registry[record_key(record)+policy_key(p)]
                    own=tuple(ref['skipped_layers'])
                    matched = ref.get('scores')
                    if matched is None:
                        matched=[score_sequence(model,ids,n,own) for ids,n in zip(record['sequence_ids'],record['target_lengths'])]
                    cache[own]=matched
                    for seed in PERMUTATION_SEEDS:
                        donated=donor[record['task'],record['sample'],seed]
                        route=registry[(record['task'],donated)+policy_key(p)]['skipped_layers']
                        key=tuple(route)
                        if key not in cache:
                            cache[key]=[score_sequence(model,ids,n,route) for ids,n in zip(record['sequence_ids'],record['target_lengths'])]
                        scores=cache[key]
                        row={'task':record['task'],'sample':record['sample'],'document':record.get('document'),
                            'document_sha256':record['document_sha256'],'policy':p['name'],'seed':p['seed'],'k':p['k'],
                            'permutation_seed':seed,'donor_sample':donated,'matched_skipped_layers':list(own),
                            'shuffled_skipped_layers':route,'same_mask':key==own,
                            'target_lengths':record['target_lengths'],'normalization_lengths':record['normalization_lengths'],
                            'matched_scores':matched,'shuffled_scores':scores,
                            'matched':score_fields(record,matched),'shuffled':score_fields(record,scores)}
                        stream.write(json.dumps(row)+'\n')
                stream.flush()
                if offset%20==0 or offset+1==len(assigned):
                    messages.put({'kind':'progress','rank':rank,'completed':offset+1,'total':len(assigned),
                        'task':record['task'],'elapsed_seconds':time.monotonic()-started})
        messages.put({'kind':'done','rank':rank})
    except BaseException:messages.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',action='store_true');args=p.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT','0'))!=4:p.error('Use four-GPU resource launcher')
    root=Path(__file__).resolve().parents[1];output=Path(os.environ['NEUROBYPASS_RUN_DIR'])
    source=root/'runs/stage7_horizontal_main_20261108_02'
    training=root/'runs/stage6_depth_train_main_20261107_01'
    selected=[p for p in policies(root,training,[10,12]) if p['kind']=='router']
    if args.probe:
        old=root/'runs/stage7_horizontal_probe_20261108_03/private_benchmark_inputs.jsonl'
        inputs=[json.loads(s) for s in old.read_text().splitlines() if json.loads(s)['task']!='wikitext_conditional']
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/home/liushenyu/opt-13b',use_fast=False,local_files_only=True)
        seen=set()
        for record in wiki_windows(read_dataset('wikitext','validation'),tokenizer):
            if record['document'] not in seen:
                inputs.append(record);seen.add(record['document'])
            if len(seen)==4:break
    else:
        if json.loads((source/'verification.json').read_text())['status']!='passed':raise ValueError('Stage7 not audited')
        private=source/'private_benchmark_inputs.jsonl'
        expected=json.loads((source/'data_manifest.json').read_text())['private_input_sha256']
        if sha256(private)!=expected:raise ValueError('Input evidence changed')
        inputs=[json.loads(s) for s in private.read_text().splitlines()]
    private=output/'private_permutation_inputs.jsonl'
    with private.open('x') as stream:
        for record in inputs:stream.write(json.dumps(record)+'\n')
    mapping=assignments(inputs);write_json(output/'assignments.json',mapping)
    config={'output':str(output),'source':str(source),'probe':args.probe,'policies':selected,
        'private_inputs':str(private),'private_inputs_sha256':sha256(private),
        'permutation_seeds':PERMUTATION_SEEDS,'assignments_sha256':sha256(output/'assignments.json'),
        'input_records':len(inputs),'expected_rows':len(inputs)*len(selected)*len(PERMUTATION_SEEDS),
        'source_quality_sha256':{} if args.probe else {path.name:sha256(path) for path in sorted(source.glob('quality_*.jsonl'))},
        'source_sha256':{path.name:sha256(path) for path in [Path(__file__),root/'tools/prompt_permutation_core_v1.py',
            root/'tools/horizontal_quality_v3.py',root/'tools/horizontal_data_v1.py',root/'tools/prompt_depth_core_v1.py',root/'tools/prompt_fixed_core_v1.py']},
        'null':'Break input/path pairing within each task while preserving exact multiset of masks per model/K/permutation. Mask identity may remain unchanged; do not force different masks.',
        'wiki':'Cross-article bijection via randomized article blocks and rotation at least max article window count. No donor comes from recipient article; not uniform over all constrained permutations.',
        'mc':'Uniform permutation conditioned on no self-assignment; mapping independent of labels/options/model scores.',
        'matched':'Reuse audited stage7 matched score rows, with batch1 GPU replay checks on all12policies*4tasks*4cards; probe recomputes routes and scores.',
        'claims':'Mechanism ablation on already evaluated tasks, not new independent benchmark. Permutation seeds are not additional independent questions; report task-level paired bootstrap and permutation variation separately.',
        'selection':'No training, no checkpoint/hyperparameter selection using these scores. Bothpure/hybrid and all3modelseeds retained.'}
    write_json(output/'protocol.json',config)
    tracked={str(path):sha256(path) for path in root.rglob('*') if path.is_file() and not path.is_relative_to(output)
        and '.git' not in path.parts and path.suffix in {'.py','.md','.json','.jsonl','.csv','.png'}}
    for path in root.parent.glob('*.py'):tracked[str(path)]=sha256(path)
    write_json(output/'preservation_before.json',tracked)
    context=mp.get_context('spawn');messages=context.Queue();processes=[context.Process(target=worker,args=(r,config,messages)) for r in range(4)]
    done=set();started=time.monotonic()
    try:
        for process in processes:process.start()
        cpus=list(map(int,os.environ['NEUROBYPASS_CPU_IDS'].split(',')));pin_threads(cpus[40:42])
        while len(done)<4:
            try:item=messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in processes) or all(p.exitcode is not None for p in processes):raise RuntimeError('Permutation worker exited early')
                continue
            if item['kind']=='error':raise RuntimeError(item['traceback'])
            if item['kind']=='done':done.add(item['rank'])
            else:stamp('permutation_progress',**{k:v for k,v in item.items() if k!='kind'})
        for process in processes:
            process.join(timeout=30)
            if process.exitcode!=0:raise RuntimeError('Permutation worker failure')
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(timeout=10)
            if process.is_alive():process.kill();process.join(timeout=5)
    for path,digest in tracked.items():
        if sha256(path)!=digest:raise ValueError('Existing evidence changed: '+path)
    count=sum(1 for path in output.glob('permutation_*.jsonl') for line in path.open())
    if count!=config['expected_rows']:raise ValueError('Incomplete permutation records')
    write_json(output/'preservation_after.json',{'preserved_files':len(tracked)})
    write_json(output/'completion.json',{'state':'completed','rows':count,'elapsed_seconds':time.monotonic()-started,'probe':args.probe})
    stamp('permutation_complete',rows=count,elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':main()
