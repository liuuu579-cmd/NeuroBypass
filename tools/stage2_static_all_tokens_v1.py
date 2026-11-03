"""Fair-objective static control: use only the original eight calibration blocks.

Additive follow-up to the per-input reference. No source or input is modified.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage1_benchmark_v2 import model_logits, quality_metrics, sha256, stamp, summarize, write_json
from stage2_greedy_reference_v1 import choose_best_candidate, compare_paired, evaluate_candidate, wait_result


def worker(rank,arguments,inbox,results):
    try:
        args=argparse.Namespace(**arguments)
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']));torch.set_num_interop_threads(1)
        torch.manual_seed(579);torch.cuda.set_device(rank)
        device=torch.device(f'cuda:{rank}')
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.float16,
              local_files_only=True,attn_implementation='sdpa').to(device).eval()
        model.requires_grad_(False);model.config.use_cache=False
        data=np.load(args.stage1_run/'private_eval_token_ids.npz')
        calibration=[torch.tensor(row,dtype=torch.long,device=device).unsqueeze(0) for row in data['calibration']]
        with torch.inference_mode():
            teachers=[F.log_softmax(model_logits(model,ids,[]).float(),dim=-1) for ids in calibration]
        results.put({'kind':'ready','rank':rank,'pid':os.getpid(),'cpu_ids':sorted(os.sched_getaffinity(0)),
                     'threads':torch.get_num_threads(),'calibration_samples':len(calibration)})
        while True:
            task=inbox.get()
            if task is None:return
            with torch.inference_mode():
                if task['kind']=='calibration':
                    costs=[]
                    for index in task['candidates']:
                        cost=sum(evaluate_candidate(model,ids,task['selected']+[index],teacher,'all_tokens')
                                 for ids,teacher in zip(calibration,teachers))/len(calibration)
                        costs.append([index,cost])
                    results.put({'kind':'calibration','rank':rank,'step':task['step'],'costs':costs})
                elif task['kind']=='test':
                    for sample in task['samples']:
                        ids=torch.tensor(data['test'][sample],dtype=torch.long,device=device).unsqueeze(0)
                        teacher=F.log_softmax(model_logits(model,ids,[]).float(),dim=-1)
                        rows=[]
                        for k in [6,16]:
                            indices=task['selected'][:k]
                            rows.append({'sample':sample,'rank':rank,'policy':'static_all_tokens','k':k,
                                         'skipped_layers':sorted(indices),
                                         **quality_metrics(model_logits(model,ids,indices),teacher,ids)})
                        results.put({'kind':'sample','rows':rows})
                    results.put({'kind':'done','rank':rank})
                else:raise ValueError('Unknown task')
    except BaseException:results.put({'kind':'error','rank':rank,'traceback':traceback.format_exc()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-run',type=Path,default=Path('/home/liushenyu/test/NeuroBypass/runs/stage1_c4_20261103_02'))
    parser.add_argument('--stage2-run',type=Path,default=Path('/home/liushenyu/test/NeuroBypass/runs/stage2_greedy_20261103_01'))
    parser.add_argument('--base-model',type=Path,default=Path('/home/liushenyu/opt-13b'))
    args=parser.parse_args()
    if int(os.environ.get('NEUROBYPASS_GPU_COUNT','0'))!=2:parser.error('Use two GPUs through resource launcher')
    output=Path(os.environ['NEUROBYPASS_RUN_DIR'])
    old_status=json.loads((args.stage2_run/'status.json').read_text())
    if old_status['state']!='completed' or old_status['exit_code']!=0:raise ValueError('Stage 2 not complete')
    experiment=json.loads((args.stage2_run/'experiment.json').read_text())
    if sha256(args.stage1_run/'private_eval_token_ids.npz')!=experiment['input_data_sha256']:raise ValueError('Data changed')
    metadata={'configuration':{k:str(v) for k,v in vars(args).items()},'program_sha256':sha256(__file__),
              'input_data_sha256':experiment['input_data_sha256'],'cpu_ids':sorted(os.sched_getaffinity(0)),
              'gpu_visibility':os.environ['CUDA_VISIBLE_DEVICES'],'objective':'Mean all-token KL over original eight calibration blocks',
              'calibration_samples':8,'test_samples':256,'stage2_sample_ids':experiment['sample_ids'],
              'scope':'Static mask selected on calibration only; no evaluation-block mask selection.'}
    write_json(output/'experiment.json',metadata)
    context=mp.get_context('spawn');results=context.Queue();inboxes=[context.Queue() for _ in range(2)]
    processes=[context.Process(target=worker,args=(rank,vars(args),inboxes[rank],results)) for rank in range(2)]
    selected=[];rows=[];started=time.monotonic()
    try:
        for process in processes:process.start()
        ready=[wait_result(results,processes) for _ in processes]
        if any(r['kind']!='ready' or r['calibration_samples']!=8 for r in ready):raise ValueError('Readiness mismatch')
        write_json(output/'workers.json',ready)
        with (output/'static_calibration_steps.jsonl').open('x') as handle:
            for step in range(1,17):
                available=[i for i in range(2,40) if i not in selected]
                for rank in range(2):inboxes[rank].put({'kind':'calibration','step':step,'selected':selected.copy(),'candidates':available[rank::2]})
                responses=[wait_result(results,processes) for _ in processes]
                if any(r['kind']!='calibration' or r['step']!=step for r in responses):raise ValueError('Calibration mismatch')
                costs=sorted([c for response in responses for c in response['costs']])
                index,cost=choose_best_candidate(costs);selected.append(index)
                record={'step':step,'selected_layer':index,'mean_calibration_all_token_kl':cost,
                        'ordered_skips':selected.copy(),'candidate_costs':costs}
                handle.write(json.dumps(record)+'\n');handle.flush()
                stamp('static_all_token_step',step=step,selected_layer=index,kl=cost)
        write_json(output/'static_policy.json',{'ordered_skips':selected,'objective':metadata['objective'],'calibration_samples':8})
        n=len(np.load(args.stage1_run/'private_eval_token_ids.npz')['test'])
        for rank in range(2):inboxes[rank].put({'kind':'test','samples':list(range(rank,n,2)),'selected':selected.copy()})
        done=0
        with (output/'sample_metrics.jsonl').open('x') as handle:
            while done<2:
                result=wait_result(results,processes)
                if result['kind']=='done':done+=1;continue
                if result['kind']!='sample':raise ValueError('Test mismatch')
                for row in result['rows']:rows.append(row);handle.write(json.dumps(row)+'\n')
                handle.flush()
        require_count=n*2
        if len(rows)!=require_count or len({(r['sample'],r['k']) for r in rows})!=require_count:raise ValueError('Missing test rows')
        stage2_rows=[json.loads(line) for line in (args.stage2_run/'sample_metrics.jsonl').read_text().splitlines()]
        subset=[r for r in rows if r['sample'] in experiment['sample_ids']]
        comparisons=[]
        for k in [6,16]:
            for metric in ['last_token_kl','all_token_mean_kl','next_token_nll']:
                comparisons.append(compare_paired(stage2_rows+subset,'static_all_tokens',k,metric,'matched_greedy_all_tokens',579))
        write_json(output/'summary.json',{'metadata':metadata,'sample_count':n,'policies':summarize(rows),
                   'stage2_subset_policies':summarize(subset),'paired_comparisons_on_stage2_subset':comparisons,
                   'elapsed_seconds':time.monotonic()-started,
                   'limitations':['Only eight calibration blocks; static search can be strengthened further.',
                                  'Stage2 subset reference searches use teacher outputs on their evaluation input.']})
        stamp('static_all_tokens_completed',samples=n,elapsed_seconds=time.monotonic()-started)
    finally:
        for inbox in inboxes:inbox.put(None)
        for process in processes:
            if process.pid is not None:process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=10)
            if process.is_alive():process.kill();process.join()


if __name__=='__main__':main()
