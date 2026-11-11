"""Static-start two-exchange router: full joint labels, paired training and audit.

Use the resource launcher with4 GPUs/42 CPUs. --probe uses training/selection
identities only, then a separate controller may launch the finite main pilot.
"""
import argparse
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from evaluate_full_context_four_gpu_v1 import policies
from full_context_data_v1 import Records
from full_context_experiment_v1 import Workers,preserve,read_lines,resources,verify_preserved,write
from headroom_ruler_core_v1 import paired,jsonl
from stage1_benchmark_v2 import sha256,stamp
from static_swap_core_v2 import build_bank,checks,greedy_indices
from static_swap_experiment_v2 import all_labels,worker


def load(path):return json.loads(path.read_text())


def prepare_quality(root,out,source,probe):
    data=Records(source['data_run']);inputs=[]
    selected=source['selection_ids'][:4] if probe else source['validation_ids']
    for sample in selected:
        raw,p=data[sample]
        inputs.append({'task':'c4_development','source_sample':sample,'route_ids':raw[:p].tolist(),
            'sequence_ids':[raw.tolist()],'target_lengths':[32],'normalization_lengths':[32],'gold':None})
    if not probe:
        # Fixed previously reviewed populations; no reserved C4 test or new adaptation.
        benchmark=read_lines(root/'runs/stage15_depth_slot_recovery_v4_20261111_01/private_benchmark_inputs.jsonl')
        rng=np.random.default_rng(170579)
        for task in ('piqa','hellaswag','winogrande','wikitext_conditional'):
            group=[r for r in benchmark if r['task']==task]
            for i in sorted(rng.choice(len(group),64,replace=False)):
                r=dict(group[int(i)]);r['source_sample']=r['sample'];inputs.append(r)
        fresh=read_lines(root/'runs/stage15_depth_slot_recovery_v4_20261111_01/private_fresh_inputs.jsonl')
        for p in (256,512,1024,1536):
            group=[r for r in fresh if r['prompt_tokens']==p]
            for i in sorted(rng.choice(len(group),8,replace=False)):
                r=group[int(i)]
                inputs.append({'task':'c4_prior_fresh','source_sample':r['sample'],'route_ids':r['ids'][:p],
                    'sequence_ids':[r['ids']],'target_lengths':[32],'normalization_lengths':[32],'gold':None})
        for r in read_lines(root/'runs/stage16_headroom_ruler_20261111_01/private_ruler_inputs.jsonl'):
            if r['split']!='evaluation':continue
            inputs.append({'task':'ruler_'+r['task'],'source_sample':r['sample'],'case':r['case'],
                'route_ids':r['prompt_ids'],'sequence_ids':[r['prompt_ids']+r['target_ids']],
                'target_lengths':[len(r['target_ids'])],'normalization_lengths':[len(r['target_ids'])],
                'gold':None,'answers':r['answers'],'answer_kind':r['answer_kind']})
    for i,r in enumerate(inputs):r['sample']=i
    jsonl(out/'private_quality_inputs.jsonl',inputs)
    write(out/'quality_manifest.json',{'task_records':dict(Counter(r['task'] for r in inputs)),
        'private_input_sha256':sha256(out/'private_quality_inputs.jsonl'),
        'selection':'Fixed before new labels/training; previously reviewed diagnostic/task cohorts, no untouched final-test claim.',
        'router_observation':'Full actual visible prompt; no appended answer/options/targets; same path across all answer options.',
        'ruler':'Same exploratory Stage16 cases and48-token completion protocol; no RULER training. Strict matching has a known extra-text limitation.'})
    return inputs


def controls(out,config):
    import torch
    labels=all_labels(out)
    costs=torch.tensor([labels[i]['kl'] for i in config['train_ids']],dtype=torch.float64)
    means=costs.mean(0);best=int(means.argmin());greedy=int(greedy_indices(means[None],config['bank'],2)[0])
    result={'train_best_route':best,'train_mean_greedy_route':greedy,
        'train_best_mean_kl':float(means[best]),'train_mean_greedy_kl':float(means[greedy]),
        'original_static_mean_kl':float(means[0]),
        'policy':'Both global fixed paths chosen from TRAIN mean costs only. Best-static uses entire same candidate bank including double exchanges; independent of prompt or development.'}
    write(out/'static_controls.json',result);return result


def summarize(out,config,inputs):
    rows=[r for i in range(4) for r in read_lines(out/f'private_quality_{i:03d}.jsonl')]
    expected=len(inputs)*14
    assert len(rows)==expected,(len(rows),expected)
    assert len({(r['sample'],r['policy'],r['seed'],r['maximum_exchanges']) for r in rows})==len(rows)
    source={r['sample']:r for r in inputs};grouped=defaultdict(list);index={}
    for r in rows:
        assert r['task']==source[r['sample']]['task'] and r['prompt_tokens']==len(source[r['sample']]['route_ids'])
        assert math.isfinite(r['kl']) and math.isfinite(r['nll'])
        if r['policy']!='full':assert len(set(r['skips']))==16 and min(r['skips'])>=2
        if r['policy'].startswith('swap_'):assert r['actual_exchanges']<=r['maximum_exchanges']
        grouped[r['task'],r['policy'],r['seed'],r['maximum_exchanges']].append(r)
        index[r['sample'],r['policy'],r['seed'],r['maximum_exchanges']]=r
    aggregates=[]
    for key,group in sorted(grouped.items(),key=lambda kv:str(kv[0])):
        row=dict(zip(('task','policy','seed','maximum_exchanges'),key));row['records']=len(group)
        row.update({n:float(np.mean([r[n] for r in group])) for n in ('kl','nll','actual_exchanges')})
        row['ppl']=math.exp(row['nll']);row['unique_masks']=len({tuple(r['skips']) for r in group})
        for n in ('correct','correct_norm','answer_recall','all_answers','strict_exact'):
            if n in group[0]:row[n]=float(np.mean([r[n] for r in group]))
        aggregates.append(row)
    comparisons=[]
    for task in sorted({r['task'] for r in inputs}):
        samples=[r['sample'] for r in inputs if r['task']==task]
        for maximum in (1,2):
            for baseline in ('original_static','train_best_static','train_mean_greedy_static','old_router_m6','swap_constant'):
                for metric in ('kl','nll','correct_norm','all_answers'):
                    if metric not in index[samples[0],'swap_prompt',config['seeds'][0],maximum]:continue
                    differences=[]
                    for sample in samples:
                        left=float(np.mean([index[sample,'swap_prompt',s,maximum][metric] for s in config['seeds']]))
                        if baseline in ('old_router_m6','swap_constant'):
                            budget=maximum if baseline=='swap_constant' else 0
                            right=float(np.mean([index[sample,baseline,s,budget][metric] for s in config['seeds']]))
                        else:right=index[sample,baseline,None,0][metric]
                        differences.append((right-left) if metric in ('kl','nll') else (left-right))
                    clusters=[source[s].get('case') or source[s].get('document') or s for s in samples]
                    comparisons.append({'task':task,'maximum_exchanges':maximum,'baseline':baseline,'metric':metric,
                        'positive_means':'prompt swap improves quality',**paired(differences,clusters,170580)})
    labels=all_labels(out);oracle=[]
    for split,ids in [('train',config['train_ids']),('selection',config['selection_ids']),
                      ('other_development',[i for i in config['validation_ids'] if i not in set(config['selection_ids'])])]:
        if not ids:continue
        import torch
        cost=torch.tensor([labels[i]['kl'] for i in ids]);routes=greedy_indices(cost,config['bank'],2)
        oracle.append({'split':split,'documents':len(ids),'static_kl':float(cost[:,0].mean()),
            'future_greedy_bank_kl':float(cost.gather(1,routes[:,None]).mean()),
            'future_best_bank_kl':float(cost.min(-1).values.mean()),
            'reference':'Uses actual future targets, offline reference only.'})
    write(out/'summary.json',{'quality_rows':len(rows),'input_records':len(inputs),'aggregates':aggregates,
        'paired_comparisons':comparisons,'candidate_bank_oracle':oracle,
        'limits':'Two-seed development pilot, K16, no recovery; all task populations previously reviewed. No deployment speed/memory claims. Uncertainty intervals do not include full training-seed uncertainty.'})
    for rank in (0,2):
        a=load(out/f'fit_result_{rank:03d}.json');b=load(out/f'fit_result_{rank+1:03d}.json')
        assert a['seed']==b['seed'] and a['initial_sha256']==b['initial_sha256'] and a['parameters']==b['parameters']
        assert a['optimizer_updates']==b['optimizer_updates']
    for policy in ('original_static','train_best_static','train_mean_greedy_static'):
        assert len({tuple(r['skips']) for r in rows if r['policy']==policy})==1
    write(out/'verification.json',{'passed':True,'quality_rows':len(rows),'input_records':len(inputs),
        'paired_initialization_and_update_budgets_match':True,'all_static_paths_fixed':True,
        'at_most_two_exchanges_and_constant_k16_checked':True,
        'all_engineering_gates_passed':all(load(out/f'engineering_{i:03d}.json')['passed'] for i in range(4)),
        'all_selection_costs_replayed':all(load(out/f'selection_replay_{i:03d}.json')['passed'] for i in range(4))})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',action='store_true');a=p.parse_args();resources()
    root=Path(__file__).resolve().parents[1];out=Path(os.environ['NEUROBYPASS_RUN_DIR']);tracked=preserve(root,out)
    write(out/'cpu_checks.json',checks());training=root/'runs/stage13_full_context_20261110_02_training'
    source,paths,available=policies(training,[16],[6]);bank=build_bank(training)
    write(out/'candidate_bank.json',bank)
    train=source['train_ids'];validation=source['validation_ids'];selection=source['selection_ids']
    if a.probe:
        data=Records(source['data_run']);ordered=sorted(train,key=lambda i:(data.rows[i]['prompt_length'],i))
        train=[ordered[i] for i in (0,len(ordered)//8,len(ordered)//4,len(ordered)//2,3*len(ordered)//4,7*len(ordered)//8,-2,-1)]
        validation=selection[:4];selection=validation
    inputs=prepare_quality(root,out,{**source,'validation_ids':validation,'selection_ids':selection},a.probe)
    names=['run_static_swap_four_gpu_v2.py','static_swap_core_v2.py','static_swap_experiment_v2.py',
           'full_context_core_v1.py','prompt_fixed_core_v1.py','full_context_experiment_v1.py']
    config={'output':str(out),'model':source['model'],'data_run':source['data_run'],'probe':a.probe,
        'bank':bank,'train_ids':train,'validation_ids':validation,'selection_ids':selection,
        'seeds':[579,580],'width':128,'epochs':1 if a.probe else 24,
        'old_policies':[p for p in available if p['kind']=='router' and p['seed'] in (579,580)],
        'old_prefix':paths['static_joint_kl'][:6],
        'training':'Full prompt embedding+positions; joint candidate KL residual versus original static. Training-mean prior + zero-initialized residual, smoothL1 normalized by training delta std; AdamW2e-4, batch16,24epochs, warmup5%/cosine, clip1.',
        'selection':'Original32 IDs select earliest minimum true candidate-table continuation KL, epochs0,1,4,...24; no test or answer at inference. Swap limit2 fixed before evaluation.',
        'static_control':'Best fixed full bank route from original train mean, plus greedy train-mean fixed and paired constant-observation router.',
        'route':'Start from static16 mask; at most1 or2 successive exchange steps; stop/no-change always eligible, static wins ties. Never change route within answer, never change K.',
        'resources':'4GPU/42CPU, workers10each/controller2',
        'source_sha256':{n:sha256(root/'tools'/n) for n in names},
        'input_manifest_sha256':sha256(out/'quality_manifest.json'),'candidate_bank_sha256':sha256(out/'candidate_bank.json')}
    # Protocol without a repeated full source-label manifest; bank file retains it.
    write(out/'protocol.json',{k:v for k,v in config.items() if k!='bank'})
    with Workers(worker,config) as workers:
        workers.receive(4,'ready');stamp('swap_engineering_passed',probe=a.probe,candidates=len(bank['routes']))
        workers.send_all({'kind':'labels'});workers.receive(4,'labels_done')
        static=controls(out,config)
        workers.send_all({'kind':'fit'});workers.receive(4,'fit_done')
        workers.send_all({'kind':'quality','static_controls':static});workers.receive(4,'done')
    summarize(out,config,inputs);verify_preserved(out,tracked)
    stamp('static_swap_complete',probe=a.probe,inputs=len(inputs),preserved_files=len(tracked))


if __name__=='__main__':main()
