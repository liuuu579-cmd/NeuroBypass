"""Audit prompt-fixed evidence, conditional greedy trajectories and immutability."""
import argparse
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines()]
def require(condition,message):
    if not condition:raise ValueError(message)


def verify(root,run,data_run):
    status=json.loads((run/'status.json').read_text());require(status['state']=='completed' and status['exit_code']==0,'Run incomplete')
    experiment=json.loads((run/'experiment.json').read_text());mode=experiment['configuration']['mode']
    manifest=json.loads((data_run/'data_manifest.json').read_text());records=manifest['records']
    require(dict(Counter(r['split'] for r in records))==manifest['split_counts'],'Split count mismatch')
    for field in ['record_sha256','record_lead_sha256','prefix_sha256']:
        require(len(set(r[field] for r in records))==len(records),'Duplicate grouping key '+field)
    require(digest(data_run/'private_prompt_continuation_ids.npy')==manifest['data_sha256']==experiment['data_sha256'],'Input array changed')
    labels=[r for path in sorted(run.glob('labels_*.jsonl')) for r in read_lines(path)]
    expected={r['sample'] for r in records if r['split'] in (['pilot'] if mode=='pilot' else ['train','validation'])}
    require(len(labels)==len(expected) and {r['sample'] for r in labels}==expected,'Incomplete/duplicate label records')
    candidate_scores=0
    for record in labels:
        selected=[]
        require(record['split']==records[record['sample']]['split'],'Label split mismatch')
        require(len(record['steps'])==6,'Incomplete path')
        for number,step in enumerate(record['steps'],1):
            require(step['step']==number and step['previous_skips']==selected,'Wrong conditional path state')
            candidates=step['candidate_costs'];costs=dict(candidates)
            require(len(costs)==len(candidates) and set(costs)==set(range(2,40))-set(selected),'Candidate set mismatch')
            require(all(math.isfinite(value) and value>=-1e-5 for value in costs.values()),'Invalid KL')
            chosen,cost=min(candidates,key=lambda pair:(pair[1],pair[0]))
            require(chosen==step['selected_layer'] and abs(cost-step['joint_continuation_kl'])<1e-8,'Non-greedy choice')
            selected.append(chosen);candidate_scores+=len(candidates)
        require(selected==record['ordered_skips'],'Path mismatch')
        require(record['keep_mask']==[int(i not in selected) for i in range(40)],'Keep/skip convention mismatch')
        require(abs(record['continuation_kl']-record['steps'][-1]['joint_continuation_kl'])<1e-6,'Final combined score mismatch')
    rows=read_lines(run/'sample_metrics.jsonl');policies=['full','static','per_input_greedy'] if mode=='pilot' else ['full','static','router','shuffled_router','fixed_input_router']
    evaluated={r['sample'] for r in records if r['split']==('pilot' if mode=='pilot' else 'test')}
    keys={(r['sample'],r['policy']) for r in rows}
    require(len(rows)==len(keys)==len(evaluated)*len(policies),'Incomplete quality evidence')
    require(keys=={(s,p) for s in evaluated for p in policies},'Unexpected policy/sample keys')
    lookup={(r['sample'],r['policy']):r for r in rows}
    for row in rows:
        require(row['continuation_tokens']==32,'Unexpected continuation length')
        require(row['k']==(0 if row['policy']=='full' else 6),'Unexpected pruning budget')
        mask=row['skipped_layers'];require(len(mask)==len(set(mask))==row['k'] and all(2<=i<40 for i in mask),'Invalid mask')
        require(all(math.isfinite(row[m]) and row[m]>=-1e-5 for m in ['continuation_kl','first_next_token_kl','continuation_nll']),'Invalid metric')
    if mode=='main':
        donors=json.loads((run/'shuffled_donors.json').read_text())
        require(set(map(int,donors))==evaluated==set(donors.values()),'Invalid donor permutation')
        for sample in evaluated:
            donor=donors[str(sample)];require(donor!=sample,'Donor fixed point')
            require(lookup[(sample,'shuffled_router')]['skipped_layers']==lookup[(donor,'router')]['skipped_layers'],'Wrong swapped mask')
        selected=json.loads((run/'router_selected_checkpoint.json').read_text())
        require(digest(selected['path'])==selected['sha256'] and selected['selection_uses_test'] is False,'Checkpoint changed')
        protocol=json.loads((run/'router_training_protocol.json').read_text())
        require(set(protocol['checkpoint_validation_documents']).isdisjoint(evaluated),'Test leaked into model selection')
        require(protocol['training_documents']==800 and protocol['base_model_frozen'] is True,'Training scope mismatch')
        training_source=root/'tools/train_prompt_fixed_v1.py'
        require(digest(training_source)==protocol['source_sha256'],'Training source changed')
    probes=json.loads((run/'cache_probes.json').read_text())
    for outcome in probes:
        for probe in outcome['probes']:
            require(probe['cached_vs_recomputed_mean_kl']<=1e-4 and probe['maximum_position_kl']<=1e-3,'Cache distribution error')
            require(probe['all_layer_lengths_verified'] and probe['retained_cache_length']==159,'Cache alignment mismatch')
            expected_bytes=2*(40-len(probe['route']))*159*5120*2
            require(probe['cache_bytes']==expected_bytes,'Cache storage mismatch')
    summary=json.loads((run/'summary.json').read_text())
    for policy in summary['policies']:
        group=[r for r in rows if r['policy']==policy['policy']]
        require(len(group)==policy['samples'],'Summary count mismatch')
        for metric in ['continuation_kl','first_next_token_kl','continuation_nll']:
            require(abs(sum(r[metric] for r in group)/len(group)-policy[metric])<1e-8,'Summary mean mismatch')
        require(abs(math.exp(policy['continuation_nll'])-policy['perplexity'])<1e-7,'PPL mismatch')
    for comparison in summary['paired_comparisons']:
        m=comparison['metric'];differences=[lookup[(s,comparison['policy'])][m]-lookup[(s,comparison['reference'])][m] for s in evaluated]
        require(abs(sum(differences)/len(differences)-comparison['mean_delta'])<1e-8,'Paired mean mismatch')
    timing=read_lines(run/'generation_timings.jsonl')
    expected_prompts=4 if mode=='pilot' else 8;expected_timing_policies=2 if mode=='pilot' else 3
    require(len(timing)==expected_prompts*expected_timing_policies*3,'Missing timing repetitions')
    for row in timing:
        require(row['new_tokens']==(32 if mode=='pilot' else 64),'Timing workload mismatch')
        require(abs(row['route_seconds']+row['prefill_seconds']+row['decode_seconds']-row['total_seconds'])<1e-6,'Timing arithmetic mismatch')
        expected_bytes=2*(40-len(row['skipped_layers']))*(128+row['new_tokens']-1)*5120*2
        require(row['cache_bytes']==expected_bytes,'Generation cache bytes mismatch')
    baseline=json.loads((root/'analysis/stage3_start_baseline_20261104.json').read_text())['files']
    for record in baseline:require(digest(root/record['path'])==record['sha256'],'Pre-existing file changed: '+record['path'])
    original=json.loads((root/'analysis/source_manifest.json').read_text())['files']
    for record in original:
        require(digest(record['source_path'])==record['sha256'],'Original source changed')
        require(digest(root/record['snapshot_path'])==record['sha256'],'Archived source changed')
    for name,sha in experiment['source_sha256'].items():require(digest(root/'tools'/name)==sha,'Experiment source changed: '+name)
    return {'mode':mode,'records':len(records),'verified_labels':len(labels),'verified_conditional_greedy_steps':len(labels)*6,
            'verified_candidate_scores':candidate_scores,'verified_metric_rows':len(rows),'verified_timing_rows':len(timing),
            'preserved_preexisting_text_evidence_files':len(baseline),'preserved_original_and_snapshot_pairs':len(original),
            'no_exact_record_prompt_or_lead_overlap_across_splits':True,
            'evidence_sha256':{p.name:digest(p) for p in [run/'experiment.json',run/'summary.json',run/'sample_metrics.jsonl']},
            'scope':'Structural, arithmetic and input/source hash checks; does not rerun model inference.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--data-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.root,args.run,args.data_run)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
