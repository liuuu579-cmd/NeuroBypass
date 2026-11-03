"""Validate completed greedy-search evidence and immutable input/source files.

This program only reads existing evidence and writes new verification/CSV paths.
It does not require torch or a GPU.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, tolerance=1e-8):
    require(abs(a-b) <= tolerance, message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();run=args.run.resolve()
    status=json.loads((run/'status.json').read_text())
    require(status['state']=='completed' and status['exit_code']==0,'Run not completed')
    experiment=json.loads((run/'experiment.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    samples=experiment['sample_ids'];sample_set=set(samples)
    objectives=experiment['configuration']['objectives'];ks=experiment['configuration']['k']
    require(len(samples)==32 and len(sample_set)==32,'Expected 32 distinct samples')
    steps=load_lines(run/'greedy_search_steps.jsonl')
    paths=load_lines(run/'greedy_paths.jsonl')
    rows=load_lines(run/'sample_metrics.jsonl')
    require(len(steps)==len(samples)*len(objectives)*max(ks),'Missing search steps')
    require(len(paths)==len(samples)*len(objectives),'Missing search paths')
    require(len(rows)==len(samples)*(1+len(ks)*(1+2*len(objectives))),'Missing metric rows')
    groups=defaultdict(list)
    for step in steps:
        require(step['sample'] in sample_set and step['objective'] in objectives,'Unexpected search')
        groups[(step['objective'],step['sample'])].append(step)
    path_map={}
    for path in paths:
        key=(path['objective'],path['sample'])
        require(key not in path_map,'Duplicate path')
        path_map[key]=path['ordered_skips']
    for key,group in groups.items():
        group.sort(key=lambda item:item['step']);previous=[]
        require([item['step'] for item in group]==list(range(1,max(ks)+1)),'Missing ordered step')
        for step in group:
            candidates=step['candidate_costs'];costs=dict(candidates)
            require(len(costs)==len(candidates),'Duplicate candidate')
            require(set(costs)==set(range(2,40))-set(previous),'Candidate set mismatch')
            require(all(math.isfinite(value) and value>=-1e-5 for value in costs.values()),'Invalid candidate KL')
            best=min(candidates,key=lambda item:(item[1],item[0]))
            require(best[0]==step['selected_layer'],'Selected candidate is not minimum')
            close(best[1],step['objective_kl'],'Chosen candidate cost mismatch')
            ordered=sorted(costs.values())
            close(ordered[1]-ordered[0],step['second_best_gap'],'Second best gap mismatch')
            previous.append(best[0])
            require(step['ordered_skips']==previous,'Search prefix mismatch')
        require(path_map[key]==previous,'Final path mismatch')
    metrics=['last_token_kl','all_token_mean_kl','next_token_nll']
    keyed={}
    for row in rows:
        key=(row['sample'],row['policy'],row['k'])
        require(key not in keyed,'Duplicate metric key');keyed[key]=row
        require(row['sample'] in sample_set,'Unexpected metric sample')
        mask=row['skipped_layers'];k=row['k']
        require(len(mask)==len(set(mask))==k and all(2<=v<40 for v in mask),'Invalid mask')
        require(row['next_token_count']==255,'Wrong next token count')
        require(all(math.isfinite(row[m]) and row[m]>=-1e-5 for m in metrics),'Invalid quality metric')
        for objective in objectives:
            if row['policy']=='matched_greedy_'+objective:
                require(mask==sorted(path_map[(objective,row['sample'])][:k]),'Matched path mismatch')
            if row['policy']=='shuffled_greedy_'+objective:
                donor=experiment['shuffled_mask_donors'][str(row['sample'])]
                require(donor!=row['sample'] and row['donor_sample']==donor,'Wrong donor')
                require(mask==sorted(path_map[(objective,donor)][:k]),'Shuffled path mismatch')
    stage1=Path(experiment['configuration']['stage1_run'])
    old={(r['sample'],r['policy'],r['k']):r for r in load_lines(stage1/'sample_metrics.jsonl')}
    for sample in samples:
        for policy,k in [('full',0)]+[('calibrated_static_greedy',k) for k in ks]:
            key=(sample,policy,k)
            for metric in metrics:
                close(keyed[key][metric],old[key][metric],'Baseline differs from stage 1',1e-5)
        for objective in objectives:
            for k in ks:
                for prefix in ['matched_greedy_','shuffled_greedy_']:
                    require((sample,prefix+objective,k) in keyed,'Missing policy on sample')
    for objective in objectives:
        for k in ks:
            matched=Counter(tuple(keyed[(s,'matched_greedy_'+objective,k)]['skipped_layers']) for s in samples)
            shuffled=Counter(tuple(keyed[(s,'shuffled_greedy_'+objective,k)]['skipped_layers']) for s in samples)
            require(matched==shuffled,'Transfer changed marginal mask distribution')
    for policy in summary['policies']:
        group=[r for r in rows if r['policy']==policy['policy'] and r['k']==policy['k']]
        require(len(group)==len(samples)==policy['samples'],'Wrong policy sample count')
        for metric in metrics:
            close(sum(r[metric] for r in group)/len(group),policy[metric]['mean'],'Summary mean mismatch')
        close(math.exp(sum(r['next_token_nll'] for r in group)/len(group)),policy['perplexity'],'PPL mismatch',1e-7)
    for comparison in summary['paired_comparisons']:
        k=comparison['k'];metric=comparison['metric']
        differences=[keyed[(s,comparison['policy'],k)][metric]-keyed[(s,comparison['reference'],k)][metric] for s in samples]
        close(sum(differences)/len(differences),comparison['mean_delta'],'Paired difference mismatch')
        close(sum(v<0 for v in differences)/len(differences),comparison['fraction_strictly_better'],'Paired win count mismatch')
    baseline=json.loads((root/'analysis/stage2_start_baseline_20261103.json').read_text())['files']
    for record in baseline:
        require(digest(root/record['path'])==record['sha256'],'Existing repository file changed: '+record['path'])
    original=json.loads((root/'analysis/source_manifest.json').read_text())['files']
    for record in original:
        require(digest(record['source_path'])==record['sha256'],'Original input file changed: '+record['source_path'])
        require(digest(root/record['snapshot_path'])==record['sha256'],'Archived input file changed: '+record['snapshot_path'])
    require(digest(stage1/'private_eval_token_ids.npz')==experiment['input_data_sha256'],'Input token file changed')
    require(digest(root/'tools/stage2_greedy_reference_v1.py')==experiment['program_sha256'],'Experiment program changed')
    result={'verified_samples':len(samples),'verified_searches':len(paths),'verified_search_steps':len(steps),
            'verified_candidate_scores':sum(len(s['candidate_costs']) for s in steps),
            'verified_metric_rows':len(rows),'preserved_preexisting_repository_files':len(baseline),
            'preserved_original_and_snapshot_pairs':len(original),'matched_stage1_baseline_verified':True,
            'same_mask_distribution_under_transfer_verified':True,
            'evidence_sha256':{p.name:digest(p) for p in [run/'experiment.json',run/'greedy_search_steps.jsonl',run/'greedy_paths.jsonl',run/'sample_metrics.jsonl',run/'summary.json']},
            'scope':'Structural, arithmetic and hash validation; does not rerun model inference.',
            'limitations':['Same-prompt full-teacher search is not a deployable policy or guaranteed optimum.',
                           'Bootstrap unit is a token block; document independence is unverified.']}
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    with args.output.with_suffix('.csv').open('x',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['policy','k','samples','last_token_kl','all_token_kl','nll','perplexity','unique_masks','most_common_mask_fraction'])
        for p in summary['policies']:
            writer.writerow([p['policy'],p['k'],p['samples'],p['last_token_kl']['mean'],p['all_token_mean_kl']['mean'],p['next_token_nll']['mean'],p['perplexity'],p['unique_masks'],p['most_common_mask_fraction']])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
