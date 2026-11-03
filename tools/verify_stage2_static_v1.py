"""Check the fair-objective static control and export the matched 32-block table."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root;run=args.run
    status=json.loads((run/'status.json').read_text())
    require(status['state']=='completed' and status['exit_code']==0,'Run incomplete')
    experiment=json.loads((run/'experiment.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    steps=read_lines(run/'static_calibration_steps.jsonl')
    rows=read_lines(run/'sample_metrics.jsonl')
    require(len(steps)==16 and len(rows)==512,'Wrong evidence counts')
    selected=[]
    for number,step in enumerate(steps,1):
        require(step['step']==number,'Wrong step order')
        costs=step['candidate_costs'];mapping=dict(costs)
        require(len(mapping)==len(costs) and set(mapping)==set(range(2,40))-set(selected),'Wrong candidate set')
        require(all(math.isfinite(v) and v>=-1e-5 for v in mapping.values()),'Invalid candidate score')
        index,cost=min(costs,key=lambda row:(row[1],row[0]))
        require(index==step['selected_layer'] and abs(cost-step['mean_calibration_all_token_kl'])<1e-8,'Wrong minimum')
        selected.append(index)
        require(selected==step['ordered_skips'],'Wrong path prefix')
    require(selected==json.loads((run/'static_policy.json').read_text())['ordered_skips'],'Policy mismatch')
    keyed={(r['sample'],r['k']):r for r in rows}
    require(len(keyed)==512 and set(keyed)=={(s,k) for s in range(256) for k in [6,16]},'Missing sample/K')
    metrics=['last_token_kl','all_token_mean_kl','next_token_nll']
    for row in rows:
        require(row['skipped_layers']==sorted(selected[:row['k']]) and row['next_token_count']==255,'Invalid metric mask')
        require(all(math.isfinite(row[m]) and row[m]>=-1e-5 for m in metrics),'Invalid metric')
    samples=experiment['stage2_sample_ids']
    for policies,subset in [(summary['policies'],rows),(summary['stage2_subset_policies'],[r for r in rows if r['sample'] in samples])]:
        for policy in policies:
            group=[r for r in subset if r['k']==policy['k']]
            require(len(group)==policy['samples'],'Summary count mismatch')
            for metric in metrics:require(abs(sum(r[metric] for r in group)/len(group)-policy[metric]['mean'])<1e-8,'Summary mean mismatch')
            require(abs(math.exp(sum(r['next_token_nll'] for r in group)/len(group))-policy['perplexity'])<1e-7,'PPL mismatch')
    stage2=Path(experiment['configuration']['stage2_run'])
    reference=read_lines(stage2/'sample_metrics.jsonl')
    matched={(r['sample'],r['k']):r for r in reference if r['policy']=='matched_greedy_all_tokens'}
    for comparison in summary['paired_comparisons_on_stage2_subset']:
        k=comparison['k'];m=comparison['metric']
        differences=[matched[(s,k)][m]-keyed[(s,k)][m] for s in samples]
        require(abs(sum(differences)/len(differences)-comparison['mean_delta'])<1e-8,'Paired mean mismatch')
    baseline=json.loads((root/'analysis/stage2_start_baseline_20261103.json').read_text())['files']
    for record in baseline:require(digest(root/record['path'])==record['sha256'],'Existing file changed: '+record['path'])
    original=json.loads((root/'analysis/source_manifest.json').read_text())['files']
    for record in original:
        require(digest(record['source_path'])==record['sha256'],'Original file changed')
        require(digest(root/record['snapshot_path'])==record['sha256'],'Archived file changed')
    stage1=Path(experiment['configuration']['stage1_run'])
    require(digest(stage1/'private_eval_token_ids.npz')==experiment['input_data_sha256'],'Data changed')
    require(digest(root/'tools/stage2_static_all_tokens_v1.py')==experiment['program_sha256'],'Program changed')
    output={'verified_calibration_blocks':8,'verified_calibration_steps':16,'verified_candidate_means':sum(len(s['candidate_costs']) for s in steps),
            'verified_test_blocks':256,'verified_metric_rows':len(rows),'stage2_subset_blocks':len(samples),
            'preserved_preexisting_repository_files':len(baseline),'preserved_original_and_snapshot_pairs':len(original),
            'evidence_sha256':{p.name:digest(p) for p in [run/'experiment.json',run/'static_calibration_steps.jsonl',run/'static_policy.json',run/'sample_metrics.jsonl',run/'summary.json']},
            'scope':'Structural, arithmetic and hash validation; does not rerun model inference.'}
    with args.output.open('x') as handle:json.dump(output,handle,ensure_ascii=False,indent=2)
    stage2_summary=json.loads((stage2/'summary.json').read_text())
    with args.output.with_suffix('.csv').open('x',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['policy','k','samples','last_token_kl','all_token_kl','nll','perplexity'])
        for policy in stage2_summary['policies']+summary['stage2_subset_policies']:
            writer.writerow([policy['policy'],policy['k'],policy['samples'],policy['last_token_kl']['mean'],policy['all_token_mean_kl']['mean'],policy['next_token_nll']['mean'],policy['perplexity']])
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
