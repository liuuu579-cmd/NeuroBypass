"""Validate complete stage-1 outputs and add paired uncertainty estimates."""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    status=json.loads((args.run/'status.json').read_text())
    assert status['state']=='completed' and status['exit_code']==0
    rows=[json.loads(line) for line in (args.run/'sample_metrics.jsonl').read_text().splitlines()]
    summary=json.loads((args.run/'summary.json').read_text())
    n=summary['sample_count']
    assert len(rows)==15*n
    keys=[(r['sample'],r['policy'],r['k']) for r in rows]
    assert len(keys)==len(set(keys))
    assert set(r['sample'] for r in rows)==set(range(n))
    for r in rows:
        assert len(r['skipped_layers'])==r['k']
        assert not set(r['skipped_layers']).intersection([0,1])
        assert r['next_token_count']==255
    groups=defaultdict(list)
    for r in rows:groups[(r['policy'],r['k'])].append(r)
    for group in groups.values():group.sort(key=lambda r:r['sample'])
    comparisons=[]
    rng=np.random.default_rng(579)
    boot_indices=rng.integers(0,n,size=(10000,n))
    for policy,k in sorted(groups):
        if policy in ['full','calibrated_static_greedy']:continue
        reference=groups[('calibrated_static_greedy',k)]
        selected=groups[(policy,k)]
        for metric in ['last_token_kl','all_token_mean_kl','next_token_nll']:
            delta=np.array([a[metric]-b[metric] for a,b in zip(selected,reference)])
            boot=delta[boot_indices].mean(axis=1)
            comparisons.append({'policy':policy,'k':k,'metric':metric,
                                'mean_delta_vs_calibrated_static':float(delta.mean()),
                                'paired_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
                                'fraction_strictly_better_than_static':float(np.mean(delta<0))})
    result={'verified_complete_samples':n,'verified_metric_rows':len(rows),
            'sampling_unit':'256-token blocks; document-level dependence is not measured.',
            'bootstrap_replicates':10000,'bootstrap_seed':579,
            'summary_sha256':hashlib.sha256((args.run/'summary.json').read_bytes()).hexdigest(),
            'comparisons':comparisons,
            'limitations':['Block bootstrap treats selected blocks as independent.',
                           'No multiplicity correction; descriptive paired intervals, not a task benchmark claim.']}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    csv_path=args.output.with_suffix('.csv')
    with csv_path.open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['policy','k','samples','mean_last_token_kl','mean_all_token_kl','ppl','unique_masks','most_common_mask_fraction'])
        for policy in summary['policies']:
            writer.writerow([policy['policy'],policy['k'],policy['samples'],policy['last_token_kl']['mean'],
                             policy['all_token_mean_kl']['mean'],policy['perplexity'],policy['unique_masks'],policy['most_common_mask_fraction']])
    print(json.dumps({'verified_samples':n,'verified_metric_rows':len(rows),
                      'comparisons':len(comparisons),'output':str(args.output)},indent=2))


if __name__=='__main__':main()
