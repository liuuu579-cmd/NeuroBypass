"""Independent read-only coverage, route, calibration and preservation audit.

Does not rerun models or overwrite launcher status after a publication failure.
Writes a separate review receipt only after all assertions pass.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.open()]


def close(actual,expected,tol=1e-10):
    assert abs(actual-expected)<=tol,(actual,expected)


def audit(root,run):
    cfg=load(run/'protocol.json');manifest=load(run/'input_manifest.json')
    original=load(run/'preservation_before.json')
    assert all(Path(name).is_file() and sha256(name)==value for name,value in original.items())
    assert all(sha256(root/'tools'/name)==value for name,value in cfg['source_sha256'].items())
    assert all(sha256(run/name)==value for name,value in manifest['private_sha256'].items())
    assert all(sha256(p['checkpoint']['path'])==p['checkpoint']['sha256'] for p in cfg['router_policies'])
    allocation=load(run/'allocation.json');assert allocation['cpu_count']==42 and len(allocation['gpu_indices'])==4
    for rank in range(4):
        e=load(run/f'engineering_{rank:03d}.json');assert e['passed'] and e['all_base_and_router_parameters_frozen']
        assert len(e['cpu_ids'])==e['threads']==10
        assert all(set(v)<=set(e['cpu_ids']) for v in e['thread_cpu_ids'].values())
    inputs=lines(run/'private_ruler_inputs.jsonl');by_input={r['sample']:r for r in inputs}
    calibration={r['sample']:r for r in inputs if r['split']=='calibration'}
    evaluation={r['sample']:r for r in inputs if r['split']=='evaluation'}
    assert len(calibration)==36 and len(evaluation)==72
    assert not {r['case'] for r in calibration.values()}&{r['case'] for r in evaluation.values()}
    assert not {r['prompt_sha256'] for r in calibration.values()}&{r['prompt_sha256'] for r in evaluation.values()}
    assert all(len(r['prompt_ids'])==r['length'] and r['length']+48<=2048 for r in inputs)
    grid=Counter((r['task'],r['length'],r['depth']) for r in evaluation.values())
    assert len(grid)==36 and set(grid.values())=={2}
    cal=[r for i in range(4) for r in lines(run/f'private_calibration_{i:03d}.jsonl')]
    assert len(cal)==36*3*6
    assert len({(r['sample'],r['k'],r['path_name']) for r in cal})==len(cal)
    assert {r['sample'] for r in cal}==set(calibration)
    selection=load(run/'static_portfolio_selection.json')['selected']
    for k in (10,12,16):
        options=[]
        for name in cfg['static_candidates']:
            group=[r['kl'] for r in cal if r['k']==k and r['path_name']==name]
            assert len(group)==36
            options.append((math.fsum(group)/36,name))
        expected,name=min(options)
        assert name==selection[str(k)]['path_name'];close(expected,selection[str(k)]['calibration_kl'])
    ruler=[r for i in range(4) for r in lines(run/f'private_ruler_scores_{i:03d}.jsonl')]
    assert len(ruler)==720 and len({(r['sample'],r['policy'],r['k']) for r in ruler})==720
    assert Counter(r['sample'] for r in ruler)==Counter({i:10 for i in evaluation})
    groups=defaultdict(list)
    for row in ruler:
        r=evaluation[row['sample']]
        assert row['case']==r['case'] and row['length']==len(r['prompt_ids'])
        assert row['target_tokens']==len(r['target_ids']) and 1<=len(row['generated_ids'])<=48
        assert len(set(row['skips']))==row['k'] and set(row['skips'])<=set(range(2,40))
        if row['policy']=='static_joint_kl':assert row['skips']==cfg['static_path'][:row['k']]
        elif row['policy']=='calibrated_static_portfolio':
            assert row['skips']==cfg['static_candidates'][selection[str(row['k'])]['path_name']][:row['k']]
        elif row['policy']=='router_m6':assert row['seed']==579 and row['skips'][:6]==cfg['static_path'][:6]
        else:assert row['policy']=='full' and row['k']==0 and row['kl']==0
        # Independently replay the RULER-style substring coverage metric.
        recall=sum(answer.lower() in row['generated_text'].lower() for answer in r['answers'])/len(r['answers'])
        close(recall,row['answer_recall']);assert row['all_answers']==int(recall==1)
        groups[row['policy'],row['k']].append(row)
    summary=load(run/'summary.json')
    for row in summary['ruler_aggregates']:
        if row['dimension']!='overall':continue
        group=groups[row['policy'],row['k']];assert len(group)==row['records']==72
        for metric in ('kl','nll','answer_recall','all_answers','strict_exact'):
            close(math.fsum(r[metric] for r in group)/len(group),row[metric])
    diagnostics=lines(run/'private_diagnostic_inputs.jsonl')
    assert len(diagnostics)==len({r['document_sha256'] for r in diagnostics})==16
    assert Counter(r['prompt_tokens'] for r in diagnostics)==Counter({256:4,512:4,1024:4,1536:4})
    searches=[r for i in range(4) for r in lines(run/f'private_search_{i:03d}.jsonl')]
    assert len(searches)==32 and len({(r['sample'],r['m']) for r in searches})==32
    candidate_count=0
    for record in searches:
        previous=cfg['static_path'][:record['m']]
        assert len(record['steps'])==16-record['m']
        for step in record['steps']:
            assert step['previous_skips']==previous and step['step']==len(previous)+1
            costs=step['candidate_costs'];assert len(costs)==38-len(previous)
            assert {i for i,_ in costs}==set(range(2,40))-set(previous)
            assert all(math.isfinite(value) for _,value in costs)
            i,value=min(costs,key=lambda v:(v[1],v[0]))
            assert i==step['selected_layer'];close(value,step['joint_continuation_kl'])
            previous=previous+[i];candidate_count+=len(costs)
        assert previous==record['ordered_skips']
    scores=[r for i in range(4) for r in lines(run/f'private_diagnostic_scores_{i:03d}.jsonl')]
    assert len(scores)==304 and len({(r['sample'],r['policy'],r['k'],r['seed']) for r in scores})==304
    assert Counter(r['sample'] for r in scores)==Counter({i:19 for i in range(16)})
    full=groups['full',0]
    return {'passed':True,'preserved_files':len(original),'ruler_evaluation_cases':72,
        'ruler_calibration_cases':36,'ruler_rows':len(ruler),'diagnostic_documents':16,
        'diagnostic_rows':len(scores),'joint_greedy_searches':len(searches),
        'joint_candidate_evaluations_recorded':candidate_count,
        'static_portfolio_selection_recomputed':True,'static_masks_fixed_across_tasks':True,
        'all_worker_thread_affinities_verified':True,'source_input_checkpoint_hashes_verified':True,
        'full_model_strict_exact_fraction':math.fsum(r['strict_exact'] for r in full)/72,
        'format_limit':'Full-model strict exact is0 under unrestricted48-token completion; use answer coverage as exploratory retrieval metric, not strict-format superiority.',
        'launcher_status_preserved':load(run/'status.json'),
        'publication_status':'Model computation and all research audits completed; first publication check failed after branch update. Verify the separate publication receipt; do not rerun GPU work.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    result=audit(root,args.run.resolve())
    with args.receipt.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
