"""Reuse completed pre-training scores only after exact input/route replay checks."""
import io
import json
from pathlib import Path

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def reuse_baseline(model,bank,data,dev,fresh,benchmark,routes,device,quality,rank,seed,arm,config,evaluate):
    old = Path(config['baseline_run']); out = Path(config['output'])
    source = old/f'quality_{rank:03d}.jsonl'
    if sha256(source) != config['baseline_quality_sha256'][str(rank)]:
        raise ValueError('Original pre-training score file changed')
    original = json.loads((old/'protocol.json').read_text())
    assert original['model']==config['model'] and original['data_run']==config['data_run']
    assert original['static_skips']==config['static_skips']
    assert original['router_policies']==config['router_policies']
    old_manifest = json.loads((old/'input_manifest.json').read_text())
    manifest = json.loads((out/'input_manifest.json').read_text())
    for key in ('train_ids','selection_ids','development_ids'):
        assert old_manifest[key]==manifest[key]
    for name in ('private_fresh_inputs.jsonl','private_benchmark_inputs.jsonl'):
        assert sha256(old/name)==sha256(out/name)
    assert sha256(old/f'private_routes_{rank:03d}.json')==sha256(out/f'private_routes_{rank:03d}.json')
    rows = read_lines(source)
    assert all((r['rank'],r['seed'],r['arm'],r['stage'])==(rank,seed,arm,'before') for r in rows)
    expected = len(dev)+len(fresh)+len(benchmark)
    registry = {(r['task'],r['sample'],r['policy']):r for r in rows}
    if len(registry)!=len(rows) or len(rows)!=expected*2:
        raise ValueError('Incomplete or duplicated baseline scoring')
    # Check exact numerical replay across both target-only and benchmark heads.
    selected = []
    for task in sorted({r['task'] for r in benchmark}):
        selected += [r for r in benchmark if r['task']==task][:1]
    replay = io.StringIO()
    evaluate(model,bank,data,dev[:2],fresh[:1],selected,routes,device,'before',replay,rank,seed,arm)
    maximum = 0.0
    replay_rows = [json.loads(s) for s in replay.getvalue().splitlines()]
    for row in replay_rows:
        reference = registry[row['task'],row['sample'],row['policy']]
        if row.keys()!=reference.keys(): raise ValueError('Baseline replay schema changed')
        for key,value in row.items():
            if isinstance(value,float):
                maximum = max(maximum,abs(value-reference[key]))
            elif value != reference[key]:
                raise ValueError('Baseline replay identity or prediction changed')
    if maximum > 1e-6: raise ValueError('Baseline numerical replay failed')
    for row in rows: quality.write(json.dumps(row,allow_nan=False)+'\n')
    quality.flush()
    write(out/f'baseline_replay_{rank:03d}.json',{'passed':True,'source_run':str(old),
        'source_quality_sha256':sha256(source),'reused_rows':len(rows),'replay_rows':len(replay_rows),
        'replay_max_abs_error':maximum,'private_inputs_and_frozen_routes_identical':True,
        'interpretation':'Only finished before-training scores reused; failed training is never reported as a finished recovery experiment.'})
