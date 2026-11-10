"""Strict policy contracts and read-only merging for the stage11 correction."""
import json
import math
from pathlib import Path

from stage1_benchmark_v2 import sha256

SEEDS = [579, 580, 581]
BUDGETS = [10, 12]
ARMS = [0, 4, 6, 8]
STAGE6 = 'runs/stage6_depth_train_main_20261107_01'
STAGE10 = 'runs/stage10_prefix_main_20261109_01'


def key(policy):
    return policy.get('name', policy.get('policy')), policy['seed'], policy['k']


def validate_policy(policy, global_path):
    k = policy['k']
    if policy['kind'] == 'static':
        expected = [] if policy['name'] == 'full' else global_path[:k]
        if policy['skips'] != expected:
            raise ValueError('Static policy differs from the frozen joint-greedy path')
    else:
        m, prefix = policy['m'], policy['prefix']
        if m not in ARMS or policy['name'] != f'm{m:02d}' or not 0 <= m <= k:
            raise ValueError('Invalid fixed-prefix arm')
        if len(prefix) != m or prefix != global_path[:m]:
            raise ValueError(f'Fixed-prefix mismatch: m={m}, actual={prefix}, expected={global_path[:m]}')
        if len(set(prefix)) != m or any(i < 2 or i >= 40 for i in prefix):
            raise ValueError('Invalid prefix layers')


def policies(root):
    root = Path(root)
    stage6, stage10 = root / STAGE6, root / STAGE10
    global_path = json.loads((stage6 / 'static_paths.json').read_text())['static_joint_kl']
    training = json.loads((stage10 / 'protocol.json').read_text())
    if global_path != training['global_path']:
        raise ValueError('Training and inference global paths differ')
    six = json.loads((stage6 / 'fixed_global_policy.json').read_text())['ordered_skips']
    if six != global_path[:6]:
        raise ValueError('Legacy six-layer prefix differs from full path')
    result = [{'name': 'full', 'kind': 'static', 'k': 0, 'skips': [], 'seed': None}]
    for k in BUDGETS:
        result.append({'name': 'static_joint_kl', 'kind': 'static', 'k': k,
                       'skips': global_path[:k], 'seed': None})
        for m in ARMS:
            run = stage6 if m in [0, 6] else stage10
            arm = {0: 'pure', 6: 'hybrid'}.get(m, f'm{m:02d}')
            for seed in SEEDS:
                checkpoint = json.loads((run / f'{arm}_seed{seed}_checkpoints.json').read_text())[str(k)]
                if sha256(checkpoint['path']) != checkpoint['sha256']:
                    raise ValueError('Frozen checkpoint digest mismatch')
                policy = {'name': f'm{m:02d}', 'kind': 'router', 'seed': seed, 'k': k,
                          'checkpoint': checkpoint, 'prefix': global_path[:m], 'm': m}
                validate_policy(policy, global_path)
                result.append(policy)
    return global_path, result


def partition(original, corrected):
    old = {key(p): p for p in original}
    new = {key(p): p for p in corrected}
    if len(old) != len(original) or len(new) != 27 or old.keys() != new.keys():
        raise ValueError('Policy coverage differs from the original frozen comparison')
    reused, rerun = [], []
    for policy in corrected:
        before = old[key(policy)]
        if policy['name'] == 'm08':
            # The only allowed change is the incorrect inference prefix.
            if {k: v for k, v in before.items() if k != 'prefix'} != {
                    k: v for k, v in policy.items() if k != 'prefix'}:
                raise ValueError('Affected checkpoint or non-prefix setting changed')
            if before['prefix'] != policy['prefix'][:6] or len(policy['prefix']) != 8:
                raise ValueError('Original failure is not the documented truncated prefix')
            rerun.append(policy)
        else:
            if before != policy:
                raise ValueError('Attempt to reuse a changed policy')
            reused.append(policy)
    if len(reused) != 21 or len(rerun) != 6:
        raise ValueError('Incorrect correction partition')
    return reused, rerun


def iter_rows(run):
    for path in sorted(Path(run).glob('quality_*.jsonl')):
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)


def combined_rows(run, config):
    reused = {key(p) for p in config['reused_policies']}
    for row in iter_rows(config['reuse_run']):
        if key(row) in reused:
            yield row
    yield from iter_rows(run)


def aggregate_seed_metric(values, wiki=False):
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError('Missing or nonfinite metric')
    mean = sum(values) / len(values)
    return math.exp(mean) if wiki else 100 * mean
