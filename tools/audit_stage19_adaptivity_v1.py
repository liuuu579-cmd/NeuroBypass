"""Classify frozen Stage19 policies independently of their experiment names.

Never equate a prompt_* policy name with actual input-conditioned execution.
Read completed quality records only; write aggregate evidence in new paths.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='stage19_parallel_20261112_03')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    codebook = root / 'runs' / (args.prefix + '_codebook')
    budget = root / 'runs' / (args.prefix + '_budget')
    result_path = root / 'analysis' / (args.prefix + '_adaptivity_audit.json')
    report_path = root / 'analysis' / (args.prefix + '_adaptivity_audit.zh-CN.md')
    assert not result_path.exists() and not report_path.exists()
    cfg = load(budget / 'protocol.json')
    inputs = read_lines(budget / 'private_inputs.jsonl')
    quality_paths = [budget / f'private_quality_{rank:03d}.jsonl' for rank in range(3)]
    for rank, path in enumerate(quality_paths):
        assert sha256(path) == load(budget / f'quality_worker_{rank:03d}.json')['quality_sha256']
    rows = [row for path in quality_paths for row in read_lines(path)]
    policies = []
    for seed in cfg['seeds']:
        for arm in ('prompt', 'constant'):
            calibrated = load(budget / f'budget_{arm}_s{seed}_calibration.json')
            assert sha256(calibrated['checkpoint']['path']) == calibrated['checkpoint']['sha256']
            for op in calibrated['operating_points']:
                items = [r for r in rows if r['seed'] == seed and r['policy'] == op['name']]
                assert len(items) == len(inputs)
                mc = [r for r in items if 'correct_norm' in r]
                assert len(mc) == 96
                counts = dict(sorted(Counter(r['k'] for r in items).items()))
                if op['kind'] == 'static':
                    assert len(counts) == 1 and all(r['index'] == op['index'] for r in items)
                    classification = 'static_fallback'
                elif len(counts) == 1:
                    classification = 'router_fixed_budget_on_this_population'
                else:
                    classification = 'input_adaptive_budget_on_this_population'
                policies.append({
                    'seed': seed, 'arm': arm, 'policy': op['name'],
                    'checkpoint_epoch': calibrated['checkpoint']['epoch'],
                    'classification': classification, 'configured_kind': op['kind'],
                    'penalty': op['penalty'], 'records': len(items), 'k_counts': counts,
                    'mean_k': float(np.mean([r['k'] for r in items])),
                    'mc_k_counts': dict(sorted(Counter(r['k'] for r in mc).items())),
                    'mc_correct_serial': sum(r['correct_norm'] for r in mc),
                    'mc_records': len(mc), 'selection_policy': op['selection_policy'],
                })
    summary = load(codebook / 'summary.json')
    assert load(codebook / 'verification.json')['passed']
    allocations = [load(p / 'allocation.json') for p in (codebook, budget)]
    assert set(allocations[0]['cpu_ids']).isdisjoint(allocations[1]['cpu_ids'])
    bank = load(budget / 'budget_bank.json')
    stronger_masks = []
    for k in (16, 20, 24):
        static = next(r['skips'] for r in bank['routes'] if r['k'] == k)
        strongest = load(codebook / f'k{k}_b08_bank.json')['routes'][0]['skips']
        stronger_masks.append({'k': k, 'nested_path_equals_codebook_train_best': static == strongest})
    result = {
        'passed': True, 'prefix': args.prefix,
        'budget_classification': policies,
        'prompt_classification_counts': dict(Counter(r['classification'] for r in policies if r['arm'] == 'prompt')),
        'static_frontier_scope': 'Stage19 budget timing uses one nested training-derived joint-greedy path. It is not an exhaustive or best-known-mask frontier.',
        'known_static_mask_comparison': stronger_masks,
        'codebook_other_development': [r for r in summary['paired_comparisons'] if r['cohort'] == 'other_development' and r['metric'] == 'kl'],
        'resources': [{'gpus': a['gpu_indices'], 'cpu_ids': a['cpu_ids'], 'cpu_count': a['cpu_count']} for a in allocations],
        'source_sha256': sha256(Path(__file__)),
        'quality_file_sha256': {str(p.relative_to(root)): sha256(p) for p in quality_paths},
        'limits': 'Exploratory previously reviewed development populations; serial accuracy here must not be paired with cached runtime. A static fallback or timing noise does not establish a dynamic advantage.',
    }
    write(result_path, result)
    lines = [
        '# 阶段19实际自适应性复核', '',
        '按冻结的校准文件和已完成质量记录分类，不根据 prompt 策略名称判断其是否真的动态。此处准确率来自串行质量协议；最终耗时须搭配缓存计时协议自己的准确率。', '',
        '## 路线1的开发集结果', '',
        '以下68条文档未参与本轮选检查点，但此前已用于开发；区间为三种子先按文档平均后进行配对bootstrap，未校正多重比较。', '',
        '| 组合库 | 提示词KL | 常量KL | 同库强静态KL | 静态减提示词KL | 95%区间 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for name in sorted({r['bank_name'] for r in result['codebook_other_development']}):
        prompt = next(r for r in result['codebook_other_development'] if r['bank_name'] == name and r['arm'] == 'prompt')
        constant = next(r for r in result['codebook_other_development'] if r['bank_name'] == name and r['arm'] == 'constant')
        lo, hi = prompt['ci95']
        lines.append(f"| {name} | {prompt['mean']:.5f} | {constant['mean']:.5f} | {prompt['static_mean']:.5f} | {prompt['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    fresh = next(r for r in summary['paired_comparisons'] if r['bank_name'] == 'k20_b08' and r['cohort'] == 'c4_prior_fresh' and r['arm'] == 'prompt' and r['metric'] == 'kl')
    lines += ['', f"K20、8组合在另一组32条既有C4文档上的KL为动态{fresh['mean']:.5f}、静态{fresh['static_mean']:.5f}，尚无该群体改善的证据，不能把68条开发集上的收益表述为已经普遍泛化。", '',
        '## 路线5实际是否随输入改变K', '',
        '每个操作点在三个种子上分别分类，不能把静态回退的重复计时波动算作动态收益。', '',
        '| 种子 | 操作点 | 实际类型 | 260条K分布 | 96题K分布 | 串行正确题数 |',
        '|---|---|---|---|---|---:|']
    labels = {'static_fallback': '退回固定K', 'router_fixed_budget_on_this_population': '调用路由器但此群体K固定',
        'input_adaptive_budget_on_this_population': '实际随输入改变K'}
    for r in policies:
        if r['arm'] != 'prompt':
            continue
        lines.append(f"| {r['seed']} | {r['policy']} | {labels[r['classification']]} | {r['k_counts']} | {r['mc_k_counts']} | {r['mc_correct_serial']}/96 |")
    lines += ['', '上述串行正确题数只描述质量与预算分配，不能与后续缓存耗时混用。最终计时汇总必须保留按种子的静态回退/实际动态区别；平均K或策略名称本身不是动态收益证据。', '',
        '当前预算计时的静态曲线沿用同一训练联合贪心前缀。K16/20/24的同候选库训练均值最佳固定组合与这些前缀不同，因此该曲线不代表所有已知强静态组合的完整前沿。若后续观察到积极动态收益，需补测这些已知静态组合；当前不据弱基线比较宣称一般动态优势。', '',
        'GPU 0/1/2与3/4/5各分配32个CPU核心，两个CPU集合互斥。只新增本审计程序、聚合JSON和报告，不修改任何原始来源。', '',
        f'[聚合审计]({result_path.name}) · [执行协议](STAGE19_ROUTE1_ROUTE5_PROTOCOL_20261112.zh-CN.md)']
    with report_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(report_path), 'classifications': result['prompt_classification_counts']}))


if __name__ == '__main__':
    main()
