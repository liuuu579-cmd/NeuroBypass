"""Read-only strengthened audit and paired summaries of stage12 pilots.

All outputs use new versioned paths. Earlier draft reports remain evidence.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prefix_downstream_correction_core_v1 import policies
from prompt_permutation_core_v1 import assignments, score_fields
from stage1_benchmark_v2 import sha256, write_json


def preserve(run):
    tracked = json.loads((run / 'preservation_before.json').read_text())
    for p, digest in tracked.items():
        assert sha256(p) == digest, p
    assert json.loads((run / 'preservation_after.json').read_text())['preserved_files'] == len(tracked)
    assert json.loads((run / 'status.json').read_text())['state'] == 'completed'
    allocation = json.loads((run / 'allocation.json').read_text())
    assert allocation['cpu_count'] == len(set(allocation['cpu_ids'])) == 42
    assert len(set(allocation['gpu_indices'])) == 4
    return len(tracked), allocation


def threads(item, rank, allocation):
    assert item['cpu_ids'] == allocation['cpu_ids'][rank*10:(rank+1)*10]
    assert item['threads'] == 10
    assert all(set(cpus) <= set(item['cpu_ids']) for cpus in item['thread_cpu_ids'].values())


def fidelity(root, run):
    count, allocation = preserve(run)
    config = json.loads((run / 'protocol.json').read_text())
    assert config['probe'] is False
    path, selected = policies(root)
    assert config['global_path'] == path
    assert config['policies'] == [p for p in selected if p['name'] == 'm06']
    for group in ['source_sha256', 'provenance_sha256']:
        for p, digest in config[group].items():
            assert sha256(p) == digest, p
    assert sha256(run / 'private_inputs.jsonl') == config['private_input_sha256']
    assert sha256(run / 'assignments.json') == config['assignments_sha256']
    inputs = [json.loads(line) for line in (run / 'private_inputs.jsonl').open()]
    records = {(r['task'], r['sample']): r for r in inputs}
    assert len(records) == len(inputs) == 324
    assert Counter(r['task'] for r in inputs) == {'c4_development': 68, 'hellaswag': 64,
        'piqa': 64, 'wikitext_conditional': 64, 'winogrande': 64}
    assigned = json.loads((run / 'assignments.json').read_text())
    assert assigned == assignments(inputs)
    donors = {(r['task'], r['sample'], f"shuffled_{r['permutation_seed']}"): r['donor_sample'] for r in assigned}
    registry = None
    for rank in range(4):
        routes = json.loads((run / f'routes_{rank:03d}.json').read_text())
        current = {(r['task'], r['sample'], r['seed'], r['k']): r['skipped_layers'] for r in routes}
        assert len(current) == len(routes) == 324*6
        if registry is None:
            registry = current
        else:
            assert current == registry
        probe = json.loads((run / f'worker_probe_{rank:03d}.json').read_text())
        threads(probe, rank, allocation)
        assert len(probe['likelihood_probes']) == 10
        for item in probe['likelihood_probes']:
            assert abs(item['score']-item['reference_score']) == item['absolute_error'] <= 2e-4
    rows = [json.loads(line) for p in sorted(run.glob('diagnostics_*.jsonl')) for line in p.open()]
    by = {}
    conditions = ['matched', 'shuffled_1701', 'shuffled_1702', 'shuffled_1703', 'static_joint_kl']
    for r in rows:
        uid = r['task'], r['sample'], r['seed'], r['k'], r['condition']
        assert uid not in by
        by[uid] = r
        record = records[r['task'], r['sample']]
        assert r['seed'] in [579, 580, 581] and r['k'] in [10, 12] and r['condition'] in conditions
        assert r['document'] == record.get('document') and r['document_sha256'] == record['document_sha256']
        if r['condition'] == 'static_joint_kl':
            expected, donor = sorted(path[:r['k']]), None
        else:
            donor = r['sample'] if r['condition'] == 'matched' else donors[r['task'], r['sample'], r['condition']]
            expected = registry[r['task'], donor, r['seed'], r['k']]
        assert r['donor_sample'] == donor and r['skipped_layers'] == expected
        assert expected == sorted(set(expected)) and len(expected) == r['k']
        assert set(path[:6]) <= set(expected) and all(2 <= i < 40 for i in expected)
        assert r['target_tokens'] == sum(record['target_lengths'])
        assert len(r['scores']) == len(r['teacher_scores']) == len(record['sequence_ids'])
        assert all(math.isfinite(v) and v <= 0 for v in r['scores'] + r['teacher_scores'])
        assert math.isfinite(r['teacher_kl_sum']) and r['teacher_kl_sum'] >= 0
        for name, value in score_fields(record, r['scores']).items():
            assert r[name] == value
    assert len(by) == len(rows) == 324*6*5
    # Repeated static controls and teacher references must agree across all
    # seeds/conditions, rather than merely passing a row-count check.
    for task, sample in records:
        teacher = by[task, sample, 579, 10, 'matched']['teacher_scores']
        for k in [10, 12]:
            ref = by[task, sample, 579, k, 'static_joint_kl']
            for seed in [579, 580, 581]:
                static = by[task, sample, seed, k, 'static_joint_kl']
                for field in ['scores', 'teacher_kl_sum', 'target_tokens']:
                    assert static[field] == ref[field]
                for condition in conditions:
                    assert by[task, sample, seed, k, condition]['teacher_scores'] == teacher
    rng, summaries = np.random.default_rng(579), []
    for task in sorted({r['task'] for r in inputs}):
        local = sorted([r for r in inputs if r['task'] == task], key=lambda r: r['sample'])
        documents = np.asarray([r.get('document') if task == 'wikitext_conditional' else r['sample'] for r in local])
        _, inverse = np.unique(documents, return_inverse=True)
        units = int(inverse.max())+1
        draws = rng.multinomial(units, np.full(units, 1/units), size=2000)
        for k in [10, 12]:
            def values(condition, field):
                if condition == 'shuffled_mean3':
                    cs = conditions[1:4]
                else:
                    cs = [condition]
                return np.asarray([np.mean([by[task, r['sample'], seed, k, c][field]
                    for seed in [579, 580, 581] for c in cs]) for r in local])
            tokens = np.asarray([sum(r['target_lengths']) for r in local], dtype=float)
            nll_tokens = np.asarray([r['target_lengths'][0] for r in local], dtype=float)
            for metric in ['teacher_kl_per_token', 'task_metric']:
                field = 'teacher_kl_sum' if metric == 'teacher_kl_per_token' else 'nll_sum' if task in ['c4_development', 'wikitext_conditional'] else 'correct' if task == 'winogrande' else 'correct_norm'
                lengths = tokens if metric == 'teacher_kl_per_token' else nll_tokens if field == 'nll_sum' else np.ones(len(local))
                a, b, static = [values(c, field) for c in ['matched', 'shuffled_mean3', 'static_joint_kl']]
                # Positive means the input-matched path is better, for every
                # metric; retain its units explicitly.
                delta = a-b if field in ['correct', 'correct_norm'] else b-a
                totals, denominators = np.zeros(units), np.zeros(units)
                np.add.at(totals, inverse, delta); np.add.at(denominators, inverse, lengths)
                boots = (draws @ totals)/(draws @ denominators)
                lo, hi = np.quantile(boots, [.025, .975])
                summary = {'task': task, 'k': k, 'metric': metric, 'input_records': len(local), 'bootstrap_units': units,
                    'unit': 'article' if task == 'wikitext_conditional' else 'document_or_question',
                    'matched': float(a.sum()/lengths.sum()), 'shuffled': float(b.sum()/lengths.sum()),
                    'static': float(static.sum()/lengths.sum()), 'benefit_of_matching': float(delta.sum()/lengths.sum()),
                    'ci95_low': float(lo), 'ci95_high': float(hi), 'bootstrap_replicates': 2000,
                    'metric_units': 'accuracy_fraction' if field in ['correct', 'correct_norm'] else 'nll_per_token' if field == 'nll_sum' else 'kl_per_token'}
                if field == 'nll_sum':
                    summary.update(matched_ppl=math.exp(summary['matched']), shuffled_ppl=math.exp(summary['shuffled']), static_ppl=math.exp(summary['static']))
                summaries.append(summary)
            oracle, own = [], []
            for record in local:
                mins, matched = [], []
                for seed in [579, 580, 581]:
                    candidates = [by[task, record['sample'], seed, k, c]['teacher_kl_sum'] for c in conditions]
                    mins.append(min(candidates)); matched.append(candidates[0])
                oracle.append(np.mean(mins)); own.append(np.mean(matched))
            summaries.append({'task': task, 'k': k, 'metric': 'candidate_set_hindsight_diagnostic',
                'matched_kl': float(np.sum(own)/tokens.sum()), 'best_of_five_kl': float(np.sum(oracle)/tokens.sum()),
                'regret_kl': float((np.sum(own)-np.sum(oracle))/tokens.sum()),
                'limitation': 'Reads future target distributions; minimum only among recorded candidates; not a deployable or global oracle.'})
    audit = {'status': 'passed', 'records': 324, 'rows': len(rows), 'preserved_files': count,
        'identical_route_registries': 4, 'likelihood_probes': 40, 'independent_score_to_prediction_check': True,
        'policy_and_donor_mask_contracts': True, 'cpu_budget': 42}
    write_json(run / 'verification_v2.json', audit)
    write_json(run / 'paired_summary_v3.json', {'audit': audit, 'results': summaries,
        'aggregation': 'Average models and permutation repeats per input, then aggregate tokens; bootstrap questions/C4 documents or Wiki articles. Positive benefit favors matching. Exploratory unadjusted intervals.',
        'supersedes': 'Earlier V2 report used only training seed579 for the shuffled column; V3 uses all three model seeds and three permutations.'})
    text = ['# 输入与路径匹配：配对诊断与完整词表KL', '',
        '本表替代此前阶段12 V2初步汇总：该表的置换列只汇总了训练种子579，新表在每个输入上先平均三个训练种子与三次置换，再统计配对区间。GPU原始记录和checkpoint均未改变。此前探针生成的无版本报告只覆盖20个输入，不能用作正式结果。', '',
        '冻结m=6，K10/12。使用C4非checkpoint选择开发文档68条，PIQA/HellaSwag/WinoGrande/Wiki各64个预先固定输入；这些任务已经被审阅，属于开发诊断。正的匹配收益表示自身路径更好；95%区间未作多重比较校正。', '',
        '| 数据 | K | 自身路径KL | 置换路径KL | 静态KL | 匹配收益及95%区间 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in summaries:
        if r['metric'] == 'teacher_kl_per_token':
            text.append(f"| {r['task']} | {r['k']} | {r['matched']:.5f} | {r['shuffled']:.5f} | {r['static']:.5f} | {r['benefit_of_matching']:+.5f} [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] |")
    text += ['', '选择题KL覆盖所有候选目标token，不使用gold选择路径；每个输入按相同prompt选一次路径。C4/Wiki覆盖真实续写目标token。KL计算读取未来目标只用于离线评价，预测器仍只看prompt。', '',
        '| 数据 | K | 自身任务分数 | 置换任务分数 | 静态任务分数 | 匹配收益95%区间 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in summaries:
        if r['metric'] != 'task_metric': continue
        scale = 100 if r['metric_units'] == 'accuracy_fraction' else 1
        text.append(f"| {r['task']} | {r['k']} | {scale*r['matched']:.5f} | {scale*r['shuffled']:.5f} | {scale*r['static']:.5f} | {scale*r['benefit_of_matching']:+.5f} [{scale*r['ci95_low']:+.5f}, {scale*r['ci95_high']:+.5f}] |")
    text += ['', '选择题为准确率百分比，PIQA/HellaSwag用acc_norm、WinoGrande用acc；C4/Wiki为NLL/token，越低越好，PPL另存JSON。Wiki按文章聚类，目标token加权；模型种子与置换次数不当作独立样本。', '',
        '此前路径多样性重分析显示m=6路径确实变化，例如HellaSwag K10三种子有1154–2287种mask。因此“置换后分数近似”不能简单归因于所有样本走同一路径。若C4匹配改善而下游不改善，提示训练域/目标迁移问题；若两处都不改善，则优先检查路由训练和标签可预测性。不能只凭同一开发诊断选择新方法后再次将其称为独立证据。', '',
        'JSON另存五个已记录候选路径的未来目标最小KL与自身路径regret，仅是候选集事后诊断；不能称为全局最优，也不能据此断定prompt必然能预测最佳路径。', '',
        f"加强验收通过{len(rows)}条记录、324个输入、40个似然探针；四张卡完整路由注册表一致，复核checkpoint、前缀、置换双射、每条供体mask及从分数重算的预测。{count}个既有文件哈希保持一致。", '']
    report = root / 'analysis/STAGE12_ROUTE_FIDELITY_PAIRED_RESULTS_V3_20261110.zh-CN.md'
    with report.open('x') as f: f.write('\n'.join(text))
    return [report, run / 'verification_v2.json', run / 'paired_summary_v3.json']


def residency(root, run):
    count, allocation = preserve(run)
    config = json.loads((run / 'protocol.json').read_text())
    for p, digest in config['source_sha256'].items():
        assert sha256(p) == digest, p
    assert sha256(config['private_inputs']) == config['private_input_sha256']
    _, selected = policies(root)
    summaries, max_error = [], 0
    for rank in range(4):
        d = json.loads((run / f'worker_{rank:03d}.json').read_text())
        threads(d, rank, allocation)
        assert d['full_gpu_load_before_selective_measurements'] is False
        assert d['append_same_first128_route_unchanged'] is True
        assert d['policy'] == config['policies'][rank] and d['policy'] in selected
        assert len(d['records']) == 12
        for probe in d['cache_probes']:
            assert probe['mean_absolute_position_kl'] <= 1e-4 and probe['max_absolute_position_kl'] <= 1e-3
        for deployment in ['selective_cpu_bank', 'all_weights_resident_dynamic', 'full_model']:
            rows = [r for r in d['records'] if r['deployment'] == deployment]
            assert len(rows) == 4 and [r['repetition'] for r in rows] == [0,1,2,3]
            previous = set()
            for r in rows:
                k = 0 if deployment == 'full_model' else d['policy']['k']
                assert len(r['skipped_layers']) == k and r['skipped_layers'] == sorted(set(r['skipped_layers']))
                assert r['new_tokens'] == 32 and r['prompt_tokens'] == d['length']
                assert r['kv_bytes'] == 2*(40-k)*r['cached_tokens']*5120*2
                assert r['cached_tokens'] == d['length']+31
                assert math.isclose(r['total_seconds'], r['ttft_seconds']+r['decode_seconds'], abs_tol=1e-9)
                assert math.isclose(r['ttft_seconds'], r['route_seconds']+r['weight_switch_seconds']+r['prefill_seconds'], abs_tol=1e-9)
                assert r['model_cuda_parameter_bytes'] < r['peak_allocated_bytes'] <= r['peak_reserved_bytes']
                if deployment == 'selective_cpu_bank':
                    keep = set(range(40))-set(r['skipped_layers'])
                    assert r['resident_decoder_layers'] == 40-k
                    assert set(r['loaded_layers']) == keep-previous
                    assert set(r['evicted_layers']) == previous-keep
                    assert r['device_to_host_weight_bytes'] == 0
                    previous = keep
                if deployment == 'all_weights_resident_dynamic':
                    own = d['records'][r['repetition']]
                    assert own['deployment'] == 'selective_cpu_bank'
                    assert r['skipped_layers'] == own['skipped_layers'] and r['generated_tokens_sha256'] == own['generated_tokens_sha256']
                    assert r['kv_bytes'] == own['kv_bytes']
                    assert r['selective_max_logit_error'] <= 1e-4
                    max_error = max(max_error, r['selective_max_logit_error'])
            warm = rows[1:] if deployment == 'selective_cpu_bank' else rows
            summaries.append({'rank': rank, 'k': d['policy']['k'], 'seed': d['policy']['seed'], 'prompt_tokens': d['length'],
                'deployment': deployment, 'requests': 4, 'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in rows)/2**30,
                'peak_reserved_gib': max(r['peak_reserved_bytes'] for r in rows)/2**30,
                'first_request_ttft_seconds': rows[0]['ttft_seconds'],
                'warm_median_ttft_seconds': float(np.median([r['ttft_seconds'] for r in warm])),
                'warm_median_switch_seconds': float(np.median([r['weight_switch_seconds'] for r in warm])),
                'mean_decode_tokens_per_second': float(np.mean([r['decode_tokens_per_second'] for r in rows])),
                'warm_loaded_layer_counts': [len(r['loaded_layers']) for r in warm],
                'warm_host_to_device_weight_bytes': [r['host_to_device_weight_bytes'] for r in warm],
                'cpu_master_decoder_weight_gib': d['cpu_master_decoder_weight_bytes']/2**30,
                'cpu_load_and_shared_gpu_setup_seconds': d['model_cpu_load_and_shared_gpu_setup_seconds']})
    audit = {'status': 'passed', 'requests': 48, 'workers': 4, 'cpu_budget': 42,
        'preserved_files': count, 'matched_route_and_generated_hash_checks': 16,
        'max_prefill_logit_error': max_error, 'selective_cache_probes': 4,
        'startup_full_gpu_copy': False,
        'limits': ['Small engineering pilot, not final latency/P95 or real multi-turn benchmark.',
                   'Peak is PyTorch allocated/reserved, not total NVML peak.']}
    write_json(run / 'verification.json', audit)
    write_json(run / 'summary.json', {'audit': audit, 'measurements': summaries})
    text = ['# 按路由加载权重：低显存可行性实测', '',
        '已实现CPU完整权重库，GPU常驻embedding/position/shared head与预测器；路由后仅复制本轮实际执行的decoder block。初始化没有把完整模型放进GPU，启动与首轮的PyTorch峰值在任何全模型GPU对照之前记录。切换时先释放旧cache并驱逐不用的层，再加载新增层，共同层继续驻留。', '',
        '设置：FP16/SDPA、batch1、m=6、生成32token，每卡4个独立请求，提示词来自冻结训练输入；K10/12、输入128/512/1024。GPU0/2为K10，GPU1/3为K12。各设置不是完整交叉网格，不跨卡计算加速比。', '',
        '| GPU | K | 输入长 | 部署 | 峰值allocated GiB | 首次TTFT秒 | 热请求TTFT中位秒 | 热换层中位秒 | decode token/s |',
        '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        text.append(f"| {r['rank']} | {r['k']} | {r['prompt_tokens']} | {r['deployment']} | {r['peak_allocated_gib']:.3f} | {r['first_request_ttft_seconds']:.3f} | {r['warm_median_ttft_seconds']:.3f} | {r['warm_median_switch_seconds']:.3f} | {r['mean_decode_tokens_per_second']:.2f} |")
    text += ['', '低显存部署成立，但当前未经优化的CPU深拷贝/H2D换层明显增加TTFT，尤其首轮加载。decode阶段全部所需层已经在GPU，速度与常驻动态方案接近。首次TTFT从已在CPU加载好权重开始，CPU读模型与GPU共享模块准备另存JSON，不能把该时间省略后称作完整冷启动。GPU常驻对照的放置准备时间没有计入请求，因此它代表稳态常驻服务。', '',
        '每个输入的路由、prefill logits和生成token哈希与全权重常驻动态方案相同；cache探针通过既有FP16容差。单次请求结束即释放cache；本pilot没有运行真实对话调度，也没有证明多轮KV复用收益。重新读同一前128token时路径不变，已在四卡验证；为了让最新用户消息影响路由，需要另行设计和训练观测窗口。', '',
        '峰值仅是PyTorch allocated/reserved，非NVML整个进程峰值，尚未做稳定重复/P95、随机交错顺序、GPU带宽优化、并发或批处理。CPU保留全权重，系统总内存并未按相同比例减少；每worker的CPU decoder权重字节另存JSON。', '',
        '静态物理删层也能获得类似GPU权重占用，且路径不变时无逐轮搬运开销。最终实验必须加入这个强对照，分别列冷启动、热切换、TTFT、端到端吞吐、decode及NVML峰值，不能只用decode速度掩盖换层成本。', '',
        f"共验收48个请求、16组路径/生成哈希一致检查、4个cache探针；prefill logits最大误差{max_error:g}，{count}个既有文件哈希不变。运行目录：`{run.relative_to(root)}`。", '']
    report = root / 'analysis/STAGE12_SELECTIVE_RESIDENCY_PILOT_RESULTS_20261110.zh-CN.md'
    with report.open('x') as f: f.write('\n'.join(text))
    return [report, run / 'verification.json', run / 'summary.json', *sorted(run.glob('worker_*.json')),
            *[run / n for n in ['protocol.json','allocation.json','status.json','preservation_after.json']]]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--fidelity-run', type=Path, required=True)
    p.add_argument('--residency-run', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    paths = fidelity(root, a.fidelity_run.resolve()) + residency(root, a.residency_run.resolve())
    paths.append(Path(__file__).resolve())
    manifest = root / 'analysis/stage12_pilots_verified_upload_v1_20261110.json'
    write_json(manifest, [str(p.relative_to(root)) for p in paths])
    print(json.dumps({'status': 'audited', 'manifest': str(manifest)}), flush=True)
