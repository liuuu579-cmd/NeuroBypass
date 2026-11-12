"""Stage19 independent consistency checks, paired development summaries and append-only publication."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from adaptive_choice_core_v3 import PREFIX,SEEDS,load,summary_rows
from full_context_experiment_v1 import read_lines,write
from headroom_ruler_core_v1 import paired
from run_static_swap_pipeline_v1 import publish
from stage1_benchmark_v2 import sha256


def finalize_codebook(rows,inputs,cfg,out):
    index={(r['sample'],r['bank_name'],r['policy'],r['seed']):r for r in rows}
    assert len(index)==len(rows)
    sources={r['sample']:r for r in inputs};selection=set(cfg['selection_ids']);comparisons=[];references=[]
    costs=load(out/'private_costs.json')
    for name in cfg['names']:
        bank=load(out/f'{name}_bank.json');n=len(bank['routes'])
        # Recompute library construction from its original full TRAIN costs.
        from adaptive_choice_core_v3 import complementary_indices
        full=load(out/'private_source_costs.json')[str(bank['k'])]
        matrix=np.array([full[str(i)] for i in cfg['train_ids']]);chosen=complementary_indices(matrix,8)[:n]
        assert chosen==[r['source_route'] for r in bank['routes']]
        for cohort in ('other_development','c4_development','c4_prior_fresh','piqa','hellaswag','winogrande','wikitext_conditional'):
            ids=[r['sample'] for r in inputs if (r['task']=='c4_development' and r['source_sample'] not in selection) if cohort=='other_development'] if cohort=='other_development' else [r['sample'] for r in inputs if r['task']==cohort]
            if not ids:continue
            for arm in ('prompt','constant'):
                for metric in ('kl','nll','correct_norm'):
                    if metric not in index[ids[0],name,arm,SEEDS[0]]:continue
                    own=np.array([np.mean([index[i,name,arm,s][metric] for s in SEEDS]) for i in ids])
                    fixed=np.array([index[i,name,'fixed_best',None][metric] for i in ids])
                    delta=fixed-own if metric!='correct_norm' else own-fixed
                    clusters=[sources[i].get('document') if sources[i].get('document') is not None else i for i in ids]
                    comparisons.append({'bank_name':name,'cohort':cohort,'arm':arm,'metric':metric,
                        'mean':float(own.mean()),'static_mean':float(fixed.mean()),**paired(delta,clusters,190579)})
        for split,ids in [('train',cfg['train_ids']),('selection',cfg['selection_ids']),('other_development',[i for i in cfg['validation_ids'] if i not in selection])]:
            if not ids:
                references.append({'bank_name':name,'split':split,'documents':0,'empty_split':True})
                continue
            a=np.array([costs[name][str(i)] for i in ids]);references.append({'bank_name':name,'split':split,'documents':len(ids),
                'fixed_kl':float(a[:,0].mean()),'future_best_bank_kl':float(a.min(1).mean()),'future_targets_used':True})
    return {'aggregates':summary_rows(rows,inputs),'paired_comparisons':comparisons,'candidate_references':references,
        'limits':'Three seeds averaged within documents; exploratory repeated development, no untouched final-test or speed claim.'}


def finalize_budget(out,cfg,inputs):
    rows=[r for rank in range(3) for r in read_lines(out/f'private_quality_{rank:03d}.jsonl')]
    bank=load(out/'budget_bank.json');expected=len(inputs)*(len(bank['routes'])+len(SEEDS)*2*len(cfg['target_k'])*len(cfg['kl_tolerances']))
    assert len(rows)==expected,(len(rows),expected)
    index={(r['sample'],r['policy'],r['seed']):r for r in rows};assert len(index)==len(rows)
    checks=[];timings=[];private=[]
    for rank,seed in enumerate(SEEDS):
        receipt=load(out/f'timing_{rank:03d}.json');assert receipt['private_timing_sha256']==sha256(out/f'private_timing_{rank:03d}.json')
        data=load(out/f'private_timing_{rank:03d}.json')
        assert len(receipt['trials'])==len(data)
        for trial in data:
            assert len(trial['requests'])==trial['records']
            for item in trial['requests']:
                qseed=None if trial['policy'].startswith('static_k') else seed
                reference=index[item['sample'],trial['policy'],qseed]
                assert reference['k']==item['k'] and reference['index']==item['index']
                if 'correct_norm' in item:checks.append(item['correct_norm']==reference['correct_norm'])
            private.append({**trial,'worker_rank':rank,'training_seed':seed})
            timings.append({k:v for k,v in private[-1].items() if k!='requests'})
    # Cached and serial FP16 scoring may disagree on near-tied options. Report
    # disagreements explicitly and use timed protocol accuracy with timed wall.
    agreement={'mc_scoring_checks':len(checks),'cached_vs_serial_mc_disagreements':sum(not v for v in checks)}
    group=defaultdict(list)
    for t in timings:group[t['policy'],t['deployment'],t['generation_tokens_per_request'],t['records']].append(t)
    time_summary=[]
    for key,items in sorted(group.items(),key=lambda z:str(z[0])):
        row=dict(zip(('policy','deployment','generation_tokens_per_request','records'),key))
        for metric in ('dataset_wall_seconds','dataset_including_gpu_setup_seconds','gpu_layout_setup_seconds',
                       'mean_k','h2d_weight_bytes','peak_allocated_bytes','peak_reserved_bytes'):
            row[metric]=float(np.mean([r[metric] for r in items]))
        row['whole_dataset_trials']=len(items)
        if items[0]['accuracy_norm'] is not None:row['accuracy_norm']=float(np.mean([r['accuracy_norm'] for r in items]))
        row['min_seconds']=min(r['dataset_wall_seconds'] for r in items);row['max_seconds']=max(r['dataset_wall_seconds'] for r in items)
        time_summary.append(row)
    frontier=[]
    for r in time_summary:
        if r['generation_tokens_per_request']==0 and r['records']==96 and 'accuracy_norm' in r:frontier.append(r)
    comparisons=[]
    # Full pairwise frontiers; an accuracy allowance is a descriptive threshold,
    # not a non-inferiority conclusion chosen after observing the experiment.
    for dynamic in [r for r in frontier if not r['policy'].startswith('static_k')]:
        for static in [r for r in frontier if r['policy'].startswith('static_k')]:
            qdelta=dynamic['accuracy_norm']-static['accuracy_norm'];speed=static['dataset_wall_seconds']/dynamic['dataset_wall_seconds']
            question_diffs=[]
            sample_ids=[r['sample'] for r in inputs if r.get('gold') is not None]
            # Timed accuracy source is the cached evaluation protocol itself.
            for sample in sample_ids:
                dyn=[t for t in private if t['policy']==dynamic['policy'] and t['deployment']==dynamic['deployment'] and t['records']==96 and not t['generation_tokens_per_request']]
                sta=[t for t in private if t['policy']==static['policy'] and t['deployment']==static['deployment'] and t['records']==96 and not t['generation_tokens_per_request']]
                if not dyn or not sta:continue
                dv=np.mean([next(r['correct_norm'] for r in t['requests'] if r['sample']==sample) for t in dyn])
                sv=np.mean([next(r['correct_norm'] for r in t['requests'] if r['sample']==sample) for t in sta])
                question_diffs.append(dv-sv)
            comparison={'dynamic':dynamic['policy'],'deployment':dynamic['deployment'],'static':static['policy'],
                'accuracy_difference_pp':qdelta*100,'speedup':speed,
                'descriptive_within_accuracy_loss_pp':{str(v):bool(qdelta>=-v/100) for v in (0,1,2)},
                'warning':'Small96-question exploratory population; descriptive thresholds do not establish non-inferiority.'}
            if question_diffs:comparison['accuracy_paired']=paired(question_diffs,None,190580)
            comparisons.append(comparison)
    write(out/'summary.json',{'quality_aggregates':summary_rows(rows,inputs),'timing_aggregates':time_summary,
        'accuracy_time_frontier':comparisons,'scoring_consistency':agreement,
        'limits':cfg['limits'],'quality_population':'96 previously reviewedMC questions,32Wiki windows,132C4 documents. Mixed260 andMC96 wall times separate; generation16 separate. No task-accuracy claim from C4 KL calibration.'})
    write(out/'timing_protocol_checks.json',{'passed':True,'quality_rows':len(rows),'timing_trials':len(timings),
        'all_timed_routes_match_frozen_quality_policies':True,**agreement})


def publish_lane(root,out,cfg,lane):
    import os
    summary=load(out/'summary.json');report=root/'analysis'/f'{PREFIX}_{lane}_results.zh-CN.md'
    if lane=='codebook':
        lines=['# 阶段19路线1：小组合库提示词路由','',
            'K=16/20/24，每个K构造嵌套4/8个完整组合。训练256条、24轮、三个配对种子579/580/581。先选训练均值最优完整组合，再按训练集逐文档最小联合KL贪心加入互补组合。全文提示词输入，整轮固定路径；不把独立层损失相加。', '',
            '动态、常量观察、同库训练均值最强固定组合均比较。以下68条未参与本轮选检查点，但已用于历史开发；不是独立最终测试。正差值表示动态降低KL，区间按文档配对bootstrap，未校正多重比较。', '',
            '| 组合库 | 动态KL | 强静态KL | 改善 | 95%区间 |','|---|---:|---:|---:|---|']
        for r in summary['paired_comparisons']:
            if r['cohort']=='other_development' and r['arm']=='prompt' and r['metric']=='kl':
                lo,hi=r['ci95'];lines.append(f"| {r['bank_name']} | {r['mean']:.5f} | {r['static_mean']:.5f} | {r['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
        lines+=['','组合多样性、常量对照、任务正确率、KL/NLL与看到未来续写的候选参考均保存在汇总中。这里只回答小库选路是否有效，不宣称吞吐或显存收益。']
    else:
        lines=['# 阶段19路线5：提示词选择剪层预算与端到端数据集耗时','',
            '固定一条训练来源的联合贪心层序列，路由器从K=0/6/8/10/12/16/20/24中选择预算，每轮固定路径。全文embedding，训练256条、24轮、三个种子，配对常量观察对照。', '',
            '预测完整组合KL，按KL加执行层数代价选择K。原32条开发身份先选检查点，再为静态K10/12/16校准平均KL容差0/.02的操作点；静态方案可以作为回退。全部任务评测前冻结操作点，不用任务答案选K。KL容差不是准确率保证。', '',
            '## 96题MC数据集耗时与准确率','',
            'PIQA/HellaSwag/WinoGrande各32题，均为既有开发群体。计时包含输入搬运、全文路由、缓存prefill和候选续写评分。每个策略完整跑同一顺序96题，在各GPU上重复3次，报告全部重复均值。静态物理删层，动态常驻或按需；GPU布局准备另列，并提供包含准备的总耗时。不是自由生成吞吐或多并发服务指标。', '',
            '| 策略 | 部署 | 准确率% | 数据集秒数 | 含GPU准备秒数 | 平均K |','|---|---|---:|---:|---:|---:|']
        for r in summary['timing_aggregates']:
            if r['records']==96 and r['generation_tokens_per_request']==0:
                lines.append(f"| {r['policy']} | {r['deployment']} | {100*r['accuracy_norm']:.3f} | {r['dataset_wall_seconds']:.3f} | {r['dataset_including_gpu_setup_seconds']:.3f} | {r['mean_k']:.3f} |")
        lines+=['','完整静态质量—耗时曲线以及相对各静态K的速度比、准确率差和配对区间均公开。0/1/2个百分点仅是预先列出的描述性权衡档位，96题的小试不足以证明统计非劣效或论文创新性。', '',
            '## 混合数据集与生成检查','',
            '另外完整计时260条混合数据，包括96题MC、32个Wiki窗口和132条C4。MC准确率与其余群体的KL/NLL分别报告，不把所有260条称作准确率数据集。另有16个请求、每个固定64个生成token的两次重复；不因某模型提早EOS而缩短输出预算。按需部署计入逐请求换层搬运，任何新路径均重建完整上下文KV。', '',
            '公开显存为部署阶段allocated/reserved峰值；进程此前执行过全模型质量评测，因此不声称它是从启动起的低显存峰值。模型CPU检查点读取与tokenization为共同准备，不计入请求计时。最终计时等待另一条线GPU工作结束，未作多轮对话或并发优化。', '',
            f"缓存评分与串行评分的MC结果不一致次数：{summary['scoring_consistency']['cached_vs_serial_mc_disagreements']} / {summary['scoring_consistency']['mc_scoring_checks']}。耗时对应缓存协议的准确率；完整质量汇总保留串行协议。"]
    lines+=['',f"[协议](../{out.relative_to(root)}/protocol.json) · [汇总](../{out.relative_to(root)}/summary.json) · [验收](../{out.relative_to(root)}/verification.json)", '',
        '使用3张GPU和32个CPU核心，worker各10核、控制器2核。全部旧程序、数据、权重及报告保留，生成输出新增；私有token、逐题结果和权重不上传。']
    with report.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    files=[report]+[root/'tools'/n for n in cfg['source_sha256']]
    files += [p for p in out.glob('*.json') if not p.name.startswith('private_') and p.name not in ('preservation_before.json','allocation.json')]
    assert all(p.stat().st_size<2*1024*1024 for p in files)
    receipt=publish(root,files,f'{PREFIX}_{lane}',f'Pilot {lane} routing with paired controls and append-only evidence',
        {**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    write(out/'publication_verified.json',receipt)
