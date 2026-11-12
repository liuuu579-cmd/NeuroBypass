"""Finite K16/20/24 deep-exchange pilot; engineering gates, audits and append-only publication."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_deep_swap_v1 import audit, development_splits
from full_context_experiment_v1 import read_lines, write
from headroom_ruler_core_v1 import paired
from run_static_swap_pipeline_v1 import publish
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def combined(root, runs, prefix):
    result={'runs':[str(p.relative_to(root)) for p in runs], 'budgets':[16,20,24],
        'training_documents':256,'epochs':12,'seeds':[579,580], 'cohorts':[],
        'limits':'Exploratory previously reviewed development cohorts; two seeds, no recovery or deployment measurements. CIs do not include all training-seed uncertainty or multiple-comparison correction.'}
    lines=['# 阶段18：加深剪枝后的小范围动态换层小试','',
        '冻结40层OPT-13B，全文提示词路由，每轮回答固定路径，K=16/20/24。每步恢复一层并新增跳过一层，最多两次，联合评分全部完整候选组合。', '',
        '三种K使用完全相同的256条训练文档、两个配对种子579/580和12轮训练。256条从原800训练文档按提示词长度等间隔选取；静态起点及16对交换候选名单沿用原800条训练来源，不使用开发答案建库。候选均值、预测器和新的固定组合只在256条上训练/选择。K16与旧候选库逐项一致后复用其已核验标签和embedding，再按本轮预算重新训练与评测，不与旧800条训练结果混作同预算对照。', '',
        '普通动态采用训练均值先验加全文编码残差，按完整组合预测代价逐步贪心，可随时停止。另保留阈值版：只有预测改善大于固定阈值才换层。阈值在选定普通动态检查点后，仅在原32条开发身份上从0/.05/.1/.2/.4/.8/1.6倍训练代价标准差及始终静态中选择；它不是概率置信度，也不保证其他输入不退化。常量观察组使用相同结构、初始化、训练预算与阈值规则。', '',
        '## 68条其他开发文档：核心比较','',
        '这些文档没有用于本轮选检查点或阈值，但此前已反复查看，不能称为新的独立测试。动态列先在每条文档内平均两个种子。区间为按文档配对bootstrap的未校正95%区间，正差值表示动态优于训练均值最强固定组合。', '',
        '| K | 方法 | KL | 相对同库强静态改善 | 95%区间 | 平均实际交换数 | 不同路径数（各种子） |',
        '|---:|---|---:|---:|---|---:|---|']
    for run in runs:
        cfg=load(run/'protocol.json');k=cfg['k'];bank=load(run/'candidate_bank.json')
        rows=[r for rank in range(4) for r in read_lines(run/f'private_quality_{rank:03d}.jsonl')]
        inputs={r['sample']:r for r in read_lines(run/'private_quality_inputs.jsonl')}
        selection=set(cfg['selection_ids'])
        tasks=sorted({r['task'] for r in rows})
        cohorts=[('other_development',[i for i,r in inputs.items() if r['task']=='c4_development' and r['source_sample'] not in selection])]
        cohorts += [(task,[i for i,r in inputs.items() if r['task']==task]) for task in tasks]
        for cohort,samples in cohorts:
            idx={(r['sample'],r['policy'],r['seed'],r['maximum_exchanges']):r for r in rows if r['sample'] in samples}
            base=np.array([idx[i,'train_best_static',None,0]['kl'] for i in samples])
            for policy,maximum in [('original_static',0),('train_best_static',0),('swap_constant',2),
                                   ('swap_prompt',1),('swap_prompt',2),('swap_constant_gated',2),('swap_prompt_gated',2),('old_router_m6',0)]:
                seeded=policy not in ('original_static','train_best_static')
                seeds=cfg['seeds'] if seeded else [None]
                values=np.array([[idx[i,policy,seed,maximum]['kl'] for seed in seeds] for i in samples]).mean(1)
                swaps=float(np.mean([idx[i,policy,s,maximum]['actual_exchanges'] for i in samples for s in seeds]))
                masks=[len({tuple(idx[i,policy,s,maximum]['skips']) for i in samples}) for s in seeds]
                # Wiki can contain multiple windows from the same article.
                clusters=[inputs[i].get('document') if inputs[i].get('document') is not None else i for i in samples]
                comparison=paired((base-values).tolist(),clusters,180579)
                row={'k':k,'cohort':cohort,'policy':policy,'maximum_exchanges':maximum,
                    'documents_or_questions':len(samples),'kl':float(values.mean()),
                    'actual_exchanges':swaps,'unique_masks_by_seed':masks,**comparison}
                for metric in ('nll','correct_norm'):
                    if metric in idx[samples[0],policy,seeds[0],maximum]:
                        row[metric]=float(np.mean([idx[i,policy,s,maximum][metric] for i in samples for s in seeds]))
                result['cohorts'].append(row)
                if cohort=='other_development':
                    lo,hi=comparison['ci95']
                    label=policy+(f'（最多{maximum}次）' if maximum else '')
                    lines.append(f"| {k} | {label} | {row['kl']:.5f} | {comparison['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {swaps:.3f} | {masks} |")
        result.setdefault('candidate_banks',[]).append({'k':k,'joint_masks':len(bank['routes'])})
    lines+=['','## 既有任务群体小规模复核','',
        'PIQA/HellaSwag/WinoGrande/Wiki每组固定32条，C4既有新来源群体32条；所有K复用同一批身份。只做探索性检查，不扩大RULER，也不宣称正式横评或吞吐/显存优势。下列动态均为两个种子均值。', '',
        '| K | 群体 | 指标 | 同库强静态 | 普通两次动态 | 阈值两次动态 |', '|---:|---|---|---:|---:|---:|']
    for k in (16,20,24):
        for task in ('c4_prior_fresh','piqa','hellaswag','winogrande','wikitext_conditional'):
            metric='correct_norm' if task in ('piqa','hellaswag','winogrande') else 'nll'
            rows=[]
            for policy in ('train_best_static','swap_prompt','swap_prompt_gated'):
                rows.append(next(r for r in result['cohorts'] if r['k']==k and r['cohort']==task and r['policy']==policy and (policy=='train_best_static' or r['maximum_exchanges']==2)))
            factor=100 if metric=='correct_norm' else 1
            lines.append(f'| {k} | {task} | {metric} | '+' | '.join(f'{r[metric]*factor:.5f}' for r in rows)+' |')
    lines+=['','## 候选收益与边界','',
        '逐输入候选最小KL及贪心参考读取真实未来续写，只用于区分候选收益空间与已学路由，不是线上质量或全局最优保证。完整分组KL/NLL/PPL、常量观察组、原静态、训练均值贪心静态、旧m6路由及阈值记录保留在每个运行目录中。不能把路径多样性、选模集收益、某单一种子或退化到始终静态当作动态优势。', '',
        '所有GPU工作使用0–3四张RTX6000Ada，42个CPU核心；worker各10核，控制器2核。工程检查覆盖候选代价与直接执行重放、真实跳层、提示词长度和KV cache一致性。旧文件哈希逐项核验，全部生成路径新增。仅公开源码、协议、审计及聚合结果，权重、token和逐样本数据留本地。', '']
    for run in runs:
        lines.append(f"- [{run.name}](../{run.relative_to(root)}/summary.json) · [协议](../{run.relative_to(root)}/protocol.json) · [审计](../{run.relative_to(root)}/independent_audit.json)")
    summary=root/'analysis'/f'{prefix}_combined.json';write(summary,result)
    report=root/'analysis'/f'STAGE18_DEEP_SWAP_PILOT_{prefix}.zh-CN.md'
    with report.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    return summary,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args();prefix=args.prefix
    assert '/' not in prefix and prefix.startswith('stage18_')
    root=Path(__file__).resolve().parents[1]
    cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=42
    os.sched_setaffinity(0,cpus[40:42])
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    completion=root/'analysis'/f'{prefix}_completion.json';assert not completion.exists()
    names=('deep_swap_core_v1.py','deep_swap_experiment_v1.py','run_deep_swap_four_gpu_v1.py','audit_deep_swap_v1.py','run_deep_swap_pipeline_v1.py')
    write(root/'analysis'/f'{prefix}_source_manifest.json',{n:sha256(root/'tools'/n) for n in names})
    runs=[];probes=[];phase='engineering';state='failed';error=None
    try:
        plan=[(True,24),(True,20),(False,16),(False,20),(False,24)]
        for probe,k in plan:
            run=root/'runs'/f'{prefix}_{"probe" if probe else "main"}_k{k:02d}'
            assert not run.exists()
            command=[sys.executable,'-B','-u',str(root/'tools/run_with_resources_v2.py'),'--gpus','0,1,2,3',
                '--run-dir',str(run),'--',sys.executable,'-B','-u',str(root/'tools/run_deep_swap_four_gpu_v1.py'),'--k',str(k)]
            if probe:command.append('--probe')
            phase=f'{"probe" if probe else "main"}_k{k}'
            print(json.dumps({'event':'deep_swap_phase','phase':phase,'run':str(run)}),flush=True)
            subprocess.run(command,cwd=root,env=env,check=True,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
            phase=f'audit_{phase}'
            checked=audit(root,run,probe);write(run/'independent_audit.json',checked)
            write(run/'development_splits.json',development_splits(run))
            print(json.dumps({'event':'deep_swap_audited',**checked}),flush=True)
            (probes if probe else runs).append(run)
        phase='combined_report';summary,report=combined(root,runs,prefix)
        configs=[load(run/'protocol.json') for run in runs]
        assert all(c['train_ids']==configs[0]['train_ids'] and c['validation_ids']==configs[0]['validation_ids'] for c in configs)
        manifests=[load(run/'quality_manifest.json') for run in runs]
        assert len({m['private_input_sha256'] for m in manifests})==1
        files=[root/'tools'/n for n in names]+[summary,report,root/'analysis'/f'{prefix}_source_manifest.json']
        for run in runs:
            files += [run/n for n in ('summary.json','verification.json','independent_audit.json','protocol.json','candidate_bank.json',
                'quality_manifest.json','static_controls.json','cpu_checks.json','development_splits.json','preservation_after.json','status.json')]
            files += [run/f'{kind}_{rank:03d}.json' for rank in range(4)
                for kind in ('engineering','fit_result','label_result','selection_replay','worker_result','gate')]
        files += [run/'independent_audit.json' for run in probes]
        assert all(not p.name.startswith('private_') and p.stat().st_size<1024*1024 for p in files)
        phase='publication'
        receipt=publish(root,files,prefix,'Pilot deeper static-start exchange routing at K16/20/24 with fixed-gain fallback and matched controls',env)
        print(json.dumps({'event':'deep_swap_published',**receipt}),flush=True)
        state='completed'
    except BaseException as exc:
        import traceback
        error=f'{type(exc).__name__}: {exc}';traceback.print_exc()
    write(completion,{'state':state,'phase':phase,'error':error,'main_runs':[str(p) for p in runs],
        'probe_runs':[str(p) for p in probes],'finished_utc':datetime.now(timezone.utc).isoformat()})
    return 0 if state=='completed' else 1


if __name__=='__main__':
    sys.exit(main())
