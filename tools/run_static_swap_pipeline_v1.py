"""Adopt audited probe, run finite static-swap pilot, audit and publish summaries."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

from audit_static_swap_v1 import audit,development_splits
from full_context_experiment_v1 import write


def load(path):return json.loads(path.read_text())


def publish(root,files,prefix,message,env):
    """A successful branch update may precede consistent readback; reverify safely."""
    upload=root/'analysis'/f'{prefix}_upload.json';write(upload,[str(p.relative_to(root)) for p in files])
    for attempt in range(1,4):
        receipt=root/'analysis'/f'{prefix}_publication_attempt{attempt}.json'
        result=subprocess.run([sys.executable,'-B',str(root/'tools/publish_additions_v7.py'),
            '--root',str(root),'--repo','liuuu579-cmd/NeuroBypass','--files',str(upload),
            '--message',message,'--receipt',str(receipt)],cwd=root,env=env)
        if result.returncode==0:return load(receipt)
    raise RuntimeError('Publication was not verified after3 safe append-only attempts; computation remains completed')


def report(root,run):
    summary=load(run/'summary.json');rows=summary['aggregates'];cfg=load(run/'protocol.json')
    def mean(task,policy,maximum,metric):
        values=[r[metric] for r in rows if r['task']==task and r['policy']==policy and r['maximum_exchanges']==maximum]
        return sum(values)/len(values)
    lines=['# 阶段17：静态组合起点的小范围动态换层','',
        '冻结OPT-13B，K=16，输入为完整提示词embedding。每轮回答从同一静态组合出发，最多做一或两次等量换层，回答过程中固定路径。恢复一层执行并跳过另一层，剪层总数不变。', '',
        '候选由800条旧训练文档的既有m0/m6专家路径与静态组合差异确定：16种原子交换，连同兼容的两次交换去重，共85个完整组合。候选名单不使用开发/测试；85个组合在800训练和100开发文档上全部重新计算32-token真实续写联合KL和NLL。两次交换的代价独立评分，不相加估计。不同交换配对产生同一组合时只评分一次，但保留所有有效的贪心转移。', '',
        '预测器采用训练平均相对KL先验，加完整提示词编码器预测的残差；零初始化残差头，smoothL1回归完整组合相对原静态的KL。推理仅使用提示词，不运行教师或真实KL；每步在当前组合和一步可达组合之间按预测代价贪心选择，允许停止、相同预测代价保留当前组合。训练覆盖所有可达第二步状态。', '',
        '两个配对种子579/580。动态提示词组与常量观察组参数量、初始化和有效更新预算相同。常量组不获得提示词内容或长度。原32开发身份仅用于epoch0/1/4/...24检查点选择，其他68也属已查看过的开发文档。另列训练均值下整个85组合库的最佳固定组合，避免把更强静态路径误当作动态收益。', '',
        '## 无恢复的质量比较','',
        '| 群体 | 指标 | 原静态 | 同候选库最强固定静态 | 旧m6 Router | 一次动态换层 | 两次动态换层 |','|---|---|---:|---:|---:|---:|---:|']
    for task in sorted({r['task'] for r in rows}):
        metrics=['kl']
        sample=next(r for r in rows if r['task']==task)
        if 'correct_norm' in sample:metrics.append('correct_norm')
        elif 'all_answers' in sample:metrics.append('all_answers')
        else:metrics.append('nll')
        for metric in metrics:
            scale=100 if metric in ('correct_norm','all_answers') else 1
            values=[mean(task,'original_static',0,metric),mean(task,'train_best_static',0,metric),
                mean(task,'old_router_m6',0,metric),mean(task,'swap_prompt',1,metric),mean(task,'swap_prompt',2,metric)]
            lines.append(f"| {task} | {metric} | "+' | '.join(f'{v*scale:.5f}' for v in values)+' |')
    lines+=['','## 两次动态换层相对同库最强固定静态的配对差异','',
        '正值表示动态改善。先在同一文档内平均两个种子，然后按文档/题目配对bootstrap；RULER按内容实例聚类。区间不作多重比较校正，不包含完整种子不确定性。', '',
        '| 群体 | 指标 | 改善 | 95%区间 |','|---|---|---:|---|']
    for r in summary['paired_comparisons']:
        if r['maximum_exchanges']==2 and r['baseline']=='train_best_static' and r['metric'] in ('kl','correct_norm','all_answers'):
            scale=100 if r['metric'] in ('correct_norm','all_answers') else 1
            lo,hi=[v*scale for v in r['ci95']]
            lines.append(f"| {r['task']} | {r['metric']} | {r['difference']*scale:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines+=['','## 开发分组及解释边界','',
        '全部质量群体此前已评测过，不宣称新的独立测试。C4_prior_fresh为Stage15既有留出群体32条；MC/Wiki每任务固定64条；RULER沿用Stage16的72题和48-token补全，只是开发复核，不更改模板。严格格式分数有额外生成内容的已知限制，不能据它声称优于完整模型。', '',
        '只做路径学习，没有联合恢复训练，不代表完整恢复效果；候选库受训练分布和16对交换限制，未覆盖所有可能的一次或两次交换。离线候选库最优代价看到真实续写，仅为诊断参考；不能作为线上性能。', '',
        '完整统计含原静态、训练均值最强静态、训练均值贪心静态、旧Router、动态和常量观察组，实际换层次数及不同路径数量。若动态不能超过同库强静态，不以路径多样性宣称动态优势。未新增吞吐量或部署显存测量。', '',
        '[汇总](../'+str(run.relative_to(root))+'/summary.json) · [协议](../'+str(run.relative_to(root))+'/protocol.json) · [独立审计](../'+str(run.relative_to(root))+'/independent_audit.json) · [选模32与其他68分列](../'+str(run.relative_to(root))+'/development_splits.json)', '',
        '使用4张RTX6000Ada、42个CPU核心；每worker10核、控制器2核。全部既有文件保全，公开源码、协议、审计与聚合结果，权重、token数组、逐题路由和生成留在本地。']
    destination=root/'analysis'/f'STAGE17_STATIC_SWAP_RESULTS_{run.name}.zh-CN.md'
    with destination.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe-run',type=Path,required=True);p.add_argument('--main-run',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];probe=a.probe_run.resolve();run=a.main_run.resolve()
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[40:42])
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    completion=root/'analysis'/f'{run.name}_completion.json';assert not run.exists() and not completion.exists()
    phase,state,error='probe_audit','failed',None
    try:
        result=audit(root,probe,True)
        assert result['passed'];print(json.dumps({'event':'swap_probe_verified',**result}),flush=True)
        phase='main'
        subprocess.run([sys.executable,'-B','-u',str(root/'tools/run_with_resources_v2.py'),'--gpus','0,1,2,3',
            '--run-dir',str(run),'--',sys.executable,'-B','-u',str(root/'tools/run_static_swap_four_gpu_v2.py')],
            cwd=root,env=env,check=True,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
        phase='main_audit';result=audit(root,run,False);write(run/'independent_audit.json',result)
        write(run/'development_splits.json',development_splits(run))
        phase='report';destination=report(root,run)
        files=[destination,Path(__file__),*[root/'tools'/n for n in ('static_swap_core_v2.py','static_swap_experiment_v2.py','run_static_swap_four_gpu_v2.py','audit_static_swap_v1.py')],
            *[run/n for n in ('summary.json','verification.json','independent_audit.json','protocol.json','candidate_bank.json',
                'quality_manifest.json','static_controls.json','cpu_checks.json','development_splits.json','preservation_after.json','status.json')],
            *[run/f'{kind}_{i:03d}.json' for i in range(4) for kind in ('engineering','fit_result','label_result','selection_replay','worker_result')],
            probe/'independent_audit.json']
        assert all(not f.name.startswith('private_') and f.stat().st_size<1024*1024 for f in files)
        phase='publication';receipt=publish(root,files,run.name,'Report static-start prompt-conditioned joint exchange router against equally optimized fixed masks',env)
        print(json.dumps({'event':'swap_results_published',**receipt}),flush=True)
        state='completed'
    except BaseException as e:
        error=f'{type(e).__name__}: {e}'
        import traceback;traceback.print_exc()
    result={'state':state,'phase':phase,'error':error,'run':str(run),'finished_utc':datetime.now(timezone.utc).isoformat()}
    write(completion,result);print(json.dumps(result),flush=True)
    return 0 if state=='completed' else 1


if __name__=='__main__':sys.exit(main())
