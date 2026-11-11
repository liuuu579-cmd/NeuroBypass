"""Recovery v4: audit forced-overflow retries, finite four-arm rerun, publish evidence."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(Path(path).read_text())


def audit(run,probe):
    root = Path(__file__).resolve().parents[1]
    config = load(run/'protocol.json'); manifest = load(run/'input_manifest.json')
    status = load(run/'status.json'); verification = load(run/'verification.json')
    assert status['state']=='completed' and config['probe']==probe and verification['passed']
    assert sha256(run/'input_manifest.json')==config['input_manifest_sha256']
    assert all(sha256(root/'tools'/n)==h for n,h in config['source_sha256'].items())
    assert all(sha256(run/n)==h for n,h in manifest['new_private_input_sha256'].items())
    assert all(sha256(p['checkpoint']['path'])==p['checkpoint']['sha256'] for p in config['router_policies'])
    original = load(run/'preservation_before.json')
    assert all(sha256(name)==h for name,h in original.items())
    assert load(run/'preservation_after.json')['preserved_files']==len(original)
    assert load(run/'allocation.json')['cpu_count']==42
    assert config['lr']==5e-5 and config['deterministic_algorithms']
    for rank in range(4):
        assert load(run/f'engineering_{rank:03d}.json')['passed']
        result = load(run/f'worker_result_{rank:03d}.json')
        assert result['passed'] and result['skipped_updates']==0
        assert result['adam_steps']==[result['updates']]
        updates = read_lines(run/f'updates_{rank:03d}.log')
        successful = [r for r in updates if r['kind']=='successful_update']
        retried = [r for r in updates if r['kind']=='overflow_retry']
        assert len(successful)==result['updates']
        assert len(retried)==result['retry_attempts']
        for attempt in retried:
            final = next(r for r in successful if r['update']==attempt['update'])
            assert final['batch_ids']==attempt['batch_ids'] and not attempt['optimizer_advanced']
        if probe:
            forced = load(run/f'forced_retry_{rank:03d}.json')
            assert forced['passed'] and forced['retry_attempts']>0
            assert forced['successful_updates']==1
        else:
            assert load(run/f'baseline_replay_{rank:03d}.json')['passed']
        assert result['adapter_replay_max_abs_error']==result['teacher_restoration_max_abs_error']==0
        assert result['trainable_parameters']==3932160
        assert result['quality_sha256']==sha256(run/f'quality_{rank:03d}.jsonl')
        assert result['frozen_routes_sha256']==sha256(run/f'private_routes_{rank:03d}.json')
        assert result['best']['sha256']==sha256(result['best']['path'])
    if not probe:
        assert manifest['train_documents']==800 and manifest['selection_documents']==32
        assert manifest['development_documents']==100 and manifest['fresh_documents']==128
        assert manifest['fresh_length_histogram']=={'1536':32,'1024':32,'512':32,'256':32}
        assert config['epochs']==8 and config['effective_batch']==8
    fresh = read_lines(run/'private_fresh_inputs.jsonl')
    old13 = load(Path(config['data_run'])/'data_manifest.json')['records']
    old14 = read_lines(root/'runs/stage14_observation_windows_20261111_01/private_inputs.jsonl')
    old_docs = {r['record_sha256'] for r in old13}|{r['document_sha256'] for r in old14}
    assert not old_docs&{r['document_sha256'] for r in fresh}
    assert len({r['document_sha256'] for r in fresh})==len(fresh)
    # Report the unselected portion of development separately from selection32.
    selection = set(manifest['selection_ids']); splits = defaultdict(list)
    full_replicates = defaultdict(list)
    for rank in range(4):
        for row in read_lines(run/f'quality_{rank:03d}.jsonl'):
            if row['task']=='development' and row['policy']=='pruned':
                key = ('selection' if row['sample'] in selection else 'nonselection',row['arm'],row['seed'],row['stage'])
                splits[key].append(row)
            if row['policy']=='full' and 'nll' in row:
                full_replicates[row['task'],row['sample']].append(row['nll'])
    full_error = max((max(v)-min(v) for v in full_replicates.values()),default=0)
    assert full_error <= 1e-5
    split_metrics = [{'split':k[0],'arm':k[1],'seed':k[2],'stage':k[3],'documents':len(v),
        'nll':sum(r['nll'] for r in v)/len(v),'kl':sum(r['kl'] for r in v)/len(v)} for k,v in sorted(splits.items())]
    return {**verification,'state':'verified','preserved_files':len(original),
        'fresh_documents_disjoint_from_stage13_and14':True,'full_teacher_replica_max_nll_error':full_error,
        'development_splits':split_metrics}


def report(path,run,result,probe_result):
    cfg = load(run/'protocol.json'); summary = load(run/'summary.json')
    rows = summary['aggregates']
    def value(task,arm,stage,policy,metric):
        group = [r[metric] for r in rows if (r['task'],r['arm'],r['stage'],r['policy'])==(task,arm,stage,policy)]
        assert len(group)==2
        return sum(group)/len(group)
    lines = ['# 阶段15：K=16 深剪后质量恢复小试','',
        '继续使用完整提示词路由。冻结 OPT-13B、完整提示词路由器和每条输入的剪层路径，先单独训练恢复适配器。动态组保留 m=6 协同贪心路由；静态组使用同一条全局联合 KL 贪心路径。两组都剪16层、执行24层，使用两个配对种子579/580。', '',
        '这是等容量的 executed-depth-slot LoRA 小试：24个执行位置各有 q/v rank8、alpha16 的适配器，一套参数跨动态路径共享，按保留层的执行顺序绑定位置；静态组使用同样结构。每组总可训练参数和每条输入启用的参数均为3,932,160。该参数绑定方式与按原始层号独立配置 LoRA 不同，不能直接当作标准 LoRA 的结论。', '',
        '旧v1在第4轮因静态seed579梯度溢出而中止，没有完整恢复测评。v4四组均从相同规则的初始参数重训，学习率统一从2e-4降至5e-5；发生非有限梯度时降低loss scale并重算同一batch，失败尝试不推进Adam、学习率或数据顺序。启用PyTorch确定性算法及确定性cuBLAS/注意力反向传播；工程小试强制触发真实FP16溢出，核验失败尝试不改变权重及Adam状态，重试更新与直接有限scale更新参数误差为0。', '',
        '训练800条旧训练文档，每条32个真实续写目标，8轮、有效batch8、每组800次有效更新和204,800个有效目标token。CE + 完整冻结模型的前向KL，权重均为1；AdamW5e-5，5%预热与余弦衰减。每个配对种子的初始化、数据顺序和预算相同。仅用原32条开发身份的 CE+KL 选最佳epoch；不更新路由器，不重新生成剪层标签。恢复前评分在验证输入、路径和数值重放一致后复用v1已完成的基线，恢复后评分全部新算。', '',
        '## 恢复前后','',
        '表中为两个种子的均值。全模型不参与恢复训练。MC/Wiki 使用事先固定的已有基准子集（PIQA/HellaSwag/Wino各256条，Wiki128个窗口），属于已查看过的基准群体；新C4评测128个文档、四种长度各32条，对 Stage13/14 完整文档、开头和token前缀去重。未知预训练及更早不可追溯数据重叠，近重复仍可能存在。', '',
        '| 评测 | 指标 | 全模型 | 动态恢复前 | 动态恢复后 | 静态恢复前 | 静态恢复后 |',
        '|---|---|---:|---:|---:|---:|---:|']
    tasks = [('fresh_c4','kl','新C4'),('fresh_c4','nll','新C4'),('wikitext_conditional','ppl','Wiki'),
        ('hellaswag','accuracy','HellaSwag'),('piqa','accuracy','PIQA'),('winogrande','accuracy','WinoGrande')]
    for task,metric,label in tasks:
        vals = [value(task,'dynamic','before','full',metric),
            value(task,'dynamic','before','pruned',metric),value(task,'dynamic','after','pruned',metric),
            value(task,'static','before','pruned',metric),value(task,'static','after','pruned',metric)]
        cells = ' | '.join(f'{v:.2%}' if metric=='accuracy' else f'{v:.5f}' for v in vals)
        lines.append(f'| {label} | {metric} | {cells} |')
    lines += ['', '## 配对改善及不确定性','',
        '正值表示右侧条件改善质量；KL/NLL/PPL取左减右，准确率取右减左。输入内先平均两个种子，按文档/题目配对bootstrap1000次，Wiki按文章重采样。区间不包含训练种子不确定性，未做多重比较校正，不能据此宣布最终论文优势。', '',
        '| 评测 | 比较 | 指标 | 改善 | 95%区间 |','|---|---|---|---:|---|']
    names = {'dynamic_recovery':'动态恢复后 vs 恢复前','static_recovery':'静态恢复后 vs 恢复前',
        'recovered_dynamic_vs_static':'恢复后动态 vs 静态'}
    for row in summary['comparisons']:
        if row['task']=='development': continue
        lo,hi = row['ci95']
        lines.append(f"| {row['task']} | {names[row['comparison']]} | {row['metric']} | {row['improvement']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', '## 开发数据与工程审计','',
        '全部100条开发数据含选模32条，不能作为独立测试。以下另列未用于本次选epoch的68条旧开发数据；它们仍是此前已查看过的开发数据。', '',
        '| 组别 | 阶段 | KL | NLL |','|---|---|---:|---:|']
    for arm in ('dynamic','static'):
        for stage in ('before','after'):
            group = [r for r in result['development_splits'] if (r['split'],r['arm'],r['stage'])==('nonselection',arm,stage)]
            if group:
                lines.append(f"| {arm} | {stage} | {sum(r['kl'] for r in group)/len(group):.5f} | {sum(r['nll'] for r in group)/len(group):.5f} |")
    lines += ['', '| 组别/种子 | 最佳epoch | 选模KL | 训练与选模秒数 | 记录的峰值allocated GiB |',
        '|---|---:|---:|---:|---:|']
    for rank in range(4):
        r = load(run/f'worker_result_{rank:03d}.json')
        lines.append(f"| {r['arm']}/{r['seed']} | {r['best']['epoch']} | {r['best']['selection_kl']:.5f} | {r['seconds_training_and_selection']:.1f} | {r['peak_training_allocated_bytes']/2**30:.2f} |")
    lines += ['', '实际训练溢出重试次数：' + '，'.join(f"worker{i}={load(run/f'worker_result_{i:03d}.json')['retry_attempts']}" for i in range(4)) + '。重试产生额外计算，计入实测耗时；有效更新、数据与目标预算相同。', '',
        '峰值计数在训练前重置、训练后评测结束时读取，包含选模及恢复后评测，不能作为部署显存或压缩收益。没有重新测恢复后的吞吐量。运行使用4张RTX6000Ada、42个CPU核心，每个工作进程10核、控制器2核。', '',
        f"工程小试与正式运行均通过：零初始化适配器等于原剪层模型、24个执行位置均有有限梯度、原参数和路由器冻结、跳过原层不执行、保存载入复现误差为0、关闭适配器回到原教师误差为0、恢复前后路径完全一致、配对组数据顺序相同、无跳过的优化器更新。另有随机小OPT的checkpoint梯度一致性与KV缓存续写一致性检查。正式运行核验{result['preserved_files']}个既有文件哈希不变。", '',
        '该实验回答固定路由下恢复训练是否有效；不等于完整恢复到全模型，也不证明输入条件路由优于静态。后续是否扩展到更多K、标准原层LoRA或交替优化路由，应依据这里的质量与不确定性决定。', '',
        f"[完整汇总](../runs/{run.name}/summary.json) · [协议](../runs/{run.name}/protocol.json) · [独立审计](../runs/{run.name}/independent_audit.json)"]
    with path.open('x') as stream: stream.write('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run',type=Path,required=True)
    parser.add_argument('--main-run',type=Path,required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; analysis = root/'analysis'
    probe,run = args.probe_run.resolve(),args.main_run.resolve(); py = sys.executable
    all_cpus = sorted(os.sched_getaffinity(0))
    # Parent uses2 cores; resource launcher restores its own42-core allocation.
    os.sched_setaffinity(0,all_cpus[40:42])
    env = {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    completion = analysis/f'{run.name}_completion.json'
    if run.exists() or completion.exists(): raise FileExistsError('Use new main and completion paths')
    phase,state,error = 'probe','failed',None
    try:
        deadline = time.monotonic()+1800
        while not (probe/'status.json').exists():
            if time.monotonic()>deadline: raise TimeoutError('Engineering probe deadline exceeded')
            time.sleep(3)
        probe_result = audit(probe,True)
        print(json.dumps({'event':'recovery_probe_verified',**probe_result}),flush=True)
        phase = 'main'
        subprocess.run([py,'-B','-u',str(root/'tools/run_with_resources_v2.py'),'--gpus','0,1,2,3','--run-dir',str(run),'--',
            py,'-B','-u',str(root/'tools/train_depth_slot_recovery_four_gpu_v4.py')],cwd=root,env=env,check=True,
            preexec_fn=lambda:os.sched_setaffinity(0,all_cpus))
        phase = 'audit'
        result = audit(run,False); write(run/'independent_audit.json',result)
        phase = 'report'
        destination = analysis/'STAGE15_DEPTH_SLOT_RECOVERY_V4_RESULTS_20261111.zh-CN.md'
        report(destination,run,result,probe_result)
        files = [destination,Path(__file__),*[root/'tools'/n for n in (
            'recovery_optimizer_v3.py','recovery_baseline_v2.py','depth_slot_recovery_core_v1.py','depth_slot_recovery_data_v1.py',
            'train_depth_slot_recovery_four_gpu_v4.py','check_depth_slot_recovery_v1.py')],
            analysis/'stage15_recovery_toy_checks_20261111_01.json',
            analysis/'stage15_recovery_failure_audit_20261111_01.json',
            *[run/n for n in ('summary.json','verification.json','independent_audit.json','protocol.json','input_manifest.json','status.json','preservation_after.json')],
            *[run/f'worker_result_{i:03d}.json' for i in range(4)],
            *[probe/n for n in ('verification.json',*[f'engineering_{i:03d}.json' for i in range(4)],*[f'forced_retry_{i:03d}.json' for i in range(4)])],
            *[run/f'baseline_replay_{i:03d}.json' for i in range(4)]]
        assert all(not p.name.startswith('private_') and p.stat().st_size<1024*1024 for p in files)
        upload = analysis/f'{run.name}_upload.json'
        write(upload,[str(p.relative_to(root)) for p in files])
        phase = 'publication'
        subprocess.run([py,'-B',str(root/'tools/publish_additions_v7.py'),'--root',str(root),'--repo','liuuu579-cmd/NeuroBypass',
            '--files',str(upload),'--message','Complete recovery v4 with deterministic gradients and exact same-batch overflow retries and paired quality evaluation',
            '--receipt',str(analysis/f'{run.name}_publication.json')],cwd=root,env=env,check=True)
        state = 'completed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    result = {'state':state,'phase':phase,'error':error,'run':str(run),'finished_utc':datetime.now(timezone.utc).isoformat()}
    write(completion,result); print(json.dumps(result),flush=True)
    return 0 if state=='completed' else 1


if __name__=='__main__': sys.exit(main())
