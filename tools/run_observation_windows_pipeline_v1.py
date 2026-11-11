"""Gate long-prompt observation evaluation on probe, audit, and publish summaries."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from full_context_experiment_v1 import write
from stage1_benchmark_v2 import sha256


def load(path):
    return json.loads(path.read_text())


def audit(run, probe):
    cfg, verification = load(run/'protocol.json'), load(run/'verification.json')
    status = load(run/'status.json')
    if status['state'] != 'completed' or cfg['probe'] != probe or not verification['passed']:
        raise ValueError('Evaluation did not pass: '+str(run))
    root = Path(__file__).resolve().parents[1]
    if any(sha256(root/'tools'/n) != h for n,h in cfg['source_sha256'].items()):
        raise ValueError('Source changed')
    if sha256(run/'input_manifest.json') != cfg['input_manifest_sha256']:
        raise ValueError('Manifest changed')
    manifest = load(run/'input_manifest.json')
    if sha256(run/'private_inputs.jsonl') != manifest['private_inputs_sha256']:
        raise ValueError('Private input changed')
    assert verification['documents'] == manifest['documents']
    assert verification['score_rows'] == verification['documents']*45
    assert all(sha256(p['checkpoint']['path']) == p['checkpoint']['sha256'] for p in cfg['policies'])
    if not probe:
        assert manifest['prompt_length_histogram'] == {'256':80,'512':80,'1024':80,'1536':80}
    else:
        assert all(load(run/f'worker_probe_{r:03d}.json')['stage13_replay_max_abs_error'] <= 1e-6 for r in range(4))
    old = load(run/'preservation_before.json')
    assert len(old) == load(run/'preservation_after.json')['preserved_files']
    if any(sha256(name) != digest for name,digest in old.items()):
        raise ValueError('Previously existing evidence changed')
    return {**verification, 'preserved_files': len(old), 'state': 'verified'}


def report(path, run, main_audit, probe_audit):
    summary = load(run/'summary.json')
    comparisons = {(r['length'],r['k'],r['metric'],r['preferred_if_positive'],r['reference']): r for r in summary['comparisons']}
    pairs = [('full','first128','全文 vs 前128'),('full','last128','全文 vs 后128'),
        ('last128','first128','后128 vs 前128'),('last128','last128_rebased','后128原位置 vs 位置归零'),
        ('full','shuffled_mean3','全文匹配 vs 同长度打乱'),('full','static_joint_kl','全文 vs 静态联合贪心')]
    lines = ['# 阶段14：同一检查点的路由观察窗口对照','',
        f"正式评测 {main_audit['documents']} 条 C4 文档、{main_audit['score_rows']:,} 条评分记录。每组提示词长度256/512/1024/1536各80条。文档来自与 Stage 13 训练及选模不同的归档字节区间，并按完整文档、文档开头和token前缀去重。属于归档 C4 的路由留出样本，无法排除 OPT 预训练或旧不可追溯数据的重叠。",
        '', '同一文档所有条件均用相同的完整 OPT 输入、32 个目标 token、m=6 前缀和剪层数 K；同一 K/种子使用完全相同的新版检查点。只有路由观察改变。后128主组保留原位置；另测位置归零。每条样本的打乱路径来自三个同长度双射供体，基座仍使用原文档。短输入一致性、答案不进入路由、跨卡路径、相同mask评分一致性及旧版全文评分重放均通过。',
        '', '## 整体配对比较','',
        '差值为对照减第一组，正值支持第一组。每个文档先平均三个模型种子；打乱组再平均三次置换。区间为按长度分层的文档配对 bootstrap 95% 区间，未校正多重比较。主要比较为 K=10 的 KL 全文对前128、全文对后128；K=12、NLL 和长度分层为补充。','',
        '| K | 指标 | 对照 | 第一组 | 第二组 | 改善 [95% 区间] |','|---:|---|---|---:|---:|---:|']
    for k in [10,12]:
        for metric in ['kl','nll']:
            for a,b,label in pairs:
                row = comparisons['all',k,metric,a,b]; lo,hi = row['ci95']
                lines.append(f"| {k} | {metric.upper()} | {label} | {row['mean_a']:.5f} | {row['mean_b']:.5f} | {row['difference_b_minus_a']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', '## 长度分层：全文相对前128和后128的 KL 改善','',
        '| 提示词长度 | K | 全文 vs 前128 | 全文 vs 后128 | 全文 vs 前128 改变路径比例 |','|---:|---:|---:|---:|---:|']
    for group in ['256','512','1024','1536']:
        for k in [10,12]:
            items = [comparisons[group,k,'kl','full',b] for b in ['first128','last128']]
            c = next(r for r in summary['route_changes'] if (r['length'],r['k'],r['a'],r['b']) == (group,k,'full','first128'))
            cells = ' | '.join(f"{r['difference_b_minus_a']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]" for r in items)
            lines.append(f"| {group} | {k} | {cells} | {c['changed_mask_fraction']:.1%} |")
    lines += ['', '## 如何解读','']
    for a,b,label in pairs[:2]:
        row = comparisons['all',10,'kl',a,b]; lo,hi = row['ci95']
        finding = '区间支持全文降低 KL' if lo > 0 else '区间支持截断组降低 KL' if hi < 0 else '区间跨零，不能判定哪组的 KL 更低'
        lines.append(f"- K=10 {label}：{finding}，差值 {row['difference_b_minus_a']:+.5f}。")
    lines += ['', '改变观察窗口可能改变路径，路径改变本身不等于质量改善。三组均用针对完整提示词训练的路由器，因此本实验识别的是这个冻结路由器部署时改变观察的影响，不能代表专门为128-token训练的路由器。评测是自然文本续写，不外推到多轮对话，也没有重新测量路由耗时或恢复微调。', '',
        f"工程小试使用旧开发样本 {probe_audit['documents']} 条，未使用正式评测样本。正式运行在4张RTX 6000 Ada上，42个CPU核心，核验 {main_audit['preserved_files']} 个既有文件哈希不变。",'',
        f"[完整汇总和路由变化](../runs/{run.name}/summary.json) · [实验协议](../runs/{run.name}/protocol.json) · [覆盖审计](../runs/{run.name}/verification.json)"]
    with path.open('x') as f: f.write('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run', type=Path, required=True)
    parser.add_argument('--main-run', type=Path, required=True)
    args = parser.parse_args(); py = sys.executable; root = Path(__file__).resolve().parents[1]
    probe, run = args.probe_run.resolve(), args.main_run.resolve()
    cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,cpus[40:42])
    env = {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    analysis = root/'analysis'; stem = run.name
    completion = analysis/f'{stem}_completion.json'
    if run.exists() or completion.exists(): raise FileExistsError('Main run and completion paths must be new')
    phase, state, error = 'probe', 'failed', None
    try:
        deadline = time.monotonic()+300
        while not (probe/'status.json').exists():
            if time.monotonic() > deadline: raise TimeoutError('Probe not finished within five minutes')
            time.sleep(3)
        probe_audit = audit(probe, True)
        print(json.dumps({'phase':'probe_verified',**probe_audit}),flush=True)
        phase = 'main'
        subprocess.run([py,'-B','-u',str(root/'tools/run_with_resources_v2.py'),'--gpus','0,1,2,3','--run-dir',str(run),'--',
            py,'-B','-u',str(root/'tools/evaluate_observation_windows_v1.py')],cwd=root,env=env,check=True,
            preexec_fn=lambda: os.sched_setaffinity(0,cpus))
        phase = 'audit'
        main_audit = audit(run,False)
        write(run/'independent_audit.json',main_audit)
        phase = 'report'
        result = analysis/'STAGE14_OBSERVATION_WINDOWS_RESULTS_20261111.zh-CN.md'
        report(result,run,main_audit,probe_audit)
        publish = analysis/f'{stem}_upload.json'
        files = [result,Path(__file__),root/'tools/evaluate_observation_windows_v1.py',
            *[run/n for n in ['summary.json','protocol.json','input_manifest.json','verification.json','independent_audit.json','status.json']],
            *[probe/n for n in ['verification.json',*[f'worker_probe_{i:03d}.json' for i in range(4)]]]]
        assert all(not p.name.startswith('private_') and p.stat().st_size < 1024*1024 for p in files)
        write(publish,[str(p.relative_to(root)) for p in files])
        phase = 'publication'
        subprocess.run([py,'-B',str(root/'tools/publish_additions_v7.py'),'--root',str(root),'--repo','liuuu579-cmd/NeuroBypass',
            '--files',str(publish),'--message','Evaluate frozen router observation windows on held-out long prompts',
            '--receipt',str(analysis/f'{stem}_publication.json')],cwd=root,env=env,check=True)
        state = 'completed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    result = {'state':state,'phase':phase,'error':error,'run':str(run),'finished_utc':datetime.now(timezone.utc).isoformat()}
    write(completion,result)
    print(json.dumps(result),flush=True)
    return 0 if state == 'completed' else 1


if __name__ == '__main__': sys.exit(main())
