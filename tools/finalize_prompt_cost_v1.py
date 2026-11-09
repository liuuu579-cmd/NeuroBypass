"""Independently check recorded generation costs, summarize and publish."""
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256, write_json


def audit(root, run):
    config = json.loads((run/'protocol.json').read_text())
    completion = json.loads((run/'completion.json').read_text())
    assert completion['state']=='completed'
    assert sha256(config['private_inputs'])==config['input_sha256']
    for name,digest in config['source_sha256'].items(): assert sha256(root/'tools'/name)==digest
    tracked=json.loads((run/'preservation_before.json').read_text())
    for name,digest in tracked.items(): assert sha256(name)==digest,name
    assert json.loads((run/'preservation_after.json').read_text())['preserved_files']==len(tracked)
    allocation=json.loads((run/'allocation.json').read_text())
    assert allocation['cpu_count']==42 and len(allocation['gpu_indices'])==4
    policies={(p['name'],p['seed'],p['k'],p['deployment']):p for p in config['policies']}
    probes=0
    for rank in range(4):
        data=json.loads((run/f'cache_probe_{rank:03d}.json').read_text())
        assert data['cpu_ids']==allocation['cpu_ids'][rank*10:(rank+1)*10]
        assert all(set(cpus)<=set(data['cpu_ids']) for cpus in data['thread_cpu_ids'].values())
        for row in data['probes']:
            assert row['cache_within_tolerance'] and row.get('physical_equivalent',True)
            assert row['mean_absolute_position_kl']<=1e-4 and row['max_absolute_position_kl']<=1e-3
            if 'physical_vs_bypass_max_logit_error' in row: assert row['physical_vs_bypass_max_logit_error']<=1e-4
            probes+=1
    rows=[json.loads(line) for path in sorted(run.glob('timing_*.jsonl')) for line in path.open()]
    assert len(rows)==completion['expected_rows']==4*(config['warmups']+config['repeats'])*len(policies)
    by_key={};grouped=defaultdict(list)
    for row in rows:
        policy_key=row['policy'],row['seed'],row['k'],row['deployment']
        policy=policies[policy_key]
        key=row['rank'],row['round'],*policy_key
        assert key not in by_key;by_key[key]=row
        assert row['length']==config['lengths_by_rank'][row['rank']]
        assert row['physical_gpu']==config['gpu_indices'][row['rank']]
        assert row['warmup']==(row['round']<config['warmups'])
        assert len(row['skipped_layers'])==len(set(row['skipped_layers']))==row['k']
        assert all(2<=i<40 for i in row['skipped_layers'])
        if policy['kind']=='router': assert set(policy['prefix'])<=set(row['skipped_layers'])
        else: assert row['skipped_layers']==sorted(policy['skips'])
        assert row['new_tokens']==config['new_tokens']
        assert row['cached_tokens']==row['length']+row['new_tokens']-1
        assert row['kv_bytes']==2*(40-row['k'])*row['cached_tokens']*5120*2
        assert row['cache_layer_lengths_verified'] and row['cache_route_change_rejected']
        assert row['peak_reserved_bytes']>=row['peak_allocated_bytes']>=row['resident_allocated_bytes']
        assert row['resident_allocated_bytes']>=row['model_cuda_parameter_bytes']+row['router_cuda_parameter_bytes']
        assert math.isclose(row['ttft_seconds'],row['route_seconds']+row['prefill_seconds'],abs_tol=1e-8)
        assert math.isclose(row['total_seconds'],row['ttft_seconds']+row['decode_seconds'],abs_tol=1e-8)
        assert math.isclose(row['decode_tokens_per_second'],(row['new_tokens']-1)/row['decode_seconds'],rel_tol=1e-9)
        assert math.isclose(row['end_to_end_tokens_per_second'],row['new_tokens']/row['total_seconds'],rel_tol=1e-9)
        assert all(row[key]>0 and math.isfinite(row[key]) for key in ['route_seconds','prefill_seconds','decode_seconds','total_seconds'])
        if not row['warmup']:grouped[row['rank'],*policy_key].append(row)
    # Every static physical and bypass trial must generate the same tokens.
    for row in rows:
        if row['deployment']=='physical':
            other=by_key[row['rank'],row['round'],row['policy'],row['seed'],row['k'],'bypass']
            assert row['generated_tokens_sha256']==other['generated_tokens_sha256']
            assert row['model_cuda_parameter_bytes']<other['model_cuda_parameter_bytes']
    summary=[]
    for key,group in grouped.items():
        assert len(group)==config['repeats']
        rank,policy,seed,k,deployment=key
        result={'rank':rank,'physical_gpu':config['gpu_indices'][rank],'length':config['lengths_by_rank'][rank],
            'policy':policy,'seed':seed,'k':k,'deployment':deployment,'repeats':len(group)}
        for field in ['route_seconds','prefill_seconds','ttft_seconds','decode_seconds','total_seconds',
            'decode_tokens_per_second','end_to_end_tokens_per_second','peak_allocated_bytes','peak_reserved_bytes',
            'resident_allocated_bytes','kv_bytes','model_cuda_parameter_bytes','router_cuda_parameter_bytes']:
            values=[r[field] for r in group]
            result[field+'_median']=float(np.median(values))
            if field in ['total_seconds','ttft_seconds']:result[field+'_p95']=float(np.quantile(values,.95))
        baseline=[by_key[rank,r['round'],'full',None,0,'bypass'] for r in group]
        result['paired_total_speedup_vs_full_median']=float(np.median([b['total_seconds']/r['total_seconds'] for b,r in zip(baseline,group)]))
        result['paired_decode_speedup_vs_full_median']=float(np.median([b['decode_seconds']/r['decode_seconds'] for b,r in zip(baseline,group)]))
        summary.append(result)
    write_json(run/'verification.json',{'status':'passed','probe':config['probe'],'verified_trials':len(rows),
        'verified_cache_probes':probes,'preserved_files':len(tracked),'cpu_budget':42,
        'static_physical_vs_bypass_generated_hashes_identical':True})
    write_json(run/'summary.json',{'probe':config['probe'],'timing':summary,'elapsed_seconds':completion['elapsed_seconds']})
    return config,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--publish',action='store_true');args=p.parse_args()
    root,run=args.root.resolve(),args.run.resolve()
    config,summary=audit(root,run)
    if config['probe']:
        print(json.dumps({'state':'probe_audited','trials':len(summary)}),flush=True);return
    table=root/'analysis/stage8_prompt_cost_20261108.csv'
    with table.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in summary for k in r)))
        writer.writeheader();writer.writerows(summary)
    report=root/'analysis/STAGE8_PROMPT_COST_RESULTS_20261108.zh-CN.md'
    text=['# 固定提示词路由的吞吐量与显存', '',
        'FP16/SDPA、batch1，每请求固定生成128token、忽略EOS。每方法5次预热、20次测量，按轮随机交错；每轮各方法输入相同。输入长度128/512/1024，另在第4张卡重复512。每个加速比仅使用同一物理卡、同一轮的完整模型作分母。', '',
        '本表列出全部训练种子，未依据计时择优。模型/静态层/检查点沿用阶段7；K=10、12为查看质量后确定的效率工作点。', '',
        '| GPU | 输入长度 | 方法 | 种子 | K | 部署 | decode token/s | 总token/s | 含路由总加速 | 峰值allocated GiB | KV MiB |',
        '| ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        text.append(f"| {r['physical_gpu']} | {r['length']} | {r['policy']} | {r['seed']} | {r['k']} | {r['deployment']} | {r['decode_tokens_per_second_median']:.2f} | {r['end_to_end_tokens_per_second_median']:.2f} | {r['paired_total_speedup_vs_full_median']:.3f} | {r['peak_allocated_bytes_median']/2**30:.3f} | {r['kv_bytes_median']/2**20:.2f} |")
    text += ['', '动态总耗时计入一次提示词路由；prefill产生首token，decode计127个后续token。静态physical部署将剪掉的层从decoder和GPU移除，验证与bypass的缓存logits及生成token哈希一致。仅为切换测量策略而保留CPU副本；权重搬运属于加载准备，不计入稳态请求。', '',
        '显存为PyTorch peak allocated/reserved，而非整进程NVML峰值；单列权重字节、路由器字节和KV张量实际字节。动态保留完整基座权重，不能从跳层比例推断总显存同比下降。allocator每请求清理，因此reserved是明确的冷allocator测量口径。', '',
        '缓存探针对照FP16完整前向和增量前向，沿用mean KL<=1e-4/max KL<=1e-3容差，仅覆盖记录的提示词/路径。不能据此保证所有生成路径数值一致。每个请求都重建完整提示词的KV；旧缓存不能换路径沿用，代码验证拒绝此操作。此表未单独测旧缓存释放或真实对话调度器。', '',
        '计时输入由训练记录组成，只测成本，不用于新质量结论。正常OPT CausalLM在prefill计算完整序列输出头；没有进行输出头、批处理或服务端调度优化。stage7质量报告中的文献基线仍是当前OPT适配，非完整官方复现。', '',f'来源：`{run.relative_to(root)}`。完整TTFT、prefill、route、P95、reserved和参数存储字段见CSV；所有单次记录及验证保留。','']
    with report.open('x') as stream:stream.write('\n'.join(text))
    if args.publish:
        files=[str(p.relative_to(root)) for p in run.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl','.log'} and not p.name.startswith('private_')]
        files +=[str(p.relative_to(root)) for p in root.glob('tools/*prompt_cost*.py')]
        files +=[str(report.relative_to(root)),str(table.relative_to(root)),'analysis/STAGE8_PROMPT_COST_PROTOCOL_20261108.zh-CN.md']
        manifest=root/'analysis/stage8_cost_completed_upload_20261108.json';files.append(str(manifest.relative_to(root)))
        write_json(manifest,sorted(set(files)))
        subprocess.run([sys.executable,'-B',str(root/'tools/publish_additions_v7.py'),'--root',str(root),
            '--repo','liuuu579-cmd/NeuroBypass','--files',str(manifest),'--message','Complete same-card cached inference throughput and memory measurements',
            '--receipt',str(root/'analysis/stage8_cost_completed_publication_20261108.json')],check=True,cwd=root)
    print(json.dumps({'report':str(report),'groups':len(summary)}),flush=True)


if __name__=='__main__':main()
