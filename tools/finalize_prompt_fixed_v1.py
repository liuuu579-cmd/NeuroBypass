"""One-shot completion audit/report/publication for an already launched run.

This is a finite local pipeline tail, not a recurring task or notification bot.
It writes only new report, verification and publication paths. No messages are
sent, and private tokens/checkpoints are excluded from publication.
"""
import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from verify_prompt_fixed_v1 import verify


def write_json(path,value):
    with path.open('x') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)


def report(root,run,verification):
    summary=json.loads((run/'summary.json').read_text())
    checkpoint=json.loads((run/'router_selected_checkpoint.json').read_text())
    history=json.loads((run/'router_training_history.json').read_text())
    selected_epoch=next(r['epoch'] for r in history if abs(r.get('continuation_kl',-1)-checkpoint['best_validation_continuation_kl'])<1e-12)
    labels={'full':'完整 OPT','static':'校准静态联合贪心','router':'真实提示词 Router',
            'shuffled_router':'交换 Router 方案','fixed_input_router':'固定训练提示词方案'}
    lines=['# 每轮提示词固定选层：Router 训练与独立测试结果','',
           '本轮已完成联合贪心标签生成、两卡 Router 监督训练和独立测试。OPT 权重冻结，尚未进行恢复微调。',
           '','Router 每轮只观察 128-token 提示词，逐步选择 6 个跳层索引；提示词与回答使用同一方案。每个后续选择都以已有剪层集合为条件。离线标签使用 32-token 自然续写的联合 teacher KL，不将未来续写输入 Router。',
           '','数据按本地原始记录分组：800 训练、100 验证、100 测试，另有 16 校准及 16 先导记录；精确记录、提示词和前32词重复过滤。原始 URL 缺失，近重复与旧特征数据重叠仍无法完全排除。',
           '',f'训练 {len(history)} 个 epoch，按固定 32 条验证记录的真实续写 KL 选择 epoch {selected_epoch}。测试集未参与标签搜索、checkpoint 选择或静态方案校准。',
           '','## 同一 100 条测试记录','',
           '| 方案 | 续写平均 KL | 首个预测 KL | 续写 PPL | 不同组合数 |',
           '| --- | ---: | ---: | ---: | ---: |']
    by_policy={p['policy']:p for p in summary['policies']}
    for name in ['full','static','router','shuffled_router','fixed_input_router']:
        p=by_policy[name]
        lines.append(f"| {labels[name]} | {p['continuation_kl']:.6f} | {p['first_next_token_kl']:.6f} | {p['perplexity']:.4f} | {p['unique_masks']} |")
    lines+=['','## 配对比较','',
            '差值为真实 Router 减去对照，负值有利于 Router。按记录配对 bootstrap 10,000 次，seed579；区间为描述性结果，未校正文档依赖或多重比较。','',
            '| 对照 | 指标 | 平均差值 | 95% 区间 | Router 更优比例 |','| --- | --- | ---: | --- | ---: |']
    for comparison in summary['paired_comparisons']:
        lo,hi=comparison['bootstrap_95pct']
        lines.append(f"| {labels[comparison['reference']]} | {comparison['metric']} | {comparison['mean_delta']:.6f} | [{lo:.6f}, {hi:.6f}] | {comparison['fraction_better']:.1%} |")
    static_kl=next(c for c in summary['paired_comparisons'] if c['reference']=='static' and c['metric']=='continuation_kl')
    match_kl=next(c for c in summary['paired_comparisons'] if c['reference']=='shuffled_router' and c['metric']=='continuation_kl')
    if static_kl['bootstrap_95pct'][1]<0 and match_kl['bootstrap_95pct'][1]<0:
        conclusion='本轮在续写 KL 上同时观察到相对静态方案和相对交换方案的改善，支持继续做数据与预算一致的模型恢复训练、下游任务和更广泛输入验证。'
    else:
        conclusion='本轮尚未同时建立相对静态方案及相对交换方案的稳定续写 KL 优势。下一步应先分析 Router 的误差、输入信息和条件路径泛化，暂不宣称动态方案胜出。'
    lines+=['',conclusion,'','## 缓存生成计时','',
            '8 条提示词，每种方案一次热身、三次测量，固定生成64个 token，忽略EOS提前停止。decode 部分为预填充之后的63步。下面为运行中位数，Router开销包含在其总耗时中。服务器共享负载与小样本限制了精度，不代表完整服务吞吐。','',
            '| 方案 | Router 秒 | Prefill 秒 | Decode 秒 | 总秒 | KV 数据 MiB |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['generation_timing_medians']:
        lines.append(f"| {labels[row['policy']]} | {row['route_seconds']:.6f} | {row['prefill_seconds']:.6f} | {row['decode_seconds']:.6f} | {row['total_seconds']:.6f} | {row['cache_bytes']/2**20:.2f} |")
    lines+=['','每轮固定跳6层时，保留层缓存长度完整、被跳层无KV，缓存数据比40层完整模型少15%。模型权重仍全部常驻，不据此宣称权重显存压缩。新一轮若换方案必须新建缓存重算完整历史。',
            '','## 核验与证据','',
            f"验收通过：{verification['verified_labels']} 条条件贪心轨迹、{verification['verified_candidate_scores']} 次候选评分、{verification['verified_metric_rows']} 条质量记录、{verification['verified_timing_rows']} 条计时。{verification['preserved_preexisting_text_evidence_files']} 个既有文本证据文件及 {verification['preserved_original_and_snapshot_pairs']} 对原始/归档文件哈希未变。",
            '',f"主运行目录：`{run.relative_to(root)}`。全过程2 GPU / 21 CPU核，通过统一资源入口执行。checkpoint SHA256：`{checkpoint['sha256']}`。权重、输入token数组和生成token序列只留本地，不上传。",
            '','全部输入是C4自然续写，尚非聊天指令、多轮对话或完整PIQA任务结果。teacher forcing质量与自由生成计时分别报告，自由生成内容没有人工质量评价。静态与动态恢复微调都未实施，不能代替双方恢复训练后的比较。',
            '','代码与先导协议见 [阶段3协议](STAGE3_PROMPT_FIXED_PROTOCOL_20261104.zh-CN.md)。本报告由完成后单次验收程序生成。','']
    path=root/'analysis/STAGE3_MAIN_RESULTS_20261104.zh-CN.md'
    with path.open('x') as handle:handle.write('\n'.join(lines))
    csv_path=root/'analysis/stage3_main_results_20261104.csv'
    with csv_path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['policy','samples','k','continuation_kl','first_next_token_kl','continuation_nll','perplexity','unique_masks'])
        writer.writeheader();writer.writerows(summary['policies'])
    return path,csv_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-run',type=Path,required=True);parser.add_argument('--wait',action='store_true')
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    root=args.root.resolve();run=args.run.resolve();started=time.monotonic()
    while not (run/'status.json').exists():
        if not args.wait:raise ValueError('Run not finished')
        if time.monotonic()-started>6*3600:raise TimeoutError('Finite pipeline tail exceeded six hours')
        time.sleep(20)
    status=json.loads((run/'status.json').read_text())
    if status['state']!='completed' or status.get('exit_code')!=0:
        write_json(root/'analysis/stage3_main_finalization_failure_20261104.json',{'run_status':status,'time':datetime.datetime.now().astimezone().isoformat()})
        raise RuntimeError('Experiment failed; evidence preserved, no success report or result publication attempted')
    verification=verify(root,run,args.data_run)
    verification_path=root/'analysis/stage3_main_verification_20261104.json';write_json(verification_path,verification)
    report_path,csv_path=report(root,run,verification)
    paths=[str(p.relative_to(root)) for p in [verification_path,report_path,csv_path]]
    for p in sorted(run.iterdir()):
        if p.is_file() and not p.name.startswith('private_') and p.suffix in {'.json','.jsonl','.log'}:
            paths.append(str(p.relative_to(root)))
    manifest=root/'analysis/stage3_completion_upload_20261104.json';paths.append(str(manifest.relative_to(root)));write_json(manifest,paths)
    if args.publish:
        result=subprocess.run([sys.executable,str(root/'tools/publish_additions_v2.py'),'--root',str(root),'--repo','liuuu579-cmd/NeuroBypass',
                               '--files',str(manifest),'--message','Report prompt-fixed greedy router training and held-out continuation evaluation'],
                              cwd=root,text=True,capture_output=True,timeout=300)
        if result.returncode!=0:raise RuntimeError(result.stderr[-4000:])
        write_json(root/'analysis/stage3_completion_publication_20261104.json',json.loads(result.stdout))
        print(result.stdout,flush=True)
    print(json.dumps({'state':'finalized','report':str(report_path),'verified':verification},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
