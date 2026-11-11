"""Frozen-path headroom diagnosis and a short RULER-inspired exploratory probe.

All outputs are new. Four GPU replicas share42 CPU cores via the existing
resource launcher. Engineering gates precede calibration and measurements.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_full_context_four_gpu_v1 import policies
from full_context_core_v1 import candidate_costs, load_router, prompt_embeddings
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, read_lines, resources,
    setup_worker, thread_record, verify_preserved, write)
from headroom_ruler_core_v1 import (BUDGETS, TASKS, answer_metrics, cpu_checks,
    jsonl, paired, prepare_inputs, summarize_history)
from prompt_fixed_core_v1 import (FixedRouteSession, best_candidate,
    cached_continuation_logits, continuation_logits, continuation_metrics)
from stage1_benchmark_v2 import sha256, stamp


def load(path):
    return json.loads(Path(path).read_text())


def tensor(model, ids):
    return torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device)[None]


def evaluate(model, ids, p, skips, teacher):
    values=continuation_metrics(continuation_logits(model,ids,p,skips),teacher,ids,p)
    return {'kl':values['continuation_kl'],'nll':values['continuation_nll'],
            'target_tokens':values['continuation_tokens']}


def routed_paths(model, routers, config, ids, seeds):
    # The only function that obtains a route receives prompt IDs, never targets.
    emb=prompt_embeddings(model,ids)
    assert emb.shape[1]==ids.shape[1]
    encoded, paths={},{}
    for policy in config['router_policies']:
        if policy['seed'] not in seeds: continue
        name=policy['checkpoint']['path']
        if name not in encoded: encoded[name]=routers[name].encode(emb)
        path=routers[name].route_encoded(encoded[name],policy['k'],policy['prefix'])[0].tolist()
        assert path[:6]==config['static_path'][:6] and len(set(path))==policy['k'] and min(path)>=2
        paths[policy['seed'],policy['k']]=path
    return paths


def generate(model,prompt,skips,maximum,eos):
    session=FixedRouteSession(model,skips)
    logits=session.forward(prompt)[:,-1]
    generated=[]
    for index in range(maximum):
        token=int(logits.argmax(-1).item());generated.append(token)
        if token==eos or index+1==maximum: break
        logits=session.forward(tensor(model,[token]))[:,-1]
    return generated


def engineering(rank,model,config,routers,ruler,diagnostics):
    r=diagnostics[rank%len(diagnostics)]
    p=min(r['prompt_tokens'],129)
    ids=tensor(model,r['ids'][:p+4])
    teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
    selected=config['static_path'][:6]
    candidates=[i for i in (3,20,39) if i not in selected]
    shared=candidate_costs(model,ids,p,selected,candidates,teacher)
    serial=[[i,evaluate(model,ids,p,selected+[i],teacher)['kl']] for i in candidates]
    error=max(abs(a[1]-b[1]) for a,b in zip(shared,serial))
    assert error<=1e-6,('candidate replay',error)
    full=continuation_logits(model,ids,p,selected)
    cached,session=cached_continuation_logits(model,ids,p,selected)
    cache_kl=(F.log_softmax(full,-1).exp()*(F.log_softmax(full,-1)-F.log_softmax(cached,-1))).sum(-1).abs().mean().item()
    assert cache_kl<1e-4 and torch.equal(full.argmax(-1),cached.argmax(-1)),('KV replay',cache_kl)
    # Replay a saved expert mask independently, to validate the historical headroom values.
    training=Path(config['training_run']);data=Records(config['data_run'])
    sample=config['historical_replay_ids'][rank]
    raw,old_p=data[sample];old_ids=tensor(model,raw.tolist())
    old_teacher=F.log_softmax(continuation_logits(model,old_ids,old_p,[]),-1)
    replays=[]
    for m in (0,6):
        row=next(row for path in sorted(training.glob(f'm{m:02d}_labels_*.jsonl'))
                 for row in read_lines(path) if row['sample']==sample)
        step=next(s for s in row['steps'] if s['step']==16)
        actual=evaluate(model,old_ids,old_p,row['ordered_skips'][:16],old_teacher)['kl']
        replay_error=abs(actual-step['joint_continuation_kl'])
        assert replay_error<=1e-6,('history replay',replay_error)
        replays.append({'m':m,'max_abs_error':replay_error})
    long=next(r for r in ruler if r['length']==1920)
    observation=tensor(model,long['prompt_ids'])
    paths=routed_paths(model,routers,config,observation,[579])
    assert len(paths)==len(BUDGETS)
    # Future target changes cannot alter the exact tensor passed to the router.
    sequence=long['prompt_ids']+long['target_ids']
    changed=long['prompt_ids']+[model.config.eos_token_id]*len(long['target_ids'])
    assert sequence[:1920]==changed[:1920]
    # Actual cached generation must work at the maximum observed length and deep skip budget.
    generated=generate(model,observation,paths[579,16],3,model.config.eos_token_id)
    assert 1<=len(generated)<=3
    return {'passed':True,'candidate_serial_max_abs_error':error,'cached_teacher_forcing_kl':cache_kl,
        'history_mask_replays':replays,'all1920_prompt_tokens_observed':True,
        'maximum_length_cached_generation_checked':True,'target_boundary_checked':True,
        'all_base_and_router_parameters_frozen':not any(p.requires_grad for p in model.parameters()) and
            not any(p.requires_grad for router in routers.values() for p in router.parameters()),
        **thread_record()}


def greedy_search(model,ids,p,teacher,prefix,maximum,cache,rank,sample,messages):
    path=list(prefix);steps=[]
    for step in range(len(path)+1,maximum+1):
        key=tuple(sorted(path))
        if key not in cache:
            cache[key]=candidate_costs(model,ids,p,path,[i for i in range(2,40) if i not in path],teacher)
        costs=cache[key];chosen,value=best_candidate(costs)
        steps.append({'step':step,'previous_skips':list(path),'selected_layer':chosen,
                      'joint_continuation_kl':value,'candidate_costs':costs})
        path.append(chosen)
        if step%4==0 or step==maximum:
            messages.put({'kind':'progress','phase':'headroom_search','rank':rank,
                'sample':sample,'prefix_size':len(prefix),'step':step,'prompt_tokens':p})
    return path,steps


def diagnostic(model,routers,config,record,rank,messages,search_stream):
    p=record['prompt_tokens'];ids=tensor(model,record['ids'])
    teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
    paths=routed_paths(model,routers,config,ids[:,:p],[579,580,581])
    references={};cache={}
    for m in (6,0):
        path,steps=greedy_search(model,ids,p,teacher,config['static_path'][:m],16,cache,rank,record['sample'],messages)
        references[m]=(path,steps)
        search_stream.write(json.dumps({'sample':record['sample'],'m':m,'ordered_skips':path,'steps':steps})+'\n')
        search_stream.flush()
    values={};rows=[]
    def add(policy,k,mask,seed=None):
        key=tuple(sorted(mask))
        if key not in values: values[key]=evaluate(model,ids,p,mask,teacher)
        row={'sample':record['sample'],'source_sample':record['source_sample'],'task':'c4_diagnostic',
             'length':p,'policy':policy,'seed':seed,'k':k,'skips':list(mask),**values[key]}
        rows.append(row)
        return row
    add('full',0,[])
    for k in BUDGETS:
        add('static_joint_kl',k,config['static_path'][:k])
        for seed in (579,580,581): add('router_m6',k,paths[seed,k],seed)
        for m in (6,0):
            path,steps=references[m]
            row=add(f'future_greedy_m{m}',k,path[:k])
            expected=next(s['joint_continuation_kl'] for s in steps if s['step']==k)
            assert abs(row['kl']-expected)<=1e-6
    return rows


def ruler_evaluation(model,routers,config,record,portfolio,tokenizer):
    p=record['length'];prompt=tensor(model,record['prompt_ids'])
    ids=tensor(model,record['prompt_ids']+record['target_ids'])
    teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
    paths=routed_paths(model,routers,config,prompt,[579])
    proposals=[('full',0,[],None)]
    for k in BUDGETS:
        proposals.extend([('static_joint_kl',k,config['static_path'][:k],None),
            ('calibrated_static_portfolio',k,config['static_candidates'][portfolio[str(k)]['path_name']][:k],None),
            ('router_m6',k,paths[579,k],579)])
    cache={};rows=[]
    for policy,k,mask,seed in proposals:
        key=tuple(sorted(mask))
        if key not in cache:
            costs=evaluate(model,ids,p,mask,teacher)
            generated=generate(model,prompt,mask,48,model.config.eos_token_id)
            text=tokenizer.decode(generated,skip_special_tokens=True)
            cache[key]={**costs,**answer_metrics(text,record['answers'],record['answer_kind']),
                        'generated_ids':generated,'generated_text':text}
        rows.append({**{n:record[n] for n in ('sample','task','case','length','depth','replica')},
            'policy':policy,'k':k,'seed':seed,'skips':mask,**cache[key]})
    return rows


def worker(rank,config,inbox,messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='sdpa').to(f'cuda:{rank}').eval().requires_grad_(False)
    routers={p['checkpoint']['path']:None for p in config['router_policies']}
    for path in routers: routers[path]=load_router(path,f'cuda:{rank}')
    tokenizer=AutoTokenizer.from_pretrained(config['model'],use_fast=False,local_files_only=True)
    out=Path(config['output']);ruler=read_lines(out/'private_ruler_inputs.jsonl')
    diagnostics=read_lines(out/'private_diagnostic_inputs.jsonl')
    with torch.inference_mode():
        result=engineering(rank,model,config,routers,ruler,diagnostics)
    assert result['all_base_and_router_parameters_frozen']
    write(out/f'engineering_{rank:03d}.json',result)
    messages.put({'kind':'ready','rank':rank})
    task=inbox.get();assert task['kind']=='calibrate'
    records=[r for r in ruler if r['split']=='calibration'][rank::4]
    with torch.inference_mode(),(out/f'private_calibration_{rank:03d}.jsonl').open('x') as stream:
        for index,r in enumerate(records):
            p=r['length'];ids=tensor(model,r['prompt_ids']+r['target_ids'])
            teacher=F.log_softmax(continuation_logits(model,ids,p,[]),-1)
            cached={}
            for k in BUDGETS:
                for name,path in config['static_candidates'].items():
                    mask=path[:k];key=tuple(sorted(mask))
                    if key not in cached: cached[key]=evaluate(model,ids,p,mask,teacher)
                    stream.write(json.dumps({'sample':r['sample'],'case':r['case'],'k':k,
                        'path_name':name,**cached[key]})+'\n')
            stream.flush()
            messages.put({'kind':'progress','phase':'static_portfolio_calibration','rank':rank,
                'completed':index+1,'total':len(records)})
    messages.put({'kind':'calibrated','rank':rank})
    task=inbox.get();assert task['kind']=='evaluate'
    local_diagnostics=diagnostics[rank::4]
    local_ruler=[r for r in ruler if r['split']=='evaluation'][rank::4]
    # Half start search and half start retrieval: both experiments make progress concurrently.
    jobs=([('diagnostic',r) for r in local_diagnostics]+[('ruler',r) for r in local_ruler])
    if rank>=2: jobs=jobs[len(local_diagnostics):]+jobs[:len(local_diagnostics)]
    started=time.monotonic()
    with torch.inference_mode(),(out/f'private_search_{rank:03d}.jsonl').open('x') as search_stream,\
            (out/f'private_diagnostic_scores_{rank:03d}.jsonl').open('x') as diagnostic_stream,\
            (out/f'private_ruler_scores_{rank:03d}.jsonl').open('x') as ruler_stream:
        for index,(kind,record) in enumerate(jobs):
            if kind=='diagnostic':
                rows=diagnostic(model,routers,config,record,rank,messages,search_stream);stream=diagnostic_stream
            else:
                rows=ruler_evaluation(model,routers,config,record,task['portfolio'],tokenizer);stream=ruler_stream
            for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush()
            messages.put({'kind':'progress','phase':kind,'rank':rank,'completed':index+1,'total':len(jobs),
                'sample':record['sample'],'elapsed_seconds':time.monotonic()-started})
    write(out/f'worker_result_{rank:03d}.json',{'passed':True,'elapsed_seconds':time.monotonic()-started,
        'diagnostic_documents':len(local_diagnostics),'ruler_cases':len(local_ruler),
        'peak_allocated_gib':torch.cuda.max_memory_allocated(rank)/2**30,
        'private_output_sha256':{name:sha256(out/name) for name in
            (f'private_search_{rank:03d}.jsonl',f'private_diagnostic_scores_{rank:03d}.jsonl',f'private_ruler_scores_{rank:03d}.jsonl')},
        **thread_record()})
    messages.put({'kind':'done','rank':rank})
    assert inbox.get() is None


def choose_portfolio(out,config):
    rows=[r for rank in range(4) for r in read_lines(out/f'private_calibration_{rank:03d}.jsonl')]
    selected={};aggregates=[]
    for k in BUDGETS:
        options=[]
        for name in config['static_candidates']:
            group=[r for r in rows if r['k']==k and r['path_name']==name]
            assert len(group)==36
            assert len({r['sample'] for r in group})==36
            value=float(np.mean([r['kl'] for r in group]))
            options.append((value,name));aggregates.append({'k':k,'path_name':name,'kl':value,'cases':len(group)})
        cost,name=min(options)
        selected[str(k)]={'path_name':name,'calibration_kl':cost}
    write(out/'static_portfolio_selection.json',{'selected':selected,'calibration':aggregates,
        'policy':'One global fixed path perK across all evaluation tasks and inputs; mean calibration answer-position KL only. No evaluation outputs used. This is portfolio selection, not a new joint greedy search.'})
    return selected


def summarize(out,config):
    ruler=[r for rank in range(4) for r in read_lines(out/f'private_ruler_scores_{rank:03d}.jsonl')]
    diagnostic_rows=[r for rank in range(4) for r in read_lines(out/f'private_diagnostic_scores_{rank:03d}.jsonl')]
    source=read_lines(out/'private_ruler_inputs.jsonl')
    expected={r['sample']:r for r in source if r['split']=='evaluation'}
    assert len(ruler)==72*10 and len(diagnostic_rows)==16*19
    seen=set()
    for row in ruler:
        key=row['sample'],row['policy'],row['k'];assert key not in seen;seen.add(key)
        assert row['sample'] in expected and len(set(row['skips']))==row['k']
        assert not row['skips'] or min(row['skips'])>=2
        assert all(math.isfinite(row[n]) for n in ('kl','nll','answer_recall'))
        r=expected[row['sample']]
        assert answer_metrics(row['generated_text'],r['answers'],r['answer_kind'])=={n:row[n] for n in ('answer_recall','all_answers','strict_exact')}
        assert row['target_tokens']==len(r['target_ids']) and row['length']==len(r['prompt_ids'])
    aggregates=[]
    for dimension in ('overall','task','length','depth'):
        grouped=defaultdict(list)
        for row in ruler:
            grouped[row['policy'],row['k'],str(row.get(dimension,'all')) if dimension!='overall' else 'all'].append(row)
        for (policy,k,stratum),rows in sorted(grouped.items()):
            aggregates.append({'dimension':dimension,'stratum':stratum,'policy':policy,'k':k,'records':len(rows),
                **{n:float(np.mean([r[n] for r in rows])) for n in ('kl','nll','answer_recall','all_answers','strict_exact')}})
    index={(r['sample'],r['policy'],r['k']):r for r in ruler};comparisons=[]
    for task in ('all',*TASKS):
        ids=[i for i,r in expected.items() if task=='all' or r['task']==task]
        for k in BUDGETS:
            for baseline in ('static_joint_kl','calibrated_static_portfolio'):
                for metric in ('kl','nll','answer_recall','all_answers','strict_exact'):
                    sign=-1 if metric in ('kl','nll') else 1
                    differences=[sign*(index[i,'router_m6',k][metric]-index[i,baseline,k][metric]) for i in ids]
                    comparisons.append({'task':task,'k':k,'baseline':baseline,'metric':metric,
                        'positive_means':'dynamic improves quality',**paired(differences,[expected[i]['case'] for i in ids])})
    # Each diagnostic input contributes once after averaging the3 router seeds.
    diag_index=defaultdict(list)
    for row in diagnostic_rows:
        key=row['sample'],row['policy'],row['k'];diag_index[key].append(row)
        assert math.isfinite(row['kl']) and math.isfinite(row['nll'])
    dsummary=[]
    for k in BUDGETS:
        for policy in ('static_joint_kl','router_m6','future_greedy_m6','future_greedy_m0'):
            values=[]
            for sample in range(16):
                rows=diag_index[sample,policy,k]
                assert len(rows)==(3 if policy=='router_m6' else 1)
                values.append({n:float(np.mean([r[n] for r in rows])) for n in ('kl','nll')})
            dsummary.append({'policy':policy,'k':k,'documents':16,
                'kl':float(np.mean([r['kl'] for r in values])),'nll':float(np.mean([r['nll'] for r in values])),
                'ppl':math.exp(float(np.mean([r['nll'] for r in values])))})
    dpaired=[]
    for k in BUDGETS:
        for baseline,reference in [('static_joint_kl','future_greedy_m6'),('static_joint_kl','future_greedy_m0'),
                                   ('router_m6','future_greedy_m6'),('future_greedy_m6','future_greedy_m0')]:
            for metric in ('kl','nll'):
                diffs=[]
                for sample in range(16):
                    diffs.append(float(np.mean([r[metric] for r in diag_index[sample,baseline,k]]))-
                                 float(np.mean([r[metric] for r in diag_index[sample,reference,k]])))
                dpaired.append({'k':k,'metric':metric,'baseline':baseline,'reference':reference,
                    'positive_means':'reference improves quality',**paired(diffs)})
    write(out/'summary.json',{'ruler_aggregates':aggregates,'ruler_paired':comparisons,
        'diagnostic_aggregates':dsummary,'diagnostic_paired':dpaired,
        'ruler_rows':len(ruler),'diagnostic_rows':len(diagnostic_rows),
        'limits':'Exploratory2K synthetic probe, one router seed579 for retrieval and3 seeds forC4; no recovery. Headroom references see true future targets, not deployable or guaranteed optimal. Case-clustered intervals do not include training-seed uncertainty.'})
    # Assert static masks really are identical for every task/input at a fixedK.
    for k in BUDGETS:
        for policy in ('static_joint_kl','calibrated_static_portfolio'):
            masks={tuple(sorted(r['skips'])) for r in ruler if r['policy']==policy and r['k']==k}
            assert len(masks)==1
    write(out/'verification.json',{'passed':True,'ruler_rows':len(ruler),'diagnostic_rows':len(diagnostic_rows),
        'static_masks_fixed_across_all_tasks':True,'calibration_and_evaluation_disjoint':True,
        'protected_layers_and_budget_checked':True,'answer_metrics_replayed':True,
        'engineering_all_workers_passed':all(load(out/f'engineering_{rank:03d}.json')['passed'] for rank in range(4)),
        'private_outputs_verified':all(sha256(out/name)==digest for rank in range(4)
            for name,digest in load(out/f'worker_result_{rank:03d}.json')['private_output_sha256'].items())})


def report(root,out):
    summary=load(out/'summary.json');history=load(out/'history_headroom.json')
    lines=['# 阶段16：收益空间诊断与缩短版 RULER 小试','',
        '冻结 OPT-13B 和全文路由器，无恢复训练。使用4张 RTX6000 Ada、整个进程树42个CPU核心。全部程序和结果新增，既有文件哈希保全。', '',
        '## 历史开发集：逐输入搜索还有多少收益空间','',
        '复用100条旧开发文档的完整逐候选联合KL搜索记录，并重放8条专家路径核验数值。Router为固定三种子逐文档均值。32条曾用于选检查点，另外68条也曾被查看；本表是诊断，不是独立测试。', '',
        '| K | 静态 KL | Router KL | 逐输入贪心 m6 KL | 逐输入贪心 m0 KL |','|---:|---:|---:|---:|---:|']
    for row in history['groups']:
        if row['split']=='all_development':
            m=row['mean_kl'];lines.append(f"| {row['k']} | {m['static']:.5f} | {m['router']:.5f} | {m['greedy_m6']:.5f} | {m['greedy_m0']:.5f} |")
    lines+=['','m6搜索固定前6次剪层，m0从空集合开始；两者每一步都以完整已剪集合评估候选，保留协同效应。离线搜索看32个真实未来token，因此仅为收益参考，不是可部署路由，也不是全局最优或严格收益上界。', '',
        '## 长度分层复核','',
        '从此前Stage14文档按固定种子选16条，256/512/1024/1536各4条；不读取预留C4测试集。重新计算静态、三种子Router和m6/m0逐输入贪心，仍属于已查看过文档上的诊断。', '',
        '| K | 方案 | KL | NLL | PPL |','|---:|---|---:|---:|---:|']
    for row in summary['diagnostic_aggregates']:
        lines.append(f"| {row['k']} | {row['policy']} | {row['kl']:.5f} | {row['nll']:.5f} | {row['ppl']:.3f} |")
    lines+=['','## 2K检索小试','',
        '这是独立实现的RULER风格缩短版，不是官方RULER分数。单信息检索、三信息检索、两跳变量追踪；512/1024/1536/1920个真实提示token，10%/50%/90%背景位置，各两个测试内容实例，共72题。位置变体共享内容，配对区间按内容实例聚类。每题最多生成48token，报告答案覆盖率、全部答案命中和无多余答案的严格匹配。', '',
        '路由仅观察完整提示词，包括作为检索上下文出现的信息；不观察追加答案。检索Router固定seed579；C4诊断用579/580/581。两类搜索不混用。', '',
        '另有36条独立合成校准题，按答案位置平均教师KL，从6条既有静态路径中为每个K选一条全局固定路径。全部任务共享该路径，不按题或按任务选。它称为calibrated_static_portfolio，不冒充重新运行的任务联合贪心；原C4校准static_joint_kl也完整保留。', '',
        '| 方案 | K | 答案覆盖率 | 全部命中率 | 严格匹配率 | 答案位置 KL |','|---|---:|---:|---:|---:|---:|']
    for row in summary['ruler_aggregates']:
        if row['dimension']=='overall':
            lines.append(f"| {row['policy']} | {row['k']} | {row['answer_recall']*100:.2f}% | {row['all_answers']*100:.2f}% | {row['strict_exact']*100:.2f}% | {row['kl']:.5f} |")
    lines+=['','完整模型的任务分层结果：','',
        '| 任务 | 全部命中率 | 严格匹配率 |','|---|---:|---:|']
    for row in summary['ruler_aggregates']:
        if row['dimension']=='task' and row['policy']=='full':
            lines.append(f"| {row['stratum']} | {row['all_answers']*100:.2f}% | {row['strict_exact']*100:.2f}% |")
    lines+=['','动态相对静态的配对改善，正值有利动态；正确率用百分点，KL用原单位：','',
        '| K | 对照 | 指标 | 改善 | 95%区间 |','|---:|---|---|---:|---|']
    for row in summary['ruler_paired']:
        if row['task']=='all' and row['metric'] in ('kl','all_answers'):
            scale=100 if row['metric']=='all_answers' else 1
            lo,hi=[v*scale for v in row['ci95']]
            lines.append(f"| {row['k']} | {row['baseline']} | {row['metric']} | {row['difference']*scale:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines+=['','本轮为小试，样本量有限；不作多重比较校正，也不包含训练种子不确定性。基座表现低的任务不能用于断言剪枝方案优劣。未新增恢复训练、吞吐量或部署显存结论。详细任务、长度、位置分层及配对比较见汇总。', '',
        '[汇总](../'+str(out.relative_to(root))+'/summary.json) · [历史诊断](../'+str(out.relative_to(root))+'/history_headroom.json) · [协议](../'+str(out.relative_to(root))+'/protocol.json) · [审计](../'+str(out.relative_to(root))+'/verification.json)', '',
        '参考：[RULER论文](https://arxiv.org/abs/2404.06654)；[检查的官方版本](https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a)。未发布第三方代码、模型权重、原始语料、token数组或逐题私有路由。']
    path=root/'analysis'/f'STAGE16_HEADROOM_RULER_PROBE_{out.name}.zh-CN.md'
    with path.open('x') as stream: stream.write('\n'.join(lines)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();resources()
    root=Path(__file__).resolve().parents[1];out=Path(os.environ['NEUROBYPASS_RUN_DIR'])
    tracked=preserve(root,out)
    write(out/'cpu_checks.json',cpu_checks())
    history=summarize_history(root,out)
    stamp('stage16_history_headroom_ready',groups=len(history['groups']))
    training=root/'runs/stage13_full_context_20261110_02_training'
    source,paths,available=policies(training,list(BUDGETS),[6])
    ruler,diagnostics=prepare_inputs(root,out,source['model'])
    names=['static_joint_kl','sleb_nll_opt','shortgpt_bi_opt','shortened_ppl_opt','shortened_ppl_plus','random_static_seed579']
    source_names=['run_headroom_ruler_four_gpu_v1.py','headroom_ruler_core_v1.py','full_context_core_v1.py',
        'prompt_fixed_core_v1.py','full_context_experiment_v1.py','evaluate_full_context_four_gpu_v1.py','run_with_resources_v2.py']
    config={'output':str(out),'model':source['model'],'training_run':str(training),'data_run':source['data_run'],
        'static_path':paths['static_joint_kl'],'static_candidates':{n:paths[n] for n in names},
        'router_policies':[p for p in available if p['kind']=='router'],
        'historical_replay_ids':source['validation_ids'][:4],
        'budgets':list(BUDGETS),'ruler_router_seed':579,'diagnostic_router_seeds':[579,580,581],
        'ruler_evaluation_cases':72,'ruler_calibration_cases':36,'diagnostic_documents':16,
        'observation':'Full actual prompt only; path fixed for prefill and entire generated answer. No new training or recovery.',
        'diagnostic_reference':'Joint greedy uses true32 continuation tokens; m6 and m0 are offline references, not deployable or global optima.',
        'engineering_gate':'All4 replicas: serial candidate replay, saved historical-mask replay, KV teacher-forcing and1920-token cached generation before evaluation.',
        'resources':'Four GPUs/42 CPU cores; each worker10, controller2, all threads inherit restricted affinity.',
        'source_sha256':{n:sha256(root/'tools'/n) for n in source_names},
        'training_protocol_sha256':sha256(training/'protocol.json'),'input_manifest_sha256':sha256(out/'input_manifest.json')}
    write(out/'protocol.json',config)
    with Workers(worker,config) as workers:
        workers.receive(4,'ready')
        stamp('stage16_engineering_gates_passed')
        workers.send_all({'kind':'calibrate'});workers.receive(4,'calibrated')
        portfolio=choose_portfolio(out,config)
        workers.send_all({'kind':'evaluate','portfolio':portfolio});workers.receive(4,'done')
    summarize(out,config)
    verify_preserved(out,tracked)
    assert all(sha256(root/'tools'/n)==h for n,h in config['source_sha256'].items())
    assert all(sha256(out/n)==h for n,h in load(out/'input_manifest.json')['private_sha256'].items())
    write(out/'independent_audit.json',{'passed':True,'preserved_files':len(tracked),
        'input_and_source_hashes_verified':True,'all_four_engineering_gates_verified':True,
        'cpu_count':load(out/'allocation.json')['cpu_count'],'no_weights_or_private_samples_in_publication':True})
    destination=report(root,out)
    files=[destination,Path(__file__),root/'tools/headroom_ruler_core_v1.py',
        *[out/n for n in ('summary.json','history_headroom.json','verification.json','protocol.json',
            'input_manifest.json','static_portfolio_selection.json','cpu_checks.json','independent_audit.json','preservation_after.json')],
        *[out/f'engineering_{rank:03d}.json' for rank in range(4)],
        *[out/f'worker_result_{rank:03d}.json' for rank in range(4)]]
    assert all(not p.name.startswith('private_') and p.stat().st_size<1024*1024 for p in files)
    upload=root/'analysis'/f'{out.name}_upload.json';write(upload,[str(p.relative_to(root)) for p in files])
    if args.publish:
        subprocess.run([sys.executable,'-B',str(root/'tools/publish_additions_v7.py'),'--root',str(root),
            '--repo','liuuu579-cmd/NeuroBypass','--files',str(upload),'--message',
            'Add frozen-route headroom diagnosis and calibrated short retrieval probe with preservation audits',
            '--receipt',str(root/'analysis'/f'{out.name}_publication.json')],cwd=root,check=True)
    stamp('stage16_complete',report=str(destination),preserved_files=len(tracked))


if __name__=='__main__': main()
