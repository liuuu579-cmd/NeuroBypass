"""Independent CPU route-multiset audit and paired input/path mechanism results."""
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from finalize_horizontal_quality_v2 import paired_interval
from prompt_permutation_core_v1 import assignments, load_reference, metric_value, policy_key, record_key
from stage1_benchmark_v2 import sha256, write_json


def finalize(root, run):
    config=json.loads((run/'protocol.json').read_text())
    completion=json.loads((run/'completion.json').read_text())
    assert completion['state']=='completed'
    assert sha256(config['private_inputs'])==config['private_inputs_sha256']
    assert sha256(run/'assignments.json')==config['assignments_sha256']
    for name,digest in config['source_sha256'].items():assert sha256(root/'tools'/name)==digest
    for name,digest in config['source_quality_sha256'].items():assert sha256(Path(config['source'])/name)==digest
    for policy in config['policies']:assert sha256(policy['checkpoint']['path'])==policy['checkpoint']['sha256']
    tracked=json.loads((run/'preservation_before.json').read_text())
    for name,digest in tracked.items():assert sha256(name)==digest,name
    assert json.loads((run/'preservation_after.json').read_text())['preserved_files']==len(tracked)
    inputs=[json.loads(s) for s in Path(config['private_inputs']).read_text().splitlines()]
    records={record_key(r):r for r in inputs}
    stored=json.loads((run/'assignments.json').read_text());assert stored==assignments(inputs)
    # Assignments must stay invariant under arbitrary answer/label mutation.
    assert assignments([{**r,'gold':999,'sequence_ids':[],'scores':[]} for r in reversed(inputs)])==stored
    mapping={(r['task'],r['sample'],r['permutation_seed']):r for r in stored}
    if not config['probe']:reference=load_reference(config['source'],config['policies'])
    else:reference=None
    allocation=json.loads((run/'allocation.json').read_text());assert allocation['cpu_count']==42
    nreplays,nprobes=0,0
    for rank in range(4):
        probe=json.loads((run/f'worker_probe_{rank:03d}.json').read_text())
        assert probe['cpu_ids']==allocation['cpu_ids'][rank*10:(rank+1)*10]
        assert all(set(cpus)<=set(probe['cpu_ids']) for cpus in probe['thread_cpu_ids'].values())
        for item in probe['likelihood_probes']:
            assert item['absolute_error']<=item['tolerance']==2e-4;nprobes+=1
        for item in probe['matched_score_replays']:
            assert item['max_score_error'] is None if config['probe'] else item['max_score_error']<=2e-4
            nreplays+=1
    policies={policy_key(p):p for p in config['policies']};seen=set();groups=defaultdict(dict)
    for path in sorted(run.glob('permutation_*.jsonl')):
        for line in path.open():
            row=json.loads(line);pk=policy_key(row);policy=policies[pk];rk=record_key(row);record=records[rk]
            unique=rk+pk+(row['permutation_seed'],);assert unique not in seen;seen.add(unique)
            assigned=mapping[rk+(row['permutation_seed'],)]
            assert row['donor_sample']==assigned['donor_sample']!=row['sample']
            assert row['document_sha256']==record['document_sha256'] and row['document']==record.get('document')
            if row['task']=='wikitext_conditional':assert assigned['donor_document']!=row['document']
            for field in ['target_lengths','normalization_lengths']:assert row[field]==record[field]
            for name in ['matched_skipped_layers','shuffled_skipped_layers']:
                assert row[name]==sorted(set(row[name])) and len(row[name])==row['k']
                assert all(2<=i<40 for i in row[name])
                assert set(policy['prefix'])<=set(row[name])
            assert row['same_mask']==(row['matched_skipped_layers']==row['shuffled_skipped_layers'])
            if reference is not None:
                own=reference[rk+pk];donor=reference[(row['task'],row['donor_sample'])+pk]
                assert row['matched_skipped_layers']==own['skipped_layers']
                assert row['shuffled_skipped_layers']==donor['skipped_layers']
                assert row['matched_scores']==own['scores']
            for arm in ['matched','shuffled']:
                scores=np.asarray(row[arm+'_scores']);assert np.isfinite(scores).all() and (scores<=0).all()
                assert len(scores)==len(record['sequence_ids'])
                if record['gold'] is None:
                    assert row[arm]['nll_sum']==-scores[0] and row[arm]['scored_tokens']==record['target_lengths'][0]
                else:
                    gold=record['gold'];prediction=int(scores.argmax())
                    assert row[arm]['gold']==gold and row[arm]['prediction']==prediction
                    assert row[arm]['correct']==int(prediction==gold)
                    if row['task']!='winogrande':
                        predicted=int((scores/np.asarray(record['normalization_lengths'])).argmax())
                        assert row[arm]['prediction_norm']==predicted and row[arm]['correct_norm']==int(predicted==gold)
            if row['same_mask']:assert row['matched_scores']==row['shuffled_scores']
            groups[(row['task'],)+pk+(row['permutation_seed'],)][row['sample']]=row
    assert len(seen)==config['expected_rows']==completion['rows']
    n_by_task=Counter(r['task'] for r in inputs);details=[];rng=np.random.default_rng(579)
    vectors={}
    for key,values in sorted(groups.items(),key=lambda item:str(item[0])):
        task,arm,seed,k,permutation=key
        assert len(values)==n_by_task[task]
        ordered=[values[s] for s in sorted(values)]
        # Stronger than checking marginal layer frequencies: full-mask multiset.
        assert Counter(tuple(r['matched_skipped_layers']) for r in ordered)==Counter(tuple(r['shuffled_skipped_layers']) for r in ordered)
        a=np.asarray([metric_value({**r,'nll_sum':r['matched'].get('nll_sum'),**r['matched']}) for r in ordered],dtype=float)
        b=np.asarray([metric_value({**r,'nll_sum':r['shuffled'].get('nll_sum'),**r['shuffled']}) for r in ordered],dtype=float)
        tokens=np.asarray([r['matched']['scored_tokens'] for r in ordered]) if task=='wikitext_conditional' else None
        documents=np.asarray([r['document'] for r in ordered])
        vectors[key]=(a,b,tokens,documents)
        difference,lo,hi=paired_interval(a,b,tokens,documents,rng)
        details.append({'task':task,'arm':arm,'seed':seed,'k':k,'permutation_seed':permutation,
            'records':len(ordered),'metric':'nll_per_token' if tokens is not None else 'accuracy',
            'matched':float(a.sum()/tokens.sum()) if tokens is not None else float(a.mean()),
            'shuffled':float(b.sum()/tokens.sum()) if tokens is not None else float(b.mean()),
            'matched_minus_shuffled':difference,'ci95_low':lo,'ci95_high':hi,
            'same_mask_fraction':float(np.mean([r['same_mask'] for r in ordered])),
            'bootstrap_unit':'article' if tokens is not None else 'question',
            'favors_matched':'negative' if tokens is not None else 'positive'})
    aggregate=[]
    for task in sorted(n_by_task):
        for arm in ['pure','hybrid']:
            for k in [10,12]:
                keys=[key for key in vectors if key[:2]==(task,arm) and key[3]==k]
                # Average model/permutation scores per question first; do not
                # inflate independent sample count by either repetition axis.
                a=np.mean([vectors[key][0] for key in keys],axis=0)
                b=np.mean([vectors[key][1] for key in keys],axis=0)
                tokens,documents=vectors[keys[0]][2:]
                difference,lo,hi=paired_interval(a,b,tokens,documents,rng)
                item={'task':task,'arm':arm,'k':k,'model_seeds':3,'permutations':3,
                    'records':n_by_task[task],'metric':'nll_per_token' if tokens is not None else 'accuracy',
                    'matched':float(a.sum()/tokens.sum()) if tokens is not None else float(a.mean()),
                    'shuffled':float(b.sum()/tokens.sum()) if tokens is not None else float(b.mean()),
                    'matched_minus_shuffled':difference,'ci95_low':lo,'ci95_high':hi,
                    'favors_matched':'negative' if tokens is not None else 'positive'}
                if tokens is not None:item.update({'matched_conditional_ppl':math.exp(item['matched']),'shuffled_conditional_ppl':math.exp(item['shuffled'])})
                aggregate.append(item)
    write_json(run/'verification.json',{'status':'passed','probe':config['probe'],'verified_rows':len(seen),
        'exact_mask_multisets_preserved':True,'wiki_cross_article_donors':True,'counterfactual_assignment_test':True,
        'matched_score_replays':nreplays,'likelihood_probes':nprobes,'preserved_files':len(tracked),'cpu_budget':42})
    write_json(run/'summary.json',{'probe':config['probe'],'aggregate':aggregate,'per_seed_permutation':details,
        'elapsed_seconds':completion['elapsed_seconds'],'bootstrap_replicates':2000,
        'limits':['Question/article bootstrap conditional on trained models and frozen permutations; does not capture retraining uncertainty.',
            'Multiple exploratory intervals uncorrected; permutations and model seeds are not extra independent questions.']})
    return config,aggregate,details


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--publish',action='store_true');args=p.parse_args()
    root,run=args.root.resolve(),args.run.resolve();config,aggregate,details=finalize(root,run)
    if config['probe']:print(json.dumps({'state':'probe_audited','rows':config['expected_rows']}));return
    csv_paths=[]
    for suffix,rows in [('aggregate',aggregate),('per_seed',details)]:
        path=root/f'analysis/stage9_permutation_{suffix}_20261109.csv';csv_paths.append(path)
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader();writer.writerows(rows)
    text=['# 输入与路径匹配的置换消融','',
        '冻结已有路由器和检查点，K=10/12、纯动态/固定6层混合、各3训练种子、3次预定置换。只打乱输入与路径配对，精确保留每个策略的整条路径多重集合。Wiki供体来自不同文章。matched复用阶段7并通过GPU重放核对。','',
        '选择题采用PIQA/HellaSwag acc_norm与WinoGrande acc，越高越好；Wiki为条件NLL/PPL，越低越好。先对每题平均三个模型和三次置换，再按题/文章bootstrap2000次；不把重复轴当成额外样本。','',
        '| 任务 | 方法 | K | 正常匹配 | 路径置换 | 匹配减置换 | 配对95%区间 |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in aggregate:
        scale=1 if r['task']=='wikitext_conditional' else 100
        text.append(f"| {r['task']} | {r['arm']} | {r['k']} | {r['matched']*scale:.5f} | {r['shuffled']*scale:.5f} | {r['matched_minus_shuffled']*scale:.5f} | [{r['ci95_low']*scale:.5f}, {r['ci95_high']*scale:.5f}] |")
    text +=['','Wiki表内是每token NLL，PPL另存CSV。选择题差值单位为百分点。区间未校正多重比较，是固定模型/置换条件下的不确定性描述。','',
        '不同输入产生不同路径本身不证明动态匹配有效。本实验直接检查配对收益；允许供体路径恰好与原路径相同，不能强迫所有输入改用不同mask而改变原路径分布。供体不看标签或选项，不训练或择优。','',
        '这些任务已用于阶段7结果审阅，本轮为机制消融，不是新独立测试。其它学习改动若依据本轮结果开发，仍需新的最终评价。固定层数m的配套训练与文献基线核查单独开展，不混入此表。','',f'来源：`{run.relative_to(root)}`。完整逐题记录保留本地；发布汇总、置换对应、源代码和验收，不上传语料、权重或token数组。','']
    report=root/'analysis/STAGE9_PERMUTATION_RESULTS_20261109.zh-CN.md'
    with report.open('x') as stream:stream.write('\n'.join(text))
    if args.publish:
        files=[str(p.relative_to(root)) for p in run.iterdir() if p.is_file() and p.suffix in {'.json','.log'} and not p.name.startswith('private_')]
        files +=[str(p.relative_to(root)) for p in root.glob('tools/*prompt_permutation*.py')]
        files +=[str(report.relative_to(root)),*[str(p.relative_to(root)) for p in csv_paths],
            'analysis/STAGE9_PERMUTATION_PROTOCOL_20261109.zh-CN.md']
        manifest=root/'analysis/stage9_permutation_completed_upload_20261109.json';files.append(str(manifest.relative_to(root)))
        write_json(manifest,sorted(set(files)))
        subprocess.run([sys.executable,'-B',str(root/'tools/publish_additions_v7.py'),'--root',str(root),'--repo','liuuu579-cmd/NeuroBypass',
            '--files',str(manifest),'--message','Complete frozen input-path permutation mechanism ablation',
            '--receipt',str(root/'analysis/stage9_permutation_completed_publication_20261109.json')],check=True,cwd=root)
    print(json.dumps({'report':str(report),'aggregate_rows':len(aggregate)}),flush=True)


if __name__=='__main__':main()
