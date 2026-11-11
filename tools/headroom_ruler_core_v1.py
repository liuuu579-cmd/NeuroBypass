"""Append-only Stage16 inputs, paired summaries, and RULER-inspired short tasks.

This is an independently implemented 2K adaptation, not the official RULER
generator or an official RULER score. No third-party source is vendored.
"""
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256

TASKS = ('niah_single', 'niah_multi', 'variable_tracking')
LENGTHS = (512, 1024, 1536, 1920)
DEPTHS = (.1, .5, .9)
BUDGETS = (10, 12, 16)
UPSTREAM = 'https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a'
NOISE = ('The grass is green. The sky is blue. The sun is yellow. '
         'The river flows past a quiet village. A bird sits in the tree. ')


def digest_ids(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest()


def jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def paired(values, clusters=None, seed=160579):
    """Mean difference and cluster bootstrap; exploratory, no multiplicity correction."""
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    if clusters is None:
        clusters = list(range(len(a)))
    groups = defaultdict(list)
    for v, c in zip(a, clusters):
        groups[str(c)].append(float(v))
    # All current input clusters have equal size; resample whole cases.
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    draws = np.random.default_rng(seed).integers(0, len(sums), (2000, len(sums)))
    means = sums[draws].sum(1)/counts[draws].sum(1)
    return {'difference': float(a.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'records': len(a), 'clusters': len(groups), 'fraction_positive': float((a > 0).mean())}


def summarize_history(root, out):
    training = root/'runs/stage13_full_context_20261110_02_training'
    dev = root/'runs/stage13_full_context_20261110_02_development'
    cfg = json.loads((training/'protocol.json').read_text())
    inputs = {r['sample']: r for r in read_lines(dev/'private_inputs.jsonl') if r['task']=='c4_development'}
    labels, sources = {}, [training/'protocol.json', dev/'protocol.json', dev/'private_inputs.jsonl']
    for m in (0, 6):
        for path in sorted(training.glob(f'm{m:02d}_labels_*.jsonl')):
            sources.append(path)
            for row in read_lines(path):
                if row['split']=='validation':
                    assert row['sample'] in inputs and row['m']==m
                    record = inputs[row['sample']]
                    assert row['prompt_length']==len(record['route_ids'])
                    assert row['sequence_sha256']==digest_ids(record['sequence_ids'][0])
                    for k in BUDGETS:
                        step = next(s for s in row['steps'] if s['step']==k)
                        assert step['previous_skips']==row['ordered_skips'][:k-1]
                        assert step['selected_layer']==row['ordered_skips'][k-1]
                        assert min(step['candidate_costs'], key=lambda v:(v[1], v[0]))[0]==step['selected_layer']
                        labels[row['sample'], m, k] = float(step['joint_continuation_kl'])
    metrics = {}
    for path in sorted(dev.glob('quality_*.jsonl')):
        sources.append(path)
        for row in read_lines(path):
            if row['task']!='c4_development' or row['condition']!='matched' or row['k'] not in BUDGETS:
                continue
            if row['policy']=='static_joint_kl' or row['policy']=='m06':
                key = row['sample'], row['k'], row['policy'], row['seed']
                assert key not in metrics
                metrics[key] = row['teacher_kl_sum']/row['target_tokens']
    private, summary = [], []
    for sample, record in sorted(inputs.items()):
        for k in BUDGETS:
            row = {'sample': sample, 'k': k, 'prompt_tokens': len(record['route_ids']),
                   'selection': sample in cfg['selection_ids'],
                   'static': metrics[sample, k, 'static_joint_kl', None],
                   'router': float(np.mean([metrics[sample,k,'m06',s] for s in (579,580,581)])),
                   'greedy_m6': labels[sample,6,k], 'greedy_m0': labels[sample,0,k]}
            row['best_known_with_future'] = min(row['static'],row['greedy_m6'],row['greedy_m0'])
            private.append(row)
    for split in ('all_development', 'selection32', 'other68'):
        for k in BUDGETS:
            rows = [r for r in private if r['k']==k and (split=='all_development' or r['selection']==(split=='selection32'))]
            assert len(rows)=={'all_development':100,'selection32':32,'other68':68}[split]
            summary.append({'split':split,'k':k,'documents':len(rows),
                'mean_kl':{n:float(np.mean([r[n] for r in rows])) for n in
                    ('static','router','greedy_m6','greedy_m0','best_known_with_future')},
                'static_minus_greedy_m6':paired([r['static']-r['greedy_m6'] for r in rows]),
                'static_minus_greedy_m0':paired([r['static']-r['greedy_m0'] for r in rows]),
                'router_minus_greedy_m6':paired([r['router']-r['greedy_m6'] for r in rows]),
                'greedy_m6_minus_m0':paired([r['greedy_m6']-r['greedy_m0'] for r in rows])})
    jsonl(out/'private_history_paired.jsonl',private)
    result = {'groups':summary,'source_sha256':{str(p.relative_to(root)):sha256(p) for p in sources},
        'scope':'Reanalysis of previously reviewed development documents. 32 used to choose checkpoints; other68 also previously reviewed.',
        'reference':'Greedy paths use32 true future tokens, not deployable and not a global optimum or guaranteed upper bound. Best-known includes static itself.',
        'router':'Full-context m6, mean per-document KL across fixed seeds579/580/581; no inference ensemble.',
        'recovery':'None. This isolates frozen-base path selection, not the Stage15 adapter results.'}
    write(out/'history_headroom.json',result)
    return result


def make_task(tokenizer, task, length, depth, replica, split):
    # Depth variants share content, while different lengths/replicas/splits are independent.
    split_number = {'calibration':1,'evaluation':2,'engineering':3}[split]
    seed = 160579 + split_number*1000000 + TASKS.index(task)*10000 + LENGTHS.index(length)*100 + replica
    rng = np.random.default_rng(seed)
    def word():
        return ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),4))
    names = []
    while len(names)<8:
        name=word()
        if name not in names and name not in ('AAAA','BBBB','CCCC'): names.append(name)
    numbers = rng.choice(np.arange(100000,999999),8,replace=False).tolist()
    if task=='niah_single':
        header = ('Read the notes and answer by copying the requested number.\n'
            'Example note: The special magic number for AAAA is 123456.\n'
            'Question: What is the special magic number for AAAA?\nAnswer: 123456\n\nNotes:\n')
        needles = [f'\nThe special magic number for {names[0]} is {numbers[0]}.\n']
        answers = [str(numbers[0])]
        footer = f'\nQuestion: What is the special magic number for {names[0]}?\nAnswer:'
        positions = [depth]
    elif task=='niah_multi':
        header = ('Read the notes and list the requested numbers in the question order.\n'
            'Example notes: The special magic number for AAAA is 123456. '
            'The special magic number for BBBB is 654321.\n'
            'Question: What are the special magic numbers for AAAA and BBBB?\nAnswer: 123456, 654321\n\nNotes:\n')
        needles = [f'\nThe special magic number for {names[j]} is {numbers[j]}.\n' for j in range(3)]
        answers = [str(v) for v in numbers[:3]]
        footer = f'\nQuestion: What are the special magic numbers for {names[0]}, {names[1]} and {names[2]}?\nAnswer:'
        positions = [max(.01,depth-.07),depth,min(.99,depth+.07)]
    else:
        header = ('Track the assignments and list all variables with the requested value.\n'
            'Example: VAR AAAA = 123456. VAR BBBB = VAR AAAA. VAR CCCC = VAR BBBB.\n'
            'Question: Which variables have value 123456?\nAnswer: AAAA, BBBB, CCCC\n\nAssignments:\n')
        needles = [f'\nVAR {names[0]} = {numbers[0]}.\n',
                   f'\nVAR {names[1]} = VAR {names[0]}.\n',f'\nVAR {names[2]} = VAR {names[1]}.\n']
        answers = names[:3]
        footer = f'\nQuestion: Which variables have value {numbers[0]}?\nAnswer:'
        positions = [max(.01,depth-.07),depth,min(.99,depth+.07)]
    # Add unrelated key/value distractors in every task; none is a target key.
    distractors = '\n'.join(f'The special magic number for {names[j]} is {numbers[j]}.' for j in range(3,6))+'\n'
    header_ids = tokenizer.encode(header+distractors,add_special_tokens=False)
    footer_ids = tokenizer.encode(footer,add_special_tokens=False)
    needle_ids = [tokenizer.encode(v,add_special_tokens=False) for v in needles]
    budget = length-1-len(header_ids)-len(footer_ids)-sum(map(len,needle_ids))
    assert budget>50
    noise = tokenizer.encode(NOISE*100,add_special_tokens=False)[:budget]
    assert len(noise)==budget
    ids, spans, cursor = [tokenizer.bos_token_id]+header_ids,[],0
    for position, piece in zip(positions,needle_ids):
        stop=int(position*budget)
        ids+=noise[cursor:stop]
        spans.append([len(ids),len(ids)+len(piece)])
        ids+=piece;cursor=stop
    ids+=noise[cursor:]+footer_ids
    assert len(ids)==length
    target=tokenizer.encode(' '+', '.join(answers),add_special_tokens=False)
    assert length+len(target)<=2048 and length+48<=2048
    assert all(answer in tokenizer.decode(ids) for answer in answers)
    return {'task':task,'split':split,'length':length,'depth':depth,'replica':replica,
        'case':f'{split}/{task}/{length}/{replica}','seed':seed,'prompt_ids':ids,'target_ids':target,
        'answers':answers,'needle_token_spans':spans,'prompt_sha256':digest_ids(ids),
        'answer_kind':'variables' if task=='variable_tracking' else 'numbers'}


def answer_metrics(text, answers, kind):
    lower=text.casefold()
    recall=sum(a.casefold() in lower for a in answers)/len(answers)
    pattern=r'(?<![A-Za-z])[A-Z]{4}(?![A-Za-z])' if kind=='variables' else r'(?<!\d)\d{6}(?!\d)'
    found=re.findall(pattern,text)
    strict=set(found)==set(answers) and len(found)==len(answers)
    return {'answer_recall':recall,'all_answers':int(recall==1),'strict_exact':int(strict)}


def prepare_inputs(root, out, model, engineering=False):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,use_fast=False,local_files_only=True)
    rows=[]
    for split,count in ([('engineering',1)] if engineering else [('calibration',1),('evaluation',2)]):
        for task in TASKS:
            for length in ((512,1920) if engineering else LENGTHS):
                for replica in range(count):
                    for depth in ((.5,) if engineering else DEPTHS):
                        row=make_task(tokenizer,task,length,depth,replica,split)
                        row['sample']=len(rows);rows.append(row)
    assert len({r['prompt_sha256'] for r in rows})==len(rows)
    jsonl(out/'private_ruler_inputs.jsonl',rows)
    # Replicate a length-stratified selection from Stage14, not reserved C4 test.
    source=root/'runs/stage14_observation_windows_20261111_01/private_inputs.jsonl'
    old=read_lines(source);rng=np.random.default_rng(160580);fresh=[]
    for length in (256,512,1024,1536):
        group=[r for r in old if r['prompt_tokens']==length]
        picks=rng.choice(len(group),1 if engineering else 4,replace=False)
        for pick in picks:
            r=group[int(pick)]
            assert digest_ids(r['ids'])==r['ids_sha256'] and len(r['ids'])==length+32
            fresh.append({**r,'task':'c4_diagnostic','source_sample':r['sample'],'sample':len(fresh)})
    jsonl(out/'private_diagnostic_inputs.jsonl',fresh)
    manifest={'ruler_records':len(rows),'ruler_split_counts':{s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},
        'diagnostic_documents':len(fresh),'diagnostic_lengths':{str(p):sum(r['prompt_tokens']==p for r in fresh) for p in (256,512,1024,1536)},
        'private_sha256':{p.name:sha256(p) for p in (out/'private_ruler_inputs.jsonl',out/'private_diagnostic_inputs.jsonl')},
        'diagnostic_source_sha256':sha256(source),'sampling_seed':160580,'upstream':UPSTREAM,
        'adaptation':'Independent RULER-inspired2K probe: one-shot completion, six-digit values, three requested keys or2-hop variable chain, synthetic noise, explicit10/50/90% insertion positions. Not official RULER.',
        'input_policy':'Router sees every actual prompt token and no appended answer. The needle necessarily contains the answer as retrieval context.',
        'calibration_policy':'Disjoint synthetic calibration cases choose one fixed existing static portfolio path perK by mean answer-position teacher KL, not evaluation accuracy. Router frozen and C4-trained.',
        'limitations':'Stage14 diagnostic documents previously evaluated; synthetic probe is exploratory; no official RULER or long-context beyond2K claim. No reserved C4 test used.'}
    write(out/'input_manifest.json',manifest)
    return rows,fresh


def cpu_checks():
    assert answer_metrics('123456',['123456'],'numbers')['strict_exact']==1
    assert answer_metrics('123456, 654321',['123456'],'numbers')['strict_exact']==0
    assert answer_metrics('AAAA, BBBB',['AAAA','BBBB'],'variables')['strict_exact']==1
    assert paired([1.,2.],['same','same'])['ci95']==[1.5,1.5]
    return {'passed':True,'extra_answer_penalty_checked':True,'whole_case_bootstrap_checked':True}
