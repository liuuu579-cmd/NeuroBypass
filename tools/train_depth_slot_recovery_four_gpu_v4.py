"""K16 recovery v2: lower LR, same-batch overflow retries and verified baseline reuse."""
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from depth_slot_recovery_core_v1 import (DepthSlots, attach, log_probs, objective,
    routed, save_adapter)
from depth_slot_recovery_data_v1 import prepare
from recovery_optimizer_v3 import matched_step, forced_retry_check
from recovery_baseline_v2 import reuse_baseline
from evaluate_full_context_four_gpu_v1 import policies
from full_context_core_v1 import load_router, prompt_only, route_prompt
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, read_lines, setup_worker,
    thread_record, verify_preserved, write)
from prompt_fixed_core_v1 import continuation_logits
from prompt_permutation_core_v1 import score_fields
from stage1_benchmark_v2 import sha256, stamp


def load(path):
    return json.loads(Path(path).read_text())


def tensor(ids, device):
    return torch.as_tensor(np.asarray(ids).copy(), dtype=torch.long, device=device)[None]


def scalar_metrics(lp, teacher, targets):
    loss, ce, kl = objective(lp, teacher, targets)
    result = {'loss': loss.item(), 'nll': ce.item(), 'kl': kl.item()}
    if not all(math.isfinite(v) for v in result.values()) or result['kl'] < -1e-6:
        raise ValueError('Nonfinite or negative metrics')
    return result


def make_routes(model, router, policy, records, inputs, fresh, benchmark, device):
    routes = {}
    with torch.no_grad():
        for key, ids in inputs:
            if key in routes: raise ValueError('Duplicate input route key')
            route = policy['skips'] if router is None else route_prompt(model, router,
                tensor(ids, device), 16, policy['prefix'])
            if len(route) != 16 or len(set(route)) != 16 or set(route)&{0,1}:
                raise ValueError('Invalid route')
            routes[key] = sorted(route)
    return routes


def engineering(model, bank, router, data, ids, routes, device, out, rank):
    sample = max(ids, key=lambda s: data.rows[s]['prompt_length'])
    array, p = data[sample]; x = tensor(array, device); mask = routes[f'c4:{sample}']
    with torch.no_grad():
        # Compare identical target-only LM-head conventions.
        reference = F.log_softmax(continuation_logits(model, x, p, mask), -1)
        with routed(model, bank, mask): initial = log_probs(model, x, p)
        with routed(model, bank, [], adapters=False): teacher = log_probs(model, x, p)
    zero_error = (reference-initial).abs().max().item()
    if zero_error != 0: raise ValueError('Zero-initialized adapter changed the baseline')
    if router is not None:
        altered = x.clone(); altered[:, p:] = (altered[:, p:]+17)%model.config.vocab_size
        with torch.no_grad():
            route_a = route_prompt(model, router, prompt_only(x,p), 16, routes['_prefix'])
            route_b = route_prompt(model, router, prompt_only(altered,p), 16, routes['_prefix'])
        if route_a != route_b or sorted(route_a) != mask:
            raise ValueError('Targets reached the router or frozen route changed')
    bank.block_calls.clear()
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    with routed(model, bank, mask):
        loss,ce,kl = objective(log_probs(model,x,p), teacher, x[:,p:])
        scaler.scale(loss).backward()
    gradients = {n: float(v.grad.float().norm()) if v.grad is not None else None for n,v in bank.named_parameters()}
    if any(v.grad is None or not torch.isfinite(v.grad).all() for v in bank.parameters()):
        raise ValueError('Missing or invalid adapter gradient')
    if any(v <= 0 for n,v in gradients.items() if n.endswith('.b')):
        raise ValueError('An active slot has no B gradient')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Base model received trainable parameters or gradients')
    if router is not None and any(p.grad is not None or p.requires_grad for p in router.parameters()):
        raise ValueError('Router was not frozen')
    if set(bank.block_calls) != set(range(40))-set(mask):
        raise ValueError('Skipped original layers executed')
    bank.zero_grad(set_to_none=True)
    record = {'passed': True, 'zero_adapter_log_probs_max_abs_error': zero_error,
        'probe_prompt_tokens': p, 'targets_excluded_from_route': True,
        'executed_original_layers': sorted(bank.block_calls), 'skips': mask,
        'all24_slots_have_finite_nonzero_B_gradients': True, 'base_and_router_frozen': True,
        'trainable_parameters': bank.trainable_parameters, 'gradients': gradients,
        'initial_loss': loss.item(), 'initial_ce': ce.item(), 'initial_kl': kl.item(),
        'checkpointing': 'non-reentrant; same route context retained through backward',
        **thread_record()}
    write(out/f'engineering_{rank:03d}.json',record)
    return record


def evaluate(model, bank, data, dev_ids, fresh, benchmark, routes, device, stage, stream, rank, seed, arm):
    def emit(row):
        row.update({'rank':rank,'seed':seed,'arm':arm,'stage':stage})
        stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
    with torch.no_grad():
        for group, rows in [('development', dev_ids), ('fresh_c4', fresh)]:
            for row in rows:
                if group == 'development':
                    sample = row; array,p = data[sample]; key = f'c4:{sample}'
                else:
                    sample = row['sample']; array,p = row['ids'],row['prompt_tokens']; key = f'fresh:{sample}'
                x = tensor(array,device)
                with routed(model,bank,[],adapters=False): teacher = log_probs(model,x,p)
                if stage == 'before':
                    emit({'task':group,'sample':sample,'policy':'full','prompt_tokens':p,
                        'tokens':x.size(1)-p,**scalar_metrics(teacher,teacher,x[:,p:])})
                mask = routes[key]
                with routed(model,bank,mask,adapters=stage=='after'): student = log_probs(model,x,p)
                emit({'task':group,'sample':sample,'policy':'pruned','prompt_tokens':p,
                    'tokens':x.size(1)-p,'skips':mask,**scalar_metrics(student,teacher,x[:,p:])})
        for row in benchmark:
            mask = routes[f"{row['task']}:{row['sample']}"]
            for name,skips,enabled in ([('full',[],False)] if stage=='before' else []) + [('pruned',mask,stage=='after')]:
                scores = []
                for ids,n in zip(row['sequence_ids'],row['target_lengths']):
                    x = tensor(ids,device)
                    with routed(model,bank,skips,adapters=enabled):
                        # Preserve the original benchmark's full-sequence LM-head numerics.
                        h = model.model(x[:,:-1], attention_mask=torch.ones_like(x[:,:-1]), use_cache=False).last_hidden_state
                        lp = F.log_softmax(model.lm_head(h)[:,-n:].float(),-1)
                    scores.append(lp.gather(-1,x[:,-n:,None]).double().sum().item())
                emit({'task':row['task'],'sample':row['sample'],'document':row.get('document',row['sample']),
                    'policy':name,'skips':skips,**score_fields(row,scores)})


def worker(rank, config, inbox, messages):
    setup_worker(rank)
    torch.use_deterministic_algorithms(True)
    from transformers import AutoModelForCausalLM
    out = Path(config['output']); device = f'cuda:{rank}'
    arm = 'dynamic' if rank%2==0 else 'static'; seed = config['seeds'][rank//2]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    router_policy = next(p for p in config['router_policies'] if p['seed']==seed)
    policy = router_policy if arm=='dynamic' else {'skips':config['static_skips']}
    router = load_router(policy['checkpoint']['path'],device) if arm=='dynamic' else None
    data = Records(config['data_run']); manifest = load(out/'input_manifest.json')
    train, selection, dev = manifest['train_ids'], manifest['selection_ids'],manifest['development_ids']
    fresh = read_lines(out/'private_fresh_inputs.jsonl'); benchmark = read_lines(out/'private_benchmark_inputs.jsonl')
    inputs = []
    for sample in sorted(set(train)|set(dev)):
        array,p = data[sample]; inputs.append((f'c4:{sample}',array[:p]))
    inputs += [(f"fresh:{r['sample']}",r['ids'][:r['prompt_tokens']]) for r in fresh]
    inputs += [(f"{r['task']}:{r['sample']}",r['route_ids']) for r in benchmark]
    routes = make_routes(model,router,policy,data,inputs,fresh,benchmark,device)
    write(out/f'private_routes_{rank:03d}.json',routes)
    bank = DepthSlots(model.config.hidden_size,24,config['rank'],config['alpha'],seed).to(device)
    attach(model,bank)
    assert bank.trainable_parameters == 3932160
    check_routes = {**routes,'_prefix':router_policy['prefix']}
    checks = engineering(model,bank,router,data,train,check_routes,device,out,rank)
    messages.put({'kind':'progress','rank':rank,'phase':'engineering_passed','arm':arm,'seed':seed,
        'max_prompt':checks['probe_prompt_tokens'],'parameters':bank.trainable_parameters})
    teacher_cache = {}
    with torch.no_grad():
        for i,sample in enumerate(sorted(set(train)|set(selection))):
            array,p = data[sample]; x = tensor(array,device)
            with routed(model,bank,[],adapters=False): teacher_cache[sample] = log_probs(model,x,p).cpu()
            if (i+1)%100==0:
                messages.put({'kind':'progress','rank':rank,'phase':'teacher_cache','documents':i+1})
    write(out/f'teacher_cache_{rank:03d}.json',{'documents':len(teacher_cache),
        'bytes':sum(t.numel()*t.element_size() for t in teacher_cache.values()),
        'dtype':'float32','source':'All40 frozen OPT layers; adapters disabled; no targets observed by router.'})
    # Exercise real FP16 overflow handling before the optimizer is created.
    if config['probe']:
        sample = max(train,key=lambda s:data.rows[s]['prompt_length'])
        array,p = data[sample]; x = tensor(array,device)
        def forced_backward(scaler):
            with routed(model,bank,routes[f'c4:{sample}']):
                loss,ce,kl = objective(log_probs(model,x,p),teacher_cache[sample].to(device),x[:,p:])
                scaler.scale(loss).backward()
            return [[loss.item(),ce.item(),kl.item()]]
        retry_check = forced_retry_check(bank,forced_backward,config['lr'])
        write(out/f'forced_retry_{rank:03d}.json',retry_check)
        if not retry_check['passed']: raise ValueError('Forced retry replay failed; numerical details saved')
        messages.put({'kind':'progress','rank':rank,'phase':'forced_retry_check',
            'retries':retry_check['retry_attempts'],'passed':retry_check['passed']})
    quality_path = out/f'quality_{rank:03d}.jsonl' 
    with quality_path.open('x') as quality:
        if config['probe']:
            evaluate(model,bank,data,dev,fresh,benchmark,routes,device,'before',quality,rank,seed,arm)
        else:
            reuse_baseline(model,bank,data,dev,fresh,benchmark,routes,device,quality,rank,seed,arm,config,evaluate)
        messages.put({'kind':'progress','rank':rank,'phase':'before_evaluation_complete'})
        optimizer = torch.optim.AdamW(bank.parameters(),lr=config['lr'],weight_decay=0.0)
        scaler = torch.amp.GradScaler('cuda',init_scale=128)
        effective_batch = config['effective_batch']; batches = math.ceil(len(train)/effective_batch)
        total_updates = config['epochs']*batches; updates = 0; skipped_updates = 0; retry_attempts = 0
        best = None; start = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
        with (out/f'updates_{rank:03d}.log').open('x') as update_stream, (out/f'epochs_{rank:03d}.log').open('x') as epoch_stream:
            for epoch in range(1,config['epochs']+1):
                ordered = np.random.default_rng(seed*1000+epoch).permutation(train).tolist()
                totals = []
                for batch_start in range(0,len(ordered),effective_batch):
                    batch = ordered[batch_start:batch_start+effective_batch]
                    updates += 1
                    warmup = max(1,round(total_updates*.05))
                    multiplier = updates/warmup if updates <= warmup else .5*(1+math.cos(math.pi*(updates-warmup)/max(1,total_updates-warmup)))
                    optimizer.param_groups[0]['lr'] = config['lr']*multiplier
                    def backward_batch(current_scaler):
                        losses = []
                        for sample in batch:
                            array,p = data[sample]; x = tensor(array,device)
                            teacher = teacher_cache[sample].to(device)
                            with routed(model,bank,routes[f'c4:{sample}']):
                                loss,ce,kl = objective(log_probs(model,x,p),teacher,x[:,p:])
                                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                                current_scaler.scale(loss/len(batch)).backward()
                            losses.append([loss.item(),ce.item(),kl.item()])
                        return losses
                    def on_retry(record):
                        update_stream.write(json.dumps({'kind':'overflow_retry','epoch':epoch,
                            'update':updates,'batch_ids':batch,**record})+'\n'); update_stream.flush()
                        messages.put({'kind':'progress','rank':rank,'phase':'same_batch_overflow_retry',
                            'epoch':epoch,'update':updates,**record})
                    losses,norm,retries = matched_step(bank,optimizer,scaler,backward_batch,on_retry)
                    retry_attempts += retries
                    finite = True
                    values = np.mean(losses,axis=0).tolist(); totals.extend(losses)
                    entry = {'epoch':epoch,'update':updates,'batch_ids':batch,'loss':values[0],
                        'ce':values[1],'kl':values[2],'lr':optimizer.param_groups[0]['lr'],
                        'grad_norm':float(norm) if finite else None,'scale':scaler.get_scale(),'skipped_updates':skipped_updates,
                        'kind':'successful_update','retry_attempts':retries}
                    update_stream.write(json.dumps(entry,allow_nan=False)+'\n'); update_stream.flush()
                    if updates%10==0 or config['probe']:
                        messages.put({'kind':'progress','rank':rank,'phase':'training','epoch':epoch,
                            'update':updates,'total_updates':total_updates,'loss':values[0],'kl':values[2],
                            'seconds':time.monotonic()-start})
                validation = []
                with torch.no_grad():
                    for sample in selection:
                        array,p = data[sample]; x = tensor(array,device)
                        with routed(model,bank,routes[f'c4:{sample}']):
                            result = scalar_metrics(log_probs(model,x,p),teacher_cache[sample].to(device),x[:,p:])
                        validation.append(result)
                selected_loss = float(np.mean([r['loss'] for r in validation]))
                path = out/f'private_adapter_{rank:03d}_epoch{epoch:03d}.pth'
                save_adapter(path,bank,{'epoch':epoch,'selection_loss':selected_loss,'seed':seed,'arm':arm,
                    'protocol_sha256':sha256(out/'protocol.json'),'adapter':'executed-depth-slot-qv-lora-v1'})
                record = {'epoch':epoch,'path':str(path),'sha256':sha256(path),'selection_loss':selected_loss,
                    'selection_nll':float(np.mean([r['nll'] for r in validation])),
                    'selection_kl':float(np.mean([r['kl'] for r in validation])),
                    'train_loss_ce_kl':np.mean(totals,axis=0).tolist()}
                epoch_stream.write(json.dumps(record,allow_nan=False)+'\n'); epoch_stream.flush()
                if best is None or selected_loss < best['selection_loss']: best = record
                messages.put({'kind':'progress','rank':rank,'phase':'epoch_complete',**record})
        duration = time.monotonic()-start
        adam_steps = sorted({int(s['step']) for s in optimizer.state.values()})
        if adam_steps != [total_updates]: raise ValueError('Adam successful-update budget mismatch')
        bank.load_state_dict(torch.load(best['path'],map_location=device,weights_only=True)['state_dict'])
        sample = selection[0]; array,p = data[sample]; x = tensor(array,device)
        with torch.no_grad(), routed(model,bank,routes[f'c4:{sample}']): replay = log_probs(model,x,p).clone()
        # Save/reload exact replay, including original-model adapter-off restoration.
        roundtrip = out/f'private_adapter_{rank:03d}_replay.pth'
        save_adapter(roundtrip,bank,{'source_checkpoint_sha256':best['sha256']})
        bank.load_state_dict(torch.load(roundtrip,map_location=device,weights_only=True)['state_dict'])
        with torch.no_grad(), routed(model,bank,routes[f'c4:{sample}']): replay2 = log_probs(model,x,p)
        replay_error = (replay-replay2).abs().max().item()
        if replay_error != 0: raise ValueError('Adapter checkpoint replay mismatch')
        with torch.no_grad(), routed(model,bank,[],adapters=False): restored = log_probs(model,x,p)
        teacher_error = (restored.cpu()-teacher_cache[sample]).abs().max().item()
        if teacher_error != 0: raise ValueError('Adapter-off teacher changed after training')
        evaluate(model,bank,data,dev,fresh,benchmark,routes,device,'after',quality,rank,seed,arm)
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Base gradient invariant failed')
    result = {'passed':True,'rank':rank,'arm':arm,'seed':seed,'best':best,
        'updates':updates,'skipped_updates':skipped_updates,'retry_attempts':retry_attempts,
        'adam_steps':adam_steps,'train_documents_per_epoch':len(train),
        'target_tokens_seen':len(train)*32*config['epochs'],'trainable_parameters':bank.trainable_parameters,
        'seconds_training_and_selection':duration,'peak_training_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'peak_training_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'adapter_replay_max_abs_error':replay_error,'teacher_restoration_max_abs_error':teacher_error,
        'quality_sha256':sha256(quality_path),'frozen_routes_sha256':sha256(out/f'private_routes_{rank:03d}.json'),
        **thread_record()}
    write(out/f'worker_result_{rank:03d}.json',result)
    messages.put({'kind':'done','rank':rank,'result':result})


def audit_and_summarize(out,config,manifest):
    results = [load(out/f'worker_result_{r:03d}.json') for r in range(4)]
    all_rows = [row for r in range(4) for row in read_lines(out/f'quality_{r:03d}.jsonl')]
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row['task'],row['arm'],row['seed'],row['stage'],row['policy'])
        grouped[key].append(row)
    expected_inputs = manifest['development_documents']+manifest['fresh_documents']+sum(manifest['benchmark_records'].values())
    for rank,result in enumerate(results):
        assert result['passed'] and result['updates'] == config['epochs']*math.ceil(manifest['train_documents']/config['effective_batch'])
        assert result['skipped_updates'] == 0, 'Overflow-skipped updates break the intended matched update budget'
        assert result['target_tokens_seen'] == manifest['train_documents']*32*config['epochs']
        assert result['trainable_parameters'] == 3932160
        assert len([r for r in all_rows if r['rank']==rank]) == expected_inputs*3
        assert sha256(result['best']['path']) == result['best']['sha256']
        assert sha256(out/f'quality_{rank:03d}.jsonl') == result['quality_sha256']
        rows = [r for r in all_rows if r['rank']==rank and r['policy']=='pruned']
        before = {(r['task'],r['sample']):r['skips'] for r in rows if r['stage']=='before'}
        after = {(r['task'],r['sample']):r['skips'] for r in rows if r['stage']=='after'}
        assert before == after and len(before)==expected_inputs
        if rank%2:
            assert all(v==sorted(config['static_skips']) for v in before.values())
        threads = result['thread_cpu_ids'].values()
        assert len(result['cpu_ids'])==10 and all(set(t)<=set(result['cpu_ids']) for t in threads)
    for a,b in ((0,1),(2,3)):
        assert [r['batch_ids'] for r in read_lines(out/f'updates_{a:03d}.log') if r['kind']=='successful_update'] == [r['batch_ids'] for r in read_lines(out/f'updates_{b:03d}.log') if r['kind']=='successful_update']
    def metrics(rows):
        task = rows[0]['task']
        if task in ('development','fresh_c4'):
            return {'kl':float(np.mean([r['kl'] for r in rows])), 'nll':float(np.mean([r['nll'] for r in rows])),
                'ppl':math.exp(float(np.mean([r['nll'] for r in rows])))}
        if task=='wikitext_conditional':
            nll = sum(r['nll_sum'] for r in rows)/sum(r['scored_tokens'] for r in rows)
            return {'nll':nll,'ppl':math.exp(nll)}
        key = 'correct' if task=='winogrande' else 'correct_norm'
        return {'accuracy':float(np.mean([r[key] for r in rows]))}
    aggregates = [{'task':key[0],'arm':key[1],'seed':key[2],'stage':key[3],'policy':key[4],
        'records':len(rows),**metrics(rows)} for key,rows in sorted(grouped.items())]
    comparisons = []
    # Average seeds inside each input, then paired bootstrap documents/questions.
    # Wiki windows resample their source articles, rather than dependent windows.
    for task in sorted({r['task'] for r in all_rows}):
        for comparison,left,right in [
                ('dynamic_recovery',('dynamic','before','pruned'),('dynamic','after','pruned')),
                ('static_recovery',('static','before','pruned'),('static','after','pruned')),
                ('recovered_dynamic_vs_static',('static','after','pruned'),('dynamic','after','pruned'))]:
            by_side = []
            for arm,stage,policy in (left,right):
                by_sample = defaultdict(list)
                for row in all_rows:
                    if (row['task'],row['arm'],row['stage'],row['policy']) == (task,arm,stage,policy):
                        by_sample[row['sample']].append(row)
                assert all(len(v)==2 for v in by_sample.values())
                by_side.append(by_sample)
            assert by_side[0].keys()==by_side[1].keys()
            samples = sorted(by_side[0]); clusters = defaultdict(list)
            for sample in samples:
                row = by_side[0][sample][0]
                clusters[row.get('document',sample)].append(sample)
            cluster_keys = list(clusters)
            metric_names = ['kl','nll'] if task in ('development','fresh_c4') else ['ppl'] if task=='wikitext_conditional' else ['accuracy']
            for metric in metric_names:
                def estimate(selected):
                    vals = []
                    for side in by_side:
                        rows = [r for sample in selected for r in side[sample]]
                        vals.append(metrics(rows)[metric])
                    # Positive always means the right-hand condition improves quality.
                    return vals, vals[1]-vals[0] if metric=='accuracy' else vals[0]-vals[1]
                values,delta = estimate(samples)
                rng = np.random.default_rng(150581)
                boot = []
                for _ in range(1000):
                    indices = rng.integers(0,len(cluster_keys),len(cluster_keys))
                    selected = [s for i in indices for s in clusters[cluster_keys[i]]]
                    boot.append(estimate(selected)[1])
                comparisons.append({'task':task,'comparison':comparison,'metric':metric,
                    'left':left,'right':right,'left_mean':values[0],'right_mean':values[1],
                    'improvement':delta,'ci95':np.quantile(boot,[.025,.975]).tolist(),'input_clusters':len(clusters)})
    summary = {'aggregates':aggregates,'comparisons':comparisons,
        'interpretation':'Two paired seeds and fixed reviewed benchmark subsets. Bootstrap clusters inputs/articles, averages both seeds inside inputs; CIs do not estimate training-seed uncertainty. Depth-slot LoRA variant, not standard per-original-layer LoRA.',
        'selection':'Mean CE+KL on32 original dev identities only. Development aggregates include selection identities; fresh C4 and task benchmarks do not select checkpoints.'}
    write(out/'summary.json',summary)
    audit = {'passed':True,'workers':4,'quality_rows':len(all_rows),'inputs_per_worker':expected_inputs,
        'same_routes_before_after':True,'same_data_order_within_seed':True,'same_parameter_and_active_budgets':True,
        'no_skipped_optimizer_updates':True,'all_threads_within_worker10core_affinity':True,
        'base_and_router_frozen':True,'teacher_and_adapter_replay_exact':True}
    write(out/'verification.json',audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--baseline-run',type=Path,default=Path(__file__).resolve().parents[1]/'runs/stage15_depth_slot_recovery_20261111_01')
    args = parser.parse_args(); os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    tracked = preserve(root,out)
    training = root/'runs/stage13_full_context_20261110_02_training'
    source,paths,available = policies(training,[16],[6])
    manifest = prepare(root,out,source,args.probe)
    filenames = ['train_depth_slot_recovery_four_gpu_v4.py','recovery_optimizer_v3.py','recovery_baseline_v2.py',
        'train_depth_slot_recovery_four_gpu_v1.py','depth_slot_recovery_core_v1.py',
        'depth_slot_recovery_data_v1.py','full_context_core_v1.py','full_context_data_v1.py',
        'full_context_experiment_v1.py','evaluate_full_context_four_gpu_v1.py','prompt_fixed_core_v1.py',
        'prompt_permutation_core_v1.py','run_with_resources_v2.py']
    config = {'model':source['model'],'data_run':source['data_run'],'output':str(out),'probe':args.probe,
        'seeds':[579,580],'k':16,'kept':24,'m':6,'rank':8,'alpha':16,'dropout':0,
        'epochs':4 if args.probe else 8,'effective_batch':2 if args.probe else 8,'lr':5e-5,
        'baseline_run':str(args.baseline_run.resolve()),
        'baseline_quality_sha256':{str(i):sha256(args.baseline_run/f'quality_{i:03d}.jsonl') for i in range(4)},
        'deterministic_algorithms':True,'cublas_workspace_config':':4096:8',
        'revision':'v4: deterministic attention gradients, from fresh initialization in all arms, lower LR5e-5; overflow attempts retry exact same batch after scale reduction, no Adam or schedule advance; complete800 successful updates; verified exact replay of existing pre-training scores.',
        'input_manifest_sha256':sha256(out/'input_manifest.json'),
        'source_training_protocol_sha256':sha256(training/'protocol.json'),
        'static_skips':paths['static_joint_kl'][:16],
        'router_policies':[p for p in available if p['name']=='m06' and p['seed'] in (579,580)],
        'source_sha256':{n:sha256(root/'tools'/n) for n in filenames},
        'observation':'Frozen Stage13 full-visible-prompt-v1; all prompt token+position embeddings; no targets/options; one fixed route for entire continuation.',
        'recovery':'Frozen FP16 OPT; FP32 q/v LoRA parameters;24 executed-depth slots shared across routes; rank8 alpha16;3,932,160 total and active parameters in each arm. Pilot parameter-tying architecture, not standard per-original-layer LoRA.',
        'loss':'Mean CE on32 real continuation targets + untempered forward KL(full frozen OPT || student), equal weights; one document microbatch, effective8; all prompt positions participate in contextual gradient, loss only on32 continuation targets.',
        'optimization':'AdamW5e-5, weight_decay0, cosine with5% warmup, clip1; FP16 frozen base, FP32 adapters; GradScaler128 with same-batch retries; non-reentrant per-retained-block checkpointing; base eval/dropout disabled.',
        'selection':'Each epoch select minimum mean CE+KL on original32 development identities. Static/dynamic within each seed share init/data order/update/token/parameter budgets. No route relabeling.',
        'limits':'Two-seed pilot, K16 only, no joint router training; no measured recovered deployment speed or memory saving; fresh archive cohort deduplicated from Stage13/14, not a pretraining-clean benchmark.'}
    write(out/'protocol.json',config)
    with Workers(worker,config) as workers: workers.receive(4,'done')
    result = audit_and_summarize(out,config,manifest)
    verify_preserved(out,tracked)
    stamp('recovery_complete',**result)


if __name__=='__main__': main()
