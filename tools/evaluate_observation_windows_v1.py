"""Frozen Stage 13 router crossover on long C4: only router observation varies."""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_full_context_four_gpu_v1 import logs, policies
from full_context_core_v1 import load_router, prompt_embeddings, prompt_only
from full_context_data_v1 import Records
from full_context_experiment_v1 import (Workers, preserve, read_lines, resources, setup_worker,
    thread_record, verify_preserved, write)
from stage1_benchmark_v2 import sha256, stamp

LENGTHS = (256, 512, 1024, 1536)
BUDGETS = (10, 12)
SEEDS = (579, 580, 581)
SHUFFLES = (2701, 2702, 2703)
WINDOWS = ('full', 'first128', 'last128', 'last128_rebased')
CONDITIONS = WINDOWS + tuple(f'shuffled_{s}' for s in SHUFFLES)
PER_LENGTH = 80


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ids_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def prepare(root, out, train, cfg, probe):
    from transformers import AutoTokenizer
    data = Records(cfg['data_run'])
    source = Path(cfg['data_run'])/'data_manifest.json'
    if sha256(source) != cfg['data_manifest_sha256']:
        raise ValueError('Stage 13 manifest digest mismatch')
    if probe:
        # Smoke-test on already evaluated development inputs, never the main cohort.
        available = {r['sample']: r['prompt_length'] for r in data.rows if r['sample'] in cfg['validation_ids']}
        rows = []
        for p in LENGTHS:
            chosen = sorted((s for s in available if available[s] >= p), key=lambda s: (available[s], s))[:4]
            if len(chosen) < 4:
                # Shorter source record may be replaced by an exact longer context
                # from the archived document only in the main run, not this probe.
                p = 1024 if p > 1024 else p
                chosen = sorted((s for s in available if available[s] >= p), key=lambda s: (available[s], s))[:4]
            if len(chosen) < 4:
                p = 768
                chosen = sorted((s for s in available if available[s] >= p), key=lambda s: (available[s], s))[:4]
            if len(chosen) != 4: raise ValueError('Insufficient existing development records for smoke test')
            if any(r['prompt_tokens'] == p for r in rows): continue
            for s in chosen:
                ids, _ = data[s]
                rows.append({'sample': len(rows), 'prompt_tokens': p, 'source_sample': s,
                    'ids': ids[:p+32].tolist()})
        # Separate exact replay of old development metrics; no truncation.
        old = root/'runs/stage13_full_context_20261110_02_development'
        replay_sample = max(available, key=available.get)
        replay_ids, p = data[replay_sample]
        match = next(r for path in sorted(old.glob('quality_*.jsonl')) for r in read_lines(path)
            if (r['task'], r['sample'], r['policy'], r['seed'], r['k'], r['condition'])
            == ('c4_development', replay_sample, 'm06', 579, 10, 'matched'))
        write(out/'private_replay.json', {'ids': replay_ids.tolist(), 'p': p, 'reference': match})
        provenance = {'source': 'Stage 13 development, smoke test only', 'source_manifest_sha256': sha256(source)}
    else:
        corpus = Path(data.manifest['corpus'])
        if sha256(corpus) != data.manifest['corpus_sha256']:
            raise ValueError('Archived C4 digest mismatch')
        start = corpus.stat().st_size*95//100
        prior_end = max(r['source_end_byte'] for r in data.rows)
        if start <= prior_end: raise ValueError('Test byte range overlaps Stage 13 sources')
        old_regions = data.manifest.get('excluded_stage1_byte_regions', [])
        original = load(data.manifest['source_manifest'])
        if sha256(data.manifest['source_manifest']) != data.manifest['source_manifest_sha256']:
            raise ValueError('Stage 3 document manifest digest mismatch')
        old_regions += original['excluded_stage1_byte_regions']
        if any(end > start for _, end in old_regions):
            raise ValueError('Test byte range overlaps recorded historical calibration/eval regions')
        seen_docs = {r['record_sha256'] for r in data.rows}
        seen_leads = {r['record_lead_sha256'] for r in data.rows}
        seen_prefix = {r['prefix_sha256'] for r in data.rows}
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'], use_fast=True, local_files_only=True)
        slow = AutoTokenizer.from_pretrained(cfg['model'], use_fast=False, local_files_only=True)
        rng = np.random.default_rng(140579)
        rows, rejected = [], Counter()
        with corpus.open('rb') as stream:
            stream.seek(start); stream.readline()
            offsets = []
            while True:
                offset = stream.tell(); raw = stream.readline()
                if not raw: break
                if len(raw) >= 350: offsets.append(offset)
            rng.shuffle(offsets)
            cursor = 0
            # Allocate longest first. Every accepted row is a different document.
            for p in reversed(LENGTHS):
                count = 0
                while count < PER_LENGTH and cursor < len(offsets):
                    offset = offsets[cursor]; cursor += 1
                    stream.seek(offset); raw = stream.readline()
                    text = raw.decode('utf-8').strip()
                    normalized = ' '.join(text.split())
                    doc = digest(normalized)
                    lead = digest(' '.join(normalized.casefold().split()[:32]))
                    if doc in seen_docs or lead in seen_leads:
                        rejected['document_or_lead_duplicate'] += 1; continue
                    content = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=p+31)
                    if len(content) != p+31:
                        rejected['too_short_for_current_length'] += 1; continue
                    ids = [tokenizer.bos_token_id]+content
                    prefix = ids_digest(ids[:128])
                    if prefix in seen_prefix:
                        rejected['token_prefix_duplicate'] += 1; continue
                    slow_ids = slow.encode(text, add_special_tokens=False, truncation=True, max_length=p+31)
                    if slow_ids != content or slow.bos_token_id != ids[0]:
                        raise ValueError('Fast and Stage 13 slow tokenizer disagree on selected document')
                    seen_docs.add(doc); seen_leads.add(lead); seen_prefix.add(prefix)
                    rows.append({'sample': len(rows), 'prompt_tokens': p, 'source_offset': offset,
                        'document_sha256': doc, 'lead_sha256': lead, 'ids': ids})
                    count += 1
                if count != PER_LENGTH: raise ValueError(f'Insufficient distinct long C4 records for {p}: {count}')
        provenance = {'source': 'Archived C4 last 5% by byte, new documents held out from Stage 13 router fitting and selection',
            'source_manifest_sha256': sha256(source), 'corpus_sha256': data.manifest['corpus_sha256'],
            'region_start': start, 'stage13_last_source_end': prior_end, 'sampling_seed': 140579,
            'selection': 'Shuffle eligible line offsets, allocate longest group first, one prefix/continuation per distinct source document.',
            'dedup': 'Exact normalized document, first32 normalized casefolded words and first128 token hashes against all Stage 13 documents and this cohort.',
            'limits': 'Same archival corpus; unknown OPT pretraining and older feature-only data overlaps; near duplicates can remain.',
            'rejected': dict(rejected), 'fast_slow_tokenizer_all_selected_exact': True}
    for r in rows:
        if not 256 <= r['prompt_tokens'] <= 2016 or len(r['ids']) != r['prompt_tokens']+32:
            raise ValueError('Invalid prompt/target split')
        r['ids_sha256'] = ids_digest(r['ids'])
    with (out/'private_inputs.jsonl').open('x') as stream:
        for r in rows: stream.write(json.dumps(r)+'\n')
    provenance.update({'documents': len(rows), 'prompt_length_histogram': dict(Counter(r['prompt_tokens'] for r in rows)),
        'private_inputs_sha256': sha256(out/'private_inputs.jsonl'), 'scored_tokens_per_document': 32})
    write(out/'input_manifest.json', provenance)
    return rows


def donors(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['prompt_tokens']].append(row['sample'])
    result = {}
    for p, group in sorted(groups.items()):
        for s in SHUFFLES:
            rng = np.random.default_rng(s+p*1000)
            for _ in range(1000):
                permuted = rng.permutation(group).tolist()
                if all(a != b for a, b in zip(group, permuted)): break
            else: raise ValueError('Could not generate within-length derangement')
            result.update({(a, s): b for a, b in zip(group, permuted)})
    return result


def observations(model, ids, p):
    prompt = prompt_only(ids, p)
    emb = prompt_embeddings(model, prompt)
    return {'full': emb, 'first128': emb[:, :128], 'last128': emb[:, -128:],
        'last128_rebased': prompt_embeddings(model, prompt[:, -128:])}


def result_metrics(lp, lq, score, targets):
    kl = (lp.exp()*(lp-lq)).sum(-1).double().sum().item()/targets
    result = {'kl': kl, 'nll': -score/targets}
    if not all(math.isfinite(v) for v in result.values()) or kl < -1e-6:
        raise ValueError('Invalid continuation metrics')
    return result


def regression(model, router, policy, out, records, rank, device):
    row = records[rank]
    p = row['prompt_tokens']; x = torch.tensor(row['ids'], device=device)[None]
    values = observations(model, x, p)
    altered = x.clone(); altered[:, p:] = 3
    checks = observations(model, altered, p)
    if any(not torch.equal(values[key], checks[key]) for key in WINDOWS):
        raise ValueError('Targets affect observations')
    if not torch.equal(values['first128'], prompt_embeddings(model, x[:, :128])):
        raise ValueError('First128 differs from original truncation semantics')
    tail = x.clone(); tail[:, p-1] = (tail[:, p-1]+1) % model.config.vocab_size
    changed = observations(model, tail, p)
    if not torch.equal(values['first128'], changed['first128']) or torch.equal(values['last128'], changed['last128']):
        raise ValueError('Tail intervention failed to reach intended observations')
    short = observations(model, x, 128)
    if any(not torch.equal(short['full'], short[key]) for key in WINDOWS):
        raise ValueError('Window conditions differ when complete prompt is exactly128 tokens')
    if torch.equal(values['last128'], values['last128_rebased']):
        raise ValueError('Tail position sensitivity contrast has no position change')
    audit = {'observed_tokens': {k: v.shape[1] for k,v in values.items()}, 'target_exclusion': True,
        'first128_legacy_equivalent': True, 'tail_token_reaches_tail': True, 'short_control_exact': True,
        'tail_position_control_distinct': True, **thread_record()}
    if (out/'private_replay.json').exists():
        replay = load(out/'private_replay.json'); ref = replay['reference']; p = replay['p']
        x = torch.tensor(replay['ids'], device=device)[None]
        mask = sorted(router.route(observations(model, x, p)['full'], policy['k'], policy['prefix'])[0].tolist())
        lp, _ = logs(model, replay['ids'], 32, [])
        lq, score = logs(model, replay['ids'], 32, mask)
        got = result_metrics(lp, lq, score, 32)
        error = max(abs(got['kl']-ref['teacher_kl_sum']/32), abs(got['nll']-ref['nll_sum']/32))
        if mask != ref['skipped_layers'] or error > 1e-6:
            raise ValueError('Stage 13 full-context replay differs')
        audit['stage13_replay_max_abs_error'] = error
    return audit


def worker(rank, config, inbox, messages):
    setup_worker(rank)
    from transformers import AutoModelForCausalLM
    device, out = f'cuda:{rank}', Path(config['out'])
    model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True, torch_dtype=torch.float16,
        attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    routes, routers = {}, {}
    for pol in config['policies']:
        path = pol['checkpoint']['path']
        if path not in routers: routers[path] = load_router(path, device)
    records = read_lines(out/'private_inputs.jsonl')
    mapping = donors(records)
    with torch.inference_mode():
        first = config['policies'][0]
        assert (first['seed'], first['k']) == (579, 10)
        write(out/f'worker_probe_{rank:03d}.json', regression(model, routers[first['checkpoint']['path']], first,
            out, records, rank, device))
        for row in records:
            p = row['prompt_tokens']; x = torch.tensor(row['ids'], device=device)[None]
            windows = observations(model, x, p); encoded = {}
            for pol in config['policies']:
                path = pol['checkpoint']['path']; router = routers[path]
                for name in WINDOWS:
                    key = path, name
                    if key not in encoded: encoded[key] = router.encode(windows[name])
                    route = router.route_encoded(encoded[key], pol['k'], pol['prefix'])[0].tolist()
                    if len(route) != pol['k'] or len(set(route)) != pol['k'] or route[:6] != pol['prefix'] or min(route) < 2:
                        raise ValueError('Frozen route contract failed')
                    routes[row['sample'], pol['k'], pol['seed'], name] = tuple(sorted(route))
        write(out/f'private_routes_{rank:03d}.json', [[*key, list(value)] for key,value in sorted(routes.items())])
        local = records[rank::4]
        with (out/f'private_scores_{rank:03d}.jsonl').open('x') as stream:
            for i, row in enumerate(local):
                sample, p = row['sample'], row['prompt_tokens']
                lp, baseline = logs(model, row['ids'], 32, [])
                cache = {(): result_metrics(lp, lp, baseline, 32)}
                def measure(mask):
                    if mask not in cache:
                        lq, score = logs(model, row['ids'], 32, mask)
                        cache[mask] = result_metrics(lp, lq, score, 32)
                    return cache[mask]
                def emit(k, seed, name, donor, mask):
                    record = {'sample': sample, 'p': p, 'k': k, 'seed': seed, 'condition': name,
                        'donor': donor, 'skips': mask, 'base_input_sha256': row['ids_sha256'], **measure(mask)}
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                emit(0, None, 'unpruned', sample, ())
                for k in BUDGETS:
                    emit(k, None, 'static_joint_kl', sample, tuple(sorted(config['global_path'][:k])))
                    for s in SEEDS:
                        for name in WINDOWS: emit(k, s, name, sample, routes[sample,k,s,name])
                        for perm in SHUFFLES:
                            donor = mapping[sample,perm]
                            emit(k, s, f'shuffled_{perm}', donor, routes[donor,k,s,'full'])
                stream.flush()
                if i%8 == 0 or i+1 == len(local):
                    messages.put({'kind': 'progress', 'phase': 'window_scores', 'rank': rank,
                        'completed': i+1, 'total': len(local), 'prompt_tokens': p})
    messages.put({'kind': 'done', 'rank': rank})


def audit(out, config, records):
    rows = {r['sample']: r for r in records}
    for r in records:
        if ids_digest(r['ids']) != r['ids_sha256']: raise ValueError('Input hash mismatch')
    mapping, registry = donors(records), None
    allocation = load(out/'allocation.json')
    assert allocation['cpu_count'] == len(set(allocation['cpu_ids'])) == 42
    assert len(set(allocation['gpu_indices'])) == 4
    for rank in range(4):
        entries = load(out/f'private_routes_{rank:03d}.json')
        current = {tuple(r[:4]): tuple(r[4]) for r in entries}
        assert len(current) == len(entries) == len(records)*len(BUDGETS)*len(SEEDS)*len(WINDOWS)
        if registry is None: registry = current
        elif current != registry: raise ValueError('Workers disagree on registered routes')
        check = load(out/f'worker_probe_{rank:03d}.json')
        cpus = allocation['cpu_ids'][rank*10:(rank+1)*10]
        assert check['cpu_ids'] == cpus and check['threads'] == 10
        assert all(set(c) <= set(cpus) for c in check['thread_cpu_ids'].values())
        if config['probe']: assert check['stage13_replay_max_abs_error'] <= 1e-6
    scores = {}
    for rank in range(4):
        sample_ids = {r['sample'] for r in records[rank::4]}
        unique = {}
        for r in read_lines(out/f'private_scores_{rank:03d}.jsonl'):
            sample,k,s,name = key = (r['sample'],r['k'],r['seed'],r['condition'])
            if key in scores or sample not in sample_ids: raise ValueError('Invalid score row coverage')
            assert r['base_input_sha256'] == rows[sample]['ids_sha256'] and r['p'] == rows[sample]['prompt_tokens']
            donor = sample
            if name == 'unpruned': mask = ()
            elif name == 'static_joint_kl': mask = tuple(sorted(config['global_path'][:k]))
            elif name in WINDOWS: mask = registry[sample,k,s,name]
            else:
                donor = mapping[sample,int(name.split('_')[-1])]
                assert rows[donor]['prompt_tokens'] == rows[sample]['prompt_tokens'] and donor != sample
                mask = registry[donor,k,s,'full']
            assert r['donor'] == donor and tuple(r['skips']) == mask and len(mask) == k
            score = (r['kl'],r['nll'])
            if (sample,mask) in unique: assert score == unique[sample,mask]
            unique[sample,mask] = score; scores[key] = r
    expected = {(r['sample'],k,s,n) for r in records for k in BUDGETS for s in SEEDS for n in CONDITIONS}
    expected |= {(r['sample'],k,None,'static_joint_kl') for r in records for k in BUDGETS}
    expected |= {(r['sample'],0,None,'unpruned') for r in records}
    if scores.keys() != expected: raise ValueError('Incomplete or extra score conditions')
    write(out/'verification.json', {'passed': True, 'documents': len(records), 'score_rows': len(scores),
        'same_base_input_for_every_condition': True, 'target_exclusion': True, 'route_registry_consensus': True,
        'within_exact_length_derangements': True, 'equal_masks_equal_scores': True,
        'stage13_metric_replay': config['probe'], 'gpu_count': 4, 'cpu_count': 42})
    return scores


def summarize(out, records, scores):
    groups = {'all': records, **{str(p): [r for r in records if r['prompt_tokens'] == p]
        for p in sorted({r['prompt_tokens'] for r in records})}}
    rows, rng = [], np.random.default_rng(140581)
    for group, documents in groups.items():
        n = len(documents)
        # Stratified overall estimate: equal sample counts at all lengths. Each
        # bootstrap draw holds the pre-specified length composition fixed.
        strata = defaultdict(list)
        for i, r in enumerate(documents): strata[r['prompt_tokens']].append(i)
        indices = np.concatenate([rng.choice(inds, size=(5000,len(inds))) for inds in strata.values()], axis=1)
        for k in BUDGETS:
            def row(doc, seed, cond): return scores[doc['sample'],k,seed,cond]
            for metric in ['kl', 'nll']:
                def values(cond):
                    if cond == 'static_joint_kl': return np.array([row(d,None,cond)[metric] for d in documents])
                    cs = [f'shuffled_{s}' for s in SHUFFLES] if cond == 'shuffled_mean3' else [cond]
                    return np.array([np.mean([row(d,s,c)[metric] for s in SEEDS for c in cs]) for d in documents])
                means = {a: values(a) for a in [*WINDOWS,'shuffled_mean3','static_joint_kl']}
                for a,b in [('full','first128'),('full','last128'),('last128','first128'),
                        ('last128','last128_rebased'),('full','shuffled_mean3'),('full','static_joint_kl')]:
                    diff = means[b]-means[a]
                    rows.append({'length': group,'documents': n,'k': k,'metric': metric,
                        'preferred_if_positive': a,'reference': b,'mean_a': float(means[a].mean()),
                        'mean_b': float(means[b].mean()),'difference_b_minus_a': float(diff.mean()),
                        'ci95': np.quantile(diff[indices].mean(1), [.025,.975]).tolist()})
    changes = []
    for group, docs in groups.items():
        for k in BUDGETS:
            for a,b in [('full','first128'),('full','last128'),('last128','first128'),('last128','last128_rebased')]:
                pairs = [(set(scores[d['sample'],k,s,a]['skips']),set(scores[d['sample'],k,s,b]['skips']))
                    for d in docs for s in SEEDS]
                changes.append({'length': group,'k': k,'a': a,'b': b,'document_seed_pairs': len(pairs),
                    'changed_mask_fraction': float(np.mean([x!=y for x,y in pairs])),
                    'mean_replaced_layers': float(np.mean([len(x-y) for x,y in pairs]))})
    write(out/'summary.json', {'comparisons': rows, 'route_changes': changes,
        'method': 'Average seeds and shuffle replicates within document. 5000 paired document bootstrap draws stratified by exact prompt length. Unadjusted 95% intervals.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args(); resources()
    start = time.monotonic(); root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    training = root/'runs/stage13_full_context_20261110_02_training'
    if load(training/'status.json')['state'] != 'completed' or not load(training/'independent_audit.json')['passed']:
        raise ValueError('Training provenance is not verified')
    tracked = preserve(root,out)
    cfg, paths, selected = policies(training,list(BUDGETS),[6])
    selected = [p for p in selected if p['kind'] == 'router']
    assert {(p['seed'],p['k']) for p in selected} == {(s,k) for s in SEEDS for k in BUDGETS}
    records = prepare(root,out,training,cfg,args.probe)
    config = {'out': str(out), 'model': cfg['model'], 'policies': selected, 'global_path': paths['static_joint_kl'],
        'probe': args.probe, 'training_protocol_sha256': sha256(training/'protocol.json'),
        'input_manifest_sha256': sha256(out/'input_manifest.json'),
        'windows': {'full':'all prompt tokens','first128':'positions0:128','last128':'last128 tokens with original position embeddings',
            'last128_rebased':'same tail tokens with positions reset to0:128'},
        'primary': 'Long C4 held out from Stage 13 router; KL contrasts full vs first128 and full vs last128 at K10. K12, NLL, length strata and other contrasts secondary.',
        'base': 'One identical full prompt plus32-target teacher-forced sequence for all conditions; only routing input changes.',
        'shuffled': 'Full-window masks from fixed seed derangements within the exact same prompt length; same K and router seed.',
        'source_sha256': {p.name: sha256(p) for p in [Path(__file__),root/'tools/full_context_core_v1.py',
            root/'tools/evaluate_full_context_four_gpu_v1.py',root/'tools/full_context_experiment_v1.py']}}
    write(out/'protocol.json',config)
    stamp('window_observation_inputs_ready', documents=len(records),probe=args.probe)
    with Workers(worker,config) as workers: workers.receive(4,'done')
    scores = audit(out,config,records)
    summarize(out,records,scores)
    verify_preserved(out,tracked)
    stamp('window_observation_complete', documents=len(records),rows=len(scores),elapsed=time.monotonic()-start)


if __name__ == '__main__': main()
