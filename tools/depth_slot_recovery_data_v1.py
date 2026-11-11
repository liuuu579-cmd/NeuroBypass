"""Freeze recovery-pilot inputs; exclude Stage 13 and Stage 14 source documents."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from full_context_data_v1 import Records
from full_context_experiment_v1 import read_lines, write
from stage1_benchmark_v2 import sha256


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def token_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest()


def prepare(root, output, source, probe):
    from transformers import AutoTokenizer
    data = Records(source['data_run'])
    if sha256(Path(source['data_run'])/'data_manifest.json') != source['data_manifest_sha256']:
        raise ValueError('Stage 13 input manifest changed')
    train, selection, dev = source['train_ids'], source['selection_ids'], source['validation_ids']
    if probe:
        # Include a maximum-length backward pass and several short examples.
        train = sorted(train, key=lambda i: (data.rows[i]['prompt_length'], i))
        train = [train[-1], train[0], train[len(train)//2], train[-2]]
        selection, dev = selection[:2], dev[:2]
    assert set(selection) <= set(dev) and not set(train)&set(dev)
    corpus = Path(data.manifest['corpus'])
    if sha256(corpus) != data.manifest['corpus_sha256']:
        raise ValueError('Archived corpus changed')
    prior14 = root/'runs/stage14_observation_windows_20261111_01/private_inputs.jsonl'
    prior = read_lines(prior14)
    docs = {r['record_sha256'] for r in data.rows} | {r['document_sha256'] for r in prior}
    leads = {r['record_lead_sha256'] for r in data.rows} | {r['lead_sha256'] for r in prior}
    prefixes = {r['prefix_sha256'] for r in data.rows} | {token_digest(r['ids'][:128]) for r in prior}
    rows, rejected = [], Counter()
    if not probe:
        tokenizer = AutoTokenizer.from_pretrained(source['model'], use_fast=True, local_files_only=True)
        slow = AutoTokenizer.from_pretrained(source['model'], use_fast=False, local_files_only=True)
        start = corpus.stat().st_size*95//100
        assert start > max(r['source_end_byte'] for r in data.rows)
        original = json.loads(Path(data.manifest['source_manifest']).read_text())
        assert sha256(data.manifest['source_manifest']) == data.manifest['source_manifest_sha256']
        assert all(end <= start for _,end in original['excluded_stage1_byte_regions'])
        rng = np.random.default_rng(150579)
        with corpus.open('rb') as stream:
            stream.seek(start); stream.readline()
            offsets = []
            while True:
                offset = stream.tell(); raw = stream.readline()
                if not raw: break
                if len(raw) >= 350: offsets.append(offset)
            rng.shuffle(offsets); cursor = 0
            for p in (1536, 1024, 512, 256):
                count = 0
                while cursor < len(offsets) and count < 32:
                    offset = offsets[cursor]; cursor += 1
                    stream.seek(offset); text = stream.readline().decode('utf-8').strip()
                    normalized = ' '.join(text.split())
                    document = digest(normalized)
                    lead = digest(' '.join(normalized.casefold().split()[:32]))
                    if document in docs or lead in leads:
                        rejected['document_or_lead_duplicate'] += 1; continue
                    content = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=p+31)
                    if len(content) != p+31:
                        rejected['short_for_length'] += 1; continue
                    ids = [tokenizer.bos_token_id]+content
                    prefix = token_digest(ids[:128])
                    if prefix in prefixes:
                        rejected['prefix_duplicate'] += 1; continue
                    assert content == slow.encode(text, add_special_tokens=False, truncation=True, max_length=p+31)
                    docs.add(document); leads.add(lead); prefixes.add(prefix)
                    rows.append({'sample': len(rows), 'prompt_tokens': p, 'ids': ids,
                        'document_sha256': document, 'lead_sha256': lead, 'source_offset': offset,
                        'ids_sha256': token_digest(ids)})
                    count += 1
                if count != 32: raise ValueError('Insufficient distinct recovery evaluation documents')
    with (output/'private_fresh_inputs.jsonl').open('x') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    benchmark_source = root/'runs/stage13_full_context_20261110_02_horizontal/private_inputs.jsonl'
    rng = np.random.default_rng(150580)
    all_benchmark = read_lines(benchmark_source)
    benchmark = []
    for task in ('piqa', 'hellaswag', 'winogrande', 'wikitext_conditional'):
        group = [r for r in all_benchmark if r['task'] == task]
        count = (2 if probe else 128) if task == 'wikitext_conditional' else (2 if probe else 256)
        selected = sorted(rng.choice(len(group), count, replace=False))
        benchmark.extend(group[i] for i in selected)
    with (output/'private_benchmark_inputs.jsonl').open('x') as stream:
        for row in benchmark: stream.write(json.dumps(row)+'\n')
    manifest = {'train_ids': train, 'selection_ids': selection, 'development_ids': dev,
        'train_documents': len(train), 'selection_documents': len(selection),
        'development_documents': len(dev), 'fresh_documents': len(rows),
        'fresh_length_histogram': dict(Counter(r['prompt_tokens'] for r in rows)),
        'benchmark_records': dict(Counter(r['task'] for r in benchmark)),
        'source_manifest_sha256': source['data_manifest_sha256'],
        'stage14_private_inputs_sha256': sha256(prior14), 'corpus_sha256': data.manifest['corpus_sha256'],
        'benchmark_source_sha256': sha256(benchmark_source),
        'new_private_input_sha256': {p.name: sha256(p) for p in
            (output/'private_fresh_inputs.jsonl', output/'private_benchmark_inputs.jsonl')},
        'fresh_sampling_seed': 150579, 'benchmark_sampling_seed': 150580,
        'fresh_policy': 'Archived C4 last5%; normalized document, first32 casefolded words, first128 tokens deduplicated against all Stage13, Stage14 and this cohort; longest lengths allocated first; 32 target tokens.',
        'selection_policy': 'Only original32 development identities select adapter epoch by mean CE+forward KL; no fresh/benchmark selection.',
        'limitations': 'Fresh cohort held out from these experiments, but same archive with unknown pretraining and older feature-only overlaps; near duplicates possible. MC/Wiki subset is from previously reviewed benchmark populations. Two seeds, recovery pilot, no final-paper claim.'}
    write(output/'input_manifest.json', manifest)
    return manifest
