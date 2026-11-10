"""Recover the original grouped documents; create new variable-length examples."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256, write_json

LENGTHS = [64, 128, 256, 512, 1024, 2016]


def prepare(root, output, base_model):
    from transformers import AutoTokenizer
    source = root / 'runs/stage3_prompt_fixed_pilot_20261104_01/data_manifest.json'
    old = json.loads(source.read_text())
    corpus = Path(old['corpus'])
    if sha256(corpus) != old['corpus_sha256']:
        raise ValueError('Original corpus digest mismatch')
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, local_files_only=True)
    assigned = {}
    rng = np.random.default_rng(579)
    for split in sorted(old['split_counts']):
        samples = [r['sample'] for r in old['records'] if r['split'] == split]
        rng.shuffle(samples)
        assigned.update({sample: LENGTHS[i % len(LENGTHS)] for i, sample in enumerate(samples)})
    records, flat, offsets = [], [], [0]
    with corpus.open('rb') as stream:
        for row in old['records']:
            stream.seek(row['source_start_byte'])
            raw = stream.read(row['source_end_byte'] - row['source_start_byte'])
            if hashlib.sha256(raw).hexdigest() != row['source_raw_sha256']:
                raise ValueError('Original document digest mismatch')
            text = raw.decode('utf8').strip()
            if hashlib.sha256(' '.join(text.split()).encode()).hexdigest() != row['record_sha256']:
                raise ValueError('Original normalized document mismatch')
            content = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2047)
            tokens = [tokenizer.bos_token_id] + content
            length = min(assigned[row['sample']], len(tokens) - 32)
            if length < 64:
                raise ValueError('Historical eligible document is unexpectedly short')
            ids = np.asarray(tokens[:length+32], dtype='<i4')
            records.append({**row, 'requested_prompt_length': assigned[row['sample']],
                'prompt_length': length, 'continuation_length': 32,
                'sequence_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                'observation_sha256': hashlib.sha256(ids[:length].tobytes()).hexdigest()})
            flat.extend(ids.tolist()); offsets.append(len(flat))
    arrays = {'private_tokens.npy': np.asarray(flat, dtype='<i4'),
        'private_offsets.npy': np.asarray(offsets, dtype='<i8')}
    for name, array in arrays.items():
        with (output / name).open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
    summary = {}
    for split in sorted(old['split_counts']):
        group = [r for r in records if r['split'] == split]
        summary[split] = {'documents': len(group), 'min': min(r['prompt_length'] for r in group),
            'max': max(r['prompt_length'] for r in group),
            'mean': float(np.mean([r['prompt_length'] for r in group])),
            'over128': sum(r['prompt_length'] > 128 for r in group),
            'at_least512': sum(r['prompt_length'] >= 512 for r in group),
            'at_least1024': sum(r['prompt_length'] >= 1024 for r in group),
            'requested_histogram': dict(Counter(r['requested_prompt_length'] for r in group))}
    manifest = {'source_manifest': str(source), 'source_manifest_sha256': sha256(source),
        'corpus': str(corpus), 'corpus_sha256': old['corpus_sha256'],
        'arrays_sha256': {name: sha256(output / name) for name in arrays},
        'seed': 579, 'requested_lengths': LENGTHS, 'continuation_tokens': 32,
        'context_limit': 2048, 'split_summary': summary, 'records': records,
        'split_policy': 'Original document and lead grouping and split identity unchanged; no cross-document concatenation.',
        'length_policy': 'Length assignment randomized within each split before reading document lengths; cap at available document minus32, never pad targets.',
        'test_policy': 'Original100 test documents remain excluded from label generation, checkpoint selection and development evaluation.',
        'limits': ['Historical development documents remain development; no new independent final test claim.',
            'Natural text continuation is not a real multi-turn chat benchmark.',
            'Short documents limit the number of long examples; report actual lengths.']}
    write_json(output / 'data_manifest.json', manifest)
    return manifest


class Records:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'data_manifest.json').read_text())
        for name, digest in self.manifest['arrays_sha256'].items():
            if sha256(self.directory / name) != digest:
                raise ValueError('Variable-length data digest mismatch')
        self.tokens = np.load(self.directory / 'private_tokens.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(self.directory / 'private_offsets.npy', mmap_mode='r', allow_pickle=False)
        self.rows = self.manifest['records']

    def __getitem__(self, sample):
        row = self.rows[sample]
        ids = np.array(self.tokens[self.offsets[sample]:self.offsets[sample+1]], copy=True)
        if (row['sample'] != sample or len(ids) != row['prompt_length']+32
                or hashlib.sha256(ids.tobytes()).hexdigest() != row['sequence_sha256']):
            raise ValueError('Corrupt or misaligned variable-length sample')
        return ids, row['prompt_length']
