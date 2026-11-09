"""Frozen, label-free within-task bijections and recorded path reuse."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PERMUTATION_SEEDS = [1701, 1702, 1703]


def policy_key(policy):
    return policy.get('name', policy.get('policy')), policy['seed'], policy['k']


def record_key(record):
    return record['task'], record['sample']


def assignments(inputs, seeds=PERMUTATION_SEEDS):
    groups = defaultdict(list)
    for record in inputs: groups[record['task']].append(record)
    result = []
    for task_index, (task, records) in enumerate(sorted(groups.items())):
        records = sorted(records, key=lambda r: r['sample'])
        for seed in seeds:
            rng = np.random.default_rng(seed + task_index*100000)
            if task == 'wikitext_conditional':
                # Group windows by article, randomize groups and within-group
                # order, then rotate at least the largest group length. Thus
                # each window receives a prompt from a DIFFERENT article.
                by_doc = defaultdict(list)
                for record in records: by_doc[record['document']].append(record)
                documents = list(by_doc); rng.shuffle(documents)
                ordered = []
                for document in documents:
                    rows = list(by_doc[document]); rng.shuffle(rows); ordered.extend(rows)
                maximum = max(map(len, by_doc.values()))
                if maximum*2 > len(ordered): raise ValueError('No cross-article bijection exists')
                shift = int(rng.integers(maximum, len(ordered)-maximum+1))
                donors = ordered[shift:] + ordered[:shift]
                pairs = list(zip(ordered, donors))
            else:
                # Uniform random permutation conditioned on no self assignment.
                for attempt in range(1000):
                    indexes = rng.permutation(len(records))
                    if np.all(indexes != np.arange(len(records))): break
                else: raise ValueError('Unable to construct derangement')
                pairs = [(record, records[index]) for record, index in zip(records, indexes)]
            assert len({b['sample'] for _, b in pairs}) == len(records)
            for recipient, donor in pairs:
                assert recipient['sample'] != donor['sample']
                if task == 'wikitext_conditional': assert recipient['document'] != donor['document']
                result.append({'task': task, 'sample': recipient['sample'], 'permutation_seed': seed,
                    'donor_sample': donor['sample'], 'recipient_document': recipient.get('document'),
                    'donor_document': donor.get('document')})
    return sorted(result, key=lambda r: (r['task'], r['sample'], r['permutation_seed']))


def load_reference(source, selected, record_keys=None):
    wanted = {policy_key(p) for p in selected}
    registry = {}
    for path in sorted(Path(source).glob('quality_*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            if policy_key(row) not in wanted: continue
            if record_keys is not None and record_key(row) not in record_keys: continue
            key = record_key(row) + policy_key(row)
            if key in registry: raise ValueError('Duplicate reference record')
            registry[key] = row
    return registry


def score_fields(record, scores):
    if record['gold'] is None:
        return {'nll_sum': -scores[0], 'scored_tokens': record['target_lengths'][0]}
    predicted = int(np.argmax(scores))
    values = {'gold': record['gold'], 'prediction': predicted, 'correct': int(predicted == record['gold'])}
    if record['task'] != 'winogrande':
        normalized = np.asarray(scores)/np.asarray(record['normalization_lengths'])
        predicted = int(normalized.argmax())
        values.update({'prediction_norm': predicted, 'correct_norm': int(predicted == record['gold'])})
    return values


def metric_value(row):
    if row['task'] == 'wikitext_conditional': return row['nll_sum']
    return row['correct'] if row['task'] == 'winogrande' else row['correct_norm']
