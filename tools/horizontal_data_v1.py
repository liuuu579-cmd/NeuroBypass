"""Read-only benchmark caches and frozen, answer-blind routing observations."""
import hashlib
import json
import re
from pathlib import Path


CACHE = Path('/home/liushenyu/.cache/huggingface/datasets')
HARNESS = Path('/home/liushenyu/test/skip/lm-evaluation-harness')
DATA = {
    'piqa': CACHE / 'piqa/plain_text/1.1.0/6c611c1a9bf220943c4174e117d3b660859665baf1d43156230116185312d011',
    'hellaswag': CACHE / 'Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76',
    'winogrande': CACHE / 'winogrande/winogrande_xl/1.1.0/a826c3d3506aefe0e9e9390dcb53271070536586bab95849876b2c1743df56e2',
    'wikitext': CACHE / 'wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3'}


def read_dataset(task, split):
    from datasets import Dataset
    return Dataset.from_file(str(DATA[task] / f'{task}-{split}.arrow'))


def clean_hella(text):
    return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')


def question(task, row):
    if task == 'piqa':
        prompt = 'Question: ' + row['goal'] + '\nAnswer:'
        options = [row['sol1'], row['sol2']]
        return prompt, [(prompt, ' ' + answer) for answer in options], options, int(row['label'])
    if task == 'hellaswag':
        prompt = clean_hella(row['activity_label'] + ': ' + row['ctx_a'] + ' ' + row['ctx_b'].capitalize())
        options = [clean_hella(text) for text in row['endings']]
        return prompt, [(prompt, ' ' + answer) for answer in options], options, int(row['label'])
    if task == 'winogrande':
        if row['sentence'].count('_') != 1: raise ValueError('Require exactly one blank')
        before, after = row['sentence'].split('_')
        suffix = after.strip()
        if not suffix: raise ValueError('Empty WinoGrande scoring suffix')
        options = [row['option1'], row['option2']]
        # Route only from the common prefix BEFORE the blank. Neither options,
        # scored suffix nor correct label can enter the router observation.
        return before.rstrip(), [(before + answer, ' ' + suffix) for answer in options], options, int(row['answer']) - 1
    raise ValueError('Unknown multiple-choice task')


def encode_pair(tokenizer, context, continuation):
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation, context = context[-spaces:] + continuation, context[:-spaces]
    prefix = tokenizer.encode(context, add_special_tokens=True)
    whole = tokenizer.encode(context + continuation, add_special_tokens=True)
    if whole[:len(prefix)] != prefix:
        raise ValueError('BPE boundary changed a context token; cannot silently misalign target')
    target = whole[len(prefix):]
    if not prefix or not target or len(target) > 2048:
        raise ValueError('Invalid context/target lengths')
    removed = max(0, len(prefix) + len(target) - 2049)
    if removed >= len(prefix): raise ValueError('Truncation removed all context')
    return prefix[removed:] + target, len(target), removed


def encode_question(task, row, index, tokenizer):
    prompt, pairs, choices, gold = question(task, row)
    if not 0 <= gold < len(pairs): raise ValueError('Invalid public benchmark label')
    observation = tokenizer.encode(prompt, add_special_tokens=True)[:128]
    if not observation: observation = [tokenizer.bos_token_id]
    sequences = [encode_pair(tokenizer, context, continuation) for context, continuation in pairs]
    return {'task': task, 'sample': index, 'route_ids': observation,
        'sequence_ids': [seq for seq, _, _ in sequences], 'target_lengths': [n for _, n, _ in sequences],
        'truncated_context_tokens': [n for _, _, n in sequences],
        'normalization_lengths': [len(text) for text in choices], 'gold': gold,
        'document_sha256': hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}


def wiki_documents(dataset):
    records, current = [], []
    for row in dataset:
        text = row['text']
        if re.match(r'^\s*= [^=].*[^=] =\s*$', text) and current:
            records.append('\n'.join(current)); current = []
        current.append(text)
    if current: records.append('\n'.join(current))
    return records


def wiki_windows(dataset, tokenizer):
    windows = []
    for document, text in enumerate(wiki_documents(dataset)):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        digest = hashlib.sha256(text.encode()).hexdigest()
        for offset in range(0, max(0, len(tokens) - 127), 384):
            ids = [tokenizer.bos_token_id] + tokens[offset:offset + 511]
            if len(ids) <= 128: continue
            windows.append({'task': 'wikitext_conditional', 'sample': len(windows), 'document': document,
                'document_sha256': digest, 'offset': offset, 'route_ids': ids[:128],
                'sequence_ids': [ids], 'target_lengths': [len(ids) - 128],
                'truncated_context_tokens': [0], 'normalization_lengths': [], 'gold': None})
    return windows


def policies(root, training_run, budgets):
    from stage1_benchmark_v2 import sha256
    paths = json.loads((training_run / 'static_paths.json').read_text())
    fixed = json.loads((training_run / 'fixed_global_policy.json').read_text())['ordered_skips']
    result = [{'name': 'full', 'kind': 'static', 'k': 0, 'skips': [], 'seed': None}]
    for k in budgets:
        for name in ['static_joint_kl', 'sleb_nll_opt', 'shortgpt_bi_opt',
                'shortened_ppl_opt', 'shortened_ppl_plus', 'random_static_seed579']:
            result.append({'name': name, 'kind': 'static', 'k': k, 'skips': paths[name][:k], 'seed': None})
        for arm in ['pure', 'hybrid']:
            for seed in [579, 580, 581]:
                checkpoint = json.loads((training_run / f'{arm}_seed{seed}_checkpoints.json').read_text())[str(k)]
                if sha256(checkpoint['path']) != checkpoint['sha256']: raise ValueError('Checkpoint digest mismatch')
                result.append({'name': arm, 'kind': 'router', 'seed': seed, 'k': k,
                    'checkpoint': checkpoint, 'prefix': [] if arm == 'pure' else fixed})
    return result
