"""CPU-only independent task-template, answer-blindness and token-boundary audit."""
import argparse
import importlib.util
import json
from pathlib import Path

from horizontal_data_v1 import HARNESS, encode_question, question, read_dataset, wiki_documents, wiki_windows
from stage1_benchmark_v2 import write_json


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/liushenyu/opt-13b', use_fast=False, local_files_only=True)
    hella = load('audited_hella', HARNESS / 'lm_eval/tasks/hellaswag/utils.py')
    wino = load('audited_wino', HARNESS / 'lm_eval/tasks/winogrande/preprocess_winogrande.py')
    counts, observations, truncated = {}, {}, {}
    for task in ['piqa', 'hellaswag', 'winogrande']:
        data = read_dataset(task, 'validation')
        token_counts, truncations = [], 0
        for index, doc in enumerate(data):
            prompt, pairs, choices, gold = question(task, doc)
            if task == 'hellaswag':
                reference_prompt = hella.preprocess(doc['activity_label'] + ': ' + doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize())
                assert prompt == reference_prompt
                assert choices == [hella.preprocess(s) for s in doc['endings']]
            elif task == 'winogrande':
                assert [p[0] for p in pairs] == wino.doc_to_choice(doc)
                assert [p[1] for p in pairs] == [' ' + wino.doc_to_target(doc)] * 2
                assert gold == wino.doc_to_text(doc)
                assert prompt == doc['sentence'].split('_')[0].rstrip()
            else:
                assert prompt == 'Question: ' + doc['goal'] + '\nAnswer:'
                assert choices == [doc['sol1'], doc['sol2']]
            encoded = encode_question(task, doc, index, tokenizer)
            assert encoded['route_ids'][0] == tokenizer.bos_token_id
            assert encoded['normalization_lengths'] == list(map(len, choices))
            token_counts.append(len(encoded['route_ids']))
            truncations += sum(encoded['truncated_context_tokens'])
            # Independently exercise pinned TemplateLM's trailing-space and
            # whole-pair tokenization convention, without copying third-party code.
            for j, (context, target) in enumerate(pairs):
                nspaces = len(context) - len(context.rstrip())
                if nspaces:
                    target, context = context[-nspaces:] + target, context[:-nspaces]
                whole = tokenizer.encode(context + target, add_special_tokens=True)
                prefix = tokenizer.encode(context, add_special_tokens=True)
                assert encoded['target_lengths'][j] == len(whole) - len(prefix)
                assert encoded['sequence_ids'][j] == whole[-2049:]
            # Counterfactual labels and choices cannot affect routing input.
            changed = dict(doc)
            if task == 'piqa': changed.update(label=1-gold, sol1='different answer one', sol2='different answer two')
            elif task == 'hellaswag': changed.update(label=str((gold+1) % 4), endings=['changed ending'] * 4)
            else:
                changed.update(answer=str(2-gold), option1='changed first', option2='changed second',
                    sentence=doc['sentence'].split('_')[0] + '_ changed future suffix')
            other = encode_question(task, changed, index, tokenizer)
            assert other['route_ids'] == encoded['route_ids']
        counts[task] = len(data)
        observations[task] = {'min': min(token_counts), 'max': max(token_counts)}
        truncated[task] = truncations
        print(json.dumps({'audited_task': task, 'records': len(data)}), flush=True)
    dataset = read_dataset('wikitext', 'test')
    documents = wiki_documents(dataset)
    windows = wiki_windows(dataset, tokenizer)
    scored = {}
    for row in windows:
        document = row['document']
        start = row['offset'] + 127
        end = start + row['target_lengths'][0]
        assert start == scored.get(document, 127)
        assert row['route_ids'] == row['sequence_ids'][0][:128]
        assert row['target_lengths'][0] == len(row['sequence_ids'][0]) - 128
        scored[document] = end
    for index, text in enumerate(documents):
        length = len(tokenizer.encode(text, add_special_tokens=False))
        assert scored.get(index, 127) == max(127, length)
    write_json(args.output, {'status': 'passed', 'multiple_choice_records': counts,
        'router_observation_tokens': observations, 'truncated_context_tokens': truncated,
        'wiki_articles': len(documents), 'wiki_windows': len(windows),
        'wiki_scored_tokens': sum(r['target_lengths'][0] for r in windows),
        'checks': ['all task templates vs pinned upstream helper functions',
            'all option token boundaries and explicit BOS', 'character normalization lengths',
            'routing invariant to labels and changed options', 'Wino routing invariant to future suffix',
            'Wiki article boundaries and nonoverlapping exhaustive conditional targets'],
        'gpu_used': False})


if __name__ == '__main__': main()
