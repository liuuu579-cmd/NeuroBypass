"""Prepare record-grouped C4 prompt/continuation data without modifying inputs."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from stage1_benchmark_v2 import sha256,write_json


def prepare_records(root,output,base_model,prompt_length=128,continuation_length=32,seed=579):
    from transformers import AutoTokenizer
    import zarr
    corpus=Path('/home/liushenyu/test/c4_subset_1024mb.txt')
    tokenizer=AutoTokenizer.from_pretrained(base_model,use_fast=False,local_files_only=True)
    known=set()
    for path in sorted(Path('/home/liushenyu/test').glob('opt13b_greedy_routing_labels_v3_part_*')):
        group=zarr.open_group(str(path),mode='r')
        for row in range(len(group['seq_lens'])):
            if group['input_ids'].attrs.get(f'row_{row}_written',False):
                token_ids=np.asarray(group['input_ids'][row],dtype='<i4')
                known.add(hashlib.sha256(token_ids[:128].tobytes()).hexdigest())
                known.add(hashlib.sha256(token_ids[1:128].tobytes()).hexdigest())
    old=json.loads((root/'runs/stage1_c4_20261103_02/data_manifest.json').read_text())
    excluded_regions=sorted(set((r['source_region_start_byte'],r['source_region_end_byte']) for r in old['samples']))
    counts={'train':800,'validation':100,'test':100,'calibration':16,'pilot':16}
    assignments=[split for split,count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(assignments)
    records=[];arrays=[];seen_documents=set();seen_prefixes=set();seen_leads=set();rejected=0
    with corpus.open('rb') as stream:
        stream.seek(int(corpus.stat().st_size*.90));stream.readline()
        while len(records)<len(assignments):
            start=stream.tell();raw=stream.readline();end=stream.tell()
            if not raw:raise ValueError('Not enough eligible records in source region')
            if any(start<b and end>a for a,b in excluded_regions):rejected+=1;continue
            text=raw.decode('utf8').strip()
            normalized=' '.join(text.split())
            document=hashlib.sha256(normalized.encode()).hexdigest()
            lead=hashlib.sha256(' '.join(normalized.casefold().split()[:32]).encode()).hexdigest()
            if not text or document in seen_documents or lead in seen_leads:rejected+=1;continue
            content=tokenizer.encode(text,add_special_tokens=False,truncation=True,max_length=prompt_length+continuation_length)
            if len(content)<prompt_length+continuation_length-1:rejected+=1;continue
            ids=np.asarray([tokenizer.bos_token_id]+content[:prompt_length+continuation_length-1],dtype='<i4')
            prefix=hashlib.sha256(ids[:prompt_length].tobytes()).hexdigest()
            content_prefix=hashlib.sha256(ids[1:prompt_length].tobytes()).hexdigest()
            if prefix in seen_prefixes or prefix in known or content_prefix in known:rejected+=1;continue
            seen_documents.add(document);seen_prefixes.add(prefix);seen_leads.add(lead)
            sample=len(records);split=assignments[sample]
            records.append({'sample':sample,'split':split,'record_sha256':document,'record_lead_sha256':lead,
                            'prefix_sha256':prefix,'input_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                            'source_start_byte':start,'source_end_byte':end,'source_raw_sha256':hashlib.sha256(raw).hexdigest()})
            arrays.append(ids)
    data_file=output/'private_prompt_continuation_ids.npy'
    with data_file.open('xb') as stream:np.save(stream,np.stack(arrays),allow_pickle=False)
    manifest={'corpus':str(corpus),'corpus_bytes':corpus.stat().st_size,'corpus_sha256':sha256(corpus),
              'data_sha256':sha256(data_file),'prompt_length':prompt_length,'continuation_length':continuation_length,
              'seed':seed,'split_counts':counts,'records':records,'rejected_records':rejected,
              'excluded_stage1_byte_regions':excluded_regions,'legacy_latest_prefix_overlap':0,
              'grouping':'One normalized source record per sample; exact document, prompt prefix and first-32-word lead deduplicated before split assignment.',
              'scope':'Natural-text continuation pilot, not an instruction-following or multi-turn chat benchmark.',
              'limitations':['Line boundaries follow the archived corpus writer; original URLs and remote document IDs were not preserved.',
                             'Near-duplicate documents beyond exact content, prompt and lead filters remain possible.',
                             'Legacy feature-only training overlap cannot be exhaustively excluded.']}
    write_json(output/'data_manifest.json',manifest)
    return manifest,data_file


def load_records(data_run):
    manifest=json.loads((data_run/'data_manifest.json').read_text())
    path=data_run/'private_prompt_continuation_ids.npy'
    if sha256(path)!=manifest['data_sha256']:raise ValueError('Data digest mismatch')
    ids=np.load(path,allow_pickle=False)
    if len(ids)!=len(manifest['records']):raise ValueError('Data/record count mismatch')
    return manifest,path
