"""Supplementary read-only checks for both BOS and content-only prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


def digest(values):
    return hashlib.sha256(np.asarray(values,dtype='<i4').tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    known=set()
    for path in args.source.glob('opt13b_greedy_routing_labels_v3_part_*'):
        g=zarr.open_group(str(path),mode='r')
        for i in range(len(g['seq_lens'])):
            if g['input_ids'].attrs.get(f'row_{i}_written',False):
                known.add(digest(g['input_ids'][i][:128]))
    data=np.load(args.run/'private_eval_token_ids.npz')
    content=set();whole=set();matches=0;duplicates=0
    counts={}
    for split in ['calibration','test']:
        counts[split]=len(data[split])
        for row in data[split]:
            first=digest(row[:128]);body=digest(row[1:129])
            matches+=int(first in known or body in known)
            duplicates+=int(body in content or digest(row) in whole)
            content.add(body);whole.add(digest(row))
    result={'counts':counts,'training_prefixes_compared':len(known),
            'bos_and_content_prefix_training_matches':matches,'duplicate_content_prefixes_or_inputs':duplicates,
            'passed':matches==0 and duplicates==0,
            'limitations':['Latest v3 training stores only; legacy feature-only sources have incomplete provenance.',
                           'Separate blocks do not guarantee separate documents.']}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
