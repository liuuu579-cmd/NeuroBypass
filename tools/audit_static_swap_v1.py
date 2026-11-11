"""Read-only audit of fixed-bank exchange labels, paired fit and final outputs."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from full_context_experiment_v1 import read_lines
from stage1_benchmark_v2 import sha256
from static_swap_core_v2 import greedy_indices


def load(path):return json.loads(path.read_text())


def audit(root,run,probe):
    cfg=load(run/'protocol.json');bank=load(run/'candidate_bank.json')
    assert cfg['probe']==probe and load(run/'status.json')['state']=='completed'
    assert load(run/'verification.json')['passed']
    assert len(bank['actions'])==16 and len(bank['routes'])==85
    assert len({tuple(r['skips']) for r in bank['routes']})==85
    original=set(bank['static_skips']);all_actions=set(range(16))
    for r in bank['routes']:
        assert len(r['skips'])==len(set(r['skips']))==16
        assert set(r['skips'])<=set(range(2,40))
        for path in r['action_paths']:
            assert set(path)<=all_actions and len(path)<=2
            restore=[bank['actions'][i][0] for i in path];skip=[bank['actions'][i][1] for i in path]
            assert len(set(restore))==len(restore) and len(set(skip))==len(skip)
            assert sorted((original-set(restore))|set(skip))==r['skips']
    assert sha256(run/'candidate_bank.json')==cfg['candidate_bank_sha256']
    assert sha256(run/'quality_manifest.json')==cfg['input_manifest_sha256']
    assert all(sha256(root/'tools'/name)==digest for name,digest in cfg['source_sha256'].items())
    assert all(sha256(name)==digest for name,digest in bank['source_sha256'].items())
    assert not set(cfg['train_ids'])&set(cfg['validation_ids'])
    assert set(cfg['selection_ids'])<=set(cfg['validation_ids'])
    records={}
    for rank in range(4):
        for row in read_lines(run/f'private_labels_{rank:03d}.jsonl'):
            assert row['sample'] not in records;records[row['sample']]=row
            assert len(row['kl'])==len(row['nll'])==85
            assert all(math.isfinite(v) for key in ('kl','nll') for v in row[key])
            assert min(row['kl'])>=-1e-5
    assert set(records)==set(cfg['train_ids'])|set(cfg['validation_ids'])
    costs=torch.tensor([records[i]['kl'] for i in cfg['train_ids']],dtype=torch.float64)
    control=load(run/'static_controls.json');mean=costs.mean(0)
    assert control['train_best_route']==int(mean.argmin())
    assert control['train_mean_greedy_route']==int(greedy_indices(mean[None],bank,2)[0])
    allocation=load(run/'allocation.json');assert allocation['cpu_count']==42 and len(allocation['gpu_indices'])==4
    for rank in range(4):
        e=load(run/f'engineering_{rank:03d}.json');assert e['passed'] and len(e['cpu_ids'])==10
        assert all(set(v)<=set(e['cpu_ids']) for v in e['thread_cpu_ids'].values())
        label=load(run/f'label_result_{rank:03d}.json')
        assert sha256(run/f'private_labels_{rank:03d}.jsonl')==label['labels_sha256']
        assert sha256(run/f'private_embeddings_{rank:03d}.pth')==label['embedding_sha256']
        f=load(run/f'fit_result_{rank:03d}.json');checkpoint=f['best']
        assert f['passed'] and sha256(checkpoint['path'])==checkpoint['sha256']
        assert f['optimizer_updates']==math.ceil(len(cfg['train_ids'])/16)*cfg['epochs']
        checkpoint_data=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
        assert checkpoint_data['bank_sha256']==cfg['candidate_bank_sha256']
        assert checkpoint_data['epoch']==checkpoint['epoch']
        prior=checkpoint_data['state_dict']['prior'].double()
        expected=(costs-costs[:,:1]).mean(0)
        assert (prior-expected).abs().max()<1e-5
    for rank in (0,2):
        left=load(run/f'fit_result_{rank:03d}.json');right=load(run/f'fit_result_{rank+1:03d}.json')
        assert left['initial_sha256']==right['initial_sha256'] and left['parameters']==right['parameters']
    inputs=read_lines(run/'private_quality_inputs.jsonl');manifest=load(run/'quality_manifest.json')
    assert sha256(run/'private_quality_inputs.jsonl')==manifest['private_input_sha256']
    assert Counter(r['task'] for r in inputs)==Counter(manifest['task_records'])
    quality=[r for rank in range(4) for r in read_lines(run/f'private_quality_{rank:03d}.jsonl')]
    assert len(quality)==len(inputs)*14
    assert Counter(r['sample'] for r in quality)==Counter({r['sample']:14 for r in inputs})
    assert len({(r['sample'],r['policy'],r['seed'],r['maximum_exchanges']) for r in quality})==len(quality)
    for rank in range(4):
        worker=load(run/f'worker_result_{rank:03d}.json');assert worker['passed'] and worker['base_model_frozen']
        assert sha256(run/f'private_quality_{rank:03d}.jsonl')==worker['quality_sha256']
    preserved=load(run/'preservation_before.json')
    assert all(sha256(name)==digest for name,digest in preserved.items())
    assert load(run/'preservation_after.json')['preserved_files']==len(preserved)
    return {'passed':True,'probe':probe,'train_documents':len(cfg['train_ids']),
        'development_documents':len(cfg['validation_ids']),'joint_masks':85,
        'complete_joint_costs':len(records)*85,'quality_rows':len(quality),
        'preserved_files':len(preserved),'static_control_recomputed_from_training_only':True,
        'candidate_source_is_training_only':True,'paired_initialization_and_update_budget_verified':True,
        'constant_k_and_action_compositions_verified':True,'source_checkpoint_input_hashes_verified':True}


def development_splits(run):
    cfg=load(run/'protocol.json');select=set(cfg['selection_ids'])
    grouped={}
    for rank in range(4):
        for r in read_lines(run/f'private_quality_{rank:03d}.jsonl'):
            if r['task']!='c4_development':continue
            split='selection' if r['source_sample'] in select else 'other_development'
            grouped.setdefault((split,r['policy'],r['seed'],r['maximum_exchanges']),[]).append(r)
    return [{'split':key[0],'policy':key[1],'seed':key[2],'maximum_exchanges':key[3],
        'documents':len(rows),'kl':float(np.mean([r['kl'] for r in rows])),
        'nll':float(np.mean([r['nll'] for r in rows]))} for key,rows in sorted(grouped.items(),key=lambda kv:str(kv[0]))]
