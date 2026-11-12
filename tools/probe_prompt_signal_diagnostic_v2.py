"""Two long-document GPU engineering checks for Stage23, without training."""
import multiprocessing as mp
import os
import queue
from pathlib import Path

from adaptive_choice_core_v3 import load
from full_context_data_v1 import Records
from full_context_experiment_v1 import preserve, verify_preserved, write
from prompt_signal_diagnostic_v2 import setup, worker
from stage1_benchmark_v2 import sha256, stamp


def main():
    setup(); root = Path(__file__).resolve().parents[1]; out = Path(os.environ['NEUROBYPASS_RUN_DIR'])
    preservation = preserve(root, out)
    old = root / 'runs/stage20_codebook_refine_20261112_02_main'
    previous = load(old / 'protocol.json'); data = Records(previous['data_run'])
    longest = sorted(previous['validation_ids'], key=lambda s: (-data.rows[s]['prompt_length'], s))[:2]
    assert all(data.rows[s]['prompt_length'] > 128 for s in longest)
    cfg = {'output': str(out), 'old_run': str(old), 'model': previous['model'], 'data_run': previous['data_run'],
        'bank_path': str(old / 'candidate_bank.json'), 'baseline': 1, 'ids': longest,
        'source_sha256': {name: sha256(root / 'tools' / name)
            for name in ('prompt_signal_diagnostic_v2.py', 'probe_prompt_signal_diagnostic_v2.py')},
        'engineering_only': True}
    write(out / 'protocol.json', cfg)
    ctx = mp.get_context('spawn'); messages = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(r, cfg, messages)) for r in range(2)]
    try:
        for p in processes:
            p.start()
        done = set()
        while len(done) < 2:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Probe process ended without receipt')
                continue
            if item['kind'] == 'error':
                raise RuntimeError(item['traceback'])
            if item['kind'] == 'done':
                assert item['rank'] not in done; done.add(item['rank'])
            else:
                stamp('prompt_probe_progress', **{k: v for k, v in item.items() if k != 'kind'})
        for p in processes:
            p.join(timeout=30)
        assert all(p.exitcode == 0 for p in processes)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate(); p.join(timeout=10)
    verify_preserved(out, preservation)
    write(out / 'verification.json', {'passed': True, 'documents': 2, 'long_prompt_8_joint_masks': True,
        'dense_identity_and_projection_chunk_agree': True, 'old_continuation_kl_replayed': True,
        'old_files_preserved': len(preservation)})
    stamp('prompt_probe_passed', documents=2)


if __name__ == '__main__':
    main()
