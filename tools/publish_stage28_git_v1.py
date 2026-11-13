"""Publish only Stage28 allowlisted additions using a fresh Git object database.

This avoids one HTTP API POST per file. Existing remote files must be identical;
the normal non-force push and complete post-push tree audit preserve old evidence.
No worktree or existing source is modified, and private artifacts are excluded.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path


PREFIX = 'stage28_window_supervision_20261113_01'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def main():
    root = Path(__file__).resolve().parents[1]; cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cpus[40:42])
    run = root / 'runs' / (PREFIX + '_main')
    assert json.loads((run / 'independent_audit.json').read_text())['passed']
    assert json.loads((root / 'analysis' / (PREFIX + '_completion.json')).read_text())['state'] == 'completed'
    paths = json.loads((root / 'analysis' / (PREFIX + '_audited_upload.json')).read_text())
    paths.append('tools/publish_stage28_git_v1.py'); assert len(paths) == len(set(paths))
    payloads = {}; expected = {}
    for name in paths:
        path = (root / name).resolve()
        assert path.is_relative_to(root) and not path.name.startswith('private_')
        assert path.suffix in ('.py', '.md', '.json') and path.stat().st_size < 1024*1024
        data = path.read_bytes(); data.decode('utf8'); payloads[name] = data
        expected[name] = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    database = root / 'runs' / (PREFIX + '_publication_git'); database.mkdir(exist_ok=False)
    write(database / 'allowlist.json', paths)
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    def git(*args, data=None):
        result = subprocess.run(['git', '-c', 'pack.threads=2', '-c', 'index.threads=2', *args], cwd=database,
            env=env, input=data, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError('Git operation failed: ' + str(args[0]) + ': ' + result.stderr.decode(errors='replace'))
        return result.stdout
    def tree(ref):
        result = {}
        for row in git('ls-tree', '-r', '-z', ref).split(b'\0'):
            if not row:
                continue
            header, name = row.split(b'\t', 1); mode, kind, digest = header.split()
            assert kind == b'blob'
            result[name.decode()] = (mode.decode(), digest.decode())
        return result
    remote = 'https://github.com/liuuu579-cmd/NeuroBypass.git'
    git('init', '--bare', '.'); git('remote', 'add', 'origin', remote)
    git('fetch', '--depth=1', '--no-tags', 'origin', 'refs/heads/main')
    base = git('rev-parse', 'FETCH_HEAD').decode().strip(); old = tree(base)
    missing = []
    for name, digest in expected.items():
        if name in old:
            assert old[name] == ('100644', digest), 'Refusing conflicting existing public path: ' + name
        else:
            missing.append(name)
    print(json.dumps({'event': 'git_preflight', 'parent': base, 'missing': len(missing), 'preserved_paths': len(old)}), flush=True)
    git('read-tree', base)
    for name in missing:
        digest = git('hash-object', '-w', '--stdin', data=payloads[name]).decode().strip()
        assert digest == expected[name]
        git('update-index', '--add', '--cacheinfo', '100644,' + digest + ',' + name)
    new_tree = git('write-tree').decode().strip()
    for key, fallback in (('GIT_AUTHOR_NAME', 'NeuroBypass automation'), ('GIT_AUTHOR_EMAIL', 'neurobypass@users.noreply.github.com')):
        config_key = 'user.name' if key.endswith('NAME') else 'user.email'
        result = subprocess.run(['git', 'config', '--global', '--get', config_key], capture_output=True, text=True)
        env[key] = result.stdout.strip() or fallback
    env['GIT_COMMITTER_NAME'] = env['GIT_AUTHOR_NAME']; env['GIT_COMMITTER_EMAIL'] = env['GIT_AUTHOR_EMAIL']
    commit = git('commit-tree', new_tree, '-p', base,
        data=b'Compare matched first32 and four-window joint KL supervision with no-input controls and independent audit\n').decode().strip()
    before = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    assert before == base, 'Remote advanced; do not attempt a stale push'
    git('push', 'origin', commit + ':refs/heads/main')
    git('fetch', '--depth=1', '--no-tags', 'origin', 'refs/heads/main')
    head = git('rev-parse', 'FETCH_HEAD').decode().strip(); final = tree(head)
    for name, original in old.items():
        assert final.get(name) == original, 'Preexisting remote evidence changed: ' + name
    for name, digest in expected.items():
        assert final.get(name) == ('100644', digest), 'Requested addition missing or changed: ' + name
    receipt = {'commit_url': 'https://github.com/liuuu579-cmd/NeuroBypass/commit/' + commit,
        'added_files': len(missing), 'verified_requested_files': len(paths), 'preserved_existing_paths': len(old),
        'final_head_commit': head, 'publication': 'normal non-force Git push from fresh bare repository; full remote tree audited',
        'original_api_upload': 'Stopped during immutable blob creation after repeated network retries; no branch update requested.'}
    write(run / 'publication_verified.json', receipt); write(database / 'verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
