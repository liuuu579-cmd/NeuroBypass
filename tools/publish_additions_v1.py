"""Publish explicitly listed new files only; refuse overwriting remote paths."""
import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--files', type=Path, required=True, help='JSON array of repository-relative new paths')
    p.add_argument('--message', required=True)
    args = p.parse_args()
    root = args.root.resolve()
    paths = json.loads(args.files.read_text())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Expected distinct, nonempty file paths')
    credential = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                                text=True, capture_output=True, check=True,
                                env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    fields = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    token = fields['password']
    def api(method, path, payload=None):
        request = urllib.request.Request('https://api.github.com/repos/' + args.repo + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                     'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    parent = api('GET', '/git/ref/heads/main')['object']['sha']
    parent_commit = api('GET', '/git/commits/' + parent)
    old_tree = api('GET', '/git/trees/' + parent_commit['tree']['sha'] + '?recursive=1')
    if old_tree.get('truncated'):
        raise RuntimeError('Cannot verify existing paths in a truncated tree')
    old = {r['path']: r for r in old_tree['tree']}
    entries, expected = [], {}
    for name in paths:
        file = (root / name).resolve()
        if not file.is_relative_to(root) or file.suffix in {'.pth', '.pt', '.bin', '.safetensors'}:
            raise ValueError('Invalid upload path: ' + name)
        if name in old:
            raise FileExistsError('Remote path already exists: ' + name)
        data = file.read_bytes()
        expected[name] = blob_sha(data)
        entries.append({'path': name, 'mode': '100644', 'type': 'blob', 'content': data.decode('utf-8')})
    tree = api('POST', '/git/trees', {'base_tree': parent_commit['tree']['sha'], 'tree': entries})
    commit = api('POST', '/git/commits', {'message': args.message, 'tree': tree['sha'], 'parents': [parent]})
    # Re-check before updating; no forced update is ever used.
    if api('GET', '/git/ref/heads/main')['object']['sha'] != parent:
        raise RuntimeError('Remote branch changed during upload; branch was not updated')
    api('PATCH', '/git/refs/heads/main', {'sha': commit['sha'], 'force': False})
    new_tree = api('GET', '/git/trees/' + tree['sha'] + '?recursive=1')
    new = {r['path']: r for r in new_tree['tree']}
    for name, sha in expected.items():
        assert new[name]['sha'] == sha, name
    for name, row in old.items():
        assert new[name]['sha'] == row['sha'], name
    print(json.dumps({'commit_url': commit['html_url'], 'added_files': len(expected),
                      'preserved_existing_paths': len(old)}, indent=2))


if __name__ == '__main__':
    main()
