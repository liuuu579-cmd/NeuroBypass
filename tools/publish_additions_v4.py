"""Append-only publication with ASCII JSON and base64 Git blob payloads.

Avoid putting every large evidence file inside one tree request. Existing paths
must match local contents exactly; conflicting paths are never overwritten.
Network retries apply only to reads and content-addressed immutable objects.
An ambiguous branch update is resolved by reading its state before proceeding.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--files', type=Path, required=True)
    parser.add_argument('--message', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = json.loads(args.files.read_text())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Expected distinct nonempty repository paths')
    if args.receipt and args.receipt.exists():
        raise FileExistsError('Publication receipt must have a new path')
    expected, contents = {}, {}
    for name in paths:
        path = (root / name).resolve()
        if (not path.is_relative_to(root) or path.suffix not in {'.py', '.md', '.json', '.jsonl', '.csv', '.log'}
                or path.name.startswith('private_') or str(Path(name)) != name):
            raise ValueError('Invalid public evidence/source path: ' + name)
        data = path.read_bytes()
        data.decode('utf-8')  # Public source/evidence must still be valid UTF-8.
        contents[name] = base64.b64encode(data).decode('ascii')
        expected[name] = blob_sha(data)
    result = subprocess.run(['git', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True, check=True,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    token = fields['password']

    def api(method, suffix, payload=None, retry=True):
        encoded = None if payload is None else json.dumps(payload, ensure_ascii=True).encode('ascii')
        attempts = 3 if retry else 1
        for attempt in range(attempts):
            try:
                request = urllib.request.Request('https://api.github.com/repos/' + args.repo + suffix,
                    data=encoded, method=method, headers={'Authorization': 'Bearer ' + token,
                        'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json',
                        'User-Agent': 'NeuroBypass-append-only-publisher'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in [429, 500, 502, 503, 504]:
                    raise RuntimeError(f'GitHub returned HTTP {error.code} for {method} {suffix}') from None
                if attempt + 1 == attempts:
                    raise RuntimeError(f'GitHub network request failed: {method} {suffix}; {type(error).__name__}') from None
                print(json.dumps({'event': 'retry_immutable_request', 'method': method,
                    'endpoint': suffix, 'attempt': attempt + 1}), flush=True)
                time.sleep(2 ** (attempt + 1))

    parent = api('GET', '/git/ref/heads/main')['object']['sha']
    parent_commit = api('GET', '/git/commits/' + parent)
    old_tree = api('GET', '/git/trees/' + parent_commit['tree']['sha'] + '?recursive=1')
    if old_tree.get('truncated'):
        raise RuntimeError('Cannot audit a truncated Git tree')
    old = {row['path']: row for row in old_tree['tree']}
    missing = []
    for name, digest in expected.items():
        if name in old:
            row = old[name]
            if row['type'] != 'blob' or row['sha'] != digest:
                raise ValueError('Refusing to overwrite a conflicting remote path: ' + name)
        else:
            missing.append(name)
    print(json.dumps({'event': 'remote_preflight', 'head_commit': parent,
        'requested_files': len(paths), 'missing_files': len(missing),
        'already_identical_files': len(paths) - len(missing)}), flush=True)
    if args.check_only:
        return
    commit = parent_commit
    if missing:
        remote_blob_ids = {row['sha'] for row in old.values() if row['type'] == 'blob'}
        entries = []
        for index, name in enumerate(missing):
            digest = expected[name]
            if digest not in remote_blob_ids:
                blob = api('POST', '/git/blobs', {'content': contents[name], 'encoding': 'base64'})
                if blob['sha'] != digest:
                    raise RuntimeError('Uploaded blob content mismatch: ' + name)
                remote_blob_ids.add(digest)
            entries.append({'path': name, 'mode': '100644', 'type': 'blob', 'sha': digest})
            print(json.dumps({'event': 'blob_ready', 'file': name,
                'completed': index + 1, 'total': len(missing)}), flush=True)
        tree = api('POST', '/git/trees', {'base_tree': parent_commit['tree']['sha'], 'tree': entries})
        created = api('POST', '/git/commits', {'message': args.message, 'tree': tree['sha'], 'parents': [parent]})
        if api('GET', '/git/ref/heads/main')['object']['sha'] != parent:
            raise RuntimeError('Remote branch changed during upload; no branch update attempted')
        try:
            api('PATCH', '/git/refs/heads/main', {'sha': created['sha'], 'force': False}, retry=False)
        except RuntimeError:
            if api('GET', '/git/ref/heads/main')['object']['sha'] != created['sha']:
                raise RuntimeError('Branch-update outcome was not confirmed; rerun preflight before retrying') from None
        commit = created
    final_head = api('GET', '/git/ref/heads/main')['object']['sha']
    final_commit = api('GET', '/git/commits/' + final_head)
    final_tree = api('GET', '/git/trees/' + final_commit['tree']['sha'] + '?recursive=1')
    if final_tree.get('truncated'):
        raise RuntimeError('Cannot audit final truncated Git tree')
    final = {row['path']: row for row in final_tree['tree']}
    for name, digest in expected.items():
        if name not in final or final[name]['sha'] != digest:
            raise RuntimeError('Public evidence missing or changed after publication: ' + name)
    for name, row in old.items():
        if row['type'] == 'blob' and (name not in final or final[name]['sha'] != row['sha']):
            raise RuntimeError('Preexisting public file changed: ' + name)
    receipt = {'commit_url': 'https://github.com/' + args.repo + '/commit/' + commit['sha'],
        'added_files': len(missing), 'already_identical_files': len(paths) - len(missing),
        'preserved_existing_paths': len(old), 'verified_requested_files': len(paths),
        'final_head_commit': final_head}
    if args.receipt:
        with args.receipt.open('x') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
