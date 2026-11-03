"""Upload this snapshot to an empty GitHub repository using the REST API.

Uses the existing git credential helper without printing its credentials. Refuses
to replace an existing branch; performs no local source edits. Large weights and
data are excluded. Run only when publication has been explicitly authorized.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def git_blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', required=True, help='owner/repository')
    args = p.parse_args()
    root = args.root.resolve()
    credential = subprocess.run(['git', 'credential', 'fill'],
                                input='protocol=https\nhost=github.com\n\n', text=True,
                                capture_output=True, check=True,
                                env={**__import__('os').environ, 'GIT_TERMINAL_PROMPT': '0'})
    fields = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    token = fields.get('password')
    if not token:
        raise RuntimeError('No existing GitHub credential is available')

    def api(method, path, payload=None):
        request = urllib.request.Request(
            'https://api.github.com' + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                     'Content-Type': 'application/json', 'X-GitHub-Api-Version': '2022-11-28'},
            method=method)
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)

    prefix = '/repos/' + args.repo
    info = api('GET', prefix)
    if not info.get('permissions', {}).get('push'):
        raise RuntimeError('Authenticated account lacks push permission')
    if api('GET', prefix + '/branches'):
        raise RuntimeError('Refusing to publish: repository already has branches')
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in {'.git', '__pycache__', '.venv'} for part in relative.parts):
            continue
        if path.suffix in {'.pth', '.pt', '.bin', '.safetensors', '.pyc'} or path.name.startswith('.env'):
            continue
        data = path.read_bytes()
        if len(data) > 10_000_000:
            raise ValueError(f'Unexpected large file: {relative}')
        files.append((str(relative), data))
    readme = (root / 'README.md').read_bytes()
    initial = api('PUT', prefix + '/contents/README.md',
                  {'message': 'Add NeuroBypass project overview', 'branch': 'main',
                   'content': base64.b64encode(readme).decode()})
    parent = initial['commit']['sha']
    parent_commit = api('GET', prefix + '/git/commits/' + parent)
    tree_entries = []
    binary_files = []
    for name, data in files:
        try:
            content = data.decode('utf-8')
        except UnicodeDecodeError:
            binary_files.append((name, data))
        else:
            mode = '100755' if (root / name).stat().st_mode & 0o111 else '100644'
            tree_entries.append({'path': name, 'mode': mode, 'type': 'blob', 'content': content})

    def upload_binary(item):
        name, data = item
        blob = api('POST', prefix + '/git/blobs',
                   {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        if blob['sha'] != git_blob_sha(data):
            raise RuntimeError('Blob hash mismatch: ' + name)
        return {'path': name, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(upload_binary, binary_files):
            tree_entries.append(result)
    tree = api('POST', prefix + '/git/trees',
               {'base_tree': parent_commit['tree']['sha'], 'tree': tree_entries})
    commit = api('POST', prefix + '/git/commits',
                 {'message': 'Archive dynamic pruning experiments and add verified diagnostics',
                  'tree': tree['sha'], 'parents': [parent]})
    # Non-forced update refuses an intervening incompatible branch update.
    api('PATCH', prefix + '/git/refs/heads/main', {'sha': commit['sha'], 'force': False})
    remote_tree = api('GET', prefix + '/git/trees/' + tree['sha'] + '?recursive=1')
    if remote_tree.get('truncated'):
        raise RuntimeError('Cannot verify a truncated tree')
    remote_blobs = {item['path']: item['sha'] for item in remote_tree['tree'] if item['type'] == 'blob'}
    for name, data in files:
        if remote_blobs.get(name) != git_blob_sha(data):
            raise RuntimeError('Uploaded file differs: ' + name)
    print(json.dumps({'url': info['html_url'], 'commit_url': commit['html_url'],
                      'commit_sha': commit['sha'], 'tree_sha': tree['sha'],
                      'verified_files': len(files), 'total_bytes': sum(len(d) for _, d in files)}, indent=2))


if __name__ == '__main__':
    main()
