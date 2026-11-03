"""Read-only verification of every remote source blob against the local copy."""
import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', required=True)
    args = p.parse_args()
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            text=True, capture_output=True, check=True,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    def get(path):
        request = urllib.request.Request('https://api.github.com/repos/' + args.repo + path,
                                          headers={'Authorization': 'Bearer ' + credential['password'],
                                                   'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    head = get('/git/ref/heads/main')['object']['sha']
    commit = get('/git/commits/' + head)
    tree = get('/git/trees/' + commit['tree']['sha'] + '?recursive=1')
    assert not tree.get('truncated')
    count = 0
    for item in tree['tree']:
        if item['type'] != 'blob':
            continue
        data = (args.root / item['path']).read_bytes()
        sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert sha == item['sha'], item['path']
        count += 1
    baseline = json.loads((args.root / 'analysis/resource_followup_baseline_20261103.json').read_text())['files']
    for item in baseline:
        data = (args.root / item['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    print(json.dumps({'repo': args.repo, 'head_commit': head, 'verified_remote_blobs': count,
                      'preserved_baseline_files': len(baseline)}, indent=2))


if __name__ == '__main__':
    main()
