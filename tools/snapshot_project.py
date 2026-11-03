"""Create a byte-preserving snapshot without writing into the source project."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--attachments', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    files = [(p, Path('legacy') / p.relative_to(source))
             for folder in (source, source / 'skip')
             for p in folder.iterdir()
             if p.is_file() and (p.suffix in {'.py', '.sh'} or p.name == 'readme.txt')]
    root_figures = ['avg_layer_importance.png', 'layer_importance_heatmap.png',
                    'skip_count_distribution.png', 'synergistic_effect_proof.png',
                    'routing_strategies_comparison.png']
    files += [(source / n, Path('evidence/figures') / n) for n in root_figures]
    files += [(p, Path('evidence/figures/skip') / p.name)
              for p in (source / 'skip').glob('*.png')]
    files += [(p, Path('evidence/reports') / p.name)
              for p in args.attachments.glob('*.docx')]
    entries = []
    for path, relative in sorted(files, key=lambda item: str(item[1])):
        if not path.exists():
            continue
        dest = output / relative
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest}')
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
        entries.append({'source_path': str(path), 'snapshot_path': str(relative),
                        'bytes': stat.st_size, 'sha256': digest,
                        'mtime_ns': stat.st_mtime_ns,
                        'mtime_asia_shanghai': datetime.fromtimestamp(
                            stat.st_mtime, ZoneInfo('Asia/Shanghai')).isoformat()})
    manifest = output / 'analysis/source_manifest.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'timezone': 'Asia/Shanghai', 'files': entries},
                                   ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(entries), 'total_bytes': sum(x['bytes'] for x in entries)}, indent=2))


if __name__ == '__main__':
    main()
