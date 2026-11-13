"""CPU audit: new pilot against frozen Stage13 per-question likelihoods."""
import argparse
import json
from pathlib import Path


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = {}
    for path in args.pilot.glob('private_quality_*.jsonl'):
        for row in lines(path):
            if row['k'] not in (0, 12, 16) or row['policy'] == 'evopress_whole_opt':
                continue
            key = tuple(row[x] for x in ('task', 'sample', 'policy', 'seed', 'k'))
            target[key] = row
    found = set()
    maximum = 0.
    for path in sorted(args.historical.glob('quality_*.jsonl')):
        for row in lines(path):
            if row['condition'] != 'matched':
                continue
            key = tuple(row[x] for x in ('task', 'sample', 'policy', 'seed', 'k'))
            if key not in target:
                continue
            new = target[key]
            assert key not in found
            assert row['document_sha256'] == new['document_sha256']
            assert row['skipped_layers'] == new['skips']
            assert row['router_observation_tokens'] == new['router_tokens']
            assert row['target_tokens'] == new['target_tokens']
            error = max(abs(a-b) for a, b in zip(row['scores'], new['scores']))
            maximum = max(maximum, error)
            assert error < 1e-5
            for name in ('correct', 'correct_norm', 'prediction', 'prediction_norm', 'nll_sum', 'scored_tokens'):
                assert row.get(name) == new.get(name)
            found.add(key)
    assert found == set(target), (len(found), len(target))
    result = {'passed': True, 'historical_rows_replayed': len(found),
        'max_abs_loglikelihood_error': maximum, 'document_hashes_routes_observation_and_targets_match': True,
        'pilot': str(args.pilot.resolve()), 'historical': str(args.historical.resolve())}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
