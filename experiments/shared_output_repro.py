"""Reproduction check for the K arms (docs/shared-output-v1-k-arms.md): common-raw at seed 300, rerun after the readout refactor, against the pilot report.

Every measure must be equal except the `substitutability` read added with the K arms.
Usage: .venv/bin/python -m experiments.shared_output_repro
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = 'common-raw/seed300'


def flat(o, p=''):
    if isinstance(o, dict):
        return {k: v for i, x in o.items() for k, v in flat(x, f'{p}/{i}').items()}
    if isinstance(o, list):
        return {k: v for i, x in enumerate(o) for k, v in flat(x, f'{p}/{i}').items()}
    return {p: o}


def main():
    old = json.loads((ROOT / 'reports' / 'shared-output-v1-pilot.json').read_text())['results'][KEY]
    new = json.loads((ROOT / 'reports' / 'shared-output-v1-pilot-repro.json').read_text())['results'][KEY]
    fields = ('snapshots', 'series', 'relearning', 'loss_every_100', 'audit', 'reader_parameters')
    a, b = flat({k: old.get(k) for k in fields}), flat({k: new.get(k) for k in fields})
    added = sorted(k for k in set(b) - set(a) if '/substitutability/' in k)
    keys = (set(a) | set(b)) - set(added)
    differing = sorted(k for k in keys if a.get(k) != b.get(k))
    out = {'run': KEY, 'measures_compared': len(keys), 'equal': len(keys) - len(differing), 'differing': differing, 'added_substitutability_measures': len(added)}
    (ROOT / 'reports' / 'shared-output-v1-repro-check.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f'{out["equal"]} of {len(keys)} measures equal; {len(differing)} differ; {len(added)} added substitutability measures', differing[:10])


if __name__ == '__main__':
    main()
