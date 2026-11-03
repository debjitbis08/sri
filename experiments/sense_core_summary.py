"""Tables for experiments.sense_core: the core's counts, symbols and relations with each image encoder, and the reference check against limited-memory main.
Usage: python -m experiments.sense_core_summary reports/sense-core-v1-pilot.json.gz [reports/limited-memory-v1-main-results.json.gz]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

POINTS = ('block1', 'block2', 'block3', 'delay', 'relearn')


def load(p):
    p = Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def cell(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '—'
    m = float(np.mean(xs))
    return f'{m:.2f} ({min(xs):.2f}–{max(xs):.2f})'


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{prefix}{k}/'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'{prefix}{k}'] = float(v)
    return out


def measures(snap):
    f = snap['four_way']['full']
    m = {f'matching {k}': v['all'] for k, v in f['matching'].items()}
    m.update({f'transfer {k}': v for k, v in f['transfer8'].items()})
    ev = snap.get('evaluate', {})
    for k, v in flat(ev.get('succession_pairwise', {})).items():
        m[f'succession {k}'] = v
    for k, v in flat(ev.get('senses', {}).get('composition', {})).items():
        m[f'composition {k}'] = v
    return m


def main(path, lm=None):
    d = load(path)
    R, arms, seeds = d['results'], d['arms'], d['seeds']
    for point in ('block1', 'block2', 'block3', 'relearn'):
        rows = {}
        for a in arms:
            for s in seeds:
                r = R.get(f'{a}/seed{s}')
                if r is None:
                    continue
                for k, v in measures(r['snapshots'][point]).items():
                    rows.setdefault(k, {}).setdefault(a, []).append(v)
        print(f'\n## {point}: mean (range) over seeds {", ".join(map(str, seeds))}')
        print(f'{"measure":52s} ' + ' '.join(f'{a:>18s}' for a in arms))
        for k in sorted(rows):
            if k.startswith(('succession', 'composition')) and point != 'block3':
                continue
            print(f'{k:52s} ' + ' '.join(f'{cell(rows[k].get(a, [])):>18s}' for a in arms))

    print('\n## paired against existing at block 3 and after relearning: mean difference (range over seeds)')
    for point in ('block3', 'relearn'):
        base = {s: measures(R[f'existing/seed{s}']['snapshots'][point]) for s in seeds if f'existing/seed{s}' in R}
        for a in arms[1:]:
            diffs = {}
            for s in seeds:
                if f'{a}/seed{s}' not in R or s not in base:
                    continue
                m = measures(R[f'{a}/seed{s}']['snapshots'][point])
                for k in m:
                    if k.startswith(('matching', 'transfer')) and k in base[s]:
                        diffs.setdefault(k, []).append(m[k] - base[s][k])
            print(f'\n{point} {a}')
            for k in sorted(diffs):
                xs = diffs[k]
                print(f'   {k:50s} {np.mean(xs):+.2f} ({min(xs):+.2f} to {max(xs):+.2f})')

    if lm:
        L = load(lm)['results']
        same = n = 0
        for s in seeds:
            mine, ref = R.get(f'existing/seed{s}'), L.get(f'sparse48/seed{s}')
            if mine is None or ref is None:
                continue
            for point in POINTS:
                a, b = measures(mine['snapshots'][point]), measures(ref['snapshots'][point])
                for k in b:
                    n += 1
                    same += abs(a.get(k, np.nan) - b[k]) < 1e-9
        print(f'\nreference: existing equals limited-memory main sparse48 in {same} of {n} measures')


if __name__ == '__main__':
    main(*sys.argv[1:])
