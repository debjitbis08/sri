"""Tables for docs/general-frozen-encoder.md, step 1, from the report. Only reads.

Usage: /data/venvs/sri-cuda/bin/python -m experiments.general_encoder_summary
"""
import gzip
import json
from pathlib import Path

import numpy as np

KINDS = ('natural', 'mnist', 'tree', 'thai_digits', 'banana', 'chinese')


def load(path):
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def report():
    p = Path('reports/general-encoder-v1-memory.json')
    return load(p if p.exists() else str(p) + '.gz')


def rng(v):
    a, b = min(v), max(v)
    return f'{a:.2f}' if abs(a - b) < .005 else f'{a:.2f}–{b:.2f}'


def mean_over_kinds(r, name, tau, view, read, score):
    return float(np.mean([r['reads'][f'{name}|{tau}|{k}-{view}'][read][score] for k in r['kinds']]))


def unseen(r, name, tau, view):
    return float(np.mean([r['reads'][f'{name}|{tau}|{k}-{view}']['unseen_at_tau'] for k in r['kinds']]))


def main():
    R = report()
    res = R['results']
    arms = ['/'.join(a) for a in R['arms']]
    print('arm | tau | mild matched, tau | strong matched, tau | unseen at tau mild, strong | answer-all address strong | units')
    for name in arms:
        for tau in R['taus']:
            if name.startswith('pixels') and tau != R['taus'][0]:
                continue
            g = lambda view, read, score='value': rng([mean_over_kinds(r, name, tau, view, read, score) for r in res.values()])
            u = lambda view: rng([unseen(r, name, tau, view) for r in res.values()])
            units = sorted(r['reads'][f'{name}|{tau}|natural-mild']['units'] for r in res.values())
            print(f"{name:28s} {tau} | {g('mild', 'matched-0.13')}, {g('mild', 'tau')} | {g('strong', 'matched-0.13')}, {g('strong', 'tau')} | {u('mild')}, {u('strong')} | {g('strong', 'answer-all', 'address')} | {units[0]}-{units[-1]}")


if __name__ == '__main__':
    main()
