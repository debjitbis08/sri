"""Tables for the label-free stability test in docs/general-frozen-encoder.md, from finished records. Only reads.

Usage: /data/venvs/sri-cuda/bin/python -m experiments.general_encoder_stability
"""
import glob
import gzip
import json

import numpy as np


def records():
    return [json.loads(gzip.decompress(open(f, 'rb').read())) for f in sorted(glob.glob('runs/general-encoder-v1-dino/done-stability/*.json.gz'))]


def rng(v):
    a, b = min(v), max(v)
    return f'{a:.2f}' if abs(a - b) < .005 else f'{a:.2f}–{b:.2f}'


def main():
    recs = records()
    print(len(recs), 'runs')
    for k, r, _ in recs:
        print(k, r['kinds'], 'centres after each kind:', [c['patch']['centres'] for c in r['centres']], [c['cls']['centres'] for c in r['centres']])
    arms = sorted({n.split('|')[0] for _, r, _ in recs for n in r['reads']['6:' + r['kinds'][-1]]})
    for view in ('mild', 'strong'):
        print(f'\n== {view}, matched value: first read (just after its own kind) -> last read (after all six); change; unseen at tau at end; resemblance same category at end')
        for arm in arms:
            first, last, unseen, res = [], [], [], []
            for _, r, _ in recs:
                kinds = r['kinds']
                for pos, kd in enumerate(kinds):
                    a = r['reads'][f'{pos + 1}:{kd}'][f'{arm}|{kd}-{view}']
                    b = r['reads'][f'6:{kinds[-1]}'][f'{arm}|{kd}-{view}']
                    first.append(a['matched-0.13']['value']); last.append(b['matched-0.13']['value'])
                    unseen.append(b['unseen_at_tau']); res.append(b['resemblance']['answered_same_class'])
            ch = np.array(last) - np.array(first)
            print(f'{arm:12s} first {np.mean(first):.2f} last {np.mean(last):.2f} | change mean {ch.mean():+.3f}, worst {ch.min():+.2f}, best {ch.max():+.2f} | unseen {np.mean(unseen):.2f} | resemblance {np.mean(res):.2f}')
    print('\n== assignment of queries to centres at the end (found arms): share going to the centre most of its kind goes to, per kind')
    for _, r, _ in recs:
        st = r['reads'][f'6:{r["kinds"][-1]}']
        for lay in ('patch', 'cls'):
            shares = []
            owner = {}
            for kd in r['kinds']:
                c = np.array(st[f'{lay}/found|{kd}-mild']['centre_of_queries'])
                owner[kd] = int(c.argmax()); shares.append(c.max() / c.sum())
            print(r['seed'], r['order'], lay, 'purity', np.round(shares, 2), 'distinct centres used', len(set(owner.values())))


if __name__ == '__main__':
    main()
