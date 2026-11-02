"""Tables for docs/natural-memory-v1-pilot.md, from finished task records (any finished subset). Only reads.

Usage: /data/venvs/sri-cuda/bin/python -m experiments.natural_memory_summary [done-memory|done-transport] [config] [tau] [matched-0.13|tau]
"""
import glob
import gzip
import json
import sys
from collections import defaultdict

ROOT = 'runs/natural-memory-v1'


def records(done='done-memory', tag=''):
    out = []
    for f in sorted(glob.glob(f'runs/natural-memory-v1{tag}/{done}/*.json.gz')):
        k, r, t = json.loads(gzip.decompress(open(f, 'rb').read()))
        out.append((k, r, t))
    return out


def rng(xs):
    if not xs:
        return '—'
    lo, hi = min(xs), max(xs)
    return f'{lo:.2f}' if abs(hi - lo) < .005 else f'{lo:.2f}–{hi:.2f}'


def table(recs, config, P, level, query, tau, read, score='value', points=('second', 'rest-4000', 'rest-16000')):
    arms = defaultdict(lambda: defaultdict(list))
    for k, r, _ in recs:
        if r['config'] != config or r['P'] != P:
            continue
        first, second = ('A', 'B') if r['order'] == 'AB' else ('B', 'A')
        for point in points:
            rr = r['reads'].get(point)
            if rr is None:
                continue
            for name, rec in rr.items():
                if name.startswith('_'):
                    continue
                e, lv, t = name.split('|')
                if lv != level or (float(t) != tau and not e.startswith('rebuild')):
                    continue
                if not rec or query not in rec:
                    continue
                q = rec[query]
                arms[e][(point, 'first')].append(q[read]['A' if first == 'A' else 'B'][score])
                arms[e][(point, 'second')].append(q[read]['B' if first == 'A' else 'A'][score])
                arms[e][(point, 'unseen')].append(q['unseen_at_tau'])
                arms[e][(point, 'thr')].append(q[read]['threshold'])
                arms[e][(point, 'units')].append(rec['units'])
    return arms


def show(recs, config, P, level, query, tau, read, score):
    arms = table(recs, config, P, level, query, tau, read, score)
    if not arms:
        return
    print(f'\n### {config} P={P} {level} {query} write-tau {tau} read {read} score {score}')
    print('arm | second: first, second | +4000: first, second | +16000: first, second | unseen@tau end | units end')
    for e in sorted(arms):
        a = arms[e]
        cell = lambda p: f"{rng(a[(p, 'first')])}, {rng(a[(p, 'second')])}"
        print(f"{e:20s} | {cell('second')} | {cell('rest-4000')} | {cell('rest-16000')} | {rng(a[('rest-16000', 'unseen')])} | {int(max(a[('rest-16000', 'units')], default=0))}")


def main():
    done = sys.argv[1] if len(sys.argv) > 1 else 'done-memory'
    recs = records(done)
    print('finished', len(recs), [k for k, _, _ in recs])
    print('gates', {k: r['gates'].get('matches_encoder_stage') for k, r, _ in recs})
    configs = sorted({r['config'] for _, r, _ in recs})
    for c in configs if len(sys.argv) < 3 else [sys.argv[2]]:
        for P in sorted({r['P'] for _, r, _ in recs if r['config'] == c}):
            for tau in (.7, .85, .95) if len(sys.argv) < 4 else [float(sys.argv[3])]:
                read = sys.argv[4] if len(sys.argv) > 4 else 'matched-0.13'
                for level, queries in (('instance-45', ('mild', 'strong')), ('instance-450', ('mild', 'strong')), ('class', ('class',))):
                    for q in queries:
                        for score in ('value', 'address') if level == 'class' else ('value',):
                            show(recs, c, P, level, q, tau, read, score)


if __name__ == '__main__':
    main()
