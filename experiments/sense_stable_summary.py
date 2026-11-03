"""Tables for experiments.sense_stable.
Usage: python3 -m experiments.sense_stable_summary reports/sense-stable-v1-pilot.json.gz [reports/sense-adapt-v1-pilot.json.gz]
Drift: reads before a kind's facts against reads after the previous kind's facts (centres opened and gate refitted, nothing written between).
Interference: reads after a kind's facts against reads just before them.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

MATCHED = 'matched-0.13'
VIEWS = ('mild', 'strong')


def load(p):
    p = Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def rng(xs):
    return f'{min(xs):+.2f}' if max(xs) - min(xs) < .005 else f'{min(xs):+.2f} to {max(xs):+.2f}'


def rngp(xs):
    return f'{min(xs):.2f}' if max(xs) - min(xs) < .005 else f'{min(xs):.2f}–{max(xs):.2f}'


def ordered(reads):
    """Stage tags in run order: the report stores keys sorted, which puts 'post' before 'pre'."""
    return sorted(reads, key=lambda t: (int(t.split(':')[0]), t.endswith(':post')))


def delta(x, y):
    """Largest change between two score lists; a query with no unit holding its fact scores -inf, and losing or gaining such a unit counts as infinite."""
    x, y = np.array(x, float), np.array(y, float)
    both_inf = np.isinf(x) & np.isinf(y) & (x == y)
    d = np.where(both_inf, 0., np.abs(np.where(both_inf, 0., x) - np.where(both_inf, 0., y)))
    return float(np.nan_to_num(d, nan=np.inf, posinf=np.inf).max())


def compare(a, b, tau):
    """Per-query changes from read a to read b of the same kind and view."""
    qa, qb = a['q'], b['q']
    same_unit = [(x, y) == (z, w) for x, y, z, w in zip(qa['centre'], qa['unit'], qb['centre'], qb['unit'])]
    return {'own': delta(qa['own'], qb['own']), 'best': delta(qa['score'], qb['score']),
            'unit_changes': int(len(same_unit) - sum(same_unit)), 'value_flips': int(sum(x != y for x, y in zip(qa['value_ok'], qb['value_ok']))),
            'accept_flips': int(sum((x >= tau) != (y >= tau) for x, y in zip(qa['score'], qb['score']))),
            'unseen': delta(a['u']['score'], b['u']['score']), 'unseen_flips': int(sum((x >= tau) != (y >= tau) for x, y in zip(a['u']['score'], b['u']['score']))),
            'queries': len(same_unit)}


def pairs(r, kind):
    """(earlier tag, later tag, kinds read in both) for drift and interference."""
    tags = ordered(r['reads'])
    out = []
    for i, t in enumerate(tags):
        if i == 0:
            continue
        prev = tags[i - 1]
        kinds = sorted({k.split('|')[1].rsplit('-', 1)[0] for k in r['reads'][prev]} & {k.split('|')[1].rsplit('-', 1)[0] for k in r['reads'][t]})
        out.append(('drift' if t.endswith(':pre') else 'interference', prev, t, kinds))
    return [p for p in out if p[0] == kind]


def main(path, adapt=None):
    d = load(path)
    R, arms, tau = d['results'], d['arms'], d['tau']

    for which in ('drift', 'interference'):
        print(f'\n## {which}: summed over stages, kinds and views; max over runs for scores')
        print(f'{"arm":12s} {"max|d own|":>11s} {"max|d best|":>12s} {"unit chg":>9s} {"value flips":>12s} {"accept flips":>13s} {"max|d unseen|":>14s} {"unseen flips":>13s} {"queries":>8s}')
        for arm in arms:
            tot = {'own': 0., 'best': 0., 'unit_changes': 0, 'value_flips': 0, 'accept_flips': 0, 'unseen': 0., 'unseen_flips': 0, 'queries': 0}
            for r in R.values():
                for _, a, b, kinds in pairs(r, which):
                    for k in kinds:
                        for v in VIEWS:
                            c = compare(r['reads'][a][f'{arm}|{k}-{v}'], r['reads'][b][f'{arm}|{k}-{v}'], tau)
                            for x in tot:
                                tot[x] = max(tot[x], c[x]) if x in ('own', 'best', 'unseen') else tot[x] + c[x]
            print(f'{arm:12s} {tot["own"]:11.4f} {tot["best"]:12.4f} {tot["unit_changes"]:9d} {tot["value_flips"]:12d} {tot["accept_flips"]:13d} {tot["unseen"]:14.4f} {tot["unseen_flips"]:13d} {tot["queries"]:8d}')

    print('\n## end minus just after teaching, kinds taught before the last (mean over kinds, range over runs; worst single kind)')
    measures = [('matched value', lambda x: x[MATCHED]['value']), ('value at tau', lambda x: x['tau']['value']), ('address, all', lambda x: x['answer-all']['address']),
                ('value, all', lambda x: x['answer-all']['value']), ('unseen at tau', lambda x: x['unseen_at_tau'])]
    for v in VIEWS:
        for name, f in measures:
            for arm in arms:
                means, worst = [], []
                for r in R.values():
                    posts = [t for t in ordered(r['reads']) if t.endswith(':post')]
                    ds = [f(r['reads'][posts[-1]][f'{arm}|{k}-{v}']) - f(r['reads'][posts[s]][f'{arm}|{k}-{v}']) for s, k in enumerate(r['kinds'][:-1])]
                    means.append(float(np.mean(ds)))
                    worst.append(min(ds) if name != 'unseen at tau' else max(ds))
                w = min(worst) if name != 'unseen at tau' else max(worst)
                print(f'{v:6s} {name:14s} {arm:12s} mean {rng(means):>16s}   worst {w:+.2f}')

    print('\n## final reads (mean over kinds; range over runs)')
    for v in VIEWS:
        for arm in arms:
            vals = {name: [] for name, _ in measures}
            for r in R.values():
                last = [t for t in ordered(r['reads']) if t.endswith(':post')][-1]
                for name, f in measures:
                    vals[name].append(float(np.mean([f(r['reads'][last][f'{arm}|{k}-{v}']) for k in r['kinds']])))
            print(f'{v:6s} {arm:12s} ' + '  '.join(f'{n} {rngp(x)}' for n, x in vals.items()))

    print('\n## cost')
    rt = Path(str(path).replace('.json.gz', '-runtime.json').replace('.json', '-runtime.json') if not str(path).endswith('-runtime.json') else path)
    if rt.exists():
        t = json.loads(rt.read_text())
        print('read seconds per query:', {k: {a: f'{s * 1e3:.2f} ms' for a, s in v['read_seconds_per_query'].items()} for k, v in t['tasks'].items()})
    for run, r in R.items():
        print(run, 'forced writes', r['forced_writes'], 'units', r['units'], 'state MB', {a: round(sum(b.values()) / 2 ** 20, 1) for a, b in r['state_bytes'].items()},
              'centres', [s['centres'] for s in r['stages']])

    if adapt:
        A = load(adapt)['results']
        same = n = 0
        keys = ('tau', 'answer-all', MATCHED, 'unseen_at_tau', 'resemblance')
        for run, r in R.items():
            for t, stage in A[run]['reads'].items():
                for key, rec in stage.items():
                    arm, rest = key.split('|')
                    if arm != 'found/online':
                        continue
                    n += 1
                    mine = r['reads'][f'{t}:post'][f'online|{rest}']
                    same += all(mine[x] == rec[x] for x in keys)
        print(f'\nreference: online equals sense_adapt found/online in {same} of {n} reads (every stage)')


if __name__ == '__main__':
    main(*sys.argv[1:])
