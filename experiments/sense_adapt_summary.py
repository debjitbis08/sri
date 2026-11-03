"""Tables for experiments.sense_adapt: final reads per arm, stability of each kind from just after teaching to the end, centres opened, and the reference check against `pilot mix`.
Usage: python3 -m experiments.sense_adapt_summary reports/sense-adapt-v1-pilot.json.gz [reports/general-encoder-v1-mix-mix.json.gz]
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

MATCHED = 'matched-0.13'


def load(p):
    p = Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def rng(xs):
    return f'{min(xs):.2f}' if max(xs) - min(xs) < .005 else f'{min(xs):.2f}–{max(xs):.2f}'


def main(path, mix=None):
    d = load(path)
    R, arms = d['results'], d['arms']
    final = {run: r['reads'][list(r['reads'])[-1]] for run, r in R.items()}
    metrics = [('matched', lambda x: x[MATCHED]['value']), ('tau', lambda x: x['tau']['value']), ('unseen@tau', lambda x: x['unseen_at_tau']), ('same class', lambda x: x['resemblance']['all_same_class'])]
    for view in ('mild', 'strong'):
        print(f'\n## final reads, {view} (mean over kinds; range over runs) | rendered kinds only, matched')
        print(f'{"arm":20s} ' + ' '.join(f'{m:>11s}' for m, _ in metrics) + '   rendered')
        for a in arms:
            cells = []
            for _, f in metrics:
                cells.append(rng([sum(f(final[run][f'{a}|{k}-{view}']) for k in R[run]['kinds']) / 6 for run in R]))
            rend = rng([sum(final[run][f'{a}|{k}-{view}'][MATCHED]['value'] for k in R[run]['kinds'] if k != 'natural') / 5 for run in R])
            print(f'{a:20s} ' + ' '.join(f'{c:>11s}' for c in cells) + f'   {rend}')

    print('\n## stability: matched value at the end minus just after teaching (mean over kinds taught before the last; range over runs), and the worst single kind')
    for view in ('mild', 'strong'):
        for a in arms:
            means, worst = [], []
            for run, r in R.items():
                stages = list(r['reads'])
                ds = []
                for s, k in enumerate(r['kinds'][:-1]):
                    first = r['reads'][stages[s]][f'{a}|{k}-{view}'][MATCHED]['value']
                    last = r['reads'][stages[-1]][f'{a}|{k}-{view}'][MATCHED]['value']
                    ds.append(last - first)
                means.append(sum(ds) / len(ds))
                worst.append(min(ds))
            print(f'{view:6s} {a:20s} mean {rng(means):>12s}   worst {min(worst):+.2f}')

    print('\n## centres opened online (shift rule); which kinds\' passing batches formed or joined each')
    for run, r in R.items():
        joined = Counter()
        centre_of_forming, forming_tags = {}, []
        for e in r['shift_log']:
            if e['joined'] is None:
                forming_tags.append(e['tag'])
            else:
                joined[(e['joined'], e['tag'])] += 1
        print(run, 'centres after each kind:', [s['centres'] for s in r['stages']], 'kinds:', r['kinds'])
        print('   joins (centre, kind): count', dict(joined), '  forming batches by kind:', dict(Counter(forming_tags)))
        print('   gate refits:', [(g['centres'], round(g['loss'], 2)) for g in r['gate_refits']], ' advance clusters:', r['advance']['clusters'])

    print('\n## TrOCR weight on queries at the end (mild), by kind')
    for a in arms:
        if 'trocr_weight' in next(iter(final.values()))[f'{a}|natural-mild']:
            print(f'{a:20s}', {k: rng([final[run][f'{a}|{k}-mild']['trocr_weight'] for run in R]) for k in ('natural', 'mnist', 'tree', 'thai_digits', 'banana', 'chinese')})

    if mix:
        m = load(mix)['results']
        same = n = 0
        for run, r in R.items():
            for key, rec in final[run].items():
                arm, rest = key.split('|')
                if arm != 'told/advance':
                    continue
                n += 1
                same += m[run]['reads'][f'joined/learned|{d["tau"]}|{rest}'] == rec
        print(f'\nreference: told/advance equals pilot mix joined/learned in {same} of {n} final reads')


if __name__ == '__main__':
    main(*sys.argv[1:])
