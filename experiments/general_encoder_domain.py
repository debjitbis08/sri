"""Per-kind tables for docs/general-frozen-encoder.md, the domain and mixing tests, from their reports. Only reads.

Usage: python3 -m experiments.general_encoder_domain reports/general-encoder-v1-domain-memory.json.gz [tau]
"""
import gzip, json, sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())
r = d['results']
kinds = ['natural', 'mnist', 'tree', 'thai_digits', 'banana', 'chinese']
arms = [a if isinstance(a, str) else '/'.join(a) for a in d['arms']]
tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85


def per_kind(a, v, f):
    return [[f(r[run]['reads'][f'{a}|{tau}|{k}-{v}']) for run in r] for k in kinds]


def rng(xs):
    return f'{min(xs):.2f}' if max(xs) - min(xs) < .005 else f'{min(xs):.2f}–{max(xs):.2f}'


mean = lambda rows: [sum(c) / len(c) for c in zip(*rows)]
for metric, f in [('matched value', lambda x: x['matched-0.13']['value']), ('tau value', lambda x: x['tau']['value']), ('unseen at tau', lambda x: x['unseen_at_tau']), ('same class, all', lambda x: x['resemblance']['all_same_class'])]:
    for v in ['mild', 'strong']:
        print(f'\n## {metric}, {v}  (per kind: mean over 4 runs; then mean over kinds as a range over runs; rendered-only mean)')
        for a in arms:
            pk = per_kind(a, v, f)
            avg = mean(pk)
            rend = mean(pk[1:])
            print(f'{a:28s} ' + ' '.join(f'{sum(c)/len(c):.2f}' for c in pk) + f' | all {rng(avg)} | rendered {rng(rend)}')
print('\nunits:', {a: sorted(r[run]['reads'][f'{a}|{tau}|natural-mild']['units'] for run in r) for a in arms})
for run, res in r.items():
    if 'gate' in res:
        g = res['gate']
        print(run, 'clusters', [(c['served_by'], {k: v for k, v in c['kinds'].items() if v}) for c in g['clusters']], 'pile-up', {n: {d: round(v, 2) for d, v in x.items()} for n, x in g['pile_up_by_domain'].items()},
              'learned trocr weight', {k: round(v, 2) for k, v in g['learned']['trocr_weight_on_training_views'].items()}, 'loss', round(g['learned']['final_loss'], 3))
