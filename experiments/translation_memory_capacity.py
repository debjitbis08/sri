"""Post hoc capacity of the associative rule on oracle codes (docs/translation-memory-v1-pilot.md).

Independent random sparse source patterns and codebook targets (150 target words, as in the task), no host: the largest load this
exact rule and coding reach at 95% recovery from exact cues, as storage per fact against the explicit stores, and the work per read.
This is a best case for the memory, since real keys are correlated and imperfect.

Usage: .venv/bin/python -m experiments.translation_memory_capacity
"""
import json

import numpy as np
import torch

from experiments import number_systems as X
from experiments import translation_store as TS

nc = X.nc
GRID = ((256, 5), (256, 10), (512, 10), (512, 20), (1024, 20), (1024, 41))
LOADS = tuple(int(round(100 * 2 ** (i / 2))) for i in range(0, 15))
SEEDS, SAMPLE, KEY_WIDTH, TARGET = (299, 298), 500, 32, 0.95


def curve(m, k, seed):
    rng = nc.stream('tm_capacity', seed, m, k)
    cb = TS.codebook(150, m, k, rng)
    mem = TS.Associative(m, m, k, k, rng)
    idx = np.stack([rng.choice(m, k, replace=False) for _ in range(max(LOADS))])
    tg = torch.from_numpy(rng.integers(150, size=max(LOADS)))
    pattern = lambda a: torch.zeros(len(a), m).scatter_(1, torch.from_numpy(a), 1.)
    out, done = {}, 0
    for n in LOADS:
        for i in range(done, n, 1000):
            j = range(i, min(i + 1000, n))
            mem.write(pattern(idx[j]), cb[tg[list(j)]])
        done = n
        pick = rng.permutation(n)[:SAMPLE]
        res, info = mem.retrieve(pattern(idx[pick]))
        y = tg[torch.from_numpy(pick)]
        acc = {str(t): float((TS.target_scores(res[t], cb).argmax(1) == y).float().mean()) for t in (1, 16)}
        out[n] = {**acc, 'weights_on': float(mem.W.mean()), 'active_connections_16_steps': float(info['active_connections'].mean())}
    return out


def main():
    torch.set_num_threads(10)
    res = {}
    for m, k in GRID:
        cs = [curve(m, k, s) for s in SEEDS]
        joint = 2 * m
        rec = {'bytes': joint * joint / 8, 'curve': {str(n): {key: float(np.mean([c[n][key] for c in cs])) for key in cs[0][n]} for n in LOADS}}
        for t in ('1', '16'):
            ok = [n for n in LOADS if rec['curve'][str(n)][t] >= TARGET]
            rec[f'max_load_{t}'] = max(ok) if ok else 0
            rec[f'bytes_per_fact_{t}'] = rec['bytes'] / rec[f'max_load_{t}'] if ok else None
        rec['flat_sparse_bytes_per_fact'] = 2 * k + 2
        rec['flat_store_bytes_per_fact'] = 4 * KEY_WIDTH + 2
        res[f'{m},{k}'] = rec
        print(m, k, 'bytes', rec['bytes'], 'max load 1-step', rec['max_load_1'], '16-step', rec['max_load_16'], 'bytes/fact', rec['bytes_per_fact_1'], 'flat-sparse', rec['flat_sparse_bytes_per_fact'], 'flat-store', rec['flat_store_bytes_per_fact'], flush=True)
    (X.ROOT / 'reports' / 'translation-memory-v1-capacity.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
