"""Post-hoc training-sufficiency check for benchmark v1 (exploratory, development worlds only).

Asks whether the established arms lose because 1500 steps / lr >= 1e-3 undertrain them.
Usage: .venv/bin/python -m experiments.learning_benchmark_training_check
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from experiments.learning_benchmark import DEV_WORLDS, canonical, init_seed, score, world_data
from src.learning_benchmark import CONDITIONS, encode, train_ensemble


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (16, 64, 256)
ARMS = dict(coarse_local_vicreg={}, dense_linear_vicreg={}, mlp_vicreg=dict(width=64),
            conv_vicreg=dict(width=8), dae_paired=dict(width=64))
LRS, STEPS = (3e-4, 1e-3), (1500, 6000)


def job(args):
    torch.set_num_threads(1)
    arm, budget, steps = args
    extra = ARMS[arm]
    wd = 1e-2 if 'linear' in arm or 'coarse' in arm else 1e-4
    data = {w: world_data(w, 'aligned') for w in DEV_WORLDS}
    members = [(lr, w) for lr in LRS for w in DEV_WORLDS]
    params, base, records = train_ensemble(
        arm, [dict(lr=lr, wd=wd, **extra) for lr, _ in members],
        np.stack([data[w][0][0][:budget] for _, w in members]),
        np.stack([data[w][0][1][:budget] for _, w in members]),
        [init_seed(w, 'aligned', arm, budget, 0) for _, w in members], steps=steps)
    out = []
    for k, (lr, w) in enumerate(members):
        r2 = score(lambda h: encode(params, base, k, h), data[w][1])
        out.append(dict(arm=arm, budget=budget, steps=steps, lr=lr, world=w, r2=r2, **records[k]))
    return out


def main():
    jobs = [(a, b, s) for a in ARMS for b in BUDGETS for s in STEPS]
    jobs.sort(key=lambda j: j[1]*j[2]*(3 if j[0] == 'conv_vicreg' else 1), reverse=True)
    rows = []
    with Pool(8) as pool:
        for out in pool.imap_unordered(job, jobs):
            rows.extend(out)
    summary = {}
    for arm in ARMS:
        for budget in BUDGETS:
            for steps in STEPS:
                for lr in LRS:
                    sel = [r for r in rows if (r['arm'], r['budget'], r['steps'], r['lr']) == (arm, budget, steps, lr)]
                    summary[f'{arm}|{budget}|{steps}|{lr}'] = dict(
                        in_distribution=float(np.mean([r['r2']['in_distribution'] for r in sel])),
                        resilience=float(np.mean([np.mean([r['r2'][c] for c in CONDITIONS[1:]]) for r in sel])),
                        unsettled=float(np.mean([r['tail_relative_change'] > .05 for r in sel])))
    path = ROOT/'reports/single-module-learning-benchmark-v1-training-check.json'
    path.write_text(json.dumps(canonical(dict(status='exploratory post-hoc check; development worlds; aligned',
                                              summary=summary, rows=rows)), indent=2, sort_keys=True)+'\n')
    for key, v in summary.items():
        print(f"{key:40s} in {v['in_distribution']:.3f} res {v['resilience']:.3f} unsettled {v['unsettled']:.2f}")


if __name__ == '__main__':
    main()
