"""Everyday relations v2 addendum (after the freeze; docs/everyday-relations-v2-results.md):
(a) shuffled-world controls for the closed-world arms (the frozen design ran shuffled controls only in open world, whose
    negatives turned out to train the held-out facts as false);
(b) fast mapping without supplied exclusions (user, 2026-11-05: 'a bird cannot be a fish' must be learned): dax and fep
    fitted from true facts only, under the Gaussian prior of the trained codes, on the real-world v2 checkpoints.
Usage: .venv/bin/python -m experiments.everyday_relations_v2_addendum
"""
import os
import resource
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from src import everyday_world_v2 as W

OUT = V2.ROOT / 'reports' / 'everyday-relations-v2-addendum'


def fm_task(t):
    arm, regime, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    world = W.build()
    model = V2.Net(arm, len(world.entities), len(world.relations), torch.Generator().manual_seed(0))
    model.load_state_dict(torch.load(V2.RUNS / f'{arm}-{regime}-real-seed{seed}.pt'))
    prior = V2.gaussian_prior(model.E)
    out = {}
    for name, (given, read, ks, row) in W.fast_mapping_specs().items():
        if name not in ('dax', 'fep'):
            continue
        cls = {'dax': 'bird', 'fep': 'fish'}[name]
        read = read + [x for x in row if x[2] in W.CLASSES]   # exclusions are read, not given
        for k in ks:
            out[f'{name}/prior-only/k{k}'] = V2.read_fit(model, world, name, given, read, [], k, seed, prior, cls)
    return f'{arm}/{regime}/seed{seed}', out, {'wall_seconds': time.perf_counter() - t0,
                                                'cpu_seconds': time.process_time() - c0}


def main():
    t0 = time.perf_counter()
    workers = int(os.environ.get('EV_WORKERS', 8))
    ctl = [(a, 'closed', 'shuffled', s) for a in ('two', 'tworel', 'flat') for s in V2.SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(V2.task, V2.RUNS / 'done', '{0}-{1}-{2}-seed{3}'), ctl, chunksize=1)
    t1 = time.perf_counter()
    fm = [(a, rg, s) for a, rg, wd in V2.CONFIGS if wd == 'real' for s in V2.SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(workers) as pool:
        fres = pool.map(fm_task, fm, chunksize=1)
    V2.save_json(f'{OUT}.json', {'shuffled_closed': {k: r for k, r, _ in res}, 'fast_mapping_prior_only': {k: r for k, r, _ in fres}})
    V2.save_json(f'{OUT}-runtime.json', V2.runtime({
        'wall_seconds': time.perf_counter() - t0, 'shuffled_wall_seconds': t1 - t0,
        'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res) + sum(t['cpu_seconds'] for _, _, t in fres)),
        'tasks': {**{k: t for k, _, t in res}, **{'fm/' + k: t for k, _, t in fres}}}))
    tests = sorted(res[0][1]['tests'])
    print('shuffled closed (bal_acc, auroc)')
    for a in ('two', 'tworel', 'flat'):
        rs = [r for k, r, _ in res if r['arm'] == a]
        print(' ', a, {t: (round(np.mean([r['tests'][t]['bal_acc'] if r['tests'][t]['bal_acc'] is not None else r['tests'][t]['acc'] for r in rs]), 2),
                           round(np.mean([r['tests'][t]['auroc'] or np.nan for r in rs]), 2)) for t in tests})
    print('fast mapping prior-only: properties+exclusions bal_acc; is-a other classes rejected')
    for a, rg, wd in V2.CONFIGS:
        if wd != 'real':
            continue
        rs = [r for k, r, _ in fres if k.startswith(f'{a}/{rg}/')]
        print(' ', f'{a}/{rg}', {k: (round(np.mean([r[k]['read']['bal_acc'] for r in rs]), 2),
                                     round(np.mean([r[k]['by_relation']['is-a']['tnr'] for r in rs]), 2)) for k in rs[0]})


if __name__ == '__main__':
    main()
