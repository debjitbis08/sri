"""Taught truth cost diagnosis, follow-up run (docs/taught-truth-cost-diagnosis.md; user-approved 2026-11-06): the
taught-truth reader (`pred`, answering by a predicted code) trained on individual T/F statements only, the same diet as
the built-in unit. Same statements, splits, seeds and steps as multi-step reader v1's one-pass arms."""
import os
import resource
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from experiments import multistep_reader as MS
from experiments import taught_truth as TT
from experiments import taught_truth_cost_diagnosis as DG
from src import everyday_truth_v2 as T2

NAME = 'taught-truth-tf-only'
RUNS, REPORT = V2.ROOT / 'runs' / NAME, V2.ROOT / 'reports' / NAME


def task(seed):
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    st, _ = T2.build()
    TT.value_idx_cache = TT.value_idx(st)
    train = DG.split(st, seed)
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    rows = train & ind                       # T/F individual statements only
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = TT.train_pred(st, 'pred', seed, g, rows)
    idx = np.flatnonzero(ind)
    lg = TT.predict_values(model, st, ind)
    vi = {v: k for k, v in enumerate(TT.T.VALUES)}
    margin = np.full(len(st.a), np.nan)
    margin[idx] = lg[:, vi['T']] - lg[:, vi['F']]
    am = np.full(len(st.a), -1)
    am[idx] = lg.argmax(1)
    res = {}
    for tst in DG.TESTS:
        m = DG.test_mask(st, tst, ind)
        yy = st.world_y[m].astype(bool)
        res[tst] = {'argmax_bal': float(.5 * ((am[m][yy] == vi['T']).mean() + (am[m][~yy] == vi['F']).mean())),
                    'T_vs_F_bal': float(DG.bal(margin[m] > 0, yy)), 'auroc': R1.auroc(margin[m], yy),
                    'margin_median': {'trained_true': float(np.median(margin[rows & (st.world_y == 1) & np.isin(st.r, np.unique(st.r[m]))])),
                                      'heldout_true': float(np.median(margin[m][yy])),
                                      'heldout_false': float(np.median(margin[m][~yy]))}}
    res['fit'] = float(DG.bal(margin[rows] > 0, st.world_y[rows].astype(bool)))
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'pred-tf-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(V2.ROOT)), 'sha256': V2.nc.sha256(ck)}
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'steps': TT.STEPS,
          'trained_statements': int(rows.sum()), 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return f'seed{seed}', res, rt


def main():
    t0 = time.perf_counter()
    with __import__('multiprocessing').get_context('fork').Pool(4, maxtasksperchild=1) as pool:
        out = pool.map(task, MS.SEEDS, chunksize=1)
    res = {k: r for k, r, _ in out}
    V2.save_json(f'{REPORT}.json', res)
    V2.save_json(f'{REPORT}-runtime.json', V2.runtime({'runner_sha256': V2.nc.sha256(__file__), 'workers': 4,
                                                       'wall_seconds': time.perf_counter() - t0,
                                                       'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in out)),
                                                       'tasks': {k: t for k, _, t in out}}))
    import json
    diag = json.loads((V2.ROOT / 'reports' / 'taught-truth-cost-diagnosis.json').read_text())
    print('test'.ljust(16), 'pred-all  pred-TF-only  unit   | AUROC: pred-all  pred-TF-only  unit | held-out true margin TF-only')
    for tst in DG.TESTS:
        pa = np.mean([diag[s][tst]['pred_argmax_bal'] for s in diag])
        pu = np.mean([diag[s][tst]['unit_bal'] for s in diag])
        aa = np.mean([diag[s][tst]['pred_auroc'] for s in diag])
        au = np.mean([diag[s][tst]['unit_auroc'] for s in diag])
        pt = np.mean([res[s][tst]['argmax_bal'] for s in res])
        at = np.mean([res[s][tst]['auroc'] for s in res])
        mt = np.median([res[s][tst]['margin_median']['heldout_true'] for s in res])
        print(tst.ljust(16), f'{pa:.2f}      {pt:.2f}          {pu:.2f}   |  {aa:.2f}       {at:.2f}         {au:.2f}  |  {mt:+.1f}')
    print('fit', [round(res[s]['fit'], 2) for s in res])


if __name__ == '__main__':
    main()
