"""Why taught truth costs on chains (docs/taught-truth-cost-diagnosis.md): reads on multi-step reader v1 checkpoints
(pred-one, unit-one), no training."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from experiments import multistep_reader as MS
from experiments import taught_truth as TT
from src import everyday_truth as T
from src import everyday_truth_v2 as T2

TESTS = ('L1-size', 'L1-anc', 'L2-named', 'P6-grandparent', 'P4', 'P3', 'P1-default')


def split(st, seed):
    rng = np.random.default_rng(seed)
    trainable = st.test == ''
    ind_idx = np.flatnonzero(trainable & (st.kind == 'individual'))
    val = np.zeros(len(st.a), bool)
    val[rng.choice(ind_idx, int(round(TT.VAL_FRAC * len(ind_idx))), replace=False)] = True
    return trainable & ~val


def test_mask(st, tst, ind):
    if tst == 'P1-default':
        return ind & (st.test == 'P1') & (st.sub == 'default')
    return ind & (st.test == tst)


def fit_logistic(x, y, w=None):
    """1-D logistic regression a*x + b by Newton steps; balanced class weights."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if w is None:
        w = np.where(y == 1, .5 / max(y.sum(), 1), .5 / max((1 - y).sum(), 1))
    a, b = 1.0, 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([(w * (p - y) * x).sum(), (w * (p - y)).sum()])
        h = np.array([[(w * p * (1 - p) * x * x).sum(), (w * p * (1 - p) * x).sum()],
                      [(w * p * (1 - p) * x).sum(), (w * p * (1 - p)).sum()]]) + 1e-6 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(h, g)
    return a, b


def bal(pred, y):
    y = np.asarray(y, bool)
    return .5 * ((pred[y]).mean() + (~pred[~y]).mean())


def quant(x):
    return [round(float(q), 2) for q in np.quantile(x, [.1, .5, .9])] if len(x) else None


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    st, _ = T2.build()
    TT.value_idx_cache = TT.value_idx(st)
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    vi = {v: k for k, v in enumerate(T.VALUES)}
    out = {}
    for seed in MS.SEEDS:
        train = split(st, seed)
        # taught truth: margins logit_T - logit_F, and argmax
        pm = TT.Pred('pred', len(st.entities), len(st.relations), torch.Generator().manual_seed(0))
        pm.load_state_dict(torch.load(MS.RUNS / f'pred-one-seed{seed}.pt'))
        lg = TT.predict_values(pm, st, np.ones(len(st.a), bool) & ind)
        idx = np.flatnonzero(ind)
        pos_of = {i: k for k, i in enumerate(idx)}
        margin_p = np.full(len(st.a), np.nan)
        argmax_p = np.full(len(st.a), -1)
        margin_p[idx] = lg[:, vi['T']] - lg[:, vi['F']]
        argmax_p[idx] = lg.argmax(1)
        # unit
        um = V2.Net('tworel', len(st.entities), len(st.relations), torch.Generator().manual_seed(0))
        um.load_state_dict(torch.load(MS.RUNS / f'unit-one-seed{seed}.pt'))
        with torch.no_grad():
            logit_u = np.full(len(st.a), np.nan)
            logit_u[idx] = um(torch.as_tensor(st.a[idx]), torch.as_tensor(st.r[idx]), torch.as_tensor(st.b[idx])).numpy()
        tr = train & ind
        y = st.world_y
        # global calibration fitted on trained rows
        ag, bg = fit_logistic(margin_p[tr], y[tr])
        rec = {}
        for tst in TESTS:
            m = test_mask(st, tst, ind)
            yy = y[m].astype(bool)
            rels = np.unique(st.r[m])
            # per-relation calibration (trained rows of the test's relations)
            cal = np.zeros(m.sum(), bool)
            for r in rels:
                rr = tr & (st.r == r)
                a, b = fit_logistic(margin_p[rr], y[rr]) if (y[rr] == 1).any() and (y[rr] == 0).any() else (ag, bg)
                mm = st.r[m] == r
                cal[mm] = a * margin_p[m][mm] + b > 0
            am = argmax_p[m]
            share = lambda sel: {v: round(float((am[sel] == k).mean()), 2) for v, k in vi.items() if (am[sel] == k).any()}
            trr = tr & np.isin(st.r, rels)
            rec[tst] = {
                'pred_argmax_bal': float(bal(am == vi['T'], yy) if False else .5 * ((am[yy] == vi['T']).mean() + (am[~yy] == vi['F']).mean())),
                'pred_T_vs_F_bal': float(bal(margin_p[m] > 0, yy)),
                'pred_calibrated_global_bal': float(bal(ag * margin_p[m] + bg > 0, yy)),
                'pred_calibrated_per_relation_bal': float(bal(cal, yy)),
                'pred_auroc': R1.auroc(margin_p[m], yy),
                'unit_bal': float(bal(logit_u[m] > 0, yy)),
                'unit_auroc': R1.auroc(logit_u[m], yy),
                'pred_value_share_heldout_true': share(yy), 'pred_value_share_heldout_false': share(~yy),
                'pred_margin_q10_50_90': {'trained_true': quant(margin_p[trr & (y == 1)]), 'trained_false': quant(margin_p[trr & (y == 0)]),
                                          'heldout_true': quant(margin_p[m][yy]), 'heldout_false': quant(margin_p[m][~yy])},
                'unit_logit_q10_50_90': {'trained_true': quant(logit_u[trr & (y == 1)]), 'trained_false': quant(logit_u[trr & (y == 0)]),
                                         'heldout_true': quant(logit_u[m][yy]), 'heldout_false': quant(logit_u[m][~yy])},
            }
        vc = F.normalize(pm.E[TT.value_idx_cache].detach(), dim=1)
        rec['value_code_cos'] = {f'{a}-{b}': round(float(vc[vi[a]] @ vc[vi[b]]), 2)
                                 for a, b in (('T', 'F'), ('T', 'U'), ('F', 'U'), ('T', 'always'), ('F', 'never'))}
        rec['temperature_scale'] = float(pm.log_scale.exp())
        out[seed] = rec
    rt = {'scope': 'whole-script timing', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    V2.save_json(V2.ROOT / 'reports' / 'taught-truth-cost-diagnosis.json', {str(k): v for k, v in out.items()})
    V2.save_json(V2.ROOT / 'reports' / 'taught-truth-cost-diagnosis-runtime.json', rt)
    seeds = list(out)
    print('test'.ljust(16), 'argmax  TvsF  cal-g  cal-r  aurocP | unit  aurocU')
    for tst in TESTS:
        g = lambda k: np.mean([out[s][tst][k] for s in seeds])
        print(tst.ljust(16), f"{g('pred_argmax_bal'):.2f}   {g('pred_T_vs_F_bal'):.2f}  {g('pred_calibrated_global_bal'):.2f}  "
              f"{g('pred_calibrated_per_relation_bal'):.2f}  {g('pred_auroc'):.2f}  | {g('unit_bal'):.2f}  {g('unit_auroc'):.2f}")
    s0 = seeds[0]
    for tst in TESTS:
        print(tst, 'shares T/F held-out', out[s0][tst]['pred_value_share_heldout_true'], out[s0][tst]['pred_value_share_heldout_false'])
        print('   pred margin', out[s0][tst]['pred_margin_q10_50_90'])
        print('   unit logit ', out[s0][tst]['unit_logit_q10_50_90'])
    print('value cos', {s: out[s]['value_code_cos'] for s in seeds}, 'scale', [round(out[s]['temperature_scale'], 1) for s in seeds])
    print(rt)


if __name__ == '__main__':
    main()
