"""Taught truth v1 (docs/taught-truth-v1.md): one predictive reader that fills the value or tail slot; truth values are
learned codes; K3 connectives, kind quantities and the kind -> individual link are taught by examples.

Usage: .venv/bin/python -m experiments.taught_truth main      (rerun to resume; EV_WORKERS, default 8)
       .venv/bin/python -m experiments.taught_truth summary
       EV_SMOKE=1 ... for a short check (100 steps, one seed)
"""
import json
import os
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from src import everyday_truth as T

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'taught-truth-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'taught-truth-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 100 if SMOKE else 3000
FIT_STEPS = 50 if SMOKE else 500
SEEDS = (440,) if SMOKE else (440, 441, 442, 443)
ARMS = ('unit', 'pred', 'pred-flat')
D, RD, HID, SLOT = 32, 16, 128, 8
LR, WD, INIT_STD, VAL_FRAC = 3e-3, 1e-4, 0.3, 0.05
GROUPS = ('individual', 'kind', 'anon', 'anon-isa', 'connective', 'fill')
S1_TESTS = ('P1', 'P1-default', 'P1-specific', 'P1-hierarchy', 'P2', 'P3', 'P4', 'P5', 'P6-colour', 'P6-grandparent',
            'G1', 'G2', 'G3', 'G4')
CONTROL_TESTS = ('P1', 'P2', 'P3', 'G1')


# ---------------------------------------------------------------- predictive reader

class Pred(nn.Module):
    def __init__(self, arm, n_ent, n_rel, g):
        super().__init__()
        self.arm = arm
        self.E = nn.Parameter(torch.randn(n_ent, D, generator=g) * INIT_STD)
        self.R = nn.Parameter(torch.randn(n_rel, RD, generator=g) * INIT_STD)
        self.mask = nn.Parameter(torch.randn(D, generator=g) * INIT_STD)
        self.slot = nn.Parameter(torch.randn(2, SLOT, generator=g) * INIT_STD)
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        qd = 2 * D if arm == 'pred-flat' else D
        if arm == 'pred':
            self.G = nn.Sequential(nn.Linear(2 * D + RD, HID), nn.ReLU(), nn.Linear(HID, D))
        self.P = nn.Sequential(nn.Linear(qd + RD + SLOT, HID), nn.ReLU(), nn.Linear(HID, D))
        R1._reinit(self, g)

    def out_codes(self, ca, rc, cb, gap):
        """ca, cb: (n, D) codes (cb ignored where gap == 1, replaced by MASK); gap: (n,) long."""
        cb = torch.where(gap[:, None] == 1, self.mask.expand_as(cb), cb)
        q = self.G(torch.cat([ca, cb, rc], 1)) if self.arm == 'pred' else torch.cat([ca, cb], 1)
        return self.P(torch.cat([q, rc, self.slot[gap]], 1))

    def logits(self, out, cand):
        return self.log_scale.exp() * F.normalize(out, dim=1) @ F.normalize(cand, dim=1).T


def value_idx(st):
    return torch.tensor([st.index(v) for v in T.VALUES])


# ---------------------------------------------------------------- training

def weights(st, rows):
    """Each statement group total 1/len(GROUPS); within individual rows, equal per (relation, value); within kind and anon,
    equal per value."""
    w = np.zeros(len(st.a), np.float32)
    present = [g for g in GROUPS if (rows & (st.kind == g)).any()]
    for g in present:
        m = rows & (st.kind == g)
        if g == 'individual':
            cells = {}
            for i in np.flatnonzero(m):
                cells.setdefault((st.r[i], st.v[i]), []).append(i)
            for idx in cells.values():
                w[idx] = 1 / (len(cells) * len(idx))
        elif g in ('kind', 'anon'):
            vals = np.unique(st.v[m])
            for v in vals:
                mm = m & (st.v == v)
                w[mm] = 1 / (len(vals) * mm.sum())
        else:
            w[m] = 1 / m.sum()
        w[m] /= len(present)
    return w


def train_pred(st, arm, seed, gen, train_rows):
    model = Pred(arm, len(st.entities), len(st.relations), gen)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    vidx = value_idx(st)
    w = torch.as_tensor(weights(st, train_rows))
    vrows = np.flatnonzero(train_rows & (st.gap == 0))
    frows = np.flatnonzero(train_rows & (st.gap == 1))
    va, vr, vb, vv = (torch.as_tensor(x[vrows]) for x in (st.a, st.r, st.b, st.v))
    fa, fr = torch.as_tensor(st.a[frows]), torch.as_tensor(st.r[frows])
    fmask = torch.zeros(len(frows), len(st.entities), dtype=torch.bool)
    for k, i in enumerate(frows):
        fmask[k, st.fill_answers[int(i)]] = True
    wv, wf = w[vrows], w[frows]
    zeros_v, ones_f = torch.zeros(len(vrows), dtype=torch.long), torch.ones(len(frows), dtype=torch.long)
    for step in range(STEPS):
        opt.zero_grad()
        out = model.out_codes(model.E[va], model.R[vr], model.E[vb], zeros_v)
        lv = F.cross_entropy(model.logits(out, model.E[vidx]), vv, reduction='none')
        loss = (lv * wv).sum()
        if len(frows):
            out_f = model.out_codes(model.E[fa], model.R[fr], model.E[fa], ones_f)
            lp = torch.log_softmax(model.logits(out_f, model.E), 1)
            lf = -torch.logsumexp(lp.masked_fill(~fmask, -1e9), 1)
            loss = loss + (lf * wf).sum()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict_values(model, st, rows):
    """(n, 7) logits for the value slot on the given rows."""
    i = np.flatnonzero(rows)
    out = model.out_codes(model.E[st.a[i]], model.R[st.r[i]], model.E[np.maximum(st.b[i], 0)],
                          torch.zeros(len(i), dtype=torch.long))
    return model.logits(out, model.E[value_idx(st)]).numpy()


def train_unit(st, seed, gen, train_rows):
    """v3 tworel with the built-in truth unit, trained on individual T/F statements only (closed world)."""
    m = train_rows & (st.kind == 'individual') & (st.world_y >= 0)
    model = V2.Net('tworel', len(st.entities), len(st.relations), gen)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    h, r, t = (torch.as_tensor(x[m]) for x in (st.a, st.r, st.b))
    y = torch.as_tensor(st.world_y[m].astype(np.float32))
    wt = V2.rel_weights(st.r[m], st.world_y[m], len(st.relations))
    for _ in range(STEPS):
        opt.zero_grad()
        (F.binary_cross_entropy_with_logits(model(h, r, t), y, reduction='none') * wt).sum().backward()
        opt.step()
    return model


# ---------------------------------------------------------------- reads

def tf_metrics(logits, world_y):
    """World-truth read: positive iff predicted T; negative correct iff predicted F; U is wrong either way."""
    vi = {v: k for k, v in enumerate(T.VALUES)}
    pred = logits.argmax(1)
    y = np.asarray(world_y).astype(bool)
    tpr = float((pred[y] == vi['T']).mean()) if y.any() else None
    tnr = float((pred[~y] == vi['F']).mean()) if (~y).any() else None
    score = logits[:, vi['T']] - logits[:, vi['F']]
    return {'n': int(len(y)), 'tpr': tpr, 'tnr': tnr,
            'bal_acc': .5 * (tpr + tnr) if tpr is not None and tnr is not None else None,
            'auroc': R1.auroc(score, y), 'u_rate': float((pred == vi['U']).mean())}


def value_accuracy(logits, expected):
    """Accuracy against expected value names, overall and per expected value."""
    pred = np.array(T.VALUES)[logits.argmax(1)]
    expected = np.asarray(expected)
    per = {v: float((pred[expected == v] == v).mean()) for v in np.unique(expected)}
    return {'acc': float((pred == expected).mean()), 'mean_recall': float(np.mean(list(per.values()))), 'per_value': per,
            'pred_counts': {v: int((pred == v).sum()) for v in T.VALUES}}


def reads_pred(model, st, world, train_rows, val_rows):
    out = {}
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    for tst in S1_TESTS:
        if tst in ('P1-default', 'P1-specific'):
            m = ind & (st.test == 'P1') & (st.sub == tst.split('-')[1])
        else:
            m = ind & (st.test == tst)
        if m.any():
            out[tst] = tf_metrics(predict_values(model, st, m), st.world_y[m])
    out['fit-individual'] = tf_metrics(predict_values(model, st, train_rows & ind), st.world_y[train_rows & ind])
    out['val-individual'] = tf_metrics(predict_values(model, st, val_rows & ind), st.world_y[val_rows & ind])
    # S2 kind quantities
    for tag, m in (('S2', (st.kind == 'kind') & (st.test == 'S2')), ('fit-kind', (st.kind == 'kind') & (st.test == ''))):
        out[tag] = value_accuracy(predict_values(model, st, m), st.sub[m])
    # S3 default read on P1 held-out animals (kind relations only), S3b anon
    names = np.array(st.entities)
    m = ind & (st.test == 'P1') & np.isin(names[st.b], [v for vals in T.KIND_RELS.values() for v in vals]) \
        & np.isin(np.array(st.relations)[st.r], list(T.KIND_RELS))
    exp = [T.DEFAULT[T.quantity(T.W3.ANIMALS[names[a]][0], st.relations[r], names[b])] for a, r, b in
           zip(st.a[m], st.r[m], st.b[m])]
    q = [T.quantity(T.W3.ANIMALS[names[a]][0], st.relations[r], names[b]) for a, r, b in zip(st.a[m], st.r[m], st.b[m])]
    lg = predict_values(model, st, m)
    out['S3-default'] = value_accuracy(lg, exp)
    out['S3-default-by-quantity'] = {qq: value_accuracy(lg[np.array(q) == qq], np.array(exp)[np.array(q) == qq])['acc']
                                     for qq in sorted(set(q))}
    for tag, mm in (('S3b', st.test == 'S3b'), ('fit-anon', (st.kind == 'anon') & (st.test == ''))):
        lg = predict_values(model, st, mm)
        exp = np.array(T.VALUES)[st.v[mm]]
        out[tag] = value_accuracy(lg, exp)
        out[tag + '-by-quantity'] = {qq: value_accuracy(lg[st.sub[mm] == qq], exp[st.sub[mm] == qq])['acc']
                                     for qq in np.unique(st.sub[mm])}
    # S4 don't know
    for tag, mm in (('S4a', st.test == 'S4a'), ('S4b-P7', ind & (st.test == 'P7')),
                    ('fit-unknown-colour', (st.sub == 'unknown-colour') & (st.test == ''))):
        lg = predict_values(model, st, mm)
        out[tag] = {'u_rate': float((lg.argmax(1) == T.VALUES.index('U')).mean()), 'n': int(mm.sum())}
    ctl = ind & np.isin(st.test, CONTROL_TESTS)
    out['S4-control-u-rate'] = float((predict_values(model, st, ctl).argmax(1) == T.VALUES.index('U')).mean())
    # connectives and fill fit
    mm = st.kind == 'connective'
    out['connectives'] = value_accuracy(predict_values(model, st, mm), np.array(T.VALUES)[st.v[mm]])['acc']
    fr = np.flatnonzero((st.kind == 'fill') & train_rows)
    with torch.no_grad():
        o = model.out_codes(model.E[st.a[fr]], model.R[st.r[fr]], model.E[st.a[fr]], torch.ones(len(fr), dtype=torch.long))
        top = model.logits(o, model.E).argmax(1).numpy()
    out['fill-hits1'] = float(np.mean([top[k] in st.fill_answers[int(i)] for k, i in enumerate(fr)]))
    return out


def reads_unit(model, st):
    out = {}
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    with torch.no_grad():
        for tst in S1_TESTS:
            if tst in ('P1-default', 'P1-specific'):
                m = ind & (st.test == 'P1') & (st.sub == tst.split('-')[1])
            else:
                m = ind & (st.test == tst)
            if m.any():
                s = model(torch.as_tensor(st.a[m]), torch.as_tensor(st.r[m]), torch.as_tensor(st.b[m])).numpy()
                out[tst] = R1.metrics(s, st.world_y[m])
    return out


def fast_map_dax(model, st, seed, arm):
    """dax given only is-a(dax, bird) = T; code kept among the known codes (prior), no exclusions given."""
    names = np.array(st.entities)
    known = [i for i, x in enumerate(st.entities) if x not in T.VALUES + ('nil',)]
    mu, P = V2.gaussian_prior(model.E[known])
    g = torch.Generator().manual_seed(seed * 1000 + 11)
    code = nn.Parameter(torch.randn(1, model.E.shape[1], generator=g) * INIT_STD)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([code], lr=V2.FIT_LR)
    ri = lambda rel: st.relations.index(rel)
    exps = T.default_expectations('bird')
    if arm == 'unit':
        target = torch.tensor([1.0])
        for _ in range(FIT_STEPS):
            opt.zero_grad()
            s = model.score_codes(code, model.R[[ri('is-a')]], model.E[[st.index('bird')]])
            dl = code[0] - mu
            (F.binary_cross_entropy_with_logits(s, target, reduction='sum') + .5 * dl @ P @ dl).backward()
            opt.step()
        with torch.no_grad():
            res = {}
            for q in ('always', 'usually', 'never'):
                items = [(rel, val, e) for rel, val, e, qq in exps if qq == q]
                s = model.score_codes(code.expand(len(items), -1), model.R[[ri(x[0]) for x in items]],
                                      model.E[[st.index(x[1]) for x in items]]).numpy()
                res[q] = float(np.mean((s > 0) == np.array([x[2] == 'T' for x in items])))
            others = [c for c in T.W3.CLASSES if c != 'bird']
            s = model.score_codes(code.expand(len(others), -1), model.R[[ri('is-a')] * len(others)],
                                  model.E[[st.index(c) for c in others]]).numpy()
            res['other_classes_rejected'] = float((s <= 0).mean())
        return res
    vT = torch.tensor([T.VALUES.index('T')])
    vidx = value_idx(st)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        out = model.out_codes(code, model.R[[ri('is-a')]], model.E[[st.index('bird')]], torch.zeros(1, dtype=torch.long))
        dl = code[0] - mu
        (F.cross_entropy(model.logits(out, model.E[vidx]), vT, reduction='sum') + .5 * dl @ P @ dl).backward()
        opt.step()
    with torch.no_grad():
        res = {}
        for q in ('always', 'usually', 'sometimes', 'never'):
            items = [(rel, val, e) for rel, val, e, qq in exps if qq == q]
            out = model.out_codes(code.expand(len(items), -1), model.R[[ri(x[0]) for x in items]],
                                  model.E[[st.index(x[1]) for x in items]], torch.zeros(len(items), dtype=torch.long))
            pred = np.array(T.VALUES)[model.logits(out, model.E[vidx]).argmax(1).numpy()]
            res[q] = float(np.mean(pred == np.array([x[2] for x in items])))
        others = [c for c in T.W3.CLASSES if c != 'bird']
        out = model.out_codes(code.expand(len(others), -1), model.R[[ri('is-a')] * len(others)],
                              model.E[[st.index(c) for c in others]], torch.zeros(len(others), dtype=torch.long))
        pred = np.array(T.VALUES)[model.logits(out, model.E[vidx]).argmax(1).numpy()]
        res['other_classes_rejected'] = float(np.mean(pred == 'F'))
    return res


# ---------------------------------------------------------------- tasks

def run(arm, seed):
    st, world = T.build()
    rng = np.random.default_rng(seed)
    trainable = st.test == ''
    ind_idx = np.flatnonzero(trainable & (st.kind == 'individual'))
    val = np.zeros(len(st.a), bool)
    val[rng.choice(ind_idx, int(round(VAL_FRAC * len(ind_idx))), replace=False)] = True
    train_rows = trainable & ~val
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if arm == 'unit':
        model = train_unit(st, seed, g, train_rows)
        res = {'tests': reads_unit(model, st)}
    else:
        model = train_pred(st, arm, seed, g, train_rows)
        res = {'tests': reads_pred(model, st, world, train_rows, val)}
    res['dax'] = fast_map_dax(model, st, seed, arm)
    res.update(arm=arm, seed=seed, params=R1.n_params(model))
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'fit_steps': FIT_STEPS, 'params': R1.n_params(model),
                 'trained_statements': int(train_rows.sum()), 'checkpoint_bytes': ck.stat().st_size}


def task(t):
    arm, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(arm, seed)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{arm}/seed{seed}', res, rt


def main():
    t0 = time.perf_counter()
    tasks = [(a, s) for a in ARMS for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(int(os.environ.get('EV_WORKERS', 8)),
                                                                 maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
                     'world_sha256': nc.sha256(T.__file__), 'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                     'tasks': {k: t for k, _, t in res}})
    st, _ = T.build()
    V2.save_json(f'{REPORT}.json', {'status': 'taught truth v1' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                    'statements': len(st.a), 'results': {k: r for k, r, _ in res}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    by = {}
    for r in res.values():
        by.setdefault(r['arm'], []).append(r)
    m = lambda xs: None if not [x for x in xs if x is not None] else round(float(np.mean([x for x in xs if x is not None])), 2)
    out = {}
    for arm, rs in by.items():
        o = {'tests': {}}
        for t in rs[0]['tests']:
            v = rs[0]['tests'][t]
            if isinstance(v, dict):
                o['tests'][t] = {k: m([r['tests'][t].get(k) for r in rs]) for k in v
                                 if not isinstance(v[k], dict) and k not in ('n', 'n_pos')}
                if 'per_value' in v:
                    o['tests'][t]['per_value'] = {k: m([r['tests'][t]['per_value'].get(k) for r in rs]) for k in v['per_value']}
                if all(isinstance(x, float) for x in v.values()):
                    o['tests'][t] = {k: m([r['tests'][t].get(k) for r in rs]) for k in v}
            else:
                o['tests'][t] = m([r['tests'][t] for r in rs])
        o['dax'] = {k: m([r['dax'][k] for r in rs]) for k in rs[0]['dax']}
        o['params'] = rs[0]['params']
        out[arm] = o
    V2.save_json(f'{REPORT}-summary.json', out)
    for arm, o in out.items():
        print('==', arm, 'params', o['params'])
        for t, v in o['tests'].items():
            print('  ', t.ljust(24), v)
        print('   dax', o['dax'])


if __name__ == '__main__':
    {'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
