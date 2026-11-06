"""Dual-route truth v1 (docs/dual-route-truth-v1.md): an intuitive scalar verdict and a taught logic layer that speaks
the same currency (support for, support against), on shared or separate codes; agreement, conflict and fixed
combination rules R1-R4.

Usage: .venv/bin/python -m experiments.dual_route_truth main      (rerun to resume; EV_WORKERS, default 6)
       .venv/bin/python -m experiments.dual_route_truth summary
       EV_SMOKE=1 ... for a short check (60 steps, one seed)
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
from experiments import taught_truth as TT
from src import everyday_truth as T
from src import everyday_truth_v2 as T2

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'dual-route-truth-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'dual-route-truth-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 60 if SMOKE else 3000
FIT_STEPS = 50 if SMOKE else 500
LOG_EVERY = 20 if SMOKE else 250
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('dual-shared', 'dual-separate')
D, RD, HID, SLOT = TT.D, TT.RD, TT.HID, TT.SLOT
SUPPORT = {'T': (1., 0.), 'F': (0., 1.), 'U': (0., 0.)}
QUANT = ('always', 'usually', 'sometimes', 'never')
TF_TESTS = ('P1-default', 'P2', 'P3', 'P4', 'P6-grandparent', 'P6-colour', 'G2', 'G4', 'L1-kind', 'L1-anc', 'L1-size',
            'L2-named')
RULES = ('R1-intuition', 'R2-logic', 'R3-logic-overrides', 'R4-conflict-unknown')


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HID), nn.ReLU(), nn.Linear(HID, o))


class Dual(nn.Module):
    def __init__(self, arm, n_ent, n_rel, g):
        super().__init__()
        self.shared = arm == 'dual-shared'
        p = lambda *s: nn.Parameter(torch.randn(*s, generator=g) * TT.INIT_STD)
        self.E_I, self.R_I = p(n_ent, D), p(n_rel, RD)
        if not self.shared:
            self.E_L_own, self.R_L_own = p(n_ent, D), p(n_rel, RD)
        self.G_I, self.S_I = mlp(2 * D + RD, D), mlp(D + RD, 1)
        self.G_L, self.S_L = mlp(2 * D + RD, D), mlp(D + RD, 2)
        self.P_L = mlp(D + RD + SLOT, D)                      # code head: kind quantities and fill-the-tail
        self.mask, self.slot = p(D), p(2, SLOT)
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.C = mlp(4 + RD, 2)                               # connective operator on support pairs
        R1._reinit(self, g)

    @property
    def E_L(self):
        return self.E_I if self.shared else self.E_L_own

    @property
    def R_L(self):
        return self.R_I if self.shared else self.R_L_own

    def intuition(self, a, r, b, ca=None):
        ca = self.E_I[a] if ca is None else ca
        rc = self.R_I[r]
        return self.S_I(torch.cat([self.G_I(torch.cat([ca, self.E_I[b], rc], 1)), rc], 1))[:, 0]

    def logic(self, a, r, b, ca=None):
        ca = self.E_L[a] if ca is None else ca
        rc = self.R_L[r]
        return torch.sigmoid(self.S_L(torch.cat([self.G_L(torch.cat([ca, self.E_L[b], rc], 1)), rc], 1)))

    def code_out(self, a, r, b, gap):
        ca, rc = self.E_L[a], self.R_L[r]
        cb = torch.where(gap[:, None] == 1, self.mask.expand(len(a), -1), self.E_L[b])
        return self.P_L(torch.cat([self.G_L(torch.cat([ca, cb, rc], 1)), rc, self.slot[gap]], 1))

    def code_logits(self, out, cand):
        return self.log_scale.exp() * F.normalize(out, dim=1) @ F.normalize(cand, dim=1).T

    def connective(self, p1, p2, r):
        return torch.sigmoid(self.C(torch.cat([p1, p2, self.R_L[r]], 1)))


# ---------------------------------------------------------------- reading states and rules

def logic_state(s):
    """s: (n, 2) support -> array of 'T', 'F', 'U', 'D'."""
    s = np.asarray(s)
    p, q = s[:, 0] > .5, s[:, 1] > .5
    return np.where(p & ~q, 'T', np.where(q & ~p, 'F', np.where(p & q, 'D', 'U')))


def rules(u, s):
    """Answers ('T', 'F', 'U', 'D') of the four fixed rules from intuition logits u and logic support s."""
    iv = np.where(np.asarray(u) > 0, 'T', 'F')
    lv = logic_state(s)
    decisive = np.isin(lv, ('T', 'F'))
    return {'R1-intuition': iv, 'R2-logic': lv, 'R3-logic-overrides': np.where(decisive, lv, iv),
            'R4-conflict-unknown': np.where(lv == iv, iv, 'U')}


def tf_bal(ans, y):
    y = np.asarray(y).astype(bool)
    return float(.5 * ((ans[y] == 'T').mean() + (ans[~y] == 'F').mean()))


# ---------------------------------------------------------------- training

def train(st, arm, seed, g, train_rows):
    model = Dual(arm, len(st.entities), len(st.relations), g)
    opt = torch.optim.Adam(model.parameters(), lr=TT.LR, weight_decay=TT.WD)
    w = torch.as_tensor(TT.weights(st, train_rows))
    vnames = np.array(T.VALUES)
    # logic support rows: individual, anon, anon-isa (values T/F/U)
    sup = np.flatnonzero(train_rows & np.isin(st.kind, ('individual', 'anon', 'anon-isa')))
    sup_t = torch.tensor([SUPPORT[vnames[v]] for v in st.v[sup]])
    # kind rows (code head over the 4 quantity codes)
    kin = np.flatnonzero(train_rows & (st.kind == 'kind'))
    qidx = torch.tensor([st.index(q) for q in QUANT])
    kin_t = torch.tensor([QUANT.index(vnames[v]) for v in st.v[kin]])
    # fill rows
    fil = np.flatnonzero(train_rows & (st.kind == 'fill'))
    fmask = torch.zeros(len(fil), len(st.entities), dtype=torch.bool)
    for k, i in enumerate(fil):
        fmask[k, st.fill_answers[int(i)]] = True
    # connectives: K3 tables on support corners
    con = np.flatnonzero(st.kind == 'connective')
    names = np.array(st.entities)
    corner = lambda x: SUPPORT.get(x, (0., 0.))
    con_p1 = torch.tensor([corner(names[a]) for a in st.a[con]])
    con_p2 = torch.tensor([corner(names[b]) if names[b] != 'nil' else (0., 0.) for b in st.b[con]])
    con_t = torch.tensor([SUPPORT[vnames[v]] for v in st.v[con]])
    con_r = torch.as_tensor(st.r[con])
    # intuition rows: individual T/F
    ii = np.flatnonzero(train_rows & (st.kind == 'individual') & (st.world_y >= 0))
    iy = torch.as_tensor(st.world_y[ii].astype(np.float32))
    iw = V2.rel_weights(st.r[ii], st.world_y[ii], len(st.relations))
    A = lambda idx: (torch.as_tensor(st.a[idx]), torch.as_tensor(st.r[idx]), torch.as_tensor(np.maximum(st.b[idx], 0)))
    sa, sr, sb = A(sup)
    ka, kr, kb = A(kin)
    fa, fr, fb = A(fil)
    ia, ir, ib = A(ii)
    log = []
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        li = (F.binary_cross_entropy_with_logits(model.intuition(ia, ir, ib), iy, reduction='none') * iw).sum()
        ls = (F.binary_cross_entropy(model.logic(sa, sr, sb), sup_t, reduction='none').sum(1) * w[sup]).sum()
        lk = (F.cross_entropy(model.code_logits(model.code_out(ka, kr, kb, torch.zeros(len(kin), dtype=torch.long)),
                                                model.E_L[qidx]), kin_t, reduction='none') * w[kin]).sum()
        lp = torch.log_softmax(model.code_logits(model.code_out(fa, fr, fb, torch.ones(len(fil), dtype=torch.long)),
                                                 model.E_L), 1)
        lf = (-torch.logsumexp(lp.masked_fill(~fmask, -1e9), 1) * w[fil]).sum()
        lc = (F.binary_cross_entropy(model.connective(con_p1, con_p2, con_r), con_t, reduction='none').sum(1) * w[con]).sum()
        loss = li + ls + lk + lf + lc
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == STEPS:
            log.append({'step': step, 'loss': float(loss.detach()), 'intuition': float(li.detach()),
                        'logic': float(ls.detach()), 'wall': time.perf_counter() - t0})
            print(f'[{arm} seed {seed}] step {step}/{STEPS} loss {float(loss.detach()):.4f} '
                  f'({time.perf_counter() - t0:.0f} s)', flush=True)
    return model, log


# ---------------------------------------------------------------- reads

@torch.no_grad()
def route_outputs(model, st, rows):
    idx = np.flatnonzero(rows)
    a, r, b = (torch.as_tensor(x[idx]) for x in (st.a, st.r, np.maximum(st.b, 0)))
    return model.intuition(a, r, b).numpy(), model.logic(a, r, b).numpy()


def reads(model, st, train_rows):
    out = {}
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    # each test: routes, rules, agreement
    pooled_err, pooled_conf, pooled_dis = [], [], []
    for tst in TF_TESTS:
        m = ind & (st.test == 'P1') & (st.sub == 'default') if tst == 'P1-default' else ind & (st.test == tst)
        u, s = route_outputs(model, st, m)
        y = st.world_y[m]
        rr = rules(u, s)
        conflict = rr['R1-intuition'] != rr['R2-logic']
        rec = {k: tf_bal(v, y) for k, v in rr.items()}
        rec['intuition_auroc'] = R1.auroc(u, y)
        rec['logic_auroc'] = R1.auroc(s[:, 0] - s[:, 1], y)
        rec['agreement'] = float(1 - conflict.mean())
        rec['logic_state_share'] = {v: float((rr['R2-logic'] == v).mean()) for v in ('T', 'F', 'U', 'D')}
        out[tst] = rec
        pooled_err += list(rr['R1-intuition'] != np.where(y == 1, 'T', 'F'))
        pooled_conf += list(conflict)
        sig = 1 / (1 + np.exp(-u))
        pooled_dis += list(np.abs(sig - s[:, 0]) + np.abs((1 - sig) - s[:, 1]))
    out['conflict-flags-intuition-errors'] = {'binary_auroc': R1.auroc(np.array(pooled_conf, float), pooled_err),
                                              'graded_auroc': R1.auroc(pooled_dis, pooled_err),
                                              'error_rate': float(np.mean(pooled_err)), 'conflict_rate': float(np.mean(pooled_conf))}
    # S3b and anon fit: expected values T/F/U
    vnames = np.array(T.VALUES)
    for tag, m in (('S3b', st.test == 'S3b'), ('fit-anon', (st.kind == 'anon') & (st.test == ''))):
        u, s = route_outputs(model, st, m)
        exp = vnames[st.v[m]]
        rr = rules(u, s)
        out[tag] = {k: {'acc': float((v == exp).mean()),
                        'by_quantity': {q: float((v[st.sub[m] == q] == exp[st.sub[m] == q]).mean()) for q in np.unique(st.sub[m])}}
                    for k, v in rr.items()}
        out[tag + '-conflict-rate-by-quantity'] = {q: float((rr['R1-intuition'] != rr['R2-logic'])[st.sub[m] == q].mean())
                                                   for q in np.unique(st.sub[m])}
    # unknown reads
    for tag, m in (('S4a', st.test == 'S4a'), ('S4b-P7', ind & (st.test == 'P7')),
                   ('control', ind & np.isin(st.test, TT.CONTROL_TESTS))):
        u, s = route_outputs(model, st, m)
        rr = rules(u, s)
        out[tag + '-u-rate'] = {k: float((v == 'U').mean()) for k, v in rr.items() if k != 'R1-intuition'}
        out[tag + '-conflict-rate'] = float((rr['R1-intuition'] != rr['R2-logic']).mean())
    # kind quantities (code head), fits
    with torch.no_grad():
        for tag, m in (('S2', (st.kind == 'kind') & (st.test == 'S2')), ('fit-kind', (st.kind == 'kind') & (st.test == ''))):
            idx = np.flatnonzero(m)
            a, r, b = (torch.as_tensor(x[idx]) for x in (st.a, st.r, st.b))
            lg = model.code_logits(model.code_out(a, r, b, torch.zeros(len(idx), dtype=torch.long)),
                                   model.E_L[[st.index(q) for q in QUANT]]).numpy()
            pred = np.array(QUANT)[lg.argmax(1)]
            out[tag] = float((pred == vnames[st.v[m]]).mean())
        con = np.flatnonzero(st.kind == 'connective')
        names = np.array(st.entities)
        p1 = torch.tensor([SUPPORT.get(names[a], (0., 0.)) for a in st.a[con]])
        p2 = torch.tensor([SUPPORT.get(names[b], (0., 0.)) for b in st.b[con]])
        res = logic_state(model.connective(p1, p2, torch.as_tensor(st.r[con])).numpy())
        out['fit-connectives'] = float((res == vnames[st.v[con]]).mean())
    u, s = route_outputs(model, st, train_rows & ind)
    rr = rules(u, s)
    out['fit-individual'] = {k: tf_bal(v, st.world_y[train_rows & ind]) for k, v in rr.items()}
    return out


def fast_map_dax(model, st, seed):
    """dax given only is-a(dax, bird) = T through both routes; code(s) kept among the known codes (prior)."""
    known = [i for i, x in enumerate(st.entities) if x not in T.VALUES + ('nil',)]
    tables = [model.E_I] if model.shared else [model.E_I, model.E_L_own]
    priors = [V2.gaussian_prior(E[known]) for E in tables]
    g = torch.Generator().manual_seed(seed * 1000 + 17)
    codes = [nn.Parameter(torch.randn(1, D, generator=g) * TT.INIT_STD) for _ in tables]
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(codes, lr=V2.FIT_LR)
    ri, bi = torch.tensor([st.relations.index('is-a')]), torch.tensor([st.index('bird')])
    cI = lambda: codes[0]
    cL = lambda: codes[0] if model.shared else codes[1]
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model.intuition(None, ri, bi, ca=cI()), torch.ones(1), reduction='sum')
        loss = loss + F.binary_cross_entropy(model.logic(None, ri, bi, ca=cL()), torch.tensor([[1., 0.]]), reduction='sum')
        for c, (mu, P) in zip(codes, priors):
            dl = c[0] - mu
            loss = loss + .5 * dl @ P @ dl
        loss.backward()
        opt.step()
    res = {}
    with torch.no_grad():
        for q in QUANT:
            items = [(rel, val, e) for rel, val, e, qq in T.default_expectations('bird') if qq == q]
            r = torch.tensor([st.relations.index(x[0]) for x in items])
            b = torch.tensor([st.index(x[1]) for x in items])
            u = model.intuition(None, r, b, ca=cI().expand(len(items), -1)).numpy()
            s = model.logic(None, r, b, ca=cL().expand(len(items), -1)).numpy()
            exp = np.array([x[2] for x in items])
            res[q] = {k: float((v == exp).mean()) for k, v in rules(u, s).items()}
            res[q]['conflict_rate'] = float((rules(u, s)['R1-intuition'] != rules(u, s)['R2-logic']).mean())
        others = [c for c in T.W3.CLASSES if c != 'bird']
        r = torch.tensor([st.relations.index('is-a')] * len(others))
        b = torch.tensor([st.index(c) for c in others])
        u = model.intuition(None, r, b, ca=cI().expand(len(others), -1)).numpy()
        s = model.logic(None, r, b, ca=cL().expand(len(others), -1)).numpy()
        res['other_classes_rejected'] = {k: float((v == 'F').mean()) for k, v in rules(u, s).items()}
    return res


# ---------------------------------------------------------------- tasks

def run(arm, seed):
    st, _ = T2.build()
    rng = np.random.default_rng(seed)
    trainable = st.test == ''
    ind_idx = np.flatnonzero(trainable & (st.kind == 'individual'))
    val = np.zeros(len(st.a), bool)
    val[rng.choice(ind_idx, int(round(TT.VAL_FRAC * len(ind_idx))), replace=False)] = True
    train_rows = trainable & ~val
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    t0 = time.process_time()
    model, log = train(st, arm, seed, g, train_rows)
    train_cpu = time.process_time() - t0
    res = {'tests': reads(model, st, train_rows), 'dax': fast_map_dax(model, st, seed), 'log': log,
           'arm': arm, 'seed': seed, 'params': R1.n_params(model)}
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'fit_steps': FIT_STEPS, 'params': R1.n_params(model),
                 'train_cpu_seconds': train_cpu, 'trained_statements': int(train_rows.sum()),
                 'checkpoint_bytes': ck.stat().st_size}


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
    workers = int(os.environ.get('EV_WORKERS', 6))
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
                     'world_sha256': nc.sha256(T2.__file__), 'workers': workers, 'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                     'tasks': {k: t for k, _, t in res}})
    V2.save_json(f'{REPORT}.json', {'status': 'dual-route truth v1' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                    'results': {k: r for k, r, _ in res}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    by = {}
    for r in res.values():
        by.setdefault(r['arm'], []).append(r)

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if vals and all(isinstance(v, dict) for v in vals):
            keys = set().union(*[v.keys() for v in vals])
            return {k: agg([v.get(k) for v in vals]) for k in sorted(keys)}
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(float(np.mean(vals)), 3) if vals else None
    out = {a: {'tests': agg([r['tests'] for r in rs]), 'dax': agg([r['dax'] for r in rs]), 'params': rs[0]['params']}
           for a, rs in by.items()}
    ms = ROOT / 'reports' / 'multistep-reader-v1-summary.json'
    if ms.exists():
        m = json.loads(ms.read_text())
        out['reference'] = {a: {t: m[a]['tests'].get(t, {}).get('bal_acc') for t in TF_TESTS} for a in ('unit-one', 'pred-one')}
    V2.save_json(f'{REPORT}-summary.json', out)
    for a in ARMS:
        if a not in out:
            continue
        print('==', a, 'params', out[a]['params'])
        print('test'.ljust(15) + ''.join(r[:12].rjust(13) for r in RULES) + '  agree  aurocI aurocL | unit  pred')
        for t in TF_TESTS:
            v = out[a]['tests'][t]
            ref = out.get('reference', {})
            print(t.ljust(15) + ''.join(f"{v[r]:.2f}".rjust(13) for r in RULES)
                  + f"  {v['agreement']:.2f}   {v['intuition_auroc']:.2f}   {v['logic_auroc']:.2f} | "
                  + ' '.join(f"{(ref.get(x, {}).get(t) or float('nan')):.2f}" for x in ('unit-one', 'pred-one')))
        for k in ('conflict-flags-intuition-errors', 'S3b', 'S3b-conflict-rate-by-quantity', 'S4a-u-rate', 'S4b-P7-u-rate',
                  'control-u-rate', 'S4a-conflict-rate', 'S2', 'fit-kind', 'fit-connectives', 'fit-individual'):
            print('  ', k, out[a]['tests'].get(k))
        print('   dax', out[a]['dax'])


if __name__ == '__main__':
    {'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
