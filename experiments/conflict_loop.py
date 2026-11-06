"""Conflict loop v1 (docs/conflict-loop-v1.md): separate-code intuition and logic routes; logic loops only when it
conflicts with intuition; half of the chain facts withheld from logic's direct training; a learned arbiter trained on
validation statements.

Usage: .venv/bin/python -m experiments.conflict_loop main      (rerun to resume; EV_WORKERS, default 6)
       .venv/bin/python -m experiments.conflict_loop summary
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
from experiments import dual_route_truth as DR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from experiments import multistep_reader as MS
from experiments import taught_truth as TT
from src import everyday_truth as T
from src import everyday_truth_v2 as T2

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'conflict-loop-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'conflict-loop-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 60 if SMOKE else 3000
ARB_STEPS = 50 if SMOKE else 500
FIT_STEPS = 50 if SMOKE else 500
LOG_EVERY = 20 if SMOKE else 250
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('arbiter', 'conflict-loop', 'conflict-loop-told')
TAU, T_MAX, H = 0.5, 3, 64
PRIOR_P, KL_W = 0.5, 0.01
CHAIN_RELS = ('grandparent-of', 'ancestor-of', 'kind-of', 'bigger-than', 'smaller-than', 'named-can', 'named-has',
              'named-kind', 'same-colour-as')
D, RD = DR.D, DR.RD
DR.STEPS = STEPS


class Model(nn.Module):
    """Dual-route (separate codes) plus a logic loop controller and an arbiter."""

    def __init__(self, n_ent, n_rel, g):
        super().__init__()
        self.dual = DR.Dual('dual-separate', n_ent, n_rel, g)
        self.init = nn.Linear(2 * D + RD, H)
        self.gru = nn.GRUCell(D + RD, H)
        self.w_rel, self.w_ans = nn.Linear(H, n_rel), nn.Linear(H, n_rel)
        self.w_halt = nn.Linear(H + 1, 1)
        self.arb = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, 3))
        R1._reinit(self, g)

    def logic_codes(self, ca, rc, cb):
        d = self.dual
        return torch.sigmoid(d.S_L(torch.cat([d.G_L(torch.cat([ca, cb, rc], 1)), rc], 1)))

    def produce(self, m, rho):
        d = self.dual
        gap = torch.ones(len(m), dtype=torch.long)
        out = d.P_L(torch.cat([d.G_L(torch.cat([m, d.mask.expand(len(m), -1), rho], 1)), rho, d.slot[gap]], 1))
        return F.normalize(out, dim=1) * d.E_L.detach().norm(dim=1).mean()

    def loop(self, a, r, b, sig_u, ca=None):
        """Returns supports per step (list, t = 0..T), halting distribution (n, T+1), produced codes."""
        d = self.dual
        ca = d.E_L[a] if ca is None else ca
        cb, rc = d.E_L[b], d.R_L[r]
        s0 = self.logic_codes(ca, rc, cb)
        state = torch.tanh(self.init(torch.cat([ca, cb, rc], 1)))
        m, sup, lams, ms = ca, [s0], [], []
        for t in range(1, T_MAX + 1):
            rho = torch.softmax(self.w_rel(state), 1) @ d.R_L
            m = self.produce(m, rho)
            ms.append(m)
            rho_a = torch.softmax(self.w_ans(state), 1) @ d.R_L
            s = self.logic_codes(m, rho_a, cb)
            sup.append(s)
            dis = (torch.abs(sig_u - s[:, 0]) + torch.abs((1 - sig_u) - s[:, 1]))[:, None]
            state = self.gru(torch.cat([m, rho], 1), state)
            if t < T_MAX:
                lams.append(torch.sigmoid(self.w_halt(torch.cat([state, dis], 1)))[:, 0])
        return sup, lams, ms


def halting(lams, n, start):
    """PonderNet halting over t = start..T_MAX given λ for t = 1..T_MAX-1 (step 0 halts with λ_0 = 0 when start = 1)."""
    ph, rem = [], torch.ones(n)
    steps = list(range(start, T_MAX + 1))
    for t in steps:
        if t == T_MAX:
            ph.append(rem)
        else:
            lam = lams[t - 1] if t >= 1 else torch.full((n,), .5)
            ph.append(rem * lam)
            rem = rem * (1 - lam)
    return torch.stack(ph, 1), steps


def mix(sup, ph, steps):
    return sum(ph[:, k:k + 1] * sup[t] for k, t in enumerate(steps)).clamp(1e-6, 1 - 1e-6)


def kl(ph):
    k = ph.shape[1]
    pr = torch.tensor([PRIOR_P * (1 - PRIOR_P) ** t for t in range(k)])
    pr = pr / pr.sum()
    return (ph * (torch.log(ph + 1e-8) - torch.log(pr))).sum(1)


def disagreement(sig_u, s):
    return torch.abs(sig_u - s[:, 0]) + torch.abs((1 - sig_u) - s[:, 1])


# ---------------------------------------------------------------- training

def train(st, arm, seed, g, train_rows):
    model = Model(len(st.entities), len(st.relations), g)
    d = model.dual
    route_params = [p for n, p in model.named_parameters() if not n.startswith('arb.')]
    opt = torch.optim.Adam(route_params, lr=TT.LR, weight_decay=TT.WD)
    w = torch.as_tensor(TT.weights(st, train_rows))
    vnames = np.array(T.VALUES)
    rels = np.array(st.relations)
    chain = train_rows & (st.kind == 'individual') & np.isin(rels[st.r], CHAIN_RELS)
    withheld = np.zeros(len(st.a), bool)
    if arm == 'conflict-loop':
        rng = np.random.default_rng([seed, 99])
        ci = np.flatnonzero(chain)
        withheld[rng.choice(ci, len(ci) // 2, replace=False)] = True
    sup = np.flatnonzero(train_rows & np.isin(st.kind, ('individual', 'anon', 'anon-isa')))
    sup_t = torch.tensor([DR.SUPPORT[vnames[v]] for v in st.v[sup]])
    wh = torch.as_tensor(withheld[sup])
    kin = np.flatnonzero(train_rows & (st.kind == 'kind'))
    qidx = torch.tensor([st.index(q) for q in DR.QUANT])
    kin_t = torch.tensor([DR.QUANT.index(vnames[v]) for v in st.v[kin]])
    fil = np.flatnonzero(train_rows & (st.kind == 'fill'))
    fmask = torch.zeros(len(fil), len(st.entities), dtype=torch.bool)
    for k, i in enumerate(fil):
        fmask[k, st.fill_answers[int(i)]] = True
    con = np.flatnonzero(st.kind == 'connective')
    names = np.array(st.entities)
    con_p1 = torch.tensor([DR.SUPPORT.get(names[a], (0., 0.)) for a in st.a[con]])
    con_p2 = torch.tensor([DR.SUPPORT.get(names[b], (0., 0.)) for b in st.b[con]])
    con_t = torch.tensor([DR.SUPPORT[vnames[v]] for v in st.v[con]])
    con_r = torch.as_tensor(st.r[con])
    ii = np.flatnonzero(train_rows & (st.kind == 'individual') & (st.world_y >= 0))
    iy = torch.as_tensor(st.world_y[ii].astype(np.float32))
    iw = V2.rel_weights(st.r[ii], st.world_y[ii], len(st.relations))
    A = lambda idx: (torch.as_tensor(st.a[idx]), torch.as_tensor(st.r[idx]), torch.as_tensor(np.maximum(st.b[idx], 0)))
    sa, sr, sb = A(sup)
    ka, kr, kb = A(kin)
    fa, fr, fb = A(fil)
    ia, ir, ib = A(ii)
    use_loop = arm != 'arbiter'
    log, t0 = [], time.perf_counter()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        li = (F.binary_cross_entropy_with_logits(d.intuition(ia, ir, ib), iy, reduction='none') * iw).sum()
        s0 = d.logic(sa, sr, sb)
        with torch.no_grad():
            sig = torch.sigmoid(d.intuition(sa, sr, sb))
            dis0 = disagreement(sig, s0)
        trig = (dis0 > TAU) if use_loop else torch.zeros(len(sup), dtype=torch.bool)
        bce0 = F.binary_cross_entropy(s0, sup_t, reduction='none').sum(1)
        direct = ~wh & ~trig
        ls = (bce0 * w[sup] * direct).sum()
        n_trig = int(trig.sum())
        lk_loop = torch.zeros(())
        if n_trig:
            ti = torch.nonzero(trig)[:, 0]
            sp, lams, _ = model.loop(sa[ti], sr[ti], sb[ti], sig[ti])
            whi = wh[ti]
            ph_all, st_all = halting(lams, len(ti), 0)
            ph_w, st_w = halting(lams, len(ti), 1)
            s_told = mix(sp, ph_all, st_all)
            s_with = mix(sp, ph_w, st_w)
            s_fin = torch.where(whi[:, None], s_with, s_told)
            bl = F.binary_cross_entropy(s_fin, sup_t[ti], reduction='none').sum(1)
            klv = torch.where(whi, kl(ph_w), kl(ph_all[:, :]))
            lk_loop = (bl * w[sup][ti]).sum() + KL_W * klv.mean()
        lk = (F.cross_entropy(d.code_logits(d.code_out(ka, kr, kb, torch.zeros(len(kin), dtype=torch.long)), d.E_L[qidx]),
                              kin_t, reduction='none') * w[kin]).sum()
        lp = torch.log_softmax(d.code_logits(d.code_out(fa, fr, fb, torch.ones(len(fil), dtype=torch.long)), d.E_L), 1)
        lf = (-torch.logsumexp(lp.masked_fill(~fmask, -1e9), 1) * w[fil]).sum()
        lc = (F.binary_cross_entropy(d.connective(con_p1, con_p2, con_r), con_t, reduction='none').sum(1) * w[con]).sum()
        loss = li + ls + lk_loop + lk + lf + lc
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == STEPS:
            rec = {'step': step, 'loss': float(loss.detach()), 'triggered': n_trig,
                   'triggered_withheld': int((trig & wh).sum()), 'withheld': int(wh.sum()), 'wall': time.perf_counter() - t0}
            log.append(rec)
            print(f'[{arm} seed {seed}] step {step}/{STEPS} loss {rec["loss"]:.4f} triggered {n_trig} '
                  f'(withheld {rec["triggered_withheld"]}/{rec["withheld"]}) ({rec["wall"]:.0f} s)', flush=True)
    return model, log, withheld


# ---------------------------------------------------------------- answering

@torch.no_grad()
def answer(model, st, rows, use_loop, ca_I=None, ca_L=None, items=None):
    """Returns dict with u, s0, s_final, d0, d_final, steps, triggered, and the first produced codes (for inspection)."""
    d = model.dual
    if items is None:
        idx = np.flatnonzero(rows)
        a, r, b = (torch.as_tensor(x[idx]) for x in (st.a, st.r, np.maximum(st.b, 0)))
    else:
        a, r, b = items
    u = d.intuition(a, r, b, ca=ca_I)
    sig = torch.sigmoid(u)
    s0 = d.logic(a, r, b, ca=ca_L)
    d0 = disagreement(sig, s0)
    n = len(u)
    s_fin, steps, trig = s0.clone(), torch.zeros(n), torch.zeros(n, dtype=torch.bool)
    first = None
    if use_loop:
        trig = d0 > TAU
        if trig.any():
            ti = torch.nonzero(trig)[:, 0]
            caL = (d.E_L[a] if ca_L is None else ca_L)[ti]
            sp, lams, ms = model.loop(None, r[ti], b[ti], sig[ti], ca=caL)
            ph, stp = halting(lams, len(ti), 0)
            s_fin[ti] = mix(sp, ph, stp)
            steps[ti] = (ph * torch.tensor(stp, dtype=torch.float)).sum(1)
            first = (ti.numpy(), ms[0])
    dfin = disagreement(sig, s_fin)
    return {'u': u.numpy(), 's0': s0.numpy(), 's': s_fin.numpy(), 'd0': d0.numpy(), 'd': dfin.numpy(),
            'steps': steps.numpy(), 'trig': trig.numpy(), 'first': first}


def arb_features(o):
    sig = 1 / (1 + np.exp(-o['u']))
    return torch.tensor(np.stack([sig, o['s'][:, 0], o['s'][:, 1], o['d0'], o['d'], o['steps'], o['trig'].astype(float)], 1),
                        dtype=torch.float32)


def train_arbiter(model, st, val_rows, use_loop, seed):
    o = answer(model, st, val_rows, use_loop)
    X = arb_features(o)
    vn = np.array(T.VALUES)[st.v[val_rows]]
    y = torch.tensor([('T', 'F', 'U').index(v) for v in vn])
    counts = torch.bincount(y, minlength=3).float().clamp(min=1)
    wcls = (1 / counts) / (1 / counts).sum() * 3
    torch.manual_seed(seed + 5)
    opt = torch.optim.Adam(model.arb.parameters(), lr=1e-2)
    for _ in range(ARB_STEPS):
        opt.zero_grad()
        F.cross_entropy(model.arb(X), y, weight=wcls).backward()
        opt.step()
    return {'val_n': int(len(y)), 'val_counts': {k: int(c) for k, c in zip('TFU', torch.bincount(y, minlength=3))}}


@torch.no_grad()
def arb_answer(model, o):
    return np.array(['T', 'F', 'U'])[model.arb(arb_features(o)).argmax(1).numpy()]


# ---------------------------------------------------------------- reads

def outputs(model, o):
    iv = np.where(o['u'] > 0, 'T', 'F')
    return {'intuition': iv, 'logic-step0': DR.logic_state(o['s0']), 'logic-final': DR.logic_state(o['s']),
            'arbiter': arb_answer(model, o)}


def true_middles(st, i):
    """True middle entities for inspection, or None."""
    names = np.array(st.entities)
    rel = st.relations[st.r[i]]
    a, b = names[st.a[i]], names[st.b[i]]
    if rel.startswith('named-'):
        return {a[:-2]}
    if rel == 'grandparent-of':
        par = T2.parents()
        return {c for c in par if a in par[c] and c in par.get(b, [])}
    if st.test[i] == 'S3b':
        return {a.rsplit('-', 1)[0]}
    return None


def reads(model, st, use_loop, withheld, train_rows):
    out = {}
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    names = np.array(st.entities)
    E = F.normalize(model.dual.E_L.detach(), dim=1)
    pe, pc, pdis = [], [], []
    for tst in DR.TF_TESTS:
        m = ind & (st.test == 'P1') & (st.sub == 'default') if tst == 'P1-default' else ind & (st.test == tst)
        o = answer(model, st, m, use_loop)
        y = st.world_y[m]
        outs = outputs(model, o)
        rec = {k: DR.tf_bal(v, y) for k, v in outs.items()}
        rec['intuition_auroc'] = R1.auroc(o['u'], y)
        rec['logic_final_auroc'] = R1.auroc(o['s'][:, 0] - o['s'][:, 1], y)
        rec['logic_step0_auroc'] = R1.auroc(o['s0'][:, 0] - o['s0'][:, 1], y)
        rec['trigger_rate'] = float(o['trig'].mean())
        rec['expected_steps_triggered'] = float(o['steps'][o['trig']].mean()) if o['trig'].any() else None
        rec['arbiter_u_rate'] = float((outs['arbiter'] == 'U').mean())
        if o['first'] is not None and tst in ('L2-named', 'P6-grandparent'):
            ti, m1 = o['first']
            idx = np.flatnonzero(m)[ti]
            near = names[(F.normalize(m1, dim=1) @ E.T).argmax(1).numpy()]
            hits = [near[k] in (true_middles(st, i) or set()) for k, i in enumerate(idx)]
            rec['first_produced_is_middle'] = float(np.mean(hits)) if hits else None
        if tst.startswith('L1'):
            L = st.sub[m]
            rec['steps_by_length'] = {s: float(o['steps'][(L == s) & o['trig']].mean()) for s in np.unique(L)
                                      if ((L == s) & o['trig']).any()}
            rec['logic_final_by_length'] = {s: DR.tf_bal(outs['logic-final'][L == s], y[L == s]) for s in np.unique(L)}
            rec['arbiter_by_length'] = {s: DR.tf_bal(outs['arbiter'][L == s], y[L == s]) for s in np.unique(L)}
        out[tst] = rec
        pe += list(outs['intuition'] != np.where(y == 1, 'T', 'F'))
        pc += list(o['d0'])
    out['conflict-flags-intuition-errors'] = {'graded_auroc': R1.auroc(pc, pe), 'error_rate': float(np.mean(pe))}
    # withheld facts (training rows logic never saw directly)
    if withheld.any():
        wm = withheld & ind
        o = answer(model, st, wm, use_loop)
        outs = outputs(model, o)
        y = st.world_y[wm]
        out['withheld-fit'] = {k: DR.tf_bal(v, y) for k, v in outs.items()}
        out['withheld-fit']['trigger_rate'] = float(o['trig'].mean())
        out['withheld-fit']['trigger_rate_true'] = float(o['trig'][y == 1].mean())
    # told chain facts (training)
    tm = train_rows & ind & ~withheld & np.isin(np.array(st.relations)[st.r], CHAIN_RELS)
    o = answer(model, st, tm, use_loop)
    out['told-chain-fit'] = {k: DR.tf_bal(v, st.world_y[tm]) for k, v in outputs(model, o).items()}
    out['told-chain-fit']['trigger_rate'] = float(o['trig'].mean())
    # unknown and S3b
    vn = np.array(T.VALUES)
    for tag, m in (('S4a', st.test == 'S4a'), ('S4b-P7', ind & (st.test == 'P7')),
                   ('control', ind & np.isin(st.test, TT.CONTROL_TESTS))):
        o = answer(model, st, m, use_loop)
        outs = outputs(model, o)
        out[tag + '-u-rate'] = {k: float((v == 'U').mean()) for k, v in outs.items() if k != 'intuition'}
        out[tag + '-trigger-rate'] = float(o['trig'].mean())
    m = st.test == 'S3b'
    o = answer(model, st, m, use_loop)
    outs = outputs(model, o)
    exp = vn[st.v[m]]
    out['S3b'] = {k: {q: float((v[st.sub[m] == q] == exp[st.sub[m] == q]).mean()) for q in np.unique(st.sub[m])}
                  for k, v in outs.items()}
    if o['first'] is not None:
        ti, m1 = o['first']
        idx = np.flatnonzero(m)[ti]
        near = names[(F.normalize(m1, dim=1) @ E.T).argmax(1).numpy()]
        out['S3b-first-produced-is-class'] = float(np.mean([near[k] in true_middles(st, i) for k, i in enumerate(idx)]))
    out['S3b-trigger-rate'] = float(o['trig'].mean())
    return out


def fast_map_dax(model, st, seed, use_loop):
    d = model.dual
    known = [i for i, x in enumerate(st.entities) if x not in T.VALUES + ('nil',)]
    priors = [V2.gaussian_prior(E[known]) for E in (d.E_I, d.E_L)]
    g = torch.Generator().manual_seed(seed * 1000 + 17)
    codes = [nn.Parameter(torch.randn(1, D, generator=g) * TT.INIT_STD) for _ in range(2)]
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(codes, lr=V2.FIT_LR)
    ri, bi = torch.tensor([st.relations.index('is-a')]), torch.tensor([st.index('bird')])
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(d.intuition(None, ri, bi, ca=codes[0]), torch.ones(1), reduction='sum')
        loss = loss + F.binary_cross_entropy(d.logic(None, ri, bi, ca=codes[1]), torch.tensor([[1., 0.]]), reduction='sum')
        for c, (mu, P) in zip(codes, priors):
            dl = c[0] - mu
            loss = loss + .5 * dl @ P @ dl
        loss.backward()
        opt.step()
    res = {}
    for q in DR.QUANT:
        items = [(rel, val, e) for rel, val, e, qq in T.default_expectations('bird') if qq == q]
        r = torch.tensor([st.relations.index(x[0]) for x in items])
        b = torch.tensor([st.index(x[1]) for x in items])
        n = len(items)
        o = answer(model, st, None, use_loop, ca_I=codes[0].detach().expand(n, -1), ca_L=codes[1].detach().expand(n, -1),
                   items=(None, r, b))
        exp = np.array([x[2] for x in items])
        res[q] = {k: float((v == exp).mean()) for k, v in outputs(model, o).items()}
        res[q]['trigger_rate'] = float(o['trig'].mean())
    return res


# ---------------------------------------------------------------- tasks

def run(arm, seed):
    st, _ = T2.build()
    TT.value_idx_cache = TT.value_idx(st)
    rng = np.random.default_rng(seed)
    trainable = st.test == ''
    ind_idx = np.flatnonzero(trainable & (st.kind == 'individual'))
    val = np.zeros(len(st.a), bool)
    val[rng.choice(ind_idx, int(round(TT.VAL_FRAC * len(ind_idx))), replace=False)] = True
    train_rows = trainable & ~val
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    t0 = time.process_time()
    model, log, withheld = train(st, arm, seed, g, train_rows)
    use_loop = arm != 'arbiter'
    arb_info = train_arbiter(model, st, val & (st.kind == 'individual'), use_loop, seed)
    train_cpu = time.process_time() - t0
    res = {'tests': reads(model, st, use_loop, withheld, train_rows), 'dax': fast_map_dax(model, st, seed, use_loop),
           'log': log, 'arbiter_training': arb_info, 'arm': arm, 'seed': seed, 'params': R1.n_params(model)}
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'arbiter_steps': ARB_STEPS, 'params': R1.n_params(model),
                 'train_cpu_seconds': train_cpu, 'checkpoint_bytes': ck.stat().st_size}


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
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'workers': workers,
                     'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                     'tasks': {k: t for k, _, t in res}})
    V2.save_json(f'{REPORT}.json', {'status': 'conflict loop v1' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
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
    out = {a: {'tests': agg([r['tests'] for r in rs]), 'dax': agg([r['dax'] for r in rs]), 'params': rs[0]['params'],
               'final_triggered': agg([r['log'][-1] for r in rs])} for a, rs in by.items()}
    V2.save_json(f'{REPORT}-summary.json', out)
    for a in ARMS:
        if a not in out:
            continue
        o = out[a]
        print('==', a, 'params', o['params'], 'last log', o['final_triggered'])
        print('test'.ljust(15) + 'intuition  step0  final  arbiter | trig  steps  arbU | aurocI aurocL0 aurocL')
        for t in DR.TF_TESTS:
            v = o['tests'][t]
            f = lambda k: '  -  ' if v.get(k) is None else f'{v[k]:.2f}'
            print(t.ljust(15) + f"{f('intuition')}       {f('logic-step0')}   {f('logic-final')}   {f('arbiter')}   | "
                  f"{f('trigger_rate')}  {f('expected_steps_triggered')}  {f('arbiter_u_rate')} | "
                  f"{f('intuition_auroc')}  {f('logic_step0_auroc')}  {f('logic_final_auroc')}")
        for k in ('withheld-fit', 'told-chain-fit', 'conflict-flags-intuition-errors', 'S4a-u-rate', 'S4b-P7-u-rate',
                  'control-u-rate', 'S3b', 'S3b-first-produced-is-class', 'S3b-trigger-rate'):
            print('  ', k, o['tests'].get(k))
        for t in ('L1-kind', 'L1-anc', 'L1-size'):
            print('  ', t, {k: o['tests'][t].get(k) for k in ('steps_by_length', 'logic_final_by_length', 'arbiter_by_length')})
        for t in ('L2-named', 'P6-grandparent'):
            print('  ', t, 'first produced is middle', o['tests'][t].get('first_produced_is_middle'))
        print('   dax', o['dax'])


if __name__ == '__main__':
    {'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
