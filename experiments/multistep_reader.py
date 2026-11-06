"""Multi-step reader v1 (docs/multistep-reader-v1.md): a generative loop (Option 2: produce the middle thing's code) on the
taught-truth reader, against the same loop on the built-in truth unit, and both one-pass readers.

Usage: .venv/bin/python -m experiments.multistep_reader main      (rerun to resume; EV_WORKERS, default 8)
       .venv/bin/python -m experiments.multistep_reader summary
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
NAME = 'multistep-reader-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'multistep-reader-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 60 if SMOKE else int(os.environ.get('EV_STEPS', 3000))
FIT_STEPS = 50 if SMOKE else 500
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('pred-one', 'pred-loop', 'unit-one', 'unit-loop')
D, RD, H = TT.D, TT.RD, 64
T_TRAIN, T_READ_LONG = 4, 6
PRIOR_P, KL_W = 0.5, 0.01
TF_TESTS = TT.S1_TESTS + ('L1-kind', 'L1-anc', 'L1-size', 'L2-named', 'L2-x', 'P6-anc')
TT.STEPS, TT.FIT_STEPS, TT.SEEDS = STEPS, FIT_STEPS, SEEDS


# ---------------------------------------------------------------- the loop

class Loop(nn.Module):
    """Weight-tied controller around a reader. kind 'pred': the taught-truth reader produces and answers. kind 'unit':
    the built-in truth unit answers and a separate generator P' produces."""

    def __init__(self, kind, n_ent, n_rel, g):
        super().__init__()
        self.kind = kind
        if kind == 'pred':
            self.reader = TT.Pred('pred', n_ent, n_rel, g)
            self.E, self.R = self.reader.E, self.reader.R
        else:
            self.judge = V2.Net('tworel', n_ent, n_rel, g)
            self.E, self.R = self.judge.E, self.judge.R
            self.mask = nn.Parameter(torch.randn(D, generator=g) * TT.INIT_STD)
            self.prod = nn.Sequential(nn.Linear(D + RD, TT.HID), nn.ReLU(), nn.Linear(TT.HID, D))
            self.prod_log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.init = nn.Linear(2 * D + RD + TT.SLOT, H)
        self.gru = nn.GRUCell(D + RD, H)
        self.w_rel = nn.Linear(H, n_rel)
        self.w_ans = nn.Linear(H, n_rel)
        self.w_halt = nn.Linear(H, 1)
        R1._reinit(self, g)

    def slot(self, gap):
        return self.reader.slot[gap] if self.kind == 'pred' else torch.zeros(len(gap), TT.SLOT)

    def mask_code(self):
        return self.reader.mask if self.kind == 'pred' else self.mask

    def produce(self, m, rho):
        if self.kind == 'pred':
            out = self.reader.out_codes(m, rho, m, torch.ones(len(m), dtype=torch.long))
        else:
            out = self.prod(torch.cat([m, rho], 1))
        return F.normalize(out, dim=1) * self.E.detach().norm(dim=1).mean()

    def answer(self, m, rho, cb, gap):
        if self.kind == 'pred':
            out = self.reader.out_codes(m, rho, cb, gap)
            cand = self.E[TT.value_idx_cache] if gap[0] == 0 else self.E
            return self.reader.logits(out, cand)
        return self.judge.score_codes(m, rho, cb)

    def forward(self, ca, r, cb, gap, T_steps):
        """Returns (answers per step, halting distribution (n, T+1), produced codes per step)."""
        cbm = torch.where(gap[:, None] == 1, self.mask_code().expand_as(cb), cb)
        s = torch.tanh(self.init(torch.cat([ca, cbm, self.R[r], self.slot(gap)], 1)))
        m = ca
        ys, lams, ms = [], [], [m]
        for t in range(T_steps + 1):
            rho_a = torch.softmax(self.w_ans(s), 1) @ self.R
            ys.append(self.answer(m, rho_a, cbm, gap))
            if t == T_steps:
                break
            lams.append(torch.sigmoid(self.w_halt(s))[:, 0])
            rho = torch.softmax(self.w_rel(s), 1) @ self.R
            m = self.produce(m, rho)
            ms.append(m)
            s = self.gru(torch.cat([m, rho], 1), s)
        ph, rem = [], torch.ones(len(ca))
        for lam in lams:
            ph.append(rem * lam)
            rem = rem * (1 - lam)
        ph.append(rem)
        return ys, torch.stack(ph, 1), ms


def geometric_prior(T_steps):
    p = torch.tensor([PRIOR_P * (1 - PRIOR_P) ** t for t in range(T_steps + 1)])
    return p / p.sum()


def kl_halt(ph, T_steps):
    pr = geometric_prior(T_steps)
    return (ph * (torch.log(ph + 1e-8) - torch.log(pr))).sum(1)


# ---------------------------------------------------------------- training

def rows_tensors(model, st, idx):
    a, r = torch.as_tensor(st.a[idx]), torch.as_tensor(st.r[idx])
    b = torch.as_tensor(np.maximum(st.b[idx], 0))
    return model.E[a], r, model.E[b], torch.as_tensor(st.gap[idx])


def mixture_logp(ys, ph):
    """log of Σ_t p(halt t) softmax(y_t): (n, C)."""
    lp = torch.stack([torch.log_softmax(y, 1) for y in ys], 1)          # (n, T+1, C)
    return torch.logsumexp(lp + torch.log(ph + 1e-8)[:, :, None], 1)


def mixture_prob_unit(ys, ph):
    return (torch.stack([torch.sigmoid(y) for y in ys], 1) * ph).sum(1).clamp(1e-6, 1 - 1e-6)


def train_loop(st, kind, seed, gen, train_rows):
    model = Loop(kind, len(st.entities), len(st.relations), gen)
    opt = torch.optim.Adam(model.parameters(), lr=TT.LR, weight_decay=TT.WD)
    frows = np.flatnonzero(train_rows & (st.gap == 1))
    fmask = torch.zeros(len(frows), len(st.entities), dtype=torch.bool)
    for k, i in enumerate(frows):
        fmask[k, st.fill_answers[int(i)]] = True
    if kind == 'pred':
        w = torch.as_tensor(TT.weights(st, train_rows))
        vrows = np.flatnonzero(train_rows & (st.gap == 0))
        vv = torch.as_tensor(st.v[vrows])
    else:
        vrows = np.flatnonzero(train_rows & (st.kind == 'individual') & (st.world_y >= 0))
        vy = torch.as_tensor(st.world_y[vrows].astype(np.float32))
        wv_unit = V2.rel_weights(st.r[vrows], st.world_y[vrows], len(st.relations))
    for step in range(STEPS):
        opt.zero_grad()
        ca, r, cb, gap = rows_tensors(model, st, vrows)
        ys, ph, _ = model(ca, r, cb, gap, T_TRAIN)
        if kind == 'pred':
            lp = mixture_logp(ys, ph)
            loss = (-lp.gather(1, vv[:, None])[:, 0] * w[vrows]).sum() + KL_W * kl_halt(ph, T_TRAIN).mean()
            fa, fr, fb, fg = rows_tensors(model, st, frows)
            ysf, phf, _ = model(fa, fr, fb, fg, T_TRAIN)
            lpf = mixture_logp(ysf, phf)
            loss = loss + (-torch.logsumexp(lpf.masked_fill(~fmask, -1e9), 1) * w[frows]).sum() \
                + KL_W * kl_halt(phf, T_TRAIN).mean()
        else:
            p = mixture_prob_unit(ys, ph)
            loss = (F.binary_cross_entropy(p, vy, reduction='none') * wv_unit).sum() + KL_W * kl_halt(ph, T_TRAIN).mean()
            # the separate producer P' learns to fill tails directly (one pass)
            fa, fr = model.E[st.a[frows]], model.R[st.r[frows]]
            out = model.prod(torch.cat([fa, fr], 1))
            lg = model.prod_log_scale.exp() * F.normalize(out, dim=1) @ F.normalize(model.E, dim=1).T
            lpf = torch.log_softmax(lg, 1)
            loss = loss + (-torch.logsumexp(lpf.masked_fill(~fmask, -1e9), 1)).mean() / 6
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------- reads

@torch.no_grad()
def loop_value_logp(model, st, rows, T_steps=T_TRAIN, ca=None):
    idx = np.flatnonzero(rows)
    a, r, b, gap = rows_tensors(model, st, idx)
    if ca is not None:
        a = ca
    ys, ph, ms = model(a, r, b, gap, T_steps)
    return ys, ph, ms


def tf_read(model, st, rows, T_steps=T_TRAIN):
    ys, ph, _ = loop_value_logp(model, st, rows, T_steps)
    y = st.world_y[rows]
    if model.kind == 'pred':
        return TT.tf_metrics(mixture_logp(ys, ph).numpy(), y)
    p = mixture_prob_unit(ys, ph).numpy()
    return R1.metrics(np.log(p) - np.log1p(-p), y)


def by_sub(fn, st, base, subs):
    return {s: fn(base & (st.sub == s)) for s in subs if (base & (st.sub == s)).any()}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and np.std(rx) > 0 and np.std(ry) > 0 else None


def inspect(model, st):
    """Nearest entity of the first produced code against the true middle thing; halting against chain length; forced
    stop accuracy."""
    out = {}
    names = np.array(st.entities)
    E = F.normalize(model.E.detach(), dim=1)
    # S3b: anonymous individual -> its class; L2-named: word -> the named animal
    for tag, rows, true_mid in (
            ('S3b', st.test == 'S3b', lambda i: names[st.a[i]].rsplit('-', 1)[0]),
            ('L2-named', (st.test == 'L2-named') & (st.world_y >= 0), lambda i: names[st.a[i]][:-2])):
        idx = np.flatnonzero(rows)
        _, ph, ms = loop_value_logp(model, st, rows)
        nearest = names[(F.normalize(ms[1], dim=1) @ E.T).argmax(1).numpy()]
        truth = np.array([true_mid(i) for i in idx])
        out[f'{tag}-first-produced-is-middle'] = float((nearest == truth).mean())
        out[f'{tag}-first-produced-top'] = dict(zip(*np.unique(nearest, return_counts=True)))
        out[f'{tag}-first-produced-top'] = {k: int(v) for k, v in sorted(out[f'{tag}-first-produced-top'].items(),
                                                                          key=lambda x: -x[1])[:5]}
        out[f'{tag}-expected-steps'] = float((ph.numpy() * np.arange(ph.shape[1])).sum(1).mean())
    L = T2.chain_length(st)
    rows = L >= 0
    _, ph, _ = loop_value_logp(model, st, rows)
    exp_steps = (ph.numpy() * np.arange(ph.shape[1])).sum(1)
    out['halting-vs-length-spearman'] = spearman(exp_steps, L[rows])
    out['expected-steps-by-length'] = {int(l): float(exp_steps[L[rows] == l].mean()) for l in np.unique(L[rows])}
    # forced stop: answer from step t alone
    for tag in ('L1-kind', 'L1-anc', 'L1-size', 'L2-named', 'P6-grandparent', 'S3b'):
        rows = (st.test == tag) & ((st.world_y >= 0) | (tag == 'S3b'))
        ys, _, _ = loop_value_logp(model, st, rows)
        res = {}
        for t, y in enumerate(ys):
            if tag == 'S3b':
                if model.kind != 'pred':
                    continue
                res[t] = TT.value_accuracy(y.numpy(), np.array(T.VALUES)[st.v[rows]])['acc']
            elif model.kind == 'pred':
                res[t] = TT.tf_metrics(y.numpy(), st.world_y[rows])['bal_acc']
            else:
                res[t] = R1.metrics(y.numpy(), st.world_y[rows])['bal_acc']
        out[f'forced-stop-{tag}'] = res
    return out


def reads_loop(model, st, train_rows, val_rows):
    out = {}
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    for tst in TF_TESTS:
        if tst in ('P1-default', 'P1-specific'):
            m = ind & (st.test == 'P1') & (st.sub == tst.split('-')[1])
        else:
            m = ind & (st.test == tst)
        if m.any():
            out[tst] = tf_read(model, st, m)
    for tst, subs in (('L1-kind', ('3', '4')), ('L1-anc', ('3', '4', '5')), ('L1-size', tuple(str(d) for d in range(3, 12))),
                      ('L2-named', ('named-can', 'named-has', 'named-kind'))):
        base = ind & (st.test == tst)
        out[tst + '-by'] = by_sub(lambda m: tf_read(model, st, m)['bal_acc'], st, base, subs)
        out[tst + '-by-T6'] = by_sub(lambda m: tf_read(model, st, m, T_READ_LONG)['bal_acc'], st, base, subs)
    out['fit-individual'] = tf_read(model, st, train_rows & ind)
    out['val-individual'] = tf_read(model, st, val_rows & ind)
    if model.kind == 'pred':
        for tag, rows in (('S2', (st.kind == 'kind') & (st.test == 'S2')), ('S3b', st.test == 'S3b'),
                          ('fit-anon', (st.kind == 'anon') & (st.test == ''))):
            ys, ph, _ = loop_value_logp(model, st, rows)
            lg = mixture_logp(ys, ph).numpy()
            exp = st.sub[rows] if tag == 'S2' else np.array(T.VALUES)[st.v[rows]]
            out[tag] = TT.value_accuracy(lg, exp)
            if tag != 'S2':
                out[tag + '-by-quantity'] = {q: TT.value_accuracy(lg[st.sub[rows] == q], exp[st.sub[rows] == q])['acc']
                                             for q in np.unique(st.sub[rows])}
        for tag, rows in (('S4a', st.test == 'S4a'), ('S4b-P7', ind & (st.test == 'P7'))):
            ys, ph, _ = loop_value_logp(model, st, rows)
            out[tag] = {'u_rate': float((mixture_logp(ys, ph).numpy().argmax(1) == T.VALUES.index('U')).mean())}
        ctl = ind & np.isin(st.test, TT.CONTROL_TESTS)
        ys, ph, _ = loop_value_logp(model, st, ctl)
        out['S4-control-u-rate'] = float((mixture_logp(ys, ph).numpy().argmax(1) == T.VALUES.index('U')).mean())
    out['inspect'] = inspect(model, st)
    return out


def reads_one(model, st, kind, train_rows, val_rows):
    if kind == 'unit':
        out = TT.reads_unit(model, st)
        ind = (st.kind == 'individual') & (st.world_y >= 0)
        with torch.no_grad():
            for tst in ('L1-kind', 'L1-anc', 'L1-size', 'L2-named', 'L2-x', 'P6-anc'):
                m = ind & (st.test == tst)
                s = model(torch.as_tensor(st.a[m]), torch.as_tensor(st.r[m]), torch.as_tensor(st.b[m])).numpy()
                out[tst] = R1.metrics(s, st.world_y[m])
            for tst, subs in (('L1-kind', ('3', '4')), ('L1-anc', ('3', '4', '5')),
                              ('L1-size', tuple(str(d) for d in range(3, 12))),
                              ('L2-named', ('named-can', 'named-has', 'named-kind'))):
                res = {}
                for sub in subs:
                    m = ind & (st.test == tst) & (st.sub == sub)
                    if m.any():
                        s = model(torch.as_tensor(st.a[m]), torch.as_tensor(st.r[m]), torch.as_tensor(st.b[m])).numpy()
                        res[sub] = R1.metrics(s, st.world_y[m])['bal_acc']
                out[tst + '-by'] = res
        return out
    out = TT.reads_pred(model, st, None, train_rows, val_rows)
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    for tst in ('L1-kind', 'L1-anc', 'L1-size', 'L2-named', 'L2-x', 'P6-anc'):
        m = ind & (st.test == tst)
        out[tst] = TT.tf_metrics(TT.predict_values(model, st, m), st.world_y[m])
    for tst, subs in (('L1-kind', ('3', '4')), ('L1-anc', ('3', '4', '5')), ('L1-size', tuple(str(d) for d in range(3, 12))),
                      ('L2-named', ('named-can', 'named-has', 'named-kind'))):
        res = {}
        for sub in subs:
            m = ind & (st.test == tst) & (st.sub == sub)
            if m.any():
                res[sub] = TT.tf_metrics(TT.predict_values(model, st, m), st.world_y[m])['bal_acc']
        out[tst + '-by'] = res
    return out


# ---------------------------------------------------------------- fast mapping through the loop

def fit_new(model, st, seed, head_fact, is_loop, kind):
    """Fit a new head code from one T fact (rel, tail), code kept among the known codes (Gaussian prior)."""
    rel, tail = head_fact
    known = [i for i, x in enumerate(st.entities) if x not in T.VALUES + ('nil',)]
    mu, P = V2.gaussian_prior(model.E[known])
    g = torch.Generator().manual_seed(seed * 1000 + 13 + len(tail))
    code = nn.Parameter(torch.randn(1, D, generator=g) * TT.INIT_STD)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([code], lr=V2.FIT_LR)
    r = torch.tensor([st.relations.index(rel)])
    cb = model.E[[st.index(tail)]]
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        dl = code[0] - mu
        if is_loop:
            ys, ph, _ = model(code, r, cb, torch.zeros(1, dtype=torch.long), T_TRAIN)
            if kind == 'pred':
                nll = -mixture_logp(ys, ph)[0, T.VALUES.index('T')]
            else:
                nll = -torch.log(mixture_prob_unit(ys, ph))[0]
        elif kind == 'pred':
            out = model.out_codes(code, model.R[r], cb, torch.zeros(1, dtype=torch.long))
            nll = F.cross_entropy(model.logits(out, model.E[TT.value_idx_cache]), torch.tensor([T.VALUES.index('T')]))
        else:
            nll = F.binary_cross_entropy_with_logits(model.score_codes(code, model.R[r], cb), torch.ones(1))
        (nll + .5 * dl @ P @ dl).backward()
        opt.step()
    return code.detach()


@torch.no_grad()
def read_items(model, st, code, items, is_loop, kind):
    """items: [(rel, tail)] -> predicted value names (pred) or yes/no (unit)."""
    n = len(items)
    ca = code.expand(n, -1)
    r = torch.tensor([st.relations.index(x[0]) for x in items])
    cb = model.E[[st.index(x[1]) for x in items]]
    gap = torch.zeros(n, dtype=torch.long)
    if is_loop:
        ys, ph, _ = model(ca, r, cb, gap, T_TRAIN)
        if kind == 'pred':
            return np.array(T.VALUES)[mixture_logp(ys, ph).argmax(1).numpy()]
        return np.where(mixture_prob_unit(ys, ph).numpy() > .5, 'T', 'F')
    if kind == 'pred':
        out = model.out_codes(ca, model.R[r], cb, gap)
        return np.array(T.VALUES)[model.logits(out, model.E[TT.value_idx_cache]).argmax(1).numpy()]
    return np.where(model.score_codes(ca, model.R[r], cb).numpy() > 0, 'T', 'F')


def fast_mapping(model, st, seed, arm):
    kind = 'pred' if arm.startswith('pred') else 'unit'
    is_loop = arm.endswith('loop')
    res = {}
    code = fit_new(model, st, seed, ('is-a', 'bird'), is_loop, kind)
    for q in ('always', 'usually', 'sometimes', 'never'):
        if kind == 'unit' and q == 'sometimes':
            continue
        items = [(rel, val, e) for rel, val, e, qq in T.default_expectations('bird') if qq == q]
        pred = read_items(model, st, code, [(x[0], x[1]) for x in items], is_loop, kind)
        exp = np.array([x[2] for x in items])
        res[f'dax-{q}'] = float((pred == exp).mean())
    others = [c for c in T.W3.CLASSES if c != 'bird']
    res['dax-other-classes-rejected'] = float((read_items(model, st, code, [('is-a', c) for c in others], is_loop, kind) == 'F').mean())
    code = fit_new(model, st, seed, ('names', 'owl'), is_loop, kind)
    owl = T.W3.ANIMALS['owl']
    items = ([('named-can', ab, ab in owl[2].split()) for ab in T.W3.W1.ABILITIES]
             + [('named-has', p, p in owl[1].split()) for p in T.W3.W1.PARTS]
             + [('named-kind', k, k in T2.kind_chain('owl')) for k in T2.KINDS])
    pred = read_items(model, st, code, [(x[0], x[1]) for x in items], is_loop, kind)
    y = np.array([x[2] for x in items])
    tpr, tnr = float((pred[y] == 'T').mean()), float((pred[~y] == 'F').mean())
    res['blick-bal-acc'] = .5 * (tpr + tnr)
    return res


@torch.no_grad()
def inference_timing(model, st, arm):
    rows = np.flatnonzero((st.kind == 'individual') & (st.world_y >= 0))[:1000]
    m = np.zeros(len(st.a), bool)
    m[rows] = True
    fn = (lambda: loop_value_logp(model, st, m)) if arm.endswith('loop') else (
        (lambda: TT.predict_values(model, st, m)) if arm.startswith('pred') else
        (lambda: model(torch.as_tensor(st.a[m]), torch.as_tensor(st.r[m]), torch.as_tensor(st.b[m]))))
    fn()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {'median_seconds_per_1000': float(np.median(ts)), 'spread': [float(min(ts)), float(max(ts))], 'repeats': 5,
            'threads': 1, 'includes': 'forward only, batched, CPU float32'}


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
    if arm == 'pred-one':
        model = TT.train_pred(st, 'pred', seed, g, train_rows)
    elif arm == 'unit-one':
        model = TT.train_unit(st, seed, g, train_rows)
    else:
        model = train_loop(st, 'pred' if arm == 'pred-loop' else 'unit', seed, g, train_rows)
    train_cpu = time.process_time() - t0
    if arm.endswith('loop'):
        res = {'tests': reads_loop(model, st, train_rows, val)}
    else:
        res = {'tests': reads_one(model, st, 'pred' if arm == 'pred-one' else 'unit', train_rows, val)}
    timing = inference_timing(model, st, arm)
    res['fast_mapping'] = fast_mapping(model, st, seed, arm)
    res.update(arm=arm, seed=seed, params=R1.n_params(model))
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'fit_steps': FIT_STEPS, 'params': R1.n_params(model), 'train_cpu_seconds': train_cpu,
                 'inference': timing, 'trained_statements': int(train_rows.sum()), 'checkpoint_bytes': ck.stat().st_size}


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
    tasks = [(a, s) for a in ('pred-loop', 'unit-loop', 'pred-one', 'unit-one') for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(int(os.environ.get('EV_WORKERS', 8)),
                                                                 maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
                     'world_sha256': nc.sha256(T2.__file__), 'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                     'tasks': {k: t for k, _, t in res}})
    st, _ = T2.build()
    V2.save_json(f'{REPORT}.json', {'status': 'multistep reader v1' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                    'statements': len(st.a), 'results': {k: r for k, r, _ in res}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    by = {}
    for r in res.values():
        by.setdefault(r['arm'], []).append(r)

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float)) and x is not None]
        return round(float(np.mean(xs)), 2) if xs else None

    def agg(vals):
        if all(isinstance(v, dict) for v in vals):
            keys = set().union(*[v.keys() for v in vals])
            return {k: agg([v.get(k) for v in vals if v.get(k) is not None]) for k in sorted(keys)}
        return mean(vals)
    out = {arm: {'tests': agg([r['tests'] for r in rs]), 'fast_mapping': agg([r['fast_mapping'] for r in rs]),
                 'params': rs[0]['params']} for arm, rs in by.items()}
    V2.save_json(f'{REPORT}-summary.json', out)
    keys = ('P1-default', 'P2', 'P3', 'P4', 'P6-grandparent', 'P6-colour', 'G2', 'G4', 'L1-kind', 'L1-anc', 'L1-size',
            'L2-named', 'L2-x', 'P6-anc')
    arms = [a for a in ('pred-one', 'pred-loop', 'unit-one', 'unit-loop') if a in out]
    print('test'.ljust(16) + ''.join(a.rjust(12) for a in arms))
    for k in keys:
        row = []
        for a in arms:
            v = out[a]['tests'].get(k, {})
            row.append('-' if not v or v.get('bal_acc') is None else f"{v['bal_acc']:.2f}")
        print(k.ljust(16) + ''.join(x.rjust(12) for x in row))
    for k in ('L1-kind-by', 'L1-anc-by', 'L1-size-by', 'L2-named-by', 'L1-anc-by-T6', 'L1-size-by-T6', 'S3b-by-quantity',
              'S4a', 'S4b-P7', 'S4-control-u-rate', 'inspect'):
        for a in arms:
            if k in out[a]['tests']:
                print(k, a, out[a]['tests'][k])
    for a in arms:
        print('fast', a, out[a]['fast_mapping'])


if __name__ == '__main__':
    {'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
