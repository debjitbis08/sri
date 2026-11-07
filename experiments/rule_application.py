"""Rule application v1 (docs/rule-application-v1.md): facts and formula rules with variables are given as context;
a pair-state network (Edge Transformer step, looped with tied weights) learns modus ponens from consequences on
problems of depth <= 2 and is tested on deeper chains, fresh symbols, larger problems, rule edits and taught
relations (oracle, reader and reader-closure facts). Arms: edge-loop, edge-untied, pair-loop.

Usage: EV_WORKERS=6 .venv/bin/python -m experiments.rule_application        (rerun to resume)
       .venv/bin/python -m experiments.rule_application summary
       EV_SMOKE=1 ... short check
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
from experiments import everyday_relations_v2 as V2

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'rule-application-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'rule-application-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = int(os.environ.get('EV_STEPS', 100)) if SMOKE else 6000
SEEDS = (460,) if SMOKE else (460, 461, 462, 463)
ARMS = ('edge-loop', 'edge-untied', 'pair-loop')
BATCH, LR = 32, 1e-3
D, HEADS, ID_D, SYM_D, VAR_D = 64, 4, 32, 16, 8
T_TRAIN = (2, 3, 4)
T_TEST = (4, 8, 16)
N_TEST = 20 if SMOKE else 500                      # per depth bucket / per extra set
N_EDIT = 20 if SMOKE else 500
DENSITY = .15
MAX_ATTEMPTS = 200

# ---------------------------------------------------------------- rule forms (x=0, y=1, z=2)

FORMS = {
    'sub': ((0, 1), [('A', (0, 1))]),
    'inv': ((0, 1), [('A', (1, 0))]),
    'chain1': ((0, 2), [('A', (0, 1)), ('B', (1, 2))]),
    'chain2': ((0, 2), [('A', (1, 0)), ('B', (1, 2))]),
    'chain3': ((0, 2), [('A', (0, 1)), ('B', (2, 1))]),
    'chain4': ((0, 2), [('A', (1, 0)), ('B', (2, 1))]),
    'conj': ((0, 1), [('A', (0, 1)), ('B', (0, 1))]),
    'sym': ((0, 1), [('H', (1, 0))]),
    'trans': ((0, 2), [('H', (0, 1)), ('H', (1, 2))]),
    'lin-l': ((0, 2), [('A', (0, 1)), ('H', (1, 2))]),
    'lin-r': ((0, 2), [('H', (0, 1)), ('A', (1, 2))]),
}
FORM_IDX = {f: i for i, f in enumerate(FORMS)}
SEED_TRAIN = ('sub', 'inv', 'chain1', 'chain2', 'chain3', 'conj')
SEED_TEST = SEED_TRAIN + ('chain4',)
REC = ('sym', 'trans', 'lin-l', 'lin-r')

_GRID = {}


def grid(n):
    if n not in _GRID:
        _GRID[n] = np.indices((n, n, n))
    return _GRID[n]


def apply_rule(S, rule, n):
    """Head matrix (n, n) derived by one rule from state S (m, n, n) bool."""
    head, (ha, hb), body, _ = rule
    V = grid(n)
    B = np.ones((n, n, n), dtype=bool)
    for r, (va, vb) in body:
        B &= S[r][V[va], V[vb]]
    other = [a for a in range(3) if a not in (ha, hb)][0]
    return np.transpose(B, (ha, hb, other)).any(2)


def closure(S0, rules, n, max_rounds=64):
    S = S0.copy()
    Dp = np.where(S0, 0, -1)
    prov = np.full(S0.shape, -1)
    for k in range(1, max_rounds + 1):
        new = np.zeros_like(S)
        newprov = np.full(S0.shape, -1)
        for rule in rules:
            Hm = apply_rule(S, rule, n) & ~S[rule[0]] & ~new[rule[0]]
            new[rule[0]] |= Hm
            newprov[rule[0]][Hm] = FORM_IDX[rule[3]]
        if not new.any():
            break
        S |= new
        Dp[new] = k
        prov[new] = newprov[new]
    return S, Dp, prov


def unit(rng, k, d):
    v = rng.standard_normal((k, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def make_rules(rng, nb, nd, seed_forms, force=None):
    rules = []
    for j in range(nd):
        H = nb + j

        def pick():
            if j > 0 and rng.random() < .3:
                return nb + int(rng.integers(j))
            return int(rng.integers(nb))
        form = force if (force is not None and j == 0) else seed_forms[rng.integers(len(seed_forms))]
        head, body = FORMS[form]
        slots = {'A': pick(), 'B': pick()}
        rules.append((H, head, [(slots[s], v) for s, v in body], form))
        if rng.random() < .5:
            form = REC[rng.integers(len(REC))]
            head, body = FORMS[form]
            slots = {'A': pick(), 'H': H}
            rules.append((H, head, [(slots[s], v) for s, v in body], form))
    return rules


def problem(rng, sizes, seed_forms, force=None):
    (n_lo, n_hi), (b_lo, b_hi), (d_lo, d_hi), (c_lo, c_hi) = sizes
    n = int(rng.integers(n_lo, n_hi + 1))
    nb, nd = int(rng.integers(b_lo, b_hi + 1)), int(rng.integers(d_lo, d_hi + 1))
    return n, nb, nd


def draw(rng, n, nb, nd, chain_len, seed_forms, force=None):
    m = nb + nd
    rules = make_rules(rng, nb, nd, seed_forms, force)
    S0 = np.zeros((m, n, n), dtype=bool)
    eye = np.eye(n, dtype=bool)
    for r in range(nb):
        S0[r] = (rng.random((n, n)) < DENSITY) & ~eye
    base_atoms = [(rel, v) for rule in rules for rel, v in rule[2] if rel < nb]
    for _ in range(int(rng.integers(1, 3))):
        if not base_atoms:
            break
        rel, (va, vb) = base_atoms[rng.integers(len(base_atoms))]
        L = int(rng.integers(chain_len[0], min(chain_len[1], n) + 1))
        path = rng.choice(n, L, replace=False)
        for i in range(L - 1):
            a, b = (path[i], path[i + 1]) if va < vb else (path[i + 1], path[i])
            S0[rel, a, b] = True
    S, Dp, prov = closure(S0, rules, n)
    return dict(n=n, nb=nb, m=m, S0=S0, rules=rules, S=S, D=Dp, prov=prov)


def acceptable(p):
    n, nb, m = p['n'], p['nb'], p['m']
    for r in range(nb, m):
        frac = p['S'][r].mean()
        if not p['S'][r].any() or frac < .05 or frac > .6:
            return False
    return True


def max_depth(p):
    return int(p['D'][p['nb']:].max())


SIZES = {'train': ((6, 10), (2, 3), (1, 3), (2, 4)), 'test': ((10, 16), (2, 4), (1, 3), (4, 8)),
         'large': ((12, 16), (2, 4), (1, 3), (4, 8))}


def generate(rng, kind, depth_ok, seed_forms, force=None, stats=None):
    (nr, br, dr, cl) = SIZES[kind]
    while True:
        n = int(rng.integers(nr[0], nr[1] + 1))
        nb, nd = int(rng.integers(br[0], br[1] + 1)), int(rng.integers(dr[0], dr[1] + 1))
        for _ in range(MAX_ATTEMPTS):
            p = draw(rng, n, nb, nd, cl, seed_forms, force)
            if stats is not None:
                stats['attempts'] = stats.get('attempts', 0) + 1
            if acceptable(p) and depth_ok(max_depth(p)):
                p['ent'], p['sym'] = unit(rng, n, ID_D), unit(rng, p['m'], SYM_D)
                return p
        if stats is not None:
            stats['redraws'] = stats.get('redraws', 0) + 1


# ---------------------------------------------------------------- batching

def collate(probs):
    B = len(probs)
    n = max(p['n'] for p in probs)
    m = max(p['m'] for p in probs)
    R = max(len(p['rules']) for p in probs)
    ent = np.zeros((B, n, ID_D), np.float32)
    emask = np.zeros((B, n), bool)
    sym = np.zeros((B, m, SYM_D), np.float32)
    rmask = np.zeros((B, m), bool)
    S0 = np.zeros((B, m, n, n), np.float32)
    Y = np.zeros((B, m, n, n), np.float32)
    Wt = np.zeros((B, m, n, n), np.float32)
    arel = np.zeros((B, R, 3), np.int64)
    ava = np.zeros((B, R, 3), np.int64)
    avb = np.zeros((B, R, 3), np.int64)
    amask = np.zeros((B, R, 3), bool)
    rumask = np.zeros((B, R), bool)
    for i, p in enumerate(probs):
        k, mm = p['n'], p['m']
        ent[i, :k], emask[i, :k] = p['ent'], True
        sym[i, :mm], rmask[i, :mm] = p['sym'], True
        S0[i, :mm, :k, :k] = p['S0']
        Y[i, :mm, :k, :k] = p['S']
        t = p['S']
        nt, nf = t.sum(), t.size - t.sum()
        Wt[i, :mm, :k, :k] = np.where(t, .5 / max(nt, 1), .5 / max(nf, 1))
        for j, (h, (ha, hb), body, _) in enumerate(p['rules']):
            rumask[i, j] = True
            atoms = [(h, ha, hb)] + [(r, va, vb) for r, (va, vb) in body]
            for a, (r, va, vb) in enumerate(atoms):
                arel[i, j, a], ava[i, j, a], avb[i, j, a], amask[i, j, a] = r, va, vb, True
    T_ = torch.as_tensor
    return dict(ent=T_(ent), emask=T_(emask), sym=T_(sym), rmask=T_(rmask), S0=T_(S0), Y=T_(Y), W=T_(Wt),
                arel=T_(arel), ava=T_(ava), avb=T_(avb), amask=T_(amask), rumask=T_(rumask))


# ---------------------------------------------------------------- model

def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class Step(nn.Module):
    def __init__(self, tri):
        super().__init__()
        self.tri = tri
        self.W0 = nn.Linear(D, D)
        if tri:
            self.q, self.k1, self.k2 = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
            self.v1, self.v2, self.o = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        else:
            self.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.rev = mlp(2 * D, D)
        self.ratt = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ff = mlp(D, D)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)

    def triangle(self, h, emask):
        B, n = h.shape[:2]
        dh = D // HEADS
        sh = lambda x: x.view(B, n, n, HEADS, dh)
        q, k1, k2 = sh(self.q(h)), sh(self.k1(h)), sh(self.k2(h))
        v1, v2 = sh(self.v1(h)), sh(self.v2(h))
        s = (torch.einsum('bxzhd,bxyhd->bxyzh', q, k1) + torch.einsum('bxzhd,byzhd->bxyzh', q, k2)) / dh ** .5
        s = s.masked_fill(~emask[:, None, :, None, None], float('-inf'))
        a = torch.softmax(s, 2)
        out = torch.einsum('bxyzh,bxyhd,byzhd->bxzhd', a, v1, v2).reshape(B, n, n, D)
        return self.o(out)

    def full(self, h, pmask):
        B, n = h.shape[:2]
        x = h.reshape(B, n * n, D)
        out, _ = self.att(x, x, x, key_padding_mask=~pmask.reshape(B, n * n), need_weights=False)
        return out.reshape(B, n, n, D)

    def forward(self, h, h0, rules, ctx):
        B, n = h.shape[:2]
        h = h + self.W0(h0)
        h = self.ln1(h + (self.triangle(h, ctx['emask']) if self.tri else self.full(h, ctx['pmask'])))
        h = h + self.rev(torch.cat([h, h.transpose(1, 2)], -1))
        x = h.reshape(B, n * n, D)
        r, _ = self.ratt(x, rules, rules, key_padding_mask=~ctx['rumask'], need_weights=False)
        h = self.ln2(h + r.reshape(B, n, n, D))
        return self.ln3(h + self.ff(h))


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.inp = mlp(2 * SYM_D + 2 * ID_D + 1, D)
        self.var = nn.Parameter(torch.randn(3, VAR_D) * .3)
        self.atom = mlp(SYM_D + 2 * VAR_D + 3, D)
        self.rule = mlp(D, D)
        if arm == 'edge-untied':
            self.steps = nn.ModuleList([Step(True) for _ in range(4)])
        else:
            self.step = Step(arm == 'edge-loop')
        self.U = nn.Linear(D, SYM_D)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, bt, T):
        ent, emask, sym, S0 = bt['ent'], bt['emask'], bt['sym'], bt['S0']
        B, n = ent.shape[:2]
        ff = torch.einsum('bmxy,bmd->bxyd', S0, sym)
        eye = torch.eye(n).expand(B, n, n)[..., None]
        x = torch.cat([ff, ff.transpose(1, 2), ent[:, :, None].expand(B, n, n, ID_D),
                       ent[:, None].expand(B, n, n, ID_D), eye], -1)
        h0 = self.inp(x)
        asym = torch.gather(sym, 1, bt['arel'].reshape(B, -1)[..., None].expand(-1, -1, SYM_D)).reshape(*bt['arel'].shape, SYM_D)
        role = torch.eye(3).expand(*bt['arel'].shape[:2], 3, 3)
        a = self.atom(torch.cat([asym, self.var[bt['ava']], self.var[bt['avb']], role], -1))
        rules = self.rule((a * bt['amask'][..., None]).sum(2))
        pmask = emask[:, :, None] & emask[:, None, :]
        ctx = dict(emask=emask, pmask=pmask, rumask=bt['rumask'])
        h = h0
        if self.arm == 'edge-untied':
            for st in self.steps:
                h = st(h, h0, rules, ctx)
        else:
            for _ in range(T):
                h = self.step(h, h0, rules, ctx)
        return torch.einsum('bmd,bxyd->bmxy', sym, self.U(h)) + self.b


# ---------------------------------------------------------------- training

def train(arm, seed):
    torch.manual_seed(seed)
    net = Net(arm)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    log, stats = [], {}
    gen_cpu = 0.
    for step in range(1, STEPS + 1):
        c0 = time.process_time()
        rng = np.random.default_rng([seed, 1, step])
        probs = [generate(rng, 'train', (lambda d: d == 2) if i < BATCH // 2 else (lambda d: d <= 2), SEED_TRAIN,
                          stats=stats) for i in range(BATCH)]
        bt = collate(probs)
        gen_cpu += time.process_time() - c0
        T = T_TRAIN[rng.integers(len(T_TRAIN))]
        opt.zero_grad()
        logit = net(bt, T)
        loss = (F.binary_cross_entropy_with_logits(logit, bt['Y'], reduction='none') * bt['W']).sum() / BATCH
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == STEPS:
            with torch.no_grad():
                pred = logit > 0
                y = bt['Y'] > .5
                valid = bt['W'] > 0
                tr = (pred & y & valid).sum() / max((y & valid).sum(), 1)
                fr = (~pred & ~y & valid).sum() / max((~y & valid).sum(), 1)
            log.append({'step': step, 'loss': loss.item(), 'bal': float(.5 * (tr + fr)), 'T': int(T)})
    for p in net.parameters():
        p.requires_grad_(False)
    return net, log, {'generation_cpu_seconds': gen_cpu, **stats}


# ---------------------------------------------------------------- evaluation helpers

@torch.no_grad()
def predict(net, probs, T, bs=50):
    """Boolean predictions (m, n, n) per problem."""
    out = [None] * len(probs)
    order = sorted(range(len(probs)), key=lambda i: probs[i]['n'])
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        bt = collate([probs[i] for i in idx])
        lg = net(bt, T)
        for j, i in enumerate(idx):
            p = probs[i]
            out[i] = (lg[j, :p['m'], :p['n'], :p['n']] > 0).numpy()
    return out


def path_lengths(p, cap=12):
    n, nb = p['n'], p['nb']
    A = p['S0'][:nb].any(0)
    A = A | A.T
    dist = np.full((n, n), -1)
    reach = np.eye(n, dtype=bool)
    frontier = reach.copy()
    for L in range(1, cap + 1):
        frontier = (frontier.astype(np.int32) @ A.astype(np.int32) > 0) & ~reach
        dist[frontier & (dist < 0)] = L
        reach |= frontier
    return dist


CONN = ((1, 2), (3, 4), (5, 6), (7, 12))


def tally(probs, preds, rel_sel=None, label_key='S', depth_key='D'):
    """Counts: T recall by depth, F recall, connected negatives by path bucket, held-out-form T recall."""
    c = {'T': {}, 'F': [0, 0], 'conn': {f'{a}-{b}': [0, 0] for a, b in CONN}, 'chain4': [0, 0]}
    for p, pr in zip(probs, preds):
        rels = rel_sel(p) if rel_sel else range(p['nb'], p['m'])
        dist = path_lengths(p)
        for r in rels:
            y, d, q = p[label_key][r], p[depth_key][r], pr[r]
            for dv in np.unique(d[y]):
                m = y & (d == dv)
                h = c['T'].setdefault(int(dv), [0, 0])
                h[0] += int(q[m].sum()); h[1] += int(m.sum())
            neg = ~y
            c['F'][0] += int((~q[neg]).sum()); c['F'][1] += int(neg.sum())
            for a, b in CONN:
                m = neg & (dist >= a) & (dist <= b)
                c['conn'][f'{a}-{b}'][0] += int((~q[m]).sum()); c['conn'][f'{a}-{b}'][1] += int(m.sum())
            if 'prov' in p and label_key == 'S':
                m = y & (p['prov'][r] == FORM_IDX['chain4'])
                c['chain4'][0] += int(q[m].sum()); c['chain4'][1] += int(m.sum())
    return c


def rate(h):
    return None if h[1] == 0 else h[0] / h[1]


def report(c):
    fr = rate(c['F'])
    T = {d: rate(h) for d, h in sorted(c['T'].items())}
    return {'T_recall': T, 'T_n': {d: h[1] for d, h in sorted(c['T'].items())}, 'F_recall': fr,
            'bal_by_depth': {d: None if (t is None or fr is None) else .5 * (t + fr) for d, t in T.items()},
            'conn_F_recall': {k: rate(h) for k, h in c['conn'].items()},
            'conn_n': {k: h[1] for k, h in c['conn'].items()}, 'chain4_T_recall': rate(c['chain4']),
            'chain4_n': c['chain4'][1]}


# ---------------------------------------------------------------- test sets

def test_sets(seed):
    stats = {}
    rng = np.random.default_rng([seed, 2])
    main = [generate(rng, 'test', (lambda d, k=k: d == k), SEED_TEST, stats=stats)
            for k in (3, 4, 5, 6) for _ in range(N_TEST)]
    held = [generate(rng, 'test', (lambda d: d <= 2) if i < N_TEST // 2 else (lambda d: 3 <= d <= 4), SEED_TEST,
                     force='chain4', stats=stats) for i in range(N_TEST)]
    large = [generate(rng, 'large', lambda d: d <= 2, SEED_TRAIN, stats=stats) for _ in range(N_TEST)]
    indist = [generate(rng, 'train', (lambda d: d == 2) if i % 2 else (lambda d: d <= 2), SEED_TRAIN, stats=stats)
              for i in range(N_TEST)]
    return dict(main=main, held=held, large=large, indist=indist), stats


def rebuild(p, S0=None, rules=None, sym=None, ent=None):
    q = dict(p)
    q['S0'] = p['S0'] if S0 is None else S0
    q['rules'] = p['rules'] if rules is None else rules
    q['m'] = q['S0'].shape[0]
    q['S'], q['D'], q['prov'] = closure(q['S0'], q['rules'], q['n'])
    q['sym'] = p['sym'] if sym is None else sym
    q['ent'] = p['ent'] if ent is None else ent
    return q


def edits(p, rng):
    """Relevant edits e1-e3 and irrelevant edits i1-i3 of a problem."""
    out = {}
    rules = p['rules']
    j = int(rng.integers(len(rules)))
    h, hv, body, form = rules[j]
    r0, (va, vb) = body[0]
    out['e1-body-args'] = rebuild(p, rules=rules[:j] + [(h, hv, [(r0, (vb, va))] + body[1:], form)] + rules[j + 1:])
    cand = [r for r in range(p['m']) if r not in (r0, h)]
    if cand:
        r1 = int(cand[rng.integers(len(cand))])
        out['e2-body-rel'] = rebuild(p, rules=rules[:j] + [(h, hv, [(r1, (va, vb))] + body[1:], form)] + rules[j + 1:])
    out['e3-head-args'] = rebuild(p, rules=rules[:j] + [(h, (hv[1], hv[0]), body, form)] + rules[j + 1:])
    out['i1-fresh-vectors'] = rebuild(p, sym=unit(rng, p['m'], SYM_D), ent=unit(rng, p['n'], ID_D))
    n, m = p['n'], p['m']
    S0 = np.concatenate([p['S0'], np.zeros((1, n, n), bool)])
    hd = int(rng.integers(p['nb'], m))
    out['i2-dead-rule'] = rebuild(p, S0=S0, rules=rules + [(hd, (0, 1), [(m, (0, 1))], 'sub')],
                                  sym=np.concatenate([p['sym'], unit(rng, 1, SYM_D)]))
    S0 = np.concatenate([p['S0'], ((rng.random((1, n, n)) < DENSITY) & ~np.eye(n, dtype=bool))])
    out['i3-unused-relation'] = rebuild(p, S0=S0, sym=np.concatenate([p['sym'], unit(rng, 1, SYM_D)]))
    return out


def edit_reads(net, probs, T, seed, rels_of=None):
    rng = np.random.default_rng([seed, 5])
    base = predict(net, probs, T)
    ed = [edits(p, rng) for p in probs]
    out = {}
    for key in ('e1-body-args', 'e2-body-rel', 'e3-head-args', 'i1-fresh-vectors', 'i2-dead-rule', 'i3-unused-relation'):
        sel = [i for i in range(len(probs)) if key in ed[i]]
        pr = predict(net, [ed[i][key] for i in sel], T)
        flip, keep, acc = [0, 0], [0, 0], [0, 0]
        for i, q in zip(sel, pr):
            p, e = probs[i], ed[i][key]
            rels = rels_of(p) if rels_of else range(p['nb'], p['m'])
            for r in rels:
                fl = p['S'][r] != e['S'][r]
                flip[0] += int((q[r][fl] == e['S'][r][fl]).sum()); flip[1] += int(fl.sum())
                keep[0] += int((q[r][~fl] == base[i][r][~fl]).sum()); keep[1] += int((~fl).sum())
                acc[0] += int((q[r] == e['S'][r]).sum()); acc[1] += q[r].size
        out[key] = {'flipped_correct': rate(flip), 'n_flipped': flip[1], 'unchanged_answer_rate': rate(keep),
                    'accuracy': rate(acc), 'problems': len(sel)}
    return out


# ---------------------------------------------------------------- taught relations

def transitive(M):
    C = M.copy()
    while True:
        N = C | ((C.astype(np.int32) @ C.astype(np.int32)) > 0)
        if (N == C).all():
            return C
        C = N


def reduce_kind(K, Sb):
    C = transitive(Sb)
    out = K.copy()
    n = len(K)
    for x in range(n):
        cs = np.flatnonzero(K[x])
        for d in cs:
            if any(c != d and C[c, d] for c in cs):
                out[x, d] = False
    return out


TAUGHT = {
    'ancestor-of': dict(base=('parent-of', 'child-of'), derived=('ancestor-of',),
                        rules=[(2, (0, 1), [(0, (0, 1))], 'sub'), (2, (0, 2), [(0, (0, 1)), (2, (1, 2))], 'lin-l')]),
    'bigger-than': dict(base=('just-bigger-than',), derived=('bigger-than',),
                        rules=[(1, (0, 1), [(0, (0, 1))], 'sub'), (1, (0, 2), [(0, (0, 1)), (1, (1, 2))], 'lin-l')]),
    'smaller-than': dict(base=('just-bigger-than',), derived=('bigger-than', 'smaller-than'),
                         rules=[(1, (0, 1), [(0, (0, 1))], 'sub'), (1, (0, 2), [(0, (0, 1)), (1, (1, 2))], 'lin-l'),
                                (2, (0, 1), [(1, (1, 0))], 'inv')]),
    'is-a': dict(base=('kind-of', 'subkind-of'), derived=('is-a',),
                 rules=[(2, (0, 1), [(0, (0, 1))], 'sub'), (2, (0, 2), [(2, (0, 1)), (1, (1, 2))], 'lin-r')]),
    'grandparent-of': dict(base=('parent-of',), derived=('grandparent-of',),
                           rules=[(1, (0, 2), [(0, (0, 1)), (0, (1, 2))], 'chain1')]),
    'child-of': dict(base=('parent-of',), derived=('child-of',), rules=[(1, (0, 1), [(0, (1, 0))], 'inv')]),
}
ENT_REL = {'ancestor-of': ('parent-of',), 'bigger-than': ('just-bigger-than',), 'smaller-than': ('just-bigger-than',),
           'is-a': ('kind-of', 'subkind-of'), 'grandparent-of': ('parent-of',), 'child-of': ('parent-of',)}


class TaughtWorld:
    def __init__(self):
        from experiments import relations_as_objects as RO
        self.W = RO.World(450)
        st = self.W.st
        self.st = st
        self.E = {x: i for i, x in enumerate(st.entities)}
        ind = st.kind == 'individual'
        self.pos, self.lab = {}, {}
        for i, rel in enumerate(self.W.real):
            m = ind & (st.r == i) & (st.world_y >= 0)
            self.lab[rel] = {(int(a), int(b)): int(y) for a, b, y in zip(st.a[m], st.r[m] * 0 + st.b[m], st.world_y[m])}
            self.pos[rel] = {k for k, v in self.lab[rel].items() if v == 1}

    def entities(self, name):
        s = set()
        for rel in ENT_REL[name]:
            for a, b in self.pos[rel]:
                s |= {a, b}
        return sorted(s)

    def true_matrix(self, rel, ents):
        ix = {e: i for i, e in enumerate(ents)}
        M = np.zeros((len(ents), len(ents)), bool)
        for a, b in self.pos[rel]:
            if a in ix and b in ix:
                M[ix[a], ix[b]] = True
        return M


def reader_matrix(model, W, rel, ents):
    from experiments import relations_as_objects as RO
    r = W.R[rel]
    trip = [(a, r, b) for a in ents for b in ents]
    lv, _ = RO.logic_answers(model.net, trip)
    M = (np.array(lv) == 'T').reshape(len(ents), len(ents))
    np.fill_diagonal(M, False)
    return M


def taught_problem(TW, name, rng, mats):
    spec = TAUGHT[name]
    ents = TW.entities(name)
    n = len(ents)
    base = [mats[b] for b in spec['base']]
    if name == 'is-a':
        base[0] = reduce_kind(base[0], base[1])
    m = len(spec['base']) + len(spec['derived'])
    S0 = np.zeros((m, n, n), bool)
    for i, M in enumerate(base):
        S0[i] = M
    p = dict(n=n, nb=len(spec['base']), m=m, S0=S0, rules=list(spec['rules']), ent=unit(rng, n, ID_D),
             sym=unit(rng, m, SYM_D), ents=ents, name=name)
    p['S'], p['D'], p['prov'] = closure(S0, p['rules'], n)
    return p


def taught_edits(p, rng):
    out = {}
    rules = p['rules']
    nb = p['nb']
    swap = [(h, hv, [(r, (v[1], v[0])) if r < nb else (r, v) for r, v in body], f) for h, hv, body, f in rules]
    out['e1-body-args'] = rebuild(p, rules=swap)
    if p['name'] == 'ancestor-of':
        out['e2-body-rel'] = rebuild(p, rules=[(h, hv, [(1 if r == 0 else r, v) for r, v in body], f)
                                               for h, hv, body, f in rules])
    h, hv, body, f = rules[0]
    out['e3-head-args'] = rebuild(p, rules=[(h, (hv[1], hv[0]), body, f)] + rules[1:])
    out['i1-fresh-vectors'] = rebuild(p, sym=unit(rng, p['m'], SYM_D), ent=unit(rng, p['n'], ID_D))
    n, m = p['n'], p['m']
    tgt = m - 1
    S0 = np.concatenate([p['S0'], np.zeros((1, n, n), bool)])
    out['i2-dead-rule'] = rebuild(p, S0=S0, rules=rules + [(tgt, (0, 1), [(m, (0, 1))], 'sub')],
                                  sym=np.concatenate([p['sym'], unit(rng, 1, SYM_D)]))
    S0 = np.concatenate([p['S0'], ((rng.random((1, n, n)) < DENSITY) & ~np.eye(n, dtype=bool))])
    out['i3-unused-relation'] = rebuild(p, S0=S0, sym=np.concatenate([p['sym'], unit(rng, 1, SYM_D)]))
    return out


def world_bal(TW, name, p, pred):
    tgt = p['m'] - 1
    ix = {e: i for i, e in enumerate(p['ents'])}
    ys, qs, cs = [], [], []
    for (a, b), y in TW.lab[name].items():
        if a in ix and b in ix and a != b:
            ys.append(y); qs.append(bool(pred[tgt, ix[a], ix[b]])); cs.append(bool(p['S'][tgt, ix[a], ix[b]]))
    ys, qs, cs = np.array(ys, bool), np.array(qs), np.array(cs)
    if len(ys) == 0 or ys.all() or (~ys).all():
        return None
    b = lambda q: float(.5 * (q[ys].mean() + (~q[~ys]).mean()))
    return {'network': b(qs), 'closure': b(cs), 'n': int(len(ys)), 'closure_world_disagree': int((cs != ys).sum())}


def taught_reads(net, arm, seed, TW, reader_model):
    rng = np.random.default_rng([seed, 7])
    Ts = (4,) if arm == 'edge-untied' else T_TEST
    out = {}
    for name in TAUGHT:
        spec = TAUGHT[name]
        ents = TW.entities(name)
        oracle_m = {b: TW.true_matrix(b, ents) for b in spec['base']}
        reader_m = {b: reader_matrix(reader_model, TW.W, b, ents) for b in spec['base']}
        po = taught_problem(TW, name, np.random.default_rng([seed, 8, len(name)]), oracle_m)
        pr = taught_problem(TW, name, np.random.default_rng([seed, 8, len(name)]), reader_m)
        tgt = lambda p: [p['m'] - 1]
        rec = {'n_entities': po['n'], 'hop': {}, 'oracle': {}, 'reader': {}}
        for b in spec['base']:
            o, r = po['S0'][spec['base'].index(b)], pr['S0'][spec['base'].index(b)]
            rec['hop'][b] = {'precision': rate([int((o & r).sum()), int(r.sum())]),
                             'recall': rate([int((o & r).sum()), int(o.sum())]), 'n_true': int(o.sum())}
        rec['hop_raw'] = {b: {'precision': rate([int((oracle_m[b] & reader_m[b]).sum()), int(reader_m[b].sum())]),
                              'recall': rate([int((oracle_m[b] & reader_m[b]).sum()), int(oracle_m[b].sum())])}
                          for b in spec['base']}
        # reader-closure against oracle (knowledge damage), labels of the oracle
        rec['damage'] = report(tally([po], [pr['S']], tgt))
        for T in Ts:
            qo = predict(net, [po], T)[0]
            qr = predict(net, [pr], T)[0]
            rec['oracle'][f'T{T}'] = {**report(tally([po], [qo], tgt)), 'world': world_bal(TW, name, po, qo)}
            rec['reader'][f'T{T}'] = {'vs_reader_closure': report(tally([pr], [qr], tgt)),
                                      'vs_oracle': report(tally([po], [qr], tgt)), 'world': world_bal(TW, name, po, qr)}
        T = max(Ts)
        base = predict(net, [po], T)[0]
        eds = taught_edits(po, rng)
        rec['edits'] = {}
        for key, e in eds.items():
            q = predict(net, [e], T)[0]
            r = po['m'] - 1
            fl = po['S'][r] != e['S'][r]
            rec['edits'][key] = {'flipped_correct': rate([int((q[r][fl] == e['S'][r][fl]).sum()), int(fl.sum())]),
                                 'n_flipped': int(fl.sum()),
                                 'unchanged_answer_rate': rate([int((q[r][~fl] == base[r][~fl]).sum()), int((~fl).sum())]),
                                 'accuracy': float((q[r] == e['S'][r]).mean())}
        out[name] = rec
    return out


# ---------------------------------------------------------------- tasks

def load_reader(seed):
    from experiments import amortized_introduction as AI
    from experiments import relations_as_objects as RO
    W = RO.World(seed - 10)
    model = AI.Model(W, torch.Generator().manual_seed(0))
    ck = ROOT / 'runs' / 'amortized-introduction-v1' / f'virtual-seed{seed - 10}.pt'
    model.load_state_dict(torch.load(ck))
    return model, W, nc.sha256(ck)


def run(arm, seed):
    torch.set_num_threads(1)
    ph = {}
    t0 = time.process_time()
    net, log, gstats = train(arm, seed)
    ph['train_cpu_seconds'] = time.process_time() - t0
    t0 = time.process_time()
    sets, tstats = test_sets(seed)
    ph['test_generation_cpu_seconds'] = time.process_time() - t0
    Ts = (4,) if arm == 'edge-untied' else T_TEST
    res = {'arm': arm, 'seed': seed, 'log': log, 'sets': {}}
    t0 = time.process_time()
    for key, probs in sets.items():
        res['sets'][key] = {f'T{T}': report(tally(probs, predict(net, probs, T))) for T in Ts}
    ph['synthetic_reads_cpu_seconds'] = time.process_time() - t0
    t0 = time.process_time()
    res['edits'] = edit_reads(net, sets['main'][::max(1, len(sets['main']) // N_EDIT)][:N_EDIT], max(Ts), seed)
    ph['edit_reads_cpu_seconds'] = time.process_time() - t0
    t0 = time.process_time()
    reader, W, ck_sha = load_reader(seed)
    TW = TaughtWorld()
    res['taught'] = taught_reads(net, arm, seed, TW, reader)
    res['reader_checkpoint_sha256'] = ck_sha
    ph['taught_reads_cpu_seconds'] = time.process_time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(net.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    rt = {**ph, 'params': int(sum(p.numel() for p in net.parameters())), 'steps': STEPS, 'batch': BATCH,
          'train_generation': gstats, 'test_generation': tstats, 'checkpoint_bytes': ck.stat().st_size,
          'test_problems': {k: len(v) for k, v in sets.items()}}
    return res, rt


def task(t):
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(*t)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{t[0]}/seed{t[1]}', res, rt


def main():
    t0 = time.perf_counter()
    workers = int(os.environ.get('EV_WORKERS', 6))
    tasks = [(a, s) for a in ARMS for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
        out = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'workers': workers,
                     'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(r['cpu_seconds'] for _, _, r in out)),
                     'tasks': {k: r for k, _, r in out}})
    V2.save_json(f'{REPORT}.json', {'status': 'rule application v1' + (' (smoke)' if SMOKE else ''),
                                    'results': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    m = lambda v: None if not [x for x in v if x is not None] else round(float(np.mean([x for x in v if x is not None])), 3)
    out = {}
    for arm in ARMS:
        R = [r for r in res.values() if r['arm'] == arm]
        if not R:
            continue
        Tk = list(R[0]['sets']['main'].keys())
        o = {'sets': {}, 'edits': {}, 'taught': {}}
        for s in ('main', 'held', 'large', 'indist'):
            o['sets'][s] = {}
            for T in Tk:
                rs = [r['sets'][s][T] for r in R]
                depths = sorted({d for x in rs for d in x['bal_by_depth']}, key=int)
                o['sets'][s][T] = {'bal_by_depth': {d: m([x['bal_by_depth'].get(d) for x in rs]) for d in depths},
                                   'T_recall': {d: m([x['T_recall'].get(d) for x in rs]) for d in depths},
                                   'F_recall': m([x['F_recall'] for x in rs]),
                                   'conn_F_recall': {k: m([x['conn_F_recall'][k] for x in rs]) for k in rs[0]['conn_F_recall']},
                                   'chain4_T_recall': m([x['chain4_T_recall'] for x in rs])}
        for k in R[0]['edits']:
            o['edits'][k] = {q: m([r['edits'][k][q] for r in R]) for q in ('flipped_correct', 'unchanged_answer_rate', 'accuracy')}
        for name in R[0]['taught']:
            tr = [r['taught'][name] for r in R]
            t = {'hop': {b: {q: m([x['hop'][b][q] for x in tr]) for q in ('precision', 'recall')} for b in tr[0]['hop']},
                 'damage_T_recall': {d: m([x['damage']['T_recall'].get(d) for x in tr]) for d in sorted({d for x in tr for d in x['damage']['T_recall']}, key=int)},
                 'damage_F_recall': m([x['damage']['F_recall'] for x in tr])}
            for src in ('oracle', 'reader'):
                t[src] = {}
                for T in tr[0][src]:
                    xs = [x[src][T] for x in tr]
                    if src == 'oracle':
                        rep = xs
                    else:
                        rep = [x['vs_reader_closure'] for x in xs]
                    depths = sorted({d for x in rep for d in x['bal_by_depth']}, key=int)
                    t[src][T] = {'bal_by_depth': {d: m([x['bal_by_depth'].get(d) for x in rep]) for d in depths},
                                 'F_recall': m([x['F_recall'] for x in rep]),
                                 'conn_F_recall': {k: m([x['conn_F_recall'][k] for x in rep]) for k in rep[0]['conn_F_recall']},
                                 'world_network': m([(x['world'] or {}).get('network') for x in xs]),
                                 'world_closure': m([(x['world'] or {}).get('closure') for x in xs])}
                    if src == 'reader':
                        vo = [x['vs_oracle'] for x in xs]
                        t[src][T]['vs_oracle_bal_by_depth'] = {d: m([x['bal_by_depth'].get(d) for x in vo])
                                                              for d in sorted({d for x in vo for d in x['bal_by_depth']}, key=int)}
            t['edits'] = {k: {q: m([x['edits'][k][q] for x in tr if k in x['edits']])
                              for q in ('flipped_correct', 'unchanged_answer_rate')} for k in tr[0]['edits']}
            o['taught'][name] = t
        out[arm] = o
        print(f'\n===== {arm}')
        for s in ('indist', 'large', 'main', 'held'):
            for T in Tk:
                x = o['sets'][s][T]
                print(f"  {s} {T}: bal by depth {x['bal_by_depth']} F {x['F_recall']} conn {x['conn_F_recall']} chain4 {x['chain4_T_recall']}")
        print(f"  edits {o['edits']}")
        for name, t in o['taught'].items():
            Tm = list(t['oracle'])[-1]
            print(f"  {name}: oracle {Tm} {t['oracle'][Tm]['bal_by_depth']} F {t['oracle'][Tm]['F_recall']} world {t['oracle'][Tm]['world_network']} "
                  f"| reader-vs-closure {t['reader'][Tm]['bal_by_depth']} | damage T {t['damage_T_recall']} F {t['damage_F_recall']} | hop {t['hop']} | edits {t['edits']}")
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
