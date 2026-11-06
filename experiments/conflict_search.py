"""Conflict search v1 (docs/conflict-search-v1.md): separate-code intuition and logic routes; whole higher-order
relations withheld from logic's one pass; on conflict, logic walks from a over entities following chosen relations
(and directions), one path at a time with backtracking, a trace and a budget, every hop checked by both routes with
the readers frozen; the controller learns from the outcome and from the search's own checks; v1's learned arbiter.

Usage: .venv/bin/python -m experiments.conflict_search main      (phase 1 then phase 2; rerun to resume; EV_WORKERS)
       .venv/bin/python -m experiments.conflict_search phase1
       .venv/bin/python -m experiments.conflict_search summary
       EV_SMOKE=1 ... for a short check (60 / 40 steps, one seed)
"""
import collections
import json
import os
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import conflict_loop as CL
from experiments import durable as DUR
from experiments import dual_route_truth as DR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from experiments import taught_truth as TT
from src import everyday_truth as T
from src import everyday_truth_v2 as T2

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'conflict-search-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'conflict-search-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 60 if SMOKE else 3000            # phase 1
P2_STEPS = 40 if SMOKE else 1000         # phase 2
ARB_STEPS = 50 if SMOKE else 500
FIT_STEPS = 50 if SMOKE else 500
LOG_EVERY = 20 if SMOKE else 250
P2_LOG_EVERY = 10 if SMOKE else 100
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('one-pass', 'soft-loop', 'bt-forward', 'bt-bidir', 'bt-forward-notrace')
SEARCH_ARMS = ('bt-forward', 'bt-bidir', 'bt-forward-notrace')
TAU, H = 0.5, 64
B, K_FIRST, K_DEEP, D_MAX = 24, 3, 2, 5
T_MAX, PRIOR_P, KL_W = 3, 0.5, 0.01      # soft-loop
WITHHELD_RELS = CL.CHAIN_RELS
WITHHELD_NO_SUPPORT = True               # pre-registered fallback (contract), taken 2026-11-06: without it the trigger fired on 0.13 of withheld rows
UNFOLD = {'grandparent-of': ('parent-of',), 'ancestor-of': ('parent-of',), 'kind-of': ('is-a', 'subkind-of'),
          'bigger-than': ('just-bigger-than',), 'smaller-than': ('just-bigger-than',), 'named-can': ('names', 'can'),
          'named-has': ('names', 'has'), 'named-kind': ('names', 'is-a', 'subkind-of'), 'same-colour-as': ('colour-of',)}
D, RD = DR.D, DR.RD
CONF = 0.25                              # intuition "confident": |sigma(u) - 0.5| > CONF


class Model(nn.Module):
    """Dual-route (separate codes) plus the search controller, the soft-loop controller and the arbiter."""

    def __init__(self, n_ent, n_rel, g):
        super().__init__()
        self.dual = DR.Dual('dual-separate', n_ent, n_rel, g)
        self.n_rel = n_rel
        # search controller
        self.init = nn.Linear(2 * D + RD + 1, H)
        self.gru = nn.GRUCell(2 * D + RD + 3, H)
        self.w_rel, self.w_dir, self.q = nn.Linear(2 * H, n_rel), nn.Linear(2 * H, 2), nn.Linear(2 * H, D)
        self.q_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.w_e = nn.Linear(2 * H + 1, 1)
        # soft-loop controller
        self.init_s = nn.Linear(2 * D + RD, H)
        self.gru_s = nn.GRUCell(D + RD, H)
        self.w_rel_s, self.w_ans_s = nn.Linear(H, n_rel), nn.Linear(H, n_rel)
        self.w_halt_s = nn.Linear(H + 1, 1)
        self.att_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.arb = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 3))
        R1._reinit(self, g)

    def logic_codes(self, ca, rc, cb):
        d = self.dual
        return torch.sigmoid(d.S_L(torch.cat([d.G_L(torch.cat([ca, cb, rc], 1)), rc], 1)))

    def intuition_codes(self, ca, rc, cb):
        d = self.dual
        return d.S_I(torch.cat([d.G_I(torch.cat([ca, cb, rc], 1)), rc], 1))[:, 0]

    def controller_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(('dual.', 'arb.'))]


def straight_through(logits, table, sample=False, mask=None):
    """Returns (chosen index, code equal to table[idx] in value with the softmax's gradient, probabilities).
    sample: draw the index from the softmax (training exploration) instead of the argmax; mask: allowed indices."""
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(logits, 1)
    idx = torch.multinomial(probs.detach(), 1)[:, 0] if sample else probs.argmax(1)
    soft = probs @ table
    return idx, table[idx] + soft - soft.detach(), probs


@torch.no_grad()
def edge_table(model):
    """The frozen logic reader's beliefs about base edges: pool0[x, rel, e] = s+(x rel e) > .5, cached per model.
    pool1 (reversed hops) is pool0 transposed over (x, e). Exact, since the readers do not change in phase 2."""
    key = ('edges', id(model.dual))
    if getattr(model, '_edges', None) is not None and model._edges[0] == key:
        return model._edges[1]
    d = model.dual
    E_L, R_L = d.E_L.detach(), d.R_L.detach()
    ne, nr = len(E_L), len(R_L)
    pool0 = torch.zeros(ne, nr, ne, dtype=torch.bool)
    er = E_L.repeat(nr, 1)
    rr = R_L.repeat_interleave(ne, 0)
    for x in range(ne):
        sc = model.logic_codes(E_L[x].expand(nr * ne, -1), rr, er)[:, 0].view(nr, ne)
        pool0[x] = sc > .5
    pool1 = pool0.permute(2, 1, 0).contiguous()
    avail = pool0.any(2) | pool1.any(2)
    model._edges = (key, (pool0, pool1, avail))
    return model._edges[1]


@torch.no_grad()
def live_edges(model, code_L):
    """pool0, pool1 (n_rel, n_ent) and avail (n_rel) for a start node given as a code (the dax)."""
    d = model.dual
    E_L, R_L = d.E_L.detach(), d.R_L.detach()
    ne, nr = len(E_L), len(R_L)
    er, rr = E_L.repeat(nr, 1), R_L.repeat_interleave(ne, 0)
    c = code_L.expand(nr * ne, -1)
    p0 = (model.logic_codes(c, rr, er)[:, 0] > .5).view(nr, ne)
    p1 = (model.logic_codes(er, rr, c)[:, 0] > .5).view(nr, ne)
    return p0, p1, p0.any(1) | p1.any(1)


def disagreement(sig_u, s):
    return torch.abs(sig_u - s[:, 0]) + torch.abs((1 - sig_u) - s[:, 1])


# ---------------------------------------------------------------- the search

class Row:
    __slots__ = ('node', 'depth', 'path', 'frames', 'frame', 'cand', 'status', 'checks', 'tries', 'fails', 'sidepaths')

    def __init__(self, a, b):
        self.node = [a, b]          # frontier per side (-1 = the start code given separately)
        self.depth = [0, 0]
        self.path = [[], []]        # per side: list of hops (frame, m, s_plus)
        self.frames = []            # every choice point visited (for the recall read)
        self.frame = None           # the frame whose candidate is being checked
        self.cand = -1
        self.status = None          # 'found' | 'exhausted' | None
        self.checks = 0
        self.tries = 0
        self.fails = 0


class Frame:
    __slots__ = ('side', 'node', 'depth', 'tried', 'left', 'ridx', 'rc', 'dprob', 'dir', 'tries', 'parent', 'confirmed', 'pool')

    def __init__(self, side, node, depth, k, parent):
        self.side, self.node, self.depth, self.left, self.parent = side, node, depth, k, parent
        self.tried = []
        self.confirmed = []
        self.pool = []              # reader-scored candidate pool size at each visit
        self.tries = []             # (logits row over entities, candidate) for the ranking loss / recall


def side_of(row, mode):
    return (row.depth[0] + row.depth[1]) % 2 if mode == 'bidir' else 0


def search(model, a, r, b, mode, trace, start_codes=None, train=False, rel_oracle=None):
    """Depth-first walk with backtracking for every row (a, r, b). Returns a dict of per-row results and loss terms.
    mode: 'forward' | 'bidir'; trace: whether the GRU trace and exclusion of tried candidates are used.
    start_codes: optional (code_I, code_L) tensors (n, D) for a start node that is not an entity (the dax)."""
    d = model.dual
    E_I, E_L, R_I, R_L = d.E_I.detach(), d.E_L.detach(), d.R_I.detach(), d.R_L.detach()
    E_norm = F.normalize(E_L, dim=1)
    n = len(a)
    a_l, r_l, b_l = a.tolist(), r.tolist(), b.tolist()
    rows = [Row(-1 if start_codes is not None else a_l[i], b_l[i]) for i in range(n)]
    rq = R_L[r]
    pool0, pool1, avail = edge_table(model)
    live = {i: live_edges(model, start_codes[1][i:i + 1]) for i in range(n)} if start_codes is not None else {}
    u_state = torch.zeros(n, H)
    need = list(range(n))           # rows that need a proposal this iteration
    rank_terms = []                 # (logits over tried-plus-positive, position of positive)
    hop_sup = [[] for _ in range(n)]

    def code(table, i, node):
        if node == -1:
            return start_codes[0 if table is E_I else 1][i]
        return table[node]

    def codes(table, idx, nodes):
        if not idx:
            return torch.zeros(0, D)
        if start_codes is None:
            return table[torch.tensor(nodes)]
        return torch.stack([code(table, i, nd) for i, nd in zip(idx, nodes)])

    for it in range(B):
        # 1. proposals for rows at a choice point
        if need:
            idx = need
            sides = [side_of(rows[i], mode) for i in idx]
            nd = [rows[i].node[s] for i, s in zip(idx, sides)]
            other = [rows[i].node[1 - s] for i, s in zip(idx, sides)]
            dep = torch.tensor([(rows[i].depth[0] + rows[i].depth[1]) / D_MAX for i in idx])[:, None]
            h = torch.tanh(model.init(torch.cat([codes(E_L, idx, nd), codes(E_L, idx, other), rq[idx], dep], 1)))
            u = u_state[idx] if trace else torch.zeros(len(idx), H)
            z = torch.cat([h, u], 1)
            # action space (amendment, smoke stage): the frozen reader's edges at the node. Relations are restricted to
            # those under which the reader relates the node to some entity; the candidate pool is the reader's entities
            # for the chosen relation and direction; the proposer Q orders the pool. Sampled in training, argmax at test.
            rel_mask = torch.stack([avail[x] if x >= 0 else live[i][2] for i, x in zip(idx, nd)])
            if rel_oracle is not None:      # diagnostic only: restrict to the question's base relations
                rel_mask = rel_mask & rel_oracle[r[idx]]
            rel_logits = model.w_rel(z).masked_fill(~rel_mask, -1e9)
            ridx, rc, _ = straight_through(rel_logits, R_L, sample=train)
            dlog = model.w_dir(z)
            dprob = torch.softmax(dlog, 1)
            dsel = torch.multinomial(dprob.detach(), 1)[:, 0] if train else dprob.argmax(1)
            logits = model.q_scale.exp() * F.normalize(model.q(z), dim=1) @ E_norm.T
            with torch.no_grad():
                pools = []
                for k, (i, x) in enumerate(zip(idx, nd)):
                    r_ = int(ridx[k])
                    p0, p1 = (pool0[x, r_], pool1[x, r_]) if x >= 0 else (live[i][0][r_], live[i][1][r_])
                    rv = (sides[k] == 1) ^ (int(dsel[k]) == 1)
                    pk = p1 if rv else p0
                    if not pk.any() and (p0 if rv else p1).any():   # only the other direction has edges: flip
                        dsel[k] = 1 - dsel[k]
                        pk = p0 if rv else p1
                    pools.append(pk)
                pool = torch.stack(pools)
                lm = torch.where(pool, logits.detach(), torch.full_like(logits, -1e9))
                pool_n = pool.sum(1).tolist()
            for k, i in enumerate(idx):
                row = rows[i]
                fr = row.frame
                if fr is None or fr.node != nd[k] or fr.side != sides[k] or fr.depth != row.depth[sides[k]]:
                    kk = K_FIRST if row.depth[0] + row.depth[1] == 0 else K_DEEP
                    fr = Frame(sides[k], nd[k], row.depth[sides[k]], kk, row.path[sides[k]][-1][0] if row.path[sides[k]] else None)
                    row.frames.append(fr)
                    row.frame = fr
                fr.pool.append(pool_n[k])
                sd_ = sides[k]          # no self-loops or cycles on this side's own path; the other side stays proposable
                for x in [row.node[sd_]] + [hp[1] for hp in row.path[sd_]] + ([a_l[i]] if sd_ == 0 and start_codes is None else []):
                    if x >= 0:
                        lm[k, x] = -1e9
                if trace:
                    for x in fr.tried:
                        lm[k, x] = -1e9
                if train and (lm[k] > -1e8).sum() > 1:
                    m = int(torch.multinomial(torch.softmax(lm[k], 0), 1))
                else:
                    m = int(lm[k].argmax())
                fr.left -= 1
                fr.ridx, fr.rc, fr.dprob, fr.dir = int(ridx[k]), rc[k], dprob[k], int(dsel[k])
                fr.tries.append((logits[k], m, rel_logits[k], int(ridx[k]), dlog[k], int(dsel[k])))
                row.cand = m
                row.tries += 1
        need = []
        # 2. one check for every row with a pending candidate
        act = [i for i in range(n) if rows[i].status is None and rows[i].cand >= 0]
        if not act:
            break
        frs = [rows[i].frame for i in act]
        sides = [fr.side for fr in frs]
        xs = [rows[i].node[s] for i, s in zip(act, sides)]
        ms = [rows[i].cand for i in act]
        rc = torch.stack([fr.rc for fr in frs])
        dprob = torch.stack([fr.dprob for fr in frs])
        dsel = torch.tensor([fr.dir for fr in frs])
        ridx = torch.tensor([fr.ridx for fr in frs])
        cx_L, cm_L = codes(E_L, act, xs), E_L[ms]
        cx_I, cm_I = codes(E_I, act, xs), E_I[ms]
        # side 1 (from b) proposes an entity related to the frontier: ask(m, rho, node); dir flips the order again
        rev = (torch.tensor(sides) == 1) ^ (dsel == 1)
        s_f = model.logic_codes(cx_L, rc, cm_L)
        s_r = model.logic_codes(cm_L, rc, cx_L)
        s_sel = torch.where(rev[:, None], s_r, s_f)
        s_soft = dprob[:, 1:2] * torch.where(torch.tensor(sides)[:, None] == 1, s_f, s_r) + \
            dprob[:, 0:1] * torch.where(torch.tensor(sides)[:, None] == 1, s_r, s_f)
        s = s_sel + s_soft - s_soft.detach()
        with torch.no_grad():
            ca_I = torch.where(rev[:, None], cm_I, cx_I)
            cb_I = torch.where(rev[:, None], cx_I, cm_I)
            u = model.intuition_codes(ca_I, R_I[ridx], cb_I)
            sig = torch.sigmoid(u)
            dis = disagreement(sig, s.detach())
            conf = (dis <= TAU) & (s[:, 0] > .5)
        # 3. outcomes
        outcome = torch.zeros(len(act), 3)
        for k, i in enumerate(act):
            row, fr, m, sd = rows[i], frs[k], ms[k], sides[k]
            row.checks += 1
            row.cand = -1
            if bool(conf[k]):
                other = 1 - sd
                start_other = a_l[i] if (other == 0 and start_codes is None) else (b_l[i] if other == 1 else -2)
                meet = m == row.node[other] or m == start_other or any(hp[1] == m for hp in row.path[other])
                fr.confirmed.append(m)
                if meet:
                    row.path[sd].append((fr, m, s[k, 0]))
                    row.status = 'found'
                    hop_sup[i] = [hp[2] for hp in row.path[0] + row.path[1]]
                    outcome[k, 0] = 1
                    if train:
                        for hp in row.path[0] + row.path[1]:
                            f2, pos = hp[0], hp[1]
                            lg, _, rl, ri, dl, di = f2.tries[-1]
                            neg = [x for x in f2.tried if x != pos]
                            cand = torch.tensor([pos] + neg)
                            rank_terms.append(F.cross_entropy(lg[cand][None], torch.zeros(1, dtype=torch.long)))
                            # amendment: the relation and direction on a found path are positives too
                            rank_terms.append(F.cross_entropy(rl[None], torch.tensor([ri])))
                            rank_terms.append(F.cross_entropy(dl[None], torch.tensor([di])))
                    continue
                if row.depth[0] + row.depth[1] + 1 < D_MAX:
                    fr.confirmed.append(m)
                    row.path[sd].append((fr, m, s[k, 0]))
                    row.node[sd] = m
                    row.depth[sd] += 1
                    row.frame = None
                    outcome[k, 0] = 1
                    need.append(i)
                    continue
                fr.confirmed.append(m)
                outcome[k, 2] = 1       # dead end at the depth limit
            else:
                outcome[k, 1] = 1
            row.fails += 1
            # failure of candidate m at frame fr: retry there or backtrack
            fr.tried.append(m)
            while True:
                if fr.left > 0 and row.checks < B:
                    row.frame = fr
                    row.node[fr.side], row.depth[fr.side] = fr.node, fr.depth
                    row.path[fr.side] = row.path[fr.side][:fr.depth]
                    need.append(i)
                    break
                # pop to the parent choice point
                par = fr.parent
                if par is None:
                    row.status = 'exhausted'
                    break
                par.tried.append(fr.node)
                fr = par
        # 4. trace update
        if trace:
            inp = torch.cat([cx_L, rc.detach(), cm_L, outcome], 1)
            new = model.gru(inp, u_state[act])
            u_new = u_state.clone()
            u_new[act] = new
            u_state = u_new
        need = [i for i in need if rows[i].status is None and rows[i].checks < B]
        for i in range(n):
            if rows[i].status is None and rows[i].checks >= B and rows[i].cand < 0:
                rows[i].status = 'exhausted'
    for row in rows:
        if row.status is None:
            row.status = 'exhausted'
    # give-up scalar from the final state
    sides = [side_of(rw, mode) for rw in rows]
    nd = [rw.node[s] for rw, s in zip(rows, sides)]
    other = [rw.node[1 - s] for rw, s in zip(rows, sides)]
    dep = torch.tensor([(rw.depth[0] + rw.depth[1]) / D_MAX for rw in rows])[:, None]
    h = torch.tanh(model.init(torch.cat([codes(E_L, list(range(n)), nd), codes(E_L, list(range(n)), other), rq, dep], 1)))
    z = torch.cat([h, u_state if trace else torch.zeros(n, H)], 1)
    e = torch.sigmoid(model.w_e(torch.cat([z, torch.tensor([rw.checks / B for rw in rows])[:, None]], 1)))[:, 0]
    found = torch.tensor([rw.status == 'found' for rw in rows])
    p = torch.stack([torch.stack(hs).prod() if hs else torch.zeros(()) for hs in hop_sup])
    s_out = torch.where(found[:, None], torch.stack([p, 1 - p], 1), torch.stack([torch.zeros(n), e], 1))
    return {'s': s_out, 'found': found, 'e': e, 'p': p, 'rows': rows, 'rank_terms': rank_terms,
            'checks': np.array([rw.checks for rw in rows]), 'fails': np.array([rw.fails for rw in rows]),
            'path_len': np.array([len(rw.path[0]) + len(rw.path[1]) for rw in rows])}


def path_facts(row, a, b, rels):
    """The found path as world facts (x, rel, y) meaning 'x rel y', and the relation program with direction marks."""
    facts, prog = [], []
    for sd in (0, 1):
        prev = a if sd == 0 else b
        for fr, m, _ in row.path[sd]:
            rel = rels[fr.ridx]
            rev = (sd == 1) ^ (fr.dir == 1)
            facts.append((m, rel, prev) if rev else (prev, rel, m))
            prog.append(('<' if fr.dir == 1 else '>') + rel + ('|b' if sd == 1 else ''))
            prev = m
    return facts, prog


# ---------------------------------------------------------------- soft loop (the cheaper alternative)

def soft_loop(model, a, r, b, sig_u, ca=None):
    d = model.dual
    E_L, R_L = d.E_L.detach(), d.R_L.detach()
    ca = E_L[a] if ca is None else ca
    cb, rc = E_L[b], R_L[r]
    n = len(r)
    s0 = model.logic_codes(ca, rc, cb)
    state = torch.tanh(model.init_s(torch.cat([ca, cb, rc], 1)))
    m, sup, lams, ms = ca, [s0], [], []
    for t in range(1, T_MAX + 1):
        _, rho, _ = straight_through(model.w_rel_s(state), R_L)
        sc = model.logic_codes(m.repeat_interleave(len(E_L), 0), rho.repeat_interleave(len(E_L), 0),
                               E_L.repeat(n, 1))[:, 0].view(n, len(E_L))
        att = torch.softmax(model.att_scale.exp() * torch.logit(sc.clamp(1e-4, 1 - 1e-4)), 1)
        m = att @ E_L
        ms.append(m)
        _, rho_a, _ = straight_through(model.w_ans_s(state), R_L)
        s = model.logic_codes(m, rho_a, cb)
        sup.append(s)
        dis = (torch.abs(sig_u - s[:, 0]) + torch.abs((1 - sig_u) - s[:, 1]))[:, None]
        state = model.gru_s(torch.cat([m, rho], 1), state)
        if t < T_MAX:
            lams.append(torch.sigmoid(model.w_halt_s(torch.cat([state, dis], 1)))[:, 0])
    return sup, lams, ms


# ---------------------------------------------------------------- phase 1: readers

def withheld_mask(st):
    return np.isin(np.array(st.relations)[st.r], WITHHELD_RELS)


def phase1(st, seed, g, train_rows):
    model = Model(len(st.entities), len(st.relations), g)
    d = model.dual
    opt = torch.optim.Adam(d.parameters(), lr=TT.LR, weight_decay=TT.WD)
    w = torch.as_tensor(TT.weights(st, train_rows))
    vnames = np.array(T.VALUES)
    wh = withheld_mask(st)
    wrel = torch.tensor([i for i, x in enumerate(st.relations) if x in WITHHELD_RELS])
    r_init = d.R_L_own.detach()[wrel].clone()
    sup = np.flatnonzero(train_rows & np.isin(st.kind, ('individual', 'anon', 'anon-isa')) & (~wh | WITHHELD_NO_SUPPORT))
    sup_t = torch.tensor([DR.SUPPORT[vnames[v]] for v in st.v[sup]])
    if WITHHELD_NO_SUPPORT:
        sup_t[torch.as_tensor(wh[sup])] = 0.
    kin = np.flatnonzero(train_rows & (st.kind == 'kind'))
    qidx = torch.tensor([st.index(q) for q in DR.QUANT])
    kin_t = torch.tensor([DR.QUANT.index(vnames[v]) for v in st.v[kin]])
    fil = np.flatnonzero(train_rows & (st.kind == 'fill') & ~wh)
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
    fa, fr_, fb = A(fil)
    ia, ir, ib = A(ii)
    log, t0 = [], time.perf_counter()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        li = (F.binary_cross_entropy_with_logits(d.intuition(ia, ir, ib), iy, reduction='none') * iw).sum()
        ls = (F.binary_cross_entropy(d.logic(sa, sr, sb), sup_t, reduction='none').sum(1) * w[sup]).sum()
        lk = (F.cross_entropy(d.code_logits(d.code_out(ka, kr, kb, torch.zeros(len(kin), dtype=torch.long)), d.E_L[qidx]),
                              kin_t, reduction='none') * w[kin]).sum()
        lp = torch.log_softmax(d.code_logits(d.code_out(fa, fr_, fb, torch.ones(len(fil), dtype=torch.long)), d.E_L), 1)
        lf = (-torch.logsumexp(lp.masked_fill(~fmask, -1e9), 1) * w[fil]).sum()
        lc = (F.binary_cross_entropy(d.connective(con_p1, con_p2, con_r), con_t, reduction='none').sum(1) * w[con]).sum()
        loss = li + ls + lk + lf + lc
        loss.backward()
        opt.step()
        with torch.no_grad():
            d.R_L_own[wrel] = r_init          # never-trained relation codes stay at their initial values
        if step % LOG_EVERY == 0 or step == STEPS:
            rec = {'step': step, 'loss': float(loss.detach()), 'wall': time.perf_counter() - t0}
            log.append(rec)
            print(f'[phase1 seed {seed}] step {step}/{STEPS} loss {rec["loss"]:.4f} ({rec["wall"]:.0f} s)', flush=True)
    return model, log


@torch.no_grad()
def rl_check(model, st, train_rows):
    """Pre-registered smoke check: what the frozen one-pass returns on untrained relation codes, and the trigger."""
    d = model.dual
    m = train_rows & (st.kind == 'individual') & withheld_mask(st)
    idx = np.flatnonzero(m)
    a, r, b = (torch.as_tensor(x[idx]) for x in (st.a, st.r, np.maximum(st.b, 0)))
    s0 = d.logic(a, r, b)
    sig = torch.sigmoid(d.intuition(a, r, b))
    d0 = disagreement(sig, s0)
    conf = (sig - .5).abs() > CONF
    states = collections.Counter(DR.logic_state(s0.numpy()).tolist())
    return {'rows': int(len(idx)), 'mean_s_plus': float(s0[:, 0].mean()), 'mean_s_minus': float(s0[:, 1].mean()),
            'states': {k: v / len(idx) for k, v in states.items()}, 'trigger_rate': float((d0 > TAU).float().mean()),
            'intuition_confident': float(conf.float().mean()),
            'trigger_rate_confident': float((d0[conf] > TAU).float().mean()) if conf.any() else None,
            'withheld_no_support': WITHHELD_NO_SUPPORT}


# ---------------------------------------------------------------- phase 2: controllers

def p2_rows(st, train_rows):
    m = train_rows & (st.kind == 'individual') & withheld_mask(st) & (st.world_y >= 0)
    return np.flatnonzero(m & (st.world_y == 1)), np.flatnonzero(m & (st.world_y == 0))


def phase2(model, st, arm, seed, train_rows):
    for p in model.dual.parameters():
        p.requires_grad_(False)
    if arm == 'one-pass':
        return []
    opt = torch.optim.Adam(model.controller_params(), lr=TT.LR)
    tr, fl = p2_rows(st, train_rows)
    rng = np.random.default_rng([seed, 7])
    mode = 'bidir' if arm == 'bt-bidir' else 'forward'
    trace = arm != 'bt-forward-notrace'
    log, t0 = [], time.perf_counter()
    for step in range(1, P2_STEPS + 1):
        idx = np.concatenate([tr, rng.choice(fl, min(len(tr), len(fl)), replace=False)])
        a, r, b = (torch.as_tensor(x[idx]) for x in (st.a, st.r, np.maximum(st.b, 0)))
        y = torch.as_tensor(st.world_y[idx].astype(np.float32))
        target = torch.stack([y, 1 - y], 1)
        opt.zero_grad()
        if arm == 'soft-loop':
            with torch.no_grad():
                sig = torch.sigmoid(model.dual.intuition(a, r, b))
            sup, lams, _ = soft_loop(model, a, r, b, sig)
            ph, stp = CL.halting(lams, len(idx), 1)
            s = CL.mix(sup, ph, stp)
            loss = F.binary_cross_entropy(s, target, reduction='none').sum(1).mean() + KL_W * CL.kl(ph).mean()
            rec = {'step': step, 'loss': float(loss.detach()), 'wall': time.perf_counter() - t0,
                   'expected_steps': float((ph * torch.tensor(stp, dtype=torch.float)).sum(1).mean())}
        else:
            o = search(model, a, r, b, mode, trace, train=True)
            fm = o['found']
            bce_found = F.binary_cross_entropy(o['s'].clamp(1e-6, 1 - 1e-6), target, reduction='none').sum(1)
            bce_exh = F.binary_cross_entropy(o['e'].clamp(1e-6, 1 - 1e-6), 1 - y, reduction='none')
            outcome = torch.where(fm, bce_found, bce_exh).mean()
            rank = torch.stack(o['rank_terms']).mean() if o['rank_terms'] else torch.zeros(())
            loss = outcome + rank
            fnd = o['found'].numpy()
            rec = {'step': step, 'loss': float(loss.detach()), 'outcome': float(outcome), 'rank': float(rank),
                   'found_true': float(fnd[:len(tr)].mean()), 'found_false': float(fnd[len(tr):].mean()),
                   'checks_true': float(o['checks'][:len(tr)].mean()), 'wall': time.perf_counter() - t0}
        loss.backward()
        opt.step()
        if step % P2_LOG_EVERY == 0 or step == P2_STEPS:
            log.append(rec)
            print(f'[{arm} seed {seed}] ' + ' '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}'
                                                    for k, v in rec.items()), flush=True)
    return log


# ---------------------------------------------------------------- answering

def answer(model, st, rows, arm, items=None, codes=None, search_all=False):
    """Both routes on the rows; the search (or soft loop) on conflicts, or on every row if search_all."""
    d = model.dual
    if items is None:
        idx = np.flatnonzero(rows)
        a, r, b = (torch.as_tensor(x[idx]) for x in (st.a, st.r, np.maximum(st.b, 0)))
    else:
        a, r, b = items
    n = len(r)
    with torch.no_grad():
        ca_I, ca_L = (None, None) if codes is None else (codes[0].expand(n, -1), codes[1].expand(n, -1))
        u = d.intuition(a, r, b, ca=ca_I)
        sig = torch.sigmoid(u)
        s0 = d.logic(a, r, b, ca=ca_L)
        d0 = disagreement(sig, s0)
    out = {'u': u.numpy(), 's0': s0.numpy(), 'd0': d0.numpy(), 'found': np.zeros(n, bool), 'exhausted': np.zeros(n, bool),
           'checks': np.zeros(n), 'fails': np.zeros(n), 'path_len': np.zeros(n), 'e': np.zeros(n), 'steps': np.zeros(n),
           'rows': [None] * n}
    trig = np.ones(n, bool) if search_all else (d0 > TAU).numpy()
    if arm == 'one-pass':
        trig = np.zeros(n, bool)
    s_fin = s0.clone()
    if trig.any():
        ti = np.flatnonzero(trig)
        tt = torch.as_tensor(ti)
        with torch.no_grad():
            if arm == 'soft-loop':
                caL = None if codes is None else ca_L[tt]
                sup, lams, _ = soft_loop(model, a[tt], r[tt], b[tt], sig[tt], ca=caL)
                ph, stp = CL.halting(lams, len(ti), 0)
                s_fin[tt] = CL.mix(sup, ph, stp)
                out['steps'][ti] = (ph * torch.tensor(stp, dtype=torch.float)).sum(1).numpy()
            else:
                mode = 'bidir' if arm == 'bt-bidir' else 'forward'
                sc = None if codes is None else (ca_I[tt], ca_L[tt])
                o = search(model, a[tt], r[tt], b[tt], mode, arm != 'bt-forward-notrace', start_codes=sc)
                s_fin[tt] = o['s']
                out['found'][ti] = o['found'].numpy()
                out['exhausted'][ti] = ~o['found'].numpy()
                out['checks'][ti], out['fails'][ti], out['path_len'][ti] = o['checks'], o['fails'], o['path_len']
                out['e'][ti] = o['e'].numpy()
                for k, i in enumerate(ti):
                    out['rows'][i] = o['rows'][k]
    out['s'] = s_fin.numpy()
    out['d'] = disagreement(sig, s_fin).numpy()
    out['trig'] = trig
    return out


def arb_features(o):
    sig = 1 / (1 + np.exp(-o['u']))
    X = np.stack([sig, o['s'][:, 0], o['s'][:, 1], o['d0'], o['found'].astype(float), o['exhausted'].astype(float),
                  o['checks'] / B, o['path_len'] / D_MAX, o['e']], 1)
    return torch.tensor(X, dtype=torch.float32)


def train_arbiter(model, st, val_rows, arm, seed):
    o = answer(model, st, val_rows, arm)
    X = arb_features(o)
    vn = np.array(T.VALUES)[st.v[val_rows]]
    y = torch.tensor([('T', 'F', 'U').index(v) for v in vn])
    counts = torch.bincount(y, minlength=3).float().clamp(min=1)
    wcls = (1 / counts) / (1 / counts).sum() * 3
    torch.manual_seed(seed + 5)
    for p in model.arb.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.arb.parameters(), lr=1e-2)
    for _ in range(ARB_STEPS):
        opt.zero_grad()
        F.cross_entropy(model.arb(X), y, weight=wcls).backward()
        opt.step()
    return {'val_n': int(len(y)), 'val_counts': {k: int(c) for k, c in zip('TFU', torch.bincount(y, minlength=3))},
            'val_trigger_rate': float(o['trig'].mean())}


@torch.no_grad()
def arb_answer(model, o):
    return np.array(['T', 'F', 'U'])[model.arb(arb_features(o)).argmax(1).numpy()]


def outputs(model, o):
    iv = np.where(o['u'] > 0, 'T', 'F')
    return {'intuition': iv, 'logic-step0': DR.logic_state(o['s0']), 'logic-final': DR.logic_state(o['s']),
            'arbiter': arb_answer(model, o)}


# ---------------------------------------------------------------- world knowledge for the diagnostic reads

class World:
    def __init__(self, st):
        rels = np.array(st.relations)
        ind = (st.kind == 'individual') & (st.world_y == 1)
        self.true = set(zip(st.a[ind].tolist(), rels[st.r[ind]].tolist(), st.b[ind].tolist()))
        self.adj = collections.defaultdict(set)      # (rel, x) -> {y}: x rel y or y rel x
        for a, rel, b in self.true:
            self.adj[(rel, a)].add(b)
            self.adj[(rel, b)].add(a)
        self.cache = {}

    def paths(self, a, rel, b):
        """Simple paths a -> b of length <= D_MAX over the base relations rel unfolds to, either direction."""
        key = (a, rel, b)
        if key in self.cache:
            return self.cache[key]
        base = UNFOLD.get(rel, ())
        out = []

        def dfs(x, path):
            if len(path) > D_MAX + 1:
                return
            if x == b and len(path) > 1:
                out.append(tuple(path))
                return
            for br in base:
                for y in self.adj.get((br, x), ()):
                    if y not in path:
                        dfs(y, path + [y])
        dfs(a, [a])
        nxt = [collections.defaultdict(set), collections.defaultdict(set)]
        for p in out:
            for j in range(len(p) - 1):
                nxt[0][p[j]].add(p[j + 1])
                nxt[1][p[j + 1]].add(p[j])
        self.cache[key] = (out, nxt)
        return self.cache[key]


def search_reads(o, st, idx, world, rels, names):
    """Faithfulness and search reads over rows idx (np array of statement indices) with results o."""
    y = st.world_y[idx]
    rec = {}
    srch = o['trig']
    if not srch.any():
        return rec
    tp, prog, cidx, recall, offpath, bwaste, pools = [], collections.Counter(), [], [], 0, [], []
    for k in np.flatnonzero(srch):
        row = o['rows'][k]
        if row is None:
            continue
        i = idx[k]
        a, rel, b = int(st.a[i]), rels[st.r[i]], int(st.b[i])
        if row.status == 'found':
            facts, pr = path_facts(row, a, b, rels)
            tp.append(all(f in world.true for f in facts))
            prog[' '.join(pr)] += 1
            cidx += [len(fr.tried) + 1 for fr, _, _ in row.path[0] + row.path[1]]
            if row.path[1]:
                onpath = {m for _, m, _ in row.path[1]}
                conf_b = [fr for fr in row.frames if fr.side == 1]
                cb = [m not in onpath for fr in conf_b for m in fr.confirmed]
                if cb:
                    bwaste.append(float(np.mean(cb)))
        pools += [x for fr in row.frames for x in fr.pool]
        if y[k] == 1 and a >= 0:
            _, nxt = world.paths(a, rel, b)
            for fr in row.frames:
                if fr.node not in nxt[fr.side]:
                    offpath += 1
                    continue
                proposed = {t[1] for t in fr.tries}
                recall.append(bool(proposed & nxt[fr.side][fr.node]))
    f = lambda v: float(np.mean(v)) if len(v) else None
    rec['search_rate'] = float(srch.mean())
    rec['found_rate_true'] = f(o['found'][srch & (y == 1)])
    rec['found_rate_false'] = f(o['found'][srch & (y == 0)])
    rec['exhausted_true'] = f(o['exhausted'][srch & (y == 1)])
    rec['exhausted_false'] = f(o['exhausted'][srch & (y == 0)])
    rec['checks_true'] = f(o['checks'][srch & (y == 1)])
    rec['checks_false'] = f(o['checks'][srch & (y == 0)])
    rec['backtracks_true'] = f(o['fails'][srch & (y == 1)])
    rec['path_len_found'] = f(o['path_len'][srch & o['found']])
    rec['true_path_rate'] = f(tp)
    rec['candidate_index_mean'] = f(cidx)
    rec['candidate_index_first'] = f([c == 1 for c in cidx])
    rec['proposer_recall'] = f(recall)
    rec['proposer_offpath_choice_points'] = offpath
    rec['backward_waste'] = f(bwaste)
    rec['pool_size_mean'] = f(pools)
    rec['pool_empty_rate'] = f([x == 0 for x in pools])
    rec['programs'] = [[p, c] for p, c in prog.most_common(3)]
    return rec


def by_sub(o, outs, y, L, key):
    return {s: DR.tf_bal(outs[key][L == s], y[L == s]) for s in np.unique(L)}


def reads(model, st, arm, train_rows):
    out = {}
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    rels, names = list(st.relations), np.array(st.entities)
    world = World(st)
    wh = withheld_mask(st)
    pe, pc = [], []
    for tst in DR.TF_TESTS:
        m = ind & (st.test == 'P1') & (st.sub == 'default') if tst == 'P1-default' else ind & (st.test == tst)
        idx = np.flatnonzero(m)
        o = answer(model, st, m, arm)
        y = st.world_y[m]
        outs = outputs(model, o)
        rec = {k: DR.tf_bal(v, y) for k, v in outs.items()}
        rec['intuition_auroc'] = R1.auroc(o['u'], y)
        rec['logic_final_auroc'] = R1.auroc(o['s'][:, 0] - o['s'][:, 1], y)
        rec['logic_step0_auroc'] = R1.auroc(o['s0'][:, 0] - o['s0'][:, 1], y)
        rec['trigger_rate'] = float(o['trig'].mean())
        rec['trigger_rate_true'] = float(o['trig'][y == 1].mean()) if (y == 1).any() else None
        rec['arbiter_u_rate'] = float((outs['arbiter'] == 'U').mean())
        if arm == 'soft-loop':
            rec['expected_steps_triggered'] = float(o['steps'][o['trig']].mean()) if o['trig'].any() else None
        if arm in SEARCH_ARMS:
            rec.update(search_reads(o, st, idx, world, rels, names))
        if wh[idx].any() and arm != 'one-pass':
            oa = answer(model, st, m, arm, search_all=True)
            rec['logic_search_all'] = DR.tf_bal(DR.logic_state(oa['s']), y)
            if arm in SEARCH_ARMS:
                rec['search_all'] = {k: v for k, v in search_reads(oa, st, idx, world, rels, names).items()
                                     if k in ('found_rate_true', 'found_rate_false', 'exhausted_true', 'true_path_rate',
                                              'proposer_recall', 'checks_true')}
        if tst.startswith('L1') or tst == 'L2-named':
            L = st.sub[m]
            rec['logic_final_by_length'] = by_sub(o, outs, y, L, 'logic-final')
            rec['arbiter_by_length'] = by_sub(o, outs, y, L, 'arbiter')
            rec['intuition_by_length'] = by_sub(o, outs, y, L, 'intuition')
            rec['n_by_length'] = {s: int((L == s).sum()) for s in np.unique(L)}
            if arm in SEARCH_ARMS:
                rec['checks_by_length_true'] = {s: float(o['checks'][(L == s) & (y == 1) & o['trig']].mean())
                                                for s in np.unique(L) if ((L == s) & (y == 1) & o['trig']).any()}
                rec['found_by_length_true'] = {s: float(o['found'][(L == s) & (y == 1) & o['trig']].mean())
                                               for s in np.unique(L) if ((L == s) & (y == 1) & o['trig']).any()}
                rec['exhausted_true_by_length'] = {s: float(o['exhausted'][(L == s) & (y == 1) & o['trig']].mean())
                                                   for s in np.unique(L) if ((L == s) & (y == 1) & o['trig']).any()}
                rec['found_count_true_by_length'] = {s: [int(o['found'][(L == s) & (y == 1)].sum()), int(((L == s) & (y == 1)).sum())]
                                                     for s in np.unique(L)}
            if arm == 'soft-loop':
                rec['steps_by_length'] = {s: float(o['steps'][(L == s) & o['trig']].mean()) for s in np.unique(L)
                                          if ((L == s) & o['trig']).any()}
        out[tst] = rec
        pe += list(outs['intuition'] != np.where(y == 1, 'T', 'F'))
        pc += list(o['d0'])
    out['conflict-flags-intuition-errors'] = {'graded_auroc': R1.auroc(pc, pe), 'error_rate': float(np.mean(pe))}
    # pooled ancestor depth 4-5
    m = ind & (st.test == 'L1-anc') & np.isin(st.sub, ('4', '5'))
    o = answer(model, st, m, arm)
    out['L1-anc-depth45'] = {k: DR.tf_bal(v, st.world_y[m]) for k, v in outputs(model, o).items()}
    out['L1-anc-depth45']['n'] = int(m.sum())
    # withheld training facts (logic never saw them one-pass)
    wm = train_rows & ind & wh
    idx = np.flatnonzero(wm)
    o = answer(model, st, wm, arm)
    outs = outputs(model, o)
    y = st.world_y[wm]
    out['withheld-fit'] = {k: DR.tf_bal(v, y) for k, v in outs.items()}
    out['withheld-fit']['trigger_rate'] = float(o['trig'].mean())
    out['withheld-fit']['trigger_rate_true'] = float(o['trig'][y == 1].mean())
    if arm in SEARCH_ARMS:
        out['withheld-fit'].update(search_reads(o, st, idx, world, rels, names))
    # per-hop base accuracy: both routes agree and confirm, on true and false base facts (train rows)
    base = sorted({x for v in UNFOLD.values() for x in v})
    bm = train_rows & ind & np.isin(np.array(rels)[st.r], base)
    o = answer(model, st, bm, 'one-pass')
    conf = (o['d0'] <= TAU) & (o['s0'][:, 0] > .5)
    yb = st.world_y[bm]
    out['per-hop'] = {'confirm_true': float(conf[yb == 1].mean()), 'confirm_false': float(conf[yb == 0].mean()),
                      'by_relation_true': {rel: float(conf[(yb == 1) & (np.array(rels)[st.r[bm]] == rel)].mean())
                                           for rel in base if ((yb == 1) & (np.array(rels)[st.r[bm]] == rel)).any()}}
    # unknown and S3b
    vn = np.array(T.VALUES)
    for tag, m in (('S4a', st.test == 'S4a'), ('S4b-P7', ind & (st.test == 'P7')),
                   ('control', ind & np.isin(st.test, TT.CONTROL_TESTS))):
        o = answer(model, st, m, arm)
        outs = outputs(model, o)
        out[tag + '-u-rate'] = {k: float((v == 'U').mean()) for k, v in outs.items() if k != 'intuition'}
        out[tag + '-trigger-rate'] = float(o['trig'].mean())
        if arm in SEARCH_ARMS and o['trig'].any():
            out[tag + '-u-rate']['arbiter_u_given_exhausted'] = float((outs['arbiter'][o['exhausted']] == 'U').mean()) \
                if o['exhausted'].any() else None
    m = st.test == 'S3b'
    o = answer(model, st, m, arm)
    outs = outputs(model, o)
    exp = vn[st.v[m]]
    out['S3b'] = {k: {q: float((v[st.sub[m] == q] == exp[st.sub[m] == q]).mean()) for q in np.unique(st.sub[m])}
                  for k, v in outs.items()}
    out['S3b-trigger-rate'] = float(o['trig'].mean())
    return out


def fast_map_dax(model, st, seed, arm):
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
        o = answer(model, st, None, arm, items=(torch.full((n,), -1), r, b), codes=(codes[0].detach(), codes[1].detach()))
        exp = np.array([x[2] for x in items])
        res[q] = {k: float((v == exp).mean()) for k, v in outputs(model, o).items()}
        res[q]['trigger_rate'] = float(o['trig'].mean())
    return res


# ---------------------------------------------------------------- tasks

def splits(st, seed):
    rng = np.random.default_rng(seed)
    trainable = st.test == ''
    ind_idx = np.flatnonzero(trainable & (st.kind == 'individual'))
    val = np.zeros(len(st.a), bool)
    val[rng.choice(ind_idx, int(round(TT.VAL_FRAC * len(ind_idx))), replace=False)] = True
    return trainable & ~val, val


def run_phase1(seed):
    st, _ = T2.build()
    TT.value_idx_cache = TT.value_idx(st)
    train_rows, _ = splits(st, seed)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    t0 = time.process_time()
    model, log = phase1(st, seed, g, train_rows)
    cpu = time.process_time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'phase1-seed{seed}.pt'
    torch.save(model.dual.state_dict(), ck)
    tr, fl = p2_rows(st, train_rows)
    res = {'seed': seed, 'log': log, 'rl_check': rl_check(model, st, train_rows), 'withheld_rows': {'T': int(len(tr)), 'F': int(len(fl))},
           'checkpoint': {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}}
    return res, {'optimizer_steps': STEPS, 'train_cpu_seconds': cpu, 'checkpoint_bytes': ck.stat().st_size}


def run_phase2(arm, seed):
    st, _ = T2.build()
    TT.value_idx_cache = TT.value_idx(st)
    train_rows, val = splits(st, seed)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Model(len(st.entities), len(st.relations), g)
    model.dual.load_state_dict(torch.load(RUNS / f'phase1-seed{seed}.pt'))
    torch.manual_seed(seed + 11)
    t0 = time.process_time()
    log = phase2(model, st, arm, seed, train_rows)
    p2_cpu = time.process_time() - t0
    arb_info = train_arbiter(model, st, val & (st.kind == 'individual'), arm, seed)
    t1 = time.process_time()
    res = {'tests': reads(model, st, arm, train_rows), 'dax': fast_map_dax(model, st, seed, arm), 'log': log,
           'arbiter_training': arb_info, 'arm': arm, 'seed': seed, 'params': R1.n_params(model),
           'controller_params': int(sum(p.numel() for p in model.controller_params())),
           'withheld_no_support': WITHHELD_NO_SUPPORT}
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'phase2_steps': P2_STEPS if arm != 'one-pass' else 0, 'arbiter_steps': ARB_STEPS,
                 'phase2_cpu_seconds': p2_cpu, 'reads_cpu_seconds': time.process_time() - t1,
                 'checkpoint_bytes': ck.stat().st_size}


def task(t):
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    if t[0] == 'phase1':
        res, rt = run_phase1(t[1])
        key = f'phase1/seed{t[1]}'
    else:
        res, rt = run_phase2(*t)
        key = f'{t[0]}/seed{t[1]}'
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return key, res, rt


def pool_map(tasks):
    workers = int(os.environ.get('EV_WORKERS', 6))
    with __import__('multiprocessing').get_context('fork').Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
        return pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1), workers


def main(phase1_only=False):
    t0 = time.perf_counter()
    res1, workers = pool_map([('phase1', s) for s in SEEDS])
    if phase1_only:
        for k, r, _ in res1:
            print(k, 'R_L check', r['rl_check'], 'withheld rows', r['withheld_rows'])
        return
    res2, _ = pool_map([(a, s) for a in ARMS for s in SEEDS])
    res = res1 + res2
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'workers': workers,
                     'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                     'tasks': {k: t for k, _, t in res}})
    V2.save_json(f'{REPORT}.json', {'status': 'conflict search v1' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                    'phase2_steps': P2_STEPS, 'B': B, 'k': [K_FIRST, K_DEEP], 'D_max': D_MAX, 'tau': TAU,
                                    'withheld_no_support': WITHHELD_NO_SUPPORT,
                                    'phase1': {k: r for k, r, _ in res1}, 'results': {k: r for k, r, _ in res2}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    rep = json.loads(open(f'{REPORT}.json').read())
    res = rep['results']
    by = {}
    for r in res.values():
        by.setdefault(r['arm'], []).append(r)

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if vals and all(isinstance(v, dict) for v in vals):
            keys = set().union(*[v.keys() for v in vals])
            return {k: agg([v.get(k) for v in vals]) for k in sorted(keys)}
        if vals and all(isinstance(v, list) for v in vals):
            return vals[0] if len(vals) == 1 else vals
        vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return round(float(np.mean(vals)), 3) if vals else None
    out = {a: {'tests': agg([r['tests'] for r in rs]), 'dax': agg([r['dax'] for r in rs]), 'params': rs[0]['params'],
               'controller_params': rs[0]['controller_params'],
               'final_log': agg([r['log'][-1] for r in rs if r['log']])} for a, rs in by.items()}
    out['phase1'] = {'rl_check': agg([r['rl_check'] for r in rep['phase1'].values()]),
                     'withheld_rows': agg([r['withheld_rows'] for r in rep['phase1'].values()])}
    V2.save_json(f'{REPORT}-summary.json', out)
    print('== phase 1 R_L check', out['phase1'])
    for a in ARMS:
        if a not in out:
            continue
        o = out[a]
        print('==', a, 'params', o['params'], 'controller', o['controller_params'], 'last log', o['final_log'])
        print('test'.ljust(15) + 'intuit step0  final  arbit | trig  found_T exh_T  truepath recall | aurocI aurocL0 aurocL | srchAll')
        for t in DR.TF_TESTS:
            v = o['tests'][t]
            f = lambda k: '  -  ' if v.get(k) is None else f'{v[k]:.2f}'
            print(t.ljust(15) + f"{f('intuition')}  {f('logic-step0')}  {f('logic-final')}  {f('arbiter')} | "
                  f"{f('trigger_rate')}  {f('found_rate_true')}  {f('exhausted_true')}  {f('true_path_rate')}  {f('proposer_recall')} | "
                  f"{f('intuition_auroc')}  {f('logic_step0_auroc')}  {f('logic_final_auroc')} | {f('logic_search_all')}")
        for k in ('L1-anc-depth45', 'withheld-fit', 'per-hop', 'conflict-flags-intuition-errors', 'S4a-u-rate',
                  'S4b-P7-u-rate', 'control-u-rate', 'S3b', 'S3b-trigger-rate'):
            print('  ', k, o['tests'].get(k))
        for t in ('L1-kind', 'L1-anc', 'L1-size', 'L2-named'):
            print('  ', t, {k: o['tests'][t].get(k) for k in ('logic_final_by_length', 'arbiter_by_length', 'checks_by_length_true',
                                                                'found_count_true_by_length', 'programs', 'steps_by_length')})
        print('   dax', o['dax'])


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'main'
    {'main': main, 'phase1': lambda: main(True), 'summary': summary}[cmd]()
