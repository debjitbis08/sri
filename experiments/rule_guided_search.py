"""Rule-guided search v1 (docs/rule-guided-search-v1.md): frozen readers confirm hops (both routes + applicability),
a committed beam walk over the confirmed graph, and a pairwise composition critic whose context is the told rules.
Arms: rules, rules-nogate, question. Evaluated on the confirmed graph and on the oracle graph.

Usage: EV_WORKERS=6 .venv/bin/python -m experiments.rule_guided_search        (rerun to resume)
       .venv/bin/python -m experiments.rule_guided_search summary
       EV_SMOKE=1 ... short check
"""
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import durable as DUR
from experiments import everyday_relations_v2 as V2
from experiments import rule_application as RA

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'rule-guided-search-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'rule-guided-search-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('rules', 'rules-nogate', 'question')
N_RULESETS = 30 if SMOKE else 300
N_SYN = 10 if SMOKE else 100
QUOTA = 10 if SMOKE else 100
N_CONJ = 5 if SMOKE else 50
N_SYN_EDIT = 30 if SMOKE else 300
STEPS = int(os.environ.get('EV_STEPS', 100)) if SMOKE else 1500
QPER = 20
W_BEAM, D_MAX = 16, 12
DFS_CAP = 20000
EXCLUDE = ('ancestor-of', 'grandparent-of', 'child-of', 'bigger-than', 'smaller-than', 'is-a')
SEED_FORMS = ('sub', 'inv', 'chain1', 'chain2', 'chain3', 'chain4')
REC = RA.REC
HID, EMB, SYM = 64, 32, 16
BUCKETS = ('1', '2', '3', '4', '5-6', '7-12')


def bucket(h):
    if h < 1:
        return None
    return str(h) if h <= 4 else ('5-6' if h <= 6 else '7-12')


# ---------------------------------------------------------------- rules on matrices

def orient(M, atom, p, q):
    _, (va, vb) = atom
    return M if (va, vb) == (p, q) else M.T


def split_atoms(rule):
    """(atom from head var a to middle, atom from middle to head var b, middle) for 2-atom path rules."""
    _, (ha, hb), body, _ = rule
    mid = [v for v in range(3) if v not in (ha, hb)][0]
    first = [x for x in body if ha in x[1]][0]
    second = [x for x in body if x is not first][0]
    return first, second, mid


def apply_rule_mat(rule, M):
    h, (ha, hb), body, form = rule
    if form == 'conj':
        return orient(M[body[0][0]], body[0], ha, hb) & orient(M[body[1][0]], body[1], ha, hb)
    if len(body) == 1:
        return orient(M[body[0][0]], body[0], ha, hb)
    a1, a2, mid = split_atoms(rule)
    return (orient(M[a1[0]], a1, ha, mid).astype(np.float32) @ orient(M[a2[0]], a2, mid, hb).astype(np.float32)) > 0


def closure_mat(base, rules, m):
    """base: list of (N, N) bool for local base relations 0..nb-1; derived nb..m-1. Returns list of matrices."""
    N = base[0].shape[0]
    M = list(base) + [np.zeros((N, N), bool) for _ in range(m - len(base))]
    for _ in range(64):
        changed = False
        for rule in rules:
            Hm = apply_rule_mat(rule, M) | M[rule[0]]
            if (Hm != M[rule[0]]).any():
                M[rule[0]] = Hm
                changed = True
        if not changed:
            break
    return M


def min_hops(base, rules, m, K=D_MAX):
    """Minimum hops of a valid path proof for every derived atom (0 = none within K). Path fragment only.
    Two-atom rules at length k use only lengths < k (each atom >= 1 hop), so only one-atom rules need a fixpoint."""
    N = base[0].shape[0]
    f32 = lambda x: x.astype(np.float32)
    D = [None] + [list(base) + [np.zeros((N, N), bool) for _ in range(m - len(base))] for _ in range(K)]
    two = [r for r in rules if r[3] != 'conj' and len(r[2]) == 2]
    one = [r for r in rules if r[3] != 'conj' and len(r[2]) == 1]
    for k in range(1, K + 1):
        if k > 1:
            for r in range(len(base), m):
                D[k][r] = D[k][r] | D[k - 1][r]
        for rule in two:
            h, (ha, hb), body, form = rule
            a1, a2, mid = split_atoms(rule)
            new = np.zeros((N, N), np.float32)
            for i in range(1, k):
                L, R = D[i][a1[0]], D[k - i][a2[0]]
                if L.any() and R.any():
                    new += f32(orient(L, a1, ha, mid)) @ f32(orient(R, a2, mid, hb))
            D[k][h] = D[k][h] | (new > 0)
        for _ in range(2 * len(one) + 2):
            changed = False
            for rule in one:
                h, (ha, hb), body, form = rule
                new = orient(D[k][body[0][0]], body[0], ha, hb)
                if (new & ~D[k][h]).any():
                    D[k][h] = D[k][h] | new
                    changed = True
            if not changed:
                break
    out = np.zeros((m, N, N), np.int16)
    for r in range(len(base), m):
        for k in range(K, 0, -1):
            out[r][D[k][r]] = k
    return out


def language(nb, rules, m, glob, cap_len=D_MAX, cap_n=3000):
    """Valid step sequences (tuples of (global relation, direction)) for each derived relation, up to cap_len."""
    rev = lambda s: tuple((g, 1 - d) for g, d in reversed(s))
    L = [{((glob[r], 0),)} for r in range(nb)] + [set() for _ in range(m - nb)]

    def ori(atom, p, q):
        r, (va, vb) = atom
        return L[r] if (va, vb) == (p, q) else {rev(s) for s in L[r]}
    for _ in range(cap_len + 2):
        changed = False
        for rule in rules:
            h, (ha, hb), body, form = rule
            if form == 'conj':
                continue
            if len(body) == 1:
                new = ori(body[0], ha, hb)
            else:
                a1, a2, mid = split_atoms(rule)
                A, B = ori(a1, ha, mid), ori(a2, mid, hb)
                new = {x + y for x in A for y in B if len(x) + len(y) <= cap_len}
            new = {s for s in new if len(s) <= cap_len} - L[h]
            if new and len(L[h]) < cap_n:
                L[h] |= set(list(new)[:cap_n - len(L[h])])
                changed = True
        if not changed:
            break
    return L


def walk_valid(program, rs):
    """Exact: does the asked relation hold between the walk's ends when the rules are closed over its edges only?"""
    k = len(program)
    n = k + 1
    loc = {g: i for i, g in enumerate(rs['glob'])}
    S0 = np.zeros((rs['m'], n, n), bool)
    for i, (g, d) in enumerate(program):
        if g not in loc:
            return False
        a, b = (i, i + 1) if d == 0 else (i + 1, i)
        S0[loc[g], a, b] = True
    S, _, _ = RA.closure(S0, rs['rules'], n)
    return bool(S[rs['ask'], 0, k])


# ---------------------------------------------------------------- readers and graphs

class Readers:
    def __init__(self, seed):
        from experiments import amortized_introduction as AI
        from experiments import relations_as_objects as RO
        self.W = W = RO.World(seed)
        self.model = AI.Model(W, torch.Generator().manual_seed(0))
        ck = ROOT / 'runs' / 'amortized-introduction-v1' / f'virtual-seed{seed}.pt'
        self.model.load_state_dict(torch.load(ck))
        self.ck_sha = nc.sha256(ck)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.net = self.model.net
        self.N = len(W.st.entities)
        trained = set(W.fr[W.fg != 2].tolist())
        ex = {W.R[x] for x in EXCLUDE}
        self.vocab = [r for r in range(len(W.real) + len(W.syn)) if r in trained and r not in ex and r in W.true]
        self.vidx = {g: i for i, g in enumerate(self.vocab)}
        self.S = 2 * len(self.vocab)
        self.kind, self.sub = W.R['kind-of'], W.R['subkind-of']

    @torch.no_grad()
    def judge(self, r):
        N = self.N
        a = torch.arange(N).repeat_interleave(N)
        b = torch.arange(N).repeat(N)
        rr = torch.full((N * N,), r)
        s, al = self.net.logic(a, rr, b)
        u = self.net.intuition(a, rr, b)
        sp, sm = s[:, 0].numpy(), s[:, 1].numpy()
        return ((sp > .5) & (sm <= .5)).reshape(N, N), (u > 0).numpy().reshape(N, N), \
            (torch.sigmoid(al) > .5).numpy().reshape(N, N), sp.reshape(N, N).astype(np.float16)

    def graphs(self):
        """Confirmed (gated, ungated), supports, oracle: dict global relation -> (N, N) bool."""
        g, ng, sup, orc = {}, {}, {}, {}
        eye = np.eye(self.N, dtype=bool)
        for r in self.vocab:
            lt, it, ap, sp = self.judge(r)
            ng[r] = lt & it & ~eye
            g[r] = ng[r] & ap
            sup[r] = sp
            M = np.zeros((self.N, self.N), bool)
            for a, b in self.W.true[r]:
                M[a, b] = True
            orc[r] = M & ~eye
        for G_ in (g, ng, orc):
            G_[self.kind] = RA.reduce_kind(G_[self.kind], G_[self.sub])
        return g, ng, sup, orc


def step_adj(graph, rd, sigma):
    V = len(rd.vocab)
    r = rd.vocab[sigma % V]
    return graph[r] if sigma < V else graph[r].T


def succ_lists(graph, rd):
    """succ[sigma][x] -> array of m with a confirmed hop sigma from x."""
    out = []
    for s in range(rd.S):
        A = step_adj(graph, rd, s)
        out.append([np.flatnonzero(A[x]) for x in range(rd.N)])
    return out


def hop_composition(graph, rd):
    W = rd.W
    c = {'true': 0, 'false': 0, 'unlabelled': 0}
    for r in rd.vocab:
        lab = W.lab.get(r, {})
        xs, ys = np.nonzero(graph[r])
        for a, b in zip(xs.tolist(), ys.tolist()):
            v = lab.get((a, b))
            c['unlabelled' if v is None else ('true' if v == 1 else 'false')] += 1
    return c


# ---------------------------------------------------------------- rule sets and questions

def unitv(rng, k):
    v = rng.standard_normal((k, SYM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def make_ruleset(rng, rd, orc, seed_forms=SEED_FORMS, force=None, min_true=10):
    W = rd.W
    rich = [r for r in rd.vocab if orc[r].sum() >= 5]
    for _ in range(500):
        nb, nd = int(rng.integers(2, 4)), int(rng.integers(1, 3))
        glob = [int(x) for x in rng.choice(rich, nb, replace=False)]
        rules = RA.make_rules(rng, nb, nd, list(seed_forms), force)
        m = nb + nd
        M = closure_mat([orc[g] for g in glob], rules, m)
        ask = m - 1
        T = M[ask] & ~np.eye(rd.N, dtype=bool)
        if T.sum() >= min_true and T.mean() < .3:
            return dict(glob=glob, nb=nb, m=m, rules=rules, ask=ask, sym=unitv(rng, m))
    raise RuntimeError('no rule set')


def labels(rs, graph, K=D_MAX):
    M = closure_mat([graph[g] for g in rs['glob']], rs['rules'], rs['m'])
    H = min_hops([graph[g] for g in rs['glob']], rs['rules'], rs['m'], K)
    return M[rs['ask']], H[rs['ask']]


def connected(graph, rd, hops=2):
    U = np.zeros((rd.N, rd.N), bool)
    for r in rd.vocab:
        U |= graph[r] | graph[r].T
    R = U.copy()
    for _ in range(hops - 1):
        R |= (R.astype(np.float32) @ U.astype(np.float32)) > 0
    return R


# ---------------------------------------------------------------- candidates (training)

def candidates(a, b, succ, pred_of_b, rd):
    """Distinct programs of 1-2 hops from a to b: int16 array (P, 2), second column -1 for one hop."""
    progs = set()
    mids = {}
    for s in range(rd.S):
        for m in succ[s][a]:
            if m == b:
                progs.add((s,))
            elif m != a:
                mids.setdefault(int(m), []).append(s)
    for s2, ms in pred_of_b.items():
        for m in ms:
            for s1 in mids.get(int(m), ()):
                progs.add((s1, s2))
    return np.array([p if len(p) == 2 else (p[0], -1) for p in progs], dtype=np.int16).reshape(-1, 2)


def preds(succT, b, rd):
    """pred[sigma] -> entities m with hop sigma from m to b."""
    return {s: succT[s][b] for s in range(rd.S) if len(succT[s][b])}


# ---------------------------------------------------------------- critic

class Critic(nn.Module):
    """Pairwise critic as frozen in the contract: c_t = sigmoid(MLP([rule-set context attended by step t; step t-1;
    step t])), 64 hidden. Steps 0..S-1; START = S (as last); END = S + 1 (as step). Distinct (last, step) pairs are
    scored once per rule set and reused across programs (computation only; the function is unchanged)."""
    def __init__(self, rd, arm):
        super().__init__()
        self.arm, self.S = arm, rd.S
        V = len(rd.vocab)
        codes = rd.net.R_L.detach()[torch.as_tensor(rd.vocab)]
        self.register_buffer('codes', torch.cat([codes, codes]))
        self.register_buffer('dirs', torch.cat([torch.zeros(V), torch.ones(V)]))
        self.register_buffer('allcodes', rd.net.R_L.detach().clone())
        self.special = nn.Parameter(torch.randn(2, EMB) * .3)
        self.step = nn.Sequential(nn.Linear(SYM + 1, HID), nn.ReLU(), nn.Linear(HID, EMB))
        self.var = nn.Parameter(torch.randn(3, 8) * .3)
        self.atom = nn.Sequential(nn.Linear(SYM + 16 + 3 + 1, HID), nn.ReLU(), nn.Linear(HID, HID))
        self.rule = nn.Sequential(nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, HID))
        self.qctx = nn.Sequential(nn.Linear(SYM, HID), nn.ReLU(), nn.Linear(HID, HID))
        self.query = nn.Linear(EMB, HID)
        self.out = nn.Sequential(nn.Linear(HID + 2 * EMB, HID), nn.ReLU(), nn.Linear(HID, 1))

    def emb_all(self):
        e = self.step(torch.cat([self.codes, self.dirs[:, None]], 1))
        return torch.cat([e, self.special])                          # (S + 2, EMB): steps, START, END

    def tokens(self, rs):
        if self.arm == 'question':
            return self.qctx(self.allcodes[rs['qrel']])[None]
        toks = []
        for h, (ha, hb), body, _ in rs['rules']:
            atoms = [(h, ha, hb, 0)] + [(r, va, vb, 1 + i) for i, (r, (va, vb)) in enumerate(body)]
            feats = []
            for r, va, vb, role in atoms:
                sym = self.allcodes[rs['glob'][r]] if r < rs['nb'] else torch.as_tensor(rs['sym'][r])
                feats.append(torch.cat([sym, self.var[va], self.var[vb], F.one_hot(torch.tensor(role), 3).float(),
                                        torch.tensor([float(r == rs['ask'])])]))
            toks.append(self.rule(self.atom(torch.stack(feats)).sum(0)))
        return torch.stack(toks)

    def score(self, toks, E, last, step):
        el, es = E[last], E[step]
        if self.arm == 'question':
            ctx = toks.expand(len(last), -1)
        else:
            a = torch.softmax(self.query(es) @ toks.T / HID ** .5, 1)
            ctx = a @ toks
        return torch.sigmoid(self.out(torch.cat([ctx, el, es], 1))[:, 0])

    @torch.no_grad()
    def table(self, rs):
        """C[last in 0..S (S = START), step in 0..S (S = END)]."""
        S = self.S
        E, toks = self.emb_all(), self.tokens(rs)
        last = torch.arange(S + 1).repeat_interleave(S + 1)
        step = torch.arange(S + 1).repeat(S + 1)
        step = torch.where(step == S, torch.full_like(step, S + 1), step)
        out = torch.cat([self.score(toks, E, last[i:i + 8192], step[i:i + 8192]) for i in range(0, len(last), 8192)])
        return out.numpy().reshape(S + 1, S + 1)


def train_critic(critic, qs, steps, seed):
    """Multiple-instance: question score = max over its programs of prod c (START, steps, END). 16 rule sets x
    (16 T + 16 F) questions per step = 256 T + 256 F."""
    opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    S = critic.S
    K = S + 2
    rng = np.random.default_rng([seed, 11])
    by = {}
    for i, q in enumerate(qs):
        if len(q['progs']):
            by.setdefault(q['rsid'], {0: [], 1: []})[q['y']].append(i)
    rsids = [r for r, d in by.items() if d[0] and d[1]]
    log = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        E = critic.emb_all()
        total, nq, pairs = 0., 0, 0
        for rsid in rng.choice(rsids, min(16, len(rsids)), replace=False):
            d = by[rsid]
            ii = list(rng.choice(d[1], min(16, len(d[1])), replace=False)) + \
                list(rng.choice(d[0], min(16, len(d[0])), replace=False))
            P = np.concatenate([qs[i]['progs'] for i in ii]).astype(np.int64)
            owner = np.concatenate([np.full(len(qs[i]['progs']), j) for j, i in enumerate(ii)])
            s1, s2 = P[:, 0], P[:, 1]
            one = s2 < 0
            L = np.concatenate([np.full(len(P), S), s1, s2[~one]])
            R = np.concatenate([s1, np.where(one, S + 1, np.maximum(s2, 0)), np.full(int((~one).sum()), S + 1)])
            prog = np.concatenate([np.arange(len(P)), np.arange(len(P)), np.flatnonzero(~one)])
            uniq, inv = np.unique(L * K + R, return_inverse=True)
            pairs += len(uniq)
            toks = critic.tokens(qs[ii[0]]['rs'])
            c = critic.score(toks, E, torch.as_tensor(uniq // K), torch.as_tensor(uniq % K))
            lc = torch.log(c.clamp_min(1e-6))[torch.as_tensor(inv)]
            lp = torch.zeros(len(P)).index_add(0, torch.as_tensor(prog), lc)
            qmax = torch.full((len(ii),), -1e9).scatter_reduce(0, torch.as_tensor(owner), lp, 'amax')
            p_ = qmax.exp().clamp(1e-6, 1 - 1e-6)
            y = torch.as_tensor([float(qs[i]['y']) for i in ii])
            total = total + F.binary_cross_entropy(p_, y, reduction='sum')
            nq += len(ii)
        loss = total / nq
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            log.append({'step': step, 'loss': loss.item(), 'distinct_pairs': pairs})
    return log


# ---------------------------------------------------------------- search

def beam_from(a, C, succ, S, rng_prio):
    """Beam from a; returns {b: (program, nodes)} for the first accepted path reaching each b."""
    allowed = C[:, :S] > .5
    endok = C[:S, S] > .5
    logC = np.log(np.clip(C, 1e-9, 1))
    beam = [(0., (a,), ())]
    found = {}
    expansions = 0
    for _ in range(D_MAX):
        ext = []
        for sc, nodes, prog in beam:
            last = prog[-1] if prog else S
            node = nodes[-1]
            for s in np.flatnonzero(allowed[last]):
                ms = succ[s][node]
                if not len(ms):
                    continue
                for m in ms:
                    m = int(m)
                    if m in nodes:
                        continue
                    expansions += 1
                    p2 = prog + (int(s),)
                    if endok[s] and m not in found:
                        found[m] = (p2, nodes + (m,))
                    ext.append((sc + logC[last, s] + rng_prio[m], nodes + (m,), p2))
        if not ext:
            break
        ext.sort(key=lambda x: -x[0])
        beam = ext[:W_BEAM]
    return found, expansions


def reach_returns(a, C, adjs, S, N):
    """Exhaustive with returns allowed: set of b reachable by an accepted walk (critic-checked), up to D_MAX hops."""
    allowed = C[:, :S] > .5
    endok = C[:S, S] > .5
    front = {S: np.eye(N, dtype=bool)[a]}
    acc = np.zeros(N, bool)
    seen = {}
    for _ in range(D_MAX):
        new = {}
        for last, vec in front.items():
            if not vec.any():
                continue
            for s in np.flatnonzero(allowed[last]):
                v = (vec.astype(np.float32) @ adjs[s].astype(np.float32)) > 0
                if s in seen:
                    v &= ~seen[s]
                if v.any():
                    new[s] = new.get(s, np.zeros(N, bool)) | v
        for s, v in new.items():
            seen[s] = seen.get(s, np.zeros(N, bool)) | v
            if endok[s]:
                acc |= v
        front = new
        if not front:
            break
    acc[a] = False
    return acc


def dfs_noreturn(a, C, succ, S, cap=DFS_CAP, max_len=D_MAX):
    allowed = C[:, :S] > .5
    endok = C[:S, S] > .5
    acc = set()
    n = 0
    stack = [((a,), S)]
    while stack:
        nodes, last = stack.pop()
        if len(nodes) - 1 >= max_len:
            continue
        for s in np.flatnonzero(allowed[last]):
            for m in succ[s][nodes[-1]]:
                m = int(m)
                if m in nodes:
                    continue
                n += 1
                if n > cap:
                    return acc, True
                if endok[s]:
                    acc.add(m)
                stack.append((nodes + (m,), int(s)))
    return acc, False


def program_accepts(C, prog, S):
    seq = (S,) + tuple(prog)
    return all(C[seq[t], prog[t]] > .5 for t in range(len(prog))) and C[prog[-1], S] > .5


def to_sigma(prog_glob, rd):
    V = len(rd.vocab)
    out = []
    for g, d in prog_glob:
        if g not in rd.vidx:
            return None
        out.append(rd.vidx[g] + d * V)
    return tuple(out)


# ---------------------------------------------------------------- taught tasks

def taught_specs(rd):
    R = rd.W.R
    P, JB, K, SB = R['parent-of'], R['just-bigger-than'], R['kind-of'], R['subkind-of']
    rec = [(1, (0, 1), [(0, (0, 1))], 'sub'), (1, (0, 2), [(0, (0, 1)), (1, (1, 2))], 'lin-l')]
    return {
        'ancestor-of': dict(glob=[P], nb=1, m=2, rules=list(rec), ask=1, qrel=R['ancestor-of']),
        'bigger-than': dict(glob=[JB], nb=1, m=2, rules=list(rec), ask=1, qrel=R['bigger-than']),
        'smaller-than': dict(glob=[JB], nb=1, m=3, rules=rec + [(2, (0, 1), [(1, (1, 0))], 'inv')], ask=2,
                             qrel=R['smaller-than']),
        'is-a': dict(glob=[K, SB], nb=2, m=3, rules=[(2, (0, 1), [(0, (0, 1))], 'sub'),
                                                     (2, (0, 2), [(2, (0, 1)), (1, (1, 2))], 'lin-r')], ask=2,
                     qrel=R['is-a']),
        'grandparent-of': dict(glob=[P], nb=1, m=2, rules=[(1, (0, 2), [(0, (0, 1)), (0, (1, 2))], 'chain1')], ask=1,
                               qrel=R['grandparent-of']),
        'child-of': dict(glob=[P], nb=1, m=2, rules=[(1, (0, 1), [(0, (1, 0))], 'inv')], ask=1, qrel=R['child-of']),
    }


def task_entities(rs, orc):
    s = set()
    for g in rs['glob']:
        xs, ys = np.nonzero(orc[g])
        s |= set(xs.tolist()) | set(ys.tolist())
    return sorted(s)


# ---------------------------------------------------------------- edits

def with_rules(rs, rules=None, m=None, sym=None):
    q = dict(rs)
    if rules is not None:
        q['rules'] = rules
    if m is not None:
        q['m'] = m
    if sym is not None:
        q['sym'] = sym
    return q


def make_edits(rs, rng, rd, taught_name=None):
    out = {}
    rules, nb, m = rs['rules'], rs['nb'], rs['m']
    if taught_name:
        out['e1-body-args'] = with_rules(rs, [(h, hv, [(r, (v[1], v[0])) if r < nb else (r, v) for r, v in body], f)
                                              for h, hv, body, f in rules])
        if taught_name == 'ancestor-of':
            q = with_rules(rs)
            q['glob'] = [rd.W.R['spouse-of']] + rs['glob'][1:]
            out['e2-body-rel'] = q
        h, hv, body, f = rules[0]
        out['e3-head-args'] = with_rules(rs, [(h, (hv[1], hv[0]), body, f)] + rules[1:])
    else:
        j = int(rng.integers(len(rules)))
        h, hv, body, f = rules[j]
        r0, (va, vb) = body[0]
        out['e1-body-args'] = with_rules(rs, rules[:j] + [(h, hv, [(r0, (vb, va))] + body[1:], f)] + rules[j + 1:])
        if r0 < nb:
            q = with_rules(rs)
            others = [g for g in rd.vocab if g not in rs['glob']]
            q['glob'] = list(rs['glob'])
            q['glob'][r0] = int(others[rng.integers(len(others))])
            out['e2-body-rel'] = q
        out['e3-head-args'] = with_rules(rs, rules[:j] + [(h, (hv[1], hv[0]), body, f)] + rules[j + 1:])
    out['i1-fresh-symbols'] = with_rules(rs, sym=unitv(rng, m))
    out['i2-dead-rule'] = with_rules(rs, rules + [(rs['ask'], (0, 1), [(m, (0, 1))], 'sub')], m + 1,
                                     np.concatenate([rs['sym'], unitv(rng, 1)]))
    out['i3-unused-rule'] = with_rules(rs, rules + [(m, (0, 1), [(0, (0, 1))], 'sub')], m + 1,
                                       np.concatenate([rs['sym'], unitv(rng, 1)]))
    return out


# ---------------------------------------------------------------- evaluation of question sets

def new_tally():
    return {'T': {k: [0, 0] for k in BUCKETS}, 'F': [0, 0], 'conn': [0, 0],
            'tax': {k: {c: 0 for c in ('no-proof', 'found', 'lost', 'rejected')} for k in BUCKETS},
            'valid': {k: [0, 0] for k in BUCKETS}, 'product_fail': {k: [0, 0] for k in BUCKETS},
            'noreturn_cost': [0, 0], 'dfs_truncated': 0, 'expansions': {k: [0, 0] for k in BUCKETS}}


def evaluate(rs, qlist, C, ctx, tl, sup=None, full=True):
    """qlist: list of (a, b, y, h_oracle, connected). ctx: graph context dict(graph, succ, adjs, rd, prio)."""
    rd, S = ctx['rd'], ctx['rd'].S
    yG, hG = labels(rs, ctx['graph'])
    by_a = {}
    for q in qlist:
        by_a.setdefault(q[0], []).append(q)
    for a, qs in by_a.items():
        found, nexp = beam_from(a, C, ctx['succ'], S, ctx['prio'])
        if full:
            ret = reach_returns(a, C, ctx['adjs'], S, rd.N)
            dfs, trunc = dfs_noreturn(a, C, ctx['succ'], S)
            tl['dfs_truncated'] += int(trunc)
        for a_, b, y, h, conn in qs:
            ans = b in found
            if y:
                k = bucket(h) or '7-12'
                tl['T'][k][0] += int(ans); tl['T'][k][1] += 1
                tl['expansions'][k][0] += nexp; tl['expansions'][k][1] += 1
                if full:
                    if hG[a, b] == 0:
                        cat = 'no-proof'
                    elif ans:
                        cat = 'found'
                    elif b in dfs or ret[b]:
                        cat = 'lost'
                    else:
                        cat = 'rejected'
                    tl['tax'][k][cat] += 1
                    tl['noreturn_cost'][0] += int(bool(ret[b]) and b not in dfs and not trunc)
                    tl['noreturn_cost'][1] += 1
                if ans:
                    prog, nodes = found[b]
                    V = len(rd.vocab)
                    glob = tuple((rd.vocab[s % V], int(s >= V)) for s in prog)
                    tl['valid'][k][0] += int(walk_valid(glob, rs)); tl['valid'][k][1] += 1
                    if sup is not None:
                        pc = C[S, prog[0]] * C[prog[-1], S]
                        for t in range(1, len(prog)):
                            pc *= C[prog[t - 1], prog[t]]
                        for t, s in enumerate(prog):
                            r = rd.vocab[s % V]
                            x, m_ = nodes[t], nodes[t + 1]
                            pc *= float(sup[r][x, m_] if s < V else sup[r][m_, x])
                        tl['product_fail'][k][0] += int(pc <= .5); tl['product_fail'][k][1] += 1
            else:
                tl['F'][0] += int(not ans); tl['F'][1] += 1
                if conn:
                    tl['conn'][0] += int(not ans); tl['conn'][1] += 1
    return tl


def summarize_tally(tl):
    r = lambda h: None if h[1] == 0 else h[0] / h[1]
    fr = r(tl['F'])
    out = {'T_recall': {k: r(v) for k, v in tl['T'].items()}, 'T_n': {k: v[1] for k, v in tl['T'].items()},
           'F_recall': fr, 'F_n': tl['F'][1], 'conn_F_recall': r(tl['conn']), 'conn_n': tl['conn'][1],
           'taxonomy': tl['tax'], 'proof_validity': {k: r(v) for k, v in tl['valid'].items()},
           'product_policy_fail': {k: r(v) for k, v in tl['product_fail'].items()},
           'noreturn_cost': r(tl['noreturn_cost']), 'dfs_truncated': tl['dfs_truncated'],
           'expansions_per_question': {k: r(v) for k, v in tl['expansions'].items()}}
    out['bal'] = {k: None if (t is None or fr is None) else .5 * (t + fr) for k, t in out['T_recall'].items()}
    return out


def language_reads(rs, C, rd, rng, max_len=D_MAX):
    L = language(rs['nb'], rs['rules'], rs['m'], rs['glob'], cap_len=max_len)[rs['ask']]
    valid = {}
    for p in L:
        sp = to_sigma(p, rd)
        if sp is not None:
            valid.setdefault(len(sp), []).append(sp)
    out = {}
    for k in range(1, max_len + 1):
        vs = valid.get(k, [])
        vset = set(vs)
        spur = []
        for _ in range(200):
            p = tuple(int(x) for x in rng.integers(rd.S, size=k))
            if p not in vset:
                spur.append(p)
        out[str(k)] = {'valid_accepted': None if not vs else float(np.mean([program_accepts(C, p, rd.S) for p in vs[:200]])),
                       'n_valid': len(vs),
                       'spurious_accepted': float(np.mean([program_accepts(C, p, rd.S) for p in spur]))}
    return out


# ---------------------------------------------------------------- run

def build_training(rd, graph, orc, seed, arm, specs):
    rng = np.random.default_rng([seed, 21])
    succ = ctxs = None
    succ = succ_lists(graph, rd)
    succT = [[None] * rd.N for _ in range(rd.S)]
    for s in range(rd.S):
        A = step_adj(graph, rd, s).T
        succT[s] = [np.flatnonzero(A[x]) for x in range(rd.N)]
    conn = connected(graph, rd)
    eye = np.eye(rd.N, dtype=bool)
    qs, stats = [], {'oracle_T_short': 0, 'covered': 0, 'F_excluded_reader_closure': 0, 'rulesets': 0}

    def add(rs, rsid, pairs_ok=None, limitT=QPER, limitF=QPER):
        yo, ho = labels(rs, orc, K=2)
        yc, hc = labels(rs, graph, K=2)
        ok = ~eye if pairs_ok is None else pairs_ok
        shortT = yo & (ho >= 1) & (ho <= 2) & ok
        cov = shortT & (hc >= 1) & (hc <= 2)
        stats['oracle_T_short'] += int(shortT.sum()); stats['covered'] += int(cov.sum())
        Fm = ~yo & ok
        stats['F_excluded_reader_closure'] += int((Fm & yc).sum())
        Fm = Fm & ~yc
        T = np.argwhere(cov)
        Fc, Fo = np.argwhere(Fm & conn), np.argwhere(Fm & ~conn)
        pick = lambda A, n: A[rng.choice(len(A), min(n, len(A)), replace=False)] if len(A) else A
        chosen = [(a, b, 1) for a, b in pick(T, limitT)] + [(a, b, 0) for a, b in pick(Fc, limitF // 2)] + \
                 [(a, b, 0) for a, b in pick(Fo, limitF - min(limitF // 2, len(Fc)))]
        for a, b, y in chosen:
            pr = preds(succT, int(b), rd)
            qs.append(dict(rs=rs, rsid=rsid, a=int(a), b=int(b), y=y, progs=candidates(int(a), int(b), succ, pr, rd)))
    if arm == 'question':
        for ti, (name, rs) in enumerate(specs.items()):
            ents = task_entities(rs, orc)
            ok = np.zeros((rd.N, rd.N), bool)
            ok[np.ix_(ents, ents)] = True
            ok &= ~eye & ~held_mask(ti, rd.N)
            add(rs, ti, ok, limitT=10 ** 6, limitF=10 ** 6)
            stats['rulesets'] += 1
    else:
        for i in range(N_RULESETS):
            rs = make_ruleset(rng, rd, orc)
            add(rs, i)
            stats['rulesets'] += 1
    stats['questions'] = len(qs)
    stats['programs'] = int(sum(len(q['progs']) for q in qs))
    stats['T_without_candidates'] = int(sum(1 for q in qs if q['y'] == 1 and not len(q['progs'])))
    return qs, stats


def held_mask(ti, N):
    return np.random.default_rng([ti, 99]).random((N, N)) < .5


def graph_ctx(graph, rd, seed):
    return dict(graph=graph, rd=rd, succ=succ_lists(graph, rd), adjs=[step_adj(graph, rd, s) for s in range(rd.S)],
                prio=np.random.default_rng([seed, 41]).random(rd.N) * 1e-6)


def run(arm, seed):
    torch.set_num_threads(1)
    ph = {}
    t0 = time.process_time()
    rd = Readers(seed)
    g, ng, sup, orc = rd.graphs()
    graph = ng if arm == 'rules-nogate' else g
    ph['graph_cpu_seconds'] = time.process_time() - t0
    specs = taught_specs(rd)
    for ti, rs in enumerate(specs.values()):
        rs['sym'] = unitv(np.random.default_rng([seed, 71, ti]), rs['m'])
    res = {'arm': arm, 'seed': seed, 'reader_checkpoint_sha256': rd.ck_sha, 'vocab': len(rd.vocab),
           'hop_composition': {'confirmed': hop_composition(graph, rd), 'gated': hop_composition(g, rd),
                               'ungated': hop_composition(ng, rd)},
           'oracle_edges': int(sum(v.sum() for v in orc.values()))}
    t0 = time.process_time()
    qs, tstats = build_training(rd, graph, orc, seed, arm, specs)
    res['training'] = tstats
    ph['candidates_cpu_seconds'] = time.process_time() - t0
    t0 = time.process_time()
    torch.manual_seed(seed)
    critic = Critic(rd, arm)
    res['log'] = train_critic(critic, qs, STEPS, seed)
    del qs
    ph['critic_cpu_seconds'] = time.process_time() - t0
    for p in critic.parameters():
        p.requires_grad_(False)
    ctxs = {'confirmed': graph_ctx(graph, rd, seed), 'oracle': graph_ctx(orc, rd, seed)}
    conn = connected(graph, rd)
    rng = np.random.default_rng([seed, 51])
    eye = np.eye(rd.N, dtype=bool)
    # taught tasks
    t0 = time.process_time()
    res['taught'] = {}
    for ti, (name, rs) in enumerate(specs.items()):
        ents = task_entities(rs, orc)
        yo, ho = labels(rs, orc)
        hm = held_mask(ti, rd.N)
        qall = [(a, b, int(yo[a, b]), int(ho[a, b]), bool(conn[a, b])) for a in ents for b in ents if a != b]
        C = critic.table(rs)
        rec = {'n_questions': len(qall), 'language': language_reads(rs, C, rd, rng)}
        for gname, ctx in ctxs.items():
            tl = evaluate(rs, qall, C, ctx, new_tally(), sup if gname == 'confirmed' else None)
            th = evaluate(rs, [q for q in qall if hm[q[0], q[1]]], C, ctx, new_tally(), full=False)
            rec[gname] = summarize_tally(tl)
            rec[gname + '_held_half'] = summarize_tally(th)
        yc, _ = labels(rs, graph)
        sel = [(a, b) for a in ents for b in ents if a != b]
        ys = np.array([yo[a, b] for a, b in sel]); cs = np.array([yc[a, b] for a, b in sel])
        rec['confirmed_closure'] = {'T_recall': float(cs[ys].mean()) if ys.any() else None,
                                    'F_recall': float((~cs[~ys]).mean())}
        if name == 'grandparent-of':
            ya, ha = labels(specs['ancestor-of'], orc)
            q34 = [(a, b, 0, 0, False) for a in ents for b in ents if a != b and ha[a, b] in (3, 4)]
            tl = evaluate(rs, q34, C, ctxs['oracle'], new_tally(), full=False)
            rec['counting_F_recall_3_4'] = summarize_tally(tl)['F_recall']
        with torch.no_grad():
            a_ = torch.as_tensor([a for a, b in sel]); b_ = torch.as_tensor([b for a, b in sel])
            u = rd.net.intuition(a_, torch.full_like(a_, rs['qrel']), b_).numpy() > 0
        rec['intuition_bal'] = float(.5 * (u[ys].mean() + (~u[~ys]).mean())) if ys.any() and (~ys).any() else None
        # edits on the oracle graph
        rec['edits'] = {}
        base = {}
        found_by_a = {a: beam_from(a, C, ctxs['oracle']['succ'], rd.S, ctxs['oracle']['prio'])[0] for a in ents}
        for key, e in make_edits(rs, rng, rd, name).items():
            ye, _ = labels(e, orc)
            Ce = critic.table(e)
            fl, kp = [0, 0], [0, 0]
            for a in ents:
                fe, _ = beam_from(a, Ce, ctxs['oracle']['succ'], rd.S, ctxs['oracle']['prio'])
                for b in ents:
                    if a == b:
                        continue
                    old, new = b in found_by_a[a], b in fe
                    if ye[a, b] != yo[a, b]:
                        fl[0] += int(new == bool(ye[a, b])); fl[1] += 1
                    else:
                        kp[0] += int(new == old); kp[1] += 1
            rec['edits'][key] = {'flipped_correct': None if not fl[1] else fl[0] / fl[1], 'n_flipped': fl[1],
                                 'unchanged_answer_rate': None if not kp[1] else kp[0] / kp[1]}
        res['taught'][name] = rec
    ph['taught_cpu_seconds'] = time.process_time() - t0
    # synthetic
    if arm != 'question':
        t0 = time.process_time()
        srng = np.random.default_rng([seed, 31])
        sets = [make_ruleset(srng, rd, orc) for _ in range(N_SYN)]
        poolT = {k: [] for k in BUCKETS}
        poolF = []
        for i, rs in enumerate(sets):
            yo, ho = labels(rs, orc)
            for a, b in np.argwhere(yo & ~eye):
                poolT[bucket(int(ho[a, b])) or '7-12'].append((i, int(a), int(b), 1, int(ho[a, b]), bool(conn[a, b])))
            Fa = np.argwhere(~yo & ~eye)
            for a, b in Fa[srng.choice(len(Fa), min(400, len(Fa)), replace=False)]:
                poolF.append((i, int(a), int(b), 0, 0, bool(conn[a, b])))
        chosen, short = [], {}
        for k in ('1', '2', '3', '4', '5-6'):
            P = poolT[k]
            take = [P[j] for j in srng.choice(len(P), min(QUOTA, len(P)), replace=False)] if P else []
            short[k] = len(take)
            chosen += take
        nT = len(chosen)
        Fc = [q for q in poolF if q[5]]; Fo = [q for q in poolF if not q[5]]
        chosen += [Fc[j] for j in srng.choice(len(Fc), min(nT // 2, len(Fc)), replace=False)] if Fc else []
        chosen += [Fo[j] for j in srng.choice(len(Fo), min(nT - nT // 2, len(Fo)), replace=False)] if Fo else []
        syn = {'quota_filled': short, 'F_n': len(chosen) - nT}
        tables = {i: critic.table(rs) for i, rs in enumerate(sets)}
        for gname, ctx in ctxs.items():
            tl = new_tally()
            for i, rs in enumerate(sets):
                ql = [q[1:] for q in chosen if q[0] == i]
                if ql:
                    full = all(q[3] <= 4 for q in ql if q[2])
                    evaluate(rs, ql, tables[i], ctx, tl, sup if gname == 'confirmed' else None, full=True)
            syn[gname] = summarize_tally(tl)
        syn['language'] = {}
        for k in range(1, 7):
            vals = [language_reads(rs, tables[i], rd, rng, max_len=6)[str(k)] for i, rs in enumerate(sets[:20])]
            syn['language'][str(k)] = {q: (lambda v: None if not v else float(np.mean(v)))(
                [x[q] for x in vals if x[q] is not None]) for q in ('valid_accepted', 'spurious_accepted')}
        # synthetic edits (oracle graph)
        ed = {}
        sel = chosen[:N_SYN_EDIT]
        for i in sorted({q[0] for q in sel}):
            rs = sets[i]
            ql = [q for q in sel if q[0] == i]
            yo, _ = labels(rs, orc)
            starts = sorted({q[1] for q in ql})
            f0 = {a: beam_from(a, tables[i], ctxs['oracle']['succ'], rd.S, ctxs['oracle']['prio'])[0] for a in starts}
            for key, e in make_edits(rs, rng, rd).items():
                ye, _ = labels(e, orc)
                Ce = critic.table(e)
                d = ed.setdefault(key, {'fl': [0, 0], 'kp': [0, 0]})
                for a in starts:
                    fe, _ = beam_from(a, Ce, ctxs['oracle']['succ'], rd.S, ctxs['oracle']['prio'])
                    for q in ql:
                        if q[1] != a:
                            continue
                        b = q[2]
                        old, new = b in f0[a], b in fe
                        if ye[a, b] != yo[a, b]:
                            d['fl'][0] += int(new == bool(ye[a, b])); d['fl'][1] += 1
                        else:
                            d['kp'][0] += int(new == old); d['kp'][1] += 1
        syn['edits'] = {k: {'flipped_correct': None if not v['fl'][1] else v['fl'][0] / v['fl'][1], 'n_flipped': v['fl'][1],
                            'unchanged_answer_rate': None if not v['kp'][1] else v['kp'][0] / v['kp'][1]}
                        for k, v in ed.items()}
        # conj diagnostic
        crng = np.random.default_rng([seed, 61])
        tl = new_tally()
        for _ in range(N_CONJ):
            try:
                rs = make_ruleset(crng, rd, orc, force='conj', min_true=5)
            except RuntimeError:
                continue
            yo, ho = labels(rs, orc)
            T = np.argwhere(yo & ~eye); Fa = np.argwhere(~yo & ~eye)
            ql = [(int(a), int(b), 1, 1, False) for a, b in T[crng.choice(len(T), min(20, len(T)), replace=False)]] + \
                 [(int(a), int(b), 0, 0, False) for a, b in Fa[crng.choice(len(Fa), min(20, len(Fa)), replace=False)]]
            evaluate(rs, ql, critic.table(rs), ctxs['oracle'], tl, full=False)
        syn['conj_diagnostic'] = summarize_tally(tl)
        res['synthetic'] = syn
        ph['synthetic_cpu_seconds'] = time.process_time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(critic.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    rt = {**ph, 'critic_params': int(sum(p.numel() for p in critic.parameters())), 'steps': STEPS,
          'training': tstats, 'checkpoint_bytes': ck.stat().st_size}
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
    V2.save_json(f'{REPORT}.json', {'status': 'rule-guided search v1' + (' (smoke)' if SMOKE else ''),
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
        o = {'hop_composition': R[0]['hop_composition'], 'training': {k: m([r['training'][k] for r in R]) for k in R[0]['training']},
             'taught': {}}
        for name in R[0]['taught']:
            tr = [r['taught'][name] for r in R]
            t = {}
            for gname in ('confirmed', 'oracle', 'confirmed_held_half', 'oracle_held_half'):
                xs = [x[gname] for x in tr]
                t[gname] = {'bal': {k: m([x['bal'][k] for x in xs]) for k in BUCKETS},
                            'T_recall': {k: m([x['T_recall'][k] for x in xs]) for k in BUCKETS},
                            'F_recall': m([x['F_recall'] for x in xs]), 'conn_F_recall': m([x['conn_F_recall'] for x in xs]),
                            'proof_validity': {k: m([x['proof_validity'][k] for x in xs]) for k in BUCKETS},
                            'taxonomy': {k: {c: int(sum(x['taxonomy'][k][c] for x in xs)) for c in ('no-proof', 'found', 'lost', 'rejected')}
                                         for k in BUCKETS}}
            t['language'] = {k: {q: m([x['language'][k][q] for x in tr]) for q in ('valid_accepted', 'spurious_accepted')}
                             for k in tr[0]['language']}
            t['confirmed_closure'] = {q: m([x['confirmed_closure'][q] for x in tr]) for q in ('T_recall', 'F_recall')}
            t['edits'] = {k: {q: m([x['edits'][k][q] for x in tr if k in x['edits']]) for q in ('flipped_correct', 'unchanged_answer_rate')}
                          for k in tr[0]['edits']}
            t['intuition_bal'] = m([x['intuition_bal'] for x in tr])
            if 'counting_F_recall_3_4' in tr[0]:
                t['counting_F_recall_3_4'] = m([x['counting_F_recall_3_4'] for x in tr])
            o['taught'][name] = t
        if 'synthetic' in R[0]:
            sy = [r['synthetic'] for r in R]
            o['synthetic'] = {'quota_filled': sy[0]['quota_filled']}
            for gname in ('confirmed', 'oracle'):
                xs = [x[gname] for x in sy]
                o['synthetic'][gname] = {'bal': {k: m([x['bal'][k] for x in xs]) for k in BUCKETS},
                                         'F_recall': m([x['F_recall'] for x in xs]), 'conn_F_recall': m([x['conn_F_recall'] for x in xs]),
                                         'proof_validity': {k: m([x['proof_validity'][k] for x in xs]) for k in BUCKETS},
                                         'taxonomy': {k: {c: int(sum(x['taxonomy'][k][c] for x in xs)) for c in ('no-proof', 'found', 'lost', 'rejected')} for k in BUCKETS},
                                         'product_policy_fail': {k: m([x['product_policy_fail'][k] for x in xs]) for k in BUCKETS},
                                         'noreturn_cost': m([x['noreturn_cost'] for x in xs])}
            o['synthetic']['language'] = {k: {q: m([x['language'][k][q] for x in sy]) for q in ('valid_accepted', 'spurious_accepted')} for k in sy[0]['language']}
            o['synthetic']['edits'] = {k: {q: m([x['edits'][k][q] for x in sy if k in x['edits']]) for q in ('flipped_correct', 'unchanged_answer_rate')} for k in sy[0]['edits']}
            o['synthetic']['conj'] = {'bal_1': m([x['conj_diagnostic']['bal']['1'] for x in sy]), 'F_recall': m([x['conj_diagnostic']['F_recall'] for x in sy])}
        out[arm] = o
        print(f"\n===== {arm}: hop composition {o['hop_composition']['confirmed']} training {o['training']}")
        for name, t in o['taught'].items():
            print(f"  {name}: oracle bal {t['oracle']['bal']} F {t['oracle']['F_recall']} | confirmed bal {t['confirmed']['bal']} F {t['confirmed']['F_recall']} "
                  f"| closure {t['confirmed_closure']} | lang {t['language']} | edits {t['edits']} | intuition {t['intuition_bal']}"
                  + (f" | counting {t['counting_F_recall_3_4']}" if 'counting_F_recall_3_4' in t else ''))
            print(f"     taxonomy oracle {t['oracle']['taxonomy']} confirmed {t['confirmed']['taxonomy']} validity conf {t['confirmed']['proof_validity']}")
        if 'synthetic' in o:
            print(f"  synthetic {json.dumps(o['synthetic'])}")
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
