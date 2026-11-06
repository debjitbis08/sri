"""Relations as objects v1 (docs/relations-as-objects-v1.md): relations get object codes; relations among relations
(five arbitrary tokens, truth computed from extensions) are ordinary facts; a held-out relation is introduced through
its relations to known ones. Arms: map (operator = f(object code)), nomap (own operator table), merged (one space).

Usage: EV_WORKERS=6 .venv/bin/python -m experiments.relations_as_objects        (rerun to resume)
       .venv/bin/python -m experiments.relations_as_objects summary
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

from experiments import conflict_search as CS
from experiments import dual_route_truth as DR
from experiments import durable as DUR
from experiments import everyday_relations_v2 as V2
from experiments import taught_truth as TT
from experiments import type_meanings as TM
from src import everyday_world as W1

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'relations-as-objects-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'relations-as-objects-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 100 if SMOKE else 3000
FIT_STEPS = 60 if SMOKE else 500
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('map', 'nomap', 'merged')
CONDS = ('rho', 'none', 'shuffled')
KS = (0, 1, 4)
BATCH, NEG_BATCH, DROP = 8192, 2048, 0.3
D, RD, H = DR.D, DR.RD, DR.HID
WORLD_SEED, HOLD_SEED = 2026, 2027
SYN_CATS = ('person', 'animal', 'object', 'property', 'sound', 'colour', 'body-part', 'ability', 'food', 'substance')
REAL_HELD = ('child-of', 'darker-than')
RHO = ('rho-1', 'rho-2', 'rho-3', 'rho-4', 'rho-5')
SHUFFLE = {0: 2, 2: 1, 1: 0, 3: 4, 4: 3}            # rho-1 -> rho-3 -> rho-2 -> rho-1, rho-4 <-> rho-5


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, H), nn.ReLU(), nn.Linear(H, o))


# ---------------------------------------------------------------- world

class World:
    def __init__(self, seed):
        st, st0, train_rows, _, members = TM.build('isa', seed)
        self.st = st
        ents = list(st.entities)
        E = {x: i for i, x in enumerate(ents)}
        cats = TM.entity_types(st0, members, len(ents))
        animals = sorted(i for i in range(len(ents)) if 'animal' in cats[i] and ents[i] not in TM.CATEGORIES)
        mem = {c: [E[x] for x in members[c]] for c in members}
        mem['animal'] = animals
        self.members = mem
        self.cats = cats
        # synthetic relations, generated once
        g = np.random.default_rng(WORLD_SEED)
        syn = []                                  # dicts: name, kind, A, B (category names), T (bool |A| x |B|), partner
        pick = lambda: SYN_CATS[g.integers(len(SYN_CATS))]

        def rand_rel(A, B, p):
            m = g.random((len(mem[A]), len(mem[B]))) < p
            if A == B:
                np.fill_diagonal(m, False)
            if m.sum() < 2:
                m.flat[g.choice(m.size, 2, replace=False)] = True
                if A == B:
                    np.fill_diagonal(m, False)
            return m
        for i in range(12):
            A, B = pick(), pick()
            m = rand_rel(A, B, .25)
            syn.append(dict(name=f'inv{i}', kind='inv-base', A=A, B=B, T=m, fam=f'inv{i}'))
            syn.append(dict(name=f'inv{i}p', kind='inverse', A=B, B=A, T=m.T.copy(), fam=f'inv{i}'))
        for i in range(8):
            A, B = pick(), pick()
            m = rand_rel(A, B, .25)
            syn.append(dict(name=f'syn{i}', kind='syn-base', A=A, B=B, T=m, fam=f'syn{i}'))
            syn.append(dict(name=f'syn{i}s', kind='synonym', A=A, B=B, T=m.copy(), fam=f'syn{i}'))
        for i in range(8):
            A, B = pick(), pick()
            broad = rand_rel(A, B, .4)
            narrow = broad & (g.random(broad.shape) < .5)
            if narrow.sum() < 2:
                narrow.flat[np.flatnonzero(broad)[:2]] = True
            syn.append(dict(name=f'imp{i}b', kind='broad', A=A, B=B, T=broad, fam=f'imp{i}'))
            syn.append(dict(name=f'imp{i}n', kind='narrow', A=A, B=B, T=narrow, fam=f'imp{i}'))
        for i in range(6):
            A = pick()
            m = rand_rel(A, A, .25)
            m = m | m.T
            syn.append(dict(name=f'sym{i}', kind='symmetric', A=A, B=A, T=m, fam=f'sym{i}'))
        self.syn = syn
        # held-out relations, chosen once
        h = np.random.default_rng(HOLD_SEED)
        held = []
        fams = {s['name']: s['fam'] for s in syn}
        for kind, n in (('inverse', 4), ('synonym', 3), ('broad', 2), ('narrow', 2), ('symmetric', 2)):
            used = {fams[x] for x in held}
            cand = [s['name'] for s in syn if s['kind'] == kind and s['fam'] not in used]
            held += list(h.choice(cand, n, replace=False))
        fam_used = {}
        for s in syn:                              # one held-out per family (broad and narrow from different families)
            if s['name'] in held:
                if s['fam'] in fam_used:
                    raise RuntimeError('two held-out relations in one family')
                fam_used[s['fam']] = s['name']
        self.held_syn = held
        # relations: real, synthetic, rho
        self.real = list(st.relations)
        self.rels = self.real + [s['name'] for s in syn] + list(RHO)
        R = {x: i for i, x in enumerate(self.rels)}
        self.R = R
        self.held = [R[x] for x in REAL_HELD] + [R[x] for x in held]
        self.syn_idx = {R[s['name']]: s for s in syn}
        # relation objects: one entity per relation; categories that are entities
        self.ents = ents + [f'rel:{x}' for x in self.rels]
        self.obj = np.array([len(ents) + i for i in range(len(self.rels))])
        self.cat_ent = {c: E[c] for c in TM.CATEGORIES + ('animal',)}
        # extensions (true pairs, labelled pairs) per relation
        ind = (st.kind == 'individual') & (st.world_y >= 0)
        self.true, self.lab = {}, {}
        for i, rel in enumerate(self.real):
            m = ind & (st.r == i)
            self.lab[i] = dict(zip(zip(st.a[m].tolist(), st.b[m].tolist()), st.world_y[m].tolist()))
        for ri, s in self.syn_idx.items():
            A, B = mem[s['A']], mem[s['B']]
            self.lab[ri] = {(A[x], B[y]): int(s['T'][x, y]) for x in range(len(A)) for y in range(len(B))
                            if A[x] != B[y]}
        for ri, lab in self.lab.items():
            self.true[ri] = {p for p, v in lab.items() if v == 1}
        self.meta = sorted(ri for ri in self.lab if self.true[ri])
        # facts: world (training split), synthetic instances, rho
        rows = []                                  # (a, r, b, y, group, rel_for_drop)
        wm = ind & train_rows & ~np.isin(np.array(self.real)[st.r], REAL_HELD)
        for a, r, b, y in zip(st.a[wm], st.r[wm], st.b[wm], st.world_y[wm]):
            rows.append((a, r, b, y, 0, -1))
        for ri, s in self.syn_idx.items():
            if ri in self.held:
                continue
            for (a, b), y in self.lab[ri].items():
                rows.append((a, ri, b, y, 1, ri))
        self.rho_rows = self.rho_facts()
        for (x, k, y, lab) in self.rho_rows:
            if self._rel_of_obj(x) in self.held or self._rel_of_obj(y) in self.held:
                continue
            rows.append((x, R[RHO[k]], y, lab, 2, -1))
        arr = np.array(rows, dtype=np.int64)
        self.fa, self.fr, self.fb, self.fy, self.fg, self.fdrop = arr.T
        # sampling weights: equal per (relation, value) cell
        cells = {}
        for i, key in enumerate(zip(self.fr.tolist(), self.fy.tolist())):
            cells.setdefault(key, []).append(i)
        w = np.zeros(len(arr))
        for idx in cells.values():
            w[idx] = 1 / (len(cells) * len(idx))
        self.fw = w
        # meaningless examples
        rng = np.random.default_rng([seed, 9])
        train0, _ = CS.splits(st0, seed)
        _, _, masks, permitted = TM.meaningless_sets(st0, train0, members, len(ents), seed)
        self.masks = {}
        for rel in permitted:
            self.masks[R[rel]] = masks[self.real.index(rel)]
        catd = np.array([bool(c) for c in cats])
        for ri, s in self.syn_idx.items():
            inA = np.array(['animal' in c if s['A'] == 'animal' else s['A'] in c for c in cats])
            inB = np.array(['animal' in c if s['B'] == 'animal' else s['B'] in c for c in cats])
            M = catd[:, None] & catd[None, :] & (~inA[:, None] | ~inB[None, :])
            np.fill_diagonal(M, False)
            self.masks[ri] = M
        neg = []
        for ri, M in self.masks.items():
            if ri in self.held:
                continue
            xy = np.argwhere(M)
            if len(xy):
                for x, y in xy[rng.choice(len(xy), min(50, len(xy)), replace=False)]:
                    neg.append((x, ri, y))
        self.neg = np.array(neg, dtype=np.int64)
        self.n_ent, self.n_rel = len(self.ents), len(self.rels)

    def _rel_of_obj(self, x):
        base = self.obj[0]
        return x - base if base <= x < base + len(self.rels) else -1

    def rho_facts(self):
        """(x_entity, rho_index, y_entity, label) for rho-1..3 over meta relations and rho-4/5 over categories."""
        out = []
        rev = {ri: {(b, a) for a, b in self.true[ri]} for ri in self.meta}
        labrev = {ri: {(b, a) for a, b in self.lab[ri]} for ri in self.meta}
        labset = {ri: set(self.lab[ri]) for ri in self.meta}

        def agree(X, Y, reverse):
            LX = labrev[X] if reverse else labset[X]
            TX = rev[X] if reverse else self.true[X]
            O = LX & labset[Y]
            if len(O) < .5 * min(len(LX), len(labset[Y])):
                return None
            return O, TX & O, self.true[Y] & O
        for X in self.meta:
            for Y in self.meta:
                inv = agree(X, Y, True)
                lab1 = int(inv is not None and inv[1] == inv[2] and len(inv[1]) >= 1)
                out.append((self.obj[X], 0, self.obj[Y], lab1))
                if X == Y:
                    continue
                same = agree(X, Y, False)
                lab2 = int(same is not None and len(same[1]) >= 2 and same[1] <= same[2])
                lab3 = int(same is not None and len(same[1]) >= 1 and same[1] == same[2])
                out.append((self.obj[X], 1, self.obj[Y], lab2))
                out.append((self.obj[X], 2, self.obj[Y], lab3))
        for X in self.meta:
            subj = {a for a, b in self.lab[X]}
            obj = {b for a, b in self.lab[X]}
            for c, ce in self.cat_ent.items():
                m = set(self.members[c])
                out.append((self.obj[X], 3, ce, int(len(m & subj) >= .5 * len(m))))
                out.append((self.obj[X], 4, ce, int(len(m & obj) >= .5 * len(m))))
        return out


# ---------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, W, arm, g):
        super().__init__()
        self.arm = arm
        p = lambda *s: nn.Parameter(torch.randn(*s, generator=g) * TT.INIT_STD)
        self.E_L, self.E_I, self.R_I = p(W.n_ent, D), p(W.n_ent, D), p(W.n_rel, RD)
        self.obj = torch.as_tensor(W.obj)
        rd = D if arm == 'merged' else RD
        if arm == 'nomap':
            self.R_L = p(W.n_rel, RD)
        if arm == 'map':
            self.f = mlp(D, RD)
        self.G_L, self.S_L, self.A_L = mlp(2 * D + rd, D), mlp(D + rd, 2), mlp(D + rd, 1)
        self.G_I, self.S_I = mlp(2 * D + RD, D), mlp(D + RD, 1)

    def op_from_obj(self, e):
        return self.f(e) if self.arm == 'map' else e

    def op(self, r, E=None, R_L=None):
        E = self.E_L if E is None else E
        if self.arm == 'nomap':
            return (self.R_L if R_L is None else R_L)[r]
        return self.op_from_obj(E[self.obj[r]])

    def logic(self, a, r, b, E=None, R_L=None):
        E = self.E_L if E is None else E
        rc = self.op(r, E, R_L)
        h = self.G_L(torch.cat([E[a], E[b], rc], 1))
        return torch.sigmoid(self.S_L(torch.cat([h, rc], 1))), self.A_L(torch.cat([h, rc], 1))[:, 0]

    def intuition(self, a, r, b, R_I=None):
        rc = (self.R_I if R_I is None else R_I)[r]
        return self.S_I(torch.cat([self.G_I(torch.cat([self.E_I[a], self.E_I[b], rc], 1)), rc], 1))[:, 0]


def train(W, arm, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    net = Net(W, arm, g)
    opt = torch.optim.Adam(net.parameters(), lr=TT.LR, weight_decay=TT.WD)
    prob = torch.as_tensor(W.fw / W.fw.sum(), dtype=torch.float)
    T_ = lambda x: torch.as_tensor(x)
    fa, fr, fb, fy, fdrop = T_(W.fa), T_(W.fr), T_(W.fb), T_(W.fy).float(), T_(W.fdrop)
    neg = T_(W.neg)
    syn_train = np.array([ri for ri in W.syn_idx if ri not in W.held])
    rng = np.random.default_rng([seed, 3])
    gs = torch.Generator().manual_seed(seed + 1)
    log = []
    for step in range(1, STEPS + 1):
        idx = torch.multinomial(prob, BATCH, replacement=True, generator=gs)
        drop = torch.as_tensor(rng.choice(syn_train, int(DROP * len(syn_train)), replace=False))
        keep = ~torch.isin(fdrop[idx], drop)
        idx = idx[keep]
        a, r, b, y = fa[idx], fr[idx], fb[idx], fy[idx]
        opt.zero_grad()
        s, al = net.logic(a, r, b)
        target = torch.stack([y, 1 - y], 1)
        l_logic = F.binary_cross_entropy(s, target, reduction='none').sum(1).mean()
        l_int = F.binary_cross_entropy_with_logits(net.intuition(a, r, b), y)
        ni = torch.randint(len(neg), (NEG_BATCH,), generator=gs)
        _, an = net.logic(neg[ni, 0], neg[ni, 1], neg[ni, 2])
        l_app = .5 * (F.binary_cross_entropy_with_logits(al, torch.ones_like(al)) +
                      F.binary_cross_entropy_with_logits(an, torch.zeros_like(an)))
        loss = l_logic + l_int + l_app
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == STEPS:
            log.append({'step': step, 'loss': float(loss), 'logic': float(l_logic), 'app': float(l_app)})
    for p in net.parameters():
        p.requires_grad_(False)
    return net, log


# ---------------------------------------------------------------- reads

def bal(ans, y):
    y = np.asarray(y).astype(bool)
    if not y.any() or y.all():
        return None
    return DR.tf_bal(ans, y)


@torch.no_grad()
def logic_answers(net, trip, E=None, R_L=None):
    t = torch.as_tensor(np.asarray(trip, dtype=np.int64))
    s, al = net.logic(t[:, 0], t[:, 1], t[:, 2], E, R_L)
    return DR.logic_state(s.numpy()), torch.sigmoid(al).numpy()


def world_reads(net, W):
    st = W.st
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    out = {}
    for tst in DR.TF_TESTS:
        m = ind & (st.test == 'P1') & (st.sub == 'default') if tst == 'P1-default' else ind & (st.test == tst)
        idx = np.flatnonzero(m)
        lv, _ = logic_answers(net, np.stack([st.a[idx], st.r[idx], st.b[idx]], 1))
        with torch.no_grad():
            u = net.intuition(*(torch.as_tensor(x[idx]) for x in (st.a, st.r, st.b))).numpy()
        out[tst] = {'logic': bal(lv, st.world_y[idx]), 'intuition': bal(np.where(u > 0, 'T', 'F'), st.world_y[idx])}
    syn = {}
    for ri, s in W.syn_idx.items():
        if ri in W.held:
            continue
        pairs = list(W.lab[ri].items())
        lv, _ = logic_answers(net, [(a, ri, b) for (a, b), _ in pairs])
        syn[s['name']] = bal(lv, [y for _, y in pairs])
    out['trained_synthetic_mean'] = float(np.mean([v for v in syn.values() if v is not None]))
    out['trained_synthetic_by_kind'] = {k: float(np.mean([syn[s['name']] for s in W.syn if s['kind'] == k and s['name'] in syn
                                                          and syn[s['name']] is not None]))
                                        for k in ('inv-base', 'inverse', 'syn-base', 'synonym', 'broad', 'narrow', 'symmetric')
                                        if any(s['kind'] == k and s['name'] in syn for s in W.syn)}
    rows = [i for i, (x, k, y, lab) in enumerate(W.rho_rows)
            if W._rel_of_obj(x) not in W.held and W._rel_of_obj(y) not in W.held]
    rr = [W.rho_rows[i] for i in rows]
    lv, _ = logic_answers(net, [(x, W.R[RHO[k]], y) for x, k, y, _ in rr])
    out['rho_fit'] = {RHO[k]: bal(lv[[j for j, q in enumerate(rr) if q[1] == k]], [q[3] for q in rr if q[1] == k])
                      for k in range(5)}
    return out


def fit_and_read(net, W, h, cond, k, seed):
    """Introduce held-out relation h; returns reads."""
    rng = np.random.default_rng([seed, h, 17])
    pairs = list(W.lab[h].items())
    ys = np.array([y for _, y in pairs])
    ti, fi = np.flatnonzero(ys == 1), np.flatnonzero(ys == 0)
    gt = rng.choice(ti, min(4, len(ti)), replace=False)
    gf = rng.choice(fi, min(4, len(fi)), replace=False)
    given = list(gt[:k]) + list(gf[:k])
    inst = [(pairs[i][0][0], h, pairs[i][0][1], pairs[i][1]) for i in given]
    rho = []
    if cond != 'none':
        for x, kk, y, lab in W.rho_rows:
            rx, ry = W._rel_of_obj(x), W._rel_of_obj(y)
            if h not in (rx, ry):
                continue
            other = ry if rx == h else rx
            if other != h and other in W.held:
                continue
            kk2 = SHUFFLE[kk] if cond == 'shuffled' else kk
            rho.append((x, W.R[RHO[kk2]], y, lab))
    data = inst + rho
    obj_h = int(W.obj[h])
    E0 = net.E_L.detach()
    trained_objs = torch.as_tensor([int(W.obj[r]) for r in range(W.n_rel) if r not in W.held])
    mu_e, P_e = V2.gaussian_prior(E0[trained_objs])
    e_h = nn.Parameter(mu_e.clone())
    params = [e_h]
    if net.arm == 'nomap':
        keep = torch.as_tensor([r for r in range(W.n_rel) if r not in W.held])
        mu_r, P_r = V2.gaussian_prior(net.R_L.detach()[keep])
        r_h = nn.Parameter(mu_r.clone())
        params.append(r_h)
    keepI = torch.as_tensor([r for r in range(W.n_rel) if r not in W.held])
    mu_i, P_i = V2.gaussian_prior(net.R_I.detach()[keepI])
    r_i = nn.Parameter(mu_i.clone())

    def tables():
        E = torch.cat([E0[:obj_h], e_h[None], E0[obj_h + 1:]])
        RL = None
        if net.arm == 'nomap':
            R0 = net.R_L.detach()
            RL = torch.cat([R0[:h], r_h[None], R0[h + 1:]])
        return E, RL
    if data:
        t = torch.as_tensor(np.array(data, dtype=np.int64))
        opt = torch.optim.Adam(params, lr=V2.FIT_LR)
        for _ in range(FIT_STEPS):
            opt.zero_grad()
            E, RL = tables()
            s, al = net.logic(t[:, 0], t[:, 1], t[:, 2], E, RL)
            yy = t[:, 3].float()
            loss = F.binary_cross_entropy(s, torch.stack([yy, 1 - yy], 1), reduction='sum')
            loss = loss + F.binary_cross_entropy_with_logits(al, torch.ones_like(al), reduction='sum')
            de = e_h - mu_e
            loss = loss + .5 * de @ P_e @ de
            if net.arm == 'nomap':
                dr = r_h - mu_r
                loss = loss + .5 * dr @ P_r @ dr
            loss.backward()
            opt.step()
    E, RL = tables()
    E, RL = E.detach(), None if RL is None else RL.detach()
    given_set = {pairs[i][0] for i in given}
    test = [(p, y) for p, y in pairs if p not in given_set]
    lv, app = logic_answers(net, [(a, h, b) for (a, b), _ in test], E, RL)
    ty = np.array([y for _, y in test])
    rec = {'bal': bal(lv, ty), 'kept_meaningful': float((app >= .5).mean())}
    M = W.masks.get(h)
    if M is not None:
        xy = np.argwhere(M)
        sel = xy[rng.choice(len(xy), min(300, len(xy)), replace=False)]
        _, app_m = logic_answers(net, [(x, h, y) for x, y in sel], E, RL)
        rec['recognized_meaningless'] = float((app_m < .5).mean())
    s = W.syn_idx.get(h)
    if s is not None and s['kind'] in ('broad', 'narrow'):
        partner = [ri for ri, q in W.syn_idx.items() if q['fam'] == s['fam'] and ri != h][0]
        pt = W.true[partner]
        if s['kind'] == 'broad':
            m = np.array([p in pt for p, _ in test])
            rec['implied_T_recall'] = float((lv[m] == 'T').mean()) if m.any() else None
        else:
            plab = W.lab[partner]
            m = np.array([plab.get(p, 1) == 0 for p, _ in test])
            rec['implied_F_recall'] = float((lv[m] == 'F').mean()) if m.any() else None
    if s is not None and s['kind'] == 'symmetric' and given:
        rv = [((b, a), W.lab[h][(b, a)]) for (a, b) in given_set if (b, a) in W.lab[h] and (b, a) not in given_set]
        if rv:
            lr_, _ = logic_answers(net, [(a, h, b) for (a, b), _ in rv], E, RL)
            rec['reverse_acc'] = float(np.mean([(v == 'T') == (y == 1) for v, (_, y) in zip(lr_, rv)]))
    # intuition, fitted from instances only (same for every condition)
    if cond == 'rho':
        if inst:
            ti_ = torch.as_tensor(np.array(inst, dtype=np.int64))
            opt = torch.optim.Adam([r_i], lr=V2.FIT_LR)
            R0 = net.R_I.detach()
            for _ in range(FIT_STEPS):
                opt.zero_grad()
                RI = torch.cat([R0[:h], r_i[None], R0[h + 1:]])
                loss = F.binary_cross_entropy_with_logits(net.intuition(ti_[:, 0], ti_[:, 1], ti_[:, 2], RI),
                                                          ti_[:, 3].float(), reduction='sum')
                dr = r_i - mu_i
                loss = loss + .5 * dr @ P_i @ dr
                loss.backward()
                opt.step()
        R0 = net.R_I.detach()
        RI = torch.cat([R0[:h], r_i.detach()[None], R0[h + 1:]])
        tt = torch.as_tensor(np.array([(a, h, b) for (a, b), _ in test], dtype=np.int64))
        with torch.no_grad():
            u = net.intuition(tt[:, 0], tt[:, 1], tt[:, 2], RI).numpy()
        rec['intuition_bal'] = bal(np.where(u > 0, 'T', 'F'), ty)
    rec['n_rho_facts'] = len(rho)
    return rec


def kind_of(W, h):
    if h in W.syn_idx:
        return W.syn_idx[h]['kind']
    return 'real-' + W.rels[h]


# ---------------------------------------------------------------- tasks

def run(arm, seed):
    torch.set_num_threads(1)
    t0 = time.process_time()
    W = World(seed)
    build_cpu = time.process_time() - t0
    t0 = time.process_time()
    net, log = train(W, arm, seed)
    train_cpu = time.process_time() - t0
    t0 = time.process_time()
    res = {'arm': arm, 'seed': seed, 'log': log, 'world': world_reads(net, W),
           'counts': {'facts': int(len(W.fa)), 'rho_rows_train': int((W.fg == 2).sum()), 'synthetic_rows': int((W.fg == 1).sum()),
                      'world_rows': int((W.fg == 0).sum()), 'meaningless': int(len(W.neg)),
                      'rho_true_by_k': {RHO[k]: int(sum(1 for q in W.rho_rows if q[1] == k and q[3] == 1)) for k in range(5)}},
           'held': {}}
    for h in W.held:
        name = W.rels[h]
        res['held'][name] = {'kind': kind_of(W, h), 'reads': {}}
        for cond in CONDS:
            for k in KS:
                res['held'][name]['reads'][f'{cond}/k{k}'] = fit_and_read(net, W, h, cond, k, seed)
    test_cpu = time.process_time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(net.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'build_cpu_seconds': build_cpu, 'train_cpu_seconds': train_cpu, 'test_cpu_seconds': test_cpu,
                 'params': int(sum(p.numel() for p in net.parameters())), 'checkpoint_bytes': ck.stat().st_size,
                 'steps': STEPS, 'fit_steps': FIT_STEPS, 'fits': len(W.held) * len(CONDS) * len(KS)}


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
    V2.save_json(f'{REPORT}.json', {'status': 'relations as objects v1' + (' (smoke)' if SMOKE else ''),
                                    'results': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    m = lambda v: None if not [x for x in v if x is not None] else round(float(np.mean([x for x in v if x is not None])), 3)
    out = {}
    keys = ('bal', 'recognized_meaningless', 'kept_meaningful', 'implied_T_recall', 'implied_F_recall', 'reverse_acc',
            'intuition_bal')
    groups = {'inverse+synonym (synthetic)': ('inverse', 'synonym'), 'real inverse': ('real-child-of', 'real-darker-than'),
              'broad': ('broad',), 'narrow': ('narrow',), 'symmetric': ('symmetric',)}
    for arm in ARMS:
        R = [r for r in res.values() if r['arm'] == arm]
        if not R:
            continue
        o = {'world': {t: {q: m([r['world'][t][q] for r in R]) for q in ('logic', 'intuition')} for t in DR.TF_TESTS},
             'trained_synthetic_mean': m([r['world']['trained_synthetic_mean'] for r in R]),
             'trained_synthetic_by_kind': {k: m([r['world']['trained_synthetic_by_kind'].get(k) for r in R])
                                           for k in R[0]['world']['trained_synthetic_by_kind']},
             'rho_fit': {k: m([r['world']['rho_fit'][k] for r in R]) for k in RHO}, 'held': {}}
        o['world_logic_mean'] = m([v['logic'] for v in o['world'].values()])
        o['world_intuition_mean'] = m([v['intuition'] for v in o['world'].values()])
        for gname, kinds in groups.items():
            o['held'][gname] = {}
            for cond in CONDS:
                for k in KS:
                    vals = [rel['reads'][f'{cond}/k{k}'] for r in R for rel in r['held'].values() if rel['kind'] in kinds]
                    o['held'][gname][f'{cond}/k{k}'] = {q: m([v.get(q) for v in vals]) for q in keys}
        out[arm] = o
        print(f'\n===== {arm}: world logic {o["world_logic_mean"]} intuition {o["world_intuition_mean"]}; trained synthetic '
              f'{o["trained_synthetic_mean"]} {o["trained_synthetic_by_kind"]}; rho fit {o["rho_fit"]}')
        for gname in groups:
            print(f'  {gname}')
            for k in KS:
                line = f'    k={k} '
                for cond in CONDS:
                    v = o['held'][gname][f'{cond}/k{k}']
                    extra = ''.join(f' {q[:8]} {v[q]}' for q in ('implied_T_recall', 'implied_F_recall', 'reverse_acc') if v[q] is not None)
                    line += f"| {cond}: bal {v['bal']} M {v['recognized_meaningless']} kept {v['kept_meaningful']}{extra} "
                line += f"| intuition {o['held'][gname][f'rho/k{k}']['intuition_bal']}"
                print(line)
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
