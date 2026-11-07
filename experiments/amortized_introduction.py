"""Amortized introduction v1 (docs/amortized-introduction-v1.md): an introduction encoder builds a relation's operator
code from its description (rho statements against trained relations and categories) and is trained in introduction
episodes on the relation's instance answers. Arms: episodes, virtual (plus virtual copy/inverse relations), regress
(encoder regressed onto stored codes).

Usage: EV_WORKERS=6 .venv/bin/python -m experiments.amortized_introduction        (rerun to resume)
       .venv/bin/python -m experiments.amortized_introduction summary
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
from experiments import relations_as_objects as RO
from experiments import taught_truth as TT

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'amortized-introduction-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'amortized-introduction-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = int(os.environ.get('EV_STEPS', 100)) if SMOKE else 3000
FIT_STEPS = 60 if SMOKE else 500
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('episodes', 'virtual', 'regress')
KS = (0, 4)
BATCH, NEG_BATCH, INTRO = RO.BATCH, RO.NEG_BATCH, 0.3
N_VIRT, VIRT_T = 16, 32                      # virtual relations per step; T (and F) pairs per virtual relation
N_F = 64                                     # sampled F statements per description
SPLIT_SEED = 2028
D, RD, H = RO.D, RO.RD, RO.H
EL, HEADS = 64, 4
REAL_B = ('parent-of', 'brighter-than')


# ---------------------------------------------------------------- descriptions

def rho_between(labX, trueX, labY, trueY):
    """rho-1..3 labels of (X, Y), as RO.World.rho_facts."""
    revL = {(b, a) for a, b in labX}
    revT = {(b, a) for a, b in trueX}
    LY = set(labY)
    out = []
    for LX, TX, rev in ((revL, revT, True), (set(labX), set(trueX), False)):
        O = LX & LY
        if len(O) < .5 * min(len(LX), len(LY)):
            out.append(None)
        else:
            out.append((TX & O, set(trueY) & O))
    inv, same = out
    l1 = int(inv is not None and inv[0] == inv[1] and len(inv[0]) >= 1)
    l2 = int(same is not None and len(same[0]) >= 2 and same[0] <= same[1])
    l3 = int(same is not None and len(same[0]) >= 1 and same[0] == same[1])
    return l1, l2, l3


class Desc:
    """Statement pools of one relation: always-included rows and an F pool. Row = (rho, dir, label, kind, idx);
    kind 0 relation partner (idx relation), 1 category (idx entity), 2 self."""
    def __init__(self, fixed, fpool):
        self.fixed, self.fpool = fixed, fpool

    def sample(self, rng, shuffle=False):
        f = self.fpool
        if len(f) > N_F:
            f = [f[i] for i in rng.choice(len(f), N_F, replace=False)]
        rows = self.fixed + f
        if shuffle:
            rows = [(RO.SHUFFLE[r[0]],) + r[1:] for r in rows]
        return rows


class Intro:
    """Descriptions of real relations (from v1's rho rows), virtual relations and the A/B split."""
    def __init__(self, W, seed):
        self.W = W
        self.trained = [r for r in W.meta if r not in W.held]
        tset = set(self.trained)
        per = {r: ([], []) for r in W.meta}
        for x, k, y, lab in W.rho_rows:
            rx = W._rel_of_obj(x)
            if k >= 3:
                per[rx][0].append((k, 0, 2 * lab - 1, 1, int(y)))
                continue
            ry = W._rel_of_obj(y)
            if rx == ry:
                per[rx][0].append((0, 0, 2 * lab - 1, 2, 0))
                continue
            for me, other, d in ((rx, ry, 0), (ry, rx, 1)):
                if other not in tset:
                    continue
                row = (k, d, 2 * lab - 1, 0, other)
                (per[me][0] if lab else per[me][1]).append(row)
        self.desc = {r: Desc(*per[r]) for r in W.meta}
        # A/B split of trained relations for virtual relations
        B = set()
        for h in W.held:
            s = W.syn_idx.get(h)
            if s is not None:
                B |= {ri for ri, q in W.syn_idx.items() if q['fam'] == s['fam'] and ri not in W.held}
        B |= {W.R[x] for x in REAL_B}
        rest = sorted(tset - B)
        g = np.random.default_rng(SPLIT_SEED)
        B |= set(g.choice(rest, len(rest) // 2, replace=False).tolist())
        self.B = sorted(B & tset)
        self.A = sorted(tset - B)
        # training extension of real relations (training split) and full extension
        train_lab = {}
        for a, r, b, y, grp in zip(W.fa, W.fr, W.fb, W.fy, W.fg):
            if grp == 0:
                train_lab.setdefault(int(r), {})[(int(a), int(b))] = int(y)
        self.dropped = []
        self.virt_train = [v for Q in self.A for v in self.virtuals(Q, train_lab.get(Q, W.lab[Q]) if Q < len(W.real)
                                                                       else W.lab[Q], check_leak=True)]
        self.virt_test_B = [v for Q in self.B for v in self.virtuals(Q, W.lab[Q])]
        self.virt_test_A = [v for Q in self.A for v in self.virtuals(Q, W.lab[Q])]

    def describe(self, lab):
        W = self.W
        true = {p for p, v in lab.items() if v}
        fixed, fpool = [], []
        fixed.append((0, 0, 2 * rho_between(lab, true, lab, true)[0] - 1, 2, 0))
        for Y in self.trained:
            for d, (l1, l2, l3) in ((0, rho_between(lab, true, W.lab[Y], W.true[Y])),
                                    (1, rho_between(W.lab[Y], W.true[Y], lab, true))):
                for k, l in enumerate((l1, l2, l3)):
                    (fixed if l else fpool).append((k, d, 2 * l - 1, 0, Y))
        subj, obj = {a for a, b in lab}, {b for a, b in lab}
        for c, ce in W.cat_ent.items():
            m = set(W.members[c])
            fixed.append((3, 0, 2 * int(len(m & subj) >= .5 * len(m)) - 1, 1, ce))
            fixed.append((4, 0, 2 * int(len(m & obj) >= .5 * len(m)) - 1, 1, ce))
        return Desc(fixed, fpool)

    def virtuals(self, Q, labQ, check_leak=False):
        W = self.W
        out = []
        for typ in ('copy', 'inverse'):
            lab = dict(labQ) if typ == 'copy' else {(b, a): y for (a, b), y in labQ.items()}
            ys = np.array(list(lab.values()))
            if not ys.any() or ys.all():
                continue
            true = {p for p, v in lab.items() if v}
            if check_leak:
                leak = False
                for h in W.held:
                    l1, _, l3 = rho_between(lab, true, W.lab[h], W.true[h])
                    if l1 or l3:
                        leak = True
                if leak:
                    self.dropped.append(f'{typ}-of-{W.rels[Q]}')
                    continue
            pairs = np.array(list(lab.keys()), dtype=np.int64)
            out.append(dict(name=f'{typ}-of-{W.rels[Q]}', Q=Q, typ=typ, desc=self.describe(lab), pairs=pairs,
                            y=ys.astype(np.int64)))
        return out


def pack(descs):
    """List of statement lists -> padded tensors."""
    L = max(len(d) for d in descs)
    arr = np.zeros((len(descs), L, 5), dtype=np.int64)
    mask = np.zeros((len(descs), L), dtype=bool)
    for i, d in enumerate(descs):
        arr[i, :len(d)] = d
        mask[i, :len(d)] = True
    return torch.as_tensor(arr), torch.as_tensor(mask)


# ---------------------------------------------------------------- model

class Encoder(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.rho = nn.Parameter(torch.randn(5, 8, generator=g) * TT.INIT_STD)
        self.self_tok = nn.Parameter(torch.randn(RD, generator=g) * TT.INIT_STD)
        self.phi = nn.Sequential(nn.Linear(8 + 2 + 3 + RD + D, H), nn.ReLU(), nn.Linear(H, EL))
        self.K, self.V = nn.Linear(EL, EL), nn.Linear(EL, EL)
        self.q = nn.Parameter(torch.randn(HEADS, EL // HEADS, generator=g))
        self.out = nn.Sequential(nn.Linear(EL, H), nn.ReLU(), nn.Linear(H, RD))

    def forward(self, arr, mask, R_L, E_L):
        rho, d, lab, kind, idx = arr.unbind(-1)
        is_r, is_c, is_s = (kind == 0), (kind == 1), (kind == 2)
        rpart = torch.where(is_r[..., None], R_L[torch.where(is_r, idx, 0)], 0.) + is_s[..., None] * self.self_tok
        cpart = torch.where(is_c[..., None], E_L[torch.where(is_c, idx, 0)], 0.)
        x = torch.cat([self.rho[rho], d[..., None].float(), lab[..., None].float(),
                       F.one_hot(kind, 3).float(), rpart, cpart], -1)
        h = self.phi(x)
        B, L, _ = h.shape
        k = self.K(h).view(B, L, HEADS, -1)
        v = self.V(h).view(B, L, HEADS, -1)
        s = torch.einsum('blhd,hd->blh', k, self.q) / (k.shape[-1] ** .5)
        s = s.masked_fill(~mask[..., None], float('-inf'))
        a = torch.softmax(s, 1)
        return self.out(torch.einsum('blh,blhd->bhd', a, v).reshape(B, -1))


class Model(nn.Module):
    def __init__(self, W, g):
        super().__init__()
        self.net = RO.Net(W, 'nomap', g)
        self.enc = Encoder(g)

    def encode(self, descs):
        arr, mask = pack(descs)
        return self.enc(arr, mask, self.net.R_L, self.net.E_L)


def train(W, I, arm, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Model(W, g)
    net = model.net
    opt = torch.optim.Adam([{'params': net.parameters(), 'weight_decay': TT.WD},
                            {'params': model.enc.parameters(), 'weight_decay': 0.}], lr=TT.LR)
    keep = W.fg != 2                                                   # rho rows leave the batch
    fa, fr, fb, fy = (torch.as_tensor(x[keep]) for x in (W.fa, W.fr, W.fb, W.fy))
    fy = fy.float()
    cells = {}
    for i, key in enumerate(zip(fr.tolist(), fy.tolist())):
        cells.setdefault(key, []).append(i)
    w = np.zeros(len(fr))
    for idx in cells.values():
        w[idx] = 1 / len(idx)
    prob = torch.as_tensor(w / w.sum(), dtype=torch.float)
    neg = torch.as_tensor(W.neg)
    trained = np.array(I.trained)
    n_intro = int(INTRO * len(trained))
    rng = np.random.default_rng([seed, 3])
    gs = torch.Generator().manual_seed(seed + 1)
    log = []
    n_rel = W.n_rel
    for step in range(1, STEPS + 1):
        idx = torch.multinomial(prob, BATCH, replacement=True, generator=gs)
        a, r, b, y = fa[idx], fr[idx], fb[idx], fy[idx]
        ni = torch.randint(len(neg), (NEG_BATCH,), generator=gs)
        na, nr, nb = neg[ni, 0], neg[ni, 1], neg[ni, 2]
        intro = rng.choice(trained, n_intro, replace=False)
        descs = [I.desc[x].sample(rng) for x in intro]
        virt = []
        if arm == 'virtual':
            virt = [I.virt_train[i] for i in rng.choice(len(I.virt_train), N_VIRT, replace=False)]
            descs += [v['desc'].sample(rng) for v in virt]
        opt.zero_grad()
        codes = model.encode(descs)
        R_eff = net.R_L
        l_reg = torch.zeros(())
        if arm == 'regress':
            tgt = net.R_L[torch.as_tensor(intro)].detach()
            l_reg = ((codes - tgt) ** 2).sum(1).mean() / (net.R_L.detach() ** 2).sum(1).mean()
        else:
            R_eff = net.R_L.clone()
            R_eff[torch.as_tensor(intro)] = codes[:n_intro]
        if virt:
            va, vb, vy, vr = [], [], [], []
            for j, v in enumerate(virt):
                ti, fi = np.flatnonzero(v['y'] == 1), np.flatnonzero(v['y'] == 0)
                sel = np.concatenate([rng.choice(ti, VIRT_T), rng.choice(fi, VIRT_T)])
                va.append(v['pairs'][sel, 0]); vb.append(v['pairs'][sel, 1]); vy.append(v['y'][sel])
                vr.append(np.full(len(sel), n_rel + j))
            R_eff = torch.cat([R_eff, codes[n_intro:]])
            T_ = lambda x: torch.as_tensor(np.concatenate(x))
            a2, r2, b2, y2 = torch.cat([a, T_(va)]), torch.cat([r, T_(vr)]), torch.cat([b, T_(vb)]), torch.cat([y, T_(vy).float()])
        else:
            a2, r2, b2, y2 = a, r, b, y
        s, al = net.logic(a2, r2, b2, None, R_eff)
        l_logic = F.binary_cross_entropy(s, torch.stack([y2, 1 - y2], 1), reduction='none').sum(1).mean()
        l_int = F.binary_cross_entropy_with_logits(net.intuition(a, r, b), y)
        _, an = net.logic(na, nr, nb, None, R_eff)
        al_real = al[:len(a)]
        l_app = .5 * (F.binary_cross_entropy_with_logits(al_real, torch.ones_like(al_real)) +
                      F.binary_cross_entropy_with_logits(an, torch.zeros_like(an)))
        loss = l_logic + l_int + l_app + l_reg
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == STEPS:
            log.append({'step': step, 'loss': loss.item(), 'logic': l_logic.item(), 'app': l_app.item(),
                        'reg': l_reg.item(), 'code_std': codes.detach().std(0).mean().item()})
    for p in model.parameters():
        p.requires_grad_(False)
    return model, log


# ---------------------------------------------------------------- reads

def with_code(net, h, c):
    R = net.R_L.detach()
    if h >= len(R):
        return torch.cat([R, c[None]])
    return torch.cat([R[:h], c[None], R[h + 1:]])


def fit_code(net, h, data, init, mu, P):
    """MAP fit of relation h's operator code on instance rows (a, h, b, y)."""
    c = nn.Parameter(init.clone())
    t = torch.as_tensor(np.array(data, dtype=np.int64))
    opt = torch.optim.Adam([c], lr=V2.FIT_LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        s, al = net.logic(t[:, 0], t[:, 1], t[:, 2], None, with_code(net, h, c))
        yy = t[:, 3].float()
        loss = F.binary_cross_entropy(s, torch.stack([yy, 1 - yy], 1), reduction='sum')
        loss = loss + F.binary_cross_entropy_with_logits(al, torch.ones_like(al), reduction='sum')
        dc = c - mu
        loss = loss + .5 * dc @ P @ dc
        loss.backward()
        opt.step()
    return c.detach()


def read_held(model, W, I, h, seed):
    net = model.net
    rng = np.random.default_rng([seed, h, 17])
    pairs = list(W.lab[h].items())
    ys = np.array([y for _, y in pairs])
    ti, fi = np.flatnonzero(ys == 1), np.flatnonzero(ys == 0)
    gt = rng.choice(ti, min(4, len(ti)), replace=False)
    gf = rng.choice(fi, min(4, len(fi)), replace=False)
    drng = np.random.default_rng([seed, h, 23])
    desc = I.desc[h].sample(drng)
    desc_sh = [(RO.SHUFFLE[r[0]],) + r[1:] for r in desc]
    with torch.no_grad():
        c_rho, c_sh = model.encode([desc, desc_sh])
    keep = torch.as_tensor([r for r in range(W.n_rel) if r not in W.held])
    mu, P = V2.gaussian_prior(net.R_L.detach()[keep])
    out = {}
    for k in KS:
        given = list(gt[:k]) + list(gf[:k])
        inst = [(pairs[i][0][0], h, pairs[i][0][1], pairs[i][1]) for i in given]
        given_set = {pairs[i][0] for i in given}
        test = [(p, y) for p, y in pairs if p not in given_set]
        conds = {'rho': c_rho if k == 0 else fit_code(net, h, inst, c_rho, c_rho, P),
                 'none': mu if k == 0 else fit_code(net, h, inst, mu, mu, P)}
        if k == 0:
            conds['shuffled'] = c_sh
        for cond, c in conds.items():
            out[f'{cond}/k{k}'] = held_reads(net, W, h, with_code(net, h, c), test, given_set, rng)
        if k == 4:
            out['intuition/k4'] = intuition_read(net, W, h, inst, test)
    return out


def held_reads(net, W, h, RL, test, given_set, rng):
    lv, app = RO.logic_answers(net, [(a, h, b) for (a, b), _ in test], None, RL)
    ty = np.array([y for _, y in test])
    rec = {'bal': RO.bal(lv, ty), 'kept_meaningful': float((app >= .5).mean())}
    M = W.masks.get(h)
    if M is not None:
        xy = np.argwhere(M)
        sel = xy[np.random.default_rng(h).choice(len(xy), min(300, len(xy)), replace=False)]
        _, app_m = RO.logic_answers(net, [(x, h, y) for x, y in sel], None, RL)
        rec['recognized_meaningless'] = float((app_m < .5).mean())
    s = W.syn_idx.get(h)
    if s is not None and s['kind'] in ('broad', 'narrow'):
        partner = [ri for ri, q in W.syn_idx.items() if q['fam'] == s['fam'] and ri != h][0]
        if s['kind'] == 'broad':
            m = np.array([p in W.true[partner] for p, _ in test])
            rec['implied_T_recall'] = float((lv[m] == 'T').mean()) if m.any() else None
        else:
            m = np.array([W.lab[partner].get(p, 1) == 0 for p, _ in test])
            rec['implied_F_recall'] = float((lv[m] == 'F').mean()) if m.any() else None
    if s is not None and s['kind'] == 'symmetric' and given_set:
        rv = [((b, a), W.lab[h][(b, a)]) for (a, b) in given_set if (b, a) in W.lab[h] and (b, a) not in given_set]
        if rv:
            lr_, _ = RO.logic_answers(net, [(a, h, b) for (a, b), _ in rv], None, RL)
            rec['reverse_acc'] = float(np.mean([(v == 'T') == (y == 1) for v, (_, y) in zip(lr_, rv)]))
    return rec


def intuition_read(net, W, h, inst, test):
    keep = torch.as_tensor([r for r in range(W.n_rel) if r not in W.held])
    mu, P = V2.gaussian_prior(net.R_I.detach()[keep])
    c = nn.Parameter(mu.clone())
    t = torch.as_tensor(np.array(inst, dtype=np.int64))
    R0 = net.R_I.detach()
    opt = torch.optim.Adam([c], lr=V2.FIT_LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        RI = torch.cat([R0[:h], c[None], R0[h + 1:]])
        loss = F.binary_cross_entropy_with_logits(net.intuition(t[:, 0], t[:, 1], t[:, 2], RI), t[:, 3].float(),
                                                  reduction='sum') + .5 * (c - mu) @ P @ (c - mu)
        loss.backward()
        opt.step()
    RI = torch.cat([R0[:h], c.detach()[None], R0[h + 1:]])
    tt = torch.as_tensor(np.array([(a, h, b) for (a, b), _ in test], dtype=np.int64))
    with torch.no_grad():
        u = net.intuition(tt[:, 0], tt[:, 1], tt[:, 2], RI).numpy()
    return {'bal': RO.bal(np.where(u > 0, 'T', 'F'), [y for _, y in test])}


def read_virtuals(model, W, virts, seed, shuffled=True):
    net = model.net
    out = {}
    rng = np.random.default_rng([seed, 31])
    for v in virts:
        desc = v['desc'].sample(rng)
        ds = [desc] + ([[(RO.SHUFFLE[r[0]],) + r[1:] for r in desc]] if shuffled else [])
        with torch.no_grad():
            codes = model.encode(ds)
        rec = {'typ': v['typ'], 'Q': W.rels[v['Q']]}
        trip = [(a, W.n_rel, b) for a, b in v['pairs']]
        for cond, c in zip(('rho', 'shuffled'), codes):
            lv, _ = RO.logic_answers(net, trip, None, with_code(net, W.n_rel, c))
            rec[cond] = RO.bal(lv, v['y'])
        out[v['name']] = rec
    return out


def read_trained_intro(model, W, I, seed):
    """Trained relations introduced through the encoder (no fit): the memorization side of the gap."""
    net = model.net
    rng = np.random.default_rng([seed, 37])
    out = {}
    for r in I.trained:
        with torch.no_grad():
            c = model.encode([I.desc[r].sample(rng)])[0]
        pairs = list(W.lab[r].items())
        lv, _ = RO.logic_answers(net, [(a, r, b) for (a, b), _ in pairs], None, with_code(net, r, c))
        lv0, _ = RO.logic_answers(net, [(a, r, b) for (a, b), _ in pairs])
        ys = [y for _, y in pairs]
        out[W.rels[r]] = {'kind': RO.kind_of(W, r), 'set': 'A' if r in I.A else 'B',
                          'encoded': RO.bal(lv, ys), 'stored': RO.bal(lv0, ys)}
    return out


# ---------------------------------------------------------------- tasks

def run(arm, seed):
    torch.set_num_threads(1)
    t0 = time.process_time()
    W = RO.World(seed)
    I = Intro(W, seed)
    build_cpu = time.process_time() - t0
    t0 = time.process_time()
    model, log = train(W, I, arm, seed)
    train_cpu = time.process_time() - t0
    t0 = time.process_time()
    world = RO.world_reads(model.net, W)
    world.pop('rho_fit', None)
    res = {'arm': arm, 'seed': seed, 'log': log, 'world': world,
           'counts': {'trained_described': len(I.trained), 'set_A': len(I.A), 'set_B': len(I.B),
                      'virtual_train': len(I.virt_train), 'virtual_dropped_leak': I.dropped,
                      'virtual_test_B': len(I.virt_test_B), 'virtual_test_A': len(I.virt_test_A),
                      'desc_sizes_held': {W.rels[h]: len(I.desc[h].fixed) + min(N_F, len(I.desc[h].fpool))
                                          for h in W.held}},
           'held': {W.rels[h]: {'kind': RO.kind_of(W, h), 'reads': read_held(model, W, I, h, seed)} for h in W.held},
           'virtual_B': read_virtuals(model, W, I.virt_test_B, seed),
           'virtual_A': read_virtuals(model, W, I.virt_test_A, seed, shuffled=False),
           'trained_intro': read_trained_intro(model, W, I, seed)}
    test_cpu = time.process_time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'build_cpu_seconds': build_cpu, 'train_cpu_seconds': train_cpu, 'test_cpu_seconds': test_cpu,
                 'params': int(sum(p.numel() for p in model.parameters())),
                 'encoder_params': int(sum(p.numel() for p in model.enc.parameters())),
                 'checkpoint_bytes': ck.stat().st_size, 'steps': STEPS, 'fit_steps': FIT_STEPS,
                 'fits': len(W.held) * 3, 'encoder_reads': len(W.held) * 2 + 2 * len(I.virt_test_B) +
                 len(I.virt_test_A) + len(I.trained)}


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
    V2.save_json(f'{REPORT}.json', {'status': 'amortized introduction v1' + (' (smoke)' if SMOKE else ''),
                                    'results': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    m = lambda v: None if not [x for x in v if x is not None] else round(float(np.mean([x for x in v if x is not None])), 3)
    keys = ('bal', 'recognized_meaningless', 'kept_meaningful', 'implied_T_recall', 'implied_F_recall', 'reverse_acc')
    groups = {'inverse+synonym (synthetic)': ('inverse', 'synonym'), 'inverse (synthetic)': ('inverse',),
              'synonym (synthetic)': ('synonym',), 'real inverse': ('real-child-of', 'real-darker-than'),
              'broad': ('broad',), 'narrow': ('narrow',), 'symmetric': ('symmetric',)}
    conds = ('rho/k0', 'shuffled/k0', 'none/k0', 'rho/k4', 'none/k4', 'intuition/k4')
    out = {}
    for arm in ARMS:
        R = [r for r in res.values() if r['arm'] == arm]
        if not R:
            continue
        o = {'world_logic_mean': m([np.mean([r['world'][t]['logic'] for t in RO.DR.TF_TESTS
                                             if r['world'][t]['logic'] is not None]) for r in R]),
             'world_intuition_mean': m([np.mean([r['world'][t]['intuition'] for t in RO.DR.TF_TESTS
                                                 if r['world'][t]['intuition'] is not None]) for r in R]),
             'trained_synthetic_mean': m([r['world']['trained_synthetic_mean'] for r in R]), 'held': {}}
        for gname, kinds in groups.items():
            o['held'][gname] = {}
            for c in conds:
                vals = [rel['reads'][c] for r in R for rel in r['held'].values() if rel['kind'] in kinds]
                o['held'][gname][c] = {q: m([v.get(q) for v in vals]) for q in keys}
        for setn in ('virtual_B', 'virtual_A'):
            o[setn] = {typ: {c: m([v.get(c) for r in R for v in r[setn].values() if v['typ'] == typ])
                             for c in ('rho', 'shuffled')} for typ in ('copy', 'inverse')}
        o['trained_intro'] = {}
        for setn in ('A', 'B'):
            for kinds, lab in ((None, 'all'), (('inverse', 'inv-base'), 'inverse'), (('synonym', 'syn-base'), 'synonym')):
                vals = [v for r in R for v in r['trained_intro'].values() if v['set'] == setn and
                        (kinds is None or v['kind'] in kinds)]
                o['trained_intro'][f'{setn}/{lab}'] = {'encoded': m([v['encoded'] for v in vals]),
                                                       'stored': m([v['stored'] for v in vals])}
        out[arm] = o
        print(f"\n===== {arm}: world logic {o['world_logic_mean']} intuition {o['world_intuition_mean']}; "
              f"trained synthetic {o['trained_synthetic_mean']}")
        for gname in groups:
            print(f'  {gname}: ' + ' | '.join(
                f"{c} {o['held'][gname][c]['bal']}" +
                ''.join(f" {q[:8]} {o['held'][gname][c][q]}" for q in keys[1:] if o['held'][gname][c][q] is not None)
                for c in conds))
        print(f"  virtual B {o['virtual_B']}  virtual A {o['virtual_A']}")
        print(f"  trained via encoder {o['trained_intro']}")
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
