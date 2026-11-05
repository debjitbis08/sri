"""Everyday relations v2 (docs/everyday-relations-v2.md).

Usage: .venv/bin/python -m experiments.everyday_relations_v2 part0     (fair fast mapping on v1 checkpoints)
       .venv/bin/python -m experiments.everyday_relations_v2 main      (rerun to resume; EV_WORKERS, default 8)
       .venv/bin/python -m experiments.everyday_relations_v2 summary
       EV_SMOKE=1 ... for a short check (100 steps, one seed)
"""
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import number_systems as X
from src import everyday_world as W1
from src import everyday_world_v2 as W

nc, ROOT = X.nc, X.ROOT
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'everyday-relations-v2' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'everyday-relations-v2.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 100 if SMOKE else 3000
FIT_STEPS = 50 if SMOKE else 500
SEEDS = (410,) if SMOKE else (410, 411, 412, 413)
LR, WD, FIT_LR, INIT_STD, D, RD, HID = 3e-3, 1e-4, 1e-2, 0.3, 32, 16, 128
FLAT_HID = 255          # matches tworel's G + S (20,897) within 0.1%
NEG_PER_POS = 4
PRIOR_SHRINK = 0.1
VAL_FRAC = 0.05
NEWREL_K = (4, 16)
CONFIGS = [('two', 'closed', 'real'), ('two', 'open', 'real'), ('tworel', 'closed', 'real'), ('tworel', 'open', 'real'),
           ('flat', 'closed', 'real'), ('flat', 'open', 'real'), ('complex', 'open', 'real'),
           ('two', 'open', 'shuffled'), ('tworel', 'open', 'shuffled'), ('flat', 'open', 'shuffled')]


def log_steps():
    if SMOKE:
        return set(range(25, STEPS + 1, 25))
    return set(range(25, 501, 25)) | set(range(750, STEPS + 1, 250))


# ---------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, arm, n_ent, n_rel, g):
        super().__init__()
        self.arm, self.d = arm, D
        ed, rd = (2 * D, 2 * D) if arm == 'complex' else (D, RD)
        self.E = nn.Parameter(torch.randn(n_ent, ed, generator=g) * INIT_STD)
        self.R = nn.Parameter(torch.randn(n_rel, rd, generator=g) * INIT_STD)
        if arm in ('two', 'tworel'):
            gin = 2 * D + (RD if arm == 'tworel' else 0)
            self.G = nn.Sequential(nn.Linear(gin, HID), nn.ReLU(), nn.Linear(HID, D))
            self.S = nn.Sequential(nn.Linear(D + RD, HID), nn.ReLU(), nn.Linear(HID, 1))
        elif arm == 'flat':
            self.S = nn.Sequential(nn.Linear(2 * D + RD, FLAT_HID), nn.ReLU(), nn.Linear(FLAT_HID, 1))
        R1._reinit(self, g)

    def q(self, ch, rc, ct):
        x = [ch, ct, rc] if self.arm == 'tworel' else [ch, ct]
        return self.G(torch.cat(x, 1))

    def score_codes(self, ch, rc, ct):
        if self.arm == 'complex':
            d = self.d
            hr, hi, rr, ri, tr, ti = ch[:, :d], ch[:, d:], rc[:, :d], rc[:, d:], ct[:, :d], ct[:, d:]
            return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(1)
        if self.arm == 'flat':
            return self.S(torch.cat([ch, ct, rc], 1))[:, 0]
        return self.S(torch.cat([self.q(ch, rc, ct), rc], 1))[:, 0]

    def forward(self, h, r, t):
        return self.score_codes(self.E[h], self.R[r], self.E[t])


@torch.no_grad()
def scores(model, world):
    return model(torch.as_tensor(world.h), torch.as_tensor(world.r), torch.as_tensor(world.t)).numpy().astype(float)


# ---------------------------------------------------------------- code fitting (fast mapping)

def gaussian_prior(E):
    """Mean and shrunk precision of the rows of E: cov' = (1 - a) cov + a (tr cov / d) I."""
    E = E.detach().double()
    mu = E.mean(0)
    cov = torch.cov(E.T)
    d = E.shape[1]
    cov = (1 - PRIOR_SHRINK) * cov + PRIOR_SHRINK * cov.trace() / d * torch.eye(d, dtype=torch.float64)
    return mu.float(), torch.linalg.inv(cov).float()


def fit_code(model, world, name, facts, gen, prior):
    """Fit one new entity code. prior None: mean BCE + weight decay (v1). prior (mu, P): MAP, sum BCE + ½ Mahalanobis²."""
    code = nn.Parameter(torch.randn(1, model.E.shape[1], generator=gen) * INIT_STD)
    opt = torch.optim.Adam([code], lr=FIT_LR, weight_decay=WD if prior is None else 0.)

    def batch(fs):
        ch = torch.cat([code if a == name else model.E[world.index(a)][None] for _, a, _, _ in fs])
        ct = torch.cat([code if c == name else model.E[world.index(c)][None] for _, _, c, _ in fs])
        rc = model.R[[world.relations.index(rel) for rel, _, _, _ in fs]]
        return model.score_codes(ch, rc, ct)

    yt = torch.tensor([float(l) for *_, l in facts])
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        if prior is None:
            loss = F.binary_cross_entropy_with_logits(batch(facts), yt)
        else:
            mu, P = prior
            dlt = code[0] - mu
            loss = F.binary_cross_entropy_with_logits(batch(facts), yt, reduction='sum') + .5 * dlt @ P @ dlt
        loss.backward()
        opt.step()
    return code.detach(), batch


def read_fit(model, world, name, given, read, row, k, seed, prior, cls=None):
    g = torch.Generator().manual_seed(seed * 1000 + 7 * k + len(name))
    for p in model.parameters():
        p.requires_grad_(False)
    facts = given[:k] + list(row)
    code, batch = fit_code(model, world, name, facts, g, prior)
    with torch.no_grad():
        s = batch(read).numpy().astype(float)
        sg = batch(facts).numpy().astype(float)
    rec = {'read': R1.metrics(s, [l for *_, l in read]),
           'given_acc': float(np.mean((sg > 0) == np.array([l for *_, l in facts]))),
           'by_relation': {rel: R1.metrics(s[[i for i, x in enumerate(read) if x[0] == rel]],
                                           [x[3] for x in read if x[0] == rel]) for rel in sorted({x[0] for x in read})}}
    if cls is not None:
        rec['centroid'] = R1.class_rank(model, world, code, cls) if model.E.shape[1] else None
    E = model.E.detach()
    rec['code_norm'], rec['entity_norm_mean'] = float(code.norm()), float(E.norm(dim=1).mean())
    return rec


# ---------------------------------------------------------------- Part 0

def part0_task(t):
    arm, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    world = W1.build()
    model = R1.Net(arm, len(world.entities), len(world.relations), torch.Generator().manual_seed(0))
    model.load_state_dict(torch.load(R1.RUNS / f'{arm}-real-seed{seed}.pt'))
    prior = gaussian_prior(model.E)
    specs = W1.fast_mapping_specs()
    out = {}
    for name, cls in (('dax', 'bird'), ('fep', 'fish')):
        given, read, ks = specs[name]
        row = [('is-a', name, c, False) for c in W1.CLASSES if c != cls] + [('is-a', name, x, False) for x in ('noun', 'verb')]
        read = [x for x in read if x[:3] not in {r_[:3] for r_ in row}]
        for fit in ('v1', 'row', 'row-prior'):
            for k in ks:
                out[f'{name}/{fit}/k{k}'] = read_fit(model, world, name, given, read, row if fit != 'v1' else [], k, seed,
                                                     prior if fit == 'row-prior' else None, cls)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'fits': len(out),
          'fit_steps': FIT_STEPS}
    return f'{arm}/seed{seed}', out, rt


def part0():
    t0 = time.perf_counter()
    tasks = [(a, s) for a in R1.ARMS for s in R1.SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(int(os.environ.get('EV_WORKERS', 8))) as pool:
        res = pool.map(part0_task, tasks, chunksize=1)
    out = {k: r for k, r, _ in res}
    rt = runtime({'wall_seconds': time.perf_counter() - t0,
                  'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                  'tasks': {k: t for k, _, t in res}})
    save_json(ROOT / 'reports' / 'everyday-relations-v2-part0.json', out)
    save_json(ROOT / 'reports' / 'everyday-relations-v2-part0-runtime.json', rt)
    print('arm      fit        dax k1/k2/k4         fep k1/k2/k4        dax norm (entity mean)  dax class rank')
    for arm in R1.ARMS:
        for fit in ('v1', 'row', 'row-prior'):
            v = lambda n, k: np.mean([out[f'{arm}/seed{s}'][f'{n}/{fit}/k{k}']['read']['bal_acc'] for s in R1.SEEDS])
            nm = np.mean([out[f'{arm}/seed{s}'][f'dax/{fit}/k1']['code_norm'] for s in R1.SEEDS])
            en = np.mean([out[f'{arm}/seed{s}'][f'dax/{fit}/k1']['entity_norm_mean'] for s in R1.SEEDS])
            cr = np.mean([out[f'{arm}/seed{s}'][f'dax/{fit}/k1']['centroid']['class_rank'] for s in R1.SEEDS])
            print(f'{arm:8s} {fit:10s} ' + '/'.join(f'{v("dax", k):.2f}' for k in (1, 2, 4)) + '       '
                  + '/'.join(f'{v("fep", k):.2f}' for k in (1, 2, 4)) + f'      {nm:.2f} ({en:.2f})          {cr:.2f}')


# ---------------------------------------------------------------- training

class Sampler:
    """Open-world negatives: each positive gets NEG_PER_POS corruptions (head or tail, uniform over the relation's set);
    known trained positives are redrawn; held-out facts are not (the learner does not know them)."""

    def __init__(self, world, y, train_pos, rng):
        self.rng, self.world = rng, world
        nE, nR = len(world.entities), len(world.relations)
        self.key = lambda h, r, t: (h * nR + r) * nE + t
        self.h, self.r, self.t = world.h[train_pos], world.r[train_pos], world.t[train_pos]
        self.known = np.sort(self.key(self.h, self.r, self.t))
        test = world.test != ''
        self.test_keys = np.sort(self.key(world.h[test], world.r[test], world.t[test]))
        self.test_true_keys = np.sort(self.key(world.h[test & (y == 1)], world.r[test & (y == 1)], world.t[test & (y == 1)]))
        hp, tp = [], []
        self.hs, self.hl, self.ts, self.tl = (np.zeros(nR, int) for _ in range(4))
        for ri, rel in enumerate(world.relations):
            self.hs[ri], self.hl[ri] = len(hp), len(world.heads[rel])
            hp += [world.index(x) for x in world.heads[rel]]
            self.ts[ri], self.tl[ri] = len(tp), len(world.tails[rel])
            tp += [world.index(x) for x in world.tails[rel]]
        self.hp, self.tp = np.array(hp), np.array(tp)
        self.drawn = self.test_hits = self.test_true_hits = 0

    def corrupt(self, h, r, t):
        side = self.rng.random(len(h)) < .5
        u = self.rng.random(len(h))
        nh = np.where(side, self.hp[self.hs[r] + (u * self.hl[r]).astype(int)], h)
        nt = np.where(side, t, self.tp[self.ts[r] + (u * self.tl[r]).astype(int)])
        return nh, nt

    def isin(self, keys, table):
        i = np.searchsorted(table, keys)
        return (i < len(table)) & (table[np.minimum(i, len(table) - 1)] == keys)

    def draw(self):
        h, r, t = (np.repeat(a, NEG_PER_POS) for a in (self.h, self.r, self.t))
        nh, nt = self.corrupt(h, r, t)
        for _ in range(20):
            bad = self.isin(self.key(nh, r, nt), self.known)
            if not bad.any():
                break
            nh[bad], nt[bad] = self.corrupt(h[bad], r[bad], t[bad])
        keys = self.key(nh, r, nt)
        self.drawn += len(keys)
        self.test_hits += int(self.isin(keys, self.test_keys).sum())
        self.test_true_hits += int(self.isin(keys, self.test_true_keys).sum())
        return nh, r, nt


def rel_weights(r, y, n_rel):
    wt = np.zeros(len(r), np.float32)
    for ri in range(n_rel):
        for lab in (0, 1):
            m = (r == ri) & (y == lab)
            if m.any():
                wt[m] = 1 / (2 * m.sum() * n_rel)
    return torch.as_tensor(wt)


def train(world, y, arm, regime, seed, gen):
    trained = world.test == ''
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(trained)
    val = np.zeros(len(y), bool)
    val[rng.choice(idx, int(round(VAL_FRAC * len(idx))), replace=False)] = True
    fit = trained & ~val
    nR = len(world.relations)
    model = Net(arm, len(world.entities), nR, gen)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sampler = None
    if regime == 'closed':
        h, r, t = (torch.as_tensor(a[fit]) for a in (world.h, world.r, world.t))
        yt = torch.as_tensor(y[fit])
        wt = rel_weights(world.r[fit], y[fit], nR)
    else:
        pos = fit & (y == 1)
        sampler = Sampler(world, y, pos, np.random.default_rng(seed + 1))
        ph, pr, pt = world.h[pos], world.r[pos], world.t[pos]
    curve, logs = [], log_steps()
    for step in range(1, STEPS + 1):
        if regime == 'open':
            nh, nr, nt = sampler.draw()
            h, r, t = (torch.as_tensor(np.r_[a, b_]) for a, b_ in ((ph, nh), (pr, nr), (pt, nt)))
            lab = np.r_[np.ones(len(ph)), np.zeros(len(nh))].astype(np.float32)
            yt = torch.as_tensor(lab)
            wt = rel_weights(np.r_[pr, nr], lab, nR)
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(h, r, t), yt, reduction='none') * wt).sum()
        loss.backward()
        opt.step()
        if step in logs:
            s = scores(model, world)
            curve.append({'step': step, 'loss': float(loss.detach()), 'train_rel_bal': R1.per_relation_bal(world, s, y, fit),
                          'tests': {k: v['bal_acc'] if v['bal_acc'] is not None else v['acc']
                                    for k, v in test_reads(world, s, y).items()}})
    neg = None if sampler is None else {'drawn': sampler.drawn, 'test_fact_draws': sampler.test_hits,
                                        'test_true_fact_draws': sampler.test_true_hits}
    return model, fit, val, curve, neg


# ---------------------------------------------------------------- reads

def test_reads(world, s, y):
    out = {}
    for tst in sorted(set(world.test) - {''}):
        m = world.test == tst
        out[tst] = R1.metrics(s[m], y[m])
        if tst in ('P2', 'P3', 'P4'):
            out[tst]['by_relation'] = {world.relations[ri]: R1.metrics(s[m & (world.r == ri)], y[m & (world.r == ri)])
                                       for ri in np.unique(world.r[m])}
        if tst == 'P4':
            out[tst]['by_distance'] = {d: R1.metrics(s[m & (world.subset == d)], y[m & (world.subset == d)])
                                       for d in sorted(set(world.subset[m]), key=int)}
        if tst == 'P1':
            for sub in ('default', 'specific'):
                ms = m & (world.subset == sub)
                out[f'P1-{sub}'] = R1.metrics(s[ms], y[ms])
    return out


def fast_mapping(model, world, seed):
    prior = gaussian_prior(model.E)
    out = {}
    for name, (given, read, ks, row) in W.fast_mapping_specs().items():
        cls = {'dax': 'bird', 'fep': 'fish'}.get(name)
        for k in ks:
            out[f'{name}/k{k}'] = read_fit(model, world, name, given, read, row, k, seed, prior, cls)
    return out


def new_relations(model, world, seed):
    out = {}
    for name, rows in W.new_relation_specs().items():
        h = torch.tensor([world.index(a) for a, _, _ in rows])
        t = torch.tensor([world.index(c) for _, c, _ in rows])
        y = np.array([l for *_, l in rows], bool)
        for k in NEWREL_K:
            rng = np.random.default_rng([seed, k, len(name)])
            ex = np.r_[rng.choice(np.flatnonzero(y), k // 2, replace=False),
                       rng.choice(np.flatnonzero(~y), k // 2, replace=False)]
            rest = np.setdiff1d(np.arange(len(rows)), ex)
            g = torch.Generator().manual_seed(seed * 1000 + k + len(name))
            rc = nn.Parameter(torch.randn(1, model.R.shape[1], generator=g) * INIT_STD)
            opt = torch.optim.Adam([rc], lr=FIT_LR, weight_decay=WD)
            yt = torch.as_tensor(y[ex].astype(np.float32))
            for _ in range(FIT_STEPS):
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(
                    model.score_codes(model.E[h[ex]], rc.expand(len(ex), -1), model.E[t[ex]]), yt).backward()
                opt.step()
            with torch.no_grad():
                s = model.score_codes(model.E[h], rc.expand(len(rows), -1), model.E[t]).numpy().astype(float)
            out[f'{name}/k{k}'] = R1.metrics(s[rest], y[rest])
    return out


@torch.no_grad()
def relation_space(model, world, y, fit, seed):
    m = fit & (y == 1)
    h, r, t = world.h[m], world.r[m], world.t[m]
    groups = np.random.default_rng(seed).permutation(len(world.entities))[h]
    ch, ct, rc = model.E[torch.as_tensor(h)], model.E[torch.as_tensor(t)], model.R[torch.as_tensor(r)]
    counts = np.bincount(r, minlength=len(world.relations))
    prop = np.array([W.PROPERTY.get(world.relations[ri], 'other') for ri in r])
    pc = np.unique(prop, return_counts=True)[1]
    out = {'n_pairs': int(m.sum()), 'chance_majority': float(counts.max() / counts.sum()),
           'chance_same_relation': float(((counts / counts.sum()) ** 2).sum()),
           'chance_same_property': float(((pc / pc.sum()) ** 2).sum()),
           'probe_pair_codes': R1.probe(torch.cat([ch, ct], 1).numpy(), r, groups)}
    if model.arm not in ('two', 'tworel'):
        out.update(probe_q=None, analogy=None, property_analogy=None)
        return out
    q = model.q(ch, rc, ct)
    out['probe_q'] = R1.probe(q.numpy(), r, groups)
    qn = F.normalize(q, dim=1)
    share = (h[:, None] == h[None]) | (h[:, None] == t[None]) | (t[:, None] == h[None]) | (t[:, None] == t[None])
    sim = qn @ qn.T
    sim[torch.as_tensor(share)] = -2
    if model.arm == 'two':
        rels = {}
        for a, rr, c in zip(h, r, t):
            rels.setdefault((a, c), set()).add(rr)
        nn_ = sim.argmax(1).numpy()
        out['analogy'] = float(np.mean([len(rels[(h[i], t[i])] & rels[(h[j], t[j])]) > 0 for i, j in enumerate(nn_)]))
    else:
        out['analogy'] = None  # q contains r by construction
    sim[torch.as_tensor(r[:, None] == r[None])] = -2   # property analogy: nearest pair from a different relation
    nn_ = sim.argmax(1).numpy()
    out['property_analogy'] = float(np.mean(prop[nn_] == prop))
    out['property_analogy_by_property'] = {p: float(np.mean(prop[nn_][prop == p] == p)) for p in np.unique(prop)}
    return out


def false_positive(world, s, y, fit):
    m = fit & (y == 0)
    return {'fpr_trained_false': float((s[m] > 0).mean()),
            'by_relation': {world.relations[ri]: float((s[m & (world.r == ri)] > 0).mean()) for ri in np.unique(world.r[m])}}


# ---------------------------------------------------------------- tasks

def run(arm, regime, worldname, seed):
    world = W.build()
    y = world.y if worldname == 'real' else W.shuffled(world, seed)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model, fit, val, curve, neg = train(world, y, arm, regime, seed, g)
    s = scores(model, world)
    res = {'arm': arm, 'regime': regime, 'world': worldname, 'seed': seed, 'params': R1.n_params(model),
           'fit': {'train_bal_acc': R1.metrics(s[fit], y[fit])['bal_acc'], 'val': R1.metrics(s[val], y[val]),
                   'train_rel_bal': R1.per_relation_bal(world, s, y, fit)},
           'false_positive': false_positive(world, s, y, fit), 'negatives': neg,
           'tests': test_reads(world, s, y), 'curve': curve,
           'relation_space': relation_space(model, world, y, fit, seed)}
    if worldname == 'real':
        res['fast_mapping'] = fast_mapping(model, world, seed)
        res['new_relations'] = new_relations(model, world, seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-{regime}-{worldname}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'fit_steps': FIT_STEPS, 'params': R1.n_params(model),
                 'trained_queries_closed': int(fit.sum()), 'negatives': neg, 'checkpoint_bytes': ck.stat().st_size}


def task(t):
    arm, regime, worldname, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(arm, regime, worldname, seed)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{arm}/{regime}/{worldname}/seed{seed}', res, rt


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
            'dirty': bool(X.git('status', '--porcelain', '--untracked-files=no')), 'contract_sha256': nc.sha256(CONTRACT),
            'runner_sha256': nc.sha256(__file__), 'world_sha256': nc.sha256(W.__file__),
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'cpu': platform.processor(), 'os': platform.platform(), 'threads_per_worker': 1, 'dtype': 'float32',
            'device': 'cpu', 'workers': int(os.environ.get('EV_WORKERS', 8)),
            'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            'rss_convention': 'ru_maxrss in KiB per process; children is the largest single child', **extra}


def main():
    t0 = time.perf_counter()
    tasks = [(a, rg, wd, s) for a, rg, wd in CONFIGS for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(int(os.environ.get('EV_WORKERS', 8)),
                                                                 maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', '{0}-{1}-{2}-seed{3}'), tasks, chunksize=1)
    world = W.build()
    tp, cp = time.perf_counter(), time.process_time()
    prior = prior_baseline(world)
    rt = runtime({'wall_seconds': time.perf_counter() - t0,
                  'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                  'prior_baseline': {'wall_seconds': time.perf_counter() - tp, 'cpu_seconds': time.process_time() - cp},
                  'tasks': {k: t for k, _, t in res}})
    save_json(f'{REPORT}.json', {'status': 'everyday relations v2' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                 'results': {k: r for k, r, _ in res}, 'prior': prior,
                                 'world': {'entities': len(world.entities), 'relations': len(world.relations),
                                           'queries': int(len(world.y)), 'true': int(world.y.sum())}})
    save_json(f'{REPORT}-runtime.json', rt)
    summary()


def prior_baseline(world):
    fit = world.test == ''
    s = np.zeros(len(world.y))
    for ri in np.unique(world.r):
        for ti in np.unique(world.t[world.r == ri]):
            m = (world.r == ri) & (world.t == ti)
            p = (world.y[m & fit].sum() + .5) / ((m & fit).sum() + 1)
            s[m] = np.log(p / (1 - p))
        m = world.r == ri
        pr = world.y[m & fit].mean()
        s[m] -= np.log(pr / (1 - pr))
    return {'tests': test_reads(world, s, world.y)}


save_json = R1.save_json
_mean = R1._mean


def summary():
    data = json.loads(open(f'{REPORT}.json').read())
    res = data['results']
    groups = {}
    for r in res.values():
        groups.setdefault(f"{r['arm']}/{r['regime']}/{r['world']}", []).append(r)
    tests = sorted({t for r in res.values() for t in r['tests']})
    out = {}
    for gk, rs in groups.items():
        o = {'train_bal_acc': _mean([r['fit']['train_bal_acc'] for r in rs]),
             'fpr_trained_false': _mean([r['false_positive']['fpr_trained_false'] for r in rs]), 'tests': {}}
        for t in tests:
            o['tests'][t] = {m: _mean([r['tests'][t].get(m) for r in rs]) for m in ('bal_acc', 'acc', 'auroc', 'tpr', 'tnr')}
            for part in ('by_relation', 'by_distance'):
                if part in rs[0]['tests'][t]:
                    o['tests'][t][part] = {k: {'bal_acc': _mean([r['tests'][t][part][k]['bal_acc'] for r in rs])[0],
                                               'auroc': _mean([r['tests'][t][part][k]['auroc'] for r in rs])[0],
                                               'tpr': _mean([r['tests'][t][part][k]['tpr'] for r in rs])[0]}
                                           for k in rs[0]['tests'][t][part]}
        if 'fast_mapping' in rs[0]:
            o['fast_mapping'] = {k: {'bal_acc': _mean([r['fast_mapping'][k]['read']['bal_acc'] for r in rs]),
                                     'by_relation': {rel: _mean([r['fast_mapping'][k]['by_relation'][rel]['bal_acc']
                                                                 if r['fast_mapping'][k]['by_relation'][rel]['bal_acc'] is not None
                                                                 else r['fast_mapping'][k]['by_relation'][rel]['acc'] for r in rs])[0]
                                                     for rel in rs[0]['fast_mapping'][k]['by_relation']}}
                                 for k in rs[0]['fast_mapping']}
            o['new_relations'] = {k: _mean([r['new_relations'][k]['bal_acc'] for r in rs]) for k in rs[0]['new_relations']}
        o['relation_space'] = {m: _mean([r['relation_space'].get(m) for r in rs])
                               for m in ('probe_q', 'probe_pair_codes', 'analogy', 'property_analogy', 'chance_majority',
                                         'chance_same_relation', 'chance_same_property')}
        if rs[0]['negatives']:
            o['negatives_test_fact_rate'] = _mean([r['negatives']['test_fact_draws'] / r['negatives']['drawn'] for r in rs])[0]
        o['early_order'] = {rel: _mean([next((c['step'] for c in r['curve'] if (c['train_rel_bal'][rel] or 0) >= .9), None)
                                        for r in rs])[0] for rel in rs[0]['fit']['train_rel_bal']}
        out[gk] = o
    out['prior'] = data['prior']['tests']
    save_json(f'{REPORT}-summary.json', out)
    cols = [c for c, _, _ in [(f'{a}/{rg}/{wd}', 0, 0) for a, rg, wd in CONFIGS] if c in out]
    short = lambda c: c.replace('/real', '').replace('/shuffled', '/sh').replace('closed', 'cl')
    print('test'.ljust(15) + ''.join(short(c).rjust(13) for c in cols))
    for t in tests:
        row = []
        for c in cols:
            m = out[c]['tests'][t]
            v = m['bal_acc'][0] if m['bal_acc'][0] is not None else m['acc'][0]
            a = m['auroc'][0]
            row.append('-' if v is None else f'{v:.2f}' + ('' if a is None else f'({a:.2f})'))
        print(t.ljust(15) + ''.join(x.rjust(13) for x in row))
    for lab, key in (('train', 'train_bal_acc'), ('fpr', 'fpr_trained_false')):
        print(lab.ljust(15) + ''.join(f"{out[c][key][0]:.2f}".rjust(13) for c in cols))


if __name__ == '__main__':
    {'part0': part0, 'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
