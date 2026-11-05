"""Everyday relations v1 (docs/everyday-relations-v1.md): the two-space core on a one-hot world of everyday facts.

Usage: .venv/bin/python -m experiments.everyday_relations main      (rerun to resume; EV_WORKERS, default 8)
       .venv/bin/python -m experiments.everyday_relations summary
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
from experiments import number_systems as X
from src import everyday_world as W

nc, ROOT = X.nc, X.ROOT
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'everyday-relations-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'everyday-relations-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 100 if SMOKE else 3000
FIT_STEPS = 50 if SMOKE else 500
LOG_EVERY = 25 if SMOKE else 250
SEEDS = (400,) if SMOKE else (400, 401, 402, 403)
LR, WD, FIT_LR, INIT_STD, RD, HID, FLAT_HID = 3e-3, 1e-4, 1e-2, 0.3, 16, 128, 230
ARMS = {'two': 32, 'two-16': 16, 'two-64': 64, 'flat': 32, 'complex': 32}
SHUFFLED_ARMS = ('two', 'flat')
VAL_FRAC = 0.05
NEWREL_K = (4, 16)


# ---------------------------------------------------------------- models

class Net(nn.Module):
    """Scores (head code, relation code, tail code). Codes are rows of embedding tables; new entities or relations are
    scored by passing their fitted code vectors directly."""

    def __init__(self, arm, n_ent, n_rel, g):
        super().__init__()
        self.arm = arm
        d = ARMS[arm]
        ed, rd = (2 * d, 2 * d) if arm == 'complex' else (d, RD)
        self.E = nn.Parameter(torch.randn(n_ent, ed, generator=g) * INIT_STD)
        self.R = nn.Parameter(torch.randn(n_rel, rd, generator=g) * INIT_STD)
        self.d = d
        if arm.startswith('two'):
            self.G = nn.Sequential(nn.Linear(2 * d, HID), nn.ReLU(), nn.Linear(HID, d))
            self.S = nn.Sequential(nn.Linear(d + RD, HID), nn.ReLU(), nn.Linear(HID, 1))
        elif arm == 'flat':
            self.S = nn.Sequential(nn.Linear(2 * d + RD, FLAT_HID), nn.ReLU(), nn.Linear(FLAT_HID, 1))
        _reinit(self, g)

    def q(self, ch, ct):
        return self.G(torch.cat([ch, ct], 1))

    def score_codes(self, ch, rc, ct):
        if self.arm == 'complex':
            d = self.d
            hr, hi, rr, ri, tr, ti = ch[:, :d], ch[:, d:], rc[:, :d], rc[:, d:], ct[:, :d], ct[:, d:]
            return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(1)
        if self.arm == 'flat':
            return self.S(torch.cat([ch, ct, rc], 1))[:, 0]
        return self.S(torch.cat([self.q(ch, ct), rc], 1))[:, 0]

    def forward(self, h, r, t):
        return self.score_codes(self.E[h], self.R[r], self.E[t])


def _reinit(model, g):
    """Default PyTorch Linear init, drawn from the run's generator (reproducible across processes)."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            bound = 1 / m.in_features ** .5
            with torch.no_grad():
                m.weight.copy_((torch.rand(m.weight.shape, generator=g) * 2 - 1) * bound)
                m.bias.copy_((torch.rand(m.bias.shape, generator=g) * 2 - 1) * bound)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- metrics

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    p = s > 0
    out = {'n': int(len(y)), 'n_pos': int(y.sum()), 'acc': float((p == y).mean()) if len(y) else None}
    out['tpr'] = float(p[y].mean()) if y.any() else None
    out['tnr'] = float((~p[~y]).mean()) if (~y).any() else None
    out['bal_acc'] = .5 * (out['tpr'] + out['tnr']) if y.any() and (~y).any() else None
    out['auroc'] = auroc(s, y)
    return out


def per_relation_bal(world, s, y, mask):
    out = {}
    for ri, rel in enumerate(world.relations):
        m = mask & (world.r == ri)
        out[rel] = metrics(s[m], y[m])['bal_acc'] if m.any() else None
    return out


def test_reads(world, s, y):
    """Per test: metrics on the held-out queries; positive-only tests also ranked against the relation's false queries."""
    out = {}
    for tst in sorted(set(world.test) - {''}):
        m = world.test == tst
        out[tst] = metrics(s[m], y[m])
        rels = np.unique(world.r[m])
        neg = np.isin(world.r, rels) & (y == 0) & (world.test == '')
        if y[m].all():
            out[tst]['auroc_vs_trained_neg'] = auroc(np.r_[s[m], s[neg]], np.r_[np.ones(m.sum()), np.zeros(neg.sum())])
        if tst == 'T1':
            for sub in ('default', 'specific'):
                ms = m & (world.subset == sub)
                out[f'T1-{sub}'] = metrics(s[ms], y[ms])
    return out


# ---------------------------------------------------------------- training

def train(world, y, arm, seed, gen):
    n = len(y)
    trained = world.test == ''
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(trained)
    val = np.zeros(n, bool)
    val[rng.choice(idx, int(round(VAL_FRAC * len(idx))), replace=False)] = True
    fit = trained & ~val
    model = Net(arm, len(world.entities), len(world.relations), gen)
    h, r, t = (torch.as_tensor(a[fit]) for a in (world.h, world.r, world.t))
    yt = torch.as_tensor(y[fit])
    wt = np.zeros(fit.sum(), np.float32)
    rf, yf = world.r[fit], y[fit]
    for ri in range(len(world.relations)):
        for lab in (0, 1):
            m = (rf == ri) & (yf == lab)
            if m.any():
                wt[m] = 1 / (2 * m.sum() * len(world.relations))
    wt = torch.as_tensor(wt)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    curve = []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(h, r, t), yt, reduction='none') * wt).sum()
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == STEPS:
            s = scores(model, world)
            curve.append({'step': step, 'loss': float(loss.detach()), 'train_rel_bal': per_relation_bal(world, s, y, fit),
                          'tests': {k: v['bal_acc'] if v['bal_acc'] is not None else v['acc']
                                    for k, v in test_reads(world, s, y).items()}})
    return model, fit, val, curve


@torch.no_grad()
def scores(model, world):
    return model(torch.as_tensor(world.h), torch.as_tensor(world.r), torch.as_tensor(world.t)).numpy().astype(float)


# ---------------------------------------------------------------- fast mapping and new relations

def fit_entity(model, world, name, facts, gen):
    """Fit one new entity code from labelled facts, model frozen."""
    code = nn.Parameter(torch.randn(1, model.E.shape[1], generator=gen) * INIT_STD)
    opt = torch.optim.Adam([code], lr=FIT_LR, weight_decay=WD)

    def batch(fs):
        ch = torch.cat([code if a == name else model.E[world.index(a)][None] for _, a, _, _ in fs])
        ct = torch.cat([code if c == name else model.E[world.index(c)][None] for _, _, c, _ in fs])
        rc = model.R[[world.relations.index(rel) for rel, _, _, _ in fs]]
        return model.score_codes(ch, rc, ct)

    yt = torch.tensor([float(l) for *_, l in facts])
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(batch(facts), yt).backward()
        opt.step()
    return code.detach(), batch


def class_rank(model, world, code, cls):
    """Rank (1 = nearest) of cls among the 6 classes by Euclidean distance from code to the class's member-code centroid,
    plus code norm against the mean animal-code norm."""
    E = model.E.detach()
    dist = {}
    for c in W.CLASSES:
        mem = [world.index(a) for a in W.ANIMALS if W.ANIMALS[a][0] == c]
        dist[c] = float((code[0] - E[mem].mean(0)).norm())
    order = sorted(dist, key=dist.get)
    norms = E[[world.index(a) for a in W.ANIMALS]].norm(dim=1)
    return {'class_rank': order.index(cls) + 1, 'nearest_class': order[0], 'code_norm': float(code.norm()),
            'animal_norm_mean': float(norms.mean())}


def fast_mapping(model, world, seed):
    out = {}
    for name, (given, read, ks) in W.fast_mapping_specs().items():
        for k in ks:
            g = torch.Generator().manual_seed(seed * 1000 + 7 * k + len(name))
            for p in model.parameters():
                p.requires_grad_(False)
            code, batch = fit_entity(model, world, name, given[:k], g)
            with torch.no_grad():
                s = batch(read).numpy().astype(float)
                sg = batch(given[:k]).numpy().astype(float)
            rec = {'read': metrics(s, [l for *_, l in read]),
                   'given_acc': float(np.mean((sg > 0) == np.array([l for *_, l in given[:k]])))}
            rec['by_relation'] = {rel: metrics(s[[i for i, x in enumerate(read) if x[0] == rel]],
                                               [x[3] for x in read if x[0] == rel])
                                  for rel in sorted({x[0] for x in read})}
            if name in ('dax', 'fep'):
                rec['centroid'] = class_rank(model, world, code, given[0][2])
            out[f'{name}/k{k}'] = rec
    return out


def new_relations(model, world, seed):
    out = {}
    for name, rows in W.new_relation_specs(world).items():
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
            out[f'{name}/k{k}'] = metrics(s[rest], y[rest])
    return out


# ---------------------------------------------------------------- relation space

def probe(Xf, labels, groups, folds=5, steps=200):
    """Multinomial logistic regression (L2 1e-3, LBFGS), cross-validated with folds grouped by head entity."""
    Xf = torch.as_tensor(Xf, dtype=torch.float32)
    lab = torch.as_tensor(labels)
    k = int(lab.max()) + 1
    accs, n = [], 0
    for f in range(folds):
        te = torch.as_tensor(groups % folds == f)
        if not te.any() or te.all():
            continue
        tr = ~te
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        A = (Xf - mu) / sd
        Wt = torch.zeros(A.shape[1], k, requires_grad=True)
        bt = torch.zeros(k, requires_grad=True)
        opt = torch.optim.LBFGS([Wt, bt], max_iter=steps, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            l = F.cross_entropy(A[tr] @ Wt + bt, lab[tr]) + 1e-3 * Wt.pow(2).sum()
            l.backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            accs.append(float(((A[te] @ Wt + bt).argmax(1) == lab[te]).float().sum()))
            n += int(te.sum())
    return sum(accs) / n


@torch.no_grad()
def relation_space(model, world, y, fit, seed):
    """On true trained facts: relation probe from q (two arms) and from [c_h, c_t] (all arms); analogy read on q."""
    m = fit & (y == 1)
    h, r, t = world.h[m], world.r[m], world.t[m]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(world.entities))
    groups = perm[h]
    ch, ct = model.E[torch.as_tensor(h)], model.E[torch.as_tensor(t)]
    counts = np.bincount(r, minlength=len(world.relations))
    out = {'n_pairs': int(m.sum()), 'chance_majority': float(counts.max() / counts.sum()),
           'chance_same_relation': float(((counts / counts.sum()) ** 2).sum()),
           'probe_pair_codes': probe(torch.cat([ch, ct], 1).numpy(), r, groups)}
    if not model.arm.startswith('two'):
        out.update(probe_q=None, analogy=None)
        return out
    q = model.q(ch, ct)
    out['probe_q'] = probe(q.numpy(), r, groups)
    rels = {}
    for a, rr, c in zip(h, r, t):
        rels.setdefault((a, c), set()).add(rr)
    qn = F.normalize(q, dim=1)
    sim = qn @ qn.T
    share = (h[:, None] == h[None]) | (h[:, None] == t[None]) | (t[:, None] == h[None]) | (t[:, None] == t[None])
    sim[torch.as_tensor(share)] = -2
    nn_ = sim.argmax(1).numpy()
    hits = [len(rels[(h[i], t[i])] & rels[(h[j], t[j])]) > 0 for i, j in enumerate(nn_)]
    out['analogy'] = float(np.mean(hits))
    out['analogy_by_relation'] = {world.relations[ri]: float(np.mean([hits[i] for i in np.flatnonzero(r == ri)]))
                                  for ri in np.unique(r)}
    return out


# ---------------------------------------------------------------- tasks

def run(arm, worldname, seed):
    world = W.build()
    y = world.y if worldname == 'real' else W.shuffled(world, seed)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model, fit, val, curve = train(world, y, arm, seed, g)
    s = scores(model, world)
    res = {'arm': arm, 'world': worldname, 'seed': seed, 'params': n_params(model),
           'fit': {'train_bal_acc': metrics(s[fit], y[fit])['bal_acc'], 'val': metrics(s[val], y[val]),
                   'train_rel_bal': per_relation_bal(world, s, y, fit)},
           'tests': test_reads(world, s, y), 'curve': curve,
           'relation_space': relation_space(model, world, y, fit, seed)}
    if worldname == 'real':
        res['fast_mapping'] = fast_mapping(model, world, seed)
        res['new_relations'] = new_relations(model, world, seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-{worldname}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'fit_steps_per_fast_map': FIT_STEPS, 'trained_queries': int(fit.sum()),
                 'params': n_params(model), 'checkpoint_bytes': ck.stat().st_size}


def task(t):
    arm, worldname, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(arm, worldname, seed)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{arm}/{worldname}/seed{seed}', res, rt


def prior_baseline(world):
    """Per (relation, tail) base rate on trained queries, scored as a logit; deterministic, no items."""
    fit = world.test == ''
    s = np.zeros(len(world.y))
    for ri in np.unique(world.r):
        for ti in np.unique(world.t[world.r == ri]):
            m = (world.r == ri) & (world.t == ti)
            p = (world.y[m & fit].sum() + .5) / ((m & fit).sum() + 1)
            s[m] = np.log(p / (1 - p))
    # threshold at the relation's base rate, so a constant 'false' is not credited
    for ri in np.unique(world.r):
        m = world.r == ri
        pr = world.y[m & fit].mean()
        s[m] -= np.log(pr / (1 - pr))
    return {'tests': test_reads(world, s, world.y)}


def main():
    t0 = time.perf_counter()
    tasks = [(a, 'real', s) for a in ARMS for s in SEEDS] + [(a, 'shuffled', s) for a in SHUFFLED_ARMS for s in SEEDS]
    workers = int(os.environ.get('EV_WORKERS', 8))
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', '{0}-{1}-seed{2}'), tasks, chunksize=1)
    tp, cp = time.perf_counter(), time.process_time()
    prior = prior_baseline(W.build())
    prior_rt = {'wall_seconds': time.perf_counter() - tp, 'cpu_seconds': time.process_time() - cp}
    rt = {'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
          'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT),
          'runner_sha256': nc.sha256(__file__), 'world_sha256': nc.sha256(W.__file__),
          'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
          'cpu': platform.processor(), 'os': platform.platform(), 'workers': workers, 'threads_per_worker': 1,
          'dtype': 'float32', 'device': 'cpu', 'wall_seconds': time.perf_counter() - t0,
          'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
          'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
          'rss_convention': 'ru_maxrss in KiB per process; children is the largest single child',
          'prior_baseline': prior_rt, 'tasks': {k: t for k, _, t in res}}
    save_json(f'{REPORT}.json', {'status': 'everyday relations v1' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                 'results': {k: r for k, r, _ in res}, 'prior': prior})
    save_json(f'{REPORT}-runtime.json', rt)
    summary()


def save_json(path, obj):
    DUR.atomic_write(path, (json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n').encode())


# ---------------------------------------------------------------- summary

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), [round(x, 3) for x in xs]) if xs else (None, [])


def summary():
    data = json.loads(open(f'{REPORT}.json').read())
    res = data['results']
    groups = {}
    for k, r in res.items():
        groups.setdefault(f"{r['arm']}/{r['world']}", []).append(r)
    out = {}
    tests = sorted({t for r in res.values() for t in r['tests']})
    for gk, rs in sorted(groups.items()):
        o = {'train_bal_acc': _mean([r['fit']['train_bal_acc'] for r in rs]),
             'val_bal_acc': _mean([r['fit']['val']['bal_acc'] for r in rs]), 'tests': {}}
        for t in tests:
            o['tests'][t] = {m: _mean([r['tests'][t].get(m) for r in rs])
                             for m in ('bal_acc', 'acc', 'auroc', 'auroc_vs_trained_neg', 'tpr', 'tnr')}
        if 'fast_mapping' in rs[0]:
            o['fast_mapping'] = {k: {'bal_acc': _mean([r['fast_mapping'][k]['read']['bal_acc'] for r in rs]),
                                     'acc': _mean([r['fast_mapping'][k]['read']['acc'] for r in rs]),
                                     'class_rank': _mean([r['fast_mapping'][k].get('centroid', {}).get('class_rank')
                                                          for r in rs])}
                                 for k in rs[0]['fast_mapping']}
            o['new_relations'] = {k: {'bal_acc': _mean([r['new_relations'][k]['bal_acc'] for r in rs]),
                                      'auroc': _mean([r['new_relations'][k]['auroc'] for r in rs])}
                                  for k in rs[0]['new_relations']}
        o['relation_space'] = {m: _mean([r['relation_space'].get(m) for r in rs])
                               for m in ('probe_q', 'probe_pair_codes', 'analogy', 'chance_majority',
                                         'chance_same_relation')}
        o['train_rel_bal'] = {rel: _mean([r['fit']['train_rel_bal'][rel] for r in rs])[0]
                              for rel in rs[0]['fit']['train_rel_bal']}
        o['steps_to_0.9'] = {rel: _mean([next((c['step'] for c in r['curve'] if (c['train_rel_bal'][rel] or 0) >= .9),
                                              None) for r in rs])[0] for rel in rs[0]['fit']['train_rel_bal']}
        out[gk] = o
    out['prior'] = data['prior']['tests']
    save_json(f'{REPORT}-summary.json', out)
    cols = [g for g in out if g != 'prior']
    print('test'.ljust(16) + ''.join(c[:14].rjust(15) for c in cols))
    for t in tests:
        row = []
        for c in cols:
            m = out[c]['tests'][t]
            v = m['bal_acc'][0] if m['bal_acc'][0] is not None else m['acc'][0]
            row.append('-' if v is None else f'{v:.2f}')
        print(t.ljust(16) + ''.join(x.rjust(15) for x in row))
    print('train'.ljust(16) + ''.join(f"{out[c]['train_bal_acc'][0]:.2f}".rjust(15) for c in cols))
    for c in cols:
        if 'fast_mapping' in out[c]:
            print(c, {k: round(v['bal_acc'][0] if v['bal_acc'][0] is not None else v['acc'][0], 2)
                      for k, v in out[c]['fast_mapping'].items()})
            print('   newrel', {k: round(v['bal_acc'][0], 2) for k, v in out[c]['new_relations'].items()})
        rs_ = out[c]['relation_space']
        print('   space', {k: None if v[0] is None else round(v[0], 2) for k, v in rs_.items()})


if __name__ == '__main__':
    {'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
