"""Everyday relations v3 (docs/everyday-relations-v3.md): closed world against few "no"s (positive-unlabelled), plus
two-word phrases. Model, prior fitting, new relations and relation-space reads come from v2's runner.

Usage: .venv/bin/python -m experiments.everyday_relations_v3 main      (rerun to resume; EV_WORKERS, default 8)
       .venv/bin/python -m experiments.everyday_relations_v3 summary
       EV_SMOKE=1 ... for a short check (100 steps, one seed)
"""
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from src import everyday_world_v3 as W

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'everyday-relations-v3' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'everyday-relations-v3.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 100 if SMOKE else 3000
SEEDS = (420,) if SMOKE else (420, 421, 422, 423)
MENTIONED_PER_TRUE = 1
CONFIGS = ([(a, rg, wd) for rg in ('closed', 'fewno') for a in ('two', 'tworel', 'flat') for wd in ('real', 'shuffled')]
           + [('complex', 'fewno', 'real')])
V2.FIT_STEPS = 50 if SMOKE else 500


def log_steps():
    return set(range(25, STEPS + 1, 25)) if SMOKE else set(range(25, 501, 25)) | set(range(750, STEPS + 1, 250))


# ---------------------------------------------------------------- few "no"s

def mentioned_negatives(world, y, fit, seed):
    """One known-false corruption per trained true fact, drawn once: replace head or tail uniformly from the relation's
    set; accept only a query that is labelled false, not held out and not in validation. Returns a boolean mask."""
    rng = np.random.default_rng([seed, 7])
    nE, nR = len(world.entities), len(world.relations)
    key = lambda h, r, t: (h * nR + r) * nE + t
    allowed = {k: i for i, k in enumerate(key(world.h, world.r, world.t)) if fit[i] and y[i] == 0}
    heads = {ri: np.array([world.index(x) for x in world.heads[rel]]) for ri, rel in enumerate(world.relations)}
    tails = {ri: np.array([world.index(x) for x in world.tails[rel]]) for ri, rel in enumerate(world.relations)}
    mask = np.zeros(len(y), bool)
    misses = 0
    for i in np.flatnonzero(fit & (y == 1)):
        h, r, t = int(world.h[i]), int(world.r[i]), int(world.t[i])
        for _ in range(MENTIONED_PER_TRUE):
            for _try in range(50):
                if rng.random() < .5:
                    k = key(int(rng.choice(heads[r])), r, t)
                else:
                    k = key(h, r, int(rng.choice(tails[r])))
                j = allowed.get(k)
                if j is not None and not mask[j]:
                    mask[j] = True
                    break
            else:
                misses += 1
    return mask, misses


def train(world, y, arm, regime, seed, gen):
    trained = world.test == ''
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(trained)
    val = np.zeros(len(y), bool)
    val[rng.choice(idx, int(round(V2.VAL_FRAC * len(idx))), replace=False)] = True
    fit = trained & ~val
    mentioned, misses = (None, 0)
    use = fit
    if regime == 'fewno':
        mentioned, misses = mentioned_negatives(world, y, fit, seed)
        use = (fit & (y == 1)) | mentioned
    nR = len(world.relations)
    model = V2.Net(arm, len(world.entities), nR, gen)
    opt = torch.optim.Adam(model.parameters(), lr=V2.LR, weight_decay=V2.WD)
    h, r, t = (torch.as_tensor(a[use]) for a in (world.h, world.r, world.t))
    yt = torch.as_tensor(y[use])
    wt = V2.rel_weights(world.r[use], y[use], nR)
    curve, logs = [], log_steps()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(h, r, t), yt, reduction='none') * wt).sum()
        loss.backward()
        opt.step()
        if step in logs:
            s = V2.scores(model, world)
            curve.append({'step': step, 'loss': float(loss.detach()), 'train_rel_bal': R1.per_relation_bal(world, s, y, use),
                          'tests': {k: v['bal_acc'] if v['bal_acc'] is not None else v['acc']
                                    for k, v in test_reads(world, s, y).items() if k != 'G4-dissociation'}})
    info = {'trained_queries': int(use.sum()), 'mentioned_false': None if mentioned is None else int(mentioned.sum()),
            'mention_misses': misses}
    return model, fit, val, use, mentioned, curve, info


# ---------------------------------------------------------------- reads

def test_reads(world, s, y):
    out = V2.test_reads(world, s, y)
    for tst in ('G2', 'G3'):
        m = world.test == tst
        out[tst]['by_relation'] = {sub: R1.metrics(s[m & (world.subset == sub)], y[m & (world.subset == sub)])
                                   for sub in sorted(set(world.subset[m]))}
    m = world.test == 'G4'
    for sub in ('silly', 'sensible'):
        out[f'G4-{sub}'] = R1.metrics(s[m & (world.subset == sub)], y[m & (world.subset == sub)])
    # dissociation: held-out silly pairs judged good-phrase true and makes-sense false
    gp = world.relations.index('good-phrase')
    idx = {(int(a), int(c)): i for i, (a, c) in enumerate(zip(world.h, world.t)) if world.r[i] == gp}
    silly = np.flatnonzero(m & (world.subset == 'silly'))
    gps = np.array([s[idx[(int(world.h[i]), int(world.t[i]))]] for i in silly])
    out['G4-dissociation'] = {'n': int(len(silly)), 'good_order_and_no_sense': float(np.mean((gps > 0) & (s[silly] <= 0))),
                              'good_order': float(np.mean(gps > 0)), 'no_sense': float(np.mean(s[silly] <= 0))}
    return out


def acceptance(world, s, y, fit, mentioned):
    """False-acceptance on false pairs not held out: unmentioned (few-no) or trained (closed)."""
    false = fit & (y == 0)
    out = {'trained_false_fpr': None, 'unmentioned_false_fpr': None, 'mentioned_false_fpr': None}
    if mentioned is None:
        out['trained_false_fpr'] = float((s[false] > 0).mean())
    else:
        un = false & ~mentioned
        out['unmentioned_false_fpr'] = float((s[un] > 0).mean())
        out['mentioned_false_fpr'] = float((s[mentioned] > 0).mean())
        out['unmentioned_by_relation'] = {world.relations[ri]: float((s[un & (world.r == ri)] > 0).mean())
                                          for ri in np.unique(world.r[un])}
    return out


def fast_mapping(model, world, seed):
    prior = V2.gaussian_prior(model.E)
    out = {}
    for name, (given, read, ks) in W.fast_mapping_specs().items():
        cls = {'dax': 'bird', 'fep': 'fish'}.get(name)
        for k in ks:
            out[f'{name}/k{k}'] = V2.read_fit(model, world, name, given, read, [], k, seed, prior, cls)
    return out


# ---------------------------------------------------------------- tasks

def run(arm, regime, worldname, seed):
    world = W.build()
    y = world.y if worldname == 'real' else W.shuffled(world, seed)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model, fit, val, use, mentioned, curve, info = train(world, y, arm, regime, seed, g)
    s = V2.scores(model, world)
    res = {'arm': arm, 'regime': regime, 'world': worldname, 'seed': seed, 'params': R1.n_params(model), 'info': info,
           'fit': {'train_bal_acc': R1.metrics(s[use], y[use])['bal_acc'], 'val': R1.metrics(s[val], y[val]),
                   'train_rel_bal': R1.per_relation_bal(world, s, y, use)},
           'acceptance': acceptance(world, s, y, fit, mentioned),
           'tests': test_reads(world, s, y), 'curve': curve,
           'relation_space': V2.relation_space(model, world, y, use, seed)}
    if worldname == 'real':
        res['fast_mapping'] = fast_mapping(model, world, seed)
        res['new_relations'] = V2.new_relations(model, world, seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-{regime}-{worldname}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'optimizer_steps': STEPS, 'fit_steps': V2.FIT_STEPS, 'params': R1.n_params(model), **info,
                 'checkpoint_bytes': ck.stat().st_size}


def task(t):
    arm, regime, worldname, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(arm, regime, worldname, seed)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{arm}/{regime}/{worldname}/seed{seed}', res, rt


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


def main():
    t0 = time.perf_counter()
    tasks = [(a, rg, wd, s) for a, rg, wd in CONFIGS for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(int(os.environ.get('EV_WORKERS', 8)),
                                                                 maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', '{0}-{1}-{2}-seed{3}'), tasks, chunksize=1)
    world = W.build()
    tp, cp = time.perf_counter(), time.process_time()
    prior = prior_baseline(world)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
                     'world_sha256': nc.sha256(W.__file__), 'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
                     'prior_baseline': {'wall_seconds': time.perf_counter() - tp, 'cpu_seconds': time.process_time() - cp},
                     'tasks': {k: t for k, _, t in res}})
    V2.save_json(f'{REPORT}.json', {'status': 'everyday relations v3' + (' (smoke)' if SMOKE else ''), 'steps': STEPS,
                                    'results': {k: r for k, r, _ in res}, 'prior': prior,
                                    'world': {'entities': len(world.entities), 'relations': len(world.relations),
                                              'queries': int(len(world.y)), 'true': int(world.y.sum())}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    data = json.loads(open(f'{REPORT}.json').read())
    res = data['results']
    groups = {}
    for r in res.values():
        groups.setdefault(f"{r['arm']}/{r['regime']}/{r['world']}", []).append(r)
    tests = sorted({t for r in res.values() for t in r['tests'] if t != 'G4-dissociation'})
    _m = R1._mean
    out = {}
    for gk, rs in groups.items():
        o = {'train_bal_acc': _m([r['fit']['train_bal_acc'] for r in rs]),
             'acceptance': {k: _m([r['acceptance'][k] for r in rs]) for k in ('trained_false_fpr', 'unmentioned_false_fpr',
                                                                             'mentioned_false_fpr')},
             'g4_dissociation': {k: _m([r['tests']['G4-dissociation'][k] for r in rs])
                                 for k in ('good_order_and_no_sense', 'good_order', 'no_sense')},
             'tests': {}}
        for t in tests:
            o['tests'][t] = {m: _m([r['tests'][t].get(m) for r in rs]) for m in ('bal_acc', 'acc', 'auroc', 'tpr', 'tnr')}
            for part in ('by_relation', 'by_distance'):
                if part in rs[0]['tests'][t]:
                    o['tests'][t][part] = {k: {q: _m([r['tests'][t][part][k][q] for r in rs])[0]
                                               for q in ('bal_acc', 'auroc', 'tpr', 'tnr')}
                                           for k in rs[0]['tests'][t][part]}
        if 'fast_mapping' in rs[0]:
            o['fast_mapping'] = {k: {'bal_acc': _m([r['fast_mapping'][k]['read']['bal_acc'] for r in rs]),
                                     'by_relation': {rel: _m([r['fast_mapping'][k]['by_relation'][rel]['bal_acc']
                                                              if r['fast_mapping'][k]['by_relation'][rel]['bal_acc'] is not None
                                                              else r['fast_mapping'][k]['by_relation'][rel]['acc']
                                                              for r in rs])[0]
                                                     for rel in rs[0]['fast_mapping'][k]['by_relation']}}
                                 for k in rs[0]['fast_mapping']}
            o['new_relations'] = {k: _m([r['new_relations'][k]['bal_acc'] for r in rs]) for k in rs[0]['new_relations']}
        o['relation_space'] = {m: _m([r['relation_space'].get(m) for r in rs])
                               for m in ('probe_q', 'probe_pair_codes', 'analogy', 'property_analogy', 'chance_same_property')}
        out[gk] = o
    out['prior'] = data['prior']['tests']
    V2.save_json(f'{REPORT}-summary.json', out)
    cols = [f'{a}/{rg}/{wd}' for a, rg, wd in CONFIGS if f'{a}/{rg}/{wd}' in out]
    short = lambda c: c.replace('/real', '').replace('/shuffled', '/sh').replace('closed', 'cl').replace('fewno', 'fn')
    print('test'.ljust(15) + ''.join(short(c).rjust(13) for c in cols))
    for t in tests:
        row = []
        for c in cols:
            m = out[c]['tests'][t]
            v = m['bal_acc'][0] if m['bal_acc'][0] is not None else m['acc'][0]
            row.append('-' if v is None else f'{v:.2f}')
        print(t.ljust(15) + ''.join(x.rjust(13) for x in row))
    for lab, f in (('fpr', lambda o: o['acceptance']['unmentioned_false_fpr'][0] if o['acceptance']['unmentioned_false_fpr'][0]
                    is not None else o['acceptance']['trained_false_fpr'][0]),
                   ('G4 dissoc', lambda o: o['g4_dissociation']['good_order_and_no_sense'][0]),
                   ('train', lambda o: o['train_bal_acc'][0])):
        print(lab.ljust(15) + ''.join(f'{f(out[c]):.2f}'.rjust(13) for c in cols))


if __name__ == '__main__':
    {'main': main, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'main']()
