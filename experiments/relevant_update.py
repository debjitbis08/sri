"""Relevant update v1 (docs/relevant-update-v1-contract.md), stage 1: which update rule learns shared structure from a known-good set of
experiences, at what cost to old knowledge, and does joint processing help at matched work?

Stages (argv[1]):
  encoders          fit the frozen key encoders for the pilot seeds (GPU helps)
  gates             key pools on the pilot seeds, then gates 1, 2 and 5
  pilot             every setting at the primary cell, the selection rule, then all cells at selected settings
  confirm-encoders  refuses before the contract is frozen: encoders for seeds 382-391
  confirm           refuses before the contract is frozen: all cells at frozen settings on seeds 382-391
  analyse           the contract's decision rule on the confirmation seeds

Usage: .venv/bin/python -m experiments.relevant_update gates; NSY_WORKERS sets the pool. NSY_SMOKE=1: small and fast.
"""
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch

from experiments import durable as DUR
from experiments import surprise_lifecycle as E
from src import relevant_update as ru
from src import surprise_lifecycle as sl

X, nc, TK = E.X, E.nc, E.TK
SMOKE = X.SMOKE
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'gates'
NAME = 'relevant-update-v1' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'relevant-update-v1-contract.md'
E.RUNS = RUNS                # pools and MLP fits live with this study
WORKERS = int(os.environ.get('NSY_WORKERS', 11))

PILOT_SEEDS = (380,) if SMOKE else (380, 381)
CONFIRM_SEEDS = tuple(range(382, 392))
TRIALS = 2 if SMOKE else 8
LOADS, PRIMARY_L = (4, 16, 32), 16
KS, PRIMARY_K = (1, 2, 4, 8), 4
PS_BAD, KINDS = (.25, .5), ('unrelated', 'same-outcome', 'same-family')
CONDITIONS, PRIMARY_COND = ('clean', 'degraded'), 'clean'
AUDIT_FAMILY, AUDIT_BINDING = 32, 8
BUDGET = 64 * 1024
OLD_LIMIT, RULE_BAR = .85, .70   # provisional

LAM = (.01, .1, 1., 10.)
NEURAL = dict(layers=(1, 2), alpha=(0., .0001, .001), eta=(.001, .003, .01, .03, .1))
GRIDS = {'rank-one': {}, 'ridge': dict(lam=LAM), 'ridge-rehearse': dict(lam=LAM, r=(8, 32, 128), w=(.25, 1., 4.)), 'ridge-interfered': dict(lam=LAM, B=(128,), r=(8, 32), w=(.25, 1., 4.)),
         'ridge-orthogonal': dict(lam=LAM, q=(4, 8, 16, 32, 64)), 'expert': dict(tau_x=(.3, .4, .5, .6, .7)), 'omega': NEURAL, 'titans-m0': NEURAL, 'titans': {**NEURAL, 'beta': (.5, .9)}}
CONTROLS = {'ridge-cycled': 'ridge', 'omega-seq': 'omega'}
TURN_PARTNER = {'omega': 'titans-m0'}
LINEAR = {'rank-one': ru.RankOne, 'ridge': ru.Ridge, 'ridge-cycled': ru.RidgeCycled, 'ridge-rehearse': ru.RidgeRehearse, 'ridge-interfered': ru.RidgeInterfered, 'ridge-orthogonal': ru.RidgeOrthogonal, 'expert': ru.Expert}
NEURAL_RULES = {'omega': ru.Omega, 'omega-seq': ru.OmegaSeq, 'titans': ru.Titans, 'titans-m0': ru.Titans}


def save_json(path, obj):
    DUR.atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=float).encode())


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'stage': STAGE, 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
            'source_sha256': nc.sha256(__file__), 'mechanisms_sha256': nc.sha256(ru.__file__), 'contract_sha256': nc.sha256(CONTRACT), 'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cpu': platform.processor(), 'workers': WORKERS, 'threads_per_worker': 1, 'dtype': 'float32',
            'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, **extra}


# ---------------------------------------------------------------- the world for one seed and condition

_cache = {}


def world(seed, cond):
    """Everything fixed per seed and condition, once per worker. Old knowledge and buffers come from the fit events only."""
    key = (seed, cond)
    if key in _cache:
        return _cache[key]
    P = E.pool(seed)
    t = E.tables(seed)
    keys = P[f'stream_{cond}']
    old = E.old_knowledge(seed, t, keys)
    C, O, isrule = E.fit_events(seed, t, keys)
    rule_C, rule_O = C[isrule], O[isrule]
    labels = np.repeat(np.arange(sl.N_KNOWN_RULES), E.FIT_RULE)
    second = rule_C.T @ rule_C
    w, V = np.linalg.eigh(second)
    base = {**old, 'rule_C': rule_C, 'rule_O': rule_O, 'rule_labels': labels, 'buffer_order': nc.stream('ru_buffer', seed, cond).permutation(len(rule_C)), 'eigvecs': V[:, ::-1].astype(np.float32)}
    table = ru.rule_outcomes(nc.stream('ru_rule_outcomes', seed), t)
    protos, ids = E.prototypes(P[f'audit_{cond}'])
    audit = audit_cues(seed, t, table, P['audit_clean'])
    out = {'P': P, 't': t, 'keys': keys, 'base': base, 'table': table, 'protos': protos, 'ids': ids, 'audit': audit, 'cond': cond, 'seed': seed}
    _cache[key] = out
    return out


def audit_cues(seed, t, table, A):
    rng = nc.stream('ru_audit', seed)
    r = sl.symbol_ranges()
    R = A.shape[1]

    def family(f, n=AUDIT_FAMILY):
        pairs = np.array([rng.choice(r['audit_items'], 2, replace=False) for _ in range(n)])
        rr = rng.integers(0, 1 << 15, (n, 3))
        return E.cue_keys(A, np.full(n, f), pairs[:, 0], pairs[:, 1], rr, R).astype(np.float32)

    def fixed(c, n=AUDIT_BINDING):
        rr = rng.integers(0, 1 << 15, (n, 3))
        return E.cue_keys(A, np.full(n, c[0]), np.full(n, c[1]), np.full(n, c[2]), rr, R).astype(np.float32)
    return {'families': {int(f): family(int(f)) for f in table}, 'known_rules': [(family(f), o) for f, o in t['known_rules']], 'known_bindings': [(fixed(c), o) for c, o in t['known_bindings']]}


def make_rule(name, cfg, seed, cond, w):
    base = w['base']
    if name in LINEAR:
        return LINEAR[name](base, cfg)
    momentum = name == 'titans'
    window = 8 if name in ('omega', 'omega-seq') else 0
    H = sl.mlp_width(64, cfg['layers'], momentum=momentum, window=window)
    net, _ = E.fit_net(H, cfg['layers'], seed, cond, base)
    rule = NEURAL_RULES[name](net, cfg)
    return rule


def decoded(rule, C, w):
    return w['ids'][E.decode(rule.answers(C), w['protos'])]


def acc(rule, C, o, w):
    return float((decoded(rule, C, w) == o).mean())


def old_reading(rule, w):
    a = w['audit']
    kr = float(np.mean([acc(rule, C, o, w) for C, o in a['known_rules']]))
    kb = float(np.mean([acc(rule, C, o, w) for C, o in a['known_bindings']]))
    return (kr + kb) / 2, kr, kb


def run_trial(name, cfg, w, trial, L, k, p, kind, turn):
    """One trial. The draws depend on the cell and trial and never on the rule, so contrasts are paired down to the examples."""
    seed, cond = w['seed'], w['cond']
    rng = nc.stream('ru_trial', seed, cond, L, k, round(p * 100), kind or 'none', trial)
    tr = ru.make_trial(rng, w['table'], L, k, p, kind)
    keys = w['keys']
    R = keys.shape[1]
    C = E.cue_keys(keys, tr['F'], tr['I'], tr['J'], tr['render'], R).astype(np.float32)
    O = keys[tr['O'], tr['render'][..., 3] % R]
    rule = make_rule(name, cfg, seed, cond, w)
    rule.begin_trial()
    unused = np.vstack([w['audit']['families'][int(f)] for f in tr['unused']])
    outs = np.array(sorted({w['table'][int(f)] for f in tr['families']}))
    start_broad = float(np.isin(decoded(rule, unused, w), outs).mean())
    immediate = []
    t0 = rule.ops
    for j, f in enumerate(tr['families']):
        if turn:
            for i in range(k):
                rule.update(C[j, i:i + 1], O[j, i:i + 1])
        else:
            rule.update(C[j], O[j])
        immediate.append(acc(rule, w['audit']['families'][int(f)], w['table'][int(f)], w))
    gen = float(np.mean([acc(rule, w['audit']['families'][int(f)], w['table'][int(f)], w) for f in tr['families']]))
    old, kr, kb = old_reading(rule, w)
    breadth = float(np.isin(decoded(rule, unused, w), outs).mean()) - start_broad
    return {'rule_generalization': gen, 'old_knowledge': old, 'known_rule': kr, 'known_binding': kb, 'breadth': breadth, 'immediate': float(np.mean(immediate)), 'ops_per_update': (rule.ops - t0) / L, 'extra_bytes': rule.extra_bytes(L, k if turn else 1)}


def cell_key(task):
    name, cfgj, seed, cond, L, k, p, kind, turn = task
    return f"{name}-{hashlib.sha256(cfgj.encode()).hexdigest()[:12]}-seed{seed}-{cond}-L{L}-k{k}-p{round(p * 100)}-{kind or 'none'}-t{int(turn)}"


class Durable:
    def __call__(self, task):
        path = RUNS / 'done' / (cell_key(task) + '.json.gz')
        if path.exists():
            try:
                return json.loads(gzip.decompress(path.read_bytes()))
            except (OSError, ValueError, EOFError):
                pass
        name, cfgj, seed, cond, L, k, p, kind, turn = task
        t0 = time.process_time()
        w = world(seed, cond)
        rows = [run_trial(name, json.loads(cfgj), w, tr, L, k, p, kind, turn) for tr in range(TRIALS)]
        res = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        out = DUR.roundtrip((cell_key(task), res, {'cpu_s': time.process_time() - t0}))
        DUR.atomic_write(path, gzip.compress(json.dumps(out, sort_keys=True).encode()))
        return out


def run_all(tasks):
    if not tasks:
        return {}
    with mp.get_context('fork').Pool(WORKERS) as p:
        return {k: (r, rt) for k, r, rt in p.imap_unordered(Durable(), tasks)}


def configs(name):
    g = GRIDS[name]
    keys = sorted(g)
    return [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]


# ---------------------------------------------------------------- gates

def gate_seed(seed):
    out = {}
    for cond in CONDITIONS:
        w = world(seed, cond)
        base = w['base']
        start = ru.LinearRule(base, {})
        old0 = old_reading(start, w)
        rng = nc.stream('ru_ceiling', seed, cond)
        r = sl.symbol_ranges()
        fams = rng.permutation(np.array(sorted(w['table'])))
        rows = {}
        for L in LOADS:
            F = np.repeat(fams[:L], E.FIT_RULE)
            pairs = np.array([rng.choice(r['items'], 2, replace=False) for _ in range(len(F))])
            rr = rng.integers(0, 1 << 15, (len(F), 4))
            K = w['keys']
            C = E.cue_keys(K, F, pairs[:, 0], pairs[:, 1], rr, K.shape[1]).astype(np.float32)
            O = K[np.array([w['table'][int(f)] for f in F]), rr[:, 3] % K.shape[1]]
            joint = ru.LinearRule(base, {})
            joint.R = sl.fit_reader(np.vstack([base['rule_C'], C]), np.vstack([base['rule_O'], O]), 1.)
            rule = float(np.mean([acc(joint, w['audit']['families'][int(f)], w['table'][int(f)], w) for f in fams[:L]]))
            old, kr, kb = old_reading(joint, w)
            rows[str(L)] = {'rule_generalization': rule, 'old_knowledge': old, 'known_rule': kr, 'known_binding': kb}
        before = start.R.copy()
        old_reading(start, w)
        unchanged = bool((before == start.R).all())
        shuffled = nc.stream('ru_shuffle', seed).permutation(len(w['protos']))
        chance = float(np.mean([(w['ids'][shuffled][E.decode(start.answers(Cf), w['protos'])] == o).mean() for Cf, o in w['audit']['known_rules']]))
        mlp = {}
        for layers in (1, 2):
            for name, momentum, window in (('titans-m0', False, 0), ('titans', True, 0), ('omega', False, 8)):
                H = sl.mlp_width(64, layers, momentum=momentum, window=window)
                net, info = E.fit_net(H, layers, seed, cond, base)
                m = ru.Titans(net, dict(alpha=0., eta=0.))
                m.begin_trial()
                mlp[f'L{layers}-H{H}'] = {**info, 'known_rule': old_reading(m, w)[1], 'known_binding': old_reading(m, w)[2]}
        out[cond] = {'start_old_knowledge': old0[0], 'start_known_rule': old0[1], 'start_known_binding': old0[2], 'joint_ceiling': rows, 'audit_unchanged': unchanged, 'shuffled_known_rule': chance, 'mlp': mlp}
    return seed, out


def fit_encoders(seeds):
    assert not SMOKE
    torch.set_num_threads(4)
    from experiments import device as DEV
    DEV.install()
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    syms = TK.symbol_splits()
    for seed in seeds:
        t0 = time.perf_counter()
        TK.train(c['arch'], c['key'], seed, syms)
        print(seed, round(time.perf_counter() - t0, 1), flush=True)


def gates():
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(WORKERS, len(PILOT_SEEDS))) as p:
        p.map(E.build_pool, PILOT_SEEDS)
    pools = time.perf_counter() - t0
    t1 = time.perf_counter()
    with mp.get_context('fork').Pool(min(WORKERS, len(PILOT_SEEDS))) as p:
        res = dict(p.map(gate_seed, PILOT_SEEDS))
    save_json(f'{REPORT}-gates.json', res)
    save_json(f'{REPORT}-gates-runtime.json', runtime({'pools_wall_s': pools, 'gates_wall_s': time.perf_counter() - t1}))
    for seed, r in res.items():
        for cond, x in r.items():
            print(seed, cond, 'start', round(x['start_old_knowledge'], 3), 'ceiling', {L: (round(v['rule_generalization'], 3), round(v['old_knowledge'], 3)) for L, v in x['joint_ceiling'].items()}, 'unchanged', x['audit_unchanged'], 'shuffled', round(x['shuffled_known_rule'], 3))
            print('   mlp', {k: (v['epochs'], round(v['known_rule'], 2), round(v['known_binding'], 2)) for k, v in x['mlp'].items()})


# ---------------------------------------------------------------- selection

def primary(name, cfg, seeds, cond=PRIMARY_COND):
    j = json.dumps(cfg, sort_keys=True)
    return [(name, j, s, cond, PRIMARY_L, PRIMARY_K, 0., None, False) for s in seeds]


def score(name, cfg, seeds, results):
    rows = [results[cell_key(t)][0] for t in primary(name, cfg, seeds)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def full_cells(name, cfg, seeds):
    """Every cell the contract declares for a rule at its frozen setting, in the primary and other conditions and loads."""
    j = json.dumps(cfg, sort_keys=True)
    out = []
    for s in seeds:
        for cond in CONDITIONS:
            for L in LOADS:
                for k in KS:
                    out.append((name, j, s, cond, L, k, 0., None, False))
                for k in (4, 8):
                    for kind in KINDS:
                        for p in PS_BAD:
                            out.append((name, j, s, cond, L, k, p, kind, False))
                    if not (name == 'expert' and L * k * ru.ENTRY > BUDGET):
                        out.append((name, j, s, cond, L, k, 0., None, True))
    return out


def ok_budget(name, cfg, w):
    rule = make_rule(name, cfg, w['seed'], w['cond'], w) if name in LINEAR else None
    return rule is None or rule.extra_bytes(max(LOADS)) <= BUDGET


def pilot():
    t0 = time.perf_counter()
    seeds = PILOT_SEEDS
    w = world(seeds[0], PRIMARY_COND)
    results, rows = {}, []
    tasks = []
    for name in GRIDS:
        for cfg in configs(name):
            if ok_budget(name, cfg, w):
                tasks += primary(name, cfg, seeds)
    print('primary tasks', len(tasks), flush=True)
    results.update(run_all(tasks))
    sel = {}
    for name in GRIDS:
        scored = [(cfg, score(name, cfg, seeds, results)) for cfg in configs(name) if ok_budget(name, cfg, w)]
        within = [x for x in scored if x[1]['old_knowledge'] >= OLD_LIMIT]
        if within:
            cfg, sc = max(within, key=lambda x: (x[1]['rule_generalization'], -x[1]['ops_per_update']))
        else:
            cfg, sc = max(scored, key=lambda x: (x[1]['old_knowledge'], x[1]['rule_generalization']))
        sel[name] = {'cfg': cfg, 'score': sc, 'within_limit': bool(within), 'settings': len(scored)}
    candidates = [n for n in sel if n != 'rank-one' and sel[n]['within_limit']] or [n for n in sel if n != 'rank-one']
    chosen = max(candidates, key=lambda n: (sel[n]['score']['rule_generalization'], -sel[n]['score']['ops_per_update']))
    for ctrl, partner in CONTROLS.items():
        sel[ctrl] = {'cfg': sel[partner]['cfg'], 'score': None, 'within_limit': None, 'settings': 0}
    print('full cells', flush=True)
    full = [t for name in sel for t in full_cells(name, sel[name]['cfg'], seeds)]
    results.update(run_all(full))
    table = {name: {'by_cell': {cell_key(t): results[cell_key(t)][0] for t in full_cells(name, sel[name]['cfg'], seeds)}, 'cfg': sel[name]['cfg'], 'primary': (sel[name]['score'] or score(name, sel[name]['cfg'], seeds, {**results})), 'within_limit': sel[name]['within_limit'], 'settings': sel[name]['settings']} for name in sel}
    save_json(f'{REPORT}-selection.json', {'chosen': chosen, 'selected': {n: sel[n]['cfg'] for n in sel}, 'table': table})
    save_json(f'{REPORT}-pilot-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'cells': len(results), 'worker_cpu_s': sum(rt.get('cpu_s', 0) for _, rt in results.values())}))
    for name in sel:
        s = table[name]['primary']
        print(name, sel[name]['cfg'], {k: round(s[k], 3) for k in ('rule_generalization', 'old_knowledge', 'breadth', 'immediate', 'ops_per_update')}, table[name]['within_limit'])
    print('chosen', chosen)


# ---------------------------------------------------------------- confirmation

def frozen():
    return 'Status: **frozen' in CONTRACT.read_text()


def confirm_encoders():
    if not frozen():
        raise SystemExit('the contract is not frozen: seeds 382-391 stay unbuilt')
    fit_encoders(CONFIRM_SEEDS)


def confirm():
    if not frozen():
        raise SystemExit('the contract is not frozen: confirmation refuses to run')
    t0 = time.perf_counter()
    sel = json.load(open(f'{REPORT}-selection.json'))['selected']
    with mp.get_context('fork').Pool(WORKERS) as p:
        p.map(E.build_pool, CONFIRM_SEEDS)
    tasks = [t for name in sel for t in full_cells(name, sel[name], CONFIRM_SEEDS)]
    results = run_all(tasks)
    DUR.atomic_write(f'{REPORT}-confirm.json.gz', gzip.compress(json.dumps({k: r for k, (r, _) in results.items()}, sort_keys=True, default=float).encode()))
    save_json(f'{REPORT}-confirm-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'cells': len(results), 'worker_cpu_s': sum(rt.get('cpu_s', 0) for _, rt in results.values())}))


def analyse():
    res = json.loads(gzip.decompress(open(f'{REPORT}-confirm.json.gz', 'rb').read()))
    s = json.load(open(f'{REPORT}-selection.json'))
    sel, chosen = s['selected'], s['chosen']
    rng = nc.stream('ru_bootstrap')

    def read(n, seed, reading, cond=PRIMARY_COND, L=PRIMARY_L, k=PRIMARY_K, turn=False):
        return res[cell_key((n, json.dumps(sel[n], sort_keys=True), seed, cond, L, k, 0., None, turn))][reading]

    def diff(a, b, reading, ka=PRIMARY_K, kb=PRIMARY_K, ta=False, tb=False):
        d = np.array([read(a, x, reading, k=ka, turn=ta) - read(b, x, reading, k=kb, turn=tb) for x in CONFIRM_SEEDS])
        m = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
        return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    def mean(n, reading):
        return float(np.mean([read(n, x, reading) for x in CONFIRM_SEEDS]))

    def three(d, margin=.05):
        verdict = 'better' if d[0] >= margin and d[1] > 0 else 'worse' if d[0] <= -margin and d[2] < 0 else 'equivalent' if d[1] >= -margin and d[2] <= margin else 'inconclusive'
        return {'difference': d[0], 'interval': [d[1], d[2]], 'reading': verdict}
    gain = diff(chosen, 'rank-one', 'rule_generalization')
    old = diff(chosen, 'rank-one', 'old_knowledge')
    br = diff(chosen, 'rank-one', 'breadth')
    beats = gain[0] >= .10 and gain[1] > 0 and old[1] >= -.02 and br[2] <= .05
    meets = {n: mean(n, 'rule_generalization') >= RULE_BAR and mean(n, 'old_knowledge') >= OLD_LIMIT for n in (chosen, 'rank-one')}
    if meets[chosen] and beats:
        outcome = 'update-found'
    elif meets[chosen]:
        outcome = 'bar-reached-superiority-unestablished'
    elif meets['rank-one']:
        outcome = 'rank-one-enough'
    elif beats:
        outcome = 'improved-below-bar'
    else:
        outcome = 'no-rule-reaches-bar'
    secondary = {}
    for a, b in (('omega', 'omega-seq'), ('ridge', 'ridge-cycled')):
        for k in (4, 8):
            secondary[f'B {a} vs {b} k{k}'] = {r: three(diff(a, b, r, k, k)) for r in ('rule_generalization', 'old_knowledge')}
    for n in {chosen, 'ridge', 'omega'}:
        for k in (4, 8):
            secondary[f'A {n} together vs in turn k{k}'] = {r: three(diff(n, n, r, k, k, False, True)) for r in ('rule_generalization', 'old_knowledge')}
    secondary['more evidence k4 vs k1'] = {'rule_generalization': three(diff(chosen, chosen, 'rule_generalization', 4, 1))}
    for n in ('ridge-rehearse', 'ridge-orthogonal'):
        secondary[f'protection {n} vs ridge'] = {'old_knowledge': three(diff(n, 'ridge', 'old_knowledge'))}
    save_json(f'{REPORT}-analysis.json', {'outcome': outcome, 'chosen': chosen, 'gain': gain, 'old': old, 'breadth': br, 'means': {n: {r: mean(n, r) for r in ('rule_generalization', 'old_knowledge', 'breadth')} for n in sel}, 'secondary': secondary})
    print('outcome', outcome, 'chosen', chosen, gain)


if __name__ == '__main__':
    {'encoders': lambda: fit_encoders(PILOT_SEEDS), 'gates': gates, 'pilot': pilot, 'confirm-encoders': confirm_encoders, 'confirm': confirm, 'analyse': analyse}[STAGE]()
