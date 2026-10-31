"""Surprise lifecycle v1 (docs/surprise-lifecycle-v1-contract.md): kept surprises, scored on later experience, against replay, recent
windows, immediate updates, slow learning and scheduled consolidation at matched bytes, as related events get far apart.

Stages (argv[1]):
  encoders   fit the frozen key encoders for the confirmation seeds (GPU helps)
  gates      key pools on the pilot seeds, then the contract's gates 1, 2, 3 and 5
  pilot      coordinate search on the pilot seeds at the primary cell, the selection rule, then gate 6
  confirm    refuses before the contract is frozen: selected settings on seeds 380-389, every cell and probe
  timing     per-event timing of the selected settings, one thread
  analyse    the contract's decision rule on the confirmation seeds

Usage: .venv/bin/python -m experiments.surprise_lifecycle gates; NSY_WORKERS sets the pool. NSY_SMOKE=1: small and fast.
"""
import gzip
import hashlib
import io
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
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import streaming_retrieval as SR
from experiments import translation_stable_keys as TK
from src import surprise_lifecycle as sl

nc = X.nc
SMOKE = X.SMOKE
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'gates'
NAME = 'surprise-lifecycle-v1' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'surprise-lifecycle-v1-contract.md'
WORKERS = int(os.environ.get('NSY_WORKERS', 11))

PILOT_SEEDS = (299,) if SMOKE else (299, 298)
CONFIRM_SEEDS = tuple(range(380, 390))
GAPS = (128, 512) if SMOKE else (128, 512, 2048, 8192, 32768)
LONG_GAPS = GAPS[-1:] if SMOKE else (2048, 8192, 32768)
QS, BUDGETS_KIB, CONDITIONS = (.2, .05), (16, 64, 256), ('degraded', 'clean')
PRIMARY = {'q': .2, 'B': 64, 'cond': 'degraded'}
R_STREAM, R_AUDIT = (8, 4) if SMOKE else (64, 16)
FIT_RULE, FIT_BINDING = 384, 32
AUDIT_RULE, AUDIT_CUE, ONCE_SAMPLE = 64, 8, 128
MAX_CONFIGS, MAX_PASSES = 1500, 3
CHECKPOINTS = (.25, .5, .75, 1.)
LIMITS = {'noise_capture': .10, 'old_knowledge': .85}   # provisional
COMPARABLE = 6.                                         # provisional

LIFE_ARMS = ('life', 'life-sketch', 'life-cuegroup', 'count', 'count-sketch', 'life-selfcount', 'life-oracle')
COMPARATORS = ('slow', 'surprise-fifo', 'fifo-replay', 'reservoir-replay', 'titans-m0', 'titans', 'omega', 'sleep-periodic')
DESCRIPTIVE = ('frozen', 'sleep-grow')
ARMS = LIFE_ARMS + COMPARATORS + DESCRIPTIVE

LIFE_GRID = [('theta', (.1, .2, .3, .4, .5, .6), .3), ('tau', (.7, .8, .9, .95), .8), ('tau_e', (.3, .5, .7, .9), .7), ('tau_s', (.3, .4, .5, .6), .5),
             ('kappa', (.4, .5, .6, .7), .5), ('m', (2, 3, 4), 3), ('h_b', (1, 2, 3), 2), ('h_r', (1, 2, 3), 2), ('h_f', (2, 3, 4), 3), ('h_n', (2, 3), 2),
             ('S', (2, 4, 8), 4), ('delta', (.02, .05), .02), ('E', (16, 64), 64)]
READ_GRID = [('theta', (.1, .2, .3, .4, .5, .6), .3), ('tau', (.7, .8, .9, .95), .8)]
ETA = ('eta', (.003, .01, .03, .1, .3), .03)
GRIDS = {'slow': READ_GRID + [ETA], 'surprise-fifo': READ_GRID + [ETA], 'fifo-replay': READ_GRID + [ETA, ('r', (1, 4), 1)],
         'reservoir-replay': READ_GRID + [ETA, ('r', (1, 4), 1)], 'sleep-periodic': READ_GRID + [('P', (1., .5), 1.), ('lam', (.1, 1., 10.), 1.)],
         'titans-m0': [('layers', (1, 2), 2), ('alpha', (0., .0001, .001), .0001), ('lr', (.001, .003, .01, .03), .01)],
         'titans': [('layers', (1, 2), 2), ('alpha', (0., .0001, .001), .0001), ('beta', (.5, .9), .9), ('lr', (.001, .003, .01, .03), .01)],
         'omega': [('layers', (1, 2), 2), ('alpha', (0., .0001, .001), .0001), ('lr', (.001, .003, .01, .03), .01), ('c', (16, 64, 256), 64)]}


def save_json(path, obj):
    DUR.atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=float).encode())


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'stage': STAGE, 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
            'dirty': bool(X.git('status', '--porcelain')), 'source_sha256': nc.sha256(__file__), 'mechanisms_sha256': nc.sha256(sl.__file__),
            'contract_sha256': nc.sha256(CONTRACT), 'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'cpu': platform.processor(), 'workers': WORKERS, 'threads_per_worker': 1, 'dtype': 'float32',
            'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, **extra}


# ---------------------------------------------------------------- key pools: the simulated world, not memory of any arm

def symbol_table(seed):
    pool = sorted(TK.symbol_splits()['pool'])
    return [int(c) for c in nc.stream('sl_table', seed).permutation(pool)[:sl.N_SYMBOLS]]


def pool_path(seed):
    return RUNS / 'pools' / f'seed{seed}.npz'


def build_pool(seed):
    """Stream and audit keys per symbol: (symbols, renderings, 32), clean and degraded. Audit renderings are the other split."""
    path = pool_path(seed)
    if path.exists():
        return
    torch.set_num_threads(1)
    syms = TK.symbol_splits()
    table = symbol_table(seed)
    model = SR.key_encoder(seed)
    level = json.loads((X.ROOT / 'reports' / 'streaming-retrieval-v1-severity.json').read_text())['degraded_level']
    out = {}
    for split, n, tag in (('train', R_STREAM, 'stream'), ('eval', R_AUDIT, 'audit')):
        imgs = np.stack([PS.renderings(cp, syms['fonts'][cp], seed, split, n) for cp in table])
        shape = imgs.shape
        flat = imgs.reshape(-1, *shape[2:])
        out[f'{tag}_clean'] = SR.encode(model, flat).reshape(shape[0], shape[1], -1).astype(np.float32)
        noisy = SR.corrupt(flat, level, nc.stream('sl_corrupt', seed, tag))
        out[f'{tag}_degraded'] = SR.encode(model, noisy).reshape(shape[0], shape[1], -1).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **out)
    DUR.atomic_write(path, buf.getvalue())


def pool(seed):
    z = np.load(pool_path(seed))
    return {k: z[k] for k in z.files}


def tables(seed, scale=1):
    return sl.make_tables(nc.stream('sl_tables', seed, scale), scale)


def world(seed, t, g, q, occurrences=8, scale=1):
    return sl.make_world(nc.stream('sl_world', seed, g, round(q * 1000), occurrences, scale), t, g, q, occurrences)


def cue_keys(P, fam, it1, it2, render, R):
    return sl.join_cues(P[fam, render[..., 0] % R], P[it1, render[..., 1] % R], P[it2, render[..., 2] % R])


def events(w, keys, chunk=4096):
    """(c, o, kind, cause) in order, o None for a query. Only the oracle arm is given kind and cause."""
    R = keys.shape[1]
    for a in range(0, w['L'], chunk):
        idx = np.arange(a, min(a + chunk, w['L']))
        r = w['render'][idx]
        C = cue_keys(keys, w['fam'][idx], w['it1'][idx], w['it2'][idx], r, R).astype(np.float32)
        O = keys[w['outcome'][idx], r[:, 3] % R]
        for j, t in enumerate(idx):
            yield C[j], (None if w['outcome'][t] < 0 else O[j]), int(w['kind'][t]), int(w['cause'][t])


# ---------------------------------------------------------------- old knowledge

def fit_events(seed, t, keys):
    """The contract's fit set: 8 rules x 384 events with random items, 24 bindings x 32. Identical for every setting."""
    rng = nc.stream('sl_fit', seed)
    r = sl.symbol_ranges()
    F, I1, I2, Oc, isrule = [], [], [], [], []
    for f, o in t['known_rules']:
        for _ in range(FIT_RULE):
            a, b = rng.choice(r['items'], 2, replace=False)
            F.append(f), I1.append(a), I2.append(b), Oc.append(o), isrule.append(True)
    for (f, a, b), o in t['known_bindings']:
        for _ in range(FIT_BINDING):
            F.append(f), I1.append(a), I2.append(b), Oc.append(o), isrule.append(False)
    F, I1, I2, Oc, isrule = map(np.array, (F, I1, I2, Oc, isrule))
    render = rng.integers(0, 1 << 15, (len(F), 4))
    R = keys.shape[1]
    C = cue_keys(keys, F, I1, I2, render, R).astype(np.float32)
    O = keys[Oc, render[:, 3] % R]
    return C, O, isrule


def old_knowledge(seed, t, keys):
    C, O, isrule = fit_events(seed, t, keys)
    R = sl.fit_reader(C[isrule], O[isrule], 1.)
    first = len(C) - sl.N_KNOWN_BINDINGS * FIT_BINDING + np.arange(sl.N_KNOWN_BINDINGS) * FIT_BINDING
    return {'R': R, 'slot_c': C[first].copy(), 'slot_o': O[first].copy(), 'C': C, 'O': O}


def fit_net(H, layers, seed, cond, old):
    path = RUNS / 'mlp' / f'seed{seed}-{cond}-H{H}-L{layers}.npz'
    rng = nc.stream('sl_mlp', seed, cond, H, layers)
    net = sl.MLP(H, layers, rng)
    if path.exists():
        z = np.load(path)
        net.set([z[f'p{i}'] for i in range(2 * (layers + 1))])
        return net, {'epochs': int(z['epochs']), 'val_loss': float(z['val_loss'])}
    info = sl.fit_mlp(net, old['C'], old['O'], rng)
    buf = io.BytesIO()
    np.savez(buf, epochs=info['epochs'], val_loss=info['val_loss'], **{f'p{i}': p for i, p in enumerate(net.params())})
    DUR.atomic_write(path, buf.getvalue())
    return net, info


# ---------------------------------------------------------------- arms

def make_arm(arm, cfg, B, seed, cond, old):
    """Everything an arm receives: old knowledge as keys, and its settings. No table, kind or cause."""
    R, sc, so = old['R'], old['slot_c'], old['slot_o']
    K = sl.fifo_capacity(B)
    if arm == 'frozen':
        return sl.ReaderArm(R, sc, so, cfg['tau'], cfg['theta'])
    if arm == 'slow':
        return sl.Slow(R, sc, so, cfg['tau'], cfg['theta'], cfg['eta'])
    if arm == 'surprise-fifo':
        return sl.SurpriseFifo(R, sc, so, cfg['tau'], cfg['theta'], cfg['eta'], K)
    if arm in ('fifo-replay', 'reservoir-replay'):
        return sl.Replay(R, sc, so, cfg['tau'], cfg['theta'], cfg['eta'], cfg['r'], K, nc.stream('sl_replay', seed), arm == 'reservoir-replay')
    if arm in ('sleep-periodic', 'sleep-grow'):
        return sl.SleepPeriodic(R, sc, so, cfg['tau'], cfg['theta'], cfg['lam'], max(int(cfg['P'] * K), 1), K, cfg.get('rank') if arm == 'sleep-grow' else None)
    if arm in ('titans-m0', 'titans', 'omega'):
        beta = cfg.get('beta', 0.) if arm == 'titans' else 0.
        c = cfg.get('c', 1) if arm == 'omega' else 1
        H = sl.mlp_width(B, cfg['layers'], momentum=beta > 0, window=c if c > 1 else 0)
        net, _ = fit_net(H, cfg['layers'], seed, cond, old)
        return sl.NeuralMemory(net, cfg['alpha'], beta, cfg['lr'], c, arm)
    if arm == 'life-oracle':
        return sl.OracleLife(R, sc, so, cfg, B)
    kw = {'life': {}, 'life-sketch': {'sketch': True}, 'life-cuegroup': {'group_by': 'cue'}, 'count': {'validate': False},
          'count-sketch': {'validate': False, 'sketch': True}, 'life-selfcount': {'self_count': True}}[arm]
    return sl.Life(R, sc, so, cfg, B, name=arm, seed_rng=nc.stream('sl_sketch', seed), **kw)


def config_ok(arm, cfg, B):
    if arm in ('life', 'life-cuegroup', 'count', 'life-selfcount', 'life-oracle'):
        return sl.feasible(B, cfg['S'], cfg['m'])
    if arm in ('life-sketch', 'count-sketch'):
        return sl.feasible(B, cfg['S'], cfg['m'], sketch=True)
    if arm == 'omega':
        return (B * sl.KIB + sl.READER_BYTES + sl.KNOWN_BYTES) // 4 > cfg['c'] * (sl.D_C + sl.D_O) + sl.mlp_params(4, cfg['layers'])
    return True


# ---------------------------------------------------------------- the audit

def audit_sets(seed, t, P, R):
    """Fixed per seed, so every arm and gap reads the same cues. Item-based cues use audit-only items. Family outcomes are by the evaluator's table."""
    rng = nc.stream('sl_audit', seed)
    r = sl.symbol_ranges()

    def family(f, n):
        pairs = np.array([rng.choice(r['audit_items'], 2, replace=False) for _ in range(n)])
        return np.full(n, f), pairs[:, 0], pairs[:, 1]

    def fixed(cue, n):
        return np.full(n, cue[0]), np.full(n, cue[1]), np.full(n, cue[2])

    def build(parts):
        out = []
        for (F, A, B), o in parts:
            render = rng.integers(0, 1 << 15, (len(F), 3))
            out.append((cue_keys(P, F, A, B, render, R).astype(np.float32), np.full(len(F), o)))
        return out
    rules = build([(family(f, AUDIT_RULE), o) for f, o in t['new_rules']])
    known_rules = build([(family(f, AUDIT_RULE // 2), o) for f, o in t['known_rules']])
    known_bindings = build([(fixed(c, AUDIT_CUE), o) for c, o in t['known_bindings']])
    bindings = build([(fixed(c, AUDIT_CUE), o) for c, o in t['bindings'] + t['exceptions']])
    rares = build([(fixed(c, AUDIT_CUE), o) for c, o in t['rares']])
    fam_out = {f: o for f, o in t['known_rules']}
    exception_family = build([(family(c[0], AUDIT_RULE // 2), fam_out[c[0]]) for c, _ in t['exceptions']])
    return {'rule': rules, 'known_rule': known_rules, 'known_binding': known_bindings, 'binding': bindings, 'rare': rares, 'exception_family': exception_family}


def noise_cues(seed, w, P, R):
    """Each noise cue with the last outcome it was shown with, as of 80% of the stream for one-offs; the same for every arm of a world."""
    rng = nc.stream('sl_noise_audit', seed, w['g'], round(w['q'] * 1000))
    out = {}
    for name in ('noise-repeat', 'noise-burst'):
        idx = np.flatnonzero(w['kind'] == sl.K[name])
        last = {}
        for i in idx:
            last[int(w['cause'][i])] = int(i)
        out[name] = np.array(sorted(last.values()))
    once = np.flatnonzero((w['kind'] == sl.K['noise-once']) & (np.arange(w['L']) < .8 * w['L']))
    out['noise-once'] = np.sort(rng.choice(once, min(ONCE_SAMPLE, len(once)), replace=False))
    sets = {}
    for name, idx in out.items():
        n = AUDIT_CUE
        F, A, B = (np.repeat(w[k][idx], n) for k in ('fam', 'it1', 'it2'))
        render = rng.integers(0, 1 << 15, (len(F), 3))
        sets[name] = (cue_keys(P, F, A, B, render, R).astype(np.float32).reshape(len(idx), n, -1), w['outcome'][idx], sl.derangement(rng, len(idx)))
    return sets


def decode(ans, protos):
    return (sl.unit(ans) @ protos.T).argmax(-1)


def read_arm(arm, sets, noise, protos, ids):
    """The contract's readings. `ids` maps a decoded prototype to its symbol index."""
    def acc(parts):
        return float(np.mean([(ids[decode(arm.answers(C), protos)] == o).mean() for C, o in parts])) if parts else None
    rule = acc(sets['rule'])
    binding = acc(sets['binding'])
    old = float(np.mean([acc(sets['known_rule']), acc(sets['known_binding']), acc(sets['exception_family'])]))
    answers, targets, der = {}, {}, {}
    for name, (C, o, d) in noise.items():
        answers[name] = ids[decode(arm.answers(C.reshape(-1, C.shape[-1])), protos)].reshape(C.shape[0], C.shape[1])
        targets[name], der[name] = o, d
    capture, parts = sl.noise_capture(answers, None, targets, der)
    return {'rule_generalization': rule, 'binding_recall': binding, 'noise_capture': capture, 'noise_capture_by_kind': parts, 'old_knowledge': old,
            'rare_recall': acc(sets['rare']), 'known_rule': acc(sets['known_rule']), 'known_binding': acc(sets['known_binding']), 'exception_family': acc(sets['exception_family'])}


def prototypes(P):
    """Evaluator-held: one prototype per outcome glyph, from audit renderings."""
    ids = sl.symbol_ranges()['outcomes']
    return sl.unit(P[ids].mean(1)).astype(np.float32), ids


def run(arm_name, cfg, seed, g, q, B, cond, occurrences=8, scale=1):
    """One arm on one stream, audited on copies' worth of pure reads at the quarter points and the end."""
    P = pool(seed)
    t = tables(seed, scale)
    w = world(seed, t, g, q, occurrences, scale)
    keys = P[f'stream_{cond}']
    old = old_knowledge(seed, tables(seed), keys)
    arm = make_arm(arm_name, cfg, B, seed, cond, old)
    protos, ids = prototypes(P[f'audit_{cond}'])     # the stream's own condition: an arm's predictions live in the key space it learned
    sets = {c: audit_sets(seed, t, P[f'audit_{c}'], R_AUDIT) for c in ('clean', 'degraded')}
    noise = {c: noise_cues(seed, w, P[f'audit_{c}'], R_AUDIT) for c in ('clean', 'degraded')}
    marks = {int(f * w['L']) for f in CHECKPOINTS}
    out = {'curve': {}}
    oracle = isinstance(arm, sl.OracleLife)
    t0 = time.perf_counter()
    n = 0
    for c, o, k, cause in events(w, keys):
        if oracle:
            arm.told(c, o, k, cause)
        else:
            arm.step(c, o)
        n += 1
        if n in marks:
            out['curve'][f'{n / w["L"]:.2f}'] = read_arm(arm, sets['clean'], noise['clean'], protos, ids)
    out['end'] = out['curve']['1.00']
    out['end_degraded_audit'] = read_arm(arm, sets['degraded'], noise['degraded'], protos, ids)
    out['ops_per_event'] = arm.ops / w['L']
    out['bytes'] = arm.persistent_bytes()
    out['log'] = getattr(arm, 'log', None)
    if isinstance(arm, sl.Life):
        out['groups_left'] = len(arm.groups)
        out['tries_max'] = max([g.tries for g in arm.groups.values()], default=0)
    return out, time.perf_counter() - t0


def task_run(task):
    arm, cfg_json, seed, g, q, B, cond, occ, scale = task
    t0 = time.process_time()
    res, wall = run(arm, json.loads(cfg_json), seed, g, q, B, cond, occ, scale)
    return task_key(task), res, {'wall_s': wall, 'cpu_s': time.process_time() - t0}


def task_key(task):
    arm, cfg_json, seed, g, q, B, cond, occ, scale = task
    return f"{arm}-{hashlib.sha256(cfg_json.encode()).hexdigest()[:12]}-seed{seed}-g{g}-q{round(q * 1000)}-B{B}-{cond}-n{occ}-s{scale}"


class Durable:
    """Records named by the task's key, since the search visits configurations in an order only known at run time."""

    def __call__(self, task):
        path = RUNS / 'done' / (task_key(task) + '.json.gz')
        if path.exists():
            try:
                return json.loads(gzip.decompress(path.read_bytes()))
            except (OSError, ValueError, EOFError):
                pass
        out = DUR.roundtrip(task_run(task))
        DUR.atomic_write(path, gzip.compress(json.dumps(out, sort_keys=True).encode()))
        return out


def run_all(tasks):
    if not tasks:
        return {}
    with mp.get_context('fork').Pool(WORKERS) as p:
        return {k: (r, rt) for k, r, rt in p.imap_unordered(Durable(), tasks)}


# ---------------------------------------------------------------- gates

def auroc(pos, neg):
    allv = np.r_[pos, neg]
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gate_seed(seed):
    P = pool(seed)
    t = tables(seed)
    out = {}
    for cond in CONDITIONS:
        keys = P[f'stream_{cond}']
        protos, ids = prototypes(P[f'audit_{cond}'])
        old = old_knowledge(seed, t, keys)
        sets = audit_sets(seed, t, P['audit_clean'], R_AUDIT)
        w = world(seed, t, 2048 if not SMOKE else 512, .2)
        noise = noise_cues(seed, w, P['audit_clean'], R_AUDIT)
        floor = sl.ReaderArm(old['R'], old['slot_c'], old['slot_o'], .8, .3)
        before = read_arm(floor, sets, noise, protos, ids)

        # gate 1: a reader fit offline on every rule event, and slots written once per cause
        rule = np.isin(w['kind'], [sl.K['rule']])
        r = w['render'][rule]
        C = cue_keys(keys, w['fam'][rule], w['it1'][rule], w['it2'][rule], r, keys.shape[1]).astype(np.float32)
        O = keys[w['outcome'][rule], r[:, 3] % keys.shape[1]]
        Co, Oo, isrule = fit_events(seed, t, keys)
        weight = FIT_RULE // 8                       # 8 events per new rule, weighted to match the 384 per known rule
        Ro = sl.fit_reader(np.vstack([Co[isrule], np.repeat(C, weight, 0)]), np.vstack([Oo[isrule], np.repeat(O, weight, 0)]), 1.)
        Rr = old['R'] + sl.fit_reader(C, O - C @ old['R'], 1.)   # the residual form, on those 8 events per rule alone, as a shadow's update would
        slots_c, slots_o = [old['slot_c']], [old['slot_o']]
        for k in ('binding', 'exception', 'rare'):
            idx = np.flatnonzero(w['kind'] == sl.K[k])
            _, first = np.unique(w['cause'][idx], return_index=True)
            idx = idx[first]
            rr = w['render'][idx]
            slots_c.append(cue_keys(keys, w['fam'][idx], w['it1'][idx], w['it2'][idx], rr, keys.shape[1]).astype(np.float32))
            slots_o.append(keys[w['outcome'][idx], rr[:, 3] % keys.shape[1]])
        best = {}
        for tau in (.7, .8, .9, .95):
            ceil = sl.ReaderArm(Ro, np.vstack(slots_c), np.vstack(slots_o), tau, .3)
            best[str(tau)] = read_arm(ceil, sets, noise, protos, ids)
        residual_fit = read_arm(sl.ReaderArm(Rr, old['slot_c'], old['slot_o'], .8, .3), sets, noise, protos, ids)

        # MLP reach on old knowledge, the contract's target, at the primary budget
        mlp = {}
        for layers in (1, 2):
            H = sl.mlp_width(64, layers)
            net, info = fit_net(H, layers, seed, cond, old)
            mem = sl.NeuralMemory(net, 0., 0., 0., 1, 'titans-m0')
            mlp[str(layers)] = {'H': H, **info, 'known_rule': read_arm(mem, sets, noise, protos, ids)['known_rule'],
                                'known_binding': read_arm(mem, sets, noise, protos, ids)['known_binding']}

        # gate 2: can surprise see a cause's first occurrence against known events?
        s_known, s_cause = [], []
        seen = set()
        for i in np.flatnonzero(w['kind'] >= 0):
            k = int(w['kind'][i])
            if k in (sl.K['known-rule'], sl.K['known-binding']) and len(s_known) < 4000:
                pick = 'known'
            elif k in (sl.K['rule'], sl.K['binding'], sl.K['exception'], sl.K['rare']) and int(w['cause'][i]) not in seen:
                seen.add(int(w['cause'][i]))
                pick = 'cause'
            else:
                continue
            rr = w['render'][i]
            c = cue_keys(keys, w['fam'][i], w['it1'][i], w['it2'][i], rr, keys.shape[1]).astype(np.float32)
            o = keys[w['outcome'][i], rr[3] % keys.shape[1]]
            p = floor._read(c)
            s = 1 - float(p @ o) / max(float(np.linalg.norm(p) * np.linalg.norm(o)), 1e-12)
            (s_known if pick == 'known' else s_cause).append(s)
        separability = auroc(np.array(s_cause), np.array(s_known))

        # gate 3: construction, and what a cue alone gives away at first occurrence
        once = w['kind'] == sl.K['noise-once']
        cues = list(zip(w['fam'][once], w['it1'][once], w['it2'][once]))
        fams_new = {}
        for i in range(w['L']):
            f = int(w['fam'][i])
            if f not in fams_new and f not in set(int(x) for x in sl.symbol_ranges()['known']):
                fams_new[f] = int(w['kind'][i])
        new_total = len(fams_new)
        new_rule = sum(1 for v in fams_new.values() if v in (sl.K['rule'], sl.K['rule-noise']))
        out[cond] = {
            'gate1': {'floor': before, 'by_tau': best, 'residual_fit': residual_fit}, 'mlp': mlp, 'gate2_auroc': separability,
            'gate3': {'unique_once': len(set(cues)) == len(cues), 'reserved_clash': bool(set(cues) & t['reserved']), 'new_families': new_total, 'new_rule_families': new_rule,
                      'posterior_rule_given_new_family': new_rule / max(new_total, 1)},
            'gate5': {'floor_noise_capture': before['noise_capture']}}
    return seed, out


def gates():
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(WORKERS, len(PILOT_SEEDS))) as p:
        p.map(build_pool, PILOT_SEEDS)
    pools_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    with mp.get_context('fork').Pool(min(WORKERS, len(PILOT_SEEDS))) as p:
        res = dict(p.map(gate_seed, PILOT_SEEDS))
    save_json(f'{REPORT}-gates.json', res)
    save_json(f'{REPORT}-gates-runtime.json', runtime({'pools_wall_s': pools_s, 'gates_wall_s': time.perf_counter() - t1}))
    for seed, r in res.items():
        for cond, x in r.items():
            best = max(x['gate1']['by_tau'].items(), key=lambda kv: kv[1]['binding_recall'] + kv[1]['rule_generalization'])
            print(seed, cond, 'floor', {k: round(x['gate1']['floor'][k], 3) for k in ('rule_generalization', 'binding_recall', 'old_knowledge', 'noise_capture')})
            print('   ceiling tau', best[0], {k: round(best[1][k], 3) for k in ('rule_generalization', 'binding_recall', 'old_knowledge', 'noise_capture', 'known_rule', 'known_binding', 'exception_family')})
            print('   residual fit', {k: round(x['gate1']['residual_fit'][k], 3) for k in ('rule_generalization', 'old_knowledge', 'known_rule', 'exception_family')})
            print('   auroc', round(x['gate2_auroc'], 3), 'mlp', x['mlp'], 'gate3', x['gate3'])


# ---------------------------------------------------------------- selection

def primary_tasks(arm, cfg, seeds):
    j = json.dumps(cfg, sort_keys=True)
    return [(arm, j, s, g, PRIMARY['q'], PRIMARY['B'], PRIMARY['cond'], 8, 1) for s in seeds for g in GAPS]


def score_config(arm, cfg, seeds, results, gaps=GAPS):
    rows = [results[task_key((arm, json.dumps(cfg, sort_keys=True), s, g, PRIMARY['q'], PRIMARY['B'], PRIMARY['cond'], 8, 1))][0] for s in seeds for g in gaps]
    e = [r['end'] for r in rows]
    return {'objective': float(np.mean([(x['rule_generalization'] + x['binding_recall']) / 2 for x in e])), 'noise_capture': float(np.mean([x['noise_capture'] for x in e])),
            'old_knowledge': float(np.mean([x['old_knowledge'] for x in e])), 'ops': float(np.mean([r['ops_per_event'] for r in rows]))}


def excess(s):
    return max(max(0., s['noise_capture'] - LIMITS['noise_capture']) / LIMITS['noise_capture'], max(0., LIMITS['old_knowledge'] - s['old_knowledge']) / LIMITS['old_knowledge'])


def rank_key(s):
    """Lower is better: within limits first by objective then operations; otherwise the smallest largest relative excess."""
    e = excess(s)
    return (0, -s['objective'], s['ops']) if e == 0 else (1, e, -s['objective'])


def search(arm, start, grid, seeds, results):
    """Coordinate descent: each setting once per pass, in the table's order; stops after a pass with no change, 3 passes, or the configuration cap."""
    cur = dict(start)
    tried = set()
    passes = 0
    B = PRIMARY['B']
    log = []
    for passes in range(1, MAX_PASSES + 1):
        changed = False
        for name, values, _ in grid:
            cands = [{**cur, name: v} for v in values]
            cands = [c for c in cands if config_ok(arm, c, B)]
            new = [c for c in cands if json.dumps(c, sort_keys=True) not in tried]
            if len(tried) + len(new) > MAX_CONFIGS:
                log.append({'stopped': 'configuration cap'})
                return cur, tried, passes, log
            tasks = [t for c in new for t in primary_tasks(arm, c, seeds)]
            results.update(run_all(tasks))
            tried.update(json.dumps(c, sort_keys=True) for c in new)
            best = min(cands, key=lambda c: (rank_key(score_config(arm, c, seeds, results)), json.dumps(c, sort_keys=True)))
            if best != cur:
                changed = True
                cur = best
            log.append({'pass': passes, 'setting': name, 'chosen': cur[name], 'score': score_config(arm, cur, seeds, results)})
            print(arm, passes, name, cur[name], {k: round(v, 3) for k, v in log[-1]['score'].items()}, flush=True)
        if not changed:
            break
    return cur, tried, passes, log


def pilot():
    t0 = time.perf_counter()
    seeds = PILOT_SEEDS
    results, sel = {}, {}
    start = {n: s for n, _, s in LIFE_GRID}
    life, tried, passes, log = search('life', {**start, 'rho': 8, 'tau_g': .4}, LIFE_GRID, seeds, results)
    sel['life'] = {'cfg': life, 'configs': len(tried), 'passes': passes, 'log': log}
    for arm in ('life-selfcount', 'life-oracle'):
        sel[arm] = {'cfg': life, 'configs': 1, 'passes': 0, 'log': []}
    sk, tried, passes, log = search('life-sketch', life, [('rho', (4, 8, 12), 8)], seeds, results)
    sel['life-sketch'] = {'cfg': sk, 'configs': len(tried), 'passes': passes, 'log': log}
    cg, tried, passes, log = search('life-cuegroup', life, [('tau_g', (.4, .5, .6), .4)], seeds, results)
    sel['life-cuegroup'] = {'cfg': cg, 'configs': len(tried), 'passes': passes, 'log': log}
    ct, tried, passes, log = search('count', life, [('h_b', (1, 2, 3), 2), ('h_r', (1, 2, 3), 2)], seeds, results)
    sel['count'] = {'cfg': ct, 'configs': len(tried), 'passes': passes, 'log': log}
    sel['count-sketch'] = {'cfg': {**ct, 'rho': sk['rho']}, 'configs': 1, 'passes': 0, 'log': []}
    for arm, grid in GRIDS.items():
        cfg, tried, passes, log = search(arm, {n: s for n, _, s in grid}, grid, seeds, results)
        sel[arm] = {'cfg': cfg, 'configs': len(tried), 'passes': passes, 'log': log}
    sel['frozen'] = {'cfg': {'tau': life['tau'], 'theta': life['theta']}, 'configs': 1, 'passes': 0, 'log': []}
    grow, tried, passes, log = search('sleep-grow', {**sel['sleep-periodic']['cfg'], 'rank': 4}, [('rank', (2, 4), 4)], seeds, results)
    sel['sleep-grow'] = {'cfg': grow, 'configs': len(tried), 'passes': passes, 'log': log}
    results.update(run_all([t for a in ARMS for t in primary_tasks(a, sel[a]['cfg'], seeds)]))

    table = {a: {'scores': score_config(a, sel[a]['cfg'], seeds, results), 'long_gaps': score_config(a, sel[a]['cfg'], seeds, results, LONG_GAPS),
                 'by_gap': {str(g): score_config(a, sel[a]['cfg'], seeds, results, (g,)) for g in GAPS}, 'within_limits': excess(score_config(a, sel[a]['cfg'], seeds, results)) == 0} for a in ARMS}
    candidate = max(('life', 'life-sketch'), key=lambda a: table[a]['long_gaps']['objective'])
    best_comp = {str(g): max(table[a]['by_gap'][str(g)]['objective'] for a in COMPARATORS) for g in GAPS}
    feasibility = {str(g): table['life-oracle']['by_gap'][str(g)]['objective'] - best_comp[str(g)] for g in LONG_GAPS}
    save_json(f'{REPORT}-selection.json', {'selected': {a: sel[a]['cfg'] for a in ARMS}, 'candidate': candidate, 'table': table, 'search': {a: sel[a]['log'] for a in ARMS},
                                           'configs_scored': {a: sel[a]['configs'] for a in ARMS}, 'gate6_oracle_headroom_on_objective': feasibility})
    save_json(f'{REPORT}-pilot-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'runs': len(results), 'worker_cpu_s': sum(rt.get('cpu_s', 0) for _, rt in results.values())}))
    for a in ARMS:
        print(a, {k: round(v, 3) for k, v in table[a]['scores'].items()}, 'long', round(table[a]['long_gaps']['objective'], 3), table[a]['within_limits'])
    print('candidate', candidate, 'headroom', feasibility)


# ---------------------------------------------------------------- confirmation

def frozen():
    return 'Status: **frozen' in CONTRACT.read_text()


def fit_encoders():
    assert not SMOKE
    torch.set_num_threads(4)
    from experiments import device as DEV
    DEV.install()
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    syms = TK.symbol_splits()
    for seed in CONFIRM_SEEDS:
        t0 = time.perf_counter()
        TK.train(c['arch'], c['key'], seed, syms)
        print(seed, round(time.perf_counter() - t0, 1), flush=True)


def confirm():
    if not frozen():
        raise SystemExit('the contract is not frozen: confirmation refuses to run, and its seeds stay unbuilt')
    t0 = time.perf_counter()
    sel = json.loads(open(f'{REPORT}-selection.json').read())['selected']
    with mp.get_context('fork').Pool(WORKERS) as p:
        p.map(build_pool, CONFIRM_SEEDS)
    tasks = []
    for a in ARMS:
        j = json.dumps(sel[a], sort_keys=True)
        for s in CONFIRM_SEEDS:
            for g in GAPS:
                for q in QS:
                    tasks.append((a, j, s, g, q, 64, 'degraded', 8, 1))
                for B in (16, 256):
                    if config_ok(a, sel[a], B):
                        tasks.append((a, j, s, g, .2, B, 'degraded', 8, 1))
                tasks.append((a, j, s, g, .2, 64, 'clean', 8, 1))
            for g in GAPS[1:]:
                tasks.append((a, j, s, g, .2, 64, 'degraded', 8, 4))
            for n in (4, 32):
                tasks.append((a, j, s, 512, .2, 64, 'degraded', n, 1))
    results = run_all(tasks)
    out = {k: r for k, (r, _) in results.items()}
    DUR.atomic_write(f'{REPORT}-confirm.json.gz', gzip.compress(json.dumps(out, sort_keys=True, default=float).encode()))
    save_json(f'{REPORT}-confirm-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'runs': len(results), 'worker_cpu_s': sum(rt.get('cpu_s', 0) for _, rt in results.values())}))


def timing():
    sel = json.loads(open(f'{REPORT}-selection.json').read())['selected']
    seed = PILOT_SEEDS[0]
    P = pool(seed)
    t = tables(seed)
    keys = P['stream_degraded']
    old = old_knowledge(seed, t, keys)
    w = world(seed, t, 512, .2)
    evs = list(events(w, keys))[:2000]
    rows = {}
    for a in ARMS:
        if a == 'life-oracle':
            continue
        times = []
        for rep in range(16):
            arm = make_arm(a, sel[a], 64, seed, 'degraded', old)
            t0 = time.perf_counter()
            for c, o, _, _ in evs:
                arm.step(c, o)
            if rep:
                times.append((time.perf_counter() - t0) / len(evs) * 1e6)
        rows[a] = {'us_per_event_median': float(np.median(times)), 'spread_iqr': float(np.subtract(*np.percentile(times, [75, 25]))), 'repeats': len(times)}
    save_json(f'{REPORT}-timing-runtime.json', runtime({'threads': 1, 'scope': 'one thread, warm-up pass dropped, 15 repeats over 2,000 events, event loading excluded', 'rows': rows}))
    print(rows)


# ---------------------------------------------------------------- the decision rule

def bootstrap(diffs, rng, reps=10000):
    d = np.asarray(diffs)
    means = d[rng.integers(len(d), size=(reps, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse():
    res = {k: v for k, v in json.loads(gzip.decompress(open(f'{REPORT}-confirm.json.gz', 'rb').read())).items()}
    s = json.loads(open(f'{REPORT}-selection.json').read())
    sel, candidate = s['selected'], s['candidate']
    rng = nc.stream('sl_bootstrap')

    def read(arm, seed, g, reading, q=.2, B=64, cond='degraded'):
        return res[task_key((arm, json.dumps(sel[arm], sort_keys=True), seed, g, q, B, cond, 8, 1))]['end'][reading]

    def diff(a, b, g, reading):
        return bootstrap([read(a, sd, g, reading) - read(b, sd, g, reading) for sd in CONFIRM_SEEDS], rng)

    def beats(cand, comp, g, reading):
        other = 'binding_recall' if reading == 'rule_generalization' else 'rule_generalization'
        m, lo, hi = diff(cand, comp, g, reading)
        o = diff(cand, comp, g, other)
        old = diff(cand, comp, g, 'old_knowledge')
        nc_ = diff(cand, comp, g, 'noise_capture')
        ok = m >= .10 and lo > 0 and o[1] >= -.03 and old[1] >= -.02 and nc_[2] <= .02
        return ok, {'gain': (m, lo, hi), reading: None, 'other': o, 'old_knowledge': old, 'noise_capture': nc_}

    def won(cand, reading):
        wins = {g: {c: beats(cand, c, g, reading) for c in COMPARATORS} for g in LONG_GAPS}
        gaps = [g for g in LONG_GAPS if all(v[0] for v in wins[g].values())]
        return len(gaps) >= 2, gaps, {str(g): {c: v[1] for c, v in wins[g].items()} for g in LONG_GAPS}

    ops = {a: float(np.mean([res[task_key((a, json.dumps(sel[a], sort_keys=True), sd, g, .2, 64, 'degraded', 8, 1))]['ops_per_event'] for sd in CONFIRM_SEEDS for g in GAPS])) for a in ARMS}
    comparable = ops[candidate] <= COMPARABLE * ops['reservoir-replay']
    rule = won(candidate, 'rule_generalization')
    binding = won(candidate, 'binding_recall')
    if rule[0] and comparable:
        outcome = 'earns-role'
    elif binding[0] and comparable:
        outcome = 'binding-store-only'
    elif rule[0] or binding[0]:
        outcome = 'higher-cost-only'
    else:
        outcome = 'no-advantage-shown'

    def three(a, b, g, reading, margin):
        m, lo, hi = diff(a, b, g, reading)
        sign = -1 if reading == 'noise_capture' else 1
        gain, l2, h2 = sign * m, *sorted((sign * lo, sign * hi))
        verdict = 'better' if gain >= margin and l2 > 0 else 'equivalent' if l2 >= -margin and h2 <= margin else 'inconclusive'
        return {'difference': m, 'interval': [lo, hi], 'reading': verdict}
    contrasts = {'test': ('life', 'count', (('noise_capture', .02), ('old_knowledge', .02))), 'grouping': ('life', 'life-cuegroup', (('rule_generalization', .05), ('noise_capture', .02))),
                 'sketch': ('life-sketch', 'life', (('rule_generalization', .05), ('binding_recall', .05))), 'sketch-control': ('count-sketch', 'count', (('noise_capture', .02),)),
                 'self-support': ('life-selfcount', 'life', (('noise_capture', .02),))}
    secondary = {name: {str(g): {r: three(a, b, g, r, mg) for r, mg in rs} for g in LONG_GAPS} for name, (a, b, rs) in contrasts.items()}
    headroom = {str(g): {r: {c: diff('life-oracle', c, g, r) for c in COMPARATORS} for r in ('rule_generalization', 'binding_recall')} for g in LONG_GAPS}
    save_json(f'{REPORT}-analysis.json', {'outcome': outcome, 'candidate': candidate, 'comparable_cost': comparable, 'ops_per_event': ops,
                                          'rule': {'won': rule[0], 'gaps': rule[1], 'detail': rule[2]}, 'binding': {'won': binding[0], 'gaps': binding[1], 'detail': binding[2]},
                                          'secondary': secondary, 'headroom': headroom})
    print('outcome', outcome, 'candidate', candidate, 'comparable', comparable, 'rule gaps', rule[1], 'binding gaps', binding[1])


if __name__ == '__main__':
    {'encoders': fit_encoders, 'gates': gates, 'pilot': pilot, 'confirm': confirm, 'timing': timing, 'analyse': analyse}[STAGE]()
