"""Streaming retrieval v1 (docs/streaming-retrieval-v1-contract.md): input-driven retrieval against nearest-memory retrieval and
temporal smoothing, on identical sensory sequences through the frozen key encoder.

Stages (argv[1]):
  severity   corruption levels and oracle noise, chosen by the contract's rule on validation symbols of seed 299
  encoders   fit the frozen key encoders for the confirmation seeds (GPU helps)
  pilot      streams and the full grid on seeds 299, 298, every condition, then the selection rule
  confirm    refuses before the contract is frozen: selected settings on seeds 370-379 at every condition and load, and the grid on
             the primary condition as a descriptive frontier
  timing     per-frame timing of the selected settings, one thread
  analyse    the contract's decision rule on the confirmation seeds

Usage: .venv/bin/python -m experiments.streaming_retrieval pilot; NSY_WORKERS sets the pool. NSY_SMOKE=1: small and fast.
"""
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import translation_stable_keys as TK
from src import streaming_retrieval as sr

nc = X.nc
SMOKE = X.SMOKE
TK.RUNS, TK.REPORT = X.ROOT / 'runs' / 'translation-memory-v1-stable-keys', X.ROOT / 'reports' / 'translation-memory-v1-stable-keys'  # the frozen encoders, smoke runs included
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
NAME = 'streaming-retrieval-v1' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'streaming-retrieval-v1-contract.md'
WORKERS = int(os.environ.get('NSY_WORKERS', 11))

PILOT_SEEDS = (299, 298)
CONFIRM_SEEDS = tuple(range(370, 380))
FRONTIER_SEEDS = CONFIRM_SEEDS[:4]
FRAMES = 1600 if SMOKE else 14400
LOADS = (30, 120, 360)
PRIMARY_LOAD = 120
N_UNFAMILIAR = 100
RENDERINGS = 8 if SMOKE else 32
HOLD = 8
CONDITIONS = ('clean', 'degraded', 'held', 'oracle')
PRIMARY = 'degraded'
ORACLE_N = 1024
LADDER = [(.01, 0), (.02, 2), (.03, 2), (.04, 3), (.05, 3), (.06, 4), (.08, 4), (.1, 4), (.15, 6), (.2, 6), (.3, 8)]  # noise std, occluder side
TARGET_DEGRADED, TARGET_BURST = .70, .20
ORACLE_CLEAN = .96

RATES = (1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8, 1 / 12, 1 / 16, 1 / 24)
GAINS = {'idp-key': (1, 2, 4, 8, 16, 32), 'idp-wide': (4, 8, 16, 32), 'idp-oracle': (64, 128, 256, 512, 1024)}
FRAME_T = {'idp-key': (.1, .25, .5, 1, 2), 'idp-wide': (.25, .5, 1), 'idp-oracle': (.25, .5, 1)}
STEPS = {'idp-key': (1, 2, 4, 16), 'idp-wide': (1, 2), 'idp-oracle': (1, 2)}
SIGMAS = {'idp-key': (0, 1, 2, 4), 'idp-wide': (0, 2), 'idp-oracle': (0, 2)}
COMPARABLE = 6.  # retrieval multiplications per frame, against EMA
# provisional until the pilot, frozen with the contract
BUDGET = {'delay_mean': 8., 'accept_unfamiliar': .10, 'dropout': .10}
METHODS = ('nearest', 'ema', 'hysteresis', 'hmm', 'idp-key', 'idp-wide')
SMOOTHERS = ('ema', 'hysteresis')


def save_json(path, obj):
    DUR.atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=float).encode())


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'stage': STAGE, 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
            'dirty': bool(X.git('status', '--porcelain')), 'source_sha256': nc.sha256(__file__), 'mechanisms_sha256': nc.sha256(sr.__file__),
            'contract_sha256': nc.sha256(CONTRACT) if CONTRACT.exists() else None, 'python': platform.python_version(),
            'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(), 'workers': WORKERS, 'threads_per_worker': 1,
            'dtype': 'float32', 'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, **extra}


# ---------------------------------------------------------------- material

def key_encoder(seed):
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    fit = torch.load(TK.RUNS / 'fits' / f"{c['arch']}-key{c['key']}-seed{seed}.pt", weights_only=False, map_location='cpu')
    model = TK.KeyEncoder(*TK.ARCHS[c['arch']], c['key'])
    model.load_state_dict(fit['checkpoints'][c['steps']]['state'])
    return model.eval()


@torch.no_grad()
def encode(model, x):
    x = np.ascontiguousarray(x.reshape(-1, 28, 28), dtype=np.float32)
    return torch.cat([model(torch.from_numpy(x[i:i + 512])) for i in range(0, len(x), 512)]).numpy()


def corrupt(imgs, level, rng):
    """Fresh noise and a fresh occluder per frame, at a level of the ladder; level None leaves the image."""
    if level is None:
        return imgs
    std, side = LADDER[level]
    out = imgs + std * rng.standard_normal(imgs.shape, dtype=np.float32)
    corner = rng.integers(0, 29 - side, (len(imgs), 2))
    for i, (r, c) in enumerate(corner):
        out[i, r:r + side, c:c + side] = 0
    return np.clip(out, 0, 1)


def severity():
    """The first ladder level whose single-frame 120-way accuracy on validation symbols, seed 299, is at or below each target, and the
    oracle's signal strength for each of the same accuracies."""
    t0 = time.perf_counter()
    path = f'{REPORT}-severity.json'
    model = key_encoder(299)
    mat = np.load(TK.RUNS / 'material' / 'seed299.npz')
    keys = encode(model, mat['val_key'])
    imgs = mat['val_eval'].reshape(-1, 28, 28)
    labels = np.repeat(np.arange(len(mat['val_key'])), mat['val_eval'].shape[1])
    rng = nc.stream('sr_severity')
    picks = [rng.permutation(len(keys))[:120] for _ in range(5)]

    def accuracy(q):
        acc = []
        for pick in picks:
            keep = np.isin(labels, pick)
            near = pick[(q[keep] @ keys[pick].T).argmax(1)]
            acc.append((near == labels[keep]).mean())
        return float(np.mean(acc))

    clean = accuracy(encode(model, imgs))
    rows = []
    for i in range(len(LADDER)):
        rows.append({'level': i, 'noise': LADDER[i][0], 'occluder': LADDER[i][1], 'accuracy': accuracy(encode(model, corrupt(imgs, i, nc.stream('sr_severity_level', i))))})
    first = lambda target: next(r['level'] for r in rows if r['accuracy'] <= target)
    out = {'clean_accuracy': clean, 'ladder': rows, 'degraded_level': first(TARGET_DEGRADED), 'burst_level': first(TARGET_BURST), 'oracle': {}}

    def oracle_accuracy(c):
        rng = nc.stream('sr_oracle_cal', round(c, 6))
        xi = rng.standard_normal((120, ORACLE_N))
        xi = np.sign(xi) / np.sqrt(ORACLE_N)
        noise = rng.standard_normal((1200, ORACLE_N))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        who = np.repeat(np.arange(120), 10)
        q = c * xi[who] + np.sqrt(1 - c * c) * noise
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return float(((q @ xi.T).argmax(1) == who).mean())

    for name, target in (('clean', ORACLE_CLEAN), ('degraded', TARGET_DEGRADED), ('burst', TARGET_BURST)):
        lo, hi = .0, .5
        for _ in range(12):
            mid = (lo + hi) / 2
            if oracle_accuracy(mid) > target:
                hi = mid
            else:
                lo = mid
        out['oracle'][name] = {'signal': hi, 'accuracy': oracle_accuracy(hi)}
    save_json(path, out)
    save_json(f'{REPORT}-severity-runtime.json', runtime({'wall_seconds': time.perf_counter() - t0}))
    print(json.dumps({k: out[k] for k in ('clean_accuracy', 'degraded_level', 'burst_level', 'oracle')}))
    for r in rows:
        print(r)


def levels():
    return json.loads(open(f'{REPORT}-severity.json').read())


def fit_encoders():
    assert not SMOKE
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    syms = TK.symbol_splits()
    for seed in CONFIRM_SEEDS:
        t0 = time.perf_counter()
        TK.train(c['arch'], c['key'], seed, syms)
        print(seed, round(time.perf_counter() - t0, 1), flush=True)


def word_table(seed):
    """A seed's stored symbols (nested across loads: the first P) and its unfamiliar symbols, all from the pool the encoder never saw."""
    pool = sorted(TK.symbol_splits()['pool'])
    perm = nc.stream('sr_table', seed).permutation(pool)
    stored, unfamiliar = [int(c) for c in perm[:max(LOADS)]], [int(c) for c in perm[-N_UNFAMILIAR:]]
    assert not set(stored) & set(unfamiliar)
    return stored, unfamiliar


def stream_path(seed, cond, load):
    return RUNS / 'streams' / f'seed{seed}-{cond}-P{load}.npz'


def schedule(seed, load):
    return sr.make_schedule(nc.stream('sr_schedule', seed, load), load, N_UNFAMILIAR, FRAMES)


def build_streams(seed):
    """Every stream of a seed. Within a seed and load the conditions share one schedule, so only the imagery differs."""
    jobs = [(c, PRIMARY_LOAD) for c in CONDITIONS] + [(PRIMARY, p) for p in LOADS if p != PRIMARY_LOAD]
    if all(stream_path(seed, c, p).exists() for c, p in jobs):
        return
    lv = levels()
    syms = TK.symbol_splits()
    stored, unfamiliar = word_table(seed)
    table = stored + unfamiliar
    model = key_encoder(seed)
    bank = {i: PS.renderings(cp, syms['fonts'][cp], seed, 'eval', RENDERINGS) for i, cp in enumerate(table)}
    key_images = np.stack([PS.renderings(cp, syms['fonts'][cp], seed, 'train', 1)[0] for cp in stored])
    keys = encode(model, key_images)
    for cond, load in jobs:
        path = stream_path(seed, cond, load)
        if path.exists():
            continue
        sch = schedule(seed, load)
        rng = nc.stream('sr_frames', seed, cond, load)
        T = len(sch['truth'])
        shown = np.where(sch['shown'] >= load, sch['shown'] - load + max(LOADS), sch['shown'])  # index into the table
        if cond == 'oracle':
            q, k = oracle_stream(seed, load, sch, lv['oracle'], rng)
        else:
            base = None if cond == 'clean' else lv['degraded_level']
            r = rng.integers(RENDERINGS, size=T)
            if cond == 'held':
                r = r[np.arange(T) // HOLD * HOLD]
            imgs = np.stack([bank[s][i] for s, i in zip(shown, r)])
            hit = sch['corrupt']
            out = corrupt(imgs, base, rng).copy() if base is not None else imgs.copy()
            out[hit] = corrupt(imgs[hit], lv['burst_level'], rng)
            q, k = encode(model, out), keys[:load]
        buf = __import__('io').BytesIO()
        np.savez(buf, q=q.astype(np.float32), keys=k.astype(np.float32))
        DUR.atomic_write(path, buf.getvalue())


def oracle_stream(seed, load, sch, oracle, rng):
    """Independent random +-1 stored patterns and random query noise tuned to the imagery's single-frame accuracy, the mechanism's setting."""
    gen = nc.stream('sr_oracle', seed, load)
    patterns = np.sign(gen.standard_normal((load + N_UNFAMILIAR, ORACLE_N))) / np.sqrt(ORACLE_N)
    shown = sch['shown']
    T = len(shown)
    c = np.where(sch['corrupt'], oracle['burst']['signal'], oracle['degraded']['signal'])[:, None]
    noise = rng.standard_normal((T, ORACLE_N))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    q = c * patterns[shown] + np.sqrt(1 - c * c) * noise
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q, patterns[:load]


def load_stream(seed, cond, load):
    z = np.load(stream_path(seed, cond, load))
    return z['q'], z['keys'], schedule(seed, load)


def similarity_scale(cond):
    """A fixed numeric scale for similarity thresholds: the median nearest similarity on the first pilot seed's stream of this condition."""
    path = RUNS / 'scales.json'
    scales = json.loads(path.read_text()) if path.exists() else {}
    if cond not in scales:
        q, k, _ = load_stream(PILOT_SEEDS[0], cond, PRIMARY_LOAD)
        scales[cond] = round(float(np.median((q @ k.T).max(1))), 4)
        DUR.atomic_write(path, json.dumps(scales, sort_keys=True).encode())
    return scales[cond]


# ---------------------------------------------------------------- settings

def grid(method, cond):
    """[(setting, thresholds)]: a setting is what changes the run; thresholds are swept on its output without rerunning it."""
    scale = similarity_scale(cond)
    sweep = [round(float(x) * scale, 4) for x in np.arange(0, 2.001, .05)]
    unit = [round(float(x), 3) for x in np.arange(0, 1.0001, .025)]
    fracs = (.5, .65, .8, 1., 1.2)
    margins = (.05, .1, .2, .3, .5)
    if method == 'nearest':
        return [({'method': 'nearest'}, sweep)]
    if method == 'ema':
        return [({'method': 'ema', 'a': a}, sweep) for a in RATES]
    if method == 'hysteresis':
        return [({'method': 'hysteresis', 'a': a, 'tau': round(f * scale, 4), 'delta': round(d * scale, 4)}, None) for a in RATES for f in fracs for d in margins]
    if method == 'hmm':
        return [({'method': 'hmm', 'kappa': round(k / scale, 3), 'hazard': h, 'tau': round(f * scale, 4)}, None)
                for k in (5, 10, 20, 40, 80) for h in (.001, .003, .01, .03, .1) for f in fracs]
    if method == 'idp-wide' and cond != PRIMARY:
        return []  # the cost of width is read on the primary condition; oracle patterns are already wide and idp-key reads them directly
    kind = 'idp-oracle' if cond == 'oracle' else method
    return [({'method': method, 'gain': g, 'tau': t, 'steps': k, 'sigma': s}, unit)
            for g in GAINS[kind] for t in FRAME_T[kind] for k in STEPS[kind] for s in SIGMAS[kind] if k < 16 or t >= .25]


def patterns_for(method, keys, q, seed):
    """Stored patterns and the similarities that drive the saliency weights. Oracle patterns are already wide and are read directly."""
    if method == 'idp-wide':
        proj = nc.stream('sr_project', seed).standard_normal((ORACLE_N, keys.shape[1])).astype(np.float32)
        pat = sr.expand(keys, proj)[1]
        return pat, sr.expand(q, proj)[0] @ pat.T / np.sqrt(ORACLE_N)
    return sr.to_patterns(keys), q @ keys.T


def run_setting(cfg, seed, cond, load, q, keys):
    """A decision source for one setting: (index, score) for a threshold sweep, or a decision array."""
    S = q @ keys.T
    m = cfg['method']
    if m == 'nearest':
        return sr.nearest(S)
    if m == 'ema':
        return sr.ema(S, cfg['a'])
    if m == 'hysteresis':
        return sr.hysteresis(S, cfg['a'], cfg['tau'], cfg['delta'])
    if m == 'hmm':
        return sr.hmm_filter(S, cfg['kappa'], cfg['hazard'], cfg['tau'])
    pat, sim = patterns_for(m, keys, q, seed)
    rng = nc.stream('sr_idp', seed, cond, load, json.dumps(cfg, sort_keys=True))
    return sr.idp(sim, pat, cfg['gain'], cfg['tau'], cfg['steps'], cfg['sigma'], rng)


def decisions(cfg, source, thr):
    return source if thr is None else sr.threshold(*source, thr)


def method_ops(method, load, D, cfg=None, cond=None):
    if method.startswith('idp'):
        N = ORACLE_N if method == 'idp-wide' else D
        return sr.ops('idp', load, D, cfg['steps'], N)
    return sr.ops(method, load, D)


# ---------------------------------------------------------------- tasks

KEEP = ('wrong_per_1000', 'delay_mean', 'accept_unfamiliar', 'dropout')


def chunks(method, cond):
    items = grid(method, cond)
    if method.startswith('idp'):
        return sorted({c['gain'] for c, _ in items})
    return [0] if items else []


def grid_task(task):
    """All settings of a method (for input-driven dynamics, of one gain) on one stream; compact rows per setting and threshold."""
    seed, cond, load, method, chunk = task
    t0, c0 = time.perf_counter(), time.process_time()
    q, keys, sch = load_stream(seed, cond, load)
    rows = []
    for cfg, thrs in grid(method, cond):
        if method.startswith('idp') and cfg['gain'] != chunk:
            continue
        src = run_setting(cfg, seed, cond, load, q, keys)
        for thr in ([None] if thrs is None else thrs):
            r = sr.score(decisions(cfg, src, thr), sch)
            rows.append({'cfg': cfg, 'thr': thr, **{k: round(r[k], 5) for k in KEEP}})
    return (f'{seed}/{cond}/P{load}/{method}/{chunk}', {'rows': rows},
            {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'settings': len(grid(method, cond)), 'worker_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def build_task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    build_streams(seed)
    return seed, time.perf_counter() - t0, time.process_time() - c0


_durable = DUR.durable(grid_task, RUNS / 'done', 'seed{0}-{1}-P{2}-{3}-{4}')


def full_task(task):
    """The full readings of one selected setting on one stream."""
    seed, cond, load, method, cfg, thr = task
    q, keys, sch = load_stream(seed, cond, load)
    r = sr.score(decisions(cfg, run_setting(cfg, seed, cond, load, q, keys), thr), sch)
    return f'{seed}/{cond}/P{load}/{method}', r


def run_all(fn, tasks):
    if not tasks:
        return []
    with mp.get_context('fork').Pool(WORKERS) as pool:
        return list(pool.imap_unordered(fn, tasks))


def pooled_rows(results, seeds, cond, load, method):
    """{(setting, thr): the mean of each kept reading across seeds}, only for settings present on every seed."""
    merged = {}
    for key, res, _ in results:
        s, c, p, m, _chunk = key.split('/')
        if c != cond or p != f'P{load}' or m != method or int(s) not in seeds:
            continue
        for row in res['rows']:
            merged.setdefault(json.dumps([row['cfg'], row['thr']], sort_keys=True), []).append(row)
    return [{'cfg': v[0]['cfg'], 'thr': v[0]['thr'], **{k: float(np.mean([x[k] for x in v])) for k in KEEP}} for v in merged.values() if len(v) == len(seeds)]


def feasible(row, budget):
    return all(row[k] <= budget[k] for k in budget)


def select_from(rows, budget, ops_of):
    """The contract's rule: among settings within budget, the fewest wrong switches, then the fewest operations. If none is within
    budget, nothing is selected and the closest setting is reported, by largest relative excess over the budget."""
    ok = [r for r in rows if feasible(r, budget)]
    if ok:
        return min(ok, key=lambda r: (r['wrong_per_1000'], ops_of(r['cfg']), json.dumps(r['cfg'], sort_keys=True), r['thr'] or 0)), True
    return min(rows, key=lambda r: (max(r[k] / budget[k] for k in budget), r['wrong_per_1000'], json.dumps(r['cfg'], sort_keys=True))), False


def tasks_for(seeds, conds=CONDITIONS, methods=METHODS):
    out = []
    for seed in seeds:
        for cond in conds:
            for method in methods:
                for chunk in chunks(method, cond):
                    out.append((seed, cond, PRIMARY_LOAD, method, chunk))
    return out


def select_all(results, seeds, budget=None):
    budget = budget or BUDGET
    D = 32
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        width = ORACLE_N if cond == 'oracle' else D
        ema_ops = sr.ops('ema', PRIMARY_LOAD, width)
        for method in METHODS:
            rows = pooled_rows(results, seeds, cond, PRIMARY_LOAD, method)
            if not rows:
                continue
            ops_of = lambda cfg, m=method: method_ops(m, PRIMARY_LOAD, width, cfg)
            tiers = {'any': rows}
            if method.startswith('idp'):
                tiers = {'comparable': [r for r in rows if ops_of(r['cfg']) <= COMPARABLE * ema_ops], 'any': rows}
            for tier, sub in tiers.items():
                if not sub:
                    continue
                row, ok = select_from(sub, budget, ops_of)
                out[cond][method if tier == 'any' else f'{method}:{tier}'] = {**row, 'within_budget': ok, 'candidates': len(sub), 'ops': ops_of(row['cfg']), 'ops_ratio_to_ema': ops_of(row['cfg']) / ema_ops}
    return out


def frontier(rows, budgets):
    """For each delay budget, the fewest wrong switches within the reject and dropout limits."""
    out = {}
    for d in budgets:
        ok = [r for r in rows if r['delay_mean'] <= d and r['accept_unfamiliar'] <= BUDGET['accept_unfamiliar'] and r['dropout'] <= BUDGET['dropout']]
        out[str(d)] = None if not ok else min(ok, key=lambda r: r['wrong_per_1000'])
    return out


# ---------------------------------------------------------------- stages

def pilot():
    t0 = time.perf_counter()
    seeds = (PILOT_SEEDS[0],) if SMOKE else PILOT_SEEDS
    for seed, wall, cpu in run_all(build_task, list(seeds)):
        print('streams', seed, round(wall, 1), flush=True)
    for cond in CONDITIONS:
        similarity_scale(cond)
    build_s = time.perf_counter() - t0
    tasks = tasks_for(seeds)
    t1 = time.perf_counter()
    results = []
    with mp.get_context('fork').Pool(WORKERS) as pool:
        for i, (key, res, rt) in enumerate(pool.imap_unordered(_durable, tasks)):
            results.append((key, res, rt))
            print(i + 1, '/', len(tasks), key, round(rt['wall_seconds'], 1), flush=True)
    results.sort(key=lambda z: z[0])
    sel = select_all(results, set(seeds))
    frontiers = {cond: {m: frontier(pooled_rows(results, set(seeds), cond, PRIMARY_LOAD, m), (4, 8, 16)) for m in METHODS} for cond in CONDITIONS}
    save_json(f'{REPORT}-pilot.json', {'seeds': list(seeds), 'budget': BUDGET, 'selected': sel, 'frontier': frontiers, 'levels': levels()})
    cpu = sum(rt['cpu_seconds'] for _, _, rt in results)
    save_json(f'{REPORT}-pilot-runtime.json', runtime({'wall_seconds': time.perf_counter() - t0, 'stream_build_wall_seconds': build_s, 'grid_wall_seconds': time.perf_counter() - t1, 'worker_cpu_seconds': cpu,
                                                       'tasks': len(results), 'settings_scored': sum(len(r['rows']) for _, r, _ in results), 'by_task': {k: rt for k, _, rt in results}}))
    print_sel(sel)


def print_sel(sel):
    for cond, row in sel.items():
        print(cond)
        for m, r in row.items():
            print(f"  {m:22s} wrong {r['wrong_per_1000']:7.2f} delay {r['delay_mean']:6.2f} accept {r['accept_unfamiliar']:.3f} dropout {r['dropout']:.3f} ok {r['within_budget']} ops {r['ops']} x{r['ops_ratio_to_ema']:.1f}  {r['cfg']} thr={r['thr']}")


def frozen():
    return CONTRACT.exists() and 'Status: **frozen' in CONTRACT.read_text()


def confirm():
    assert SMOKE or frozen(), 'the contract must be frozen before confirmation'
    t0 = time.perf_counter()
    seeds = CONFIRM_SEEDS[:1] if SMOKE else CONFIRM_SEEDS
    built = run_all(build_task, list(seeds))
    build_s = time.perf_counter() - t0
    sel = json.loads(open(f'{REPORT}-pilot.json').read())['selected']
    t1 = time.perf_counter()
    jobs = []
    for seed in seeds:
        for cond in CONDITIONS:
            for name, r in sel[cond].items():
                jobs.append((seed, cond, PRIMARY_LOAD, name, r['cfg'], r['thr']))
        for load in LOADS:
            if load != PRIMARY_LOAD:
                for name, r in sel[PRIMARY].items():
                    jobs.append((seed, PRIMARY, load, name, r['cfg'], r['thr']))
    full = dict(run_all(full_task, jobs))
    selected_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    results = []
    frontier_seeds = seeds[:len(FRONTIER_SEEDS)]
    frontier_methods = tuple(m for m in METHODS if m != 'idp-wide')
    tasks = tasks_for(frontier_seeds, conds=(PRIMARY,), methods=frontier_methods)
    with mp.get_context('fork').Pool(WORKERS) as pool:
        for i, (key, res, rt) in enumerate(pool.imap_unordered(_durable, tasks)):
            results.append((key, res, rt))
            if i % 10 == 0:
                print(i + 1, '/', len(tasks), flush=True)
    results.sort(key=lambda z: z[0])
    fr = {m: frontier(pooled_rows(results, set(frontier_seeds), PRIMARY, PRIMARY_LOAD, m), (4, 8, 16)) for m in frontier_methods}
    with gzip.open(f'{REPORT}-confirm.json.gz', 'wt') as f:
        json.dump({'seeds': list(seeds), 'frontier_seeds': list(frontier_seeds), 'selected': sel, 'readings': full, 'frontier_primary': fr}, f, sort_keys=True, default=float)
    save_json(f'{REPORT}-confirm-runtime.json', runtime({'wall_seconds': time.perf_counter() - t0, 'stream_build_wall_seconds': build_s, 'selected_wall_seconds': selected_s, 'frontier_wall_seconds': time.perf_counter() - t2,
                                                         'frontier_worker_cpu_seconds': sum(rt['cpu_seconds'] for _, _, rt in results), 'stream_build_worker_cpu_seconds': sum(c for _, _, c in built), 'selected_runs': len(jobs), 'frontier_tasks': len(results), 'settings_scored': sum(len(r['rows']) for _, r, _ in results)}))


def timing():
    """One thread, repeated passes over the first frames of a pilot stream, after a warm-up pass. Decisions only; the encoder is timed separately."""
    sel = json.loads(open(f'{REPORT}-pilot.json').read())['selected'][PRIMARY]
    seed = PILOT_SEEDS[0]
    q, keys, _ = load_stream(seed, PRIMARY, PRIMARY_LOAD)
    n, passes = 200 if SMOKE else 2000, 5 if SMOKE else 15
    q = q[:n]
    out = {}
    for name, r in sel.items():
        t = []
        for i in range(passes + 1):
            t0 = time.perf_counter()
            src = run_setting(r['cfg'], seed, PRIMARY, PRIMARY_LOAD, q, keys)
            decisions(r['cfg'], src, r['thr'])
            t.append((time.perf_counter() - t0) / n * 1e6)
        t = t[1:]
        out[name] = {'cfg': r['cfg'], 'us_per_frame_median': float(np.median(t)), 'us_per_frame_iqr': float(np.subtract(*np.percentile(t, [75, 25]))), 'passes': passes, 'frames_per_pass': n, 'multiplications_per_frame': r['ops'],
                     'state_floats': sr.state_floats('idp' if name.startswith('idp') else name, PRIMARY_LOAD, keys.shape[1], ORACLE_N if name.startswith('idp-wide') else None),
                     'scope': 'includes the similarity matrix-vector work per frame as a batch, and allocation; excludes the encoder'}
    model = key_encoder(seed)
    x = torch.zeros(256, 28, 28)
    t = []
    with torch.no_grad():
        for i in range(21):
            t0 = time.perf_counter()
            model(x)
            t.append((time.perf_counter() - t0) / len(x) * 1e6)
    out['encoder'] = {'us_per_frame_median': float(np.median(t[1:])), 'us_per_frame_iqr': float(np.subtract(*np.percentile(t[1:], [75, 25]))), 'passes': 20, 'frames_per_pass': len(x), 'batched': True,
                      'multiplications_per_frame': 16 * 9 * 14 * 14 + 32 * 16 * 9 * 7 * 7 + 32 * 49 * 128 + 128 * 32, 'persistent_bytes': 839040}
    save_json(f'{REPORT}-timing-runtime.json', runtime({'timings': out}))
    for k, v in out.items():
        print(k, {a: v[a] for a in ('us_per_frame_median', 'us_per_frame_iqr', 'multiplications_per_frame') if a in v})


def bootstrap(diffs, rng, reps=10000):
    d = np.asarray(diffs, float)
    means = d[rng.integers(len(d), size=(reps, len(d)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


MARGINS = {'delay_mean': 1., 'accept_unfamiliar': .02, 'dropout': .02}
MIN_REDUCTION = .20


def analyse():
    with gzip.open(f'{REPORT}-confirm.json.gz', 'rt') as f:
        z = json.load(f)
    seeds, read, sel = z['seeds'], z['readings'], z['selected'][PRIMARY]
    get = lambda m, k, cond=PRIMARY, load=PRIMARY_LOAD: np.array([read[f'{s}/{cond}/P{load}/{m}'][k] for s in seeds], float)
    smoother = min(SMOOTHERS, key=lambda m: (not sel[m]['within_budget'], sel[m]['wrong_per_1000']))
    rng = nc.stream('sr_bootstrap')
    table = {cond: {m: {k: float(get(m, k, cond).mean()) for k in ('wrong_per_1000', 'wrong_outside_bursts_per_1000', 'delay_mean', 'delay_median', 'delay_miss', 'accept_unfamiliar', 'reject_delay_mean', 'dropout', 'settled_correct', 'frame_correct')}
                    for m in z['selected'][cond]} for cond in CONDITIONS}
    capture = {m: {f'{g}:{k}': float(np.nanmean(np.array([read[f'{s}/{PRIMARY}/P{PRIMARY_LOAD}/{m}'][g][k] if read[f'{s}/{PRIMARY}/P{PRIMARY_LOAD}/{m}'][g][k] is not None else np.nan for s in seeds], float)))
                   for g, keys in (('capture_by_kind', sr.BURST_KINDS), ('capture_by_length', [str(n) for n in sr.BURST_LENGTHS])) for k in keys} for m in sel}
    loads = {str(p): {m: {k: float(get(m, k, PRIMARY, p).mean()) for k in ('wrong_per_1000', 'delay_mean', 'accept_unfamiliar', 'dropout')} for m in sel} for p in LOADS}
    contrasts = {}
    for name in ('idp-key:comparable', 'idp-key', 'idp-wide', 'hmm'):
        if name not in sel:
            continue
        d = get(name, 'wrong_per_1000') - get(smoother, 'wrong_per_1000')
        base = get(smoother, 'wrong_per_1000').mean()
        guards = {k: {'difference': float((get(name, k) - get(smoother, k)).mean()), 'ci95': bootstrap(get(name, k) - get(smoother, k), rng), 'margin': MARGINS[k]} for k in MARGINS}
        ci = bootstrap(d, rng)
        reduction = float(-d.mean() / base) if base else 0.
        guards_ok = all(g['ci95'][1] <= g['margin'] for g in guards.values())
        contrasts[name] = {'against': smoother, 'wrong_difference': float(d.mean()), 'ci95': ci, 'relative_reduction': reduction, 'guards': guards, 'guards_hold': guards_ok,
                           'beats': bool(ci[1] < 0 and reduction >= MIN_REDUCTION and guards_ok), 'ops_ratio_to_ema': sel[name]['ops_ratio_to_ema'], 'within_budget_on_pilot': sel[name]['within_budget']}
    primary = contrasts.get('idp-key:comparable', {})
    outcome = 'earns-role' if primary.get('beats') else ('beats-only-at-higher-cost' if any(contrasts[n]['beats'] for n in ('idp-key', 'idp-wide') if n in contrasts) else 'does-not-beat')
    out = {'seeds': seeds, 'smoother': smoother, 'table': table, 'capture': capture, 'loads': loads, 'contrasts': contrasts, 'outcome': outcome, 'frontier_primary': z['frontier_primary']}
    save_json(f'{REPORT}-analysis.json', out)
    print(outcome, smoother)
    print(json.dumps(contrasts, indent=1, default=float))


if __name__ == '__main__':
    {'severity': severity, 'encoders': fit_encoders, 'pilot': pilot, 'confirm': confirm, 'timing': timing, 'analyse': analyse}[STAGE]()
