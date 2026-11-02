"""Shared-core sparse memory v1: the refresh test. Can stored values keep up with a learning host without a frozen copy of it and without replaying pairs? Replays the long-term memory host's cached stream as
experiments/shared_core_moving_values.py does, and refreshes stored values at the end of blocks 4, 5 and 6 from probe images: evaluation collections held out of reading (their item index even); reading uses the odd ones.

Usage: NSY_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_refresh pilot
"""
import json
import time

import numpy as np

from experiments import shared_core_moving_values as MV
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from src.sparse_local_memory import Memory

nc, X, P = S0.nc, S0.X, S0.P
NAME = 'shared-core-sparse-memory-v1-refresh'
REPORT = X.ROOT / 'reports' / NAME
H, K, TAU = MV.H, MV.K, MV.TAU
N, E = S1.N, S0.EVAL_RENDERS
RIDGE = 1e-3
PROBES = {'map-135': np.arange(0, P.EVAL_PER, 2), 'map-27': np.array([0, 2, 4])}
ARMS = ('none', 'map-135', 'map-27', 'map-135-half', 'snap', 'snap-train', 'oracle-count', 'oracle-train')
NO_BLOCK6 = ('snap-train', 'oracle-train')  # the cache has no training collections in block 6


def affine(x, y):
    """Ridge least squares y = (x - mx) M + my; x, y (n, d)."""
    mx, my = x.mean(0), y.mean(0)
    a, b = (x - mx).astype(np.float64), (y - my).astype(np.float64)
    M = np.linalg.solve(a.T @ a + RIDGE * np.eye(a.shape[1]), a.T @ b)
    return lambda v: ((v - mx) @ M + my).astype(np.float32)


def query_items(seed, tag):
    """Each evaluation query's collection item index j, and its candidates' order, from the stream that drew them."""
    rng = nc.stream('ap_eval_candidates', seed, tag)
    order = np.stack([rng.permutation(9) for _ in range(N * E)])
    return order, rng.integers(P.EVAL_PER, size=N * E)


def probe_table(evb, items):
    """(9, EVAL_PER, d) with the code of each collection image at a checkpoint, from the cached candidates of all three evaluation sets."""
    d = evb['recall__A___c_live'].shape[-1]
    out = np.full((9, P.EVAL_PER, d), np.nan, np.float32)
    for tag, (order, j) in items.items():
        c = evb[f'recall__{tag}___c_live']
        for t in range(len(j)):
            out[order[t], j[t]] = c[t]
    return out


def split_refresh(arm, old_v, new_v, nz_old, nz_new, table, b, majority, tp):
    """The refreshed values of each unit's part written before this block, and of its part written in it. table[b] holds probe codes at each checkpoint."""
    old_v, new_v = old_v.copy(), new_v.copy()
    if arm == 'none':
        return old_v, new_v
    now, prev = table[b], table[b - 1]
    if arm.startswith('map'):
        j = PROBES[arm[:-5] if arm.endswith('-half') else arm]
        f = affine(prev[:, j].reshape(-1, prev.shape[-1]), now[:, j].reshape(-1, now.shape[-1]))
        old_v[nz_old] = f(old_v[nz_old])
        if arm.endswith('-half') and nz_new.any():  # writes in this block lag by half of it on average
            new_v[nz_new] += .5 * (f(new_v[nz_new]) - new_v[nz_new])
        return old_v, new_v
    j = PROBES['map-135']
    p_now, p_prev = now[:, j].mean(1), prev[:, j].mean(1)
    if arm == 'snap':
        for mask, v, ref in ((nz_old, old_v, p_prev), (nz_new, new_v, p_now)):
            if mask.any():
                v[mask] = p_now[(S1.unit(v[mask]) @ S1.unit(ref).T).argmax(1)]
        return old_v, new_v
    if arm == 'snap-train':
        for mask, v, ref in ((nz_old, old_v, tp[b - 1]), (nz_new, new_v, tp[b])):
            if mask.any():
                v[mask] = tp[b][(S1.unit(v[mask]) @ S1.unit(ref).T).argmax(1)]
        return old_v, new_v
    target = p_now if arm == 'oracle-count' else tp[b]
    return target[majority], target[majority]


def map_quality(table, tp):
    """How well a drift map fitted on even collection items predicts: odd items at the same step, and the training rendering's count prototypes. R^2 of residual against no refresh: 1 is perfect, 0 no better than leaving codes alone, below 0 worse."""
    out = {}
    even, odd = PROBES['map-135'], np.arange(1, P.EVAL_PER, 2)
    for b in (4, 5, 6):
        x, y = table[b - 1], table[b]
        f = affine(x[:, even].reshape(-1, x.shape[-1]), y[:, even].reshape(-1, y.shape[-1]))
        gain = lambda a, t: float(1 - ((f(a) - t) ** 2).sum() / ((a - t) ** 2).sum())
        row = {'held_out_items': gain(x[:, odd].reshape(-1, x.shape[-1]), y[:, odd].reshape(-1, y.shape[-1]))}
        if tp.get(b) is not None and tp.get(b - 1) is not None:
            row['training_prototypes'] = gain(tp[b - 1], tp[b])
        row['probe_drift_rms'] = float(np.sqrt(((x[:, even] - y[:, even]) ** 2).sum(-1).mean()))
        out[str(b)] = row
    return out


def one_seed(seed):
    d = MV.CACHE / f'sparse48-seed{seed}-full-N45'
    c = S1.load(seed)
    ev = {b: dict(np.load(d / f'eval-{b}.npz')) for b in MV.BLOCKS}
    items = {t: query_items(seed, t) for t in ('A', 'B', 'unknown')}
    gate = {f'candidate_order_{t}': bool(np.array_equal(items[t][0] + 1, ev[3][f'recall__{t}___counts'])) for t in 'AB'}
    table = {b: probe_table(ev[b], items) for b in MV.BLOCKS}
    gate['probe_table_complete'] = bool(all(np.isfinite(table[b]).all() for b in MV.BLOCKS))
    base = Memory(H, K, TAU, 32, S1.CORE)
    frozen = Memory(H, K, TAU, 32, S1.CORE)
    counts = np.zeros((H, 9), np.int64)
    arms = {a: np.zeros((H, S1.CORE), np.float32) for a in ARMS}
    tp = {3: c['protos'], 6: None}
    reads = {a: {} for a in ARMS}
    reads['frozen-block3'] = {}
    streams = {'A': S1.stream('A', seed, c), 'B': S1.stream('B', seed, c)}

    def read_all(b, tags):
        for name in tags:
            s = MV.eval_set(ev[b], name)
            odd = items[name][1] % 2 == 1
            for a in ARMS:
                if a in NO_BLOCK6 and b == 6:
                    reads[a].setdefault(str(b), {})[name] = None
                    continue
                base.u, keep = arms[a], base.u
                content, ok = base.read(S1.unit(s['q'][odd]))
                base.u = keep
                out = S1.decode(content, ok, s['truth'][odd], s['cand'][odd], s['counts'][odd] - 1, tp[b] if tp[b] is not None else c['protos'])
                if tp[b] is None:
                    out['prototype'] = None
                reads[a].setdefault(str(b), {})[name] = out
            o = MV.eval_set(ev[3], name)
            content, ok = frozen.read(S1.unit(o['q'][odd]))
            reads['frozen-block3'].setdefault(str(b), {})[name] = S1.decode(content, ok, o['truth'][odd], o['cand'][odd], o['counts'][odd] - 1, c['protos'])

    for b, tag in ((4, 'A'), (5, 'B'), (6, None)):
        u_before, n_before = base.u.copy(), base.n.copy()
        if tag is not None:
            z = np.load(d / f'train-{b}.npz')
            s = streams[tag]
            t, target = z['t_live'], z['target'].astype(int)
            for zq, v, fv, m in zip(S1.unit(z['q_frozen']), t, c['coll_train'][s[:, 2], c['counts'][s[:, 0]] - 1], target):
                I = base.write(zq, v)
                frozen.write(zq, fv)
                counts[I, m] += 1
            tp[b] = np.stack([t[-MV.WINDOW:][target[-MV.WINDOW:] == m].mean(0) for m in range(9)])
        delta, dn = base.u - u_before, base.n - n_before
        majority = counts.argmax(1)
        nz_old, nz_new = n_before > 0, dn > 0
        for a in ARMS:
            if a in NO_BLOCK6 and b == 6:
                continue
            old_v = np.divide(arms[a], np.maximum(n_before, 1)[:, None])
            new_v = np.divide(delta, np.maximum(dn, 1)[:, None])
            ov, nv = split_refresh(a, old_v, new_v, nz_old, nz_new, table, b, majority, tp)
            arms[a] = ov * n_before[:, None] + nv * dn[:, None]
        read_all(b, 'A' if b == 4 else 'AB')
    gate['none_equals_base'] = bool(np.array_equal(arms['none'], base.u) or np.allclose(arms['none'], base.u, atol=1e-2))
    return {'reads': reads, 'gate': gate, 'map_quality': map_quality(table, tp), 'units': base.h, 'forced': base.forced, 'probes': {k: int(9 * len(v)) for k, v in PROBES.items()}}


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    out = {str(s): one_seed(s) for s in S0.SEEDS}
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, A then B only, refresh at block ends; reading on collection items with odd index, probes from even', 'setting': {'H': H, 'k': K, 'tau': TAU}, 'results': out})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'replay on cached codes', 'device': 'cpu', 'workers': 1, 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}))


if __name__ == '__main__':
    main()
