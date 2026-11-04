"""Relation codes check v1, part A2: context-normalized identity (docs/relation-codes-check-v1.md). No training of the system.

Per checkpoint and level, each kind's codes are standardized over that kind's own evaluation items (temporal/context normalization:
the context is the set of items of one kind, all nine counts, 30 each). Collections are standardized over their own items too.
Identity by nearest collection-count mean and by the ridge count axis (both fitted on the first half of collections), raw and
context-normalized; plus the rank code for reference.
Usage: .venv/bin/python -m experiments.relation_codes_cn   (TS_WORKERS; rerun to resume)
"""
import platform
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relation_codes_check as R
sys.argv = _argv
import numpy as np
import torch

T, S0, X, LM, nc, ns = R.T, R.S0, R.X, R.LM, R.nc, R.ns
DEV, DUR, FI = R.DEV, R.DUR, R.FI
NAME = 'relation-codes-check-v1-cn'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
N = 9


def ident(codes):
    k = min(len(codes[kd][n]) for kd in codes for n in codes[kd])
    half = np.arange(k) < k // 2
    lab = np.repeat(np.arange(1, N + 1), k)
    full = {kd: np.concatenate([codes[kd][n][:k] for n in range(1, N + 1)]) for kd in codes}
    out = {}
    for norm in ('raw', 'context'):
        if norm == 'raw':
            z = full
        else:
            z = {kd: (v - v.mean(0)) / (v.std(0) + 1e-6) for kd, v in full.items()}
        C = z['collection']; ch = np.tile(half, N)
        mu = np.stack([C[ch & (lab == n)].mean(0) for n in range(1, N + 1)])
        m0 = C[ch].mean(0)
        w = np.linalg.solve((C[ch] - m0).T @ (C[ch] - m0) + np.eye(C.shape[1]), (C[ch] - m0).T @ (lab[ch] - 5.0))
        for kd, v in z.items():
            s = ~ch if kd == 'collection' else np.ones(len(lab), bool)
            x, y = v[s], lab[s]
            near = ((x[:, None] - mu[None]) ** 2).sum(2).argmin(1) + 1
            axis = np.clip(np.rint((x - m0) @ w + 5.0), 1, N)
            out.setdefault(T.SHORT[kd], {})[norm] = {'nearest': float((near == y).mean()), 'nearest_within1': float((np.abs(near - y) <= 1).mean()),
                                                     'axis': float((axis == y).mean()), 'axis_within1': float((np.abs(axis - y) <= 1).mean())}
    return out


def task(t):
    arm, seed = t
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    t0, c0 = time.perf_counter(), time.process_time()
    res, ck = {}, {}
    for point in R.POINTS:
        enc, model, ck[point] = T.load(arm, seed, point)
        codes = R.item_codes(enc, model)
        res[point] = {lv: ident(codes[lv]) for lv in R.LEVELS}
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return f'{arm}/seed{seed}', {'reads': res, 'checkpoints': ck}, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    tasks = [(a, s) for a in R.ARMS for s in T.SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(T.WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'context-normalized identity; no training', 'arms': R.ARMS, 'seeds': T.SEEDS, 'points': R.POINTS,
                                     'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(R.CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': T.WORKERS, 'threads_per_worker': FI.THREADS, **DEV.info(),
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


if __name__ == '__main__':
    main()
