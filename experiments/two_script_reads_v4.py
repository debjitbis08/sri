"""Where the alignment mismatch is (docs/two-script-reads-v1.md, "Where the mismatch is"). No training; no readers.

Each item's code at two levels: the projection output (the clamped input) and core_inner (what readers read).
Per level, fitted on the first half of collection items only (second half scored):
  nearest mean   the collection count whose mean code is nearest (Euclidean)
  count axis     ridge regression code -> count; reported as the mean predicted count per true count, and rounded accuracy
Usage: .venv/bin/python -m experiments.two_script_reads_v4   (TS_WORKERS; rerun to resume)
"""
import platform
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import two_script_reads as T
sys.argv = _argv
import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI

SOX, S0, P, D, X, LM, nc, ns = T.SOX, T.S0, T.P, T.D, T.X, T.LM, T.nc, T.ns
NAME = 'two-script-reads-v4'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
ARMS = ('gp', 'common-k-cand')
LEVELS = ('projection', 'core_inner')
N = len(ns.NUMBERS)


@torch.no_grad()
def levels(enc, model, x, medium):
    clamp = enc[medium](torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) if medium == 'image' else np.ascontiguousarray(x)))
    y = D.B.Solver(model, medium).solve(clamp)
    o, m, i = model.sizes
    return {'projection': clamp.numpy(), 'input_inner': y[:, m:m + i].numpy(), 'core_inner': model.split(y)['core_inner'].numpy()}


@torch.no_grad()
def reads(enc, model):
    raw = {'collection': ({n: P.DOTS['succ'][n][:T.E] for n in ns.NUMBERS}, 'image'),
           'taps': ({n: D.TAPS[('eval', n)][:T.E] for n in ns.NUMBERS}, 'sequence')}
    for lang in T.SCRIPTS:
        raw[lang] = ({n: X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]][:T.E] for n in ns.NUMBERS}, 'image')
    k = min(len(v[n]) for v, _ in raw.values() for n in ns.NUMBERS)
    codes = {}
    for kind, (v, medium) in raw.items():
        per = [levels(enc, model, v[n][:k], medium) for n in ns.NUMBERS]
        codes[kind] = {lv: np.concatenate([p[lv] for p in per]) for lv in LEVELS}
    lab = np.repeat(ns.NUMBERS, k)
    half = np.tile(np.arange(k) < k // 2, N)
    out = {}
    for lv in LEVELS:
        C = codes['collection'][lv]
        mu = np.stack([C[half & (lab == n)].mean(0) for n in ns.NUMBERS])
        Xf = C[half]; m0 = Xf.mean(0); sd = Xf.std(0) + 1e-6
        Z = (Xf - m0) / sd
        w = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ (lab[half] - 5.0))
        res = {}
        for kind in raw:
            sel = ~half if kind == 'collection' else np.ones(len(lab), bool)
            x, y = codes[kind][lv][sel], lab[sel]
            near = ((x[:, None] - mu[None]) ** 2).sum(2).argmin(1) + 1
            pred = ((x - m0) / sd) @ w + 5.0
            res[T.SHORT[kind]] = {'nearest_mean': float((near == y).mean()), 'nearest_within1': float((np.abs(near - y) <= 1).mean()),
                                  'axis_rounded': float((np.clip(np.rint(pred), 1, 9) == y).mean()),
                                  'axis_mean_by_count': [float(pred[y == n].mean()) for n in ns.NUMBERS],
                                  'nearest_mode_by_count': [int(np.bincount(near[y == n], minlength=10).argmax()) for n in ns.NUMBERS],
                                  'distance_to_collection_line': float(np.sqrt(((x[:, None] - mu[None]) ** 2).sum(2).min(1)).mean() /
                                                                       np.sqrt(((C[~half][:, None] - mu[None]) ** 2).sum(2).min(1)).mean())}
        out[lv] = res
    return out


def task(t):
    arm, seed = t
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    t0, c0 = time.perf_counter(), time.process_time()
    res, ck = {}, {}
    for point in T.POINTS:
        enc, model, ck[point] = T.load(arm, seed, point)
        res[point] = reads(enc, model)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return f'{arm}/seed{seed}', {'reads': res, 'checkpoints': ck}, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    tasks = [(a, s) for a in ARMS for s in T.SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(T.WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'code-space placement on the shared output pilot checkpoints; no training, no readers', 'arms': ARMS,
                                     'seeds': T.SEEDS, 'points': T.POINTS, 'levels': LEVELS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(T.CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': T.WORKERS, 'threads_per_worker': FI.THREADS, 'dtype': 'float32', **DEV.info(),
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


if __name__ == '__main__':
    main()
