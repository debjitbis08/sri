"""Identity from order (docs/two-script-reads-v1.md, "Identity from order"). No training of the system; the v1 checkpoints, arms, seeds, items.

  order profile   s_c(x) = mean over M's participants of count c of (more(x, m) - more(m, x)) / 2, for c = 1..9.
  placement       argmax_n [sum_{c<n} s_c - sum_{c>n} s_c]   (fixed rule, a diagnostic)
  learned read    linear softmax read profile -> count, fitted on the first half of collection and taps items only; scored on the rest
  same            identification over M as in the follow-up (a count's score is its best participant's logit)
Per kind: accuracy of each, within one count, and the recovered share (derived accuracy on items `same` misidentifies).

Usage: .venv/bin/python -m experiments.two_script_reads_v3   (TS_WORKERS, default 4; rerun to resume)
       .venv/bin/python -m experiments.two_script_reads_v3 summary [arm]
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
NAME = 'two-script-reads-v3'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
FIT_KINDS = ('collection', 'taps')
N = len(ns.NUMBERS)
SIGN = np.sign(np.arange(1, N + 1)[:, None] - np.arange(1, N + 1)[None])  # (n, c): +1 if c < n


def placement(s):
    return (s @ SIGN.T).argmax(1) + 1


def fit_read(Xf, yf, seed):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(N, N).double()
    x, y = torch.from_numpy(Xf).double(), torch.from_numpy(yf - 1).long()
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(x), y) + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda s: lin(torch.from_numpy(s).double()).argmax(1).numpy() + 1


@torch.no_grad()
def reads(enc, model, seed):
    img = lambda x: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), 'image')
    items = {'collection': {n: img(P.DOTS['succ'][n][:T.E]) for n in ns.NUMBERS},
             'taps': {n: D.codes_for(enc, model, D.TAPS[('eval', n)][:T.E], 'sequence') for n in ns.NUMBERS}}
    for lang in T.SCRIPTS:
        items[lang] = {n: img(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]][:T.E]) for n in ns.NUMBERS}
    k = min(len(v[n]) for v in items.values() for n in ns.NUMBERS)
    M = model.participants()
    inv_n = np.array([n for n, _ in SOX.W['inventory']])
    cols = [np.flatnonzero(inv_n == n) for n in ns.NUMBERS]
    K, d = len(M), M.shape[1]

    def profile(x):
        n = len(x)
        xa, mb = x[:, None].expand(-1, K, -1).reshape(-1, d), M[None].expand(n, -1, -1).reshape(-1, d)
        a = ((model.relation_logits(xa, mb, 2)
              - model.relation_logits(mb, xa, 2)) / 2).reshape(n, K).numpy()
        return np.stack([a[:, c].mean(1) for c in cols], 1)

    def same_id(x):
        lg = model.identify(x, M).numpy()
        return np.stack([lg[:, c].max(1) for c in cols], 1).argmax(1) + 1

    prof, sid, lab = {}, {}, {}
    for kind in T.KINDS:
        xs = torch.cat([items[kind][n][:k] for n in ns.NUMBERS])
        prof[kind], sid[kind], lab[kind] = profile(xs), same_id(xs), np.repeat(ns.NUMBERS, k)
    half = np.tile(np.arange(k) < k // 2, N)  # first half of each count's items
    read = fit_read(np.concatenate([prof[kd][half] for kd in FIT_KINDS]), np.concatenate([lab[kd][half] for kd in FIT_KINDS]), seed)
    out = {}
    for kind in T.KINDS:
        sel = ~half if kind in FIT_KINDS else np.ones(len(lab[kind]), bool)
        y, s, sm = lab[kind][sel], prof[kind][sel], sid[kind][sel]
        r = {'same': float((sm == y).mean()), 'n_items': int(sel.sum()), 'same_wrong_n': int((sm != y).sum()),
             'profile_mean_by_count': prof[kind].reshape(N, k, N).mean(1).tolist()}
        for name, pred in (('placement', placement(s)), ('learned', read(s))):
            ok, wrong = pred == y, sm != y
            r[name] = float(ok.mean())
            r[name + '_within1'] = float((np.abs(pred - y) <= 1).mean())
            r[name + '_recovered'] = float(ok[wrong].mean()) if wrong.any() else None
            r[name + '_by_count'] = [float(ok[y == n].mean()) for n in ns.NUMBERS]
        out[T.SHORT[kind]] = r
    return out


def task(t):
    arm, seed = t
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    t0, c0 = time.perf_counter(), time.process_time()
    res, ck = {}, {}
    for point in T.POINTS:
        enc, model, ck[point] = T.load(arm, seed, point)
        res[point] = reads(enc, model, seed)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return f'{arm}/seed{seed}', {'reads': res, 'checkpoints': ck}, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    tasks = [(a, s) for a in T.ARMS for s in T.SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(T.WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'identity from order on the shared output pilot checkpoints; no training of the system', 'arms': T.ARMS,
                                     'seeds': T.SEEDS, 'points': T.POINTS, 'kinds': T.KINDS, 'fit_kinds': FIT_KINDS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(T.CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': T.WORKERS, 'threads_per_worker': FI.THREADS, 'dtype': 'float32', **DEV.info(),
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'rss_convention': 'ru_maxrss in KiB; children is the largest single child',
        'tasks': {k: t for k, _, t in res}})


def summary(arm='common-k-cand'):
    import json
    res = json.loads(REPORT.with_suffix('.json').read_text())['results']
    keys = ('same', 'placement', 'placement_within1', 'placement_recovered', 'learned', 'learned_within1', 'learned_recovered')
    for point in T.POINTS:
        print(f'\n{arm}, {point}\n| kind | ' + ' | '.join(keys) + ' |')
        for kind in T.KINDS:
            sk = T.SHORT[kind]
            cells = []
            for key in keys:
                v = [res[f'{arm}/seed{s}']['reads'][point][sk][key] for s in T.SEEDS]
                v = [x for x in v if x is not None]
                cells.append(f'{np.mean(v):.2f} ({min(v):.2f}-{max(v):.2f})' if v else '-')
            print(f'| {sk} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary(*sys.argv[2:])
    else:
        main()
