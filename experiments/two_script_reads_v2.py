"""Two-script follow-up reads (docs/two-script-reads-v1.md, "Follow-up reads"). No training; the v1 checkpoints, arms, seeds and items.

  comparison by distance   Part D's comparison read for every ordered pair of kinds, split by |a - b| = 1, 2, 3+.
  matching via collections for every ordered pair (A, B) of kinds: identify each item's count through `same` against M (the 72 participant
                           collections; a count's score is its best participant's logit), then
                             both_correct   the item of A and the gallery's item of the same count are both identified as that count;
                             mediated       pick the gallery item whose identified-count distribution (softmax over the 9 count scores) best
                                            matches the query's (dot product);
                             direct         v1's matching read (same(query, gallery)), recomputed on the same items;
                             agreement      the share of queries where direct and mediated pick the same gallery item;
                             direct_given_both / direct_given_not_both   direct accuracy split by both_correct.
                           Identification accuracy over M is reported per kind.

Usage: .venv/bin/python -m experiments.two_script_reads_v2   (TS_WORKERS, default 4; rerun to resume)
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
NAME = 'two-script-reads-v2'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
BANDS = (('1', lambda d: d == 1), ('2', lambda d: d == 2), ('3+', lambda d: d >= 3))


@torch.no_grad()
def reads(enc, model):
    img = lambda x: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), 'image')
    items = {'collection': {n: img(P.DOTS['succ'][n][:T.E]) for n in ns.NUMBERS},
             'taps': {n: D.codes_for(enc, model, D.TAPS[('eval', n)][:T.E], 'sequence') for n in ns.NUMBERS}}
    for lang in T.SCRIPTS:
        items[lang] = {n: img(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]][:T.E]) for n in ns.NUMBERS}
    k = min(len(v[n]) for v in items.values() for n in ns.NUMBERS)
    items = {kind: {n: v[n][:k] for n in ns.NUMBERS} for kind, v in items.items()}
    M = model.participants()
    inv_n = np.array([n for n, _ in SOX.W['inventory']])
    cols = [np.flatnonzero(inv_n == n) for n in ns.NUMBERS]

    def count_scores(x):
        """(len(x), 9): each count's best participant logit."""
        lg = model.identify(x, M).numpy()
        return np.stack([lg[:, c].max(1) for c in cols], 1)

    cs = {kind: {n: count_scores(items[kind][n]) for n in ns.NUMBERS} for kind in T.KINDS}
    ident = {kind: {n: cs[kind][n].argmax(1) + 1 for n in ns.NUMBERS} for kind in T.KINDS}
    prob = {kind: {n: torch.from_numpy(cs[kind][n]).softmax(1).numpy() for n in ns.NUMBERS} for kind in T.KINDS}
    out = {'identification': {T.SHORT[kind]: float(np.mean([(ident[kind][n] == n).mean() for n in ns.NUMBERS])) for kind in T.KINDS}, 'pairs': {}}
    for a in T.KINDS:
        for b in T.KINDS:
            # comparison by distance
            hits = {name: [] for name, _ in BANDS}
            for x in ns.NUMBERS:
                for y in ns.NUMBERS:
                    if x == y:
                        continue
                    h = (model.relation_logits(items[a][x], items[b][y], 2) > 0).numpy() == (x > y)
                    for name, sel in BANDS:
                        if sel(abs(x - y)):
                            hits[name].append(h)
            comp = {name: float(np.concatenate(h).mean()) for name, h in hits.items()}
            # matching via collections
            gal = torch.stack([items[b][m] for m in ns.NUMBERS], 1)  # (k, 9, d), item index aligned
            gal_p = np.stack([prob[b][m] for m in ns.NUMBERS], 1)  # (k, 9 gallery items, 9 counts)
            gal_id = np.stack([ident[b][m] for m in ns.NUMBERS], 1)  # (k, 9)
            r = {key: [] for key in ('direct', 'mediated', 'both_correct', 'agreement')}
            given = {True: [], False: []}
            for n in ns.NUMBERS:
                direct = SOX.D.pick(model, items[a][n], gal)
                mediated = np.einsum('ic,ijc->ij', prob[a][n], gal_p).argmax(1)
                both = (ident[a][n] == n) & (gal_id[:, n - 1] == n)
                r['direct'].append(direct == n - 1); r['mediated'].append(mediated == n - 1)
                r['both_correct'].append(both); r['agreement'].append(direct == mediated)
                for flag in (True, False):
                    given[flag].append((direct == n - 1)[both == flag])
            match = {key: float(np.concatenate(v).mean()) for key, v in r.items()}
            for flag, name in ((True, 'direct_given_both'), (False, 'direct_given_not_both')):
                v = np.concatenate(given[flag])
                match[name] = float(v.mean()) if len(v) else None
                match[name + '_n'] = int(len(v))
            out['pairs'][f'{T.SHORT[a]}>{T.SHORT[b]}'] = {'comparison_by_distance': comp, 'matching_via_collections': match}
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
    tasks = [(a, s) for a in T.ARMS for s in T.SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(T.WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'follow-up reads on the shared output pilot checkpoints; no training', 'arms': T.ARMS, 'seeds': T.SEEDS,
                                     'points': T.POINTS, 'kinds': T.KINDS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(T.CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': T.WORKERS, 'threads_per_worker': FI.THREADS, 'dtype': 'float32', **DEV.info(),
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'rss_convention': 'ru_maxrss in KiB; children is the largest single child',
        'tasks': {k: t for k, _, t in res}})


if __name__ == '__main__':
    main()
