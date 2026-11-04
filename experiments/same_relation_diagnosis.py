"""Is the core's learned `same` relation symmetric and discriminative across senses, before and after relearning? (docs/same-relation-diagnosis-v1.md)

Existing checkpoints only, no training: step 8's calibrated core (identical to stage 2's no-replay), seeds 300-303, at block 3, the delay and relearning.
`same` is a learned comparator g(a, b) = g(cat(a, b, token)). It is trained only through a 9-way softmax with the query in the first slot (a form or a
taps sequence) and nine collections in the second, so the loss constrains only score differences across collections for a fixed query.

All reads use identical pairs at every checkpoint: the 270 evaluation collections C, the 270 evaluation taps sequences T and the English evaluation
forms F (30 per count), every pair scored in both orders.
  symmetry        g(c, t) against g(t, c) on the same pairs: correlation, mean absolute difference in units of the scores' spread, and how the
                  violation grows with the distance between the two codes
  discrimination  pair level, threshold-free: AUROC of same-count against different-count pairs, per order; the trained pairs g(f, c) and
                  g(t, c) as references; and the order-free code distance
  ranking         nine candidates, one per count (fixed draws): query c ranking taps by g(c, t) (the evaluated read) or by g(t, c); query t
                  ranking collections by g(t, c) (trained) or by g(c, t). Each also with every candidate's mean score over all queries removed,
                  which cancels any per-candidate offset (hubs) and leaves pair-level discrimination
  geometry        within each checkpoint: class means of C, T, F; same-count distances between senses in units of the mean distance between
                  collection classes
Usage: SMR_WORKERS=4 /data/venvs/sri-cuda/bin/python -m experiments.same_relation_diagnosis
"""
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import collections_taps_diagnosis as CTD
sys.argv = _argv
import numpy as np
import torch

SC, S0, P, D, X, nc = CTD.SC, CTD.S0, CTD.P, CTD.D, CTD.SC.X, CTD.nc
SEEDS, POINTS = CTD.SEEDS, CTD.POINTS
REPORT = X.ROOT / 'reports' / 'same-relation-diagnosis-v1'
DRAWS = 20  # candidate sets per query


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = np.empty(len(s)); r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def scores(model, A, B):
    """(len(A), len(B)) matrix of g(a, b) for the `same` head."""
    out = []
    for i in range(0, len(A), 30):
        a = A[i:i + 30]
        out.append(P.same_logits(model, a[:, None].expand(-1, len(B), -1).reshape(-1, A.shape[1]), B[None].expand(len(a), -1, -1).reshape(-1, A.shape[1])).reshape(len(a), len(B)))
    return torch.cat(out).numpy()


def rank(S, qy, cy, draws):
    """Nine-candidate accuracy: S (queries, candidates) scores, qy and cy their counts; draws (queries, DRAWS, 9) candidate indices, one per count."""
    hit = []
    for i in range(len(S)):
        sc = S[i][draws[i]]  # (DRAWS, 9), column k is count k + 1
        hit.append((sc.argmax(1) + 1 == qy[i]).mean())
    return float(np.mean(hit))


def task(seed):
    t0 = time.perf_counter()
    torch.set_num_threads(SC.LM.FI.THREADS)
    enc, model = SC.make_model(seed, 'sparse48', CTD.ENC)
    E = P.EVAL_PER
    y = np.repeat(np.arange(1, 10), E)
    rng = nc.stream('srd_draws', seed)
    by_count = [np.flatnonzero(y == n) for n in range(1, 10)]
    draws = np.stack([np.stack([[rng.choice(b) for b in by_count] for _ in range(DRAWS)]) for _ in range(len(y))])  # the same for every read and checkpoint
    out = {}
    for p in POINTS:
        CTD.apply(enc, model, CTD.load(seed, p))
        with torch.no_grad():
            code = lambda x, m: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), m)
            C = torch.cat([code(P.DOTS['succ'][n], 'image') for n in range(1, 10)])
            T = torch.cat([code(D.TAPS[('eval', n)], 'sequence') for n in range(1, 10)])
            F = torch.cat([code(X.DATA[('english', 'eval')][0][P.EVAL[('english', n, 'idx')]], 'image') for n in range(1, 10)])
        ct, tc = scores(model, C, T), scores(model, T, C).T  # both (collections, taps), the same pairs
        fc, tt, cc = scores(model, F, C), scores(model, T, T), scores(model, C, C)
        same = y[:, None] == y[None]
        dist = torch.cdist(C, T).numpy()
        viol = np.abs(ct - tc)
        spread = np.std(np.concatenate([ct.ravel(), tc.ravel()]))
        near = dist <= np.median(dist)
        r = {
            'symmetry': {'correlation': float(np.corrcoef(ct.ravel(), tc.ravel())[0, 1]), 'mean_abs_difference_over_spread': float(viol.mean() / spread),
                         'same_count_pairs': float(viol[same].mean() / spread), 'different_count_pairs': float(viol[~same].mean() / spread),
                         'nearer_half_by_code_distance': float(viol[near].mean() / spread), 'farther_half': float(viol[~near].mean() / spread),
                         'violation_distance_correlation': float(np.corrcoef(viol.ravel(), dist.ravel())[0, 1])},
            'discrimination_auroc': {'g(c,t)': auroc(ct[same], ct[~same]), 'g(t,c)': auroc(tc[same], tc[~same]),
                                     'g(f,c) trained': auroc(fc[same], fc[~same]), 'g(t,t)': auroc(tt[same], tt[~same]), 'g(c,c)': auroc(cc[same], cc[~same]),
                                     '-distance(c,t)': auroc(-dist[same], -dist[~same])},
            'ranking': {}}
        for name, S, Q in (('c -> t by g(c,t) (evaluated)', ct, 'c'), ('c -> t by g(t,c)', tc, 'c'), ('t -> c by g(t,c) (trained)', tc.T, 't'), ('t -> c by g(c,t)', ct.T, 't')):
            raw = rank(S, y, y, draws)
            cen = rank(S - S.mean(0, keepdims=True), y, y, draws)  # each candidate's mean over all queries removed
            r['ranking'][name] = {'raw': raw, 'candidate_centred': cen}
        with torch.no_grad():
            Cm, Tm, Fm = [torch.stack([X_[y == n].mean(0) for n in range(1, 10)]) for X_ in (C, T, F)]
            unit = float(torch.cdist(Cm, Cm)[~torch.eye(9, dtype=bool)].mean())
            r['geometry'] = {'collection_class_spacing': unit,
                             'same_count_c_t': float(torch.diagonal(torch.cdist(Cm, Tm)).mean() / unit),
                             'same_count_c_f': float(torch.diagonal(torch.cdist(Cm, Fm)).mean() / unit),
                             'same_count_t_f': float(torch.diagonal(torch.cdist(Tm, Fm)).mean() / unit),
                             'taps_class_spacing': float(torch.cdist(Tm, Tm)[~torch.eye(9, dtype=bool)].mean() / unit),
                             'english_forms_to_collections_nearest': float((torch.cdist(Fm, Cm).argmin(1) == torch.arange(9)).float().mean())}
        out[p] = r
    return seed, out, time.perf_counter() - t0


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    S0.prepare()
    SC.LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('SMR_WORKERS', 4))) as pool:
        res = pool.map(task, SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'diagnostics on existing checkpoints; no training', 'seeds': SEEDS, 'points': POINTS, 'draws_per_query': DRAWS,
                                    'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'same relation diagnosis', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                                      'task_wall_seconds': {f'seed{s}': w for s, _, w in res}}))


if __name__ == '__main__':
    main()
