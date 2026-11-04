"""Why does relearning lower the core's collections -> taps reading? Diagnostics on existing checkpoints only (no training).

The core is step 8's calibrated core (`joined-mlp-calibrated`, identical to stage 2's `no-replay`), seeds 300-303, at block 3, the delay and relearning.
Every checkpoint is read under identical conditions: the same query collections, the same tap candidates in the same order, and the same stored dot values.

Reads:
  accuracy     collections -> taps (query a collection, nine taps candidates) for fresh evaluation collections and for memory's stored dot values
               (stage 2's units, labelled by their majority count); taps -> collections, the reverse, separately
  order        the same pairs scored with the comparator's arguments swapped: the core's `same` relation is a learned comparator g(a, b) whose
               training always puts the collection second
  geometry     within each checkpoint (never across): count separability of collection and taps codes by nearest class mean, and cross-sense
               alignment of class means by Euclidean distance (order-free)
  errors       which counts fail, signed error, and how often each candidate count wins (hubs)
  drift        relative change of each parameter group and of the codes of the same inputs, delay -> relearn
  hybrids      the delay checkpoint with one parameter group taken from relearning, and the reverse, read only after the codes' drift is reported
Usage: SMR_WORKERS=4 /data/venvs/sri-cuda/bin/python -m experiments.collections_taps_diagnosis
"""
import os
import pickle
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import sense_memory_replay as SMR
sys.argv = _argv
import numpy as np
import torch

SMC, SC, S0, P, D, nc = SMR.SMC, SMR.SC, SMR.S0, SMR.P, SMR.D, SMR.nc
SEEDS = (300, 301, 302, 303)
POINTS = ('block3', 'delay', 'relearn')
ENC = 'joined-mlp-calibrated'
REPORT = SC.X.ROOT / 'reports' / 'collections-taps-diagnosis-v1'
GROUPS = {'image projection': ('image', 'head.'), 'taps encoder': ('sequence', ''), 'core projection': ('sri', 'proj.'), 'relation tokens': ('sri', 'token.'),
          'comparator g': ('sri', 'g.'), 'predictor P': ('sri', 'P.'), 'scale': ('sri', 'scale')}


def load(seed, point):
    return torch.load(SC.RUNS / ENC / 'checkpoints' / f'sparse48-seed{seed}-{point}.pt', weights_only=False, map_location='cpu')


def apply(enc, model, ck):
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); model.load_state_dict(ck['sri'])
    for m in (enc['image'], enc['sequence'], model):
        m.eval()


def hybrid(base, donor, group):
    part, prefix = GROUPS[group]
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for k, t in donor[part].items():
        if k.startswith(prefix) and not k.startswith('_'):
            out[part][k] = t
    return out


def fixed_draws(seed, n_values):
    rng = nc.stream('ctd_draws', seed)
    E, T = P.EVAL_PER, len(D.TAPS[('eval', 1)])
    n = 9 * E
    return {'order': np.stack([rng.permutation(9) for _ in range(n + n_values)]), 'taps_j': rng.integers(T, size=(n + n_values, 9)),
            'rev_order': np.stack([rng.permutation(9) for _ in range(9 * T)]), 'coll_j': rng.integers(E, size=(9 * T, 9))}


@torch.no_grad()
def read(enc, model, values, vcounts, dr):
    """All reads at one checkpoint. Collections: the 30 evaluation collections per count; values: memory's stored dot values."""
    code = lambda x, m: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), m)
    colls = {n: code(P.DOTS['succ'][n], 'image') for n in range(1, 10)}
    taps = {n: code(D.TAPS[('eval', n)], 'sequence') for n in range(1, 10)}
    sv = D.B.Solver(model, 'image')
    vals = model.split(sv.solve(enc['image'].from_codes(values)))['core_inner']
    E, T = P.EVAL_PER, len(D.TAPS[('eval', 1)])
    qc = torch.cat([colls[n] for n in range(1, 10)] + [vals])
    counts = np.concatenate([np.repeat(np.arange(1, 10), E), vcounts])
    nq = len(qc)
    order, tj = dr['order'][:nq], dr['taps_j'][:nq]
    cand = torch.stack([torch.stack([taps[int(k) + 1][tj[i, k]] for k in order[i]]) for i in range(nq)])
    d = qc.shape[1]
    q = qc[:, None].expand_as(cand).reshape(-1, d)
    c = cand.reshape(-1, d)
    s_fwd = P.same_logits(model, q, c).reshape(nq, 9).numpy()  # the evaluated order: g(collection, taps)
    s_rev = P.same_logits(model, c, q).reshape(nq, 9).numpy()  # the trained order: g(taps, collection)
    pick_f = order[np.arange(nq), s_fwd.argmax(1)] + 1
    pick_r = order[np.arange(nq), s_rev.argmax(1)] + 1
    # the reverse direction: each evaluation taps sequence against nine evaluation collections
    tq = torch.cat([taps[n] for n in range(1, 10)])
    tcounts = np.repeat(np.arange(1, 10), T)
    ro, cj = dr['rev_order'], dr['coll_j']
    cc = torch.stack([torch.stack([colls[int(k) + 1][cj[i, k]] for k in ro[i]]) for i in range(len(tq))])
    s_t2c = P.same_logits(model, tq[:, None].expand_as(cc).reshape(-1, d), cc.reshape(-1, d)).reshape(len(tq), 9).numpy()
    pick_t2c = ro[np.arange(len(tq)), s_t2c.argmax(1)] + 1
    # geometry within this checkpoint
    C = torch.stack([colls[n].mean(0) for n in range(1, 10)]); Tm = torch.stack([taps[n].mean(0) for n in range(1, 10)])
    sep = lambda X, M: float(np.mean([(torch.cdist(X[n], M).argmin(1) == n - 1).float().mean() for n in range(1, 10)]))
    align = (torch.cdist(C, Tm).argmin(1) == torch.arange(9)).float().mean().item()
    align_rev = (torch.cdist(Tm, C).argmin(1) == torch.arange(9)).float().mean().item()
    nE = 9 * E
    f, v = slice(0, nE), slice(nE, nq)
    win = np.bincount(pick_f[f], minlength=10)[1:] / nE
    out = {'collections_to_taps': {'fresh': float((pick_f[f] == counts[f]).mean()), 'stored': float((pick_f[v] == counts[v]).mean()),
                                   'fresh_by_count': [float((pick_f[f][counts[f] == n] == n).mean()) for n in range(1, 10)],
                                   'fresh_signed_error': float((pick_f[f] - counts[f]).mean()), 'fresh_within_one': float((np.abs(pick_f[f] - counts[f]) <= 1).mean()),
                                   'winning_count_share': win.tolist()},
           'collections_to_taps_swapped_order': {'fresh': float((pick_r[f] == counts[f]).mean()), 'stored': float((pick_r[v] == counts[v]).mean())},
           'taps_to_collections': float((pick_t2c == tcounts).mean()),
           'geometry': {'collections_separable': sep(colls_list(colls), C), 'taps_separable': sep(colls_list(taps), Tm),
                        'collection_means_nearest_taps_mean': align, 'taps_means_nearest_collection_mean': align_rev,
                        'same_count_distance': float(torch.diagonal(torch.cdist(C, Tm)).mean()), 'other_count_distance': float(torch.cdist(C, Tm)[~torch.eye(9, dtype=bool)].mean())},
           'margin_fresh': float(np.mean([s_fwd[i, np.flatnonzero(order[i] + 1 == counts[i])[0]] - np.max(np.delete(s_fwd[i], np.flatnonzero(order[i] + 1 == counts[i])[0])) for i in range(nE)]))}
    return out, {'colls': torch.cat([colls[n] for n in range(1, 10)]), 'taps': tq}


def colls_list(dct):
    return {n: dct[n] for n in dct}


def task(seed):
    t0 = time.perf_counter()
    torch.set_num_threads(SMC.FI.THREADS)
    prep = pickle.loads((SMR.RUNS / f'prep-seed{seed}.pkl').read_bytes())  # stage 2's memory: the stored dot values memory holds
    mem = SMR.LiveMemory(prep)
    mem.open_run(0); mem.write_upto('A', len(mem.order['A'])); mem.open_run(1); mem.write_upto('B', len(mem.order['B']))
    keys = mem.units()
    values = np.stack([mem.pair(k)[1] for k in keys]).astype(np.float32)
    vcounts = np.array([mem.describe(k)[1] for k in keys])
    dr = fixed_draws(seed, len(values))
    enc, model = SC.make_model(seed, 'sparse48', ENC)
    cks = {p: load(seed, p) for p in POINTS}
    out, codes = {}, {}
    for p in POINTS:
        apply(enc, model, cks[p])
        out[p], codes[p] = read(enc, model, values, vcounts, dr)
    # drift, delay -> relearn: parameters by group, and codes of the same inputs
    drift = {}
    for g, (part, prefix) in GROUPS.items():
        a = [t.float() for k, t in cks['delay'][part].items() if k.startswith(prefix) and torch.is_tensor(t) and t.is_floating_point()]
        b = [t.float() for k, t in cks['relearn'][part].items() if k.startswith(prefix) and torch.is_tensor(t) and t.is_floating_point()]
        na = torch.sqrt(sum((x ** 2).sum() for x in a)); nd = torch.sqrt(sum(((y - x) ** 2).sum() for x, y in zip(a, b)))
        drift[g] = float(nd / na)
    rel = lambda x, y: float(((y - x).norm(dim=1) / x.norm(dim=1)).mean())
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y).mean())
    drift['codes'] = {'collections': {'relative_change': rel(codes['delay']['colls'], codes['relearn']['colls']), 'cosine': cos(codes['delay']['colls'], codes['relearn']['colls'])},
                      'taps': {'relative_change': rel(codes['delay']['taps'], codes['relearn']['taps']), 'cosine': cos(codes['delay']['taps'], codes['relearn']['taps'])}}
    # hybrids: one group swapped, both directions
    hyb = {}
    for g in GROUPS:
        if g == 'scale':
            continue
        apply(enc, model, hybrid(cks['delay'], cks['relearn'], g))
        r, _ = read(enc, model, values, vcounts, dr)
        apply(enc, model, hybrid(cks['relearn'], cks['delay'], g))
        r2, _ = read(enc, model, values, vcounts, dr)
        hyb[g] = {'delay_with_relearn_group': {'collections_to_taps': r['collections_to_taps']['fresh'], 'stored': r['collections_to_taps']['stored'], 'taps_to_collections': r['taps_to_collections']},
                  'relearn_with_delay_group': {'collections_to_taps': r2['collections_to_taps']['fresh'], 'stored': r2['collections_to_taps']['stored'], 'taps_to_collections': r2['taps_to_collections']}}
    return seed, {'reads': out, 'drift': drift, 'hybrids': hyb, 'stored_values': int(len(values))}, time.perf_counter() - t0


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    S0.prepare()
    SC.LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('SMR_WORKERS', 4))) as pool:
        res = pool.map(task, SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'diagnostics on existing checkpoints; no training', 'encoder': ENC, 'seeds': SEEDS, 'points': POINTS,
                                    'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'diagnosis', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                                      'task_wall_seconds': {f'seed{s}': w for s, _, w in res}}))


if __name__ == '__main__':
    main()
