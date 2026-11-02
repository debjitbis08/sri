"""Shared-core sparse memory v1: does A survive B while the host keeps learning? Replays the long-term memory host's cached stream (block 4 teaches A, block 5 teaches B, block 6 has no pairs)
through local memory on frozen-encoder keys. Addressing is identical in every condition, so any difference comes from values and reading. CPU, seconds per seed.

Usage: NSY_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_moving_values pilot
"""
import json
import time

import numpy as np

from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from src.sparse_local_memory import Memory

nc, X = S0.nc, S0.X
NAME = 'shared-core-sparse-memory-v1-moving-values'
REPORT = X.ROOT / 'reports' / NAME
CACHE = S0.REFERENCE
H, K, TAU = 128, 1, .85
BLOCKS = (3, 4, 5, 6)
WINDOW = 4000


def prototypes(cand, counts):
    """Per count, the mean code of the checkpoint's own evaluation candidates. cand (q, 9, d), counts (q, 9) in 1..9."""
    return np.stack([cand[counts == m].mean(0) for m in range(1, 10)])


def eval_set(z, tag):
    g = lambda f: z[f'recall__{tag}___{f}']
    return {'q': g('q_frozen'), 'cand': g('c_live'), 'counts': g('counts').astype(int), 'truth': g('truth').astype(int), 'logits': g('logits')}


def read(mem, keys, cand, counts, truth, train_protos):
    """Memory level: nearest prototype of training collections at this checkpoint, or None if the cache has none. Evaluation collections are rendered differently, so their prototypes are a separate reading."""
    content, ok = mem.read(S1.unit(keys))
    out = S1.decode(content, ok, truth, cand, counts - 1, prototypes(cand, counts))
    out['prototype_evaluation_collections'] = out.pop('prototype')
    out['prototype'] = S1.decode(content, ok, truth, cand, counts - 1, train_protos)['prototype'] if train_protos is not None else None
    return out


def reference_read(z):
    """Each checkpoint on its own: the right count's prototype as content, the readers' cached answers, and single collections against the prototypes."""
    out = {}
    for tag in 'AB':
        s = eval_set(z, tag)
        P = prototypes(s['cand'], s['counts'])
        out[tag] = {'perfect_contents': S1.decode(P[s['truth'] - 1], np.ones(len(s['truth']), bool), s['truth'], s['cand'], s['counts'] - 1, P),
                    'readers_alone': float((s['counts'][np.arange(len(s['truth'])), s['logits'].argmax(1)] == s['truth']).mean())}
    s = eval_set(z, 'A')
    flat, c = s['cand'].reshape(-1, s['cand'].shape[-1]), s['counts'].ravel()
    P = prototypes(s['cand'], s['counts'])
    near = (S1.unit(flat) @ S1.unit(P).T).argmax(1) + 1
    out['single_collection_to_count'] = {'accuracy': float((near == c).mean()), 'within_one': float((np.abs(near - c) <= 1).mean())}
    return out


def readings(moving, frozen, ev, b, tags, tp):
    """Moving or frozen stored values, read with the checkpoint's codes or with block 3's. Candidates and queries are the same images at every checkpoint."""
    out = {}
    for name in tags:
        now, old = eval_set(ev[b], name), eval_set(ev[3], name)
        at = lambda mem, s, protos: read(mem, s['q'], s['cand'], s['counts'], s['truth'], protos)
        out[name] = {'moving_values_current_reading': at(moving, now, tp[b]), 'moving_values_block3_reading': at(moving, old, tp[3]), 'frozen_values_block3_reading': at(frozen, old, tp[3]), 'frozen_values_current_reading': at(frozen, now, tp[b])}
    return out


def other_source(d, ev, src, width):
    """Values and reading both from a steadier code in the cache: 'slow' is the host's moving average, half-life 2,000 updates; 'frozen' is the frozen encoder, which never moves."""
    mem = Memory(H, K, TAU, 32, width)
    tp, out = {}, {}
    for b, tag in ((4, 'A'), (5, 'B')):
        z = np.load(d / f'train-{b}.npz')
        t, target = z[f't_{src}'], z['target'].astype(int)
        for zq, v in zip(S1.unit(z['q_frozen']), t):
            mem.write(zq, v)
        tp[b] = np.stack([t[-WINDOW:][target[-WINDOW:] == m].mean(0) for m in range(9)])
    tp[6] = tp[5] if src == 'frozen' else None
    mem4 = Memory(H, K, TAU, 32, width)
    z = np.load(d / 'train-4.npz')
    for zq, v in zip(S1.unit(z['q_frozen']), z[f't_{src}']):
        mem4.write(zq, v)

    def row(m, b, name):
        e = eval_set(ev[b], name)
        return read(m, e['q'], ev[b][f'recall__{name}___c_{src}'], e['counts'], e['truth'], tp[b])
    return {'4': {'A': row(mem4, 4, 'A')}, '5': {n: row(mem, 5, n) for n in 'AB'}, '6': {n: row(mem, 6, n) for n in 'AB'}}


def one_seed(seed):
    d = CACHE / f'sparse48-seed{seed}-full-N45'
    info = json.load(open(d.parent / f'sparse48-seed{seed}-full-N45.json'))
    assert info['hashes']['3'] == json.load(open(S0.REPORT.parent / 'shared-core-sparse-memory-v1-stage0-gate.json'))['seeds'][str(seed)]['hash'], 'cache is not the stage 0 sparse48 host'
    c = S1.load(seed)
    ev = {b: dict(np.load(d / f'eval-{b}.npz')) for b in BLOCKS}
    moving, frozen = Memory(H, K, TAU, 32, S1.CORE), Memory(H, K, TAU, 32, S1.CORE)
    gate = {}
    res = {'after': {}}
    drift = {}
    streams = {'A': S1.stream('A', seed, c), 'B': S1.stream('B', seed, c)}
    tp = {3: c['protos'], 6: None}  # block 6 has no pair items in the cache, so no training collections
    for b, tag in ((4, 'A'), (5, 'B')):
        z = np.load(d / f'train-{b}.npz')
        s = streams[tag]
        q, t, target = z['q_frozen'], z['t_live'], z['target'].astype(int)
        assert len(q) == len(s) == len(t), (len(q), len(s), len(t))
        count = c['counts'][s[:, 0]]
        gate[f'block{b}_targets_match_stream'] = bool(np.array_equal(target + 1, count))
        gate[f'block{b}_max_key_difference_from_stage1'] = float(np.abs(q - c['key_train'][s[:, 0] * S0.TRAIN_RENDERS + s[:, 1]]).max())
        frozen_values = c['coll_train'][s[:, 2], count - 1]
        for zq, v, fv in zip(S1.unit(q), t, frozen_values):
            moving.write(zq, v)
            frozen.write(zq, fv)
        drift[f'block{b}_written_value_cosine_with_block3'] = float((S1.unit(t) * S1.unit(frozen_values)).sum(1).mean())
        res[f'units_after_{b}'], res[f'forced_after_{b}'] = moving.h, moving.forced
        tp[b] = np.stack([t[-WINDOW:][target[-WINDOW:] == m].mean(0) for m in range(9)])  # the block's last writes, so slightly behind its end
        res['after'][str(b)] = readings(moving, frozen, ev, b, 'A' if b == 4 else 'AB', tp)
        if b == 4:  # the frozen reference through stage 1's own reading, for the reproduction gate
            gate['frozen_reference_A_after_A_prototype'] = S1.read_sets(frozen, S1.unit(c['key_eval']), c, ('A',))['A']['prototype']
    res['after']['6'] = readings(moving, frozen, ev, 6, 'AB', tp)  # no writes in block 6
    gate['frozen_reference_A_after_both_prototype'] = S1.read_sets(frozen, S1.unit(c['key_eval']), c, ('A',))['A']['prototype']
    grid = json.load(open(S1.REPORT.parent / f'{S1.NAME}.json'))['grid'][f'{seed}/sense-only/{H}/{K}/{TAU}']['AB']
    gate['stage1_A_after_A_prototype'] = grid['old_after_old']['prototype']
    gate['stage1_A_after_both_prototype'] = grid['old_after_both']['prototype']
    res['frozen_addresses_equal_moving'] = bool(np.array_equal(moving.a, frozen.a) and np.array_equal(moving.n, frozen.n))
    res['unknown_coverage'] = float(moving.read(S1.unit(eval_set(ev[6], 'unknown')['q']))[1].mean())
    res['slow_values_current_reading'] = other_source(d, ev, 'slow', S1.CORE)
    res['frozen_encoder_values'] = other_source(d, ev, 'frozen', 32)
    res['checkpoint'] = {str(b): reference_read(ev[b]) for b in BLOCKS}
    pr = {b: prototypes(eval_set(ev[b], 'A')['cand'], eval_set(ev[b], 'A')['counts']) for b in BLOCKS}
    drift['count_prototype_cosine_with_block3'] = {str(b): float((S1.unit(pr[b]) * S1.unit(pr[3])).sum(1).mean()) for b in BLOCKS}
    res['drift'] = drift
    res['gate'] = gate
    return res


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    out = {str(s): one_seed(s) for s in S0.SEEDS}
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, A then B only; the long-term host keeps learning through blocks 4 to 6; memory, keys and setting as the stage 1 pilot sense-only', 'setting': {'H': H, 'k': K, 'tau': TAU}, 'results': out})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'replay on cached codes', 'device': 'cpu', 'workers': 1, 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}))


if __name__ == '__main__':
    main()
