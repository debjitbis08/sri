"""Shared-core sparse memory v1: sensory traces. Memory stores what was experienced rather than a host code; the particular is read in sensory space, and meaning at read time, by passing the trace through the host as it is then.

Stages (argv[2]):
  hosts  GPU: sparse48 through blocks 1-6 as the long-term memory host, keeping weights at the end of blocks 4, 5, 6; each block's state must equal the cached run's
  read   GPU: re-encode traces at each checkpoint; the count pairs, then a symbol-to-symbol fixture, which has no meaning to read

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_traces pilot hosts
"""
import json
import os
import sys
import time

os.environ.setdefault('NAP_OPTION', 'A')
os.environ.setdefault('NAP_ARMS', 'sparse48')
os.environ.setdefault('NAP_N', '45')
os.environ.setdefault('NAP_L', '2000')

import numpy as np
import torch

from experiments import arbitrary_pairings as AP
from experiments import durable as DUR
from experiments import limited_memory as LM
from experiments import shared_core_moving_values as MV
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import streaming_retrieval as SR
from src.sparse_local_memory import Memory

nc, X, P, D, SP, DEV, FI = S0.nc, S0.X, S0.P, S0.D, S0.SP, S0.DEV, S0.FI
NAME = 'shared-core-sparse-memory-v1-traces'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
H, K, TAU = MV.H, MV.K, MV.TAU
N, E = S1.N, S0.EVAL_RENDERS
EXEMPLARS = 8


def ck_path(seed, b):
    return RUNS / 'checkpoints' / f'seed{seed}-block{b}.pt'


def fit(task):
    seed, = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    withheld = D.withheld_counts(seed)
    sets, row, _ = AP.build_sets(seed)
    enc, model, opt = LM.setup('sparse48', seed)
    LM.CURRENT['end'] = None
    rng = nc.stream('partA_train', seed, 'A-order')
    for lang in P.LANGS:
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    hashes = {'3': S0.state_hash(enc, model)}
    for b, taught in {4: sets['A'], 5: sets['B'], 6: None}.items():
        crng = nc.stream('ap_context', seed, b)
        prng = nc.stream('ap_pairs', seed, b, {4: 'A', 5: 'B', 6: 'none'}[b])
        for _ in range(AP.L):
            items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
            if taught is not None:
                AP.pair_items(items, match, taught, row, prng)
            LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True)
        hashes[str(b)] = S0.state_hash(enc, model)
        LM.save_atomic(LM.weights(enc, model, withheld, 'sparse48', seed), ck_path(seed, b))
    ref = json.load(open(MV.CACHE / f'sparse48-seed{seed}-full-N45.json'))['hashes']
    return f'seed{seed}', {'hashes': hashes, 'reproduces_cache': {b: hashes[b] == ref[b] for b in hashes}, 'updates': 3 * D.STEPS + 3 * AP.L}, S0.timing(t0, c0)


def hosts():
    t0 = time.perf_counter()
    S0.prepare()
    AP.POOL.update(S0.PS.survivors())
    res = S0.pool_map(DUR.durable(fit, RUNS / 'done-hosts', 'seed{0}'), [(s,) for s in S0.SEEDS])
    S0.save_json(f'{REPORT}-hosts.json', {k: r for k, r, _ in res})
    S0.save_json(f'{REPORT}-hosts-runtime.json', S0.runtime({'phase': 'host runs, 12,000 updates each', 'wall_seconds': time.perf_counter() - t0, 'fits': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))
    print({k: r['reproduces_cache'] for k, r, _ in res}, flush=True)


def load_host(seed, b):
    ck = torch.load(ck_path(seed, b), weights_only=False, map_location='cpu')
    enc, model, _ = LM.setup('sparse48', seed)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    for m in (*enc.values(), model):
        m.requires_grad_(False)
    return enc, model, lambda x: model.proj['image'](enc['image'](x))


def table(fn):
    """Codes of every training collection, (collections, 9, d), by whichever function."""
    return torch.stack([S0.batched(fn, P.DOTS['train'][:, k]) for k in range(9)], 1).cpu().numpy().astype(np.float32)


def unit_values(unit_of, rows, T, how):
    """Per unit, a stored trace's code under table T. rows (w, 2): collection and count index of each write, in order; unit_of (w,) its unit. Traces kept: the first, the first EXEMPLARS, or all."""
    limit = {'exemplar-1': 1, f'exemplar-{EXEMPLARS}': EXEMPLARS, 'all': len(rows)}[how]
    val, seen = np.zeros((H, T.shape[-1]), np.float32), np.zeros(H, np.int64)
    for w in range(len(rows)):
        i = unit_of[w]
        if seen[i] < limit:
            val[i] += T[rows[w, 0], rows[w, 1]]
            seen[i] += 1
    return val / np.maximum(seen, 1)[:, None]


def read_with(mem, val, q, cand, counts, truth, protos):
    keep, width = mem.u, mem.m
    mem.u, mem.m = val * mem.n[:, None], val.shape[1]
    content, ok = mem.read(S1.unit(q))
    mem.u, mem.m = keep, width
    return S1.decode(content, ok, truth, cand, counts - 1, protos)


def pairs_seed(seed, key):
    c = S1.load(seed)
    d = MV.CACHE / f'sparse48-seed{seed}-full-N45'
    ev = {b: dict(np.load(d / f'eval-{b}.npz')) for b in MV.BLOCKS}
    streams = {'A': S1.stream('A', seed, c), 'B': S1.stream('B', seed, c)}
    mems, writes, units, live = {}, [], [], []
    mem = Memory(H, K, TAU, 32, S1.CORE)
    for b, tag in ((4, 'A'), (5, 'B')):
        z = np.load(d / f'train-{b}.npz')
        s = streams[tag]
        for zq, v in zip(S1.unit(z['q_frozen']), z['t_live']):
            units.append(int(mem.write(zq, v)[0]))
        writes.append(np.stack([s[:, 2], c['counts'][s[:, 0]] - 1], 1))
        live.append(mem.u.copy())
        snap = Memory(H, K, TAU, 32, S1.CORE)
        snap.a, snap.n, snap.h, snap.u = mem.a.copy(), mem.n.copy(), mem.h, mem.u.copy()
        mems[b] = snap
    mems[6] = mems[5]
    rows = np.concatenate(writes)
    units = np.array(units)
    KT = table(key)
    res = {'gate': {'cached_targets_match': bool(all(np.array_equal(np.load(d / f'train-{b}.npz')['target'], writes[i][:, 1]) for i, b in enumerate((4, 5))))}, 'after': {}}
    P_key = KT.mean(0)
    for b in (4, 5, 6):
        enc, model, code = load_host(seed, b)
        T = table(code)
        protos = T.mean(0)
        n_w = len(streams['A']) if b == 4 else len(rows)
        mem_b = mems[b]
        hold = np.maximum(mem_b.n, 1)[:, None]
        row = {}
        vals = {f'reencode-{how}': unit_values(units[:n_w], rows[:n_w], T, how) for how in ('exemplar-1', f'exemplar-{EXEMPLARS}', 'all')}
        vals['stale-copy'] = mem_b.u / hold
        sensory = {f'sensory-{how}': unit_values(units[:n_w], rows[:n_w], KT, how) for how in ('exemplar-1', 'all')}
        for tag in ('A',) if b == 4 else ('A', 'B'):
            s = MV.eval_set(ev[b], tag)
            q, cand, counts, truth = s['q'], s['cand'], s['counts'], s['truth']
            out = {k: read_with(mem_b, v, q, cand, counts, truth, protos) for k, v in vals.items()}
            cf = ev[b][f'recall__{tag}___c_frozen']
            out.update({k: read_with(mem_b, v, q, cf, counts, truth, P_key) for k, v in sensory.items()})
            out['perfect_contents'] = S1.decode(protos[truth - 1], np.ones(len(truth), bool), truth, cand, counts - 1, protos)
            row[tag] = out
        res['after'][str(b)] = row
        res.setdefault('training_prototype_cosine_with_block3', {})[str(b)] = float((S1.unit(protos) * S1.unit(c['protos'])).sum(1).mean())
    res['units'], res['forced'] = mem.h, mem.forced
    return res


def symbols_seed(seed, key, code6):
    """Symbol to symbol, on the frozen encoder: arbitrary, no count involved. 45 cues to 45 target symbols in each of two disjoint sets, both orders; answers among nine rendered target symbols. Values: the first trace, or the mean of traces; the host's code of the same trace is read at block 6."""
    train, ev, _ = S0.test_material(seed)
    R = S0.TRAIN_RENDERS
    Kt = S0.batched(key, train.reshape(-1, 28, 28)).cpu().numpy().astype(np.float32).reshape(len(train), R, -1)
    Ke = S0.batched(key, ev.reshape(-1, 28, 28)).cpu().numpy().astype(np.float32).reshape(len(ev), E, -1)
    Ct = S0.batched(code6, train.reshape(-1, 28, 28)).cpu().numpy().astype(np.float32).reshape(len(train), R, -1)
    Ce = S0.batched(code6, ev.reshape(-1, 28, 28)).cpu().numpy().astype(np.float32).reshape(len(ev), E, -1)
    target_of = {'A': 90 + np.arange(N), 'B': 135 + np.arange(N)}
    cue_of = {'A': np.arange(N), 'B': N + np.arange(N)}
    rng = nc.stream('sc_traces_symbols', seed)
    draws = {t: rng.integers(0, [N, R, R], size=(S1.STEPS * S1.BATCH, 3)) for t in 'AB'}
    q_sets = {}
    for t in 'AB':
        wrong = np.stack([rng.permutation(np.delete(target_of[t], i))[:8] for i in range(N) for _ in range(E)])
        cand_rows = np.concatenate([np.repeat(target_of[t], E)[:, None], wrong], 1)
        rend = rng.integers(E, size=cand_rows.shape)
        q_sets[t] = (np.repeat(cue_of[t], E), np.tile(np.arange(E), N), cand_rows, rend)

    def read(mem, val, t, space):
        rows, r, cand_rows, rend = q_sets[t]
        keep, width = mem.u, mem.m
        mem.u, mem.m = val * mem.n[:, None], val.shape[1]
        content, ok = mem.read(S1.unit(Ke[rows, r]))
        mem.u, mem.m = keep, width
        pool = space[cand_rows, rend]
        return float((((S1.unit(content)[:, None] * S1.unit(pool)).sum(-1).argmax(1) == 0) & ok).sum() / len(rows)), float(ok.mean())

    out = {}
    for first, second in (('A', 'B'), ('B', 'A')):
        mem = Memory(H, K, TAU, 32, 32)
        first_key, first_code, seen = np.zeros((H, 32), np.float32), np.zeros((H, S1.CORE), np.float32), np.zeros(H, bool)
        mean_key = np.zeros((H, 32), np.float32)
        rec = {}
        for phase, t in enumerate((first, second)):
            d = draws[t]
            for i, r_c, r_t in d:
                cue, tgt = cue_of[t][i], target_of[t][i]
                v = Kt[tgt, r_t]
                I = int(mem.write(S1.unit(Kt[cue, r_c]), v)[0])
                mean_key[I] += v
                if not seen[I]:
                    first_key[I], first_code[I], seen[I] = v, Ct[tgt, r_t], True
            hold = np.maximum(mem.n, 1)[:, None]
            if phase == 0:
                rec['old_after_old'] = {'first_trace': read(mem, first_key, first, Ke)[0], 'mean_of_traces': read(mem, mean_key / hold, first, Ke)[0]}
        hold = np.maximum(mem.n, 1)[:, None]
        rec['old_after_both'] = {'first_trace': read(mem, first_key, first, Ke)[0], 'mean_of_traces': read(mem, mean_key / hold, first, Ke)[0], 'host_code_first_trace_block6': read(mem, first_code, first, Ce)[0]}
        rec['new_after_both'] = {'first_trace': read(mem, first_key, second, Ke)[0], 'mean_of_traces': read(mem, mean_key / hold, second, Ke)[0], 'host_code_first_trace_block6': read(mem, first_code, second, Ce)[0]}
        rec['units'], rec['forced'] = mem.h, mem.forced
        out[f'{first}{second}'] = rec
    return out


def read():
    t0, c0 = time.perf_counter(), time.process_time()
    S0.prepare()
    S0.set_up_task('sparse48')
    out = {}
    with torch.no_grad():
        for seed in S0.SEEDS:
            key_net = SR.key_encoder(seed).to(DEV.NAME if DEV.ACTIVE else 'cpu').requires_grad_(False)
            key = lambda x: key_net(x)
            pairs = pairs_seed(seed, key)
            _, _, code6 = load_host(seed, 6)
            out[str(seed)] = {'pairs': pairs, 'symbols': symbols_seed(seed, key, code6)}
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds; A then B for count pairs, both orders for symbols; memory setting and keys as the stage 1 pilot sense-only', 'setting': {'H': H, 'k': K, 'tau': TAU}, 'exemplars': EXEMPLARS, 'results': out})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'reads at checkpoints', 'wall_seconds': time.perf_counter() - t0, 'parent_cpu_seconds': time.process_time() - c0}))


if __name__ == '__main__':
    {'hosts': hosts, 'read': read}[sys.argv[2] if len(sys.argv) > 2 else 'hosts']()
