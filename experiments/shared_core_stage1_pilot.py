"""Shared-core sparse memory v1: the small stage 1 pilot on the locked conjunctive route, without trainable comparators, probes or the stress condition. Memory learns on cached codes.

Stages (argv[2]):
  codes  GPU, one process: core, frozen-encoder and whitened codes of the pair and fixture material, from stage 0's sparse48 checkpoints
  run    CPU: the grid on each address in both orders, then the cross-sense fixture, then selection by the contract's rule

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_stage1_pilot pilot codes
       NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_stage1_pilot pilot run
"""
import io
import itertools
import json
import multiprocessing as mp
import os
import resource
import sys
import time

import numpy as np

from experiments import durable as DUR
from experiments import shared_core_stage0 as S0
from src.sparse_local_memory import Memory

nc, X, DEV, P, D = S0.nc, S0.X, S0.DEV, S0.P, S0.D
NAME = 'shared-core-sparse-memory-v1-stage1-pilot' + ('-smoke' if S0.SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
N = S0.LOADS[0] // 2
STEPS, BATCH = (20, 16) if S0.SMOKE else (2000, 16)
HS, KS, TAUS = (128, 256, 512), (1, 4), (.5, .7, .85)
ADDRESSES = ('conjunctive', 'sensory', 'sense-only')
FIXTURE_ADDRESSES = ('core', 'conjunctive', 'sensory', 'sense-only')
FIXTURE_RENDERS, FIXTURE_H = 64, 128
FIXTURE_LANG = 'devanagari_digits'
WORKERS = int(os.environ.get('NSY_WORKERS', 6))
CORE = 16
STORE_MARGIN, ACQUIRE_FLOOR, LOSS_BAR = .15, .50, .05


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def whiten_fit(x):
    """W_K: mean and whitening matrix of the top directions, closed form, no labels. x (n, w)."""
    mean = x.mean(0)
    val, vec = np.linalg.eigh(np.cov((x - mean).T.astype(np.float64)))
    val, vec = val[::-1], vec[:, ::-1]
    return mean, (vec / np.sqrt(np.maximum(val, 1e-12))).astype(np.float32)


# ---------------------------------------------------------------- codes (GPU)

def codes_path(seed):
    return RUNS / 'codes' / f'seed{seed}.npz'


def candidate_images(rng, truth_count):
    order = np.stack([rng.permutation(9) for _ in range(truth_count)])
    j = rng.integers(P.EVAL_PER, size=truth_count)
    return order, np.stack([P.DOTS['succ'][int(k) + 1][j[t]] for t in range(truth_count) for k in order[t]])


def derangement(rng):
    while True:
        p = rng.permutation(9)
        if not np.any(p == np.arange(9)):
            return p + 1


def make_codes(seed):
    import torch
    from experiments import shared_core_value as V
    from experiments import streaming_retrieval as SR
    with torch.no_grad():
        enc, model = V.host(seed)
        core = lambda x: model.proj['image'](enc['image'](x))
        tap = lambda x: model.proj['sequence'](enc['sequence'](x))
        key = SR.key_encoder(seed).to(DEV.NAME if DEV.ACTIVE else 'cpu').requires_grad_(False)
        fn = lambda f, x: S0.batched(f, x).cpu().numpy().astype(np.float32)
        train, ev, _ = S0.test_material(seed)
        n = 3 * N
        out = {}
        out['core_train'], out['core_eval'] = fn(core, train[:n].reshape(-1, 28, 28)), fn(core, ev[:n].reshape(-1, 28, 28))
        out['key_train'], out['key_eval'] = fn(key, train[:n].reshape(-1, 28, 28)), fn(key, ev[:n].reshape(-1, 28, 28))
        out['coll_train'] = np.stack([fn(core, P.DOTS['train'][:, k]) for k in range(9)], 1)
        bg = S0.background_images(seed)[:, 0]
        out['w_mean'], out['w_matrix'] = whiten_fit(fn(key, bg))
        sets = S0.PS.draw_sets(seed, N, S0.PS.survivors())
        out['counts'] = np.array([{**sets['A'], **sets['B']}[c] for c in [*sets['A'], *sets['B']]])
        out['n_A'] = np.array(len(sets['A']))
        for tag, off in (('A', 0), ('B', N)):
            order, imgs = V.candidates(N, S0.EVAL_RENDERS, seed, tag)
            out[f'cand_{tag}'], out[f'order_{tag}'] = fn(core, imgs).reshape(N * S0.EVAL_RENDERS, 9, -1), order
        # the fixture: the most recently taught script, a derangement of counts
        rng = nc.stream('sc1_fixture', seed)
        target = derangement(rng)
        xt, yt = X.DATA[(FIXTURE_LANG, 'train')]
        idx = np.concatenate([rng.permutation(np.flatnonzero(yt == m))[:FIXTURE_RENDERS if not S0.SMOKE else 8] for m in S0.ns.NUMBERS])
        idx = idx[rng.permutation(len(idx))]
        out['fx_target'], out['fx_write_count'] = target, yt[idx].astype(int)
        out['fx_core'], out['fx_key'] = fn(core, xt[idx]), fn(key, xt[idx])
        out['fx_collection'] = rng.integers(len(P.DOTS['train']), size=len(idx))
        forms = np.concatenate([X.DATA[(FIXTURE_LANG, 'eval')][0][P.EVAL[(FIXTURE_LANG, m, 'idx')]] for m in S0.ns.NUMBERS])
        out['fx_eval_core'], out['fx_eval_key'] = fn(core, forms), fn(key, forms)
        out['fx_taps_core'] = fn(tap, np.concatenate([D.TAPS[('eval', m)] for m in S0.ns.NUMBERS]))
        E = P.EVAL_PER
        out['fx_query_count'] = np.repeat(np.arange(1, 10), E)
        order, imgs = candidate_images(nc.stream('sc1_fixture_candidates', seed), 9 * E)
        out['fx_cand'], out['fx_order'] = fn(core, imgs).reshape(9 * E, 9, -1), order
        buf = io.BytesIO()
        np.savez(buf, **out)
        DUR.atomic_write(codes_path(seed), buf.getvalue())
    return {k: v.shape for k, v in out.items()}


# ---------------------------------------------------------------- run (CPU)

CODES = {}


def load(seed):
    if seed not in CODES:
        z = np.load(codes_path(seed))
        c = {k: z[k] for k in z.files}
        c['protos'] = c['coll_train'].mean(0)
        CODES[seed] = c
    return CODES[seed]


def cues(c, address, core, key):
    """Unit cue vectors for an address, from raw core and frozen-encoder codes. None marks a part the sense does not have."""
    sensory = unit((key - c['w_mean']) @ c['w_matrix']) if key is not None else None
    if address == 'core':
        return unit(core)
    if address == 'sense-only':
        return unit(key)
    if address == 'sensory':
        return sensory
    return np.concatenate([unit(core), sensory], -1) / np.sqrt(2)


def stream(tag, seed, c):
    """The arbitrary pairings sampler for one set, as (symbol row, rendering, collection)."""
    rng = nc.stream('ap_pairs', seed, {'A': 4, 'B': 5}[tag], tag)
    off = 0 if tag == 'A' else N
    out = np.empty((STEPS * BATCH, 3), np.int64)
    n_coll = len(c['coll_train'])
    for t in range(len(out)):
        out[t] = off + int(rng.integers(N)), int(rng.integers(S0.TRAIN_RENDERS)), int(rng.integers(n_coll))
    return out


def decode(content, ok, truth, cand, order, protos):
    """Accuracy with abstentions wrong. Memory level: nearest count prototype. Answer level: among nine rendered candidates."""
    q = len(truth)
    answered = lambda pick: float(((pick == truth) & ok).sum() / q)
    pos = lambda s: order[np.arange(q), s.argmax(1)] + 1
    return {'prototype': answered((unit(content) @ unit(protos).T).argmax(1) + 1), 'euclidean': answered(pos(-((content[:, None] - cand) ** 2).sum(-1))),
            'cosine': answered(pos((unit(content)[:, None] * unit(cand)).sum(-1))), 'coverage': float(ok.mean())}


def read_sets(mem, Zq, c, sets):
    out = {}
    for tag in sets:
        off = 0 if tag == 'A' else N
        E = S0.EVAL_RENDERS
        rows = np.repeat(np.arange(off, off + N), E)
        z = Zq[rows * E + np.tile(np.arange(E), N)]
        content, ok = mem.read(z)
        out[tag] = decode(content, ok, c['counts'][rows], c[f'cand_{tag}'], c[f'order_{tag}'], c['protos'])
    return out


def one_grid(task):
    seed, address, H, k, tau = task
    t0, c0 = time.perf_counter(), time.process_time()
    c = load(seed)
    R = S0.TRAIN_RENDERS
    Zt = cues(c, address, c['core_train'], c['key_train'])
    Zq = cues(c, address, c['core_eval'], c['key_eval'])
    res = {}
    write_s, writes, peak = 0., 0, 0
    for first, second in (('A', 'B'), ('B', 'A')):
        mem = Memory(H, k, tau, Zt.shape[1], CORE)
        targets = np.zeros((H, 9), np.int64)
        rec = {}
        for phase, tag in enumerate((first, second)):
            s = stream(tag, seed, c)
            w0 = time.process_time()
            for row, ren, col in s:
                count = c['counts'][row]
                I = mem.write(Zt[row * R + ren], c['coll_train'][col, count - 1])
                targets[I, count - 1] += 1
            write_s += time.process_time() - w0
            writes += len(s)
            rec[f'units_after_{phase + 1}'], rec[f'forced_after_{phase + 1}'] = mem.h, mem.forced
            if phase == 0:
                rec['old_after_old'] = read_sets(mem, Zq, c, (first,))[first]
        both = read_sets(mem, Zq, c, (first, second))
        rec['old_after_both'], rec['new_after_both'] = both[first], both[second]
        E = S0.EVAL_RENDERS
        rec['unknown_coverage'] = float(mem.read(Zq[2 * N * E:3 * N * E])[1].mean())
        rec['bytes_used'], rec['bytes_limit'] = mem.bytes_used(), mem.bytes_limit()
        rec['conflicting_targets_per_unit'] = float((targets[:mem.h] > 0).sum(1).mean())
        res[f'{first}{second}'] = rec
    return f'{seed}/{address}/{H}/{k}/{tau}', res, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'write_cpu_seconds': write_s, 'writes': writes}


def explicit_store(seed, address):
    """One normalized cue prototype and contents per symbol, grouped by the symbol: an oracle privilege, a diagnostic."""
    c = load(seed)
    R, E = S0.TRAIN_RENDERS, S0.EVAL_RENDERS
    Zt = cues(c, address, c['core_train'], c['key_train'])
    Zq = cues(c, address, c['core_eval'], c['key_eval'])
    S = 2 * N
    cue, val, cnt = np.zeros((S, Zt.shape[1]), np.float32), np.zeros((S, CORE), np.float32), np.zeros(S)
    streams = {t: stream(t, seed, c) for t in 'AB'}
    res = {}
    for first, second in (('A', 'B'), ('B', 'A')):
        cue[:] = 0; val[:] = 0; cnt[:] = 0
        rec = {}
        for tag in (first, second):
            s = streams[tag]
            np.add.at(cue, s[:, 0], Zt[s[:, 0] * R + s[:, 1]])
            np.add.at(val, s[:, 0], c['coll_train'][s[:, 2], c['counts'][s[:, 0]] - 1])
            np.add.at(cnt, s[:, 0], 1)
            taught = np.flatnonzero(cnt)
            proto, content = unit(cue[taught]), val[taught] / cnt[taught, None]

            def read(t):
                off = 0 if t == 'A' else N
                rows = np.repeat(np.arange(off, off + N), E)
                near = (Zq[rows * E + np.tile(np.arange(E), N)] @ proto.T).argmax(1)
                return decode(content[near], np.ones(len(near), bool), c['counts'][rows], c[f'cand_{t}'], c[f'order_{t}'], c['protos'])
            if tag == first:
                rec['old_after_old'] = read(first)
        rec['old_after_both'], rec['new_after_both'] = read(first), read(second)
        res[f'{first}{second}'] = rec
    return res


def one_fixture(task):
    seed, address, k, tau = task
    c = load(seed)
    Zw = cues(c, address, c['fx_core'], c['fx_key'])
    mem = Memory(FIXTURE_H, k, tau, Zw.shape[1], CORE)
    targets = c['fx_target'][c['fx_write_count'] - 1]
    for z, t, col in zip(Zw, targets, c['fx_collection']):
        mem.write(z, c['coll_train'][col, t - 1])
    truth = c['fx_target'][c['fx_query_count'] - 1]
    args = truth, c['fx_cand'], c['fx_order'], c['protos']
    same = decode(*mem.read(cues(c, address, c['fx_eval_core'], c['fx_eval_key'])), *args)
    q = len(truth)
    if address in ('sensory', 'sense-only'):  # the frozen encoder has no tap path
        taps = decode(np.zeros((q, CORE), np.float32), np.zeros(q, bool), *args)
    elif address == 'core':
        taps = decode(*mem.read(unit(c['fx_taps_core'])), *args)
    else:
        taps = decode(*mem.read(unit(c['fx_taps_core']), slice(0, CORE)), *args)
    return f'{seed}/{address}/{k}/{tau}', {'same_sense': same, 'taps': taps, 'units': mem.h}, {}


def worst(rec, field, decoder='prototype'):
    return min(rec[o][field][decoder] for o in ('AB', 'BA'))


def select(address, grid, stores, seeds, decoder='prototype'):
    """The contract's rule on pilot means, worst order. Order loss = old after old - old after both."""
    rows = []
    for H, k, tau in itertools.product(HS, KS, TAUS):
        recs = [grid[f'{s}/{address}/{H}/{k}/{tau}'] for s in seeds]
        mean = lambda o, f: float(np.mean([r[o][f][decoder] for r in recs]))
        st = lambda o, f: float(np.mean([stores[f'{s}/{address}'][o][f][decoder] for s in seeds]))
        row = {'H': H, 'k': k, 'tau': tau, 'bytes_limit': recs[0]['AB']['bytes_limit'], 'bytes_used': float(np.mean([max(r[o]['bytes_used'] for o in ('AB', 'BA')) for r in recs])), 'forced': float(np.mean([r[o]['forced_after_2'] for r in recs for o in ('AB', 'BA')])),
               'units': float(np.mean([r[o]['units_after_2'] for r in recs for o in ('AB', 'BA')])), 'unknown_coverage': float(np.mean([r[o]['unknown_coverage'] for r in recs for o in ('AB', 'BA')])), 'conflicts': float(np.mean([r[o]['conflicting_targets_per_unit'] for r in recs for o in ('AB', 'BA')]))}
        for o in ('AB', 'BA'):
            row[o] = {f: mean(o, f) for f in ('old_after_old', 'old_after_both', 'new_after_both')}
            row[o]['old_loss'] = row[o]['old_after_old'] - row[o]['old_after_both']
            row[o]['coverage_old_after_both'] = float(np.mean([r[o]['old_after_both']['coverage'] for r in recs]))
            row[o]['answer'] = {d: {f: mean_d for f, mean_d in ((f, float(np.mean([r[o][f][d] for r in recs]))) for f in ('old_after_old', 'old_after_both', 'new_after_both'))} for d in ('euclidean', 'cosine')}
        row['worst_old_loss'] = max(row[o]['old_loss'] for o in ('AB', 'BA'))
        row['eligible'] = all(row[o][f] >= ACQUIRE_FLOOR and row[o][f] >= st(o, f) - STORE_MARGIN for o in ('AB', 'BA') for f in ('old_after_old', 'new_after_both'))
        rows.append(row)
    eligible = [r for r in rows if r['eligible']]
    if not eligible:
        return {'selected': None, 'rule': 'none eligible', 'rows': rows}
    good = [r for r in eligible if r['worst_old_loss'] <= LOSS_BAR]
    if good:
        sel, how = min(good, key=lambda r: (r['bytes_limit'], -r['tau'], r['k'])), 'smallest bytes with worst-order loss within the bar'
    else:
        sel, how = min(eligible, key=lambda r: (r['worst_old_loss'], r['bytes_limit'])), 'lowest worst-order loss, marked: none within the bar'
    return {'selected': sel, 'rule': how, 'rows': rows}


def run():
    t0 = time.perf_counter()
    seeds = S0.SEEDS
    tasks = [(s, a, H, k, t) for s in seeds for a in ADDRESSES for H, k, t in itertools.product(HS, KS, TAUS)]
    run_grid = DUR.durable(one_grid, RUNS / 'done-grid', '{0}-{1}-{2}-{3}-{4}')
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = list(pool.imap_unordered(run_grid, tasks))
        fx = list(pool.imap_unordered(one_fixture, [(s, a, k, t) for s in seeds for a in FIXTURE_ADDRESSES for k in KS for t in TAUS]))
    grid = {k: r for k, r, _ in res}
    stores = {f'{s}/{a}': explicit_store(s, a) for s in seeds for a in ADDRESSES}
    selection = {a: select(a, grid, stores, seeds) for a in ADDRESSES}
    fixture = {k: r for k, r, _ in fx}
    chosen = {}
    for a in FIXTURE_ADDRESSES:
        cand = []
        for k, t in itertools.product(KS, TAUS):
            rs = [fixture[f'{s}/{a}/{k}/{t}'] for s in seeds]
            same, taps = float(np.mean([r['same_sense']['prototype'] for r in rs])), float(np.mean([r['taps']['prototype'] for r in rs]))
            cand.append({'k': k, 'tau': t, 'same_sense': same, 'taps': taps, 'score': (same + taps) / 2, 'answer_same_sense': float(np.mean([r['same_sense']['euclidean'] for r in rs])), 'answer_taps': float(np.mean([r['taps']['euclidean'] for r in rs])),
                         'taps_by_seed': [r['taps']['prototype'] for r in rs], 'same_by_seed': [r['same_sense']['prototype'] for r in rs], 'taps_coverage': [r['taps']['coverage'] for r in rs], 'units': [r['units'] for r in rs]})
        chosen[a] = {'selected': max(cand, key=lambda r: (round(r['score'], 12), r['tau'], -r['k'])), 'all': cand}
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory small pilot on the locked route; no trainable comparator, probe or stress condition', 'selection': selection, 'stores': stores, 'fixture': chosen, 'grid': grid})
    cpu = sum(t['cpu_seconds'] for _, _, t in res)
    w = sum(t['write_cpu_seconds'] for _, _, t in res)
    n = sum(t['writes'] for _, _, t in res)
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'CPU: 18 settings x 3 addresses x 2 orders x 2 seeds, fixture, stores', 'device': 'cpu', 'workers': WORKERS, 'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': cpu, 'write_cpu_seconds': w, 'writes': n, 'mean_write_microseconds_with_python_overhead': 1e6 * w / max(n, 1)}))


def fixture_ceiling():
    """What these codes allow on the fixture, with the count of each write given to the store: an oracle privilege. Nine prototypes per address; similarity gaps; raw and mean-centred."""
    out = {}
    for seed in S0.SEEDS:
        c = load(seed)
        qc, E = c['fx_query_count'], P.EVAL_PER
        rec = {}
        targets = c['fx_target'][qc - 1]
        for name, W, Q, part in (('core_raw', c['fx_core'], {'same_sense': c['fx_eval_core'], 'taps': c['fx_taps_core']}, None), ('frozen_raw', c['fx_key'], {'same_sense': c['fx_eval_key']}, None),
                                 ('core_centred_on_forms', c['fx_core'] - c['fx_core'].mean(0), {'same_sense': c['fx_eval_core'] - c['fx_core'].mean(0), 'taps': c['fx_taps_core'] - c['fx_core'].mean(0)}, None),
                                 ('core_centred_per_sense', c['fx_core'] - c['fx_core'].mean(0), {'same_sense': c['fx_eval_core'] - c['fx_core'].mean(0), 'taps': c['fx_taps_core'] - c['fx_taps_core'].mean(0)}, None)):
            proto = unit(np.stack([unit(W[c['fx_write_count'] == m]).mean(0) for m in range(1, 10)]))
            euc = np.stack([W[c['fx_write_count'] == m].mean(0) for m in range(1, 10)])
            rec[name] = {}
            for q, x in Q.items():
                sim = unit(x) @ proto.T
                near = sim.argmax(1)
                row = np.arange(len(x))
                same = sim[row, qc - 1]
                best_other = np.where(sim.argsort(1)[:, -1] == qc - 1, np.sort(sim, 1)[:, -2], sim.max(1))
                rec[name][q] = {'query_count': float((near + 1 == qc).mean()), 'query_count_within_one': float((np.abs(near + 1 - qc) <= 1).mean()), 'euclidean_query_count': float((((x[:, None] - euc[None]) ** 2).sum(-1).argmin(1) + 1 == qc).mean()),
                                'similarity_right_count': float(same.mean()), 'similarity_best_other': float(best_other.mean()), 'share_right_above_0.5': float((same >= .5).mean()), 'share_right_above_0.7': float((same >= .7).mean()), 'share_right_above_0.85': float((same >= .85).mean())}
            rec[name]['form_prototypes_cosine'] = {'neighbouring_counts': float(np.mean([proto[i] @ proto[i + 1] for i in range(8)])), 'all_pairs': float((proto @ proto.T)[np.triu_indices(9, 1)].mean())}
        rec['targets_fixed_points'] = int((c['fx_target'] == np.arange(1, 10)).sum())
        out[str(seed)] = rec
    S0.save_json(f'{REPORT}-fixture-ceiling.json', {'status': 'diagnostic: an oracle store on the fixture material', 'results': out})
    for s, r in out.items():
        for k, v in r.items():
            print(s, k, v if not isinstance(v, dict) else {a: ({x: round(y, 2) for x, y in b.items()}) for a, b in v.items()})


def script_ceiling():
    """GPU. Per script, on the block-3 core: an oracle store of nine form prototypes, read from the script's held-out forms and from taps, by cosine in the code and through the host's head."""
    import torch
    from experiments import shared_core_value as V
    S0.prepare()
    out = {}
    with torch.no_grad():
        for seed in S0.SEEDS:
            enc, model = V.host(seed)
            core = lambda x: model.proj['image'](enc['image'](x))
            tap = lambda x: model.proj['sequence'](enc['sequence'](x))
            taps = S0.batched(tap, np.concatenate([D.TAPS[('eval', m)] for m in S0.ns.NUMBERS]))
            qc = np.repeat(np.arange(1, 10), P.EVAL_PER)
            withheld = D.withheld_counts(seed)
            rec = {'withheld': {k: list(map(int, v)) for k, v in withheld.items()}}
            for lang in P.LANGS:
                xt, yt = X.DATA[(lang, 'train')]
                rng = nc.stream('sc1_script_ceiling', seed, lang)
                proto = torch.stack([S0.batched(core, xt[rng.permutation(np.flatnonzero(yt == m))[:FIXTURE_RENDERS]]).mean(0) for m in S0.ns.NUMBERS])
                forms = S0.batched(core, np.concatenate([X.DATA[(lang, 'eval')][0][P.EVAL[(lang, m, 'idx')]] for m in S0.ns.NUMBERS]))
                unit_t = lambda v: torch.nn.functional.normalize(v, dim=-1)
                exp = lambda q: proto[None].expand(len(q), -1, -1)
                r = {}
                for name, q in (('forms', forms), ('taps', taps)):
                    cos = (unit_t(q) @ unit_t(proto).T).argmax(1).cpu().numpy() + 1
                    head = S0.G.pair(model, q, exp(q), 1).argmax(1).cpu().numpy() + 1
                    sim = (unit_t(q) @ unit_t(proto).T).cpu().numpy()
                    r[name] = {'cosine': float((cos == qc).mean()), 'cosine_within_one': float((np.abs(cos - qc) <= 1).mean()), 'head': float((head == qc).mean()), 'similarity_right': float(sim[np.arange(len(qc)), qc - 1].mean()), 'similarity_mean_all': float(sim.mean())}
                rec[lang] = r
            out[str(seed)] = rec
    S0.save_json(f'{REPORT}-script-ceiling.json', {'status': 'diagnostic: an oracle store of form prototypes per script', 'results': out})
    for s, r in out.items():
        for k, v in r.items():
            print(s, k, v if k == 'withheld' else {a: {x: round(y, 2) for x, y in b.items()} for a, b in v.items()})


if __name__ == '__main__':
    stage = sys.argv[2] if len(sys.argv) > 2 else 'run'
    if stage == 'codes':
        t0 = time.perf_counter()
        S0.prepare()
        for s in S0.SEEDS:
            print(s, make_codes(s), flush=True)
        S0.save_json(f'{REPORT}-codes-runtime.json', S0.runtime({'phase': 'code cache', 'wall_seconds': time.perf_counter() - t0, 'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
    elif stage == 'script-ceiling':
        script_ceiling()
    elif stage == 'fixture-ceiling':
        fixture_ceiling()
    else:
        run()
