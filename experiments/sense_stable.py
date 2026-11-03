"""Fixing centre and gate drift in online sight memory, including value decoding (docs/sense-stable-v1-pilot.md; follows docs/sense-adapt-v1-pilot.md).

Same facts, streams, shift rule and gate refits as `experiments.sense_adapt` found/online. Arms:
  online         as sense_adapt found/online (must read exactly as there): one memory; every cue, target, query and candidate coded on its own nearest centre; the current gate
  kept           each unit keeps the centre it was written under, for key and value: a fact's cue picks the centre, and its target is coded on the same centre;
                 units are partitioned by centre, so writes never mix coordinate systems; a read codes the query on every centre, scores each centre's units in that
                 centre's coordinates, and the best unit wins; candidates are coded on the winning unit's centre; the current gate
  kept+frozen    as kept, and each centre keeps the gate as refitted when it opened (the query-dependent function, not an average weight)
Invariant expected of kept+frozen: an unchanged query's score against an unchanged old unit, and its value decoding, stay exactly fixed when centres open or the gate refits.
Reads after every stage twice: `pre`, after the kind's passing inputs (centres opened, gate refitted) and before its facts; `post`, after its facts.
pre[s] against post[s-1] isolates representation drift (no writes between); post[s] against pre[s] isolates interference from new units.

Usage: NFI_DEVICE=cuda SC0_WORKERS=2 OPENBLAS_NUM_THREADS=1 /data/venvs/sri-cuda/bin/python -m experiments.sense_stable pilot
"""
import os
import sys
import time

import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import general_encoder as GE
from experiments import natural_memory as NM
from experiments import natural_memory_sequence as SEQ
from experiments import sense_adapt as SA
from experiments import shared_core_stage0 as S0
from src import senses as SN

nc = S0.nc
NAME = 'sense-stable-v1' + os.environ.get('SS_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
SEEDS, TAU, N, MODELS = SA.SEEDS, SA.TAU, SA.N, SA.MODELS
PER_CENTRE = 2048  # units per centre's partition; about 200 are used per kind
ARMS = ('online', 'kept', 'kept+frozen')


def floats(x):
    return [float(v) for v in x]


def stable_task(task):
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    S0.set_up_task('sparse48')
    kinds = SEQ.kind_order(seed, order)
    data = {k: SEQ.kind_data(k, seed) for k in kinds}
    index = {k: i for i, k in enumerate(kinds)}
    wrng = nc.stream('nm_seq_writes', seed, order)
    qrng = nc.stream('nm_seq_queries', seed)
    q = {k: {v: NM.view_arrays(data[k]['cues'], qrng, c) for v, c in NM.VIEWS.items()} for k in SEQ.KINDS}
    qu = {k: {v: NM.view_arrays(data[k]['unseen'], qrng, c) for v, c in NM.VIEWS.items()} for k in SEQ.KINDS}
    dev = DEV.NAME if DEV.ACTIVE else 'cpu'
    sense = SN.Sense(MODELS, dev, chunk=64)
    E = sense.names

    # ---- online centres and gate refits, exactly as sense_adapt
    online = SA.ShiftCentres(nc.stream('ge_adapt_boot', seed, order))
    prng, vrng = nc.stream('ge_passing', seed, order), nc.stream('ge_adapt_views', seed, order)
    kept_codes, refits = [], []
    gate = {'now': None, 'snap': []}  # snap[j]: the gate as refitted when centre j opened

    def refit_gate():
        raw = [np.concatenate([NM.unit(k[n][0]) for k in kept_codes]) for n in E]
        views = []
        for n in E:
            C = online.centre_array(n)
            views.append(np.stack([np.concatenate([SA.coded(k[n][i], C, online.assign({'dinov2-s': k['dinov2-s'][i]})) for k in kept_codes]) for i in range(2)]))
        g, loss = SN.Gate.train(raw, views, seed, GE.GATE_STEPS, GE.GATE_BATCH, GE.GATE_LR, GE.GATE_TEMP)
        gate['now'] = g
        refits.append({'centres': len(online.centres), 'loss': loss, 'codes': int(len(raw[0]))})
        return g

    D = SN.get('dinov2-s').dim
    facts = len(SEQ.KINDS) * N
    single = NM.Tagged(SEQ.CAPACITY, TAU, 2 * D, 2 * D, facts)
    parts_mem = {arm: [] for arm in ('kept', 'kept+frozen')}  # per centre

    def centre_parts(f, j):
        return [SN.centred(f[n], online.centre_array(n)[j][None]) for n in E]

    def own_parts(f):
        a = online.assign(f)
        return [SA.coded(f[n], online.centre_array(n), a) for n in E]

    def gate_weights(g, f):
        w = g([NM.unit(f[n]) for n in E])
        return [w[:, i] for i in range(len(E))]

    read_seconds = {arm: 0. for arm in ARMS}
    read_queries = {arm: 0 for arm in ARMS}

    def read(arm, f, fcand, rows_k):
        """Best unit for each query: (centre, unit, score, fact row of the unit, value correct), and the query's best score against units holding its own fact."""
        t = time.perf_counter()
        nq = len(next(iter(f.values())))
        if arm == 'online':
            Z = SN.join(own_parts(f), gate_weights(gate['now'], f))
            S = Z @ single.a[:single.h].T
            cen = np.zeros(nq, np.int64)
            i = S.argmax(1)
            s = S[np.arange(nq), i]
            trow = single.tags[:single.h].argmax(1)
            val = (single.u[:single.h] / single.n[:single.h, None])[i]
            ok = (val @ SN.join(own_parts(fcand)).T).argmax(1) == np.arange(N) if fcand is not None else None
            row = trow[i]
            own = np.where(trow[None] == rows_k[:, None], S, -np.inf).max(1) if rows_k is not None else None
        else:
            mems = parts_mem[arm]
            blocks, cen_of, unit_of, trows = [], [], [], []
            for j, mem in enumerate(mems):
                if not mem.h:
                    continue
                g = gate['snap'][j] if arm == 'kept+frozen' else gate['now']
                Z = SN.join(centre_parts(f, j), gate_weights(g, f))
                blocks.append(Z @ mem.a[:mem.h].T)
                cen_of.append(np.full(mem.h, j))
                unit_of.append(np.arange(mem.h))
                trows.append(mem.tags[:mem.h].argmax(1))
            S, cen_of, unit_of, trow = np.concatenate(blocks, 1), np.concatenate(cen_of), np.concatenate(unit_of), np.concatenate(trows)
            b = S.argmax(1)
            s = S[np.arange(nq), b]
            cen, i = cen_of[b], unit_of[b]
            ok = None
            if fcand is not None:
                val = np.stack([mems[c].u[u] / mems[c].n[u] for c, u in zip(cen, i)])
                Zc = {c: SN.join(centre_parts(fcand, c)) for c in np.unique(cen)}
                ok = np.array([(val[r] @ Zc[cen[r]].T).argmax() == r for r in range(nq)])
            own = np.where(trow[None] == rows_k[:, None], S, -np.inf).max(1) if rows_k is not None else None
            row = trow[b]
        read_seconds[arm] += time.perf_counter() - t
        read_queries[arm] += nq
        return {'centre': cen, 'unit': i, 'score': s, 'row': row, 'value_ok': ok, 'own': own}  # a unit is identified by (centre, unit)

    labels = {k: GE.kind_labels(k, seed) for k in kinds}
    cue_labels = np.stack([labels[k][0] for k in kinds])
    truth = np.arange(N)
    cache, reads, stages, taught, forming_imgs = {}, {}, [], [], []
    forced = {arm: 0 for arm in ARMS}

    def read_all(tag):
        stage = {}
        for k in taught:
            rows_k = SEQ.fact_row(index[k], truth)
            for view in NM.VIEWS:
                fq, fu, fcand = cache[k][f'q-{view}'], cache[k][f'u-{view}'], cache[k]['cand']
                for arm in ARMS:
                    r, ru = read(arm, fq, fcand, rows_k), read(arm, fu, None, None)
                    rec = {**GE.score(r['row'] == rows_k, r['score'], r['value_ok'], ru['score'], TAU),
                           'resemblance': GE.resemblance(ru['row'], ru['score'], TAU, index[k], labels[k][1], cue_labels),
                           'q': {'centre': r['centre'].tolist(), 'unit': r['unit'].tolist(), 'score': floats(r['score']), 'value_ok': r['value_ok'].tolist(), 'own': floats(r['own'])},
                           'u': {'score': floats(ru['score'])}}
                    stage[f'{arm}|{k}-{view}'] = rec
        reads[tag] = stage

    for s_i, kind in enumerate(kinds):
        passing = np.ascontiguousarray(data[kind]['pool'][prng.permutation(len(data[kind]['pool']))[:SA.PASSING]], dtype=np.float32)
        froze = []
        for b in range(0, SA.PASSING, SA.BATCH):
            imgs = passing[b:b + SA.BATCH]
            j, new = online.observe(sense(imgs), tag=kind)
            if j is None or new:
                forming_imgs.append(imgs)
            if new:
                x = np.concatenate(forming_imgs)
                f1, f2 = sense(NM.view_arrays(x, vrng, NM.VIEWS['mild'])), sense(NM.view_arrays(x, vrng, NM.VIEWS['strong']))
                kept_codes.append({n: np.stack([f1[n], f2[n]]) for n in E})
                forming_imgs = []
                froze.append(j)
                gate['snap'].append(refit_gate())
                for arm in parts_mem:
                    parts_mem[arm].append(NM.Tagged(PER_CENTRE, TAU, 2 * D, 2 * D, facts))
        del passing
        stages.append({'kind': kind, 'centres': len(online.centres), 'froze': froze, 'thresholds': list(online.thresholds)})
        if taught:
            read_all(f'{s_i + 1}:{kind}:pre')

        plan = NM.plan_writes(wrng, np.arange(N), SEQ.EXPOSURES, SEQ.BLOCK)
        cues, targets, rows = [], [], []
        for batch in plan:
            if not len(batch):
                continue
            cues.append(NM.view_arrays(data[kind]['cues'][batch], wrng, NM.VIEWS['mild']))
            targets.append(NM.view_arrays(data[kind]['targets'][batch], wrng, NM.VIEWS['mild']))
            rows.append(SEQ.fact_row(index[kind], batch))
        cues, targets, rows = np.concatenate(cues), np.concatenate(targets), np.concatenate(rows)
        fc, ft = sense(cues), sense(targets)
        z, v = SN.join(own_parts(fc)), SN.join(own_parts(ft))
        for i in range(len(rows)):
            single.write_tagged(z[i], v[i], rows[i])
        a = online.assign(fc)  # the cue picks the centre for cue and target
        for j in np.unique(a):
            m = a == j
            zj, vj = SN.join(centre_parts({n: fc[n][m] for n in E}, j)), SN.join(centre_parts({n: ft[n][m] for n in E}, j))
            for arm in parts_mem:
                for zi, vi, ri in zip(zj, vj, rows[m]):
                    parts_mem[arm][j].write_tagged(zi, vi, ri)
        taught.append(kind)
        if kind not in cache:
            cache[kind] = {'cand': sense(data[kind]['targets']), **{f'q-{v}': sense(q[kind][v]) for v in NM.VIEWS}, **{f'u-{v}': sense(qu[kind][v]) for v in NM.VIEWS}}
        read_all(f'{s_i + 1}:{kind}:post')

    forced = {'online': single.forced, **{arm: int(sum(m.forced for m in parts_mem[arm])) for arm in parts_mem}}
    units = {'online': single.h, **{arm: [m.h for m in parts_mem[arm]] for arm in parts_mem}}
    kept_bytes = int(sum(sum(a.nbytes for a in k.values()) for k in kept_codes))
    gate_bytes = 4 * gate['now'].parameters()
    centre_bytes = 4 * len(online.centres) * sum(SN.get(n).dim for n in E)
    state = {'online': {'units': single.bytes_used(), 'centres': centre_bytes, 'gates': gate_bytes, 'kept_codes': kept_bytes},
             'kept': {'units': int(sum(m.bytes_used() for m in parts_mem['kept'])), 'centres': centre_bytes, 'gates': gate_bytes, 'kept_codes': kept_bytes},
             'kept+frozen': {'units': int(sum(m.bytes_used() for m in parts_mem['kept+frozen'])), 'centres': centre_bytes, 'gates': gate_bytes * len(gate['snap']), 'kept_codes': kept_bytes}}
    latency = {arm: read_seconds[arm] / max(read_queries[arm], 1) for arm in ARMS}
    res = {'reads': reads, 'kinds': kinds, 'seed': seed, 'order': order, 'stages': stages, 'shift_log': online.log, 'gate_refits': refits,
           'forced_writes': forced, 'units': units, 'state_bytes': state}
    return f'{seed}/{order}', res, {**S0.timing(t0, c0), 'senses': sense.describe(), 'read_seconds_per_query': latency}


def pilot():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(stable_task, RUNS / 'done-pilot', '{0}-{1}'), [(s, o) for s in SEEDS for o in SEQ.ORDERS])
    S0.save_json(f'{REPORT}-pilot.json', {'status': 'exploratory, pilot seeds, both orders; frozen DINOv2-S and TrOCR-small through src.senses; online centres and gate as sense_adapt; centre kept per unit for key and value, gate frozen per centre; reads before and after each kind\'s facts',
                                          'arms': ARMS, 'tau': TAU, 'per_centre_units': PER_CENTRE, 'licences': SN.licences(), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-pilot-runtime.json', S0.runtime({'phase': 'two frozen sight models, online centres and gates, three memory designs, reads before and after each kind', 'wall_seconds': time.perf_counter() - t0,
                                                             'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'pilot': pilot}[sys.argv[1]]()
