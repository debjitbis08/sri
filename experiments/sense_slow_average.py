"""Memory addressed through a slow average of the learning key encoder (docs/sense-slow-average-v1-pilot.md). The working encoder keeps learning; memory reads and writes keys through a parameter average, with no map on keys and no image held. Stored codes simply age with the average.

Values: counts in host codes, moved by a live map every window, as in the windows pilot; symbols in the code of the arm's own encoder, with no map.

Thresholds: every memory writes at tau. Reads are given at tau, at a live threshold and at matched false acceptance. The live threshold is one number per memory: a running estimate of the similarity at which ALPHA of inputs passing now, which memory is not being taught, would get an answer. Matched false acceptance uses the unseen queries and only scores.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_slow_average pilot
"""
import copy
import os
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import sense_consolidation as SC
from experiments import sense_consolidation_moving as M
from experiments import sense_consolidation_traces as T
from experiments import sense_drift_structure as DS
from experiments import sense_image_free as F
from experiments import sense_image_free_windows as W
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import shared_core_unseen_traces as UT
from experiments import translation_stable_keys as TK

nc, P = S0.nc, S0.P
NAME = 'sense-slow-average-v1' + os.environ.get('SI_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
N, E, R = F.N, F.E, F.R
BLOCK, BATCH, WINDOW = F.BLOCK, F.BATCH, W.WINDOW
BETAS = (.99, .999, .9999)
ALPHA = .13  # the stored images' rate of answering unseen symbols
LIVE_STEP = .01
LIVE_PER_UPDATE = 16
STREAMS = ('stationary', 'shift', 'natural')
ENCODERS = ('working', *[f'ema-{b}' for b in BETAS], 'frozen')


def arms():
    return {**{e: dict(encoder=e, H=2048) for e in ENCODERS}, 'ema-0.999-H512': dict(encoder='ema-0.999', H=512)}


def live_step(q, best, alpha=ALPHA, step=LIVE_STEP):
    """Stochastic approximation: q moves until alpha of the best similarities exceed it."""
    return float(np.clip(q + step * (float((best > q).mean()) - alpha), F.FLOOR, .99))


def update_average(average, model, beta):
    with torch.no_grad():
        for p, q in zip(average.parameters(), model.parameters()):
            p.mul_(beta).add_(q, alpha=1 - beta)


def read_at(vals, mem, thr, Zq, Zf, mask, view, tags):
    keep = mem.thr.copy()
    mem.thr[:mem.h] = thr
    out = F.read_arm(vals, mem, Zq, Zf, mask, view, tags)
    mem.thr[:] = keep
    return out


def read_all(vals, mem, live, Zq, Zf, mask, view, tags):
    unseen = Zq[2 * N * E:3 * N * E]
    out = {'tau': read_at(vals, mem, F.TAU, Zq, Zf, mask, view, tags)}
    if live is not None:
        out['live'] = read_at(vals, mem, live, Zq, Zf, mask, view, tags)
        out['live_threshold'] = live
    h = mem.h
    best = (unseen @ mem.a[:h].T).max(1) if h else np.zeros(1)
    for target in W.TARGETS:
        out[f'matched-{target}'] = read_at(vals, mem, float(np.quantile(best, 1 - target)), Zq, Zf, mask, view, tags)
    return out


def learn(task):
    seed, lr, order, stream_kind = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    M.LS.TAU = F.TAU
    c = S1.load(seed)
    mat = F.load_material(seed)
    model = SM.encoder('unlabelled', seed)
    frozen = SM.encoder('unlabelled', seed).requires_grad_(False).eval()
    opt = torch.optim.Adam(model.parameters(), lr)
    averages = {f'ema-{b}': copy.deepcopy(model).requires_grad_(False).eval() for b in BETAS}
    nets = {'working': model, **averages, 'frozen': frozen}
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    fonts = mat['other_font'].reshape(-1, 28, 28)
    rng = nc.stream('sm_learning', seed, str(lr))
    prng = nc.stream('sa_probes', seed, str(lr), order, stream_kind)
    streams = {t: S1.stream(t, seed, c) for t in 'AB'}
    first, second = SC.ORDERS[order]
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    unseen_images = UT.source_images(seed)['new-radius']
    x_natural = DS.natural()[0]
    pool = {'shift': np.concatenate([P.DOTS['train'].reshape(-1, 28, 28), *[S0.X.DATA[(f, 'eval')][0] for f in UT.FORMATS[:3]]]).astype(np.float32), 'natural': x_natural[:-min(1000, len(x_natural) // 3)]}.get(stream_kind)
    values = {d: F.Values(d, seed, order, c, mat, unseen_images) for d in ('counts', 'symbols')}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    code = lambda m: (lambda x: np_(enc(m, x)))
    free = {d: {k: SC.RadiusMemory(a['H'], M.LS.K, F.TAU, 32, F.VALUE_WIDTH[d]) for k, a in arms().items()} for d in values}
    specs = M.arms()
    imaged = {d: {k: T.Arm(**specs[k]) for k in F.IMAGE_ARMS} for d in values}
    for d in imaged:
        for k, arm in imaged[d].items():
            arm.mem = SC.RadiusMemory(specs[k]['H'], M.LS.K, F.TAU, 32, F.VALUE_WIDTH[d])
    live = {k: F.TAU for k in arms()}
    first_coll = np.full(len(cue), -1, np.int64)
    reads, at = {}, 0
    host_now, host_next = W.HostCopy(seed).load(S0.checkpoint_path('sparse48', seed)), W.HostCopy(seed)
    write_table = values['counts'].table(host_now)
    with torch.no_grad():
        moved = {'start_cues': {e: np_(enc(n, cue)) for e, n in nets.items()}}

    for b, tag in ((4, first), (5, second), (6, None)):
        s = streams[tag] if tag else None
        for u in range(BLOCK):
            if pool is not None and b >= 5:
                base = torch.from_numpy(np.ascontiguousarray(pool[rng.integers(len(pool), size=TK.CLASSES)], dtype=np.float32))
            else:
                base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=TK.CLASSES), rng.integers(n_r, size=TK.CLASSES)], dtype=np.float32))
            x = torch.stack([SRL.augment_with(base, rng, SRL.AUGMENTS['full']), SRL.augment_with(base, rng, SRL.AUGMENTS['full'])], 1).reshape(-1, 28, 28)
            model.train()
            z = model(x)
            loss = SRL.info_nce(z @ z.T / SRL.T, partner)
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            for beta, avg in zip(BETAS, averages.values()):
                update_average(avg, model, beta)
            at += 1
            with torch.no_grad():
                passing = {e: S1.unit(np_(enc(n, base[:LIVE_PER_UPDATE].numpy()))) for e, n in nets.items()}
            for k, a in arms().items():
                mem = free['counts'][k]  # addresses do not depend on values, so one estimate serves both
                if mem.h:
                    live[k] = live_step(live[k], (passing[a['encoder']] @ mem.a[:mem.h].T).max(1))
            if s is not None:
                part = s[u * BATCH:(u + 1) * BATCH]
                flat = part[:, 0] * R + part[:, 1]
                images = np.ascontiguousarray(mat['tgt_eval'][part[:, 0], part[:, 2] % F.VALUE_RENDERS], dtype=np.float32)
                with torch.no_grad():
                    cues = {e: np_(enc(n, cue[flat])) for e, n in nets.items()}
                    vals_now = {e: np_(enc(n, images)) for e, n in nets.items()}
                for i in range(len(part)):
                    sym, col = int(part[i, 0]), int(part[i, 2])
                    if first_coll[flat[i]] < 0:
                        first_coll[flat[i]] = col
                    for d, vals in values.items():
                        for k, a in arms().items():
                            e = a['encoder']
                            v = write_table[col % len(write_table), vals.fact_row[sym]] if d == 'counts' else vals_now[e][i]
                            free[d][k].write(cues[e][i], v)
                        for arm in imaged[d].values():
                            arm.write(cues['working'][i], np.zeros(F.VALUE_WIDTH[d], np.float32), flat[i])
            if at % WINDOW == 0:
                host_next.load(W.host_path(seed, order, at))
                hmap = W.RunningMap(S1.CORE)
                for _ in range(4):
                    probe = F.live_images(cue, tag, prng)[:W.HOST_PROBES // 4]
                    hmap.add(host_now(probe), host_next(probe))
                fn = hmap.fn()
                for mem in free['counts'].values():
                    h = mem.h
                    if h:
                        mem.u[:h] = fn(mem.u[:h] / mem.n[:h, None]) * mem.n[:h, None]
                host_now, host_next = host_next, host_now
                write_table = values['counts'].table(host_now)
        with torch.no_grad():
            Zq = {e: S1.unit(np_(enc(n, query))) for e, n in nets.items()}
            Zf = {e: S1.unit(np_(enc(n, fonts))) for e, n in nets.items()}
            table_cue = np_(enc(model, cue))
            now_cues = {e: np_(enc(n, cue)) for e, n in nets.items()}
        tags = first if b == 4 else 'AB'
        rec_b = {}
        for d, vals in values.items():
            rec = {}
            views = {e: vals.view(host_now if d == 'counts' else code(n), b) for e, n in nets.items()}
            for e, n in nets.items():
                rec[f'ceiling-{e}'] = {t: vals.ceiling(vals.table(host_now if d == 'counts' else code(n)), views[e], t) for t in tags}
            for k, a in arms().items():
                e, mem = a['encoder'], free[d][k]
                rec[k] = {**read_all(vals, mem, live[k], Zq[e], Zf[e], mat['font_mask'], views[e], tags), 'units': mem.h, 'forced': mem.forced, 'payload_bytes': 4 * mem.h * (32 + F.VALUE_WIDTH[d] + 2), 'allocated_bytes': sum(x.nbytes for x in (mem.c, mem.a, mem.u, mem.n, mem.thr))}
            fresh = vals.table(host_now if d == 'counts' else code(model))
            for k, arm in imaged[d].items():
                h = arm.mem.h
                arm.mem.u[:h] = M.unit_values(arm.tu, arm.tr, h, first_coll, vals.row_counts, fresh) * arm.mem.n[:h, None]
                arm.tick(table_cue, vals.score)
                h = arm.mem.h
                arm.mem.u[:h] = M.unit_values(arm.tu, arm.tr, h, first_coll, vals.row_counts, fresh) * arm.mem.n[:h, None]
                rec[f'images-{k}'] = {**W.read_all(vals, arm.mem, Zq['working'], Zf['working'], mat['font_mask'], views['working'], tags), 'units': h, 'forced': arm.mem.forced, 'payload_bytes': 4 * h * (32 + 2) + len(arm.tu) * M.BYTES_PER_EXPERIENCE}
            rec_b[d] = rec
        reads[str(b)] = {'reads': rec_b, 'cue_movement': {e: float(np.median(np.arccos(np.clip((S1.unit(moved['start_cues'][e]) * S1.unit(now_cues[e])).sum(1), -1., 1.)))) for e in nets}}
        moved['start_cues'] = now_cues
    return f'{seed}/{lr}/{order}/{stream_kind}', {'blocks': reads, 'seed': seed, 'lr': lr, 'order': order, 'stream': stream_kind}, S0.timing(t0, c0)


def main():
    t0 = time.perf_counter()
    S0.prepare()
    F.material()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done', '{0}-{1}-{2}-{3}'), [(s, lr, o, k) for s in S0.SEEDS for lr in SC.LRS for o in SC.ORDERS for k in STREAMS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; memory through a parameter average of the key encoder; no image held but in the images reference; live threshold from passing inputs; matched false acceptance only scores', 'arms': arms(), 'betas': BETAS, 'alpha': ALPHA, 'natural_images': len(DS.natural()[0]), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'key encoder, averages and memory writes on both sets of values', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
