"""Many kinds of image in turn, one memory (docs/natural-memory-v1-pilot.md). The stream shows each kind for a while, then facts on that kind are taught, then the next kind arrives. Every set is read after every later one, in a seeded order and its reverse. Does keeping a frozen copy per stretch of time hold as copies grow, and do false answers grow with them?

A fact is an image and its own target image of the same kind. Queries are new views. No arm holds an image except the rebuild reference, which only scores.

Usage: NFI_DEVICE=cuda SC0_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.natural_memory_sequence pilot
"""
import copy
import os
import time

import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import natural_memory as NM
from experiments import natural_memory_mixed as MX
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import translation_stable_keys as TK

nc = S0.nc
NAME = 'natural-memory-v1-sequence' + os.environ.get('NM_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
KINDS = ('natural', 'mnist', 'tree', 'thai_digits', 'banana', 'chinese')
ORDERS = ('forward', 'reverse')
SEEDS = tuple(int(s) for s in os.environ.get('NM_SEEDS', '392,393').split(','))
PRELUDE = int(os.environ.get('NM_PRELUDE', 2000))
BLOCK = int(os.environ.get('NM_BLOCK', 2000))
REST = int(os.environ.get('NM_REST', 2000))
N, EXPOSURES, N_CANDIDATES = 45, 8, 150
TAU = .85
INTERVALS = (700, 1300, 3100)
BETAS = (.999, .9999)
CAPACITY = 8192


def kind_order(seed, order):
    """The seed's order of kinds, or its reverse."""
    perm = [KINDS[i] for i in nc.stream('nm_kinds', seed).permutation(len(KINDS))]
    return perm if order == 'forward' else perm[::-1]


def fact_row(kind_index, fact):
    """A kind's facts have their own rows in the scoring table."""
    return kind_index * N + fact


def kind_data(kind, seed):
    """Stream pool, and from a split the stream never shows: cues, unseen images and targets. Grey 28 x 28 in [0, 1]."""
    rng = nc.stream('nm_kind_facts', seed, kind)
    if kind == 'natural':
        d = NM.data()
        pool, held = NM.grey28(d['x_train']), NM.grey28(d['x_test'])
    else:
        pool, held = np.ascontiguousarray(S0.X.DATA[(kind, 'train')][0], dtype=np.float32), np.ascontiguousarray(S0.X.DATA[(kind, 'eval')][0], dtype=np.float32)
    pick = rng.permutation(len(held))[:2 * N + N_CANDIDATES]
    return {'pool': pool, 'cues': held[pick[:N]], 'unseen': held[pick[N:2 * N]], 'targets': held[pick[2 * N:]]}


def learn(task):
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    S0.set_up_task('sparse48')
    model = SM.encoder('unlabelled', seed)
    opt = torch.optim.Adam(model.parameters(), NM.LR)
    frozen = copy.deepcopy(model).requires_grad_(False).eval()
    averages = {f'ema-{b}': copy.deepcopy(model).requires_grad_(False).eval() for b in BETAS}
    kinds = kind_order(seed, order)
    data = {k: kind_data(k, seed) for k in kinds}
    index = {k: i for i, k in enumerate(kinds)}
    rng = nc.stream('nm_seq_stream', seed, order)
    wrng = nc.stream('nm_seq_writes', seed, order)
    qrng = nc.stream('nm_seq_queries', seed)
    q = {k: {v: NM.view_arrays(data[k]['cues'], qrng, c) for v, c in NM.VIEWS.items()} for k in KINDS}
    qu = {k: {v: NM.view_arrays(data[k]['unseen'], qrng, c) for v, c in NM.VIEWS.items()} for k in KINDS}
    partner = NM.partners(TK.CLASSES)
    fx = {k: NM.fixed_codes(k, 'small', seed) for k in ('pixels', 'pca')}
    nets = {'working': model, **averages, 'frozen-start': frozen}
    coders = {e: NM.encode_fn(m) for e, m in nets.items()}
    coders.update({k: v[0] for k, v in fx.items()})
    width = {**{e: NM.KEY for e in nets}, **{k: v[1] for k, v in fx.items()}}
    facts = len(KINDS) * N
    mems = {e: NM.Tagged(CAPACITY, TAU, width[e], width[e], facts) for e in coders}
    version_arms = {'versions': [], **{f'every-{k}': [] for k in INTERVALS}}
    held, cache = [], {}

    def code(e, x, key=None):
        if e in fx or e == 'frozen-start':
            if key is not None and (e, key) in cache:
                return cache[(e, key)]
            out = NM.unit(coders[e](x))
            if key is not None:
                cache[(e, key)] = out
            return out
        return NM.unit(coders[e](x))

    def fork(versions):
        if versions and not versions[-1]['mem'].h:
            versions.pop()
        net = copy.deepcopy(model).requires_grad_(False).eval()
        versions.append({'net': net, 'code': NM.encode_fn(net), 'mem': NM.Tagged(CAPACITY, TAU, NM.KEY, NM.KEY, facts)})

    def write_step(kind, batch):
        if not len(batch):
            return
        cues = NM.view_arrays(data[kind]['cues'][batch], wrng, NM.VIEWS['mild'])
        targets = NM.view_arrays(data[kind]['targets'][batch], wrng, NM.VIEWS['mild'])
        rows = fact_row(index[kind], batch)
        held.append((cues, targets, rows))
        for e in coders:
            z, v = code(e, cues), code(e, targets)
            for i in range(len(rows)):
                mems[e].write_tagged(z[i], v[i], rows[i])
        for versions in version_arms.values():
            ver = versions[-1]
            z, v = NM.unit(ver['code'](cues)), NM.unit(ver['code'](targets))
            for i in range(len(rows)):
                ver['mem'].write_tagged(z[i], v[i], rows[i])

    def score(tag_ok, sim, val_ok, unseen_sim):
        out = {'unseen_at_tau': float((unseen_sim >= TAU).mean())}
        for name, thr in {'tau': TAU, 'answer-all': -2., 'matched-0.13': NM.matched_threshold(unseen_sim, .13)}.items():
            ok = sim >= thr
            out[name] = {'threshold': float(thr), 'coverage': float(ok.mean()), 'address': float((ok & tag_ok).mean()), 'value': float((ok & val_ok).mean())}
        return out

    def read_one(mem, Z, Zu, Zc, rows, truth):
        h = mem.h
        i, s = mem.nearest(Z)
        _, su = mem.nearest(Zu)
        val = mem.u[:h] / mem.n[:h, None]
        return score(mem.tags[:h][i].argmax(1) == rows, s, (val[i] @ Zc.T).argmax(1) == truth, su)

    def read_all(taught):
        out = {}
        truth = np.arange(N)
        for kind in taught:
            rows = fact_row(index[kind], truth)
            cand = data[kind]['targets']
            for v in NM.VIEWS:
                name = f'{kind}-{v}'
                for e, mem in mems.items():
                    out[f'{e}|{name}'] = {**read_one(mem, code(e, q[kind][v], name), code(e, qu[kind][v], name + '-u'), code(e, cand, kind + '-c'), rows, truth), 'units': mem.h}
                for arm, all_versions in version_arms.items():
                    versions = [x for x in all_versions if x['mem'].h]
                    near = [x['mem'].nearest(NM.unit(x['code'](q[kind][v]))) for x in versions]
                    w, i, s = MX.pick_answer(np.stack([n[1] for n in near]), np.stack([n[0] for n in near]))
                    su = np.max(np.stack([x['mem'].nearest(NM.unit(x['code'](qu[kind][v])))[1] for x in versions]), 0)
                    tag_ok, val_ok = np.zeros(N, bool), np.zeros(N, bool)
                    for k, x in enumerate(versions):
                        at = w == k
                        if at.any():
                            m = x['mem']
                            tag_ok[at] = m.tags[:m.h][i[at]].argmax(1) == rows[at]
                            val_ok[at] = ((m.u[:m.h] / m.n[:m.h, None])[i[at]] @ NM.unit(x['code'](cand)).T).argmax(1) == truth[at]
                    out[f'{arm}|{name}'] = {**score(tag_ok, s, val_ok, su), 'units': sum(x['mem'].h for x in versions), 'copies': len(versions)}
        return out

    def rebuild_reads(taught):
        """Reference: everything rewritten through the working encoder now. Holds images, so it only scores."""
        out = {}
        cues = np.concatenate([x[0] for x in held]); targets = np.concatenate([x[1] for x in held]); r_ = np.concatenate([x[2] for x in held])
        z, val = code('working', cues), code('working', targets)
        mem = NM.Tagged(CAPACITY, TAU, NM.KEY, NM.KEY, facts)
        for i in range(len(r_)):
            mem.write_tagged(z[i], val[i], r_[i])
        truth = np.arange(N)
        for kind in taught:
            rows = fact_row(index[kind], truth)
            Zc = code('working', data[kind]['targets'])
            for v in NM.VIEWS:
                out[f'rebuild-images|{kind}-{v}'] = {**read_one(mem, code('working', q[kind][v]), code('working', qu[kind][v]), Zc, rows, truth), 'units': mem.h}
        return out

    reads, u_all = {}, 0
    train_wall = write_wall = 0.
    taught = []

    def train_step(kind):
        nonlocal u_all, train_wall
        for k in INTERVALS:
            if u_all % k == 0:
                fork(version_arms[f'every-{k}'])
        t = time.perf_counter()
        pool = data[kind]['pool']
        base = torch.from_numpy(np.ascontiguousarray(pool[rng.integers(len(pool), size=TK.CLASSES)], dtype=np.float32))
        x = torch.stack([NM.stream_views(base, rng, 'strong'), NM.stream_views(base, rng, 'strong')], 1).reshape(-1, 28, 28)
        z = model(x)
        loss = SRL.info_nce(z @ z.T / NM.T, partner)
        opt.zero_grad(); loss.backward(); opt.step()
        for beta, avg in zip(BETAS, averages.values()):
            NM.update_average(avg, model, beta)
        u_all += 1
        train_wall += time.perf_counter() - t

    for kind in kinds:
        for _ in range(PRELUDE):
            train_step(kind)
        fork(version_arms['versions'])
        plan = NM.plan_writes(wrng, np.arange(N), EXPOSURES, BLOCK)
        for u in range(BLOCK):
            train_step(kind)
            if u % NM.WRITE_EVERY == NM.WRITE_EVERY - 1:
                t = time.perf_counter()
                with torch.no_grad():
                    write_step(kind, plan[u // NM.WRITE_EVERY])
                write_wall += time.perf_counter() - t
        taught.append(kind)
        with torch.no_grad():
            reads[f'{len(taught)}:{kind}'] = {**read_all(taught), **rebuild_reads(taught)}
    for _ in range(REST):
        train_step(kinds[-1])
    with torch.no_grad():
        reads['end'] = {**read_all(taught), **rebuild_reads(taught)}
    res = {'reads': reads, 'kinds': kinds, 'seed': seed, 'order': order, 'updates': u_all, 'train_wall_seconds': train_wall, 'write_wall_seconds': write_wall}
    return f'{seed}/{order}', res, S0.timing(t0, c0)


def main():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done', '{0}-{1}'), [(s, o) for s in SEEDS for o in ORDERS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, an order of kinds and its reverse; one memory; code only but for the rebuild reference', 'kinds': KINDS, 'intervals': INTERVALS, 'betas': BETAS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder, averages, memory writes and reads across kinds', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
