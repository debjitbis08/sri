"""One memory, two kinds of input: glyph facts and natural-image facts, in both orders, with the stream changing between them (docs/natural-memory-v1-pilot.md). This is the trade a learning encoder faces: a frozen glyph encoder should keep glyph facts and miss natural ones, a learning encoder the reverse.

Orders:
  GN  glyph facts on a glyph stream; the stream turns natural; a prelude; natural facts; rest
  NG  a natural prelude from the same glyph-trained start; natural facts; the stream returns to glyphs; a prelude; glyph facts; rest

Arms write and read one memory holding both sets, except `versions`, which keeps one frozen copy of the encoder per block and one memory per copy: a read tries every copy and takes the most similar answer. No arm holds an image except the rebuild reference, which only scores.

Usage: NFI_DEVICE=cuda SC0_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.natural_memory_mixed pilot
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
from experiments import sense_image_free as SIF
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import translation_stable_keys as TK

nc = S0.nc
NAME = 'natural-memory-v1-mixed' + os.environ.get('NM_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
N, E, R = SIF.N, SIF.E, SIF.R
BETAS = (.999, .9999)
BLOCK = int(os.environ.get('NM_BLOCK', 2000))
PRELUDE = int(os.environ.get('NM_PRELUDE', 4000))
REST = int(os.environ.get('NM_REST', 4000))
GLYPH_EXPOSURES, NATURAL_EXPOSURES = 64, 8
TAUS = (.85, .7)
ORDERS = {'GN': ('glyph', 'natural'), 'NG': ('natural', 'glyph')}
SEEDS = (392, 393)
FACTS = 2 * N  # glyph facts then natural facts
CAPACITY = 8192
MATCHED = (.13,)
INTERVALS = tuple(int(k) for k in os.environ.get('NM_INTERVALS', '').split(',') if k)


def pick_answer(sim_by_version, unit_by_version):
    """Per query, the version whose nearest unit is most similar. Arrays (versions, q). Returns the version, its unit and similarity."""
    v = sim_by_version.argmax(0)
    q = np.arange(sim_by_version.shape[1])
    return v, unit_by_version[v, q], sim_by_version[v, q]


def timeline(order, block=BLOCK, prelude=PRELUDE, rest=REST):
    """Stream kind and teaching per update range, as (start, end, stream, taught). Both orders begin from the glyph-trained encoder."""
    first, second = ORDERS[order]
    if order == 'GN':
        parts = [(block, 'glyph', 'glyph'), (prelude, 'natural', None), (block, 'natural', 'natural'), (rest, 'natural', None)]
    else:
        parts = [(prelude, 'natural', None), (block, 'natural', 'natural'), (prelude, 'glyph', None), (block, 'glyph', 'glyph'), (rest, 'glyph', None)]
    out, at = [], 0
    for length, stream, taught in parts:
        out.append((at, at + length, stream, taught))
        at += length
    return out


def learn(task):
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    S0.set_up_task('sparse48')
    model = SM.encoder('unlabelled', seed)
    opt = torch.optim.Adam(model.parameters(), NM.LR)
    frozen = copy.deepcopy(model).requires_grad_(False).eval()
    averages = {f'ema-{b}': copy.deepcopy(model).requires_grad_(False).eval() for b in BETAS}
    bg = S0.background_images(seed)
    n_cls, n_r = bg.shape[:2]
    train_g, ev_g, _ = S0.test_material(seed)
    cue_g = np.ascontiguousarray(train_g[:N], dtype=np.float32)
    query_g = np.ascontiguousarray(ev_g[:N].reshape(-1, 28, 28), dtype=np.float32)
    unseen_g = np.ascontiguousarray(ev_g[2 * N:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    mat_g = SIF.load_material(seed)
    tgt_g = mat_g['tgt_eval']
    cand_g = np.ascontiguousarray(tgt_g[:, SIF.VALUE_RENDERS], dtype=np.float32)
    mat_n = NM.memory_material(seed, 'small', 'AB')
    cue_n = mat_n['cue_images'][:N]
    q_n = {v: mat_n['q'][v][:N] for v in NM.VIEWS}
    qu_n = {v: mat_n['qu'][v][:N] for v in NM.VIEWS}
    cand_n = mat_n['candidates']
    tgt_n = mat_n['target_images'][:N]
    train_n = NM.data()['x_train']
    rng = nc.stream('nm_stream', seed, 'mixed', order)
    wrng = nc.stream('nm_mixed_writes', seed, order)
    partner = NM.partners(TK.CLASSES)
    fx = {k: NM.fixed_codes(k, 'small', seed) for k in ('pixels', 'pca')}
    nets = {'working': model, **averages, 'frozen-start': frozen}
    coders = {e: NM.encode_fn(m) for e, m in nets.items()}
    coders.update({k: v[0] for k, v in fx.items()})
    width = {**{e: NM.KEY for e in nets}, **{k: v[1] for k, v in fx.items()}}
    mems = {(e, tau): NM.Tagged(CAPACITY, tau, width[e], width[e], FACTS) for e in coders for tau in (TAUS if e != 'pixels' else (.85,))}
    # version arms: `versions` forks at each taught block, which it is told of; `every-K` forks every K updates, blind to blocks. Each copy holds its own memory. A copy that took no writes is replaced, not kept.
    version_arms = {'versions': [], **{f'every-{k}': [] for k in INTERVALS}}
    held = []
    cache = {}
    plan = {'glyph': NM.plan_writes(wrng, np.arange(N), GLYPH_EXPOSURES, BLOCK), 'natural': NM.plan_writes(wrng, np.arange(N), NATURAL_EXPOSURES, BLOCK)}

    def code(e, x, key=None):
        if e in fx or e == 'frozen-start':
            if key is not None and (e, key) in cache:
                return cache[(e, key)]
            out = NM.unit(coders[e](x))
            if key is not None:
                cache[(e, key)] = out
            return out
        return NM.unit(coders[e](x))

    def write_step(kind, step):
        facts = plan[kind][step]
        if not len(facts):
            return
        if kind == 'glyph':
            cues = np.ascontiguousarray(cue_g[facts, wrng.integers(R, size=len(facts))], dtype=np.float32)
            targets = np.ascontiguousarray(tgt_g[facts, wrng.integers(SIF.VALUE_RENDERS, size=len(facts))], dtype=np.float32)
            rows = facts
        else:
            cues = NM.view_arrays(NM.prepare_input(cue_n[facts], 'small'), wrng, NM.VIEWS['mild'])
            targets = NM.view_arrays(NM.prepare_input(tgt_n[facts], 'small'), wrng, NM.VIEWS['mild'])
            rows = facts + N
        held.append((cues, targets, rows))
        for e in coders:
            z, v = code(e, cues), code(e, targets)
            for tau in (TAUS if e != 'pixels' else (.85,)):
                for i in range(len(rows)):
                    mems[(e, tau)].write_tagged(z[i], v[i], rows[i])
        for versions in version_arms.values():
            ver = versions[-1]
            z, v = NM.unit(ver['code'](cues)), NM.unit(ver['code'](targets))
            for tau in TAUS:
                for i in range(len(rows)):
                    ver['mem'][tau].write_tagged(z[i], v[i], rows[i])

    def score_one(answer, kn_rows, unseen_sim, tags, tau):
        """answer: (unit tag rows, similarity, value picked is right)."""
        tag_ok, sim, val_ok = answer
        out = {'unseen_at_tau': float((unseen_sim >= tau).mean())}
        for name, thr in {'tau': tau, 'answer-all': -2., **{f'matched-{r}': NM.matched_threshold(unseen_sim, r) for r in MATCHED}}.items():
            ok = sim >= thr
            out[name] = {'threshold': float(thr), 'coverage': float(ok.mean()), 'address': float((ok & tag_ok).mean()), 'value': float((ok & val_ok).mean())}
        return out

    def sets():
        """Query sets: (name, images, fact rows, truth in candidates, unseen images, candidates)."""
        out = [('glyph', query_g, np.repeat(np.arange(N), E), np.repeat(np.arange(N), E), unseen_g, cand_g)]
        for v in NM.VIEWS:
            out.append((f'natural-{v}', q_n[v], np.arange(N) + N, np.arange(N), qu_n[v], cand_n))
        return out

    def read_memory(mem, Zq, Zu, Zc, rows, truth, tau):
        h = mem.h
        if not h:
            return None
        i, s = mem.nearest(Zq)
        _, su = mem.nearest(Zu)
        val = mem.u[:h] / mem.n[:h, None]
        pick = (val[i] @ Zc.T).argmax(1)
        return score_one((mem.tags[:h][i].argmax(1) == rows, s, pick == truth), rows, su, mem.tags[:h], tau)

    def read_all():
        out = {}
        for name, q, rows, truth, un, cand in sets():
            for (e, tau), mem in mems.items():
                r = read_memory(mem, code(e, q, name), code(e, un, name + '-unseen'), code(e, cand, name + '-cand'), rows, truth, tau)
                if r is not None:
                    out[f'{e}|{name}|{tau}'] = {**r, 'units': mem.h, 'payload_bytes': mem.payload_bytes()}
            for arm, all_versions in version_arms.items():
                versions = [ver for ver in all_versions if ver['mem'][TAUS[0]].h]
                if not versions:
                    continue
                Zs = [NM.unit(ver['code'](q)) for ver in versions]
                Zus = [NM.unit(ver['code'](un)) for ver in versions]
                Zcs = [NM.unit(ver['code'](cand)) for ver in versions]
                for tau in TAUS:
                    near = [ver['mem'][tau].nearest(Z) for ver, Z in zip(versions, Zs)]
                    v, i, s = pick_answer(np.stack([n[1] for n in near]), np.stack([n[0] for n in near]))
                    su = np.max(np.stack([ver['mem'][tau].nearest(Z)[1] for ver, Z in zip(versions, Zus)]), 0)
                    tag_ok, val_ok = np.zeros(len(q), bool), np.zeros(len(q), bool)
                    for k, ver in enumerate(versions):
                        at = v == k
                        mem = ver['mem'][tau]
                        if at.any() and mem.h:
                            val = mem.u[:mem.h] / mem.n[:mem.h, None]
                            tag_ok[at] = mem.tags[:mem.h][i[at]].argmax(1) == rows[at]
                            val_ok[at] = (val[i[at]] @ Zcs[k].T).argmax(1) == truth[at]
                    out[f'{arm}|{name}|{tau}'] = {**score_one((tag_ok, s, val_ok), rows, su, None, tau), 'units': sum(ver['mem'][tau].h for ver in versions), 'payload_bytes': sum(ver['mem'][tau].payload_bytes() for ver in versions), 'copies': len(versions)}
            # reference: everything rewritten through the working encoder now; it holds images, so it only scores
            if held:
                cues = np.concatenate([x[0] for x in held]); targets = np.concatenate([x[1] for x in held]); r_ = np.concatenate([x[2] for x in held])
                z, val_codes = code('working', cues), code('working', targets)
                mem = NM.Tagged(CAPACITY, .85, NM.KEY, NM.KEY, FACTS)
                for i in range(len(r_)):
                    mem.write_tagged(z[i], val_codes[i], r_[i])
                rec = read_memory(mem, code('working', q), code('working', un), code('working', cand), rows, truth, .85)
                out[f'rebuild-images|{name}|0.85'] = {**rec, 'units': mem.h, 'payload_bytes': mem.payload_bytes()}
        return out

    def fork(versions):
        if versions and not versions[-1]['mem'][TAUS[0]].h:
            versions.pop()
        net = copy.deepcopy(model).requires_grad_(False).eval()
        versions.append({'net': net, 'code': NM.encode_fn(net), 'mem': {tau: NM.Tagged(CAPACITY, tau, NM.KEY, NM.KEY, FACTS) for tau in TAUS}})

    tl = timeline(order)
    reads = {}
    train_wall, write_wall = 0., 0.
    moved_cue = NM.unit(coders['working'](query_g[:225]))
    for lo, hi, stream, taught in tl:
        if taught:
            fork(version_arms['versions'])
        for u in range(lo, hi):
            for k in INTERVALS:
                if u % k == 0:
                    fork(version_arms[f'every-{k}'])
            t = time.perf_counter()
            if stream == 'natural':
                base = torch.from_numpy(NM.prepare_input(train_n[rng.integers(len(train_n), size=TK.CLASSES)], 'small'))
                x = torch.stack([NM.stream_views(base, rng, 'strong'), NM.stream_views(base, rng, 'strong')], 1)
            else:
                base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=TK.CLASSES), rng.integers(n_r, size=TK.CLASSES)], dtype=np.float32))
                x = torch.stack([SRL.augment_with(base, rng, SRL.AUGMENTS['full']), SRL.augment_with(base, rng, SRL.AUGMENTS['full'])], 1)
            x = x.reshape(-1, 28, 28)
            z = model(x)
            loss = SRL.info_nce(z @ z.T / NM.T, partner)
            opt.zero_grad(); loss.backward(); opt.step()
            for beta, avg in zip(BETAS, averages.values()):
                NM.update_average(avg, model, beta)
            train_wall += time.perf_counter() - t
            t = time.perf_counter()
            if taught and (u - lo) % NM.WRITE_EVERY == NM.WRITE_EVERY - 1:
                with torch.no_grad():
                    write_step(taught, (u - lo) // NM.WRITE_EVERY)
            write_wall += time.perf_counter() - t
        with torch.no_grad():
            reads[f'{hi}:{stream}:{taught}'] = read_all()
    res = {'reads': reads, 'seed': seed, 'order': order, 'timeline': tl, 'train_wall_seconds': train_wall, 'write_wall_seconds': write_wall}
    return f'{seed}/{order}', res, S0.timing(t0, c0)


def main():
    t0 = time.perf_counter()
    S0.prepare()
    SIF.material()
    NM.data()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done', '{0}-{1}'), [(s, o) for s in SEEDS for o in ORDERS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; glyph and natural facts in one memory, the stream changing between them', 'betas': BETAS, 'taus': TAUS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder, averages, memory writes and reads on both kinds', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
