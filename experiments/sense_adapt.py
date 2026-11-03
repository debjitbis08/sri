"""Sense integration plan, steps 3 to 5 (docs/sense-integration-plan.md): no labels anywhere, new kinds after setup, stability as sight adapts.

Two frozen sight models through src.senses (DINOv2-S and the TrOCR-small encoder), joined keys, on step 1's facts, draws and memory (experiments.general_encoder).
Arms:
  told/advance        domain centres told the domain, gate learned in advance on all kinds: experiments.general_encoder `pilot mix` joined/learned (must read exactly as there)
  found/advance       two centres found with no labels (2-means on DINOv2 codes) in advance, gate learned in advance           -- step 3
  found/online        centres opened online by a batch-shift rule from passing inputs; gate refitted from stored codes as centres open -- steps 4 and 5
  found/online-equal  the same centres, equal weights: what the gate adds
  dinov2/online       DINOv2 alone on the same centres: what TrOCR adds
Every taught kind is read after every later kind.

Online rule, set before running: before a kind's facts, PASSING inputs pass in batches of BATCH. A batch joins the nearest frozen centre when the cosine of its mean
DINOv2 code with that centre reaches the centre's threshold; otherwise it joins the centre being formed, which freezes after FORM inputs. A centre's threshold is the
QUANTILE of the cosines between BOOTS bootstrapped batch means of its members (batches of BATCH, drawn with replacement) and as many bootstrapped centres
(all members, drawn with replacement), so that the centre's own estimation error counts. Each input is coded on its nearest frozen centre.
When a centre freezes, its forming inputs give two views each (mild, strong); only their codes are kept, and the gate is refitted from scratch on all kept codes.
Assumes a new kind arrives for a while at a time, as in these streams. Labels only score.

Usage: NFI_DEVICE=cuda SC0_WORKERS=2 OPENBLAS_NUM_THREADS=1 /data/venvs/sri-cuda/bin/python -m experiments.sense_adapt pilot
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
from experiments import shared_core_stage0 as S0
from src import senses as SN

nc = S0.nc
NAME = 'sense-adapt-v1' + os.environ.get('SA_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
SEEDS = GE.SEEDS
TAU = GE.TAUS[0]
N = GE.N
MODELS = ('dinov2-s', 'trocr-s')
PASSING, BATCH, FORM, QUANTILE, BOOTS = 1024, 256, 512, .01, 200
ARMS = ('told/advance', 'found/advance', 'found/online', 'found/online-equal', 'dinov2/online')


def coded(raw, C, a):
    """Raw codes, each centred on its own centre: C (k, d) centres, a (n,) assignments."""
    return SN.centred(raw, C[a])


class ShiftCentres:
    """Centres opened by the batch-shift rule. Routing and the shift test use DINOv2's raw codes; each model keeps its own mean over the same members."""
    def __init__(self, rng, route='dinov2-s'):
        self.rng, self.route = rng, route
        self.centres, self.thresholds, self.forming = [], [], []
        self.log = []

    def routing(self):
        return SN.unit(np.stack([c[self.route] for c in self.centres]))

    def observe(self, batch, tag=None):
        """batch: {model: raw codes (BATCH, d)}. Returns the frozen centre it joined, or None while a centre forms; and whether a centre froze."""
        m = SN.unit(batch[self.route].mean(0))
        if self.centres:
            s = self.routing() @ m
            j = int(s.argmax())
            if s[j] >= self.thresholds[j]:
                self.log.append({'tag': tag, 'joined': j, 'cos': float(s[j]), 'threshold': self.thresholds[j]})
                return j, False
        self.forming.append(batch)
        self.log.append({'tag': tag, 'joined': None, 'cos': float((self.routing() @ m).max()) if self.centres else None, 'forming': sum(len(b[self.route]) for b in self.forming)})
        if sum(len(b[self.route]) for b in self.forming) < FORM:
            return None, False
        members = {n: np.concatenate([b[n] for b in self.forming]) for n in batch}
        c = {n: z.mean(0) for n, z in members.items()}
        r = members[self.route]
        idx, cdx = self.rng.integers(0, len(r), size=(BOOTS, BATCH)), self.rng.integers(0, len(r), size=(BOOTS, len(r)))
        boot = (SN.unit(r[idx].mean(1)) * SN.unit(r[cdx].mean(1))).sum(1)  # a fresh batch against a centre that is itself estimated from the members
        self.centres.append(c)
        self.thresholds.append(float(np.quantile(boot, QUANTILE)))
        self.forming = []
        return len(self.centres) - 1, True

    def assign(self, codes):
        return (SN.unit(codes[self.route]) @ self.routing().T).argmax(1)

    def centre_array(self, n):
        return np.stack([c[n] for c in self.centres])


def adapt_task(task):
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

    # ---- in advance, as in `pilot mix`: centre samples of every kind, then a gate on two views of every kind (same streams, so told/advance reads as there)
    crng = nc.stream('ge_centres', seed)
    sample = lambda pool: np.ascontiguousarray(pool[crng.permutation(len(pool))[:GE.CENTRE_IMAGES]], dtype=np.float32)
    samples = {k: sense(sample(data[k]['pool'])) for k in kinds}
    ck = {n: {k: samples[k][n].mean(0) for k in kinds} for n in E}
    cd = {n: {'natural': ck[n]['natural'], 'rendered': np.mean([ck[n][k] for k in kinds if k != 'natural'], 0)} for n in E}
    pooled = {n: np.concatenate([samples[k][n] for k in kinds]) for n in E}
    member_kind = np.concatenate([[k] * len(samples[k]['dinov2-s']) for k in kinds])
    CL, a = GE.two_means(NM.unit(pooled['dinov2-s']))
    cc = {n: np.stack([pooled[n][a == j].mean(0) for j in range(2)]) for n in E}
    cluster_of = lambda f: (NM.unit(f['dinov2-s']) @ CL.T).argmax(1)
    advance_info = {'clusters': [{k: int(((a == j) & (member_kind == k)).sum()) for k in kinds} for j in range(2)]}
    del samples, pooled

    grng = nc.stream('ge_mix_gate', seed)
    g_imgs = {k: np.ascontiguousarray(data[k]['pool'][grng.permutation(len(data[k]['pool']))[:GE.GATE_IMAGES]], dtype=np.float32) for k in kinds}
    v1 = {k: sense(NM.view_arrays(g_imgs[k], grng, NM.VIEWS['mild'])) for k in kinds}
    v2 = {k: sense(NM.view_arrays(g_imgs[k], grng, NM.VIEWS['strong'])) for k in kinds}
    cat = lambda f: np.concatenate([f(k) for k in kinds])
    raw1 = [cat(lambda k: NM.unit(v1[k][n])) for n in E]
    told_views = [np.stack([cat(lambda k: GE.centred(v1[k][n], cd[n][GE.domain_of(k)])), cat(lambda k: GE.centred(v2[k][n], cd[n][GE.domain_of(k)]))]) for n in E]
    gate_told, loss = SN.Gate.train(raw1, told_views, seed, GE.GATE_STEPS, GE.GATE_BATCH, GE.GATE_LR, GE.GATE_TEMP)
    advance_info['told_gate_loss'] = loss
    found_views = [np.stack([cat(lambda k: coded(v1[k][n], cc[n], cluster_of(v1[k]))), cat(lambda k: coded(v2[k][n], cc[n], cluster_of(v2[k])))]) for n in E]
    gate_found, loss = SN.Gate.train(raw1, found_views, seed, GE.GATE_STEPS, GE.GATE_BATCH, GE.GATE_LR, GE.GATE_TEMP)
    advance_info['found_gate_loss'] = loss
    del v1, v2, g_imgs, raw1, told_views, found_views

    # ---- online state
    online = ShiftCentres(nc.stream('ge_adapt_boot', seed, order))
    prng, vrng = nc.stream('ge_passing', seed, order), nc.stream('ge_adapt_views', seed, order)
    kept = []  # per frozen centre: raw codes of two views of its forming inputs, {model: (2, n, d)}; no images
    gate_online = {'fn': None, 'log': []}

    def refit_gate():
        raw = [np.concatenate([NM.unit(k[n][0]) for k in kept]) for n in E]
        views = []
        for n in E:
            C = online.centre_array(n)
            views.append(np.stack([np.concatenate([coded(k[n][i], C, online.assign({'dinov2-s': k['dinov2-s'][i]})) for k in kept]) for i in range(2)]))
        g, loss = SN.Gate.train(raw, views, seed, GE.GATE_STEPS, GE.GATE_BATCH, GE.GATE_LR, GE.GATE_TEMP)
        gate_online['fn'] = g
        gate_online['log'].append({'centres': len(online.centres), 'loss': loss, 'codes': int(len(raw[0]))})

    # ---- how each arm codes a set of raw codes, and weights its queries
    def parts(arm, kind, f):
        if arm == 'told/advance':
            return [GE.centred(f[n], cd[n][GE.domain_of(kind)]) for n in E]
        if arm == 'found/advance':
            cl = cluster_of(f)
            return [coded(f[n], cc[n], cl) for n in E]
        a = online.assign(f)
        if arm == 'dinov2/online':
            return [coded(f['dinov2-s'], online.centre_array('dinov2-s'), a)]
        return [coded(f[n], online.centre_array(n), a) for n in E]

    def weights(arm, f):
        g = {'told/advance': gate_told, 'found/advance': gate_found, 'found/online': gate_online['fn']}.get(arm)
        if g is None:
            return None
        w = g([NM.unit(f[n]) for n in E])
        return [w[:, i] for i in range(len(E))]

    facts = len(SEQ.KINDS) * N
    D = SN.get('dinov2-s').dim
    mems = {arm: NM.Tagged(SEQ.CAPACITY, TAU, D if arm == 'dinov2/online' else 2 * D, D if arm == 'dinov2/online' else 2 * D, facts) for arm in ARMS}
    labels = {k: GE.kind_labels(k, seed) for k in kinds}
    cue_labels = np.stack([labels[k][0] for k in kinds])
    truth = np.arange(N)
    cache, reads, stages, taught, forming_imgs = {}, {}, [], [], []
    write_wall = 0.

    for kind in kinds:
        # passing inputs, before the kind's facts; discarded after
        passing = np.ascontiguousarray(data[kind]['pool'][prng.permutation(len(data[kind]['pool']))[:PASSING]], dtype=np.float32)
        froze = []
        for b in range(0, PASSING, BATCH):
            imgs = passing[b:b + BATCH]
            j, new = online.observe(sense(imgs), tag=kind)
            if j is None or new:
                forming_imgs.append(imgs)
            if new:  # two views of the inputs that formed the centre; only their codes are kept
                x = np.concatenate(forming_imgs)
                f1, f2 = sense(NM.view_arrays(x, vrng, NM.VIEWS['mild'])), sense(NM.view_arrays(x, vrng, NM.VIEWS['strong']))
                kept.append({n: np.stack([f1[n], f2[n]]) for n in E})
                forming_imgs = []
                froze.append(j)
                refit_gate()
        del passing
        if not online.centres:
            raise RuntimeError('no centre frozen before the first facts')
        stages.append({'kind': kind, 'centres': len(online.centres), 'froze': froze, 'thresholds': list(online.thresholds)})

        t = time.perf_counter()
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
        for arm in ARMS:
            z, v = SN.join(parts(arm, kind, fc)), SN.join(parts(arm, kind, ft))
            for i in range(len(rows)):
                mems[arm].write_tagged(z[i], v[i], rows[i])
        write_wall += time.perf_counter() - t
        taught.append(kind)

        # read every taught kind
        stage = {}
        for k in taught:
            if k not in cache:
                cache[k] = {'cand': sense(data[k]['targets']), **{f'q-{v}': sense(q[k][v]) for v in NM.VIEWS}, **{f'u-{v}': sense(qu[k][v]) for v in NM.VIEWS}}
            rows_k = SEQ.fact_row(index[k], truth)
            for view in NM.VIEWS:
                fq, fu, fcand = cache[k][f'q-{view}'], cache[k][f'u-{view}'], cache[k]['cand']
                for arm in ARMS:
                    mem = mems[arm]
                    wq, wu = weights(arm, fq), weights(arm, fu)
                    i, s = mem.nearest(SN.join(parts(arm, k, fq), wq))
                    iu, su = mem.nearest(SN.join(parts(arm, k, fu), wu))
                    h = mem.h
                    val = (mem.u[:h] / mem.n[:h, None])[i]
                    Zc = SN.join(parts(arm, k, fcand))
                    rec = {**GE.score(mem.tags[:h][i].argmax(1) == rows_k, s, (val @ Zc.T).argmax(1) == truth, su, TAU), 'units': h,
                           'resemblance': GE.resemblance(mem.tags[:h][iu].argmax(1), su, TAU, index[k], labels[k][1], cue_labels)}
                    if wq is not None:
                        rec['trocr_weight'] = float(np.mean(wq[1]))
                    if arm.endswith('online'):
                        rec['centre_of_queries'] = np.bincount(online.assign(fq), minlength=len(online.centres)).tolist()
                    stage[f'{arm}|{k}-{view}'] = rec
        reads[f'{len(taught)}:{kind}'] = stage

    res = {'reads': reads, 'kinds': kinds, 'seed': seed, 'order': order, 'stages': stages, 'shift_log': online.log, 'gate_refits': gate_online['log'], 'advance': advance_info, 'write_wall_seconds': write_wall}
    return f'{seed}/{order}', res, {**S0.timing(t0, c0), 'senses': sense.describe()}


def pilot():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(adapt_task, RUNS / 'done-pilot', '{0}-{1}'), [(s, o) for s in SEEDS for o in SEQ.ORDERS])
    S0.save_json(f'{REPORT}-pilot.json', {'status': 'exploratory, pilot seeds, both orders; frozen DINOv2-S and TrOCR-small through src.senses; centres and gate with no labels, in advance and online; every taught kind read after every later kind',
                                          'arms': ARMS, 'tau': TAU, 'rule': {'passing': PASSING, 'batch': BATCH, 'form': FORM, 'quantile': QUANTILE, 'boots': BOOTS},
                                          'gate': {'steps': GE.GATE_STEPS, 'batch': GE.GATE_BATCH, 'lr': GE.GATE_LR, 'temperature': GE.GATE_TEMP}, 'licences': SN.licences(),
                                          'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-pilot-runtime.json', S0.runtime({'phase': 'two frozen sight models, centres and gates in advance and online, writes and repeated reads', 'wall_seconds': time.perf_counter() - t0,
                                                             'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'pilot': pilot}[sys.argv[1]]()
