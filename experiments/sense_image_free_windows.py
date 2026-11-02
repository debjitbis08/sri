"""Short synchronized windows, no image held, and transport separated from acceptance (docs/sense-image-free-v1-pilot.md, Windows). Follows external review of the first image-free run.

Every stored code is in one version. Memory reads and writes use a published copy of each network while the working copy trains. At a window's end, inputs passing now go through the published copy and a candidate of the working copy; only running totals are kept, never an image. Then a map moves all stored codes, and the candidate is published.

Thresholds are separate from transport: fixed; widened by estimated drift; or background, each unit's running high quantile of its similarity to inputs memory is not being taught. Reads are also given at matched false acceptance, by shifting all thresholds so a stated share of unseen queries is answered; that calibration uses the unseen queries and only scores.

Stages (argv[2]):
  hosts     GPU: sparse48 through blocks 4-6 in both orders, weights every window
  learning  GPU: key encoder, memory arms on both sets of values

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_image_free_windows pilot learning
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import limited_memory as LM
from experiments import sense_consolidation as SC
from experiments import sense_consolidation_moving as M
from experiments import sense_consolidation_traces as T
from experiments import sense_image_free as F
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import shared_core_unseen_traces as UT
from experiments import translation_stable_keys as TK

nc, P, D, SP, AP = S0.nc, S0.P, S0.D, S0.SP, M.AP
NAME = 'sense-image-free-v1-windows' + os.environ.get('SI_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
N, E, R = F.N, F.E, F.R
BLOCK, BATCH = F.BLOCK, F.BATCH
WINDOW = int(os.environ.get('SI_WINDOW', 250))
COLLECT = int(os.environ.get('SI_COLLECT', 16))  # updates at a window's end during which passing inputs fit the map
HOST_PROBES = 2048
QUANTILE, QUANTILE_STEP, BG_PER_UPDATE = .999, .02, 16
Q_LOW, Q_HIGH = F.FLOOR, .98
TARGETS = (.05, .13)
LAMBDA = F.LAMBDA


class RunningMap:
    """fit_map from running totals of code pairs; no pair or image is kept. Holds (d + 1)(d + 1 + d) numbers."""
    def __init__(self, d, lam=LAMBDA):
        self.d, self.lam, self.n, self.yy = d, lam, 0, 0.
        self.xx, self.xy = np.zeros((d + 1, d + 1)), np.zeros((d + 1, d))

    def add(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        xb = np.c_[x, np.ones(len(x))]
        self.xx += xb.T @ xb
        self.xy += xb.T @ y
        self.yy += float((y ** 2).sum())
        self.n += len(x)

    def fn(self):
        d, n = self.d, self.n
        s = max(self.yy / (n * d), 1e-12)
        I0 = np.eye(d + 1)[:, :d]
        A = np.linalg.solve(self.xx / (n * s) + self.lam / d * np.eye(d + 1), self.xy / (n * s) + self.lam / d * I0)
        return lambda Z: (np.c_[np.asarray(Z, np.float64), np.ones(len(Z))] @ A).astype(np.float32)


class Drift:
    """The map, and how far inputs moved, with and without a map fitted on earlier arrivals. Angles are single numbers; the pairs go."""
    def __init__(self, d):
        self.all, self.early, self.angles, self.left = RunningMap(d), RunningMap(d), [], []

    def add(self, old, new, half):
        self.all.add(old, new)
        if half:
            fn = self.early.fn()
            self.angles += np.arccos(np.clip((S1.unit(old) * S1.unit(new)).sum(1), -1., 1.)).tolist()
            self.left += np.arccos(np.clip((S1.unit(fn(old)) * S1.unit(new)).sum(1), -1., 1.)).tolist()
        else:
            self.early.add(old, new)

    def result(self):
        return {'map': self.all.fn(), 'angle': float(np.median(self.angles)), 'residual': float(np.median(self.left))}


class Store(F.Store):
    """Adds the background threshold: a running high quantile, by the Robbins-Monro rule, of each unit's similarity to inputs memory is not being taught."""
    def __init__(self, H, k, tau, d, m, background=False):
        super().__init__(H, k, tau, d, m)
        self.background = background
        self.q = np.full(H, tau, np.float32)

    def observe(self, Z):
        h = self.h
        if not h:
            return
        over = (Z @ self.a[:h].T > self.q[:h]).sum(0)
        self.q[:h] += QUANTILE_STEP * (QUANTILE * over - (1 - QUANTILE) * (len(Z) - over))
        self.q[:h] = np.clip(self.q[:h], Q_LOW, Q_HIGH)
        self.thr[:h] = self.q[:h]

    def to_balls(self):
        """A merge covers members' balls, whichever rule set them."""
        if self.background:
            self.theta[:self.h] = np.arccos(self.thr[:self.h])

    def from_balls(self):
        if self.background:
            self.q[:] = self.tau
            self.q[:self.h] = np.clip(np.cos(self.theta[:self.h]), Q_LOW, Q_HIGH)
            self.thr[:] = self.tau
            self.thr[:self.h] = self.q[:self.h]


def matched(mem, unseen, target):
    """The shift to all thresholds at which this share of unseen queries is answered. Scoring only."""
    h = mem.h
    return float(np.quantile((unseen @ mem.a[:h].T - mem.thr[:h]).max(1), 1 - target)) if h else 0.


class Arm(F.Arm):
    """thresholds: fixed, widen, background. transport: move addresses by the live map. values: stale, live, oracle."""
    def __init__(self, thresholds, transport, values, merge=None, H=2048, m=S1.CORE, frozen=False):
        self.keys, self.values, self.merge, self.frozen = 'frozen' if frozen else 'x', values, merge, frozen
        self.thresholds, self.transport = thresholds, transport
        self.mem = Store(H, M.LS.K, F.TAU, 32, m, thresholds == 'background')
        self.symbols = np.zeros((H, 3 * N), np.int64)
        self.events = []

    def publish(self, drift, maps):
        if self.frozen is False:
            if self.transport:
                self.mem.move_addresses(drift['map'])
            if self.thresholds == 'widen':
                self.mem.widen(drift['residual'] if self.transport else drift['angle'])
        if self.values != 'stale':
            self.mem.move_values(maps[self.values])

    def block_end(self, score):
        event = {'units_before': self.mem.h}
        if self.merge is not None:
            self.mem.to_balls()
            event.update(self.consolidate(score))
            self.mem.from_balls()
        event['units'] = self.mem.h
        self.events.append(event)

    def allocated(self):
        m = self.mem
        return sum(x.nbytes for x in (m.c, m.a, m.u, m.n, m.thr, m.theta, m.q))


def arms():
    return {'fixed': dict(thresholds='fixed', transport=False, values='live'), 'widen': dict(thresholds='widen', transport=False, values='live'), 'background': dict(thresholds='background', transport=False, values='live'),
            'transport-fixed': dict(thresholds='fixed', transport=True, values='live'), 'transport-widen': dict(thresholds='widen', transport=True, values='live'), 'transport-background': dict(thresholds='background', transport=True, values='live'),
            'transport-fixed-stale': dict(thresholds='fixed', transport=True, values='stale'), 'transport-fixed-oracle': dict(thresholds='fixed', transport=True, values='oracle'), 'frozen': dict(thresholds='fixed', transport=False, values='live', frozen=True),
            'transport-fixed-merge': dict(thresholds='fixed', transport=True, values='live', merge=.8), 'transport-background-merge': dict(thresholds='background', transport=True, values='live', merge=.8),
            'transport-fixed-H512': dict(thresholds='fixed', transport=True, values='live', H=512), 'transport-fixed-merge-H512': dict(thresholds='fixed', transport=True, values='live', merge=.8, H=512)}


# ---------------------------------------------------------------- hosts

def host_path(seed, order, at):
    return RUNS / 'hosts' / f'seed{seed}-{order}-{at}.pt'


def fit_host(task):
    seed, order = task
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
    first, second = SC.ORDERS[order]
    hashes, at = {}, 0
    for b, tag in ((4, first), (5, second), (6, None)):
        crng = nc.stream('ap_context', seed, b)
        prng = nc.stream('ap_pairs', seed, {'A': 4, 'B': 5}[tag], tag) if tag else None
        for _ in range(F.AP_L):
            items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
            if tag:
                AP.pair_items(items, match, sets[tag], row, prng)
            LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True)
            at += 1
            if at % WINDOW == 0:
                LM.save_atomic(LM.weights(enc, model, withheld, 'sparse48', seed), host_path(seed, order, at))
        hashes[str(b)] = S0.state_hash(enc, model)
    old = json.load(open(S0.X.ROOT / 'reports' / ('shared-core-sparse-memory-v1-traces-hosts.json' if order == 'AB' else 'sense-consolidation-v1-moving-hosts.json')))[f'seed{seed}']['hashes']
    return f'seed{seed}/{order}', {'hashes': hashes, 'reproduces_block_hosts': {b: hashes[b] == old[b] for b in hashes}, 'updates': 3 * D.STEPS + 3 * F.AP_L}, S0.timing(t0, c0)


def hosts():
    t0 = time.perf_counter()
    S0.prepare()
    AP.POOL.update(S0.PS.survivors())
    res = S0.pool_map(DUR.durable(fit_host, RUNS / 'done-hosts', 'seed{0}-{1}'), [(s, o) for s in S0.SEEDS for o in SC.ORDERS])
    S0.save_json(f'{REPORT}-hosts.json', {k: r for k, r, _ in res})
    S0.save_json(f'{REPORT}-hosts-runtime.json', S0.runtime({'phase': 'hosts with weights every window, 12,000 updates each', 'wall_seconds': time.perf_counter() - t0, 'fits': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))
    print({k: r['reproduces_block_hosts'] for k, r, _ in res}, flush=True)


class HostCopy:
    def __init__(self, seed):
        self.enc, self.model, _ = LM.setup('sparse48', seed)
        for m in (*self.enc.values(), self.model):
            m.requires_grad_(False)

    def load(self, path):
        ck = torch.load(path, weights_only=False, map_location='cpu')
        self.model.load_state_dict(ck['sri']); self.enc['image'].load_state_dict(ck['image']); self.enc['sequence'].load_state_dict(ck['sequence'])
        return self

    def __call__(self, x):
        return S0.batched(lambda y: self.model.proj['image'](self.enc['image'](y)), x).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- reads

def read_all(vals, mem, Zq, Zf, mask, view, tags):
    out = F.read_arm(vals, mem, Zq, Zf, mask, view, tags)
    unseen = Zq[2 * N * E:3 * N * E]
    keep = mem.thr.copy()
    out['matched'] = {}
    for target in TARGETS:
        mem.thr[:mem.h] = keep[:mem.h] + matched(mem, unseen, target)
        out['matched'][str(target)] = F.read_arm(vals, mem, Zq, Zf, mask, view, tags)
        mem.thr[:] = keep
    return out


def learn(task):
    seed, lr, order, stream_kind = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    M.LS.TAU = F.TAU
    c = S1.load(seed)
    mat = F.load_material(seed)
    model = SM.encoder('unlabelled', seed)
    start = SM.encoder('unlabelled', seed).requires_grad_(False)
    published = copy.deepcopy(model).requires_grad_(False).eval()
    candidate = None
    opt = torch.optim.Adam(model.parameters(), lr)
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    fonts = mat['other_font'].reshape(-1, 28, 28)
    rng = nc.stream('sm_learning', seed, str(lr))
    prng = nc.stream('siw_probes', seed, str(lr), order, stream_kind)
    streams = {t: S1.stream(t, seed, c) for t in 'AB'}
    first, second = SC.ORDERS[order]
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    unseen_images = UT.source_images(seed)['new-radius']
    shift_pool = np.concatenate([P.DOTS['train'].reshape(-1, 28, 28), *[S0.X.DATA[(f, 'eval')][0] for f in UT.FORMATS[:3]]]).astype(np.float32) if stream_kind == 'shift' else None
    values = {d: F.Values(d, seed, order, c, mat, unseen_images) for d in ('counts', 'symbols')}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    sense = lambda m: (lambda x: np_(enc(m, x)))
    with torch.no_grad():
        start_cues, start_queries, start_fonts = np_(enc(start, cue)), np_(enc(start, query)), np_(enc(start, fonts))
    Zqs, Zfs = S1.unit(start_queries), S1.unit(start_fonts)
    free = {d: {k: Arm(**v, m=F.VALUE_WIDTH[d]) for k, v in arms().items()} for d in values}
    specs = M.arms()
    imaged = {d: {k: T.Arm(**specs[k]) for k in F.IMAGE_ARMS} for d in values}
    for d in imaged:
        for k, arm in imaged[d].items():
            arm.mem = SC.RadiusMemory(specs[k]['H'], M.LS.K, F.TAU, 32, F.VALUE_WIDTH[d])
    background_arms = [(d, a) for d in free for a in free[d].values() if a.thresholds == 'background']
    first_coll = np.full(len(cue), -1, np.int64)
    reads, drifts = {}, {}
    host_now, host_next = HostCopy(seed).load(S0.checkpoint_path('sparse48', seed)), HostCopy(seed)
    write_table = values['counts'].table(host_now)
    drift, at = None, 0

    for b, tag in ((4, first), (5, second), (6, None)):
        s = streams[tag] if tag else None
        window_drift = []
        for u in range(BLOCK):
            if stream_kind == 'shift' and b >= 5:
                base = torch.from_numpy(np.ascontiguousarray(shift_pool[rng.integers(len(shift_pool), size=TK.CLASSES)], dtype=np.float32))
            else:
                base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=TK.CLASSES), rng.integers(n_r, size=TK.CLASSES)], dtype=np.float32))
            x = torch.stack([SRL.augment_with(base, rng, SRL.AUGMENTS['full']), SRL.augment_with(base, rng, SRL.AUGMENTS['full'])], 1).reshape(-1, 28, 28)
            model.train()
            z = model(x)
            loss = SRL.info_nce(z @ z.T / SRL.T, partner)
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            at += 1
            passing = [base.numpy()]
            if background_arms:
                Zb = S1.unit(np_(enc(published, base[:BG_PER_UPDATE].numpy())))
                for _, arm in background_arms:
                    arm.mem.observe(Zb)
            if s is not None:
                part = s[u * BATCH:(u + 1) * BATCH]
                flat = part[:, 0] * R + part[:, 1]
                images = np.ascontiguousarray(mat['tgt_eval'][part[:, 0], part[:, 2] % F.VALUE_RENDERS], dtype=np.float32)
                passing += [cue[flat], images]
                with torch.no_grad():
                    live = np_(model(torch.from_numpy(cue[flat])))
                    cue_codes, value_codes = np_(enc(published, cue[flat])), np_(enc(published, images))
                for i in range(len(part)):
                    sym, col = int(part[i, 0]), int(part[i, 2])
                    if first_coll[flat[i]] < 0:
                        first_coll[flat[i]] = col
                    for d, vals in values.items():
                        v = write_table[col % len(write_table), vals.fact_row[sym]] if d == 'counts' else value_codes[i]
                        for arm in free[d].values():
                            arm.write(start_cues[flat[i]] if arm.frozen else cue_codes[i], v, sym)
                        for arm in imaged[d].values():
                            arm.write(live[i], np.zeros(F.VALUE_WIDTH[d], np.float32), flat[i])
            if at % WINDOW == WINDOW - COLLECT:
                candidate = copy.deepcopy(model).requires_grad_(False).eval()
                drift, updates = Drift(32), 0
            if candidate is not None:
                images = np.ascontiguousarray(np.concatenate(passing), dtype=np.float32)
                drift.add(np_(enc(published, images)), np_(enc(candidate, images)), updates >= COLLECT // 2)
                updates += 1
            del passing
            if at % WINDOW == 0:
                key = drift.result()
                window_drift.append((key['angle'], key['residual']))
                host_next.load(host_path(seed, order, at))
                hmap = RunningMap(S1.CORE)
                for _ in range(4):
                    images = F.live_images(cue, tag, prng)[:HOST_PROBES // 4]
                    hmap.add(host_now(images), host_next(images))
                new_table = values['counts'].table(host_next)
                old_symbol, new_symbol = values['symbols'].table(sense(published)), values['symbols'].table(sense(candidate))
                maps = {'counts': {'live': hmap.fn(), 'oracle': F.fit_map(write_table.reshape(-1, S1.CORE), new_table.reshape(-1, S1.CORE))},
                        'symbols': {'live': key['map'], 'oracle': F.fit_map(old_symbol.reshape(-1, 32), new_symbol.reshape(-1, 32))}}
                for d in free:
                    for arm in free[d].values():
                        arm.publish(key, maps[d])
                published, candidate = candidate, None
                host_now, host_next = host_next, host_now
                write_table = new_table
        with torch.no_grad():
            table_cue, qw, fw = np_(enc(model, cue)), np_(enc(model, query)), np_(enc(model, fonts))
            qp, fp = np_(enc(published, query)), np_(enc(published, fonts))
        drifts[str(b)] = {'angle': float(np.sum([a for a, _ in window_drift])), 'residual': float(np.sum([r for _, r in window_drift])), 'windows': window_drift}
        Zq, Zf, Zqw, Zfw = S1.unit(qp), S1.unit(fp), S1.unit(qw), S1.unit(fw)
        tags = first if b == 4 else 'AB'
        reads[str(b)] = {}
        for d, vals in values.items():
            code = host_now if d == 'counts' else sense(published)
            view = vals.view(code, b)
            rec = {'ceiling': {t: vals.ceiling(vals.table(code), view, t) for t in tags}}
            for name, arm in free[d].items():
                arm.block_end(vals.score)
                frozen = arm.frozen
                rec[name] = {**read_all(vals, arm.mem, Zqs if frozen else Zq, Zfs if frozen else Zf, mat['font_mask'], view, tags), 'units': arm.mem.h, 'forced': arm.mem.forced, 'payload_bytes': arm.bytes(), 'allocated_bytes': arm.allocated()}
                if arm.thresholds == 'background':
                    rec[name]['threshold_quantiles'] = [float(v) for v in np.quantile(arm.mem.thr[:arm.mem.h], (.05, .5, .95))]
            code_w = host_now if d == 'counts' else sense(model)
            view_w = vals.view(code_w, b)
            fresh = vals.table(code_w)
            for name, arm in imaged[d].items():
                h = arm.mem.h
                arm.mem.u[:h] = M.unit_values(arm.tu, arm.tr, h, first_coll, vals.row_counts, fresh) * arm.mem.n[:h, None]
                arm.tick(table_cue, vals.score)
                h = arm.mem.h
                arm.mem.u[:h] = M.unit_values(arm.tu, arm.tr, h, first_coll, vals.row_counts, fresh) * arm.mem.n[:h, None]
                rec[f'images-{name}'] = {**read_all(vals, arm.mem, Zqw, Zfw, mat['font_mask'], view_w, tags), 'units': h, 'forced': arm.mem.forced, 'payload_bytes': 4 * h * (32 + 2) + len(arm.tu) * M.BYTES_PER_EXPERIENCE}
            reads[str(b)][d] = rec
    res = {'reads': reads, 'key_drift': drifts, 'events': {d: {k: a.events for k, a in free[d].items()} for d in free}, 'seed': seed, 'lr': lr, 'order': order, 'stream': stream_kind}
    return f'{seed}/{lr}/{order}/{stream_kind}', res, S0.timing(t0, c0)


def learning():
    t0 = time.perf_counter()
    S0.prepare()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done', '{0}-{1}-{2}-{3}'), [(s, lr, o, k) for s in S0.SEEDS for lr in SC.LRS for o in SC.ORDERS for k in F.STREAMS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; published copies keep every stored code in one version; running totals only, no image held by any arm but the images reference; thresholds separate from transport; matched false acceptance only scores', 'window': WINDOW, 'collect': COLLECT, 'arms': arms(), 'image_arms': F.IMAGE_ARMS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'key encoder with windows and memory writes on both sets of values', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'hosts': hosts, 'learning': learning}[sys.argv[2]]()
