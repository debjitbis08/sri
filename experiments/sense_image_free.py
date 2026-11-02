"""Memory with no stored images, with keys, values and host all learning (docs/sense-image-free-v1-pilot.md). A unit holds an address, a value code, a count and an angular radius. Old codes are kept current, if at all, by maps fitted on inputs passing now, which are then discarded; each map needs the networks as they were at the block's start, and those copies are counted. Stored images remain only as a reference arm. True symbols and counts only score merges and reads.

Two sets of values:
  counts   the earlier task: 90 facts share nine counts, host codes, read among nine candidates; counts recur in the host's stream
  symbols  each fact has its own target symbol, read among 150, in the code of the key encoder as it learns, since the host's code does not identify symbols (the hosts stage measures that); a target is seen only while its own set is taught

Stages (argv[2]):
  material  renders targets and other-font queries
  hosts     GPU: sparse48 taught the symbol pairs, in both orders: the reason symbol values do not go through the host
  learning  GPU: key encoder on a stationary or shifting stream, memory arms on both sets of values

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_image_free pilot learning
"""
import copy
import io
import os
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import sense_consolidation as SC
from experiments import sense_consolidation_evidence as EV
from experiments import sense_consolidation_moving as M
from experiments import sense_consolidation_traces as T
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import shared_core_unseen_traces as UT
from experiments import shared_core_value as V
from experiments import translation_memory as TM
from experiments import translation_stable_keys as TK

nc, P, D, SP, PS = S0.nc, S0.P, S0.D, S0.SP, S0.PS
NAME = 'sense-image-free-v1' + os.environ.get('SI_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
N, E, R = M.N, M.E, M.R
BLOCK = int(os.environ.get('SI_BLOCK', M.BLOCK))
BATCH = M.BATCH
AP_L = int(os.environ.get('SI_HOST_BLOCK', M.AP.L))
TAU, FLOOR, GATE = SC.TAU, SC.FLOOR, EV.GATE
N_TARGETS, VALUE_RENDERS, TARGET_TRAIN = 150, 5, 64
LAMBDA = .01  # the translation draft's anchor map setting; not tuned here
PROBES = 2048
STREAMS = ('stationary', 'shift')
VALUE_WIDTH = {'counts': S1.CORE, 'symbols': 32}


# ---------------------------------------------------------------- maps, pure

def fit_map(X, Y, lam=LAMBDA):
    """Identity-centred ridge with a bias: Y ~ X A + b, regularized toward no change, as the translation draft's anchor map. Returns the function it defines."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    d = X.shape[1]
    Xb = np.c_[X, np.ones(len(X))]
    s = max(float((Y ** 2).mean()), 1e-12)
    I0 = np.eye(d + 1)[:, :d]
    A = np.linalg.solve(Xb.T @ Xb / (len(X) * s) + lam / d * np.eye(d + 1), Xb.T @ Y / (len(X) * s) + lam / d * I0)
    return lambda Z: (np.c_[np.asarray(Z, np.float64), np.ones(len(Z))] @ A).astype(np.float32)


def median_angle(a, b):
    """The median angle between paired vectors, in radians."""
    return float(np.median(np.arccos(np.clip((S1.unit(a) * S1.unit(b)).sum(1), -1., 1.))))


def key_drift(old, new):
    """From probes under the old and new encoder: the map, and the median angle each probe moved with no correction and with the map, the latter on probes the map did not fit."""
    half = len(old) // 2
    fn = fit_map(old[:half], new[:half])
    return {'map': fit_map(old, new), 'angle': median_angle(old, new), 'residual': median_angle(fn(old[half:]), new[half:])}


# ---------------------------------------------------------------- memory

class Store(SC.RadiusMemory):
    """A unit's radius is an angle. New units start at the write threshold. A merged unit's ball covers both members' balls; drift widens every ball; thr is the cosine of the angle, within [FLOOR, TAU]."""
    def __init__(self, H, k, tau, d, m):
        super().__init__(H, k, tau, d, m)
        self.theta = np.full(H, np.arccos(tau), np.float32)

    def settle_thresholds(self):
        self.thr[:] = self.tau
        self.thr[:self.h] = np.clip(np.cos(self.theta[:self.h]), FLOOR, self.tau)

    def merge_pairs(self, lower, upper):
        m = S1.unit(self.c[lower] + self.c[upper])
        cover = lambda at: self.theta[at] + np.arccos(np.clip((self.a[at] * m).sum(1), -1., 1.))
        theta = np.maximum(cover(lower), cover(upper))
        self.theta[lower] = theta
        h = self.h
        keep = np.ones(h, bool)
        keep[upper] = False
        survivors = self.theta[:h][keep].copy()
        out = super().merge_pairs(lower, upper)
        self.theta[:] = np.arccos(self.tau)
        self.theta[:len(survivors)] = survivors
        self.settle_thresholds()
        return out

    def widen(self, angle):
        """Drift is taken as roughly orthogonal to the query's offset from the address, as most directions are in high dimension, so the cosines multiply. The sum of the angles would be the worst case, and answers almost anything."""
        self.theta[:self.h] = np.arccos(np.clip(np.cos(self.theta[:self.h]) * np.cos(angle), -1., 1.))
        self.settle_thresholds()

    def move_addresses(self, fn):
        h = self.h
        if h:
            size = np.linalg.norm(self.c[:h], axis=1, keepdims=True)
            self.a[:h] = S1.unit(fn(self.a[:h]))
            self.c[:h] = self.a[:h] * size

    def move_values(self, fn):
        h = self.h
        if h:
            n = self.n[:h, None]
            self.u[:h] = fn(self.u[:h] / n) * n


def merge_keeps_distinctions(a, u, theta, lower, upper, gate):
    """For each proposed pair: would the merged ball hold another unit's address whose value differs? True where it would not. No traces needed. a, u (h, .); theta (h,)."""
    c = a[lower] + a[upper]
    m = S1.unit(c)
    vm = S1.unit(S1.unit(u[lower]) + S1.unit(u[upper]))
    cover = lambda at: theta[at] + np.arccos(np.clip((a[at] * m).sum(1), -1., 1.))
    r = np.cos(np.maximum(cover(lower), cover(upper)))
    r = np.clip(r, FLOOR, TAU)
    S = a @ m.T
    h = len(a)
    member = np.zeros((h, len(lower)), bool)
    member[lower, np.arange(len(lower))] = member[upper, np.arange(len(lower))] = True
    return ~((S >= r) & (S1.unit(u) @ vm.T < gate) & ~member).any(0)


class Arm:
    """keys: frozen, fixed (no correction, thresholds stay), radius (thresholds widen by the drift), transport (addresses move by a map, thresholds widen by what it leaves). values: stale, background, live, oracle (a map fitted on the stored values' own images; a reference, not a mechanism)."""
    def __init__(self, keys, values, merge=None, H=2048, m=S1.CORE):
        self.keys, self.values, self.merge, self.frozen = keys, values, merge, keys == 'frozen'
        self.mem = Store(H, M.LS.K, TAU, 32, m)
        self.symbols = np.zeros((H, 2 * N + N), np.int64)  # scoring only
        self.events = []

    def write(self, z, v, sym):
        for i in self.mem.write(z, v):
            self.symbols[i, sym] += 1

    def block_end(self, drift, maps, score):
        if self.keys == 'radius':
            self.mem.widen(drift['angle'])
        elif self.keys == 'transport':
            self.mem.move_addresses(drift['map'])
            self.mem.widen(drift['residual'])
        if self.values != 'stale':
            self.mem.move_values(maps[self.values])
        event = {'units_before': self.mem.h}
        if self.merge is not None:
            event.update(self.consolidate(score))
        event['units'] = self.mem.h
        self.events.append(event)

    def consolidate(self, score):
        mem = self.mem
        h0 = mem.h
        main = self.symbols[:h0].argmax(1)
        labels, blocked, vetoes = np.arange(h0), np.zeros((h0, h0), bool), 0
        while mem.h > 1:
            h = mem.h
            a, val = mem.a[:h], S1.unit(mem.u[:h])
            S = a @ a.T
            S[S < self.merge] = -np.inf
            S[val @ val.T < GATE] = -np.inf
            S[blocked[:h, :h]] = -np.inf
            np.fill_diagonal(S, -np.inf)
            near = S.argmax(1)
            lower = np.arange(h)
            lower = lower[(near[near[lower]] == lower) & (lower < near[lower]) & (S[lower, near[lower]] > -np.inf)]
            if not len(lower):
                break
            upper = near[lower]
            ok = merge_keeps_distinctions(a.copy(), mem.u[:h].copy(), mem.theta[:h].copy(), lower, upper, GATE)
            blocked[lower[~ok], upper[~ok]] = blocked[upper[~ok], lower[~ok]] = True
            vetoes += int((~ok).sum())
            lower, upper = lower[ok], upper[ok]
            if not len(lower):
                continue
            step = mem.merge_pairs(lower, upper)
            labels = step[labels]
            i, j = np.nonzero(blocked[:h, :h])
            blocked[:] = False
            blocked[step[i], step[j]] = True
            blocked[np.arange(mem.h), np.arange(mem.h)] = False
        table = np.zeros_like(self.symbols)
        np.add.at(table, labels, self.symbols[:h0])
        self.symbols = table
        return {**SC.merge_errors(main, score, labels), 'vetoes': vetoes}

    def bytes(self):
        return 4 * self.mem.h * (32 + self.mem.m + 3)  # address, value sum, count, radius, write total of the address


def arms():
    """Image-free arms vary one thing from values-live (radius keys, live maps, no merging), then merging on that base. Stored images are the reference."""
    out = {'values-stale': dict(keys='radius', values='stale'), 'values-background': dict(keys='radius', values='background'), 'values-live': dict(keys='radius', values='live'), 'values-oracle': dict(keys='radius', values='oracle'),
           'keys-fixed': dict(keys='fixed', values='live'), 'keys-transport': dict(keys='transport', values='live'), 'keys-frozen': dict(keys='frozen', values='live'),
           'merge-0.8': dict(keys='radius', values='live', merge=.8), 'merge-0.7': dict(keys='radius', values='live', merge=.7), 'transport-merge-0.8': dict(keys='transport', values='live', merge=.8),
           'values-live-H512': dict(keys='radius', values='live', H=512), 'merge-0.8-H512': dict(keys='radius', values='live', merge=.8, H=512)}
    return out


IMAGE_ARMS = ('rebuild', 'cover-0.25-distinct')


# ---------------------------------------------------------------- material

def material_path(seed):
    return RUNS / 'material' / f'seed{seed}.npz'


def make_material(seed):
    """Targets are pool symbols outside the load sets. Fact i's target is target i, in a seeded order; the other 60 are distractors, never a fact's. Other-font queries where a cue symbol has a second font."""
    pool = PS.survivors()
    train, ev, _ = S0.test_material(seed)
    cps = [int(c) for c in np.load(S0.material_path(seed))['cps']]
    rest = sorted(set(pool) - set(cps))
    rng = nc.stream('si_targets', seed)
    pick = [int(c) for c in rng.permutation(rest)[:N_TARGETS]]
    cover = TK.fonts_covering(sorted(cps[:2 * N]))
    frng = nc.stream('si_fonts', seed)
    other, mask = np.zeros((2 * N, E, 28, 28), np.float32), np.zeros(2 * N, bool)
    for i, c in enumerate(cps[:2 * N]):
        fonts = [f for f in cover.get(c, ()) if f not in pool[c]]
        if fonts:
            other[i] = PS.renderings(c, [fonts[int(frng.integers(len(fonts)))]], seed, 'eval', E)
            mask[i] = True
    buf = io.BytesIO()
    np.savez_compressed(buf, tgt_train=np.stack([PS.renderings(c, pool[c], seed, 'train', TARGET_TRAIN) for c in pick]), tgt_eval=np.stack([PS.renderings(c, pool[c], seed, 'eval', VALUE_RENDERS + 1) for c in pick]), other_font=other, font_mask=mask)
    DUR.atomic_write(material_path(seed), buf.getvalue())


def material():
    for s in S0.SEEDS:
        if not material_path(s).exists():
            make_material(s)


LOADED = {}


def load_material(seed):
    if seed not in LOADED:
        z = np.load(material_path(seed))
        LOADED[seed] = {k: z[k] for k in z.files}
    return LOADED[seed]


# ---------------------------------------------------------------- symbol hosts

def host_path(seed, b, order):
    return RUNS / 'hosts' / f'seed{seed}-{order}-block{b}.pt'


def symbol_trials(train, tgt_train, words, rng):
    """As translation trials, but a word's candidates come only from its set's targets and the never-taught distractors, so an earlier set's targets do not recur. A target's index equals its fact's index."""
    w = rng.integers(len(words), size=TM.TRIALS)
    w = words[w]
    j = rng.integers(R, size=TM.TRIALS)
    targets = list(dict.fromkeys(w.tolist()))
    pool = [t for t in words.tolist() if t not in targets]
    fill = np.concatenate([rng.permutation(pool), 2 * N + rng.permutation(N_TARGETS - 2 * N)])
    cands = np.array(targets + fill[:TM.CANDIDATES - len(targets)].astype(int).tolist())
    return {'src': train[w, j], 'cand': tgt_train[cands, rng.integers(TARGET_TRAIN, size=TM.CANDIDATES)], 'label': np.array([targets.index(int(x)) for x in w]), 'words': w.tolist(), 'renders': j.tolist()}


@torch.no_grad()
def host_recall(enc, model, ev, tgt, words):
    enc['image'].eval()
    code = G.codes_fn(enc, model)
    q = code(ev[words].reshape(-1, 28, 28), 'image')
    k = code(tgt[:, VALUE_RENDERS], 'image')
    pick = G.pair(model, q, k[None].expand(len(q), -1, -1), 1).argmax(1).cpu().numpy()
    enc['image'].train()
    return float((pick == np.repeat(words, E)).mean())


def fit_symbol_host(task):
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    TM.install_loss()
    withheld = D.withheld_counts(seed)
    mat = load_material(seed)
    train, ev, _ = S0.test_material(seed)
    train, ev = np.ascontiguousarray(train[:2 * N], dtype=np.float32), np.ascontiguousarray(ev[:2 * N], dtype=np.float32)
    enc, model, opt = LM.setup('sparse48', seed)
    LM.CURRENT['end'] = None
    TM.STATE['enc'] = enc
    TM.STATE['trials'] = None
    rng = nc.stream('partA_train', seed, 'A-order')
    for lang in P.LANGS:
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    hashes, recall = {'3': S0.state_hash(enc, model)}, {}
    first, second = SC.ORDERS[order]
    words = {'A': np.arange(N), 'B': np.arange(N, 2 * N)}
    for b, tag in ((4, first), (5, second), (6, None)):
        crng, trng = nc.stream('ap_context', seed, b), nc.stream('si_trials', seed, b, order)
        TM.STATE['losses'] = []
        for _ in range(AP_L):
            TM.STATE['trials'] = symbol_trials(train, mat['tgt_train'], words[tag], trng) if tag else None
            LM.train_update(enc, model, opt, SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
        TM.STATE['trials'] = None
        hashes[str(b)] = S0.state_hash(enc, model)
        recall[str(b)] = {t: host_recall(enc, model, ev, mat['tgt_eval'], words[t]) for t in 'AB'}
        LM.save_atomic(LM.weights(enc, model, withheld, 'sparse48', seed), host_path(seed, b, order))
    ref = S0.reference_hash(seed)
    return f'seed{seed}/{order}', {'hashes': hashes, 'block3_reproduces': hashes['3'] == ref, 'host_recall_of_targets': recall, 'updates': 3 * D.STEPS + 3 * AP_L}, S0.timing(t0, c0)


def hosts():
    t0 = time.perf_counter()
    S0.prepare()
    material()
    res = S0.pool_map(DUR.durable(fit_symbol_host, RUNS / 'done-hosts', 'seed{0}-{1}'), [(s, o) for s in S0.SEEDS for o in SC.ORDERS])
    S0.save_json(f'{REPORT}-hosts.json', {k: r for k, r, _ in res})
    S0.save_json(f'{REPORT}-hosts-runtime.json', S0.runtime({'phase': 'symbol hosts, 12,000 updates each', 'wall_seconds': time.perf_counter() - t0, 'fits': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))
    print({k: (r['block3_reproduces'], r['host_recall_of_targets']) for k, r, _ in res}, flush=True)


def load_host(seed, b, order, data):
    path = S0.checkpoint_path('sparse48', seed) if b == 3 else M.ck_path(seed, b, order)
    ck = torch.load(path, weights_only=False, map_location='cpu')
    enc, model, _ = LM.setup('sparse48', seed)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    for m in (*enc.values(), model):
        m.requires_grad_(False)
    return lambda x: S0.batched(lambda y: model.proj['image'](enc['image'](y)), x).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- what each read needs

def live_images(cue, tag, rng):
    """Images of the kinds the host's stream shows in this block, drawn from what it draws from: its context collections, the taught cues and those cues' collections. Discarded after the fit. Not the exact draws, since only the population matters to a map."""
    dots = P.DOTS['train']
    n = PROBES // 2 if tag else PROBES
    out = [dots[rng.integers(len(dots), size=n), rng.integers(9, size=n)]]
    if tag:
        off = 0 if tag == 'A' else N
        rows = off + rng.integers(N, size=PROBES // 4)
        out.append(cue[rows * R + rng.integers(R, size=len(rows))])
        out.append(dots[rng.integers(len(dots), size=PROBES // 4), rng.integers(9, size=PROBES // 4)])
    return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)


def taught_targets(tgt_train, tag, rng):
    """Value images passing now: this set's targets, none before or after it. Nothing while no set is taught."""
    if not tag:
        return np.zeros((0, 28, 28), np.float32)
    off = 0 if tag == 'A' else N
    return tgt_train[off + rng.integers(N, size=PROBES // 2), rng.integers(TARGET_TRAIN, size=PROBES // 2)]


class Values:
    """One set of values: the codes written now, and the maps and reads at a block's end."""
    def __init__(self, data, seed, order, c, mat, unseen_images):
        self.data, self.seed, self.order, self.c, self.mat = data, seed, order, c, mat
        self.images = unseen_images if data == 'counts' else mat['tgt_eval'][:2 * N, :VALUE_RENDERS].transpose(1, 0, 2, 3)
        self.fact_row = c['counts'] - 1 if data == 'counts' else np.arange(3 * N)
        self.score = c['counts'] if data == 'counts' else np.arange(3 * N)
        self.row_counts = self.fact_row + 1

    def table(self, code):
        return np.stack([code(self.images[:, k]) for k in range(self.images.shape[1])], 1)

    def view(self, code, b):
        out = dict(self.c)
        if self.data == 'counts':
            trained = np.stack([code(P.DOTS['train'][:, k]) for k in range(9)], 1)
            for t in 'AB':
                out[f'cand_{t}'] = code(V.candidates(N, E, self.seed, t)[1]).reshape(N * E, 9, -1)
            out['protos'] = trained.mean(0)
        else:
            out['cand'] = code(self.mat['tgt_eval'][:, VALUE_RENDERS])
        return out

    def read(self, mem, Z, rows, view, tag):
        """Z: unit keys of queries about symbol rows. Abstentions wrong."""
        content, ok = mem.read(Z)
        if self.data == 'counts':
            at = (rows - (0 if tag == 'A' else N))
            per = (at * E + self.query_index) if self.query_index is not None else None
            return S1.decode(content, ok, self.c['counts'][rows], view[f'cand_{tag}'][per], view[f'order_{tag}'][per], view['protos'])
        pick = ((content[:, None] - view['cand'][None]) ** 2).sum(-1).argmin(1)
        return {'symbol': float(((pick == rows) & ok).sum() / len(rows)), 'coverage': float(ok.mean())}

    def ceiling(self, table, view, tag):
        off = 0 if tag == 'A' else N
        if self.data == 'counts':
            truth = self.c['counts'][np.repeat(np.arange(off, off + N), E)]
            return S1.decode(view['protos'][truth - 1], np.ones(len(truth), bool), truth, view[f'cand_{tag}'], view[f'order_{tag}'], view['protos'])
        rows = np.arange(off, off + N)
        codes = table[:, rows].reshape(-1, table.shape[-1])
        pick = ((codes[:, None] - view['cand'][None]) ** 2).sum(-1).argmin(1)
        return {'symbol': float((pick == np.tile(rows, VALUE_RENDERS)).mean())}


def read_arm(values, mem, Zq, Zf, mask, view, tags):
    """Both sets on the same-font queries, on other-font queries where the symbol has a second font, and how often unseen symbols get an answer."""
    out = {'unknown_coverage': float(mem.read(Zq[2 * N * E:3 * N * E])[1].mean())}
    for tag in tags:
        off = 0 if tag == 'A' else N
        rows = np.repeat(np.arange(off, off + N), E)
        values.query_index = np.tile(np.arange(E), N)
        out[tag] = values.read(mem, Zq[rows * E + values.query_index], rows, view, tag)
        have = mask[rows]
        values.query_index = np.tile(np.arange(E), N)[have]
        out[f'{tag}-other-font'] = values.read(mem, Zf[rows[have] * E + values.query_index], rows[have], view, tag)
        out[f'{tag}-font-symbols'] = int(have.sum() // E)
    return out


def learn(task):
    seed, lr, order, stream_kind = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    M.LS.TAU = TAU
    c = S1.load(seed)
    mat = load_material(seed)
    model = SM.encoder('unlabelled', seed)
    start = SM.encoder('unlabelled', seed).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr)
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    fonts = mat['other_font'].reshape(-1, 28, 28)
    rng = nc.stream('sm_learning', seed, str(lr))  # as the earlier pilot, so a stationary stream gives the same encoder
    prng = nc.stream('si_probes', seed, str(lr), order, stream_kind)
    streams = {t: S1.stream(t, seed, c) for t in 'AB'}
    first, second = SC.ORDERS[order]
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    bg_probe = np.ascontiguousarray(bg[prng.integers(n_cls, size=PROBES), prng.integers(n_r, size=PROBES)], dtype=np.float32)
    unseen_images = UT.source_images(seed)['new-radius']
    shift_pool = np.concatenate([P.DOTS['train'].reshape(-1, 28, 28), *[S0.X.DATA[(f, 'eval')][0] for f in UT.FORMATS[:3]]]).astype(np.float32) if stream_kind == 'shift' else None
    values = {d: Values(d, seed, order, c, mat, unseen_images) for d in ('counts', 'symbols')}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        start_cues, start_queries, start_fonts = np_(enc(start, cue)), np_(enc(start, query)), np_(enc(start, fonts))
    free = {d: {k: Arm(**v, m=VALUE_WIDTH[d]) for k, v in arms().items()} for d in values}
    specs = M.arms()
    imaged = {d: {k: T.Arm(**specs[k]) for k in IMAGE_ARMS} for d in values}
    for d in imaged:
        for k, arm in imaged[d].items():
            arm.mem = SC.RadiusMemory(specs[k]['H'], M.LS.K, TAU, 32, VALUE_WIDTH[d])
    first_coll = np.full(len(cue), -1, np.int64)
    reads, drifts = {}, {}
    current = {'counts': load_host(seed, 3, order, 'counts')}
    sense = lambda m: (lambda x: np_(enc(m, x)))

    for b, tag in ((4, first), (5, second), (6, None)):
        write_table = values['counts'].table(current['counts'])
        s = streams[tag] if tag else None
        old_key = copy.deepcopy(model).requires_grad_(False).eval()
        key_probes = []
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
            key_probes.append(base[int(prng.integers(len(base)))].numpy())
            if s is not None:
                part = s[u * BATCH:(u + 1) * BATCH]
                flat = part[:, 0] * R + part[:, 1]
                key_probes.append(cue[flat[int(prng.integers(len(flat)))]])
                with torch.no_grad():
                    live = np_(model(torch.from_numpy(cue[flat])))
                    live_values = np_(model(torch.from_numpy(np.ascontiguousarray(mat['tgt_eval'][part[:, 0], part[:, 2] % VALUE_RENDERS], dtype=np.float32))))
                for i in range(len(part)):
                    sym, col = int(part[i, 0]), int(part[i, 2])
                    if first_coll[flat[i]] < 0:
                        first_coll[flat[i]] = col
                    for d, vals in values.items():
                        v = write_table[col % len(write_table), vals.fact_row[sym]] if d == 'counts' else live_values[i]
                        for arm in free[d].values():
                            arm.write(start_cues[flat[i]] if arm.frozen else live[i], v, sym)
                        for arm in imaged[d].values():
                            arm.write(live[i], np.zeros(VALUE_WIDTH[d], np.float32), flat[i])
        with torch.no_grad():
            probes = np.ascontiguousarray(np.stack(key_probes), dtype=np.float32)
            drift = key_drift(np_(enc(old_key, probes)), np_(enc(model, probes)))
            table_cue, q, f = np_(enc(model, cue)), np_(enc(model, query)), np_(enc(model, fonts))
        drifts[str(b)] = {'angle': drift['angle'], 'residual': drift['residual']}
        Zq, Zqs = S1.unit(q), S1.unit(start_queries)
        Zf, Zfs = S1.unit(f), S1.unit(start_fonts)
        tags = first if b == 4 else 'AB'
        reads[str(b)] = {}
        for d, vals in values.items():
            if d == 'counts':
                old, new = current[d], load_host(seed, b, order, d)
                live_x = live_images(cue, tag, prng)
            else:
                old, new = sense(old_key), sense(model)
                live_x = np.ascontiguousarray(np.concatenate([probes, taught_targets(mat['tgt_train'], tag, prng)]), dtype=np.float32)
            fresh = vals.table(new)
            maps = {'background': fit_map(old(bg_probe), new(bg_probe)), 'live': fit_map(old(live_x), new(live_x)), 'oracle': fit_map(vals.table(old).reshape(-1, fresh.shape[-1]), fresh.reshape(-1, fresh.shape[-1]))}
            view = vals.view(new, b)
            rec = {'ceiling': {t: vals.ceiling(fresh, view, t) for t in tags}}
            for name, arm in free[d].items():
                arm.block_end(drift, maps, vals.score)
                frozen = arm.frozen
                rec[name] = {**read_arm(vals, arm.mem, Zqs if frozen else Zq, Zfs if frozen else Zf, mat['font_mask'], view, tags), 'units': arm.mem.h, 'forced': arm.mem.forced, 'bytes': arm.bytes()}
            for name, arm in imaged[d].items():
                h = arm.mem.h
                arm.mem.u[:h] = M.unit_values(arm.tu, arm.tr, h, first_coll, vals.row_counts, fresh) * arm.mem.n[:h, None]
                arm.tick(table_cue, vals.score)
                h = arm.mem.h
                arm.mem.u[:h] = M.unit_values(arm.tu, arm.tr, h, first_coll, vals.row_counts, fresh) * arm.mem.n[:h, None]
                rec[f'images-{name}'] = {**read_arm(vals, arm.mem, Zq, Zf, mat['font_mask'], view, tags), 'units': h, 'forced': arm.mem.forced, 'bytes': 4 * h * (32 + 2) + len(arm.tu) * M.BYTES_PER_EXPERIENCE}
            reads[str(b)][d] = rec
            if d == 'counts':
                current[d] = new
        del old_key, key_probes, probes
    res = {'reads': reads, 'key_drift': drifts, 'events': {d: {k: a.events for k, a in free[d].items()} for d in free}, 'seed': seed, 'lr': lr, 'order': order, 'stream': stream_kind}
    return f'{seed}/{lr}/{order}/{stream_kind}', res, S0.timing(t0, c0)


def learning():
    t0 = time.perf_counter()
    S0.prepare()
    material()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done', '{0}-{1}-{2}-{3}'), [(s, lr, o, k) for s in S0.SEEDS for lr in SC.LRS for o in SC.ORDERS for k in STREAMS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; keys, values and host all learning; no arm but the images reference stores an image; maps from inputs passing now; true symbols only score merges and reads', 'arms': arms(), 'image_arms': IMAGE_ARMS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'key encoder with memory writes on both sets of values, reads through hosts', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'material': material, 'hosts': hosts, 'learning': learning}[sys.argv[2]]()
