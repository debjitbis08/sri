"""Consolidation alongside address rebuilding, on the unlabelled contrastive encoder (docs/sense-consolidation-v1-pilot.md). Rebuilding asks where a stored experience lies under the current encoder; consolidation asks whether units now represent the same thing and can be combined. Values and reading stay on block-3 host codes, so differences come from addresses and merging. No mechanism here sees a symbol label; true symbols and counts are used only to score merges.

Stages (argv[2]):
  probe     CPU: frozen keys; how well address and content similarity separate units of one symbol, and merging at each threshold
  learning  GPU: the encoder keeps learning; rebuild and consolidate as a 2x2 and at smaller capacities, in both orders

Usage: NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.sense_consolidation pilot probe
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_learning_senses as LS
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import translation_stable_keys as TK
from src.sparse_local_memory import Memory

nc = S0.nc
NAME = 'sense-consolidation-v1'
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
N, E, R = S1.N, S0.EVAL_RENDERS, S0.TRAIN_RENDERS
BLOCK, BATCH = LS.BLOCK, LS.BATCH
TAU = 0.85  # write and read threshold, as selected on frozen keys in the sense-memory step
WORKERS = int(os.environ.get('NSY_WORKERS', 6))
ORDERS = {'AB': ('A', 'B'), 'BA': ('B', 'A')}
PROBE_ADDRESS = (.75, .8, .85, .9, .95)
PROBE_CONTENT = (None, .5, .8, .9, .95)
ORACLE_ADDRESS = (None, .6, .75, .8, .85, .9)
READ_TAUS = (.85, .8, .75, .7, .6, .5)
LRS = (1e-3, 1e-2)
PREPARE = None  # run before a tick, with the encoder as it is then; used by the evidence stage
FLOOR = 0.5  # the lowest read threshold in the probe's sweep; set after seeing it
TRACE_BYTES = 784  # one stored 28 x 28 image, as one byte per pixel


def merge_errors(main, counts, labels):
    """Scoring only. main: each old unit's main symbol. A merged group's members whose symbol, or count, is not the group's most common."""
    wrong_symbol, wrong_count, merged = 0, 0, 0
    for g in np.unique(labels):
        m = main[labels == g]
        if len(m) < 2:
            continue
        merged += len(m) - 1
        wrong_symbol += len(m) - np.bincount(m).max()
        wrong_count += len(m) - np.bincount(counts[m]).max()
    return {'merged': int(merged), 'wrong_symbol': int(wrong_symbol), 'wrong_count': int(wrong_count)}


def oracle_merge(mem, table, tau):
    """Scoring reference, not a mechanism: true symbols allow a merge, so no merge joins two symbols. tau None merges all of a symbol's units into one. Returns labels."""
    h = mem.h
    main = table[:h].argmax(1)
    labels, at = np.zeros(h, np.int64), 0
    parts = []
    for sym in np.unique(main):
        idx = np.flatnonzero(main == sym)
        sub = Memory(len(idx), 1, mem.tau, mem.d, mem.m)
        sub.c[:], sub.a[:], sub.u[:], sub.n[:] = mem.c[idx], mem.a[idx], mem.u[idx], mem.n[idx]
        sub.h = len(idx)
        labels[idx] = sub.consolidate(-2. if tau is None else tau) + at
        at += sub.h
        parts.append(sub)
    for x, name in ((mem.c, 'c'), (mem.a, 'a'), (mem.u, 'u'), (mem.n, 'n')):
        x[:] = 0
        x[:at] = np.concatenate([getattr(p, name)[:p.h] for p in parts])
    mem.h = at
    return labels


def read_sweep(mem, Zq, c):
    """The read threshold alone moves; writes keep TAU. Memory level, abstentions wrong, both sets, and how often unseen symbols get an answer."""
    keep, out = mem.tau, {}
    for t in READ_TAUS:
        mem.tau = t
        r = S1.read_sets(mem, Zq, c, 'AB')
        out[str(t)] = {'A': r['A']['prototype'], 'B': r['B']['prototype'], 'coverage': r['B']['coverage'], 'unknown_coverage': float(mem.read(Zq[2 * N * E:3 * N * E])[1].mean())}
    mem.tau = keep
    return out


def probe_task(task):
    seed, mode, tau_a, tau_c = task
    t0, c0 = time.perf_counter(), time.process_time()
    c = SM.use_encoder('unlabelled', seed)
    Zt, Zq = S1.unit(c['key_train']), S1.unit(c['key_eval'])
    streams = {t: S1.stream(t, seed, c) for t in 'AB'}
    mem = Memory(2048, 1, TAU, 32, S1.CORE)
    table = np.zeros((2048, 2 * N), np.int64)
    out = {'seed': seed, 'mode': mode, 'tau_address': tau_a, 'tau_content': tau_c, 'blocks': {}}
    for tag in ('A', 'B'):
        for sym, r, col in streams[tag]:
            i = int(mem.write(Zt[sym * R + r], c['coll_train'][col, c['counts'][sym] - 1])[0])
            table[i, sym] += 1
        before, sep, errors = mem.h, None, None
        main = table[:before].argmax(1)
        if mode == 'none':
            a, val = mem.a[:before], S1.unit(mem.u[:before] / mem.n[:before, None])
            upper = np.triu_indices(before, 1)
            sa, sc = (a @ a.T)[upper], (val @ val.T)[upper]
            same, count = main[upper[0]] == main[upper[1]], c['counts'][main[upper[0]]] == c['counts'][main[upper[1]]]
            q = lambda x: [float(v) for v in np.quantile(x, (.01, .1, .5, .9, .99))]
            sep = {'address_same_symbol': q(sa[same]), 'address_other_symbol': q(sa[~same]), 'content_same_count': q(sc[count]), 'content_other_count': q(sc[~count]), 'pairs_same_symbol': int(same.sum()), 'pairs_other_symbol': int((~same).sum())}
            for t in PROBE_ADDRESS:
                sep[f'pairs_at_{t}'] = {'same_symbol': int((same & (sa >= t)).sum()), 'other_symbol': int((~same & (sa >= t)).sum()), 'other_count': int((~count & (sa >= t)).sum())}
        else:
            labels = mem.consolidate(tau_a, tau_c) if mode == 'merge' else oracle_merge(mem, table, tau_a)
            errors = merge_errors(main, c['counts'], labels)
            new = np.zeros_like(table)
            np.add.at(new, labels, table[:before])
            table = new
        reads = S1.read_sets(mem, Zq, c, 'A' if tag == 'A' else 'AB')
        reads['unknown_coverage'] = float(mem.read(Zq[2 * N * E:3 * N * E])[1].mean())
        kept = table[:mem.h]
        out['blocks'][tag] = {'reads': reads, 'units_before': before, 'units': mem.h, 'forced': mem.forced, 'errors': errors, 'separation': sep, 'symbols_held': int(len(np.unique(kept.argmax(1)))), 'purity': float(kept.max(1).sum() / max(kept.sum(), 1))}
    out['read_sweep'] = read_sweep(mem, Zq, c)
    return f'{seed}/{mode}/{tau_a}/{tau_c}', out, S0.timing(t0, c0)


def probe():
    t0 = time.perf_counter()
    tasks = [(s, 'none', None, None) for s in S0.SEEDS] + [(s, 'merge', a, g) for s in S0.SEEDS for a, g in itertools.product(PROBE_ADDRESS, PROBE_CONTENT)] + [(s, 'oracle', a, None) for s in S0.SEEDS for a in ORACLE_ADDRESS]
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = list(pool.imap_unordered(DUR.durable(probe_task, RUNS / 'done-probe', '{0}-{1}-{2}-{3}'), tasks))
    S0.save_json(f'{REPORT}-probe.json', {'status': 'exploratory, pilot seeds, frozen keys, A then B, one merge after each block; true symbols and counts only score merges and define the oracle reference', 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-probe-runtime.json', S0.runtime({'phase': 'CPU probe', 'device': 'cpu', 'workers': WORKERS, 'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)), 'tasks': len(res)}))


class RadiusMemory(Memory):
    """Memory in which each unit may have its own acceptance threshold, for reads and writes. All start at tau, so with none set this is Memory."""
    def __init__(self, H, k, tau, d, m):
        super().__init__(H, k, tau, d, m)
        self.thr = np.full(H, tau, np.float32)

    def _winners(self, sims):
        good = np.flatnonzero(sims >= self.thr[:len(sims)])
        return good if len(good) <= 1 else good[np.argsort(-sims[good], kind='stable')[:self.k]]


class Arm(LS.Arm):
    """Rebuilding and consolidation are separate steps, in that order: where do the stored experiences lie now, then do these units represent the same thing."""
    def __init__(self, H, every=2000, rebuilds=True, merge=None, gate=None, radius=None, frozen=False, keep=8):
        super().__init__(keep, every, H)
        self.mem = RadiusMemory(H, LS.K, TAU, 32, S1.CORE)
        self.rebuilds, self.merge, self.gate, self.radius, self.frozen = rebuilds, merge, gate, radius, frozen
        self.events, self.seconds = [], 0.

    def tick(self, table, counts):
        t = time.perf_counter()
        if self.rebuilds:
            self.rebuild(table)
        event = {'units_before': self.mem.h}
        if self.merge is not None:
            event.update(self.consolidate(counts))
        if self.radius is not None:
            self.set_radius(table)
        self.seconds += time.perf_counter() - t
        event['units'] = self.mem.h
        self.events.append(event)

    def consolidate(self, counts):
        tu, tr = np.array(self.tu), np.array(self.tr)
        h = self.mem.h
        main = np.array([np.bincount(tr[tu == i] // R).argmax() for i in range(h)])  # scoring only
        labels = self.mem.consolidate(self.merge, self.gate)
        errors = merge_errors(main, counts, labels)
        return self.settle(labels, tu, tr, errors)

    def settle(self, labels, tu, tr, errors):
        """Traces follow their units, capped at keep, taken in turn from each merged unit."""
        h = len(labels)
        rank = np.zeros(len(tu), np.int64)
        taken = np.zeros(h, np.int64)
        for t, u in enumerate(tu):
            rank[t] = taken[u]
            taken[u] += 1
        new = labels[tu]
        kept = np.lexsort((tu, rank, new))
        new_sorted = new[kept]
        place = np.arange(len(kept)) - np.searchsorted(new_sorted, new_sorted)
        kept = kept[place < self.keep]
        self.tu, self.tr = new[kept].tolist(), tr[kept].tolist()
        self.seen[:] = 0
        self.seen[:self.mem.h] = np.bincount(self.tu, minlength=self.mem.h)
        self.mem.thr[:] = TAU
        return errors

    def set_radius(self, table):
        """A unit's threshold covers its own kept traces under the current encoder, never above tau, never below the floor."""
        tu, tr = np.array(self.tu), np.array(self.tr)
        sims = (self.mem.a[tu] * table[tr]).sum(1)
        self.mem.thr[:] = TAU
        for i in np.unique(tu):
            self.mem.thr[i] = np.clip(np.quantile(sims[tu == i], self.radius), FLOOR, TAU)


def arms():
    base = dict(gate=.8)
    out = {'frozen': dict(H=2048, frozen=True, every=None, rebuilds=False, keep=0), 'neither': dict(H=2048, every=None, rebuilds=False, keep=0), 'rebuild': dict(H=2048), 'rebuild-radius': dict(H=2048, radius=0.),
           'merge-radius': dict(H=2048, rebuilds=False, merge=.8, radius=0., **base), 'both-fixed-tau': dict(H=2048, merge=.8, **base), 'both-no-gate': dict(H=2048, merge=.8, radius=0.),
           'both-q25': dict(H=2048, merge=.8, radius=.25, **base), 'rebuild-1024': dict(H=1024), 'rebuild-512-e500': dict(H=512, every=500)}
    for m in (.75, .8, .85, .9):
        out[f'both-{m}'] = dict(H=2048, merge=m, radius=0., **base)
    for H, e in ((1024, 2000), (512, 2000), (512, 500), (256, 500), (128, 500)):
        out[f'both-0.8-H{H}-e{e}'] = dict(H=H, every=e, merge=.8, radius=0., **base)
    return out


def learn(task):
    seed, lr, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    LS.TAU = TAU
    c = S1.load(seed)
    model, start = SM.encoder('unlabelled', seed), SM.encoder('unlabelled', seed).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr)
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    rng = nc.stream('sm_learning', seed, str(lr))  # the sense-memory step's stream: the same encoder path, in either order
    first, second = ORDERS[order]
    streams = {t: S1.stream(t, seed, c) for t in 'AB'}
    active = {k: Arm(**v) for k, v in arms().items()}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    start_cues, start_queries = np_(enc(start, cue)), np_(enc(start, query))
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    reads = {}

    def read_point(b, tags):
        model.eval()
        with torch.no_grad():
            q = np_(enc(model, query))
        reads[str(b)] = {}
        for name, arm in active.items():
            Zq = S1.unit(start_queries if arm.frozen else q)
            out = S1.read_sets(arm.mem, Zq, c, tags)
            out.update(units=arm.mem.h, forced=arm.mem.forced, traces=len(arm.tu), unknown_coverage=float(arm.mem.read(Zq[2 * N * E:3 * N * E])[1].mean()), bytes=4 * arm.mem.h * (32 + S1.CORE + 2) + len(arm.tu) * TRACE_BYTES)
            reads[str(b)][name] = out

    step = 0
    for b, tag in ((4, first), (5, second), (6, None)):
        s = streams[tag] if tag else None
        for u in range(BLOCK):
            base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=TK.CLASSES), rng.integers(n_r, size=TK.CLASSES)], dtype=np.float32))
            x = torch.stack([SRL.augment_with(base, rng, SRL.AUGMENTS['full']), SRL.augment_with(base, rng, SRL.AUGMENTS['full'])], 1).reshape(-1, 28, 28)
            model.train()
            z = model(x)
            loss = SRL.info_nce(z @ z.T / SRL.T, partner)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            model.eval()
            if s is not None:
                part = s[u * BATCH:(u + 1) * BATCH]
                flat = part[:, 0] * R + part[:, 1]
                with torch.no_grad():
                    live = np_(model(torch.from_numpy(cue[flat])))
                v = c['coll_train'][part[:, 2], c['counts'][part[:, 0]] - 1]
                for i in range(len(part)):
                    for arm in active.values():
                        arm.write(start_cues[flat[i]] if arm.frozen else live[i], v[i], flat[i])
            if step % LS.TICK == 0:
                due = [a for a in active.values() if a.every and step % a.every == 0]
                if due:
                    with torch.no_grad():
                        table = np_(enc(model, cue))
                        if PREPARE:
                            PREPARE(model, enc, seed)
                    for a in due:
                        a.tick(start_cues if a.frozen else table, c['counts'])
        read_point(b, first if b == 4 else 'AB')
    res = {'reads': reads, 'events': {k: a.events for k, a in active.items()}, 'seed': seed, 'lr': lr, 'order': order}
    rt = S0.timing(t0, c0)
    rt['tick_seconds'] = {k: a.seconds for k, a in active.items()}
    return f'{seed}/{lr}/{order}', res, rt


def learning():
    t0 = time.perf_counter()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done-learning', '{0}-{1}-{2}'), [(s, lr, o) for s in S0.SEEDS for lr in LRS for o in ORDERS])
    S0.save_json(f'{REPORT}-learning.json', {'status': 'exploratory, pilot seeds, both orders; the encoder keeps learning from augmentations, no symbol knowledge; values and reading on block-3 host codes; true symbols and counts only score merges', 'arms': arms(), 'floor': FLOOR, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-learning-runtime.json', S0.runtime({'phase': 'encoder training with memory writes', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'probe': probe, 'learning': learning}[sys.argv[2]]()
