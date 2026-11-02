"""Consolidation as compressing a thing's varied appearances while preserving distinctions from other things (docs/sense-consolidation-v1-pilot.md, Evidence). Two label-free tests on top of rebuilding:
compression evidence: do two units answer each other's traces? Distinction test: would the merged unit, with the radius covering its traces, also take in traces with a different content, or more than a small share of a reference set of other things? True symbols and counts only score merges.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_consolidation_evidence pilot
"""
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import sense_consolidation as SC
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1

NAME = 'sense-consolidation-v1-evidence'
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
R, TAU, FLOOR = SC.R, SC.TAU, SC.FLOOR
REFERENCE_ITEMS = 256  # other things: background images, fixed in number
GATE = .8
IMAGES = {}


def shared_evidence(T, owner, h, phi):
    """T (traces, units) similarities; owner: each trace's unit. Each trace's nearest other unit is the one it would next answer to. Allowed where each unit gives the other at least phi of its traces."""
    T = T.copy()
    T[np.arange(len(T)), owner] = -np.inf
    count = np.zeros((h, h))
    np.add.at(count, (owner, T.argmax(1)), 1)
    return mutual(count, owner, h, phi)


def cover_evidence(T, owner, h, phi, tau):
    """Allowed where each unit's address answers at least phi of the other's traces at tau."""
    count = np.zeros((h, h))
    np.add.at(count, owner, (T >= tau).astype(np.float64))
    return mutual(count, owner, h, phi)


def mutual(count, owner, h, phi):
    share = count / np.maximum(np.bincount(owner, minlength=h), 1)[:, None]
    return (share >= phi) & (share.T >= phi)


def distinct(Z, owner, c, u, lower, upper, gate, reference=None, eps=0.):
    """Would each merged pair preserve distinctions? Its radius covers its own traces, clipped as a unit's radius is. It fails if that radius takes in a trace of another unit whose content differs, or more than eps of the reference set."""
    m, vm = S1.unit(c[lower] + c[upper]), S1.unit(u[lower] + u[upper])
    S = Z @ m.T
    member = (owner[:, None] == lower) | (owner[:, None] == upper)
    radius = np.clip(np.where(member, S, np.inf).min(0), FLOOR, TAU)
    conflict = S1.unit(u[owner]) @ vm.T < gate
    ok = ~((S >= radius) & conflict & ~member).any(0)
    if reference is not None:
        ok &= (reference @ m.T >= radius).mean(0) <= eps
    return ok


class Arm(SC.Arm):
    reference = None

    def __init__(self, evidence=None, phi=None, veto=False, eps=None, **kw):
        super().__init__(**kw)
        self.evidence, self.phi, self.veto, self.eps = evidence, phi, veto, eps

    def tick(self, table, counts):
        self.table = table
        super().tick(table, counts)

    def consolidate(self, counts):
        if self.evidence is None and not self.veto and self.eps is None:
            return super().consolidate(counts)
        mem, tu, tr = self.mem, np.array(self.tu), np.array(self.tr)
        Z = self.table[tr]
        h0 = mem.h
        main = np.array([np.bincount(tr[tu == i] // R).argmax() for i in range(h0)])  # scoring only
        labels, blocked, vetoes = np.arange(h0), np.zeros((h0, h0), bool), 0
        while mem.h > 1:
            h, owner = mem.h, labels[tu]
            a = mem.a[:h]
            S = a @ a.T
            S[S < self.merge] = -np.inf
            val = S1.unit(mem.u[:h])
            S[val @ val.T < self.gate] = -np.inf
            S[blocked[:h, :h]] = -np.inf
            if self.evidence:
                T = Z @ a.T
                E = shared_evidence(T, owner, h, self.phi) if self.evidence == 'shared' else cover_evidence(T, owner, h, self.phi, TAU)
                S[~E] = -np.inf
            np.fill_diagonal(S, -np.inf)
            near = S.argmax(1)
            lower = np.arange(h)
            lower = lower[(near[near[lower]] == lower) & (lower < near[lower]) & (S[lower, near[lower]] > -np.inf)]
            if not len(lower):
                break
            upper = near[lower]
            if self.veto or self.eps is not None:
                ok = distinct(Z, owner, mem.c[:h], mem.u[:h], lower, upper, self.gate if self.veto else -np.inf, Arm.reference if self.eps is not None else None, self.eps or 0.)
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
        out = self.settle(labels, tu, tr, SC.merge_errors(main, counts, labels))
        out['vetoes'] = vetoes
        return out

    def set_radius(self, table):
        super().set_radius(table)
        if self.eps is not None:
            h = self.mem.h
            self.mem.thr[:h] = np.clip(np.maximum(self.mem.thr[:h], np.quantile(self.mem.a[:h] @ Arm.reference.T, 1 - self.eps, axis=1)), FLOOR, TAU)


def prepare(model, enc, seed):
    if seed not in IMAGES:
        bg = S0.background_images(seed)
        rng = S0.nc.stream('sc_reference', seed)
        IMAGES[seed] = np.ascontiguousarray(bg[rng.integers(bg.shape[0], size=REFERENCE_ITEMS), rng.integers(bg.shape[1], size=REFERENCE_ITEMS)], dtype=np.float32)
    Arm.reference = S1.unit(enc(model, IMAGES[seed]).detach().cpu().numpy().astype(np.float32))


def arms():
    base = dict(gate=GATE, radius=0.)
    new = {'cover-0.5': dict(evidence='cover', phi=.5, merge=.6), 'cover-0.25': dict(evidence='cover', phi=.25, merge=.6), 'shared-0.5': dict(evidence='shared', phi=.5, merge=.6), 'shared-0.25': dict(evidence='shared', phi=.25, merge=.6),
           'address-0.8-distinct': dict(merge=.8, veto=True), 'address-0.7-distinct': dict(merge=.7, veto=True), 'cover-0.25-distinct': dict(evidence='cover', phi=.25, merge=.6, veto=True), 'shared-0.25-distinct': dict(evidence='shared', phi=.25, merge=.6, veto=True),
           'address-0.8-others': dict(merge=.8, eps=.01), 'cover-0.25-distinct-others': dict(evidence='cover', phi=.25, merge=.6, veto=True, eps=.01)}
    out = {'rebuild': dict(H=2048), 'rebuild-512-e500': dict(H=512, every=500), 'both-0.9': dict(H=2048, merge=.9, **base), 'both-0.85': dict(H=2048, merge=.85, **base), 'both-0.8': dict(H=2048, merge=.8, **base), 'both-0.8-H512': dict(H=512, merge=.8, **base)}
    for k, v in new.items():
        out[k] = dict(H=2048, **base, **v)
        out[f'{k}-H512'] = dict(H=512, **base, **v)
    return out


def main():
    t0 = time.perf_counter()
    res = S0.pool_map(DUR.durable(SC.learn, RUNS / 'done', '{0}-{1}-{2}'), [(s, lr, o) for s in S0.SEEDS for lr in SC.LRS for o in SC.ORDERS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; the encoder keeps learning from augmentations, no symbol knowledge; values and reading on block-3 host codes; true symbols and counts only score merges', 'arms': arms(), 'reference_items': REFERENCE_ITEMS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder training with memory writes', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


SC.Arm, SC.arms, SC.PREPARE = Arm, arms, prepare

if __name__ == '__main__':
    main()
