"""Which traces should a consolidated unit keep? (docs/sense-consolidation-v1-pilot.md, Trace selection.) Tests whether the eight-trace cap causes the loss left after wrong merges fell.

Policies, each under the same merge rules:
  turns   the earlier rule: eight, taken in turn from each merged unit
  spread  eight chosen by farthest-point selection when units merge; a full unit takes no new traces
  intake  as spread, but a unit may hold up to 16 between ticks, thinned to eight at each tick, so new appearances can enter
  all     every distinct trace of a unit: the ceiling, not a proposal; radius from the lowest similarity
  all-q05 every distinct trace; radius from the 5th percentile, which tolerates outliers

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_consolidation_traces pilot
"""
import time

import numpy as np

from experiments import durable as DUR
from experiments import sense_consolidation as SC
from experiments import sense_consolidation_evidence as EV
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1

NAME = 'sense-consolidation-v1-traces'
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
CAP = 8
RULES = {'address-0.8-distinct': dict(merge=.8, veto=True), 'address-0.7-distinct': dict(merge=.7, veto=True), 'cover-0.25-distinct': dict(evidence='cover', phi=.25, merge=.6, veto=True)}
POLICIES = {'turns': dict(keep=8), 'spread': dict(keep=8), 'intake': dict(keep=16), 'all': dict(keep=None), 'all-q05': dict(keep=None, radius=.05)}


def spread_select(Z, k):
    """Farthest-point selection on unit vectors: start from the trace farthest from their mean, then repeatedly add the trace least similar to any chosen. Returns indices, in the order chosen."""
    if len(Z) <= k:
        return np.arange(len(Z))
    closest = Z @ S1.unit(Z.mean(0))
    first = int(closest.argmin())
    chosen = [first]
    closest = Z @ Z[first]
    for _ in range(k - 1):
        closest[chosen] = np.inf
        i = int(closest.argmin())
        chosen.append(i)
        closest = np.maximum(closest, Z @ Z[i])
    return np.array(chosen)


class Arm(EV.Arm):
    def __init__(self, traces='turns', **kw):
        super().__init__(**kw)
        self.traces = traces
        self.pairs = set()

    def write(self, z, v, flat):
        if self.keep is not None:
            return super().write(z, v, flat)
        i = int(self.mem.write(z, v)[0])
        if (i, flat) not in self.pairs:
            self.pairs.add((i, flat))
            self.tu.append(i)
            self.tr.append(flat)

    def settle(self, labels, tu, tr, errors):
        errors['traces_before'] = len(tu)
        if self.traces == 'turns':
            return super().settle(labels, tu, tr, errors)
        new = labels[tu]
        if self.keep is None:
            self.pairs = set(zip(new.tolist(), tr.tolist()))
            pairs = sorted(self.pairs)
            self.tu, self.tr = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            Z, out_u, out_r = self.table[tr], [], []
            for u in range(self.mem.h):
                at = np.flatnonzero(new == u)
                # duplicates of one image add no spread
                at = at[np.unique(tr[at], return_index=True)[1]]
                at = at[spread_select(Z[at], CAP)]
                out_u += [u] * len(at)
                out_r += tr[at].tolist()
            self.tu, self.tr = out_u, out_r
            self.seen[:] = 0
            self.seen[:self.mem.h] = np.bincount(out_u, minlength=self.mem.h)
        self.mem.thr[:] = SC.TAU
        return errors


def arms():
    base = dict(gate=EV.GATE, radius=0.)
    out = {'rebuild': dict(H=2048), 'rebuild-all': dict(H=2048, traces='all', keep=None)}
    for r, rule in RULES.items():
        for p, policy in POLICIES.items():
            setting = {**base, **rule, **policy, 'traces': p.split('-')[0]}
            out[f'{r}-{p}'] = dict(H=2048, **setting)
            out[f'{r}-{p}-H512'] = dict(H=512, **setting)
    return out


def main():
    t0 = time.perf_counter()
    res = S0.pool_map(DUR.durable(SC.learn, RUNS / 'done', '{0}-{1}-{2}'), [(s, lr, o) for s in S0.SEEDS for lr in SC.LRS for o in SC.ORDERS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; the encoder keeps learning from augmentations, no symbol knowledge; values and reading on block-3 host codes; true symbols and counts only score merges; keeping all traces is a ceiling', 'arms': arms(), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder training with memory writes', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


SC.Arm, SC.arms, SC.PREPARE = Arm, arms, None

if __name__ == '__main__':
    main()
