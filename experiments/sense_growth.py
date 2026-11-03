"""When should sight open a centre? Growth rules on streams built to stress them (docs/sense-growth-v1-pilot.md).

Only the decision to open centres is tested, on DINOv2-S codes (the routing model of experiments.sense_adapt), cached once per kind; labels only score.
Rules:
  shift          experiments.sense_adapt.ShiftCentres: a batch joins the nearest centre if its mean reaches the centre's threshold, otherwise it goes to a
                 forming pool, which never expires; FORM pooled inputs open a centre
  shift-window   the same, with stale rejections expiring: the pool keeps only batches from the last WINDOW batches
  pile-up        each input goes to its nearest centre; a centre is flagged when the inputs it receives in a batch, centred on it, bunch more than samples
                 of its own members do (the length of their mean unit code above the PILE_Q quantile of bootstrapped member samples, their size rounded down to a power of two);
                 the flagged centre's inputs are split in two, the old centre held fixed (two-means on raw codes, the new side started from the inputs
                 leaning the way of the bunching); those nearer the new side go to a forming pool with the same window; FORM pooled inputs open a centre
Streams (batches of BATCH; kinds in experiments.natural_memory_sequence's order for the seed, forward and reverse):
  runs           each kind for 4 batches, as in the pilots, then a familiar tail: TAIL batches of known kinds in runs of 1 to 8 batches
  interleaved    a new kind every 8 batches; each batch half the newest kind and half the earlier ones; then a tail of even mixtures of all kinds
  gradual        each kind fades into the next over 16 batches; then a tail of even mixtures
  returning      runs of 4 batches in the order 0 1 2 0 1 3 4 0 5 2 (kind positions), then the familiar tail
Usage: NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.sense_growth pilot codes; then ... pilot main
"""
import os
import sys
import time
from collections import Counter

import numpy as np

from experiments import device as DEV
from experiments import natural_memory as NM
from experiments import natural_memory_sequence as SEQ
from experiments import sense_adapt as SA
from experiments import shared_core_stage0 as S0
from src import senses as SN

nc = S0.nc
NAME = 'sense-growth-v1'
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
SEED = int(os.environ.get('SG_SEED', 392))
PER_KIND = 9000
BATCH, FORM, QUANTILE, BOOTS = SA.BATCH, SA.FORM, SA.QUANTILE, SA.BOOTS
WINDOW = 4
PILE_Q, PILE_MIN, PILE_BOOTS = .999, 32, 300
TAIL = 200
RULES = ('shift', 'shift-window', 'pile-up')
STREAMS = ('runs', 'interleaved', 'gradual', 'returning')


def codes():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    DEV.install()
    sense = SN.Sense(('dinov2-s',), DEV.NAME if DEV.ACTIVE else 'cpu', chunk=64)
    out = {}
    rng = nc.stream('sg_pool', SEED)
    for k in SEQ.KINDS:
        pool = SEQ.kind_data(k, SEED)['pool']
        x = np.ascontiguousarray(pool[rng.permutation(len(pool))[:PER_KIND]], dtype=np.float32)
        out[k] = np.concatenate([sense(x[i:i + 512])['dinov2-s'] for i in range(0, len(x), 512)])
        print(k, len(x), f'{time.perf_counter() - t0:.0f} s', flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(RUNS / f'codes-seed{SEED}.npz', **out)
    S0.save_json(f'{REPORT}-codes-runtime.json', S0.runtime({'phase': 'DINOv2 codes of each kind\'s pool', 'wall_seconds': time.perf_counter() - t0, 'senses': sense.describe()}))


# ---------------------------------------------------------------- rules

class Shift(SA.ShiftCentres):
    """sense_adapt's rule on raw DINOv2 codes, with an optional window on the forming pool. Members are kept for scoring."""
    def __init__(self, rng, window=None):
        super().__init__(rng)
        self.window, self.t, self.stamps, self.members = window, 0, [], []
        self.pool_labels = []

    def step(self, z, labels):
        self.t += 1
        if self.window is not None:
            keep = [i for i, s in enumerate(self.stamps) if self.t - s < self.window]
            self.forming = [self.forming[i] for i in keep]
            self.pool_labels = [self.pool_labels[i] for i in keep]
            self.stamps = [self.stamps[i] for i in keep]
        n_before = len(self.forming)
        j, new = self.observe({'dinov2-s': z})
        if j is None or new:
            self.stamps.append(self.t)
            self.pool_labels.append(labels)
        if new:
            self.members.append(np.concatenate(self.pool_labels))
            self.stamps, self.pool_labels = [], []
        return new

    def assign_all(self, z):
        return self.assign({'dinov2-s': z}) if self.centres else np.full(len(z), -1)


class PileUp:
    """Opens a centre when inputs routed to an existing centre bunch up there more than its members do."""
    def __init__(self, rng, window=WINDOW):
        self.rng, self.window = rng, window
        self.centres, self.member_codes, self.members, self.thresholds = [], [], [], []
        self.pool, self.pool_labels, self.stamps, self.t = [], [], [], 0
        self.log = []

    def threshold(self, j, m):
        """PILE_Q quantile of the pile-up of m members centred on a bootstrapped centre, m rounded down to a power of two."""
        b = 1 << int(np.floor(np.log2(max(m, 2))))  # rounded down: smaller samples bunch more, so the threshold stays conservative
        key = (j, b)
        if key not in self.thresholds[j]:
            r = self.member_codes[j]
            idx = self.rng.integers(0, len(r), size=(PILE_BOOTS, min(b, len(r))))
            cdx = self.rng.integers(0, len(r), size=(PILE_BOOTS, len(r)))
            c = r[cdx].mean(1)
            p = np.linalg.norm(SN.unit(r[idx] - c[:, None]).mean(1), axis=1)
            self.thresholds[j][key] = float(np.quantile(p, PILE_Q))
        return self.thresholds[j][key]

    @staticmethod
    def split(z, c_old, lean, steps=10):
        """Which of a flagged centre's inputs are new: two-means on raw codes with the old centre held fixed, the new one started from the inputs leaning
        the way of the bunching."""
        new = lean.copy()
        for _ in range(steps):
            if not new.any():
                break
            c_new = z[new].mean(0)
            nxt = ((z - c_new) ** 2).sum(1) < ((z - c_old) ** 2).sum(1)
            if (nxt == new).all():
                break
            new = nxt
        return new

    def assign_all(self, z):
        if not self.centres:
            return np.full(len(z), -1)
        return (SN.unit(z) @ SN.unit(np.stack(self.centres)).T).argmax(1)

    def step(self, z, labels):
        self.t += 1
        keep = [i for i, s in enumerate(self.stamps) if self.t - s < self.window]
        self.pool, self.pool_labels, self.stamps = [self.pool[i] for i in keep], [self.pool_labels[i] for i in keep], [self.stamps[i] for i in keep]
        if not self.centres:
            novel = np.ones(len(z), bool)
        else:
            a = self.assign_all(z)
            novel = np.zeros(len(z), bool)
            for j in np.unique(a):
                m = a == j
                if m.sum() < PILE_MIN:
                    continue
                y = SN.unit(z[m] - self.centres[j])
                v = y.mean(0)
                p = float(np.linalg.norm(v))
                thr = self.threshold(j, int(m.sum()))
                flagged = p > thr
                self.log.append({'t': self.t, 'centre': int(j), 'n': int(m.sum()), 'pile_up': p, 'threshold': thr, 'flagged': bool(flagged)})
                if flagged:
                    novel[np.flatnonzero(m)[self.split(z[m], self.centres[j], y @ (v / max(p, 1e-12)) > 0)]] = True
        if novel.any():
            self.pool.append(z[novel])
            self.pool_labels.append(labels[novel])
            self.stamps.append(self.t)
        if sum(len(p) for p in self.pool) >= FORM:
            r = np.concatenate(self.pool)
            self.centres.append(r.mean(0))
            self.member_codes.append(r)
            self.members.append(np.concatenate(self.pool_labels))
            self.thresholds.append({})
            self.pool, self.pool_labels, self.stamps = [], [], []
            return True
        return False


# ---------------------------------------------------------------- streams

def stream(name, kinds, rng):
    """List of batches, each a list of (kind position, count) parts, and the index where the tail starts."""
    K = len(kinds)
    batches = []
    if name in ('runs', 'returning'):
        order = list(range(K)) if name == 'runs' else [0, 1, 2, 0, 1, 3, 4, 0, 5, 2]
        for k in order:
            batches += [[(k, BATCH)]] * 4
        tail = len(batches)
        while len(batches) < tail + TAIL:
            k, n = int(rng.integers(K)), int(rng.integers(1, 9))
            batches += [[(k, BATCH)]] * n
        batches = batches[:tail + TAIL]
    elif name == 'interleaved':
        for k in range(K):
            for _ in range(8):
                if k == 0:
                    batches.append([(0, BATCH)])
                else:
                    old = rng.multinomial(BATCH // 2, np.ones(k) / k)
                    batches.append([(k, BATCH - BATCH // 2)] + [(i, int(n)) for i, n in enumerate(old) if n])
        tail = len(batches)
        for _ in range(TAIL // 2):
            batches.append([(i, int(n)) for i, n in enumerate(rng.multinomial(BATCH, np.ones(K) / K)) if n])
    elif name == 'gradual':
        batches += [[(0, BATCH)]] * 4
        for k in range(1, K):
            for s in range(16):
                f = (s + 1) / 16
                n = int(round(f * BATCH))
                batches.append([(k - 1, BATCH - n), (k, n)] if n < BATCH else [(k, BATCH)])
        tail = len(batches)
        for _ in range(TAIL // 4):
            batches.append([(i, int(n)) for i, n in enumerate(rng.multinomial(BATCH, np.ones(K) / K)) if n])
    return batches, tail


def run_one(rule, sname, order, codes_by_kind):
    kinds = SEQ.kind_order(SEED, order)
    srng = nc.stream('sg_stream', SEED, sname, order)
    batches, tail = stream(sname, kinds, srng)
    cursor = {k: 0 for k in kinds}
    perm = {k: nc.stream('sg_draw', SEED, sname, order, k).permutation(len(codes_by_kind[k])) for k in kinds}
    rrng = nc.stream('sg_rule', SEED, rule, sname, order)
    r = Shift(rrng) if rule == 'shift' else Shift(rrng, WINDOW) if rule == 'shift-window' else PileUp(rrng)
    opened, routing = [], []
    for t, parts in enumerate(batches):
        z, lab = [], []
        for k, n in parts:
            kind = kinds[k]
            idx = perm[kind][(cursor[kind] + np.arange(n)) % len(perm[kind])]
            cursor[kind] += n
            z.append(codes_by_kind[kind][idx])
            lab.append(np.full(n, k))
        z, lab = np.concatenate(z), np.concatenate(lab)
        mix = srng.permutation(len(z))
        z, lab = z[mix], lab[mix]
        if r.step(z, lab):
            opened.append(t)
        if t >= len(batches) - 50 and r.centres:
            routing.append((r.assign_all(z), lab))
    # score: each centre's majority kind and purity from the inputs that formed it; routing purity on the last 50 batches
    major, purity = [], []
    for m in r.members:
        c = Counter(m.tolist())
        k, n = c.most_common(1)[0]
        major.append(int(k))
        purity.append(n / len(m))
    seen = sorted({k for parts in batches for k, _ in parts})
    route_ok = float(np.mean(np.concatenate([np.array(major)[a] == lab for a, lab in routing]))) if routing else None
    return {'centres': len(r.centres), 'kinds': len(seen), 'opened_at': opened, 'tail_starts': tail, 'opened_in_tail': sum(t >= tail for t in opened),
            'majority_kind': major, 'purity': purity, 'kinds_with_a_pure_centre': len({k for k, p in zip(major, purity) if p >= .9}),
            'extra_centres': len(r.centres) - len(set(major)), 'routing_purity_last_50': route_ok, 'inputs': int(sum(n for parts in batches for _, n in parts)),
            'tail_inputs': int(sum(n for parts in batches[tail:] for _, n in parts))}


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    z = np.load(RUNS / f'codes-seed{SEED}.npz')
    codes_by_kind = {k: z[k] for k in SEQ.KINDS}
    res = {}
    for rule in RULES:
        for sname in STREAMS:
            for order in SEQ.ORDERS:
                t = time.perf_counter()
                res[f'{rule}/{sname}/{order}'] = out = run_one(rule, sname, order, codes_by_kind)
                print(rule, sname, order, {k: out[k] for k in ('centres', 'kinds_with_a_pure_centre', 'extra_centres', 'opened_in_tail', 'routing_purity_last_50')}, f'{time.perf_counter() - t:.0f} s', flush=True)
    S0.save_json(f'{REPORT}-pilot.json', {'status': 'exploratory; centre-opening rules on stress streams, DINOv2 codes, labels only score', 'seed': SEED, 'rules': RULES, 'streams': STREAMS,
                                          'settings': {'batch': BATCH, 'form': FORM, 'quantile': QUANTILE, 'boots': BOOTS, 'window': WINDOW, 'pile_q': PILE_Q, 'pile_min': PILE_MIN, 'pile_boots': PILE_BOOTS, 'tail': TAIL},
                                          'results': res})
    S0.save_json(f'{REPORT}-pilot-runtime.json', S0.runtime({'phase': 'growth rules on stress streams', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}))


if __name__ == '__main__':
    {'codes': codes, 'main': main}[sys.argv[2]]()
