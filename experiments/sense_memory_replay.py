"""Step 10, stage 2: latent replay from memory during the core's training, read with memory removed (docs/sense-memory-replay-v1-design.md).

The step 8 calibrated core (sparse48 with frozen DINOv2-S + TrOCR-small and a small projection) is trained exactly as limited-memory main. Facts from
stage 1 (new pool symbols seen with dot collections) are written into kept-centre memory during block 2 (set A) and block 3 (set B). Replay arms add 16
stored (symbol, collection) pairs to every update as a separate matching term, with the replacement slot chosen by the current core (gradients stopped)
or, in a diagnostic arm, by the unit's true count. Replay of stored frozen codes is rehearsal: latent replay from memory.

Arms: no-replay, memory-replay, buffer-replay (reservoir of single feature pairs, fixed capacity declared before the run), memory-replay-true-count.
Stages:
  prepare  GPU: frozen codes, centres and gates, the write plan, and the full memory once, for the buffer's fixed capacity.
  main     the core runs (three workers), with reads at every snapshot.
Usage:
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.sense_memory_replay pilot prepare
  NFI_DEVICE=cuda SMR_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_memory_replay pilot main
Smoke: `smoke` in place of `pilot` (18 symbols per set, 8 writes, 50 updates per block; the delay and relearning keep their lengths).
"""
import os
import pickle
import sys
import time

SMOKE = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
if SMOKE:
    os.environ.setdefault('NCA_STEPS', '50')
_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import sense_memory_core as SMC  # imports sense_core, which sets limited_memory's stage
sys.argv = _argv
import numpy as np
import torch
import torch.nn.functional as F

from experiments import transition_pilot as TP

SC, S0, P, D, X, LM, nc = SMC.SC, SMC.S0, SMC.P, SMC.D, SMC.X, SMC.SC.LM, SMC.nc
SN, NM, PS, SA, GE, DEV, DUR = SMC.SN, SMC.NM, SMC.PS, SMC.SA, SMC.GE, SMC.DEV, SMC.DUR

NAME = 'sense-memory-replay-v1' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
SEEDS = tuple(int(s) for s in os.environ.get('SMR_SEEDS', '300' if SMOKE else '300,301,302,303').split(','))
ARMS = tuple(os.environ.get('SMR_ARMS', 'no-replay,memory-replay,buffer-replay,memory-replay-true-count').split(','))
WORKERS = int(os.environ.get('SMR_WORKERS', 3))
ENCODER = 'joined-mlp-calibrated'
MODELS = SMC.MODELS
N_SET = 18 if SMOKE else 45
WRITES = 8 if SMOKE else 32
QUERIES = SMC.QUERIES
REPLAY = 16
WIDTH = 768  # raw frozen code: DINOv2-S 384 + TrOCR-small 384
PAIR_BYTES = 2 * WIDTH * 4
TAU = SMC.TAU


def phase(step):
    S = D.STEPS
    if step <= S:
        return 'block1'
    if step <= 2 * S:
        return 'block2'
    if step <= 3 * S:
        return 'block3'
    return 'delay' if step <= 3 * S + LM.DELAY else 'relearn'


# ---------------------------------------------------------------- prepare (memory never depends on the core)

def prepare_task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    DEV.install()
    sense = SN.Sense(MODELS, DEV.NAME if DEV.ACTIVE else 'cpu', chunk=64)
    pool, sets, unseen, passing_syms = SMC.symbols(seed) if not SMOKE else smoke_symbols(seed)
    enc_s = 0.

    def code(x):
        nonlocal enc_s
        e0 = time.perf_counter()
        f = sense(np.ascontiguousarray(x, dtype=np.float32))
        enc_s += time.perf_counter() - e0
        return f

    def joined(f):
        return np.concatenate([f[n] for n in MODELS], -1).astype(np.float32)

    table = SC.table(MODELS)
    row0 = {'A': 0, 'B': N_SET}
    writes, cue, val, query = [], [], [], {}
    for name in ('A', 'B'):
        cps = list(sets[name])
        tr = np.stack([PS.renderings(cp, pool[cp], seed, 'train', WRITES) for cp in cps])
        f = code(tr.reshape(-1, 28, 28))
        ftr = {n: f[n].reshape(len(cps), WRITES, -1) for n in MODELS}
        ev = np.stack([PS.renderings(cp, pool[cp], seed, 'eval', QUERIES) for cp in cps])
        query[name] = {'images': ev, 'codes': joined(code(ev.reshape(-1, 28, 28)))}
        rng = nc.stream('smr_writes', seed, name)
        plan = [(s, r, int(rng.integers(len(P.DOTS['train']))), sets[name][cps[s]]) for s in range(len(cps)) for r in range(WRITES)]
        for i in rng.permutation(len(plan)):
            s, r, e, n = plan[i]
            writes.append({'set': name, 'fact': row0[name] + s, 'symbol': s, 'render': r, 'dot': e, 'count': int(n)})
            cue.append(np.concatenate([ftr[m][s, r] for m in MODELS]))
            val.append(np.concatenate([table[m][table['index'][SC.image_key(np.ascontiguousarray(P.DOTS['train'][e, n - 1], dtype=np.float32))]] for m in MODELS]))
    ev = np.stack([PS.renderings(cp, pool[cp], seed, 'eval', QUERIES) for cp in unseen])
    query['unseen'] = {'images': ev, 'codes': joined(code(ev.reshape(-1, 28, 28)))}

    # passing runs, centres and gates, as stage 1
    prng, vrng = nc.stream('smc_passing', seed), nc.stream('smc_views', seed)
    pairs = [(cp, r) for cp in passing_syms for r in range(SMC.PASS_RENDERS)]
    passing = SMC.PASSING if not SMOKE else 512
    pairs = [pairs[i] for i in prng.permutation(len(pairs))[:2 * passing]]
    rendered = {cp: PS.renderings(cp, pool[cp], seed, 'train', SMC.PASS_RENDERS) for cp in sorted({cp for cp, _ in pairs})}
    rule = SMC.WindowShift(nc.stream('smc_rule', seed))
    opened, kept = {0: [], 1: []}, []
    for run in (0, 1):
        x = np.stack([rendered[cp][r] for cp, r in pairs[run * passing:(run + 1) * passing]])
        for b in range(0, passing, SA.BATCH):
            j, new, forming = rule.observe_images(code(x[b:b + SA.BATCH]), x[b:b + SA.BATCH])
            if new:
                f1, f2 = code(NM.view_arrays(forming, vrng, NM.VIEWS['mild'])), code(NM.view_arrays(forming, vrng, NM.VIEWS['strong']))
                kept.append({n: np.stack([f1[n], f2[n]]) for n in MODELS})
                raw = [np.concatenate([NM.unit(k[n][0]) for k in kept]) for n in MODELS]
                views = [np.stack([np.concatenate([SA.coded(k[n][i], rule.centre_array(n), rule.assign({'dinov2-s': k['dinov2-s'][i]})) for k in kept]) for i in range(2)])
                         for n in MODELS]
                g, _ = SN.Gate.train(raw, views, seed, GE.GATE_STEPS, GE.GATE_BATCH, GE.GATE_LR, GE.GATE_TEMP)
                opened[run].append((j, g))
    prep = {'seed': seed, 'sets': {k: {int(cp): int(n) for cp, n in v.items()} for k, v in sets.items()}, 'writes': writes,
            'cue': np.stack(cue), 'value': np.stack(val), 'query': query, 'centres': [{n: c[n].astype(np.float32) for n in MODELS} for c in rule.centres],
            'opened': opened}
    mem = LiveMemory(prep)  # the whole write sequence once: final bytes fix the buffer's capacity before any core run
    mem.open_run(0)
    mem.write_upto('A', sum(w['set'] == 'A' for w in writes))
    mem.open_run(1)
    mem.write_upto('B', sum(w['set'] == 'B' for w in writes))
    final = mem.bytes()
    prep['buffer_capacity'] = int(final['total'] // PAIR_BYTES)
    prep['final_memory'] = {**final, 'units': mem.n_units(), 'writes': len(writes), 'units_per_fact': mem.units_per_fact()}
    prep['runtime'] = S0.runtime({'phase': 'prepare', 'seed': seed, 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                  'frozen_encoding_seconds': enc_s, 'senses': sense.describe()})
    RUNS.mkdir(parents=True, exist_ok=True)
    DUR.atomic_write(RUNS / f'prep-seed{seed}.pkl', pickle.dumps(prep))
    print('prepare', seed, f'{time.perf_counter() - t0:.0f} s', prep['final_memory'], 'buffer capacity', prep['buffer_capacity'], flush=True)


def smoke_symbols(seed):
    pool, sets, unseen, passing = SMC.symbols(seed)
    take = {k: {} for k in sets}
    for k, d in sets.items():  # balanced: N_SET / 9 symbols per count
        for cp, n in d.items():
            if sum(v == n for v in take[k].values()) < N_SET // 9:
                take[k][cp] = n
    return pool, take, unseen[:N_SET], passing


# ---------------------------------------------------------------- memory and buffer during training

class LiveMemory:
    """Kept-centre memory with raw values, plus each unit's raw cue sum (needed to replay the pair). Writes happen at their scheduled updates."""
    def __init__(self, prep):
        self.prep = prep
        D_ = SN.get('dinov2-s').dim
        self.km = SN.KeptMemory(MODELS, lambda: NM.Tagged(SMC.PER_CENTRE, TAU, 2 * D_, WIDTH, 2 * N_SET), frozen_gate=True, values='raw')
        self.order = {k: [i for i, w in enumerate(prep['writes']) if w['set'] == k] for k in ('A', 'B')}
        self.done = {'A': 0, 'B': 0}
        self.log, self.cue_sum, self.runs = {}, {}, set()

    def open_run(self, run):
        if run in self.runs:
            return
        self.runs.add(run)
        for j, g in self.prep['opened'][run]:
            self.km.open_centre(self.prep['centres'][j], g)

    def split(self, x):
        return {'dinov2-s': x[:, :384], 'trocr-s': x[:, 384:]}

    def write_upto(self, name, k):
        ids = self.order[name][self.done[name]:k]
        if not ids:
            return []
        self.done[name] = k
        cue, val = self.split(self.prep['cue'][ids]), self.split(self.prep['value'][ids])
        rows = np.array([self.prep['writes'][i]['fact'] for i in ids])
        for t, j, units in self.km.write(cue, val, SMC.assign(cue, self.km.centres), rows):
            for u in units:
                key = (j, int(u))
                self.log.setdefault(key, []).append(ids[t])
                self.cue_sum[key] = self.cue_sum.get(key, 0.) + self.prep['cue'][ids[t]]
        return ids

    def units(self):
        return list(self.log)

    def n_units(self):
        return len(self.log)

    def pair(self, key):
        j, u = key
        m = self.km.parts[j]
        return self.cue_sum[key] / len(self.log[key]), m.u[u] / m.n[u]

    def describe(self, key):
        """(majority fact, majority count with ties to the count written first, holds more than one fact, holds more than one count)."""
        ws = [self.prep['writes'][i] for i in self.log[key]]
        facts = [w['fact'] for w in ws]
        counts = [w['count'] for w in ws]
        tally = {}
        for c in counts:
            tally[c] = tally.get(c, 0) + 1
        best = max(tally.values())
        maj = next(c for c in counts if tally[c] == best)
        return int(np.bincount(facts).argmax()), int(maj), len(set(facts)) > 1, len(set(counts)) > 1

    def units_per_fact(self):
        f = {}
        for key, ws in self.log.items():
            for i in ws:
                f.setdefault(self.prep['writes'][i]['fact'], set()).add(key)
        return float(np.mean([len(v) for v in f.values()])) if f else 0.

    def bytes(self):
        s = self.km.state_bytes()
        cues = 4 * WIDTH * self.n_units()
        return {**s, 'raw_cues': cues, 'total': s['units_occupied'] + s['centres'] + s['gates'] + cues}


class Reservoir:
    """Algorithm R over the stream of writes, with a fixed capacity: until full it keeps every write; then every write so far is kept with equal probability."""
    def __init__(self, capacity, rng):
        self.capacity, self.rng, self.items, self.seen = capacity, rng, [], 0

    def add(self, ids):
        for i in ids:
            self.seen += 1
            if len(self.items) < self.capacity:
                self.items.append(i)
            else:
                k = int(self.rng.integers(self.seen))
                if k < self.capacity:
                    self.items[k] = i


# ---------------------------------------------------------------- the run

class Replay:
    """Everything an arm adds to limited-memory main: scheduled writes, the replay term, the reads at every snapshot, and the logs."""
    def __init__(self, arm, seed, prep):
        self.arm, self.seed, self.prep = arm, seed, prep
        self.mem = LiveMemory(prep)
        self.mem.open_run(0)
        self.buffer = Reservoir(prep['buffer_capacity'], nc.stream('smr_reservoir', seed)) if arm == 'buffer-replay' else None
        self.rng = nc.stream('smr_replay', seed, arm)
        self.enc = None
        self.stats = {'slot': {}, 'exposure_fact': np.zeros(2 * N_SET, np.int64), 'exposure_count': np.zeros(9, np.int64),
                      'mixed_fact_replays': 0, 'mixed_count_replays': 0, 'replays': 0, 'lambda': [], 'data_match_items': [], 'replay_loss': []}
        self.bytes, self.reads, self.added = {}, {}, False

    def sync_writes(self, step):
        S = D.STEPS
        for name, b in (('A', 2), ('B', 3)):
            start = (b - 1) * S
            if step <= start:
                continue
            if name == 'B':
                self.mem.open_run(1)
            n = len(self.mem.order[name])
            k = n if step >= b * S else (step - start) * n // S
            ids = self.mem.write_upto(name, k)
            if self.buffer is not None:
                self.buffer.add(ids)

    def sample(self):
        """REPLAY pairs: (cue raw, value raw, true slot, fact, mixed fact, mixed count, set)."""
        out = []
        if self.arm == 'buffer-replay':
            items = self.buffer.items
            if not items:
                return out
            for i in self.rng.integers(len(items), size=REPLAY):
                w = self.prep['writes'][items[i]]
                out.append((self.prep['cue'][items[i]], self.prep['value'][items[i]], w['count'] - 1, w['fact'], False, False, w['set']))
        else:
            units = self.mem.units()
            if not units:
                return out
            for i in self.rng.integers(len(units), size=REPLAY):
                key = units[i]
                c, v = self.mem.pair(key)
                fact, maj, mf, mc = self.mem.describe(key)
                out.append((c, v, maj - 1, fact, mf, mc, 'A' if fact < N_SET else 'B'))
        return out

    def extra(self, orig):
        def loss(model, material, c, cand):
            base = orig(model, material, c, cand)
            step = LM.CURRENT['step']
            self.sync_writes(step)
            if self.arm == 'no-replay':
                return base
            items = self.sample()
            if not items:
                return base
            x = torch.from_numpy(np.stack([it[0] for it in items] + [it[1] for it in items]).astype(np.float32))
            z = P.Coupled(model).solve(self.enc['image'].from_codes(x))[:, P.INNER]
            q, v = z[:REPLAY], z[REPLAY:]
            true = torch.tensor([it[2] for it in items], device=z.device)
            d = z.shape[1]
            with torch.no_grad():  # the slot the current core reads the stored collection at, before the update
                lg = P.same_logits(model, v.detach()[:, None].expand(-1, 9, -1).reshape(-1, d), cand.detach()[None].expand(REPLAY, -1, -1).reshape(-1, d)).reshape(REPLAY, 9)
                read = lg.argmax(1)
            slot = true if self.arm == 'memory-replay-true-count' else read
            C = cand[None].expand(REPLAY, -1, -1).clone()
            C[torch.arange(REPLAY, device=z.device), slot] = v
            logits = P.same_logits(model, q[:, None].expand_as(C).reshape(-1, d), C.reshape(-1, d)).reshape(REPLAY, 9)
            lr = F.cross_entropy(logits, slot)
            lam = REPLAY / len(material[2])
            ph = phase(step)
            ok = (read == true).cpu().numpy()
            st = self.stats
            for it, o in zip(items, ok):
                for key in ((ph, it[6]), (ph, 'mixed-count' if it[5] else 'single-count')):
                    a = st['slot'].setdefault(f'{key[0]}|{key[1]}', [0, 0])
                    a[0] += 1
                    a[1] += int(o)
                st['exposure_fact'][it[3]] += 1
                st['exposure_count'][it[2]] += 1
                st['mixed_fact_replays'] += int(it[4])
                st['mixed_count_replays'] += int(it[5])
            st['replays'] += len(items)
            if step % 100 == 0:
                st['lambda'].append(lam)
                st['data_match_items'].append(len(material[2]))
                st['replay_loss'].append(round(float(lr.detach()), 4))
            return base + lam * lr
        return loss

    def snapshot(self, orig):
        def snap(enc, model, withheld, seed, point, disrupt):
            out = orig(enc, model, withheld, seed, point, disrupt)
            out['smr'] = self.read(enc, model, point)
            return out
        return snap

    @torch.no_grad()
    def read(self, enc, model, point):
        """With memory removed: each set's fresh renderings through the core alone. With memory present (secondary): stage 1's end-to-end read."""
        img = enc['image']
        if not self.added:
            for k in ('A', 'B', 'unseen'):
                x = self.prep['query'][k]['images'].reshape(-1, 28, 28)
                img.add_codes([SC.image_key(np.ascontiguousarray(r, dtype=np.float32)) for r in x], self.prep['query'][k]['codes'])
            self.added = True
        img.eval(); enc['sequence'].eval()
        host = SMC.Host.__new__(SMC.Host)
        host.enc, host.model, host.frozen, host.cache = enc, model, True, {}
        res = {}
        for name in ('A', 'B'):
            counts = np.repeat([self.prep['sets'][name][cp] for cp in self.prep['sets'][name]], QUERIES)
            dr = SMC.draws(self.seed, name, len(counts), counts)
            ones = np.ones(len(counts), bool)
            s, _ = SMC.score(host, host.images(self.prep['query'][name]['images'].reshape(-1, 28, 28)), ones, counts, dr)
            res[f'{name}|core-only'] = {k: float(v.mean()) for k, v in s.items()}
            if self.mem.done[name]:
                q = self.mem.split(self.prep['query'][name]['codes'])
                r = self.mem.km.read(q)
                ok = r['score'] >= TAU
                s, _ = SMC.score(host, host.raw(r['value']), ok, counts, dr)
                res[f'{name}|with-memory'] = {k: float(v.mean()) for k, v in s.items()}
        img.train(); enc['sequence'].train()
        self.bytes[point] = {'memory': self.mem.bytes(), 'memory_units': self.mem.n_units(), 'writes': int(sum(self.mem.done.values())),
                             'buffer_items': len(self.buffer.items) if self.buffer is not None else None,
                             'buffer_bytes': PAIR_BYTES * len(self.buffer.items) if self.buffer is not None else None}
        return res

    def report(self):
        st = self.stats
        return {'slot_accuracy': {k: {'n': a[0], 'accuracy': a[1] / a[0]} for k, a in st['slot'].items()},
                'exposure_per_fact': st['exposure_fact'].tolist(), 'exposure_per_count': st['exposure_count'].tolist(),
                'replays': st['replays'], 'mixed_fact_replays': st['mixed_fact_replays'], 'mixed_count_replays': st['mixed_count_replays'],
                'lambda_every_100': st['lambda'], 'data_match_items_every_100': st['data_match_items'], 'replay_loss_every_100': st['replay_loss'],
                'bytes': self.bytes, 'buffer_capacity': self.prep['buffer_capacity'], 'final_memory': self.prep['final_memory'],
                'units_per_fact_end': self.mem.units_per_fact()}


def replay_task(task):
    arm, seed = task
    prep = pickle.loads((RUNS / f'prep-seed{seed}.pkl').read_bytes())
    rp = Replay(arm, seed, prep)
    orig_extra, orig_snap, orig_setup, orig_runs = TP.EXTRA_LOSS, LM.snapshot, LM.setup, SC.RUNS

    def setup(a, s):
        enc, model, opt = orig_setup(a, s)
        rp.enc = enc
        return enc, model, opt
    TP.EXTRA_LOSS, LM.snapshot, LM.setup, SC.RUNS = rp.extra(orig_extra), rp.snapshot(orig_snap), setup, RUNS / arm
    try:
        _, res, rt = SC.run((ENCODER, seed))
    finally:
        TP.EXTRA_LOSS, LM.snapshot, LM.setup, SC.RUNS = orig_extra, orig_snap, orig_setup, orig_runs
    rt['replay_items'] = rp.stats['replays']
    rt['image_lookups'], rt['image_cache_misses'] = rp.enc['image'].lookups, rp.enc['image'].misses
    rt['extra_item_passes'] = 2 * rp.stats['replays']
    return f'{arm}/seed{seed}', {**res, 'arm': arm, 'smr': rp.report()}, rt


def prepare_main():
    S0.prepare()
    for seed in SEEDS:
        if not (RUNS / f'prep-seed{seed}.pkl').exists():
            prepare_task(seed)


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    X.make_model = None
    tasks = [(a, s) for a in ARMS for s in SEEDS]
    ctx = __import__('multiprocessing').get_context('fork')
    fn = DUR.durable(replay_task, RUNS / 'done', '{0}-seed{1}')
    with ctx.Pool(WORKERS) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}-pilot.json', {'status': 'exploratory, pilot seeds 300-303; stage 2: latent replay from memory (rehearsal); reads with memory removed',
                                          'arms': ARMS, 'seeds': SEEDS, 'replay_per_update': REPLAY, 'writes_per_symbol': WRITES, 'set_size': N_SET, 'tau': TAU,
                                          'licences': SN.licences(), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-pilot-runtime.json', S0.runtime({'phase': 'core runs with replay', 'wall_seconds': time.perf_counter() - t0, 'workers': WORKERS,
                                                             'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'prepare': prepare_main, 'main': main}[sys.argv[2]]()
