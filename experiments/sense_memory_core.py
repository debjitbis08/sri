"""Step 10, stage 1: memory on frozen sight supplies facts to a changing core (docs/sense-memory-core-v1-design.md).

A fact is a new symbol from the pairing pool seen with a collection of dots. Kept-centre memory with a frozen gate per centre (src.senses.KeptMemory),
addressed by the symbol's frozen code, stores the collection's raw frozen code as its value. A read recalls the value and passes it through the current
calibrated projection and core (the step 8 checkpoints), which answers dots, taps (across senses) and comparison. No core is trained here, and memory
never changes the core.

Stages:
  memory   GPU: frozen codes of the symbols, centres opened by the windowed shift rule on runs of passing symbols, gate refits, writes in both orders,
           and retrieval for every query on the `first` and `both` memory states. Memory does not depend on the host, so this runs once per seed.
  read     CPU: every read through every host checkpoint (calibrated and existing, block 1, block 3, delay, relearn).
Usage:
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.sense_memory_core pilot memory
  SMC_WORKERS=4 /data/venvs/sri-cuda/bin/python -m experiments.sense_memory_core pilot read
"""
import hashlib
import os
import pickle
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import sense_core as SC  # sets limited_memory's argv itself
sys.argv = _argv
import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import general_encoder as GE
from experiments import natural_memory as NM
from experiments import pairing_symbols as PS
from experiments import sense_adapt as SA
from experiments import sense_growth as SG
from src import senses as SN

S0, P, D, X, nc = SC.S0, SC.P, SC.D, SC.X, SC.S0.nc
SMOKE = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
NAME = 'sense-memory-core-v1' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
SEEDS = tuple(int(s) for s in os.environ.get('SMC_SEEDS', '300' if SMOKE else '300,301,302,303').split(','))
WORKERS = int(os.environ.get('SMC_WORKERS', 4))
MODELS = SC.MODELS
N_SET = 45 if not SMOKE else 18  # symbols per set, balanced over counts 1-9
WRITES, QUERIES = 8, 5  # training renderings written per symbol; evaluation renderings queried
PASSING, PASS_RENDERS = 1024 if not SMOKE else 512, 6  # passing renderings per run, renderings drawn per passing symbol
TAU, PER_CENTRE, WINDOW = SA.TAU, 2048, SG.WINDOW
TAU_LOW = GE.TAUS[1]  # the earlier pilots' second threshold (0.70), a secondary abstention read only; writes always use TAU
HOSTS = ('joined-mlp-calibrated', 'existing')
POINTS = ('block1', 'block3', 'delay', 'relearn')
READ_AT = {'first': POINTS, 'both': POINTS[1:]}
WRITE_POINT = {'first': 'block1', 'second': 'block3'}  # when each set is written: decides which host a copy is made by
ORDERS = {1: ('A', 'B'), 2: ('B', 'A')}


# ---------------------------------------------------------------- memory stage

class WindowShift(SA.ShiftCentres):
    """sense_adapt's batch-shift rule with stale forming batches expiring after WINDOW batches (sense_growth's shift-window), keeping the forming
    images so the gate can be refitted from their views when a centre opens."""
    def __init__(self, rng, window=WINDOW):
        super().__init__(rng)
        self.window, self.t, self.stamps, self.imgs = window, 0, [], []

    def observe_images(self, batch, imgs):
        self.t += 1
        keep = [i for i, s in enumerate(self.stamps) if self.t - s < self.window]
        self.forming, self.stamps, self.imgs = [self.forming[i] for i in keep], [self.stamps[i] for i in keep], [self.imgs[i] for i in keep]
        j, new = self.observe(batch)
        if new:
            x = np.concatenate(self.imgs + [imgs])
            self.stamps, self.imgs = [], []
            return j, True, x
        if j is None:
            self.stamps.append(self.t)
            self.imgs.append(imgs)
        return j, False, None


def assign(codes, centres):
    """Nearest of the given centres by DINOv2 raw codes (the rule's routing), for the centres open at the time."""
    C = SN.unit(np.stack([c['dinov2-s'] for c in centres]))
    return (SN.unit(codes['dinov2-s']) @ C.T).argmax(1)


def symbols(seed):
    pool = PS.survivors()
    sets = PS.draw_sets(seed, N_SET, pool)
    rest = sorted(set(pool) - set(sets['A']) - set(sets['B']))
    perm = [int(c) for c in nc.stream('smc_symbols', seed).permutation(rest)]
    return pool, {'A': sets['A'], 'B': sets['B']}, perm[:N_SET], perm[N_SET:]


def memory_task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    DEV.install()
    dev = DEV.NAME if DEV.ACTIVE else 'cpu'
    sense = SN.Sense(MODELS, dev, chunk=64)
    pool, sets, unseen, passing_syms = symbols(seed)
    enc_seconds = 0.

    def code(x):
        nonlocal enc_seconds
        e0 = time.perf_counter()
        f = sense(np.ascontiguousarray(x, dtype=np.float32))
        enc_seconds += time.perf_counter() - e0
        return f

    # renderings: written (train) and queried (eval); unseen symbols are queried only
    imgs, codes = {}, {}
    for name in ('A', 'B'):
        cps = list(sets[name])
        imgs[name, 'train'] = np.stack([PS.renderings(cp, pool[cp], seed, 'train', WRITES) for cp in cps])
        imgs[name, 'eval'] = np.stack([PS.renderings(cp, pool[cp], seed, 'eval', QUERIES) for cp in cps])
    imgs['unseen', 'eval'] = np.stack([PS.renderings(cp, pool[cp], seed, 'eval', QUERIES) for cp in unseen])
    for k, x in imgs.items():
        f = code(x.reshape(-1, 28, 28))
        codes[k] = {n: f[n].reshape(*x.shape[:2], -1) for n in MODELS}

    # dot values: one collection of the symbol's count per write, from the training collections (codes from the step 8 cache)
    table = SC.table(MODELS)
    plan = {}
    for name in ('A', 'B'):
        rng = nc.stream('smc_writes', seed, name)
        counts = [sets[name][cp] for cp in sets[name]]
        w = [(s, r, int(rng.integers(len(P.DOTS['train']))), counts[s]) for s in range(len(counts)) for r in range(WRITES)]
        plan[name] = [w[i] for i in rng.permutation(len(w))]
    dot_rows = {}
    for name in ('A', 'B'):
        for s, r, e, n in plan[name]:
            dot_rows[e, n] = table['index'][SC.image_key(np.ascontiguousarray(P.DOTS['train'][e, n - 1], dtype=np.float32))]

    def dot_codes(keys):
        return {m: np.stack([table[m][dot_rows[k]] for k in keys]) for m in MODELS}

    # passing runs and the rule, once: memory's centres do not depend on what is written
    prng, vrng = nc.stream('smc_passing', seed), nc.stream('smc_views', seed)
    pairs = [(cp, r) for cp in passing_syms for r in range(PASS_RENDERS)]
    pairs = [pairs[i] for i in prng.permutation(len(pairs))[:2 * PASSING]]
    rendered = {cp: PS.renderings(cp, pool[cp], seed, 'train', PASS_RENDERS) for cp in sorted({cp for cp, _ in pairs})}
    rule = WindowShift(nc.stream('smc_rule', seed))
    opened, kept_codes = {}, []
    for run in (0, 1):
        x = np.stack([rendered[cp][r] for cp, r in pairs[run * PASSING:(run + 1) * PASSING]])
        opened[run] = []
        for b in range(0, PASSING, SA.BATCH):
            j, new, forming = rule.observe_images(code(x[b:b + SA.BATCH]), x[b:b + SA.BATCH])
            if new:
                f1, f2 = code(NM.view_arrays(forming, vrng, NM.VIEWS['mild'])), code(NM.view_arrays(forming, vrng, NM.VIEWS['strong']))
                kept_codes.append({n: np.stack([f1[n], f2[n]]) for n in MODELS})
                raw = [np.concatenate([NM.unit(k[n][0]) for k in kept_codes]) for n in MODELS]
                views = []
                for n in MODELS:
                    C = rule.centre_array(n)
                    views.append(np.stack([np.concatenate([SA.coded(k[n][i], C, rule.assign({'dinov2-s': k['dinov2-s'][i]})) for k in kept_codes]) for i in range(2)]))
                g, loss = SN.Gate.train(raw, views, seed, GE.GATE_STEPS, GE.GATE_BATCH, GE.GATE_LR, GE.GATE_TEMP)
                opened[run].append({'centre': j, 'gate': g, 'gate_loss': loss, 'gate_codes': int(len(raw[0])), 'forming': int(len(forming))})
    centres = [{n: c[n].astype(np.float32) for n in MODELS} for c in rule.centres]

    out = {'seed': seed, 'sets': {k: {int(cp): int(n) for cp, n in v.items()} for k, v in sets.items()}, 'unseen': unseen,
           'centres_opened': {run: [o['centre'] for o in v] for run, v in opened.items()}, 'thresholds': list(rule.thresholds),
           'gate_losses': [o['gate_loss'] for v in opened.values() for o in v], 'query_images': {k: imgs[k, 'eval'] for k in ('A', 'B', 'unseen')},
           'query_codes': {k: codes[k, 'eval'] for k in ('A', 'B', 'unseen')}, 'plan': plan, 'orders': {}}
    D_ = SN.get('dinov2-s').dim
    row0 = {'A': 0, 'B': N_SET}
    for order, (first, second) in ORDERS.items():
        km = SN.KeptMemory(MODELS, lambda: NM.Tagged(PER_CENTRE, TAU, 2 * D_, 2 * D_, 2 * N_SET), frozen_gate=True, values='raw')
        log, first_value, writes = {}, {}, []  # (centre, unit) -> write ids; (centre, unit) -> first raw value written; write table
        states = {}

        def open_run(run):
            for o in opened[run]:
                km.open_centre(centres[o['centre']], o['gate'])

        def write(name, role):
            cue = {n: np.stack([codes[name, 'train'][n][s, r] for s, r, _, _ in plan[name]]) for n in MODELS}
            tgt = dot_codes([(e, n) for _, _, e, n in plan[name]])
            rows = np.array([row0[name] + s for s, _, _, _ in plan[name]])
            base = len(writes)
            writes.extend({'set': name, 'role': role, 'symbol': s, 'render': r, 'dot': e, 'count': n} for s, r, e, n in plan[name])
            v = np.concatenate([tgt[n] for n in MODELS], 1)
            for k, j, units in km.write(cue, tgt, assign(cue, km.centres), rows):
                for u in units:
                    log.setdefault((j, int(u)), []).append(base + k)
                    first_value.setdefault((j, int(u)), v[k].copy())

        def retrieve(state, names):
            res = {}
            for name in names:
                q = {n: codes[name, 'eval'][n].reshape(-1, codes[name, 'eval'][n].shape[-1]) for n in MODELS}
                r = km.read(q)
                units = list(zip(r['centre'].tolist(), r['unit'].tolist()))
                held = [log[u] for u in units]
                fact = [int(np.bincount([row0[writes[w]['set']] + writes[w]['symbol'] for w in h]).argmax()) for h in held]
                res[name] = {'score': r['score'], 'centre': r['centre'], 'unit': r['unit'], 'value': r['value'].astype(np.float32),
                             'value_single': np.stack([first_value[u] for u in units]), 'held': held, 'fact': np.array(fact)}
            # bypass: each written fact's own unit, the one holding most of its writes
            own = {}
            for (j, u), ws in log.items():
                for w in ws:
                    f = row0[writes[w]['set']] + writes[w]['symbol']
                    own.setdefault(f, {}).setdefault((j, u), 0)
                    own[f][j, u] += 1
            bypass = {}
            for f, units in own.items():
                (j, u) = max(units, key=lambda k: (units[k], -k[0], -k[1]))
                mem = km.parts[j]
                bypass[f] = {'unit': (j, u), 'value': (mem.u[u] / mem.n[u]).astype(np.float32), 'value_single': first_value[j, u], 'held': log[j, u],
                             'units_of_fact': len(units)}
            states[state] = {'reads': res, 'bypass': bypass, 'units': [m.h for m in km.parts], 'forced': int(sum(m.forced for m in km.parts)),
                             'state_bytes': km.state_bytes(), 'single_value_bytes': 4 * len(first_value) * 2 * D_}

        open_run(0)
        write(first, 'first')
        retrieve('first', (first, second, 'unseen'))
        open_run(1)
        write(second, 'second')
        retrieve('both', (first, second, 'unseen'))
        # batch-one retrieval latency on the final state
        q = {n: codes['unseen', 'eval'][n].reshape(-1, codes['unseen', 'eval'][n].shape[-1]) for n in MODELS}
        lat = []
        for i in range(min(200, len(q['dinov2-s']))):
            l0 = time.perf_counter()
            km.read({n: q[n][i:i + 1] for n in MODELS})
            lat.append(time.perf_counter() - l0)
        out['orders'][order] = {'first': first, 'second': second, 'writes': writes, 'states': states, 'batch_one_read_seconds': float(np.median(lat))}
    out['runtime'] = S0.runtime({'phase': 'memory', 'seed': seed, 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                 'frozen_encoding_seconds': enc_seconds, 'senses': sense.describe()})
    RUNS.mkdir(parents=True, exist_ok=True)
    DUR.atomic_write(RUNS / f'memory-seed{seed}.pkl', pickle.dumps(out))
    print('memory', seed, f'{time.perf_counter() - t0:.0f} s', 'centres', len(centres), {o: out['orders'][o]['states']['both']['units'] for o in ORDERS}, flush=True)


# ---------------------------------------------------------------- read stage

def param_hash(*mods):
    h = hashlib.sha256()
    for m in mods:
        for k, t in sorted(list(m.named_parameters()) + list(m.named_buffers())):
            h.update(k.encode())
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Host:
    def __init__(self, encoder, seed, mem):
        self.encoder = encoder
        self.enc, self.model = SC.make_model(seed, 'sparse48', encoder)
        self.frozen = isinstance(self.enc['image'], SC.FrozenImage)
        if self.frozen:  # the symbols' frozen codes, so core-only reads look them up instead of encoding live
            for k in ('A', 'B', 'unseen'):
                x = mem['query_images'][k].reshape(-1, 28, 28)
                c = np.concatenate([mem['query_codes'][k][n].reshape(len(x), -1) for n in MODELS], 1)
                self.enc['image'].add_codes([SC.image_key(np.ascontiguousarray(r, dtype=np.float32)) for r in x], c)

    def load(self, seed, point):
        ck = torch.load(SC.RUNS / self.encoder / 'checkpoints' / f'sparse48-seed{seed}-{point}.pt', weights_only=False)
        self.enc['image'].load_state_dict(ck['image'])
        self.enc['sequence'].load_state_dict(ck['sequence'])
        self.model.load_state_dict(ck['sri'])
        for m in (self.enc['image'], self.enc['sequence'], self.model):
            m.eval()
        self.cache = {}
        return self

    def hash(self):
        return param_hash(self.enc['image'], self.enc['sequence'], self.model)

    @torch.no_grad()
    def images(self, x):
        return D.codes_for(self.enc, self.model, np.ascontiguousarray(x, dtype=np.float32), 'image')

    @torch.no_grad()
    def raw(self, v):
        """Core codes of raw frozen codes given directly (a recalled value)."""
        sv = D.B.Solver(self.model, 'image')
        return self.model.split(sv.solve(self.enc['image'].from_codes(v)))['core_inner']

    @torch.no_grad()
    def taps(self, k):
        if ('taps', k) not in self.cache:
            self.cache['taps', k] = D.codes_for(self.enc, self.model, D.TAPS[('eval', k)], 'sequence')
        return self.cache['taps', k]

    def dots(self, k):
        if ('dots', k) not in self.cache:
            self.cache['dots', k] = self.images(P.DOTS['succ'][k])
        return self.cache['dots', k]


def draws(seed, name, n_q, counts):
    """Candidates and partners for a set's queries, the same for every host, arm, order and state."""
    rng = nc.stream('smc_candidates', seed, name)
    E = P.EVAL_PER
    order = np.stack([rng.permutation(9) for _ in range(n_q)])
    dj, tj = rng.integers(E, size=(n_q, 9)), rng.integers(len(D.TAPS[('eval', 1)]), size=(n_q, 9))
    fresh = (dj[np.arange(n_q), counts - 1] + 1 + rng.integers(E - 1, size=n_q)) % E  # a fresh collection, never the candidate of its count
    partner = np.array([rng.choice(np.flatnonzero(counts != counts[i])) for i in range(n_q)])
    # taps -> symbol: per trial a count, a taps item, and one query of each count as the nine candidates
    t_count = np.repeat(np.arange(1, 10), max(1, n_q // 9 // QUERIES))
    t_item = rng.integers(len(D.TAPS[('eval', 1)]), size=len(t_count))
    t_cand = np.stack([[rng.choice(np.flatnonzero(counts == k)) for k in range(1, 10)] for _ in t_count])
    return {'order': order, 'dj': dj, 'tj': tj, 'fresh': fresh, 'partner': partner, 't_count': t_count, 't_item': t_item, 't_cand': t_cand}


@torch.no_grad()
def score(host, Q, ok, counts, dr):
    """R2 dots, R3 taps, R4 comparison and R3' taps -> symbol for query core codes Q; queries with ok False abstain and count as wrong."""
    n_q = len(Q)
    dc = torch.stack([torch.stack([host.dots(int(k) + 1)[dr['dj'][i, k]] for k in dr['order'][i]]) for i in range(n_q)])
    tc = torch.stack([torch.stack([host.taps(int(k) + 1)[dr['tj'][i, k]] for k in dr['order'][i]]) for i in range(n_q)])
    r2 = (dr['order'][np.arange(n_q), D.pick(host.model, Q, dc)] + 1 == counts) & ok
    r3 = (dr['order'][np.arange(n_q), D.pick(host.model, Q, tc)] + 1 == counts) & ok
    p = dr['partner']
    r4 = ((host.model.relation_logits(Q, Q[torch.as_tensor(p)], 2) > 0).numpy() == (counts > counts[p])) & ok & ok[p]
    T = torch.stack([host.taps(int(k))[i] for k, i in zip(dr['t_count'], dr['t_item'])])
    C = Q[torch.as_tensor(dr['t_cand'])]
    lg = P.same_logits(host.model, T[:, None].expand_as(C).reshape(-1, C.shape[-1]), C.reshape(-1, C.shape[-1])).reshape(len(T), 9).numpy()
    lg[~ok[dr['t_cand']]] = -np.inf
    r3s = (lg.argmax(1) + 1 == dr['t_count']) & ok[dr['t_cand']][np.arange(len(T)), dr['t_count'] - 1]
    picks = {'r2': dr['order'][np.arange(n_q), D.pick(host.model, Q, dc)] + 1, 'r3': dr['order'][np.arange(n_q), D.pick(host.model, Q, tc)] + 1}
    return {'r2': r2, 'r3': r3, 'r4': r4, 'r3_taps_to_symbol': r3s}, picks


def read_task(task):
    seed, = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    mem = pickle.loads((RUNS / f'memory-seed{seed}.pkl').read_bytes())
    hosts = {e: Host(e, seed, mem) for e in HOSTS}
    table = SC.table(MODELS)
    row0 = {'A': 0, 'B': N_SET}
    counts = {k: np.repeat([mem['sets'][k][cp] for cp in mem['sets'][k]], QUERIES) for k in ('A', 'B')}
    facts = {k: row0[k] + np.repeat(np.arange(N_SET), QUERIES) for k in ('A', 'B')}
    dr = {k: draws(seed, k, len(counts[k]), counts[k]) for k in ('A', 'B')}
    raw_dots = {k: np.concatenate([table[n][[table['index'][SC.image_key(np.ascontiguousarray(r, dtype=np.float32))] for r in P.DOTS['succ'][k]]] for n in MODELS], 1)
                for k in range(1, 10)}

    def joined(v):
        return SN.join([SN.unit(v[:, :384]), SN.unit(v[:, 384:])])

    results, picks, hashes, lat = {}, {}, {}, {}
    for e, host in hosts.items():
        for point in POINTS:
            host.load(seed, point)
            h0 = host.hash()
            # copies need the codes each write's host gave its dots: the host at the write's time (block 1 for the first set, block 3 for the second)
            for order, o in mem['orders'].items():
                for state, st in o['states'].items():
                    if point not in READ_AT[state]:
                        continue
                    for role in ('first', 'second') if state == 'both' else ('first',):
                        name = o[role]
                        r = st['reads'][name]
                        ok = r['score'] >= TAU
                        R0 = ok & (r['fact'] == facts[name])
                        arms = {}
                        if host.frozen:
                            l0 = time.perf_counter()
                            arms['trace'] = (host.raw(r['value']), ok)
                            lat.setdefault(e, []).append((time.perf_counter() - l0) / len(r['value']))
                            arms[f'trace@{TAU_LOW:.2f}'] = (arms['trace'][0], r['score'] >= TAU_LOW)
                            arms['trace-single'] = (host.raw(r['value_single']), ok)
                            bp = st['bypass']
                            arms['trace-bypass'] = (host.raw(np.stack([bp[f]['value'] for f in facts[name]])), np.ones(len(ok), bool))
                            arms['trace-single-bypass'] = (host.raw(np.stack([bp[f]['value_single'] for f in facts[name]])), np.ones(len(ok), bool))
                        arms['core-only'] = (host.images(mem['query_images'][name].reshape(-1, 28, 28)), np.ones(len(ok), bool))
                        arms['fresh-dots'] = (torch.stack([host.dots(int(c))[j] for c, j in zip(counts[name], dr[name]['fresh'])]), np.ones(len(ok), bool))
                        low = r['score'] >= TAU_LOW
                        out = {'R0_retrieval': float(R0.mean()), 'answered': float(ok.mean()),
                               f'R0_retrieval@{TAU_LOW:.2f}': float((low & (r['fact'] == facts[name])).mean()), f'answered@{TAU_LOW:.2f}': float(low.mean())}
                        for arm, (Q, okq) in arms.items():
                            s, pk = score(host, Q, okq, counts[name], dr[name])
                            out[arm] = {k: float(v.mean()) for k, v in s.items()}
                            if arm in ('trace', 'trace-single'):
                                out[arm + '|retrieved'] = {k: float(v[R0].mean()) if R0.any() and k != 'r3_taps_to_symbol' else None for k, v in s.items()}
                            if arm in ('trace', 'trace-bypass'):
                                picks[f'{e}|{point}|{order}|{state}|{role}|{arm}'] = {k: v.tolist() for k, v in pk.items()}
                        results[f'{e}|{point}|{order}|{state}|{role}'] = out
            hashes[e, point] = (h0, host.hash())
        # copies: values coded by the host at write time, read through every later host
    for e, host in hosts.items():
        write_codes = {}
        for wp in ('block1', 'block3'):
            host.load(seed, wp)
            for order, o in mem['orders'].items():
                ws = o['writes']
                write_codes[wp, order] = host.images(np.stack([P.DOTS['train'][w['dot'], w['count'] - 1] for w in ws]))
        for point in POINTS:
            host.load(seed, point)
            for order, o in mem['orders'].items():
                for state, st in o['states'].items():
                    if point not in READ_AT[state]:
                        continue
                    for role in ('first', 'second') if state == 'both' else ('first',):
                        name = o[role]
                        r = st['reads'][name]
                        ok = r['score'] >= TAU
                        # a unit's copy is the mean of the codes its writes were given, each at its own write time
                        ws = o['writes']
                        Q = torch.stack([torch.stack([write_codes[WRITE_POINT[ws[w]['role']], order][w] for w in held]).mean(0) for held in r['held']])
                        s, _ = score(host, Q, ok, counts[name], dr[name])
                        results[f'{e}|{point}|{order}|{state}|{role}'][('copy' if host.frozen else 'copy-existing')] = {k: float(v.mean()) for k, v in s.items()}
    # sight only, no host: the recalled raw value against candidate collections' raw codes
    for order, o in mem['orders'].items():
        for state, st in o['states'].items():
            for role in ('first', 'second') if state == 'both' else ('first',):
                name, r = o[role], st['reads'][o[role]]
                ok = r['score'] >= TAU
                d = dr[name]
                n_q = len(ok)
                cand = np.stack([np.stack([raw_dots[int(k) + 1][d['dj'][i, k]] for k in d['order'][i]]) for i in range(n_q)])
                S = np.einsum('qd,qkd->qk', joined(r['value']), joined(cand.reshape(-1, 768)).reshape(n_q, 9, -1))
                r1 = (d['order'][np.arange(n_q), S.argmax(1)] + 1 == counts[name]) & ok
                results.setdefault(f'sight|{order}|{state}|{role}', {})['sense-only'] = {'r1': float(r1.mean())}
            u = st['reads']['unseen']
            results[f'sight|{order}|{state}|unseen'] = {'R5_unseen_answered': float((u['score'] >= TAU).mean()), f'R5_unseen_answered@{TAU_LOW:.2f}': float((u['score'] >= TAU_LOW).mean())}
    mem_summary = {order: {'first': o['first'], 'second': o['second'], 'batch_one_read_seconds': o['batch_one_read_seconds'],
                           'states': {s: {'units': st['units'], 'forced': st['forced'], 'state_bytes': st['state_bytes'], 'single_value_bytes': st['single_value_bytes'],
                                          'units_per_fact': float(np.mean([b['units_of_fact'] for b in st['bypass'].values()]))} for s, st in o['states'].items()}}
                   for order, o in mem['orders'].items()}
    rt = S0.runtime({'phase': 'read', 'seed': seed, 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                     'host_read_seconds_per_value': {e: float(np.median(v)) for e, v in lat.items()}, 'memory_stage': mem['runtime']})
    same = all(a == b for a, b in hashes.values())
    return f'seed{seed}', {'results': results, 'picks': picks, 'memory': mem_summary, 'centres_opened': mem['centres_opened'], 'gate_losses': mem['gate_losses'],
                           'hosts_unchanged_by_reads': same}, rt


def memory_main():
    S0.prepare()
    for seed in SEEDS:
        if not (RUNS / f'memory-seed{seed}.pkl').exists():
            memory_task(seed)


def read_main():
    t0 = time.perf_counter()
    S0.prepare()
    SC.LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    fn = DUR.durable(read_task, RUNS / 'done', 'seed{0}')
    with ctx.Pool(min(WORKERS, len(SEEDS))) as pool:
        res = pool.map(fn, [(s,) for s in SEEDS], chunksize=1)
    S0.save_json(f'{REPORT}-pilot.json', {'status': 'exploratory, pilot seeds 300-303 (the step 8 seeds), both orders; stage 1: memory never trains the core',
                                          'tau': TAU, 'per_centre_units': PER_CENTRE, 'set_size': N_SET, 'writes_per_symbol': WRITES, 'queries_per_symbol': QUERIES,
                                          'passing_per_run': PASSING, 'window': WINDOW, 'licences': SN.licences(), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-pilot-runtime.json', S0.runtime({'phase': 'stage 1 reads', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res},
                                                             'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'memory': memory_main, 'read': read_main}[sys.argv[2]]()
