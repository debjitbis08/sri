"""Long-term memory v1 (docs/long-term-memory-v1-contract.md): a memory network combined with sparse48 in the forward pass, trained on the readers'
errors, unable to change the core. Host runs write a trajectory cache; memory variants replay from it offline.

Stages (argv[1]):
  encoders          frozen key encoders for the pilot seeds (GPU)
  hosts             sparse48 with the cache hook, sparse48 without it (gate 1, first pilot seed), and mixed, on the pilot seeds (GPU)
  gates             gates 1-5, from the cache
  pilot             the sequential sweep on the pilot seeds, offline on CPU
  confirm-hosts     refuses before the contract is frozen: encoders and host runs for seeds 394-403
  confirm           refuses before the contract is frozen: candidate, six ablations and references
  analyse           the contract's decision rule

Usage: NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.long_term_memory hosts; .venv/bin/python -m experiments.long_term_memory pilot.
"""
import copy
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time

os.environ.setdefault('NAP_OPTION', 'A')
os.environ.setdefault('NAP_ARMS', 'sparse48')
_N = os.environ.get('LTM_N')
if _N:
    os.environ['NAP_N'] = _N
elif not os.environ.get('NSY_SMOKE'):
    os.environ.setdefault('NAP_N', '45')
if not os.environ.get('NSY_SMOKE'):
    os.environ.setdefault('NAP_L', '2000')

import numpy as np
import torch

from experiments import arbitrary_pairings as AP
from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import sequence_pilot as SP
from experiments import streaming_retrieval as SR
from experiments import transition_pilot as TP
from experiments import translation_stable_keys as TK
from src import long_term_memory as ltm

nc, ns = X.nc, X.ns
SMOKE = X.SMOKE
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'gates'
NAME = 'long-term-memory-v1' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'long-term-memory-v1-contract.md'
WORKERS = int(os.environ.get('NSY_WORKERS', 11))
HOST_WORKERS = int(os.environ.get('LTM_HOST_WORKERS', 3))

PILOT_SEEDS = (299,) if SMOKE else (392, 393)
CONFIRM_SEEDS = tuple(range(394, 404))
N, L = AP.N, AP.L
BLOCKS = (3, 4, 5, 6)
SLOW_HALF_LIFE = 2000
KEYS = {'live': 16, 'slow': 16, 'frozen': 32}
TRANSFER = ('chinese/withheld', 'devanagari_digits/withheld')
ELIGIBLE, NEAR = .03, .03
START = dict(form='mlp', projections='learned', key='frozen', eta=.01, beta=.9, C=16, p=.05)
SETTINGS = [('eta', (.001, .003, .01, .03)), ('beta', (0., .9)), ('C', (1, 16, 64)), ('p', (.02, .05, .1))]


def save_json(path, obj):
    DUR.atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=float).encode())


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'stage': STAGE, 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
            'source_sha256': nc.sha256(__file__), 'mechanisms_sha256': nc.sha256(ltm.__file__), 'contract_sha256': nc.sha256(CONTRACT), 'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cpu': platform.processor(), 'workers': WORKERS, 'host_workers': HOST_WORKERS, 'dtype': 'float32', **DEV.info(),
            'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, **extra}


def save_npz(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    DUR.atomic_write(path, buf.getvalue())


def cache_dir(arm, seed, mode):
    return RUNS / 'cache' / f'{arm}-seed{seed}-{mode}-N{N}'


# ---------------------------------------------------------------- the host, with its hook

CAP = {'on': False}
_same = P.same_logits


def capturing_same_logits(model, a, b):
    out = _same(model, a, b)
    if CAP['on']:
        CAP.update(a=a.detach(), b=b.detach(), logits=out.detach(), on=False)
    return out


P.same_logits = capturing_same_logits


class Keys:
    """Codes by key source for images, on the current device, drawing no random numbers and changing no host state."""

    def __init__(self, enc, model, seed, mode):
        self.enc, self.model = enc, model
        self.slow = None
        self.frozen = None
        if mode == 'full':
            self.slow = {'enc': copy.deepcopy(enc['image']), 'model': copy.deepcopy(model)}
            self.slow['enc'].eval()
            self.slow['enc'].requires_grad_(False)
            self.slow['model'].requires_grad_(False)
            self.frozen = SR.key_encoder(seed).to(torch.get_default_device() if DEV.ACTIVE else 'cpu').requires_grad_(False)
            self.r = 2. ** (-1. / SLOW_HALF_LIFE)

    @torch.no_grad()
    def slow_codes(self, x):
        sol = P.Coupled(self.slow['model'])
        return sol.solve(self.slow['enc'](torch.from_numpy(np.ascontiguousarray(x))))[:, P.INNER]

    @torch.no_grad()
    def frozen_codes(self, x):
        return torch.cat([self.frozen(torch.from_numpy(np.ascontiguousarray(x[i:i + 512], dtype=np.float32))) for i in range(0, len(x), 512)])

    @torch.no_grad()
    def step_slow(self):
        for a, b in zip(self.slow['enc'].parameters(), self.enc['image'].parameters()):
            a.lerp_(b, 1 - self.r)
        for a, b in zip(self.slow['model'].parameters(), self.model.parameters()):
            a.lerp_(b, 1 - self.r)
        for a, b in zip(self.slow['model'].buffers(), self.model.buffers()):
            a.copy_(b)

    def all_codes(self, x, live):
        out = {'live': live}
        if self.slow is not None:
            out['slow'] = self.slow_codes(x)
            out['frozen'] = self.frozen_codes(x)
        return out


def to_np(t):
    return t.detach().cpu().numpy().astype(np.float32)


def hooked_update(enc, model, opt, material, rng, keys, sink):
    items, _, match = material[:3]
    ci = copy.deepcopy(rng).integers(len(P.DOTS['train']), size=9)     # the update's own first draw, without taking it
    CAP['on'] = True
    out = LM.train_update(enc, model, opt, material, rng, True)
    CAP['on'] = False
    n = len(match)
    if not n:
        keys.step_slow()
        return out
    a, b, logits = CAP['a'].reshape(n, 9, -1)[:, 0], CAP['b'].reshape(n, 9, -1), CAP['logits'].reshape(n, 9)
    nn_ = np.array([m[2] for m in match]) - 1
    keep = [i for i, m in enumerate(match) if items[m[0]][0] != 'taps' and items[m[1]][0] != 'taps']
    cand_x = np.stack([P.DOTS['train'][ci[k], k] for k in range(9)])
    cand_live = torch.stack([b[int(np.flatnonzero(nn_ != k)[0]) if (nn_ != k).any() else 0, k] for k in range(9)])
    sink['n'].append(len(keep))
    if keep:
        ki = torch.as_tensor(keep)
        q_x = P.images([items[match[i][0]] for i in keep])
        t_x = P.images([items[match[i][1]] for i in keep])
        rows = torch.as_tensor(nn_[keep])
        sink['logits'].append(to_np(logits[ki]))
        sink['target'].append(nn_[keep].astype(np.int8))
        for name, q, t, c in (('live', a[ki], b[ki, rows], cand_live),
                              ('slow', keys.slow_codes(q_x), keys.slow_codes(t_x), keys.slow_codes(cand_x)),
                              ('frozen', keys.frozen_codes(q_x), keys.frozen_codes(t_x), keys.frozen_codes(cand_x))):
            sink[f'q_{name}'].append(to_np(q))
            sink[f't_{name}'].append(to_np(t))
            sink[f'c_{name}'].append(to_np(c))
    keys.step_slow()
    return out


def new_sink():
    return {k: [] for k in ('n', 'logits', 'target', *(f'{p}_{s}' for p in 'qtc' for s in KEYS))}


def flush(sink, path):
    arrays = {'n': np.array(sink['n'], np.int32)}
    for k, v in sink.items():
        if k != 'n':     # a context-only block has no image-image items, so memory has nothing to train on there
            arrays[k] = (np.concatenate(v) if k[0] != 'c' else np.stack(v)) if v else np.zeros((0,), np.float32)
    save_npz(path, **arrays)


@torch.no_grad()
def capture_eval(enc, model, keys, seed, withheld, sets, ev, unknown_ev):
    """Everything a memory variant needs to read an evaluation point: codes by key source, readers' logits, each candidate's count, the truth."""
    enc['image'].eval(); enc['sequence'].eval()
    code = G.codes_fn(enc, model)
    out = {}

    def add(name, qx, cx, counts, truth):
        q = code(qx, 'image')
        n, k = cx.shape[:2]
        cc = code(cx.reshape(-1, 28, 28), 'image').reshape(n, k, -1)
        logits = G.pair(model, q, cc, 1)
        out[f'{name}|logits'], out[f'{name}|counts'], out[f'{name}|truth'] = to_np(logits), counts.astype(np.int8), truth.astype(np.int8)
        for src, (qc, ccc) in {'live': (q, cc), **({} if keys.slow is None else {'slow': (keys.slow_codes(qx), keys.slow_codes(cx.reshape(-1, 28, 28)).reshape(n, k, -1)), 'frozen': (keys.frozen_codes(qx), keys.frozen_codes(cx.reshape(-1, 28, 28)).reshape(n, k, -1))})}.items():
            out[f'{name}|q_{src}'], out[f'{name}|c_{src}'] = to_np(qc), to_np(ccc)

    for tag, taught, renders in (('A', sets['A'], ev['A']), ('B', sets['B'], ev['B']), ('unknown', {c: 1 for c in unknown_ev[0]}, unknown_ev[1])):
        n, E = renders.shape[:2]
        rng = nc.stream('ap_eval_candidates', seed, tag)
        order = np.stack([rng.permutation(9) for _ in range(n * E)])
        j = rng.integers(P.EVAL_PER, size=n * E)
        imgs = np.stack([P.DOTS['succ'][int(k) + 1][j[t]] for t in range(n * E) for k in order[t]]).reshape(n * E, 9, 28, 28)
        add(f'recall/{tag}', renders.reshape(-1, 28, 28), imgs, order + 1, np.repeat([taught[c] for c in taught], E))
    for lang in P.LANGS:
        groups = {}
        for cnt in ns.NUMBERS:
            grp = 'withheld' if cnt in withheld.get(lang, ()) else 'paired'
            groups.setdefault(grp, []).append(cnt)
        for grp, cnts in groups.items():
            qs, cs, ks, ts = [], [], [], []
            for cnt in cnts:
                qs.append(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, cnt, 'idx')]])
                E = P.EVAL_PER
                cs.append(np.stack([P.EVAL[(lang, cnt, i)][0] for i in range(E)]))
                ks.append(np.stack([P.EVAL[(lang, cnt, i)][1] for i in range(E)]))
                ts.append(np.full(len(qs[-1]), cnt))
            add(f'match/{lang}/{grp}', np.concatenate(qs), np.concatenate(cs), np.concatenate(ks), np.concatenate(ts))
    enc['image'].train(); enc['sequence'].train()
    return out


def state_hash(enc, model):
    h = hashlib.sha256()
    for sd in (enc['image'].state_dict(), enc['sequence'].state_dict(), model.state_dict()):
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def unknown_set(seed, sets):
    """Symbols never taught, for reading confident errors; from the same pool, seeded by the seed alone."""
    used = set(sets['A']) | set(sets['B'])
    rest = [c for c in sorted(AP.POOL) if c not in used]
    pick = [int(c) for c in nc.stream('ltm_unknown', seed).permutation(rest)[:min(N, len(rest))]]
    return pick, np.stack([PS.renderings(c, AP.POOL[c], seed, 'eval', AP.EVAL_RENDERS) for c in pick])


def run_host(task):
    """mode 'full': sparse48 with the hook; 'plain': no hook, for gate 1; 'light': no cache and live codes only, for references."""
    arm, seed, mode = task
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    t0, c0 = time.perf_counter(), time.process_time()
    withheld = D.withheld_counts(seed)
    sets, row, ev = AP.build_sets(seed)
    unknown = unknown_set(seed, sets)
    enc, model, opt = LM.setup(arm, seed)
    LM.CURRENT['end'] = None
    rng = nc.stream('partA_train', seed, 'A-order')
    out_dir = cache_dir(arm, seed, mode)
    keys = Keys(enc, model, seed, mode) if mode != 'plain' else None
    hashes, sizes = {}, {'items': 0, 'updates': 0}

    def one(material, sink):
        if mode == 'full':
            hooked_update(enc, model, opt, material, rng, keys, sink)
            sizes['updates'] += 1
        else:
            LM.train_update(enc, model, opt, material, rng, True)

    def end_block(b, sink):
        hashes[str(b)] = state_hash(enc, model)
        if mode == 'full':
            sizes['items'] += int(np.sum(sink['n']))
            flush(sink, out_dir / f'train-{b}.npz')
        if mode != 'plain':
            save_npz(out_dir / f'eval-{b}.npz', **{k.replace('/', '__').replace('|', '___'): v for k, v in capture_eval(enc, model, keys, seed, withheld, sets, ev, unknown).items()})

    sink = new_sink()
    for b, lang in enumerate(P.LANGS):
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            one(SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), sink)
    end_block(3, sink)
    steps = {4: sets['A'], 5: sets['B'], 6: None}
    for b, taught in steps.items():
        sink = new_sink()
        crng = nc.stream('ap_context', seed, b)
        prng = nc.stream('ap_pairs', seed, b, {4: 'A', 5: 'B', 6: 'none'}[b])
        for _ in range(L):
            items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
            if taught is not None:
                AP.pair_items(items, match, taught, row, prng)
            one((items, pairs, match, *rest), sink)
        end_block(b, sink)
    if mode == 'full':
        flat = {}
        for tag in ('A', 'B'):
            cps = list(sets[tag])
            tr = X.DATA[('pairsym', 'train')][0].reshape(-1, AP.TRAIN_RENDERS, 28, 28)
            flat[f'{tag}_keys'] = to_np(keys.frozen_codes(np.stack([tr[row[c]][0] for c in cps])))
            flat[f'{tag}_counts'] = np.array([sets[tag][c] for c in cps], np.int8)
        save_npz(out_dir / 'flat.npz', **flat)
    info = {'hashes': hashes, **sizes, 'wall_s': time.perf_counter() - t0, 'cpu_s': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cache_bytes': sum(f.stat().st_size for f in out_dir.glob('*')) if out_dir.exists() else 0}
    if DEV.ACTIVE:
        info['gpu_peak_bytes'] = torch.cuda.max_memory_allocated()
    save_json(out_dir.parent / f'{arm}-seed{seed}-{mode}-N{N}.json', info)
    print(arm, seed, mode, round(info['wall_s']), 's', info['items'], 'items', info['cache_bytes'] // 10 ** 6, 'MB', flush=True)
    return task, info


def fit_encoders(seeds):
    assert not SMOKE
    torch.set_num_threads(4)
    DEV.install()
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    syms = TK.symbol_splits()
    for seed in seeds:
        t0 = time.perf_counter()
        TK.train(c['arch'], c['key'], seed, syms)
        print(seed, round(time.perf_counter() - t0, 1), flush=True)


def prepare():
    X.prepare(); P.prepare(); D.prepare()
    LM.install()
    PSL = PS.survivors()
    AP.POOL.update(PSL)


def run_hosts(tasks):
    t0 = time.perf_counter()
    prepare()
    tasks = [t for t in tasks if not (cache_dir(*t).parent / f'{t[0]}-seed{t[1]}-{t[2]}-N{N}.json').exists()]
    with mp.get_context('fork').Pool(min(HOST_WORKERS, max(len(tasks), 1))) as pool:
        res = list(pool.imap_unordered(run_host, tasks))
    save_json(f'{REPORT}-{STAGE}-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'runs': [list(t) for t, _ in res]}))


def hosts():
    tasks = [('sparse48', s, 'full') for s in PILOT_SEEDS] + [('mixed', s, 'light') for s in PILOT_SEEDS] + [('sparse48', PILOT_SEEDS[0], 'plain')]
    run_hosts(tasks)


# ---------------------------------------------------------------- replay

_c = {}


def load_cache(seed, arm='sparse48', mode='full'):
    key = (seed, arm, mode)
    if key not in _c:
        d = cache_dir(arm, seed, mode)
        train = None
        if mode == 'full':
            train = {b: dict(np.load(d / f'train-{b}.npz')) for b in BLOCKS}
        ev = {b: dict(np.load(d / f'eval-{b}.npz')) for b in BLOCKS}
        _c[key] = {'train': train, 'eval': ev}
    return _c[key]


def eval_name(name, field):
    return f"{name.replace('/', '__')}___{field}"


def read_set(ev, name, src, mem):
    logits = torch.from_numpy(ev[eval_name(name, 'logits')])
    counts = torch.from_numpy(ev[eval_name(name, 'counts')]).long()
    truth = torch.from_numpy(ev[eval_name(name, 'truth')]).long()
    alone = ltm.accuracy(logits, counts, truth)
    if mem is None:
        return {'readers': alone, 'combined': alone, 'memory': None}
    q, c = torch.from_numpy(ev[eval_name(name, f'q_{src}')]), torch.from_numpy(ev[eval_name(name, f'c_{src}')])
    with torch.no_grad():
        term = mem.term(q, c)
    return {'readers': alone, 'combined': ltm.accuracy(logits + term, counts, truth), 'memory': ltm.accuracy(term, counts, truth),
            'changed': float(((logits + term).argmax(1) != logits.argmax(1)).float().mean())}


def match_names():
    return [f'match/{lang}/{g}' for lang in P.LANGS for g in ('paired', 'withheld')]


def readings(ev, src, mem):
    out = {t: read_set(ev, f'recall/{t}', src, mem) for t in ('A', 'B', 'unknown')}
    for name in match_names():
        if eval_name(name, 'logits') in ev:
            out[name] = read_set(ev, name, src, mem)
    tr = [out[f'match/{k}']['combined'] for k in TRANSFER]
    out['answer_transfer'] = float(np.mean(tr))
    out['answer_transfer_readers'] = float(np.mean([out[f'match/{k}']['readers'] for k in TRANSFER]))
    return out


def replay(cfg, seed):
    torch.set_num_threads(1)
    cache = load_cache(seed)
    src = cfg['key']
    d = KEYS[src]
    mem = ltm.Memory(cfg['form'], d, d, cfg['projections'], cfg['eta'], cfg['beta'], cfg['C'], cfg['p'], seed=int(nc.stream('ltm_memory', seed, cfg['form'], src, cfg['projections']).integers(1 << 30)))
    q_all, reads = 0, {}
    t0 = time.process_time()
    losses = {}
    for b in BLOCKS:
        tr = cache['train'][b]
        n = tr['n']
        offs = np.r_[0, np.cumsum(n)]
        q_, t_, c_, lg, tg = tr[f'q_{src}'], tr[f't_{src}'], tr[f'c_{src}'], tr['logits'], tr['target']
        cu = 0
        ls = []
        for u, m in enumerate(n):
            if not m:
                continue
            a, z = offs[u], offs[u + 1]
            C = np.repeat(c_[cu][None], m, 0).copy()
            C[np.arange(m), tg[a:z]] = t_[a:z]
            ls.append(mem.update(torch.from_numpy(q_[a:z]), torch.from_numpy(C), torch.from_numpy(lg[a:z]), torch.from_numpy(tg[a:z].astype(np.int64))))
            cu += 1
        losses[str(b)] = float(np.mean(ls[-200:])) if ls else None
        reads[str(b)] = readings(cache['eval'][b], src, mem)
    return {'reads': reads, 'losses': losses, 's': float(mem.s.detach()), 'floats': ltm.param_floats(cfg['form'], d, d), 'cpu_s': time.process_time() - t0}


def replay_key(cfg, seed):
    j = json.dumps(cfg, sort_keys=True)
    return f"{hashlib.sha256(j.encode()).hexdigest()[:12]}-seed{seed}-N{N}"


class Replay:
    def __call__(self, task):
        cfg, seed = task
        path = RUNS / 'replays' / f'{replay_key(cfg, seed)}.json.gz'
        if path.exists():
            try:
                return task, json.loads(gzip.decompress(path.read_bytes()))
            except (OSError, ValueError, EOFError):
                pass
        out = DUR.roundtrip(replay(cfg, seed))
        DUR.atomic_write(path, gzip.compress(json.dumps(out, sort_keys=True).encode()))
        return task, out


def run_replays(tasks):
    with mp.get_context('fork').Pool(min(WORKERS, max(len(tasks), 1))) as pool:
        return {replay_key(c, s): r for (c, s), r in pool.imap_unordered(Replay(), tasks)}


def flat_read(seed, b):
    """`flat-store`: each pair's frozen key and count, written once; A from block 4, B from block 5."""
    d = cache_dir('sparse48', seed, 'full')
    z = np.load(d / 'flat.npz')
    ev = load_cache(seed)['eval'][b]
    keys, vals = [], []
    if b >= 4:
        keys.append(z['A_keys']); vals.append(z['A_counts'])
    if b >= 5:
        keys.append(z['B_keys']); vals.append(z['B_counts'])
    if not keys:
        return {'A': None, 'B': None}
    K, V = np.vstack(keys), np.concatenate(vals)
    out = {}
    for t in ('A', 'B'):
        q = ev[eval_name(f'recall/{t}', 'q_frozen')]
        out[t] = float((V[(q @ K.T).argmax(1)] == ev[eval_name(f'recall/{t}', 'truth')]).mean())
    return out


# ---------------------------------------------------------------- gates

@torch.no_grad()
def prototype_acc(cache_ev, name_prefix):
    return None


def gates():
    t0 = time.perf_counter()
    out = {}
    for seed in PILOT_SEEDS:
        d = cache_dir('sparse48', seed, 'full')
        info = json.load(open(d.parent / f'{d.name}.json'))
        r = {'hashes': info['hashes'], 'items_per_update': info['items'] / max(info['updates'], 1), 'cache_bytes': info['cache_bytes'], 'host_wall_s': info['wall_s']}
        plain = d.parent / f'sparse48-seed{seed}-plain-N{N}.json'
        if plain.exists():
            pi = json.load(open(plain))
            r['gate1_identical'] = pi['hashes'] == info['hashes']
            r['plain_hashes'] = pi['hashes']
        ev = load_cache(seed)['eval'][6]
        # gate 2: can the frozen encoder tell counts and symbols apart?
        q = ev[eval_name('recall/A', 'q_frozen')]
        cc = ev[eval_name('recall/A', 'c_frozen')]
        counts = ev[eval_name('recall/A', 'counts')]
        truth = ev[eval_name('recall/A', 'truth')]
        flat = cc.reshape(-1, cc.shape[-1])
        labels = counts.reshape(-1)
        half = len(flat) // 2
        protos = np.stack([flat[:half][labels[:half] == k].mean(0) for k in range(1, 10)])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        r['gate2_count_identity'] = float(((flat[half:] @ protos.T).argmax(1) + 1 == labels[half:]).mean())
        E = AP.EVAL_RENDERS
        sym = np.repeat(np.arange(len(q) // E), E)
        pm = np.stack([q[sym == k].mean(0) for k in range(len(q) // E)])
        pm /= np.linalg.norm(pm, axis=1, keepdims=True)
        r['gate2_symbol_identity_leave_in'] = float(((q @ pm.T).argmax(1) == sym).mean())
        # gate 3: silence
        mem = ltm.Memory('mlp', 32, 32, seed=seed)
        name = 'recall/A'
        lg = torch.from_numpy(ev[eval_name(name, 'logits')])
        t = mem.combined(torch.from_numpy(q), torch.from_numpy(cc), lg)
        r['gate3_silent'] = bool((t == lg).all())
        # gate 4: storage ceiling, at the end of block 6
        r['gate4_flat'] = {str(b): flat_read(seed, b) for b in (4, 5, 6)}
        # gate 5: audit leaves state alone; shuffled targets score chance
        h = mem.state_hash()
        readings(ev, 'frozen', mem)
        r['gate5_audit_unchanged'] = h == mem.state_hash()
        rng = nc.stream('ltm_shuffle', seed)
        r['gate5_shuffled'] = float((counts[np.arange(len(q)), np.argmax(ev[eval_name(name, 'logits')], 1)] == rng.permutation(truth)).mean())
        reads = readings(ev, 'live', None)
        r['sparse48'] = {k: reads[k]['readers'] for k in ('A', 'B')}
        r['sparse48']['answer_transfer'] = reads['answer_transfer_readers']
        r['sparse48']['by_block_A'] = {str(b): read_set(load_cache(seed)['eval'][b], 'recall/A', 'live', None)['readers'] for b in BLOCKS}
        r['sparse48']['by_block_B'] = {str(b): read_set(load_cache(seed)['eval'][b], 'recall/B', 'live', None)['readers'] for b in BLOCKS}
        out[str(seed)] = r
    save_json(f'{REPORT}-gates.json', out)
    save_json(f'{REPORT}-gates-runtime.json', runtime({'wall_s': time.perf_counter() - t0}))
    print(json.dumps({k: {x: y for x, y in v.items() if 'hashes' not in x} for k, v in out.items()}, indent=1, default=float))


# ---------------------------------------------------------------- selection

def score(cfg, seeds, results):
    rows = [results[replay_key(cfg, s)]['reads']['6'] for s in seeds]
    base = [load_baseline(s) for s in seeds]
    a, b, t, t0 = [], [], [], []
    for r, b0 in zip(rows, base):
        a.append(r['A']['combined']); b.append(r['B']['combined']); t.append(r['answer_transfer']); t0.append(b0)
    return {'retained_A': float(np.mean(a)), 'retained_B': float(np.mean(b)), 'answer_transfer': float(np.mean(t)), 'sparse48_transfer': float(np.mean(t0)), 'objective': float(np.mean(a + b)), 'eligible': float(np.mean(t)) >= float(np.mean(t0)) - ELIGIBLE}


def load_baseline(seed):
    return readings(load_cache(seed)['eval'][6], 'live', None)['answer_transfer_readers']


def better(a, b):
    """Is score a preferred to score b: eligible first, then objective; if neither is eligible, the smaller shortfall."""
    if a['eligible'] != b['eligible']:
        return a['eligible']
    if a['eligible']:
        return a['objective'] > b['objective']
    return a['answer_transfer'] > b['answer_transfer']


def best_of(cands, seeds, results):
    cur = None
    for c in cands:
        if cur is None or better(score(c, seeds, results), score(cur, seeds, results)):
            cur = c
    return cur


def stage(cands, seeds, results):
    results.update(run_replays([(c, s) for c in cands for s in seeds]))
    return best_of(cands, seeds, results)


def pilot():
    t0 = time.perf_counter()
    seeds = PILOT_SEEDS
    results, log = {}, []
    cur = dict(START)

    def note(name, cands, chosen):
        log.append({'stage': name, 'candidates': [{'cfg': c, 'score': score(c, seeds, results)} for c in cands], 'chosen': chosen})
        print(name, {c[name] if name in c else '': round(score(c, seeds, results)['objective'], 3) for c in cands}, '->', chosen.get(name), {k: round(v, 3) for k, v in score(chosen, seeds, results).items() if k != 'eligible'}, flush=True)
    cands = [{**cur, 'form': f} for f in ltm.FORMS]
    cur = stage(cands, seeds, results); note('form', cands, cur)
    cands = [{**cur, 'projections': p} for p in ('learned', 'fixed')]
    cur = stage(cands, seeds, results); note('projections', cands, cur)
    cands = [{**cur, 'key': k} for k in ('live', 'slow', 'frozen')]
    best = stage(cands, seeds, results)
    pref = None
    for k in ('live', 'slow'):
        c = {**cur, 'key': k}
        if score(c, seeds, results)['eligible'] and score(c, seeds, results)['objective'] >= score(best, seeds, results)['objective'] - NEAR:
            pref = c
            break
    cur = pref or best; note('key', cands, cur)
    for name, values in SETTINGS:
        if name == 'p' and cur['form'] != 'k-sparse':
            continue
        cands = [{**cur, name: v} for v in values]
        cur = stage(cands, seeds, results); note(name, cands, cur)
    final = {str(s): results[replay_key(cur, s)] for s in seeds}
    save_json(f'{REPORT}-selection.json', {'candidate': cur, 'score': score(cur, seeds, results), 'log': log, 'final': final, 'sparse48': {str(s): {b: {k: read_set(load_cache(s)['eval'][int(b)], f'recall/{k}', 'live', None)['readers'] for k in 'AB'} for b in map(str, BLOCKS)} for s in seeds},
                                           'flat': {str(s): {str(b): flat_read(s, b) for b in (4, 5, 6)} for s in seeds}})
    save_json(f'{REPORT}-pilot-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'replays': len(results), 'worker_cpu_s': sum(r['cpu_s'] for r in results.values())}))
    print('candidate', cur, score(cur, seeds, results))


# ---------------------------------------------------------------- confirmation

def frozen():
    return 'Status: **frozen' in CONTRACT.read_text()


def ablations(c):
    out = [{**c, 'projections': 'fixed' if c['projections'] == 'learned' else 'learned'}]
    out += [{**c, 'key': k} for k in KEYS if k != c['key']]
    out += [{**c, 'form': f} for f in ltm.FORMS if f != c['form']]
    out.append({**c, 'C': 16 if c['C'] == 1 else 1})
    return out


def confirm_hosts():
    if not frozen():
        raise SystemExit('the contract is not frozen: seeds 394-403 stay unbuilt')
    fit_encoders(CONFIRM_SEEDS)
    run_hosts([('sparse48', s, 'full') for s in CONFIRM_SEEDS] + [('mixed', s, 'light') for s in CONFIRM_SEEDS])


def confirm():
    if not frozen():
        raise SystemExit('the contract is not frozen: confirmation refuses to run')
    t0 = time.perf_counter()
    c = json.load(open(f'{REPORT}-selection.json'))['candidate']
    results = run_replays([(x, s) for x in [c] + ablations(c) for s in CONFIRM_SEEDS])
    DUR.atomic_write(f'{REPORT}-confirm.json.gz', gzip.compress(json.dumps(results, sort_keys=True, default=float).encode()))
    save_json(f'{REPORT}-confirm-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'replays': len(results), 'worker_cpu_s': sum(r['cpu_s'] for r in results.values())}))


def boot(d, rng):
    d = np.asarray(d)
    m = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse():
    res = json.loads(gzip.decompress(open(f'{REPORT}-confirm.json.gz', 'rb').read()))
    c = json.load(open(f'{REPORT}-selection.json'))['candidate']
    rng = nc.stream('ltm_bootstrap')

    def r(cfg, s, field):
        x = res[replay_key(cfg, s)]['reads']['6']
        return x['answer_transfer'] if field == 'answer_transfer' else x[field]['combined']

    base = {s: readings(load_cache(s)['eval'][6], 'live', None) for s in CONFIRM_SEEDS}
    gA = boot([r(c, s, 'A') - base[s]['A']['readers'] for s in CONFIRM_SEEDS], rng)
    gB = boot([r(c, s, 'B') - base[s]['B']['readers'] for s in CONFIRM_SEEDS], rng)
    tr = boot([r(c, s, 'answer_transfer') - base[s]['answer_transfer_readers'] for s in CONFIRM_SEEDS], rng)
    near = boot([r(c, s, 'A') - flat_read(s, 6)['A'] for s in CONFIRM_SEEDS], rng)
    retains = gA[0] >= .15 and gA[1] > 0 and gB[0] >= .15 and gB[1] > 0
    verdict = 'holds' if tr[1] >= -.05 else 'lost' if tr[0] <= -.05 and tr[2] < 0 else 'inconclusive'
    if retains and verdict == 'holds':
        outcome = 'long-term-works' if near[1] >= -.15 else 'retains-store-gap'
    elif retains and verdict == 'lost':
        outcome = 'retains-at-transfer-cost'
    elif retains:
        outcome = 'retains-transfer-inconclusive'
    else:
        outcome = 'no-retention-shown'

    def three(d, margin=.05):
        return {'difference': d[0], 'interval': [d[1], d[2]], 'reading': 'better' if d[0] >= margin and d[1] > 0 else 'equivalent' if d[1] >= -margin and d[2] <= margin else 'inconclusive'}
    secondary = {json.dumps(x, sort_keys=True): {f: three(boot([r(c, s, f) - r(x, s, f) for s in CONFIRM_SEEDS], rng)) for f in ('A', 'B', 'answer_transfer')} for x in ablations(c)}
    save_json(f'{REPORT}-analysis.json', {'outcome': outcome, 'candidate': c, 'retained_A_gain': gA, 'retained_B_gain': gB, 'answer_transfer_difference': tr, 'transfer_reading': verdict, 'near_store': near, 'secondary': secondary})
    print('outcome', outcome, gA, gB, tr, near)


if __name__ == '__main__':
    if STAGE in ('encoders', 'hosts', 'gates', 'pilot', 'confirm-hosts', 'confirm', 'analyse'):
        {'encoders': lambda: fit_encoders(PILOT_SEEDS), 'hosts': hosts, 'gates': gates, 'pilot': pilot, 'confirm-hosts': confirm_hosts, 'confirm': confirm, 'analyse': analyse}[STAGE]()
