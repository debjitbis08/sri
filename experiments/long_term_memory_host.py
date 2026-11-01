"""Long-term memory v1 (docs/long-term-memory-v1-contract.md): the host, its capture hook and the trajectory cache. Kept apart from replay, so that
changing replay does not invalidate host runs. Nothing here depends on memory.
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
HOST_WORKERS = int(os.environ.get('LTM_HOST_WORKERS', 3))
PILOT_SEEDS = (299,) if SMOKE else (392, 393)
CONFIRM_SEEDS = tuple(range(394, 404))
N, L = AP.N, AP.L
BLOCKS = (3, 4, 5, 6)
SLOW_HALF_LIFE = 2000
KEYS = {'live': 16, 'slow': 16, 'frozen': 32}
HOST_SOURCES = (AP, LM, P, D, G, SP, TP, FI, DEV, TK, X)


def save_json(path, obj):
    DUR.atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=float).encode())


def save_npz(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    DUR.atomic_write(path, buf.getvalue())


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'stage': STAGE, 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
            'contract_sha256': nc.sha256(CONTRACT), 'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(), 'host_workers': HOST_WORKERS,
            'dtype': 'float32', **DEV.info(), 'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, **extra}


def cache_dir(arm, seed, mode):
    return RUNS / 'cache' / f'{arm}-seed{seed}-{mode}-N{N}'


def info_path(arm, seed, mode):
    return cache_dir(arm, seed, mode).parent / f'{arm}-seed{seed}-{mode}-N{N}.json'


def encoder_file(seed):
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    return TK.RUNS / 'fits' / f"{c['arch']}-key{c['key']}-seed{seed}.pt"


def host_provenance(arm, seed, mode):
    """What a host run's output depends on: this module, the host's modules, the encoder for a full run, the software and device."""
    out = {'host_module': nc.sha256(__file__), 'sources': {m.__name__: nc.sha256(m.__file__) for m in HOST_SOURCES}, 'task': [arm, seed, mode], 'N': N, 'L': L}
    if mode == 'full':
        out['encoder'] = nc.sha256(encoder_file(seed))
    return out


def host_env():
    return {'torch': torch.__version__, 'device': os.environ.get('NFI_DEVICE', 'cpu')}


def host_valid(arm, seed, mode, same_env=True):
    """A finished host run is reused only if what it depends on is unchanged and its files are as written. Reruns also need the same software and
    device; readers of the cache, which run under another environment, pass same_env=False."""
    p = info_path(arm, seed, mode)
    if not p.exists():
        return False
    info = json.load(open(p))
    if info.get('provenance') != host_provenance(arm, seed, mode) or (same_env and info.get('env') != host_env()):
        return False
    d = cache_dir(arm, seed, mode)
    return all((d / f).exists() and nc.sha256(d / f) == h for f, h in info.get('files', {}).items())


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
        sink['n'].append(0)
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
    files = {f.name: nc.sha256(f) for f in sorted(out_dir.glob('*'))} if out_dir.exists() else {}
    info = {'hashes': hashes, **sizes, 'provenance': host_provenance(arm, seed, mode), 'env': host_env(), 'files': files}
    save_json(info_path(arm, seed, mode), info)
    timing = {'wall_s': time.perf_counter() - t0, 'cpu_s': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'cache_bytes': sum(f.stat().st_size for f in out_dir.glob('*')) if out_dir.exists() else 0, **({'gpu_peak_bytes': torch.cuda.max_memory_allocated()} if DEV.ACTIVE else {})}
    save_json(info_path(arm, seed, mode).with_suffix('.timing.json'), timing)
    print(arm, seed, mode, round(timing['wall_s']), 's', info['items'], 'items', timing['cache_bytes'] // 10 ** 6, 'MB', flush=True)
    return task, info, timing


def fit_encoders(seeds):
    assert not SMOKE
    torch.set_num_threads(4)
    DEV.install()
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    syms = TK.symbol_splits()
    for seed in seeds:
        if encoder_file(seed).exists():
            continue
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
    assert not torch.cuda.is_initialized(), 'CUDA must not be initialised before the pool forks: fit encoders in their own process'  # is_available() itself would break the fork
    prepare()
    todo = [t for t in tasks if not host_valid(*t)]
    with mp.get_context('fork').Pool(min(HOST_WORKERS, max(len(todo), 1))) as pool:
        res = list(pool.imap_unordered(run_host, todo))
    save_json(f'{REPORT}-{STAGE}-runtime.json', runtime({'wall_s': time.perf_counter() - t0, 'executed': [list(t) for t, _, _ in res], 'reused': [list(t) for t in tasks if t not in todo], 'timing': {str(t): x for t, _, x in res}}))


def hosts():
    tasks = [('sparse48', s, 'full') for s in PILOT_SEEDS] + [('mixed', s, 'light') for s in PILOT_SEEDS] + [('sparse48', PILOT_SEEDS[0], 'plain')]
    run_hosts(tasks)


