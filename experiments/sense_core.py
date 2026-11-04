"""Sense integration plan, step 8: does frozen sight support the core's counts, symbols and relational use? (docs/sense-core-v1-pilot.md)

The core is `sparse48`, trained and read exactly as in the limited-memory study's main stage (experiments.limited_memory run_main: three scripts in
blocks, a delay and relearning, with matching, the senses evaluation and the eight transfer measures at every snapshot). Only the image encoder changes:
  existing        the core's own image encoder (a small CNN trained with the core)
  joined-mlp      frozen DINOv2-S patch mean and frozen TrOCR-small class token, each standardized, concatenated, then a trained MLP 768 -> 128 -> out,
                  the same shape as the existing encoder's last two layers
  joined-linear   the same frozen blocks, then one trained linear layer
  dinov2-mlp      DINOv2-S alone, then the MLP
The core gets the raw frozen blocks through a small projection, not memory's centred, query-weighted code. Memory is not used here; its baseline stays
kept+frozen (docs/sense-stable-v1-pilot.md).

Frozen codes are computed once for every image the core sees (stage `codes`) and looked up by image content during training; an image not in the
cache is encoded live and counted.

Usage (the cache is built in order: codes, then capture, then codes again, which adds only what is new):
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.sense_core pilot codes
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.sense_core pilot capture
  NFI_DEVICE=cuda SC_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_core pilot main
The calibrated rerun (statistics from block 1's training images only, docs/sense-core-v1-pilot.md):
  NFI_DEVICE=cuda SC_WORKERS=3 SC_ARMS=joined-mlp-calibrated SC_OUT=calibrated /data/venvs/sri-cuda/bin/python -m experiments.sense_core pilot main
"""
import hashlib
import json
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0], 'pilot', 'main']  # experiments.limited_memory reads its stage from argv at import
os.environ.setdefault('NFI_TAG', '-sense-core')  # so nothing is written over the limited-memory study's own runs
import numpy as np
import torch
from torch import nn

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_systems as X
from experiments import shared_core_stage0 as S0
from src import senses as SN
sys.argv = _argv

NAME = 'sense-core-v1' + ('-smoke' if X.SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CODES = X.ROOT / 'runs' / 'sense-core-v1' / 'codes.npz'  # shared by smoke and pilot; smoke images it lacks are encoded live
MODELS = ('dinov2-s', 'trocr-s')
CALIBRATION = ('english/train', 'dots/train')  # block 1's training images: what the system has before anything else is taught
ARMS = tuple(os.environ.get('SC_ARMS', 'existing,joined-mlp,joined-linear,dinov2-mlp').split(','))
# encoder arm -> (frozen models, head, statistics). The first pilot's three frozen arms fit statistics on every cached image; joined-mlp-calibrated fits
# them on block 1's training images only, then freezes them.
ENCODERS = {'joined-mlp': (MODELS, 'mlp', 'all-cached'), 'joined-linear': (MODELS, 'linear', 'all-cached'), 'dinov2-mlp': (('dinov2-s',), 'mlp', 'all-cached'),
            'joined-mlp-calibrated': (MODELS, 'mlp', CALIBRATION)}
SEEDS = tuple(int(s) for s in os.environ.get('SC_SEEDS', '300,301,302,303').split(','))
WORKERS = int(os.environ.get('SC_WORKERS', 3))
HIDDEN = 128


def image_key(x):
    return int.from_bytes(hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float32).tobytes(), digest_size=8).digest(), 'little')


def sources():
    """Every image array the core's training and reads draw from."""
    out = {}
    for lang in (*P.LANGS, P.TRANSFER):  # the transfer script, Thai digits, is read in evaluation
        for split in ('train', 'eval'):
            out[f'{lang}/{split}'] = X.DATA[(lang, split)][0]
    out['dots/train'] = P.DOTS['train'].reshape(-1, 28, 28)
    for k, v in P.DOTS['succ'].items():
        out[f'dots/succ/{k}'] = v
    out['eval-items'] = np.concatenate([v[0] for k, v in sorted(P.EVAL.items(), key=str) if isinstance(v, tuple) and getattr(v[0], 'ndim', 0) == 3])  # candidate sets built at prepare time
    captured = RUNS / 'captured-eval.npy'  # the remaining evaluation images, the same for every seed, captured once from a snapshot (see docs)
    if captured.exists():
        out['eval-captured'] = np.load(captured)
    return out


def codes():
    """Frozen codes of every source image, once. Sources already in the cache are kept, so adding a source encodes only it."""
    t0, c0 = time.perf_counter(), time.process_time()
    S0.prepare()
    DEV.install()
    sense = SN.Sense(MODELS, DEV.NAME if DEV.ACTIVE else 'cpu', chunk=64)
    keys, parts, names = [], {n: [] for n in MODELS}, []
    have = set()
    if CODES.exists():
        z = np.load(CODES)
        keys.append(z['keys'])
        for n in MODELS:
            parts[n].append(z[n])
        have = set(int(k) for k in z['keys'])
    for name, x in sources().items():
        if all(image_key(r) in have for r in x[:64]) and image_key(x[-1]) in have:
            names.append((name, 0))
            continue
        x = np.ascontiguousarray(x, dtype=np.float32)
        for i in range(0, len(x), 512):
            f = sense(x[i:i + 512])
            for n in MODELS:
                parts[n].append(f[n])
        keys.append(np.array([image_key(r) for r in x], np.uint64))
        names.append((name, len(x)))
        print(name, len(x), f'{time.perf_counter() - t0:.0f} s', flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = RUNS / 'codes.tmp.npz'
    np.savez(tmp, keys=np.concatenate(keys), **{n: np.concatenate(parts[n]) for n in MODELS})
    os.replace(tmp, CODES)
    S0.save_json(f'{REPORT}-codes-{os.environ.get("SC_PART", "part1")}-runtime.json', S0.runtime({'phase': 'frozen codes of the core\'s images', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                                            'sources': names, 'senses': sense.describe()}))


def fit_stats(models, table, names):
    """Per-coordinate mean and standard deviation of the frozen codes of the named sources, fitted once and then frozen with the projection.
    'all-cached' (the first pilot) fits on every cached image, evaluation and later blocks included, which is transductive."""
    src = sources()
    if names == 'all-cached':
        raw = np.concatenate([table[n] for n in models], 1)
        used = [('all-cached', len(raw))]
    else:
        rows = [table['index'][image_key(r)] for name in names for r in np.ascontiguousarray(src[name], dtype=np.float32)]
        raw = np.concatenate([table[n][rows] for n in models], 1)
        used = [(name, len(src[name])) for name in names]
    mean, std = raw.mean(0), raw.std(0) + 1e-6
    return {'mean': mean, 'std': std, 'provenance': {'models': {n: SN.get(n).describe() for n in models}, 'input': 'raw 28x28 core image, as each model\'s inputs spec',
                                                     'calibration': used, 'images': int(len(raw)),
                                                     'sha256': hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()}}


class FrozenImage(nn.Module):
    """Image encoder for the core: frozen codes looked up by image content, standardized per coordinate with fixed statistics, concatenated, then a
    small trained projection. The statistics are buffers and their provenance (models, weights, calibration set) is extra state, so a checkpoint
    carries everything that defines the projection's input; the code table itself is data and is not saved."""

    def __init__(self, models, head, out, table, stats=None):
        super().__init__()
        self.models = tuple(models)
        self.index, self.raw = table['index'], np.concatenate([table[n] for n in self.models], 1)
        width = self.raw.shape[1]
        if stats is None:  # smoke before a cache exists: fitted on the first images seen
            stats = {'mean': np.zeros(width, np.float32), 'std': np.ones(width, np.float32), 'provenance': None}
        self.register_buffer('mean', torch.as_tensor(np.asarray(stats['mean'], np.float32)).clone())
        self.register_buffer('std', torch.as_tensor(np.asarray(stats['std'], np.float32)).clone())
        self.provenance = stats['provenance']
        self.feats = None  # standardized device table, rebuilt when the table or the statistics change
        self.head = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, out)) if head == 'mlp' else nn.Linear(width, out)
        self.misses, self.lookups, self.live = 0, 0, None

    def get_extra_state(self):
        return {'models': list(self.models), 'provenance': self.provenance}

    def set_extra_state(self, state):
        if list(state['models']) != list(self.models):
            raise ValueError(f'checkpoint is for {state["models"]}, encoder is {list(self.models)}')
        self.provenance = state['provenance']

    def _load_from_state_dict(self, *a, **k):
        super()._load_from_state_dict(*a, **k)
        self.feats = None

    def lookup(self, x):
        a = x.detach().cpu().numpy().astype(np.float32, copy=False)
        rows = np.empty(len(a), np.int64)
        missing = []
        for i, r in enumerate(a):
            j = self.index.get(image_key(r))
            if j is None:
                missing.append(i)
            else:
                rows[i] = j
        self.lookups += len(a)
        if missing:  # encode live and add to the table; the statistics do not change
            self.misses += len(missing)
            if self.live is None:
                self.live = SN.Sense(self.models, DEV.NAME if DEV.ACTIVE else 'cpu', chunk=64)
            f = self.live(np.ascontiguousarray(a[missing]))
            new = np.concatenate([f[n] for n in self.models], 1)
            base = len(self.raw)
            self.raw = np.concatenate([self.raw, new])
            for k, i in enumerate(missing):
                self.index[image_key(a[i])] = base + k
                rows[i] = base + k
            if self.provenance is None:
                self.mean.copy_(torch.from_numpy(self.raw.mean(0))), self.std.copy_(torch.from_numpy(self.raw.std(0) + 1e-6))
                self.provenance = {'calibration': 'first images seen (smoke only)'}
            self.feats = None
        return rows

    def add_codes(self, keys, raw):
        """Add frozen codes (n, width) for images with the given content keys, so they are looked up rather than encoded live. Statistics do not change."""
        new = [i for i, k in enumerate(keys) if int(k) not in self.index]
        base = len(self.raw)
        self.raw = np.concatenate([self.raw, np.asarray(raw, np.float32)[new]])
        for t, i in enumerate(new):
            self.index[int(keys[i])] = base + t
        self.feats = None

    def from_codes(self, raw):
        """The projection of raw frozen codes (n, width) given directly, for example a value recalled from memory: standardized with the fixed statistics,
        then the head. No image and no lookup."""
        m, s = self.mean.cpu().numpy(), self.std.cpu().numpy()
        z = torch.from_numpy(((np.asarray(raw, np.float32) - m) / s).astype(np.float32))
        return self.head(z.to(self.mean.device))

    def forward(self, x):
        rows = self.lookup(x)
        if self.feats is None:
            m, s = self.mean.cpu().numpy(), self.std.cpu().numpy()
            self.feats = torch.from_numpy(((self.raw - m) / s).astype(np.float32))
            if DEV.ACTIVE:
                self.feats = self.feats.to(DEV.NAME)
        return self.head(self.feats[torch.as_tensor(rows, device=self.feats.device)])


TABLE = {}


def table(models):
    if 'keys' not in TABLE:
        if CODES.exists():
            z = np.load(CODES)
            keys = z['keys']
            TABLE.update(keys=keys, **{n: z[n] for n in MODELS})
        else:  # smoke before the cache exists: everything encoded live
            keys = np.zeros(0, np.uint64)
            TABLE.update(keys=keys, **{n: np.zeros((0, SN.get(n).dim), np.float32) for n in MODELS})
        TABLE['index'] = {int(k): i for i, k in enumerate(keys)}
    return {'index': dict(TABLE['index']), **{n: TABLE[n] for n in models}}


ORIG_MAKE_MODEL = LM.make_model


def make_model(seed, arm, encoder):
    enc, model = ORIG_MAKE_MODEL(seed, arm)
    if encoder != 'existing':
        models, head, cal = ENCODERS[encoder]
        out = enc['image'].out.out_features
        t = table(models)
        stats = fit_stats(models, t, cal) if len(t['index']) else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(S0.nc.stream('sc_head', seed, encoder.replace('-calibrated', '')).integers(2 ** 62)))  # the same initial head
            enc['image'] = FrozenImage(models, head, out, t, stats)
    return enc, model


def run(task):
    encoder, seed = task
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    LM.RUNS = RUNS / encoder
    X.make_model = lambda s: DEV.build(lambda: make_model(s, 'sparse48', encoder))
    key, res, rt = LM.run_main(('sparse48', seed))
    if encoder != 'existing':
        rt['image_lookups'] = None
    if DEV.ACTIVE:
        rt['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return f'{encoder}/seed{seed}', {**res, 'encoder': encoder}, rt


def counted_run(task):
    """run, plus the encoder's cache misses (read from the encoder after training)."""
    holder = {}
    orig = LM.setup

    def setup(arm, seed):
        enc, model, opt = orig(arm, seed)
        holder['enc'] = enc
        return enc, model, opt
    LM.setup = setup
    try:
        key, res, rt = run(task)
    finally:
        LM.setup = orig
    e = holder['enc']['image']
    if isinstance(e, FrozenImage):
        rt['image_lookups'], rt['image_cache_misses'] = e.lookups, e.misses
    rt['image_encoder_parameters'] = int(sum(p.numel() for p in holder['enc']['image'].parameters()))
    return key, res, rt


def capture():
    """The evaluation images that are in no source array: one block-3 snapshot of an untrained frozen-encoder core, every image it reads that the sources
    lack, saved for `codes` to add. They are the same for every seed (checked on seeds 300 and 301)."""
    S0.prepare()
    LM.install()
    known = {image_key(r) for x in sources().values() for r in x}
    seen = {}

    def lookup(self, x):
        a = x.detach().cpu().numpy()
        for r in a:
            k = image_key(r)
            if k not in known:
                seen[k] = r.copy()
        return np.array([self.index.get(image_key(r), 0) for r in a])
    FrozenImage.lookup = lookup
    seed = SEEDS[0]
    X.make_model = lambda s: make_model(s, 'sparse48', 'joined-mlp')
    LM.CURRENT.update(arm='sparse48', seed=seed, step=0, r=None)
    enc, model, _ = LM.setup('sparse48', seed)
    LM.snapshot(enc, model, D.withheld_counts(seed), seed, 'block3', True)
    np.save(RUNS / 'captured-eval.npy', np.stack(list(seen.values())) if seen else np.zeros((0, 28, 28), np.float32))
    print('captured', len(seen))


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    X.make_model = None  # set per task
    tasks = [(a, s) for a in ARMS for s in SEEDS]
    ctx = __import__('multiprocessing').get_context('fork')
    fn = DUR.durable(counted_run, RUNS / 'done', '{0}-seed{1}')
    with ctx.Pool(WORKERS) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    out = os.environ.get('SC_OUT', 'pilot')  # the first pilot's report is frozen; reruns write their own
    S0.save_json(f'{REPORT}-{out}.json', {'status': 'exploratory, pilot; sparse48 trained and read as limited-memory main; only the image encoder differs', 'arms': ARMS, 'seeds': SEEDS,
                                          'hidden': HIDDEN, 'licences': SN.licences(), 'statistics': {a: ENCODERS[a][2] for a in ARMS if a in ENCODERS}, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-{out}-runtime.json', S0.runtime({'phase': 'sparse48 with four image encoders', 'wall_seconds': time.perf_counter() - t0, 'workers': WORKERS,
                                                             'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'codes': codes, 'capture': capture, 'main': main}[sys.argv[2]]()
