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
ARMS = ('existing', 'joined-mlp', 'joined-linear', 'dinov2-mlp')
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


class FrozenImage(nn.Module):
    """Image encoder for the core: frozen codes looked up by image content, standardized per model with fixed statistics from the cache, concatenated,
    then a small trained projection. Only the projection is a parameter, so checkpoints hold only it."""

    def __init__(self, models, head, out, table):
        super().__init__()
        self.models = tuple(models)
        self.index, self.raw = table['index'], np.concatenate([table[n] for n in self.models], 1)
        self.mean, self.std = (self.raw.mean(0), self.raw.std(0) + 1e-6) if len(self.raw) else (None, None)
        self.feats = None  # device tensor, built on first use
        width = self.raw.shape[1]
        self.head = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, out)) if head == 'mlp' else nn.Linear(width, out)
        self.misses, self.lookups, self.live = 0, 0, None

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
        if missing:  # encode live and add to the table
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
            self.feats = None
        return rows

    def forward(self, x):
        rows = self.lookup(x)
        if self.mean is None:  # statistics from the first images seen, when there was no cache (smoke only)
            self.mean, self.std = self.raw.mean(0), self.raw.std(0) + 1e-6
        if self.feats is None:
            self.feats = torch.from_numpy(((self.raw - self.mean) / self.std).astype(np.float32))
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
        models, head = {'joined-mlp': (MODELS, 'mlp'), 'joined-linear': (MODELS, 'linear'), 'dinov2-mlp': (('dinov2-s',), 'mlp')}[encoder]
        out = enc['image'].out.out_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(S0.nc.stream('sc_head', seed, encoder).integers(2 ** 62)))
            enc['image'] = FrozenImage(models, head, out, table(models))
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
    S0.save_json(f'{REPORT}-pilot.json', {'status': 'exploratory, pilot; sparse48 trained and read as limited-memory main; only the image encoder differs', 'arms': ARMS, 'seeds': SEEDS,
                                          'hidden': HIDDEN, 'licences': SN.licences(), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-pilot-runtime.json', S0.runtime({'phase': 'sparse48 with four image encoders', 'wall_seconds': time.perf_counter() - t0, 'workers': WORKERS,
                                                             'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'codes': codes, 'capture': capture, 'main': main}[sys.argv[2]]()
