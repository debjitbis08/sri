"""Memory on natural images (docs/natural-memory-v1-pilot.md). Does the key encoder's learning pay for memory, once what is stored is something a glyph encoder handles poorly?

Stage `encoders` measures, before any memory, what each network can do on CIFAR-10 as it learns from an unlabelled stream: finding an image from a new view of it, and finding its class. Fixed baselines: pixels and PCA. The stream is CIFAR-10's training split; everything scored comes from the test split, which no stream shows.

Usage: NFI_DEVICE=cuda SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.natural_memory pilot encoders
"""
import copy
import hashlib
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import sense_image_free as SIF
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import translation_stable_keys as TK
from src.sparse_local_memory import Memory

nc = S0.nc
NAME = 'natural-memory-v1' + os.environ.get('NM_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
CIFAR = S0.X.ROOT / 'runs' / 'external' / 'cifar10' / 'cifar10-full.npz'
KEY = 32
T = TK.TEMPERATURE
LR = 1e-3
SEEDS = tuple(int(s) for s in os.environ.get('NM_SEEDS', '392,393').split(','))
GLYPH_SEEDS = (392, 393)
UPDATES = int(os.environ.get('NM_UPDATES', 40000))
SNAPSHOTS = tuple(u for u in (0, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000) if u <= UPDATES)
GALLERY, KNN = 1000, 20
TAUS = (.7, .85, .95)
# network, start, augmentation
CONFIGS = {'small-glyph-affine': ('small', 'glyph', 'affine'), 'small-glyph-strong': ('small', 'glyph', 'strong'), 'small-scratch-strong': ('small', 'scratch', 'strong'),
           'deep-grey-strong': ('deep-grey', 'scratch', 'strong'), 'deep-colour-strong': ('deep-colour', 'scratch', 'strong')}
BASES = {'small': 64, 'deep-grey': 64, 'deep-colour': 64}
# memory stage: settings set after the encoder stage (see the pilot report)
BETAS = tuple(float(b) for b in os.environ.get('NM_BETAS', '.999,.9999,.99999').split(','))
LOAD, N_CANDIDATES = 450, 1500
LEVELS = {'instance-45': 45, 'instance-450': 450, 'class': 4}
EXPOSURES, CLASS_PER_STEP, WRITE_EVERY = 8, 32, 8
BLOCK = int(os.environ.get('NM_BLOCK', 2000))
REST = tuple(int(r) for r in os.environ.get('NM_REST', '4000,16000').split(','))
PRELUDES = tuple(int(p) for p in os.environ.get('NM_PRELUDES', '4000,16000').split(','))
WRITE_TAUS = tuple(float(t) for t in os.environ.get('NM_TAUS', '.7,.85,.95').split(','))
TAU_MAIN, CAPACITY, MATCHED = .85, 8192, (.13, .05)
SDC_SIGMA = .3  # the paper's setting for its normalized embeddings, not tuned here
TRANSPORT = {'sdc': lambda old, new: (lambda a: sdc_move(a, old, new)), 'linear': lambda old, new: (lambda a: unit(SIF.fit_map(old, new)(a)))} if os.environ.get('NM_TRANSPORT') else {}
TRANSPORT_EVERY, TRANSPORT_PROBES = 500, 2048
DONE = 'done-transport' if TRANSPORT else 'done-memory'
MEMORY_CONFIGS = tuple(os.environ.get('NM_CONFIGS', 'small-glyph-strong,deep-colour-strong').split(','))
VIEWS = {'mild': dict(area=.7, jitter=.1, flip=False, grey=0.), 'strong': dict(area=.2, jitter=.4, flip=True, grey=.2)}


# ---------------------------------------------------------------- data

LOADED = {}


def grey28(x):
    """uint8 (n, 3, 32, 32) -> float (n, 28, 28) in [0, 1], as the drift measurement's natural images."""
    return x.mean(1)[:, 2:30, 2:30].astype(np.float32) / 255


def colour32(x):
    return x.astype(np.float32) / 255


def data():
    if not LOADED:
        z = np.load(CIFAR)
        LOADED.update({k: z[k] for k in z.files})
    return LOADED


def split_test(seed):
    """One seeded partition of the test split, so nothing scored ever appears in a stream. Returns index arrays: special (cues, targets, unseen of the instance facts), gallery of instance retrieval, writes and queries of class facts."""
    perm = nc.stream('nm_test_split', seed).permutation(10000)
    special, rest = perm[:285], perm[285:]
    half = len(rest) // 2
    return {'special': special, 'write': rest[:half], 'query': rest[half:], 'gallery': rest[half:][:GALLERY]}


def prepare_input(x, arch):
    return colour32(x) if arch == 'deep-colour' else grey28(x)


# ---------------------------------------------------------------- networks

class Block(nn.Module):
    def __init__(self, a, b, stride):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(a, b, 3, stride, 1, bias=False), nn.Conv2d(b, b, 3, 1, 1, bias=False)
        self.n1, self.n2 = nn.GroupNorm(8, b), nn.GroupNorm(8, b)
        self.skip = nn.Identity() if a == b and stride == 1 else nn.Sequential(nn.Conv2d(a, b, 1, stride, bias=False), nn.GroupNorm(8, b))

    def forward(self, x):
        return F.relu(self.n2(self.c2(F.relu(self.n1(self.c1(x))))) + self.skip(x))


class DeepEncoder(nn.Module):
    """A quarter-width ResNet-18 for small images, the width this GPU can train through a night. Group normalisation, so no running statistics: a fixed function of the input and its weights, which a parameter average can track. Key: a two-layer head, unit length."""
    def __init__(self, channels, width=16, key=KEY):
        super().__init__()
        self.colour = channels == 3
        self.first = nn.Sequential(nn.Conv2d(channels, width, 3, 1, 1, bias=False), nn.GroupNorm(8, width), nn.ReLU())
        widths = [width, width, 2 * width, 2 * width, 4 * width, 4 * width, 8 * width, 8 * width]
        strides = [1, 1, 2, 1, 2, 1, 2, 1]
        ins = [width] + widths[:-1]
        self.blocks = nn.Sequential(*[Block(a, b, s) for a, b, s in zip(ins, widths, strides)])
        self.fc, self.out = nn.Linear(8 * width, 8 * width), nn.Linear(8 * width, key)

    def features(self, x):
        if not self.colour:
            x = x[:, None]
        return self.blocks(self.first(x)).mean((2, 3))

    def forward(self, x):
        return F.normalize(self.out(F.relu(self.fc(self.features(x)))), dim=1)


def small_features(m, x):
    return F.relu(m.fc(F.relu(m.c2(F.relu(m.c1(x[:, None])))).flatten(1)))


def features(m, x):
    return m.features(x) if isinstance(m, DeepEncoder) else small_features(m, x)


def build(arch, start, seed):
    device = DEV.NAME if DEV.ACTIVE else 'cpu'
    if start == 'glyph':
        return SM.encoder('unlabelled', seed)
    if arch == 'small':
        return TK.build('base', KEY, 40_000 + seed)
    with torch.device('cpu'):
        torch.manual_seed(int(nc.stream('nm_init', seed, arch).integers(2 ** 62)))
        m = DeepEncoder(3 if arch == 'deep-colour' else 1)
    return m.to(device)


# ---------------------------------------------------------------- views

def views(x, rng, cfg):
    """A view of each image: x tensor (n, h, w) or (n, c, h, w) in [0, 1]. A crop of at least `area`, resized; a flip; brightness, contrast and, in colour, saturation; sometimes grey. Hue jitter is left out. Parameters from a numpy stream."""
    grey_input = x.dim() == 3
    c = x[:, None] if grey_input else x
    n, size = len(c), c.shape[-1]
    s = rng.uniform(cfg['area'], 1, n)
    r = np.exp(rng.uniform(np.log(.75), np.log(4 / 3), n))
    w, h = np.minimum(np.sqrt(s * r), 1), np.minimum(np.sqrt(s / r), 1)
    tx, ty = rng.uniform(-(1 - w), 1 - w), rng.uniform(-(1 - h), 1 - h)
    flip = np.where(rng.random(n) < .5, -1., 1.) if cfg['flip'] else np.ones(n)
    theta = torch.from_numpy(np.stack([np.stack([w * flip, np.zeros(n), tx], 1), np.stack([np.zeros(n), h, ty], 1)], 1).astype(np.float32))
    out = F.grid_sample(c, F.affine_grid(theta, (n, c.shape[1], size, size), align_corners=False), align_corners=False, padding_mode='border')
    j = cfg['jitter']
    gain = lambda: torch.from_numpy(rng.uniform(1 - j, 1 + j, n).astype(np.float32))[:, None, None, None]
    out = out * gain()
    mean = out.mean((1, 2, 3), keepdim=True)
    out = mean + (out - mean) * gain()
    if c.shape[1] == 3:
        lum = out.mean(1, keepdim=True)
        out = lum + (out - lum) * gain()
        if cfg['grey']:
            turn = torch.from_numpy((rng.random(n) < cfg['grey']).astype(np.float32))[:, None, None, None]
            out = turn * out.mean(1, keepdim=True) + (1 - turn) * out
    out = out.clamp(0, 1)
    return out[:, 0] if grey_input else out


def stream_views(base, rng, aug):
    if aug == 'affine':
        return SRL.augment_with(base, rng, SRL.AUGMENTS['full'])
    return views(base, rng, VIEWS['strong'])


def partners(n):
    """Each of n bases seen twice, base by base: which item is the same image."""
    return torch.arange(2 * n).reshape(-1, 2).flip(1).reshape(-1)


# ---------------------------------------------------------------- measures, pure

def auc(pos, neg):
    """Probability that a positive scores above a negative, ties half. Rank form."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    all_ = np.concatenate([pos, neg])
    order = all_.argsort(kind='stable')
    ranks = np.empty(len(all_))
    ranks[order] = np.arange(1, len(all_) + 1)
    values, inverse, counts = np.unique(all_, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + 1 + ends) / 2)[inverse]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def instance_scores(g, q, unseen):
    """g (n, d) unit codes of n images, q (n, d) a new view of each, unseen (u, d) views of other images. Finding the image, how near its own code lies, how near the nearest other, and how well a threshold separates known from unseen."""
    S = q @ g.T
    own = np.diag(S).copy()
    other = S.copy()
    np.fill_diagonal(other, -2.)
    best = other.max(1)
    best_unseen = (unseen @ g.T).max(1)
    out = {'top1': float((S.argmax(1) == np.arange(len(g))).mean()), 'own_median': float(np.median(own)), 'other_median': float(np.median(best)), 'auc_unseen': auc(S.max(1), best_unseen)}
    for tau in TAUS:
        out[f'own_at_{tau}'] = float((own >= tau).mean())
        out[f'other_at_{tau}'] = float((best >= tau).mean())
        out[f'unseen_at_{tau}'] = float((best_unseen >= tau).mean())
    return out


def knn_class(zw, yw, zq, yq, k=KNN, classes=10):
    """Cosine k-nearest neighbours, weighted by similarity. Also the top neighbour alone, which is what a one-winner memory reads."""
    S = zq @ zw.T
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    votes = np.zeros((len(zq), classes), np.float32)
    rows = np.arange(len(zq))[:, None]
    np.add.at(votes, (rows, yw[top]), np.maximum(S[rows, top], 0))
    return {'knn': float((votes.argmax(1) == yq).mean()), 'nearest': float((yw[S.argmax(1)] == yq).mean())}


# ---------------------------------------------------------------- stage: encoders

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def encode_fn(m):
    return lambda x: S0.batched(lambda b: m(b), x).detach().cpu().numpy().astype(np.float32)


def feature_fn(m):
    return lambda x: S0.batched(lambda b: features(m, b), x).detach().cpu().numpy().astype(np.float32)


def view_arrays(x, rng, cfg):
    out = []
    for i in range(0, len(x), 512):
        out.append(views(torch.from_numpy(np.ascontiguousarray(x[i:i + 512])), rng, cfg).detach().cpu().numpy())
    return np.concatenate(out)


def eval_material(seed, arch):
    d, sp = data(), split_test(seed)
    x, y = d['x_test'], d['y_test']
    rng = nc.stream('nm_eval_views', seed, arch)
    fix = lambda idx: prepare_input(x[idx], arch)
    g, u = fix(sp['gallery']), fix(sp['write'][:GALLERY])
    out = {'gallery': g, 'write': fix(sp['write']), 'query': fix(sp['query']), 'y_write': y[sp['write']], 'y_query': y[sp['query']]}
    for v, cfg in VIEWS.items():
        out[f'q_{v}'], out[f'u_{v}'] = view_arrays(g, rng, cfg), view_arrays(u, rng, cfg)
    return out


def evaluate(fn, mat):
    """fn: images -> codes. Everything an encoder is scored on at one point."""
    g = unit(fn(mat['gallery']))
    out = {v: instance_scores(g, unit(fn(mat[f'q_{v}'])), unit(fn(mat[f'u_{v}']))) for v in VIEWS}
    out['class'] = knn_class(unit(fn(mat['write'])), mat['y_write'], unit(fn(mat['query'])), mat['y_query'])
    return out, g


def encoder_path(config, seed, u):
    return RUNS / 'encoders' / f'{config}-seed{seed}' / f'u{u}.pt'


def fit_encoder(task):
    config, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    arch, start, aug = CONFIGS[config]
    model = build(arch, start, seed)
    opt = torch.optim.Adam(model.parameters(), LR)
    mat = eval_material(seed, arch)
    train = data()['x_train']
    rng = nc.stream('nm_stream', seed, config)
    n = BASES[arch]
    partner = partners(n)
    curve, g0, losses, train_wall = {}, None, [], 0.
    for u in range(UPDATES + 1):
        if u in SNAPSHOTS:
            model.eval()
            with torch.no_grad():
                rec, g = evaluate(encode_fn(model), mat)
                if u == SNAPSHOTS[-1] or u == 0:
                    rec['feature_class'] = knn_class(unit(feature_fn(model)(mat['write'])), mat['y_write'], unit(feature_fn(model)(mat['query'])), mat['y_query'])
            rec['moved_from_start'] = float(np.median(np.arccos(np.clip((g * g0).sum(1), -1, 1)))) if g0 is not None else 0.
            rec['moved_since_last'] = float(np.median(np.arccos(np.clip((g * prev).sum(1), -1, 1)))) if g0 is not None else 0.
            rec['loss'] = float(np.mean(losses[-200:])) if losses else None
            curve[str(u)] = rec
            g0 = g if g0 is None else g0
            prev = g
            path = encoder_path(config, seed, u)
            buf = io.BytesIO()
            torch.save({'model': DEV.to_cpu(model.state_dict()), 'opt': DEV.to_cpu(opt.state_dict()), 'updates': u}, buf)
            DUR.atomic_write(path, buf.getvalue())
            model.train()
        if u == UPDATES:
            break
        t = time.perf_counter()
        base = torch.from_numpy(prepare_input(train[rng.integers(len(train), size=n)], arch))
        x = torch.stack([stream_views(base, rng, aug), stream_views(base, rng, aug)], 1)
        x = x.reshape(-1, *x.shape[2:])
        z = model(x)
        loss = SRL.info_nce(z @ z.T / T, partner)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        train_wall += time.perf_counter() - t
    res = {'curve': curve, 'config': config, 'seed': seed, 'params': int(sum(p.numel() for p in model.parameters())), 'train_wall_seconds': train_wall}
    return f'{config}/{seed}', res, S0.timing(t0, c0)


def fit_fixed(task):
    """Pixels and PCA on grey and colour inputs: no learning from the stream beyond PCA's fit, on 5,000 stream images."""
    kind, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    name, arch = kind.split(':')
    mat = eval_material(seed, arch)
    train = data()['x_train']
    rng = nc.stream('nm_pca', seed, arch)
    fit = prepare_input(train[rng.permutation(len(train))[:5000]], arch).reshape(5000, -1).astype(np.float64)
    mean = fit.mean(0)
    if name == 'pixels':
        fn = lambda x: np.asarray(x.reshape(len(x), -1), np.float64) - mean
        bytes_needed = mean.size * 4
    else:
        _, s, vt = np.linalg.svd(fit - mean, full_matrices=False)
        w = vt[:KEY] / s[:KEY, None] * np.sqrt(len(fit))
        fn = lambda x: (np.asarray(x.reshape(len(x), -1), np.float64) - mean) @ w.T
        bytes_needed = (mean.size + w.size) * 4
    rec, _ = evaluate(lambda x: fn(x).astype(np.float32), mat)
    return f'{kind}/{seed}', {'kind': kind, 'seed': seed, 'scores': rec, 'bytes': bytes_needed}, S0.timing(t0, c0)


def encoders():
    t0 = time.perf_counter()
    data()
    tasks = [(c, s) for c, (_, start, _) in CONFIGS.items() for s in SEEDS if start != 'glyph' or s in GLYPH_SEEDS]
    fixed = [(f'{n}:{a}', s) for n in ('pixels', 'pca') for a in ('small', 'deep-colour') for s in SEEDS]
    res_fixed = S0.pool_map(DUR.durable(fit_fixed, RUNS / 'done-fixed', '{0}-{1}'), fixed)
    S0.save_json(f'{REPORT}-fixed.json', {k: r for k, r, _ in res_fixed})
    res = S0.pool_map(DUR.durable(fit_encoder, RUNS / 'done-encoders', '{0}-{1}'), tasks)
    S0.save_json(f'{REPORT}-encoders.json', {'status': 'exploratory; what each network can do on CIFAR-10 as it learns from an unlabelled stream, before any memory', 'configs': CONFIGS, 'views': VIEWS, 'snapshots': SNAPSHOTS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-encoders-runtime.json', S0.runtime({'phase': 'encoders on the natural stream, scored at snapshots; fixed baselines', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res + res_fixed}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res + res_fixed))}))


# ---------------------------------------------------------------- stage: memory

class Tagged(Memory):
    """Memory with a scoring table beside it: which fact each unit took. The table is never read by the memory itself; it says what a unit holds, so an address can be scored apart from its value."""
    def __init__(self, H, tau, d, m, facts):
        super().__init__(H, 1, tau, d, m)
        self.tags = np.zeros((H, facts), np.int16)

    def write_tagged(self, z, v, fact):
        I = self.write(z, v)
        for i in I:
            self.tags[i, fact] = min(int(self.tags[i, fact]) + 1, 32767)
        return I

    def payload_bytes(self):
        return 4 * self.h * (self.d + self.m + 1)  # address, value sum, count

    def nearest(self, Z):
        """Best unit and its similarity for each unit query. A one-winner read, in one step."""
        if not self.h:
            return np.zeros(len(Z), np.int64), np.full(len(Z), -2., np.float32)
        S = Z @ self.a[:self.h].T
        i = S.argmax(1)
        return i, S[np.arange(len(Z)), i]


def margin_of(S, k):
    """S (q, h) similarities to units. Best unit, its similarity, and how far it stands above the mean of the next k units. A code that cannot tell inputs apart gives every unit nearly the same similarity, so a small margin. Local scaling, as in CSLS (Conneau et al. 2018), here on one's own memory."""
    q, h = S.shape
    i = S.argmax(1)
    best = S[np.arange(q), i]
    if h < 2:
        return i, best, np.zeros(q, np.float32)
    k = min(k, h - 1)
    top = -np.partition(-S, k, axis=1)[:, :k + 1]  # the k + 1 largest
    return i, best, best - (top.sum(1) - best) / k


def taus_for(encoder):
    """Pixels are wide, so one write threshold only."""
    return (TAU_MAIN,) if encoder == 'pixels' else WRITE_TAUS


def plan_writes(rng, facts, exposures, updates, every=WRITE_EVERY):
    """A seeded order of writes, spread over a block: each fact `exposures` times, cut into one list per write step."""
    order = rng.permutation(np.repeat(np.asarray(facts), exposures))
    steps = updates // every
    return [order[i * len(order) // steps:(i + 1) * len(order) // steps] for i in range(steps)]


def read_scores(unit_at, sim, thr, truth, tags, pick_value, value_truth):
    """One read at one threshold. Abstentions are wrong. unit_at, sim: nearest unit and similarity of each query. truth: the fact each query is about. pick_value: the candidate each answer's content is nearest to."""
    ok = sim >= thr
    return {'coverage': float(ok.mean()), 'address': float((ok & (tags[unit_at].argmax(1) == truth)).mean()), 'value': float((ok & (pick_value == value_truth)).mean())}


def matched_threshold(unseen_sim, rate):
    """The threshold at which `rate` of unseen queries would be answered. Uses the unseen queries, so it only scores."""
    return float(np.quantile(unseen_sim, 1 - rate))


def memory_material(seed, arch, order):
    """Facts, in the seed's order. instance: a named image and its own target image. class: any image of a class and that class's own target image. Queries are new views, or for classes other images never written."""
    d, sp = data(), split_test(seed)
    x, y = d['x_test'], d['y_test']
    rng = nc.stream('nm_facts', seed)
    pool = rng.permutation(sp['write'])
    cues, unseen, tg = pool[:2 * LOAD], pool[2 * LOAD:3 * LOAD], pool[3 * LOAD:3 * LOAD + N_CANDIDATES]
    classes = rng.permutation(10)
    A, B, U = classes[:4], classes[4:8], classes[8:]
    first, second = (('A', 'B') if order == 'AB' else ('B', 'A'))
    fix = lambda idx: prepare_input(x[idx], arch)
    qrng = nc.stream('nm_query_views', seed, arch)
    q = {v: view_arrays(fix(cues), qrng, c) for v, c in VIEWS.items()}
    qu = {v: view_arrays(fix(unseen), qrng, c) for v, c in VIEWS.items()}
    known = np.flatnonzero(np.isin(y[sp['query']], np.r_[A, B]))
    other = np.flatnonzero(np.isin(y[sp['query']], U))
    remap = np.full(10, -1)
    remap[np.r_[A, B]] = np.arange(8)
    return {'first': first, 'second': second, 'cue_images': x[cues], 'target_images': x[tg], 'candidates': fix(tg), 'q': q, 'qu': qu,
            'class_write': x[sp['write']], 'class_write_y': remap[y[sp['write']]], 'class_query': fix(sp['query'][known]), 'class_query_y': remap[y[sp['query'][known]]], 'class_unseen': fix(sp['query'][other])}


def set_facts(level, tag):
    """The facts of one set, per level. A fact's target is the candidate of the same index."""
    if level == 'class':
        return np.arange(4) if tag == 'A' else np.arange(4, 8)
    n = LEVELS[level]
    return np.arange(n) if tag == 'A' else np.arange(LOAD, LOAD + n)


def fact_index(level, truth):
    """Rows in a level's tag table: instance facts in B start at LOAD, and small loads keep their rows compact."""
    if level == 'class':
        return truth
    n = LEVELS[level]
    return np.where(truth >= LOAD, truth - LOAD + n, truth)


def digest(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def update_average(average, model, beta):
    with torch.no_grad():
        for p, q in zip(average.parameters(), model.parameters()):
            p.mul_(beta).add_(q, alpha=1 - beta)


def fixed_codes(kind, arch, seed):
    """Pixels or PCA fitted on 5,000 stream images: a code that never learns. Returns images -> unit codes, and its width."""
    train = data()['x_train']
    rng = nc.stream('nm_pca', seed, arch)
    fit = prepare_input(train[rng.permutation(len(train))[:5000]], arch).reshape(5000, -1).astype(np.float64)
    mean = fit.mean(0)
    if kind == 'pixels':
        return (lambda x: unit((np.asarray(x.reshape(len(x), -1), np.float64) - mean).astype(np.float32))), mean.size
    _, s, vt = np.linalg.svd(fit - mean, full_matrices=False)
    w = vt[:KEY] / s[:KEY, None] * np.sqrt(len(fit))
    return (lambda x: unit(((np.asarray(x.reshape(len(x), -1), np.float64) - mean) @ w.T).astype(np.float32))), KEY


def sdc_move(points, old, new, sigma=SDC_SIGMA):
    """Semantic drift compensation (Yu et al. 2020, eq. 10-11): each point moves by a Gaussian-weighted mean of how nearby live inputs moved. points (h, d); old, new (p, d) the same inputs under the old and new encoder. Returns unit points. Where no input lies near, the weights are tiny but their ratio is still defined, so it moves by the nearest inputs' shift."""
    d2 = ((points ** 2).sum(1, keepdims=True) + (old ** 2).sum(1)[None] - 2 * points @ old.T)
    logw = -d2 / (2 * sigma ** 2)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    return unit(points + (w @ (new - old)) / w.sum(1, keepdims=True))


def move_memory(mem, fn):
    """Addresses and mean contents move by fn; write totals and counts keep their sizes, so later writes weigh as before."""
    h = mem.h
    if not h:
        return
    size = np.linalg.norm(mem.c[:h], axis=1, keepdims=True)
    mem.a[:h] = fn(mem.a[:h])
    mem.c[:h] = mem.a[:h] * size
    n = mem.n[:h, None]
    mem.u[:h] = fn(mem.u[:h] / n) * n


def learn_memory(task):
    config, seed, order, P = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    arch, start, aug = CONFIGS[config]
    model = build(arch, start, seed)
    initial = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.Adam(model.parameters(), LR)
    averages = {f'ema-{b}': copy.deepcopy(model).requires_grad_(False).eval() for b in BETAS}
    train = data()['x_train']
    rng = nc.stream('nm_stream', seed, config)  # as the encoder stage, so the working encoder is the same one
    wrng = nc.stream('nm_writes', seed, config, order, P)
    n = BASES[arch]
    partner = partners(n)
    mat = memory_material(seed, arch, order)
    fx = {k: fixed_codes(k, arch, seed) for k in ('pixels', 'pca')}
    teach = {mat['first']: (P, P + BLOCK), mat['second']: (P + BLOCK, P + 2 * BLOCK)}
    end_second = P + 2 * BLOCK
    checks = {P + BLOCK: 'first', end_second: 'second', **{end_second + r: f'rest-{r}' for r in REST}}
    plans = {t: {lv: plan_writes(wrng, set_facts(lv, t), EXPOSURES, BLOCK) if lv != 'class' else None for lv in LEVELS} for t in 'AB'}
    class_rows = {t: np.flatnonzero(np.isin(mat['class_write_y'], set_facts('class', t))) for t in 'AB'}
    mems, width = {}, {}
    frozen_at_p = None
    moving = lambda: {'working': model, **averages}
    coders, cache, held = {}, {}, {lv: [] for lv in LEVELS}
    fact_total = {lv: (8 if lv == 'class' else 2 * LEVELS[lv]) for lv in LEVELS}
    reads, gates, moved, baseline = {}, {}, {}, {}
    cue_probe = prepare_input(mat['cue_images'][:GALLERY // 2], arch)

    def set_up_memories():
        nonlocal frozen_at_p, coders, width
        frozen_at_p = copy.deepcopy(model).requires_grad_(False).eval()
        nets = {**moving(), 'frozen-start': initial, 'frozen-at-teaching': frozen_at_p}
        coders = {e: encode_fn(m) for e, m in nets.items()}
        coders.update({k: v[0] for k, v in fx.items()})
        width = {**{e: KEY for e in nets}, **{k: v[1] for k, v in fx.items()}}
        if TRANSPORT:
            coders = {e: coders[e] for e in ('working', 'frozen-at-teaching')}
            coders.update({f'moved-{k}': encode_fn(model) for k in TRANSPORT})
            width = {e: KEY for e in coders}
        for e in coders:
            for lv in LEVELS:
                for tau in taus_for(e):
                    mems[(e, lv, tau)] = Tagged(CAPACITY, tau, width[e], width[e], fact_total[lv])

    def code(e, x, key=None):
        """Unit codes. A code that never changes is computed once for a named set of images."""
        if e in fx or e.startswith('frozen'):
            if key is not None and (e, key) in cache:
                return cache[(e, key)]
            out = unit(coders[e](x))
            if key is not None:
                cache[(e, key)] = out
            return out
        return unit(coders[e](x))

    def write_step(tag, step):
        batches = []
        for lv in LEVELS:
            if lv == 'class':
                rows = class_rows[tag][wrng.integers(len(class_rows[tag]), size=CLASS_PER_STEP)]
                facts = mat['class_write_y'][rows]
                images = mat['class_write'][rows]
            else:
                facts = plans[tag][lv][step]
                images = mat['cue_images'][facts]
            if not len(facts):
                continue
            batches.append((lv, facts, images))
        if not batches:
            return
        images = np.concatenate([b[2] for b in batches])
        facts = np.concatenate([b[1] for b in batches])
        cues = view_arrays(prepare_input(images, arch), wrng, VIEWS['mild'])
        targets = view_arrays(prepare_input(mat['target_images'][facts], arch), wrng, VIEWS['mild'])
        z = {e: code(e, cues) for e in coders}
        v = {e: code(e, targets) for e in coders}
        at = 0
        for lv, f, im in batches:
            rows = fact_index(lv, f)
            held[lv].append((cues[at:at + len(f)], targets[at:at + len(f)], rows))
            for e in coders:
                for tau in taus_for(e):
                    mem = mems[(e, lv, tau)]
                    for i in range(len(f)):
                        mem.write_tagged(z[e][at + i], v[e][at + i], rows[i])
            at += len(f)

    def read_level(mem, lv, e, rec, tags, codes):
        """All reads of one memory: per set, at its own threshold and at matched false answers."""
        cand = codes['cand']
        h = mem.h
        if not h:
            return
        val = mem.u[:h] / mem.n[:h, None]
        for q_name, (Zk, facts_k, Zu) in codes['queries'].items():
            i_u, s_u = mem.nearest(Zu)
            i_k, s_k = mem.nearest(Zk)
            pick = ((val[i_k] @ cand.T)).argmax(1)  # nearest stored-value direction among candidates, on unit codes
            row = {'unseen_at_tau': float((s_u >= mem.tau).mean())}
            thresholds = {'tau': mem.tau, 'answer-all': -2., **{f'matched-{r}': matched_threshold(s_u, r) for r in MATCHED}}
            for name, thr in thresholds.items():
                row[name] = {'threshold': float(thr)}
                for t in tags:
                    sel = np.isin(facts_k, set_facts(lv, t)) if lv != 'class' else np.isin(facts_k, set_facts('class', t))
                    row[name][t] = read_scores(i_k[sel], s_k[sel], thr, fact_index(lv, facts_k[sel]), mem.tags[:h], pick[sel], facts_k[sel])
            rec[q_name] = row
        rec['units'], rec['forced'], rec['payload_bytes'] = h, mem.forced, mem.payload_bytes()

    def queries_for(e, lv, tags_all):
        """Unit codes of a level's queries through one encoder."""
        out = {}
        if lv == 'class':
            out['class'] = (code(e, mat['class_query'], 'class_query'), mat['class_query_y'], code(e, mat['class_unseen'], 'class_unseen'))
            return out
        n = LEVELS[lv]
        keep = np.r_[np.arange(n), np.arange(LOAD, LOAD + n)]
        for v in VIEWS:
            out[v] = (code(e, mat['q'][v], f'q_{v}')[keep], keep.astype(np.int64), code(e, mat['qu'][v], f'qu_{v}')[:n])
        return out

    def read_all(name, tags):
        t_read = time.perf_counter()
        out = {}
        for e in coders:
            cand = code(e, mat['candidates'], 'candidates')
            for lv in LEVELS:
                q = queries_for(e, lv, tags)
                for tau in taus_for(e):
                    rec = {}
                    read_level(mems[(e, lv, tau)], lv, e, rec, tags, {'cand': cand, 'queries': q})
                    out[f'{e}|{lv}|{tau}'] = rec
        # reference: memory rebuilt now from every view written, through the working encoder. It holds images, so it only scores.
        for lv in LEVELS:
            if not held[lv]:
                continue
            cues = np.concatenate([h[0] for h in held[lv]])
            targets = np.concatenate([h[1] for h in held[lv]])
            rows = np.concatenate([h[2] for h in held[lv]])
            z, v = code('working', cues), code('working', targets)
            cand = code('working', mat['candidates'])
            q = queries_for('working', lv, tags)
            mem = Tagged(CAPACITY, TAU_MAIN, KEY, KEY, fact_total[lv])
            for i in range(len(rows)):
                mem.write_tagged(z[i], v[i], rows[i])
            rec = {}
            read_level(mem, lv, 'working', rec, tags, {'cand': cand, 'queries': q})
            out[f'rebuild-images|{lv}|{TAU_MAIN}'] = rec
        out['_read_seconds'] = time.perf_counter() - t_read
        return out

    write_wall, train_wall, move_wall = 0., 0., 0.
    total = end_second + max(REST)
    trng = nc.stream('nm_transport', seed, config, order, P)
    window_net = None
    for u in range(total):
        if u == P:
            gates['working_at_P'] = digest(model)
            path = encoder_path(config, seed, P)
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=False)['model']
                gates['matches_encoder_stage'] = bool(all(torch.equal(saved[k].cpu(), val.detach().cpu()) for k, val in model.state_dict().items()))
            set_up_memories()
            with torch.no_grad():
                start_codes = {e: unit(encode_fn(m)(cue_probe)) for e, m in moving().items()}
        t = time.perf_counter()
        base = torch.from_numpy(prepare_input(train[rng.integers(len(train), size=n)], arch))
        x = torch.stack([stream_views(base, rng, aug), stream_views(base, rng, aug)], 1)
        x = x.reshape(-1, *x.shape[2:])
        z = model(x)
        loss = SRL.info_nce(z @ z.T / T, partner)
        opt.zero_grad(); loss.backward(); opt.step()
        for beta, avg in zip(BETAS, averages.values()):
            update_average(avg, model, beta)
        train_wall += time.perf_counter() - t
        t = time.perf_counter()
        for tag, (lo, hi) in teach.items():
            if lo <= u < hi and (u - lo) % WRITE_EVERY == WRITE_EVERY - 1:
                with torch.no_grad():
                    write_step(tag, (u - lo) // WRITE_EVERY)
        write_wall += time.perf_counter() - t
        t = time.perf_counter()
        if TRANSPORT and u >= P:
            if window_net is None:
                window_net = copy.deepcopy(model).requires_grad_(False).eval()
            elif (u + 1 - P) % TRANSPORT_EVERY == 0:
                probe = prepare_input(train[trng.integers(len(train), size=TRANSPORT_PROBES)], arch)  # live inputs, discarded after this step
                with torch.no_grad():
                    old, new = unit(encode_fn(window_net)(probe)), unit(encode_fn(model)(probe))
                for k, make in TRANSPORT.items():
                    fn = make(old, new)
                    for lv in LEVELS:
                        for tau in WRITE_TAUS:
                            move_memory(mems[(f'moved-{k}', lv, tau)], fn)
                window_net.load_state_dict(model.state_dict())
        move_wall += time.perf_counter() - t
        if u + 1 in checks:
            label = checks[u + 1]
            with torch.no_grad():
                reads[label] = read_all(label, mat['first'] if label == 'first' else 'AB')
                moved[label] = {e: float(np.median(np.arccos(np.clip((start_codes[e] * unit(encode_fn(m)(cue_probe))).sum(1), -1, 1)))) for e, m in moving().items()}
    res = {'reads': reads, 'moved_from_teaching_start': moved, 'gates': gates, 'config': config, 'seed': seed, 'order': order, 'P': P, 'updates': total, 'train_wall_seconds': train_wall, 'write_wall_seconds': write_wall, 'move_wall_seconds': move_wall}
    return f'{config}/{seed}/{order}/{P}', res, S0.timing(t0, c0)


def memory():
    t0 = time.perf_counter()
    data()
    tasks = [(c, s, o, p) for c in MEMORY_CONFIGS for s in SEEDS if CONFIGS[c][1] != 'glyph' or s in GLYPH_SEEDS for o in ('AB', 'BA') for p in PRELUDES]
    res = S0.pool_map(DUR.durable(learn_memory, RUNS / DONE, '{0}-{1}-{2}-{3}'), tasks)
    res = [DUR.load(f) for f in sorted((RUNS / DONE).glob('*.json.gz'))]  # every finished task, since networks may run in separate commands
    out = f'{REPORT}-transport' if TRANSPORT else REPORT
    S0.save_json(f'{out}.json', {'status': 'exploratory, pilot seeds, both orders; natural images held as codes only, except the rebuild reference', 'betas': BETAS, 'levels': LEVELS, 'exposures': EXPOSURES, 'write_taus': WRITE_TAUS, 'rest': REST, 'block': BLOCK, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{out}-runtime.json', S0.runtime({'phase': 'encoder, averages, memory writes and reads', 'last_command_wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'encoders': encoders, 'memory': memory}[sys.argv[2]]()
