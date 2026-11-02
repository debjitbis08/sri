"""What is the movement of a learning encoder made of, and does the answer depend on the images used to measure it? (docs/sense-drift-structure-v1.md.) Images here only measure; nothing is a memory mechanism.

A code's path through training is split three ways, by input family:
  jitter     does not accumulate; from the structure function S(tau) = a + b tau + c tau^2, fitted non-negatively (a, b, c: jitter, diffusion, trend)
  mapped     the share of a block's movement a linear map predicts, fitted on one family or held-out half, read on another
  remainder  the rest
Also: slow directions by the generalized eigenproblem C_move w = lambda C w, fitted on one family and read on others, and whether they keep symbol identity; and parameter averages (EMA) as a calm copy.

Generality is tested on four axes: input family, including natural images and noise; network (a contrastive key encoder, a host trained on other tasks, an encoder trained from scratch on natural images); stream (stationary, shift, natural); seed, including fresh seeds for the from-scratch encoder.

Natural images: as much of CIFAR-10's binary archive as has been fetched, parsed without unpickling, grey, centre-cropped to 28.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_drift_structure pilot
"""
import copy
import io
import os
import time
import zlib

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import nnls

from experiments import durable as DUR
from experiments import sense_consolidation as SC
from experiments import sense_image_free as F
from experiments import sense_image_free_windows as W
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import shared_core_unseen_traces as UT
from experiments import translation_stable_keys as TK

nc, P = S0.nc, S0.P
NAME = 'sense-drift-structure-v1' + os.environ.get('SI_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
CIFAR = S0.X.ROOT / 'runs' / 'external' / 'cifar10' / 'cifar-10-binary.tar.gz'
N, E, R = F.N, F.E, F.R
WINDOW, BLOCK = W.WINDOW, F.BLOCK
SPAN = BLOCK // WINDOW  # snapshots per block
PER_FAMILY = int(os.environ.get('SD_PER_FAMILY', 512))
BETAS = (.99, .999)
SCRATCH_STEPS = int(os.environ.get('SD_SCRATCH', 8000))
SLOW = (4, 8, 16)
MATERIAL_SEED = {392: 392, 393: 393, 404: 392, 405: 393}


# ---------------------------------------------------------------- natural images

def parse_cifar(raw):
    """Whatever complete records the possibly partial gzip holds. No pickle and nothing written to disk. Returns grey centre crops (n, 28, 28) and labels."""
    data = zlib.decompressobj(31).decompress(raw)
    images, labels, at = [], [], 0
    while at + 512 <= len(data):
        head = data[at:at + 512]
        if not any(head):
            break
        name = head[:100].rstrip(b'\0').decode()
        size = int(head[124:136].rstrip(b'\0 ') or b'0', 8)
        body = data[at + 512:at + 512 + size]
        if name.endswith('.bin') and 'batch' in name:
            count = len(body) // 3073
            rec = np.frombuffer(body[:count * 3073], np.uint8).reshape(count, 3073)
            labels.append(rec[:, 0])
            images.append(rec[:, 1:].reshape(count, 3, 32, 32).mean(1)[:, 2:30, 2:30].astype(np.float32) / 255)
        at += 512 + (size + 511) // 512 * 512
    return (np.concatenate(images), np.concatenate(labels)) if images else (np.zeros((0, 28, 28), np.float32), np.zeros(0, np.uint8))


def natural():
    path = RUNS / 'natural.npz'
    if not path.exists():
        x, y = parse_cifar(CIFAR.read_bytes())
        buf = io.BytesIO()
        np.savez_compressed(buf, x=x, y=y)
        DUR.atomic_write(path, buf.getvalue())
    z = np.load(path)
    return z['x'], z['y']


# ---------------------------------------------------------------- families

def sample(x, n, rng):
    return np.ascontiguousarray(x[rng.permutation(len(x))[:n]], dtype=np.float32)


def noise_images(n, rng):
    white = rng.random((n, 28, 28), dtype=np.float32)
    coarse = rng.random((n, 7, 7), dtype=np.float32)
    return white, coarse.repeat(4, 1).repeat(4, 2)


def families(seed):
    """Input families, each PER_FAMILY images or fewer, all fixed for a seed. The last natural images are left out of any training stream."""
    m = MATERIAL_SEED[seed]
    rng = nc.stream('sd_families', seed)
    mat = F.load_material(m)
    train, ev, _ = S0.test_material(m)
    bg = S0.background_images(m)
    x_natural, _ = natural()
    held = x_natural[-min(1000, len(x_natural) // 3):]
    out = {'background': sample(bg.reshape(-1, 28, 28), PER_FAMILY, rng), 'cues': sample(train[:2 * N].reshape(-1, 28, 28), PER_FAMILY, rng), 'unseen-symbols': sample(ev[2 * N:3 * N].reshape(-1, 28, 28), PER_FAMILY, rng),
           'targets': sample(mat['tgt_eval'].reshape(-1, 28, 28), PER_FAMILY, rng), 'other-fonts': sample(mat['other_font'][mat['font_mask']].reshape(-1, 28, 28), PER_FAMILY, rng),
           'dots': sample(P.DOTS['train'].reshape(-1, 28, 28), PER_FAMILY, rng), 'mnist': sample(S0.X.DATA[('mnist', 'eval')][0], PER_FAMILY, rng), 'dice': sample(S0.X.DATA[('dice', 'eval')][0], PER_FAMILY, rng),
           'natural': sample(held, PER_FAMILY, rng)}
    for f in UT.FORMATS:
        out[f] = sample(S0.X.DATA[(f, 'eval')][0], PER_FAMILY, rng)
    out['white-noise'], out['coarse-noise'] = noise_images(PER_FAMILY, rng)
    q = np.ascontiguousarray(ev[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    c = np.ascontiguousarray(train[:2 * N, :E].reshape(-1, 28, 28), dtype=np.float32)
    x, y = S0.X.DATA[('mnist', 'eval')]
    pick = rng.permutation(len(x))[:600]
    labelled = {'symbols': (c, np.repeat(np.arange(2 * N), E), q, np.repeat(np.arange(2 * N), E)), 'mnist': (x[pick[:300]].astype(np.float32), y[pick[:300]], x[pick[300:]].astype(np.float32), y[pick[300:]])}
    return out, labelled


# ---------------------------------------------------------------- measures, pure

def structure(path, horizon):
    """path (T, n, d) unit codes. S(tau), fitted as a + b tau + c tau^2 with non-negative parts; shares at the horizon; net over path length."""
    T = len(path)
    taus = np.arange(1, min(T, horizon + 1))
    s = np.array([((path[t:] - path[:-t]) ** 2).sum(-1).mean() for t in taus])
    (a, b, c), _ = nnls(np.c_[np.ones(len(taus)), taus, taus ** 2], s)
    total = a + b * horizon + c * horizon ** 2
    steps = np.sqrt(((path[1:] - path[:-1]) ** 2).sum(-1)).sum(0)
    net = np.sqrt(((path[-1] - path[0]) ** 2).sum(-1))
    return {'jitter': float(a / total), 'diffusion': float(b * horizon / total), 'trend': float(c * horizon ** 2 / total), 'movement': float(np.sqrt(total)), 'net_over_path': float((net / np.maximum(steps, 1e-12)).mean())}


def explained(fit_old, fit_new, old, new):
    """The share of the movement old -> new that a map fitted on another set predicts; negative if it adds movement."""
    fn = F.fit_map(fit_old, fit_new)
    return 1 - float(((fn(old) - new) ** 2).sum() / max(((old - new) ** 2).sum(), 1e-12))


def slow_directions(z, move, eps=1e-4):
    """Directions by rising lambda = w'C_move w / w'C w. z (n, d) codes; move (n, d) movements."""
    C = np.cov(z.T.astype(np.float64)) + eps * np.eye(z.shape[1])
    Cm = (move.astype(np.float64).T @ move.astype(np.float64)) / len(move)
    val, vec = eigh(Cm, C)
    return val, vec


def rayleigh(w, z, move, eps=1e-4):
    """w (d, k): the movement variance over code variance per direction, on another set."""
    C = np.cov(z.T.astype(np.float64)) + eps * np.eye(z.shape[1])
    Cm = (move.astype(np.float64).T @ move.astype(np.float64)) / len(move)
    return np.array([float(w[:, i] @ Cm @ w[:, i] / (w[:, i] @ C @ w[:, i])) for i in range(w.shape[1])])


def recall(stored, y_stored, query, y_query, w=None):
    if w is not None:
        stored, query = stored @ w, query @ w
    stored, query = S1.unit(stored), S1.unit(query)
    return float((y_stored[(query @ stored.T).argmax(1)] == y_query).mean())


def averaged(path, width=1):
    """Each snapshot's code averaged with its neighbours, so the map sees trend rather than jitter."""
    return np.stack([path[max(0, t - width):t + width + 1].mean(0) for t in range(len(path))])


def analyse(paths, labelled_paths, segments):
    """paths: family -> (T, n, d) unit codes. segments: name -> (start, end) snapshot indices one block apart."""
    names = list(paths)
    out = {}
    for seg, (a, b) in segments.items():
        horizon = b - a
        rec = {'structure': {f: structure(paths[f][a:b + 1], horizon) for f in names}}
        half = {f: len(paths[f]) and paths[f].shape[1] // 2 for f in names}
        # the map: fitted on the first half of a family, read on the second half of every family
        table = {}
        smooth = {f: averaged(paths[f]) for f in names}
        trend = {}
        for f in names:
            h = half[f]
            table[f], trend[f] = {}, {}
            for g in names:
                k = half[g]
                table[f][g] = explained(paths[f][a, :h], paths[f][b, :h], paths[g][a, k:], paths[g][b, k:])
                trend[f][g] = explained(smooth[f][a, :h], smooth[f][b, :h], smooth[g][a, k:], smooth[g][b, k:])
        rec['map_explains'], rec['map_explains_averaged'] = table, trend
        # slow directions, fitted on each family's first half
        slow = {}
        for f in names:
            h = half[f]
            val, vec = slow_directions(paths[f][a, :h], paths[f][b, :h] - paths[f][a, :h])
            row = {'lambda': [float(v) for v in val], 'read': {}}
            for g in names:
                k = half[g]
                row['read'][g] = {str(n): float(np.mean(rayleigh(vec[:, :n], paths[g][a, k:], paths[g][b, k:] - paths[g][a, k:]))) for n in SLOW}
            for lab, (st, ys, qu, yq) in labelled_paths.items():
                row.setdefault('recall', {})[lab] = {'fresh': recall(st[b], ys, qu[b], yq), 'stale': recall(st[a], ys, qu[b], yq), **{f'stale-slow-{n}': recall(st[a], ys, qu[b], yq, vec[:, :n]) for n in SLOW}, **{f'fresh-slow-{n}': recall(st[b], ys, qu[b], yq, vec[:, :n]) for n in SLOW}}
            slow[f] = row
        rec['slow'] = slow
        out[seg] = rec
    return out


# ---------------------------------------------------------------- paths

def encode_all(fn, fam, lab):
    return {k: fn(v) for k, v in fam.items()}, {k: (fn(a), b, fn(c), d) for k, (a, b, c, d) in lab.items()}


def stack(snaps):
    fam = {k: S1.unit(np.stack([s[0][k] for s in snaps])) for k in snaps[0][0]}
    lab = {k: (S1.unit(np.stack([s[1][k][0] for s in snaps])), snaps[0][1][k][1], S1.unit(np.stack([s[1][k][2] for s in snaps])), snaps[0][1][k][3]) for k in snaps[0][1]}
    return fam, lab


def segments_for(kind):
    return {'block-4': (0, SPAN), 'block-5': (SPAN, 2 * SPAN), 'block-6': (2 * SPAN, 3 * SPAN), 'blocks-4-6': (0, 3 * SPAN)}


def contrast_step(model, opt, base, rng, partner):
    x = torch.stack([SRL.augment_with(base, rng, SRL.AUGMENTS['full']), SRL.augment_with(base, rng, SRL.AUGMENTS['full'])], 1).reshape(-1, 28, 28)
    model.train()
    z = model(x)
    loss = SRL.info_nce(z @ z.T / SRL.T, partner)
    opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def run_key(task):
    """A contrastive encoder keeps learning: the pilots' glyph encoder, or one trained from scratch on natural images. stream: stationary, shift, natural."""
    kind, seed, lr, stream = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    fam, lab = families(seed)
    bg = S0.background_images(MATERIAL_SEED[seed])
    x_natural = natural()[0]
    pool_natural = x_natural[:-min(1000, len(x_natural) // 3)]
    shift_pool = np.concatenate([P.DOTS['train'].reshape(-1, 28, 28), *[S0.X.DATA[(f, 'eval')][0] for f in UT.FORMATS[:3]]]).astype(np.float32)
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    rng = nc.stream('sm_learning', seed, str(lr)) if kind == 'glyph' else nc.stream('sd_natural', seed, str(lr))

    def batch(pool):
        return torch.from_numpy(np.ascontiguousarray(pool[rng.integers(len(pool), size=TK.CLASSES)], dtype=np.float32))

    if kind == 'glyph':
        model = SM.encoder('unlabelled', seed)
    else:
        model = TK.build('base', 32, 30_000 + seed)
        opt0 = torch.optim.Adam(model.parameters(), TK.LR)
        for _ in range(SCRATCH_STEPS):
            contrast_step(model, opt0, batch(pool_natural), rng, partner)
    opt = torch.optim.Adam(model.parameters(), lr)
    ema = {b: copy.deepcopy(model).requires_grad_(False).eval() for b in BETAS}
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    code = lambda m: (lambda x: np_(S0.batched(lambda y: m(y), x)))
    snaps = {'working': [encode_all(code(model), fam, lab)], **{f'ema-{b}': [encode_all(code(ema[b]), fam, lab)] for b in BETAS}}
    at = 0
    for b in (4, 5, 6):
        for _ in range(BLOCK):
            if kind == 'glyph' and not (stream != 'stationary' and b >= 5):
                base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=TK.CLASSES), rng.integers(n_r, size=TK.CLASSES)], dtype=np.float32))
            elif kind == 'glyph' and stream == 'shift':
                base = batch(shift_pool)
            else:
                base = batch(pool_natural)
            contrast_step(model, opt, base, rng, partner)
            with torch.no_grad():
                for beta, e in ema.items():
                    for p, q in zip(e.parameters(), model.parameters()):
                        p.mul_(beta).add_(q, alpha=1 - beta)
            at += 1
            if at % WINDOW == 0:
                snaps['working'].append(encode_all(code(model), fam, lab))
                for beta, e in ema.items():
                    snaps[f'ema-{beta}'].append(encode_all(code(e), fam, lab))
    out = {}
    for name, s in snaps.items():
        paths, labelled = stack(s)
        out[name] = analyse(paths, labelled, segments_for(stream)) if name == 'working' else {seg: {'structure': {f: structure(paths[f][a:e + 1], e - a) for f in paths}, 'recall': {k: {'fresh': recall(v[0][e], v[1], v[2][e], v[3]), 'stale': recall(v[0][a], v[1], v[2][e], v[3])} for k, v in labelled.items()}} for seg, (a, e) in segments_for(stream).items()}
    return f'key/{kind}/{seed}/{lr}/{stream}', {'kind': kind, 'seed': seed, 'lr': lr, 'stream': stream, 'n': {k: len(v) for k, v in fam.items()}, 'results': out}, S0.timing(t0, c0)


def run_host(task):
    """The host, from the windows pilot's saved weights: a different network and objective, on the same inputs."""
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    fam, lab = families(seed)
    host = W.HostCopy(seed).load(S0.checkpoint_path('sparse48', seed))
    snaps = [encode_all(host, fam, lab)]
    for at in range(WINDOW, 3 * BLOCK + 1, WINDOW):
        host.load(W.host_path(seed, order, at))
        snaps.append(encode_all(host, fam, lab))
    paths, labelled = stack(snaps)
    return f'host/{seed}/{order}', {'kind': 'host', 'seed': seed, 'order': order, 'results': {'working': analyse(paths, labelled, segments_for('host'))}}, S0.timing(t0, c0)


def run(task):
    return run_host(task[1:]) if task[0] == 'host' else run_key(task)


def main():
    t0 = time.perf_counter()
    S0.prepare()
    RUNS.mkdir(parents=True, exist_ok=True)
    natural()
    tasks = [('host', s, o) for s in S0.SEEDS for o in SC.ORDERS]
    tasks += [('glyph', s, lr, k) for s in S0.SEEDS for lr in SC.LRS for k in ('stationary', 'shift', 'natural')]
    tasks += [('scratch', s, lr, 'natural') for s in (392, 393, 404, 405) for lr in SC.LRS]
    res = S0.pool_map(DUR.durable(run_named, RUNS / 'done', '{0}'), [('-'.join(str(v) for v in t), tuple(t)) for t in tasks])
    n = len(natural()[0])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory measurement; images only measure', 'natural_images_fetched': n, 'window': WINDOW, 'block': BLOCK, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder paths with snapshots and measures', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


def run_named(item):
    return run(item[1])


if __name__ == '__main__':
    main()
