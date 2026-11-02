"""A relational machine at the sense level (docs/sense-relational-v1-pilot.md): the core's design with a different relation, 'same thing, seen again'. Exploratory, pilot seeds.

Stages (argv[2]):
  features  Is identity still in the host's encoder output? Heads trained on stored features of background symbols, read on unseen symbols. The host's encoder, its code, and an untrained encoder as a control.
  machines  A small encoder trained from scratch with its matching head, from two sources of 'same': two renderings of a symbol, or two augmentations of one image with no symbol knowledge.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_relational pilot features
"""
import json
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import device as DEV
from experiments import durable as DUR
from experiments import shared_core_stage0 as S0
from experiments import shared_core_value as V
from experiments import streaming_retrieval as SR
from experiments import translation_stable_keys as TK
from src import number_core as ncore

nc, X, FI = S0.nc, S0.X, S0.FI
NAME = 'sense-relational-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CLASSES, T = TK.CLASSES, TK.TEMPERATURE
HEAD_STEPS, MACHINE_STEPS, READS = 4000, 8000, (2000, 8000)
HIDDEN, STRUCT = 64, 48
FEATURES = ('host-encoder', 'host-code', 'untrained-encoder')
HEADS = ('linear', 'mlp', 'comparator')
MACHINES = (('comparator-dense', 'renderings'), ('comparator-sparse', 'renderings'), ('cosine', 'augment'), ('comparator-dense', 'augment'), ('comparator-sparse', 'augment'))


class Comparator(nn.Module):
    """g(a, b) on two codes, as the core's comparator without a relation token: two hidden layers. Sparse: 48 first-layer units reading one coordinate per operand, moved by RigL as in sparse48."""

    def __init__(self, d, sparse, seed):
        super().__init__()
        self.d, self.sparse = d, sparse
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(nc.stream('sr_comparator', seed, d, str(sparse)).integers(2 ** 62)))
            first = nn.Linear(2 * d, STRUCT if sparse else HIDDEN)
            if sparse:
                FI.CODE = d
                FI.reader_mask(first, d, nc.stream('sr_mask', seed, 'a'), nc.stream('sr_mask', seed, 'b'))
                first.register_buffer('swaps', torch.zeros((), dtype=torch.long))
            self.first = first
            self.rest = nn.Sequential(nn.ReLU(), nn.Linear(STRUCT if sparse else HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, a, b):
        return self.rest(self.first(torch.cat([a, b], -1))).squeeze(-1)

    def matrix(self, x, y, size=8192):
        """Scores (len(x), len(y))."""
        n, m = len(x), len(y)
        a, b = x[:, None].expand(-1, m, -1).reshape(n * m, -1), y[None].expand(n, -1, -1).reshape(n * m, -1)
        return torch.cat([self(a[i:i + size], b[i:i + size]) for i in range(0, n * m, size)]).reshape(n, m)


def info_nce(score, partner):
    """score (n, n), self excluded, partner (n,): which item is the same thing."""
    return F.cross_entropy(score.masked_fill(torch.eye(len(score), dtype=torch.bool), float('-inf')), partner)


def pair_labels():
    return torch.arange(2 * CLASSES).reshape(-1, 2).flip(1).reshape(-1)


def rigl(comp, step, end, opt, phase):
    """FI.rigl_step around one update: phase 'before' collects the dense gradient, 'after' moves connections."""
    lin = comp.first
    due = comp.sparse and step % FI.DT == 0 and step <= end
    if phase == 'before' and due:
        lin.collect, lin.probe = True, torch.zeros_like(lin.weight, requires_grad=True)
    elif phase == 'after' and due:
        lin.collect = False
        FI.CODE = comp.d
        FI.rigl_step(lin, [torch.arange(comp.d), torch.arange(comp.d, 2 * comp.d)], opt, FI.ALPHA * (1 + np.cos(np.pi * step / end)) / 2, 'zero', None, None)
        del lin.probe


@torch.no_grad()
def reader_use(comp, codes):
    """Live first-layer units, and distinct code coordinates they read. A unit is live if it fires on a sampled pair."""
    if not comp.sparse:
        return None
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(codes), size=(2, 2000))
    act = (comp.first(torch.cat([codes[i], codes[j]], 1)) > 0).any(0)
    m = comp.first.mask[act][:, :2 * comp.d]
    reads = (m[:, :comp.d] + m[:, comp.d:]).sum(0)
    return {'live_units': int(act.sum()), 'distinct_coordinates': int((reads > 0).sum()), 'of': comp.d}


@torch.no_grad()
def shape(t):
    x = t.reshape(-1, t.shape[-1]).double()
    x = x - x.mean(0)
    val = torch.linalg.eigvalsh(x.T @ x / len(x)).clamp_min(0)
    return {'participation_ratio': float(val.sum() ** 2 / val.pow(2).sum()), 'top_share': float(val.max() / val.sum())}


@torch.no_grad()
def identity_reads(fn, comp, train, ev, counts):
    """Unseen symbols. Cosine on the learned code, as memory's addresses use it, and through the comparator against each symbol's prototype."""
    t, q = S0.split_codes(fn, train, ev)
    out = {'shape': shape(t[:S0.LOADS[0]])}
    for L in S0.LOADS:
        r = S0.address_reading(t[:L], q[:L], counts[L])
        row = {k: r[k] for k in ('identity', 'store', 'same_symbol_similarity', 'other_symbol_similarity')}
        if comp is not None:
            proto = t[:L].mean(1)
            flat = q[:L].reshape(L * q.shape[1], -1)
            near = comp.matrix(flat, proto).argmax(1).cpu().numpy()
            truth = np.repeat(np.arange(L), q.shape[1])
            row['identity_comparator'] = float((near == truth).mean())
            row['store_comparator'] = float((counts[L][near] == counts[L][truth]).mean())
        out[str(L)] = row
    return out


def timing(t0, c0):
    return S0.timing(t0, c0)


# ---------------------------------------------------------------- step 1

@torch.no_grad()
def feature_fn(seed, feature):
    enc, model = V.host(seed)
    for m in enc.values():
        m.eval()
    if feature == 'host-encoder':
        return lambda x: enc['image'](x)
    if feature == 'host-code':
        return lambda x: model.proj['image'](enc['image'](x))
    torch.manual_seed(int(nc.stream('sr_untrained', seed).integers(2 ** 62)))
    device = DEV.NAME if DEV.ACTIVE else 'cpu'
    with torch.device('cpu'):
        net = ncore.ImageEncoder(64)
    net = net.to(device).requires_grad_(False)
    return lambda x: net(x)


def fit_head(task):
    seed, feature, head = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    fn = feature_fn(seed, feature)
    train, ev, counts = S0.test_material(seed)
    bg = S0.background_images(seed)
    n_cls, n_r = bg.shape[:2]
    with torch.no_grad():
        F_bg = S0.batched(fn, bg.reshape(-1, 28, 28)).reshape(n_cls, n_r, -1)
    mean, sd = F_bg.reshape(-1, F_bg.shape[-1]).mean(0), F_bg.reshape(-1, F_bg.shape[-1]).std(0).clamp_min(1e-6)
    standard = lambda f: (f - mean) / sd
    d = F_bg.shape[-1]
    rng = nc.stream('sr_head', seed, feature, head)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(int(rng.integers(2 ** 62)))
        net = {'linear': nn.Linear(d, 32), 'mlp': nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 32)), 'comparator': Comparator(d, False, seed)}[head]
    net = net.to(DEV.NAME) if DEV.ACTIVE else net
    opt = torch.optim.Adam(net.parameters(), TK.LR)
    partner = pair_labels()
    for _ in range(HEAD_STEPS):
        cls = rng.choice(n_cls, CLASSES, replace=False)
        view = np.stack([rng.choice(n_r, 2, replace=False) for _ in cls])
        x = standard(F_bg[torch.as_tensor(cls)[:, None], torch.as_tensor(view)].reshape(2 * CLASSES, d))
        if head == 'comparator':
            score = net.matrix(x, x)
        else:
            z = F.normalize(net(x), dim=1)
            score = z @ z.T / T
        loss = info_nce(score, partner)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        code = (lambda x: standard(fn(x))) if head == 'comparator' else (lambda x: net(standard(fn(x))))
        out = identity_reads(code, net if head == 'comparator' else None, train, ev, counts)
    out.update({'parameters': sum(p.numel() for p in net.parameters()), 'final_loss': float(loss.detach())})
    return f'{seed}/{feature}/{head}', out, timing(t0, c0)


# ---------------------------------------------------------------- step 2

def augment(x, rng):
    """Two random views of the images: rotation within 10 degrees, scale 0.9 to 1.1, shift within 2 pixels, a little pixel noise."""
    n = len(x)
    a, s = np.radians(rng.uniform(-10, 10, n)), rng.uniform(.9, 1.1, n)
    sh = rng.uniform(-2, 2, (n, 2)) * 2 / 28
    theta = torch.from_numpy(np.stack([np.stack([s * np.cos(a), -s * np.sin(a), sh[:, 0]], 1), np.stack([s * np.sin(a), s * np.cos(a), sh[:, 1]], 1)], 1).astype(np.float32))
    grid = F.affine_grid(theta, (n, 1, 28, 28), align_corners=False)
    out = F.grid_sample(x[:, None], grid, align_corners=False, padding_mode='zeros')[:, 0]
    return (out + torch.from_numpy(rng.normal(0, .05, (n, 28, 28)).astype(np.float32))).clamp(0, 1)


def fit_machine(task):
    seed, match, source = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    train, ev, counts = S0.test_material(seed)
    bg = S0.background_images(seed)
    n_cls, n_r = bg.shape[:2]
    rng = nc.stream('sr_machine', seed, match, source)
    model = TK.build('base', 32, 10_000 + seed)
    comp = None
    params = list(model.parameters())
    if match != 'cosine':
        comp = Comparator(32, match.endswith('sparse'), seed)
        comp = comp.to(DEV.NAME) if DEV.ACTIVE else comp
        params += list(comp.parameters())
    opt = torch.optim.Adam(params, TK.LR)
    partner = pair_labels()
    end = FI.END_FRACTION * MACHINE_STEPS
    scale = np.sqrt(32)  # a unit code has coordinates near 0.18: the comparator sees it at unit variance
    out = {'reads': {}}
    for step in range(1, MACHINE_STEPS + 1):
        if source == 'renderings':
            cls = rng.choice(n_cls, CLASSES, replace=False)
            view = np.stack([rng.choice(n_r, 2, replace=False) for _ in cls])
            x = torch.from_numpy(np.ascontiguousarray(bg[cls[:, None], view].reshape(-1, 28, 28), dtype=np.float32))
        else:  # no symbol knowledge: random images, so a batch may hold two renderings of one symbol
            base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=CLASSES), rng.integers(n_r, size=CLASSES)], dtype=np.float32))
            x = torch.stack([augment(base, rng), augment(base, rng)], 1).reshape(-1, 28, 28)
        if comp is not None:
            rigl(comp, step, end, opt, 'before')
        model.train()
        z = model(x)
        score = (z @ z.T / T) if comp is None else comp.matrix(z * scale, z * scale)
        loss = info_nce(score, partner)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.)
        opt.step()
        if comp is not None:
            rigl(comp, step, end, opt, 'after')
        if step in READS:
            model.eval()
            with torch.no_grad():
                code = lambda b: model(b) * scale
                out['reads'][str(step)] = identity_reads(code, comp, train, ev, counts)
                if comp is not None:
                    t = S0.batched(code, train[:90].reshape(-1, 28, 28))
                    out['reads'][str(step)]['readers'] = reader_use(comp, t)
    out.update({'parameters': sum(p.numel() for p in params), 'final_loss': float(loss.detach())})
    return f'{seed}/{match}/{source}', out, timing(t0, c0)


def frozen_reference(task):
    seed, = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    train, ev, counts = S0.test_material(seed)
    key = SR.key_encoder(seed).to(DEV.NAME if DEV.ACTIVE else 'cpu').requires_grad_(False)
    with torch.no_grad():
        out = {'reads': {'8000': identity_reads(lambda b: key(b), None, train, ev, counts)}}
    return f'{seed}/cosine/renderings', out, timing(t0, c0)


# ---------------------------------------------------------------- robustness

def font_material(seed):
    """Unseen pool symbols drawn in more than one font: stored renderings in the symbol's own font, queries in another. Same-font queries for the same symbols beside them."""
    pool = S0.PS.survivors()
    cover = TK.fonts_covering(sorted(pool))  # every font drawing the symbol; PS.covering keeps only its one
    multi = [c for c in sorted(pool) if len(cover.get(c, ())) >= 2 and any(f not in pool[c] for f in cover[c])]
    rng = nc.stream('sr_fonts', seed)
    pick = [int(c) for c in rng.permutation(multi)[:S0.LOADS[0]]]
    train = np.stack([S0.PS.renderings(c, pool[c], seed, 'train', S0.TRAIN_RENDERS) for c in pick])
    same = np.stack([S0.PS.renderings(c, pool[c], seed, 'eval', S0.EVAL_RENDERS) for c in pick])
    other = [[f for f in cover[c] if f not in pool[c]] for c in pick]
    cross = np.stack([S0.PS.renderings(c, [o[int(rng.integers(len(o)))]], seed, 'eval', S0.EVAL_RENDERS) for c, o in zip(pick, other)])
    return train, same, cross, len(multi)


@torch.no_grad()
def robust_reads(fn, seed):
    train, ev, counts = S0.test_material(seed)
    t, q = S0.split_codes(fn, train, ev)
    out = {str(L): {k: v for k, v in S0.address_reading(t[:L], q[:L], counts[L]).items() if k in ('identity', 'store', 'same_symbol_similarity', 'other_symbol_similarity')} for L in S0.LOADS}
    ftrain, fsame, fcross, n_multi = font_material(seed)
    fake = np.repeat(np.arange(1, 10), len(ftrain) // 9 + 1)[:len(ftrain)]
    ft, fq = S0.split_codes(fn, ftrain, fsame)
    _, fx = S0.split_codes(fn, ftrain[:1], fcross)
    out['fonts'] = {'symbols': len(ftrain), 'symbols_with_a_second_font': n_multi, 'same_font': S0.address_reading(ft, fq, fake)['identity'], 'other_font': S0.address_reading(ft, fx, fake)['identity'],
                    'other_font_same_symbol_similarity': S0.address_reading(ft, fx, fake)['same_symbol_similarity'], 'other_font_other_symbol_similarity': S0.address_reading(ft, fx, fake)['other_symbol_similarity']}
    return out


AUGMENTS = {'full': (10., .1, 2., .05), 'no-rotation-scale': (0., 0., 2., .05), 'shift-only': (0., 0., 2., 0.)}


def augment_with(x, rng, cfg):
    deg, sc, px, noise = cfg
    n = len(x)
    a, s = np.radians(rng.uniform(-deg, deg, n)), rng.uniform(1 - sc, 1 + sc, n)
    sh = rng.uniform(-px, px, (n, 2)) * 2 / 28
    theta = torch.from_numpy(np.stack([np.stack([s * np.cos(a), -s * np.sin(a), sh[:, 0]], 1), np.stack([s * np.sin(a), s * np.cos(a), sh[:, 1]], 1)], 1).astype(np.float32))
    out = F.grid_sample(x[:, None], F.affine_grid(theta, (n, 1, 28, 28), align_corners=False), align_corners=False, padding_mode='zeros')[:, 0]
    return (out + torch.from_numpy(rng.normal(0, noise, (n, 28, 28)).astype(np.float32))).clamp(0, 1) if noise else out


def fit_robust(task):
    seed, name = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    device = DEV.NAME if DEV.ACTIVE else 'cpu'
    if name == 'frozen-renderings':
        key = SR.key_encoder(seed).to(device).requires_grad_(False)
        return f'{seed}/{name}', robust_reads(lambda b: key(b), seed), timing(t0, c0)
    if name == 'pixels':
        return f'{seed}/{name}', robust_reads(lambda b: b.reshape(len(b), -1), seed), timing(t0, c0)
    if name == 'untrained':
        net = TK.build('base', 32, 20_000 + seed).requires_grad_(False)
        return f'{seed}/{name}', robust_reads(lambda b: net(b), seed), timing(t0, c0)
    bg = S0.background_images(seed)
    n_cls, n_r = bg.shape[:2]
    rng = nc.stream('sr_robust', seed, name)
    model = TK.build('base', 32, 30_000 + seed)
    opt = torch.optim.Adam(model.parameters(), TK.LR)
    partner = pair_labels()
    for _ in range(MACHINE_STEPS):
        base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=CLASSES), rng.integers(n_r, size=CLASSES)], dtype=np.float32))
        z = model(torch.stack([augment_with(base, rng, AUGMENTS[name]), augment_with(base, rng, AUGMENTS[name])], 1).reshape(-1, 28, 28))
        loss = info_nce(z @ z.T / T, partner)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval().requires_grad_(False)
    LM_path = RUNS / 'encoders' / f'{name}-seed{seed}.pt'
    S0.LM.save_atomic(model.state_dict(), LM_path)
    return f'{seed}/{name}', {**robust_reads(lambda b: model(b), seed), 'checkpoint': {'file': str(LM_path.relative_to(X.ROOT)), 'sha256': nc.sha256(LM_path)}}, timing(t0, c0)


# ---------------------------------------------------------------- fan-in and two-stage reading

SHORTLISTS = (1, 3, 5, 10, 20)
# arm: (how the encoder trains, fan-in per operand or None for dense, RigL)
FANIN = {'through-k4-rigl': ('through', 4, True), 'through-k8-rigl': ('through', 8, True), 'through-k16-rigl': ('through', 16, True), 'through-k8-fixed': ('through', 8, False),
         'joint-dense': ('joint', None, False), 'joint-k8-rigl': ('joint', 8, True), 'on-top-dense': ('on-top', None, False), 'on-top-k8-rigl': ('on-top', 8, True)}


class WideComparator(Comparator):
    """The same reader: sparse first layer of 48 units, dense second layer, score; each first-layer unit reads k coordinates per operand."""

    def __init__(self, d, k, rigl_on, seed):
        nn.Module.__init__(self)
        self.d, self.sparse = d, bool(k and rigl_on)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(nc.stream('sr_comparator', seed, d, str(k), str(rigl_on)).integers(2 ** 62)))
            first = nn.Linear(2 * d, STRUCT if k else HIDDEN)
            if k:
                FI.CODE = d
                FI.reader_mask(first, d // k, nc.stream('sr_mask', seed, 'a', k), nc.stream('sr_mask', seed, 'b', k))
                first.register_buffer('swaps', torch.zeros((), dtype=torch.long))
            self.first = first
            self.rest = nn.Sequential(nn.ReLU(), nn.Linear(STRUCT if k else HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))


@torch.no_grad()
def two_stage(comp, proto, q, truth, counts=None):
    """Cosine shortlists K stored symbols, the comparator picks among them. K = all stored is comparator alone. Passes through the comparator per query = K."""
    u_q, u_p = F.normalize(q, dim=-1), F.normalize(proto, dim=-1)
    order = (u_q @ u_p.T).argsort(1, descending=True)
    out = {'cosine_alone': float((order[:, 0].cpu().numpy() == truth).mean())}
    for K in (*SHORTLISTS, len(proto)):
        top = order[:, :K]
        a = q[:, None].expand(-1, K, -1).reshape(-1, q.shape[-1])
        b = proto[top.reshape(-1)]
        s = torch.cat([comp(a[i:i + 8192], b[i:i + 8192]) for i in range(0, len(a), 8192)]).reshape(len(q), K)
        pick = top[torch.arange(len(q)), s.argmax(1)].cpu().numpy()
        out[f'K{K}'] = float((pick == truth).mean())
    return out


@torch.no_grad()
def fanin_reads(code, comp, seed, fonts):
    train, ev, counts = S0.test_material(seed)
    t, q = S0.split_codes(code, train, ev)
    out = {'shape': shape(t[:S0.LOADS[0]]), 'readers': reader_use_k(comp, t[:S0.LOADS[0]].reshape(-1, t.shape[-1]))}
    for L in S0.LOADS:
        out[str(L)] = two_stage(comp, t[:L].mean(1), q[:L].reshape(L * q.shape[1], -1), np.repeat(np.arange(L), q.shape[1]))
    ftrain, fsame, fcross = fonts
    ft, fq = S0.split_codes(code, ftrain, fsame)
    _, fx = S0.split_codes(code, ftrain[:1], fcross)
    truth = np.repeat(np.arange(len(ftrain)), fx.shape[1])
    out['same_font'] = two_stage(comp, ft.mean(1), fq.reshape(-1, fq.shape[-1]), truth)
    out['other_font'] = two_stage(comp, ft.mean(1), fx.reshape(-1, fx.shape[-1]), truth)
    return out


@torch.no_grad()
def reader_use_k(comp, codes):
    if not hasattr(comp.first, 'mask'):
        return None
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(codes), size=(2, 2000))
    act = (comp.first(torch.cat([codes[i], codes[j]], 1)) > 0).any(0)
    m = comp.first.mask[act][:, :2 * comp.d]
    reads = (m[:, :comp.d] + m[:, comp.d:]).sum(0)
    return {'live_units': int(act.sum()), 'distinct_coordinates': int((reads > 0).sum()), 'of': comp.d, 'top4_share': float(reads.sort(descending=True).values[:4].sum() / reads.sum())}


def font_cache(seed):
    path = RUNS / 'fonts' / f'seed{seed}.npz'
    if not path.exists():
        import io
        a, b, c, _ = font_material(seed)
        buf = io.BytesIO()
        np.savez_compressed(buf, train=a, same=b, cross=c)
        DUR.atomic_write(path, buf.getvalue())
    z = np.load(path)
    return z['train'], z['same'], z['cross']


def fit_fanin(task):
    seed, name = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    how, k, rigl_on = FANIN[name]
    bg = S0.background_images(seed)
    n_cls, n_r = bg.shape[:2]
    rng = nc.stream('sr_fanin', seed, name)
    model = TK.build('base', 32, 40_000 + seed)
    if how == 'on-top':
        model.load_state_dict(torch.load(RUNS / 'encoders' / f'full-seed{seed}.pt', map_location='cpu'))
    comp = WideComparator(32, k, rigl_on, seed)
    comp = comp.to(DEV.NAME) if DEV.ACTIVE else comp
    params = list(comp.parameters()) + ([] if how == 'on-top' else list(model.parameters()))
    opt = torch.optim.Adam(params, TK.LR)
    partner = pair_labels()
    end, scale = FI.END_FRACTION * MACHINE_STEPS, np.sqrt(32)
    losses = []
    for step in range(1, MACHINE_STEPS + 1):
        base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=CLASSES), rng.integers(n_r, size=CLASSES)], dtype=np.float32))
        x = torch.stack([augment_with(base, rng, AUGMENTS['full']), augment_with(base, rng, AUGMENTS['full'])], 1).reshape(-1, 28, 28)
        rigl(comp, step, end, opt, 'before')
        if how == 'on-top':
            with torch.no_grad():
                z = model(x)
        else:
            z = model(x)
        loss = info_nce(comp.matrix(z * scale, z * scale), partner)
        if how == 'joint':
            loss = loss + info_nce(z @ z.T / T, partner)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.)
        opt.step()
        rigl(comp, step, end, opt, 'after')
        losses.append(float(loss.detach()))
    model.eval().requires_grad_(False)
    with torch.no_grad():
        out = fanin_reads(lambda b: model(b) * scale, comp, seed, font_cache(seed))
    out.update({'loss_last_200': float(np.mean(losses[-200:])), 'comparator_parameters': sum(p.numel() for p in comp.parameters())})
    return f'{seed}/{name}', out, timing(t0, c0)


def main(stage):
    t0 = time.perf_counter()
    for s in S0.SEEDS:
        S0.test_material(s)
    if stage == 'fanin':
        for s_ in S0.SEEDS:
            font_cache(s_)
        res = S0.pool_map(DUR.durable(fit_fanin, RUNS / 'done-fanin', '{0}-{1}'), [(s, n) for s in S0.SEEDS for n in FANIN])
    elif stage == 'robust':
        res = S0.pool_map(DUR.durable(fit_robust, RUNS / 'done-robust', '{0}-{1}'), [(s, n) for s in S0.SEEDS for n in ('pixels', 'untrained', 'frozen-renderings', *AUGMENTS)])
    elif stage == 'features':
        res = S0.pool_map(DUR.durable(fit_head, RUNS / 'done-features', '{0}-{1}-{2}'), [(s, f, h) for s in S0.SEEDS for f in FEATURES for h in HEADS])
    else:
        res = S0.pool_map(DUR.durable(frozen_reference, RUNS / 'done-machines', 'frozen-{0}'), [(s,) for s in S0.SEEDS])
        res += S0.pool_map(DUR.durable(fit_machine, RUNS / 'done-machines', '{0}-{1}-{2}'), [(s, m, src) for s in S0.SEEDS for m, src in MACHINES])
    S0.save_json(f'{REPORT}-{stage}.json', {'status': 'exploratory, pilot seeds', 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-{stage}-runtime.json', S0.runtime({'phase': stage, 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main(sys.argv[2] if len(sys.argv) > 2 else 'features')
