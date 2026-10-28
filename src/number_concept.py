"""Data, complex networks, losses, training and evaluation for number concept network v1.

Specification: docs/number-concept-network-v1-spec.md (draft; frozen later as
docs/number-concept-network-v1-contract.md). Stage 2 receives only dice pixels and
within-dice classes, the frozen successor S and, in anchored arms, the saved digit-1
centroid; digit images, other centroids and probe weights reach only the evaluator.
"""
from collections import Counter
import gzip
import hashlib
from pathlib import Path
import zlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT_SEED = 2026102801
B = 256
D = 8
TEMPERATURE = .1
NORM_FLOOR, MODRELU_EPS, SD_FLOOR = 1e-6, 1e-6, 1e-6
LR, BETAS, ADAM_EPS, CLIP = 1e-3, (.9, .999), 1e-8, 5.
PROBE_LR, PROBE_STEPS = .01, 500
DIGITS = tuple(range(10))
FACES = tuple(range(1, 7))
RANDOM_COUNTS = tuple(range(1, 10))
VARIANTS = ('rotation', 'mlp')
ARMS = {  # arm: (pairs, lambda_succ, lambda_anchor, donor operator)
    'U': ('correct', 1., 0., False),
    'A': ('correct', 1., 1., False),
    'SH': ('shuffled', 1., 1., False),
    'AO': (None, 0., 1., False),
    'SW': ('correct', 1., 1., True),
}
MNIST_FILES = {  # official IDX archives; SHA-256 of the .gz files (MD5s match torchvision's list)
    'train_images': ('train-images-idx3-ubyte.gz', '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'),
    'train_labels': ('train-labels-idx1-ubyte.gz', '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'),
    'test_images': ('t10k-images-idx3-ubyte.gz', '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6'),
    'test_labels': ('t10k-labels-idx1-ubyte.gz', 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'),
}
MNIST_VALIDATION_PER_DIGIT = 600
DICE_POOLS = {'standard': {'train': 2000, 'validation': 300, 'test': 500},
              'random': {'validation': 300, 'test': 500}}
RADIUS, BRIGHTNESS, JITTER, NOISE_SD = (1.4, 2.2), (.7, 1.), 2., .02
RANDOM_BOX = (4., 23.)
PLACEMENT_CAP = 1_000_000
NOMINAL = (7., 13.5, 20.)
SUBPIXEL = np.array([-.375, -.125, .125, .375])
GENERATOR_VERSION = 'ncn-dice-v1'


class NumericalFailure(RuntimeError):
    """Nonfinite loss, gradient or latent: stops the run; its seed is not replaced."""


class PlacementExhausted(RuntimeError):
    """A preassigned image found no valid layout within the proposal cap; dataset construction aborts."""


# ---------------------------------------------------------------- random streams

def _key(k):
    return zlib.crc32(k.encode()) if isinstance(k, str) else int(k)


def stream(*keys):
    """Named independent generator: extra draws in one stream never move another."""
    return np.random.default_rng(np.random.SeedSequence([ROOT_SEED, *map(_key, keys)]))


# ---------------------------------------------------------------- MNIST

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _idx(path):
    data = gzip.decompress(Path(path).read_bytes())
    dims = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i:8 + 4 * i], 'big') for i in range(dims))
    return np.frombuffer(data, np.uint8, offset=4 + 4 * dims).reshape(shape).copy()


def load_mnist(directory):
    """Verified official MNIST: uint8 images and labels plus source checksums."""
    out, checks = {}, {}
    for key, (name, digest) in MNIST_FILES.items():
        path = Path(directory) / name
        got = sha256(path)
        if got != digest:
            raise ValueError(f'{name}: SHA-256 {got} does not match {digest}')
        out[key] = _idx(path)
        checks[name] = got
    return out, checks


def mnist_split(train_labels):
    """Fixed stratified validation reserve: 600 per digit; the rest is the training partition."""
    rng = stream('mnist_split')
    val = np.sort(np.concatenate([rng.permutation(np.flatnonzero(train_labels == d))[:MNIST_VALIDATION_PER_DIGIT]
                                  for d in DIGITS]))
    train = np.setdiff1d(np.arange(len(train_labels)), val)
    return train, val


def class_subset(labels, classes, per_class, *keys):
    """Seeded per-class subset (pilot pools); positions into `labels`, sorted."""
    rng = stream('subset', *keys)
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in classes]))


# ---------------------------------------------------------------- dice

def face_centers(count):
    lo, mid, hi = NOMINAL
    c = (mid, mid)
    layouts = {1: [c], 2: [(lo, lo), (hi, hi)], 3: [(lo, lo), c, (hi, hi)],
               4: [(lo, lo), (lo, hi), (hi, lo), (hi, hi)], 5: [(lo, lo), (lo, hi), c, (hi, lo), (hi, hi)],
               6: [(lo, lo), (mid, lo), (hi, lo), (lo, hi), (mid, hi), (hi, hi)]}
    return np.array(layouts[count])  # (row, col); pixel centres at integer coordinates


def quarter_turn(centers, k):
    """Rotate (row, col) positions by k quarter-turns about the image centre (13.5, 13.5)."""
    c = np.array(centers, dtype=float)
    for _ in range(int(k) % 4):
        c = np.stack([c[:, 1], 27. - c[:, 0]], 1)
    return c


def min_separation(p):
    """Smallest pairwise centre distance per proposal; p has shape (m, n, 2)."""
    n = p.shape[1]
    if n < 2:
        return np.full(len(p), np.inf)
    i, j = np.triu_indices(n, 1)
    return np.sqrt(((p[:, i] - p[:, j]) ** 2).sum(-1)).min(1)


def place(layout, count, radius, rng, cap=PLACEMENT_CAP):
    """Exact rejection sampling of positions only, for a fixed preassigned radius.

    Proposals are drawn in a fixed doubling batch schedule, so the accepted layout and
    the proposal count are deterministic functions of the image's placement stream.
    """
    need, used, size = 2 * radius + 1, 0, 16
    while used < cap:
        m = min(size, cap - used)
        if layout == 'standard':
            p = face_centers(count)[None] + rng.uniform(-JITTER, JITTER, (m, count, 2))
        else:
            p = rng.uniform(*RANDOM_BOX, (m, count, 2))
        ok = np.flatnonzero(min_separation(p) >= need)
        if len(ok):
            return p[ok[0]], used + ok[0] + 1
        used += m
        size = min(size * 2, 65536)
    raise PlacementExhausted(f'{layout} count {count} radius {radius:.4f}: {cap} proposals exhausted')


def render(centers, radius, brightness, noise_rng):
    """4x4 subpixel coverage of the pip union, scaled by brightness, plus clipped Gaussian noise."""
    g = (np.arange(28)[:, None] + SUBPIXEL[None]).reshape(-1)
    d2 = np.full((112, 112), np.inf)
    for r, c in centers:
        d2 = np.minimum(d2, (g[:, None] - r) ** 2 + (g[None, :] - c) ** 2)
    cover = (d2 <= radius ** 2).reshape(28, 4, 28, 4).mean((1, 3))
    x = brightness * cover + noise_rng.normal(0, NOISE_SD, (28, 28))
    return np.clip(x, 0, 1).astype(np.float32)


def image_id(count, index):
    return count * 100_000 + index


def dice_metadata(layout, split, counts, per_count):
    """Preassign radius, brightness and quarter-turn for every image ID before any placement.

    One stream per (layout, split), drawn in ID order with count-independent distributions.
    """
    ids = np.array([image_id(c, i) for c in counts for i in range(per_count)])
    labels = np.repeat(np.array(counts), per_count)
    rng = stream('dice_meta', layout, split)
    return {'ids': ids, 'labels': labels, 'radius': rng.uniform(*RADIUS, len(ids)),
            'brightness': rng.uniform(*BRIGHTNESS, len(ids)), 'quarter': rng.integers(0, 4, len(ids))}


def build_dice(layout, split, counts=None, per_count=None, cap=PLACEMENT_CAP):
    """Fixed dice pool. On exhaustion the exception carries the assignment manifest."""
    counts = counts or (FACES if layout == 'standard' else RANDOM_COUNTS)
    per_count = per_count or DICE_POOLS[layout][split]
    meta = dice_metadata(layout, split, counts, per_count)
    n = len(meta['ids'])
    images = np.zeros((n, 28, 28), np.float32)
    centers = np.full((n, max(counts), 2), np.nan)
    attempts = np.zeros(n, np.int64)
    for k, (iid, count) in enumerate(zip(meta['ids'], meta['labels'])):
        try:
            p, used = place(layout, int(count), meta['radius'][k], stream('dice_place', layout, split, int(iid)), cap)
        except PlacementExhausted as e:
            e.manifest = {k2: v.tolist() for k2, v in meta.items()}
            e.image = {'id': int(iid), 'count': int(count), 'radius': float(meta['radius'][k]),
                       'brightness': float(meta['brightness'][k]), 'quarter': int(meta['quarter'][k])}
            raise
        p = quarter_turn(p, meta['quarter'][k])
        centers[k, :count] = p
        attempts[k] = used
        images[k] = render(p, meta['radius'][k], meta['brightness'][k], stream('dice_noise', layout, split, int(iid)))
    return {**meta, 'images': images, 'centers': centers, 'attempts': attempts, 'layout': layout, 'split': split}


def placement_summary(pool):
    out = {}
    for c in np.unique(pool['labels']):
        m = pool['labels'] == c
        a = pool['attempts'][m]
        out[str(int(c))] = {'images': int(m.sum()), 'proposals_total': int(a.sum()), 'proposals_max': int(a.max()),
                            'acceptance_rate': float(m.sum() / a.sum()), 'radius_mean': float(pool['radius'][m].mean())}
        if pool['layout'] == 'standard':
            q = pool['quarter'][m]
            raw = np.stack([quarter_turn(pool['centers'][i, :c], -qq) for i, qq in zip(np.flatnonzero(m), q)])
            jit = raw - face_centers(int(c))[None]
            out[str(int(c))]['accepted_jitter_sd'] = float(jit.std())
            out[str(int(c))]['accepted_jitter_abs_mean'] = float(np.abs(jit).mean())
    return out


# ---------------------------------------------------------------- complex networks

def to_blocks(x):
    """(N, 28, 28) -> (N, 196, 4): row-major 2x2 blocks, row-major within a block."""
    n = x.shape[0]
    return x.reshape(n, 14, 2, 14, 2).permute(0, 1, 3, 2, 4).reshape(n, 196, 4)


def from_blocks(b):
    n = b.shape[0]
    return b.reshape(n, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(n, 28, 28)


def complex_glorot(rng, fan_in, fan_out):
    """Rayleigh magnitude with sigma = 1/sqrt(fan_in + fan_out), uniform phase on [-pi, pi)."""
    sigma = 1 / np.sqrt(fan_in + fan_out)
    mag = sigma * np.sqrt(-2 * np.log1p(-rng.random((fan_out, fan_in))))
    phase = rng.uniform(-np.pi, np.pi, (fan_out, fan_in))
    return torch.from_numpy(mag * np.exp(1j * phase)).to(torch.complex64)


class CLinear(nn.Module):
    def __init__(self, fan_in, fan_out, rng):
        super().__init__()
        self.weight = nn.Parameter(complex_glorot(rng, fan_in, fan_out))
        self.bias = nn.Parameter(torch.zeros(fan_out, dtype=torch.complex64))

    def forward(self, z):
        return z @ self.weight.T + self.bias


class ModReLU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.bias = nn.Parameter(torch.full((n,), -.01))

    def forward(self, z):
        a = z.abs()
        return F.relu(a + self.bias) * z / (a + MODRELU_EPS)


class Encoder(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.block, self.act1 = CLinear(4, 4, rng), ModReLU(4)
        self.hidden, self.act2 = CLinear(784, 128, rng), ModReLU(128)
        self.out = CLinear(128, D, rng)

    def forward(self, x):
        h = self.act1(self.block(to_blocks(x).to(torch.complex64)))
        return self.out(self.act2(self.hidden(h.reshape(len(x), 784))))


class Decoder(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.inp, self.act1 = CLinear(D, 128, rng), ModReLU(128)
        self.hidden, self.act2 = CLinear(128, 784, rng), ModReLU(784)
        self.block = CLinear(4, 4, rng)

    def forward(self, z):
        h = self.act2(self.hidden(self.act1(self.inp(z))))
        return from_blocks(self.block(h.reshape(len(z), 196, 4)).real)  # BCE logits


class Modality(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.encoder, self.decoder = Encoder(rng), Decoder(rng)


class Rotation(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.alpha = nn.Parameter(torch.from_numpy(rng.uniform(-np.pi, np.pi, D)).float())

    def forward(self, z):
        return z * torch.polar(torch.ones_like(self.alpha), self.alpha)


class SuccessorMLP(nn.Module):
    def __init__(self, rng, hidden=32):
        super().__init__()
        self.inp, self.act, self.out = CLinear(D, hidden, rng), ModReLU(hidden), CLinear(hidden, D, rng)

    def forward(self, z):
        return self.out(self.act(self.inp(z)))


def make_successor(variant, rng):
    return Rotation(rng) if variant == 'rotation' else SuccessorMLP(rng)


def init_digit(seed):
    return Modality(stream('init', 'digit', seed))


def init_dice(seed):
    """Shared by every arm and both S variants within a seed."""
    return Modality(stream('init', 'dice', seed))


def init_successor(seed, variant):
    return make_successor(variant, stream('init', 'successor', variant, seed))


def state_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def param_bytes(module):
    """Real float32 scalars 4 bytes, complex64 8 bytes."""
    return int(sum(p.numel() * p.element_size() for p in module.parameters()))


# ---------------------------------------------------------------- similarity and losses

def flat(z):
    """Complex (N, d) -> real (N, 2d) as [Re, Im]."""
    return torch.cat([z.real, z.imag], -1)


def unit(z):
    f = flat(z)
    return f / f.norm(dim=-1, keepdim=True).clamp_min(NORM_FLOOR)


def cosine_matrix(a, b):
    return unit(a) @ unit(b).T


def positive_mask(target_labels):
    """Row b's positives: every target whose label equals row b's assigned target label."""
    return target_labels[None, :] == target_labels[:, None]


def successor_loss(q, k, target_labels):
    """Successor-conditioned multi-positive supervised contrastive loss (temperature 0.1).

    q[b] = S(E(x_b)); k[c] = E(x_c+); target_labels[c] is the label of the target image
    assigned to row c, so a shuffled control permutes k and target_labels together.
    """
    logits = cosine_matrix(q, k) / TEMPERATURE
    logp = logits - torch.logsumexp(logits, 1, keepdim=True)
    mask = positive_mask(target_labels).float()
    return -((mask * logp).sum(1) / mask.sum(1)).mean()


def anchor_loss(z, c1):
    return (z - c1).abs().pow(2).sum() / max(float(c1.abs().pow(2).sum()), NORM_FLOOR)


# ---------------------------------------------------------------- sampling

def balanced_counts(n_classes, size, rng):
    """As equal as possible; a shuffle decides which classes take the remainder."""
    counts = np.full(n_classes, size // n_classes)
    counts[rng.permutation(n_classes)[:size % n_classes]] += 1
    return counts


class Sampler:
    """With-replacement class-balanced batches from a fixed pool, one stream per role."""

    def __init__(self, labels, classes, successor_sources, seed_keys):
        self.by_class = {c: np.flatnonzero(labels == c) for c in classes}
        self.classes, self.sources = tuple(classes), tuple(successor_sources)
        self.rec_rng = stream('sample', *seed_keys, 'reconstruction')
        self.pair_rng = stream('sample', *seed_keys, 'pairs')
        self.shuffle_rng = stream('sample', *seed_keys, 'shuffle')
        self.seen = {'reconstruction': set(), 'source': set(), 'target': set()}

    def _draw(self, rng, classes, counts):
        return np.concatenate([rng.choice(self.by_class[c], n) for c, n in zip(classes, counts)])

    def reconstruction(self, size=B):
        idx = self._draw(self.rec_rng, self.classes, balanced_counts(len(self.classes), size, self.rec_rng))
        self.seen['reconstruction'].update(idx.tolist())
        return idx

    def pairs(self, size=B, shuffled=False):
        counts = balanced_counts(len(self.sources), size, self.pair_rng)
        src = self._draw(self.pair_rng, self.sources, counts)
        tgt = self._draw(self.pair_rng, [c + 1 for c in self.sources], counts)
        if shuffled:
            tgt = tgt[self.shuffle_rng.permutation(size)]
        self.seen['source'].update(src.tolist())
        self.seen['target'].update(tgt.tolist())
        return src, tgt


# ---------------------------------------------------------------- training

def steps_for(pool_size, epochs):
    return epochs * -(-pool_size // B)


def train(modality, successor, images, labels, sampler, steps, *, train_successor, pairs, lam_succ,
          anchor=None, c1=None, lam_anchor=0., log_every=10):
    """One stage. Stage 1: modality and S train jointly. Stage 2: S is frozen, gradients pass
    through S's input into the dice encoder. Raises NumericalFailure, carrying the partial log.
    """
    params = list(modality.parameters()) + (list(successor.parameters()) if train_successor else [])
    for p in successor.parameters():
        p.requires_grad_(train_successor)
    opt = torch.optim.Adam(params, lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=0)
    labels_t = torch.from_numpy(np.asarray(labels))
    a = None if anchor is None else anchor[None]
    hist = {k: [] for k in ('rec', 'succ', 'anchor', 'total', 'grad_norm')}
    counts = Counter()
    clips = 0
    for step in range(steps):
        rec = sampler.reconstruction()
        parts = [images[rec]]
        if pairs:
            src, tgt = sampler.pairs(shuffled=pairs == 'shuffled')
            parts += [images[src], images[tgt]]
        if a is not None:
            parts.append(a)
        z = modality.encoder(torch.cat(parts))
        l_rec = F.binary_cross_entropy_with_logits(modality.decoder(z[:B]), parts[0])
        loss, l_succ, l_anc = l_rec, torch.tensor(0.), torch.tensor(0.)
        if pairs:
            l_succ = successor_loss(successor(z[B:2 * B]), z[2 * B:3 * B], labels_t[tgt])
            loss = loss + lam_succ * l_succ
            counts['successor_applications'] += B
        if a is not None:
            l_anc = anchor_loss(z[-1], c1)
            loss = loss + lam_anchor * l_anc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        if not (torch.isfinite(loss) and torch.isfinite(norm) and torch.isfinite(z).all()):
            e = NumericalFailure(f'nonfinite value at step {step}')
            e.log = {'failed_step': step, 'history': hist, 'clips': clips}
            raise e
        if norm > CLIP:
            clips += 1
            for g in grads:
                g.mul_(CLIP / (norm + 1e-6))
        opt.step()
        counts['optimizer_steps'] += 1
        counts['reconstruction_exposures'] += B
        counts['source_exposures'] += B if pairs else 0
        counts['target_exposures'] += B if pairs else 0
        counts['anchor_exposures'] += 1 if a is not None else 0
        counts['encoded_images'] += len(z)
        counts['decoded_images'] += B
        if step % log_every == 0 or step == steps - 1:
            for k, v in (('rec', l_rec), ('succ', l_succ), ('anchor', l_anc), ('total', loss), ('grad_norm', norm)):
                hist[k].append(float(v.detach()))
    counts['clips'] = clips
    for role, s in sampler.seen.items():
        counts[f'distinct_{role}_examples'] = len(s)
    for p in successor.parameters():
        p.requires_grad_(False)
    return {'steps': steps, 'log_every': log_every, 'history': hist, 'clips': clips,
            'clip_fraction': clips / max(steps, 1)}, dict(counts)


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def encode(encoder, images, batch=2048):
    return torch.cat([encoder(images[i:i + batch]) for i in range(0, len(images), batch)])


def centroids(z, labels, classes):
    labels = torch.as_tensor(np.asarray(labels))
    return torch.stack([z[labels == c].mean(0) for c in classes])


def predict(z, cents, classes=DIGITS):
    """Largest cosine to the centroids; torch.argmax returns the first maximum, i.e. the smallest class."""
    return np.asarray(classes)[cosine_matrix(z, cents).argmax(1).numpy()]


def per_class_accuracy(pred, labels, classes):
    return {str(c): float((pred[labels == c] == c).mean()) for c in classes if (labels == c).any()}


def confusion(pred, labels, rows, cols=DIGITS):
    return [[int(((labels == r) & (pred == c)).sum()) for c in cols] for r in rows]


def fit_probe(f_train, y_train, f_eval, y_eval, classes=10, steps=PROBE_STEPS, record_every=10):
    """Linear softmax probe: train-only standardization, zero init, full-batch Adam, final weights."""
    mu, sd = f_train.mean(0), f_train.std(0).clamp_min(SD_FLOOR)
    xt, xe = (f_train - mu) / sd, (f_eval - mu) / sd
    yt = torch.as_tensor(np.asarray(y_train))
    w = torch.zeros(xt.shape[1], classes, requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=PROBE_LR)
    curve = []
    for step in range(steps):
        loss = F.cross_entropy(xt @ w + b, yt)
        if step % record_every == 0:
            curve.append(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(F.cross_entropy(xt @ w + b, yt))
        pt, pe = (xt @ w + b).argmax(1).numpy(), (xe @ w + b).argmax(1).numpy()
    ye = np.asarray(y_eval)
    return {'train_accuracy': float((pt == np.asarray(y_train)).mean()), 'accuracy': float((pe == ye).mean()),
            'per_class': per_class_accuracy(pe, ye, range(classes)),
            'loss_curve': curve + [final], 'record_every': record_every, 'steps': steps}


@torch.no_grad()
def compose(successor, z, k):
    for _ in range(k):
        z = successor(z)
    return z


def composition_sources(modality, k):
    """Primary within-support source counts for k applications."""
    return tuple(range(0, 10 - k)) if modality == 'digit' else tuple(range(1, 7 - k))


@torch.no_grad()
def composition(successor, z, labels, cents, sources, k):
    pred = predict(compose(successor, z, k), cents)
    per, n = {}, {}
    for s in sources:
        m = labels == s
        per[str(s)] = float((pred[m] == s + k).mean())
        n[str(s)] = int(m.sum())
    return {'macro_accuracy': float(np.mean(list(per.values()))), 'per_source': per, 'denominators': n}


def rel_residual(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / max(float(b.abs().pow(2).sum().sqrt()), NORM_FLOOR))


@torch.no_grad()
def latent_stats(z, labels, classes):
    labels = np.asarray(labels)
    norms = z.abs().pow(2).sum(1).sqrt()
    cents = centroids(z, labels, classes)
    out = {'near_zero_norms': int((norms < NORM_FLOOR).sum()), 'nonfinite': int((~torch.isfinite(z)).sum()),
           'norm_mean': {}, 'rms_spread': {}}
    for c, cen in zip(classes, cents):
        m = torch.as_tensor(labels == c)
        out['norm_mean'][str(c)] = float(norms[m].mean())
        out['rms_spread'][str(c)] = float((z[m] - cen).abs().pow(2).sum(1).mean().sqrt())
    return out


@torch.no_grad()
def margins(z, labels, cents, rows):
    """Cosine to the correct digit centroid minus the best other centroid, mean per true class."""
    cm = cosine_matrix(z, cents).numpy()
    labels = np.asarray(labels)
    out = {}
    for r in rows:
        m = labels == r
        if not m.any() or r > 9:
            continue
        other = np.delete(cm[m], r, axis=1).max(1)
        out[str(r)] = float((cm[m, r] - other).mean())
    return out


@torch.no_grad()
def recurrence(successor, cents, classes):
    """||S(c_n) - c_(n+1)|| / ||c_(n+1)|| along a modality's own centroids."""
    s = successor(cents)
    return {f'{a}->{b}': rel_residual(s[i], cents[i + 1]) for i, (a, b) in enumerate(zip(classes[:-1], classes[1:]))}


@torch.no_grad()
def per_example_successor(successor, z, labels, cents, classes):
    labels = np.asarray(labels)
    s = successor(z)
    out = {}
    for i, c in enumerate(classes[:-1]):
        m = torch.as_tensor(labels == c)
        tgt = cents[i + 1]
        out[f'{c}->{classes[i + 1]}'] = float(((s[m] - tgt).abs().pow(2).sum(1).sqrt()).mean()
                                             / max(float(tgt.abs().pow(2).sum().sqrt()), NORM_FLOOR))
    return out


@torch.no_grad()
def operator_discrepancy(donor, recipient, z):
    r = recipient(z)
    return float((donor(z) - r).abs().pow(2).sum(1).mean() / max(float(r.abs().pow(2).sum(1).mean()), NORM_FLOOR))


def offsets(pred, labels, rows):
    m = np.isin(labels, rows)
    d = pred[m] - labels[m]
    vals, n = np.unique(d, return_counts=True)
    hist = {str(int(v)): int(c) for v, c in zip(vals, n)}
    best = int(np.argmax(n))
    return {'histogram': hist, 'largest_offset': int(vals[best]), 'largest_offset_fraction': float(n[best] / m.sum())}


@torch.no_grad()
def evaluate_digits(modality, successor, train_images, train_labels, eval_images, eval_labels):
    """Feasibility gates, composition and mechanism diagnostics for one stage-1 checkpoint."""
    zt, ze = encode(modality.encoder, train_images), encode(modality.encoder, eval_images)
    yt, ye = np.asarray(train_labels), np.asarray(eval_labels)
    cents = centroids(zt, yt, DIGITS)
    pred = predict(ze, cents)
    per = per_class_accuracy(pred, ye, DIGITS)
    with torch.enable_grad():
        probe = fit_probe(flat(zt), yt, flat(ze), ye)
        probe_abs = fit_probe(zt.abs(), yt, ze.abs(), ye)
    out = {'centroid_accuracy': float((pred == ye).mean()), 'centroid_per_digit': per,
           'centroid_macro_accuracy': float(np.mean(list(per.values()))),
           'probe_full': probe, 'probe_magnitude': probe_abs,
           'composition': {str(k): composition(successor, ze, ye, cents, composition_sources('digit', k), k)
                           for k in (1, 2, 3)},
           'recurrence_residuals': recurrence(successor, cents, DIGITS),
           'per_example_successor_error': per_example_successor(successor, zt, yt, cents, DIGITS),
           'latents_train': latent_stats(zt, yt, DIGITS), 'latents_eval': latent_stats(ze, ye, DIGITS),
           'centroid_cosines': cosine_matrix(cents, cents).tolist(), 'margins': margins(ze, ye, cents, DIGITS),
           'eval_size': int(len(ye))}
    out['gates'] = {'centroid': out['centroid_accuracy'] >= .95, 'probe': probe['accuracy'] >= .95}
    return out, cents


@torch.no_grad()
def evaluate_dice(modality, successor, digit_cents, dice_train, dice_train_labels, anchor, pools):
    """Cross-modal readout against all ten digit centroids. pools: name -> (images, labels, standard?)."""
    zt = encode(modality.encoder, dice_train)
    yt = np.asarray(dice_train_labels)
    dice_cents = centroids(zt, yt, FACES)
    za = modality.encoder(anchor[None])[0]
    out = {'anchor_residual': rel_residual(za, digit_cents[1]),
           'recurrence_residuals_dice': recurrence(successor, dice_cents, FACES),
           'per_example_successor_error_dice': per_example_successor(successor, zt, yt, dice_cents, FACES),
           'dice_to_digit_centroid_residuals': {str(n): rel_residual(dice_cents[i], digit_cents[n])
                                                for i, n in enumerate(FACES)},
           'latents_train': latent_stats(zt, yt, FACES), 'pools': {}}
    for name, (images, labels, standard) in pools.items():
        z = encode(modality.encoder, images)
        y = np.asarray(labels)
        pred = predict(z, digit_cents)
        rows = FACES if standard else RANDOM_COUNTS
        per = per_class_accuracy(pred, y, rows)
        r = {'per_count': per, 'confusion': confusion(pred, y, rows), 'confusion_rows': list(rows),
             'latents': latent_stats(z, y, rows), 'margins': margins(z, y, digit_cents, rows), 'size': int(len(y))}
        if standard:
            r['primary_macro_2_6'] = float(np.mean([per[str(c)] for c in range(2, 7)]))
            r['count_1_accuracy'] = per['1']
            r['offsets_2_6'] = offsets(pred, y, range(2, 7))
            r['offsets_1_6'] = offsets(pred, y, FACES)
            r['count_1_centroid_residual'] = rel_residual(z[torch.as_tensor(y == 1)].mean(0), digit_cents[1])
            r['composition'] = {}
            for k in (1, 2, 3):
                r['composition'][str(k)] = composition(successor, z, y, digit_cents, composition_sources('dice', k), k)
                ext = tuple(s for s in FACES if 7 <= s + k <= 9)
                r['composition'][f'{k}_extrapolation'] = composition(successor, z, y, digit_cents, ext, k)
        else:
            r['macro_1_6'] = float(np.mean([per[str(c)] for c in range(1, 7)]))
            r['macro_7_9'] = float(np.mean([per[str(c)] for c in range(7, 10)]))
        out['pools'][name] = r
    return out


# ---------------------------------------------------------------- images

def write_png(path, array):
    """Grayscale 8-bit PNG from a [0, 1] array, without an imaging dependency."""
    a = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    raw = b''.join(b'\x00' + row.tobytes() for row in a)

    def chunk(kind, data):
        return (len(data).to_bytes(4, 'big') + kind + data
                + (zlib.crc32(kind + data) & 0xffffffff).to_bytes(4, 'big'))

    head = a.shape[1].to_bytes(4, 'big') + a.shape[0].to_bytes(4, 'big') + bytes([8, 0, 0, 0, 0])
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', head) + chunk(b'IDAT', zlib.compress(raw, 9))
                           + chunk(b'IEND', b''))


def tile(rows, pad=2):
    """rows: list of lists of 28x28 arrays -> one padded mosaic."""
    h, w = len(rows), max(len(r) for r in rows)
    out = np.full((h * (28 + pad) + pad, w * (28 + pad) + pad), .5, np.float32)
    for i, r in enumerate(rows):
        for j, img in enumerate(r):
            y, x = pad + i * (28 + pad), pad + j * (28 + pad)
            out[y:y + 28, x:x + 28] = img
    return out


# ---------------------------------------------------------------- acceptance checks

def acceptance_checks():
    """Implementation checks from the specification's checklist that need no real data.

    They verify conventions and algebra, not that learned image codes obey the model.
    """
    rng = stream('acceptance')
    torch.manual_seed(0)
    out = {}
    x = torch.from_numpy(rng.random((3, 28, 28)).astype(np.float32))
    out['block_roundtrip'] = bool(torch.equal(from_blocks(to_blocks(x)), x))
    out['block_order_row_major'] = bool(torch.equal(to_blocks(x)[0, 1], x[0, 0:2, 2:4].reshape(-1)))
    m = Modality(stream('acceptance', 'modality'))
    z = m.encoder(x)
    out['latent_shape_dtype'] = bool(z.shape == (3, D) and z.dtype == torch.complex64)
    out['decoder_shape_real'] = bool(m.decoder(z).shape == (3, 28, 28) and not m.decoder(z).is_complex())
    finite = {}
    for variant in VARIANTS:
        mm, s = Modality(stream('acceptance', variant)), make_successor(variant, stream('acceptance', 'S', variant))
        zz = mm.encoder(x)
        loss = (F.binary_cross_entropy_with_logits(mm.decoder(zz), x)
                + successor_loss(s(zz), mm.encoder(x.flip(0)), torch.tensor([1, 2, 1])))
        loss.backward()
        finite[variant] = bool(all(torch.isfinite(p.grad).all() for p in [*mm.parameters(), *s.parameters()]
                                   if p.grad is not None))
    out['finite_complex_gradients'] = finite

    # Complex Adam equals Adam on the real/imaginary view.
    w0 = torch.from_numpy(rng.normal(size=5) + 1j * rng.normal(size=5)).to(torch.complex64)
    target = torch.from_numpy(rng.normal(size=5) + 1j * rng.normal(size=5)).to(torch.complex64)
    pc, pr = nn.Parameter(w0.clone()), nn.Parameter(torch.view_as_real(w0.clone()).clone())
    oc, orr = torch.optim.Adam([pc], lr=LR, betas=BETAS, eps=ADAM_EPS), torch.optim.Adam([pr], lr=LR, betas=BETAS, eps=ADAM_EPS)
    for _ in range(5):
        for p, o, f in ((pc, oc, lambda p: (p - target).abs().pow(2).sum()),
                        (pr, orr, lambda p: (torch.view_as_complex(p) - target).abs().pow(2).sum())):
            o.zero_grad()
            f(p).backward()
            o.step()
    out['complex_adam_equals_real_view'] = bool(torch.allclose(torch.view_as_real(pc.detach()), pr.detach(), atol=1e-6))

    # Stage 2: S frozen, dice encoder still receives gradients through S's input.
    s = Rotation(stream('acceptance', 'freeze'))
    s_before = s.alpha.detach().clone()
    dice = Modality(stream('acceptance', 'dice'))
    labels = np.repeat(np.arange(1, 7), 2)
    imgs = torch.from_numpy(rng.random((12, 28, 28)).astype(np.float32))
    samp = Sampler(labels, FACES, range(1, 6), ('acceptance',))
    w_before = dice.encoder.out.weight.detach().clone()
    train(dice, s, imgs, labels, samp, 1, train_successor=False, pairs='correct', lam_succ=1.)
    out['stage2_successor_frozen'] = bool(torch.equal(s.alpha, s_before) and not s.alpha.requires_grad)
    zz = dice.encoder(imgs[:4])
    successor_loss(s(zz[:2]), zz[2:], torch.tensor([2, 3])).backward()
    out['stage2_gradient_through_successor'] = bool(dice.encoder.out.weight.grad.abs().sum() > 0
                                                    and s.alpha.grad is None)
    out['stage2_encoder_updated'] = bool(not torch.equal(dice.encoder.out.weight, w_before))

    # Positive masks follow assigned target labels, including after a shuffle.
    t = torch.tensor([2, 3, 2, 4])
    out['positive_mask'] = bool(torch.equal(positive_mask(t), torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)))
    perm = torch.tensor([3, 0, 1, 2])
    out['shuffled_mask_rebuilt'] = bool(torch.equal(positive_mask(t[perm]), t[perm][None] == t[perm][:, None]))
    sh = Sampler(labels, FACES, range(1, 6), ('acceptance', 'sh'))
    src, tgt = sh.pairs(size=10, shuffled=True)
    out['shuffle_preserves_target_marginal'] = bool(sorted(labels[tgt]) == sorted(labels[src] + 1))
    out['balanced_counts'] = bool(sorted(balanced_counts(9, 256, rng)) == [28] * 5 + [29] * 4)

    # Readout: ten candidates and smallest-class ties.
    cents = torch.zeros(10, D, dtype=torch.complex64)
    cents[3, 0] = cents[5, 0] = 1
    out['tie_smallest_digit'] = bool(predict(cents[3:4], cents)[0] == 3)
    out['ten_candidates'] = bool(len(DIGITS) == 10)
    out['composition_boundaries'] = bool(composition_sources('digit', 2) == tuple(range(8))
                                         and composition_sources('dice', 2) == (1, 2, 3, 4))

    # Exact rotation orbit algebra.
    alpha = torch.from_numpy(rng.uniform(-np.pi, np.pi, D)).float()
    rot = lambda v, k=1: v * torch.polar(torch.ones(D), alpha * k)
    c0 = torch.from_numpy(rng.normal(size=D) + 1j * rng.normal(size=D)).to(torch.complex64)
    orbit = torch.stack([rot(c0, n) for n in range(10)])
    out['orbit_constant_magnitudes'] = bool(torch.allclose(orbit.abs(), orbit.abs()[0:1].expand(10, D), atol=1e-5))
    q = torch.polar(torch.ones(D), torch.from_numpy(rng.uniform(-np.pi, np.pi, D)).float())
    dice_codes = orbit[1:7]
    scores = cosine_matrix(rot(dice_codes[:-1]), dice_codes[1:])
    scores_q = cosine_matrix(rot(dice_codes[:-1] * q), dice_codes[1:] * q)
    out['phase_change_preserves_relation_scores'] = bool(torch.allclose(scores, scores_q, atol=1e-5))
    out['phase_change_changes_unanchored_predictions'] = bool(
        (predict(dice_codes * q, orbit) != np.arange(1, 7)).any())
    anchored = torch.stack([rot(orbit[1], n - 1) for n in FACES])
    out['exact_anchor_propagation'] = bool((predict(anchored, orbit) == np.arange(1, 7)).all())
    return out
