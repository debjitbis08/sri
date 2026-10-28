"""Morse data, SRI energy and exact equilibria for number core network v1.

Specification: docs/number-core-network-v1-spec.md (draft). Image (MNIST + dice) and
Morse front ends feed 16 -> 8 -> 4 linear modules; image and Morse modules couple to a
core at corresponding layers through learned translations, and the core couples to an
output module the same way. Each example clamps one input module's outer layer and
solves the quadratic energy over that module's hidden layers, the core and the output.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src import number_concept as nc

WIDTHS = {'narrow': (16, 8, 4), 'wide': (64, 32, 16)}  # outer, middle, inner for every module
SIZES = WIDTHS['narrow']  # default for tests; the runner passes sizes explicitly
LAMBDA = 1e-3
# Per-unit coupling strengths. At narrow width they spend budget 4 per edge
# (uniform 28 units x 1/7, inner 4 units x 1); the per-unit values are kept at every width.
TOPOLOGIES = {'uniform': (1 / 7,) * 3, 'inner': (0., 0., 1.)}
MORSE = {0: '-----', 1: '.----', 2: '..---', 3: '...--', 4: '....-', 5: '.....',
         6: '-....', 7: '--...', 8: '---..', 9: '----.'}
MORSE_LEN, MORSE_UNIT, MORSE_JITTER, MORSE_OFFSET, MORSE_AMP, MORSE_NOISE = 128, (3., 5.), .2, 4, (.7, 1.), .1
MORSE_POOLS = {'train': 2000, 'validation': 300, 'test': 500}
LABELLED = (1, 3, 5)  # labelled dice and Morse numbers; MNIST is labelled at every digit


# ---------------------------------------------------------------- Morse

def morse_signal(code, rng):
    """One jittered, noisy on/off signal for a dot-dash string; returns (signal, end sample)."""
    u = rng.uniform(*MORSE_UNIT)
    t = int(rng.integers(0, MORSE_OFFSET + 1))
    x = np.zeros(MORSE_LEN, np.float32)
    amp = rng.uniform(*MORSE_AMP)
    for i, symbol in enumerate(code):
        if i:
            t += max(1, int(round(u * (1 + rng.uniform(-MORSE_JITTER, MORSE_JITTER)))))
        d = max(1, int(round(u * (1 if symbol == '.' else 3) * (1 + rng.uniform(-MORSE_JITTER, MORSE_JITTER)))))
        if t + d > MORSE_LEN:
            raise AssertionError(f'Morse code {code} overflows the window')
        x[t:t + d] = amp
        t += d
    return x + rng.normal(0, MORSE_NOISE, MORSE_LEN).astype(np.float32), t


def build_morse(split, per_code=None):
    """Signals for codes 0-9 (labels are code digits); one named stream per (split, code, index)."""
    per_code = per_code or MORSE_POOLS[split]
    sig, codes, ends = [], [], []
    for c in range(10):
        for i in range(per_code):
            s, end = morse_signal(MORSE[c], nc.stream('morse', split, c, i))
            sig.append(s); codes.append(c); ends.append(end)
    return {'signals': np.stack(sig), 'codes': np.array(codes), 'ends': np.array(ends)}


def scramble():
    """Fixed derangement: number n is shown by code perm[n]; the dot count is not monotone in n."""
    rng = nc.stream('morse_scramble')
    dots = np.array([MORSE[c].count('.') for c in range(10)])
    while True:
        perm = rng.permutation(10)
        d = dots[perm[1:6]]
        if np.all(perm != np.arange(10)) and not (np.all(np.diff(d) > 0) or np.all(np.diff(d) < 0)):
            return perm


def morse_numbers(codes, perm):
    """Number labels of Morse signals under a code assignment (identity = standard Morse)."""
    inverse = np.argsort(perm)
    return inverse[codes]


# ---------------------------------------------------------------- tally marks

TALLY_COUNTS = tuple(range(1, 11))
TALLY_POOLS = {'train': 2000, 'validation': 300, 'test': 500}
TALLY = dict(length=(10., 14.), half_width=(.5, .8), spacing=(2.2, 2.8), group_gap=(4., 5.), slant=(-6., 6.),
             stroke_jitter=(.3, 1., 5.), brightness=(.7, 1.), noise=.02, margin=1.)


def tally_segments(count, rng):
    """Line segments (x0, y0, x1, y1) for a tally of `count`: gates of four strokes plus a diagonal, then loose strokes."""
    p = TALLY
    length, spacing, gap = rng.uniform(*p['length']), rng.uniform(*p['spacing']), rng.uniform(*p['group_gap'])
    slant = np.radians(rng.uniform(*p['slant']))
    dx, dy, dang = p['stroke_jitter']
    gates, loose = divmod(count, 5)
    segs, x = [], 0.
    groups = [4] * gates + ([loose] if loose else [])
    for gi, n in enumerate(groups):
        xs = []
        for _ in range(n):
            cx, a = x + rng.uniform(-dx, dx), slant + np.radians(rng.uniform(-dang, dang))
            top, bot = -length / 2 + rng.uniform(-dy, dy), length / 2 + rng.uniform(-dy, dy)
            segs.append((cx + top * np.sin(a), top, cx + bot * np.sin(a), bot))
            xs.append(cx)
            x += spacing
        if gi < gates:  # strike-through of the gate, bottom-left to top-right, overhanging the outer strokes
            over = rng.uniform(.6, 1.4)
            segs.append((xs[0] - over, length * rng.uniform(.25, .4), xs[-1] + over, -length * rng.uniform(.25, .4)))
        x += gap - spacing
    segs = np.array(segs)
    return segs


def render_tally(count, rng):
    """28 x 28 tally image: anti-aliased segments (4 x 4 subpixels), centred with a random shift, brightness and noise."""
    p = TALLY
    segs = tally_segments(count, rng)
    hw = rng.uniform(*p['half_width'])
    xs, ys = segs[:, [0, 2]], segs[:, [1, 3]]
    lo, hi = p['margin'] + hw, 28 - p['margin'] - hw
    room_x = (hi - lo) - (xs.max() - xs.min())
    room_y = (hi - lo) - (ys.max() - ys.min())
    if room_x < 0 or room_y < 0:
        raise AssertionError(f'tally {count} does not fit the canvas')
    ox = lo - xs.min() + rng.uniform(0, room_x)
    oy = lo - ys.min() + rng.uniform(0, room_y)
    segs = segs + np.array([ox, oy, ox, oy])
    g = (np.arange(28)[:, None] + nc.SUBPIXEL[None]).reshape(-1) + .5
    py, px = np.meshgrid(g, g, indexing='ij')
    d2 = np.full(px.shape, np.inf)
    for x0, y0, x1, y1 in segs:
        vx, vy = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * vx + (py - y0) * vy) / (vx * vx + vy * vy), 0, 1)
        d2 = np.minimum(d2, (px - x0 - t * vx) ** 2 + (py - y0 - t * vy) ** 2)
    cover = (d2 <= hw ** 2).reshape(28, 4, 28, 4).mean((1, 3))
    x = rng.uniform(*p['brightness']) * cover + rng.normal(0, p['noise'], (28, 28))
    return np.clip(x, 0, 1).astype(np.float32)


def build_tally(split, per_count=None):
    """Tally images for counts 1-10; one named stream per (split, count, index)."""
    per_count = per_count or TALLY_POOLS[split]
    imgs, labels = [], []
    for c in TALLY_COUNTS:
        for i in range(per_count):
            imgs.append(render_tally(c, nc.stream('tally', split, c, i)))
            labels.append(c)
    return {'images': np.stack(imgs), 'labels': np.array(labels)}


# ---------------------------------------------------------------- front ends

class ImageEncoder(nn.Module):
    def __init__(self, out=SIZES[0]):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 16, 3, 2, 1), nn.Conv2d(16, 32, 3, 2, 1)
        self.fc, self.out = nn.Linear(32 * 49, 128), nn.Linear(128, out)

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x[:, None]))))
        return self.out(F.relu(self.fc(h.flatten(1))))


class MorseEncoder(nn.Module):
    def __init__(self, out=SIZES[0]):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(1, 16, 5, 2, 2), nn.Conv1d(16, 32, 5, 2, 2)
        self.fc, self.out = nn.Linear(32 * 32, 128), nn.Linear(128, out)

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x[:, None]))))
        return self.out(F.relu(self.fc(h.flatten(1))))


# ---------------------------------------------------------------- SRI part

MODULES = ('image', 'morse', 'core', 'output')
EDGES = (('image', 'core'), ('morse', 'core'), ('core', 'output'))


class SRI(nn.Module):
    """Linear modules, layer-wise translations, core successor, judge and label scalars."""
    def __init__(self, topology, generator, sizes=SIZES):
        super().__init__()
        self.sizes, self.state = tuple(sizes), sum(sizes)
        o, m, i = self.sizes
        self.W = nn.ParameterDict({k: nn.Parameter(torch.randn(o, m, generator=generator) * .25) for k in MODULES})
        self.V = nn.ParameterDict({k: nn.Parameter(torch.randn(m, i, generator=generator) * .25) for k in MODULES})
        self.kappa = TOPOLOGIES[topology]
        self.T = nn.ParameterDict({f'{a}-{b}-{l}': nn.Parameter(torch.eye(n)) for a, b in EDGES
                                   for l, n in enumerate(self.sizes) if self.kappa[l] > 0})
        self.A, self.t = nn.Parameter(torch.eye(i)), nn.Parameter(torch.zeros(i))
        self.judge_bias, self.judge_scale = nn.Parameter(torch.ones(2)), nn.Parameter(torch.zeros(2))
        self.label_scale = nn.Parameter(torch.tensor(5.))  # beta = softplus(5) ~ 5 at initialization

    def S(self, c):
        return c @ self.A.T + self.t

    def hessian(self, kind):
        """Energy Hessian over [kind module | core | output] states (3 x 28)."""
        order = (kind, 'core', 'output')
        STATE = self.state
        Q = torch.zeros(3 * STATE, 3 * STATE)
        o, m, i = self.sizes
        for j, k in enumerate(order):
            b = j * STATE
            x, h, c = slice(b, b + o), slice(b + o, b + o + m), slice(b + o + m, b + STATE)
            W, V = self.W[k], self.V[k]
            Q[x, x] += torch.eye(o) * (1 + (LAMBDA if k == 'core' else 0))
            Q[x, h] -= W; Q[h, x] -= W.T
            Q[h, h] += W.T @ W + (1 + LAMBDA) * torch.eye(m)
            Q[h, c] -= V; Q[c, h] -= V.T
            Q[c, c] += V.T @ V + LAMBDA * torch.eye(i)
        for a, bm in ((kind, 'core'), ('core', 'output')):
            ja, jb = order.index(a), order.index(bm)
            off = 0
            for l, n in enumerate(self.sizes):
                if self.kappa[l] > 0:
                    T, k = self.T[f'{a}-{bm}-{l}'], self.kappa[l]
                    sa, sb = slice(ja * STATE + off, ja * STATE + off + n), slice(jb * STATE + off, jb * STATE + off + n)
                    Q[sb, sb] += k * torch.eye(n)
                    Q[sa, sa] += k * T.T @ T
                    Q[sb, sa] -= k * T
                    Q[sa, sb] -= k * T.T
                off += n
        return Q

    def equilibrium_map(self, kind):
        """M: free state = M @ clamped outer, from one Cholesky factorization."""
        Q = self.hessian(kind)
        clamp = torch.arange(self.sizes[0])
        free = torch.arange(self.sizes[0], 3 * self.state)
        L = torch.linalg.cholesky(Q[free][:, free])
        return -torch.cholesky_solve(Q[free][:, clamp], L), Q

    def split(self, y):
        """Free state -> dict of named parts."""
        o, m, i = self.sizes  # free layout: input hidden (m + i), core, output
        STATE = self.state
        p = {'input_inner': y[:, m:m + i]}
        core = y[:, m + i:m + i + STATE]
        out = y[:, m + i + STATE:]
        p['core'], p['core_inner'] = core, core[:, o + m:]
        p['output_outer'] = out[:, :o]
        return p


def label_codes(dim=SIZES[0]):
    """Ten fixed orthonormal label codes (rows) in the output outer layer."""
    g = torch.from_numpy(nc.stream('label_codes', dim).normal(size=(dim, 10))).float()
    q, _ = torch.linalg.qr(g)
    return q.T.contiguous()
