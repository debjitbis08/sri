"""Generator, phase-bank estimators and evaluator metrics for locality adaptation v1.

Contract: docs/locality-adaptation-v1-contract.md. Fitters receive only observed
pairs, dimensions and a declared phase; world truth is used only by the evaluator.
"""
from contextlib import contextmanager
from collections import Counter
import time

import numpy as np
from scipy.linalg import solve_triangular


D, B = 32, 8
RANK = D//B
PHASES = B
BUDGETS = (4, 8, 16, 32, 64, 128, 256)
PRIMARY_BUDGETS = (16, 32, 64, 128, 256)
REGIMES = ('aligned', 'half_shifted', 'unknown_shift', 'dense')
CONDITIONS = ('in_distribution', 'texture_16', 'extra_noise')
ARMS = ('adaptive_cca', 'fixed_cca_all', 'global_cca_all', 'fixed_cca_fit', 'global_cca_fit',
        'adaptive_spectral', 'fixed_spectral_all', 'global_spectral_all')
TEXTURE, ISO, EXTRA, RIDGE = 4., .25, 2., .1
ROOT_SEED = 2026102407
EIG_TIE, AXIS_FLOOR, SCORE_TIE, RANK_FLOOR = 1e-12, 1e-10, 1e-12, 1e-10


class ScoringFailure(RuntimeError):
    """Nonfinite value, nonpositive regularized covariance or rank failure: stops scoring."""


class Meter:
    """Per-arm stage timers and work counters (runtime JSON only, never scientific JSON)."""
    def __init__(self):
        self.seconds, self.counts = Counter(), Counter()

    @contextmanager
    def time(self, stage):
        start = time.perf_counter()
        yield
        self.seconds[stage] += time.perf_counter()-start


# ---------------------------------------------------------------- generator

def dct_basis(b=B):
    j = np.arange(b)+.5
    out = np.stack([np.cos(np.pi*m*j/b) for m in range(b)], 1)*np.sqrt(2/b)
    out[:, 0] = 1/np.sqrt(b)
    return out


def rng_for(stage, world, part, extra=()):
    return np.random.default_rng(np.random.SeedSequence([ROOT_SEED, stage, world, part, *extra]))


def shift_matrix(d, s):
    """(R h)_i = h_{i-s}: a block occupying sites 0..b-1 moves to s..s+b-1."""
    return np.roll(np.eye(d), s, axis=0)


def make_world(stage, world, d=D, b=B, extra=()):
    """Benchmark v1 template recipe; part 0 templates, 1 dense QR, 2 unknown shift."""
    k, basis = d//b, dct_basis(b)
    rng = rng_for(stage, world, 0, extra)
    u, v = np.zeros((d, k)), np.zeros((d, k))
    for j in range(k):
        sites = slice(j*b, (j+1)*b)
        t = basis[:, :3]@rng.standard_normal(3)
        n = basis[:, 3:]@rng.standard_normal(5)
        u[sites, j], v[sites, j] = t/np.linalg.norm(t), n/np.linalg.norm(n)
    q, r = np.linalg.qr(rng_for(stage, world, 1, extra).standard_normal((d, d)))
    q = q*np.where(np.diag(r) < 0, -1., 1.)[None, :]
    shift = int(rng_for(stage, world, 2, extra).integers(1, b))
    rotations = dict(aligned=np.eye(d), half_shifted=shift_matrix(d, b//2),
                     unknown_shift=shift_matrix(d, shift), dense=q)
    return dict(u=u, v=v, shift=shift, rotations=rotations, d=d, b=b)


def base_stream(stage, world, d=D, b=B, n=max(BUDGETS), extra=()):
    """Part 3: c, g1, g2, e1, e2, drawn in that order and shared by all regimes."""
    k, rng = d//b, rng_for(stage, world, 3, extra)
    return tuple(rng.standard_normal((n, size)) for size in (k, k, k, d, d))


def views(world, regime, stream):
    c, g1, g2, e1, e2 = stream
    u, v, r = world['u'], world['v'], world['rotations'][regime]
    view = lambda g, e: (c@u.T+np.sqrt(TEXTURE)*g@v.T+np.sqrt(ISO)*e)@r.T
    return view(g1, e1), view(g2, e2)


def split(n):
    """One-based indices divisible by four validate; the rest fit."""
    validation = np.arange(1, n+1) % 4 == 0
    return ~validation, validation


def groups(phase, d=D, b=B):
    """G_{p,j} = {(p+b j+t) mod d : t < b}, one row per group."""
    return (phase+np.arange(d).reshape(d//b, b)) % d


# ---------------------------------------------------------------- estimators

def statistics(x, y):
    m = len(x)
    return (x.T@y+y.T@x)/(2*m), (x.T@x+y.T@y)/(2*m)


def canonical_basis(q):
    """Basis of span(q) from coordinate axes projected in index order, Gram-Schmidt."""
    projector, chosen = q@q.T, []
    for i in range(len(q)):
        r = projector[:, i].copy()
        for c in chosen:
            r -= (c@r)*c
        norm = np.linalg.norm(r)
        if norm > AXIS_FLOOR:
            chosen.append(r/norm)
            if len(chosen) == q.shape[1]:
                return np.array(chosen).T
    raise ScoringFailure('canonical tie basis did not reach the eigenspace dimension')


def canonical_top(matrix, count):
    """Largest algebraic eigenvectors with the contract's tie grouping."""
    vals, vecs = np.linalg.eigh(matrix)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    tol = EIG_TIE*max(1., float(np.abs(vals).max()))
    out, start = [], 0
    while len(out) < count:
        stop = start+1
        while stop < len(vals) and vals[stop-1]-vals[stop] <= tol:
            stop += 1
        group = vecs[:, start:stop]
        if stop-start > 1:
            group = canonical_basis(group)
        out.extend(group.T[:count-len(out)])
        start = stop
    return np.array(out).T


def orient(w):
    """Make each column's first largest-magnitude coordinate positive."""
    idx = np.argmax(np.abs(w), axis=0)
    return w*np.where(w[idx, np.arange(w.shape[1])] < 0, -1., 1.)


def cca_top(k, t, count, meter):
    """Top generalized eigenvectors of (k, t + r I), r = RIDGE tr(t)/dim, via Cholesky whitening."""
    reg = t+RIDGE*np.trace(t)/len(t)*np.eye(len(t))
    try:
        l = np.linalg.cholesky(reg)
    except np.linalg.LinAlgError as error:
        raise ScoringFailure(f'nonpositive regularized covariance: {error}') from None
    meter.counts[f'cholesky_{len(t)}'] += 1
    white = solve_triangular(l, solve_triangular(l, k, lower=True).T, lower=True)
    vecs = canonical_top((white+white.T)/2, count)
    meter.counts[f'eigh_{len(t)}'] += 1
    w = solve_triangular(l.T, vecs, lower=False)
    return orient(w/np.linalg.norm(w, axis=0))


def spectral_top(k, count, meter):
    meter.counts[f'eigh_{len(k)}'] += 1
    return orient(canonical_top(k, count))


def fit_local(x, y, phase, method, b=B, meter=None):
    """One unit feature per group of the declared phase; returns (d/b, b) weights.

    Block statistics are built independently per group: no reuse across phases or arms."""
    meter = meter or Meter()
    d = x.shape[1]
    out = np.zeros((d//b, b))
    for j, g in enumerate(groups(phase, d, b)):
        with meter.time('covariance'):
            kg, tg = statistics(x[:, g], y[:, g])
        with meter.time('factorization'):
            out[j] = (cca_top(kg, tg, 1, meter) if method == 'cca' else spectral_top(kg, 1, meter))[:, 0]
    meter.counts['fits'] += 1
    meter.counts['fit_row_exposures'] += len(x)
    return finite(out)


def fit_global(x, y, method, rank, meter=None):
    meter = meter or Meter()
    with meter.time('covariance'):
        k, t = statistics(x, y)
    with meter.time('factorization'):
        w = cca_top(k, t, rank, meter) if method == 'cca' else spectral_top(k, rank, meter)
    meter.counts['fits'] += 1
    meter.counts['fit_row_exposures'] += len(x)
    return finite(w)


def finite(a):
    if not np.all(np.isfinite(a)):
        raise ScoringFailure('nonfinite value')
    return a


def local_basis(weights, phase, d):
    """Dense (d, k) basis of a local code; columns are orthonormal by disjoint support."""
    out = np.zeros((d, len(weights)))
    for j, g in enumerate(groups(phase, d, weights.shape[1])):
        out[g, j] = weights[j]
    return out


def orthonormal_basis(w):
    """Span basis by reduced QR; rank k is required, never repaired."""
    s = np.linalg.svd(w, compute_uv=False)
    if s[-1] <= RANK_FLOOR*s[0]:
        raise ScoringFailure(f'rank failure: singular values {s}')
    return np.linalg.qr(w)[0]


def encode_local(h, phase, weights):
    return (h[..., groups(phase, h.shape[-1], weights.shape[1])]*weights).sum(-1)


def reconstruct_local(z, phase, weights, d):
    out = np.zeros(z.shape[:-1]+(d,))
    out[..., groups(phase, d, weights.shape[1])] = weights*z[..., None]
    return out


def validation_score(xv, yv, phase, weights):
    """mean(h1^T P h2) through the deployed O(d) local code."""
    return float(np.mean(np.sum(encode_local(xv, phase, weights)*encode_local(yv, phase, weights), -1)))


def select(scores):
    """Maximum score; scores within SCORE_TIE*max(1,|max|) tie and the smallest phase wins."""
    scores = np.asarray(scores)
    best = scores.max()
    return int(np.flatnonzero(scores >= best-SCORE_TIE*max(1., abs(best)))[0])


def fit_adaptive(x, y, method, b=B, meter=None):
    """Fit every phase on the fitting split, select on the validation split, never refit."""
    meter = meter or Meter()
    fit, val = split(len(x))
    candidates = [fit_local(x[fit], y[fit], p, method, b, meter) for p in range(b)]
    with meter.time('validation'):
        scores = [validation_score(x[val], y[val], p, w) for p, w in enumerate(candidates)]
        chosen = select(finite(np.array(scores)))
    meter.counts['candidate_scores'] += len(scores)
    meter.counts['validation_row_exposures'] += len(scores)*int(val.sum())
    return chosen, candidates, scores


def fit_arm(arm, x, y, b=B, meter=None):
    """Returns (orthonormal basis, local phase or None, adaptive record or None)."""
    meter = meter or Meter()
    d, rank = x.shape[1], x.shape[1]//b
    method = 'cca' if 'cca' in arm else 'spectral'
    if arm.startswith('adaptive'):
        chosen, candidates, scores = fit_adaptive(x, y, method, b, meter)
        return local_basis(candidates[chosen], chosen, d), chosen, dict(
            candidates=[local_basis(w, p, d) for p, w in enumerate(candidates)], scores=scores,
            weights=candidates[chosen])
    if arm.endswith('_fit'):
        fit, _ = split(len(x))
        x, y = x[fit], y[fit]
    if arm.startswith('fixed'):
        return local_basis(fit_local(x, y, 0, method, b, meter), 0, d), 0, None
    return orthonormal_basis(fit_global(x, y, method, rank, meter)), None, None


# ---------------------------------------------------------------- evaluator only

def signal_covariance(world, regime):
    r = world['rotations'][regime]
    return r@world['u']@world['u'].T@r.T


def noise_covariance(world, regime, condition='in_distribution'):
    u, v, r = world['u'], world['v'], world['rotations'][regime]
    d = len(u)
    texture = 4*TEXTURE if condition == 'texture_16' else TEXTURE
    n = texture*v@v.T+ISO*np.eye(d)
    if condition == 'extra_noise':
        n += EXTRA*(np.eye(d)-u@u.T-v@v.T)
    return r@n@r.T


def phase_floors(s, b=B):
    """Best population signal loss of each phase: 1 - sum_g lambda_max(S_gg)/k."""
    d = len(s)
    k = np.trace(s)
    return [float(1-sum(np.linalg.eigvalsh(s[np.ix_(g, g)])[-1] for g in groups(p, d, b))/k)
            for p in range(b)]


def signal_loss(basis, s):
    return float(1-np.trace(basis.T@s@basis)/np.trace(s))


def projected_noise(basis, s, n):
    return float(np.trace(basis.T@n@basis)/np.trace(s))


def nmse(basis, s, n):
    """E||P h - s||^2 / E||s||^2 = [tr((I-P)S) + tr(PN)]/tr(S)."""
    return signal_loss(basis, s)+projected_noise(basis, s, n)
