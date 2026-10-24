"""Layered module v2: constant-width geometry, field drive, covariance-aware readouts and the population reference.

See docs/layered-module-v2-contract.md. Extends src/layered_module.py without changing it: the shrinking geometry
and the soft tie use the v1 classes unchanged, so v1 configurations reproduce exactly.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import beta as beta_dist, norm

from src.layered_module import LAMBDA, N1, Disk, Module, _nearest


GEOMETRIES = ('shrinking', 'constant')
DRIVES = ('tie', 'field')


# ---------------------------------------------------------------- graph

def disk_from_sizes(L, gamma, sizes):
    """A Disk with the given ring sizes, built by the v1 rule (rings, exact nearest-angle radial edges, centre)."""
    d = object.__new__(Disk)
    d.L, d.gamma = L, gamma
    d.sizes = list(sizes)
    d.offsets = list(np.cumsum([0] + d.sizes))
    d.centre = d.offsets[-1]
    d.n = d.centre + 1
    lateral, radial = set(), set()
    for l, n in enumerate(d.sizes):
        o = d.offsets[l]
        for j in range(n):
            lateral.add((l, tuple(sorted((o + j, o + (j + 1) % n)))))
    for l in range(L - 1):
        n_a, n_b = d.sizes[l], d.sizes[l + 1]
        o_a, o_b = d.offsets[l], d.offsets[l + 1]
        hit = set()
        for j in range(n_a):
            k = _nearest(j, n_a, n_b)
            radial.add((o_a + j, o_b + k))
            hit.add(k)
        for k in range(n_b):
            if k not in hit:
                radial.add((o_a + _nearest(k, n_b, n_a), o_b + k))
    for j in range(d.sizes[-1]):
        radial.add((d.offsets[L - 1] + j, d.centre))
    d.lateral = sorted(lateral)
    d.radial = sorted(radial)
    K = np.zeros((d.n, d.n))
    for l, (i, j) in d.lateral:
        Disk._add(K, i, j, float(gamma) ** l)
    for i, j in d.radial:
        Disk._add(K, i, j, 1.0)
    d.K = K
    d.ext = np.zeros(d.n)
    d.ext[:d.sizes[0]] = 1.0
    return d


def make_disk(geometry, L, gamma):
    if geometry == 'shrinking':
        return Disk(L, gamma)
    if geometry == 'constant':
        return disk_from_sizes(L, gamma, [N1] * L)
    raise ValueError(geometry)


# ---------------------------------------------------------------- drives

class FieldModule(Module):
    """Field drive: E = 1/2 u^T K u + sum V(u) - sum_{ring 1} h_i u_i (no exterior tie)."""

    def energy(self, u, h):
        return 0.5 * u @ self.disk.K @ u + self.pot.V(u).sum() - self._full_h(h) @ u

    def grad(self, u, h):
        return self.disk.K @ u + self.pot.dV(u) - self._full_h(h)

    def hess(self, u):
        return self.disk.K + np.diag(self.pot.d2V(u))

    def direct_linear(self, h):
        """Direct sparse solve of H u = P1 h (linear potential only)."""
        if self.pot.kind != 'linear':
            raise ValueError('direct solve needs a linear potential')
        return spsolve(csr_matrix(self.hess(np.zeros(self.disk.n))), self._full_h(h))


def make_module(disk, potential, drive):
    if drive == 'tie':
        return Module(disk, potential)
    if drive == 'field':
        return FieldModule(disk, potential)
    raise ValueError(drive)


def drive_gain(drive, lam=LAMBDA):
    return lam if drive == 'tie' else 1.0


def linear_map(mod, drive):
    """Dense response matrix G (units x 32): u = G h for a linear potential."""
    B = np.zeros((mod.disk.n, mod.disk.sizes[0]))
    B[np.arange(mod.disk.sizes[0]), np.arange(mod.disk.sizes[0])] = drive_gain(drive, mod.lam)
    return np.linalg.solve(mod.hess(np.zeros(mod.disk.n)), B)


def mode_gains(n_freq, L, gamma, mu, drive, n=N1, lam=LAMBDA):
    """Per-layer gain g_n(l) of angular mode n_freq >= 1 in the constant-width linear module (tridiagonal solve)."""
    e = 2 * (1 - np.cos(2 * np.pi * n_freq / n))
    A = np.zeros((L, L))
    for l in range(L):
        A[l, l] = float(gamma) ** l * e + mu + (2.0 if l > 0 else 1.0)
        if l > 0:
            A[l, l - 1] = -1.0
        if l < L - 1:
            A[l, l + 1] = -1.0
    if drive == 'tie':
        A[0, 0] += lam
    b = np.zeros(L)
    b[0] = drive_gain(drive, lam)
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------- spectra

def ring_basis(n):
    """Orthonormal real Fourier basis of R^n (any n); returns (rows, mode index per row)."""
    t = 2 * np.pi * np.arange(n) / n
    rows, modes = [np.full(n, 1 / np.sqrt(n))], [0]
    for k in range(1, -(-n // 2)):
        rows += [np.cos(k * t) * np.sqrt(2 / n), np.sin(k * t) * np.sqrt(2 / n)]
        modes += [k, k]
    if n % 2 == 0:
        rows.append(np.cos(n // 2 * t) / np.sqrt(n))
        modes.append(n // 2)
    return np.array(rows), np.array(modes)


def spectrum(X, labels, mask, classes=4):
    """Per mode k: within-class variance P_k (averaged over classes) and between-class variance B_k of the means."""
    basis, modes = ring_basis(X.shape[1])
    C = X @ basis.T
    within, means = [], []
    for c in range(classes):
        sel = mask & (labels == c)
        if sel.any():
            within.append(C[sel].var(axis=0, ddof=0))
            means.append(C[sel].mean(axis=0))
    if not within:
        return None
    w, b = np.mean(within, axis=0), np.var(np.array(means), axis=0, ddof=0)
    out = {}
    for k in sorted(set(modes.tolist())):
        sel = modes == k
        out[str(k)] = dict(P=float(w[sel].sum()), B=float(b[sel].sum()))
    return out


# ---------------------------------------------------------------- readouts

def _class_means(train, labels, ok, classes):
    means = []
    for c in range(classes):
        sel = ok & (labels == c)
        if not sel.any():
            return None
        means.append(train[sel].mean(axis=0))
    return np.array(means)


def _score(scores, test_labels, test_ok):
    """Strict and secondary accuracy from a (test, class) score array; argmax ties go to the lowest class."""
    pred = np.argmax(scores, axis=1)
    correct = int(np.sum((pred == test_labels) & test_ok))
    n_ok = int(test_ok.sum())
    return dict(strict=correct / len(test_labels), secondary=correct / n_ok if n_ok else None, null=None)


def null_readout(n_test, reason):
    return dict(strict=0.0, secondary=None, null=reason)


def _min_pair(means, apply_inv):
    best = None
    for a in range(len(means)):
        for b in range(a + 1, len(means)):
            d = means[a] - means[b]
            v = float(np.sqrt(max(d @ apply_inv(d), 0.0)))
            best = v if best is None else min(best, v)
    return best


def lda(train, train_labels, train_ok, test, test_labels, test_ok, classes=4):
    """Pooled-covariance LDA (Cholesky, equal priors); returns (accuracy, empirical min Mahalanobis or None)."""
    means = _class_means(train, train_labels, train_ok, classes)
    if means is None:
        return null_readout(len(test), 'class_without_training_drive'), None
    n_ok, p = int(train_ok.sum()), train.shape[1]
    if n_ok - classes < p:
        return null_readout(len(test), 'too_few_training_drives'), None
    R = np.vstack([train[train_ok & (train_labels == c)] - means[c] for c in range(classes)])
    S = R.T @ R / (n_ok - classes)
    try:
        cf = cho_factor(S, lower=True)
    except np.linalg.LinAlgError:
        return null_readout(len(test), 'cholesky_failed'), None
    W = cho_solve(cf, means.T)                                          # (p, classes): Sigma^-1 m_c
    scores = test @ W - 0.5 * np.sum(means.T * W, axis=0)
    return _score(scores, test_labels, test_ok), _min_pair(means, lambda d: cho_solve(cf, d))


def lda_explicit_scores(train, train_labels, train_ok, test, classes=4):
    """Reference LDA scores with an explicit inverse (gate and tests only)."""
    means = _class_means(train, train_labels, train_ok, classes)
    R = np.vstack([train[train_ok & (train_labels == c)] - means[c] for c in range(classes)])
    Si = np.linalg.inv(R.T @ R / (int(train_ok.sum()) - classes))
    return np.array([[m @ Si @ x - 0.5 * m @ Si @ m for m in means] for x in test])


def lda_cholesky_scores(train, train_labels, train_ok, test, classes=4):
    means = _class_means(train, train_labels, train_ok, classes)
    R = np.vstack([train[train_ok & (train_labels == c)] - means[c] for c in range(classes)])
    cf = cho_factor(R.T @ R / (int(train_ok.sum()) - classes), lower=True)
    W = cho_solve(cf, means.T)
    return test @ W - 0.5 * np.sum(means.T * W, axis=0)


def projection_features(X, n_freq):
    """Coefficients on the orthonormal cos(n phi), sin(n phi) vectors of the ring; None unless n_l > 2 n."""
    n = X.shape[-1]
    if n <= 2 * n_freq:
        return None
    t = 2 * np.pi * np.arange(n) / n
    B = np.stack([np.cos(n_freq * t), np.sin(n_freq * t)]) * np.sqrt(2 / n)
    return X @ B.T


class Population:
    """Layer covariance Sigma = (alpha^2/2) G G^T + sigma^2 I and class means G s_c, handled through the SVD of G.

    G is n_l x 32 with n_l <= 32, so U is square and Sigma^-1 = U diag(1/((alpha^2/2) s^2 + sigma^2)) U^T exactly.
    Whitening this way stays well conditioned even when mode gains span many orders of magnitude.
    """

    def __init__(self, G, signals, alpha, sigma):
        U, s, _ = np.linalg.svd(G, full_matrices=False)
        if U.shape[0] != U.shape[1]:
            raise ValueError('layer map must have at most as many units as drive inputs')
        self.U = U
        self.d = (alpha ** 2 / 2) * s ** 2 + sigma ** 2
        self.means = signals @ G.T                                      # (classes, n_l)
        self.white = (self.means @ U) / np.sqrt(self.d)                 # means lie in range(G): exact whitening

    def apply_inv(self, x):
        return ((x @ self.U) / self.d) @ self.U.T

    def min_mahalanobis(self):
        return _min_pair(self.white, lambda d: d)

    def reference(self, z):
        """Monte Carlo Bayes accuracy: draw i is paired with class i mod 4; returns (accuracy, errors, draws)."""
        W = self.white
        _, sv, vt = np.linalg.svd(W, full_matrices=False)
        # the means are +-a, +-b, so they span at most 2 dimensions; rounding can add tiny further components
        r = min(2, int(np.sum(sv > 1e-12 * max(1.0, sv[0])))) if sv.size else 0
        coords = W @ vt[:r].T if r else np.zeros((len(W), 0))           # means in an orthonormal basis of their span
        cls = np.arange(len(z)) % len(W)
        x = coords[cls] + z[:, :r]
        scores = x @ coords.T - 0.5 * np.sum(coords ** 2, axis=1)
        errors = int(np.sum(np.argmax(scores, axis=1) != cls))
        return 1 - errors / len(z), errors, len(z)

    def oracle_lda(self, est_means, test, test_labels, test_ok):
        W = self.apply_inv(est_means)                                   # rows: Sigma^-1 m_c (Sigma symmetric)
        scores = test @ W.T - 0.5 * np.sum(est_means * W, axis=1)
        return _score(scores, test_labels, test_ok)


def clopper_pearson(k, n, level):
    """Two-sided Clopper-Pearson interval for a binomial proportion at the given confidence level."""
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def closed_form_reference(g, alpha, sigma):
    """Constant-width A_ref = (1 - Q(4 g / (sqrt 2 sqrt(g^2 alpha^2/2 + sigma^2))))^2; returns the error rate too."""
    q = float(norm.sf(4 * g / (np.sqrt(2) * np.sqrt(g * g * alpha ** 2 / 2 + sigma ** 2))))
    acc = (1 - q) ** 2
    return acc, q * (2 - q)


__all__ = ['DRIVES', 'FieldModule', 'GEOMETRIES', 'Population', 'clopper_pearson', 'closed_form_reference',
           'disk_from_sizes', 'drive_gain', 'lda', 'lda_cholesky_scores', 'lda_explicit_scores', 'linear_map',
           'make_disk', 'make_module', 'mode_gains', 'null_readout', 'projection_features', 'ring_basis', 'spectrum']
