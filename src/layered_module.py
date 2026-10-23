"""Layered module v1 (Stage 1): one disk-shaped layered module with continuous states.

See docs/layered-module-v1-contract.md. Units are ordered ring 1 (exterior) .. ring L, then the centre.
Energy E(u) = 1/2 u^T K u + sum_i V(u_i) + lambda/2 sum_{i in ring 1} (u_i - h_i)^2, with K the weighted
graph Laplacian (lateral weights gamma^(l-1), radial and centre weights 1).
"""
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


N1 = 32
LAMBDA = 1.0
CERT_GRAD = 1e-8
CERT_EIG = 1e-8
POLISH_GRAD = 1e-12
LBFGS_OPTIONS = dict(maxiter=20000, maxcor=20, ftol=0.0, gtol=1e-11, maxls=50)


# ---------------------------------------------------------------- graph

def ring_sizes(L, n1=N1):
    """n_l = max(4, floor(n1 (L-l+1)/L + 1/2)), rounding half up, exactly."""
    return [max(4, int(Fraction(n1 * (L - l + 1), L) + Fraction(1, 2))) for l in range(1, L + 1)]


def _circ(a, b):
    """Circular distance between fractions of a turn, exactly."""
    d = abs(a - b) % 1
    return min(d, 1 - d)


def _nearest(j, n_from, n_to):
    a = Fraction(j, n_from)
    return min(range(n_to), key=lambda k: (_circ(a, Fraction(k, n_to)), k))


class Disk:
    """Layered disk graph: rings 1..L (exterior first) and one centre unit."""
    def __init__(self, L, gamma, n1=N1):
        self.L, self.gamma = L, gamma
        self.sizes = ring_sizes(L, n1)
        self.offsets = list(np.cumsum([0] + self.sizes))
        self.centre = self.offsets[-1]
        self.n = self.centre + 1
        lateral, radial = set(), set()
        for l, n in enumerate(self.sizes):
            o = self.offsets[l]
            for j in range(n):
                lateral.add((l, tuple(sorted((o + j, o + (j + 1) % n)))))
        for l in range(L - 1):
            n_a, n_b = self.sizes[l], self.sizes[l + 1]
            o_a, o_b = self.offsets[l], self.offsets[l + 1]
            hit = set()
            for j in range(n_a):
                k = _nearest(j, n_a, n_b)
                radial.add((o_a + j, o_b + k))
                hit.add(k)
            for k in range(n_b):
                if k not in hit:
                    radial.add((o_a + _nearest(k, n_b, n_a), o_b + k))
        for j in range(self.sizes[-1]):
            radial.add((self.offsets[L - 1] + j, self.centre))
        self.lateral = sorted(lateral)
        self.radial = sorted(radial)
        K = np.zeros((self.n, self.n))
        for l, (i, j) in self.lateral:
            self._add(K, i, j, float(gamma) ** l)
        for i, j in self.radial:
            self._add(K, i, j, 1.0)
        self.K = K
        self.ext = np.zeros(self.n)
        self.ext[:self.sizes[0]] = 1.0

    @staticmethod
    def _add(K, i, j, w):
        K[i, i] += w
        K[j, j] += w
        K[i, j] -= w
        K[j, i] -= w

    def layer(self, l):
        """Unit indices of ring l (1-based)."""
        return slice(self.offsets[l - 1], self.offsets[l])

    def connected(self):
        seen, stack = {0}, [0]
        adj = [[] for _ in range(self.n)]
        for _, (i, j) in self.lateral:
            adj[i].append(j)
            adj[j].append(i)
        for i, j in self.radial:
            adj[i].append(j)
            adj[j].append(i)
        while stack:
            for v in adj[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == self.n


# ---------------------------------------------------------------- potentials and energy

class Potential:
    """('linear', mu): V = mu u^2 / 2.  ('double_well', beta): V = beta (u^2 - 1)^2 / 4."""
    def __init__(self, kind, value):
        if kind not in ('linear', 'double_well'):
            raise ValueError(kind)
        self.kind, self.value = kind, float(value)

    def V(self, u):
        if self.kind == 'linear':
            return 0.5 * self.value * u * u
        return 0.25 * self.value * (u * u - 1.0) ** 2

    def dV(self, u):
        return self.value * u if self.kind == 'linear' else self.value * u * (u * u - 1.0)

    def d2V(self, u):
        return np.full_like(u, self.value) if self.kind == 'linear' else self.value * (3.0 * u * u - 1.0)

    @property
    def name(self):
        return f"{'mu' if self.kind == 'linear' else 'beta'}={self.value:g}"


class Module:
    def __init__(self, disk, potential, lam=LAMBDA):
        self.disk, self.pot, self.lam = disk, potential, lam

    def _full_h(self, h):
        H = np.zeros(self.disk.n)
        H[:self.disk.sizes[0]] = h
        return H

    def energy(self, u, h):
        H = self._full_h(h)
        r = (u - H) * self.disk.ext
        return 0.5 * u @ self.disk.K @ u + self.pot.V(u).sum() + 0.5 * self.lam * r @ r

    def grad(self, u, h):
        H = self._full_h(h)
        return self.disk.K @ u + self.pot.dV(u) + self.lam * self.disk.ext * (u - H)

    def hess(self, u):
        return self.disk.K + np.diag(self.pot.d2V(u) + self.lam * self.disk.ext)

    def direct_linear(self, h):
        """Direct sparse solve of H u = lambda P1 h (linear potential only)."""
        if self.pot.kind != 'linear':
            raise ValueError('direct solve needs a linear potential')
        return spsolve(csr_matrix(self.hess(np.zeros(self.disk.n))), self.lam * self._full_h(h))

    def settle(self, h, start=None):
        """L-BFGS-B, then Newton polish; returns (u, certified, info)."""
        u0 = np.zeros(self.disk.n) if start is None else np.asarray(start, dtype=float)
        res = minimize(self.energy, u0, args=(h,), jac=self.grad, method='L-BFGS-B', options=LBFGS_OPTIONS)
        u = res.x
        newton = 0
        for _ in range(20):
            g = self.grad(u, h)
            if np.max(np.abs(g)) < POLISH_GRAD:
                break
            try:
                step = np.linalg.solve(self.hess(u), g)
            except np.linalg.LinAlgError:
                break
            e0, g0, t, moved = self.energy(u, h), np.max(np.abs(g)), 1.0, False
            for _ in range(31):
                cand = u - t * step
                e1 = self.energy(cand, h)
                if e1 < e0 or (e1 <= e0 + 1e-12 * max(1.0, abs(e0))
                               and np.max(np.abs(self.grad(cand, h))) < g0):
                    u, moved = cand, True
                    break
                t *= 0.5
            newton += 1
            if not moved:
                break
        gmax = float(np.max(np.abs(self.grad(u, h))))
        lmin = float(np.linalg.eigvalsh(self.hess(u))[0])
        return u, gmax < CERT_GRAD and lmin > CERT_EIG, dict(grad=gmax, min_eig=lmin, newton_steps=newton,
                                                            lbfgs_iterations=int(res.nit))


# ---------------------------------------------------------------- drives

def theta(n=N1):
    return 2 * np.pi * np.arange(n) / n


def signal(n_freq, c, n=N1):
    return np.cos(n_freq * theta(n) + np.pi * c / 2)


def fourier_basis(n=N1):
    """Orthonormal real Fourier basis of R^n (n even): DC, cos/sin k=1..n/2-1, Nyquist; rows are vectors."""
    t = theta(n)
    rows = [np.full(n, 1 / np.sqrt(n))]
    for k in range(1, n // 2):
        rows.append(np.cos(k * t) * np.sqrt(2 / n))
        rows.append(np.sin(k * t) * np.sqrt(2 / n))
    rows.append(np.cos(n // 2 * t) / np.sqrt(n))
    return np.array(rows)


def nuisance_unit(seed, shape, n=N1):
    """Unit-RMS nuisance draws: Fourier coefficients ~ N(0,1), combined, rescaled to RMS exactly 1."""
    coef = np.random.default_rng(seed).standard_normal(shape + (n,))
    z = coef @ fourier_basis(n)
    return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True))


# ---------------------------------------------------------------- measures

def layer_variance(states, mask):
    """Population variance per unit over certified rows, averaged over units; None if no rows."""
    X = states[mask]
    return float(X.var(axis=0, ddof=0).mean()) if len(X) else None


def within_class_variance(states, labels, mask):
    vals = []
    for c in sorted(set(labels.tolist())):
        sel = mask & (labels == c)
        if sel.any():
            vals.append(states[sel].var(axis=0, ddof=0).mean())
    return float(np.mean(vals)) if vals else None


def nearest_mean(train, train_labels, train_ok, test, test_labels, test_ok, classes=4):
    """Strict and secondary accuracy of a nearest-class-mean classifier (ties to the lowest class)."""
    means = {}
    for c in range(classes):
        sel = train_ok & (train_labels == c)
        if sel.any():
            means[c] = train[sel].mean(axis=0)
    correct = 0
    for x, y, ok in zip(test, test_labels, test_ok):
        if not ok or y not in means:
            continue
        d = [(float(np.sum((x - means[c]) ** 2)), c) for c in sorted(means)]
        correct += min(d)[1] == y
    n_ok = int(test_ok.sum())
    return correct / len(test), (correct / n_ok if n_ok else None)


def min_mahalanobis(train, train_labels, train_ok, sigma0, classes=4):
    """Minimum pairwise Mahalanobis distance between training class means (pooled covariance + sigma0^2 I)."""
    means, resid = [], []
    for c in range(classes):
        sel = train_ok & (train_labels == c)
        if not sel.any():
            return None
        m = train[sel].mean(axis=0)
        means.append(m)
        resid.append(train[sel] - m)
    R = np.vstack(resid)
    dof = len(R) - classes
    if dof <= 0:
        return None
    S = R.T @ R / dof + sigma0 ** 2 * np.eye(train.shape[1])
    Si = np.linalg.inv(S)
    best = None
    for a in range(classes):
        for b in range(a + 1, classes):
            d = means[a] - means[b]
            v = float(np.sqrt(d @ Si @ d))
            best = v if best is None else min(best, v)
    return best


def rms_distance(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


__all__ = ['CERT_EIG', 'CERT_GRAD', 'Disk', 'LAMBDA', 'LBFGS_OPTIONS', 'Module', 'N1', 'POLISH_GRAD', 'Potential',
           'fourier_basis', 'layer_variance', 'min_mahalanobis', 'nearest_mean', 'nuisance_unit', 'ring_sizes',
           'rms_distance', 'signal', 'theta', 'within_class_variance']
