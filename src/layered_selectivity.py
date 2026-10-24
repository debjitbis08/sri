"""Layered selectivity v1: a constant-width disk with a finite-range band-selective interior lateral penalty.

See docs/layered-selectivity-v1-contract.md. Units are ordered ring 1 (exterior) .. ring L, then the centre.
E = 1/2 ||u_1 - h||^2 + 1/2 sum_{l>=2} ||u_l - u_{l-1}||^2 + 1/2 sum_{l>=2} u_l^T K u_l + 1/2 sum_j (u_{L,j} - c)^2.
The energy is evaluated term by term; the gradient and Hessian come from the assembled spatial matrix, so the
derivative gate compares two independent constructions. Settling reuses the v1 Module (L-BFGS-B, Newton polish,
certification) unchanged.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from src.layered_module import Module, Potential


N = 32
ARMS = ('selective', 'no_lateral', 'ordinary_smoothing', 'wrong_bands')
BANDS = dict(selective=(1, 8), wrong_bands=(1, 4))
EIG = 2 - 2 * np.cos(2 * np.pi * np.arange(N) / N)                     # e_k, cycle Laplacian spectrum
J_FLOOR = 1e-12


def cycle_laplacian(n=N):
    eye = np.eye(n)
    return 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)


def band_symbols(bands):
    """q_k = [(e_k - e_a)(e_k - e_b)]^2, the excluded index set, and tau = 1 / min q_k over the other modes."""
    a, b = bands
    q = ((EIG - EIG[a]) * (EIG - EIG[b])) ** 2
    excluded = sorted({a, b, N - a, N - b})
    tau = 1 / min(q[k] for k in range(N) if k not in excluded)
    return q, excluded, float(tau)


def lateral(arm):
    """Interior lateral stiffness K (32 x 32) and its Fourier symbol p_k, built independently of each other."""
    C = cycle_laplacian()
    if arm == 'no_lateral':
        return np.zeros((N, N)), np.zeros(N)
    if arm == 'ordinary_smoothing':
        return C, EIG.copy()
    if arm in BANDS:
        a, b = BANDS[arm]
        q, _, tau = band_symbols(BANDS[arm])
        B = (C - EIG[a] * np.eye(N)) @ (C - EIG[b] * np.eye(N))
        K = tau * B.T @ B
        return (K + K.T) / 2, tau * q                                    # exactly symmetric in floating point
    raise ValueError(arm)


def band_operator(arm):
    """B with K = tau B^T B (band arms only); used for the term-by-term energy."""
    a, b = BANDS[arm]
    C = cycle_laplacian()
    _, _, tau = band_symbols(BANDS[arm])
    return (C - EIG[a] * np.eye(N)) @ (C - EIG[b] * np.eye(N)), tau


class SelectiveDisk:
    """Constant-width disk: L rings of 32, radial j <-> j, centre tied to ring L, interior lateral K on rings 2..L.

    Exposes the attributes the v1 Module uses: n, K (the full spatial Hessian minus the exterior tie), ext, sizes.
    """

    def __init__(self, arm, L):
        self.arm, self.L = arm, L
        self.sizes = [N] * L
        self.offsets = [l * N for l in range(L + 1)]
        self.centre = L * N
        self.n = L * N + 1
        self.lateral_K, self.symbol = lateral(arm)
        eye = np.eye(N)
        K = np.zeros((self.n, self.n))
        for l in range(2, L + 1):
            a, z = self.layer(l - 1), self.layer(l)
            K[a, a] += eye
            K[z, z] += eye + self.lateral_K
            K[a, z] -= eye
            K[z, a] -= eye
        z = self.layer(L)
        K[z, z] += eye
        K[self.centre, self.centre] += N
        K[z, self.centre] -= 1
        K[self.centre, z] -= 1
        self.K = K
        self.ext = np.zeros(self.n)
        self.ext[:N] = 1.0

    def layer(self, l):
        """Unit indices of ring l (1-based)."""
        return slice(self.offsets[l - 1], self.offsets[l])

    @property
    def hessian(self):
        return self.K + np.diag(self.ext)


class SelectivityModule(Module):
    """v1 Module with a zero local potential; the energy is evaluated term by term (contract section 2)."""

    def __init__(self, disk):
        super().__init__(disk, Potential('linear', 0.0), lam=1.0)
        self._B = band_operator(disk.arm) if disk.arm in BANDS else None
        self._chol = cho_factor(disk.hessian, lower=True)

    def energy(self, u, h):
        d, L = self.disk, self.disk.L
        rings = [u[d.layer(l)] for l in range(1, L + 1)]
        e = 0.5 * np.sum((rings[0] - h) ** 2)
        for l in range(1, L):
            e += 0.5 * np.sum((rings[l] - rings[l - 1]) ** 2)
            r = rings[l]
            if d.arm == 'ordinary_smoothing':
                e += 0.5 * np.sum((r - np.roll(r, 1)) ** 2)
            elif self._B is not None:
                B, tau = self._B
                Br = B @ r
                e += 0.5 * tau * Br @ Br
        return e + 0.5 * np.sum((rings[-1] - u[d.centre]) ** 2)

    def direct(self, h):
        """SPD direct solve of the full spatial Hessian (Cholesky)."""
        f = np.zeros(self.disk.n)
        f[:N] = h
        return cho_solve(self._chol, f)

    def transfer(self):
        """Dense spatial transfer G (units x 32): u = G h, by the SPD direct solve."""
        f = np.zeros((self.disk.n, N))
        f[:N] = np.eye(N)
        return cho_solve(self._chol, f)


# ---------------------------------------------------------------- Fourier (tridiagonal) reference

def mode_gains(symbol, L):
    """Gains g_k(l), shape (L, 32), from the per-mode symmetric tridiagonal system with the centre eliminated.

    Diagonal 2, 2+p_k, ..., 2+p_k, 1+p_k+b_k (b_0 = 0, b_k = 1 otherwise); for L = 2 it is (2, 1+p_k+b_k).
    """
    out = np.zeros((L, N))
    for k in range(N):
        p, b = symbol[k], (0.0 if k == 0 else 1.0)
        H = np.diag([2.0] + [2.0 + p] * (L - 2) + [1.0 + p + b])
        H += np.diag([-1.0] * (L - 1), 1) + np.diag([-1.0] * (L - 1), -1)
        rhs = np.zeros(L)
        rhs[0] = 1.0
        out[:, k] = np.linalg.solve(H, rhs)
    return out


def circulant(g):
    """Real circulant matrix with Fourier eigenvalues g (g_k = g_{32-k})."""
    return np.fft.ifft(np.fft.fft(np.eye(N), axis=0) * g[:, None], axis=0).real


def fourier_transfer(symbol, L):
    """Ring transfers (L, 32, 32) from the tridiagonal gains; independent of the spatial assembly."""
    return np.array([circulant(row) for row in mode_gains(symbol, L)])


def population(symbol, L, n_freq, alpha, sigma):
    """Population P, S, suppression, J_n, Gaussian-reference accuracy and the spherical known-mean lower bound."""
    g = mode_gains(symbol, L)
    P = alpha ** 2 / 64 * np.sum(g * g, axis=1)
    a = g[:, n_freq]
    S = a ** 2 / 2 + P
    v = alpha ** 2 * a[-1] ** 2 / 2 + sigma ** 2
    return dict(signal_gains=a.tolist(), P=P.tolist(), S=S.tolist(), suppression=float(P[-1] / P[0]),
                J=float(P[-1] / P[0] / (a[-1] / a[0]) ** 2) if a[-1] > 0 else None,
                gaussian_accuracy=float(norm.cdf(4 * a[-1] / np.sqrt(alpha ** 2 * a[-1] ** 2 + 2 * sigma ** 2)) ** 2),
                spherical_lower_bound=float(max(0.0, 1 - 2 * np.exp(-4 * a[-1] ** 2 / v))))


# ---------------------------------------------------------------- measures

def signal_power(states):
    """Vsig per unit: mean over classes and units of the squared deviation from the class-average state."""
    return float(np.mean((states - states.mean(axis=0, keepdims=True)) ** 2))


def selectivity(P1, PL, V1, VL, floor=J_FLOOR):
    """J_n = [P(L)/P(1)] / [Vsig(L)/Vsig(1)]; (None, reason) when a denominator is below the floor."""
    for name, x in (('P1', P1), ('Vsig1', V1), ('VsigL', VL)):
        if x is None:
            return None, f'{name}_unavailable'
        if x < floor:
            return None, f'{name}_below_floor'
    return (PL / P1) / (VL / V1), None


def max_coupling_distance(K, tol=0.0):
    """Largest circular distance |i - j| with |K_ij| > tol."""
    idx = np.argwhere(np.abs(K) > tol)
    if not len(idx):
        return 0
    d = np.abs(idx[:, 0] - idx[:, 1])
    return int(np.max(np.minimum(d, N - d)))


__all__ = ['ARMS', 'BANDS', 'EIG', 'J_FLOOR', 'N', 'SelectiveDisk', 'SelectivityModule', 'band_operator',
           'band_symbols', 'circulant', 'cycle_laplacian', 'fourier_transfer', 'lateral', 'max_coupling_distance',
           'mode_gains', 'population', 'selectivity', 'signal_power']
