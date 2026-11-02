"""Local sparse memory with learned addresses (docs/shared-core-sparse-memory-v1-contract.md, Mechanism). Units are allocated on mismatch by a vigilance tau; an address is the
running mean direction of the cues a unit took; contents are running means. A write changes only the selected units; there is no optimizer and nothing global. Numpy, float32."""
import numpy as np


class Memory:
    def __init__(self, H, k, tau, d, m):
        self.H, self.k, self.tau, self.d, self.m = H, k, tau, d, m
        self.c, self.a = np.zeros((H, d), np.float32), np.zeros((H, d), np.float32)
        self.u, self.n = np.zeros((H, m), np.float32), np.zeros(H, np.int64)
        self.h, self.forced = 0, 0

    def bytes_limit(self):
        return 4 * self.H * (self.d + self.m + 1)

    def bytes_used(self):
        return 4 * self.h * (self.d + self.m + 1)

    def _winners(self, sims):
        """Up to k units at or above tau, highest first, ties to the lower index."""
        good = np.flatnonzero(sims >= self.tau)
        if len(good) <= 1 or self.k == 1 and len(good) == 1:
            return good
        return good[np.argsort(-sims[good], kind='stable')[:self.k]]

    def write(self, z, v):
        """One association; returns the units written. z is a unit vector. Nothing else changes."""
        h = self.h
        if not np.any(z):
            return np.zeros(0, int)
        sims = self.a[:h] @ z
        I = self._winners(sims)
        if not len(I):
            if h < self.H:
                I = np.array([h])
                self.h = h + 1
            else:
                I = np.array([int(np.argmax(sims))])
                self.forced += 1
        self.c[I] += z
        self.a[I] = self.c[I] / np.linalg.norm(self.c[I], axis=1, keepdims=True)
        self.u[I] += v
        self.n[I] += 1
        return I

    def read(self, Z, part=None):
        """Z (q, d') unit vectors -> contents (q, m) and answered (q,). part: a slice of the address the query is compared with, renormalized; the query then carries only that part."""
        a = self.a[:self.h]
        if part is not None:
            a = a[:, part]
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        out, ok = np.zeros((len(Z), self.m), np.float32), np.zeros(len(Z), bool)
        if not self.h:
            return out, ok
        S = Z @ a.T
        val = self.u[:self.h] / self.n[:self.h, None]
        for i, s in enumerate(S):
            I = self._winners(s)
            if len(I) and np.any(Z[i]):
                out[i], ok[i] = val[I].mean(0), True
        return out, ok
