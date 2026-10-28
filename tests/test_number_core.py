import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import number_concept as nc  # noqa: E402
from src import number_core as ncore  # noqa: E402


def explicit_energy(sri, kind, s):
    """Energy written term by term for states s (3 modules x [x, h, c])."""
    o, m, i = sri.sizes
    order = (kind, 'core', 'output')
    parts = {}
    for j, k in enumerate(order):
        z = s[j * sri.state:(j + 1) * sri.state]
        parts[k] = (z[:o], z[o:o + m], z[o + m:])
    e = 0.
    for k, (x, h, c) in parts.items():
        e = e + .5 * (x - sri.W[k] @ h).pow(2).sum() + .5 * (h - sri.V[k] @ c).pow(2).sum()
        e = e + .5 * ncore.LAMBDA * (h.pow(2).sum() + c.pow(2).sum() + (x.pow(2).sum() if k == 'core' else 0))
    for a, b in ((kind, 'core'), ('core', 'output')):
        for l in range(3):
            if sri.kappa[l] > 0:
                e = e + .5 * sri.kappa[l] * (parts[b][l] - sri.T[f'{a}-{b}-{l}'] @ parts[a][l]).pow(2).sum()
    return e


class NumberCoreTests(unittest.TestCase):
    def test_morse_fits_window_and_codes(self):
        self.assertEqual(sorted(ncore.MORSE), list(range(10)))
        rng = nc.stream('test_morse')
        for c in range(10):
            for _ in range(300):
                x, end = ncore.morse_signal(ncore.MORSE[c], rng)
                self.assertLessEqual(end, ncore.MORSE_LEN)
                self.assertEqual(x.shape, (ncore.MORSE_LEN,))

    def test_scramble_is_derangement_and_non_monotone(self):
        perm = ncore.scramble()
        self.assertTrue(np.all(perm != np.arange(10)))
        self.assertEqual(sorted(perm.tolist()), list(range(10)))
        codes = perm[np.arange(10)]
        self.assertTrue(np.array_equal(ncore.morse_numbers(codes, perm), np.arange(10)))

    def test_hessian_matches_explicit_energy(self):
        for topology, width in ((t, w) for t in ncore.TOPOLOGIES for w in ncore.WIDTHS):
            sri = ncore.SRI(topology, torch.Generator().manual_seed(0), ncore.WIDTHS[width])
            with torch.no_grad():
                for p in sri.T.values():
                    p.add_(torch.randn(p.shape) * .3)
            for kind in ('image', 'morse'):
                Q = sri.hessian(kind).detach()
                s = torch.randn(3 * sri.state)
                a, b = float(.5 * s @ Q @ s), float(explicit_energy(sri, kind, s))
                self.assertLess(abs(a - b) / abs(b), 1e-5)

    def test_equilibrium_zero_gradient(self):
        for width in ncore.WIDTHS.values():
            sri = ncore.SRI('uniform', torch.Generator().manual_seed(1), width)
            M, Q = sri.equilibrium_map('image')
            x = torch.randn(5, width[0])
            full = torch.cat([x, x @ M.T], 1)
            grad = (full @ Q.T).detach()[:, width[0]:]
            self.assertLess(float(grad.abs().max()), 1e-3)

    def test_label_codes_orthonormal(self):
        for width in ncore.WIDTHS.values():
            q = ncore.label_codes(width[0])
            self.assertTrue(torch.allclose(q @ q.T, torch.eye(10), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
