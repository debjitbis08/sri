import unittest

import numpy as np

from src.layered_module import (
    Disk, Module, Potential, fourier_basis, layer_variance, min_mahalanobis, nearest_mean, nuisance_unit,
    ring_sizes, signal, within_class_variance,
)


class GraphTests(unittest.TestCase):
    def test_ring_sizes_round_half_up(self):
        self.assertEqual(ring_sizes(2), [32, 16])
        self.assertEqual(ring_sizes(3), [32, 21, 11])
        self.assertEqual(ring_sizes(6), [32, 27, 21, 16, 11, 5])
        self.assertEqual(ring_sizes(2, n1=5), [5, 4])                  # 2.5 rounds up to 3, then minimum 4
        self.assertEqual(ring_sizes(4, n1=10), [10, 8, 5, 4])           # 7.5 -> 8, 5, 2.5 -> 3 -> minimum 4

    def test_small_disk_edges_by_hand(self):
        d = Disk(2, 1, n1=8)                                            # rings 8 and 4, then the centre
        self.assertEqual(d.sizes, [8, 4])
        self.assertEqual(d.n, 13)
        outer = {i: j for i, j in d.radial if i < 8}
        # unit j of ring 1 (angle j/8) -> nearest of k/4; j odd is a tie between two k, lowest wins
        self.assertEqual([outer[j] - 8 for j in range(8)], [0, 0, 1, 1, 2, 2, 3, 0])   # 7/8 ties 3/4 and 0 circularly
        self.assertEqual(sum(1 for _, j in d.radial if j == d.centre), 4)
        self.assertTrue(d.connected())

    def test_backfill_connects_every_inner_unit(self):
        d = Disk(2, 1, n1=4)                                            # rings 4 and 4 (minimum), no back-fill needed
        self.assertEqual(d.sizes, [4, 4])
        inner = {j for i, j in d.radial if i < 4}
        self.assertEqual(inner, {4, 5, 6, 7})

    def test_laplacian_and_gamma_weights(self):
        for g in (1, 2):
            d = Disk(3, g)
            self.assertTrue(np.allclose(d.K.sum(axis=1), 0))
            self.assertTrue(np.allclose(d.K, d.K.T))
            o = d.offsets[2]
            self.assertEqual(-d.K[o, o + 1], float(g) ** 2)              # ring 3 lateral weight gamma^2


class EnergyTests(unittest.TestCase):
    def test_derivatives_match_finite_differences(self):
        rng = np.random.default_rng(1)
        for pot in (Potential('linear', 0), Potential('linear', 1), Potential('double_well', 2)):
            mod = Module(Disk(3, 2, n1=8), pot)
            u, h = rng.standard_normal(mod.disk.n), rng.standard_normal(8)
            e = np.eye(mod.disk.n) * 1e-6
            fd = [(mod.energy(u + e[i], h) - mod.energy(u - e[i], h)) / 2e-6 for i in range(mod.disk.n)]
            self.assertTrue(np.allclose(fd, mod.grad(u, h), atol=1e-6))
            fdh = np.array([(mod.grad(u + e[i], h) - mod.grad(u - e[i], h)) / 2e-6 for i in range(mod.disk.n)])
            self.assertTrue(np.allclose(fdh, mod.hess(u), atol=1e-5))

    def test_linear_settle_equals_direct_solve_and_superposes(self):
        rng = np.random.default_rng(2)
        mod = Module(Disk(3, 1), Potential('linear', 0))
        h1, h2 = rng.standard_normal(32), rng.standard_normal(32)
        u1, c1, info = mod.settle(h1)
        u2, _, _ = mod.settle(h2)
        u12, _, _ = mod.settle(h1 + h2)
        self.assertTrue(c1)
        self.assertLess(info['grad'], 1e-12)
        self.assertLess(np.max(np.abs(u1 - mod.direct_linear(h1))), 1e-10)
        self.assertLess(np.linalg.norm(u12 - u1 - u2), 1e-9)

    def test_harmonic_interior_is_neighbour_average(self):
        mod = Module(Disk(3, 1), Potential('linear', 0))
        u, _, _ = mod.settle(np.random.default_rng(3).standard_normal(32))
        K = mod.disk.K
        for i in range(32, mod.disk.n):                                 # interior units: (K u)_i = 0
            self.assertAlmostEqual(float(K[i] @ u), 0.0, places=10)

    def test_double_well_certifies_minimum_and_rejects_saddle(self):
        mod = Module(Disk(2, 1, n1=8), Potential('double_well', 1))
        u, cert, info = mod.settle(np.zeros(8))                         # u = 0 is a stationary saddle here
        self.assertFalse(cert)
        self.assertLess(info['min_eig'], 0)
        u, cert, _ = mod.settle(np.full(8, 0.5))
        self.assertTrue(cert)
        self.assertTrue(np.all(u > 0))


class DriveTests(unittest.TestCase):
    def test_fourier_basis_orthonormal(self):
        B = fourier_basis()
        self.assertEqual(B.shape, (32, 32))
        self.assertTrue(np.allclose(B @ B.T, np.eye(32)))

    def test_nuisance_unit_rms_and_determinism(self):
        z = nuisance_unit(5, (2, 3))
        self.assertTrue(np.allclose(np.sqrt(np.mean(z ** 2, axis=-1)), 1))
        self.assertTrue(np.array_equal(z, nuisance_unit(5, (2, 3))))

    def test_signal_rms(self):
        for n in (1, 8):
            for c in range(4):
                self.assertAlmostEqual(float(np.sqrt(np.mean(signal(n, c) ** 2))), 1 / np.sqrt(2))


class MeasureTests(unittest.TestCase):
    def test_variances_skip_unsettled(self):
        X = np.array([[0.0], [2.0], [100.0]])
        ok = np.array([True, True, False])
        self.assertEqual(layer_variance(X, ok), 1.0)
        self.assertEqual(within_class_variance(X, np.array([0, 1, 1]), ok), 0.0)
        self.assertIsNone(layer_variance(X, np.zeros(3, dtype=bool)))

    def test_nearest_mean_strict_secondary_and_ties(self):
        tr = np.array([[0.0], [0.0], [2.0], [2.0]])
        tl = np.array([0, 0, 1, 1])
        te = np.array([[0.1], [1.0], [1.9], [5.0]])
        el = np.array([0, 0, 1, 1])
        ok = np.array([True, True, True, False])
        strict, secondary = nearest_mean(tr, tl, np.ones(4, bool), te, el, ok, classes=2)
        self.assertEqual(strict, 3 / 4)                                  # 1.0 ties -> class 0 (correct); 5.0 unsettled
        self.assertEqual(secondary, 1.0)

    def test_nearest_mean_uses_training_only(self):
        tr, tl = np.array([[0.0], [1.0]]), np.array([0, 1])
        te, el = np.array([[0.9], [0.9]]), np.array([0, 0])              # test labels must not move the means
        strict, _ = nearest_mean(tr, tl, np.ones(2, bool), te, el, np.ones(2, bool), classes=2)
        self.assertEqual(strict, 0.0)

    def test_missing_class_counts_wrong(self):
        tr, tl = np.array([[0.0]]), np.array([0])
        te, el = np.array([[0.0], [3.0]]), np.array([0, 1])
        strict, _ = nearest_mean(tr, tl, np.ones(1, bool), te, el, np.ones(2, bool), classes=2)
        self.assertEqual(strict, 0.5)

    def test_mahalanobis_isotropic_case(self):
        tr = np.array([[0.0], [0.0], [2.0], [2.0]])
        tl = np.array([0, 0, 1, 1])
        d = min_mahalanobis(tr, tl, np.ones(4, bool), 1.0, classes=2)
        self.assertAlmostEqual(d, 2.0)                                   # zero within-class spread, sigma0 = 1


if __name__ == '__main__':
    unittest.main()
