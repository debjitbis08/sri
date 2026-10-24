import unittest

import numpy as np

from src.layered_module import Disk, Module, Potential, signal, within_class_variance
from src.layered_module_v2 import (
    FieldModule, Population, clopper_pearson, closed_form_reference, disk_from_sizes, lda, lda_cholesky_scores,
    lda_explicit_scores, linear_map, make_disk, make_module, mode_gains, projection_features, ring_basis, spectrum,
)


class GeometryTests(unittest.TestCase):
    def test_builder_reproduces_v1_disk(self):
        for L in (2, 3, 6):
            a, b = Disk(L, 2), disk_from_sizes(L, 2, Disk(L, 2).sizes)
            self.assertEqual(a.radial, b.radial)
            self.assertEqual(a.lateral, b.lateral)
            self.assertTrue(np.array_equal(a.K, b.K))

    def test_constant_width_radial_edges_are_identity(self):
        d = make_disk('constant', 3, 1)
        self.assertEqual(d.sizes, [32, 32, 32])
        self.assertEqual(d.n, 97)
        ring = sorted((i, j) for i, j in d.radial if j != d.centre)
        self.assertEqual(ring, [(o + j, o + 32 + j) for o in (0, 32) for j in range(32)])
        self.assertTrue(d.connected())

    def test_tie_uses_v1_module(self):
        mod = make_module(make_disk('shrinking', 2, 1), Potential('linear', 0), 'tie')
        self.assertIs(type(mod), Module)


class FieldDriveTests(unittest.TestCase):
    def test_field_derivatives_match_finite_differences(self):
        rng = np.random.default_rng(4)
        for pot in (Potential('linear', 1), Potential('double_well', 1)):
            mod = FieldModule(disk_from_sizes(3, 2, [8, 8, 8]), pot)
            u, h = rng.standard_normal(mod.disk.n), rng.standard_normal(8)
            e = np.eye(mod.disk.n) * 1e-6
            fd = [(mod.energy(u + e[i], h) - mod.energy(u - e[i], h)) / 2e-6 for i in range(mod.disk.n)]
            self.assertTrue(np.allclose(fd, mod.grad(u, h), atol=1e-6))
            fdh = np.array([(mod.grad(u + e[i], h) - mod.grad(u - e[i], h)) / 2e-6 for i in range(mod.disk.n)])
            self.assertTrue(np.allclose(fdh, mod.hess(u), atol=1e-5))

    def test_field_linear_settle_equals_direct_solve_and_linear_map(self):
        mod = FieldModule(make_disk('constant', 3, 1), Potential('linear', 1))
        h = np.random.default_rng(5).standard_normal(32)
        u, cert, _ = mod.settle(h)
        self.assertTrue(cert)
        self.assertLess(np.max(np.abs(u - mod.direct_linear(h))), 1e-10)
        self.assertLess(np.max(np.abs(u - linear_map(mod, 'field') @ h)), 1e-10)

    def test_tie_linear_map_includes_lambda(self):
        mod = make_module(make_disk('shrinking', 3, 1), Potential('linear', 0), 'tie')
        h = np.random.default_rng(6).standard_normal(32)
        self.assertLess(np.max(np.abs(mod.direct_linear(h) - linear_map(mod, 'tie') @ h)), 1e-10)

    def test_mode_gains_match_settled_response(self):
        t = 2 * np.pi * np.arange(32) / 32
        for drive, mu in (('tie', 0), ('tie', 1), ('field', 1)):
            for gamma in (1, 2):
                mod = make_module(make_disk('constant', 4, gamma), Potential('linear', mu), drive)
                for n in (1, 8):
                    g = mode_gains(n, 4, gamma, mu, drive)
                    u, _, _ = mod.settle(np.sin(n * t))
                    for l in range(1, 5):
                        self.assertLess(np.max(np.abs(u[mod.disk.layer(l)] - g[l - 1] * np.sin(n * t))), 1e-10)


class SpectrumTests(unittest.TestCase):
    def test_ring_basis_orthonormal_odd_and_even(self):
        for n in (5, 8, 11, 32):
            B, modes = ring_basis(n)
            self.assertEqual(B.shape, (n, n))
            self.assertTrue(np.allclose(B @ B.T, np.eye(n)))
            self.assertEqual(int(modes.max()), n // 2)

    def test_spectrum_conserves_within_class_variance(self):
        rng = np.random.default_rng(7)
        X, lab, ok = rng.standard_normal((40, 11)), np.repeat(np.arange(4), 10), np.ones(40, bool)
        ok[3] = False
        spec = spectrum(X, lab, ok)
        self.assertAlmostEqual(sum(v['P'] for v in spec.values()), 11 * within_class_variance(X, lab, ok))

    def test_pure_mode_lands_in_its_band(self):
        X = np.array([signal(8, c) for c in range(4)])
        spec = spectrum(X, np.arange(4), np.ones(4, bool))
        self.assertGreater(spec['8']['B'], 0.1)
        self.assertAlmostEqual(sum(v['B'] for k, v in spec.items() if k != '8'), 0.0)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.centres = rng.standard_normal((4, 3)) * 2
        self.tr = np.vstack([self.centres[c] + rng.standard_normal((10, 3)) for c in range(4)])
        self.te = np.vstack([self.centres[c] + rng.standard_normal((10, 3)) for c in range(4)])
        self.lab = np.repeat(np.arange(4), 10)
        self.ok = np.ones(40, bool)

    def test_cholesky_scores_match_explicit_inverse(self):
        a = lda_cholesky_scores(self.tr, self.lab, self.ok, self.te)
        b = lda_explicit_scores(self.tr, self.lab, self.ok, self.te)
        self.assertLess(np.max(np.abs(a - b)), 1e-9)

    def test_lda_strict_secondary_and_unsettled(self):
        te_ok = self.ok.copy()
        te_ok[:5] = False
        acc, maha = lda(self.tr, self.lab, self.ok, self.te, self.lab, te_ok)
        self.assertIsNone(acc['null'])
        self.assertAlmostEqual(acc['strict'] * 40, acc['secondary'] * 35)
        self.assertGreater(maha, 0)

    def test_lda_null_rules(self):
        acc, maha = lda(self.tr[:6], self.lab[:6], self.ok[:6], self.te, self.lab, self.ok)
        self.assertEqual((acc['strict'], acc['null'], maha), (0.0, 'class_without_training_drive', None))
        few = np.isin(np.arange(40), [0, 1, 10, 11, 20, 30])            # 6 certified - 4 classes = 2 < 3 dims
        acc, _ = lda(self.tr, self.lab, few, self.te, self.lab, self.ok)
        self.assertEqual((acc['strict'], acc['null']), (0.0, 'too_few_training_drives'))
        seven = np.isin(np.arange(40), [0, 1, 10, 11, 20, 21, 30])      # 7 - 4 = 3 = 3 dims: allowed
        acc, _ = lda(self.tr, self.lab, seven, self.te, self.lab, self.ok)
        self.assertIsNone(acc['null'])

    def test_lda_fit_ignores_test_set(self):
        a, m1 = lda(self.tr, self.lab, self.ok, self.te, self.lab, self.ok)
        b, m2 = lda(self.tr, self.lab, self.ok, self.te * 0 + 99, np.zeros(40, int), self.ok)
        self.assertEqual(m1, m2)

    def test_projection_requires_faithful_mode(self):
        self.assertIsNone(projection_features(np.zeros((3, 16)), 8))
        F = projection_features(np.array([signal(8, 0)]), 8)
        self.assertTrue(np.allclose(F, [[4.0, 0.0]]))                   # cos(8 theta) has norm 4 on 32 units


class ReferenceTests(unittest.TestCase):
    def test_population_inverse_matches_dense(self):
        rng = np.random.default_rng(9)
        G = rng.standard_normal((5, 32)) * 0.3
        pop = Population(G, np.array([signal(1, c) for c in range(4)]), 0.7, 0.2)
        S = 0.7 ** 2 / 2 * G @ G.T + 0.04 * np.eye(5)
        x = rng.standard_normal((3, 5))
        self.assertTrue(np.allclose(pop.apply_inv(x), x @ np.linalg.inv(S)))

    def test_monte_carlo_matches_closed_form_constant_width(self):
        z = np.random.default_rng(10).standard_normal((200000, 2))
        mod = make_module(make_disk('constant', 3, 1), Potential('linear', 0), 'tie')
        G = linear_map(mod, 'tie')
        g = mode_gains(8, 3, 1, 0, 'tie')
        sig = np.array([signal(8, c) for c in range(4)])
        for l in (2, 3):
            pop = Population(G[mod.disk.layer(l)], sig, 1.0, 0.1)
            acc, errors, n = pop.reference(z)
            _, err = closed_form_reference(g[l - 1], 1.0, 0.1)
            lo, hi = clopper_pearson(errors, n, 1 - 1e-6)
            self.assertTrue(lo <= err <= hi, (l, err, errors, lo, hi))

    def test_identical_means_give_chance(self):
        pop = Population(np.zeros((4, 32)) + 1e-30, np.array([signal(1, c) for c in range(4)]), 1.0, 1.0)
        acc, _, _ = pop.reference(np.random.default_rng(11).standard_normal((1000, 2)))
        self.assertEqual(acc, 0.25)

    def test_clopper_pearson_at_zero_errors(self):
        lo, hi = clopper_pearson(0, 10 ** 6, 1 - 1e-6)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1.450855e-05, places=10)
        self.assertTrue(lo <= closed_form_reference(0.27, 0.5, 0.03)[1] <= hi)


if __name__ == '__main__':
    unittest.main()
