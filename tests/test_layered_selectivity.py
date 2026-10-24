import unittest

import numpy as np

from src.layered_module import nearest_mean, nuisance_unit, signal
from src.layered_selectivity import (
    ARMS, EIG, N, SelectiveDisk, SelectivityModule, band_symbols, circulant, fourier_transfer, lateral,
    max_coupling_distance, mode_gains, population, selectivity, signal_power,
)


class PenaltyTests(unittest.TestCase):
    def test_selective_nullspace_is_exactly_modes_1_and_8(self):
        K, p = lateral('selective')
        t = 2 * np.pi * np.arange(N) / N
        for k in (1, 8):
            for v in (np.cos(k * t), np.sin(k * t)):
                self.assertLess(np.max(np.abs(K @ v)), 1e-9)
        others = [k for k in range(N) if k not in (1, 8, 24, 31)]
        self.assertAlmostEqual(min(p[k] for k in others), 1.0, places=12)

    def test_tau_matches_design_analysis(self):
        self.assertAlmostEqual(band_symbols((1, 8))[2], 169.2824418989257, places=9)

    def test_symbol_is_spectrum_of_spatial_penalty(self):
        for arm in ARMS:
            K, p = lateral(arm)
            self.assertTrue(np.allclose(K, circulant(p), atol=1e-8 * max(1.0, p.max())))

    def test_coupling_ranges(self):
        self.assertEqual(max_coupling_distance(lateral('selective')[0]), 4)
        self.assertEqual(max_coupling_distance(lateral('wrong_bands')[0]), 4)
        self.assertEqual(max_coupling_distance(lateral('ordinary_smoothing')[0]), 1)
        self.assertEqual(max_coupling_distance(lateral('no_lateral')[0]), 0)

    def test_ordinary_smoothing_is_cycle_laplacian(self):
        self.assertTrue(np.allclose(lateral('ordinary_smoothing')[1], EIG))


class DiskTests(unittest.TestCase):
    def test_energy_identity_and_gradient(self):
        rng = np.random.default_rng(1)
        for arm in ARMS:
            mod = SelectivityModule(SelectiveDisk(arm, 3))
            u, h = rng.standard_normal(mod.disk.n), rng.standard_normal(N)
            f = np.zeros(mod.disk.n)
            f[:N] = h
            quad = 0.5 * u @ mod.disk.hessian @ u - f @ u + 0.5 * h @ h
            self.assertAlmostEqual(mod.energy(u, h) / quad, 1.0, places=12)
            self.assertTrue(np.allclose(mod.grad(u, h), mod.disk.hessian @ u - f))

    def test_energy_zero_at_constant_state(self):
        mod = SelectivityModule(SelectiveDisk('no_lateral', 4))
        h = np.full(N, 0.7)
        self.assertAlmostEqual(mod.energy(np.full(mod.disk.n, 0.7), h), 0.0)

    def test_settle_matches_direct_solve(self):
        rng = np.random.default_rng(2)
        for arm in ARMS:
            mod = SelectivityModule(SelectiveDisk(arm, 2))
            h = rng.standard_normal(N)
            u, cert, _ = mod.settle(h)
            self.assertTrue(cert)
            self.assertLess(np.max(np.abs(u - mod.direct(h))), 1e-9)

    def test_spatial_transfer_matches_tridiagonal(self):
        for arm in ARMS:
            for L in (2, 4):
                mod = SelectivityModule(SelectiveDisk(arm, L))
                G = mod.transfer()
                rings = np.array([G[mod.disk.layer(l)] for l in range(1, L + 1)])
                self.assertLess(np.max(np.abs(rings - fourier_transfer(mod.disk.symbol, L))), 1e-9)


class GainTests(unittest.TestCase):
    def test_target_gains_are_algebraic(self):
        for L in (2, 4, 6):
            g = mode_gains(lateral('selective')[1], L)
            want = (L - np.arange(1, L + 1) + 1) / (L + 1)
            for k in (1, 8, 24, 31):
                self.assertTrue(np.allclose(g[:, k], want, atol=1e-12))

    def test_no_lateral_keeps_dc(self):
        g = mode_gains(np.zeros(N), 4)
        self.assertTrue(np.allclose(g[:, 0], 1.0))

    def test_population_predictions(self):
        p = population(lateral('selective')[1], 4, 8, 1.0, 0.1)
        self.assertAlmostEqual(p['suppression'], 0.0165802463, places=9)
        self.assertAlmostEqual(p['J'], 0.2652839410, places=9)
        self.assertAlmostEqual(p['gaussian_accuracy'], 0.9989094623, places=9)
        self.assertAlmostEqual(population(np.zeros(N), 4, 1, 1.0, 0.1)['J'], 1.7197696737, places=9)


class MeasureTests(unittest.TestCase):
    def test_signal_power_of_pure_classes(self):
        sig = np.array([signal(8, c) for c in range(4)])
        self.assertAlmostEqual(signal_power(sig), 0.5)
        self.assertAlmostEqual(signal_power(0.2 * sig), 0.02)

    def test_selectivity_floor_and_value(self):
        self.assertEqual(selectivity(1.0, 0.1, 0.5, 1e-13), (None, 'VsigL_below_floor'))
        self.assertEqual(selectivity(1.0, 0.1, None, 0.1), (None, 'Vsig1_unavailable'))
        J, reason = selectivity(2.0, 0.5, 0.5, 0.25)
        self.assertIsNone(reason)
        self.assertAlmostEqual(J, 0.5)

    def test_uniform_scaling_gives_unit_j(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((4, 50, N))
        within = lambda a: a.var(axis=1).mean()
        sig = np.array([signal(1, c) for c in range(4)])
        J, _ = selectivity(within(x), within(0.3 * x), signal_power(sig), signal_power(0.3 * sig))
        self.assertAlmostEqual(J, 1.0)

    def test_each_test_decision_ignores_other_test_drives(self):
        rng = np.random.default_rng(4)
        tr, te = rng.standard_normal((100, N)) + 0.3, rng.standard_normal((100, N))
        lab, ok = np.repeat(np.arange(4), 25), np.ones(100, bool)
        batch, _ = nearest_mean(tr, lab, ok, te, lab, ok)
        single = [nearest_mean(tr, lab, ok, te[[i]], lab[[i]], ok[[i]])[0] for i in range(100)]
        self.assertAlmostEqual(batch, np.mean(single))

    def test_nuisance_rms(self):
        z = nuisance_unit([20261030, 0, 0], (2, 4, 50), N)
        self.assertTrue(np.allclose(np.sqrt(np.mean(z ** 2, axis=-1)), 1.0))


if __name__ == '__main__':
    unittest.main()
