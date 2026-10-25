"""Independent checks for the contracted task and masked linear algebra."""
import unittest

import numpy as np

from src import abstraction_preflight as a
from src import single_module_plasticity as v1


class Task(unittest.TestCase):
    def test_hand_scenes_and_split(self):
        labels, angles, clean, test = a.scenes()
        np.testing.assert_array_equal(angles[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(angles[8], [0, 2, 4, 6])
        np.testing.assert_array_equal(angles[16], [0, 3, 6, 1])
        self.assertEqual(int(test.sum()), 8)
        self.assertEqual([int(test[labels[:, 0] == r].sum()) for r in (1, 2, 3)], [3, 3, 2])
        np.testing.assert_allclose(np.sum(clean**2, axis=2), 1)
        self.assertEqual(len(a.task_audit()['orientation_matched_pairs']), 96)

    def test_ties_are_explicit(self):
        self.assertEqual(a.tied_argmin(np.array([[1+5e-13, 1., 2.]]))[0], 0)
        self.assertEqual(a.tied_argmin(np.array([[1+2e-12, 1., 2.]]))[0], 1)

    def test_rendering_and_relation_oracle(self):
        _, angles, clean, _ = a.scenes()
        rng = np.random.default_rng(67)
        matrices = np.stack([v1.qf(rng.normal(size=(8, 8))) for _ in range(4)])
        x = a.render(clean, matrices)
        np.testing.assert_allclose(np.sum(x*x, axis=1), 4)
        np.testing.assert_array_equal(a.decode(x[:, :16], matrices, (0, 1)), angles[:, :2])
        np.testing.assert_array_equal(a.decode(x[:, 16:], matrices, (2, 3)), angles[:, 2:])

    def test_affine_floor_and_two_rule_control(self):
        labels, _, clean, _ = a.scenes()
        x = clean.reshape(24, 32)
        for rules in ((1, 2), (1, 3), (2, 3)):
            select = np.isin(labels[:, 0], rules)
            weights, _ = a.affine_fit(x[select, :16], x[select, 16:], a.Meter())
            pred = np.c_[x[select, :16], np.ones(select.sum())]@weights
            np.testing.assert_allclose(pred, x[select, 16:], atol=1e-12)
        weights, metadata = a.affine_fit(x[:, :16], x[:, 16:], a.Meter())
        loss = np.sum((np.c_[x[:, :16], np.ones(24)]@weights-x[:, 16:])**2)/48
        self.assertEqual(metadata['rank'], 5)
        self.assertAlmostEqual(loss, (8-2*np.sqrt(2))/21, places=12)


class Energy(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.p = a.random_params(self.rng)
        self.k = v1.block_curvatures(self.p)
        self.observed = self.rng.normal(size=32)*a.MASK

    def test_mask_only_changes_data_term(self):
        routes = .3+.7*self.rng.random((4, 4))
        h_full = v1.hessians(self.k, routes[None])[0]
        h_mask = a.masked_hessian(self.k, routes, a.MASK)
        np.testing.assert_allclose(h_full-h_mask, np.diag(np.r_[.1*(1-a.MASK), np.zeros(48)]), atol=1e-15)
        vector = self.rng.normal(size=80)
        self.assertAlmostEqual(a.masked_energy(vector, self.observed, self.p, routes, np.ones(32)),
                               v1.energy(vector, self.observed, self.p, routes), places=12)

    def test_energy_polarization_hessian(self):
        routes = .3+.7*self.rng.random((4, 4))
        hessian = a.masked_hessian(self.k, routes, a.MASK)
        zero = np.zeros(80)
        energy = lambda v: a.masked_energy(v, self.observed, self.p, routes, a.MASK)
        for _ in range(12):
            u, v = self.rng.normal(size=(2, 80))
            self.assertAlmostEqual(energy(u+v)-energy(u)-energy(v)+energy(zero), u@hessian@v, places=10)

    def test_original_equilibrium_matches_frozen_implementation(self):
        expected, _ = v1.Module(v1.copy_params(self.p)).infer(self.observed[None])
        pilot, _ = a.solve_checked(self.p, self.k, np.ones((4, 4)), np.ones(32), self.observed, a.Meter())
        gates = v1.gates(self.p, pilot[None])
        routes, _ = v1.routes(gates, 'matched')
        result, _ = a.solve_checked(self.p, self.k, routes[0], np.ones(32), self.observed, a.Meter())
        np.testing.assert_allclose(result[:32], expected[0], atol=1e-12)

    def test_masked_equilibrium_and_hidden_input_ignored(self):
        state, checks = a.solve_checked(self.p, self.k, np.ones((4, 4)), a.MASK, self.observed, a.Meter())
        changed = self.observed.copy()
        changed[16:] = self.rng.normal(size=16)*100
        alternate, _ = a.solve_checked(self.p, self.k, np.ones((4, 4)), a.MASK, changed, a.Meter())
        np.testing.assert_allclose(state, alternate, atol=1e-12)
        self.assertLess(checks['relative_residual'], 1e-10)

    def test_odd_symmetry_of_matched_response(self):
        results = []
        for sign in (1, -1):
            pilot, _ = a.solve_checked(self.p, self.k, np.ones((4, 4)), a.MASK, sign*self.observed, a.Meter())
            routes, _ = v1.routes(v1.gates(self.p, pilot[None]), 'matched')
            state, _ = a.solve_checked(self.p, self.k, routes[0], a.MASK, sign*self.observed, a.Meter())
            results.append(state)
        np.testing.assert_allclose(results[0], -results[1], atol=1e-12)

    def test_one_observed_coordinate_is_positive_definite(self):
        mask = np.zeros(32)
        mask[7] = 1
        h = a.masked_hessian(self.k, np.ones((4, 4)), mask)
        self.assertGreater(np.linalg.eigvalsh(h)[0], 0)


if __name__ == '__main__':
    unittest.main()
