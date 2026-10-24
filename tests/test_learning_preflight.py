"""Checks of preflight equivalences, information boundaries and spectral conventions."""
import unittest

import numpy as np

from src.learning_preflight import (covariance_error_expectation, cross_covariance,
    factor_local, global_spectral, local_spectral, population_local, run_gates,
    subspace_loss, task_world)


class PreflightTests(unittest.TestCase):
    def test_spatial_parameter_and_gradient_equivalences(self):
        result = run_gates()
        self.assertLess(result['same_parameter_loss_gradient_max_error'], 1e-6)

    def test_eigenvectors_use_algebraic_not_absolute_order(self):
        h1 = np.sqrt(2)*np.diag([10., 1.])
        h2 = np.sqrt(2)*np.diag([-10., 1.])
        v = global_spectral(cross_covariance(h1, h2, 2), rank=1)
        np.testing.assert_allclose(v@v.T, np.diag([0., 1.]), atol=1e-12)

    def test_locality_floor_and_rotational_equivalence(self):
        u, rotations = task_world(3)
        self.assertLess(population_local(u)[0], 1e-12)
        for rotation in rotations[1:]:
            truth = rotation@u
            floor, _ = population_local(truth)
            self.assertGreater(floor, .05)
            self.assertAlmostEqual(subspace_loss(local_spectral(truth@truth.T), truth), floor)
            self.assertLess(subspace_loss(global_spectral(truth@truth.T), truth), 1e-12)

    def test_covariance_moments_reduce_to_wishart_and_declared_formula(self):
        u, _ = task_world(4)
        s = u@u.T
        self.assertAlmostEqual(covariance_error_expectation(s, 0., 10), 2.)
        for nu in (.25, 1.):
            self.assertAlmostEqual(covariance_error_expectation(s, nu, 16), (20+132*nu+528*nu**2)/16)
            blocks = sum(covariance_error_expectation(s[i:i+8, i:i+8], nu, 16) for i in range(0, 32, 8))
            self.assertAlmostEqual(blocks, (8+36*nu+144*nu**2)/16)

    def test_factorization_rejects_unmatched_support(self):
        u, rotations = task_world(1)
        with self.assertRaises(ValueError):
            factor_local((rotations[2]@u).T)

    def test_covariance_prefix_length_checked(self):
        a = np.zeros((4, 32))
        for pairs in (0, 5):
            with self.assertRaises(ValueError):
                cross_covariance(a, a, pairs)


if __name__ == '__main__':
    unittest.main()
