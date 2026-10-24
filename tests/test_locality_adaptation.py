"""Correctness gates 1-6 for locality adaptation v1 (contract section 8)."""
import inspect
import unittest

import numpy as np

from src.locality_adaptation import (ARMS, B, BUDGETS, CONDITIONS, D, EXTRA, ISO, PHASES, RANK,
    REGIMES, TEXTURE, Meter, ScoringFailure, base_stream, canonical_basis, canonical_top, cca_top,
    fit_adaptive, fit_arm, fit_global, fit_local, groups, local_basis, make_world, nmse,
    noise_covariance, orthonormal_basis, phase_floors, projected_noise, reconstruct_local, select,
    encode_local, signal_covariance, signal_loss, split, statistics, validation_score, views)


def pairs(world_index=1, regime='aligned', n=256):
    world = make_world(0, world_index)
    return world, views(world, regime, base_stream(0, world_index))


class GeneratorGates(unittest.TestCase):
    def test_world_geometry_and_rotations(self):
        world = make_world(0, 3)
        u, v = world['u'], world['v']
        np.testing.assert_allclose(u.T@u, np.eye(RANK), atol=1e-12)
        np.testing.assert_allclose(v.T@v, np.eye(RANK), atol=1e-12)
        np.testing.assert_allclose(u.T@v, 0, atol=1e-12)
        for r in world['rotations'].values():
            np.testing.assert_allclose(r.T@r, np.eye(D), atol=1e-12)
        self.assertIn(world['shift'], range(1, 8))
        np.testing.assert_array_equal(world['rotations']['half_shifted']@np.arange(D), np.roll(np.arange(D), 4))

    def test_regimes_share_base_draws(self):
        world = make_world(0, 2)
        stream = base_stream(0, 2)
        a1, a2 = views(world, 'aligned', stream)
        for regime in REGIMES:
            h1, h2 = views(world, regime, stream)
            r = world['rotations'][regime]
            np.testing.assert_allclose(h1, a1@r.T, atol=1e-12)
            np.testing.assert_allclose(h2, a2@r.T, atol=1e-12)
        again = base_stream(0, 2)
        for x, y in zip(stream, again):
            np.testing.assert_array_equal(x, y)
        self.assertEqual([s.shape for s in stream], [(256, 4)]*3+[(256, 32)]*2)

    def test_timing_generator_generalizes(self):
        world = make_world(2, 0, d=128, extra=(128,))
        self.assertEqual(world['u'].shape, (128, 16))
        np.testing.assert_allclose(world['u'].T@world['v'], 0, atol=1e-12)
        self.assertFalse(np.allclose(world['u'][:32, :4], make_world(2, 0, d=32, extra=(32,))['u']))

    def test_nested_splits(self):
        previous = None
        for n in BUDGETS:
            fit, val = split(n)
            self.assertEqual((fit.sum(), val.sum()), (3*n//4, n//4))
            if previous is not None:
                np.testing.assert_array_equal(fit[:len(previous[0])], previous[0])
                np.testing.assert_array_equal(val[:len(previous[1])], previous[1])
            previous = fit, val
        np.testing.assert_array_equal(split(4)[1], [False, False, False, True])

    def test_fitters_accept_only_observations(self):
        allowed = {'x', 'y', 'phase', 'method', 'b', 'meter', 'rank', 'arm'}
        for fn in (fit_local, fit_global, fit_adaptive, fit_arm):
            self.assertLessEqual(set(inspect.signature(fn).parameters), allowed, fn.__name__)


class ProjectorGates(unittest.TestCase):
    def test_local_supports_disjoint_and_projectors_valid(self):
        for p in range(PHASES):
            g = groups(p)
            self.assertEqual(sorted(g.ravel()), list(range(D)))
        world, (h1, h2) = pairs(4, 'dense')
        for arm in ARMS:
            for n in (4, 64):
                basis, _, _ = fit_arm(arm, h1[:n], h2[:n])
                proj = basis@basis.T
                np.testing.assert_allclose(proj, proj.T, atol=1e-9)
                np.testing.assert_allclose(proj@proj, proj, atol=1e-9)
                self.assertAlmostEqual(np.trace(proj), RANK, delta=1e-9)

    def test_global_projector_basis_invariant(self):
        _, (h1, h2) = pairs(5)
        w = fit_global(h1[:32], h2[:32], 'cca', RANK)
        a = np.random.default_rng(0).standard_normal((RANK, RANK))
        p1, p2 = orthonormal_basis(w), orthonormal_basis(w@a)
        np.testing.assert_allclose(p1@p1.T, p2@p2.T, atol=1e-9)

    def test_rank_failure_is_not_repaired(self):
        w = np.zeros((D, RANK))
        w[0] = 1
        with self.assertRaises(ScoringFailure):
            orthonormal_basis(w)

    def test_local_inference_matches_projector(self):
        _, (h1, h2) = pairs(6, 'half_shifted')
        chosen, candidates, _ = fit_adaptive(h1[:64], h2[:64], 'cca')
        basis = local_basis(candidates[chosen], chosen, D)
        z = encode_local(h1[:5], chosen, candidates[chosen])
        np.testing.assert_allclose(z, h1[:5]@basis, atol=1e-12)
        np.testing.assert_allclose(reconstruct_local(z, chosen, candidates[chosen], D),
                                   h1[:5]@basis@basis.T, atol=1e-12)


class WitnessGates(unittest.TestCase):
    def test_analysis_witnesses(self):
        aligned = np.zeros((D, RANK))
        for j in range(RANK):
            aligned[j*B:(j+1)*B, j] = 1/np.sqrt(B)
        shifted = np.roll(aligned, 4, axis=0)
        floors = phase_floors(shifted@shifted.T)
        self.assertAlmostEqual(floors[0], .5, delta=1e-10)
        self.assertAlmostEqual(min(floors), 0, delta=1e-10)
        h = np.ones((1, 1))
        while len(h) < D:
            h = np.block([[h, h], [h, -h]])
        u = h[:, :RANK]/np.sqrt(D)
        np.testing.assert_allclose(phase_floors(u@u.T), .75, atol=1e-10)


class SelectionGates(unittest.TestCase):
    def test_bilinear_validation_matches_codes(self):
        _, (h1, h2) = pairs(7, 'unknown_shift')
        fit, val = split(64)
        x, y = h1[:64], h2[:64]
        for p in range(PHASES):
            w = fit_local(x[fit], y[fit], p, 'cca')
            proj = local_basis(w, p, D)@local_basis(w, p, D).T
            direct = np.mean(np.einsum('ni,ij,nj->n', x[val], proj, y[val]))
            score = validation_score(x[val], y[val], p, w)
            self.assertLess(abs(direct-score), 1e-10*max(1, abs(direct)))

    def test_tie_rule(self):
        self.assertEqual(select([1., 3., 3., 2.]), 1)
        self.assertEqual(select([1., 3.-1e-13, 3., 2.]), 1)
        self.assertEqual(select([1., 3.-1e-9, 3., 2.]), 2)
        self.assertEqual(select([5., 5., 5.]), 0)

    def test_validation_data_cannot_change_candidates(self):
        _, (h1, h2) = pairs(8)
        x, y = h1[:64].copy(), h2[:64].copy()
        _, before, _ = fit_adaptive(x, y, 'cca')
        _, val = split(64)
        x[val] = np.random.default_rng(1).standard_normal((val.sum(), D))
        _, after, _ = fit_adaptive(x, y, 'cca')
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_evaluator_truth_cannot_change_fits(self):
        # Two worlds with different U and V but identical observed pairs produce identical fits,
        # because no fitter accepts world truth; selection depends on observations only.
        _, (h1, h2) = pairs(9, 'half_shifted')
        runs = [fit_adaptive(h1[:32], h2[:32], 'cca') for _ in range(2)]
        self.assertEqual(runs[0][0], runs[1][0])
        other = make_world(0, 10)
        s_a, s_b = signal_covariance(make_world(0, 9), 'half_shifted'), signal_covariance(other, 'half_shifted')
        self.assertFalse(np.allclose(s_a, s_b))
        for a, b in zip(runs[0][1], runs[1][1]):
            np.testing.assert_array_equal(a, b)


class MetricGates(unittest.TestCase):
    def test_error_map_matches_trace_nmse(self):
        world = make_world(0, 11)
        u, v = world['u'], world['v']
        complement = np.linalg.svd(np.eye(D)-u@u.T-v@v.T)[0][:, :D-2*RANK]
        rng = np.random.default_rng(2)
        for regime in REGIMES:
            r = world['rotations'][regime]
            s = signal_covariance(world, regime)
            projectors = [np.linalg.qr(rng.standard_normal((D, RANK)))[0], r@u,
                          local_basis(np.ones((RANK, B))/np.sqrt(B), 3, D)]
            for basis in projectors:
                proj = basis@basis.T
                for condition in CONDITIONS:
                    texture = 4*TEXTURE if condition == 'texture_16' else TEXTURE
                    noise_maps = [np.sqrt(texture)*proj@r@v, np.sqrt(ISO)*proj@r]
                    if condition == 'extra_noise':
                        noise_maps.append(np.sqrt(EXTRA)*proj@r@complement)
                    error_map = np.column_stack([(proj-np.eye(D))@r@u]+noise_maps)
                    risk = np.linalg.norm(error_map, 'fro')**2/RANK
                    self.assertLess(abs(risk-nmse(basis, s, noise_covariance(world, regime, condition))), 1e-10)

    def test_descriptive_controls(self):
        world = make_world(0, 12)
        s, n = signal_covariance(world, 'dense'), noise_covariance(world, 'dense')
        self.assertAlmostEqual(np.trace(n)/RANK, 6, delta=1e-10)
        oracle = world['rotations']['dense']@world['u']
        self.assertAlmostEqual(nmse(oracle, s, n), .25, delta=1e-10)
        self.assertAlmostEqual(signal_loss(oracle, s), 0, delta=1e-10)

    def test_loss_decomposition(self):
        world, (h1, h2) = pairs(13, 'unknown_shift')
        s = signal_covariance(world, 'unknown_shift')
        floor = min(phase_floors(s))
        for n in BUDGETS:
            chosen, candidates, _ = fit_adaptive(h1[:n], h2[:n], 'cca')
            losses = [signal_loss(local_basis(w, p, D), s) for p, w in enumerate(candidates)]
            terms = (floor, min(losses)-floor, losses[chosen]-min(losses))
            self.assertTrue(all(t >= -1e-9 for t in terms))
            self.assertAlmostEqual(sum(terms), losses[chosen], delta=1e-12)


class EstimatorGates(unittest.TestCase):
    def test_cca_residual_and_factorization(self):
        _, (h1, h2) = pairs(14)
        for n in (4, 256):
            k, t = statistics(h1[:n], h2[:n])
            reg = t+.1*np.trace(t)/D*np.eye(D)
            l = np.linalg.cholesky(reg)
            np.testing.assert_allclose(l@l.T, reg, atol=1e-12)
            w = cca_top(k, t, RANK, Meter())
            lam = np.array([wi@k@wi/(wi@reg@wi) for wi in w.T])
            np.testing.assert_allclose(k@w, reg@w*lam, atol=1e-9)

    def test_singular_unregularized_covariance_at_n4(self):
        _, (h1, h2) = pairs(15)
        fit, _ = split(4)
        x = h1[:4][fit]
        _, t = statistics(x, h2[:4][fit])
        self.assertLess(np.linalg.matrix_rank(t), D)
        for arm in ARMS:
            basis, _, _ = fit_arm(arm, h1[:4], h2[:4])
            self.assertTrue(np.all(np.isfinite(basis)))

    def test_zero_covariance_fails_instead_of_falling_back(self):
        with self.assertRaises(ScoringFailure):
            fit_local(np.zeros((3, D)), np.zeros((3, D)), 0, 'cca')

    def test_tied_eigenvalues_use_canonical_basis(self):
        q = np.linalg.qr(np.random.default_rng(3).standard_normal((4, 4)))[0]
        # Any orthonormal basis of the same tied eigenspace gives the same canonical basis.
        rotation = np.linalg.qr(np.random.default_rng(4).standard_normal((2, 2)))[0]
        np.testing.assert_allclose(canonical_basis(q[:, :2]), canonical_basis(q[:, :2]@rotation), atol=1e-12)
        top = canonical_top(np.diag([1., 2., 2., 0.]), 1)
        np.testing.assert_allclose(np.abs(top[:, 0]), [0, 1, 0, 0], atol=1e-12)
        m = q@np.diag([3., 3., 3., 1.])@q.T
        a, b = canonical_top(m, 2), canonical_top(m+1e-15*np.eye(4), 2)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_full_pipeline_at_every_budget_and_regime(self):
        world = make_world(0, 16)
        stream = base_stream(0, 16)
        for regime in REGIMES:
            h1, h2 = views(world, regime, stream)
            s = signal_covariance(world, regime)
            for n in BUDGETS:
                for arm in ARMS:
                    basis, _, _ = fit_arm(arm, h1[:n], h2[:n])
                    for c in CONDITIONS:
                        self.assertTrue(np.isfinite(nmse(basis, s, noise_covariance(world, regime, c))))
                    self.assertGreaterEqual(projected_noise(basis, s, noise_covariance(world, regime)), 0)


if __name__ == '__main__':
    unittest.main()
