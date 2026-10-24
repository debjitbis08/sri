"""Correctness gates for representation reuse v1 (contract section 8)."""
import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

import src.locality_adaptation as v1
from src.locality_adaptation import (B, D, Meter, ScoringFailure, fit_global, fit_local, groups,
    local_basis, nmse, orthonormal_basis, phase_floors, signal_loss)
from src.locality_adaptation_v2 import fit_cv_refit, folds
import src.representation_reuse as rr
from src.representation_reuse import (ANGLES, ARMS, CHECKPOINTS, CONDITIONS, FRESH_ARMS, ROOT_SEED,
    SOURCE_ARMS, STREAM, acquire_global, acquire_local, add_pair_full, add_pair_local, block_sums,
    dense_basis, drifted_signal, error_map_nmse, full_sums, make_learner, make_world, metrics,
    noise_covariances, observe, principal_angle_cosines, signal_covariance, solve_local, stream)
from experiments.representation_reuse import (FROZEN, V1_SOURCES, V2_SOURCES, charged_counts,
    run_world)


ROOT = Path(__file__).resolve().parents[1]


def fixture(w=0, stage=0):
    world = make_world(stage, w)
    xs, ys = observe(world, world['u'], stream(stage, w, 3))
    return world, xs, ys


def run_branch(learner, arm, h1, h2, m):
    for i in range(m):
        if arm not in FROZEN:
            learner.observe(h1[i], h2[i])
        if i+1 in CHECKPOINTS and arm not in FROZEN:
            learner.checkpoint()
    return learner


class FrozenDependencyGate(unittest.TestCase):
    def test_v1_v2_sources_match_their_manifests(self):
        for manifest_name, names in (('locality-adaptation-v1-manifest.json', V1_SOURCES),
                                     ('locality-adaptation-v2-manifest.json', V2_SOURCES)):
            manifest = json.loads((ROOT/'reports'/manifest_name).read_text())
            for name in names:
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                                 manifest['hashes'][name], name)


class WorldGates(unittest.TestCase):
    def test_orthonormal_supports_and_principal_angles(self):
        for w in range(4):
            world = make_world(0, w)
            u, v, wd = world['u'], world['v'], world['w']
            full = np.hstack([u, v, wd])
            np.testing.assert_allclose(full.T@full, np.eye(3*u.shape[1]), atol=1e-12)
            r, k = world['rotation'], u.shape[1]
            for j, g in enumerate(groups(world['phase'], D, B)):
                outside = np.setdiff1d(np.arange(D), g)
                for col in (u, v, wd):
                    self.assertTrue(np.all((r@col)[outside, j] == 0))
            for theta in ANGLES:
                ut = drifted_signal(world, theta)
                np.testing.assert_allclose(ut.T@ut, np.eye(k), atol=1e-12)
                np.testing.assert_allclose(ut.T@v, 0, atol=1e-12)
                np.testing.assert_allclose(principal_angle_cosines(u, ut), np.cos(np.deg2rad(theta)), atol=1e-12)
                floors = phase_floors(signal_covariance(world, ut))
                self.assertLess(abs(floors[world['phase']]), 1e-9)
            np.testing.assert_array_equal(drifted_signal(world, 0), u)
            np.testing.assert_allclose(drifted_signal(world, 90).T@u, 0, atol=1e-15)

    def test_zero_drift_residual_stops(self):
        u, v = rr.templates(0, 0)
        original = rr.rng_for
        try:
            class Zero:
                def standard_normal(self, shape):
                    return np.zeros(shape)
            rr.rng_for = lambda *args, **kw: Zero()
            with self.assertRaises(ScoringFailure):
                rr.drift_directions(u, v, 0, 0)
        finally:
            rr.rng_for = original


class RandomnessGates(unittest.TestCase):
    def direct(self, stage, w, part):
        return np.random.default_rng(np.random.SeedSequence([ROOT_SEED, stage, w, part]))

    def test_templates_match_v1_recipe_under_scoped_root(self):
        original = v1.ROOT_SEED
        try:
            v1.ROOT_SEED = ROOT_SEED
            reference = v1.make_world(1, 5)
        finally:
            v1.ROOT_SEED = original
        self.assertEqual(v1.ROOT_SEED, 2026102407)
        world = make_world(1, 5)
        np.testing.assert_array_equal(world['u'], reference['u'])
        np.testing.assert_array_equal(world['v'], reference['v'])

    def test_direct_draws_and_namespaces(self):
        world = make_world(1, 3)
        self.assertEqual(world['phase'], int(self.direct(1, 3, 2).integers(0, 8)))
        source, target = stream(1, 3, 3), stream(1, 3, 4)
        for part, draws in ((3, source), (4, target)):
            rng = self.direct(1, 3, part)
            for a, shape in zip(draws, ((256, 4), (256, 4), (256, 4), (256, 32), (256, 32))):
                np.testing.assert_array_equal(a, rng.standard_normal(shape))
        self.assertFalse(np.array_equal(source[0], target[0]))
        a = self.direct(1, 3, 5).standard_normal((4, 8))
        u0, v0 = world['u'][:8, 0], world['v'][:8, 0]
        r = a[0]-u0*(u0@a[0])-v0*(v0@a[0])
        np.testing.assert_array_equal(world['w'][:8, 0], r/np.linalg.norm(r))
        self.assertNotEqual(ROOT_SEED, v1.ROOT_SEED)
        v1.base_stream(1, 3)
        np.testing.assert_array_equal(stream(1, 3, 3)[0], source[0])

    def test_target_draws_shared_across_angles(self):
        world = make_world(0, 1)
        draws = stream(0, 1, 4)
        h0 = observe(world, drifted_signal(world, 0), draws)
        h90 = observe(world, drifted_signal(world, 90), draws)
        r = world['rotation']
        np.testing.assert_allclose((h0[0]-h90[0])@r, draws[0]@(world['u']-drifted_signal(world, 90)).T, atol=1e-12)

    def test_every_cv_pair_validated_once(self):
        for n in (*rr.SOURCE_BUDGETS, *CHECKPOINTS):
            counts = np.bincount(folds(n), minlength=4)
            self.assertEqual(counts.sum(), n)
            self.assertTrue(np.all(counts == n//4))

    def test_source_selection_matches_v2(self):
        for w in range(3):
            _, xs, ys = fixture(w)
            for s in rr.SOURCE_BUDGETS:
                ours = acquire_local(xs[:s], ys[:s])
                phase, weights, scores = fit_cv_refit(xs[:s], ys[:s])
                self.assertEqual(ours['phase'], phase)
                np.testing.assert_array_equal(ours['weights'], weights)
                self.assertEqual(ours['scores'], list(scores))
                np.testing.assert_array_equal(acquire_global(xs[:s], ys[:s])['basis'],
                                              orthonormal_basis(fit_global(xs[:s], ys[:s], 'cca', 4)))


class EvaluatorGates(unittest.TestCase):
    def test_oracle_controls(self):
        world = make_world(0, 2)
        noises = noise_covariances(world)
        r = world['rotation']
        for theta in ANGLES:
            s = signal_covariance(world, drifted_signal(world, theta))
            src = metrics(r@world['u'], s, noises)
            sin2 = np.sin(np.deg2rad(theta))**2
            self.assertAlmostEqual(src['signal_loss'], sin2, places=12)
            for c, iso in (('in_distribution', .25), ('low_noise', .01), ('texture_16', .25)):
                self.assertAlmostEqual(src['nmse'][c], sin2+iso, places=12)
                self.assertAlmostEqual(metrics(r@drifted_signal(world, theta), s, noises)['nmse'][c], iso, places=12)
            self.assertAlmostEqual(float(np.trace(noises['in_distribution'])/4), 6., places=12)

    def test_error_map_agrees_with_trace(self):
        world = make_world(0, 2)
        noises = noise_covariances(world)
        rng = np.random.default_rng(7)
        for theta in (0, 30, 90):
            s = signal_covariance(world, drifted_signal(world, theta))
            for _ in range(5):
                basis = np.linalg.qr(rng.standard_normal((D, 4)))[0]
                for n in noises.values():
                    self.assertLess(abs(error_map_nmse(basis, s, n)-nmse(basis, s, n)), 1e-10)


class StatisticGates(unittest.TestCase):
    def setUp(self):
        self.world, self.xs, self.ys = fixture(4)
        self.h1, self.h2 = observe(self.world, drifted_signal(self.world, 30), stream(0, 4, 4))

    def test_incremental_sums_match_batch(self):
        s, m = 64, 32
        local, glob = acquire_local(self.xs[:s], self.ys[:s]), acquire_global(self.xs[:s], self.ys[:s])
        c, a = local['c'].copy(), local['a'].copy()
        cf, af = glob['c'].copy(), glob['a'].copy()
        for i in range(m):
            add_pair_local(c, a, self.h1[i], self.h2[i], local['phase'])
            add_pair_full(cf, af, self.h1[i], self.h2[i])
        x, y = np.vstack([self.xs[:s], self.h1[:m]]), np.vstack([self.ys[:s], self.h2[:m]])
        cb, ab = block_sums(x, y, local['phase'])
        np.testing.assert_allclose(c, cb, atol=1e-10)
        np.testing.assert_allclose(a, ab, atol=1e-10)
        cb, ab = full_sums(x, y)
        np.testing.assert_allclose(cf, cb, atol=1e-10)
        np.testing.assert_allclose(af, ab, atol=1e-10)

    def test_pooled_and_reuse_match_direct_fits(self):
        s = 64
        local, glob = acquire_local(self.xs[:s], self.ys[:s]), acquire_global(self.xs[:s], self.ys[:s])
        s_t = signal_covariance(self.world, drifted_signal(self.world, 30))
        n = noise_covariances(self.world)['in_distribution']
        for m in (4, 16, 64):
            x, y = np.vstack([self.xs[:s], self.h1[:m]]), np.vstack([self.ys[:s], self.h2[:m]])
            cases = dict(
                pooled_local=local_basis(fit_local(x, y, local['phase'], 'cca'), local['phase'], D),
                pooled_global=orthonormal_basis(fit_global(x, y, 'cca', 4)),
                phase_reuse_target=local_basis(fit_local(self.h1[:m], self.h2[:m], local['phase'], 'cca'),
                                               local['phase'], D))
            for arm, reference in cases.items():
                learner = make_learner(arm, local, glob, D)
                for i in range(m):
                    learner.observe(self.h1[i], self.h2[i])
                learner.checkpoint()
                basis = dense_basis(learner.deploy(), D)
                np.testing.assert_allclose(basis@basis.T, reference@reference.T, atol=1e-9)
                self.assertLess(abs(nmse(basis, s_t, n)-nmse(reference, s_t, n)), 1e-9)
                if arm.startswith('pooled'):
                    self.assertEqual(learner.total() if arm == 'pooled_global' else learner.source_count+learner.count,
                                     s+m)

    def test_pooled_zero_deploys_source_fit(self):
        local, glob = acquire_local(self.xs, self.ys), acquire_global(self.xs, self.ys)
        np.testing.assert_array_equal(make_learner('pooled_local', local, glob, D).deploy()[2], local['weights'])
        np.testing.assert_array_equal(make_learner('pooled_global', local, glob, D).deploy()[2], glob['basis'])


class IsolationGates(unittest.TestCase):
    def setUp(self):
        self.world, self.xs, self.ys = fixture(5)
        self.h1, self.h2 = observe(self.world, drifted_signal(self.world, 60), stream(0, 5, 4))
        self.local, self.glob = acquire_local(self.xs, self.ys), acquire_global(self.xs, self.ys)

    def test_branches_do_not_mutate_source_checkpoint(self):
        snapshot = {k: np.copy(v) for k, v in {**self.local, **{f'g_{k}': v for k, v in self.glob.items()}}.items()
                    if isinstance(v, np.ndarray)}
        for arm in SOURCE_ARMS:
            run_branch(make_learner(arm, self.local, self.glob, D), arm, self.h1, self.h2, STREAM)
        for k, v in snapshot.items():
            current = self.glob[k[2:]] if k.startswith('g_') else self.local[k]
            np.testing.assert_array_equal(current, v)

    def test_frozen_never_change_and_fresh_ignore_source(self):
        for arm in FROZEN:
            learner = make_learner(arm, self.local, self.glob, D)
            before = learner.deploy()[2].copy()
            run_branch(learner, arm, self.h1, self.h2, STREAM)
            np.testing.assert_array_equal(learner.deploy()[2], before)
        other_local, other_glob = acquire_local(self.xs[:16], self.ys[:16]), acquire_global(self.xs[:16], self.ys[:16])
        for arm in FRESH_ARMS:
            a = run_branch(make_learner(arm, self.local, self.glob, D), arm, self.h1, self.h2, 64).deploy()
            b = run_branch(make_learner(arm, other_local, other_glob, D), arm, self.h1, self.h2, 64).deploy()
            np.testing.assert_array_equal(a[2], b[2])

    def test_no_world_truth_and_no_raw_source_rows(self):
        for fn in (acquire_local, acquire_global, solve_local, rr.solve_global):
            self.assertNotIn('world', inspect.signature(fn).parameters)
        for arm in ARMS:
            learner = make_learner(arm, self.local, self.glob, D)
            self.assertNotIn('world', vars(learner))
            for value in vars(learner).values():
                if isinstance(value, np.ndarray) and value.shape in {(s, D) for s in rr.SOURCE_BUDGETS}:
                    self.assertTrue(arm == 'fresh_local_cv' and not np.any(value), arm)

    def test_evaluator_truth_cannot_change_fits(self):
        altered = dict(self.world, v=np.roll(self.world['v'], 1, axis=0))
        rows = []
        for world in (self.world, altered):
            learner = run_branch(make_learner('phase_reuse_target', self.local, self.glob, D),
                                 'phase_reuse_target', self.h1, self.h2, 32)
            rows.append(learner.deploy()[2])
            noise_covariances(world)
        np.testing.assert_array_equal(*rows)


class ArmGates(unittest.TestCase):
    def test_development_world_rows_finite_and_nulls_explicit(self):
        meters = {name: Meter() for name in ('source_local', 'source_global', *ARMS)}
        info, rows = run_world(0, 6, meters, Meter())
        for r in rows:
            if r['defined']:
                values = [r['L'], r['source_L'], *r['E'].values(), *r['source_E'].values()]
                self.assertTrue(np.all(np.isfinite(values)))
            else:
                self.assertEqual((r['target_budget'], r['reason']), (0, 'no target pairs'))
                self.assertIn(r['arm'], ('phase_reuse_target', *FRESH_ARMS))
        self.assertEqual(len(rows), len(ANGLES)*(3*len(SOURCE_ARMS)+len(FRESH_ARMS))*8)

    def test_nonpositive_covariance_fails(self):
        with self.assertRaises(ScoringFailure):
            solve_local(np.zeros((4, 8, 8)), np.zeros((4, 8, 8)), 1, Meter())

    def test_projector_check_rejects_rank_deficient(self):
        with self.assertRaises(ScoringFailure):
            rr.check_projector(np.zeros((D, 4)), 4)


class CostGates(unittest.TestCase):
    def test_counters_match_closed_form(self):
        world, xs, ys = fixture(7)
        h1, h2 = observe(world, drifted_signal(world, 15), stream(0, 7, 4))
        s, m = 16, 32
        for arm in ARMS:
            meter = Meter()
            local = acquire_local(xs[:s], ys[:s], meter=meter) if arm in rr.LOCAL_ACQUISITION_ARMS else None
            glob = acquire_global(xs[:s], ys[:s], meter=meter) if arm in rr.GLOBAL_ACQUISITION_ARMS else None
            run_branch(make_learner(arm, local or acquire_local(xs[:s], ys[:s]),
                                    glob or acquire_global(xs[:s], ys[:s]), D, meter), arm, h1, h2, m)
            expected, got = charged_counts(arm, s, m), meter.counts
            fits = expected.get('local_fits', 0)
            self.assertEqual(got['fits'], fits+expected.get('global_fits', 0), arm)
            self.assertEqual(got['candidate_scores'], expected.get('candidate_scores', 0), arm)
            self.assertEqual(got['fit_row_exposures'], expected.get('fit_row_exposures', 0), arm)
            self.assertEqual(got['validation_row_exposures'], expected.get('validation_row_exposures', 0), arm)
            self.assertEqual(got['target_pairs'], expected['target_pairs'], arm)
            self.assertEqual(got['source_pairs'], expected.get('source_pairs', 0), arm)
            solves = got['local_solves']+got['global_solves']-expected.get('local_solves', 0)-expected.get('global_solves', 0)
            self.assertEqual(solves, expected.get('checkpoint_solves', 0), arm)
            self.assertEqual(got['cv_runs'], expected.get('cv_runs', 0), arm)

    def test_source_charged_once_per_branch(self):
        for arm in SOURCE_ARMS:
            self.assertEqual(charged_counts(arm, 256, 256)['source_pairs'], 256)
            self.assertEqual(charged_counts(arm, 256, 4)['source_pairs'], 256)
        self.assertNotIn('source_pairs', charged_counts('fresh_local_cv', 256, 256))


if __name__ == '__main__':
    unittest.main()
