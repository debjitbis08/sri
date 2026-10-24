"""Correctness gates 1-7 for locality adaptation v2 (contract section 8)."""
import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from src.locality_adaptation import (BUDGETS, CONDITIONS, D, PHASES, REGIMES, base_stream,
    encode_local, fit_adaptive, fit_local, local_basis, make_world, nmse, noise_covariance,
    phase_floors, signal_covariance, signal_loss, split, views)
from src.locality_adaptation_v2 import (ARMS, FOLDS, ROOT_SEED, all_n_candidates, cv_scores,
    fit_arm, fit_cv_refit, fit_holdout_refit, folds, refit)


ROOT = Path(__file__).resolve().parents[1]
V1_SOURCES = ('src/locality_adaptation.py', 'experiments/locality_adaptation.py',
              'tests/test_locality_adaptation.py')


def pairs(world_index, regime='aligned'):
    world = make_world(0, world_index)
    return world, views(world, regime, base_stream(0, world_index))


class FrozenV1Gate(unittest.TestCase):
    def test_v1_sources_match_v1_manifest(self):
        manifest = json.loads((ROOT/'reports/locality-adaptation-v1-manifest.json').read_text())
        for name in V1_SOURCES:
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), manifest['hashes'][name], name)

    def test_fresh_seed_space(self):
        import src.locality_adaptation as v1
        from experiments.locality_adaptation_v2 import use_v2_seed
        self.assertNotEqual(ROOT_SEED, 2026102407)
        original = v1.ROOT_SEED
        try:
            use_v2_seed()
            stream = base_stream(1, 3)
            expected = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 1, 3, 3])).standard_normal((256, 4))
            np.testing.assert_array_equal(stream[0], expected)
        finally:
            v1.ROOT_SEED = original
        self.assertFalse(np.array_equal(base_stream(1, 3)[0], stream[0]))


class FoldGates(unittest.TestCase):
    def test_folds_partition_nested_and_match_v1_split(self):
        previous = None
        for n in BUDGETS:
            fold = folds(n)
            self.assertEqual([int((fold == f).sum()) for f in range(FOLDS)], [n//4]*4)
            np.testing.assert_array_equal(fold == 0, split(n)[1])
            if previous is not None:
                np.testing.assert_array_equal(fold[:len(previous)], previous)
            previous = fold

    def test_fitters_accept_only_observations(self):
        allowed = {'x', 'y', 'phase', 'b', 'meter', 'arm'}
        for fn in (cv_scores, refit, fit_cv_refit, fit_holdout_refit, fit_arm, all_n_candidates):
            self.assertLessEqual(set(inspect.signature(fn).parameters), allowed, fn.__name__)


class ScoreGates(unittest.TestCase):
    def test_cv_score_matches_independent_bilinear_sum(self):
        _, (h1, h2) = pairs(1, 'unknown_shift')
        for n in (4, 16, 64):
            x, y = h1[:n], h2[:n]
            fold = folds(n)
            direct = np.zeros(PHASES)
            for p in range(PHASES):
                for i in range(n):
                    train = fold != fold[i]
                    basis = local_basis(fit_local(x[train], y[train], p, 'cca'), p, D)
                    direct[p] += x[i]@basis@basis.T@y[i]
            direct /= n
            np.testing.assert_allclose(cv_scores(x, y), direct, rtol=1e-10, atol=1e-10)

    def test_fold_candidates_ignore_their_validation_fold(self):
        _, (h1, h2) = pairs(2)
        x, y = h1[:32].copy(), h2[:32].copy()
        fold = folds(32)
        before = [fit_local(x[fold != 1], y[fold != 1], p, 'cca') for p in range(PHASES)]
        x[fold == 1] = np.random.default_rng(0).standard_normal(((fold == 1).sum(), D))
        after = [fit_local(x[fold != 1], y[fold != 1], p, 'cca') for p in range(PHASES)]
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)


class RefitGates(unittest.TestCase):
    def test_refit_is_bitwise_direct_fit(self):
        for w_index, regime in ((3, 'aligned'), (4, 'half_shifted'), (5, 'dense')):
            _, (h1, h2) = pairs(w_index, regime)
            for n in (4, 16, 256):
                x, y = h1[:n], h2[:n]
                for fit in (fit_cv_refit, fit_holdout_refit):
                    chosen, w, _ = fit(x, y)
                    np.testing.assert_array_equal(w, fit_local(x, y, chosen, 'cca'))
                    if chosen == 0:
                        _, _, fixed = fit_arm('fixed_cca_all', x, y)
                        np.testing.assert_array_equal(w, fixed['weights'])
                    np.testing.assert_array_equal(w, all_n_candidates(x, y)[chosen])

    def test_aligned_phase0_selection_equals_fixed(self):
        _, (h1, h2) = pairs(6)
        chosen, w, _ = fit_cv_refit(h1, h2)
        self.assertEqual(chosen, 0)
        np.testing.assert_array_equal(w, fit_arm('fixed_cca_all', h1, h2)[2]['weights'])

    def test_holdout_arm_is_v1_adaptive(self):
        _, (h1, h2) = pairs(7, 'half_shifted')
        chosen, candidates, scores = fit_adaptive(h1[:32], h2[:32], 'cca')
        basis, phase, record = fit_arm('holdout_cca', h1[:32], h2[:32])
        self.assertEqual(phase, chosen)
        np.testing.assert_array_equal(record['weights'], candidates[chosen])
        self.assertEqual(record['scores'], scores)


class DecompositionGates(unittest.TestCase):
    def test_l_decomposition_and_oracle_reference(self):
        for w_index, regime in ((8, 'aligned'), (9, 'unknown_shift'), (10, 'dense')):
            world, (h1, h2) = pairs(w_index, regime)
            s = signal_covariance(world, regime)
            floors = phase_floors(s)
            structural = min(p for p, f in enumerate(floors) if f-min(floors) <= 1e-10)
            for n in BUDGETS:
                x, y = h1[:n], h2[:n]
                losses = [signal_loss(local_basis(w, p, D), s) for p, w in enumerate(all_n_candidates(x, y))]
                for arm in ('cv_refit_cca', 'holdout_refit_cca', 'oracle_structural_phase_cca_all'):
                    basis, phase, _ = fit_arm(arm, x, y, phase=structural)
                    selected = signal_loss(basis, s)
                    terms = (min(floors), min(losses)-min(floors), selected-min(losses))
                    self.assertTrue(all(t >= -1e-9 for t in terms), (arm, n, terms))
                    self.assertAlmostEqual(sum(terms), selected, delta=1e-12)
                if regime == 'aligned':
                    oracle = fit_arm('oracle_structural_phase_cca_all', x, y, phase=structural)[2]['weights']
                    np.testing.assert_array_equal(oracle, fit_arm('fixed_cca_all', x, y)[2]['weights'])


class PipelineGates(unittest.TestCase):
    def test_full_pipeline_finite_including_n4(self):
        world = make_world(0, 11)
        stream = base_stream(0, 11)
        for regime in REGIMES:
            h1, h2 = views(world, regime, stream)
            s = signal_covariance(world, regime)
            floors = phase_floors(s)
            structural = int(np.argmin(floors))
            for n in BUDGETS:
                for arm in ARMS:
                    basis, _, _ = fit_arm(arm, h1[:n], h2[:n], phase=structural)
                    proj = basis@basis.T
                    np.testing.assert_allclose(proj@proj, proj, atol=1e-9)
                    for c in CONDITIONS:
                        self.assertTrue(np.isfinite(nmse(basis, s, noise_covariance(world, regime, c))))

    def test_one_pair_per_fold_at_n4(self):
        _, (h1, h2) = pairs(12)
        self.assertEqual(sorted(folds(4)), [0, 1, 2, 3])
        self.assertTrue(np.all(np.isfinite(cv_scores(h1[:4], h2[:4]))))


if __name__ == '__main__':
    unittest.main()
