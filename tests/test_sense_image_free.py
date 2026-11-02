import unittest

import numpy as np

from experiments.sense_image_free import FLOOR, TAU, Store, fit_map, key_drift, median_angle, merge_keeps_distinctions, symbol_trials, N, R, N_TARGETS, TARGET_TRAIN


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Maps(unittest.TestCase):
    def test_a_map_recovers_a_change_of_coordinates_on_points_it_did_not_fit(self):
        rng = np.random.default_rng(0)
        Q = np.linalg.qr(rng.normal(size=(8, 8)))[0]
        X = rng.normal(size=(600, 8))
        fn = fit_map(X[:500], X[:500] @ Q, lam=1e-6)
        self.assertLess(np.abs(fn(X[500:]) - X[500:] @ Q).max(), 1e-2)

    def test_no_change_stays_no_change(self):
        X = np.random.default_rng(1).normal(size=(100, 4))
        self.assertLess(np.abs(fit_map(X, X)(X) - X).max(), 1e-4)

    def test_key_drift_reports_what_the_map_leaves(self):
        rng = np.random.default_rng(2)
        Q = np.linalg.qr(rng.normal(size=(16, 16)))[0]
        X = unit(rng.normal(size=(800, 16)))
        d = key_drift(X, X @ Q)
        self.assertGreater(d['angle'], 1.)
        self.assertLess(d['residual'], .1)

    def test_median_angle(self):
        self.assertAlmostEqual(median_angle(np.array([[1., 0.]]), np.array([[0., 1.]])), np.pi / 2, places=5)


class Radius(unittest.TestCase):
    def store(self, zs, vs):
        m = Store(8, 1, TAU, 2, 2)
        for z, v in zip(zs, vs):
            m.write(np.array(z, np.float32), np.array(v, np.float32))
        return m

    def test_a_new_unit_reads_as_plain_memory(self):
        m = self.store([[1., 0.]], [[1., 0.]])
        self.assertAlmostEqual(float(m.thr[0]), TAU, places=5)

    def test_a_merged_ball_covers_both_members_balls(self):
        a, b = unit(np.array([1., .3])), unit(np.array([1., -.3]))
        m = self.store([a, b], [[1., 0.], [1., 0.]])
        self.assertEqual(m.h, 2) if a @ b < TAU else None
        before = float(m.theta[0])
        apart = np.arccos(a @ b)
        m.merge_pairs(np.array([0]), np.array([1]))
        self.assertEqual(m.h, 1)
        self.assertAlmostEqual(float(m.theta[0]), before + apart / 2, places=4)
        self.assertLess(float(m.thr[0]), TAU)
        self.assertAlmostEqual(float(m.thr[1]), TAU, places=5)

    def test_merged_statistics_are_still_sums(self):
        a, b = np.array([1., 0.], np.float32), np.array([0., 1.], np.float32)
        m = self.store([a, b], [[2., 0.], [0., 4.]])
        m.merge_pairs(np.array([0]), np.array([1]))
        self.assertEqual(int(m.n[0]), 2)
        np.testing.assert_allclose(m.u[0], [2., 4.])

    def test_merging_keeps_each_survivors_radius_in_place(self):
        vecs = [unit(np.array([np.cos(t), np.sin(t)])) for t in (0., 1., 2.)]
        m = self.store(vecs, [[1., 0.]] * 3)
        m.theta[2] = 1.
        m.merge_pairs(np.array([0]), np.array([1]))
        self.assertEqual(m.h, 2)
        self.assertAlmostEqual(float(m.theta[1]), 1., places=5)

    def test_drift_lowers_thresholds_to_no_less_than_the_floor(self):
        m = self.store([[1., 0.]], [[1., 0.]])
        m.widen(.1)
        self.assertAlmostEqual(float(m.thr[0]), TAU * np.cos(.1), places=5)
        m.widen(1.4)
        self.assertAlmostEqual(float(m.thr[0]), FLOOR, places=5)

    def test_moving_keeps_direction_unit_and_write_weight(self):
        m = self.store([[1., 0.], [1., 0.]], [[1., 0.], [1., 0.]])
        m.move_addresses(lambda z: z[:, ::-1] * 3)
        np.testing.assert_allclose(m.a[0], [0., 1.], atol=1e-6)
        np.testing.assert_allclose(m.c[0], [0., 2.], atol=1e-6)

    def test_moving_values_acts_on_the_mean_not_the_sum(self):
        m = self.store([[1., 0.], [1., 0.]], [[1., 0.], [3., 0.]])
        m.move_values(lambda v: v + 10)
        np.testing.assert_allclose(m.u[0], [24., 20.])


class Distinctions(unittest.TestCase):
    def test_a_ball_holding_a_unit_of_another_value_is_refused(self):
        a = unit(np.array([[1., 0.], [np.cos(.4), np.sin(.4)], [np.cos(.2), np.sin(.2)]], np.float32))
        u = np.array([[1., 0.], [1., 0.], [0., 1.]], np.float32)
        ok = merge_keeps_distinctions(a, u, np.full(3, .1, np.float32), np.array([0]), np.array([1]), .8)
        self.assertFalse(ok[0])

    def test_a_ball_holding_a_unit_of_the_same_value_is_allowed(self):
        a = unit(np.array([[1., 0.], [np.cos(.4), np.sin(.4)], [np.cos(.2), np.sin(.2)]], np.float32))
        u = np.array([[1., 0.], [1., 0.], [1., 0.]], np.float32)
        ok = merge_keeps_distinctions(a, u, np.full(3, .1, np.float32), np.array([0]), np.array([1]), .8)
        self.assertTrue(ok[0])

    def test_a_far_unit_of_another_value_does_not_refuse(self):
        a = unit(np.array([[1., 0.], [np.cos(.1), np.sin(.1)], [-1., 0.]], np.float32))
        u = np.array([[1., 0.], [1., 0.], [0., 1.]], np.float32)
        ok = merge_keeps_distinctions(a, u, np.full(3, .1, np.float32), np.array([0]), np.array([1]), .8)
        self.assertTrue(ok[0])


class Trials(unittest.TestCase):
    def test_an_earlier_sets_targets_never_appear_as_candidates(self):
        rng = np.random.default_rng(3)
        train = np.zeros((2 * N, R, 28, 28), np.float32)
        tgt = np.zeros((N_TARGETS, TARGET_TRAIN, 28, 28), np.float32)
        for i in range(N_TARGETS):
            tgt[i] = i
        words = np.arange(N, 2 * N)
        for _ in range(20):
            t = symbol_trials(train, tgt, words, rng)
            seen = set(t['cand'][:, 0, 0].astype(int).tolist())
            self.assertFalse(seen & set(range(N)))
            self.assertEqual(len(t['cand']), 32)
            self.assertEqual(len(seen), 32)

    def test_a_trials_label_points_at_its_own_target(self):
        rng = np.random.default_rng(4)
        train = np.zeros((2 * N, R, 28, 28), np.float32)
        tgt = np.zeros((N_TARGETS, TARGET_TRAIN, 28, 28), np.float32)
        for i in range(N_TARGETS):
            tgt[i] = i
        t = symbol_trials(train, tgt, np.arange(N), rng)
        got = t['cand'][t['label'], 0, 0].astype(int).tolist()
        self.assertEqual(got, t['words'])


if __name__ == '__main__':
    unittest.main()
