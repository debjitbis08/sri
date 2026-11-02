import unittest

import numpy as np

from experiments import sense_image_free as F
from experiments.sense_image_free_windows import Arm, Drift, QUANTILE, RunningMap, Store, matched, TARGETS


class Totals(unittest.TestCase):
    def test_running_totals_give_the_same_map_as_all_the_pairs(self):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(300, 6)), rng.normal(size=(300, 6))
        m = RunningMap(6)
        for i in range(0, 300, 50):
            m.add(X[i:i + 50], Y[i:i + 50])
        np.testing.assert_allclose(m.fn()(X), F.fit_map(X, Y, F.LAMBDA)(X), atol=1e-4)

    def test_drift_reports_what_a_map_fitted_on_earlier_arrivals_leaves(self):
        rng = np.random.default_rng(1)
        Q = np.linalg.qr(rng.normal(size=(16, 16)))[0]
        d = Drift(16)
        for half in (False, True):
            X = F.S1.unit(rng.normal(size=(600, 16)))
            d.add(X, X @ Q, half)
        r = d.result()
        self.assertGreater(r['angle'], 1.)
        self.assertLess(r['residual'], .15)


class Background(unittest.TestCase):
    def test_the_threshold_settles_at_the_high_quantile(self):
        rng = np.random.default_rng(2)
        m = Store(2, 1, .85, 2, 1, background=True)
        m.write(np.array([1., 0.], np.float32), np.zeros(1, np.float32))
        for _ in range(4000):
            ang = rng.normal(1.2, .3, 16)
            m.observe(np.c_[np.cos(ang), np.sin(ang)].astype(np.float32))
        Z = rng.normal(1.2, .3, 200000)
        want = np.quantile(np.cos(Z), QUANTILE)
        self.assertLess(abs(float(m.q[0]) - want), .06)
        self.assertAlmostEqual(float(m.thr[0]), float(m.q[0]))

    def test_a_merge_leaves_a_threshold_covering_both_balls(self):
        m = Store(4, 1, .85, 2, 1, background=True)
        a = np.array([np.cos(.3), np.sin(.3)], np.float32), np.array([np.cos(.3), -np.sin(.3)], np.float32)
        for z in a:
            m.write(z, np.zeros(1, np.float32))
        m.h = 2
        m.to_balls()
        m.merge_pairs(np.array([0]), np.array([1]))
        m.from_balls()
        self.assertLess(float(m.thr[0]), .85)
        self.assertEqual(float(m.thr[0]), float(m.q[0]))


class Matched(unittest.TestCase):
    def test_the_shift_answers_the_stated_share(self):
        rng = np.random.default_rng(3)
        m = Store(50, 1, .85, 8, 1)
        unit = F.S1.unit(rng.normal(size=(50, 8)).astype(np.float32))
        m.a[:50], m.h = unit, 50
        Z = F.S1.unit(rng.normal(size=(1000, 8)).astype(np.float32))
        for t in TARGETS:
            d = matched(m, Z, t)
            got = float(((Z @ unit.T - (m.thr[:50] + d)).max(1) >= 0).mean())
            self.assertLess(abs(got - t), .01)


class Arms(unittest.TestCase):
    def test_frozen_keys_do_not_move(self):
        a = Arm('fixed', False, 'stale', m=1, frozen=True)
        a.mem.write(np.array([1.] + [0.] * 31, np.float32), np.zeros(1, np.float32))
        before = a.mem.a[0].copy()
        a.publish({'map': lambda z: z * 0 + 1, 'angle': 1., 'residual': 1.}, {})
        np.testing.assert_array_equal(a.mem.a[0], before)

    def test_transport_without_widening_leaves_thresholds_alone(self):
        a = Arm('fixed', True, 'stale', m=1)
        a.mem.write(np.array([1.] + [0.] * 31, np.float32), np.zeros(1, np.float32))
        t = a.mem.thr[0]
        a.publish({'map': lambda z: z[:, ::-1], 'angle': 1., 'residual': 1.}, {})
        self.assertEqual(a.mem.thr[0], t)
        self.assertEqual(float(a.mem.a[0, 31]), 1.)


if __name__ == '__main__':
    unittest.main()
