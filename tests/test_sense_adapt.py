import unittest

import numpy as np

from experiments import sense_adapt as SA


def batch(rng, mean, n=SA.BATCH):
    z = rng.normal(size=(n, 16)).astype(np.float32) + mean
    return {'dinov2-s': z, 'trocr-s': -z}


class ShiftRule(unittest.TestCase):
    def test_one_centre_per_stream_of_a_kind_and_old_kinds_rejoin(self):
        rng = np.random.default_rng(0)
        A, B = np.zeros(16, np.float32) + 1, np.eye(16, dtype=np.float32)[0] * 4 + 1
        c = SA.ShiftCentres(np.random.default_rng(1))
        got = [c.observe(batch(rng, A)) for _ in range(4)] + [c.observe(batch(rng, B)) for _ in range(4)] + [c.observe(batch(rng, A))]
        self.assertEqual(len(c.centres), 2)
        self.assertEqual([g[0] for g in got], [None, 0, 0, 0, None, 1, 1, 1, 0])
        self.assertEqual([g[1] for g in got].count(True), 2)
        np.testing.assert_array_equal(c.assign(batch(rng, B, 5)), 1)
        self.assertEqual(c.centre_array('trocr-s').shape, (2, 16))


if __name__ == '__main__':
    unittest.main()
