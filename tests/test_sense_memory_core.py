import sys
import unittest

import numpy as np


def module():
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        from experiments import sense_memory_core as SMC
    finally:
        sys.argv = argv
    return SMC


class WindowShiftRule(unittest.TestCase):
    def setUp(self):
        self.SMC = module()

    def batch(self, rng, centre, n=256):
        z = centre + .05 * rng.normal(size=(n, len(centre)))
        return {'dinov2-s': z.astype(np.float32), 'trocr-s': z.astype(np.float32)}, np.zeros((n, 28, 28), np.float32)

    def test_opens_on_a_run_returns_its_images_and_expires_stale_batches(self):
        rng = np.random.default_rng(0)
        r = self.SMC.WindowShift(np.random.default_rng(1), window=4)
        a, b = rng.normal(size=16), rng.normal(size=16)
        j, new, x = r.observe_images(*self.batch(rng, a))
        self.assertEqual((j, new, x), (None, False, None))
        j, new, x = r.observe_images(*self.batch(rng, a))
        self.assertEqual((j, new), (0, True))
        self.assertEqual(len(x), 512)  # the forming images, for the gate's views
        self.assertEqual(r.observe_images(*self.batch(rng, a))[:2], (0, False))  # familiar: joins
        self.assertEqual(r.observe_images(*self.batch(rng, b))[:2], (None, False))  # one novel batch starts forming
        for _ in range(4):
            self.assertEqual(r.observe_images(*self.batch(rng, a))[:2], (0, False))
        self.assertEqual(len(r.forming), 0)  # the stale novel batch expired
        self.assertEqual(r.observe_images(*self.batch(rng, b))[:2], (None, False))  # so one more novel batch does not open a centre


class Assign(unittest.TestCase):
    def test_routes_by_dinov2_among_open_centres(self):
        SMC = module()
        cs = [{'dinov2-s': np.array([1., 0.]), 'trocr-s': np.zeros(2)}, {'dinov2-s': np.array([0., 1.]), 'trocr-s': np.zeros(2)}]
        z = {'dinov2-s': np.array([[.9, .1], [.1, .9]]), 'trocr-s': np.zeros((2, 2))}
        np.testing.assert_array_equal(SMC.assign(z, cs), [0, 1])
        np.testing.assert_array_equal(SMC.assign(z, cs[:1]), [0, 0])


if __name__ == '__main__':
    unittest.main()
