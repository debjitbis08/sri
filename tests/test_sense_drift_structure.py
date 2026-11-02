import gzip
import io
import tarfile
import unittest

import numpy as np

from experiments.sense_drift_structure import explained, parse_cifar, rayleigh, recall, slow_directions, structure


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Structure(unittest.TestCase):
    def test_noise_around_a_fixed_code_is_jitter(self):
        rng = np.random.default_rng(0)
        base = unit(rng.normal(size=(200, 16)))
        path = unit(base[None] + .05 * rng.normal(size=(25, 200, 16)))
        r = structure(path, 8)
        self.assertGreater(r['jitter'], .9)
        self.assertLess(r['net_over_path'], .3)

    def test_steady_movement_is_trend(self):
        rng = np.random.default_rng(1)
        base, way = unit(rng.normal(size=(200, 16))), unit(rng.normal(size=(200, 16)))
        path = np.stack([base + .02 * t * way for t in range(25)])
        r = structure(path, 8)
        self.assertGreater(r['trend'], .9)
        self.assertGreater(r['net_over_path'], .95)

    def test_a_random_walk_is_diffusion(self):
        rng = np.random.default_rng(2)
        path = np.cumsum(.01 * rng.normal(size=(25, 2000, 16)), 0)
        self.assertGreater(structure(path, 8)['diffusion'], .7)


class Map(unittest.TestCase):
    def test_a_shared_rotation_is_explained_on_another_set(self):
        rng = np.random.default_rng(3)
        Q = np.linalg.qr(np.eye(16) + .3 * rng.normal(size=(16, 16)))[0]
        a, b = rng.normal(size=(400, 16)), rng.normal(size=(100, 16))
        self.assertGreater(explained(a, a @ Q, b, b @ Q), .95)

    def test_private_movement_is_not_explained(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=(400, 16)), rng.normal(size=(100, 16))
        self.assertLess(explained(a, a + rng.normal(size=a.shape), b, b + rng.normal(size=b.shape)), .1)


class Slow(unittest.TestCase):
    def test_finds_the_direction_that_does_not_move(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(2000, 4))
        move = rng.normal(size=(2000, 4)) * np.array([1., 1., 1., .01])
        val, vec = slow_directions(z, move)
        self.assertGreater(abs(vec[3, 0]), 5 * max(abs(vec[0, 0]), abs(vec[1, 0]), abs(vec[2, 0])))
        self.assertLess(val[0], .01)
        self.assertLess(rayleigh(vec[:, :1], rng.normal(size=(500, 4)), rng.normal(size=(500, 4)) * np.array([1., 1., 1., .01]))[0], .05)

    def test_recall_reads_the_nearest_label(self):
        s = np.array([[1., 0.], [0., 1.]])
        self.assertEqual(recall(s, np.array([3, 4]), np.array([[.9, .1]]), np.array([3])), 1.)


class Cifar(unittest.TestCase):
    def archive(self, count):
        rec = np.zeros((count, 3073), np.uint8)
        rec[:, 0] = np.arange(count) % 10
        rec[:, 1:] = 200
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as t:
            info = tarfile.TarInfo('cifar-10-batches-bin/data_batch_1.bin')
            info.size = rec.size
            t.addfile(info, io.BytesIO(rec.tobytes()))
        return gzip.compress(buf.getvalue())

    def test_complete_archive(self):
        x, y = parse_cifar(self.archive(5))
        self.assertEqual(x.shape, (5, 28, 28))
        self.assertEqual(y.tolist(), [0, 1, 2, 3, 4])
        self.assertAlmostEqual(float(x.max()), 200 / 255, places=5)

    def test_a_cut_archive_gives_only_whole_records(self):
        raw = self.archive(2000)
        x, _ = parse_cifar(raw[:len(raw) // 2])
        self.assertLess(len(x), 2000)
        self.assertEqual(x.shape[1:], (28, 28))


if __name__ == '__main__':
    unittest.main()
