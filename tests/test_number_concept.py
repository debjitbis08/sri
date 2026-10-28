from pathlib import Path
import unittest

import numpy as np
import torch

from src import number_concept as nc


ROOT = Path(__file__).resolve().parents[1]


class AcceptanceChecks(unittest.TestCase):
    def test_all_pass(self):
        for name, value in nc.acceptance_checks().items():
            with self.subTest(name):
                self.assertTrue(all(value.values()) if isinstance(value, dict) else value)


class Dice(unittest.TestCase):
    def test_metadata_preassigned_and_count_independent(self):
        a = nc.dice_metadata('standard', 'train', nc.FACES, 50)
        b = nc.dice_metadata('standard', 'train', nc.FACES, 50)
        np.testing.assert_array_equal(a['radius'], b['radius'])
        pool = nc.build_dice('standard', 'validation', per_count=20)
        meta = nc.dice_metadata('standard', 'validation', nc.FACES, 20)
        np.testing.assert_array_equal(pool['radius'], meta['radius'])  # rejection never redraws radius

    def test_separation_and_deterministic(self):
        for layout, counts in (('standard', nc.FACES), ('random', (7, 9))):
            p1 = nc.build_dice(layout, 'test', counts=counts, per_count=5)
            p2 = nc.build_dice(layout, 'test', counts=counts, per_count=5)
            np.testing.assert_array_equal(p1['images'], p2['images'])
            for c, r, n in zip(p1['centers'], p1['radius'], p1['labels']):
                sep = nc.min_separation(c[None, :n])[0]
                self.assertGreaterEqual(sep, 2 * r + 1)
            self.assertTrue(((p1['images'] >= 0) & (p1['images'] <= 1)).all())

    def test_quarter_turn_and_cap(self):
        c = nc.face_centers(2)
        np.testing.assert_allclose(nc.quarter_turn(nc.quarter_turn(c, 1), 3), c)
        np.testing.assert_allclose(nc.quarter_turn(c, 1), [[7, 20], [20, 7]])
        with self.assertRaises(nc.PlacementExhausted):
            nc.place('random', 9, 2.2, nc.stream('test'), cap=10)

    def test_splits_use_distinct_streams(self):
        a = nc.build_dice('standard', 'validation', per_count=3)
        b = nc.build_dice('standard', 'test', per_count=3)
        self.assertFalse(np.array_equal(a['images'], b['images']))


class Sampling(unittest.TestCase):
    def test_pairs_are_successors_and_balanced(self):
        labels = np.repeat(np.arange(10), 30)
        s = nc.Sampler(labels, nc.DIGITS, range(9), ('t',))
        src, tgt = s.pairs()
        np.testing.assert_array_equal(labels[tgt], labels[src] + 1)
        self.assertLessEqual(np.ptp(np.bincount(labels[src])[:9]), 1)
        rec = s.reconstruction()
        self.assertLessEqual(np.ptp(np.bincount(labels[rec], minlength=10)), 1)

    def test_shuffle_stream_does_not_move_other_draws(self):
        labels = np.repeat(np.arange(1, 7), 20)
        a, b = (nc.Sampler(labels, nc.FACES, range(1, 6), ('t',)) for _ in range(2))
        sa, ta = a.pairs(shuffled=False)
        sb, tb = b.pairs(shuffled=True)
        np.testing.assert_array_equal(sa, sb)
        self.assertEqual(sorted(ta.tolist()), sorted(tb.tolist()))
        np.testing.assert_array_equal(a.reconstruction(), b.reconstruction())

    def test_initialization_pairing(self):
        self.assertEqual(nc.state_hash(nc.init_dice(3)), nc.state_hash(nc.init_dice(3)))
        self.assertNotEqual(nc.state_hash(nc.init_dice(3)), nc.state_hash(nc.init_dice(4)))
        self.assertEqual(nc.state_hash(nc.init_digit(3)), nc.state_hash(nc.init_digit(3)))


class Losses(unittest.TestCase):
    def test_successor_loss_floor(self):
        """Perfect alignment leaves the positive-multiplicity entropy floor log|P|."""
        k = torch.zeros(4, nc.D, dtype=torch.complex64)
        k[:2, 0], k[2:, 1] = 1, 1
        loss = nc.successor_loss(k * 1e3, k, torch.tensor([1, 1, 2, 2]))
        self.assertAlmostEqual(float(loss), np.log(2), places=4)

    def test_anchor_loss_relative(self):
        c = torch.ones(nc.D, dtype=torch.complex64)
        self.assertAlmostEqual(float(nc.anchor_loss(2 * c, c)), 1., places=6)

    def test_initialization_scale(self):
        w = nc.complex_glorot(nc.stream('t'), 400, 600)
        self.assertAlmostEqual(float(w.abs().pow(2).mean()), 2 / 1000, delta=2e-4)


@unittest.skipUnless((ROOT / 'data/mnist' / nc.MNIST_FILES['train_labels'][0]).exists(), 'MNIST not downloaded')
class Mnist(unittest.TestCase):
    def test_split(self):
        raw, _ = nc.load_mnist(ROOT / 'data/mnist')
        train, val = nc.mnist_split(raw['train_labels'])
        self.assertEqual(len(train), 54000)
        self.assertFalse(set(train) & set(val))
        np.testing.assert_array_equal(np.bincount(raw['train_labels'][val]), [600] * 10)


if __name__ == '__main__':
    unittest.main()
