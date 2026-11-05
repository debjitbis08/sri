import unittest

import numpy as np

from src import everyday_world_v3 as W


class TestWorldV3(unittest.TestCase):
    def setUp(self):
        self.w = W.build()

    def q(self, rel, h, t):
        w = self.w
        return np.flatnonzero((w.r == W.RELATIONS.index(rel)) & (w.h == w.index(h)) & (w.t == w.index(t)))[0]

    def test_sizes(self):
        self.assertEqual((len(self.w.entities), len(self.w.relations), len(self.w.y), int(self.w.y.sum())),
                         (205, 37, 17231, 2163))

    def test_phrases(self):
        w = self.w
        self.assertEqual(w.y[self.q('good-phrase', 'dog-w', 'run-w')], 1)
        self.assertEqual(w.y[self.q('good-phrase', 'run-w', 'dog-w')], 0)
        self.assertEqual(w.y[self.q('good-phrase', 'big-w', 'whale-w')], 1)
        self.assertEqual(w.y[self.q('good-phrase', 'whale-w', 'big-w')], 0)
        self.assertEqual(w.y[self.q('makes-sense', 'eagle-w', 'fly-w')], 1)
        self.assertEqual(w.y[self.q('makes-sense', 'dog-w', 'fly-w')], 0)
        self.assertEqual(w.y[self.q('makes-sense', 'big-w', 'whale-w')], 1)
        self.assertEqual(w.y[self.q('makes-sense', 'small-w', 'whale-w')], 0)

    def test_g2_g3(self):
        w = self.w
        self.assertEqual(w.test[self.q('names', 'robin-w', 'robin')], '')
        self.assertEqual(w.test[self.q('is-a', 'robin-w', 'noun')], 'G2')
        self.assertEqual(w.test[self.q('good-phrase', 'cow-w', 'run-w')], '')
        self.assertEqual(w.test[self.q('good-phrase', 'run-w', 'cow-w')], '')
        self.assertEqual(w.test[self.q('good-phrase', 'cow-w', 'fly-w')], 'G3')
        self.assertEqual(w.test[self.q('names', 'cow-w', 'cow')], 'G3-names')

    def test_g1_both_orders(self):
        w = self.w
        m = w.test == 'G1'
        pairs = {(int(a), int(c)) for a, c in zip(w.h[m], w.t[m])}
        self.assertTrue(all((c, a) in pairs for a, c in pairs))

    def test_fast_mapping_names(self):
        for name, (given, read, ks) in W.fast_mapping_specs().items():
            for rel, h, t, _ in given + read:
                self.assertIn(rel, W.RELATIONS)
                for x in (h, t):
                    self.assertTrue(x == name or x in self.w.entities, x)


if __name__ == '__main__':
    unittest.main()
