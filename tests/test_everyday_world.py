import unittest

import numpy as np

from src import everyday_world as W
from experiments import everyday_relations as R


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.w = W.build()

    def test_sizes(self):
        self.assertEqual(len(self.w.entities), 185)
        self.assertEqual(len(self.w.y), 7088)
        self.assertEqual(int(self.w.y.sum()), 645)

    def test_queries_unique(self):
        keys = set(zip(self.w.h, self.w.r, self.w.t))
        self.assertEqual(len(keys), len(self.w.y))

    def test_trees_isomorphic(self):
        f = W.family_facts()
        for rel, pairs in f.items():
            a = {p for p in pairs if p[0] in W.TREE_A}
            self.assertEqual({(W.b(x), W.b(y)) for x, y in a}, pairs - a, rel)

    def test_heldout_labels(self):
        w = self.w
        i = lambda rel, h, t: np.flatnonzero((w.r == W.RELATIONS.index(rel)) & (w.h == w.index(h)) & (w.t == w.index(t)))[0]
        self.assertEqual(w.test[i('can', 'bat', 'fly')], 'T3b')
        self.assertEqual(w.y[i('can', 'bat', 'fly')], 1)
        self.assertEqual(w.test[i('bigger-than', 'whale', 'horse')], 'T4')
        self.assertEqual(w.test[i('bigger-than', 'whale', 'elephant')], '')
        self.assertEqual(w.test[i('opposite', 'dark', 'light')], 'T5')
        self.assertEqual(w.test[i('opposite', 'light', 'dark')], '')
        self.assertEqual(w.y[i('can-subject', 'eagle-w', 'fly-w')], 1)
        self.assertEqual(w.test[i('can-subject', 'eagle-w', 'fly-w')], 'T8')
        self.assertEqual(w.y[i('grandparent-of', W.b('andrew'), W.b('colin'))], 1)
        self.assertEqual(w.test[i('grandparent-of', W.b('andrew'), W.b('colin'))], 'T7')

    def test_shuffled_keeps_counts(self):
        ys = W.shuffled(self.w, 400)
        for ri in range(len(W.RELATIONS)):
            m = self.w.r == ri
            self.assertEqual(ys[m].sum(), self.w.y[m].sum())
        self.assertGreater((ys != self.w.y).sum(), 100)

    def test_fast_mapping_names_exist(self):
        for name, (given, read, ks) in W.fast_mapping_specs().items():
            for rel, h, t, _ in given + read:
                self.assertIn(rel, W.RELATIONS)
                for x in (h, t):
                    self.assertTrue(x == name or x in self.w.entities, x)
            self.assertLessEqual(max(ks), len(given))

    def test_auroc(self):
        self.assertEqual(R.auroc([0, 1, 2, 3], [0, 0, 1, 1]), 1.0)
        self.assertEqual(R.auroc([1, 1], [0, 1]), 0.5)
        self.assertIsNone(R.auroc([1, 2], [1, 1]))


if __name__ == '__main__':
    unittest.main()
