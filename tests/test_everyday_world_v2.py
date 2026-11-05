import unittest

import numpy as np

from src import everyday_world_v2 as W


class TestWorldV2(unittest.TestCase):
    def setUp(self):
        self.w = W.build()

    def test_sizes(self):
        self.assertEqual((len(self.w.entities), len(self.w.relations), len(self.w.y), int(self.w.y.sum())),
                         (189, 36, 14179, 1864))

    def test_symmetric_and_inverse_tables(self):
        for rel in W.SYMMETRIC:
            self.assertEqual(self.w.facts[rel], {(y, x) for x, y in self.w.facts[rel]}, rel)
        for a, b in W.INVERSE_PAIRS:
            self.assertEqual(self.w.facts[b], {(y, x) for x, y in self.w.facts[a]}, (a, b))

    def test_no_contradiction_in_held_rows(self):
        """For P2, P3, P5 every held-out head has its whole row held out (no trained query with that head)."""
        w = self.w
        for tst in ('P2', 'P3', 'P5'):
            m = w.test == tst
            for ri in np.unique(w.r[m]):
                for h in np.unique(w.h[m & (w.r == ri)]):
                    row = (w.r == ri) & (w.h == h)
                    self.assertTrue((w.test[row] != "").all(), (tst, w.relations[ri], w.entities[h]))

    def test_column_support_kept(self):
        """P2: the reversed fact of each held-out true fact is trained (unless itself held out)."""
        w = self.w
        m = (w.test == 'P2') & (w.y == 1)
        trained = 0
        for h, r, t in zip(w.h[m], w.r[m], w.t[m]):
            j = np.flatnonzero((w.r == r) & (w.h == t) & (w.t == h))[0]
            trained += w.test[j] == ''
        self.assertGreater(trained / m.sum(), 0.5)

    def test_scales(self):
        for comp, inv, order, p1, p2 in W.SCALES.values():
            self.assertIn((order[0], order[-1]), self.w.facts[comp])
            self.assertIn((order[0], p1), self.w.facts['is'])
            self.assertIn((p1, p2), self.w.facts['opposite'])

    def test_fast_mapping_names(self):
        for name, (given, read, ks, row) in W.fast_mapping_specs().items():
            for rel, h, t, _ in given + read + row:
                self.assertIn(rel, W.RELATIONS)
                for x in (h, t):
                    self.assertTrue(x == name or x in self.w.entities, x)


if __name__ == '__main__':
    unittest.main()
