import unittest

import numpy as np

from experiments.sense_consolidation_evidence import cover_evidence, distinct, shared_evidence


def unit(*deg):
    r = np.radians(deg)
    return np.stack([np.cos(r), np.sin(r)], 1).astype(np.float32)


class Evidence(unittest.TestCase):
    def test_shared_requires_both_units_to_give_each_other_traces(self):
        a = unit(0, 10, 90)
        z = unit(4, 6, 89)
        owner = np.array([0, 1, 2])
        E = shared_evidence(z @ a.T, owner, 3, .5)
        self.assertTrue(E[0, 1] and E[1, 0])
        self.assertFalse(E[0, 2] or E[2, 0] or E[1, 2])

    def test_cover_counts_traces_each_address_answers(self):
        a = unit(0, 10, 90)
        z = unit(4, 6, 80)
        E = cover_evidence(z @ a.T, np.array([0, 1, 2]), 3, .5, .95)
        self.assertTrue(E[0, 1] and E[1, 0])
        self.assertFalse(E[2, 1] or E[1, 2])

    def _pair(self):
        c = unit(0, 40, 20)
        u = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
        return c, u, np.array([0]), np.array([1])

    def test_distinct_fails_when_a_different_content_lies_inside_the_merged_radius(self):
        c, u, lo, up = self._pair()
        z = unit(0, 40, 20)
        self.assertFalse(distinct(z, np.array([0, 1, 2]), c, u, lo, up, .8)[0])

    def test_distinct_allows_the_same_content_inside(self):
        c, u, lo, up = self._pair()
        u[2] = [1, 0]
        self.assertTrue(distinct(unit(0, 40, 20), np.array([0, 1, 2]), c, u, lo, up, .8)[0])

    def test_distinct_uses_the_reference_set(self):
        c, u, lo, up = self._pair()
        u[2] = [1, 0]
        z = unit(0, 40, 170)
        far = unit(170, 171)
        near = unit(20, 21)
        owner = np.array([0, 1, 2])
        self.assertTrue(distinct(z, owner, c, u, lo, up, .8, far, 0.)[0])
        self.assertFalse(distinct(z, owner, c, u, lo, up, .8, near, 0.)[0])


if __name__ == '__main__':
    unittest.main()
