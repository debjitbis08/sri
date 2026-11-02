import unittest

import numpy as np

from src.sparse_local_memory import Memory


def unit(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)], np.float32)


class Fixtures(unittest.TestCase):
    def test_moved_competitor_changes_the_winner_with_no_allocation_or_forced_write(self):
        m = Memory(4, 1, .7, 2, 1)
        m.write(unit(-40), np.array([1.], np.float32))
        m.write(unit(60), np.array([2.], np.float32))
        q = unit(0)[None]
        self.assertEqual(m.read(q)[0][0, 0], 1.)
        m.write(unit(25), np.array([2.], np.float32))
        m.write(unit(25), np.array([2.], np.float32))
        self.assertEqual((m.h, m.forced), (2, 0))
        self.assertGreater(float(m.a[1] @ unit(0)), .8)
        self.assertEqual(m.read(q)[0][0, 0], 2.)

    def test_disjoint_write_leaves_old_state_and_read_unchanged(self):
        m = Memory(4, 1, .9, 2, 1)
        m.write(unit(0), np.array([1.], np.float32))
        before = m.a[0].copy(), m.u[0].copy(), m.n[0]
        q = unit(1)[None]
        r = m.read(q)[0].copy()
        m.write(unit(90), np.array([5.], np.float32))
        self.assertTrue(np.array_equal(m.a[0], before[0]) and np.array_equal(m.u[0], before[1]) and m.n[0] == before[2])
        self.assertTrue(np.array_equal(m.read(q)[0], r))

    def test_abstains_below_tau_and_counts_a_forced_write_when_full(self):
        m = Memory(1, 1, .9, 2, 1)
        m.write(unit(0), np.array([1.], np.float32))
        self.assertFalse(m.read(unit(90)[None])[1][0])
        m.write(unit(90), np.array([3.], np.float32))
        self.assertEqual((m.h, m.forced, m.n[0]), (1, 1, 2))
        self.assertEqual(m.read(unit(44)[None])[0][0, 0], 2.)

    def test_top_k_takes_the_mean_of_winners(self):
        m = Memory(4, 2, .1, 2, 1)
        m.tau = 2.  # force allocation
        m.write(unit(0), np.array([1.], np.float32))
        m.write(unit(20), np.array([3.], np.float32))
        m.write(unit(80), np.array([9.], np.float32))
        m.tau = .1
        self.assertAlmostEqual(float(m.read(unit(10)[None])[0][0, 0]), 2.)

    def test_core_part_read(self):
        m = Memory(4, 1, .9, 4, 1)
        m.write(np.array([1, 0, 1, 0], np.float32) / np.sqrt(2), np.array([1.], np.float32))
        m.write(np.array([0, 1, 0, 1], np.float32) / np.sqrt(2), np.array([2.], np.float32))
        q = np.array([[1, 0]], np.float32)
        self.assertEqual(m.read(q, slice(0, 2))[0][0, 0], 1.)

    def test_read_changes_nothing(self):
        m = Memory(4, 1, .5, 2, 1)
        m.write(unit(0), np.array([1.], np.float32))
        a, n = m.a.copy(), m.n.copy()
        m.read(unit(10)[None])
        self.assertTrue(np.array_equal(a, m.a) and np.array_equal(n, m.n))


if __name__ == '__main__':
    unittest.main()
