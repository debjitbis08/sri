import unittest

import numpy as np

from experiments.sense_consolidation_moving import R, arms, unit_values


class UnitValues(unittest.TestCase):
    def test_a_units_value_is_the_mean_code_of_its_experiences_value_images(self):
        table = np.zeros((2, 9, 1), np.float32)
        table[0, 2] = 10.  # collection 0, count 3
        table[1, 2] = 20.  # collection 1, count 3
        table[0, 4] = 100.  # collection 0, count 5
        counts = np.array([3, 5])
        first_coll = np.zeros(2 * R, np.int64)
        first_coll[1] = 1
        tr = [0, 1, R]
        out = unit_values([0, 0, 1], tr, 2, first_coll, counts, table)
        self.assertEqual(out[:, 0].tolist(), [15., 100.])

    def test_value_source_may_have_fewer_collections_than_were_drawn(self):
        table = np.ones((3, 9, 1), np.float32)
        table[1] = 7.
        first_coll = np.full(R, 4, np.int64)
        self.assertEqual(unit_values([0], [0], 1, first_coll, np.array([2]), table)[0, 0], 7.)


class Arms(unittest.TestCase):
    def test_every_arm_keeps_experiences_so_every_arm_has_values(self):
        self.assertTrue(all(a['keep'] == 8 for a in arms().values()))


if __name__ == '__main__':
    unittest.main()
