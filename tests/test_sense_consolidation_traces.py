import unittest

import numpy as np

from experiments.sense_consolidation_traces import spread_select


def unit(*deg):
    r = np.radians(deg)
    return np.stack([np.cos(r), np.sin(r)], 1).astype(np.float32)


class Spread(unittest.TestCase):
    def test_keeps_everything_when_within_the_cap(self):
        self.assertEqual(spread_select(unit(0, 10), 8).tolist(), [0, 1])

    def test_chooses_the_extremes_over_a_tight_cluster(self):
        z = unit(0, 1, 2, 3, 40, -40)
        chosen = spread_select(z, 2)
        self.assertEqual(sorted(chosen.tolist()), [4, 5])

    def test_third_choice_fills_the_gap_between_extremes(self):
        z = unit(-40, 40, 0, 35)
        self.assertEqual(sorted(spread_select(z, 3).tolist()), [0, 1, 2])

    def test_returns_exactly_k_distinct_traces(self):
        z = unit(*range(0, 60, 5))
        chosen = spread_select(z, 4)
        self.assertEqual(len(set(chosen.tolist())), 4)


if __name__ == '__main__':
    unittest.main()
