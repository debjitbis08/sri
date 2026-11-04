import sys
import unittest

import numpy as np


def module():
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        from experiments import sense_memory_replay as SMR
    finally:
        sys.argv = argv
    return SMR


class ReservoirSampling(unittest.TestCase):
    def test_keeps_everything_until_full_then_a_uniform_sample_of_all_writes(self):
        SMR = module()
        hits = np.zeros(400)
        for t in range(2000):
            r = SMR.Reservoir(100, np.random.default_rng(t))
            r.add(range(50))
            self.assertEqual(len(r.items), 50) if t == 0 else None
            r.add(range(50, 400))  # capacity never grows
            self.assertEqual(len(r.items), 100)
            hits[r.items] += 1
        p = hits / 2000
        self.assertLess(abs(p[:200].mean() - p[200:].mean()), .01)  # old writes are as likely to be kept as recent ones
        self.assertLess(abs(p.mean() - .25), .005)


class MixedUnits(unittest.TestCase):
    def test_majority_count_ties_go_to_the_count_written_first(self):
        SMR = module()
        mem = SMR.LiveMemory.__new__(SMR.LiveMemory)
        mem.prep = {'writes': [{'fact': 3, 'count': 5}, {'fact': 7, 'count': 2}, {'fact': 7, 'count': 2}, {'fact': 3, 'count': 5}, {'fact': 3, 'count': 5}]}
        mem.log = {(0, 0): [1, 0], (0, 1): [0, 3, 4], (0, 2): [1, 2, 0, 3]}
        self.assertEqual(mem.describe((0, 0)), (3, 2, True, True))  # one write each: the first written count wins; facts tie to the lower id
        self.assertEqual(mem.describe((0, 1)), (3, 5, False, False))
        self.assertEqual(mem.describe((0, 2)), (3, 2, True, True))


class Phases(unittest.TestCase):
    def test_phase_boundaries_follow_the_curriculum(self):
        SMR = module()
        S, Dl = SMR.D.STEPS, SMR.LM.DELAY
        self.assertEqual([SMR.phase(s) for s in (1, S, S + 1, 2 * S + 1, 3 * S + 1, 3 * S + Dl, 3 * S + Dl + 1)],
                         ['block1', 'block1', 'block2', 'block3', 'delay', 'delay', 'relearn'])


if __name__ == '__main__':
    unittest.main()
