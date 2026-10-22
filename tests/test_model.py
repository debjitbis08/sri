import random
import unittest
from src.model import example, settle, dream_seed

class BaselineTests(unittest.TestCase):
    def test_full_closure(self):
        m = example()
        r = settle(m, {0})
        self.assertEqual(r.active, frozenset(m.nodes))
        self.assertLessEqual(r.rounds, len(m.nodes))
    def test_threshold_monotonicity(self):
        m=example()
        for lo in (0, 20, 50, 100, 200):
            for hi in (lo, 220, 255):
                if hi >= lo:
                    self.assertTrue(settle(m,{0},hi).active <= settle(m,{0},lo).active)
    def test_budget_is_partial(self):
        m=example()
        for k in range(6):
            self.assertTrue(settle(m,{0},max_rounds=k).active <= settle(m,{0}).active)
    def test_endogenous_sampling_deterministic(self):
        m=example()
        a,b=random.Random(27),random.Random(27)
        self.assertEqual([dream_seed(m,a) for _ in range(20)],
                         [dream_seed(m,b) for _ in range(20)])
    def test_local_structural_change(self):
        m=example()
        before=settle(m,{0},threshold=100)
        self.assertNotIn(3,before.active)
        m.link(2,3,130)
        after=settle(m,{0},threshold=100)
        self.assertIn(3,after.active)
        self.assertTrue(before.active <= after.active)

if __name__ == '__main__': unittest.main()
