import unittest

import numpy as np
import torch

from experiments.sense_slow_average import ALPHA, arms, live_step, update_average


class Live(unittest.TestCase):
    def test_the_threshold_settles_where_alpha_of_passing_inputs_are_answered(self):
        rng = np.random.default_rng(0)
        q = .85
        for _ in range(8000):
            q = live_step(q, rng.normal(.6, .08, 16))
        got = float((rng.normal(.6, .08, 200000) > q).mean())
        self.assertLess(abs(got - ALPHA), .04)

    def test_it_stays_in_bounds(self):
        q = .85
        for _ in range(100000):
            q = live_step(q, np.full(16, -1.))
        self.assertGreaterEqual(q, .5)


class Average(unittest.TestCase):
    def test_an_average_moves_by_one_minus_beta(self):
        a, b = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        with torch.no_grad():
            a.weight.fill_(0.); b.weight.fill_(1.)
        update_average(a, b, .9)
        self.assertAlmostEqual(float(a.weight), .1, places=6)


class Arms(unittest.TestCase):
    def test_every_encoder_has_an_arm(self):
        self.assertEqual({a['encoder'] for a in arms().values()}, {'working', 'ema-0.99', 'ema-0.999', 'ema-0.9999', 'frozen'})


if __name__ == '__main__':
    unittest.main()
