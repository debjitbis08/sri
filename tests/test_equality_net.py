import unittest

import torch

from src import equality_net as EQ


class EqualityNetTest(unittest.TestCase):
    def test_op_targets(self):
        t = EQ.op_targets(torch.tensor([1, 4, 5, 9]), torch.tensor([9, 2, 5, 1]))
        self.assertEqual(t[:, 0].tolist(), [1, 0, 0, 0])  # more: b > a
        self.assertEqual(t[:, 1].tolist(), [1, 0, 1, 1])  # sum10

    def test_substitution_zero_for_identical_codes(self):
        for arm in ('eq-en', 'two-ctx-eq-en'):
            m = EQ.EqNet(arm, 8, torch.Generator().manual_seed(0))
            c = torch.randn(5, EQ.D)
            self.assertAlmostEqual(float(EQ.substitution_loss(m, torch.randn(6, EQ.D), c, c.clone(), True)), 0.0, places=6)
            self.assertGreater(float(EQ.substitution_loss(m, torch.randn(6, EQ.D), c, torch.randn(5, EQ.D), True)), 0.0)

    def test_modes(self):
        self.assertEqual([EQ.en_mode(a) for a in EQ.ARMS], [None, 'label', 'eq', None, 'label', 'eq', 'label', 'eq'])
        self.assertEqual([EQ.zh_mode(a) for a in EQ.ARMS], [None] * 6 + ['label', 'eq'])
        self.assertEqual(EQ.EqNet('two-ctx-eq-en', 8, torch.Generator()).arm, 'two-ctx')

    def test_same_count_pairs(self):
        i, j = EQ.same_count_pairs(torch.tensor([1, 2, 2]), torch.tensor([2, 3, 1]))
        self.assertEqual(sorted(zip(i.tolist(), j.tolist())), [(0, 2), (1, 0), (2, 0)])


if __name__ == '__main__':
    unittest.main()
