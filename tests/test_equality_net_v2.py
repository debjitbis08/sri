import unittest

import torch

from src import equality_net_v2 as EQ


class EqualityV2Test(unittest.TestCase):
    def test_targets_and_properties(self):
        ya, yb = torch.tensor([3, 3, 2, 8]), torch.tensor([3, 5, 8, 2])
        t = EQ.op_targets(ya, yb)
        self.assertEqual(t[:, 0].tolist(), [1, 0, 0, 0])
        self.assertEqual(t[:, 1].tolist(), [0, 1, 1, 0])
        self.assertEqual(t[:, 2].tolist(), [0, 0, 1, 1])
        r = EQ.op_targets(yb, ya)
        self.assertTrue(torch.equal(t[:, 0], r[:, 0]) and torch.equal(t[:, 2], r[:, 2]))  # eq, sum10 symmetric
        self.assertEqual((t[:, 1] + r[:, 1]).tolist(), [0, 1, 1, 1])  # more antisymmetric, irreflexive

    def test_modes_and_bases(self):
        self.assertEqual([EQ.buffer_modes(a) for a in EQ.ARMS], [('label', None), ('one', None), ('both', None), ('label', 'label'),
                                                                 ('one', 'one'), ('label', None), ('one', None)])
        self.assertEqual(EQ.RelNet('count-rel-en', 8, torch.Generator()).arm, 'count')
        self.assertEqual(EQ.RelNet('rel-en', 8, torch.Generator()).arm, 'two-ctx')

    def test_op_loss_runs_one_order(self):
        m = EQ.RelNet('rel-en', 8, torch.Generator().manual_seed(0))
        L = EQ.op_loss(m, torch.randn(4, EQ.D), torch.tensor([1, 2, 3, 4]), torch.randn(3, EQ.D), torch.tensor([2, 3, 9]), EQ.CROSS_OPS)
        self.assertTrue(torch.isfinite(L))


if __name__ == '__main__':
    unittest.main()
