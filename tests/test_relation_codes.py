import unittest

import torch

from src import relation_codes as RC


class RelationCodesTest(unittest.TestCase):
    def test_context_norm_per_group(self):
        c = torch.randn(10, 4) * 3 + 2
        g = torch.tensor([0] * 5 + [1] * 5)
        z = RC.context_norm(c, g)
        for k in (0, 1):
            self.assertTrue(torch.allclose(z[g == k].mean(0), torch.zeros(4), atol=1e-5))

    def test_all_pairs_targets(self):
        ca, cb = torch.zeros(2, 3), torch.zeros(3, 3)
        ya, yb = torch.tensor([1, 4]), torch.tensor([2, 4, 9])
        _, _, d = RC.all_pairs(ca, ya, cb, yb)
        self.assertEqual((d - (RC.N - 1)).tolist(), [1, 3, 8, -2, 0, 5])

    def test_relational_identity_recovers_offsets(self):
        g = torch.Generator().manual_seed(0)
        m = RC.RelationNet('one', 8, g)
        axis = torch.zeros(RC.D); axis[0] = 1.0
        with torch.no_grad():
            m.r.copy_(torch.stack([d * axis for d in range(-8, 9)]))
            m.log_scale.fill_(3.0)
        mu = torch.stack([n * axis for n in range(1, 10)])
        c = torch.stack([n * axis for n in (2, 5, 9)])
        self.assertEqual(RC.relational_identity(m, c, mu).tolist(), [2, 5, 9])

    def test_frozen_arm_freezes_structure(self):
        m = RC.RelationNet('two-frozen', 8, torch.Generator().manual_seed(0))
        m.freeze_structure()
        self.assertFalse(m.r.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in m.G.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.E.parameters()))


if __name__ == '__main__':
    unittest.main()
