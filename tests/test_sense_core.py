import sys
import unittest

import numpy as np
import torch


def module():
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        from experiments import sense_core as SC
    finally:
        sys.argv = argv
    return SC


class FrozenImageEncoder(unittest.TestCase):
    def setUp(self):
        self.SC = module()
        rng = np.random.default_rng(0)
        self.imgs = rng.uniform(size=(5, 28, 28)).astype(np.float32)
        self.codes = {'dinov2-s': rng.normal(size=(5, 384)).astype(np.float32), 'trocr-s': rng.normal(size=(5, 384)).astype(np.float32)}
        self.table = {'index': {self.SC.image_key(r): i for i, r in enumerate(self.imgs)}, **self.codes}

    def test_looks_up_by_content_and_standardizes_with_fixed_statistics(self):
        enc = self.SC.FrozenImage(('dinov2-s', 'trocr-s'), 'linear', 7, self.table)
        order = [3, 0, 3, 4]
        self.assertEqual(list(enc.lookup(torch.from_numpy(self.imgs[order]))), order)
        raw = np.concatenate([self.codes['dinov2-s'], self.codes['trocr-s']], 1)
        out = enc(torch.from_numpy(self.imgs[order]))
        expect = enc.head(torch.from_numpy(((raw - raw.mean(0)) / (raw.std(0) + 1e-6)).astype(np.float32)))[order]
        torch.testing.assert_close(out, expect)
        self.assertEqual(enc.misses, 0)

    def test_only_the_projection_is_saved(self):
        enc = self.SC.FrozenImage(('dinov2-s',), 'mlp', 7, {'index': self.table['index'], 'dinov2-s': self.codes['dinov2-s']})
        self.assertEqual(sorted(enc.state_dict()), ['head.0.bias', 'head.0.weight', 'head.2.bias', 'head.2.weight'])
        self.assertEqual(sum(p.numel() for p in enc.parameters()), 384 * 128 + 128 + 128 * 7 + 7)


if __name__ == '__main__':
    unittest.main()
