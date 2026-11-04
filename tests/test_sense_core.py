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

    def stats(self, models, rows=slice(None)):
        raw = np.concatenate([self.codes[n] for n in models], 1)[rows]
        return {'mean': raw.mean(0), 'std': raw.std(0) + 1e-6, 'provenance': {'calibration': [('test', len(raw))]}}

    def test_looks_up_by_content_and_standardizes_with_fixed_statistics(self):
        models = ('dinov2-s', 'trocr-s')
        st = self.stats(models, slice(0, 3))  # fitted on a calibration subset, not on everything in the table
        enc = self.SC.FrozenImage(models, 'linear', 7, self.table, st)
        order = [3, 0, 3, 4]
        self.assertEqual(list(enc.lookup(torch.from_numpy(self.imgs[order]))), order)
        raw = np.concatenate([self.codes[n] for n in models], 1)
        out = enc(torch.from_numpy(self.imgs[order]))
        expect = enc.head(torch.from_numpy(((raw - st['mean']) / st['std']).astype(np.float32)))[order]
        torch.testing.assert_close(out, expect)
        self.assertEqual(enc.misses, 0)

    def test_checkpoint_carries_statistics_and_provenance(self):
        models = ('dinov2-s', 'trocr-s')
        enc = self.SC.FrozenImage(models, 'mlp', 7, self.table, self.stats(models, slice(0, 3)))
        sd = enc.state_dict()
        self.assertEqual(sorted(k for k in sd if not k.startswith('head.')), ['_extra_state', 'mean', 'std'])
        x = torch.from_numpy(self.imgs[[1, 4]])
        want = enc(x)
        # a fresh encoder over an expanded table, with different statistics, reads identically once the checkpoint is loaded
        rng = np.random.default_rng(1)
        extra = {n: np.concatenate([self.codes[n], rng.normal(size=(50, 384)).astype(np.float32)]) for n in models}
        other = self.SC.FrozenImage(models, 'mlp', 7, {'index': dict(self.table['index']), **extra}, {'mean': np.zeros(768), 'std': np.ones(768), 'provenance': None})
        other(x)  # builds its standardized table with the old statistics first
        other.load_state_dict(sd)
        torch.testing.assert_close(other(x), want)
        self.assertEqual(other.provenance, {'calibration': [('test', 3)]})

    def test_checkpoint_for_other_models_is_refused(self):
        enc = self.SC.FrozenImage(('dinov2-s', 'trocr-s'), 'linear', 7, self.table, self.stats(('dinov2-s', 'trocr-s')))
        sd = enc.state_dict()
        other = self.SC.FrozenImage(('trocr-s', 'dinov2-s'), 'linear', 7, self.table, self.stats(('trocr-s', 'dinov2-s')))
        with self.assertRaises(ValueError):
            other.load_state_dict(sd)

    def test_only_the_projection_is_trained(self):
        enc = self.SC.FrozenImage(('dinov2-s',), 'mlp', 7, {'index': self.table['index'], 'dinov2-s': self.codes['dinov2-s']}, self.stats(('dinov2-s',)))
        self.assertEqual(sorted(n for n, _ in enc.named_parameters()), ['head.0.bias', 'head.0.weight', 'head.2.bias', 'head.2.weight'])
        self.assertEqual(sum(p.numel() for p in enc.parameters()), 384 * 128 + 128 + 128 * 7 + 7)


if __name__ == '__main__':
    unittest.main()
