import sys
import unittest

import numpy as np
import torch


def modules():
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        from experiments import sense_core as SC
    finally:
        sys.argv = argv
    from src import shared_output as SO
    return SC, SO


class FakeImage(torch.nn.Module):
    """from_codes like FrozenImage: raw codes (numpy) -> a trained projection."""

    def __init__(self, width, out):
        super().__init__()
        self.head = torch.nn.Linear(width, out)

    def from_codes(self, raw):
        return self.head(torch.from_numpy(np.asarray(raw, np.float32)))


def build(arm, seed=300):
    SC, SO = modules()
    SC.LM.SB.install()  # the standard network's solvers, as the runner installs them
    enc, model = SC.LM.make_model(seed, 'sparse48')
    width = 12
    enc = {'image': FakeImage(width, enc['image'].out.out_features), 'sequence': enc['sequence']}
    raw = np.random.default_rng(seed).normal(size=(18, width)).astype(np.float32)
    SO.convert(model, arm, enc, raw, 7)
    return SC, SO, enc, model


class Readout(unittest.TestCase):
    def test_parameters_match_across_arms(self):
        counts = {}
        for arm in ('gp', 'gp-grounded', 'common', 'common-raw'):
            _, SO, _, model = build(arm)
            counts[arm] = SO.reader_parameters(model)
        self.assertEqual(counts['gp'], counts['gp-grounded'])
        self.assertEqual(counts['common'], counts['common-raw'])
        self.assertLessEqual(abs(counts['common'] - counts['gp']), 30)  # the continuation token and the empty slot against b_c and scale

    def test_first_layers_keep_sparse48_masks(self):
        SC, _, _, gp = build('gp')
        _, _, _, common = build('common')
        for (_, a), (_, b) in zip(SC.LM.readers(gp), SC.LM.readers(common)):
            torch.testing.assert_close(a[0].struct.mask, b[0].struct.mask)
            torch.testing.assert_close(a[0].struct.weight, b[0].struct.weight)

    def test_relational_losses_do_not_reach_identification(self):
        """Gradient rule: a relational loss gives identification no gradient. In the grounded arms the raw item gets none, and M's states get some."""
        for arm in ('gp-grounded', 'common'):
            SC, SO, enc, model = build(arm)
            M = model.encode_participants(None)
            model._M = M
            M.retain_grad()
            x = torch.randn(5, 16, requires_grad=True)
            y = torch.randn(5, 16, requires_grad=True)
            model.relation_logits(x, y, SO.NEXT).sum().backward()
            self.assertIsNone(x.grad)
            self.assertIsNone(y.grad)
            self.assertGreater(float(M.grad.abs().sum()), 0)
            # and identification itself is still trained by the same read
            model.zero_grad()
            x2 = torch.randn(5, 16, requires_grad=True)
            model.relation_logits(x2, y.detach(), SO.SAME).sum().backward()
            self.assertGreater(float(x2.grad.abs().sum()), 0)

    def test_raw_arms_pass_gradient_to_the_items(self):
        for arm in ('gp', 'common-raw'):
            _, SO, _, model = build(arm)
            x = torch.randn(5, 16, requires_grad=True)
            model.relation_logits(x, torch.randn(5, 16), SO.NEXT).sum().backward()
            self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_grounding_substitutes_identified_items(self):
        """Substitutivity: two inputs identified with the same participant (hard) give identical relational reads, whatever their own codes."""
        _, SO, _, model = build('common')
        with torch.no_grad():
            M = model.participants()
            model.hard = True
            x = M[3:4] + 0.0
            lg = model.identify(x, M)
            y = M[lg.argmax(1)] + 1e-3 * torch.randn_like(x)
            if model.identify(y, M).argmax(1) == lg.argmax(1):
                a = model.cont_logits(x, x, M[None])
                b = model.cont_logits(y, x, M[None])
                torch.testing.assert_close(a, b)

    def test_continuation_logits_shapes_and_predict(self):
        for arm in ('gp', 'gp-grounded', 'common', 'common-raw'):
            _, _, _, model = build(arm)
            with torch.no_grad():
                p, c, cand = torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 9, 16)
                self.assertEqual(tuple(model.cont_logits(p, c, cand).shape), (4, 9))
                self.assertEqual(tuple(model.predict(p, c).shape), (4, 16))


class Helpers(unittest.TestCase):
    def test_threshold_maximizes_balanced_accuracy(self):
        _, SO = modules()
        t, bal = SO.threshold([2., 3., 4.], [0., 1., 2.5])
        self.assertEqual(t, 2.)  # t = 2 and t = 3 both give 5/6; ties take the lowest
        self.assertAlmostEqual(bal, 5 / 6)
        t, bal = SO.threshold([5., 6.], [0., 1.])
        self.assertEqual((t, bal), (5., 1.))

    def test_acceptance_weights_classes_equally(self):
        _, SO = modules()
        lg = torch.zeros(2, 9)
        target = torch.tensor([0, 4])
        # at zero logits each term is log 2; positive weight 8 balances one positive against eight negatives
        self.assertAlmostEqual(float(SO.acceptance(lg, target)), float(np.log(2) * (8 + 8) / 9), places=5)


class Audit(unittest.TestCase):
    def test_names_never_reach_relational_rows(self):
        """Names have their own kind, so next negatives, comparison draws and sequence rows, which select by kind or episode, cannot reach them."""
        argv = sys.argv
        sys.argv = [argv[0]]
        try:
            from experiments import shared_output as SOX
        finally:
            sys.argv = argv
        SOX.S0.prepare()
        SOX.W.update(names={'counts': np.repeat(np.arange(1, 10), 5)}, name_rng=np.random.default_rng(0))
        SOX.LM.CURRENT['step'] = 2 * SOX.D.STEPS
        rng = np.random.default_rng(1)
        eps = SOX.P.episodes('devanagari_digits', (), rng, 4)
        items, pairs, match, seqs = SOX.batch_material(eps, SOX.D.context_episodes(rng, 2), 'taps')
        named = {k for k, it in enumerate(items) if it[0] == 'name'}
        self.assertEqual(len(named), 9)
        self.assertFalse(named & {i for p in pairs for i in p})
        self.assertFalse(named & {i for s in seqs for _, st in s for i in st.values()})
        self.assertEqual({m[0] for m in match} & named, named)
        SOX.LM.CURRENT['step'] = 3 * SOX.D.STEPS  # the delay: no names
        items, *_ = SOX.batch_material([], SOX.D.context_episodes(rng, 2), 'taps')
        self.assertFalse(any(it[0] == 'name' for it in items))


if __name__ == '__main__':
    unittest.main()
