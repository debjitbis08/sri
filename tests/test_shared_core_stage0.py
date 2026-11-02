import copy
import unittest

import numpy as np
import torch

from experiments import limited_memory as LM
from experiments import shared_core_stage0 as S0


def host(config, seed=7):
    S0.set_width(config)
    return S0.make_host(seed, config)


class Arms(unittest.TestCase):
    def test_sparse48_is_the_inherited_host(self):
        enc, model = host('sparse48')
        S0.set_width('sparse48')
        enc2, model2 = LM.make_model(7, 'sparse48')
        for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_reader_fan_in_and_count_do_not_grow_with_the_code(self):
        for config, width in (('sparse48', 16), ('wide32', 32), ('wide32-recon-l1', 32), ('split-recon-bg-l1', 16)):
            _, model = host(config)
            for _, seq in LM.readers(model):
                m = seq[0].struct.mask
                self.assertEqual(m.shape[0], 48)
                self.assertTrue((m[:, :width].sum(1) == 1).all())
                self.assertTrue((m[:, width:2 * width].sum(1) == 1).all())

    def test_split_starts_as_sparse48_and_readers_ignore_the_added_coordinates(self):
        _, base = host('sparse48')
        _, model = host('split-recon-bg-l1')
        self.assertTrue(torch.equal(model.proj['image'].weight[:16], base.proj['image'].weight))
        self.assertEqual(model.proj['image'].weight.shape, (32, 64))
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(4, 32, generator=g), torch.randn(4, 32, generator=g)
        a2, b2 = a.clone(), b.clone()
        a2[:, 16:], b2[:, 16:] = 9., -9.
        self.assertTrue(torch.equal(model.relation_logits(a, b, 1), model.relation_logits(a2, b2, 1)))
        self.assertTrue(torch.equal(model.predict(a, b), model.predict(a2, b2)))

    def test_added_rows_get_no_relational_gradient(self):
        _, model = host('split-recon-bg-l1')
        f = torch.randn(4, 64, generator=torch.Generator().manual_seed(1))
        c = model.proj['image'](f)
        model.relation_logits(c, c.flip(0), 1).sum().backward()
        grad = model.proj['image'].weight.grad
        self.assertEqual(float(grad[16:].abs().sum()), 0.)
        self.assertGreater(float(grad[:16].abs().sum()), 0.)

    def test_width_constants_match_the_solver(self):
        for config in S0.CONFIGS:
            _, model = host(config)
            self.assertEqual(model.inner_slice(), S0.P.INNER)
            self.assertEqual(S0.D.SIZES[2], model.sizes[2])

    def test_decoder_size_and_output(self):
        dec = S0.Decoder(32)
        self.assertEqual(sum(p.numel() for p in dec.parameters()), 211265)
        out = dec(torch.zeros(3, 32))
        self.assertEqual(out.shape, (3, 28, 28))
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_predict_logits_is_unchanged_on_a_full_code(self):
        _, model = host('wide32')
        g = torch.Generator().manual_seed(2)
        pred, cand = torch.randn(3, 32, generator=g), torch.randn(3, 9, 32, generator=g)
        wrapped = S0.predict_logits_on_read(S0.ORIG_PREDICT_LOGITS)
        self.assertTrue(torch.equal(wrapped(model, pred, cand), S0.ORIG_PREDICT_LOGITS(model, pred, cand)))


class Readings(unittest.TestCase):
    def test_address_reading_on_separated_and_collapsed_codes(self):
        g = torch.Generator().manual_seed(3)
        centres = torch.randn(18, 32, generator=g)
        counts = np.repeat(np.arange(1, 10), 2)
        train = centres[:, None] + .01 * torch.randn(18, 8, 32, generator=g)
        ev = centres[:, None] + .01 * torch.randn(18, 2, 32, generator=g)
        r = S0.address_reading(train, ev, counts)
        self.assertEqual(r['identity'], 1.)
        self.assertEqual(r['store'], 1.)
        line = torch.ones(18, 8, 32)
        r = S0.address_reading(line, torch.ones(18, 2, 32), counts)
        self.assertLess(r['identity'], .2)


class Lock(unittest.TestCase):
    def table(self, change):
        res = {}
        for c in S0.CONFIGS:
            for s in S0.SEEDS:
                addr = {'identity': .2, 'store': .2}
                res[f'{c}/seed{s}'] = {'address': {str(S0.LOADS[0]): dict(addr)}, 'address_before_projection': {str(S0.LOADS[0]): dict(addr)}, 'transfer8_mean': .5,
                                       'transfer8': {'chinese_withheld_matching': .1, 'devanagari_withheld_matching': .3}}
        change(res)
        return S0.lock_table(res)

    def passing(self, res, config, identity=.9, transfer=.5):
        for s in S0.SEEDS:
            res[f'{config}/seed{s}']['address'][str(S0.LOADS[0])].update(identity=identity, store=identity)
            res[f'{config}/seed{s}']['transfer8_mean'] = transfer

    def test_no_arm_passes_gives_conjunctive_on_the_frozen_encoder(self):
        d = self.table(lambda r: None)['decision']
        self.assertEqual((d['route'], d['host'], d['address_source']), ('conjunctive', 'sparse48', 'frozen encoder'))

    def test_identity_without_relations_does_not_pass(self):
        d = self.table(lambda r: self.passing(r, 'wide32-recon-bg-l1', transfer=.4))['decision']
        self.assertEqual(d['route'], 'conjunctive')
        self.assertEqual(d['candidate'], 'wide32-recon-bg-l1')

    def test_cost_then_transfer_then_lambda_break_ties(self):
        def change(r):
            self.passing(r, 'wide32-recon-bg-l1', transfer=.5)
            self.passing(r, 'split-recon-bg-l1', transfer=.55)
            self.passing(r, 'split-recon-bg-l0.1', transfer=.55)
        self.assertEqual(self.table(change)['decision']['host'], 'split-recon-bg-l0.1')
        self.assertEqual(self.table(lambda r: (change(r), self.passing(r, 'wide32-recon-l1')))['decision']['host'], 'wide32-recon-l1')


if __name__ == '__main__':
    unittest.main()
