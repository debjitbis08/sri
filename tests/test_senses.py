import unittest

import numpy as np
import torch
import torch.nn.functional as Fn

from src import senses as SN


def old_joined(za, zb, wa=.5, wb=.5):
    """experiments.general_encoder.joined as it was at commit b2e3750."""
    w = lambda x: np.asarray(x, np.float32)[..., None] if np.ndim(x) else np.float32(x)
    return np.sqrt(2.) * np.concatenate([w(wa) * za, w(wb) * zb], -1).astype(np.float32)


def old_train_gate(Ra, Rb, Ca, Cb, seed, steps=300, batch=512, lr=1e-2, temp=.1):
    """experiments.general_encoder.train_gate as it was at commit b2e3750."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    W = torch.nn.Linear(Ra.shape[1] + Rb.shape[1], 2)
    with torch.no_grad():
        W.weight.copy_(.01 * torch.randn(W.weight.shape, generator=g, device='cpu'))
        W.bias.zero_()
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    X = torch.as_tensor(np.concatenate([Ra, Rb], 1))
    A1, A2, B1, B2 = (torch.as_tensor(c) for c in (Ca[0], Ca[1], Cb[0], Cb[1]))
    n = len(X)
    for _ in range(steps):
        idx = torch.randperm(n, generator=g, device='cpu')[:batch].to(X.device)
        w = W(X[idx]).softmax(1)
        S = w[:, :1] * (A1[idx] @ A2[idx].T) + w[:, 1:] * (B1[idx] @ B2[idx].T)
        loss = Fn.cross_entropy(S / temp, torch.arange(len(idx), device=X.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    W.requires_grad_(False)
    return lambda ra, rb: W(torch.as_tensor(np.concatenate([ra, rb], 1))).softmax(1).detach().cpu().numpy(), float(loss)


def units(rng, n, d):
    return SN.unit(rng.normal(size=(n, d)).astype(np.float32))


class Joining(unittest.TestCase):
    def test_two_models_match_the_earlier_join_exactly(self):
        rng = np.random.default_rng(0)
        za, zb = units(rng, 7, 5), units(rng, 7, 3)
        w = rng.uniform(size=7).astype(np.float32)
        np.testing.assert_array_equal(SN.join([za, zb], [.5, .5]), old_joined(za, zb))
        np.testing.assert_array_equal(SN.join([za, zb], [w, 1 - w]), old_joined(za, zb, w, 1 - w))

    def test_query_weights_give_weighted_cosines_for_three_models(self):
        rng = np.random.default_rng(1)
        k, q = [units(rng, 6, d) for d in (4, 5, 3)], [units(rng, 6, d) for d in (4, 5, 3)]
        K = SN.join(k)
        np.testing.assert_allclose(np.linalg.norm(K, axis=1), 1, atol=1e-6)
        w = rng.dirichlet(np.ones(3), size=6).astype(np.float32)
        got = (SN.join(q, [w[:, i] for i in range(3)]) * K).sum(1)
        np.testing.assert_allclose(got, sum(w[:, i] * (q[i] * k[i]).sum(1) for i in range(3)), atol=1e-5)

    def test_a_missing_model_adds_nothing(self):
        rng = np.random.default_rng(2)
        j = SN.Joint(('dinov2-s', 'trocr-s'))
        a, b = rng.normal(size=(4, 384)).astype(np.float32), rng.normal(size=(4, 384)).astype(np.float32)
        old = j.key({'dinov2-s': a}, None)
        np.testing.assert_array_equal(old[:, 384:], 0)
        q = j.key({'dinov2-s': a, 'trocr-s': b}, None, [.5, .5])
        np.testing.assert_allclose((q * old).sum(1), .5, atol=1e-5)

    def test_centres_per_model_and_group(self):
        j = SN.Joint(('dinov2-s',))
        z = np.ones((2, 384), np.float32)
        j.set_centre('dinov2-s', 'natural', np.full(384, .5, np.float32))
        np.testing.assert_allclose(j.code('dinov2-s', 'natural', z), SN.centred(z, .5))
        np.testing.assert_allclose(j.code('dinov2-s', None, z), SN.unit(z))


class Gating(unittest.TestCase):
    def test_two_model_gate_matches_the_earlier_gate_exactly(self):
        rng = np.random.default_rng(3)
        Ra, Rb = units(rng, 64, 6), units(rng, 64, 4)
        Ca, Cb = np.stack([units(rng, 64, 6), units(rng, 64, 6)]), np.stack([units(rng, 64, 4), units(rng, 64, 4)])
        old, old_loss = old_train_gate(Ra, Rb, Ca, Cb, 7, steps=20, batch=32)
        new, new_loss = SN.Gate.train([Ra, Rb], [Ca, Cb], 7, steps=20, batch=32)
        self.assertEqual(old_loss, new_loss)
        np.testing.assert_array_equal(old(Ra, Rb), new([Ra, Rb]))
        self.assertEqual(new.parameters(), 2 * 10 + 2)

    def test_gate_learns_to_trust_the_model_that_tells_views_apart(self):
        rng = np.random.default_rng(4)
        good, bad = units(rng, 128, 8), units(rng, 1, 8).repeat(128, 0)
        views = [np.stack([good, good]), np.stack([bad, bad])]
        g, _ = SN.Gate.train([good, bad], views, 0, steps=100, batch=64)
        self.assertGreater(g([good, bad])[:, 0].mean(), .9)


class Interface(unittest.TestCase):
    def test_registered_models_declare_what_the_plan_asks(self):
        for name in ('dinov2-s', 'trocr-s'):
            d = SN.get(name).describe()
            for k in ('name', 'sense', 'layer', 'dim', 'licence', 'path', 'sha256', 'inputs', 'reference'):
                self.assertIn(k, d)
            self.assertFalse(d['licence']['verified'])

    def test_registering_a_different_model_under_a_taken_name_fails(self):
        with self.assertRaises(ValueError):
            SN.register(SN.SenseModel('dinov2-s', 'sight', 'x', 'cls', 1))

    def test_encoder_chunks_and_counts_cost(self):
        spec = SN.SenseModel('test-mean', 'sight', 'x', 'code', 1)
        enc = SN.Encoder(spec, lambda x: {'code': x.mean((1, 2))[:, None], 'other': x}, chunk=3, parameters=0)
        x = np.random.default_rng(5).uniform(size=(8, 4, 4)).astype(np.float32)
        np.testing.assert_allclose(enc(x)[:, 0], x.mean((1, 2)), rtol=1e-6)
        self.assertEqual(enc.cost()['inputs'], 8)


@unittest.skipUnless(all(SN.get(n).impl().WEIGHTS.exists() for n in ('dinov2-s', 'trocr-s')), 'weights not fetched')
class RealModels(unittest.TestCase):
    def test_codes_equal_the_earlier_path(self):
        from experiments import dinov2_frozen as DINO
        from experiments import general_encoder as GE
        from experiments import trocr_frozen as TR
        x = np.random.default_rng(6).uniform(size=(3, 28, 28)).astype(np.float32)
        sense = SN.Sense(('dinov2-s', 'trocr-s'), chunk=64)
        got = sense(x)
        old = {'dinov2-s': GE.run_layers(DINO.layers(DINO.load('cpu')), x, keep=('patch',), size=64)['patch'],
               'trocr-s': GE.run_layers(TR.layers(TR.load('cpu'), invert=True), x, keep=('cls',), size=64)['cls']}
        for n in old:
            np.testing.assert_array_equal(got[n], old[n])
        self.assertTrue(all(c['cost']['parameters'] > 2e7 for c in sense.describe().values()))


if __name__ == '__main__':
    unittest.main()
