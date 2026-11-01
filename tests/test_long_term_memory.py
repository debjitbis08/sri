import unittest

import numpy as np
import torch

from src import long_term_memory as ltm


class Budget(unittest.TestCase):
    def test_width_fits_the_budget_for_every_setting(self):
        for f in ltm.FORMS:
            for dk, dv in ((16, 16), (32, 32)):
                self.assertLessEqual(ltm.param_floats(f, dk, dv) * ltm.CHARGED_COPIES * 4, ltm.BUDGET)

    def test_width_is_set_per_form(self):
        self.assertEqual(ltm.width('mlp'), ltm.width('k-sparse'))
        m16, m32 = ltm.Memory('mlp', 16, 16), ltm.Memory('mlp', 32, 32, projections='fixed')
        self.assertEqual(m16.M[0].shape, m32.M[0].shape)


class Behaviour(unittest.TestCase):
    def batch(self, n=8, d=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(n, d, generator=g)
        C = torch.randn(n, 9, d, generator=g)
        return q, C, torch.zeros(n, 9), torch.randint(9, (n,), generator=g)

    def test_starts_silent(self):
        for f in ltm.FORMS:
            m = ltm.Memory(f, 16, 16)
            q, C, lg, _ = self.batch()
            lg = lg + torch.randn(8, 9)
            self.assertTrue((m.combined(q, C, lg) == lg).all())

    def test_learns_a_paired_association(self):
        torch.manual_seed(0)
        q, C, lg, t = self.batch(n=9)
        t = torch.arange(9)
        for f in ltm.FORMS:
            m = ltm.Memory(f, 16, 16, eta=.1, beta=.9, C=1, p=.25)
            first = m.update(q, C, lg, t)
            for _ in range(600):
                last = m.update(q, C, lg, t)
            self.assertLess(last, first, f)

    def test_writes_little_when_readers_already_answer(self):
        q, C, _, t = self.batch()
        right = torch.zeros(8, 9).scatter(1, t[:, None], 30.)
        a = ltm.Memory('mlp', 16, 16, eta=.1, C=1, seed=1)
        a.update(q, C, right, t)
        self.assertLess(float(a.acc[0].abs().sum()) + float(a.S[-3].abs().sum()), 1e-6)

    def test_fixed_projections_never_move(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, projections='fixed', eta=.1, C=1)
        wk = m.W_K.clone()
        for _ in range(5):
            m.update(q, C, lg, t)
        self.assertTrue((m.W_K == wk).all())
        self.assertGreater(float(m.s.abs()), 0)

    def test_learned_projections_move(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=.1, C=1)
        wk = m.W_K.clone()
        for _ in range(5):
            m.update(q, C, lg, t)
        self.assertFalse((m.W_K == wk).all())

    def test_updates_apply_only_every_c(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=.1, C=4)
        s0 = m.s.clone()
        for _ in range(3):
            m.update(q, C, lg, t)
        self.assertTrue((m.s == s0).all())
        m.update(q, C, lg, t)
        self.assertFalse((m.s == s0).all())

    def test_chunking_matches_total_learning_at_small_steps(self):
        q, C, lg, t = self.batch()
        one, four = ltm.Memory('linear', 16, 16, eta=1e-4, beta=0., C=1, seed=2), ltm.Memory('linear', 16, 16, eta=1e-4, beta=0., C=4, seed=2)
        for _ in range(4):
            one.update(q, C, lg, t)
            four.update(q, C, lg, t)
        self.assertLess(abs(float(one.s) - float(four.s)), abs(float(one.s)) * .05 + 1e-7)

    def test_momentum_is_beta_to_the_c(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=.01, beta=.9, C=2)
        m.S[-3] = torch.ones_like(m.S[-3])
        for _ in range(2):
            m.update(q, C, lg, t)
        self.assertAlmostEqual(float(m.S[-3].abs().mean()), .81, delta=.3)

    def test_clipping(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=1., beta=0., C=1)
        m.s.data.fill_(1e6)
        before = [p.clone() for p in m.params]
        m.update(q, C, lg, t)
        moved = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(m.params, before)))
        self.assertLessEqual(float(moved), ltm.CLIP + 1e-3)

    def test_k_sparse_keeps_few_units(self):
        m = ltm.Memory('k-sparse', 16, 16, p=.05)
        H = ltm.width('k-sparse')
        k = torch.randn(4, ltm.D)
        h = torch.tanh(k @ m.M[0] + m.M[1])
        keep = max(1, round(.05 * H))
        top = h.abs().topk(keep, dim=-1).indices
        self.assertEqual(top.shape[1], keep)
        # different inputs reach different weights: the gradient is local
        loss = m.recall(k[:1]).sum()
        g = torch.autograd.grad(loss, m.M[2])[0]
        self.assertEqual(int((g.abs().sum(1) > 0).sum()), keep)

    def test_reading_leaves_state_alone_after_a_partial_chunk(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('mlp', 16, 16, eta=.1, C=4)
        m.update(q, C, lg, t)
        h = m.state_hash()
        m.combined(q, C, lg)
        self.assertEqual(h, m.state_hash())

    def test_state_hash_sees_every_component(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=.1, C=4)
        m.update(q, C, lg, t)
        h = m.state_hash()
        m.t += 1
        self.assertNotEqual(h, m.state_hash())
        m.t -= 1
        m.S[0] = m.S[0] + 1
        self.assertNotEqual(h, m.state_hash())
        m.S[0] = m.S[0] - 1
        m.acc[0] += 1
        self.assertNotEqual(h, m.state_hash())

    def test_a_partial_chunk_completes_during_a_delay(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=.1, beta=0., C=4)
        s0 = m.s.clone()
        m.update(q, C, lg, t)
        m.tick(); m.tick()
        self.assertTrue((m.s == s0).all())
        m.tick()
        self.assertFalse((m.s == s0).all())
        self.assertEqual(m.t, 4)

    def test_momentum_runs_on_during_a_delay(self):
        q, C, lg, t = self.batch()
        m = ltm.Memory('linear', 16, 16, eta=.1, beta=.9, C=1)
        m.update(q, C, lg, t)
        before = m.s.clone()
        m.tick()
        self.assertFalse((m.s == before).all())

    def test_chunk_with_a_delay_equals_scaled_gradient(self):
        q, C, lg, t = self.batch()
        a, b = ltm.Memory('linear', 16, 16, eta=.1, beta=0., C=2, seed=3), ltm.Memory('linear', 16, 16, eta=.1, beta=0., C=1, seed=3)
        a.update(q, C, lg, t); a.tick()
        b.update(q, C, lg, t)
        self.assertAlmostEqual(float(a.s), float(b.s) , places=5)

    def test_train_block_ticks_every_host_update(self):
        n = np.array([2, 0, 0, 1, 0])
        g = np.random.default_rng(0)
        tr = {'n': n, 'target': np.array([0, 1, 2], np.int8), 'logits': np.zeros((3, 9), np.float32), 'q_live': g.standard_normal((3, 16)).astype(np.float32), 't_live': g.standard_normal((3, 16)).astype(np.float32), 'c_live': g.standard_normal((2, 9, 16)).astype(np.float32)}
        m = ltm.Memory('linear', 16, 16, C=2)
        ltm.train_block(m, tr, 'live', 'live')
        self.assertEqual(m.t, 5)

    def test_train_block_keeps_keys_and_values_apart(self):
        n = np.array([1])
        g = np.random.default_rng(0)
        tr = {'n': n, 'target': np.array([0], np.int8), 'logits': np.zeros((1, 9), np.float32), 'q_frozen': g.standard_normal((1, 32)).astype(np.float32), 't_slow': g.standard_normal((1, 16)).astype(np.float32), 'c_slow': g.standard_normal((1, 9, 16)).astype(np.float32)}
        m = ltm.Memory('linear', 32, 16, C=1)
        ltm.train_block(m, tr, 'frozen', 'slow')
        self.assertEqual(m.t, 1)

    def test_accuracy_uses_candidate_counts(self):
        s = torch.tensor([[0., 5., 0.], [9., 0., 0.]])
        counts = torch.tensor([[3, 7, 1], [2, 4, 8]])
        self.assertEqual(ltm.accuracy(s, counts, torch.tensor([7, 4])), .5)

    def test_nearest_store(self):
        keys = torch.eye(3)
        out = ltm.nearest_store(keys, torch.tensor([5, 6, 7]), torch.tensor([[0., 1., 0.]]))
        self.assertEqual(int(out[0]), 6)


if __name__ == '__main__':
    unittest.main()
