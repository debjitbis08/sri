import unittest

import numpy as np

from src import relevant_update as ru
from src import surprise_lifecycle as sl


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Base:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        fam = unit(self.rng.standard_normal(sl.D_C))
        self.fam = fam

    def family_cues(self, fam, n):
        return unit(fam * np.sqrt(1 / 3) + np.sqrt(2 / 3) * unit(self.rng.standard_normal((n, sl.D_C)))).astype(np.float32)

    def base(self):
        rng = np.random.default_rng(1)
        C = unit(rng.standard_normal((80, sl.D_C))).astype(np.float32)
        O = unit(rng.standard_normal((80, sl.D_O))).astype(np.float32)
        w, V = np.linalg.eigh(C.T @ C)
        return {'R': np.zeros((sl.D_C, sl.D_O), np.float32), 'slot_c': np.eye(sl.D_C, dtype=np.float32)[:1], 'slot_o': np.eye(sl.D_O, dtype=np.float32)[:1], 'rule_C': C, 'rule_O': O,
                'rule_labels': np.repeat(np.arange(8), 10), 'buffer_order': np.arange(80), 'eigvecs': V[:, ::-1].astype(np.float32)}


class Rules(unittest.TestCase):
    def setUp(self):
        self.b = Base()
        self.base = self.b.base()
        self.o = unit(self.b.rng.standard_normal(sl.D_O)).astype(np.float32)

    def set(self, k=4):
        return self.b.family_cues(self.b.fam, k), np.tile(self.o, (k, 1))

    def generalizes(self, rule):
        return float((sl.unit(rule.answers(self.b.family_cues(self.b.fam, 16))) @ self.o).mean())

    def test_dual_ridge_matches_the_primal(self):
        C, O = self.set()
        lam = .5
        primal = np.linalg.solve(C.T @ C + lam * np.eye(sl.D_C), C.T @ O)
        np.testing.assert_allclose(ru.dual_ridge(C, O, lam), primal, atol=1e-3)

    def test_each_linear_rule_learns_a_family(self):
        for cls, cfg in ((ru.RankOne, {}), (ru.Ridge, {'lam': .1}), (ru.RidgeCycled, {'lam': .1}), (ru.RidgeRehearse, {'lam': .1, 'r': 8, 'w': .25}), (ru.RidgeInterfered, {'lam': .1, 'B': 16, 'r': 8, 'w': .25}), (ru.RidgeOrthogonal, {'lam': .1, 'q': 4}), (ru.Expert, {'tau_x': .2})):
            r = cls(self.base, cfg)
            r.begin_trial()
            r.update(*self.set())
            self.assertGreater(self.generalizes(r), .3, cls.__name__)

    def test_ridge_cycled_approaches_ridge_at_small_lambda(self):
        C, O = self.set(4)
        a = ru.Ridge(self.base, {'lam': 1e-3})
        a.update(C, O)
        b = ru.RidgeCycled(self.base, {'lam': 1e-3})
        for _ in range(10):
            b.update(C, O)
        np.testing.assert_allclose(C @ a.R, C @ b.R, atol=.05)

    def test_orthogonal_update_leaves_the_protected_directions_alone(self):
        r = ru.RidgeOrthogonal(self.base, {'lam': .1, 'q': 8})
        r.update(*self.set())
        np.testing.assert_allclose(r.U.T @ r.R, 0, atol=1e-4)

    def test_rehearse_buffer_is_fixed_and_even(self):
        r = ru.RidgeRehearse(self.base, {'lam': .1, 'r': 16, 'w': 1.})
        first = r.bc.copy()
        r.update(*self.set())
        np.testing.assert_array_equal(first, r.bc)
        idx = ru.even_indices(self.base['rule_labels'], 16, self.base['buffer_order'])
        self.assertTrue((np.bincount(self.base['rule_labels'][idx]) == 2).all())
        self.assertEqual(r.extra_bytes(16), 16 * 512)

    def test_rehearsal_protects_a_nearby_old_answer(self):
        base = dict(self.base)
        C, O = self.set(4)
        base['rule_C'] = np.tile(C[:1] * 0 + self.b.family_cues(self.b.fam, 8), (10, 1))
        base['rule_O'] = np.tile(unit(self.b.rng.standard_normal((8, sl.D_O))).astype(np.float32), (10, 1))
        base['R'] = sl.fit_reader(base['rule_C'], base['rule_O'], .1)
        plain = ru.Ridge(base, {'lam': .1})
        reh = ru.RidgeRehearse(base, {'lam': .1, 'r': 80, 'w': 4.})
        plain.update(C, O)
        reh.update(C, O)
        loss = lambda r: float(((base['rule_C'] @ r.R - base['rule_O']) ** 2).sum())
        self.assertLess(loss(reh), loss(plain))

    def test_keep_retains_an_earlier_rule_better_than_ridge(self):
        rng = np.random.default_rng(4)
        fams = [unit(rng.standard_normal(sl.D_C)) for _ in range(6)]
        fams = [unit(.8 * fams[0] + .6 * f) for f in fams]        # correlated families, as real keys are
        outs = unit(rng.standard_normal((6, sl.D_O))).astype(np.float32)
        plain, keep = ru.Ridge(self.base, {'lam': .01}), ru.RidgeKeep(self.base, {'lam': .01, 'r': 8, 'w': .25})
        for f, o in zip(fams, outs):
            C = self.b.family_cues(f, 4)
            O = np.tile(o, (4, 1))
            plain.update(C, O)
            keep.update(C, O)
        test = self.b.family_cues(fams[0], 32)
        score = lambda r: float((sl.unit(test @ r.R) @ outs[0]).mean())
        self.assertGreater(score(keep), score(plain))
        self.assertEqual(keep.extra_bytes(6), (8 + 24) * 512)

    def test_interfered_picks_the_most_damaged(self):
        r = ru.RidgeInterfered(self.base, {'lam': .1, 'B': 16, 'r': 8, 'w': 1.})
        self.assertEqual(len(r.bc), 16)
        self.assertEqual(r.extra_bytes(16), 16 * 512)
        r.update(*self.set())

    def test_expert_leaves_r_and_gates_locally(self):
        r = ru.Expert(self.base, {'tau_x': .3})
        before = r.R.copy()
        r.update(*self.set())
        np.testing.assert_array_equal(before, r.R)
        far = unit(self.b.rng.standard_normal((16, sl.D_C))).astype(np.float32)
        np.testing.assert_allclose(r.answers(far), far @ r.R, atol=1e-6)
        self.assertGreater(self.generalizes(r), .5)
        self.assertEqual(r.extra_bytes(32), 32 * 512)
        self.assertEqual(r.extra_bytes(32, 8), 131072)

    def test_answers_change_no_state(self):
        r = ru.Ridge(self.base, {'lam': .1})
        r.update(*self.set())
        before = r.R.copy()
        r.answers(self.set()[0])
        np.testing.assert_array_equal(before, r.R)


class Neural(unittest.TestCase):
    def setUp(self):
        self.b = Base()
        self.C = self.b.family_cues(self.b.fam, 4)
        self.O = unit(self.b.rng.standard_normal((4, sl.D_O))).astype(np.float32)

    def net(self, seed=2):
        return sl.MLP(6, 1, np.random.default_rng(seed))

    def test_omega_and_seq_have_the_same_exposures(self):
        a, b = ru.Omega(self.net(), dict(alpha=0., eta=.01)), ru.OmegaSeq(self.net(), dict(alpha=0., eta=.01))
        a.begin_trial()
        b.begin_trial()
        a.update(self.C, self.O)
        b.update(self.C, self.O)
        self.assertEqual(a.ops, b.ops)
        self.assertEqual(a.ops, 2 * 10 * a.net.count())

    def test_seq_coincides_with_omega_at_small_steps(self):
        a, b = ru.Omega(self.net(), dict(alpha=0., eta=1e-4)), ru.OmegaSeq(self.net(), dict(alpha=0., eta=1e-4))
        n0 = self.net()
        a.begin_trial()
        b.begin_trial()
        a.update(self.C, self.O)
        b.update(self.C, self.O)
        move = [np.abs(x - y).max() for x, y in zip(a.net.params(), n0.params())]
        gap = [np.abs(x - y).max() for x, y in zip(a.net.params(), b.net.params())]
        self.assertLess(max(gap), 1e-5)
        self.assertGreater(max(move), 1e-5)

    def test_seq_differs_from_omega_at_large_steps(self):
        a, b = ru.Omega(self.net(), dict(alpha=0., eta=.3)), ru.OmegaSeq(self.net(), dict(alpha=0., eta=.3))
        a.begin_trial()
        b.begin_trial()
        a.update(self.C, self.O)
        b.update(self.C, self.O)
        self.assertGreater(max(np.abs(x - y).max() for x, y in zip(a.net.params(), b.net.params())), 1e-3)

    def test_decay_is_one_minus_alpha_and_total_matches(self):
        a = ru.OmegaSeq(self.net(), dict(alpha=.1, eta=0.))
        c = ru.Omega(self.net(), dict(alpha=.1, eta=0.))
        p0 = [x.copy() for x in a.net.params()]
        a.begin_trial()
        c.begin_trial()
        a.update(self.C, self.O)
        c.update(self.C, self.O)
        np.testing.assert_allclose(a.net.params()[0], p0[0] * .9 ** 4, atol=1e-5)
        np.testing.assert_allclose(c.net.params()[0], p0[0] * .9 ** 4, atol=1e-5)

    def test_momentum_persists_within_a_trial_and_resets(self):
        t = ru.Titans(self.net(), dict(alpha=0., eta=.01, beta=.9))
        t.begin_trial()
        t.update(self.C, self.O)
        self.assertGreater(float(np.abs(t.S[0]).sum()), 0)
        t.begin_trial()
        self.assertEqual(float(np.abs(t.S[0]).sum()), 0)

    def test_titans_learns(self):
        t = ru.Titans(self.net(), dict(alpha=0., eta=.05, beta=0.))
        t.begin_trial()
        before = float(((t.answers(self.C) - self.O) ** 2).sum())
        for _ in range(20):
            t.update(self.C, self.O)
        self.assertLess(float(((t.answers(self.C) - self.O) ** 2).sum()), before)


class Trials(unittest.TestCase):
    def setUp(self):
        self.t = sl.make_tables(np.random.default_rng(1))
        self.table = ru.rule_outcomes(np.random.default_rng(2), self.t)
        self.r = sl.symbol_ranges()

    def test_rule_outcomes_avoid_known_rules(self):
        known = {o for _, o in self.t['known_rules']}
        self.assertEqual(len(self.table), 36)
        self.assertFalse(set(self.table.values()) & known)

    def trial(self, p=0., kind=None, L=16, k=4):
        return ru.make_trial(np.random.default_rng(3), self.table, L, k, p, kind)

    def test_clean_set_shape(self):
        tr = self.trial()
        self.assertEqual(tr['F'].shape, (16, 4))
        self.assertEqual(len(set(tr['families'])), 16)
        self.assertEqual(len(tr['unused']), 20)
        for j, f in enumerate(tr['families']):
            self.assertTrue((tr['F'][j] == f).all())
            self.assertTrue((tr['O'][j] == self.table[int(f)]).all())
        self.assertTrue((tr['I'] != tr['J']).all())
        self.assertTrue(np.isin(tr['I'], self.r['items']).all())

    def test_contaminants_count_and_source(self):
        for k, p, n in ((4, .25, 1), (4, .5, 2), (8, .25, 2), (8, .5, 4)):
            for kind in ('unrelated', 'same-outcome', 'same-family'):
                tr = self.trial(p, kind, k=k)
                self.assertTrue((tr['contaminated'].sum(1) == n).all())
                bad = tr['contaminated']
                fam = tr['F'][bad]
                if kind == 'same-family':
                    self.assertFalse(np.isin(fam, self.r['binding']).any())
                    for j, f in enumerate(tr['families']):
                        self.assertTrue((tr['O'][j][bad[j]] != self.table[int(f)]).all())
                else:
                    self.assertTrue(np.isin(fam, self.r['binding']).all())
                    self.assertFalse(np.isin(fam, tr['unused']).any())
                if kind == 'same-outcome':
                    for j, f in enumerate(tr['families']):
                        self.assertTrue((tr['O'][j] == self.table[int(f)]).all())

    def test_draws_depend_only_on_the_generator(self):
        a, b = self.trial(), self.trial()
        np.testing.assert_array_equal(a['I'], b['I'])
        np.testing.assert_array_equal(a['render'], b['render'])


if __name__ == '__main__':
    unittest.main()
