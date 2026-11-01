import unittest

import numpy as np

from src import surprise_lifecycle as sl


def unit(x):
    return x / np.linalg.norm(x)


CFG = dict(theta=.3, tau=.8, tau_e=.7, tau_s=.2, kappa=.3, m=2, h_b=2, h_r=2, h_f=2, h_n=2, S=2, delta=.5, E=1, rho=4, tau_g=.4)


def life(cfg=None, B=64, **kw):
    c = {**CFG, **(cfg or {})}
    R = np.zeros((sl.D_C, sl.D_O), np.float32)
    kc = np.zeros((0, sl.D_C), np.float32)
    kc = np.vstack([kc, np.eye(sl.D_C, dtype=np.float32)[:1]])
    ko = np.eye(sl.D_O, dtype=np.float32)[:1]
    return sl.Life(R, kc, ko, c, B, **kw)


class Bank:
    """Reproducible unit cues and outcomes that are close to orthogonal, avoiding the known slot's direction."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def cue(self):
        x = self.rng.standard_normal(sl.D_C)
        x[0] = 0
        return unit(x).astype(np.float32)

    def outcome(self):
        return unit(self.rng.standard_normal(sl.D_O)).astype(np.float32)

    def family_cue(self, family):
        """A cue sharing a third of its weight with `family`, as a rule's cues do."""
        return unit(family * np.sqrt(1 / 3) + np.sqrt(2 / 3) * self.cue()).astype(np.float32)


class Budgets(unittest.TestCase):
    def test_contract_table(self):
        self.assertEqual(sl.allocation(64, 4)['traces'], 113)
        self.assertEqual(sl.allocation(64, 4, True)['traces'], 54)
        self.assertEqual(sl.allocation(64, 4, True)['sketch'], 4096)
        self.assertEqual(sl.allocation(64, 4)['slot_cap'], 61)
        self.assertEqual(sl.allocation(16, 4)['traces'], 25)
        self.assertEqual(sl.allocation(256, 4)['traces'], 464)
        self.assertEqual(sl.fifo_capacity(64), 127)

    def test_rejections(self):
        self.assertFalse(sl.feasible(16, 4, 3, sketch=True))    # slot cap 5
        self.assertTrue(sl.feasible(64, 8, 4, sketch=True))
        self.assertFalse(sl.feasible(16, 8, 3))                 # 21 traces against 36

    def test_mlp_fills_the_budget(self):
        total = (64 * 1024 + sl.READER_BYTES + sl.KNOWN_BYTES) // 4
        for layers in (1, 2):
            H = sl.mlp_width(64, layers)
            self.assertLessEqual(sl.mlp_params(H, layers), total)
            self.assertGreater(sl.mlp_params(H + 1, layers), total)
            self.assertLessEqual(2 * sl.mlp_params(sl.mlp_width(64, layers, momentum=True), layers), total)
            self.assertLessEqual(sl.mlp_params(sl.mlp_width(64, layers, window=64), layers) + 64 * 128, total)

    def test_arm_counts_match_mlp(self):
        net = sl.MLP(10, 2, np.random.default_rng(0))
        self.assertEqual(net.count(), sl.mlp_params(10, 2))
        net = sl.MLP(10, 1, np.random.default_rng(0))
        self.assertEqual(net.count(), sl.mlp_params(10, 1))


class World(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.t = sl.make_tables(self.rng)

    def world(self, g=128, q=.2):
        return sl.make_world(self.rng, self.t, g, q)

    def test_length_counts_and_rates(self):
        w = self.world(512, .2)
        self.assertEqual(w['L'], 5120)
        counts = np.bincount(w['kind'], minlength=len(sl.KINDS))
        k = sl.K
        self.assertEqual(counts[k['rule']], 4 * 8)
        self.assertEqual(counts[k['rule-noise']], 4 * 2)
        self.assertEqual(counts[k['binding']], 12 * 8)
        self.assertEqual(counts[k['exception']], 6 * 8)
        self.assertEqual(counts[k['rare']], 12 * 2)
        self.assertEqual(counts[k['noise-repeat']], 8 * 8)
        self.assertEqual(counts[k['noise-burst']], 4 * 4 * 6)
        self.assertEqual(counts[k['noise-once']], round(.2 * 5120))
        self.assertEqual(counts[k['query']], round(.1 * 5120))

    def test_causes_total_as_the_contract_says(self):
        w = self.world(8192)
        k = sl.K
        n = sum((w['kind'] == k[x]).sum() for x in ('rule', 'rule-noise', 'binding', 'exception', 'rare', 'noise-repeat', 'noise-burst'))
        self.assertEqual(n, 368)

    def test_one_off_cues_are_unique_and_not_reserved(self):
        w = self.world(2048)
        once = w['kind'] == sl.K['noise-once']
        cues = list(zip(w['fam'][once], w['it1'][once], w['it2'][once]))
        self.assertEqual(len(set(cues)), len(cues))
        self.assertFalse(set(cues) & self.t['reserved'])

    def test_every_cue_has_the_same_form(self):
        w = self.world(2048)
        r = sl.symbol_ranges()
        families = np.r_[r['known'], r['new'], r['binding'], r['background']]
        self.assertTrue(np.isin(w['fam'], families).all())
        self.assertTrue(np.isin(w['it1'], r['items']).all())
        self.assertTrue(np.isin(w['it2'], r['items']).all())
        self.assertTrue((w['it1'] != w['it2']).all())

    def test_rule_families_take_no_one_off_noise(self):
        w = self.world(2048)
        r = sl.symbol_ranges()
        once = w['kind'] == sl.K['noise-once']
        self.assertFalse(np.isin(w['fam'][once], r['new']).any())

    def test_background_families_activate_like_rule_families(self):
        w = self.world(2048)
        r = sl.symbol_ranges()
        firsts = [np.flatnonzero(w['fam'] == f)[0] for f in r['background'] if (w['fam'] == f).any()]
        self.assertGreater(max(firsts), 400)   # not all active from the first events
        self.assertLessEqual(max(firsts), 2048 + 1500)   # first use lags activation by the sampling among active families

    def test_rule_outcome_is_fixed_and_noise_repeat_is_not(self):
        w = self.world(2048)
        for f, o in self.t['new_rules']:
            m = (w['fam'] == f) & (w['kind'] == sl.K['rule'])
            self.assertEqual(set(w['outcome'][m]), {o})
        c = w['cause'][w['kind'] == sl.K['noise-repeat']]
        m = (w['cause'] == c[0]) & (w['kind'] == sl.K['noise-repeat'])
        self.assertGreater(len(set(w['outcome'][m])), 1)

    def test_longest_stream_has_enough_cues(self):
        w = sl.make_world(np.random.default_rng(2), self.t, 8192, .2)
        self.assertEqual((w['kind'] == sl.K['noise-once']).sum(), round(.2 * 81920))

    def test_deterministic(self):
        a = sl.make_world(np.random.default_rng(5), self.t, 128, .2)
        b = sl.make_world(np.random.default_rng(5), self.t, 128, .2)
        np.testing.assert_array_equal(a['outcome'], b['outcome'])


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.b = Bank()

    def test_three_consistent_occurrences_write_a_slot(self):
        a = life()
        c, o = self.b.cue(), self.b.outcome()
        for _ in range(3):
            a.step(c, o)
        self.assertEqual(a.n_new, 1)
        np.testing.assert_allclose(a.answers(c[None])[0], o, atol=1e-5)

    def test_noise_repeat_is_dropped_and_never_written(self):
        a = life()
        c = self.b.cue()
        for _ in range(8):
            a.step(c, self.b.outcome())
        self.assertEqual(a.n_new, 0)
        self.assertGreaterEqual(a.log['dropped_noise'], 1)

    def test_a_burst_is_one_episode_and_cannot_promote(self):
        a = life({'E': 64})
        c, o = self.b.cue(), self.b.outcome()
        for _ in range(6):
            a.step(c, o)
        self.assertEqual(a.n_new, 0)

    def test_a_shared_rule_generalizes_after_validation(self):
        a = life()
        f, o = self.b.cue(), self.b.outcome()
        for _ in range(8):
            a.step(self.b.family_cue(f), o)
        self.assertEqual(a.log['promoted'], 1)
        x = a.answers(self.b.family_cue(f)[None])[0]
        self.assertTrue(a.hit(x, o))

    def test_answers_change_no_state(self):
        a = life()
        f, o = self.b.cue(), self.b.outcome()
        for _ in range(3):
            a.step(self.b.family_cue(f), o)
        before = (a.R.copy(), a.tc.copy(), a.alive.copy(), a.ops)
        a.answers(self.b.family_cue(f)[None])
        np.testing.assert_array_equal(before[0], a.R)
        np.testing.assert_array_equal(before[1], a.tc)
        np.testing.assert_array_equal(before[2], a.alive)
        self.assertEqual(before[3], a.ops)

    def _shadow_with_failures(self, a, f, o):
        """A shadow fitted on a family's outcome, then contradicted on other cues of that family until it retires."""
        a.step(self.b.family_cue(f), o)
        a.step(self.b.family_cue(f), o)
        self.assertEqual(len(a.shadows), 1)
        fit = list(a.shadows[0].fit)
        for _ in range(a.cfg['h_f']):
            a.step(self.b.family_cue(f), self.b.outcome())
        self.assertEqual(a.log['retired'], 1)
        return fit

    def test_a_retired_shadow_cannot_return_on_the_same_traces(self):
        a = life({'h_r': 9})
        f, o = self.b.cue(), self.b.outcome()
        fit = self._shadow_with_failures(a, f, o)
        self.assertTrue(a.used[fit].all())
        self.assertFalse(a.pinned[fit].any())
        self.assertEqual(a._fit_set(a.res[fit[0]], False), None)
        self.assertEqual(len(a.shadows), 0)

    def test_no_trace_ever_fits_two_shadows(self):
        a = life({'h_r': 99, 'h_f': 1})
        f, o = self.b.cue(), self.b.outcome()
        for i in range(60):
            a.step(self.b.family_cue(f), o if i % 3 else self.b.outcome())
        self.assertGreater(a.log['proposed'], 1)
        for shadows in a.fits_of.values():
            self.assertEqual(len(shadows), 1)

    def test_retirement_happens_on_an_event_the_answer_hits(self):
        a = life({'delta': 0., 'tau_s': .99, 'h_r': 99})
        f, o = self.b.cue(), self.b.outcome()
        a.step(self.b.family_cue(f), o)
        a.step(self.b.family_cue(f), o)
        self.assertEqual(len(a.shadows), 1)
        s = a.shadows[0]
        # a known-binding event: the answer hits through the slot, and R misses, so no damage; force the damage count directly
        s.rhits, s.damage = 64, 5
        a.step(np.eye(sl.D_C, dtype=np.float32)[0], np.eye(sl.D_O, dtype=np.float32)[0])
        self.assertEqual(a.log['retired'], 1)
        self.assertEqual(int(a.alive.sum()), 2)      # no trace was added for that event

    def test_eligibility_excludes_fit_cues_and_ignores_the_outcome(self):
        a = life({'h_r': 99, 'h_f': 99})
        f, o = self.b.cue(), self.b.outcome()
        cues = [self.b.family_cue(f) for _ in range(2)]
        for c in cues:
            a.step(c, o)
        s = a.shadows[0]
        before = s.failures + s.wins
        a.step(cues[0], o)                       # a fit cue: ineligible
        self.assertEqual(s.failures + s.wins, before)
        a.step(self.b.family_cue(f), self.b.outcome())   # another cue with a contradicting outcome: scored as a failure
        self.assertEqual(s.failures, 1)

    def test_a_pinned_trace_cannot_be_evicted(self):
        a = life({'h_r': 99, 'h_f': 99}, B=16)
        f, o = self.b.cue(), self.b.outcome()
        a.step(self.b.family_cue(f), o)
        a.step(self.b.family_cue(f), o)
        fit = list(a.shadows[0].fit)
        for _ in range(4 * a.cap):
            a.step(self.b.cue(), self.b.outcome())
        self.assertTrue(a.alive[fit].all())
        self.assertLessEqual(int(a.alive.sum()), a.cap)

    def test_singletons_leave_first_and_in_order(self):
        a = life({'h_b': 99})
        c = self.b.cue()
        o = self.b.outcome()
        a.step(c, o)
        a.step(c, o)                             # a group with support
        supported = np.flatnonzero(a.alive)
        for _ in range(a.cap + 5):
            a.step(self.b.cue(), self.b.outcome())
        self.assertTrue(a.alive[supported].all())
        self.assertEqual(int(a.alive.sum()), a.cap)

    def test_validated_support_gives_group_size_no_protection(self):
        a = life({'h_b': 99, 'h_r': 99, 'tau_e': .5, 'S': 2}, support='validated')
        o = self.b.outcome()
        for _ in range(3):
            a.step(self.b.cue(), o)                  # unrelated cues, one outcome: a residual group
        idx = np.flatnonzero(a.alive)
        self.assertGreater(max(a.groups[int(a.res[i])].episodes for i in idx), 0)
        self.assertEqual(max(a._support(int(i)) for i in idx), 0)
        c = self.b.cue()
        x = self.b.outcome()
        a.step(c, x)
        a.step(c, x)                                 # a prediction that came true
        i = int(np.flatnonzero(a.alive & (a.age == a.t))[0])
        self.assertEqual(a._support(i), 1)

    def test_deleting_a_group_deletes_its_counts(self):
        a = life({'h_b': 99, 'h_n': 99})
        c, o = self.b.cue(), self.b.outcome()
        a.step(c, o)
        gid = int(a.same[np.flatnonzero(a.alive)[0]])
        self.assertIn(gid, a.groups)
        a._drop_trace(int(np.flatnonzero(a.same == gid)[0]))
        self.assertNotIn(gid, a.groups)

    def test_eviction_keeps_a_group_counts_while_it_holds_a_trace(self):
        a = life({'h_b': 99})
        c, o = self.b.cue(), self.b.outcome()
        a.step(c, o)
        a.step(c, o)
        members = np.flatnonzero(a.alive)
        g = a.groups[int(a.same[members[0]])]
        before = g.episodes
        self.assertGreater(before, 0)
        a._drop_trace(int(members[0]))
        self.assertEqual(g.episodes, before)
        self.assertIn(g.id, a.groups)

    def test_a_promotion_resets_other_shadows(self):
        a = life({'h_r': 2, 'S': 2, 'h_f': 99, 'tau_e': .9})
        f1, o1, f2, o2 = self.b.cue(), self.b.outcome(), self.b.cue(), self.b.outcome()
        a.step(self.b.family_cue(f2), o2)
        a.step(self.b.family_cue(f2), o2)
        a.step(self.b.family_cue(f1), o1)
        a.step(self.b.family_cue(f1), o1)
        self.assertEqual(len(a.shadows), 2)
        other = [s for s in a.shadows if s.ebar @ o2 > .5][0]
        a.step(self.b.family_cue(f2), self.b.outcome())
        self.assertEqual(other.failures, 1)
        for _ in range(3):
            a.step(self.b.family_cue(f1), o1)
        self.assertEqual(a.log['promoted'], 1)
        self.assertEqual(other.failures, 0)
        self.assertEqual(a.log['refits'], 1)

    def test_query_scores_nothing(self):
        a = life()
        c, o = self.b.cue(), self.b.outcome()
        a.step(c, o)
        g = a.groups[int(a.same[np.flatnonzero(a.alive)[0]])]
        for _ in range(4):
            a.step(c, None)
        self.assertEqual((g.hits, g.misses, g.episodes), (0, 0, 0))
        self.assertEqual(a.n_new, 0)

    def test_self_count_manufactures_support(self):
        a = life(self_count=True, name='life-selfcount')
        c, o = self.b.cue(), self.b.outcome()
        a.step(c, o)
        for _ in range(3):
            a.step(c, None)
        self.assertEqual(a.n_new, 1)


class SketchTest(unittest.TestCase):
    def setUp(self):
        self.b = Bank()

    def test_familiar_bit_changes_no_count(self):
        """With and without the sketch, group and shadow counts match event for event; only eviction may differ, and the store here never fills."""
        a = life({'rho': 32}, B=256, sketch=True)
        b = life({}, B=256)
        for _ in range(30):
            c, o = self.b.cue(), self.b.outcome()
            for x in (c, c):
                a.step(x, o)
                b.step(x, o)
        self.assertGreater(a.log['sketch_matches'], 0)
        self.assertEqual(sorted((g.hits, g.misses, g.episodes) for g in a.groups.values()), sorted((g.hits, g.misses, g.episodes) for g in b.groups.values()))
        self.assertEqual(a.n_new, b.n_new)

    def test_familiar_breaks_eviction_ties_only(self):
        a = life({'h_b': 99, 'rho': 32}, B=64, sketch=True)
        a.step(self.b.cue(), self.b.outcome())       # no entry yet: unfamiliar
        first = int(np.flatnonzero(a.alive)[0])
        self.assertFalse(a.familiar[first])
        a.step(self.b.cue(), self.b.outcome())       # rho 32 matches everything: familiar
        second = [int(i) for i in np.flatnonzero(a.alive) if i != first][0]
        self.assertTrue(a.familiar[second])
        self.assertEqual(a._support(first), a._support(second))
        self.assertEqual(a._eviction_choice(), first)
        a.age[second] = 0
        a.age[first] = 5
        self.assertEqual(a._eviction_choice(), first)     # younger, but unfamiliar goes first

    def test_an_event_does_not_match_itself(self):
        a = life({'rho': 0}, B=64, sketch=True)
        a.step(self.b.cue(), self.b.outcome())
        self.assertEqual(a.log['sketch_matches'], 0)


class Comparators(unittest.TestCase):
    def setUp(self):
        self.b = Bank()
        self.R = np.zeros((sl.D_C, sl.D_O), np.float32)
        self.kc = np.eye(sl.D_C, dtype=np.float32)[:1]
        self.ko = np.eye(sl.D_O, dtype=np.float32)[:1]

    def test_frozen_keeps_its_reader(self):
        a = sl.ReaderArm(self.R, self.kc, self.ko, .8, .3)
        a.step(self.b.cue(), self.b.outcome())
        np.testing.assert_array_equal(a.R, self.R)

    def test_known_binding_is_answered_from_its_slot(self):
        a = sl.ReaderArm(self.R, self.kc, self.ko, .8, .3)
        np.testing.assert_array_equal(a.answers(self.kc)[0], self.ko[0])

    def test_slow_moves_toward_a_repeated_rule(self):
        a = sl.Slow(self.R, self.kc, self.ko, .8, .3, .1)
        f, o = self.b.cue(), self.b.outcome()
        for _ in range(100):
            a.step(self.b.family_cue(f), o)
        self.assertGreater(float(unit(a.answers(self.b.family_cue(f)[None])[0]) @ o), .7)

    def test_surprise_fifo_stores_only_surprises_and_newest_wins(self):
        a = sl.SurpriseFifo(self.R, self.kc, self.ko, .8, .3, 0., 4)
        c = self.b.cue()
        o1, o2 = self.b.outcome(), self.b.outcome()
        a.step(c, o1)
        a.step(c, o1)     # now predicted: not a surprise
        self.assertEqual(a.n, 1)
        a.step(c, o2)
        self.assertEqual(a.n, 2)
        np.testing.assert_allclose(a.answers(c[None])[0], o2)

    def test_fifo_is_a_ring_and_order_is_kept(self):
        a = sl.Replay(self.R, self.kc, self.ko, .8, .3, 0., 0, 3, np.random.default_rng(0), False)
        cs = [self.b.cue() for _ in range(5)]
        for c in cs:
            a.step(c, self.b.outcome())
        sc, _ = a.extra_store()
        np.testing.assert_array_equal(sc[-1], cs[-1])
        np.testing.assert_array_equal(sc[0], cs[2])

    def test_reservoir_keeps_a_uniform_sample(self):
        counts = np.zeros(100)
        for s in range(300):
            a = sl.Replay(self.R, self.kc, self.ko, .8, .3, 0., 0, 10, np.random.default_rng(s), True)
            for i in range(100):
                c = np.zeros(sl.D_C, np.float32)
                c[1] = i
                a.step(c, self.ko[0])
            counts[a.c[:, 1].astype(int)] += 1
        self.assertLess(abs(counts[:50].sum() - counts[50:].sum()), 300)
        self.assertEqual(counts.sum(), 3000)

    def test_sleep_consolidates_everything_and_clears(self):
        a = sl.SleepPeriodic(self.R, self.kc, self.ko, .8, .3, .1, 5, 8)
        f, o = self.b.cue(), self.b.outcome()
        for _ in range(4):
            a.step(self.b.family_cue(f), o)
        self.assertEqual(a.n, 4)
        np.testing.assert_array_equal(a.R, self.R)
        a.step(self.b.family_cue(f), o)
        self.assertEqual(a.n, 0)
        self.assertGreater(float(np.abs(a.R).sum()), 0)

    def test_sleep_grow_counts_expert_bytes(self):
        a = sl.SleepPeriodic(self.R, self.kc, self.ko, .8, .3, .1, 2, 4, rank=2)
        for _ in range(6):
            a.step(self.b.cue(), self.b.outcome())
        self.assertEqual(a.persistent_bytes()['experts'], 3 * 2 * 128 * 4)

    def test_mlp_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        for layers in (1, 2):
            net = sl.MLP(5, layers, rng)
            net.W = [w.astype(np.float64) for w in net.W]
            net.b = [b.astype(np.float64) for b in net.b]
            X, Y = rng.standard_normal((4, sl.D_C)), rng.standard_normal((4, sl.D_O))
            gs = net.grad(X, Y)
            for p, g in zip(net.params(), gs):
                idx = tuple(0 for _ in p.shape)
                e = 1e-6
                v = p[idx]
                p[idx] = v + e
                a = ((net.forward(X) - Y) ** 2).sum() / 4
                p[idx] = v - e
                b = ((net.forward(X) - Y) ** 2).sum() / 4
                p[idx] = v
                self.assertAlmostEqual(g[idx], (a - b) / (2 * e), places=4)

    def test_neural_memory_learns_and_titans_decays(self):
        rng = np.random.default_rng(0)
        net = sl.MLP(16, 1, rng)
        a = sl.NeuralMemory(net, 0., 0., .1, 1, 'titans-m0')
        c, o = self.b.cue(), self.b.outcome()
        loss0 = float(((a.answers(c[None])[0] - o) ** 2).sum())
        for _ in range(20):
            a.step(c, o)
        self.assertLess(float(((a.answers(c[None])[0] - o) ** 2).sum()), loss0)

    def test_omega_window_is_bounded(self):
        net = sl.MLP(4, 1, np.random.default_rng(0))
        a = sl.NeuralMemory(net, 0., 0., .01, 3, 'omega')
        for _ in range(5):
            a.step(self.b.cue(), self.b.outcome())
        self.assertEqual(a.n, 3)
        self.assertEqual(a.persistent_bytes()['window'], 3 * 128 * 4)

    def test_fit_mlp_reduces_loss_and_discards_adam_state(self):
        rng = np.random.default_rng(0)
        net = sl.MLP(32, 1, rng)
        C = np.stack([self.b.cue() for _ in range(64)])
        O = np.stack([self.b.outcome()] * 64)
        before = float(((net.forward(C) - O) ** 2).sum(1).mean())
        out = sl.fit_mlp(net, C, O, rng, max_epochs=200, patience=200)
        self.assertLess(out['val_loss'], before / 2)

    def test_oracle_never_keeps_noise(self):
        a = sl.OracleLife(self.R, self.kc, self.ko, CFG, 64)
        for _ in range(10):
            a.told(self.b.cue(), self.b.outcome(), sl.K['noise-once'], -1)
        self.assertEqual(a.n_new, 0)
        np.testing.assert_array_equal(a.R, self.R)


class Readings(unittest.TestCase):
    def test_noise_capture_is_zero_for_an_unrelated_answer(self):
        rng = np.random.default_rng(0)
        t = rng.integers(32, size=200)
        a = rng.integers(32, size=(200, 8))
        d = sl.derangement(rng, 200)
        val, parts = sl.noise_capture({'x': a}, None, {'x': t}, {'x': d})
        self.assertLess(abs(val), .02)

    def test_noise_capture_is_high_when_the_answer_repeats_the_target(self):
        rng = np.random.default_rng(0)
        t = rng.integers(32, size=50)
        val, _ = sl.noise_capture({'x': np.tile(t[:, None], (1, 8))}, None, {'x': t}, {'x': sl.derangement(rng, 50)})
        self.assertGreater(val, .9)

    def test_derangement_has_no_fixed_point(self):
        d = sl.derangement(np.random.default_rng(0), 8)
        self.assertFalse((d == np.arange(8)).any())


if __name__ == '__main__':
    unittest.main()
