import itertools
import json
from pathlib import Path
import random
import unittest

from src.core_abstraction import (
    C0, FULL, FULL_MASK, INPUT_FROZEN, K_CLAMPS, NF, PAIRS, Responses, WakeFilter, acquire, assemble, assess,
    assignment, blocks, index, mirror_core, parts, potential, potential_closed_form, simulate, wake_targets,
    waking_vector,
)
from src.shared_input_mediation import A, B, I1, I2, K, N_VECTORS, QUERIES, TERNARY, explore, vector
from tests.test_shared_input_mediation import oracle


ROOT = Path(__file__).resolve().parents[1]
SPLITS = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-splits.json').read_text())['splits']


def oracle_Y(qi, w, mode):
    x = QUERIES[qi]
    if mode == FULL:
        terms = oracle(x, w)[1]
    elif mode == INPUT_FROZEN:
        terms = oracle(x, w, ((I1, x.e1), (I2, x.e2)))[1]
    elif mode == NF:
        terms = oracle(x, w, nf=True)[1]
    else:
        terms = oracle(x, w, ((K, mode[1]),))[1]
    return frozenset((t[A], t[B]) for t in terms)


def with_core(w, c):
    pa, pb, _, r = parts(w)
    return assemble(pa, pb, c, r)


class FakeResp:
    """Marginals from a table keyed (qi, w, sat, mode); default {0}. Y is the pair of marginals."""
    def __init__(self, table):
        self.table = table

    def marginal(self, qi, w, sat, mode=FULL):
        return self.table.get((qi, w, sat, mode), frozenset({0}))

    def Y(self, qi, w, mode=FULL):
        return frozenset(itertools.product(self.marginal(qi, w, 'A', mode), self.marginal(qi, w, 'B', mode)))


class BasicsTests(unittest.TestCase):
    def test_index_matches_vector(self):
        self.assertEqual([index(vector(h)) for h in range(N_VECTORS)], list(range(N_VECTORS)))

    def test_parts_round_trip_and_orders(self):
        for h in range(0, N_VECTORS, 13):
            w = vector(h)
            self.assertEqual(assemble(*parts(w)), w)
        self.assertEqual(len(PAIRS), 9)
        self.assertEqual(PAIRS[0], (-3, -3))
        self.assertEqual(PAIRS[4], C0)

    def test_core_potential_closed_form(self):
        for c in PAIRS:
            for a in TERNARY:
                for b in TERNARY:
                    self.assertEqual(potential(c, a, b), potential_closed_form(c, a, b))

    def test_assignment_alternates(self):
        for s in SPLITS:
            a, b = assignment(s['train'])
            self.assertEqual((len(a), len(b)), (8, 7))
            self.assertEqual(sorted(a + b), sorted(s['train']))
            self.assertEqual(sorted(s['train'])[0::2], a)
            self.assertEqual(assignment(s['train'], True), (b, a))


class DynamicsTests(unittest.TestCase):
    def test_simulate_matches_oracle_in_every_mode(self):
        rng = random.Random(1)
        hs = [index((3, 3, 3, 0, 3, -3, 0)), index((3, 0, 0, -3, 3, -3, -3))] + [rng.randrange(N_VECTORS)
                                                                               for _ in range(12)]
        for h in hs:
            w = vector(h)
            for qi in range(0, 81, 4):
                for mode in (FULL, INPUT_FROZEN, NF) + K_CLAMPS:
                    self.assertEqual(simulate(QUERIES[qi], w, mode), oracle_Y(qi, w, mode), (h, qi, mode))

    def test_core_sign_symmetry(self):
        rng = random.Random(2)
        for _ in range(60):
            w = vector(rng.randrange(N_VECTORS))
            m = with_core(w, mirror_core(parts(w)[2]))
            for qi in range(81):
                x = QUERIES[qi]
                for mode in (FULL, INPUT_FROZEN, NF):
                    self.assertEqual(simulate(x, w, mode), simulate(x, m, mode))
                for v in TERNARY:
                    self.assertEqual(simulate(x, w, ('k', v)), simulate(x, m, ('k', -v)))

    def test_waking_environment_isolates_satellite(self):
        rng = random.Random(3)
        for _ in range(40):
            p, r = rng.choice(PAIRS), rng.choice((-3, 0, 3))
            for sat, other in (('A', B), ('B', A)):
                w = waking_vector(sat, p, r)
                self.assertEqual(parts(w)[2], C0)
                for qi in range(81):
                    terms = explore(QUERIES[qi], w).terminals
                    self.assertTrue(all(t[K] == 0 and t[other] == 0 for t in terms))

    def test_waking_targets_ignore_core_and_partner(self):
        resp = Responses()
        rng = random.Random(4)
        for _ in range(20):
            w = vector(rng.randrange(N_VECTORS))
            pa, pb, _, r = parts(w)
            u = assemble(pa, rng.choice(PAIRS), rng.choice(PAIRS), r)
            ids = list(range(81))
            self.assertEqual(wake_targets(resp, w, ids, []), wake_targets(resp, u, ids, []))


class FilterTests(unittest.TestCase):
    def test_teacher_survives_and_order_independent_and_idempotent(self):
        resp = Responses()
        rng = random.Random(5)
        for _ in range(12):
            teacher = vector(rng.randrange(N_VECTORS))
            pa, pb, _, r = parts(teacher)
            s = rng.choice(SPLITS)
            ids_a, ids_b = assignment(s['train'], rng.random() < 0.5)
            wake = wake_targets(resp, teacher, ids_a, ids_b)
            masks = set()
            for order in s['orders']:
                fa, fb = acquire(resp, r, order, ids_a, wake)
                masks.add((fa.mask, fb.mask))
                self.assertIn(pa, fa.blocks())
                self.assertIn(pb, fb.blocks())
                before = (fa.mask, fb.mask)
                for qi in order:
                    (fa if qi in ids_a else fb).update(qi, wake[('A' if qi in ids_a else 'B', qi)])
                self.assertEqual((fa.mask, fb.mask), before)
            self.assertEqual(len(masks), 1)

    def test_filter_only_simulates_waking_vectors(self):
        seen = []

        class Spy(Responses):
            def marginal(self, qi, w, sat, mode=FULL):
                seen.append((w, sat, mode))
                return super().marginal(qi, w, sat, mode)
        f = WakeFilter('B', -3, Spy())
        f.update(10, frozenset({0}))
        self.assertTrue(seen)
        for w, sat, mode in seen:
            self.assertEqual((sat, mode, w[:2], w[4:]), ('B', FULL, (0, 0), (0, 0, -3)))
        self.assertEqual(blocks(FULL_MASK), list(PAIRS))


class AssessTests(unittest.TestCase):
    """Predicate logic on a synthetic response table; no teacher scoring."""
    PA, PB, R = (3, 0), (0, 3), 0
    T = (9, 9, 9, 9, 9, 9, 9)        # opaque teacher key (only used for table lookup)
    MIX, TRAIN = 5, 40

    def run_for(self):
        return dict(train=[self.TRAIN], mixed=[self.MIX], within=[], r=self.R,
                    wake={('A', self.TRAIN): frozenset({0})})

    def table(self, c, cut_a_correct=False, frozen_differs=True):
        one = frozenset({1})
        w = lambda core: (self.PA[0], self.PA[1], self.PB[0], self.PB[1], core[0], core[1], self.R)
        t = {(self.MIX, self.T, 'B', FULL): one, (self.TRAIN, self.T, 'A', FULL): frozenset({0}),
             (self.MIX, w(c), 'B', FULL): one}
        if cut_a_correct:
            t[(self.MIX, w((0, c[1])), 'B', FULL)] = one
        if frozen_differs:
            t[(self.MIX, w(c), 'B', INPUT_FROZEN)] = one
        return t

    def check(self, c, **kw):
        return assess(FakeResp(self.table(c, **kw)), self.run_for(), self.T, self.PA, self.PB)

    def test_qualifying(self):
        out = self.check((3, 3))
        self.assertEqual((out['outcome'], out['qualifying']), ('qualifying', [[3, 3]]))
        rec = out['cores']['3,3']
        self.assertEqual(rec['route_specific_witnesses'], [['B', self.MIX]])
        self.assertEqual(rec['satellites']['B']['gains'], [self.MIX])

    def test_route_unresolved(self):
        out = self.check((3, -3), frozen_differs=False)
        self.assertEqual((out['outcome'], out['route_unresolved']), ('route_unresolved_only', [[3, -3]]))

    def test_cut_keeps_gain_is_not_witness(self):
        self.assertEqual(self.check((3, 3), cut_a_correct=True)['outcome'], 'no_core_dependent_witness')

    def test_single_link_core_cannot_qualify(self):
        out = self.check((3, 0))
        self.assertEqual(out['outcome'], 'no_core_dependent_witness')
        self.assertTrue(out['cores']['3,0']['transfer'])

    def test_interference_and_no_transfer(self):
        run = self.run_for()
        run['wake'] = {('A', self.TRAIN): frozenset({1})}
        out = assess(FakeResp(self.table((3, 3))), run, self.T, self.PA, self.PB)
        self.assertEqual(out['outcome'], 'assembly_interference')
        self.assertEqual(assess(FakeResp({}), self.run_for(), self.T, self.PA, self.PB)['outcome'],
                         'no_transfer_core')

    def test_summary_mode_agrees(self):
        for kw in ({}, dict(frozen_differs=False), dict(cut_a_correct=True)):
            full = assess(FakeResp(self.table((3, 3), **kw)), self.run_for(), self.T, self.PA, self.PB)
            short = assess(FakeResp(self.table((3, 3), **kw)), self.run_for(), self.T, self.PA, self.PB,
                           detail=False)
            self.assertEqual((full['qualifying'], full['route_unresolved']),
                             (short['qualifying'], short['route_unresolved']))


if __name__ == '__main__':
    unittest.main()
