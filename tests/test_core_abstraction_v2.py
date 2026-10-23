import json
from pathlib import Path
import random
import unittest

from src.core_abstraction import C0, FULL, FULL_MASK, INPUT_FROZEN, PAIRS, Responses, assemble, assignment, parts
from src.core_abstraction_v2 import (
    CORE_BIT, CORE_ORDER, CoreFilter, acquire_core_masks, connected_targets, consolidate, cores, evaluate,
    executable_branch, first_core, in_mask, sign_classes, witness_checks,
)
from src.shared_input_mediation import N_VECTORS, vector
from tests.test_core_abstraction import FakeResp


ROOT = Path(__file__).resolve().parents[1]
SPLITS = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-splits.json').read_text())['splits']


def mask_of(cs):
    m = 0
    for c in cs:
        m |= 1 << CORE_BIT[c]
    return m


class OrderTests(unittest.TestCase):
    def test_minimal_change_order(self):
        self.assertEqual(CORE_ORDER, ((0, 0), (-3, 0), (0, -3), (0, 3), (3, 0),
                                      (-3, -3), (-3, 3), (3, -3), (3, 3)))
        self.assertEqual(first_core(FULL_MASK), C0)
        self.assertIsNone(first_core(0))
        self.assertEqual(cores(mask_of([(3, 3), (0, 3)])), [(0, 3), (3, 3)])
        self.assertTrue(in_mask(mask_of([(3, -3)]), (3, -3)))
        self.assertFalse(in_mask(mask_of([(3, -3)]), (-3, 3)))
        self.assertEqual(sign_classes(mask_of([(3, -3), (-3, 3), (0, 3)])), [(-3, 3), (0, -3)])

    def test_consolidate_and_branch_rule(self):
        self.assertIsNone(consolidate((mask_of([(3, 3)]), mask_of([(0, 3)]))))
        self.assertEqual(consolidate((FULL_MASK, mask_of([(3, 3), (3, 0)]))), (3, 0))
        b = [((0, 0), (0, 0)), ((0, 3), (0, 0)), ((3, 0), (0, 0))]
        masks = {b[0]: (mask_of([(3, 3)]), mask_of([(0, 3)])), b[1]: (FULL_MASK, mask_of([(3, 3)])),
                 b[2]: (FULL_MASK, FULL_MASK)}
        self.assertEqual(executable_branch(b, masks), (b[1], False))
        bad = {p: (mask_of([(3, 3)]), mask_of([(0, 3)])) for p in b}
        self.assertEqual(executable_branch(b, bad), (b[0], True))


class FilterTests(unittest.TestCase):
    def test_core_filter_only_simulates_its_branch(self):
        seen = []

        class Spy(Responses):
            def marginal(self, qi, w, sat, mode=FULL):
                seen.append((w, sat, mode))
                return super().marginal(qi, w, sat, mode)
        branch = ((3, 0), (0, -3))
        f = CoreFilter('A', branch, -3, Spy())
        f.update(10, frozenset({0}))
        self.assertEqual(len(seen), 9)
        for w, sat, mode in seen:
            self.assertEqual((w[:4], w[6], sat, mode), ((3, 0, 0, -3), -3, 'A', FULL))

    def test_teacher_core_survives_symmetric_and_order_independent(self):
        resp = Responses()
        rng = random.Random(7)
        for _ in range(15):
            w = vector(rng.randrange(N_VECTORS))
            pa, pb, ct, r = parts(w)
            s = rng.choice(SPLITS)
            ids_a, ids_b = assignment(s['train'], rng.random() < 0.5)
            obs = connected_targets(resp, w, ids_a, ids_b)
            branches = [(pa, pb), (pa, (0, 0))]
            results = [acquire_core_masks(resp, branches, r, o, ids_a, obs)[0] for o in s['orders']]
            self.assertTrue(all(x == results[0] for x in results))
            ma, mb = results[0][(pa, pb)]
            self.assertTrue(in_mask(ma & mb, ct))
            for m in (ma, mb):
                self.assertEqual([in_mask(m, c) for c in PAIRS], [in_mask(m, (-c[0], -c[1])) for c in PAIRS])

    def test_connected_targets_are_full_teacher_marginals(self):
        resp = Responses()
        w = (3, 0, 0, -3, 3, -3, -3)
        obs = connected_targets(resp, w, [10], [12])
        self.assertEqual(obs[('A', 10)], frozenset(y[0] for y in resp.Y(10, w)))
        self.assertEqual(obs[('B', 12)], frozenset(y[1] for y in resp.Y(12, w)))


class PredicateTests(unittest.TestCase):
    """Synthetic response tables; no teacher scoring."""
    P = ((3, 0), (0, 3))
    R = 0
    T = (9, 9, 9, 9, 9, 9, 9)
    MIX, MIX2, TRAIN = 5, 6, 40
    ONE = frozenset({1})

    def w(self, c):
        return assemble(*self.P, c, self.R)

    def run_for(self):
        return dict(train=[self.TRAIN], mixed=[self.MIX, self.MIX2], within=[], r=self.R,
                    wake={('A', self.TRAIN): frozenset({0})})

    def base(self, c=(3, 3)):
        return {(self.MIX, self.T, 'B', FULL): self.ONE, (self.MIX, self.w(c), 'B', FULL): self.ONE,
                (self.MIX, self.w(c), 'B', INPUT_FROZEN): self.ONE}

    def test_feasible_core_dependent_and_route(self):
        ev = evaluate(FakeResp(self.base()), self.run_for(), self.T, self.P, (3, 3))
        self.assertTrue(ev['preservation_nonvacuous'] and ev['preserved'] and ev['transfer'])
        self.assertEqual((ev['core_dependent_gains'], ev['route_gains']), ([('B', self.MIX)], [('B', self.MIX)]))
        self.assertTrue(ev['feasible'] and ev['feasible_route'])
        self.assertEqual(ev['p0_sizes'], {'A': 1, 'B': 1})

    def test_any_mixed_regression_blocks_transfer(self):
        t = self.base()
        t[(self.MIX2, self.T, 'A', FULL)] = self.ONE
        t[(self.MIX2, self.w(C0), 'A', FULL)] = self.ONE
        ev = evaluate(FakeResp(t), self.run_for(), self.T, self.P, (3, 3))
        self.assertTrue(ev['mixed_regression'])
        self.assertFalse(ev['transfer'] or ev['feasible'])

    def test_vacuous_and_lost_preservation(self):
        t = self.base()
        t[(self.TRAIN, self.T, 'A', FULL)] = self.ONE          # A wrong at c0 on its only training query
        ev = evaluate(FakeResp(t), self.run_for(), self.T, self.P, (3, 3))
        self.assertEqual(ev['p0_sizes'], {'A': 0, 'B': 1})
        self.assertFalse(ev['preservation_nonvacuous'] or ev['feasible'])
        t = self.base()
        t[(self.TRAIN, self.w((3, 3)), 'B', FULL)] = self.ONE  # B correct at c0, broken at (3,3)
        ev = evaluate(FakeResp(t), self.run_for(), self.T, self.P, (3, 3))
        self.assertEqual(ev['lost'], [['B', self.TRAIN]])
        self.assertFalse(ev['preserved'])

    def test_cut_that_keeps_gain_is_not_core_dependent(self):
        t = self.base()
        t[(self.MIX, self.w((0, 3)), 'B', FULL)] = self.ONE
        ev = evaluate(FakeResp(t), self.run_for(), self.T, self.P, (3, 3))
        self.assertTrue(ev['transfer'])
        self.assertEqual(ev['core_dependent_gains'], [])

    def test_witness_donor_and_comparator_controls(self):
        resp = FakeResp(self.base())
        comps = {'I': C0, 'IR': C0}
        # recipient B's own mask alone would install c0 (fails), and contains (3,3) which differs: donor needed
        w = witness_checks(resp, self.run_for(), self.T, self.P, (3, 3), ('B', self.MIX),
                           (FULL_MASK, mask_of([C0, (3, 3)])), comps, set())
        self.assertTrue(w['recipient_only_fails'] and w['recipient_mask_has_alternative'] and w['donor_control'])
        self.assertTrue(w['comparator_control'])
        # recipient's own mask already pins the sign class of (3,3): no donor contribution
        t = self.base()
        t[(self.MIX, self.w((-3, -3)), 'B', FULL)] = self.ONE   # the mirror behaves identically
        resp = FakeResp(t)
        w = witness_checks(resp, self.run_for(), self.T, self.P, (3, 3), ('B', self.MIX),
                           (FULL_MASK, mask_of([(3, 3), (-3, -3)])), comps, set())
        self.assertFalse(w['donor_control'])
        # memorization of the exact query defeats the comparator control
        w = witness_checks(resp, self.run_for(), self.T, self.P, (3, 3), ('B', self.MIX),
                           (FULL_MASK, mask_of([C0, (3, 3)])), comps, {('B', self.MIX)})
        self.assertFalse(w['comparator_control'])


if __name__ == '__main__':
    unittest.main()
