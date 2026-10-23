from collections import Counter
from itertools import product
import json
from pathlib import Path
import random
import unittest

from experiments.shared_core_abstraction_v3 import disjoint_witness, groups_of, joint_dp
from src.core_abstraction import C0, FULL, FULL_MASK, PAIRS, Responses, assemble, parts
from src.core_abstraction_v2 import CORE_BIT, CoreFilter, connected_targets, first_core
from src.shared_core_abstraction_v3 import (
    atomic_masks, behavioral, direct_mask, joint_state, members, predictions, select_branch, select_single,
    source_state, structural, subset_masks, task_key, witness_conditions,
)
from src.shared_input_mediation import N_VECTORS, vector
from tests.test_core_abstraction import FakeResp


ROOT = Path(__file__).resolve().parents[1]
SPLITS = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-splits.json').read_text())['splits']
OPP = [(3, -3), (-3, 3)]
MATCH = [(3, 3), (-3, -3)]
A_ONLY = [(3, 0), (-3, 0)]
B_ONLY = [(0, 3), (0, -3)]


def mask_of(cs):
    m = 0
    for c in cs:
        m |= 1 << CORE_BIT[c]
    return m


def random_class_mask(rng):
    """Random union of sign classes (core-sign symmetric), as real masks are."""
    m = 0
    for cls in ([C0], OPP, MATCH, A_ONLY, B_ONLY):
        if rng.random() < 0.6:
            m |= mask_of(cls)
    return m


class StateTests(unittest.TestCase):
    def test_source_and_joint_states(self):
        self.assertEqual(source_state(0), 'empty')
        self.assertEqual(source_state(mask_of(OPP)), 'identifies')
        self.assertEqual(source_state(mask_of(OPP + MATCH)), 'ambiguous')
        self.assertEqual(source_state(FULL_MASK), 'uninformative')
        self.assertEqual(joint_state(mask_of(OPP)), 'two_link')
        self.assertEqual(joint_state(mask_of(A_ONLY)), 'other_single')
        self.assertEqual(joint_state(mask_of([C0])), 'other_single')
        self.assertEqual(joint_state(mask_of(OPP + [C0])), 'ambiguous')

    def test_structural_fixtures(self):
        comp_a, comp_b = mask_of(OPP + MATCH), mask_of(OPP + A_ONLY)
        self.assertTrue(structural(comp_a, comp_b, (3, -3)))
        self.assertTrue(structural(comp_a, comp_b, None))
        self.assertFalse(structural(comp_a, comp_b, (3, 3)))                            # teacher class excluded
        self.assertFalse(structural(mask_of(OPP), mask_of(OPP)))                         # redundant: both identify
        self.assertFalse(structural(FULL_MASK, mask_of(OPP)))                            # uninformative + identifying
        self.assertFalse(structural(mask_of(OPP + MATCH), mask_of(A_ONLY + B_ONLY)))     # empty intersection
        self.assertFalse(structural(mask_of(A_ONLY + MATCH), mask_of(A_ONLY + OPP)))     # one-link intersection
        self.assertFalse(structural(mask_of(OPP + MATCH + A_ONLY), mask_of(OPP + MATCH + B_ONLY)))   # ambiguous


class BehavioralTests(unittest.TestCase):
    P = ((0, 0), (0, 0))

    def resp(self, match_response):
        table = {}
        for c in OPP:
            table[(1, assemble(*self.P, c, 0), 'A', FULL)] = frozenset({1})
        for c in MATCH:
            table[(1, assemble(*self.P, c, 0), 'A', FULL)] = frozenset({match_response})
        table[(1, assemble(*self.P, C0, 0), 'A', FULL)] = frozenset({1})
        return FakeResp(table)

    def target(self, s, qi):
        return frozenset({1}) if (s, qi) == ('A', 1) else frozenset({0})

    def test_complementary_masks_resolve_response(self):
        resp = self.resp(-1)
        ma, mb = mask_of(OPP + MATCH), mask_of(OPP + A_ONLY)
        self.assertEqual(predictions(resp, self.P, 0, 'A', 1, ma), {frozenset({1}), frozenset({-1})})
        self.assertEqual(behavioral(resp, self.P, 0, ma, mb, [1, 2], self.target), [('A', 1)])

    def test_behavioral_equivalence_despite_coefficient_ambiguity(self):
        resp = self.resp(1)                                                       # MATCH predicts as OPP does
        self.assertEqual(behavioral(resp, self.P, 0, mask_of(OPP + MATCH), mask_of(OPP + A_ONLY), [1], self.target), [])

    def test_lucky_single_source_guess_does_not_block(self):
        resp = self.resp(-1)
        ma, mb = mask_of(OPP + MATCH), mask_of([C0] + OPP + B_ONLY)
        self.assertEqual(first_core(mb), C0)
        self.assertEqual(resp.marginal(1, assemble(*self.P, C0, 0), 'A'), frozenset({1}))   # correct by luck
        self.assertEqual(behavioral(resp, self.P, 0, ma, mb, [1], self.target), [('A', 1)])

    def test_empty_mask_certifies_nothing(self):
        resp = self.resp(-1)
        self.assertEqual(predictions(resp, self.P, 0, 'A', 1, 0), frozenset())
        self.assertEqual(behavioral(resp, self.P, 0, mask_of(OPP + MATCH), mask_of(A_ONLY + B_ONLY), [1], self.target), [])

    def test_witness_conditions_cut_each_link(self):
        resp = self.resp(-1)
        wc = witness_conditions(resp, self.P, 0, (-3, 3), ('A', 1), self.target)
        self.assertEqual(wc, dict(c0_incorrect=False, installed_exact=True, cuts_lose=True))
        table = dict(resp.table)
        table[(1, assemble(*self.P, (-3, 0), 0), 'A', FULL)] = frozenset({1})
        wc = witness_conditions(FakeResp(table), self.P, 0, (-3, 3), ('A', 1), self.target)
        self.assertFalse(wc['cuts_lose'])


class SubsetTests(unittest.TestCase):
    def test_incremental_equals_direct(self):
        rng = random.Random(3)
        atoms = [random_class_mask(rng) for _ in range(9)]
        masks = subset_masks(atoms)
        self.assertEqual(masks[0], FULL_MASK)
        self.assertTrue(all(masks[J] == direct_mask(atoms, J) for J in range(1 << 9)))

    def test_members_and_order(self):
        ids = (10, 14, 16)
        self.assertEqual(members(ids, 0b101), [10, 16])
        self.assertLess(task_key(ids, 0b100), task_key(ids, 0b011))
        self.assertLess(task_key(ids, 0b011), task_key(ids, 0b101))

    def test_branch_rules(self):
        b = [((0, 0), (0, 0)), ((0, 3), (0, 0)), ((3, 0), (0, 0))]
        t = {b[0]: (mask_of(OPP), mask_of(MATCH)), b[1]: (0, FULL_MASK), b[2]: (FULL_MASK, mask_of(OPP))}
        self.assertEqual(select_branch(b, t), (b[2], False))
        self.assertEqual(select_single(b, t, 0), (b[0], (-3, 3), False))
        self.assertEqual(select_single(b, t, 1), (b[0], (-3, -3), False))
        bad = {p: (mask_of(OPP), mask_of(MATCH)) for p in b}
        self.assertEqual(select_branch(b, bad), (b[0], True))
        self.assertEqual(select_single(b, {p: (0, 0) for p in b}, 0), (b[0], C0, True))


class EnvelopeTests(unittest.TestCase):
    def brute(self, atoms_a, atoms_b, disjoint):
        n = len(atoms_a)
        out = Counter()
        for choice in product(range(4), repeat=n):
            if disjoint and 3 in choice:
                continue
            ma = mb = FULL_MASK
            t = 0
            for j, ch in enumerate(choice):
                if ch in (1, 3):
                    ma &= atoms_a[j]
                    t += 1
                if ch in (2, 3):
                    mb &= atoms_b[j]
                    t += 1
            out[(ma, mb, t)] += 1
        return out

    def test_dp_and_convolution_match_brute_force(self):
        rng = random.Random(11)
        for _ in range(5):
            n = 5
            A = [random_class_mask(rng) for _ in range(n)]
            B = [random_class_mask(rng) for _ in range(n)]
            ids = tuple(range(n))
            self.assertEqual(+joint_dp(A, B, False), +self.brute(A, B, False))
            self.assertEqual(+joint_dp(A, B, True), +self.brute(A, B, True))
            ga, _ = groups_of(subset_masks(A), ids)
            gb, _ = groups_of(subset_masks(B), ids)
            conv = Counter()
            for ma, va in ga.items():
                for mb, vb in gb.items():
                    for i, x in enumerate(va):
                        for j, y in enumerate(vb):
                            if x and y:
                                conv[(ma, mb, i + j)] += x * y
            self.assertEqual(+conv, +self.brute(A, B, False))

    def test_groups_canonical_subset(self):
        ids = (10, 20, 30)
        atoms = [mask_of(OPP + MATCH), mask_of(OPP + MATCH), mask_of(OPP)]
        counts, canon = groups_of(subset_masks(atoms), ids)
        self.assertEqual(counts[mask_of(OPP + MATCH)], [0, 2, 1, 0])
        self.assertEqual(canon[(mask_of(OPP + MATCH), 1)], [10])
        self.assertEqual(canon[(mask_of(OPP), 2)], [10, 30])

    def test_disjoint_witness_is_first_valid(self):
        ids = (1, 2, 3, 4)
        A = [mask_of(OPP + MATCH), FULL_MASK, mask_of(OPP + A_ONLY), FULL_MASK]
        B = [FULL_MASK, mask_of(OPP + A_ONLY), FULL_MASK, mask_of(OPP + MATCH)]
        ma, mb = subset_masks(A), subset_masks(B)

        def q(x, y):
            return structural(x, y)
        w = disjoint_witness(ids, ma, mb, q, 2)
        self.assertEqual(w, dict(J_A=[1], J_B=[2], overlap=0))
        brute = sorted((members(ids, Ja), members(ids, Jb)) for Ja in range(16) for Jb in range(16)
                       if not Ja & Jb and bin(Ja).count('1') + bin(Jb).count('1') == 2 and q(ma[Ja], mb[Jb]))
        self.assertEqual((w['J_A'], w['J_B']), brute[0])


class RealFilterTests(unittest.TestCase):
    def test_atomic_masks_match_sequential_filter(self):
        resp = Responses()
        rng = random.Random(5)
        for _ in range(6):
            w = vector(rng.randrange(N_VECTORS))
            pa, pb, ct, r = parts(w)
            s = rng.choice(SPLITS)
            ids = tuple(sorted(s['train']))
            obs = connected_targets(resp, w, ids, ids)
            atoms = atomic_masks(resp, (pa, pb), r, ids, obs)
            ma = subset_masks([atoms[('A', q)] for q in ids])
            J = rng.randrange(1, 1 << len(ids))
            f = CoreFilter('A', (pa, pb), r, resp)
            for qi in rng.sample(members(ids, J), len(members(ids, J))):
                f.update(qi, obs[('A', qi)])
            self.assertEqual(f.mask, ma[J])
            self.assertTrue(all(atoms[k] >> CORE_BIT[ct] & 1 for k in atoms))


if __name__ == '__main__':
    unittest.main()
