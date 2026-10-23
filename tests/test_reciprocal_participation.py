import inspect
import itertools
import json
import random
import unittest
from unittest import mock

import src.reciprocal_participation as rp
from src.reciprocal_participation import (
    COEFFS, DOMAINS, N_HYPOTHESES, QUERIES, TEACHERS, ZERO, BlockLearner, ExactMemory, Query, VersionSpace,
    clamp_sets, controls, core_disconnected, delta, energy, explore, field, hypothesis, hypothesis_index,
    independent_product, make_split, outputs,
)


# ---------------------------------------------------------------- independent oracle
# Written from the contract formula; uses no production energy, delta, field or traversal helper.

def oracle_energy(z, e, w):
    i1, i2, k, a, b = z
    return (2 * (i1 - e[0]) ** 2 + 2 * (i2 - e[1]) ** 2 + 2 * a ** 2 + 2 * b ** 2 + k ** 2
            - w[0] * i1 * a - w[1] * i2 * a - w[2] * i1 * b - w[3] * i2 * b - w[4] * k * a - w[5] * k * b)


ORACLE_DOMAINS = {0: (-1, 0, 1), 1: (0, 1), 2: (-1, 0)}


def oracle_graph(q, w):
    """Reachable states and terminals by depth-first traversal from (e1, e2, 0, 0, 0)."""
    e = (q.e1, q.e2)
    allowed = [(-1, 0, 1)] * 3 + [ORACLE_DOMAINS[q.da], ORACLE_DOMAINS[q.db]]
    start = (q.e1, q.e2, 0, 0, 0)
    seen, stack, terminals = {start}, [start], set()
    while stack:
        z = stack.pop()
        here = oracle_energy(z, e, w)
        moved = False
        for pos in range(5):
            for v in allowed[pos]:
                if v != z[pos]:
                    nz = list(z)
                    nz[pos] = v
                    nz = tuple(nz)
                    if oracle_energy(nz, e, w) < here:
                        moved = True
                        if nz not in seen:
                            seen.add(nz)
                            stack.append(nz)
        if not moved:
            terminals.add(z)
    return seen, terminals


class SemanticsTests(unittest.TestCase):
    def test_schema_matches_contract(self):
        self.assertEqual(DOMAINS, ((-1, 0, 1), (0, 1), (-1, 0)))
        self.assertEqual(COEFFS, ('w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB'))
        self.assertEqual(len(QUERIES), 81)
        self.assertEqual(sorted(QUERIES, key=lambda q: (q.e1, q.e2, q.da, q.db)), list(QUERIES))
        self.assertEqual((hypothesis(0), hypothesis(1), hypothesis(728)), ((-3,) * 6, (-3,) * 5 + (0,), (3,) * 6))
        self.assertEqual([hypothesis_index(hypothesis(h)) for h in range(N_HYPOTHESES)], list(range(729)))
        self.assertEqual(hypothesis(hypothesis_index(ZERO)), ZERO)

    def test_input_validation(self):
        for bad in ((2, 0, 0, 0), (0, 0, 3, 0), (0, 0, 0, True), (0.0, 0, 0, 0)):
            with self.assertRaises((ValueError, TypeError)):
                Query(*bad)
        q = QUERIES[0]
        for w in ((1, 0, 0, 0, 0, 0), (0,) * 5, (0, 0, 0, 0, 0, 3.0)):
            with self.assertRaises(ValueError):
                explore(q, w)
        with self.assertRaises(ValueError):
            explore(Query(0, 0, 1, 0), ZERO, start=(0, 0, 0, -1, 0))   # forbidden sign for D1
        with self.assertRaises(ValueError):
            explore(q, ZERO, cap=-1)
        with self.assertRaises(Exception):
            q.e1 = 1

    def test_delta_and_field_agree_with_energy(self):
        for h in range(0, 729, 7):
            w = hypothesis(h)
            for q in QUERIES[::5]:
                for z in explore(q, ZERO, start=None, frozen=()).reachable | {q.start()}:
                    for c in range(5):
                        for v in q.allowed(c):
                            nz = z[:c] + (v,) + z[c + 1:]
                            self.assertEqual(delta(z, q, w, c, v), energy(nz, q, w) - energy(z, q, w))
                            self.assertEqual(oracle_energy(z, (q.e1, q.e2), w), energy(z, q, w))


class OracleTests(unittest.TestCase):
    def test_dynamics_match_independent_traversal_on_all_59049_cases(self):
        cases = 0
        for h in range(N_HYPOTHESES):
            w = hypothesis(h)
            for q in QUERIES:
                r = explore(q, w)
                seen, terminals = oracle_graph(q, w)
                self.assertEqual(r.status, 'complete')
                self.assertEqual(r.reachable, frozenset(seen))
                self.assertEqual(r.terminals, frozenset(terminals))
                self.assertEqual(r.Y, frozenset((t[3], t[4]) for t in terminals))
                self.assertTrue(r.acyclic)
                cases += 1
        self.assertEqual(cases, 59049)

    def test_every_reachable_state_keeps_domains_and_evidence(self):
        for w in TEACHERS.values():
            for q in QUERIES:
                r = explore(q, w)
                for z in r.reachable:
                    self.assertIn(z[3], DOMAINS[q.da])
                    self.assertIn(z[4], DOMAINS[q.db])
                for t in r.terminals:
                    path = r.shortest_path(t)
                    state = q.start()
                    for step in path:
                        self.assertEqual(step['source'], state)
                        self.assertLess(energy(step['target'], q, w), energy(state, q, w))
                        self.assertEqual(sum(u != v for u, v in zip(state, step['target'])), 1)
                        state = step['target']
                    self.assertEqual(state, t)


class ControlledTests(unittest.TestCase):
    def test_ties_do_not_move(self):
        # Zero-delta candidates occur in 1,152 canonical cases (none under the four teachers);
        # none of them is ever an edge.
        ties = 0
        for h in range(0, N_HYPOTHESES, 3):
            w = hypothesis(h)
            for q in QUERIES:
                r = explore(q, w)
                for z in r.reachable:
                    for c in range(5):
                        for v in q.allowed(c):
                            if v != z[c] and delta(z, q, w, c, v) == 0:
                                ties += 1
                                nz = z[:c] + (v,) + z[c + 1:]
                                if nz in r.parent and r.parent[nz] is not None:
                                    self.assertNotEqual(r.parent[nz][0], z)
        self.assertGreater(ties, 0)
        # a = 1 -> -1 under zero field is an exact tie: the terminal keeps a = 1.
        q = Query(0, 0, 0, 0)
        z = (0, 0, 0, 1, 0)
        self.assertEqual(delta(z, q, ZERO, 3, -1), 0)
        r = explore(q, ZERO, start=z, frozen=(0, 1, 2, 4))
        self.assertEqual(r.reachable, frozenset({z, (0, 0, 0, 0, 0)}))
        self.assertNotIn((0, 0, 0, -1, 0), r.reachable)

    def test_truncation_returns_no_certified_set(self):
        q, w = Query(1, 0, 0, 0), TEACHERS['T1']
        full = explore(q, w)
        zero = explore(q, w, cap=0)
        self.assertEqual((zero.status, zero.Y, zero.terminals, zero.acyclic, zero.inspections),
                         ('truncated', None, None, None, 0))
        exact = explore(q, w, cap=full.inspections)
        self.assertEqual((exact.status, exact.Y, exact.terminals), ('complete', full.Y, full.terminals))
        short = explore(q, w, cap=full.inspections - 1)
        self.assertEqual((short.status, short.Y), ('truncated', None))
        self.assertLessEqual(short.certified_terminals, full.terminals)
        for cap in range(full.inspections):
            self.assertIsNone(explore(q, w, cap=cap).Y)

    def test_simultaneous_opposing_contributions_cancel(self):
        w = TEACHERS['T1']                          # w_1A = 3, w_KA = 3
        q = Query(1, 0, 0, 0)
        reinforce, cancel = (1, 0, 1, 0, 0), (1, 0, -1, 0, 0)
        self.assertEqual((field(reinforce, w, 3), field(cancel, w, 3)), (6, 0))
        self.assertLess(delta(reinforce, q, w, 3, 1), 0)
        self.assertGreater(delta(cancel, q, w, 3, 1), 0)
        self.assertLess(delta((1, 0, 0, 0, 0), q, w, 3, 1), 0)

    def test_local_rules_depend_on_neighbors_when_coupled(self):
        # A's move 0 -> 1 has delta 2 - (w_1A i1 + w_2A i2 + w_KA k): it depends on k iff w_KA != 0.
        q = Query(0, 0, 0, 0)
        for h in range(N_HYPOTHESES):
            w = hypothesis(h)
            ds = {kv: delta((0, 0, kv, 0, 0), q, w, 3, 1) for kv in (-1, 0, 1)}
            self.assertEqual(len(set(ds.values())) > 1, w[4] != 0)
            di = {iv: delta((iv, 0, 0, 0, 0), q, w, 3, 1) for iv in (-1, 0, 1)}
            self.assertEqual(len(set(di.values())) > 1, w[0] != 0)
            # reverse: an i1 move depends on a iff w_1A != 0, on b iff w_1B != 0
            da = {av: delta((0, 0, 0, av, 0), q, w, 0, 1) for av in (-1, 0, 1)}
            self.assertEqual(len(set(da.values())) > 1, w[0] != 0)
            dk = {av: delta((0, 0, 0, av, 0), q, w, 2, 1) for av in (-1, 0, 1)}
            self.assertEqual(len(set(dk.values())) > 1, w[4] != 0)

    def test_core_clamp_keeps_core_fixed(self):
        for w in TEACHERS.values():
            for q in QUERIES:
                for kv in (-1, 0, 1):
                    r = explore(q, w, start=(q.e1, q.e2, kv, 0, 0), frozen=(2,))
                    self.assertEqual({z[2] for z in r.reachable}, {kv})

    def test_control_invariants(self):
        for h in range(0, N_HYPOTHESES, 4):
            w = hypothesis(h)
            for q in QUERIES:
                c = controls(q, w)
                self.assertEqual(c['uncoupled'], frozenset({(0, 0)}))
                self.assertEqual(c['input_feedback_removed'], c['input_frozen'])   # expected equivalence
                frozen = explore(q, w, frozen=(0, 1))
                self.assertEqual({z[:2] for z in frozen.reachable}, {(q.e1, q.e2)})
                self.assertEqual({z[2] for z in explore(q, core_disconnected(w)).reachable}, {0})
                for z in explore(q, w, input_feedback=False).reachable:
                    self.assertEqual(z[:2], (q.e1, q.e2))

    def test_fixed_input_disconnected_dynamics_factorize(self):
        for h in range(N_HYPOTHESES):
            w = core_disconnected(hypothesis(h))
            for q in QUERIES:
                r = explore(q, w, frozen=(0, 1))
                self.assertEqual(r.Y, independent_product(q, w))

    def test_clamps_never_impose_forbidden_values(self):
        q = Query(0, 0, 1, 2)
        sets = clamp_sets(q, TEACHERS['T2'], 4)
        self.assertEqual(set(sets), set(DOMAINS[2]))
        for v in sets.values():
            self.assertLessEqual(v, set(DOMAINS[1]))


class SplitTests(unittest.TestCase):
    def test_split_shapes(self):
        for seed in range(5):
            sp = make_split(seed)
            tr, wi, mx = set(sp['train']), set(sp['within_context']), set(sp['mixed_evidence'])
            self.assertEqual((len(tr), len(wi), len(mx)), (15, 30, 36))
            self.assertEqual(tr | wi | mx, set(range(81)))
            self.assertFalse(tr & wi or tr & mx or wi & mx)
            for i in mx:
                self.assertTrue(QUERIES[i].e1 and QUERIES[i].e2)
            for o, order in enumerate(sp['orders']):
                expect = sorted(sp['train'])
                random.Random(10000 * seed + o).shuffle(expect)
                self.assertEqual(order, expect)
            for ci, e in enumerate(rp.AXIS_CONTEXTS):
                ids = [i for i, q in enumerate(QUERIES) if (q.e1, q.e2) == e]
                random.Random(1000 * seed + ci).shuffle(ids)
                self.assertEqual(sorted(ids[:3]), sorted(i for i in tr if (QUERIES[i].e1, QUERIES[i].e2) == e))

    def test_split_uses_no_labels(self):
        def forbidden(*a, **k):
            raise AssertionError('label access during sampling')
        with mock.patch.object(rp, 'explore', forbidden), mock.patch.object(rp, 'outputs', forbidden):
            make_split(3)


class LearnerTests(unittest.TestCase):
    def stream(self, w, order):
        return [(QUERIES[i], outputs(QUERIES[i], w)[0]) for i in order]

    def test_apis_have_no_teacher_access(self):
        for cls in (VersionSpace, BlockLearner, ExactMemory):
            self.assertEqual(list(inspect.signature(cls.update).parameters), ['self', 'x', 'feedback'])
            self.assertEqual(list(inspect.signature(cls.predict).parameters), ['self', 'x'])

    def test_reference_retains_teacher_feedback_and_is_order_independent(self):
        sp = make_split(1)
        for w in TEACHERS.values():
            t = hypothesis_index(w)
            finals = set()
            for order in sp['orders']:
                vs = VersionSpace()
                items = self.stream(w, order)
                for n, (q, f) in enumerate(items):
                    vs.update(q, f)
                    self.assertTrue(vs.bits >> t & 1)
                    for q2, f2 in items[:n + 1]:
                        certain, possible, certified = vs.predict(q2)
                        self.assertEqual(certified, f2)
                finals.add(vs.bits)
            self.assertEqual(len(finals), 1)
            for q in QUERIES:
                certain, possible, certified = vs.predict(q)
                y = outputs(q, w)[0]
                self.assertTrue(certain <= y <= possible)
                if certified is not None:
                    self.assertEqual(certified, y)

    def test_contradictory_feedback(self):
        vs = VersionSpace()
        q = QUERIES[40]
        self.assertEqual(vs.update(q, {(0, 0)}), 'updated')
        before = vs.bits
        self.assertEqual(vs.update(q, {(1, 1)}), 'inconsistent_feedback')
        self.assertTrue(vs.failed)
        self.assertEqual(vs.bits, before)
        self.assertEqual(vs.update(q, {(0, 0)}), 'inconsistent_feedback')

    def test_positive_only_keeps_supersets(self):
        vs = VersionSpace(positive_only=True)
        q = Query(1, 0, 0, 0)
        f = outputs(q, TEACHERS['T1'])[0]
        vs.update(q, f)
        for h in vs.survivors():
            self.assertLessEqual(f, outputs(q, hypothesis(h))[0])

    def test_block_learner_one_update_per_example_no_replay(self):
        bl = BlockLearner()
        self.assertEqual(len(list(bl.neighbors())), 24)
        items = self.stream(TEACHERS['T1'], make_split(0)['orders'][0])
        for q, f in items:
            before = bl.w
            with mock.patch.object(rp, 'QUERIES', ()):       # no access to stored queries
                bl.update(q, f)
            changed = [j for j in range(6) if bl.w[j] != before[j]]
            self.assertTrue(not changed or any(set(changed) <= set(b) for b in rp.BLOCKS))
        self.assertEqual(set(vars(bl)), {'w', 'evaluations', 'inspections', 'commits', 'coefficients_changed'})

    def test_block_learner_commits_only_strict_improvement(self):
        bl = BlockLearner()
        q = Query(0, 0, 0, 0)
        self.assertEqual(bl.update(q, outputs(q, ZERO)[0]), 'unchanged')
        self.assertEqual(bl.evaluations, 1)
        q = Query(1, 0, 0, 0)
        f = outputs(q, TEACHERS['T1'])[0]
        status = bl.update(q, f)
        loss = len(outputs(q, bl.w)[0] ^ f)
        self.assertEqual(status == 'updated', loss < len(outputs(q, ZERO)[0] ^ f))

    def test_memory_returns_unknown_for_unseen(self):
        m = ExactMemory()
        m.update(QUERIES[0], {(0, 0)})
        self.assertEqual((m.predict(QUERIES[0]), m.predict(QUERIES[1])), (frozenset({(0, 0)}), None))

    def test_deterministic_serialization(self):
        a, b = VersionSpace(), VersionSpace()
        items = self.stream(TEACHERS['T2'], make_split(0)['train'])
        for q, f in items:
            a.update(q, f)
        for q, f in reversed(items):
            b.update(q, f)
        self.assertEqual(json.dumps(a.serialize()), json.dumps(b.serialize()))


def oracle_terminal_outputs(q, w, freeze_input):
    e = (q.e1, q.e2)
    allowed = [(-1, 0, 1)] * 3 + [ORACLE_DOMAINS[q.da], ORACLE_DOMAINS[q.db]]
    start = (q.e1, q.e2, 0, 0, 0)
    seen, stack, out = {start}, [start], set()
    while stack:
        z = stack.pop()
        moved = False
        for pos in range(2 if freeze_input else 0, 5):
            for v in allowed[pos]:
                if v != z[pos]:
                    nz = z[:pos] + (v,) + z[pos + 1:]
                    if oracle_energy(nz, e, w) < oracle_energy(z, e, w):
                        moved = True
                        if nz not in seen:
                            seen.add(nz)
                            stack.append(nz)
        if not moved:
            out.add(z)
    return out


class PhaseAFailureCertificateTests(unittest.TestCase):
    def test_declared_teachers_have_no_feedback_effect_on_responses(self):
        # Gate condition 3 fails for T1-T3: input revision occurs, yet Y equals the input-frozen Y.
        for name, w in TEACHERS.items():
            revised = 0
            for q in QUERIES:
                full, frozen = oracle_terminal_outputs(q, w, False), oracle_terminal_outputs(q, w, True)
                self.assertEqual({t[3:] for t in full}, {t[3:] for t in frozen})
                revised += any(t[:2] != (q.e1, q.e2) for t in full)
            self.assertEqual(revised, 0 if name == 'T0' else 16)

    def test_single_satellite_input_couplings_never_change_responses(self):
        # Exhaustive over the 225 hypotheses whose input coordinates each couple to at most one satellite.
        n = 0
        for h in range(N_HYPOTHESES):
            w = hypothesis(h)
            if (w[0] == 0 or w[2] == 0) and (w[1] == 0 or w[3] == 0):
                n += 1
                for q in QUERIES:
                    self.assertEqual(explore(q, w).Y, explore(q, w, frozen=(0, 1)).Y)
        self.assertEqual(n, 225)
