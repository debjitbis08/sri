import unittest

import inspect
import json
import random
from unittest import mock

import src.selective_mediation as sm
from src.selective_mediation import (
    CONTEXTS, DOMAINS, GATES, NO_MEDIATION, PERMISSIVE, PRIMITIVES, QUERIES, TEACHERS, CoreFreeLearner,
    ExactQueryMemory, Query, VersionSpaceLearner, conflict_class, direct_response, hypothesis_gates,
    hypothesis_id, infer, make_split, response, verify_path,
)


# ---------------------------------------------------------------- independent oracle
# Written from the contract text only: no production successor, gate, domain or BFS helper.

ORACLE_DOMAINS = [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 1, 2, 3}]
ORACLE_CONTEXTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
ORACLE_FLIPS = [('a', 1), ('a', 2), ('b', 1), ('b', 2)]   # A_low, A_high, B_low, B_high


def oracle_enabled(h, c):
    gates = [(h // 64) % 4, (h // 16) % 4, (h // 4) % 4, h % 4]
    return tuple({0: False, 1: c[0] == 1, 2: c[1] == 1, 3: True}[g] for g in gates)


_closures = {}


def oracle_closure(enabled, da, db):
    """Reachability matrix over the 16 states (a, b) -> 4a+b by Warshall's algorithm."""
    key = (enabled, da, db)
    if key not in _closures:
        A, B = ORACLE_DOMAINS[da], ORACLE_DOMAINS[db]
        n = 16
        reach = [[i == j for j in range(n)] for i in range(n)]
        for a in A:
            for b in B:
                for on, (side, bit) in zip(enabled, ORACLE_FLIPS):
                    if on:
                        a2, b2 = (a ^ bit, b) if side == 'a' else (a, b ^ bit)
                        if a2 in A and b2 in B:
                            reach[4 * a + b][4 * a2 + b2] = True
        for k in range(n):
            for i in range(n):
                if reach[i][k]:
                    for j in range(n):
                        if reach[k][j]:
                            reach[i][j] = True
        _closures[key] = reach
    return _closures[key]


def oracle_response(q, h):
    reach = oracle_closure(oracle_enabled(h, ORACLE_CONTEXTS[q.context]), q.domain_a, q.domain_b)
    return frozenset(t for t in range(4) if reach[4 * q.a0 + q.b0][5 * t])


def oracle_distance(q, h, t):
    """Shortest number of executable steps to (t, t) by repeated relational composition."""
    enabled = oracle_enabled(h, ORACLE_CONTEXTS[q.context])
    A, B = ORACLE_DOMAINS[q.domain_a], ORACLE_DOMAINS[q.domain_b]
    layer, seen, d = {(q.a0, q.b0)}, {(q.a0, q.b0)}, 0
    while (t, t) not in layer:
        nxt = set()
        for a, b in layer:
            for on, (side, bit) in zip(enabled, ORACLE_FLIPS):
                s = (a ^ bit, b) if side == 'a' else (a, b ^ bit)
                if on and s[0] in A and s[1] in B and s not in seen:
                    nxt.add(s)
        seen |= nxt
        layer, d = nxt, d + 1
    return d


def find(pred, hypotheses=range(256)):
    for h in hypotheses:
        for q in QUERIES:
            if pred(q, h):
                return q, h
    raise AssertionError('no fixture in the frozen space')


class SchemaTests(unittest.TestCase):
    def test_constants_match_contract(self):
        self.assertEqual([set(d) for d in DOMAINS], ORACLE_DOMAINS)
        self.assertEqual(list(CONTEXTS), ORACLE_CONTEXTS)
        self.assertEqual(GATES, ('never', 'c0', 'c1', 'always'))
        self.assertEqual([p[0] for p in PRIMITIVES], ['A_low', 'A_high', 'B_low', 'B_high'])
        self.assertEqual(len(QUERIES), 576)
        self.assertEqual(sorted(QUERIES, key=Query.key), list(QUERIES))
        self.assertEqual(sum(q.context == 0 for q in QUERIES), 144)

    def test_hypothesis_indexing_is_lexicographic(self):
        self.assertEqual([hypothesis_id(hypothesis_gates(h)) for h in range(256)], list(range(256)))
        self.assertEqual(hypothesis_gates(1), (0, 0, 0, 1))
        self.assertEqual(hypothesis_gates(64), (1, 0, 0, 0))
        self.assertEqual(TEACHERS, dict(T0=72, T1=105, T2=193, T3=0))
        self.assertEqual((NO_MEDIATION, PERMISSIVE), (0, 255))

    def test_input_validation(self):
        for bad in ((7, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 4), (0, 0, 0, 0, True), (0, 0, 0.0, 0, 0)):
            with self.assertRaises((ValueError, TypeError)):
                Query(*bad)
        q = QUERIES[0]
        for bad in (-1, 256, '1'):
            with self.assertRaises(ValueError):
                infer(q, bad)
        for bad in (-1, 1.5):
            with self.assertRaises(ValueError):
                infer(q, 0, cap=bad)
        with self.assertRaises(TypeError):
            infer((0, 0, 0, 0, 0), 0)
        with self.assertRaises(Exception):
            q.a0 = 1


class ExactInferenceTests(unittest.TestCase):
    def test_bfs_matches_independent_closure_on_all_147456_cases(self):
        cases = 0
        for h in range(256):
            for q in QUERIES:
                r = infer(q, h)
                truth = oracle_response(q, h)
                self.assertEqual((r.status, r.response, r.feasible), ('complete', truth, bool(truth)))
                self.assertEqual(direct_response(q, hypothesis_gates(h)), truth)
                self.assertLessEqual(r.visited, 16)
                self.assertLessEqual(r.edge_checks, 4 * r.visited)
                if truth:
                    self.assertEqual(r.representative, min(truth))
                    self.assertEqual(len(r.path), oracle_distance(q, h, r.representative))
                    self.assertEqual(verify_path(q, r), dict(domain=0, gate=0, step=0, evidence=0))
                else:
                    self.assertEqual((r.representative, r.path), (None, ()))
                for state, _ in r.invalid_attempts:
                    self.assertTrue(state[0] in DOMAINS[q.domain_a] and state[1] in DOMAINS[q.domain_b])
                cases += 1
        self.assertEqual(cases, 147456)

    def test_uncapped_work_examines_four_primitives_per_visited_state(self):
        for h in (0, 72, 105, 193, 255):
            for q in QUERIES:
                r = infer(q, h)
                self.assertEqual(r.edge_checks, 4 * r.visited)

    def test_zero_and_exact_boundary_budgets(self):
        q = Query(6, 6, 0, 0, 3)                      # already compatible, everything reachable
        r0 = infer(q, PERMISSIVE, cap=0)
        self.assertEqual((r0.status, r0.feasible, r0.response, r0.partial_response, r0.edge_checks),
                         ('truncated', None, None, frozenset({0}), 0))
        full = infer(q, PERMISSIVE)
        exact = infer(q, PERMISSIVE, cap=full.edge_checks)
        self.assertEqual((exact.status, exact.response), ('complete', full.response))
        short = infer(q, PERMISSIVE, cap=full.edge_checks - 1)
        self.assertEqual((short.status, short.feasible, short.response, short.path, short.representative),
                         ('truncated', None, None, (), None))
        self.assertLessEqual(short.partial_response, full.response)
        # A capped search is never called infeasible, even if nothing was discovered yet.
        q2 = Query(4, 1, 0, 1, 3)
        for cap in range(infer(q2, PERMISSIVE).edge_checks):
            self.assertIsNone(infer(q2, PERMISSIVE, cap=cap).feasible)

    def test_certificate_lists_real_state_changes(self):
        for h in (72, 105, 193, 255):
            for q in QUERIES:
                r = infer(q, h)
                for s in r.path:
                    self.assertNotEqual(s.source, s.target)
                self.assertEqual(r.certificate()['primitives'], [s.primitive for s in r.path])


class FixtureTests(unittest.TestCase):
    """Fixtures by bounded enumeration in the frozen space; existence is not learned performance."""

    def test_already_compatible(self):
        q = Query(0, 0, 0, 0, 0)
        r = infer(q, NO_MEDIATION)
        self.assertEqual((r.response, r.path, conflict_class(q, r.response)), (frozenset({0}), (), 'already_compatible'))

    def test_repair_requires_a(self):
        q, h = find(lambda q, h: q.a0 != q.b0 and response(q, h) and all(t == q.b0 for t in response(q, h)),
                    TEACHERS.values())
        self.assertEqual({s.primitive[0] for s in infer(q, h).path}, {'A'})

    def test_repair_requires_b(self):
        q, h = find(lambda q, h: q.a0 != q.b0 and response(q, h) and all(t == q.a0 for t in response(q, h)),
                    TEACHERS.values())
        self.assertEqual({s.primitive[0] for s in infer(q, h).path}, {'B'})

    def test_repair_requires_both(self):
        q, h = find(lambda q, h: response(q, h) and all(t not in (q.a0, q.b0) for t in response(q, h)),
                    TEACHERS.values())
        self.assertEqual({s.primitive[0] for s in infer(q, h).path}, {'A', 'B'})

    def test_disjoint_domain_impossibility(self):
        q = Query(0, 1, 0, 1, 3)
        self.assertEqual(response(q, PERMISSIVE), frozenset())
        self.assertEqual(conflict_class(q, response(q, PERMISSIVE)), 'unresolved_disjoint_domain')

    def test_permission_only_impossibility(self):
        q = Query(4, 1, 0, 1, 3)
        self.assertEqual((response(q, TEACHERS['T3']), response(q, PERMISSIVE)), (frozenset(), frozenset({1})))
        self.assertEqual(conflict_class(q, frozenset()), 'unresolved_teacher_permission')

    def test_shared_domain_but_unreachable_intersection(self):
        # Movement is executable, the domains share modes, yet no shared mode is reachable.
        q, h = find(lambda q, h: q.a0 != q.b0 and DOMAINS[q.domain_a] & DOMAINS[q.domain_b]
                    and not response(q, h) and infer(q, h).visited > 1, TEACHERS.values())
        self.assertEqual(conflict_class(q, response(q, h)), 'unresolved_teacher_permission')

    def test_all_enabled_obstruction_class_is_empty(self):
        # Every declared domain is connected under its XOR moves, so with all gates enabled the
        # reachable set is D_A x D_B: intersecting domains are always resolvable.
        for q in QUERIES:
            self.assertEqual(response(q, PERMISSIVE), DOMAINS[q.domain_a] & DOMAINS[q.domain_b])
            for h in TEACHERS.values():
                self.assertNotEqual(conflict_class(q, response(q, h)), 'unresolved_all_enabled')

    def test_several_valid_outputs(self):
        q, h = find(lambda q, h: len(response(q, h)) >= 2, TEACHERS.values())
        self.assertEqual(infer(q, h).representative, min(response(q, h)))

    def test_domain_invalid_attempts_are_diagnostics_only(self):
        q = Query(5, 5, 0, 2, 3)                       # A_low and B_low leave {0, 2}
        r = infer(q, PERMISSIVE)
        self.assertTrue(r.invalid_attempts)
        self.assertEqual(r.response, frozenset({0, 2}))
        for s in r.path:
            self.assertIn(s.target[0], DOMAINS[5])
            self.assertIn(s.target[1], DOMAINS[5])



class SplitTests(unittest.TestCase):
    def test_split_shapes_disjointness_and_context(self):
        for seed in range(5):
            sp = make_split(seed)
            train, within, unseen = set(sp['train']), set(sp['within_context']), set(sp['unseen_context'])
            self.assertEqual((len(train), len(within), len(unseen)), (48, 384, 144))
            self.assertFalse(train & within or train & unseen or within & unseen)
            self.assertEqual(train | within | unseen, set(range(576)))
            self.assertNotIn(3, {QUERIES[i].context for i in train | within})
            self.assertEqual({QUERIES[i].context for i in unseen}, {3})
            for c in range(3):
                self.assertEqual(sum(QUERIES[i].context == c for i in train), 16)
            self.assertEqual(sp['train'], sorted(sp['train'], key=lambda i: QUERIES[i].key()))
            for o, order in enumerate(sp['orders']):
                expected = list(sp['train'])
                random.Random(10000 * seed + o).shuffle(expected)
                self.assertEqual(order, expected)

    def test_split_follows_declared_rng_recipe(self):
        sp = make_split(2)
        for c in range(3):
            idx = [i for i, q in enumerate(QUERIES) if q.context == c]
            random.Random(1000 * 2 + c).shuffle(idx)
            self.assertEqual(sorted(idx[:16]), sorted(i for i in sp['train'] if QUERIES[i].context == c))

    def test_sampling_has_no_label_access(self):
        def forbidden(*a, **k):
            raise AssertionError('split sampling consulted inference')
        with mock.patch.object(sm, 'infer', forbidden), mock.patch.object(sm, 'realize', forbidden), \
                mock.patch.object(sm, 'response', forbidden), mock.patch.object(sm, 'direct_response', forbidden):
            self.assertEqual(make_split(0), make_split(0))
        self.assertEqual(list(inspect.signature(make_split).parameters), ['seed'])

    def test_frozen_manifest_matches_regeneration(self):
        from experiments.selective_mediation import MANIFEST, manifest
        self.assertEqual(json.loads(MANIFEST.read_text()), manifest())


def train(learner, items):
    return [learner.update(q, fb) for q, fb in items]


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.split = make_split(0)

    def stream(self, teacher, order):
        return [(QUERIES[i], response(QUERIES[i], teacher)) for i in order]

    def test_learner_api_has_no_teacher_access(self):
        for cls in (VersionSpaceLearner, CoreFreeLearner, ExactQueryMemory):
            for name in ('update', 'predict'):
                params = list(inspect.signature(getattr(cls, name)).parameters)
                self.assertEqual(params, ['self', 'query', 'feedback'] if name == 'update' else ['self', 'query'])
        self.assertNotIn('TEACHERS', inspect.getsource(VersionSpaceLearner))

    def test_clean_feedback_retains_teacher_and_all_observed_sets(self):
        for teacher in TEACHERS.values():
            learner = VersionSpaceLearner()
            items = self.stream(teacher, self.split['orders'][0])
            for k, (q, fb) in enumerate(items):
                learner.update(q, fb)
                self.assertTrue(learner.bits >> teacher & 1)
                for q2, fb2 in items[:k + 1]:
                    for h in learner.survivors():
                        self.assertEqual(response(q2, h), fb2)

    def test_final_V_is_identical_across_orders(self):
        for positive_only in (False, True):
            for teacher in TEACHERS.values():
                finals = set()
                for order in self.split['orders']:
                    learner = VersionSpaceLearner(positive_only=positive_only)
                    train(learner, self.stream(teacher, order))
                    finals.add(learner.bits)
                self.assertEqual(len(finals), 1)

    def test_certain_subset_teacher_subset_possible(self):
        teacher = TEACHERS['T1']
        learner = VersionSpaceLearner()
        train(learner, self.stream(teacher, self.split['orders'][1][:24]))
        for q in QUERIES:
            sets = [response(q, h) for h in learner.survivors()]
            certain, possible = frozenset.intersection(*sets), frozenset.union(*sets)
            self.assertTrue(certain <= response(q, teacher) <= possible)

    def test_core_free_learner_is_observationally_equivalent(self):
        for teacher in TEACHERS.values():
            core, free = VersionSpaceLearner(), CoreFreeLearner()
            for q, fb in self.stream(teacher, self.split['orders'][2]):
                core.update(q, fb)
                free.update(q, fb)
                self.assertEqual(core.bits, free.bits)
            for q in QUERIES:
                self.assertEqual(core.predict(q).response, free.predict(q))

    def test_contradictory_feedback_is_reported_without_reset(self):
        q = Query(6, 6, 0, 1, 3)
        learner = VersionSpaceLearner()
        self.assertEqual(learner.update(q, {0, 1}), 'updated')
        before = learner.bits
        self.assertEqual(learner.update(q, set()), 'inconsistent_feedback')
        self.assertTrue(learner.failed)
        self.assertEqual(learner.bits, before)          # no replacement model, no silent reset
        self.assertIsNone(learner.predict(q))
        self.assertEqual(learner.update(q, {0, 1}), 'inconsistent_feedback')

    def test_teachers_indistinguishable_on_training_subset_but_not_held_out(self):
        # In context 00 every c0/c1 gate is closed, so T0 and T3 agree on all of it.
        subset = [q for q in QUERIES if q.context == 0]
        a, b = VersionSpaceLearner(), VersionSpaceLearner()
        train(a, [(q, response(q, TEACHERS['T0'])) for q in subset])
        train(b, [(q, response(q, TEACHERS['T3'])) for q in subset])
        self.assertEqual(a.bits, b.bits)
        self.assertTrue(a.bits >> TEACHERS['T0'] & 1 and a.bits >> TEACHERS['T3'] & 1)
        held = [q for q in QUERIES if response(q, TEACHERS['T0']) != response(q, TEACHERS['T3'])]
        self.assertTrue(held)
        self.assertNotIn(0, {q.context for q in held})

    def test_positive_only_keeps_supersets_and_ignores_empty_feedback(self):
        q = Query(6, 6, 0, 1, 3)
        learner = VersionSpaceLearner(positive_only=True)
        self.assertEqual(learner.update(q, set()), 'unchanged')
        self.assertEqual(learner.bits, sm.ALL_HYPOTHESES)
        learner.update(q, {1})
        self.assertTrue(learner.bits >> PERMISSIVE & 1)
        for h in learner.survivors():
            self.assertLessEqual({1}, response(q, h))

    def test_deterministic_serialization(self):
        a, b = VersionSpaceLearner(), VersionSpaceLearner()
        items = self.stream(TEACHERS['T2'], self.split['orders'][0])
        train(a, items)
        train(b, list(reversed(items)))
        self.assertEqual(json.dumps(a.serialize(), sort_keys=True), json.dumps(b.serialize(), sort_keys=True))
        self.assertEqual(len(a.serialize()['membership']), 64)

    def test_exact_memory_and_no_mediation_control(self):
        mem = ExactQueryMemory()
        q, unseen = Query(4, 1, 0, 1, 2), Query(4, 1, 0, 1, 1)
        mem.update(q, {1})
        self.assertEqual((mem.predict(q), mem.predict(unseen)), (frozenset({1}), response(unseen, NO_MEDIATION)))
        self.assertEqual(mem.predict(Query(6, 6, 2, 2, 3)), frozenset({2}))


if __name__ == '__main__':
    unittest.main()
