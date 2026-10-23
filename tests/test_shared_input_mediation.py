import unittest

import src.reciprocal_participation as v1
from src.shared_input_mediation import (
    A, B, I1, I2, K, N_VECTORS, QUERIES, ZERO, Query, controls, delta, energy, explore, input_energy,
    legacy_input_feedback_removed, start_state, vector, zeroed,
)


# ---------------------------------------------------------------- independent oracle
# From the contract's written formulas only: no production energy, delta, field or traversal.

ORACLE_DOMAINS = {0: (-1, 0, 1), 1: (0, 1), 2: (-1, 0)}


def oracle_EC(z, e, w):
    i1, i2, k, a, b = z
    return (2 * (i1 - e[0]) ** 2 + 2 * (i2 - e[1]) ** 2 + 2 * a ** 2 + 2 * b ** 2 + k ** 2
            - w[0] * i1 * a - w[1] * i2 * a - w[2] * i1 * b - w[3] * i2 * b - w[4] * k * a - w[5] * k * b
            - w[6] * i1 * i2)


def oracle_EI(z, e, w):
    return 2 * (z[0] - e[0]) ** 2 + 2 * (z[1] - e[1]) ** 2 - w[6] * z[0] * z[1]


def oracle(q, w, clamps=(), nf=False):
    """(reachable, terminals) by depth-first traversal; clamps replace and freeze coordinates."""
    e = (q.e1, q.e2)
    allowed = [(-1, 0, 1)] * 3 + [ORACLE_DOMAINS[q.da], ORACLE_DOMAINS[q.db]]
    z0 = [q.e1, q.e2, 0, 0, 0]
    for c, v in clamps:
        z0[c] = v
    z0 = tuple(z0)
    fixed = {c for c, _ in clamps}
    seen, stack, terms = {z0}, [z0], set()
    while stack:
        z = stack.pop()
        moved = False
        for p in range(5):
            if p in fixed:
                continue
            for v in allowed[p]:
                if v == z[p]:
                    continue
                n = z[:p] + (v,) + z[p + 1:]
                rule = oracle_EI if nf and p < 2 else oracle_EC
                if rule(n, e, w) < rule(z, e, w):
                    moved = True
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
        if not moved:
            terms.add(z)
    return frozenset(seen), frozenset(terms)


def run_clamped(q, w, clamps=(), nf=False):
    return explore(q, w, start=start_state(q, clamps), frozen=tuple(c for c, _ in clamps), nf=nf)


class OracleTests(unittest.TestCase):
    def test_family_C_matches_oracle_on_all_177147_cases(self):
        # Includes the 59,049 w_12 = 0 (family S) cases; they are not additional coverage.
        n = 0
        for h in range(N_VECTORS):
            w = vector(h)
            for q in QUERIES:
                r = explore(q, w)
                reach, terms = oracle(q, w)
                self.assertEqual((r.status, r.reachable, r.terminals), ('complete', reach, terms))
                self.assertEqual(r.Y, frozenset((t[3], t[4]) for t in terms))
                self.assertTrue(r.acyclic)
                n += 1
        self.assertEqual(n, 177147)

    def test_zero_internal_coupling_reduces_to_frozen_v1(self):
        for h in range(v1.N_HYPOTHESES):
            w6 = v1.hypothesis(h)
            for q in QUERIES:
                old, new = v1.explore(q, w6), explore(q, w6 + (0,))
                self.assertEqual((old.reachable, old.terminals, old.Y), (new.reachable, new.terminals, new.Y))
                self.assertEqual(v1.explore(q, w6, input_feedback=False).Y, legacy_input_feedback_removed(q, w6 + (0,)))

    def test_NF_matches_oracle_directed_rule(self):
        for h in range(0, N_VECTORS, 5):
            w = vector(h)
            for q in QUERIES[::4]:
                r = explore(q, w, nf=True)
                self.assertEqual((r.reachable, r.terminals), oracle(q, w, nf=True))
                self.assertTrue(r.acyclic)

    def test_energy_and_delta_formulas(self):
        for h in range(0, N_VECTORS, 11):
            w = vector(h)
            for q in QUERIES[::7]:
                for z in explore(q, w).reachable:
                    self.assertEqual(energy(z, q, w), oracle_EC(z, (q.e1, q.e2), w))
                    self.assertEqual(input_energy(z, q, w), oracle_EI(z, (q.e1, q.e2), w))
                    for c in range(5):
                        for v in q.allowed(c):
                            n = z[:c] + (v,) + z[c + 1:]
                            self.assertEqual(delta(z, q, w, c, v), energy(n, q, w) - energy(z, q, w))
                            if c < 2:
                                self.assertEqual(delta(z, q, w, c, v, nf=True),
                                                 input_energy(n, q, w) - input_energy(z, q, w))
                            else:
                                self.assertEqual(delta(z, q, w, c, v, nf=True), delta(z, q, w, c, v))


class FixtureTests(unittest.TestCase):
    def test_NF_keeps_internal_coupling_and_differs_from_edge_deletion(self):
        q, w = Query(1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 3)
        # Internal coupling pulls i2 toward i1 = e1 = 1 under NF: input-internal dynamics remain.
        self.assertIn(1, explore(q, w, nf=True).projection(I2))
        self.assertEqual(explore(q, zeroed(w, 'w_12'), nf=True).projection(I2), frozenset({0}))
        # Removing feedback (NF) keeps I -> A; deleting the symmetric edge w_1A removes it.
        w = (3, 0, 0, 0, 0, 0, 0)
        self.assertEqual(explore(q, w, nf=True).projection(A), frozenset({1}))
        self.assertEqual(explore(q, zeroed(w, 'w_1A'), nf=True).projection(A), frozenset({0}))

    def test_NF_equals_frozen_input_for_family_S_but_not_assumed_for_C(self):
        for h in range(0, N_VECTORS, 3):
            w = vector(h)
            if w[6] == 0:
                for q in QUERIES:
                    self.assertEqual(explore(q, w, nf=True).Y, explore(q, w, frozen=(I1, I2)).Y)
        q, w = Query(1, 0, 0, 0), (0, 0, 0, 3, 0, 0, 3)
        self.assertNotEqual(explore(q, w, nf=True).Y, explore(q, w, frozen=(I1, I2)).Y)

    def test_clamps_stay_in_domain_and_keep_evidence(self):
        q = Query(-1, 1, 1, 2)
        with self.assertRaises(ValueError):
            run_clamped(q, ZERO, ((A, -1),))            # forbidden sign in D1
        r = run_clamped(q, (3, 3, 3, 0, 3, -3, 0), ((A, 1), (I1, 0)))
        for z in r.reachable:
            self.assertEqual((z[A], z[I1]), (1, 0))
            self.assertIn(z[B], (-1, 0))
        self.assertEqual((r.query.e1, r.query.e2), (-1, 1))

    def test_interventions_are_fresh_runs(self):
        q, w = Query(0, 1, 0, 0), (3, 3, 3, 0, 3, -3, 0)
        first = run_clamped(q, w, ((A, 1),))
        run_clamped(q, w, ((A, -1),))
        again = run_clamped(q, w, ((A, 1),))
        self.assertEqual((first.reachable, first.terminals), (again.reachable, again.terminals))
        self.assertEqual(first.start, (0, 1, 0, 1, 0))

    def test_ties_truncation_and_validation(self):
        q = Query(0, 0, 0, 0)
        z = (0, 0, 0, 1, 0)
        self.assertEqual(delta(z, q, ZERO, A, -1), 0)
        r = explore(q, ZERO, start=z, frozen=(I1, I2, K, B))
        self.assertNotIn((0, 0, 0, -1, 0), r.reachable)
        w = (0, 0, 0, -3, 3, -3, 3)
        q = Query(1, 0, 0, 0)
        full = explore(q, w)
        self.assertEqual((explore(q, w, cap=0).status, explore(q, w, cap=0).Y), ('truncated', None))
        self.assertEqual(explore(q, w, cap=full.inspections).Y, full.Y)
        self.assertIsNone(explore(q, w, cap=full.inspections - 1).Y)
        for bad in ((0,) * 6, (1,) + (0,) * 6, (0,) * 6 + (3.0,)):
            with self.assertRaises(ValueError):
                explore(q, bad)
        with self.assertRaises(ValueError):
            legacy_input_feedback_removed(q, w)
        with self.assertRaises(ValueError):
            vector(N_VECTORS)

    def test_multiple_terminals_and_controls_have_terminals(self):
        multi = 0
        for q in QUERIES:
            for w in ((3, 3, 3, 0, 3, -3, 0), (3, 0, 0, -3, 3, -3, 3)):
                c = controls(q, w)
                self.assertTrue(all(c.values()))
                multi += len(explore(q, w).terminals) > 1
        self.assertGreater(multi, 0)

    def test_local_fields_include_internal_coupling(self):
        q = Query(0, 0, 0, 0)
        for w12 in (-3, 3):
            w = (0, 0, 0, 0, 0, 0, w12)
            d = {i2: delta((0, i2, 0, 0, 0), q, w, I1, 1) for i2 in (-1, 0, 1)}
            self.assertEqual(len(set(d.values())), 3)
            d = {i1: delta((i1, 0, 0, 0, 0), q, w, I2, 1, nf=True) for i1 in (-1, 0, 1)}
            self.assertEqual(len(set(d.values())), 3)



class ScoringRunTests(unittest.TestCase):
    def test_every_scoring_run_for_3prime_and_8_matches_oracle(self):
        from experiments.shared_input_mediation import CHAINS, TEACHERS, Runner, chain, condition_3_prime
        runner = Runner()
        for name, w in TEACHERS.items():
            condition_3_prime(w, runner)
            for direction in ('forward', 'reverse'):
                chain(w, CHAINS[direction][name], direction, runner)
        self.assertGreater(len(runner.cache), 1000)
        for (q, w, clamps, nf), r in runner.cache.items():
            self.assertEqual((r.reachable, r.terminals), oracle(q, w, clamps, nf))
            for c, v in clamps:
                self.assertIn(v, q.allowed(c))
                self.assertEqual({z[c] for z in r.reachable}, {v})

    def test_witness_ablations_reuse_query_and_clamp_pair(self):
        from experiments.shared_input_mediation import CHAINS, TEACHERS, Runner, chain
        name = 'C1'
        runner = Runner()
        out = chain(TEACHERS[name], CHAINS['forward'][name], 'forward', runner)
        for rec in out['pairs']:
            self.assertEqual(set(rec['mask']['d_links']), set(CHAINS['forward'][name]['links']))
            for link in rec['mask']['d_links']:
                x = QUERIES[rec['query']]
                wz = zeroed(TEACHERS[name], link)
                self.assertIn((x, wz, ((A, rec['v']),), False), runner.cache)
                self.assertIn((x, wz, ((A, rec['v2']),), False), runner.cache)

    def test_family_C_donor_and_recipient_are_not_interchanged(self):
        from experiments.shared_input_mediation import CHAINS
        for n in ('C1', 'C2', 'C3'):
            f, r = CHAINS['forward'][n], CHAINS['reverse'][n]
            self.assertNotEqual(f['donor'], f['recipient'])
            self.assertEqual((r['donor'], r['recipient']), (f['recipient'], f['donor']))
        self.assertEqual((CHAINS['forward']['C3']['donor'], CHAINS['forward']['C3']['recipient']), ('i2', 'i1'))


if __name__ == '__main__':
    unittest.main()
