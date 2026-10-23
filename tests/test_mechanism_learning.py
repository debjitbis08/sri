import hashlib
import inspect
import json
from pathlib import Path
import unittest

import src.mechanism_learning as ml
from src.mechanism_learning import (
    BASE_KEYS, TEMPLATES, BlockLearner, Memory, Simulator, VersionSpace, chain_tuples, compute_hypothesis,
    hypothesis_index, observation_keys, tuple_qualifies, vector, witness_keys,
)
from src.shared_input_mediation import QUERIES, Query
from tests.test_shared_input_mediation import oracle


ROOT = Path(__file__).resolve().parents[1]
S1, C2, S2, T1 = (3, 3, 3, 0, 3, -3, 0), (3, 0, 0, -3, 3, -3, -3), (3, 3, 3, 0, 3, 3, 0), (3, 0, 0, -3, 3, -3, 0)
NAMES = ('w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12')


# ---------------------------------------------------------------- independent battery oracle
# Built on the shared-input oracle traversal; does not call production explore, chain or battery code.

def _zero(w, name):
    return tuple(0 if NAMES[j] == name else v for j, v in enumerate(w))


def _pairs(terms):
    return sum(1 << (3 * (t[3] + 1) + (t[4] + 1)) for t in {(None, None, None, t[3], t[4]) for t in terms})


def _triples(terms, other):
    return sum(1 << (9 * (i + 1) + 3 * (j + 1) + (s + 1)) for i, j, s in {(t[0], t[1], t[other]) for t in terms})


class _Oracle:
    def __init__(self, w):
        self.w, self.memo = w, {}

    def T(self, q, w=None, clamps=(), nf=False):
        w = self.w if w is None else w
        k = (q, w, clamps, nf)
        if k not in self.memo:
            self.memo[k] = oracle(q, w, clamps, nf)[1]
        return self.memo[k]


def oracle_chain(o, q, template):
    """All subconditions computed explicitly for every tuple, from oracle terminal sets."""
    spec = {'SF': (3, 4, 0, 0, ('w_1A', 'w_1B')), 'SR': (4, 3, 0, 0, ('w_1B', 'w_1A')),
            'CF': (3, 4, 0, 1, ('w_1A', 'w_12', 'w_2B')), 'CR': (4, 3, 1, 0, ('w_2B', 'w_12', 'w_1A'))}[template]
    src, tgt, j, l, links = spec
    dom = {3: {0: (-1, 0, 1), 1: (0, 1), 2: (-1, 0)}[q.da], 4: {0: (-1, 0, 1), 1: (0, 1), 2: (-1, 0)}[q.db]}[src]
    P = lambda terms, c: {t[c] for t in terms}
    out = set()
    for v in dom:
        for v2 in dom:
            if v == v2:
                continue
            Tv, Tv2 = o.T(q, clamps=((src, v),)), o.T(q, clamps=((src, v2),))
            a = P(Tv, j) != P(Tv2, j) and P(Tv, l) != P(Tv2, l)
            c_full = P(Tv, tgt) != P(Tv2, tgt)
            c_nf = P(o.T(q, clamps=((src, v),), nf=True), tgt) == P(o.T(q, clamps=((src, v2),), nf=True), tgt)
            ec = ((0, q.e1), (1, q.e2))
            c_ie = P(o.T(q, clamps=((src, v),) + ec), tgt) == P(o.T(q, clamps=((src, v2),) + ec), tgt)
            d = all(P(o.T(q, _zero(o.w, k), ((src, v),)), tgt) == P(o.T(q, _zero(o.w, k), ((src, v2),)), tgt)
                    for k in links)
            for r in (-1, 0, 1):
                for r2 in (-1, 0, 1):
                    if r == r2:
                        continue
                    present = r in P(Tv, l) and r2 in P(Tv2, l)
                    b = present and (P(o.T(q, clamps=((src, v), (l, r))), tgt)
                                     != P(o.T(q, clamps=((src, v), (l, r2))), tgt))
                    if a and b and c_full and c_nf and c_ie and d:
                        out.add((v, v2, r, r2))
    return out


def oracle_atoms(w, wkeys):
    """Independent value for every battery atom, plus exhaustive qualifying-tuple sets."""
    o = _Oracle(tuple(w))
    tuples = {}
    for qi, q in enumerate(QUERIES):
        for t in TEMPLATES:
            found = oracle_chain(o, q, t)
            if found:
                tuples[t] = tuples.get(t, []) + [(qi,) + x for x in sorted(found)]
    present = {('B8W', t) + x for t, lst in tuples.items() for x in lst}
    vec = []
    for key in list(BASE_KEYS) + list(wkeys):
        item = key[0]
        if item == 'B8W':
            vec.append(int(tuple(key) in present))
            continue
        qi = key[2] if item in ('B5', 'B8E') else key[1]
        q = QUERIES[qi]
        if item == 'B1':
            vec.append(_pairs(o.T(q)))
        elif item == 'B2':
            vec.append(_pairs(o.T(q, nf=True)))
        elif item == 'B3':
            vec.append(_pairs(o.T(q, clamps=((0, q.e1), (1, q.e2)))))
        elif item == 'B4':
            vec.append(_pairs(o.T(q, _zero(_zero(o.w, 'w_KA'), 'w_KB'))))
        elif item == 'B5':
            vec.append(_pairs(o.T(q, _zero(o.w, key[1]))))
        elif item == 'B6':
            vec.append(_triples(o.T(q, clamps=((3, key[2]),)), 4))
        elif item == 'B7':
            vec.append(_triples(o.T(q, clamps=((4, key[2]),)), 3))
        elif item == 'B8E':
            vec.append(int(any(x[0] == qi for x in tuples.get(key[1], []))))
    return vec, tuples


TEACHER_RAW = {}


def raw(w):
    if w not in TEACHER_RAW:
        TEACHER_RAW[w] = compute_hypothesis(hypothesis_index(w))
    return TEACHER_RAW[w]


def wkeys():
    return witness_keys([raw(w)[1] for w in (S1, C2, S2, T1)])


class BatteryTests(unittest.TestCase):
    def test_indexing(self):
        self.assertEqual([hypothesis_index(vector(h)) for h in range(2187)], list(range(2187)))
        self.assertEqual(vector(hypothesis_index(S1)), S1)

    def test_teacher_batteries_match_oracle(self):
        wk = wkeys()
        for w in (S1, C2):
            vals, tuples, _ = raw(w)
            ovec, otup = oracle_atoms(w, wk)
            self.assertEqual(ml.atom_vector(vals, tuples, wk), tuple(ovec))
            self.assertEqual({k: sorted(v) for k, v in tuples.items()}, {k: sorted(v) for k, v in otup.items()})

    def test_b8_reproduces_frozen_shared_input_tuples(self):
        si = json.loads((ROOT / 'reports/shared-input-mediation-v1-gate.json').read_text())
        for name, w, fwd, rev in (('S1', S1, 'SF', 'SR'), ('S2', S2, 'SF', 'SR'), ('C2', C2, 'CF', 'CR')):
            t = raw(w)[1]
            self.assertEqual(sorted(map(list, t.get(fwd, []))),
                             sorted(si['teachers'][name]['conditions']['8_causal_chain_A_input_B']['qualifying']))
            self.assertEqual(sorted(map(list, t.get(rev, []))),
                             sorted(si['teachers'][name]['reverse_chain_B_input_A']['qualifying']))
        self.assertTrue(raw(S2)[1].get('SR'))              # S2 reverse chain is positive
        self.assertFalse(raw(S2)[1].get('SF'))

    def test_b8_true_false_and_invalid_tuples(self):
        qi, v, v2, r, r2 = raw(S1)[1]['SF'][0]
        q = QUERIES[qi]
        self.assertTrue(tuple_qualifies(q, S1, 'SF', v, v2, r, r2))
        self.assertFalse(tuple_qualifies(q, S2, 'SF', v, v2, r, r2))
        self.assertFalse(tuple_qualifies(q, T1, 'SF', v, v2, r, r2))
        with self.assertRaises(ValueError):
            tuple_qualifies(Query(0, 0, 1, 0), S1, 'SF', -1, 0, 0, 1)       # -1 outside D1
        with self.assertRaises(ValueError):
            tuple_qualifies(q, S1, 'SF', v, v2, 0, 0)
        # Recipient values absent from the source-clamped terminal projections score false.
        sim = ml.Sim()
        Tv = sim.terminals(q, S1, ((3, v),))
        absent = [x for x in (-1, 0, 1) if x not in {t[0] for t in Tv}]
        for x in absent:
            for y in (-1, 0, 1):
                if y != x:
                    self.assertFalse(tuple_qualifies(q, S1, 'SF', v, v2, x, y))

    def test_existence_atoms_aggregate_per_query(self):
        vals, tuples, _ = raw(C2)
        for qi in range(len(QUERIES)):
            for t in TEMPLATES:
                a = BASE_KEYS.index(('B8E', t, qi))
                self.assertEqual(vals[a], int(any(x[0] == qi for x in tuples.get(t, []))))
        fwd_any = any(vals[BASE_KEYS.index(('B8E', t, qi))] for t in ('SF', 'CF') for qi in range(81))
        self.assertTrue(fwd_any)

    def test_zero_chain_link_forbids_template(self):
        for w in (S2, T1):
            vals, tuples, _ = raw(w)
            self.assertNotIn('CF', tuples)                 # w_12 = 0 is a chain link of CF
        self.assertNotIn('SF', raw(T1)[1])                 # w_1B = 0 is a chain link of SF

    def test_core_sign_symmetry_preserves_battery(self):
        wk = wkeys()
        for w in (S1, C2, (0, 3, -3, 3, 3, 0, 3)):
            m = w[:4] + (-w[4], -w[5]) + w[6:]
            self.assertEqual(ml.reconstruct(hypothesis_index(w), wk), ml.reconstruct(hypothesis_index(m), wk))

    def test_partition_scoped_equivalence_matches_direct_comparison(self):
        from experiments.mechanism_learning import atom_query
        wk = wkeys()
        keys = list(BASE_KEYS) + wk
        vs = {w: ml.reconstruct(hypothesis_index(w), wk) for w in (S1, S2, T1, (3, 3, 3, 0, -3, 3, 0))}
        Q = set(range(10, 40))
        for a in vs:
            for b in vs:
                scoped = all(vs[a][i] == vs[b][i] for i, k in enumerate(keys)
                             if k[0] != 'B1' and atom_query(k) in Q)
                direct = all(vs[a][keys.index(k)] == vs[b][keys.index(k)] for k in keys
                             if k[0] in ml.MECHANISM_ITEMS and (k[2] if k[0] in ('B5', 'B8E', 'B8W') else k[1]) in Q)
                self.assertEqual(scoped, direct)
        self.assertEqual(vs[S1], vs[(3, 3, 3, 0, -3, 3, 0)])


class LearnerTests(unittest.TestCase):
    def fb(self, w, q, arm):
        sim = Simulator(arm)
        return {k: sim.predict(w, q, k) for k in observation_keys(q, arm)}

    def test_api_isolation_and_simulator_keys(self):
        for cls in (VersionSpace, BlockLearner, Memory):
            self.assertEqual(list(inspect.signature(cls.update).parameters), ['self', 'x', 'feedback'])
        sim = Simulator('O')
        q = QUERIES[40]
        with self.assertRaises(PermissionError):
            sim.predict(S1, q, ('B6', 0))
        with self.assertRaises(PermissionError):
            sim.predict(S1, q, ('B2',))
        self.assertEqual(observation_keys(q, 'I'), [('B1',)] + [('B6', v) for v in q.allowed(3)]
                         + [('B7', v) for v in q.allowed(4)])
        self.assertEqual(observation_keys(q, 'O'), [('B1',)])
        self.assertEqual(ml.pair_mask([(0, 0), (0, 0)]), ml.pair_mask([(0, 0)]))
        src = inspect.getsource(ml.VersionSpace) + inspect.getsource(ml.BlockLearner) + inspect.getsource(ml.Simulator)
        for forbidden in ('TEACHER', 'BASE_KEYS', "'B8", 'table', "'B2", "'B5"):
            self.assertNotIn(forbidden, src)

    def test_contradictory_feedback(self):
        vs = VersionSpace(Simulator('O'))
        q = QUERIES[40]
        self.assertEqual(vs.update(q, {('B1',): frozenset({(0, 0)})}), 'updated')
        before = vs.bits
        self.assertEqual(vs.update(q, {('B1',): frozenset({(1, 1)})}), 'inconsistent_feedback')
        self.assertEqual((vs.bits, vs.failed), (before, True))

    def test_positive_only_containment_in_both_arms(self):
        q = QUERIES[46]
        for arm in ('O', 'I'):
            f = self.fb(S1, q, arm)
            vs = VersionSpace(Simulator(arm), positive_only=True)
            vs.update(q, f)
            sim = Simulator(arm)
            self.assertTrue(vs.bits >> hypothesis_index(S1) & 1)
            for h in range(0, 2187, 97):
                if vs.bits >> h & 1:
                    for k, s in f.items():
                        self.assertLessEqual(s, sim.predict(vector(h), q, k))
        empty = VersionSpace(Simulator('O'), positive_only=True)
        empty.update(q, {('B1',): frozenset()})
        self.assertEqual(empty.bits, ml.ALL_BITS)

    def test_arm_I_survivors_subset_of_arm_O_and_order_independent(self):
        ids = [10, 14, 16, 32, 33]
        for w in (S1, C2):
            finals = {}
            for arm in ('O', 'I'):
                for order in (ids, ids[::-1]):
                    vs = VersionSpace(Simulator(arm))
                    for i in order:
                        vs.update(QUERIES[i], self.fb(w, QUERIES[i], arm))
                    finals.setdefault(arm, set()).add(vs.bits)
            self.assertEqual((len(finals['O']), len(finals['I'])), (1, 1))
            self.assertEqual(next(iter(finals['I'])) & ~next(iter(finals['O'])), 0)

    def test_block_learner_rules_and_typed_loss(self):
        bl = BlockLearner(Simulator('I'))
        self.assertEqual(len(bl.neighbors()), 26)
        q = QUERIES[46]
        f = self.fb(S1, q, 'I')
        before = bl.w
        bl.update(q, f)
        changed = [j for j in range(7) if bl.w[j] != before[j]]
        self.assertTrue(any(set(changed) <= set(b) for b in ml.BLOCKS))
        self.assertEqual(bl.evaluations, 27 * len(f))         # every candidate scored on every typed key
        self.assertEqual(set(vars(bl)), {'sim', 'w', 'evaluations', 'commits', 'coefficients_changed'})
        expected = sum(len(Simulator('I').predict(bl.w, q, k) ^ s) for k, s in f.items())
        self.assertEqual(bl.loss(bl.w, q, f), expected)

    def test_memory_unknown_for_unobserved_keys(self):
        m = Memory()
        q = QUERIES[46]
        f = self.fb(S1, q, 'O')
        m.update(q, f)
        self.assertEqual(m.predict(q, ('B1',)), f[('B1',)])
        self.assertIsNone(m.predict(q, ('B6', 0)))
        self.assertIsNone(m.predict(QUERIES[47], ('B1',)))


class ProvenanceTests(unittest.TestCase):
    def test_prior_artifacts_unchanged(self):
        m = json.loads((ROOT / 'reports/learned-input-mediation-v1-manifest.json').read_text())
        for p, h in m['prior_artifacts'].items():
            self.assertEqual(hashlib.sha256((ROOT / p).read_bytes()).hexdigest(), h, p)
        self.assertEqual(hashlib.sha256((ROOT / m['contract']).read_bytes()).hexdigest(), m['contract_sha256'])

    def test_completion_requires_passing_tests(self):
        from experiments.mechanism_learning import completion_status
        ok, bad = dict(passed=True), dict(passed=False)
        self.assertEqual(completion_status(ok, ok), 'complete')
        self.assertEqual(completion_status(ok, bad), 'stopped_failed_tests')
        self.assertEqual(completion_status(bad, ok), 'stopped_failed_correctness_gate')
        self.assertEqual(completion_status(bad, bad), 'stopped_failed_correctness_gate')


if __name__ == '__main__':
    unittest.main()
