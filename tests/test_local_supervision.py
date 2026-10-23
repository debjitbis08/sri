import hashlib
import inspect
import json
from pathlib import Path
import unittest

from src.local_supervision import (
    MIXED_COMBINATIONS, ProjectedSimulator, axis_phase, coverage_subset, feedback_cost, make_simulator,
    observation_keys, oplus_order, pool_timelines, project, rate, timeline,
)
from src.mechanism_learning import BlockLearner, Memory, Simulator, VersionSpace
from src.shared_input_mediation import A, B, QUERIES
from tests.test_shared_input_mediation import oracle


ROOT = Path(__file__).resolve().parents[1]
S1, C2 = (3, 3, 3, 0, 3, -3, 0), (3, 0, 0, -3, 3, -3, -3)
SPLITS = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-splits.json').read_text())


def teacher_feedback(w, x, cond):
    sim = make_simulator(cond)
    return {k: sim.predict(w, x, k) for k in observation_keys(x, cond)}


class ObservationTests(unittest.TestCase):
    def test_key_schemas(self):
        x = QUERIES[10]
        self.assertEqual(observation_keys(x, 'O'), [('B1',)])
        self.assertEqual(observation_keys(x, 'O+'), [('B1',)])
        ib = observation_keys(x, 'IB')
        self.assertEqual(ib[0], ('B1',))
        self.assertEqual([k[1] for k in ib if k[0] == 'B6x'], sorted(x.allowed(A)))
        self.assertEqual([k[1] for k in ib if k[0] == 'B7x'], sorted(x.allowed(B)))
        self.assertEqual({k[0] for k in ib}, {'B1', 'B6x', 'B7x'})
        self.assertEqual(len(observation_keys(x, 'I')), len(ib))

    def test_projection_against_oracle(self):
        sim = ProjectedSimulator()
        for w in (S1, C2, (0,) * 7, (-3, 3, 0, 3, -3, 0, 3)):
            for qi in (0, 10, 40, 80):
                x = QUERIES[qi]
                for c, other, item in ((A, B, 'B6x'), (B, A, 'B7x')):
                    for v in x.allowed(c):
                        expect = frozenset(t[other] for t in oracle(x, w, ((c, v),))[1])
                        self.assertEqual(sim.predict(w, x, (item, v)), expect)

    def test_ib_service_returns_modes_only(self):
        sim = ProjectedSimulator()
        x = QUERIES[14]
        for key in observation_keys(x, 'IB')[1:]:
            y = sim.predict(S1, x, key)
            self.assertIsInstance(y, frozenset)
            self.assertTrue(all(type(v) is int and v in (-1, 0, 1) for v in y))
        with self.assertRaises(PermissionError):
            sim.predict(S1, x, ('B6', x.allowed(A)[0]))
        with self.assertRaises(PermissionError):
            Simulator('I').predict(S1, x, ('B6x', x.allowed(A)[0]))

    def test_projection_deduplicates_before_loss_and_cost(self):
        triples = frozenset({(-1, 0, 1), (0, 1, 1), (1, 1, 0)})
        self.assertEqual(project(triples), frozenset({0, 1}))
        x = QUERIES[14]
        for qi in range(len(QUERIES)):
            fi = teacher_feedback(C2, QUERIES[qi], 'I')
            if any(len(f) > len(project(f)) for k, f in fi.items() if k[0] != 'B1'):
                x = QUERIES[qi]
                break
        fi, fib = teacher_feedback(C2, x, 'I'), teacher_feedback(C2, x, 'IB')
        for k, f in fi.items():
            if k[0] != 'B1':
                self.assertEqual(fib[(k[0] + 'x', k[1])], project(f))
        ci, cib = feedback_cost(fi), feedback_cost(fib)
        self.assertEqual(cib['positive_tuples'], sum(len(f) for f in fib.values()))
        self.assertLess(cib['positive_tuples'], ci['positive_tuples'])
        L = BlockLearner(ProjectedSimulator())
        self.assertEqual(L.loss((0,) * 7, x, fib),
                         sum(len(ProjectedSimulator().predict((0,) * 7, x, k) ^ f) for k, f in fib.items()))

    def test_cost_accounting_per_key_type(self):
        x = QUERIES[10]
        nA, nB = len(x.allowed(A)), len(x.allowed(B))
        o, ib, i = (feedback_cost(teacher_feedback(S1, x, c)) for c in ('O', 'IB', 'I'))
        self.assertEqual((o['entries'], o['supervised_slots'], o.get('clamp_runs', 0), o['canonical_requests']), (1, 9, 0, 1))
        self.assertEqual((ib['entries'], ib['supervised_slots'], ib['clamp_runs']), (1 + nA + nB, 9 + 3 * (nA + nB), nA + nB))
        self.assertEqual((i['entries'], i['supervised_slots'], i['clamp_runs']), (1 + nA + nB, 9 + 27 * (nA + nB), nA + nB))

    def test_version_space_subset_chain_on_a_prefix(self):
        sims = {c: make_simulator(c) for c in ('O', 'IB', 'I')}
        V = {c: VersionSpace(sims[c]) for c in sims}
        for qi in SPLITS['splits'][0]['orders'][0][:2]:
            x = QUERIES[qi]
            for c in V:
                V[c].update(x, teacher_feedback(S1, x, c))
        self.assertEqual(V['I'].bits & ~V['IB'].bits, 0)
        self.assertEqual(V['IB'].bits & ~V['O'].bits, 0)


class SplitTests(unittest.TestCase):
    def test_coverage_subset(self):
        for s in SPLITS['splits']:
            cov = coverage_subset(s['seed'], s['mixed_evidence'], QUERIES)
            self.assertEqual(cov, coverage_subset(s['seed'], list(reversed(s['mixed_evidence'])), QUERIES))
            self.assertEqual(len(cov), 12)
            self.assertTrue(set(cov) <= set(s['mixed_evidence']))
            for e in MIXED_COMBINATIONS:
                self.assertEqual(sum((QUERIES[q].e1, QUERIES[q].e2) == e for q in cov), 3)
            rest = [q for q in s['mixed_evidence'] if q not in cov]
            self.assertEqual(len(rest), 24)
            self.assertFalse(set(cov) & set(s['train']) | set(cov) & set(s['within_context']))

    def test_coverage_digest_rule(self):
        s = SPLITS['splits'][2]
        cov = coverage_subset(s['seed'], s['mixed_evidence'], QUERIES)
        e = MIXED_COMBINATIONS[0]
        pool = [q for q in s['mixed_evidence'] if (QUERIES[q].e1, QUERIES[q].e2) == e]
        d = {q: hashlib.sha256(f'local-learning-supervision-v1:coverage:{s["seed"]}:{QUERIES[q].e1},{QUERIES[q].e2},'
                                f'{QUERIES[q].da},{QUERIES[q].db}'.encode()).hexdigest() for q in pool}
        self.assertEqual(sorted(q for q in cov if q in pool), sorted(sorted(pool, key=lambda q: (d[q], q))[:3]))

    def test_oplus_order(self):
        s = SPLITS['splits'][1]
        cov = coverage_subset(s['seed'], s['mixed_evidence'], QUERIES)
        order = oplus_order(s['seed'], 3, s['orders'][3], cov)
        self.assertEqual(order[:15], s['orders'][3])
        self.assertEqual(sorted(order[15:]), cov)
        self.assertEqual(order, oplus_order(s['seed'], 3, s['orders'][3], list(reversed(cov))))
        self.assertNotEqual(order[15:], oplus_order(s['seed'], 4, s['orders'][4], cov)[15:])

    def test_oplus_prefix_identity(self):
        s = SPLITS['splits'][0]
        cov = coverage_subset(s['seed'], s['mixed_evidence'], QUERIES)
        axis = s['orders'][0][:4]
        plus = oplus_order(s['seed'], 0, axis, cov[:2])
        learners = {}
        for name, order in (('O', axis), ('O+', plus)):
            L, M, states = BlockLearner(make_simulator(name)), Memory(), []
            for qi in order:
                x = QUERIES[qi]
                fb = teacher_feedback(S1, x, name)
                L.update(x, fb)
                M.update(x, fb)
                states.append((L.w, dict(M.table)))
            learners[name] = states
        self.assertEqual(learners['O'], learners['O+'][:4])


class TimelineTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(timeline(False, [False, False])['category'], 'never_acquired')
        self.assertEqual(timeline(False, [True, True, True])['category'], 'held')
        self.assertEqual(timeline(False, [False, True, True])['category'], 'held')
        self.assertEqual(timeline(False, [True, False, True])['category'], 'recovered')
        self.assertEqual(timeline(False, [True, True, False])['category'], 'forgotten')

    def test_latency_censoring_and_cycles(self):
        t = timeline(False, [False, False, True, False, True, False, True])
        self.assertEqual((t['latency'], t['loss_events'], t['reacquisitions']), (2, 2, 2))
        self.assertEqual(timeline(False, [True])['latency'], 0)
        never = timeline(False, [False, False, False])
        self.assertIsNone(never['latency'])
        p = pool_timelines([never, timeline(False, [True])])
        self.assertEqual(p['latency']['censored_never_acquired'], 1)
        self.assertEqual(p['latency']['acquired'], 1)
        self.assertEqual(p['latency']['mean'], 0)
        self.assertEqual(timeline(False, [False, True])['reacquisitions'], 0)

    def test_before_presentation_and_break(self):
        t = timeline(True, [False, True])
        self.assertTrue(t['broken_at_presentation'])
        self.assertFalse(t['acquired_at_presentation'])
        self.assertEqual(t['category'], 'held')
        p = pool_timelines([timeline(True, [True]), timeline(False, [True]), t])
        self.assertEqual(p['newly_acquired_at_presentation'], 1)
        self.assertEqual(p['correct_before'], 2)
        self.assertEqual(p['broken_at_presentation'], 1)

    def test_rates_and_null_denominators(self):
        p = pool_timelines([timeline(False, [True, False]), timeline(False, [True, True]),
                            timeline(False, [False]), timeline(False, [True, False, True])])
        self.assertEqual((p['ever_acquired'], p['forgotten'], p['final_correct'], p['presented']), (3, 1, 2, 4))
        self.assertEqual(p['forgetting_rate'], round(1 / 3, 6))
        self.assertEqual(p['final_retention'], 0.5)
        self.assertEqual(p['ever_lost_rate'], round(2 / 3, 6))
        empty = pool_timelines([])
        self.assertIsNone(empty['final_retention'])
        self.assertIsNone(empty['forgetting_rate'])
        self.assertIsNone(rate(0, 0))
        self.assertIsNone(pool_timelines([timeline(False, [False])])['forgetting_rate'])

    def test_oplus_axis_phase(self):
        c = axis_phase([True, True, False, False],
                       [[True, True], [False, True], [False, True], [False, False]],
                       [True, True, True, False])
        self.assertEqual(c['correct_at_15'], 2)
        self.assertEqual(c['correct_at_15_ever_lost'], 1)
        self.assertEqual(c['correct_at_15_wrong_at_27'], 0)
        self.assertEqual(c['wrong_at_15_acquired_during_coverage'], 1)
        self.assertEqual(c['wrong_at_15_correct_at_27'], 1)
        self.assertEqual(c['wrong_at_15_wrong_at_27'], 1)
        self.assertEqual(c['correct_at_15_correct_at_27'], 2)


class BoundaryTests(unittest.TestCase):
    def test_evaluator_paths_take_no_learner_simulator(self):
        from experiments import local_supervision as ls
        for f in (ls.feedback, ls.map_ok, ls.b1_ok, ls.ib_memory):
            self.assertNotIn('sim', inspect.signature(f).parameters)
        sim = ProjectedSimulator()
        sim.predict(S1, QUERIES[10], ('B1',))
        before = (sim.requests, len(sim.memo))
        ls.ib_boundary_ok({('B1',): frozenset(), ('B6x', 0): frozenset({1})})
        self.assertEqual((sim.requests, len(sim.memo)), before)

    def test_ib_boundary_check(self):
        from experiments.local_supervision import ib_boundary_ok
        self.assertTrue(ib_boundary_ok({('B1',): frozenset({(0, 0)}), ('B6x', 0): frozenset({1, -1})}))
        self.assertFalse(ib_boundary_ok({('B6', 0): frozenset({(0, 0, 1)})}))
        self.assertFalse(ib_boundary_ok({('B6x', 0): frozenset({(0, 0, 1)})}))

    def test_completion_requires_passing_tests(self):
        from experiments.local_supervision import completion_status
        self.assertEqual(completion_status(dict(passed=True), dict(passed=False)), 'stopped_failed_tests')

    def test_prior_artifacts_unchanged(self):
        path = ROOT / 'reports/local-learning-supervision-v1-manifest.json'
        if not path.exists():
            self.skipTest('manifest not frozen yet')
        m = json.loads(path.read_text())
        for p, h in m['prior_artifacts'].items():
            self.assertEqual(hashlib.sha256((ROOT / p).read_bytes()).hexdigest(), h, p)
        self.assertEqual(hashlib.sha256((ROOT / m['contract']).read_bytes()).hexdigest(), m['contract_sha256'])


if __name__ == '__main__':
    unittest.main()
