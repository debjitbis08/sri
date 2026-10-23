import hashlib
import inspect
import json
from pathlib import Path
import random
import unittest

from src.local_supervision import make_simulator, observation_keys
from src.mechanism_learning import BLOCKS, BlockLearner, N_VECTORS, QUERIES, hypothesis_index, vector
from src.update_diagnosis import (
    VariantLearner, b0_choice, blocks_differ, class_distance, classify, n1, n2, proximity, repetition_probe,
    reverse_reachable, witness,
)


ROOT = Path(__file__).resolve().parents[1]
Z = (0,) * 7
SPLITS = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-splits.json').read_text())


def edit(w, block, vals):
    u = list(w)
    for j, v in zip(BLOCKS[block], vals):
        u[j] = v
    return tuple(u)


def table_loss(table, default):
    return lambda u: table.get(u, default)


class MockSim:
    """predict returns a set whose size is the declared loss against empty feedback."""
    def __init__(self, table, default):
        self.table, self.default = table, default
        self.requests = 0

    def predict(self, w, x, key):
        self.requests += 1
        return frozenset(range(self.table.get((w, x), self.table.get(w, self.default))))


class NeighbourhoodTests(unittest.TestCase):
    def test_counts_and_block_differences(self):
        rng = random.Random(0)
        hs = [0, N_VECTORS - 1] + [rng.randrange(N_VECTORS) for _ in range(10)]
        for w12 in (-3, 0, 3):
            hs.append(hypothesis_index((3, -3, 0, 3, -3, 0, w12)))
        for h in hs:
            w = vector(h)
            a, b = n1(w), n2(w)
            self.assertEqual((len(a), len(set(a)), len(b), len(set(b))), (26, 26, 240, 240))
            self.assertTrue(all(blocks_differ(w, u) == 1 for u in a))
            self.assertTrue(all(blocks_differ(w, u) == 2 for u in b))
            self.assertFalse(set(a) & set(b))

    def test_n1_matches_v1_order(self):
        L = BlockLearner(None)
        for h in (0, 5, 1093):
            L.w = vector(h)
            self.assertEqual(n1(L.w), L.neighbors())


class ClassificationTests(unittest.TestCase):
    def test_e0_and_ties_at_a_local_minimum(self):
        self.assertEqual(classify(Z, lambda u: 0, lambda u: 0, n1(Z), None)['cls'], 'E0')
        tie = {u: 4 for u in n1(Z)}                    # equal-loss neighbours only: still stuck
        tie[Z] = 4
        T = edit(edit(edit(Z, 0, (3, 3)), 1, (3, 3)), 2, (3, 3))
        tie[T] = 0
        r = classify(Z, table_loss(tie, 9), lambda u: 0, n1(Z), b0_choice(Z, table_loss(tie, 9)))
        self.assertEqual((r['cls'], r['min_blocks_to_lower']), ('E1b', 3))

    def test_e1a_two_block_improvement(self):
        lo = {Z: 5, edit(edit(Z, 0, (3, 3)), 1, (3, 3)): 2}
        loss = table_loss(lo, 7)
        self.assertEqual(classify(Z, loss, lambda u: 0, n1(Z), b0_choice(Z, loss))['cls'], 'E1a')
        r = classify(Z, loss, lambda u: 0, n1(Z) + n2(Z), edit(edit(Z, 0, (3, 3)), 1, (3, 3)), scope_n2=True)
        self.assertEqual(r['cls'], 'E2a')              # SCOPE: the same state is not stuck, so E1a cannot occur

    def test_e2b_versus_e2c(self):
        good, bad = edit(Z, 0, (3, 3)), edit(Z, 0, (-3, -3))
        lo = {Z: 5, good: 2, bad: 1}
        loss = table_loss(lo, 9)
        chosen = b0_choice(Z, loss)
        self.assertEqual(chosen, bad)
        r = classify(Z, loss, lambda u: int(u == bad), n1(Z), chosen)
        self.assertEqual((r['cls'], r['e2b_kind'], r['best_preserving_loss']), ('E2b', 'tradeoff', 2))
        lo[good] = 1                                    # equal loss: tie-selection opportunity
        loss = table_loss(lo, 9)
        chosen = b0_choice(Z, loss)
        brk = lambda u: int(u == chosen)
        self.assertEqual(classify(Z, loss, brk, n1(Z), chosen)['e2b_kind'], 'tie_selection')
        r = classify(Z, loss, lambda u: int(u in (good, bad)), n1(Z), chosen)
        self.assertEqual(r['cls'], 'E2c')
        self.assertEqual(classify(Z, loss, lambda u: 0, n1(Z), chosen)['cls'], 'E2a')

    def test_proximity_categories(self):
        T = edit(Z, 0, (3, 3))
        members = [T, edit(Z, 1, (3, 3))]
        lo = {Z: 4, T: 4, members[1]: 6}
        p = proximity(Z, table_loss(lo, 9), members)
        self.assertEqual(p['d'], 1)
        self.assertEqual(p['N1']['counts'], {'improving': 0, 'neutral': 1, 'worse': 1})
        self.assertEqual(p['N1']['best'], 'neutral')
        self.assertEqual(class_distance(Z, members), 1)
        self.assertEqual(p['N2']['edits'], 0)


class GraphTests(unittest.TestCase):
    # nodes 0..5 on a path-like graph; target 0
    ADJ = [[1, 4], [0, 2], [1, 3], [2], [0, 5], [4]]

    def test_strict_route_through_increasing_distance(self):
        # 3 -> 2 -> 1 -> 0 descends; node 5 reaches 0 only via 4 (strict as well)
        losses = [0, 1, 2, 3, 2, 4]
        down = reverse_reachable(losses, self.ADJ, [0], strict=True)
        self.assertEqual(witness(down, 3), [3, 2, 1, 0])
        self.assertIn(5, down)

    def test_plateau_bridge(self):
        losses = [0, 2, 2, 3, 1, 1]
        down = reverse_reachable(losses, self.ADJ, [0], strict=True)
        le = reverse_reachable(losses, self.ADJ, [0], strict=False)
        self.assertNotIn(2, down)                       # 2 -> 1 is a plateau
        self.assertIn(2, le)
        self.assertEqual(witness(le, 2), [2, 1, 0])
        self.assertIn(5, le)

    def test_uphill_required(self):
        losses = [0, 5, 2, 3, 1, 1]
        le = reverse_reachable(losses, self.ADJ, [0], strict=False)
        self.assertNotIn(2, le)
        self.assertNotIn(3, le)
        self.assertTrue(set(reverse_reachable(losses, self.ADJ, [0], strict=True)) <= set(le))

    def test_greedy_repetition_misses_an_existing_descending_route(self):
        A, B = edit(Z, 2, (3, 3)), edit(Z, 0, (3, 3))
        T = edit(B, 1, (3, 3))
        lo = {Z: 5, A: 1, B: 3, T: 0}
        loss = table_loss(lo, 10)
        probe = repetition_probe(Z, loss, lambda u: set())
        self.assertEqual(probe['terminal'], A)
        idx = {vector(h): h for h in range(N_VECTORS)}
        adj = [[hypothesis_index(u) for u in n1(vector(h))] for h in range(N_VECTORS)]
        L = [loss(vector(h)) for h in range(N_VECTORS)]
        down = reverse_reachable(L, adj, [idx[T]], strict=True)
        self.assertIn(idx[Z], down)
        self.assertNotIn(idx[A], down)

    def test_local_minimum_nesting(self):
        from experiments.update_diagnosis import stream_end

        class Fake:
            pass
        T = edit(edit(Z, 0, (3, 3)), 1, (3, 3))
        members = [T, edit(T, 2, (3, 3))]
        cases = {'two_block_only': ({Z: 5, T: 0}, True, False), 'both': ({Z: 5, T: 0, members[1]: 0}, True, False),
                 'neither': ({Z: 5, edit(Z, 3, (3,)): 1, T: 0}, False, False)}
        for name, (lo, n1_min, le2_min) in cases.items():
            f = Fake()
            f.loss = {('S1', 0): [lo.get(vector(h), 9) for h in range(N_VECTORS)]}
            e = stream_end(f, 'S1', [0], hypothesis_index(Z), members, {hypothesis_index(m) for m in members})
            self.assertEqual((e['N1_local_minimum'], e['N_le2_local_minimum']), (n1_min, le2_min), name)
            self.assertTrue(e['N1_local_minimum'] or not e['N_le2_local_minimum'])
        f = Fake()
        f.loss = {('S1', 0): [3 if vector(h) == Z else 9 for h in range(N_VECTORS)]}
        e = stream_end(f, 'S1', [0], hypothesis_index(Z), members, set())
        self.assertEqual((e['N1_local_minimum'], e['N_le2_local_minimum'], e['cumulative_loss']), (True, True, 3))


class RepetitionTests(unittest.TestCase):
    def test_probe_two_commits_and_within_experience_recovery(self):
        A = edit(Z, 0, (3, 3))
        B = edit(A, 1, (3, 3))
        lo = {Z: 4, A: 2, B: 0}
        probe = repetition_probe(Z, table_loss(lo, 9), lambda u: {7} if u == A else set())
        self.assertEqual((probe['commits'], probe['decisions'], probe['candidate_losses']), (2, 3, 81))
        self.assertEqual(probe['lost_at_any_commit'], [7])
        self.assertEqual(probe['wrong_at_end'], [])

    def test_repeat_stopping_check_cost(self):
        x = QUERIES[10]
        fb = {('B1',): frozenset()}
        A = edit(Z, 0, (3, 3))
        B = edit(A, 1, (3, 3))
        for table, commits in (({Z: 0}, 0), ({Z: 2, A: 1, A + (): 1}, 1), ({Z: 4, A: 2, B: 0}, 2)):
            L = VariantLearner(MockSim(table, 9), 'REPEAT')
            trace = L.update(x, fb)
            self.assertEqual(L.commits, commits)
            self.assertEqual(L.decisions, commits + 1)
            self.assertEqual(L.candidate_losses, 27 * (commits + 1))
            self.assertFalse(trace[-1]['committed'])


class ReplayTests(unittest.TestCase):
    def test_damaging_edit_can_win_on_summed_past_loss(self):
        x, x1, x2 = QUERIES[10], QUERIES[11], QUERIES[12]
        P, D = edit(Z, 0, (3, 3)), edit(Z, 1, (3, 3))   # preserving (0,4) versus damaging (1,0)
        table = {Z: 2, P: 1, D: 1, (P, x1): 0, (P, x2): 4, (D, x1): 1, (D, x2): 0, (Z, x1): 0, (Z, x2): 4}
        L = VariantLearner(MockSim(table, 5), 'REPLAY')
        L.history = [(x1, {('B1',): frozenset()}), (x2, {('B1',): frozenset()})]
        L.update(x, {('B1',): frozenset()})
        self.assertEqual(L.w, D)
        self.assertEqual(L.history_predictions, 27 * 2)

    def test_replay_commits_exactly_when_b0_would(self):
        sim = make_simulator('IB')
        R = VariantLearner(sim, 'REPLAY')
        S1 = (3, 3, 3, 0, 3, -3, 0)
        for qi in SPLITS['splits'][0]['orders'][0][:6]:
            x = QUERIES[qi]
            fb = {k: sim.predict(S1, x, k) for k in observation_keys(x, 'IB')}
            B = VariantLearner(make_simulator('IB'), 'B0')
            B.w = R.w
            b_commit = B.decide(x, fb)[0]
            r_trace = R.update(x, fb)
            self.assertEqual(r_trace[0]['committed'], b_commit)

    def test_first_experience_selection_equals_b0(self):
        sim = make_simulator('IB')
        S1 = (3, 3, 3, 0, 3, -3, 0)
        x = QUERIES[SPLITS['splits'][1]['orders'][2][0]]
        fb = {k: sim.predict(S1, x, k) for k in observation_keys(x, 'IB')}
        R, B = VariantLearner(sim, 'REPLAY'), VariantLearner(sim, 'B0')
        R.update(x, fb)
        B.update(x, fb)
        self.assertEqual(R.w, B.w)


class VariantTests(unittest.TestCase):
    def test_cost_counters(self):
        S1 = (3, 3, 3, 0, 3, -3, 0)
        order = SPLITS['splits'][0]['orders'][0][:3]
        keys = [len(observation_keys(QUERIES[q], 'IB')) for q in order]
        for variant, cand in (('B0', 27), ('SCOPE', 267), ('REPEAT', 27), ('REPLAY', 27)):
            sim = make_simulator('IB')
            L = VariantLearner(sim, variant)
            decs = []
            for qi in order:
                x = QUERIES[qi]
                fb = {k: sim.predict(S1, x, k) for k in observation_keys(x, 'IB')}
                decs.append(len(L.update(x, fb)))
            self.assertEqual(L.candidate_losses, cand * sum(decs))
            self.assertEqual(L.evaluations, sum(cand * d * k for d, k in zip(decs, keys)))
            hist = sum(cand * sum(keys[:t]) for t in range(3)) if variant == 'REPLAY' else 0
            self.assertEqual(L.history_predictions, hist)
            self.assertEqual(len(L.history), 3 if variant == 'REPLAY' else 0)
            if variant != 'REPEAT':
                self.assertEqual(decs, [1, 1, 1])

    def test_b0_matches_v1_block_learner_and_frozen_stream(self):
        lls = json.loads((ROOT / 'reports/local-learning-supervision-v1-learning.json').read_text())
        ref = next(s for s in lls['streams'] if s['condition'] == 'IB' and s['teacher'] == 'C2'
                   and s['split'] == 3 and s['order'] == 1)
        C2 = (3, 0, 0, -3, 3, -3, -3)
        sim_a, sim_b = make_simulator('IB'), make_simulator('IB')
        A, B = VariantLearner(sim_a, 'B0'), BlockLearner(sim_b)
        for qi in ref['queries']:
            x = QUERIES[qi]
            fb = {k: sim_a.predict(C2, x, k) for k in observation_keys(x, 'IB')}
            A.update(x, fb)
            B.update(x, fb)
            self.assertEqual(A.w, B.w)
        self.assertEqual(hypothesis_index(A.w), ref['final_index'])
        self.assertEqual((A.evaluations, A.commits), (B.evaluations, B.commits))

    def test_learner_api_receives_no_diagnostics(self):
        self.assertEqual(list(inspect.signature(VariantLearner.update).parameters), ['self', 'x', 'feedback'])
        self.assertEqual(list(inspect.signature(VariantLearner.__init__).parameters), ['self', 'sim', 'variant'])

    def test_decision_and_experience_denominators(self):
        from experiments.update_diagnosis import all_decisions_summary, decision_summary
        e = lambda labs, failing: dict(failing=failing, in_teacher_class=False,
                                       counterfactual_b0=dict(cls=labs[0]), decisions=[dict(cls=c) for c in labs])
        ex = [e(['E2a', 'E2a', 'E1b'], False), e(['E1a'], True)]
        ex[0]['decisions'][2]['min_blocks_to_lower'] = 3
        ex[0]['counterfactual_b0']['cls'] = 'E2a'
        d = decision_summary(ex)
        self.assertEqual((d['experiences'], d['failing'], d['counts']['E2a'], d['failing_counts']['E1a']), (2, 1, 1, 1))
        self.assertEqual(d['fraction_of_failing']['E1a'], 1.0)
        self.assertEqual(all_decisions_summary(ex)['decisions'], 4)
        self.assertIsNone(decision_summary([])['fraction_of_failing']['E0'])

    def test_completion_requires_passing_tests(self):
        from experiments.update_diagnosis import completion_status
        self.assertEqual(completion_status(dict(passed=True), dict(passed=False)), 'stopped_failed_tests')

    def test_prior_artifacts_unchanged(self):
        path = ROOT / 'reports/local-update-diagnosis-v1-manifest.json'
        if not path.exists():
            self.skipTest('manifest not frozen yet')
        m = json.loads(path.read_text())
        for p, h in m['prior_artifacts'].items():
            self.assertEqual(hashlib.sha256((ROOT / p).read_bytes()).hexdigest(), h, p)
        self.assertEqual(hashlib.sha256((ROOT / m['contract']).read_bytes()).hexdigest(), m['contract_sha256'])


if __name__ == '__main__':
    unittest.main()
