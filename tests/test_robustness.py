import json
from itertools import product
from pathlib import Path
import unittest
from experiments.robustness import certificate
from src.recurrent import Network, INPUTS, INTERNAL_STATES, EDGE_MASKS, SCHEDULES, orbit
from src.robustness import (update_node, fair_graph, fair_outcomes,
                            guaranteed_response, fair_lasso)


class RobustnessTests(unittest.TestCase):
    def test_atomic_updates_agree_with_all_ordered_sweeps(self):
        for rules in product(('or', 'xor'), repeat=3):
            for edges in EDGE_MASKS:
                network = Network(rules, edges)
                for evidence in INPUTS:
                    for initial in INTERNAL_STATES:
                        for schedule in SCHEDULES[1:]:
                            state = initial
                            for node in schedule:
                                state = update_node(network, evidence, state, 'kab'.index(node))
                            expected = Network(rules, edges, schedule).step(evidence + initial)
                            self.assertEqual(evidence + state, expected)

    def test_fair_schedule_missed_by_seven_orders(self):
        network = Network(('or', 'or', 'xor'))
        evidence, initial = (0, 1), (0, 0, 0)
        for schedule in SCHEDULES:
            self.assertEqual(orbit(Network(network.rules, schedule=schedule),
                                   evidence + initial)['period'], 1)
        state, trace = initial, []
        for node in (2, 2, 0, 1):
            state = update_node(network, evidence, state, node)
            trace.append(state)
        self.assertEqual(state, initial)
        self.assertIn((0, 0, 1), trace)
        analysis = fair_graph(network, evidence)
        # This fair cycling SCC has an exit; bottom-SCC-only analysis is unsound.
        component = ((0, 0, 0), (0, 0, 1))
        self.assertIn(component, fair_outcomes(analysis, initial))
        self.assertTrue(any(t not in component for s in component for t in analysis['graph'][s]))
        self.assertFalse(guaranteed_response(analysis, initial, (1, 1)))

    def test_fair_lassos_are_executable_and_cover_every_label(self):
        for rules in product(('or', 'xor'), repeat=3):
            for edges in EDGE_MASKS:
                for evidence in INPUTS:
                    network = Network(rules, edges)
                    analysis = fair_graph(network, evidence)
                    for initial in INTERNAL_STATES:
                        outcomes = fair_outcomes(analysis, initial)
                        self.assertTrue(outcomes)
                        for component in outcomes:
                            lasso = fair_lasso(analysis, initial, component)
                            state = initial
                            for step in lasso['prefix']:
                                state = update_node(network, evidence, state, step['node'])
                                self.assertEqual(state, step['state'])
                            self.assertEqual(state, lasso['loop_start'])
                            visited = {state}
                            for step in lasso['loop']:
                                state = update_node(network, evidence, state, step['node'])
                                self.assertEqual(state, step['state'])
                                visited.add(state)
                            self.assertEqual(state, lasso['loop_start'])
                            self.assertEqual({s['node'] for s in lasso['loop']}, {0, 1, 2})
                            self.assertEqual(visited, set(component))

    def test_unfair_stalling_does_not_defeat_OR_convergence(self):
        for edges in EDGE_MASKS:
            network = Network(edges=edges)
            links = [link for link, enabled in zip(((1, 0), (2, 0), (0, 1), (0, 2)), edges)
                     if enabled]
            for evidence in INPUTS:
                analysis = fair_graph(network, evidence)
                for initial in INTERNAL_STATES:
                    active = {i for i, v in enumerate(initial) if v}
                    active |= {i + 1 for i, v in enumerate(evidence) if v}
                    for _ in range(3):
                        active |= {b for a, b in links if a in active}
                    expected = tuple(int(i in active) for i in range(3))
                    self.assertTrue(guaranteed_response(analysis, initial, expected, (0, 1, 2)))

    def test_SCCs_against_exhaustive_recurrent_subsets(self):
        # Independent finite oracle: inspect every possible recurring state set.
        for rules in product(('or', 'xor'), repeat=3):
            for evidence in INPUTS:
                analysis = fair_graph(Network(rules), evidence)
                graph = analysis['graph']
                recurrent_states = set()
                for mask in range(1, 256):
                    subset = {s for i, s in enumerate(INTERNAL_STATES) if mask & (1 << i)}
                    if {n for s in subset for n, t in enumerate(graph[s]) if t in subset} != {0, 1, 2}:
                        continue
                    connected = True
                    for source in subset:
                        seen, pending = {source}, [source]
                        while pending:
                            for target in graph[pending.pop()]:
                                if target in subset and target not in seen:
                                    seen.add(target)
                                    pending.append(target)
                        if seen != subset:
                            connected = False
                            break
                    if connected:
                        recurrent_states |= subset
                self.assertEqual(recurrent_states,
                                 {s for c in analysis['fair_components'] for s in c})

    def test_report_and_main_witness_recovery(self):
        report = certificate()
        self.assertEqual(report['summary'], dict(full_star_graphs=32, initial_cases=256,
            all_fair_settle=100, unique_stable_response=88, fixed_points=43,
            recover_all_single_bit_response=6,
            preserve_all_single_deletions_from_established=20))
        row = next(r for r in report['full_star_cases']
                   if r['rules'] == ('or', 'or', 'xor') and r['evidence'] == (0, 1))
        for fixed in row['fixed_points']:
            self.assertEqual([p['same_response_all_fair'] for p in fixed['single_bit_recovery']],
                             [False, True, False])
            self.assertEqual([p['from_established_state_all_fair']
                              for p in fixed['single_edge_deletions']], [True, True, True, False])
        frozen = Path(__file__).resolve().parents[1] / 'reports' / 'robustness.json'
        self.assertEqual(frozen.read_bytes(),
                         (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())


if __name__ == '__main__':
    unittest.main()
