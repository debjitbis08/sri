import json
from itertools import product
from pathlib import Path
import unittest
from experiments.recurrent import certificate
from src.recurrent import (Network, INPUTS, INTERNAL_STATES, EDGE_MASKS, SCHEDULES,
                           A_ONLY, FULL, orbit, attractors, stable_context)


class RecurrentTests(unittest.TestCase):
    def test_exact_contextual_trace_and_path_control(self):
        initial = (0, 1, 0, 0, 0)
        result = orbit(Network(), initial)
        self.assertEqual(result['transient'],
                         [(0, 1, 0, 0, 0), (0, 1, 0, 0, 1), (0, 1, 1, 0, 1)])
        self.assertEqual(result['cycle'], [(0, 1, 1, 1, 1)])
        isolated = orbit(Network(edges=A_ONLY), initial)
        self.assertTrue(stable_context(result, isolated, 3))
        # Acyclic B -> K -> A reproduces the fixed response: recurrence is not essential.
        path = orbit(Network(edges=(False, True, True, False)), initial)
        self.assertEqual(path['cycle'], result['cycle'])

    def test_cycles_and_schedule_dependence(self):
        rules = ('or', 'or', 'xor')
        result = orbit(Network(rules), (0, 0, 0, 0, 1))
        self.assertEqual(result['period'], 2)
        self.assertEqual(result['cycle'], [(0, 0, 1, 1, 0), (0, 0, 1, 1, 1)])
        initial = (0, 1, 0, 0, 0)
        self.assertEqual(orbit(Network(rules), initial)['cycle'], [(0, 1, 1, 1, 0)])
        self.assertEqual(orbit(Network(rules, FULL, 'kab'), initial)['cycle'],
                         [(0, 1, 1, 1, 1)])

    def test_complete_graphs_and_direct_fixed_points(self):
        for rules in product(('or', 'xor'), repeat=3):
            for schedule in SCHEDULES:
                for edges in EDGE_MASKS:
                    network = Network(rules, edges, schedule)
                    for evidence in INPUTS:
                        basins = attractors(network, evidence)
                        self.assertEqual(sum(b['basin_size'] for b in basins), 8)
                        actual_fixed = {b['cycle'][0] for b in basins if b['period'] == 1}
                        expected_fixed = {evidence + s for s in INTERNAL_STATES
                                          if network.step(evidence + s) == evidence + s}
                        self.assertEqual(actual_fixed, expected_fixed)
                        for internal in INTERNAL_STATES:
                            initial = evidence + internal
                            result = orbit(network, initial)
                            self.assertLessEqual(result['transitions'], 8)
                            self.assertTrue(all(s[:2] == evidence for s in
                                                result['transient'] + result['cycle']))
                        for basin in basins:
                            cycle = basin['cycle']
                            for i, state in enumerate(cycle):
                                self.assertEqual(network.step(state), cycle[(i + 1) % len(cycle)])

    def test_or_is_reachability_closure_for_every_mask_and_schedule(self):
        # Independent graph-reachability oracle, not recurrent update simulation.
        for edges in EDGE_MASKS:
            links = [link for link, enabled in zip(((1, 0), (2, 0), (0, 1), (0, 2)), edges)
                     if enabled]
            for evidence in INPUTS:
                for internal in INTERNAL_STATES:
                    active = {i for i, bit in enumerate(internal) if bit}
                    active |= {i + 1 for i, bit in enumerate(evidence) if bit}
                    for _ in range(3):
                        active |= {b for a, b in links if a in active}
                    expected = evidence + tuple(int(i in active) for i in range(3))
                    for schedule in SCHEDULES:
                        result = orbit(Network(edges=edges, schedule=schedule), evidence + internal)
                        self.assertEqual(result['cycle'], [expected])
                        self.assertLessEqual(result['transitions'], 4)

    def test_frozen_report_and_coverage(self):
        report = certificate()
        frozen = Path(__file__).resolve().parents[1] / 'reports' / 'recurrent.json'
        self.assertEqual(frozen.read_bytes(),
                         (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())
        self.assertEqual(sum(r['maps'] for r in report['enumeration']), 3584)
        self.assertEqual(sum(r['trajectories'] for r in report['enumeration']), 28672)
        for row in report['enumeration']:
            self.assertEqual(sum(row['trajectory_period_counts'].values()), 512)

    def test_invalid_configuration(self):
        for kwargs in ({'rules': ('or',)}, {'rules': ('or', 'and', 'or')},
                       {'edges': (True,)}, {'schedule': 'kka'}):
            with self.assertRaises(ValueError):
                Network(**kwargs)
        with self.assertRaises(ValueError):
            Network().step((0, 0, 0, 0, 2))


if __name__ == '__main__':
    unittest.main()
