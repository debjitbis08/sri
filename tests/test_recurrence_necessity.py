import json
from pathlib import Path
import unittest
from experiments.recurrence_necessity import certificate
from src.recurrent import (Network, EDGE_MASKS, FULL, A_ONLY, B_ONLY, orbit,
                           stable_projection, acyclic, recurrence_test)


class RecurrenceNecessityTests(unittest.TestCase):
    def test_projection_uses_every_cycle_phase(self):
        result = orbit(Network(('xor', 'or', 'xor'), (False, True, True, False)),
                       (0, 0, 0, 0, 1))
        self.assertEqual(result['period'], 2)
        self.assertEqual(stable_projection(result, (3, 4)), (1, 1))
        self.assertIsNone(stable_projection(result, (2, 3, 4)))
        shifted = dict(result, cycle=result['cycle'][1:] + result['cycle'][:1])
        self.assertEqual(stable_projection(shifted, (3, 4)), (1, 1))

    def test_unrelated_oscillator_does_not_disqualify_isolated_A(self):
        network = Network(('or', 'or', 'xor'), A_ONLY)
        result = orbit(network, (0, 1, 0, 0, 0))
        self.assertEqual(result['period'], 2)
        self.assertEqual(stable_projection(result, (3,)), (0,))
        self.assertIsNone(stable_projection(result, (4,)))

    def test_feedback_stabilization_witness(self):
        network = Network(('or', 'or', 'xor'))
        initial = (0, 1, 0, 0, 0)
        result = recurrence_test(network, initial)
        self.assertTrue(result['recurrence_essential'])
        self.assertEqual(result['satellite_response'], (1, 0))
        self.assertEqual(result['independent_response'], (0, 0))
        self.assertEqual(len(result['acyclic_controls']), 9)
        self.assertTrue(all(c['satellite_response'] is None for c in result['acyclic_controls']))
        self.assertEqual(orbit(network, initial)['cycle'], [(0, 1, 1, 1, 0)])

    def test_hidden_core_cycle_is_not_positive_evidence(self):
        result = recurrence_test(Network(('xor', 'or', 'xor')), (0, 0, 0, 0, 1))
        self.assertTrue(result['contextual'])
        self.assertFalse(result['recurrence_essential'])
        self.assertEqual(result['matching_acyclic_masks'], [(False, True, True, False)])

    def test_all_OR_propagation_remains_negative_control(self):
        result = recurrence_test(Network(), (0, 1, 0, 0, 0))
        self.assertTrue(result['contextual'])
        self.assertFalse(result['recurrence_essential'])
        self.assertIn((False, True, True, False), result['matching_acyclic_masks'])

    def test_three_edge_settled_control_boundary(self):
        result = recurrence_test(Network(('xor', 'or', 'xor'),
                                         (False, True, True, True)), (0, 0, 0, 0, 1))
        self.assertTrue(result['recurrence_essential'])
        self.assertTrue(result['all_acyclic_observations_stable'])
        self.assertEqual(result['satellite_response'], (1, 0))
        self.assertEqual({c['satellite_response'] for c in result['acyclic_controls']},
                         {(0, 1), (1, 1)})

    def test_star_cycle_detection_against_topological_sort(self):
        for edges in EDGE_MASKS:
            remaining = {0, 1, 2}
            links = [link for link, enabled in zip(((1, 0), (2, 0), (0, 1), (0, 2)), edges)
                     if enabled]
            while remaining:
                roots = {n for n in remaining if not any(b == n and a in remaining
                                                        for a, b in links)}
                if not roots:
                    break
                remaining -= roots
            self.assertEqual(acyclic(edges), not remaining)
        self.assertEqual(sum(acyclic(edges) for edges in EDGE_MASKS), 9)

    def test_exact_counts_minimality_and_frozen_report(self):
        report = certificate()
        counts = report['full_star_counts']
        self.assertEqual(counts, dict(cases=1792, full_fixed=961,
                         isolated_observations_stable=841, contextual=224,
                         recurrence_essential=80, essential_from_zero=22,
                         essential_with_all_acyclic_observations_stable=0))
        by_edges = report['counts_by_edge_count']
        self.assertEqual(sum(row['cases'] for row in by_edges), 28672)
        self.assertEqual([row['recurrence_essential'] for row in by_edges], [0, 0, 0, 82, 80])
        self.assertEqual(sum(report['minimum_edge_witness']['edges']), 3)
        self.assertEqual(by_edges[3]['essential_with_all_acyclic_observations_stable'], 2)
        self.assertEqual(len(report['full_star_witnesses']), 80)
        frozen = Path(__file__).resolve().parents[1] / 'reports' / 'recurrence-necessity.json'
        self.assertEqual(frozen.read_bytes(),
                         (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())


if __name__ == '__main__':
    unittest.main()
