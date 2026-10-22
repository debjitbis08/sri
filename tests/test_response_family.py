import json
from itertools import product
from pathlib import Path
import unittest
from experiments.response_family import certificate
from src.recurrent import Network, EDGE_MASKS, INPUTS, INTERNAL_STATES, SCHEDULES, orbit
from src.robustness import update_node
from src.response_family import (REGIMES, recurrent_states, observation,
                                 assess_input, assess_pair)


class ResponseFamilyTests(unittest.TestCase):
    def test_OR_tradeoff_against_closed_form(self):
        for edges in EDGE_MASKS:
            p = int(edges[0] and edges[3])  # A reaches B.
            q = int(edges[1] and edges[2])  # B reaches A.
            for regime in REGIMES:
                cases = [assess_input(('or',) * 3, edges, evidence, regime) for evidence in INPUTS]
                for evidence, case in zip(INPUTS, cases):
                    x, y = evidence
                    expected = (int(x or (q and y)), int(y or (p and x)))
                    self.assertEqual(case['response'], expected)
                    self.assertEqual(case['recoverable'], expected == (1, 1))
                self.assertEqual(len({c['response'] for c in cases}), 4 - p - q)
                self.assertEqual(sum(c['recoverable'] for c in cases), 1 + p + q)

    def test_fixed_sweeps_include_every_atomic_phase(self):
        for rules in product(('or', 'xor'), repeat=3):
            for evidence in INPUTS:
                for initial in INTERNAL_STATES:
                    expected = set()
                    network = Network(rules)
                    for schedule in SCHEDULES[1:]:
                        cycle = orbit(Network(rules, schedule=schedule), evidence + initial)['cycle']
                        for state in cycle:
                            internal = state[2:]
                            expected.add(internal)
                            for node in schedule:
                                internal = update_node(network, evidence, internal, 'kab'.index(node))
                                expected.add(internal)
                    self.assertEqual(set(recurrent_states(rules, network.edges, evidence, initial,
                                                          'fixed_sweeps')), expected)

    def test_restricted_schedules_remain_a_separate_property(self):
        args = (('or', 'or', 'xor'), (True,) * 4, (0, 1), (0, 0, 0))
        self.assertIsNone(observation(recurrent_states(*args, 'all_fair')))
        self.assertEqual(observation(recurrent_states(*args, 'fixed_sweeps')), (1, 1))
        self.assertFalse(assess_input(args[0], args[1], args[2], 'fixed_sweeps')['recoverable'])

    def test_all_gates_are_required_and_deletion_must_be_common(self):
        a = dict(evidence=(0, 0), response=(0, 1), independent_response=(0, 0),
                 contextual=True, recoverable=True, preserving_deletions=[0])
        b = dict(evidence=(1, 0), response=(1, 1), independent_response=(1, 1),
                 contextual=False, recoverable=True, preserving_deletions=[1])
        self.assertFalse(assess_pair(a, b)['passes'])
        b['preserving_deletions'] = [0, 1]
        self.assertTrue(assess_pair(a, b)['passes'])
        b['recoverable'] = False
        self.assertFalse(assess_pair(a, b)['passes'])

    def test_same_snapshot_preservation_is_not_reconstruction(self):
        # Removing B->K from this OR path retains latched 111, but cannot rebuild
        # that pair at evidence 01 after a zero reset. It must fail the edit gate.
        edges = (False, True, True, False)
        case = assess_input(('or',) * 3, edges, (0, 1), 'all_fair')
        self.assertEqual(case['response'], (1, 1))
        self.assertNotIn(1, case['preserving_deletions'])
        modified = (False, False, True, False)
        self.assertEqual(observation(recurrent_states(('or',) * 3, modified, (0, 1),
                                                     (1, 1, 1), 'all_fair')), (1, 1))
        self.assertEqual(observation(recurrent_states(('or',) * 3, modified, (0, 1),
                                                     (0, 0, 0), 'all_fair')), (0, 1))

    def test_every_recovery_failure_has_a_reproducible_bad_outcome(self):
        report = certificate()
        for result in report['regimes']:
            for network in result['networks']:
                for case in network['inputs']:
                    failure = case['first_recovery_failure']
                    if failure is None:
                        continue
                    self.assertIn(failure['state'], case['recurrent_states'])
                    self.assertEqual(sum(a != b for a, b in zip(failure['state'], failure['disturbed'])), 1)
                    actual = recurrent_states(network['rules'], network['edges'], case['evidence'],
                                              failure['disturbed'], result['regime'])
                    self.assertEqual(actual, failure['recurrent_states'])
                    self.assertTrue(any(state[1:] != case['response'] for state in actual))

    def test_exhaustive_obstruction_and_frozen_certificate(self):
        report = certificate()
        expected = ((217, 208, 14, 40, 10), (270, 250, 34, 48, 22))
        for result, (stable, distinct, contextual, recovered, edits) in zip(report['regimes'], expected):
            self.assertEqual(result['successive_gate_counts'], dict(candidates=768,
                stable=stable, distinguishable=distinct, contextual=contextual,
                recoverable=0, structurally_tolerant=0))
            self.assertEqual(result['distinguishable_and_recoverable'], 0)
            self.assertEqual(result['recoverable_input_cases'], recovered)
            self.assertEqual(result['recoverable_response_inventory'], [(1, 1)])
            self.assertEqual(result['stable_distinguishable_contextual_structurally_tolerant'], edits)
            self.assertEqual(len(result['networks']), 128)
            self.assertTrue(all(len(n['families']) == 6 for n in result['networks']))
            self.assertTrue(all(not f['passes'] for n in result['networks'] for f in n['families']))
        tradeoff = report['all_OR_tradeoff']
        self.assertEqual(len(tradeoff['strict_edge_additions']), 65)
        self.assertTrue(all(r['delta_R'] == -r['delta_D'] and r['delta_R'] >= 0
                            for r in tradeoff['strict_edge_additions']))
        frozen = Path(__file__).resolve().parents[1] / 'reports' / 'response-family.json'
        self.assertEqual(frozen.read_bytes(),
                         (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())


if __name__ == '__main__':
    unittest.main()
