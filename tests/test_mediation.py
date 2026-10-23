from dataclasses import replace
from itertools import combinations, product
import json
from pathlib import Path
import unittest

from experiments.mediation import certificate, example, render
from src.mediation import (CoreConfiguration, InputModule, OutputModule, Satellite,
                           Structure, initial_configuration, realize, valid_response)


class MediationTests(unittest.TestCase):
    def test_mediation_and_matched_control(self):
        structure = example()
        before = initial_configuration(structure, InputModule(('A', 'B')))
        original = repr(before)
        direct = realize(structure, before, mediated=False)
        joint = realize(structure, before)
        self.assertEqual((direct.status, direct.feasible), ('complete', False))
        self.assertEqual((joint.status, joint.feasible), ('complete', True))
        self.assertEqual(joint.configuration.output.bindings,
                         (('A', 'local-role', 'a'), ('B', 'local-role', 'b')))
        self.assertTrue(valid_response(structure, joint.configuration, True))
        self.assertFalse(valid_response(structure, joint.configuration, False))
        self.assertEqual(joint.configuration.input, before.input)
        self.assertEqual(repr(before), original)  # no partial module mutations
        self.assertEqual(joint.work.candidate_extensions, 3)
        self.assertEqual(joint.work.collision_rejections, 1)

    def test_triple_conflict_survives_although_every_pair_fits(self):
        structure = example()
        for pair in combinations(('A', 'B', 'S'), 2):
            self.assertTrue(realize(structure, initial_configuration(
                structure, InputModule(pair))).feasible)
        result = realize(structure, initial_configuration(
            structure, InputModule(('A', 'B')), ('S',)))
        self.assertEqual((result.status, result.feasible, result.configuration),
                         ('complete', False, None))
        self.assertEqual(result.work.terminal_assignments, 0)

    def test_context_changes_existing_participants_under_same_evidence(self):
        structure = example(('a', 'b', 'c'), ('a',))
        evidence = InputModule(('A', 'B'))
        off = realize(structure, initial_configuration(structure, evidence))
        on = realize(structure, off.configuration, context=('S',))
        self.assertEqual(off.configuration.input, on.configuration.input)
        self.assertEqual(off.configuration.output.bindings[:2],
                         (('A', 'local-role', 'a'), ('B', 'local-role', 'b')))
        self.assertEqual(on.configuration.output.bindings[:2],
                         (('A', 'local-role', 'b'), ('B', 'local-role', 'c')))

    def test_selector_matches_independent_full_product_oracle(self):
        # Tiny correctness oracle only: implementation never builds this product.
        domains = [d for size in (1, 2, 3) for d in combinations(('a', 'b', 'c'), size)]
        for a, b in product(domains, repeat=2):
            structure = Structure((Satellite('A', 'r', a, a[0]),
                                   Satellite('B', 'r', b, b[0])))
            before = initial_configuration(structure, InputModule(('B', 'A')))
            for mediated in (False, True):
                possibilities = product(a, b) if mediated else [(a[0], b[0])]
                allowed = [pair for pair in possibilities if len(set(pair)) == 2]
                result = realize(structure, before, mediated)
                self.assertEqual(result.feasible, bool(allowed))
                if allowed:
                    self.assertEqual(tuple(row[2] for row in result.configuration.output.bindings),
                                     min(allowed))

    def test_overlap_agreement_is_not_just_output_validity(self):
        structure = example()
        state = realize(structure, initial_configuration(
            structure, InputModule(('A', 'B')))).configuration
        self.assertFalse(valid_response(structure, replace(
            state, core=CoreConfiguration()), True))
        self.assertFalse(valid_response(structure, replace(
            state, output=OutputModule()), True))
        bad_b = replace(state.satellites[1], site='a')
        self.assertFalse(valid_response(structure, replace(
            state, satellites=(state.satellites[0], bad_b, state.satellites[2])), True))
        with self.assertRaises(ValueError):
            realize(structure, replace(state, core=CoreConfiguration()))

    def test_caps_never_turn_unknown_into_impossible_or_commit_partial_state(self):
        structure = example()
        before = initial_configuration(structure, InputModule(('A', 'B')))
        for cap in (0, 1, 2):
            result = realize(structure, before, max_extensions=cap)
            self.assertEqual((result.status, result.feasible, result.configuration),
                             ('truncated', None, None))
            self.assertEqual(result.work.candidate_extensions, cap)
        self.assertTrue(realize(structure, before, max_extensions=3).feasible)
        for cap in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                realize(structure, before, max_extensions=cap)

    def test_renaming_resources_preserves_feasibility_not_tie_break_choice(self):
        mapping = {'a': 'z', 'b': 'x'}
        source = example()
        renamed = Structure(tuple(Satellite(s.name, s.role,
            tuple(mapping[v] for v in s.sites), mapping[s.direct_site])
            for s in source.satellites))
        for participants in (('A', 'B'), ('A', 'B', 'S')):
            for mediated in (False, True):
                results = [realize(model, initial_configuration(model,
                           InputModule(participants)), mediated).feasible
                           for model in (source, renamed)]
                self.assertEqual(*results)

    def test_lossy_interface_counterexample(self):
        # Same role/direct-site descriptors, different required feasibility.
        narrow = Structure((Satellite('A', 'r', ('a',), 'a'),
                            Satellite('B', 'r', ('a',), 'a')))
        broad = Structure((narrow.satellites[0], Satellite('B', 'r', ('a', 'b'), 'a')))
        descriptor = lambda model: tuple((s.role, s.direct_site) for s in model.satellites)
        self.assertEqual(descriptor(narrow), descriptor(broad))
        self.assertFalse(realize(narrow, initial_configuration(narrow, InputModule(('A', 'B')))).feasible)
        self.assertTrue(realize(broad, initial_configuration(broad, InputModule(('A', 'B')))).feasible)

    def test_invalid_evidence_and_inventory(self):
        for names in (('missing',), ('A', 'A')):
            with self.assertRaises(ValueError):
                initial_configuration(example(), InputModule(names))
        with self.assertRaises(ValueError):
            initial_configuration(example(), InputModule(('A',)), ('A',))
        with self.assertRaises(ValueError):
            Satellite('A', 'r', ('a',), 'b')
        with self.assertRaises(ValueError):
            Structure((example().satellites[0],) * 2)

    def test_frozen_report(self):
        frozen = Path(__file__).resolve().parents[1] / 'reports' / 'mediation-example.json'
        expected = json.loads(frozen.read_text())
        actual = certificate()
        # Runtime provenance legitimately differs across Python installations.
        actual['runtime'] = expected['runtime']
        self.assertEqual(render(actual), frozen.read_text())
        self.assertEqual(render(certificate()), render(certificate()))


if __name__ == '__main__':
    unittest.main()
