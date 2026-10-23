import hashlib
import json
from pathlib import Path
import random
import unittest

import numpy as np

from src.core_mediation import (
    BOUND, configurations, constraints, farkas, joint_dependent, orbits, parameters,
    solve, target_sets, verify,
)


ROOT = Path(__file__).resolve().parents[1]
XOR = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))


def table(**entries):
    return {tuple(int(p) if p.isdigit() else p for p in k.split('_')): v
            for k, v in entries.items()}


class CoreMediationTests(unittest.TestCase):
    def test_contract_analytic_agreement_example_meets_level_1(self):
        tables = table(uA_1=1, uB_1=1, wA=2, wB=2)
        self.assertTrue(all(verify('G', 2, tables, ((0, 0, 0), (1, 1, 1)), False).values()))

    def test_encoder_agrees_with_independent_evaluator(self):
        rng = random.Random(7)
        for cls, n in (('S', 3), ('F', 2), ('T', 2), ('G', 2), ('D', 3)):
            names = parameters(cls, n)
            for _ in range(200):
                theta = [rng.randint(-3, 3) for _ in names]
                targets = rng.choice([t for t in target_sets(n, 1)] + [t for t in target_sets(n, 2)])
                strict = rng.random() < .5
                _, hard, disjunctions = constraints(cls, n, targets, strict)
                x = np.array(theta)
                encoded = (all(row @ x <= rhs for row, rhs in hard)
                           and all(any(row @ x <= rhs for row, rhs in d) for d in disjunctions))
                exact = verify(cls, n, dict(zip(names, theta)), targets, strict)
                self.assertEqual(encoded, all(exact.values()), (cls, n, targets, theta))

    def test_controls_without_core_coupling_are_certified_impossible(self):
        for cls in 'UD':
            result = solve(cls, 2, ((0, 0, 0), (1, 1, 1)), False)
            self.assertEqual(result['outcome'], 'impossible')

    def test_xor_core_needs_ternary_at_two_core_modes(self):
        self.assertEqual(solve('S', 2, XOR, False)['outcome'], 'impossible')
        self.assertEqual(solve('F', 2, XOR, False)['outcome'], 'impossible')
        result = solve('T', 2, XOR, True)
        self.assertEqual(result['outcome'], 'realizable')
        self.assertTrue(all(result['checks'].values()))

    def test_adjacent_targets_are_impossible_for_every_energy(self):
        self.assertEqual(solve('T', 2, ((0, 0, 0), (0, 1, 0)), False)['outcome'], 'impossible')

    def test_farkas_rejects_feasible_system(self):
        rows = [(np.array([1.]), 0), (np.array([-1.]), 1)]
        self.assertFalse(farkas(rows, 1))
        self.assertTrue(farkas([(np.array([1.]), -1), (np.array([-1.]), 0)], 1))

    def test_joint_dependence_and_orbit_partition(self):
        self.assertTrue(joint_dependent(XOR))
        self.assertFalse(joint_dependent(((0, 0, 0), (0, 1, 0), (1, 0, 1))))
        for cls, n in (('S', 3), ('G', 2)):
            for level in (1, 2):
                self.assertEqual(sum(orbits(cls, n, level).values()),
                                 sum(1 for _ in target_sets(n, level)))
        self.assertEqual(len(configurations(4)), 16)

    def test_frozen_report_matches_sources_and_witnesses(self):
        report = json.loads((ROOT/'reports/core-mediation-v1.json').read_text())
        for path, digest in report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest, path)
        for case in report['cases']:
            for row in case['results']:
                if row['outcome'] == 'realizable':
                    tables = {tuple(int(p) if p.isdigit() else p for p in k.split('.')): v
                              for k, v in row['witness'].items()}
                    targets = tuple(map(tuple, row['targets']))
                    self.assertTrue(all(verify(row['cls'], row['n'], tables,
                                               targets, row['strict']).values()))
                    self.assertTrue(all(abs(v) <= BOUND for v in tables.values()))
                self.assertNotEqual(row['outcome'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
