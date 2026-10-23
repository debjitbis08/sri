from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import unittest

from experiments.model_comparison import GRID, certificate, jsonable, render
from src.model_comparison import (
    D1, Work, algebra_response, algebra_system, cell_boundary, cell_driven,
    cell_matrix, cell_withdrawal_limit, contraction, difference, eigendiagnostic,
    iterate, matmul, matvec, norm, rank, relation_minima, relation_response,
    replacements, residual_bound, solve, spectral_matrix,
)


class ComparisonTests(unittest.TestCase):
    def test_rational_elimination_and_inconsistency(self):
        self.assertEqual(solve(((2, 1), (1, -1)), (1, 2)), (F(1), F(-1)))
        self.assertIsNone(solve(((1, 1), (2, 2)), (0, 1)))
        self.assertEqual(solve(((1, 1),), (1,), previous=(9, 3)), (F(-2), F(3)))
        self.assertEqual(solve(((1, 1),), (1,), previous=(0, 2), modulus=3), (2, 2))

    def test_spectral_closed_form_and_joint_equations(self):
        for g, e in product(GRID, (-1, 1)):
            q = spectral_matrix(g)
            b = (F(e), F(0), F(0), F(0))
            k = F(e) * (1+g)/(4+6*g)
            expected = ((e+k)/2, k, k/2, g*k/(1+g))
            self.assertEqual(solve(q, b), expected)
            self.assertEqual(matvec(q, expected), b)
            self.assertEqual(contraction(q, F(1, 16)), F(15, 16))
            self.assertEqual(solve(spectral_matrix(g, False), b)[2], 0)

    def test_eigenvectors_residual_orthogonality_and_degenerate_control(self):
        for matrix in [spectral_matrix(g) for g in GRID] + [spectral_matrix(0, False)]:
            modes = eigendiagnostic(matrix)
            self.assertEqual(modes['status'], 'complete')
            self.assertLess(modes['max_residual'], 1e-10)
            for i, v in enumerate(modes['eigenvectors']):
                for j, w in enumerate(modes['eigenvectors']):
                    self.assertAlmostEqual(sum(a*b for a, b in zip(v, w)), int(i == j))
        self.assertEqual(eigendiagnostic(spectral_matrix(1), max_rotations=0)['status'], 'truncated')

    def test_residual_bound_on_independent_rational_points(self):
        for g in GRID:
            q, b = spectral_matrix(g), (F(1), F(0), F(0), F(0))
            exact = solve(q, b)
            for candidate in product((F(-1), F(0), F(1)), repeat=4):
                self.assertLessEqual(norm(difference(candidate, exact)), residual_bound(q, b, candidate))

    def test_affine_response_against_full_enumeration(self):
        for e, context in product((1, 2), (0, 1)):
            rows, rhs = algebra_system(e, context)
            admissible = [z for z in product(range(3), repeat=4) if all(
                (sum(a*b for a, b in zip(row, z))-b) % 3 == 0 for row, b in zip(rows, rhs))]
            self.assertEqual(len(admissible), 1)
            expected = (e, (2*e+context) % 3, (-e-context) % 3, context)
            self.assertEqual(admissible[0], expected)
            for start in product(range(3), repeat=4):
                self.assertEqual(algebra_response(e, context, start), expected)

    def test_algebra_persistence_is_free_coordinate_retention(self):
        for e, context in product((1, 2), (0, 1)):
            z = algebra_response(e, context)
            self.assertEqual(algebra_response(None, context, z), z)
            damaged = (z[0], z[1], (z[2]+1) % 3, z[3])
            repaired = algebra_response(None, context, damaged)
            self.assertNotEqual(repaired[2], z[2])
            self.assertEqual(algebra_response(e, context, damaged), z)

    def test_relational_minima_against_analytic_two_choices(self):
        for g, context, e in product(GRID[1:], (0, 1), (0, 1)):
            work = Work()
            minima, cost = relation_minima(e, context, g, work=work)
            self.assertEqual(work.candidates, 16)
            if not context:
                expected = ((e, 0, e, context),)
            elif g < 1:
                expected = ((e, 0, 1-e, context),)
            elif g > 1:
                expected = ((e, 1, e, context),)
            else:
                expected = ((e, 0, 1-e, context), (e, 1, e, context))
            self.assertEqual(minima, expected)
            self.assertEqual(cost, 0 if not context else min(g, 1))
            for z in minima:
                self.assertEqual(relation_response(e, context, g, z), z)
                self.assertEqual(relation_response(None, context, g, z), z)

    def test_tie_requires_all_recurrent_responses_not_only_default(self):
        minima, _ = relation_minima(0, 1, F(1))
        source = minima[1]
        damaged = (1-source[0],) + source[1:]
        restored = relation_response(0, 1, F(1), damaged)
        self.assertNotEqual(restored[2], source[2])

    def test_cellular_chain_identity_homology_and_exact_eigenspaces(self):
        for attached in (False, True):
            d2 = cell_boundary(attached)
            self.assertEqual(matmul(D1, d2), ((0,),)*3)
            self.assertEqual(3-rank(D1)-rank(d2), int(not attached))
            for g in GRID:
                matrix = cell_matrix(g, attached)
                self.assertEqual(matvec(matrix, (1, 1, 1)), (3*g if attached else 0,)*3)
                self.assertEqual(matvec(matrix, (1, -1, 0)), (3, -3, 0))
                self.assertEqual(3-rank(matrix), int(not attached or not g))

    def test_cellular_driven_closed_form_and_withdrawal(self):
        for attached, g, e in product((False, True), GRID, (-1, 1)):
            response = cell_driven(g, attached, e)
            expected = F(e)*(1-g)/(1+2*g) if attached else F(e)
            self.assertEqual(response, (F(e), expected, expected))
            self.assertEqual(cell_withdrawal_limit(g, attached, response),
                             (F(0),)*3 if attached and g else (F(e),)*3)
        # Same cell counts and graph edges, distinct attachment changes persistence.
        self.assertNotEqual(cell_withdrawal_limit(F(1, 4), False, (F(1),)*3),
                            cell_withdrawal_limit(F(1, 4), True, (F(1),)*3))

    def test_cellular_unprotected_harmonic_memory(self):
        state = (F(1), F(1), F(1))
        damaged = (F(2), F(1), F(1))
        self.assertEqual(cell_withdrawal_limit(F(1), False, damaged), (F(4, 3),)*3)
        matrix = cell_matrix(1, False)
        evolved = iterate(matrix, (0,)*3, damaged, F(1, 8), 20)
        self.assertEqual(sum(evolved), sum(damaged))

    def test_complete_certificate_and_frozen_report(self):
        report, _ = certificate()
        self.assertEqual(report['status'], 'complete')
        families = report['families']
        self.assertEqual([len(families[k]['cases']) for k in families], [10, 4, 16, 20])
        for row in families['spectral']['cases']:
            self.assertTrue(all(t['certified'] and t['within_tolerance'] for t in row['recovery']))
            self.assertTrue(all(c['certified'] for c in row['spectral_cutoffs']))
            self.assertTrue(row['omission_certified'])
            self.assertLessEqual(norm(row['withdrawal_after_128']), row['withdrawal_bound'])
        self.assertEqual(families['finite_algebra']['direction_dimensions'], [2, 1, 0])
        for row in families['finite_algebra']['cases']:
            self.assertTrue(row['unique_from_all_starts'])
            self.assertTrue(all(t['restored'] for t in row['driven_recovery']))
            self.assertTrue(any(not t['restores_output'] for t in row['withdrawal_recovery']))
        relational = families['graded_relation']
        self.assertEqual(sum(not r['unique_output'] for r in relational['cases']), 2)
        self.assertEqual(relational['higher_arity']['reconstructed'], 16)
        self.assertEqual(relational['weak_term_counterexample']['output_error'], 1)
        self.assertTrue(any(not t['restores_output'] for r in relational['cases']
                            for t in r['driven_recovery']))
        for row in families['cellular']['cases']:
            self.assertTrue(row['withdrawal_certified'])
            self.assertTrue(row['omission_certified'])
            self.assertTrue(all(t['certified'] for t in row['recovery']))
        path = Path(__file__).resolve().parents[1] / 'reports' / 'model-comparison-v1.json'
        expected = json.loads(path.read_text())
        actual = jsonable(report)
        actual['runtime'] = expected['runtime']
        # Float eigendiagnostics have platform-dependent last bits. Independently
        # checked above; compare the exact certificate and their completion status.
        for row, frozen in zip(actual['families']['spectral']['cases'],
                               expected['families']['spectral']['cases']):
            self.assertEqual(row['spectrum']['status'], frozen['spectrum']['status'])
            row['spectrum'] = frozen['spectrum']
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
