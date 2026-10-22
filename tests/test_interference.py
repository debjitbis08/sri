import unittest
from itertools import permutations, product
from experiments.interference import certificate
from src.interference import Algebra, XOR, SATURATING, OR, response


class InterferenceTests(unittest.TestCase):
    def test_candidates_and_negative_control(self):
        for algebra in (XOR, SATURATING, OR):
            self.assertTrue(algebra.associative())
            self.assertTrue(algebra.commutative())
        self.assertEqual(response(XOR, 1, 1), (frozenset(), frozenset({1})))
        self.assertEqual(response(SATURATING, 1, 1), (frozenset({2}), frozenset({1})))
        for a, b in product(range(2), repeat=2):
            joint, independent = response(OR, a, b)
            self.assertEqual(joint, independent)

    def test_edge_ablations(self):
        for algebra in (XOR, SATURATING):
            for inputs in product(range(len(algebra.table)), repeat=2):
                for edges in ((False, False), (True, False), (False, True)):
                    joint, independent = response(algebra, *inputs, edges)
                    self.assertEqual(joint, independent)

    def test_invariant_under_nonzero_relabeling(self):
        for algebra in (XOR, SATURATING):
            n = len(algebra.table)
            for labels in permutations(range(1, n)):
                mapping = (0,) + labels
                table = [[0] * n for _ in range(n)]
                for a, b in product(range(n), repeat=2):
                    table[mapping[a]][mapping[b]] = mapping[algebra.table[a][b]]
                renamed = Algebra(tuple(map(tuple, table)))
                for a, b in product(range(n), repeat=2):
                    expected = tuple(frozenset(mapping[x] for x in signature)
                                     for signature in response(algebra, a, b))
                    self.assertEqual(response(renamed, mapping[a], mapping[b]), expected)

    def test_exhaustive_certificate(self):
        report = certificate()
        self.assertEqual([r['tables'] for r in report['enumeration']], [1, 2, 27])
        self.assertEqual(report['smallest_witness']['states'], 2)
        for row in report['enumeration']:
            self.assertEqual(row['cases'], row['monoids'] * 4 * row['states'] ** 2)
            self.assertEqual(row['witnesses_by_edge_count'][:2], [0, 0])
        self.assertEqual(report['enumeration'][0]['witnesses'], 0)

    def test_invalid_tables(self):
        for table in ((), ((0, 1),), ((0, 2), (1, 0)), ((1,),)):
            with self.assertRaises(ValueError):
                Algebra(table)


if __name__ == '__main__':
    unittest.main()
