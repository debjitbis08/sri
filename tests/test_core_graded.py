from itertools import product
import random
import unittest

from src.core_graded import GradedSystem, census, encode, evaluate, pulse_then_release, strict_minimum
from src.core_mediation import energy as mode_energy, parameters


WITNESS = {('uA', 1): 1, ('ABK', 0, 0, 1): 1, ('ABK', 0, 1, 0): 1,
           ('ABK', 1, 1, 0): 1, ('ABK', 1, 0, 1): -1}
TARGETS = ((0, 0, 0), (0, 1, 1), (1, 0, 1))


class _Restricted(GradedSystem):
    """Only one module's units may flip; used to test the single-module proposition."""
    def __init__(self, base, module):
        self.__dict__.update(base.__dict__, cache={})
        self.module = module

    def successors(self, s):
        e = self.energy(s)
        return tuple(t for t in (s ^ (1 << (4*self.module+i)) for i in range(4))
                     if self.energy(t) < e)


class CoreGradedTests(unittest.TestCase):
    def test_energy_matches_scaled_mode_energy_at_prototypes(self):
        system = GradedSystem('T', WITNESS, 8, 1)
        for x in product((0, 1), repeat=3):
            self.assertEqual(system.energy(encode(x)), 64*8*mode_energy('T', WITNESS, x))

    def test_single_module_transition_threshold_proposition(self):
        rng = random.Random(11)
        for _ in range(60):
            cls = rng.choice('STF')
            tables = {name: rng.randint(-4, 4) for name in parameters(cls, 2)}
            sigma, lam = rng.choice((1, 4, 8)), rng.choice((0, 1, 2, 3))
            base = GradedSystem(cls, tables, sigma, lam)
            for x in product((0, 1), repeat=3):
                for m in range(3):
                    y = list(x); y[m] ^= 1
                    delta = mode_energy(cls, tables, tuple(y)) - mode_energy(cls, tables, x)
                    reached = encode(y) in _Restricted(base, m).explore(encode(x))[0]
                    self.assertEqual(reached, sigma*delta < -4*lam, (cls, tables, x, m))

    def test_worked_example_meets_all_criteria_at_unit_barrier_one(self):
        result = evaluate('T', WITNESS, TARGETS, 8, 1, 8)
        self.assertTrue(result['P1'] and result['P2'] and result['joint'])
        first = result['pulses'][0]
        self.assertEqual((first['final'], first['core_removed'], first['core_clamped']),
                         ([[1, 0, 1]], [[0, 0]], [[0, 0, 0]]))

    def test_strict_minimum_rejects_equal_energy_neighbours(self):
        flat = GradedSystem('T', {}, 8, 0)
        self.assertEqual(flat.successors(encode((0, 0, 0))), ())
        self.assertFalse(strict_minimum(flat, encode((0, 0, 0))))
        self.assertTrue(strict_minimum(GradedSystem('T', WITNESS, 8, 1), encode((0, 0, 0))))

    def test_global_census_reports_spurious_terminals(self):
        result = census(GradedSystem('T', WITNESS, 8, 1), TARGETS)
        self.assertEqual((result['terminals'], result['spurious_terminals']), (99, 96))
        self.assertEqual(sum(result['basins'].values()), 4096)
        self.assertEqual(result['committed_to_target'], 2013)

    def test_barrier_at_or_above_gain_blocks_mediation(self):
        for lam in (2, 3):
            result = evaluate('T', WITNESS, TARGETS, 8, lam, 8)
            self.assertTrue(result['P1'] and result['P2'])
            self.assertFalse(result['joint'])
            self.assertEqual(pulse_then_release(GradedSystem('T', WITNESS, 8, lam),
                                                (0, 0, 0), 1, 8), [(1, 0, 0)])


if __name__ == '__main__':
    unittest.main()
