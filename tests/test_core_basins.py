from fractions import Fraction
import json
from pathlib import Path
import random
import unittest

from src.core_basins import analyze, basins, predicted_terminal_prototypes, tie_free
from src.core_graded import GradedSystem, encode, popcount
from src.core_mediation import parameters


ROOT = Path(__file__).resolve().parents[1]
WITNESS = {('uA', 1): 1, ('ABK', 0, 0, 1): 1, ('ABK', 0, 1, 0): 1,
           ('ABK', 1, 1, 0): 1, ('ABK', 1, 0, 1): -1}
TARGETS = ((0, 0, 0), (0, 1, 1), (1, 0, 1))


class CoreBasinTests(unittest.TestCase):
    def test_tie_free_settings_have_only_predicted_prototype_terminals(self):
        rng = random.Random(5)
        checked = 0
        while checked < 25:
            cls = rng.choice('STF')
            tables = {name: rng.randint(-4, 4) for name in parameters(cls, 2)}
            sigma, lam = rng.choice((5, 7, 9, 13)), (rng.randint(1, 3), rng.randint(1, 3))
            if not tie_free(sigma, lam):
                continue
            system = GradedSystem(cls, tables, sigma, lam)
            terminals = [s for s in range(4096) if not system.successors(s)]
            self.assertTrue(all(popcount(w) in (0, 4) for s in terminals for w in system.words(s)))
            self.assertEqual(sorted(tuple(system.readouts(s)) for s in terminals),
                             sorted(predicted_terminal_prototypes(cls, tables, sigma, lam)))
            checked += 1

    def test_tie_control_has_off_prototype_terminals_and_tie_free_neighbour_does_not(self):
        self.assertFalse(tie_free(8, 1))
        self.assertEqual(analyze('T', WITNESS, TARGETS, 8, 1)['off_prototype_terminals'], 96)
        result = analyze('T', WITNESS, TARGETS, 9, 1)
        self.assertEqual((result['terminals'], result['mean_target_probability_uniform_start']),
                         (3, 1.0))

    def test_random_schedule_probabilities_sum_to_one(self):
        system = GradedSystem('T', WITNESS, 9, (1, 2))
        _, prob = basins(system, TARGETS)
        self.assertTrue(all(sum(p.values()) == Fraction(1) for p in prob.values()))
        self.assertEqual(prob[encode((0, 1, 1))], {(0, 1, 1): Fraction(1)})

    def test_frozen_report_confirms_propositions(self):
        report = json.loads((ROOT/'reports/core-graded-basins.json').read_text())
        self.assertTrue(report['propositions_confirmed_everywhere'])
        self.assertEqual(len(report['settings']), 14*3*9)


if __name__ == '__main__':
    unittest.main()
