import hashlib
import json
from pathlib import Path
import random
import unittest

from src.core_mediation import configurations, parameters
from src.core_structured import (
    CORE_CODES, Module, SAT_CODE, System, distance_one_code, pulse_response,
    readout_energy, recovery,
)


ROOT = Path(__file__).resolve().parents[1]
XOR = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
XOR_TABLES = {('ABK', a, b, k): (0 if k == a ^ b else 2)
              for a in (0, 1) for b in (0, 1) for k in (0, 1)}


class CoreStructuredTests(unittest.TestCase):
    def test_codes_have_declared_separation_and_tie_readout(self):
        self.assertEqual(Module(SAT_CODE).min_separation(), 4)
        self.assertEqual([Module(CORE_CODES[n]).min_separation() for n in (2, 3, 4)], [6, 4, 5])
        self.assertEqual(Module(distance_one_code(3)).min_separation(), 1)
        self.assertIsNone(Module(SAT_CODE).readout[0b0011])

    def test_bottom_rule_uses_max_attained_value_plus_margin(self):
        energy = readout_energy('T', 2, XOR_TABLES)
        # u_A takes 0+1 and psi takes 2+1 at a bottom readout; u_B, u_K are 0.
        self.assertEqual(energy[(None, 0, 0)], 4)
        self.assertEqual(energy[(1, 0, 1)], 0)

    def test_every_prototype_configuration_is_stable_for_any_tables(self):
        # Separation >= 3: one flip cannot change a readout, so it costs +lambda.
        rng = random.Random(3)
        for cls, n in (('T', 2), ('S', 3), ('F', 4), ('G', 2)):
            for _ in range(5):
                tables = {name: rng.randint(-16, 16) for name in parameters(cls, n)}
                system = System(cls, n, tables, rng.choice((1, 2, 4, 8)), mu=rng.choice((None, 4)))
                for x in configurations(n):
                    self.assertEqual(system.successors(system.encode(x)), ())

    def test_single_flips_recover_locally_with_separated_codes(self):
        rows = recovery(System('T', 2, XOR_TABLES, 1), (0, 1, 1))
        flips = [r for r in rows if r['kind'] == 'flip1']
        self.assertTrue(all(r['state_recovered'] and not r['other_module_readout_changed']
                            for r in flips))

    def test_distance_one_core_lets_a_flip_change_its_readout(self):
        report = json.loads((ROOT/'reports/core-mediation-v1.json').read_text())
        case = next(c for c in report['cases']
                    if (c['cls'], c['n'], c['level'], c['strict']) == ('S', 3, 1, True))
        row = next(r for r in case['results'] if r['targets'] == [[0, 0, 0], [1, 1, 1]])
        tables = {tuple(int(p) if p.isdigit() else p for p in k.split('.')): v
                  for k, v in row['witness'].items()}
        system = System('S', 3, tables, 1, core_code=distance_one_code(3))
        rows = [r for x in ((0, 0, 0), (1, 1, 1)) for r in recovery(system, x)
                if r['kind'] == 'flip1' and r['module'] == 2]
        self.assertFalse(all(r['readout_recovered'] for r in rows))

    def test_pulse_moves_a_but_not_the_core(self):
        out = pulse_response('T', 2, XOR_TABLES, 1, (0, 0, 0), 1, 32)
        self.assertEqual(out['final'], [(1, 0, 0)])

    def test_frozen_report_matches_sources(self):
        report = json.loads((ROOT/'reports/core-mediation-v1-phase-c.json').read_text())
        for path, digest in report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest, path)
        self.assertEqual(len(report['witnesses']), 45)


if __name__ == '__main__':
    unittest.main()
