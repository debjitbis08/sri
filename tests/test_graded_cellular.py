from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import unittest

from experiments.graded_cellular import render, sensitivity_probes
from src.graded_cellular import (
    Model, ReleasedFlow, VARIANTS, energy, gradient, integrate,
    mean, pulse, real_roots, stationary_modes,
)


ROOT = Path(__file__).resolve().parents[1]


class GradedCellularTests(unittest.TestCase):
    def test_gradient_against_independent_finite_difference(self):
        state = (.2, -.4, .8, -.3)
        for name, cellular, beta, face in VARIANTS:
            model = Model(name, cellular, beta, face, 1., 1.)
            exact = gradient(model, state)
            for index in range(4):
                left, right = list(state), list(state)
                left[index] -= 1e-6
                right[index] += 1e-6
                numeric = (energy(model, right)-energy(model, left))/2e-6
                self.assertAlmostEqual(exact[index], numeric, places=7)

    def test_root_special_cases_and_double_root_retention(self):
        combined = Model('combined', True, 1., 0., 0., 1.)
        modes = stationary_modes(combined)
        self.assertEqual([round(m['mean'], 10) for m in modes], [-1., 0., 1.])
        self.assertEqual([m['kind'] for m in modes], ['isolated_minimum', 'unstable', 'isolated_minimum'])
        critical = stationary_modes(replace(combined, gamma=2.))
        self.assertEqual([round(m['mean'], 10) for m in critical], [-.5, 1.])
        self.assertEqual([m['kind'] for m in critical], ['marginal', 'isolated_minimum'])
        self.assertEqual(len(stationary_modes(replace(combined, gamma=4.))), 1)
        self.assertIsNone(real_roots(0, 0, 0))
        self.assertEqual(real_roots(0, 0, 1), ())

    def test_cell_attachment_threshold(self):
        model = Model('disk', True, 1., .25, 0., 1.)
        modes = stationary_modes(model)
        self.assertAlmostEqual(modes[0]['mean'], -math.sqrt(7)/4)
        self.assertAlmostEqual(modes[-1]['mean'], math.sqrt(7)/4)
        collapsed = stationary_modes(replace(model, face=.5))
        self.assertEqual(len(collapsed), 1)
        self.assertAlmostEqual(collapsed[0]['mean'], 0.)
        self.assertEqual(collapsed[0]['kind'], 'isolated_minimum')

    def test_reduced_release_matches_full_coordinate_rk4(self):
        start = (.7, -.2, .1, -.3)
        for name, cellular, beta, face in VARIANTS:
            model = Model(name, cellular, beta, face, .25, 1.)
            full = integrate(lambda s: tuple(-v for v in gradient(model, s)), start, 1., 1/128)
            reduced = ReleasedFlow(model, duration=1., dt=1/128).run(start)
            self.assertLess(max(abs(a-b) for a, b in zip(full['state'], reduced['final'])), 1e-8)

    def test_pulse_matches_full_clamped_coordinate_flow(self):
        model = Model('combined', True, 1., 0., 1., 1.)
        for evidence in (-1, 1):
            def flow(state):
                grad = gradient(model, state)
                return (0.,) + tuple(-v for v in grad[1:])
            full = integrate(flow, (evidence, 0., 0., 1.), 1., 1/64)
            reduced = pulse(model, evidence, duration=1., dt=1/64)
            self.assertLess(max(abs(a-b) for a, b in zip(full['state'], reduced['state'])), 1e-12)
            self.assertEqual(reduced['state'][0], evidence)

    def test_controls_fail_for_different_analytic_reasons(self):
        start = (1., 1., .75, 1.)
        combined = Model('combined', True, 1., 0., 0., 1.)
        joint = ReleasedFlow(combined).run(start)
        self.assertEqual(joint['status'], 'converged_numerically')
        self.assertAlmostEqual(joint['output'], 1., places=7)
        cellular = ReleasedFlow(replace(combined, beta=0.)).run(start)
        self.assertAlmostEqual(cellular['output'], mean(start), places=7)
        relational = ReleasedFlow(replace(combined, cellular=False)).run(start)
        self.assertAlmostEqual(relational['output'], 1.-1/6, places=7)
        self.assertTrue(all(m['kind'] != 'isolated_minimum' for m in
                            stationary_modes(replace(combined, cellular=False))))

    def test_large_disturbance_can_cross_basin(self):
        model = Model('combined', True, 1., 0., 0., 1.)
        result = ReleasedFlow(model).run((1., 1., -3., 1.))
        self.assertAlmostEqual(result['output'], -1., places=7)

    def test_finite_time_nonconvergence_remains_unresolved(self):
        model = Model('cellular', True, 0., 0., .25, 1.)
        result = ReleasedFlow(model).run((-1., -1., -1., 1.))
        self.assertEqual(result['status'], 'unresolved')
        # Analytical unique stationary mode remains available independently.
        self.assertEqual(len(stationary_modes(model)), 1)

    def test_context_symmetry_and_sensitivity_probes(self):
        positive = Model('combined', True, 1., 0., 1., 1.)
        negative = replace(positive, context=-1.)
        state = (.8, -.2, .3, -.4)
        a = ReleasedFlow(positive, duration=4.).run(state)
        b = ReleasedFlow(negative, duration=4.).run(tuple(-v for v in state))
        self.assertLess(max(abs(x+y) for x, y in zip(a['final'], b['final'])), 1e-12)
        probes = sensitivity_probes()
        for probe in probes['context_saddle_node']:
            self.assertEqual([r['isolated_modes'] for r in probe['comparison']], [2, 1])
        self.assertEqual([r['isolated_modes'] for r in probes['face_bifurcation']], [2, 1])

    def test_report_scope_and_all_case_invariants(self):
        path = ROOT/'reports'/'graded-cellular-v1.json'
        report = json.loads(path.read_text())
        self.assertEqual(render(report), path.read_text())
        for name, digest in report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)
        self.assertEqual(len(report['cases']), 60)
        self.assertEqual(sum(r['basins']['sample_count'] for r in report['cases']), 4860)
        self.assertEqual(report['scope_discount']['architecture_grade'], 'not_evaluated')
        self.assertIn('graded layer connections', report['scope_discount']['unimplemented'])
        self.assertIn('multiple layers per module', report['scope_discount']['unimplemented'])
        for row in report['cases']:
            self.assertLessEqual(row['numerical_audit']['max_energy_increase'], 1e-10)
            self.assertTrue(row['numerical_audit']['all_pulse_refinements_agree'])
            model = Model(**row['model'])
            for mode in row['stationary_modes']:
                self.assertLess(max(map(abs, gradient(model, mode['state']))), 1e-10)
            for trial in row['disturbances']:
                if trial['status'] == 'unresolved':
                    self.assertIsNone(trial['restores_output'])
                    self.assertIsNone(trial['restores_full_state'])
            if row['model']['name'] == 'combined' and row['model']['gamma'] in (0., .25, 1.):
                self.assertEqual(row['isolated_stable_modes'], 2)
                self.assertTrue(row['pulse_distinction'])
                for magnitude in ('0.25', '1.0'):
                    self.assertEqual(row['recovery_counts'][magnitude],
                                     dict(total=16, restored=16, changed=0, unresolved=0))

    def test_invalid_domains_and_step_sizes(self):
        with self.assertRaises(ValueError):
            Model('bad', False, 1., 1., 0., 1.)
        with self.assertRaises(ValueError):
            integrate(lambda z: z, (0.,), 1., 0.)
        with self.assertRaises(ValueError):
            integrate(lambda z: z, (0.,), 1., .3)


if __name__ == '__main__':
    unittest.main()
