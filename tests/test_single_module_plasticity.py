import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import experiments.single_module_plasticity as runner
import src.representation_reuse as reuse
from src.locality_adaptation import B, D, Meter, ScoringFailure, groups, local_basis
from src.locality_adaptation_v2 import fit_cv_refit
from src.single_module_plasticity import (ANCHOR, BLOCKS, H_BASE, LATERAL, M, OFFSETS, PROFILES, RANK, RINGS,
    SECTORS, UPDATES_PER_BATCH, WIDTHS, Meter2, Module, acquire, block_curvatures, branch_signal, clean, clip,
    complete_basis, copy_params, direct_reference, energy, errors, fixed_route_map, gates, hessians,
    implicit_gradients, initial_params, linear_error_map_risk, linear_trace_risk, make_world, observe, qf, retract,
    rhs, routes, schur_states, score, sector_indices, stream, tangent, terminal_q, warm_up)


ROOT = Path(__file__).resolve().parents[1]


def fixture(stage=0, world=0):
    w = make_world(stage, world)
    xs, ys = observe(w, w['u'], stream(stage, world, 3))
    local, p = acquire(xs, ys)
    return w, xs, ys, local, p


WORLD, XS, YS, LOCAL, P_INIT = fixture()
P_SOURCE, _ = warm_up(P_INIT, XS, YS)


def random_params(rng, phase=3, unit=True):
    maps = [np.stack([np.linalg.qr(rng.standard_normal((SECTORS[l], SECTORS[l+1])))[0] for _ in range(RANK)])
            for l in range(RINGS-1)]
    v = rng.standard_normal((RANK, 2))
    if unit:
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return dict(phase=phase, maps=maps, v=v)


class Dependencies(unittest.TestCase):
    def test_frozen_dependencies_match_their_manifests(self):
        runner.check_dependencies(ROOT/'reports')


class Generator(unittest.TestCase):
    def test_reimplemented_recipe_matches_reuse_under_scoped_root(self):
        with mock.patch.object(reuse, 'ROOT_SEED', 2026102410):
            ref = reuse.make_world(1, 5)
            draws = reuse.stream(1, 5, 4)
        ours = make_world(1, 5)
        for key in ('u', 'v', 'w', 'rotation'):
            np.testing.assert_array_equal(ours[key], ref[key])
        self.assertEqual(ours['phase'], ref['phase'])
        for a, b in zip(stream(1, 5, 4), draws):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(reuse.ROOT_SEED, 2026102409)

    def test_signals_and_drift_geometry(self):
        w = make_world(0, 3)
        u, v, dr = w['u'], w['v'], w['w']
        np.testing.assert_allclose(np.hstack([u, v, dr]).T@np.hstack([u, v, dr]), np.eye(3*RANK), atol=1e-12)
        self.assertIn(w['jstar'], range(RANK))
        np.testing.assert_array_equal(branch_signal(w, 'stable'), u)
        reg, glo = branch_signal(w, 'regional'), branch_signal(w, 'global')
        cos = np.einsum('ij,ij->j', reg, u)
        np.testing.assert_allclose(cos, [.5 if j == w['jstar'] else 1. for j in range(RANK)], atol=1e-12)
        np.testing.assert_allclose(np.einsum('ij,ij->j', glo, u), .5, atol=1e-12)
        np.testing.assert_allclose(reg.T@reg, np.eye(RANK), atol=1e-12)

    def test_stream_namespaces_are_isolated(self):
        firsts = [stream(0, 0, part)[0][0, 0] for part in (3, 4, 7, 8, 9)]
        self.assertEqual(len(set(firsts)), 5)
        np.testing.assert_array_equal(stream(0, 0, 4)[0], stream(0, 0, 4)[0])
        self.assertNotEqual(stream(0, 0, 4)[0][0, 0], stream(1, 0, 4)[0][0, 0])

    def test_observation_recipe(self):
        draws = stream(0, 0, 8)
        x, y = observe(WORLD, WORLD['u'], draws)
        c, g1, g2, e1, e2 = draws
        r = WORLD['rotation']
        np.testing.assert_allclose(x, (c@WORLD['u'].T+2*g1@WORLD['v'].T+.5*e1)@r.T)
        np.testing.assert_allclose(clean(WORLD, WORLD['u'], draws), c@WORLD['u'].T@r.T)


class Geometry(unittest.TestCase):
    def test_sector_indices_hand_case(self):
        idx = sector_indices(3)
        np.testing.assert_array_equal(idx[0][0], [3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_array_equal(idx[0][3], [27, 28, 29, 30, 31, 0, 1, 2])
        np.testing.assert_array_equal(idx[1][1], [38, 39, 40, 41, 42, 43])
        np.testing.assert_array_equal(idx[2][2], [64, 65, 66, 67])
        np.testing.assert_array_equal(idx[3][3], [78, 79])
        self.assertEqual(sorted(np.concatenate([i.ravel() for i in idx])), list(range(M)))

    def test_lateral_and_anchor_hand_case(self):
        self.assertAlmostEqual(H_BASE[0, 0], ANCHOR+2*LATERAL)
        self.assertAlmostEqual(H_BASE[0, 31], -LATERAL)
        self.assertAlmostEqual(H_BASE[32, 55], -LATERAL)
        self.assertAlmostEqual(H_BASE[72, 79], -LATERAL)
        self.assertAlmostEqual(H_BASE[33, 33], 2*LATERAL)
        self.assertEqual(H_BASE[31, 32], 0)
        self.assertEqual(H_BASE[55, 56], 0)

    def test_basis_completion_and_initialization(self):
        rng = np.random.default_rng(0)
        w = rng.standard_normal(8)
        q = complete_basis(w)
        np.testing.assert_allclose(q.T@q, np.eye(8), atol=1e-12)
        np.testing.assert_allclose(q[:, 0], w/np.linalg.norm(w))
        np.testing.assert_allclose(complete_basis(np.eye(8)[2]), np.eye(8)[:, [2, 0, 1, 3, 4, 5, 6, 7]], atol=1e-15)
        p = initial_params(LOCAL['phase'], LOCAL['weights'])
        for j in range(RANK):
            through = p['maps'][0][j]@p['maps'][1][j]@p['maps'][2][j]@p['v'][j]
            np.testing.assert_allclose(through, LOCAL['weights'][j], atol=1e-12)
        np.testing.assert_array_equal(p['maps'][1][0], np.eye(6)[:, :4])
        np.testing.assert_array_equal(p['v'][0], [1., 0.])

    def test_acquired_phase_matches_v2(self):
        phase, weights, _ = fit_cv_refit(XS, YS)
        self.assertEqual(LOCAL['phase'], phase)
        np.testing.assert_array_equal(LOCAL['weights'], weights)
        self.assertEqual(P_INIT['phase'], phase)


class EnergyAndSolves(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.p = random_params(self.rng)
        self.k = block_curvatures(self.p)
        self.a = .3+.7*self.rng.random((3, RINGS, RANK))
        self.h = self.rng.standard_normal((3, D))

    def test_residual_expansion_matches_quadratic_form(self):
        hs = hessians(self.k, self.a)
        for n in range(3):
            x = self.rng.standard_normal(M)
            quad = .5*x@hs[n]@x-rhs(self.h)[n]@x+ANCHOR/2*self.h[n]@self.h[n]
            self.assertAlmostEqual(energy(x, self.h[n], self.p, self.a[n]), quad, delta=1e-8*abs(quad))

    def test_energy_gradient_and_hessian_by_differences(self):
        h, x = hessians(self.k, self.a)[0], self.rng.standard_normal(M)
        grad = h@x-rhs(self.h)[0]
        eps = 1e-5
        for i in (0, 31, 40, 79):
            e = np.eye(M)[i]
            fd = (energy(x+eps*e, self.h[0], self.p, self.a[0])-energy(x-eps*e, self.h[0], self.p, self.a[0]))/(2*eps)
            self.assertAlmostEqual(fd, grad[i], delta=1e-8*max(1, abs(grad[i])))
            big = 1e-2  # exact for a quadratic up to rounding
            fd2 = (energy(x+big*e, self.h[0], self.p, self.a[0])-2*energy(x, self.h[0], self.p, self.a[0])
                   + energy(x-big*e, self.h[0], self.p, self.a[0]))/big**2
            self.assertAlmostEqual(fd2, h[i, i], delta=1e-8)
        self.assertGreater(np.linalg.eigvalsh(h).min(), 0)

    def test_full_and_schur_solves_agree(self):
        module = Module(copy_params(self.p), meter=Meter2())
        state, _, _ = module.second(self.h, self.a)
        ref = schur_states(self.k, self.a, self.h)
        np.testing.assert_allclose(state, ref, rtol=1e-9, atol=1e-9)
        hs = hessians(self.k, self.a)
        res = np.linalg.norm(rhs(self.h)-np.einsum('nij,nj->ni', hs, state), axis=1)
        self.assertLess(res.max(), 1e-10)

    def test_nonpositive_hessian_stops(self):
        module = Module(copy_params(self.p), meter=Meter2())
        with mock.patch('src.single_module_plasticity.H_BASE', -np.eye(M)):
            with self.assertRaises(ScoringFailure):
                module.second(self.h, np.zeros_like(self.a))


class Gradients(unittest.TestCase):
    def loss(self, p, x, t, a):
        module = Module(p, meter=Meter2())
        state, _, _ = module.second(x, a)
        return np.sum((state[:, :D]-t)**2)/(4*D*(len(x)//2))

    def check(self, p, unit):
        rng = np.random.default_rng(2)
        x, t = rng.standard_normal((4, D)), rng.standard_normal((4, D))
        a = .3+.7*rng.random((4, RINGS, RANK))
        grads, loss = Module(copy_params(p), meter=Meter2()).gradient(x, t, a)
        self.assertAlmostEqual(loss, self.loss(copy_params(p), x, t, a))
        for block in range(RINGS):
            arr = p['maps'][block] if block < 3 else p['v']
            direction = rng.standard_normal(arr.shape)
            plus, minus = copy_params(p), copy_params(p)
            for q, sign in ((plus, 1), (minus, -1)):
                if block < 3:
                    q['maps'][block] = arr+sign*1e-6*direction
                else:
                    q['v'] = arr+sign*1e-6*direction
            fd = (self.loss(plus, x, t, a)-self.loss(minus, x, t, a))/2e-6
            analytic = np.sum(grads[block]*direction)
            self.assertAlmostEqual(fd, analytic, delta=1e-5*max(1e-3, abs(analytic)), msg=(block, unit))

    def test_fixed_route_gradient_on_manifold(self):
        self.check(P_SOURCE, True)

    def test_fixed_route_gradient_off_manifold_terminal(self):
        p = random_params(np.random.default_rng(3), unit=False)
        q = terminal_q(p['v'])
        self.assertGreater(np.abs(q@q-q).max(), 1e-3)
        self.check(p, False)


class Updates(unittest.TestCase):
    def test_tangent_projection(self):
        rng = np.random.default_rng(4)
        p = random_params(rng)
        grads = [rng.standard_normal(m.shape) for m in p['maps']]+[rng.standard_normal(p['v'].shape)]
        tan = tangent(p, grads)
        for maps, g in zip(p['maps'], tan[:3]):
            s = np.einsum('jab,jac->jbc', maps, g)
            np.testing.assert_allclose(s+np.swapaxes(s, 1, 2), 0, atol=1e-12)
        np.testing.assert_allclose(np.sum(p['v']*tan[3], 1), 0, atol=1e-12)

    def test_clipping(self):
        g = np.stack([np.full((2, 2), 2.), np.full((2, 2), .1)])
        out = clip(g)
        self.assertAlmostEqual(np.linalg.norm(out[0]), 1.)
        np.testing.assert_array_equal(out[1], g[1])

    def test_qf_identity_and_positive_diagonal(self):
        p = random_params(np.random.default_rng(5))
        np.testing.assert_allclose(qf(p['maps'][0]), p['maps'][0], atol=1e-12)
        with self.assertRaises(ScoringFailure):
            qf(np.zeros((1, 4, 2)))

    def test_retraction_uses_rate_and_bbar(self):
        rng = np.random.default_rng(6)
        p = random_params(rng)
        grads = [1e-3*rng.standard_normal(m.shape) for m in p['maps']]+[1e-3*rng.standard_normal(p['v'].shape)]
        bbar = .5+rng.random((RINGS, RANK))
        q = copy_params(p)
        retract(q, grads, 'graded', bbar)
        tan = tangent(p, grads)
        for l in range(3):
            expect = qf(p['maps'][l]-(.1*PROFILES['graded'][l]*bbar[l])[:, None, None]*clip(tan[l]))
            np.testing.assert_allclose(q['maps'][l], expect, atol=1e-14)
        v = p['v']-(.1*PROFILES['graded'][3]*bbar[3])[:, None]*clip(tan[3])
        np.testing.assert_allclose(q['v'], v/np.linalg.norm(v, axis=1, keepdims=True), atol=1e-14)

    def test_profiles(self):
        np.testing.assert_allclose(PROFILES['graded'], np.array([1, .5, .25, .125])/np.mean([1, .5, .25, .125]))
        np.testing.assert_allclose(PROFILES['reversed'], PROFILES['graded'][::-1])
        for v in PROFILES.values():
            self.assertAlmostEqual(np.mean(v), 1.)

    def test_update_recomputes_routes_and_keeps_constraints(self):
        module = Module(copy_params(P_SOURCE), 'graded', 'matched', Meter2())
        module.learn_batch(XS[:8], YS[:8])
        self.assertEqual(module.meter.counts['route_evaluations'], UPDATES_PER_BATCH*16)
        self.assertEqual(module.meter.counts['pilot_factorizations'], UPDATES_PER_BATCH)
        self.assertEqual(module.meter.counts['second_factorizations'], UPDATES_PER_BATCH*16)
        self.assertEqual(module.meter.counts['adjoint_solves'], UPDATES_PER_BATCH*16)
        for maps in module.p['maps']:
            gram = np.einsum('jab,jac->jbc', maps, maps)
            np.testing.assert_allclose(gram, np.broadcast_to(np.eye(maps.shape[2]), gram.shape), atol=1e-9)

    def test_warm_up_is_fixed_route_and_shares_factor(self):
        meter = Meter2()
        warm_up(P_INIT, XS[:16], YS[:16], meter)
        self.assertEqual(meter.counts['fixed_route_factorizations'], 2*UPDATES_PER_BATCH)
        self.assertEqual(meter.counts['route_evaluations'], 0)
        self.assertEqual(meter.counts['second_factorizations'], 0)


class Routes(unittest.TestCase):
    def setUp(self):
        self.module = Module(copy_params(P_SOURCE), 'graded', 'matched', Meter2())
        self.x = XS[:16]

    def test_gate_bounds_and_mean_one_b(self):
        g, a, b, _ = self.module.route(self.x)
        self.assertTrue(np.all((g > .1) & (g <= 1)))
        self.assertTrue(np.all((a >= .325) & (a <= 1)))
        np.testing.assert_allclose(b.mean(2), 1, atol=1e-14)
        am, bm = routes(g, 'mean')
        np.testing.assert_array_equal(bm, 1.)
        np.testing.assert_allclose(am, .25+.75*g.mean(2, keepdims=True).repeat(RANK, 2))
        a_s, b_s = routes(g, 'shifted')
        np.testing.assert_array_equal(a_s[:, :, 0], a[:, :, 1])
        np.testing.assert_allclose(np.sort(b_s, 2), np.sort(b, 2), rtol=1e-15)

    def test_matching_residual_zero_gives_one(self):
        z = np.zeros((1, M))
        z[0, sector_indices(P_SOURCE['phase'])[3][0]] = P_SOURCE['v'][0]
        for l in (2, 1, 0):
            idx = sector_indices(P_SOURCE['phase'])
            z[0, idx[l][0]] = P_SOURCE['maps'][l][0]@z[0, idx[l+1][0]]
        self.assertTrue(np.allclose(gates(P_SOURCE, z)[0, :, 0], 1.))

    def test_target_cannot_change_routes_or_output(self):
        g1, a1, _, _ = self.module.route(self.x)
        out1, _ = self.module.infer(self.x)
        grads_a, _ = self.module.gradient(self.x, YS[:16], a1)
        grads_b, _ = self.module.gradient(self.x, -YS[:16], a1)
        g2, _, _, _ = self.module.route(self.x)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(out1, self.module.infer(self.x)[0])
        self.assertFalse(np.allclose(grads_a[0], grads_b[0]))


class Evaluator(unittest.TestCase):
    def test_linear_trace_risk_matches_error_map(self):
        rng = np.random.default_rng(7)
        module = Module(copy_params(P_SOURCE), meter=Meter2())
        m = fixed_route_map(module, .3+.7*rng.random((RINGS, RANK)))
        s = WORLD['rotation']@WORLD['u']@WORLD['u'].T@WORLD['rotation'].T
        n = WORLD['rotation']@(4*WORLD['v']@WORLD['v'].T+.25*np.eye(D))@WORLD['rotation'].T
        self.assertAlmostEqual(linear_trace_risk(m, s, n), linear_error_map_risk(m, s, n), delta=1e-12)
        r = WORLD['rotation']@WORLD['u']
        self.assertAlmostEqual(linear_trace_risk(r@r.T, s, n), .25, delta=1e-12)

    def test_routed_evaluator_matches_samplewise_errors(self):
        draws = stream(0, 0, 8)
        x, s = observe(WORLD, WORLD['u'], draws)[0], clean(WORLD, WORLD['u'], draws)
        module = Module(copy_params(P_SOURCE), meter=Meter2())
        out, _ = module.infer(x[:20])
        explicit = np.mean([np.sum((Module(copy_params(P_SOURCE), meter=Meter2()).infer(x[i:i+1])[0][0]-s[i])**2)
                            for i in range(20)])/RANK
        self.assertAlmostEqual(score(out, s[:20], WORLD['phase'])['E'], explicit, delta=1e-12)
        np.testing.assert_allclose(errors(out, s[:20], WORLD['phase']).sum(1), np.sum((out-s[:20])**2, 1))

    def test_controls_on_draws(self):
        sets = runner.evaluation_sets(WORLD, WORLD['u'], stream(0, 0, 8), stream(0, 0, 9))
        c = runner.controls(WORLD, sets, WORLD['u'])
        self.assertAlmostEqual(c['A']['zero'], 1, delta=.25)
        self.assertAlmostEqual(c['A']['raw'], 6, delta=.6)
        self.assertAlmostEqual(c['A']['oracle_projection'], .25, delta=.06)

    def test_evaluator_truth_cannot_change_updates(self):
        draws = stream(0, 0, 4)
        h1, h2 = observe(WORLD, branch_signal(WORLD, 'regional'), draws)
        quiet = Module(copy_params(P_SOURCE), 'graded', 'matched', Meter2())
        noisy = Module(copy_params(P_SOURCE), 'graded', 'matched', Meter2())
        ev = Meter2()
        sets = runner.evaluation_sets(WORLD, WORLD['u'], stream(0, 0, 8), stream(0, 0, 9))
        for i in range(0, 16, 8):
            quiet.learn_batch(h1[i:i+8], h2[i:i+8])
            runner.evaluate_layered(noisy, sets, WORLD['phase'], ev)
            noisy.learn_batch(h1[i:i+8], h2[i:i+8])
        for a, b in zip(quiet.p['maps']+[quiet.p['v']], noisy.p['maps']+[noisy.p['v']]):
            np.testing.assert_array_equal(a, b)


class Branches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        draws_b, draws_r = stream(0, 0, 4), stream(0, 0, 7)
        signal = branch_signal(WORLD, 'regional')
        cls.data = dict(B=observe(WORLD, signal, draws_b), **{'return': observe(WORLD, WORLD['u'], draws_r)})
        cls.sets = runner.evaluation_sets(WORLD, signal, stream(0, 0, 8), stream(0, 0, 9))
        cls.meters, cls.ev = runner.new_meters(), Meter2()
        cls.before = copy_params(P_SOURCE)
        cls.rows = runner.layered_branch('graded_matched', P_SOURCE, cls.data, cls.sets, WORLD, cls.meters,
                                         cls.ev, dict(world=0, branch='regional'))
        cls.frozen = runner.layered_branch('frozen_layered', P_SOURCE, cls.data, cls.sets, WORLD, cls.meters,
                                           cls.ev, dict(world=0, branch='regional'))

    def test_checkpoints_nested_and_continuous(self):
        marks = [(r['episode'], r['m']) for r in self.rows]
        self.assertEqual(marks, [('B', m) for m in runner.B_CHECKPOINTS]+[('return', m) for m in runner.RETURN_CHECKPOINTS])
        norms = [r['update_norm'].sum() for r in self.rows]
        self.assertTrue(all(b > a for a, b in zip(norms, norms[1:])))

    def test_source_checkpoint_is_not_mutated(self):
        for a, b in zip(P_SOURCE['maps']+[P_SOURCE['v']], self.before['maps']+[self.before['v']]):
            np.testing.assert_array_equal(a, b)

    def test_frozen_never_changes(self):
        self.assertTrue(all(np.all(r['param_change'] == 0) for r in self.frozen))
        self.assertEqual(self.meters['frozen_layered'].counts['parameter_updates'], 0)

    def test_actual_work_matches_charged_adaptation(self):
        counts, charged = self.meters['graded_matched'].counts, runner.charged_counts('graded_matched')
        for key in ('pilot_factorizations', 'pilot_solves', 'route_evaluations', 'second_factorizations',
                    'second_solves', 'adjoint_solves'):
            self.assertEqual(counts[key], charged[key], key)
        self.assertEqual(counts['parameter_updates'], charged['adaptation_updates'])

    def test_direct_references_continue_through_return(self):
        meters = runner.new_meters()
        rows = runner.direct_branch('phase_target_cca', LOCAL, self.data, self.sets, WORLD, meters, Meter2(),
                                    dict(world=0, branch='regional'))
        self.assertFalse(rows[0]['defined'])
        self.assertEqual(meters['phase_target_cca'].counts['target_pairs'], 512)
        learner = direct_reference('pooled_local_cca', LOCAL)
        self.assertTrue(learner.defined_at_zero)

    def test_schur_checked_at_every_checkpoint(self):
        self.assertTrue(all(r['schur_max_rel'] <= 1e-9 for r in self.rows))
        self.assertEqual(self.ev.counts['schur_checked_inputs'], 2*8*len(self.rows+self.frozen))


if __name__ == '__main__':
    unittest.main()
