import unittest

import numpy as np
import torch
from scipy.linalg import block_diag

from src import abstraction_preflight as pre
from src import masked_capacity as c
from src import relational_capacity as rc
from src import single_module_plasticity as v1

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:   # already set by a test module imported earlier in the same process
    pass
torch.use_deterministic_algorithms(True)

CONFIGS = (('S2', None), ('P0', None), ('D1', .01))


def fixture(name, seed):
    ps = [pre.random_params(np.random.default_rng([seed, m])) for m in (0, 1)]
    return [dict(maps=ps[0]['maps'])] if name == 'S2' else [dict(maps=p['maps'], v=p['v']) for p in ps]


def scenes(rows=(0, 12, 49)):
    _, _, clean, _ = c.task()
    return clean.reshape(72, 32)[list(rows)]


class Setup(unittest.TestCase):
    def test_seed_streams_are_paired_and_reproducible(self):
        for world in rc.WORLDS:
            for start in rc.STARTS:
                s2, p0, d1 = (rc.initial_parameters(n, world, start) for n in ('S2', 'P0', 'D1'))
                for a, b in zip(p0, d1):
                    for x, y in zip(a['maps'], b['maps']):
                        np.testing.assert_array_equal(x, y)
                    np.testing.assert_array_equal(a['v'], b['v'])
                for x, y in zip(s2[0]['maps'], p0[0]['maps']):
                    np.testing.assert_array_equal(x, y)
                self.assertNotIn('v', s2[0])
                rng = np.random.default_rng(np.random.SeedSequence([rc.ROOT_SEED, world, start, 1]))
                maps = [np.stack([v1.qf(rng.standard_normal((s, t))) for _ in range(4)])
                        for s, t in ((8, 6), (6, 4), (4, 2))]
                v = rng.standard_normal((4, 2))
                for x, y in zip(maps, d1[1]['maps']):
                    np.testing.assert_array_equal(x, y)
                np.testing.assert_array_equal(v/np.linalg.norm(v, axis=1, keepdims=True), d1[1]['v'])
        self.assertFalse(np.array_equal(rc.initial_parameters('D1', 0, 0)[0]['maps'][0],
                                        rc.initial_parameters('D1', 0, 1)[0]['maps'][0]))
        self.assertEqual(rc.stored_floats(rc.initial_parameters('S2', 0, 0)), 320)
        self.assertEqual(rc.stored_floats(rc.initial_parameters('D1', 0, 0)), 656)

    def test_declared_constants(self):
        self.assertAlmostEqual(rc.TAU, 0.2545917550240833, places=13)
        self.assertAlmostEqual(rc.weak_coupling_threshold(), 6.896773292194916e-4, places=15)
        self.assertTrue(all(mu > rc.weak_coupling_threshold() for mu in rc.MUS))
        for name in rc.LAYOUTS:
            self.assertAlmostEqual(sum(sum(s['anchors']) for s in rc.LAYOUTS[name]), rc.ALPHA)


class Energy(unittest.TestCase):
    def test_hessian_matches_residual_expansion_and_torch(self):
        rng = np.random.default_rng(801)
        x = scenes()
        for name, mu in CONFIGS:
            params = fixture(name, 802)
            model = rc.Composition(name, mu, params)
            ks = [model.curvatures(m) for m in range(model.count)]
            for system in rc.LAYOUTS[name]:
                routes = [.325+.675*rng.random(16 if 'v' in params[m] else 12) for m in system['modules']]
                h = rc.numpy_hessian(params, system, routes, mu or 0.)
                with torch.no_grad():
                    ht = model.hessian(system, ks, [torch.tensor(r)[None] for r in routes])[0].numpy()
                np.testing.assert_allclose(ht, h, atol=1e-12)
                observed = x[0, :32]*pre.MASK
                energy = lambda z: rc.numpy_energy(z, observed, params, system, routes, mu or 0.)
                u, w = rng.normal(size=(2, len(h)))
                zero = np.zeros(len(h))
                self.assertAlmostEqual(energy(u+w)-energy(u)-energy(w)+energy(zero), u@h@w, places=9)
                # linear term: gradient at zero equals -rhs
                b = rc.numpy_rhs(system, observed)
                self.assertAlmostEqual(energy(u)-energy(zero)-u@h@u/2, -u@b, places=9)

    def test_spd_for_random_parameters_and_routes(self):
        rng = np.random.default_rng(803)
        for name, mu in (*CONFIGS[:2], *(('D1', m) for m in rc.MUS)):
            for seed in range(5):
                params = fixture(name, 900+seed)
                for system in rc.LAYOUTS[name]:
                    routes = [.325+.675*rng.random(16 if 'v' in params[m] else 12) for m in system['modules']]
                    self.assertGreater(np.linalg.eigvalsh(rc.numpy_hessian(params, system, routes, mu or 0.))[0], 0)

    def test_parallel_blocks_match_dense_equivalent(self):
        params, x = fixture('P0', 804), scenes()
        out, _, _ = rc.numpy_prediction(params, 'P0', None, x)
        for n, scene in enumerate(x):
            observed = scene[:32]*pre.MASK
            pilots = [np.linalg.solve(rc.numpy_hessian(params, s, [np.ones(16)], 0.), rc.numpy_rhs(s, observed))
                      for s in rc.LAYOUTS['P0']]
            routes = [.25+.75*rc.numpy_gates(params[m], z[None])[0] for m, z in enumerate(pilots)]
            h = block_diag(*[rc.numpy_hessian(params, s, [r], 0.) for s, r in zip(rc.LAYOUTS['P0'], routes)])
            b = np.r_[[rc.numpy_rhs(s, observed) for s in rc.LAYOUTS['P0']]].ravel()
            state = np.linalg.solve(h, b)
            np.testing.assert_allclose(out[n], (state[16:32]+state[96:112])/2, atol=1e-12)


class Inference(unittest.TestCase):
    def test_torch_numpy_schur_and_pilot_screen(self):
        x = scenes()
        for name, mu in CONFIGS:
            params = fixture(name, 805)
            with torch.no_grad():
                routed, fixed = rc.Composition(name, mu, params)(torch.tensor(x[:, :32]*pre.MASK))
            expected, pilot, checks = rc.numpy_prediction(params, name, mu, x)
            np.testing.assert_allclose(routed.numpy(), expected, atol=1e-10)
            np.testing.assert_allclose(fixed.numpy(), pilot, atol=1e-10)
            self.assertLessEqual(checks['schur_disagreement']['max'], 1e-9)

    def test_gates_use_collective_pilot_slice(self):
        params, x, mu = fixture('D1', 806), scenes(), .1
        expected, _, _ = rc.numpy_prediction(params, 'D1', mu, x)
        system = rc.LAYOUTS['D1'][0]
        isolated = []
        for scene in x:
            observed = scene[:32]*pre.MASK
            alone = dict(modules=(0,), anchors=(rc.ALPHA,))
            z_in = np.linalg.solve(rc.numpy_hessian(params, alone, [np.ones(16)], 0.), rc.numpy_rhs(alone, observed))
            routes = [.25+.75*rc.numpy_gates(params[0], z_in[None])[0],
                      .25+.75*rc.numpy_gates(params[1], np.zeros((1, 80)))[0]]
            y = np.linalg.solve(rc.numpy_hessian(params, system, routes, mu), rc.numpy_rhs(system, observed))
            isolated.append(y[96:112])
        self.assertGreater(np.abs(np.array(isolated)-expected).max(), 1e-6)

    def test_hidden_inputs_and_targets_do_not_enter_inference(self):
        x = scenes()
        for name, mu in CONFIGS:
            model = rc.Composition(name, mu, fixture(name, 807))
            changed = x.copy()
            changed[:, 16:] = 5.
            with torch.no_grad():
                a = model(torch.tensor(x[:, :32]*pre.MASK))
                b = model(torch.tensor(changed[:, :32]))
            for p, q in zip(a, b):
                np.testing.assert_allclose(p.numpy(), q.numpy(), atol=1e-12)


class Gradients(unittest.TestCase):
    def numerical_max(self, model, name, mu, x, tensor, index, eps=1e-6):
        flat = tensor.view(-1)
        original = float(flat[index].detach())
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                flat[index] = original+sign*eps
            pred = rc.numpy_prediction(model.numpy_parameters(), name, mu, x, checked=False)[0]
            values.append(np.max(np.sum((pred-x[:, 16:])**2, axis=1)/2))
        with torch.no_grad():
            flat[index] = original
        return (values[0]-values[1])/(2*eps)

    def test_max_objective_gradient_away_from_ties(self):
        x = scenes()
        for name, mu in CONFIGS:
            model = rc.Composition(name, mu, fixture(name, 808))
            objective, losses = model.objective(torch.tensor(x[:, :32]*pre.MASK), torch.tensor(x[:, 16:]))
            top = np.sort(losses.detach().numpy())
            self.assertGreater(top[-1]-top[-2], 1e-4)
            objective.backward()
            for tensor in [*model.maps, *model.vs]:   # v entries move off the sphere: Q^T Q curvature
                for index in (0, tensor.numel()//2, tensor.numel()-1):
                    numerical = self.numerical_max(model, name, mu, x, tensor, index)
                    self.assertAlmostEqual(float(tensor.grad.view(-1)[index]), numerical,
                                           delta=2e-6+1e-4*abs(numerical))

    def test_exact_tie_averages_active_gradients(self):
        x = scenes((5, 5))
        for name, mu in CONFIGS:
            params = fixture(name, 809)
            tied = rc.Composition(name, mu, params)
            objective, losses = tied.objective(torch.tensor(x[:, :32]*pre.MASK), torch.tensor(x[:, 16:]))
            self.assertEqual(float(losses[0]), float(losses[1]))
            objective.backward()
            single = rc.Composition(name, mu, params)
            single.objective(torch.tensor(x[:1, :32]*pre.MASK), torch.tensor(x[:1, 16:]))[0].backward()
            for a, b in zip(tied.parameters(), single.parameters()):
                np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), atol=1e-13)

    def test_tangents_and_retraction(self):
        x = scenes()
        for name, mu in CONFIGS:
            model = rc.Composition(name, mu, fixture(name, 810))
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            model.objective(torch.tensor(x[:, :32]*pre.MASK), torch.tensor(x[:, 16:]))[0].backward()
            model.project_gradients()
            for d in model.maps:
                t = d.detach().transpose(-1, -2)@d.grad
                self.assertLess(float(abs(t+t.transpose(-1, -2)).max()), 1e-10)
            for v in model.vs:
                self.assertLess(float(abs((v.detach()*v.grad).sum(dim=1)).max()), 1e-10)
            optimizer.step()
            model.retract()
            rc.check_constraints(model.numpy_parameters())


class Screen(unittest.TestCase):
    def test_screen_bounds_any_witness(self):
        _, _, clean, _ = c.task()
        x = clean.reshape(72, 32)
        target = x[:, 16:]
        same = rc.route_screen(target, target, target)
        self.assertTrue(same['excludes_setting'])            # routed == fixed: d=0 < tau
        rng = np.random.default_rng(811)
        fixed = target+rng.normal(size=target.shape)
        record = rc.route_screen(target, fixed, target)
        self.assertGreaterEqual(record['d']+1e-12, record['parameter_specific_threshold'])


if __name__ == '__main__':
    unittest.main()
