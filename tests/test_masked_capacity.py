import unittest

import numpy as np
import torch

from src import masked_capacity as c
from src import abstraction_preflight as pre
from src import single_module_plasticity as v1

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


class Task(unittest.TestCase):
    def test_factorial_split_and_closed_orbits(self):
        labels, _, clean, held = c.task()
        self.assertEqual(held.sum(), 24)
        np.testing.assert_allclose(np.sum(clean[:, :2]**2, axis=(1, 2)), 2.)
        np.testing.assert_allclose(np.sum(clean[:, 2:]**2, axis=(1, 2)), 2.)
        for r in (1, 2, 3):
            for q in range(3):
                subset = (labels[:, 0] == r) & (labels[:, 1] == q)
                self.assertEqual(int(held[subset].sum()), 8 if q == r-1 else 0)

    def test_template_choices_do_not_read_hidden_targets(self):
        _, _, clean, held = c.task()
        x = clean.reshape(72, 32)
        matrices = c.rendering(0)
        rng = np.random.default_rng(701)
        changed = x.copy()
        changed[:, 16:] = rng.normal(size=(72, 16))
        for kind in ('lookup', 'signed_lookup', 'rotation_scale_lookup', 'factored_template'):
            _, a = c.template_prediction(x, matrices, ~held, kind)
            _, b = c.template_prediction(changed, matrices, ~held, kind)
            np.testing.assert_array_equal(a['chosen_scene'], b['chosen_scene'])
            np.testing.assert_allclose(a['min_visible_squared_distance'], b['min_visible_squared_distance'])

    def test_common_scale_gap_and_factored_reference(self):
        _, _, clean, held = c.task()
        x = clean.reshape(72, 32)
        source, query = clean[0, :2, :2], clean[8, :2, :2]
        scale = np.sum(source*query)/np.sum(source**2)
        self.assertAlmostEqual(np.sum((scale*source-query)**2)/2, .2, places=12)
        pred, _ = c.template_prediction(x, c.rendering(0), ~held, 'factored_template')
        np.testing.assert_allclose(pred, x[:, 16:], atol=1e-12)


class Module(unittest.TestCase):
    def setUp(self):
        self.p = pre.random_params(np.random.default_rng(702))
        _, _, clean, _ = c.task()
        self.x = clean.reshape(72, 32)[[0, 12, 49]]
        self.inputs = torch.tensor(self.x*pre.MASK, dtype=torch.float64)
        self.targets = torch.tensor(self.x[:, 16:], dtype=torch.float64)
        self.model = c.MaskedModule(self.p)

    def test_numpy_and_schur_equivalence(self):
        with torch.no_grad():
            actual = self.model(self.inputs).numpy()[:, 16:]
        expected, _ = c.numpy_prediction(self.p, self.x)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)

    def test_mask_hidden_input_independence(self):
        changed = self.inputs.clone()
        changed[:, 16:] = 4.
        with torch.no_grad():
            np.testing.assert_allclose(self.model(changed).numpy(), self.model(self.inputs).numpy(), atol=1e-12)

    def test_full_gradient_including_gates(self):
        loss = self.model.loss(self.inputs, self.targets)
        loss.backward()
        tensors = [*self.model.maps, self.model.v]
        epsilon = 1e-6
        for tensor in tensors:
            grad = tensor.grad.detach().numpy().copy()
            for index in (0, tensor.numel()//2, tensor.numel()-1):
                flat = tensor.view(-1)
                original = float(flat[index].detach())
                with torch.no_grad():
                    flat[index] = original+epsilon
                plus = c.numpy_prediction(self.model.numpy_parameters(), self.x, checked=False)[0]
                with torch.no_grad():
                    flat[index] = original-epsilon
                minus = c.numpy_prediction(self.model.numpy_parameters(), self.x, checked=False)[0]
                with torch.no_grad():
                    flat[index] = original
                lp = np.sum((plus-self.x[:, 16:])**2)/len(self.x)/2
                lm = np.sum((minus-self.x[:, 16:])**2)/len(self.x)/2
                numerical = (lp-lm)/(2*epsilon)
                self.assertAlmostEqual(float(grad.flat[index]), numerical, delta=2e-6+1e-4*abs(numerical))

    def test_tangents_and_optimizer_retraction(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.05)
        self.model.loss(self.inputs, self.targets).backward()
        self.model.project_gradients()
        for d in self.model.maps:
            tangency = d.detach().transpose(-1, -2)@d.grad
            self.assertLess(float(abs(tangency+tangency.transpose(-1, -2)).max()), 1e-10)
        self.assertLess(float(abs((self.model.v.detach()*self.model.v.grad).sum(dim=1)).max()), 1e-10)
        optimizer.step()
        self.model.retract()
        v1.check_constraints(self.model.numpy_parameters())

    def test_hessian_energy_expansion(self):
        k, _ = self.model.curvatures()
        np.testing.assert_allclose(k.detach().numpy(), v1.block_curvatures(self.p), atol=1e-12)
        rng = np.random.default_rng(703)
        routes = .325+.675*rng.random((4, 4))
        h = pre.masked_hessian(k.detach().numpy(), routes, pre.MASK)
        u, v = rng.normal(size=(2, 80))
        energy = lambda z: pre.masked_energy(z, self.x[0]*pre.MASK, self.p, routes, pre.MASK)
        self.assertAlmostEqual(energy(u+v)-energy(u)-energy(v)+energy(np.zeros(80)), u@h@v, places=10)


if __name__ == '__main__':
    unittest.main()
