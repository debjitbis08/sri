"""Gates for the single-module learning benchmark v1 (contract section 7)."""
import unittest

import numpy as np
import torch

from src.learning_benchmark import (ARMS, BLOCK, CoarseLocal, D, M, coefficients, encode,
    evaluation_sets, fixed_averaging, local_cca, probe_r2, task_world, train_ensemble,
    training_pairs)


class BenchmarkTests(unittest.TestCase):
    def test_coarse_arm_is_explicit_restriction_map(self):
        torch.manual_seed(0)
        model = CoarseLocal()
        a, b = np.zeros((8, D)), np.zeros((M, 8))
        for i in range(8):
            a[i, i*4:(i+1)*4] = model.a[i].detach().numpy()
        for k in range(M):
            b[k, k*2:(k+1)*2] = model.b[k].detach().numpy()
        h = np.random.default_rng(0).standard_normal((5, D)).astype(np.float32)
        out = model(torch.from_numpy(h)).detach().numpy()
        np.testing.assert_allclose(out, h@(b@a).T, atol=1e-5)

    def test_world_geometry(self):
        u, v, rotations = task_world(5, seed=1)
        np.testing.assert_allclose(u.T@u, np.eye(M), atol=1e-12)
        np.testing.assert_allclose(v.T@v, np.eye(M), atol=1e-12)
        np.testing.assert_allclose(u.T@v, 0, atol=1e-12)
        for r in rotations.values():
            np.testing.assert_allclose(r.T@r, np.eye(D), atol=1e-12)
        for k in range(M):
            outside = np.delete(u[:, k], np.arange(k*BLOCK, (k+1)*BLOCK))
            self.assertEqual(np.abs(outside).max(), 0)

    def test_sign_conditioning(self):
        rng = np.random.default_rng(0)
        self.assertTrue(np.all(np.prod(coefficients(rng, 500)[:, :2], 1) >= 0))
        self.assertTrue(np.all(np.prod(coefficients(rng, 500, heldout=True)[:, :2], 1) <= 0))

    def test_prefixes_are_nested_and_deterministic(self):
        w = task_world(2, seed=1)
        a = training_pairs(w, 'shifted', 2, seed=1)
        b = training_pairs(w, 'shifted', 2, seed=1)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[0].shape, (256, D))

    def test_probe_floor_and_ceiling(self):
        w = task_world(3, seed=1)
        probe, tests = evaluation_sets(w, 'aligned', 3, seed=1)
        constant = probe_r2(np.ones((len(probe[0]), M)), probe[1],
                            {k: (np.ones((len(h), M)), c) for k, (h, c) in tests.items()})
        self.assertLess(constant['in_distribution'], .02)
        oracle = probe_r2(probe[0]@w[0], probe[1], {k: (h@w[0], c) for k, (h, c) in tests.items()})
        self.assertGreater(oracle['in_distribution'], .7)
        # Fixed averaging has no access to the texture-free templates.
        avg = fixed_averaging()
        fixed = probe_r2(probe[0]@avg, probe[1], {k: (h@avg, c) for k, (h, c) in tests.items()})
        self.assertLess(fixed['in_distribution'], oracle['in_distribution'])

    def test_local_cca_recovers_aligned_templates_with_many_pairs(self):
        w = task_world(4, seed=1)
        h1, h2 = training_pairs(w, 'aligned', 4, seed=1)
        basis = local_cca(h1, h2)
        self.assertGreater(np.mean(np.abs(np.sum(basis*w[0], 0))), .9)

    def test_vectorized_trainer_is_deterministic(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(0)
        h1, h2 = rng.standard_normal((2, 3, 8, D))
        cfgs = ARMS['mlp_vicreg']['grid'][0::2]  # width 16, three learning rates
        runs = [train_ensemble('mlp_vicreg', cfgs, h1, h2, [1, 2, 3], steps=20) for _ in range(2)]
        for k in range(3):
            np.testing.assert_array_equal(encode(*runs[0][:2], k, h1[k]), encode(*runs[1][:2], k, h1[k]))
        # Members are independent: member 0 matches a single-member run.
        single = train_ensemble('mlp_vicreg', cfgs[:1], h1[:1], h2[:1], [1], steps=20)
        np.testing.assert_allclose(encode(*single[:2], 0, h1[0]), encode(*runs[0][:2], 0, h1[0]), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
