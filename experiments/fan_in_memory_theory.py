"""Verify the conditional identities in docs/fan-in-memory-theory.md.

No neural training, random sampling, tuning, or changes to historical artifacts.
"""
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np


def main():
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[1]
    checks = []
    counts = {'linear_solves': 0, 'matrix_inversions': 0,
              'symmetric_eigendecompositions': 0, 'candidate_edges_scored': 0}

    def close(name, a, b, atol=1e-9):
        assert np.allclose(a, b, atol=atol, rtol=1e-9), (name, a, b)
        checks.append({'name': name, 'max_absolute_error': float(np.max(np.abs(np.asarray(a) - b)))})

    N = 27
    truth = np.tile(np.arange(1., 10.), 3)
    edges, bridge = [], []
    for sense in range(3):
        for n in range(8):
            edges.append((9 * sense + n, 9 * sense + n + 1))
            bridge.append(False)
    for n in range(9):
        edges.append((n, 9 + n)); bridge.append(False)
        if n + 1 not in (4, 6):
            edges.append((n, 18 + n)); bridge.append(True)
    B = np.zeros((len(edges), N))
    for row, (i, j) in enumerate(edges):
        B[row, i], B[row, j] = -1, 1
    bridge = np.array(bridge)
    anchor = np.zeros(N); anchor[0] = 1
    increments = B @ truth
    v = np.zeros(N); v[21], v[12] = 1, -1  # withheld form 4 versus taps 4
    u = np.zeros(N); u[18:] = 1 / 3  # unit norm, constant over form chain
    epsilon = .001
    graph_results = []
    for alpha in (1., .1, .01, 0.):
        weights = np.where(bridge, alpha, 1.)
        A = B.T @ (weights[:, None] * B) + np.outer(anchor, anchor)
        ev = np.linalg.eigvalsh(A); counts['symmetric_eigendecompositions'] += 1
        close(f'bridge Rayleigh identity alpha={alpha}', u @ A @ u, 7 * alpha / 9)
        assert ev[0] <= 7 * alpha / 9 + 1e-10
        result = {'bridge_weight': alpha, 'min_eigenvalue': float(ev[0]),
                  'rayleigh_upper_bound': 7 * alpha / 9}
        if alpha:
            q = np.linalg.solve(A, B.T @ (weights * increments) + anchor)
            counts['linear_solves'] += 1
            close(f'exact grounding alpha={alpha}', q, truth)
            inv = np.linalg.inv(A); counts['matrix_inversions'] += 1
            R = float(v @ inv @ v)
            e = inv @ v * math.sqrt(2 * epsilon / R)
            close(f'sharp energy alpha={alpha}', .5 * e @ A @ e, epsilon)
            close(f'sharp query bound alpha={alpha}', v @ e, math.sqrt(2 * epsilon * R))
            result.update({'query_resistance': R, 'sharp_error_at_energy_0_001': float(v @ e)})
            if alpha == 1:
                base_A, base_inv, base_R = A.copy(), inv.copy(), R
        else:
            close('ungrounded script null vector', A @ u, np.zeros(N))
            shifted = truth + 20 * u
            residual = B @ shifted - increments
            close('shift preserves remaining constraints', weights * residual, np.zeros(len(edges)))
            result['undetermined_query_shift'] = float(v @ (shifted - truth))
        graph_results.append(result)

    candidates = []
    # Both edges were already permissible observations; no withheld direct label.
    for name, i, j in (('reinforce form-3 grounding', 2, 20),
                        ('reinforce collection-8-to-9 succession', 7, 8)):
        b = np.zeros(N); b[j], b[i] = 1, -1
        w = 1.
        z = base_inv @ b
        predicted = w * (v @ z) ** 2 / (1 + w * b @ z)
        new_inv = np.linalg.inv(base_A + w * np.outer(b, b))
        counts['matrix_inversions'] += 1; counts['candidate_edges_scored'] += 1
        close(name + ' inverse', new_inv, base_inv - w * np.outer(z, z) / (1 + w * b @ z))
        actual = base_R - v @ new_inv @ v
        close(name + ' resistance reduction', actual, predicted)
        assert actual >= -1e-10
        candidates.append({'name': name, 'query_resistance_reduction': float(actual)})

    X = np.array([[1., 0.], [0., .1]])
    residual = np.array([2., 1.])
    scale = np.array([1., 100.])
    growth = []
    for name, features in (('original', X), ('rescaled', X * scale)):
        g = -features.T @ residual
        h = (features ** 2).sum(0)
        gain = g ** 2 / (2 * h)
        direct = []
        for j in range(2):
            w = -g[j] / h[j]
            direct.append(.5 * residual @ residual - .5 * np.sum((residual - w * features[:, j]) ** 2))
        close(name + ' coordinate loss decrease', gain, direct)
        growth.append({'name': name, 'raw_gradient_choice': int(np.abs(g).argmax()),
                       'normalized_choice': int(gain.argmax()), 'gain': gain.tolist()})
    close('rescaling invariant gain', growth[0]['gain'], growth[1]['gain'])
    assert growth[0]['raw_gradient_choice'] != growth[1]['raw_gradient_choice']

    n, m, sigma2 = 16, 16, 1.
    threshold = sigma2 * (n + m) / (n * m)
    bias_results = []
    for bias2 in (0., threshold, 4 * threshold):
        all_mse = sigma2 / (n + m) + (m / (n + m)) ** 2 * bias2
        new_mse = sigma2 / n
        close(f'bias variance difference {bias2}', all_mse - new_mse,
              (m / (n + m)) ** 2 * (bias2 - threshold))
        bias_results.append({'old_bias_squared': bias2, 'retain_mse': all_mse, 'discard_mse': new_mse})

    eig = np.array([0., 1., 2.])
    theta0, target = np.array([3., 2., 1.]), np.array([0., 2., 0.])
    flow = []
    for t in (0., .5, 2.):
        theta = target + np.exp(-eig * t) * (theta0 - target)
        derivative = -eig * np.exp(-eig * t) * (theta0 - target)
        close(f'quadratic ODE t={t}', derivative, -eig * (theta - target))
        close(f'null and compatible preservation t={t}', theta[:2], theta0[:2])
        close(f'quadratic drift t={t}', theta - theta0, (1 - np.exp(-eig * t)) * (target - theta0))
        flow.append({'t': t, 'theta': theta.tolist()})
    eta, curvature, initial, tolerance = .2, 3., 2., .1
    exposure = math.log(initial / tolerance) / (eta * curvature)
    close('consolidation deadline equality', initial * math.exp(-eta * curvature * exposure), tolerance)

    report = {'fixture': {'nodes': N, 'edges': len(edges), 'form_bridges': int(bridge.sum()),
                          'withheld_counts': [4, 6], 'anchor': 'collection 1 = 1'},
              'checks': checks, 'graph': graph_results, 'candidate_edges': candidates,
              'growth': growth, 'bias_threshold_squared': threshold, 'bias_variance': bias_results,
              'quadratic_flow': flow, 'deadline_exposure': exposure,
              'binding_entropy_bits': math.log2(math.factorial(9)),
              'binding_fixed_length_bits': math.ceil(math.log2(math.factorial(9))),
              'counts': counts, 'scope': 'Illustrative algebra; no neural model or empirically fitted mechanism'}
    prefix = root / 'reports/fan-in-memory-theory'
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    info = io.StringIO()
    with contextlib.redirect_stdout(info):
        np.show_config()
    paths = [Path(__file__), root / 'docs/fan-in-memory-theory.md']
    runtime = {'command': '.venv/bin/python -m experiments.fan_in_memory_theory',
               'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
               'hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
               'numpy_build_config': info.getvalue(), 'dtype': 'float64', 'workers': 1,
               'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
               'actual_blas_thread_count': None, 'blas_thread_note': 'Not queried; build and environment recorded',
               'seed_policy': 'Explicit arrays; no random sampling', 'neural_fits': 0,
               'independent_examples': 0, 'optimizer_steps': 0, 'tuning_candidates': 0,
               'analytical_operations': counts, 'checks_passed': len(checks),
               'wall_seconds': time.perf_counter() - start_wall, 'process_cpu_seconds': time.process_time() - start_cpu,
               'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'timing_scope': 'Whole main through provenance; excludes imports and final runtime write; not a speed benchmark'}
    prefix.with_name(prefix.name + '-runtime.json').write_text(json.dumps(runtime, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'checks_passed': len(checks), 'graph': graph_results,
                      'candidates': candidates, 'growth': growth, 'wall_seconds': runtime['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
