"""Fixed-array checks for docs/decaying-readers-theory.md; no neural training."""
import hashlib
import contextlib
import io
import json
import math
import os
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time
import warnings

import numpy as np


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[1]
    checks = []

    def close(name, actual, expected):
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-10), (name, actual, expected)
        checks.append({'name': name, 'max_error': float(np.max(np.abs(np.asarray(actual) - expected)))})

    # Conditional unrolling of arbitrary optimizer increments.
    r = 2 ** (-1 / 5)
    updates = np.array([.1, -.3, .7, -.05])
    u = .4
    for delta in updates:
        u = r * (u + delta)
    close('increment filter', u, r ** 4 * .4 + sum(r ** (4 - t) * d for t, d in enumerate(updates)))
    close('constant update fixed point', r * (r * .2 / (1 - r) + .2), r * .2 / (1 - r))
    eta, grad, old = .03, .8, 1.2
    penalty = (1 - r) / (r * eta)
    close('SGD penalty equivalence', r * (old - eta * grad), old - r * eta * (grad + penalty * old))

    # Actual two-layer downstream ReLU scaling symmetry on explicit features.
    hs, hm = np.array([.4, 1.1]), np.array([.7, .2])
    vs = np.array([[.2, -.1], [1., .5]])
    vm = np.array([[.7, .9], [-.3, .8]])
    bias, out = np.array([.2, -.1]), np.array([1.3, -.4])
    reference = out @ np.maximum(vs @ hs + vm @ hm + bias, 0)
    for c in (.1, .001):
        score = (out / c) @ np.maximum(c * vs @ hs + c * vm @ hm + c * bias, 0)
        close(f'ReLU rescaling c={c}', score, reference)

    # Pure decay bound for one memory unit with nonzero baseline features.
    z0, dz, outgoing = np.array([.5, -.2]), np.array([.3, .6]), np.array([[.7, -.4]])
    for attenuation in (.5, .0625, 1 / 65536):
        contribution = attenuation * outgoing @ np.maximum(z0 + attenuation * dz, 0)
        bound = attenuation * np.linalg.norm(outgoing, 2) * (np.linalg.norm(np.maximum(z0, 0)) + attenuation * np.linalg.norm(dz))
        assert np.linalg.norm(contribution) <= bound + 1e-12
        checks.append({'name': f'functional bound r^t={attenuation}', 'slack': float(bound - np.linalg.norm(contribution))})

    a, b, eta, y, steps = 1., 4., .01, 1., 6000
    v = np.array([a, b])
    equilibrium = np.array([y / a, 0.])
    results = []
    for half_life in (2000, 500, 125):
        kappa = math.log(2) / half_life
        r = math.exp(-kappa)
        M = eta * np.outer(v, v) + np.diag([0., kappa])
        eigenvalues = np.linalg.eigvalsh(M)
        trace = eta * (a * a + b * b) + kappa
        det = eta * a * a * kappa
        slow = 2 * det / (trace + math.sqrt(trace * trace - 4 * det))
        close(f'flow eigenvalues H={half_life}', eigenvalues, [slow, trace - slow])
        D = np.diag([1., r])
        Q = D @ (np.eye(2) - eta * np.outer(v, v))
        drive = D @ (eta * v * y)
        close(f'discrete equilibrium H={half_life}', np.linalg.solve(np.eye(2) - Q, drive), equilibrium)
        radius = max(abs(np.linalg.eigvals(Q)))
        assert radius < 1
        state = np.zeros(2)
        for _ in range(steps):
            state = D @ (state - eta * v * (v @ state - y))
        direct = equilibrium + np.linalg.matrix_power(Q, steps) @ (-equilibrium)
        close(f'discrete recurrence H={half_life}', state, direct)
        results.append({'H': half_life, 'parameter_attenuation_after_2000': math.exp(-kappa * 2000),
                        'flow_handoff_half_life': math.log(2) / slow,
                        'weak_decay_approximation': half_life * (1 + b * b / (a * a)),
                        'discrete_spectral_radius': float(radius),
                        'slow_contribution_at_6000': float(a * state[0]),
                        'fast_contribution_at_6000': float(b * state[1]),
                        'prediction_at_6000': float(v @ state)})

    f = lambda s, m: max(s + m - 1, 0)
    interaction = f(.75, .75) - f(.75, 0) - f(0, .75) + f(0, 0)
    close('nonadditive ablation', interaction, .5)
    # Two equally likely teacher outputs at the same student-visible scalar.
    teacher = np.array([0., 2.])
    for student in (0., 1., 2.):
        close(f'conditional projection student={student}', np.mean((teacher - student) ** 2),
              np.var(teacher) + (teacher.mean() - student) ** 2)

    report = {'scope': 'Conditional mathematics, not a trained-network prediction', 'checks': checks,
              'two_path_fixture': {'a': a, 'b': b, 'eta': eta, 'y': y, 'steps': steps},
              'half_lives': results, 'ablation_interaction': interaction,
              'irreducible_distillation_mse': float(np.var(teacher)),
              'decayed_parameters': {'g': 1680, 'P': 1552}, 'saved_start_float32_bytes': 4736}
    prefix = root / 'reports/decaying-readers-theory'
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    paths = [Path(__file__), root / 'docs/decaying-readers-theory.md', root / 'docs/limited-memory-proposal.md']
    backend = io.StringIO()
    with contextlib.redirect_stdout(backend), warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        np.show_config()
    runtime = {'command': '.venv/bin/python -m experiments.decaying_readers_theory',
               'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
               'hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
               'dtype': 'float64', 'seed_policy': 'No random sampling', 'workers': 1,
               'numpy_build_config': backend.getvalue(),
               'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
               'backend_threads': None, 'backend_note': 'Not queried; matrices are at most 2 by 2',
               'neural_fits': 0, 'tuning_candidates': 0, 'independent_training_examples': 0,
               'analytical_recurrence_steps': 3 * steps, 'two_by_two_linear_solves': 3,
               'eigendecompositions': 6, 'checks_passed': len(checks),
               'wall_seconds': time.perf_counter() - wall, 'process_cpu_seconds': time.process_time() - cpu,
               'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'timing_scope': 'Whole main excluding imports and final runtime write; not a speed benchmark'}
    prefix.with_name(prefix.name + '-runtime.json').write_text(json.dumps(runtime, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'checks_passed': len(checks), 'half_lives': results, 'wall_seconds': runtime['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
