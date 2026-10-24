"""Run only the declared analytical/spectral single-module learning preflight.

Usage: .venv/bin/python -m experiments.learning_preflight [--out-dir reports]
No neural models or additional dependencies.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from src.learning_preflight import (BUDGETS, D, M, METHODS, NOISE, REGIMES,
    SEED, covariance_error_expectation, cross_covariance, fixed_averaging,
    global_spectral, local_spectral, population_local, run_gates,
    subspace_loss, task_world, training_stream)


WORLDS, STREAMS = 64, 8
ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-learning-preflight'
SOURCES = ('docs/single-module-learning-preflight.md', 'src/learning_preflight.py',
           'experiments/learning_preflight.py', 'tests/test_learning_preflight.py', 'requirements.txt')


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError('nonfinite report value')
        return float(f'{value:.12g}')
    if isinstance(value, np.integer):
        return int(value)
    return value


def dump(path, value):
    path.write_text(json.dumps(canonical(value), indent=2, sort_keys=True)+'\n')


def stats(world_means):
    mean = float(np.mean(world_means))
    se = float(np.std(world_means, ddof=1)/np.sqrt(len(world_means)))
    return dict(mean=mean, world_mean_range=[float(np.min(world_means)), float(np.max(world_means))],
                standard_error=se, approximate_95_interval=[mean-1.96*se, mean+1.96*se])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gates = run_gates()
    manifest = dict(scope='analytical equivalence and spectral development preflight only',
                    worlds=WORLDS, streams_per_world=STREAMS, pair_budgets=BUDGETS,
                    nuisance_variances=NOISE, regimes=REGIMES, methods=METHODS, seed=SEED,
                    hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     scipy=scipy.__version__, blas_threads=1))
    # Record specification/code before generating finite-sample training streams.
    dump(args.out_dir/f'{PREFIX}-manifest.json', manifest)
    shape = (WORLDS, STREAMS, len(NOISE), len(REGIMES), len(BUDGETS), len(METHODS))
    losses = np.zeros(shape)
    covariance_errors = np.zeros((WORLDS, STREAMS, len(NOISE), len(BUDGETS), 2))
    floors, gaps = np.zeros((WORLDS, len(REGIMES))), np.zeros((WORLDS, len(REGIMES), M))
    rotation_error = 0.
    floor_violation = 0.
    fixed = fixed_averaging()
    for w in range(WORLDS):
        truth, rotations = task_world(w)
        truths = [rotation@truth for rotation in rotations]
        for ri, u in enumerate(truths):
            floors[w, ri], gaps[w, ri] = population_local(u)
        for stream in range(STREAMS):
            coeff, e1, e2 = training_stream(w, stream)
            signal = coeff@truth.T
            for ni, nu in enumerate(NOISE):
                first, second = signal+np.sqrt(nu)*e1, signal+np.sqrt(nu)*e2
                for bi, budget in enumerate(BUDGETS):
                    matrix = cross_covariance(first, second, budget)
                    difference = matrix-truth@truth.T
                    covariance_errors[w, stream, ni, bi, 0] = np.sum(difference*difference)
                    covariance_errors[w, stream, ni, bi, 1] = sum(
                        np.sum(difference[i:i+8, i:i+8]**2) for i in range(0, D, 8))
                    reference_loss = subspace_loss(global_spectral(matrix), truth)
                    for ri, (rotation, u) in enumerate(zip(rotations, truths)):
                        rotated = rotation@matrix@rotation.T
                        estimates = (global_spectral(rotated), local_spectral(rotated), fixed)
                        values = [subspace_loss(v, u) for v in estimates]
                        losses[w, stream, ni, ri, bi] = values
                        rotation_error = max(rotation_error, abs(values[0]-reference_loss))
                        floor_violation = max(floor_violation, floors[w, ri]-values[1])
        if (w+1) % 16 == 0:
            print(f'Completed {w+1}/{WORLDS} task worlds', flush=True)
    assert rotation_error < 1e-9, rotation_error
    assert floor_violation < 1e-9, floor_violation
    gates.update(global_joint_rotation_loss_max_error=rotation_error,
                 local_estimator_population_floor_max_violation=floor_violation)
    means = losses.mean(axis=1)
    covariance_means = covariance_errors.mean(axis=1)
    summary, population, thresholds, covariance = [], {}, [], []
    for ri, regime in enumerate(REGIMES):
        population[regime] = dict(local_floor=stats(floors[:, ri]), global_floor=0.,
                                  local_gap_min=float(gaps[:, ri].min()),
                                  local_gap_mean=float(gaps[:, ri].mean()))
    for ni, nu in enumerate(NOISE):
        for ri, regime in enumerate(REGIMES):
            for bi, budget in enumerate(BUDGETS):
                values = means[:, ni, ri, bi]
                metrics = {method: stats(values[:, mi]) for mi, method in enumerate(METHODS)}
                summary.append(dict(regime=regime, nuisance_variance=nu, pairs=budget,
                                    loss=metrics, local_minus_global=stats(values[:, 1]-values[:, 0]),
                                    local_excess_over_population_floor=stats(values[:, 1]-floors[:, ri]),
                                    expected_new_input_relative_mse={method: {
                                        'same_noise': float(values[:, mi].mean()+nu),
                                        'four_times_noise': float(values[:, mi].mean()+4*nu)}
                                        for mi, method in enumerate(METHODS)}))
            for mi, method in enumerate(METHODS):
                curve = means[:, ni, ri, :, mi].mean(axis=0)
                crossing = next((budget for budget, loss in zip(BUDGETS, curve) if loss <= .1), None)
                thresholds.append(dict(regime=regime, nuisance_variance=nu, method=method,
                                       first_budget_mean_loss_at_most_0_1=crossing,
                                       status='reached' if crossing is not None else 'not reached by 256'))
        for bi, budget in enumerate(BUDGETS):
            covariance.append(dict(nuisance_variance=nu, pairs=budget,
                                   global_empirical=stats(covariance_means[:, ni, bi, 0]),
                                   global_exact=(20+132*nu+528*nu**2)/budget,
                                   local_blocks_empirical=stats(covariance_means[:, ni, bi, 1]),
                                   local_blocks_exact=(8+36*nu+144*nu**2)/budget))
    result = dict(completion='complete', scope=manifest['scope'], gates=gates,
                  manifest_sha256=hashlib.sha256((args.out_dir/f'{PREFIX}-manifest.json').read_bytes()).hexdigest(),
                  population=population, learning_curves=summary, descriptive_thresholds=thresholds,
                  covariance_moment_checks=covariance,
                  world_means=dict(axis_order=['world', 'nuisance_variance', 'regime', 'pair_budget', 'method'],
                                   nuisance_variances=NOISE, regimes=REGIMES, pair_budgets=BUDGETS,
                                   methods=METHODS, values=means, local_floors=floors,
                                   local_gaps=gaps),
                  interpretation_limits=['Development data, not neural confirmation.',
                    'Independent replication unit is the world; eight streams averaged within each world.',
                    'All test-risk predictions integrate independent new observations exactly.',
                    'No neural models trained; no retained-context learning rule or nonlinear candidate implemented.'])
    dump(args.out_dir/f'{PREFIX}-results.json', result)
    dump(args.out_dir/f'{PREFIX}-runtime.json', dict(elapsed_seconds=time.perf_counter()-started))
    print('Completed spectral preflight; no neural suite was run.')
    for regime, row in population.items():
        print(f'{regime}: population local floor {row["local_floor"]["mean"]:.5f}')
    for row in summary:
        if row['pairs'] in (16, 64, 256):
            print(row['regime'], row['nuisance_variance'], row['pairs'],
                  {method: round(value['mean'], 4) for method, value in row['loss'].items()})


if __name__ == '__main__':
    main()
