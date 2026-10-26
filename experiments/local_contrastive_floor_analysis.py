"""Post-hoc noise-floor analysis of local contrastive consolidation v1.

Recomputes each committed final parameter set's 32-step whole-module predictions
and scores them three ways: observed MSE, MSE minus the oracle floor on the same
sample, and avoidable MSE against the noiseless conditional mean. Reads the
committed results JSON only; the frozen v1 artifacts are not modified.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
from experiments import local_consolidation as base
from experiments import local_contrastive_consolidation as lcc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/local-contrastive-consolidation-v1-results.json'
PREFIX = ROOT / 'reports/local-contrastive-consolidation-v1-floor-analysis'
NEAR_ZERO = 1e-5  # avoidable MSE per coordinate below this is treated as zero


def predictions(model, p, test):
    """32-step whole-module predictions, shape (3 masks, n, 8), as in lcc.quick_rmse."""
    q, flat, out = lcc.build(model, p), test.reshape(len(test), 24), []
    for m in range(3):
        M = model.module_mask(m)
        y, _ = lcc.settle(q, M, flat)
        out.append(y[:, M[3]])
    return np.stack(out)


def ridge_predictions(train, test):
    """Same fit as base.direct_predictor, returning predictions instead of errors."""
    out = []
    for m in range(3):
        other = [i for i in range(3) if i != m]
        design = np.column_stack((train[:, other].reshape(len(train), 16), np.ones(len(train))))
        test_design = np.column_stack((test[:, other].reshape(len(test), 16), np.ones(len(test))))
        weights = np.linalg.solve(design.T @ design + .1 * np.eye(17), design.T @ train[:, m])
        out.append(test_design @ weights)
    return np.stack(out)


def score(pred, test, mean):
    """pred, mean: (3, n, 8); test: (n, 3, 8). Mask m hides module m."""
    target = np.transpose(test, (1, 0, 2))
    observed = np.mean((pred - target) ** 2)
    floor = np.mean((target - mean) ** 2)
    avoidable = np.mean((pred - mean) ** 2)
    cross = -2 * np.mean((pred - mean) * (target - mean))
    assert abs(observed - floor - avoidable - cross) < 1e-12
    return {'observed_rmse': float(np.sqrt(observed)), 'observed_mse': float(observed),
            'floor_mse': float(floor), 'noise_subtracted_mse': float(observed - floor),
            'avoidable_mse': float(avoidable), 'cross_term': float(cross)}


def summary(rows, reference):
    out = {}
    for name, values in rows.items():
        out[name] = {}
        for key in ('observed_rmse', 'noise_subtracted_mse', 'avoidable_mse', 'cross_term'):
            v = np.array([r[key] for r in values])
            out[name][key] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}
        a = np.array([r['avoidable_mse'] for r in values])
        ref = np.array([r['avoidable_mse'] for r in rows[reference]])
        out[name]['avoidable_ratio_to_' + reference] = (None if ref.mean() < NEAR_ZERO
                                                        else float(a.mean() / ref.mean()))
        out[name]['seeds_below_noise_subtracted_zero'] = int(sum(r['noise_subtracted_mse'] < 0 for r in values))
    return out


def main():
    result, runtime = Path(str(PREFIX) + '-results.json'), Path(str(PREFIX) + '-runtime.json')
    if result.exists() or runtime.exists():
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    source = json.loads(SOURCE.read_text())
    rows, per_seed = {}, {}
    for seed, row in sorted(source['seeds'].items(), key=lambda kv: int(kv[0])):
        seed = int(seed)
        waking, test, _ = base.fixture(seed)
        maps = lcc.private_maps(seed)
        projector = np.einsum('mij,mkj->mik', maps, maps)
        # Nuisance is orthogonal to both pattern columns, so projecting onto the
        # pattern span recovers the noiseless conditional mean exactly.
        mean = np.transpose(np.einsum('mij,nmj->nmi', projector, test), (1, 0, 2))
        per_seed[str(seed)] = {}
        uniform = lcc.Model('uniform')
        entries = [('1_frozen', uniform, lcc.initialize(uniform, seed), row['frozen'])]
        for name, arm in row['arms'].items():
            model = lcc.Model(arm['family'])
            p = {k: np.array(v) for k, v in arm['final_parameters'].items()}
            entries.append((name, model, p, arm['evaluations'][-1]))
        for name, model, p, recorded in entries:
            s = score(predictions(model, p, test), test, mean)
            assert abs(s['observed_rmse'] - recorded['whole_module_rmse_32_step']['combined']) < 1e-12, name
            rows.setdefault(name, []).append(s)
            per_seed[str(seed)][name] = s
        s = score(ridge_predictions(np.concatenate(waking), test), test, mean)
        assert abs(s['observed_rmse'] - row['ridge'][-1]['rmse']) < 1e-12
        rows.setdefault('10_ridge', []).append(s)
        per_seed[str(seed)]['10_ridge'] = s
    science = {'source': str(SOURCE.relative_to(ROOT)), 'near_zero_avoidable_mse': NEAR_ZERO,
               'definitions': {'floor_mse': 'mean squared nuisance of the hidden module on this test sample',
                               'noise_subtracted_mse': 'observed minus floor; can be negative on a finite sample',
                               'avoidable_mse': 'mean squared error against the noiseless conditional mean (evaluator only)',
                               'cross_term': 'observed - floor - avoidable = -2 mean((pred - mean)(target - mean))'},
               'summary': summary(rows, '2a_clamped_energy_C'), 'seeds': per_seed}
    result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
    files = (Path(__file__), Path(lcc.__file__), Path(base.__file__), SOURCE, result)
    info = {'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'command': ' '.join(sys.argv), 'python': sys.version, 'numpy': np.__version__,
            'os': platform.platform(), 'dtype': 'float64',
            'thread_environment': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'counts': {'settles': lcc.COUNTS.get('settles', 0), 'implicit_factorizations': lcc.COUNTS.get('implicit_factorizations', 0),
                       'parameter_sets_scored': sum(len(v) for v in rows.values())},
            'complexity': 'per parameter set: 3 masks x one O(S^3) factorization plus 32 O(n S^2) steps, S <= 42, n = 24',
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'timing_scope': 'whole run including JSON parsing; not a latency benchmark'}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(json.dumps(science['summary'], indent=1))


if __name__ == '__main__':
    main()
