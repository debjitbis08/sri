"""Post-hoc analyses for correlated nuisance v1 (not part of the frozen contract or its criteria).

1. Recentred reduced-rank regression. The frozen comparator keeps the ridge
   intercept after projecting the slopes, which shifts every prediction by
   x̄ B (I - V₂V₂ᵀ). Here the intercept is reset so the training-mean prediction
   is preserved. Slopes, and therefore the nuisance gain, are unchanged.
2. Symmetric and antisymmetric nuisance gain. In distribution both observed
   modules carry the same shared latent, so only the sum of the two blocks of
   G_h = A_h [U_o1 ⊕ U_o2] is identified by in-distribution data; the
   difference is seen only after decorrelation. Both parts are reported
   relative to the full oracle's symmetric gain.

Reads the committed results JSON and recomputes from final parameters; the
frozen artifacts are not modified.
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
from scipy.linalg import block_diag
from experiments import correlated_nuisance as cn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/correlated-nuisance-v1-results.json'
PREFIX = ROOT / 'reports/correlated-nuisance-v1-posthoc'


def recentred(train_set):
    ridge, reduced = cn.linear_fit(train_set), cn.linear_fit(train_set, 2)
    out = []
    for m, (wf, w) in enumerate(zip(ridge, reduced)):
        other = [i for i in range(3) if i != m]
        xbar = train_set[:, other].reshape(len(train_set), 16).mean(axis=0)
        w = w.copy()
        w[16] = wf[16] + xbar @ (wf[:16] - w[:16])
        out.append(w)
    return out


def split_gain(A, d):
    sym = anti = 0.
    for h in range(3):
        o = [m for m in range(3) if m != h]
        G = A[h] @ block_diag(d['U'][o[0]], d['U'][o[1]])
        sym += float(np.sum(((G[:, :2] + G[:, 2:]) / 2) ** 2))
        anti += float(np.sum(((G[:, :2] - G[:, 2:]) / 2) ** 2))
    star = 3 * 2 * cn.oracle_scale(d) ** 2  # oracle: (G1 + G2)/2 = k U_h, antisymmetric part 0
    return {'symmetric_relative_to_oracle': float(np.sqrt(sym / star)),
            'antisymmetric_relative_to_oracle': float(np.sqrt(anti / star))}


def main():
    result, runtime = Path(str(PREFIX) + '-results.json'), Path(str(PREFIX) + '-runtime.json')
    if result.exists() or runtime.exists():
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    source = json.loads(SOURCE.read_text())
    seeds = sorted(source['seeds'], key=int)
    rrr, gains = {}, {}
    for seed in seeds:
        data = cn.fixture(int(seed))
        for regime, d in data.items():
            fits = recentred(np.concatenate(d['waking']))
            frozen_fits = cn.linear_fit(np.concatenate(d['waking']), 2)
            assert all(np.array_equal(a[:16], b[:16]) for a, b in zip(fits, frozen_fits))
            s = cn.score(lambda x: cn.linear_predict(fits, x), [w[:16].T for w in fits], d)
            rrr.setdefault(regime, []).append({w: s[w]['combined']['avoidable_mse'] for w in cn.SETS})
            if d['s'] == 0:
                continue
            for name, record in source['seeds'][seed]['conditions'].items():
                if record['regime'] != regime:
                    continue
                p = {k: np.array(v) for k, v in record['final_parameters'].items()}
                model = cn.model_for(record['family'], record['budget'])
                gains.setdefault(name, []).append(split_gain(cn.completion_maps(model, p, 'exact'), d))
            for comp, rank in (('ridge', None), ('reduced_rank_2', 2)):
                f = cn.linear_fit(np.concatenate(d['waking']), rank)
                gains.setdefault(f'{regime}_{comp}', []).append(split_gain([w[:16].T for w in f], d))
    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
    science = {
        'source': str(SOURCE.relative_to(ROOT)), 'seeds': seeds,
        'recentred_reduced_rank': {r: {'by_seed': rows, 'mean_X_in': mean(rows, 'in_distribution'),
                                       'mean_X_dec': mean(rows, 'decorrelated')} for r, rows in rrr.items()},
        'gain_split': {n: {'by_seed': rows, 'mean_symmetric': mean(rows, 'symmetric_relative_to_oracle'),
                           'mean_antisymmetric': mean(rows, 'antisymmetric_relative_to_oracle')}
                       for n, rows in gains.items()},
        'definitions': {'recentred_intercept': 'ridge intercept + x̄ (B_ridge - B_reduced); slopes as frozen',
                        'gain_split': 'G = A_h [U_o1 ⊕ U_o2] split into (G1+G2)/2 and (G1-G2)/2, Frobenius norms '
                                      'pooled over hidden modules, relative to the full oracle (k U_h, 0); exact equilibrium'}}
    result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
    files = (Path(__file__), Path(cn.__file__), SOURCE, result)
    info = {'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'command': ' '.join(sys.argv), 'python': sys.version, 'numpy': np.__version__,
            'os': platform.platform(), 'dtype': 'float64',
            'complexity': 'per seed and regime: two ridge fits and one 8-column SVD; per condition one completion '
                          'map (three O(S^3) solves) and 8x16x4 products',
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'timing_scope': 'whole run including JSON parsing; not a latency benchmark'}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
