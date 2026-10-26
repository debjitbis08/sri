"""Full-model activity-prior intervention at fixed learned parameters (post hoc; no frozen contract).

For inner-channel diagnostic v1's final parameters, rebuild Q with the middle-layer
and inner-layer activity priors changed separately or together (positive values only),
and measure exact-equilibrium signal gain, signal error, avoidable errors, nuisance
gain and conditioning. Parameters are never retrained. Reads the committed results;
the frozen artifacts are not modified.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
from experiments import local_contrastive_consolidation as lcc
from experiments import correlated_nuisance as cn
from experiments import inner_channel_diagnostic as icd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/inner-channel-diagnostic-v1-results.json.gz'
PREFIX = ROOT / 'reports/prior-intervention-analysis'
BASE = .05  # Hessian of the 0.025 activity priors in lcc.build
VALUES = (.05, .02, .01, .005, .002, .001)
VARIATIONS = ('middle', 'inner', 'both')
ARMS = ('E', 'M', 'S', 'S_prime', 'U')


def prior_q(model, p, middle, inner):
    """lcc.build with the middle (h) and inner (c) activity-prior Hessians replaced."""
    q = lcc.build(model, p)
    for m in range(3):
        h, c = model.sl(m, 1), model.sl(m, 2)
        q[np.ix_(h, h)] += (middle - BASE) * np.eye(len(h))
        q[np.ix_(c, c)] += (inner - BASE) * np.eye(len(c))
    return q


def predict(model, q, x):
    flat = x.reshape(len(x), 24)
    return np.stack([lcc.exact_free(q, model.module_mask(h), flat)[:, model.module_mask(h)[3]] for h in range(3)])


def maps(model, q):
    out = []
    for h in range(3):
        M = model.module_mask(h)
        basis = np.zeros((16, 24))
        basis[np.arange(16), M[1]] = 1
        out.append(lcc.exact_free(q, M, basis)[:, M[3]].T)
    return out


def conditioning(model, q):
    masked = max(float(np.linalg.cond(q[np.ix_(model.module_mask(h)[2], model.module_mask(h)[2])])) for h in range(3))
    return {'max_masked_free_block_condition': masked, 'min_eigenvalue_Q': float(np.linalg.eigvalsh(q)[0])}


def measure(model, q, d):
    out = {}
    P = np.einsum('mij,mkj->mik', d['maps'], d['maps'])
    for name in cn.SETS:
        x = d['test'] if name == 'in_distribution' else d['decorrelated']
        sig, full = cn.targets(x, d)
        pred = predict(model, q, x)
        m = cn.set_metrics(pred, x, full if name == 'in_distribution' else sig, sig, d['maps'], {'combined': np.arange(len(x))})
        out[f'X_{name}'] = m['combined']['avoidable_mse']
        out[f'signal_error_{name}'] = m['combined']['signal_error_mse']
        mu = np.transpose(sig, (1, 0, 2))
        Py = np.einsum('mij,mnj->mni', P, pred)
        out[f'signal_gain_{name}'] = float(np.sum(Py * mu) / np.sum(mu ** 2))
    out['gamma'] = cn.gain(maps(model, q), d)['gamma']
    out.update(conditioning(model, q))
    return out


def settings():
    rows = [('base', BASE, BASE)]
    for v in VALUES[1:]:
        rows += [('middle', v, BASE), ('inner', BASE, v), ('both', v, v)]
    return rows


def main():
    result, runtime = Path(str(PREFIX) + '-results.json'), Path(str(PREFIX) + '-runtime.json')
    if result.exists() or runtime.exists():
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    source = json.loads(gzip.decompress(SOURCE.read_bytes()))
    seeds = sorted(source['seeds'], key=int)
    worst_base = 0.
    arms = {}
    for regime in ('weak', 'independent'):
        for arm in ARMS:
            for bname in ('2', '10'):
                name = f'{regime}_{arm}_{bname}'
                rows = {}
                for seed in seeds:
                    rec = source['seeds'][seed]['arms'][name]
                    r = rec['starts'][rec['chosen_start']] if 'starts' in rec else rec
                    p = {k: np.array(v) for k, v in r['final_parameters'].items()}
                    model = icd.model_for(arm, rec['budget'])
                    d = cn.fixture(int(seed))[regime]
                    recorded = r['evaluations'][-1]['nuisance']['exact']
                    for label, mid, inn in settings():
                        m = measure(model, prior_q(model, p, mid, inn), d)
                        if label == 'base':  # unchanged priors reproduce the diagnostic's recorded values
                            for w in cn.SETS:
                                worst_base = max(worst_base, abs(m[f'X_{w}'] - recorded[w]['combined']['avoidable_mse']))
                        rows.setdefault(f'{label}_{mid:g}_{inn:g}', []).append(m)
                arms[name] = {k: {'by_seed': v, 'mean': {f: (None if v[0][f] is None else float(np.mean([x[f] for x in v])))
                                                         for f in v[0]}}
                              for k, v in rows.items()}
    assert worst_base < 1e-12
    science = {'source': str(SOURCE.relative_to(ROOT)), 'seeds': seeds, 'values': list(VALUES), 'base_prior': BASE,
               'check_base_reproduces_recorded_X': worst_base, 'arms': arms,
               'definitions': {'setting_key': 'variation_middlePrior_innerPrior (Hessian values; 0.05 is the trained model)',
                               'signal_gain': 'regression coefficient of the pattern-span component of the exact completion '
                                              'on mu_sig, pooled over hidden modules',
                               'parameters': 'final learned parameters, never retrained'}}
    result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
    files = (Path(__file__), Path(icd.__file__), Path(cn.__file__), Path(lcc.__file__), SOURCE, result)
    info = {'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'command': ' '.join(sys.argv), 'python': sys.version, 'numpy': np.__version__,
            'os': platform.platform(), 'dtype': 'float64',
            'complexity': 'per arm, seed and prior setting: one Q build, six masked exact solves on 24 examples, three '
                          '16-column map solves, one 42x42 eigendecomposition and three 26x26 condition numbers',
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'timing_scope': 'whole run including JSON parsing; not a latency benchmark'}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
