"""Analytical scaling check for the inner channel (post hoc; not part of any frozen contract).

1. Reparameterization identity: scaling W, V by (s, t) and activities by (1/s, 1/(st))
   reweights every deeper energy term relative to outer reconstruction.
2. Hidden-module effective prior on c: lambda_c I + (lambda_h/(1+lambda_h)) V^T V.
3. Closed-form completion gain of a two-module scalar chain, checked against a direct solve.
4. For inner-channel diagnostic v1's inner arms: measured signal gain, the share of signal
   error that is a scalar gain loss, the three-module scalar-chain gain from learned top
   singular values, its sensitivity to priors, weights and coupling, and weight-decay cost.

Reads the committed results; the frozen artifacts are not modified.
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
from experiments import clamped_energy_ablation as cea
from experiments import correlated_nuisance as cn
from experiments import inner_channel_diagnostic as icd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/inner-channel-diagnostic-v1-results.json.gz'
PREFIX = ROOT / 'reports/energy-scaling-analysis'
LAM = .05  # Hessian of the 0.025(||h||^2 + ||c||^2) activity priors
ARMS = ('E', 'M', 'S', 'S_prime')


def energy(x, h, c, W, V, T, kappa, weights=(1., 1., LAM, LAM, 1.)):
    """Explicit inner-family energy with term weights (outer error, h error, h prior, c prior, coupling)."""
    a, b, ph, pc, k = weights
    e = 0.
    for m in range(3):
        e += .5 * a * np.sum((x[m] - W[m] @ h[m]) ** 2) + .5 * b * np.sum((h[m] - V[m] @ c[m]) ** 2)
        e += .5 * ph * np.sum(h[m] ** 2) + .5 * pc * np.sum(c[m] ** 2)
    for i in (0, 1):
        e += .5 * k * kappa * np.sum((c[i + 1] - T[i] @ c[i]) ** 2)
    return e


def identity_check(rng):
    worst = 0.
    for _ in range(20):
        W, V = rng.normal(size=(3, 8, 4)), rng.normal(size=(3, 4, 2))
        T = rng.normal(size=(2, 2, 2))
        x, h, c = rng.normal(size=(3, 8)), rng.normal(size=(3, 4)), rng.normal(size=(3, 2))
        s, t, kappa = rng.uniform(.2, 5), rng.uniform(.2, 5), rng.uniform(.1, 5)
        scaled = energy(x, h / s, c / (s * t), s * W, t * V, T, kappa)
        canonical = energy(x, h, c, W, V, T, kappa,
                           (1., 1 / s ** 2, LAM / s ** 2, LAM / (s * t) ** 2, 1 / (s * t) ** 2))
        worst = max(worst, abs(scaled - canonical) / max(1., abs(canonical)))
    return worst


def hidden_prior_check(rng):
    """Schur complement of a hidden module's x, h block onto its c, without coupling, from lcc.build."""
    model = cn.model_for('inner', 2.)
    p = lcc.initialize(model, 5)
    p['W'], p['V'] = rng.normal(size=p['W'].shape), rng.normal(size=p['V'].shape)
    for k in model.keys:
        if k.startswith('T'):
            p[k] = np.zeros_like(p[k])  # coupling contributions removed by zero translations and kappa below
    model0 = lcc.Model('inner', (0., 0., 0.))
    q = lcc.build(model0, {k: p[k] for k in ('W', 'V')})
    worst = 0.
    for m in range(3):
        xh, c = np.concatenate([model0.sl(m, 0), model0.sl(m, 1)]), model0.sl(m, 2)
        schur = q[np.ix_(c, c)] - q[np.ix_(c, xh)] @ np.linalg.solve(q[np.ix_(xh, xh)], q[np.ix_(xh, c)])
        predicted = LAM * np.eye(2) + LAM / (1 + LAM) * p['V'][m].T @ p['V'][m]
        worst = max(worst, float(np.max(abs(schur - predicted))))
    return worst


def two_module_closed_form(wo, vo, wh, vh, tau, kappa, lh=LAM, lc=LAM):
    """Gain = balance x four shrink factors, each 1 when the priors vanish."""
    P = lc + lh * vh ** 2 / (1 + lh)
    keff = kappa * P / (kappa + P)
    Q = lc + keff * tau ** 2
    factors = {'balance': wh * vh * tau / (wo * vo), 'hidden_h_prior': 1 / (1 + lh),
               'hidden_c_coupling_vs_prior': kappa / (kappa + P), 'observed_c_level': vo ** 2 / (vo ** 2 + Q),
               'observed_outer': wo ** 2 / (wo ** 2 + lh + Q / (vo ** 2 + Q))}
    return float(np.prod(list(factors.values()))), factors


def scalar_q(w, v, tau, kappa, n, lh=LAM, lc=LAM):
    Q = np.zeros((3 * n, 3 * n))
    for m in range(n):
        x, h, c = 3 * m, 3 * m + 1, 3 * m + 2
        Q[x, x] += 1
        Q[x, h] -= w[m]
        Q[h, x] -= w[m]
        Q[h, h] += w[m] ** 2 + 1 + lh
        Q[h, c] -= v[m]
        Q[c, h] -= v[m]
        Q[c, c] += v[m] ** 2 + lc
    for a in range(n - 1):
        ca, cb, t = 3 * a + 2, 3 * a + 5, tau[a]
        Q[cb, cb] += kappa
        Q[ca, ca] += kappa * t * t
        Q[cb, ca] -= kappa * t
        Q[ca, cb] -= kappa * t
    return Q


def chain_gain(w, v, tau, kappa, n=3, lh=LAM, lc=LAM):
    """Scalar n-module chain; observed outer activities set to 1; gain of each hidden module's outer unit."""
    Q, out = scalar_q(w, v, tau, kappa, n, lh, lc), []
    for hid in range(n):
        obs = [3 * m for m in range(n) if m != hid]
        free = [i for i in range(3 * n) if i not in obs]
        y = np.linalg.solve(Q[np.ix_(free, free)], -Q[np.ix_(free, obs)] @ np.ones(len(obs)))
        out.append(float(y[free.index(3 * hid)]))
    return out


def closed_form_check(rng):
    worst = 0.
    for _ in range(50):
        wo, vo, wh, vh, tau = rng.uniform(.3, 5, 5)
        kappa = rng.uniform(.1, 10)
        Q = scalar_q([wo, wh], [vo, vh], [tau], kappa, 2)
        free = [1, 2, 3, 4, 5]
        y = np.linalg.solve(Q[np.ix_(free, free)], -Q[np.ix_(free, [0])] @ np.ones(1))
        worst = max(worst, abs(y[free.index(3)] - two_module_closed_form(wo, vo, wh, vh, tau, kappa)[0]))
    return float(worst)


def top(a):
    return float(np.linalg.svd(a, compute_uv=False)[0])


def arm_analysis(source, regime, arm, bname):
    rows = []
    for seed in sorted(source['seeds'], key=int):
        rec = source['seeds'][seed]['arms'][f'{regime}_{arm}_{bname}']
        r = rec['starts'][rec['chosen_start']] if 'starts' in rec else rec
        p = {k: np.array(v) for k, v in r['final_parameters'].items()}
        model, kappa = icd.model_for(arm, rec['budget']), rec['budget'] / 2
        d = cn.fixture(int(seed))[regime]
        x = d['decorrelated']
        pred = cn.predictions(model, p, x, 'exact')
        mu = np.transpose(cea.noiseless(x, d['maps']), (1, 0, 2))
        P = np.einsum('mij,mkj->mik', d['maps'], d['maps'])
        Py = np.einsum('mij,mnj->mni', P, pred)
        g = float(np.sum(Py * mu) / np.sum(mu ** 2))
        err = float(np.mean((Py - mu) ** 2))
        w = [top(p['W'][m]) for m in range(3)]
        v = [top(p['V'][m]) for m in range(3)]
        tau = [top(p[f'T{a}_2']) for a in (0, 1)]
        sens = {'base': chain_gain(w, v, tau, kappa), 'no_h_prior': chain_gain(w, v, tau, kappa, lh=0.),
                'no_c_prior': chain_gain(w, v, tau, kappa, lc=0.), 'no_priors': chain_gain(w, v, tau, kappa, lh=0., lc=0.),
                'W_times_10': chain_gain([10 * a for a in w], v, tau, kappa),
                'kappa_times_100': chain_gain(w, v, tau, 100 * kappa)}
        data = np.concatenate(d['waking'])
        rows.append({'seed': seed, 'measured_signal_gain': g, 'signal_error': err,
                     'scalar_gain_share_of_signal_error': float(np.mean(((g - 1) * mu) ** 2) / err),
                     'top_singular_values': {'W': w, 'V': v, 'T_inner': tau},
                     'chain_gain_by_hidden_module': sens['base'], 'chain_gain': float(np.mean(sens['base'])),
                     'chain_sensitivity': {k: float(np.mean(val)) for k, val in sens.items()},
                     'S_objective': icd.masked_loss(model, p, data.reshape(32, 24), icd.signal_target(data, d)),
                     'decay_penalty': float(.5 * lcc.DECAY * (np.sum(p['W'] ** 2) + np.sum(p['V'] ** 2)))})
    mean = lambda f: float(np.mean([f(r) for r in rows]))
    summary = {'measured_signal_gain': mean(lambda r: r['measured_signal_gain']),
               'chain_gain': mean(lambda r: r['chain_gain']),
               'scalar_gain_share_of_signal_error': mean(lambda r: r['scalar_gain_share_of_signal_error']),
               'chain_sensitivity': {k: mean(lambda r: r['chain_sensitivity'][k]) for k in rows[0]['chain_sensitivity']},
               'mean_top_W': mean(lambda r: np.mean(r['top_singular_values']['W'])),
               'mean_top_V': mean(lambda r: np.mean(r['top_singular_values']['V'])),
               'mean_top_T_inner': mean(lambda r: np.mean(r['top_singular_values']['T_inner'])),
               'S_objective': mean(lambda r: r['S_objective']), 'decay_penalty': mean(lambda r: r['decay_penalty'])}
    return {'by_seed': rows, 'summary': summary}


def main():
    result, runtime = Path(str(PREFIX) + '-results.json'), Path(str(PREFIX) + '-runtime.json')
    if result.exists() or runtime.exists():
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    rng = np.random.default_rng(np.random.SeedSequence([0, 5005]))
    checks = {'reparameterization_identity_max_relative_error': identity_check(rng),
              'hidden_module_effective_prior_max_error': hidden_prior_check(rng),
              'two_module_closed_form_max_error': closed_form_check(rng)}
    assert checks['reparameterization_identity_max_relative_error'] < 1e-12
    assert checks['hidden_module_effective_prior_max_error'] < 1e-12
    assert checks['two_module_closed_form_max_error'] < 1e-12
    # Worked example at the clamped-energy learner's typical values.
    examples = {}
    for kappa in (1., 5.):
        g, f = two_module_closed_form(2.5, 1.41, 2.5, 1.41, 1., kappa)
        examples[f'kappa_{kappa:g}'] = {'gain': g, 'factors': f,
                                        'values': {'w': 2.5, 'v': 1.41, 'tau': 1., 'lambda': LAM}}
    source = json.loads(gzip.decompress(SOURCE.read_bytes()))
    arms = {f'{regime}_{arm}_{bname}': arm_analysis(source, regime, arm, bname)
            for regime in ('weak', 'independent') for arm in ARMS for bname in ('2', '10')}
    science = {'source': str(SOURCE.relative_to(ROOT)), 'checks': checks, 'two_module_examples': examples,
               'arms': arms, 'lambda': LAM,
               'definitions': {
                   'measured_signal_gain': 'regression coefficient of the pattern-span component of the exact '
                                           'completion on mu_sig, decorrelated test set, pooled over hidden modules',
                   'chain_gain': 'three-module scalar chain with each module\'s top singular values of W, V and the '
                                 'inner translations, kappa = b/2, lambda = 0.05; mean over hidden modules',
                   'decay_penalty': '0.5 * 1e-3 * (||W||^2 + ||V||^2)'}}
    result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
    files = (Path(__file__), Path(icd.__file__), Path(cn.__file__), Path(lcc.__file__), SOURCE, result)
    info = {'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'command': ' '.join(sys.argv), 'python': sys.version, 'numpy': np.__version__,
            'os': platform.platform(), 'dtype': 'float64',
            'complexity': 'per arm and seed: three masked exact solves on 24 examples, 18 scalar-chain solves of '
                          'at most 9 dimensions, one masked loss; checks: 20 + 50 random cases',
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'timing_scope': 'whole run including JSON parsing; not a latency benchmark'}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
