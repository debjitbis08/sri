"""Layered module v1 (Stage 1): depth response of one disk module, linear versus double-well potentials.

Run `.venv/bin/python -m experiments.layered_module [--out-dir DIR] [--manifest-only]`.
"""
import os

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):      # before numpy: one BLAS thread
    os.environ[_v] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

import numpy as np
import scipy

from experiments.mechanism_learning import canonical, completion_status, dump, sha
from src.layered_module import (
    CERT_EIG, CERT_GRAD, LAMBDA, LBFGS_OPTIONS, N1, POLISH_GRAD, Disk, Module, Potential, layer_variance,
    min_mahalanobis, nearest_mean, nuisance_unit, ring_sizes, rms_distance, signal, within_class_variance,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'layered-module-v1'
CONTRACT = 'docs/layered-module-v1-contract.md'
DEPTHS = (2, 3, 4, 6)
GAMMAS = (1, 2)
POTENTIALS = (('linear', 0), ('linear', 1), ('double_well', 0.5), ('double_well', 1), ('double_well', 2))
ALPHAS = (0.5, 1)
FAMILIES = (('F-low', 1), ('F-high', 8))
SIGMAS = (0.03, 0.1, 0.3)
SIGMA_PROFILE = 0.1
CLASSES, DRAWS, TRAIN = 4, 50, 25
SEED_NUISANCE, SEED_READOUT, SEED_STARTS, SEED_CHECK = 20261024, 20261025, 20261026, 20261027
N_STARTS, MERGE_TOL = 16, 1e-6
T2_MAX, T3_MIN = 0.25, 0.9
RULES = [
    'graph: rings n_l = max(4, floor(32 (L-l+1)/L + 1/2)); lateral ring edges weight gamma^(l-1); radial edges '
    'weight 1 by exact nearest-angle rule (ties lowest index) plus back-fill; centre joined to ring L with weight 1',
    'energy: 1/2 u^T K u + sum V(u) + 1/2 lambda sum_ring1 (u - h)^2, lambda = 1; V linear mu u^2/2 or '
    'double well beta (u^2-1)^2/4, on every unit',
    'settling: L-BFGS-B from u = 0 with declared options, then Newton polish (<= 20 steps, backtracking 1/2 up to 30 '
    'halvings; accept if E lowers, or E rises <= 1e-12 max(1,|E|) with max|grad| strictly lower) toward '
    'max|grad| < 1e-12; certified iff max|grad| < 1e-8 and min Hessian eigenvalue > 1e-8',
    'drives: h = cos(n theta + pi c / 2) + nuisance; F-low n = 1, F-high n = 8; nuisance = orthonormal real Fourier '
    'basis (DC, cos/sin 1..15, Nyquist) with N(0,1) coefficients from default_rng(20261024), shape (2,4,50,32), '
    'rescaled per draw to RMS alpha/sqrt(2)',
    'halves: draws 0-24 train, 25-49 test; readout noise sigma0 * eps, eps from default_rng([20261025, L, l, family])',
    'measures: S population variance over all certified drives; P within-class variance averaged over classes; '
    'nearest-class-mean decoding (strict: unsettled test drives wrong, out of 100); Mahalanobis on training half '
    'with pooled covariance (dof N_train - 4) + sigma0^2 I',
    'profile at sigma0 = 0.1 in each family: T1 S(1) >= S(l) all l; T2 P(L)/P(1) <= 0.25; T3 strict D(L) >= 0.9; '
    'eligible only if all 400 canonical settlings certified',
    'multistability: 8 probes (draw 0 of each class and family), 16 starts N(0,1) from default_rng([20261026, L, s]), '
    'greedy merge of certified states at RMS < 1e-6',
    'superposition: pairs i = 0..7, h1 = F-low class i mod 4 draw 25+i, h2 = F-high class (i+1) mod 4 draw 25+i; '
    'defect null if denominator < 1e-12 or any state unsettled',
    'candidate depth: double well beta = 1, gamma = 1, alpha = 1, smallest L meeting the profile',
    'reported floats rounded to 10 significant digits',
    'BLAS pinned to one thread (OPENBLAS/OMP/MKL_NUM_THREADS=1) before numpy import',
]


def manifest():
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                requirements_sha256=sha('requirements.txt'), numpy=np.__version__, scipy=scipy.__version__, blas_threads=1,
                n1=N1, lam=LAMBDA, depths=list(DEPTHS), ring_sizes={str(L): ring_sizes(L) for L in DEPTHS},
                gammas=list(GAMMAS), potentials=[list(p) for p in POTENTIALS], alphas=list(ALPHAS),
                families={k: v for k, v in FAMILIES}, sigmas=list(SIGMAS), sigma_profile=SIGMA_PROFILE,
                classes=CLASSES, draws=DRAWS, train_draws=TRAIN,
                seeds=dict(nuisance=SEED_NUISANCE, readout=SEED_READOUT, starts=SEED_STARTS, checks=SEED_CHECK),
                lbfgs_options=LBFGS_OPTIONS, certification=dict(grad=CERT_GRAD, min_eig=CERT_EIG),
                polish_grad=POLISH_GRAD, starts=N_STARTS, merge_tol=MERGE_TOL,
                thresholds=dict(T2_max=T2_MAX, T3_min=T3_MIN), rules=RULES)
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    m = manifest()
    path = out / f'{PREFIX}-manifest.json'
    frozen = ROOT / 'reports' / f'{PREFIX}-manifest.json'
    ref = path if path.exists() else frozen if frozen.exists() else None
    if ref is not None and json.loads(ref.read_text()) != m:
        sys.exit('manifest differs from the frozen file; refusing to run')
    if not path.exists():
        dump(path, m)
    return m


def rnd(x):
    """Round floats to 10 significant digits and convert numpy scalars, recursively."""
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return float(f'{x:.10g}')
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    return x


def drives(nuis, fam, n_freq, alpha):
    """Drive array (class, draw, 32) for one family."""
    sig = np.array([signal(n_freq, c) for c in range(CLASSES)])
    return sig[:, None, :] + nuis[fam] * (alpha / np.sqrt(2))


def settle_all(mod, H):
    states = np.zeros(H.shape[:2] + (mod.disk.n,))
    ok = np.zeros(H.shape[:2], dtype=bool)
    worst = dict(grad=0.0, min_eig=np.inf)
    for c in range(H.shape[0]):
        for d in range(H.shape[1]):
            u, cert, info = mod.settle(H[c, d])
            states[c, d], ok[c, d] = u, cert
            worst['grad'] = max(worst['grad'], info['grad'])
            worst['min_eig'] = min(worst['min_eig'], info['min_eig'])
    return states, ok, worst


def family_measures(disk, states, ok, L, fam):
    labels = np.repeat(np.arange(CLASSES), DRAWS).reshape(CLASSES, DRAWS)
    flat_ok, flat_lab = ok.reshape(-1), labels.reshape(-1)
    layers = [disk.layer(l) for l in range(1, L + 1)]
    S = [layer_variance(states.reshape(-1, disk.n)[:, sl], flat_ok) for sl in layers]
    P = [within_class_variance(states.reshape(-1, disk.n)[:, sl], flat_lab, flat_ok) for sl in layers]
    centre = states.reshape(-1, disk.n)[:, [disk.centre]]
    out = dict(certified=dict(train=int(ok[:, :TRAIN].sum()), test=int(ok[:, TRAIN:].sum())), S=S, P=P,
               centre=dict(S=layer_variance(centre, flat_ok), P=within_class_variance(centre, flat_lab, flat_ok)),
               suppression_ratio=None if not P[0] or P[-1] is None else P[-1] / P[0])
    D, M = {}, {}
    tr_lab, te_lab = labels[:, :TRAIN].reshape(-1), labels[:, TRAIN:].reshape(-1)
    tr_ok, te_ok = ok[:, :TRAIN].reshape(-1), ok[:, TRAIN:].reshape(-1)
    for l, sl in enumerate(layers, start=1):
        X = states[:, :, sl]
        eps = np.random.default_rng([SEED_READOUT, L, l, fam]).standard_normal(X.shape)
        for s0 in SIGMAS:
            Y = X + s0 * eps
            tr, te = Y[:, :TRAIN].reshape(-1, X.shape[-1]), Y[:, TRAIN:].reshape(-1, X.shape[-1])
            strict, secondary = nearest_mean(tr, tr_lab, tr_ok, te, te_lab, te_ok, CLASSES)
            D.setdefault(str(s0), []).append(dict(strict=strict, secondary=secondary))
            M.setdefault(str(s0), []).append(min_mahalanobis(tr, tr_lab, tr_ok, s0, CLASSES))
    out.update(D=D, mahalanobis=M)
    return out


def profile(rec):
    S, P = rec['S'], rec['P']
    t1 = all(s is not None for s in S) and all(S[0] >= s for s in S)
    t2 = rec['suppression_ratio'] is not None and rec['suppression_ratio'] <= T2_MAX
    t3 = rec['D'][str(SIGMA_PROFILE)][-1]['strict'] >= T3_MIN
    return dict(T1=t1, T2=t2, T3=t3)


def multistability(mod, nuis, alpha, L):
    out = []
    starts = [np.random.default_rng([SEED_STARTS, L, s]).standard_normal(mod.disk.n) for s in range(N_STARTS)]
    for fam, (fname, n_freq) in enumerate(FAMILIES):
        H = drives(nuis, fam, n_freq, alpha)
        for c in range(CLASSES):
            found, unsettled = [], 0
            for u0 in starts:
                u, cert, _ = mod.settle(H[c, 0], start=u0)
                if not cert:
                    unsettled += 1
                elif not any(rms_distance(u, v) < MERGE_TOL for v in found):
                    found.append(u)
            out.append(dict(family=fname, cls=c, distinct_minima=len(found), unsettled_starts=unsettled))
    return out


def superposition(mod, nuis, alpha, L):
    low, high = drives(nuis, 0, 1, alpha), drives(nuis, 1, 8, alpha)
    sl = mod.disk.layer(L)
    rows, linear_ok = [], True
    for i in range(8):
        h1, h2 = low[i % 4, TRAIN + i], high[(i + 1) % 4, TRAIN + i]
        (u1, c1, _), (u2, c2, _), (u12, c12, _) = mod.settle(h1), mod.settle(h2), mod.settle(h1 + h2)
        diff = float(np.linalg.norm(u12[sl] - u1[sl] - u2[sl]))
        den = float(np.linalg.norm(u12[sl]))
        defect = None if den < 1e-12 or not (c1 and c2 and c12) else diff / den
        rows.append(dict(pair=i, defect=defect, absolute=diff))
        if mod.pot.kind == 'linear':
            linear_ok &= diff <= 1e-8 * max(1.0, float(np.linalg.norm(u1[sl]) + np.linalg.norm(u2[sl])))
    vals = [r['defect'] for r in rows if r['defect'] is not None]
    return dict(pairs=rows, mean=float(np.mean(vals)) if vals else None, nulls=8 - len(vals)), linear_ok


def run_config(nuis, L, gamma, pot, alpha):
    disk = Disk(L, gamma)
    mod = Module(disk, Potential(*pot))
    fams, gates = {}, dict(linear_direct=True)
    worst = dict(grad=0.0, min_eig=np.inf)
    for fam, (fname, n_freq) in enumerate(FAMILIES):
        H = drives(nuis, fam, n_freq, alpha)
        states, ok, w = settle_all(mod, H)
        worst['grad'] = max(worst['grad'], w['grad'])
        worst['min_eig'] = min(worst['min_eig'], w['min_eig'])
        if pot[0] == 'linear':
            dev = max(float(np.max(np.abs(states[c, d] - mod.direct_linear(H[c, d]))))
                      for c in range(CLASSES) for d in range(DRAWS))
            gates['linear_direct'] &= dev <= 1e-7
            gates.setdefault('linear_direct_max_deviation', 0.0)
            gates['linear_direct_max_deviation'] = max(gates['linear_direct_max_deviation'], dev)
        rec = family_measures(disk, states, ok, L, fam)
        rec['profile'] = profile(rec)
        fams[fname] = rec
    eligible = all(f['certified']['train'] + f['certified']['test'] == CLASSES * DRAWS for f in fams.values())
    meets = eligible and all(all(f['profile'].values()) for f in fams.values())
    sup, sup_ok = superposition(mod, nuis, alpha, L)
    gates['superposition_linear'] = sup_ok
    return dict(L=L, gamma=gamma, potential=Potential(*pot).name, kind=pot[0], alpha=alpha, units=disk.n,
                eligible=eligible, meets_profile=meets, families=fams,
                multistability=multistability(mod, nuis, alpha, L), superposition=sup,
                settling=dict(worst_grad=worst['grad'], worst_min_eig=worst['min_eig'])), gates


# ---------------------------------------------------------------- gates independent of the scored grid

def graph_gate():
    ok = all(Disk(L, 1).sizes == ring_sizes(L) for L in DEPTHS)
    ok &= [ring_sizes(L) for L in DEPTHS] == [[32, 16], [32, 21, 11], [32, 24, 16, 8], [32, 27, 21, 16, 11, 5]]
    for L in DEPTHS:
        for g in GAMMAS:
            d = Disk(L, g)
            edges = [e for _, e in d.lateral] + d.radial
            ok &= d.connected() and len(edges) == len(set(edges)) and np.allclose(d.K.sum(axis=1), 0)
            ok &= sum(1 for i, j in d.radial if j == d.centre) == d.sizes[-1]
    return ok


def derivative_gate():
    worst = 0.0
    for L in DEPTHS:
        for g in GAMMAS:
            for pot in POTENTIALS:
                mod = Module(Disk(L, g), Potential(*pot))
                rng = np.random.default_rng([SEED_CHECK, L, g, POTENTIALS.index(pot)])
                u, h = rng.standard_normal(mod.disk.n), rng.standard_normal(N1)
                e = np.eye(mod.disk.n) * 1e-6
                fd_g = np.array([(mod.energy(u + e[i], h) - mod.energy(u - e[i], h)) / 2e-6 for i in range(mod.disk.n)])
                fd_h = np.array([(mod.grad(u + e[i], h) - mod.grad(u - e[i], h)) / 2e-6 for i in range(mod.disk.n)])
                worst = max(worst, float(np.max(np.abs(fd_g - mod.grad(u, h)))),
                            float(np.max(np.abs(fd_h - mod.hess(u)))))
    return worst <= 1e-5, worst


def attenuation_gate():
    mod = Module(Disk(4, 1), Potential('linear', 0))
    sl = mod.disk.layer(4)
    r = [float(np.sqrt(np.mean(mod.settle(signal(n, 0))[0][sl] ** 2))) for n in (1, 8)]
    return r[1] < r[0], r


# ---------------------------------------------------------------- summary

def summary(configs):
    out = dict(configs=len(configs), meeting_profile=sum(c['meets_profile'] for c in configs),
               eligible=sum(c['eligible'] for c in configs))
    by = {}
    for c in configs:
        row = by.setdefault(f"{c['potential']}/gamma={c['gamma']}/alpha={c['alpha']:g}", [])
        if c['meets_profile']:
            row.append(c['L'])
    out['depths_meeting_profile'] = by
    cand = sorted(c['L'] for c in configs if c['potential'] == 'beta=1' and c['gamma'] == 1 and c['alpha'] == 1
                  and c['meets_profile'])
    out['candidate_depth_stage2'] = cand[0] if cand else None
    fails = {}
    for c in configs:
        if c['meets_profile']:
            continue
        for fname, f in c['families'].items():
            for t, v in f['profile'].items():
                if not v:
                    key = f"{c['kind']}/{fname}/{t}"
                    fails[key] = fails.get(key, 0) + 1
        if not c['eligible']:
            fails[f"{c['kind']}/unsettled"] = fails.get(f"{c['kind']}/unsettled", 0) + 1
    out['failures'] = dict(sorted(fails.items()))
    table = {}
    for c in configs:
        k = f"L={c['L']}/gamma={c['gamma']}/alpha={c['alpha']:g}"
        table.setdefault(k, {})[c['potential']] = {s: c['families']['F-high']['D'][s][-1]['strict'] for s in
                                                    map(str, SIGMAS)}
    out['F_high_interior_strict_accuracy'] = table
    return out


def run_tests():
    names = ['tests.test_layered_module']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(modules=names, run=res.testsRun, failures=len(res.failures), errors=len(res.errors),
                passed=res.wasSuccessful())


def main(argv=None):
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    ap.add_argument('--manifest-only', action='store_true')
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    gate = dict(contract=sha(CONTRACT) == m['contract_sha256'],
                environment=(np.__version__, scipy.__version__) == (m['numpy'], m['scipy']))
    if not all(gate.values()):
        sys.exit(f'contract or environment differs: {gate}')
    t0 = time.perf_counter()
    gate['graph'] = graph_gate()
    gate['derivatives'], deriv_worst = derivative_gate()
    gate['attenuation_sanity'], atten = attenuation_gate()
    nuis = nuisance_unit(SEED_NUISANCE, (len(FAMILIES), CLASSES, DRAWS))
    configs, lin_dev = [], 0.0
    gate['linear_direct'] = gate['superposition_linear'] = True
    for L in DEPTHS:
        for g in GAMMAS:
            for pot in POTENTIALS:
                for a in ALPHAS:
                    rec, gs = run_config(nuis, L, g, pot, a)
                    configs.append(rec)
                    gate['linear_direct'] &= gs['linear_direct']
                    gate['superposition_linear'] &= gs['superposition_linear']
                    lin_dev = max(lin_dev, gs.get('linear_direct_max_deviation', 0.0))
    seconds = time.perf_counter() - t0
    gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
    tests = run_tests()
    completion = completion_status(gate, tests)
    prov = dict(manifest_sha256=m['sha256'], contract_sha256=m['contract_sha256'],
                code={p: sha(p) for p in ('src/layered_module.py', 'experiments/layered_module.py',
                                          'tests/test_layered_module.py')})
    diagnostics = dict(derivative_max_error=deriv_worst, attenuation_rms_n1_n8=atten,
                       linear_direct_max_deviation=lin_dev)
    dump(out / f'{PREFIX}-results.json', rnd(dict(
        experiment=PREFIX, completion=completion, correctness_gate=gate, gate_diagnostics=diagnostics,
        tests=tests, provenance=prov, summary=summary(configs), configs=configs)))
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds=dict(grid_and_gates=round(seconds, 3)), total=round(time.perf_counter() - started, 3),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_rss_scope='whole process',
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps(gate))


if __name__ == '__main__':
    main()
