"""Layered selectivity v1: confirmation of the analytically designed band-selective disk.

Run `.venv/bin/python -m experiments.layered_selectivity [--out-dir DIR] [--manifest-only | --gates-only]
[--workers N]`. `--gates-only` runs the pre-confirmation gates (no section 3 draws are generated).
"""
import os

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):      # before numpy: one BLAS thread
    os.environ[_v] = '1'

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

import numpy as np
import scipy

from experiments.layered_module import rnd
from experiments.mechanism_learning import canonical, completion_status, dump, sha
from src.layered_module import (
    CERT_EIG, CERT_GRAD, LBFGS_OPTIONS, POLISH_GRAD, layer_variance, nearest_mean, nuisance_unit, signal,
    within_class_variance,
)
from src.layered_module_v2 import clopper_pearson
from src.layered_selectivity import (
    ARMS, BANDS, J_FLOOR, N, SelectiveDisk, SelectivityModule, band_symbols, fourier_transfer, lateral,
    max_coupling_distance, mode_gains, population, selectivity, signal_power,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'layered-selectivity-v1'
CONTRACT = 'docs/layered-selectivity-v1-contract.md'
CODE = ('src/layered_module.py', 'src/layered_module_v2.py', 'src/layered_selectivity.py',
        'experiments/layered_module.py', 'experiments/mechanism_learning.py', 'experiments/layered_selectivity.py',
        'tests/test_layered_selectivity.py')
PRIOR = ('reports/layered-module-design-analysis.json',)
DEPTHS, ALPHAS = (2, 4), (0.5, 1.0)
FAMILIES = (('F-low', 1), ('F-high', 8))
CLASSES, DRAWS, TRAIN = 4, 50, 25
REPLICATES = 20
SIGMA = 0.1
SEED = 20261030                         # [SEED, 0, r] nuisance; [SEED, 1, r, l, family] readout; [SEED, 2, arm, L] gates
T2_MAX, T3_MIN, T4_MAX = 0.25, 0.9, 0.5
SUCCESS_MIN = 18
PRIMARY = ('selective', 4, 1.0)
CI_LEVEL = 0.95
FD_STEP, FD_TOL, ENERGY_IDENTITY_TOL = 1e-6, 1e-5, 1e-12
TRANSFER_TOL, DIRECT_TOL, GAIN_TOL, MONOTONE_TOL = 1e-9, 1e-7, 1e-9, 1e-10
MAX_DISTANCE = dict(selective=4, no_lateral=0, ordinary_smoothing=1, wrong_bands=4)
PREDICTIONS = dict(selective=dict(signal_gains=[0.8, 0.6, 0.4, 0.2], suppression=0.0165802463, J=0.2652839410,
                                  gaussian_accuracy=0.9989094623, spherical_lower_bound=0.9903441),
                   no_lateral=dict(signal_gains=[0.8, 0.6, 0.4, 0.2], suppression=0.1074856046, J=1.7197696737))
RULES = [
    'disk: L rings of 32 and one centre; radial j <-> j and ring L <-> centre, unit strength; unit exterior tie; no '
    'exterior lateral penalty; no local potential; interior lateral energy 1/2 u_l^T K u_l on rings 2..L',
    'arms: selective K = tau B^T B, B = (C - e_1 I)(C - e_8 I), tau = 1 / min q_k over k not in {1,8,24,31}; '
    'no_lateral K = 0; ordinary_smoothing K = C; wrong_bands as selective with modes 1, 4 (excluded {1,4,28,31})',
    'grid: arms x L in {2, 4} x alpha in {0.5, 1}; 20 replicates; primary selective L = 4 alpha = 1',
    'settling: v1 L-BFGS-B from u = 0 and Newton polish; certified iff max|grad| < 1e-8 and min Hessian eigenvalue '
    '> 1e-8; every canonical state within 1e-7 (inf-norm) of the Cholesky direct solve',
    'nuisance: v1 Fourier basis, N(0,1) coefficients from default_rng([20261030, 0, r]) shape (2,4,50,32), rescaled '
    'per draw to RMS alpha/sqrt(2); same draws for every arm, depth and alpha; draws 0-24 train, 25-49 test',
    'readout: absolute sigma = 0.1 times eps from default_rng([20261030, 1, r, l, family]) shape (4,50,32), l 1-based, '
    'family 0 = F-low, 1 = F-high; v1 nearest_mean on all 32 coordinates, means from certified noisy training states',
    'pure signal: 4 drives per family and configuration, canonical start; Vsig(l) = mean_{c,j} (u - mean_d u)^2',
    'criteria per family: T1 S(1) >= S(l); T2 P(L)/P(1) <= 0.25; T3 strict NM at layer L >= 0.9; T4 J_n <= 0.5, '
    'J_n = [P(L)/P(1)] / [Vsig(L)/Vsig(1)], unavailable (T4 not met) if a pure-signal drive is unsettled or P(1), '
    'Vsig(1) or Vsig(L) < 1e-12; replicate eligible iff all 400 canonical settlings certified',
    'replicate success: eligible and T1-T4 in both families; primary success iff >= 18 of 20; Clopper-Pearson 95%',
    'gates: Hessian symmetric PD; lateral coupling distance per arm; radial/centre structure; derivatives by central '
    'differences step 1e-6 with |fd - analytic| <= 1e-5 max(1, max|analytic|) and the quadratic energy identity to '
    '1e-12 relative; spatial versus tridiagonal transfer 1e-9; gains in [0, 1] and nonincreasing inward to 1e-10; '
    'pure-signal responses equal declared gains to 1e-9; NM equals an independent vectorized implementation',
    'parallel over (configuration, replicate) units with one BLAS thread per process; results do not depend on '
    'worker count; reported floats rounded to 10 significant digits',
]


def configs():
    return [(arm, L, a) for arm in ARMS for L in DEPTHS for a in ALPHAS]


def config_key(arm, L, alpha):
    return f'{arm}/L={L}/alpha={alpha:g}'


def manifest():
    taus = {arm: band_symbols(b)[2] for arm, b in BANDS.items()}
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                requirements_sha256=sha('requirements.txt'), numpy=np.__version__, scipy=scipy.__version__,
                blas_threads=1, code={p: sha(p) for p in CODE}, prior_artifacts={p: sha(p) for p in PRIOR},
                n=N, arms=list(ARMS), bands={k: list(v) for k, v in BANDS.items()}, tau=taus,
                max_lateral_stiffness={arm: float(np.max(lateral(arm)[1])) for arm in ARMS},
                depths=list(DEPTHS), alphas=list(ALPHAS), families={k: v for k, v in FAMILIES},
                classes=CLASSES, draws=DRAWS, train_draws=TRAIN, replicates=REPLICATES, sigma=SIGMA,
                seeds=dict(base=SEED, nuisance=[SEED, 0, 'r'], readout=[SEED, 1, 'r', 'l', 'family'],
                           gates=[SEED, 2, 'arm_index', 'L']),
                primary=dict(arm=PRIMARY[0], L=PRIMARY[1], alpha=PRIMARY[2], successes_required=SUCCESS_MIN),
                thresholds=dict(T2_max=T2_MAX, T3_min=T3_MIN, T4_max=T4_MAX, J_floor=J_FLOOR, ci_level=CI_LEVEL),
                tolerances=dict(fd_step=FD_STEP, fd=FD_TOL, energy_identity=ENERGY_IDENTITY_TOL,
                                transfer=TRANSFER_TOL, direct=DIRECT_TOL, gain=GAIN_TOL, monotone=MONOTONE_TOL),
                lbfgs_options=LBFGS_OPTIONS, certification=dict(grad=CERT_GRAD, min_eig=CERT_EIG),
                polish_grad=POLISH_GRAD, predictions=PREDICTIONS, rules=RULES)
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    m = manifest()
    path = out / f'{PREFIX}-manifest.json'
    frozen = ROOT / 'reports' / f'{PREFIX}-manifest.json'
    ref = path if path.exists() else frozen if frozen.exists() else None
    if ref is not None and json.loads(ref.read_text()) != m:
        sys.exit('manifest differs from the frozen file (contract, code or environment changed); refusing to run')
    if not path.exists():
        dump(path, m)
    return m


# ---------------------------------------------------------------- pre-confirmation gates (no section 3 draws)

def structure_gate(arm, L):
    disk = SelectiveDisk(arm, L)
    H = disk.hessian
    ev = np.linalg.eigvalsh(H)
    out = dict(symmetric=bool(np.array_equal(H, H.T)), positive_definite=bool(ev[0] > 0),
               eigenvalue_range=[float(ev[0]), float(ev[-1])], condition=float(ev[-1] / ev[0]))
    dist = max_coupling_distance(disk.lateral_K)
    out['lateral_distance'] = dist
    ok = dist == MAX_DISTANCE[arm]
    eye = np.eye(N)
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            block = H[disk.layer(i), disk.layer(j)]
            if abs(i - j) == 1:
                ok &= np.array_equal(block, -eye)
            elif i != j:
                ok &= not block.any()
    ok &= np.array_equal(H[:N, :N], 2 * eye)                            # exterior: tie + radial, no lateral
    c = disk.centre
    ok &= H[c, c] == N and np.array_equal(H[c, disk.layer(L)], -np.ones(N))
    ok &= not H[c, :disk.offsets[L - 1]].any()
    out['graph'] = bool(ok)
    out['passed'] = out['symmetric'] and out['positive_definite'] and out['graph']
    return out


def derivative_gate(arm, L):
    mod = SelectivityModule(SelectiveDisk(arm, L))
    rng = np.random.default_rng([SEED, 2, ARMS.index(arm), L])
    u, h = rng.standard_normal(mod.disk.n), rng.standard_normal(N)
    e = np.eye(mod.disk.n) * FD_STEP
    fd_g = np.array([(mod.energy(u + e[i], h) - mod.energy(u - e[i], h)) / (2 * FD_STEP) for i in range(mod.disk.n)])
    fd_h = np.array([(mod.grad(u + e[i], h) - mod.grad(u - e[i], h)) / (2 * FD_STEP) for i in range(mod.disk.n)])
    g, H = mod.grad(u, h), mod.hess(u)
    f = np.zeros(mod.disk.n)
    f[:N] = h
    E, quad = mod.energy(u, h), 0.5 * u @ mod.disk.hessian @ u - f @ u + 0.5 * h @ h
    out = dict(grad_abs=float(np.max(np.abs(fd_g - g))), grad_scale=float(np.max(np.abs(g))),
               hess_abs=float(np.max(np.abs(fd_h - H))), hess_scale=float(np.max(np.abs(H))),
               grad_rel=float(np.max(np.abs(fd_g - g)) / max(1.0, np.max(np.abs(g)))),
               hess_rel=float(np.max(np.abs(fd_h - H)) / max(1.0, np.max(np.abs(H)))),
               energy_identity_rel=float(abs(E - quad) / max(1.0, abs(E))))
    out['passed'] = (out['grad_rel'] <= FD_TOL and out['hess_rel'] <= FD_TOL
                     and out['energy_identity_rel'] <= ENERGY_IDENTITY_TOL)
    return out


def transfer_gate(arm, L):
    mod = SelectivityModule(SelectiveDisk(arm, L))
    G = mod.transfer()
    rings = np.array([G[mod.disk.layer(l)] for l in range(1, L + 1)])
    spectral = fourier_transfer(mod.disk.symbol, L)
    err = float(np.max(np.abs(rings - spectral)))
    centre = float(np.max(np.abs(G[mod.disk.centre] - rings[-1].mean(axis=0))))
    g = mode_gains(mod.disk.symbol, L)
    mono = bool(np.max(g) <= 1 + MONOTONE_TOL and np.min(g) >= -MONOTONE_TOL
                and np.all(np.diff(g, axis=0) <= MONOTONE_TOL))
    return dict(max_error=err, centre_max_error=centre, gains_monotone_bounded=mono,
                passed=err <= TRANSFER_TOL and centre <= TRANSFER_TOL and mono)


def nm_reference(train, train_labels, test, classes=CLASSES):
    """Independent vectorized nearest class mean (all rows certified); argmin ties go to the lowest class."""
    means = np.array([train[train_labels == c].mean(axis=0) for c in range(classes)])
    d = ((test[:, None, :] - means[None]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def nm_gate():
    rng = np.random.default_rng([SEED, 3])
    worst = 0
    for trial in range(20):
        centres = rng.standard_normal((CLASSES, N)) * 0.5
        tl, el = np.repeat(np.arange(CLASSES), TRAIN), np.repeat(np.arange(CLASSES), DRAWS - TRAIN)
        tr, te = centres[tl] + rng.standard_normal((len(tl), N)), centres[el] + rng.standard_normal((len(el), N))
        if trial == 0:                                                  # exact ties between equal means
            tr[tl == 1] = tr[tl == 0]
        ok_tr, ok_te = np.ones(len(tl), bool), np.ones(len(el), bool)
        strict, secondary = nearest_mean(tr, tl, ok_tr, te, el, ok_te, CLASSES)
        ref = float(np.mean(nm_reference(tr, tl, te) == el))
        worst = max(worst, abs(strict - ref), abs(secondary - ref))
    return dict(max_difference=worst, passed=worst == 0)


def pure_signal(arm, L):
    """Pure-signal settlings for one configuration: Vsig per layer and the declared-gain gate (no nuisance)."""
    mod = SelectivityModule(SelectiveDisk(arm, L))
    g = mode_gains(mod.disk.symbol, L)
    out, worst = {}, 0.0
    for fname, n_freq in FAMILIES:
        sig = np.array([signal(n_freq, c) for c in range(CLASSES)])
        states, cert = [], True
        for s in sig:
            u, ok, _ = mod.settle(s)
            cert &= ok
            states.append(u)
            for l in range(1, L + 1):
                worst = max(worst, float(np.max(np.abs(u[mod.disk.layer(l)] - g[l - 1, n_freq] * s))))
        states = np.array(states)
        V = [signal_power(states[:, mod.disk.layer(l)]) for l in range(1, L + 1)] if cert else None
        out[fname] = dict(certified=bool(cert), Vsig=V, Vsig_input=signal_power(sig))
    return out, worst


# ---------------------------------------------------------------- one (configuration, replicate)

def run_unit(task):
    (arm, L, alpha), r, pure = task
    mod = SelectivityModule(SelectiveDisk(arm, L))
    nuis = nuisance_unit([SEED, 0, r], (len(FAMILIES), CLASSES, DRAWS), N)
    labels = np.repeat(np.arange(CLASSES), DRAWS).reshape(CLASSES, DRAWS)
    tr_lab, te_lab, flat_lab = labels[:, :TRAIN].reshape(-1), labels[:, TRAIN:].reshape(-1), labels.reshape(-1)
    fams, direct_dev, opt = {}, 0.0, dict(lbfgs=[], newton_max=0, worst_grad=0.0, worst_min_eig=np.inf)
    for fam, (fname, n_freq) in enumerate(FAMILIES):
        sig = np.array([signal(n_freq, c) for c in range(CLASSES)])
        H = sig[:, None, :] + nuis[fam] * (alpha / np.sqrt(2))
        states = np.zeros((CLASSES, DRAWS, mod.disk.n))
        ok = np.zeros((CLASSES, DRAWS), bool)
        for c in range(CLASSES):
            for d in range(DRAWS):
                u, cert, info = mod.settle(H[c, d])
                states[c, d], ok[c, d] = u, cert
                direct_dev = max(direct_dev, float(np.max(np.abs(u - mod.direct(H[c, d])))))
                opt['lbfgs'].append(info['lbfgs_iterations'])
                opt['newton_max'] = max(opt['newton_max'], info['newton_steps'])
                opt['worst_grad'] = max(opt['worst_grad'], info['grad'])
                opt['worst_min_eig'] = min(opt['worst_min_eig'], info['min_eig'])
        flat, flat_ok = states.reshape(-1, mod.disk.n), ok.reshape(-1)
        rings = [mod.disk.layer(l) for l in range(1, L + 1)]
        S = [layer_variance(flat[:, sl], flat_ok) for sl in rings]
        P = [within_class_variance(flat[:, sl], flat_lab, flat_ok) for sl in rings]
        P_in = within_class_variance(H.reshape(-1, N), flat_lab, np.ones(len(flat_lab), bool))
        centre = flat[:, [mod.disk.centre]]
        tr_ok, te_ok = ok[:, :TRAIN].reshape(-1), ok[:, TRAIN:].reshape(-1)
        nm = []
        for l, sl in enumerate(rings, start=1):
            X = states[:, :, sl]
            Y = X + SIGMA * np.random.default_rng([SEED, 1, r, l, fam]).standard_normal(X.shape)
            strict, secondary = nearest_mean(Y[:, :TRAIN].reshape(-1, N), tr_lab, tr_ok,
                                             Y[:, TRAIN:].reshape(-1, N), te_lab, te_ok, CLASSES)
            nm.append(dict(strict=strict, secondary=secondary))
        supp = None if not P[0] or P[-1] is None else P[-1] / P[0]
        V = pure[fname]['Vsig']
        J, reason = selectivity(P[0], P[-1], V[0] if V else None, V[-1] if V else None)
        crit = dict(T1=all(s is not None for s in S) and all(S[0] >= s for s in S),
                    T2=supp is not None and supp <= T2_MAX, T3=nm[-1]['strict'] >= T3_MIN,
                    T4=J is not None and J <= T4_MAX)
        fams[fname] = dict(
            certified=dict(train=int(ok[:, :TRAIN].sum()), test=int(ok[:, TRAIN:].sum())), S=S, P=P,
            suppression_ratio=supp, centre=dict(S=layer_variance(centre, flat_ok),
                                                P=within_class_variance(centre, flat_lab, flat_ok)),
            input_P=P_in, P_input_relative=[p / P_in if p is not None else None for p in P],
            Vsig=V, Vsig_input_relative=[v / pure[fname]['Vsig_input'] for v in V] if V else None,
            Vsig_ratio=V[-1] / V[0] if V and V[0] >= J_FLOOR else None,
            J=J, J_unavailable=reason, nm=nm, criteria=crit)
    eligible = all(f['certified']['train'] + f['certified']['test'] == CLASSES * DRAWS for f in fams.values())
    profile = eligible and all(f['criteria'][t] for f in fams.values() for t in ('T1', 'T2', 'T3'))
    success = profile and all(f['criteria']['T4'] for f in fams.values())
    lb = opt.pop('lbfgs')
    opt.update(lbfgs_mean=float(np.mean(lb)), lbfgs_max=int(np.max(lb)))
    return dict(replicate=r, eligible=eligible, profile_pass=profile, success=success, families=fams,
                optimizer=opt, direct_max_deviation=direct_dev)


# ---------------------------------------------------------------- summary

def _stats(xs):
    xs = [x for x in xs if x is not None]
    return dict(mean=float(np.mean(xs)), min=float(np.min(xs)), max=float(np.max(xs)), n=len(xs)) if xs else None


def config_summary(c):
    reps = c['replicates']
    k = sum(r['success'] for r in reps)
    fails = {}
    for r in reps:
        if not r['eligible']:
            fails['ineligible'] = fails.get('ineligible', 0) + 1
        for fname, f in r['families'].items():
            for t, v in f['criteria'].items():
                if not v:
                    fails[f'{fname}/{t}'] = fails.get(f'{fname}/{t}', 0) + 1
    fam = {}
    for fname, _ in FAMILIES:
        fs = [r['families'][fname] for r in reps]
        fam[fname] = dict(accuracy=_stats([f['nm'][-1]['strict'] for f in fs]), J=_stats([f['J'] for f in fs]),
                          J_unavailable=sum(f['J'] is None for f in fs),
                          suppression=_stats([f['suppression_ratio'] for f in fs]))
    return dict(key=c['key'], arm=c['arm'], L=c['L'], alpha=c['alpha'], replicates=len(reps),
                eligible=sum(r['eligible'] for r in reps), profile_passes=sum(r['profile_pass'] for r in reps),
                successes=k, success_interval=list(clopper_pearson(k, len(reps), CI_LEVEL)),
                failures=dict(sorted(fails.items())), families=fam)


def primary_outcome(s):
    if s['successes'] >= SUCCESS_MIN:
        return 'success'
    if s['profile_passes'] >= SUCCESS_MIN:
        return 'profile_pass_without_demonstrated_selectivity'
    return 'not_met'


def matched_controls(idx):
    rows, tally = [], {}
    for arm in ARMS[1:]:
        for L in DEPTHS:
            for a in ALPHAS:
                sel, ctl = idx[config_key('selective', L, a)], idx[config_key(arm, L, a)]
                for rs, rc in zip(sel['replicates'], ctl['replicates']):
                    for fname, _ in FAMILIES:
                        fs, fc = rs['families'][fname], rc['families'][fname]
                        row = dict(control=arm, L=L, alpha=a, replicate=rs['replicate'], family=fname,
                                   both_eligible=rs['eligible'] and rc['eligible'],
                                   accuracy_selective=fs['nm'][-1]['strict'], accuracy_control=fc['nm'][-1]['strict'],
                                   accuracy_difference=fs['nm'][-1]['strict'] - fc['nm'][-1]['strict'],
                                   J_selective=fs['J'], J_control=fc['J'],
                                   J_difference=fs['J'] - fc['J'] if fs['J'] is not None and fc['J'] is not None
                                   else None, J_control_unavailable=fc['J_unavailable'])
                        rows.append(row)
                        t = tally.setdefault(f'{arm}/L={L}/alpha={a:g}/{fname}', dict(acc=[], J=[], J_null=0))
                        t['acc'].append(row['accuracy_difference'])
                        t['J'].append(row['J_difference'])
                        t['J_null'] += row['J_difference'] is None
    summary = {k: dict(accuracy_difference=_stats(v['acc']), J_difference=_stats(v['J']),
                       J_difference_unavailable=v['J_null']) for k, v in tally.items()}
    return dict(summary=summary, rows=rows)


def predictions():
    out = {}
    for arm, L, a in configs():
        sym = lateral(arm)[1]
        out[config_key(arm, L, a)] = {fname: population(sym, L, n, a, SIGMA) for fname, n in FAMILIES}
    check = {}
    for arm, pred in PREDICTIONS.items():
        got = out[config_key(arm, 4, 1.0)]
        for fname, _ in FAMILIES:
            for k, v in pred.items():
                g = got[fname][k]
                dev = float(np.max(np.abs(np.array(g) - np.array(v))))
                check[f'{arm}/{fname}/{k}'] = dict(declared=v, computed=g, deviation=dev, agrees=dev <= 1e-6)
    return dict(population=out, contract_section5=check)


def summary(results):
    idx = {c['key']: c for c in results}
    per = [config_summary(c) for c in results]
    prim = next(s for s in per if s['key'] == config_key(*PRIMARY))
    return dict(primary=dict(prim, successes_required=SUCCESS_MIN, outcome=primary_outcome(prim)),
                configs=per, matched_controls=matched_controls(idx), stage2_selection=None)


def run_tests():
    names = ['tests.test_layered_selectivity']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(modules=names, run=res.testsRun, failures=len(res.failures), errors=len(res.errors),
                passed=res.wasSuccessful())


def pre_gates(pool):
    gate, diag = {}, {}
    cells = [(arm, L) for arm in ARMS for L in DEPTHS]
    diag['structure'] = {f'{a}/L={L}': structure_gate(a, L) for a, L in cells}
    diag['derivatives'] = {f'{a}/L={L}': derivative_gate(a, L) for a, L in cells}
    diag['transfer'] = {f'{a}/L={L}': transfer_gate(a, L) for a, L in cells}
    diag['nearest_mean'] = nm_gate()
    pure = pool.starmap(pure_signal, [(arm, L) for arm, L, _ in configs()])
    diag['pure_signal_gain_max_deviation'] = max(w for _, w in pure)
    for k in ('structure', 'derivatives', 'transfer'):
        gate[k] = all(v['passed'] for v in diag[k].values())
    gate['nearest_mean'] = diag['nearest_mean']['passed']
    gate['pure_signal_gains'] = diag['pure_signal_gain_max_deviation'] <= GAIN_TOL
    pure_by = {config_key(*c): p for c, (p, _) in zip(configs(), pure)}
    return gate, diag, pure_by


def main(argv=None):
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--gates-only', action='store_true')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    gate = dict(contract=sha(CONTRACT) == m['contract_sha256'],
                code=all(sha(p) == h for p, h in m['code'].items()),
                environment=(np.__version__, scipy.__version__) == (m['numpy'], m['scipy']),
                prior_artifacts=all(sha(p) == h for p, h in m['prior_artifacts'].items()))
    if not all(gate.values()):
        sys.exit(f'contract, code, environment or prior artifacts differ: {gate}')
    t0 = time.perf_counter()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        pre, diag, pure = pre_gates(pool)
        gate.update(pre)
        if args.gates_only:
            print(json.dumps(rnd(dict(gate=gate, diagnostics=diag)), indent=1))
            return
        results = []
        if all(gate.values()):                                          # section 3 draws only after the gates pass
            tasks = [((arm, L, a), r, pure[config_key(arm, L, a)]) for arm, L, a in configs()
                     for r in range(REPLICATES)]
            units = pool.map(run_unit, tasks, chunksize=1)
            for i, (arm, L, a) in enumerate(configs()):
                reps = units[i * REPLICATES:(i + 1) * REPLICATES]
                results.append(dict(key=config_key(arm, L, a), arm=arm, L=L, alpha=a,
                                    pure_signal=pure[config_key(arm, L, a)], replicates=reps))
    if results:
        dev = max(r['direct_max_deviation'] for c in results for r in c['replicates'])
        gate['canonical_direct'] = dev <= DIRECT_TOL
        diag['canonical_direct_max_deviation'] = dev
    seconds = time.perf_counter() - t0
    gate['passed'] = all(v for k, v in gate.items() if k != 'passed') and bool(results)
    tests = run_tests()
    completion = completion_status(gate, tests)
    prov = dict(manifest_sha256=m['sha256'], contract_sha256=m['contract_sha256'], code=m['code'])
    body = dict(experiment=PREFIX, completion=completion, correctness_gate=gate, gate_diagnostics=diag,
                tests=tests, provenance=prov, predictions=predictions())
    if results:
        body.update(summary=summary(results), configs=results)
    dump(out / f'{PREFIX}-results.json', rnd(body))
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds=dict(gates_and_confirmation=round(seconds, 3)), total=round(time.perf_counter() - started, 3),
        workers=args.workers, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_scope='parent process', python=platform.python_version(), machine=platform.machine(),
        platform=platform.platform(), out_dir=str(out), command=' '.join(sys.argv),
        git_revision=git('rev-parse', 'HEAD'), dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps(gate))


if __name__ == '__main__':
    main()
