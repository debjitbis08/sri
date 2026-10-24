"""Layered module v2: geometric loss, attenuation and readout limits; soft-tie and field drives.

Run `.venv/bin/python -m experiments.layered_module_v2 [--out-dir DIR] [--manifest-only]`.
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

from experiments import layered_module as v1
from experiments.mechanism_learning import canonical, completion_status, dump, sha
from src.layered_module import (
    CERT_EIG, CERT_GRAD, LAMBDA, LBFGS_OPTIONS, N1, POLISH_GRAD, Potential, nearest_mean, nuisance_unit, ring_sizes,
    signal,
)
from src.layered_module_v2 import (
    GEOMETRIES, Population, clopper_pearson, closed_form_reference, lda, lda_cholesky_scores, lda_explicit_scores,
    linear_map, make_disk, make_module, mode_gains, null_readout, projection_features, spectrum,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'layered-module-v2'
CONTRACT = 'docs/layered-module-v2-contract.md'
V1_RESULTS = 'reports/layered-module-v1-results.json'
DEPTHS, GAMMAS, ALPHAS, FAMILIES = v1.DEPTHS, v1.GAMMAS, v1.ALPHAS, v1.FAMILIES
DRIVE_POTENTIALS = (
    ('tie', ('linear', 0)), ('tie', ('linear', 1)),
    ('tie', ('double_well', 0.5)), ('tie', ('double_well', 1)), ('tie', ('double_well', 2)),
    ('field', ('linear', 1)),
    ('field', ('double_well', 0.5)), ('field', ('double_well', 1)), ('field', ('double_well', 2)),
)
SIGMAS_ABS = v1.SIGMAS
SIGMAS_REL = (0.03, 0.1, 0.3)
NOISE = tuple(('abs', s) for s in SIGMAS_ABS) + tuple(('rel', s) for s in SIGMAS_REL)
SIGMA_PROFILE = 0.1
READING_ABS, READING_REL = 'abs:0.1', 'rel:0.1'
CLASSES, DRAWS, TRAIN = v1.CLASSES, v1.DRAWS, v1.TRAIN
SEED_NUISANCE, SEED_READOUT, SEED_STARTS, SEED_CHECK = 20261024, 20261025, 20261026, 20261027
SEED_REFERENCE, MC_DRAWS = 20261027, 10 ** 6
SEED_GATES = 20261028
R_MIN = 1e-12
GATE_LEVEL = 1 - 1e-6
T2_MAX, T3_MIN = v1.T2_MAX, v1.T3_MIN
READING_MARGIN = 0.10
PREDICTION_TOLERANCE = 0.1
RULES = [
    'geometries: shrinking (v1 ring sizes and Disk class unchanged) and constant (every ring 32, built by the v1 '
    'radial rule, which gives j <-> j); centre joined to ring L',
    'drives: tie (v1 Module unchanged, lambda = 1) or field E = 1/2 u^T K u + sum V(u) - sum_ring1 h u; field arm '
    'potentials linear mu = 1 and double well beta 0.5, 1, 2',
    'settling, certification, multistability, superposition, drives, nuisance, halves and readout eps exactly as v1',
    'pure-signal drives (4 per family, no nuisance) from the canonical start; R_l = RMS over units and classes of '
    'layer l; null if any unsettled or R_l < 1e-12',
    'readout noise sigma * eps with sigma = sigma0 in {0.03, 0.1, 0.3} (absolute) or sigma_rel R_l with sigma_rel in '
    '{0.03, 0.1, 0.3} (relative)',
    'readouts fitted on noisy certified training states: NM (v1); LDA pooled covariance dof N_cert,train - 4, '
    'Cholesky, equal priors, argmax ties lowest; null if class missing, N_cert,train - 4 < n_l or Cholesky fails; '
    'mode projection onto orthonormal cos/sin(n phi) when n_l > 2n, NM in 2-D; oracle-covariance LDA (linear) with '
    'LDA means and population Sigma',
    'strict accuracy out of 100 (unsettled and null count wrong) for profile accounting; null readouts are '
    'unavailable for readings',
    'population reference (linear): G from a dense solve; Sigma = (alpha^2/2) G G^T + sigma^2 I handled through the '
    'SVD of G; Bayes accuracy by Monte Carlo, 1e6 2-D standard normal draws from default_rng(20261027), draw i '
    'paired with class i mod 4, nearest whitened mean; Clopper-Pearson interval at level 1 - 1e-6',
    'spectra: orthonormal real Fourier basis of each ring; P_k within-class variance averaged over classes, B_k '
    'between-class variance of class means, cos and sin summed; certified drives of both halves, no readout noise',
    'profile at sigma0 = 0.1 (T1, T2, T3 >= 0.9 strict at layer L) with NM and with LDA as the T3 readout; '
    'eligible only if all 400 canonical settlings certified; nothing selected for Stage 2',
    'readings G, A, R as contract section 9, margin 0.10, with validity conditions; not-evaluable cases listed',
    'reported floats rounded to 10 significant digits; BLAS pinned to one thread before numpy import',
]


def config_key(geometry, drive, pot, L, gamma, alpha):
    return f"{geometry}/{drive}/{Potential(*pot).name}/L={L}/gamma={gamma}/alpha={alpha:g}"


def manifest():
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                requirements_sha256=sha('requirements.txt'), numpy=np.__version__, scipy=scipy.__version__, blas_threads=1,
                n1=N1, lam=LAMBDA, depths=list(DEPTHS),
                ring_sizes=dict(shrinking={str(L): ring_sizes(L) for L in DEPTHS},
                                constant={str(L): [N1] * L for L in DEPTHS}),
                geometries=list(GEOMETRIES), drive_potentials=[[d, list(p)] for d, p in DRIVE_POTENTIALS],
                gammas=list(GAMMAS), alphas=list(ALPHAS), families={k: v for k, v in FAMILIES},
                noise=dict(absolute=list(SIGMAS_ABS), relative=list(SIGMAS_REL), relative_min_scale=R_MIN),
                sigma_profile=SIGMA_PROFILE, classes=CLASSES, draws=DRAWS, train_draws=TRAIN,
                seeds=dict(nuisance=SEED_NUISANCE, readout=SEED_READOUT, starts=SEED_STARTS, checks=SEED_CHECK,
                           reference=SEED_REFERENCE, gates=SEED_GATES),
                reference=dict(draws=MC_DRAWS, interval='clopper-pearson', level=GATE_LEVEL),
                lbfgs_options=LBFGS_OPTIONS, certification=dict(grad=CERT_GRAD, min_eig=CERT_EIG),
                polish_grad=POLISH_GRAD, starts=v1.N_STARTS, merge_tol=v1.MERGE_TOL,
                thresholds=dict(T2_max=T2_MAX, T3_min=T3_MIN, reading_margin=READING_MARGIN,
                                prediction_tolerance=PREDICTION_TOLERANCE),
                prior_artifacts={V1_RESULTS: sha(V1_RESULTS)}, rules=RULES)
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


# ---------------------------------------------------------------- one configuration

def readout_scale(mod, n_freq, L):
    """R_l per layer from the four pure-signal drives (canonical start); None where undefined."""
    us, ok = [], True
    for c in range(CLASSES):
        u, cert, _ = mod.settle(signal(n_freq, c))
        ok &= cert
        us.append(u)
    out = []
    for l in range(1, L + 1):
        sl = mod.disk.layer(l)
        r = float(np.sqrt(np.mean(np.array([u[sl] for u in us]) ** 2))) if ok else None
        out.append(r if r is not None and r >= R_MIN else None)
    return out, ok


def layer_readouts(X, ok, fam, n_freq, L, l, R, pop_layer, alpha, z_mc, gates):
    """All readouts for one layer, for every noise setting; pop_layer is G_l for the linear arm, else None."""
    labels = np.repeat(np.arange(CLASSES), DRAWS).reshape(CLASSES, DRAWS)
    tr_lab, te_lab = labels[:, :TRAIN].reshape(-1), labels[:, TRAIN:].reshape(-1)
    tr_ok, te_ok = ok[:, :TRAIN].reshape(-1), ok[:, TRAIN:].reshape(-1)
    eps = np.random.default_rng([SEED_READOUT, L, l, fam]).standard_normal(X.shape)
    sig = np.array([signal(n_freq, c) for c in range(CLASSES)])
    out = {}
    for kind, s in NOISE:
        key = f'{kind}:{s:g}'
        if kind == 'rel' and R is None:
            out[key] = None
            continue
        sigma = s if kind == 'abs' else s * R
        Y = X + sigma * eps
        tr, te = Y[:, :TRAIN].reshape(-1, X.shape[-1]), Y[:, TRAIN:].reshape(-1, X.shape[-1])
        strict, secondary = nearest_mean(tr, tr_lab, tr_ok, te, te_lab, te_ok, CLASSES)
        rec = dict(sigma=sigma, nm=dict(strict=strict, secondary=secondary, null=None))
        rec['lda'], rec['mahalanobis_empirical'] = lda(tr, tr_lab, tr_ok, te, te_lab, te_ok, CLASSES)
        ftr, fte = projection_features(tr, n_freq), projection_features(te, n_freq)
        if ftr is None:
            rec['projection'] = null_readout(len(te), 'mode_not_representable')
        else:
            p_strict, p_secondary = nearest_mean(ftr, tr_lab, tr_ok, fte, te_lab, te_ok, CLASSES)
            rec['projection'] = dict(strict=p_strict, secondary=p_secondary, null=None)
        if pop_layer is not None:
            pop = Population(pop_layer, sig, alpha, sigma)
            acc, errors, n = pop.reference(z_mc)
            lo, hi = clopper_pearson(errors, n, GATE_LEVEL)
            rec['reference'] = dict(accuracy=acc, errors=errors, draws=n, error_interval=[lo, hi])
            rec['mahalanobis_population'] = pop.min_mahalanobis()
            means = [tr[tr_ok & (tr_lab == c)].mean(axis=0) for c in range(CLASSES) if (tr_ok & (tr_lab == c)).any()]
            rec['oracle_lda'] = (pop.oracle_lda(np.array(means), te, te_lab, te_ok) if len(means) == CLASSES
                                 else null_readout(len(te), 'class_without_training_drive'))
            if gates is not None:                                       # constant width: closed-form check
                g = gates['gain'][l - 1]
                cf_acc, cf_err = closed_form_reference(g, alpha, sigma)
                rec['reference']['closed_form'] = cf_acc
                inside = lo <= cf_err <= hi
                gates['reference_ok'] &= inside
                gates['reference_checks'] += 1
                if not inside:
                    gates['reference_failures'].append(dict(layer=l, noise=key, closed_form_error=cf_err,
                                                            mc_errors=errors, interval=[lo, hi]))
        else:
            rec['reference'] = rec['mahalanobis_population'] = rec['oracle_lda'] = None
        out[key] = rec
    return out


def lda_profile(rec, readouts):
    S = rec['S']
    t1 = all(s is not None for s in S) and all(S[0] >= s for s in S)
    t2 = rec['suppression_ratio'] is not None and rec['suppression_ratio'] <= T2_MAX
    t3 = readouts[READING_ABS][-1]['lda']['strict'] >= T3_MIN
    return dict(T1=t1, T2=t2, T3=t3)


def run_config(nuis, z_mc, geometry, drive, pot, L, gamma, alpha):
    disk = make_disk(geometry, L, gamma)
    mod = make_module(disk, Potential(*pot), drive)
    linear = pot[0] == 'linear'
    G = linear_map(mod, drive) if linear else None
    fams = {}
    gates = dict(linear_direct=True, linear_direct_max_deviation=0.0, mode_gains=True, mode_gain_max_deviation=0.0,
                 reference_ok=True, reference_checks=0, reference_failures=[])
    worst = dict(grad=0.0, min_eig=np.inf)
    for fam, (fname, n_freq) in enumerate(FAMILIES):
        H = v1.drives(nuis, fam, n_freq, alpha)
        states, ok, w = v1.settle_all(mod, H)
        worst['grad'] = max(worst['grad'], w['grad'])
        worst['min_eig'] = min(worst['min_eig'], w['min_eig'])
        if linear:
            dev = max(float(np.max(np.abs(states[c, d] - mod.direct_linear(H[c, d]))))
                      for c in range(CLASSES) for d in range(DRAWS))
            gates['linear_direct'] &= dev <= 1e-7
            gates['linear_direct_max_deviation'] = max(gates['linear_direct_max_deviation'], dev)
        rec = v1.family_measures(disk, states, ok, L, fam)
        rec.pop('mahalanobis')                                          # v1 definition counts readout noise twice
        rec['profile_nm'] = v1.profile(rec)
        R, sig_ok = readout_scale(mod, n_freq, L)
        rec['pure_signal_certified'] = sig_ok
        rec['R'] = R
        labels = np.repeat(np.arange(CLASSES), DRAWS).reshape(-1)
        cf_gates = None
        if linear and geometry == 'constant':
            cf_gates = dict(gain=mode_gains(n_freq, L, gamma, pot[1], drive), reference_ok=True,
                            reference_checks=0, reference_failures=[])
        rec['spectrum'], readouts = [], {}
        for l in range(1, L + 1):
            sl = disk.layer(l)
            rec['spectrum'].append(spectrum(states[:, :, sl].reshape(-1, disk.sizes[l - 1]), labels, ok.reshape(-1)))
            per = layer_readouts(states[:, :, sl], ok, fam, n_freq, L, l, R[l - 1],
                                 G[sl] if linear else None, alpha, z_mc, cf_gates)
            for key, v in per.items():
                readouts.setdefault(key, []).append(v)
        if cf_gates is not None:
            gates['reference_ok'] &= cf_gates['reference_ok']
            gates['reference_checks'] += cf_gates['reference_checks']
            gates['reference_failures'] += [dict(f, family=fname) for f in cf_gates['reference_failures']]
        rec['readouts'] = readouts
        rec['profile_lda'] = lda_profile(rec, readouts)
        fams[fname] = rec
    if linear and geometry == 'constant':                               # per-mode gains of pure cos / sin drives
        for n_freq in (1, 8):
            g = mode_gains(n_freq, L, gamma, pot[1], drive)
            t = 2 * np.pi * np.arange(N1) / N1
            for pattern in (np.cos(n_freq * t), np.sin(n_freq * t)):
                u, _, _ = mod.settle(pattern)
                dev = max(float(np.max(np.abs(u[disk.layer(l)] - g[l - 1] * pattern))) for l in range(1, L + 1))
                gates['mode_gains'] &= dev <= 1e-10
                gates['mode_gain_max_deviation'] = max(gates['mode_gain_max_deviation'], dev)
    eligible = all(f['certified']['train'] + f['certified']['test'] == CLASSES * DRAWS for f in fams.values())
    sup, sup_ok = v1.superposition(mod, nuis, alpha, L)
    gates['superposition_linear'] = sup_ok
    rec = dict(key=config_key(geometry, drive, pot, L, gamma, alpha), geometry=geometry, drive=drive, L=L,
               gamma=gamma, potential=Potential(*pot).name, kind=pot[0], alpha=alpha, units=disk.n,
               ring_sizes=disk.sizes, eligible=eligible,
               meets_profile_nm=eligible and all(all(f['profile_nm'].values()) for f in fams.values()),
               meets_profile_lda=eligible and all(all(f['profile_lda'].values()) for f in fams.values()),
               families=fams, multistability=v1.multistability(mod, nuis, alpha, L), superposition=sup,
               settling=dict(worst_grad=worst['grad'], worst_min_eig=worst['min_eig']))
    return rec, gates


# ---------------------------------------------------------------- gates independent of the scored grid

def graph_gate():
    ok = v1.graph_gate()
    for L in DEPTHS:
        for g in GAMMAS:
            d = make_disk('constant', L, g)
            ok &= d.sizes == [N1] * L
            edges = [e for _, e in d.lateral] + d.radial
            ok &= d.connected() and len(edges) == len(set(edges)) and np.allclose(d.K.sum(axis=1), 0)
            ring_radial = {(i, j) for i, j in d.radial if j != d.centre}
            ok &= ring_radial == {(d.offsets[l] + j, d.offsets[l + 1] + j) for l in range(L - 1) for j in range(N1)}
            ok &= sum(1 for i, j in d.radial if j == d.centre) == N1
    return bool(ok)


def derivative_gate():
    worst = 0.0
    for geometry in GEOMETRIES:
        for L in DEPTHS:
            for g in GAMMAS:
                for idx, (drive, pot) in enumerate(DRIVE_POTENTIALS):
                    mod = make_module(make_disk(geometry, L, g), Potential(*pot), drive)
                    rng = np.random.default_rng([SEED_GATES, GEOMETRIES.index(geometry), L, g, idx])
                    u, h = rng.standard_normal(mod.disk.n), rng.standard_normal(N1)
                    e = np.eye(mod.disk.n) * 1e-6
                    fd_g = np.array([(mod.energy(u + e[i], h) - mod.energy(u - e[i], h)) / 2e-6
                                     for i in range(mod.disk.n)])
                    fd_h = np.array([(mod.grad(u + e[i], h) - mod.grad(u - e[i], h)) / 2e-6
                                     for i in range(mod.disk.n)])
                    worst = max(worst, float(np.max(np.abs(fd_g - mod.grad(u, h)))),
                                float(np.max(np.abs(fd_h - mod.hess(u)))))
    return worst <= 1e-5, worst


def lda_gate():
    """Cholesky scores match an explicit inverse; accuracy invariant under an invertible feature transform."""
    rng = np.random.default_rng(SEED_GATES)
    p, n_tr, n_te = 6, 25, 25
    centres = rng.standard_normal((CLASSES, p)) * 1.5
    tr = np.vstack([centres[c] + rng.standard_normal((n_tr, p)) for c in range(CLASSES)])
    te = np.vstack([centres[c] + rng.standard_normal((n_te, p)) for c in range(CLASSES)])
    tl, el = np.repeat(np.arange(CLASSES), n_tr), np.repeat(np.arange(CLASSES), n_te)
    ok_tr, ok_te = np.ones(len(tr), bool), np.ones(len(te), bool)
    diff = float(np.max(np.abs(lda_cholesky_scores(tr, tl, ok_tr, te) - lda_explicit_scores(tr, tl, ok_tr, te))))
    A = rng.standard_normal((p, p)) + 3 * np.eye(p)
    a1, _ = lda(tr, tl, ok_tr, te, el, ok_te)
    a2, _ = lda(tr @ A.T, tl, ok_tr, te @ A.T, el, ok_te)
    return diff <= 1e-9 and a1['strict'] == a2['strict'], dict(score_max_difference=diff,
                                                              accuracy=a1['strict'], transformed=a2['strict'])


V1_COMPARED = ('certified', 'S', 'P', 'suppression_ratio', 'D')


def replication_gate(configs):
    """Shrinking-geometry, tie-drive configurations reproduce v1's reported values exactly after rounding."""
    old = {(c['L'], c['gamma'], c['potential'], c['alpha']): c for c in json.loads((ROOT / V1_RESULTS).read_text())['configs']}
    compared, mismatches = 0, []
    for c in configs:
        if c['geometry'] != 'shrinking' or c['drive'] != 'tie':
            continue
        o = old[(c['L'], c['gamma'], c['potential'], c['alpha'])]
        new = v1.rnd(c)
        for fname in new['families']:
            for k in V1_COMPARED:
                if new['families'][fname][k] != o['families'][fname][k]:
                    mismatches.append(f"{c['key']}/{fname}/{k}")
        for k in ('multistability', 'superposition', 'eligible'):
            if new[k] != o[k]:
                mismatches.append(f"{c['key']}/{k}")
        compared += 1
    return compared == 80 and not mismatches, dict(compared=compared, mismatches=mismatches)


# ---------------------------------------------------------------- readings and summary

def _layer(c, fname, key, l):
    seq = c['families'][fname]['readouts'][key]
    return seq[l - 1]


def _avail(entry, readout):
    return entry is not None and entry.get(readout) is not None and entry[readout]['null'] is None


def _not_evaluable(row, reasons):
    return dict(row, outcome='not_evaluable', reasons=reasons)


def reading_g(idx):
    rows = []
    for drive, pot in DRIVE_POTENTIALS:
        name, linear = Potential(*pot).name, pot[0] == 'linear'
        for L in DEPTHS:
            for g in GAMMAS:
                for a in ALPHAS:
                    con = idx[config_key('constant', drive, pot, L, g, a)]
                    shr = idx[config_key('shrinking', drive, pot, L, g, a)]
                    for fname, _ in FAMILIES:
                        for l in range(2, L + 1):
                            row = dict(drive=drive, potential=name, L=L, gamma=g, alpha=a, family=fname, layer=l)
                            reasons = [f'{x["geometry"]}_ineligible' for x in (con, shr) if not x['eligible']]
                            ec, es = _layer(con, fname, READING_REL, l), _layer(shr, fname, READING_REL, l)
                            reasons += [f'{n}_relative_scale_null' for n, e in (('constant', ec), ('shrinking', es))
                                        if e is None]
                            reasons += [f'{n}_lda_null' for n, e in (('constant', ec), ('shrinking', es))
                                        if e is not None and e['lda']['null'] is not None]
                            if linear:
                                reasons += [f'{n}_reference_missing' for n, e in (('constant', ec), ('shrinking', es))
                                            if e is not None and e['reference'] is None]
                            if reasons:
                                rows.append(_not_evaluable(row, reasons))
                                continue
                            d_lda = ec['lda']['strict'] - es['lda']['strict']
                            row.update(lda_constant=ec['lda']['strict'], lda_shrinking=es['lda']['strict'])
                            if linear:
                                d_ref = ec['reference']['accuracy'] - es['reference']['accuracy']
                                row.update(reference_constant=ec['reference']['accuracy'],
                                           reference_shrinking=es['reference']['accuracy'])
                                row['outcome'] = ('shown' if d_ref >= READING_MARGIN and d_lda >= READING_MARGIN
                                                  else 'predicted_not_confirmed' if d_ref >= READING_MARGIN
                                                  else 'not_shown')
                            else:
                                row['outcome'] = ('consistent_with_geometric_loss' if d_lda >= READING_MARGIN
                                                  else 'not_indicated')
                            rows.append(row)
    return rows


def reading_a(idx):
    rows = []
    for drive, pot in DRIVE_POTENTIALS:
        if pot[0] != 'linear':
            continue
        for L in DEPTHS:
            for g in GAMMAS:
                for a in ALPHAS:
                    c = idx[config_key('constant', drive, pot, L, g, a)]
                    for fname, _ in FAMILIES:
                        row = dict(drive=drive, potential=Potential(*pot).name, L=L, gamma=g, alpha=a, family=fname)
                        ea, er = _layer(c, fname, READING_ABS, L), _layer(c, fname, READING_REL, L)
                        reasons = [] if c['eligible'] else ['ineligible']
                        if er is None:
                            reasons.append('relative_scale_null')
                        reasons += [f'{n}_lda_null' for n, e in (('absolute', ea), ('relative', er))
                                    if e is not None and e['lda']['null'] is not None]
                        if reasons:
                            rows.append(_not_evaluable(row, reasons))
                            continue
                        vals = dict(reference_absolute=ea['reference']['accuracy'],
                                    reference_relative=er['reference']['accuracy'],
                                    lda_absolute=ea['lda']['strict'], lda_relative=er['lda']['strict'])
                        suff = (vals['reference_absolute'] < T3_MIN and vals['reference_relative'] >= T3_MIN
                                and vals['lda_absolute'] < T3_MIN and vals['lda_relative'] >= T3_MIN)
                        rows.append(dict(row, **vals, outcome='sufficient' if suff else 'not_shown'))
    return rows


def reading_r(configs):
    rows = []
    for c in configs:
        for fname, _ in FAMILIES:
            for key in (READING_ABS, READING_REL):
                row = dict(config=c['key'], family=fname, noise=key)
                e = _layer(c, fname, key, c['L'])
                if not c['eligible'] or e is None:
                    rows.append(_not_evaluable(row, (['ineligible'] if not c['eligible'] else [])
                                               + (['relative_scale_null'] if e is None else [])))
                    continue
                sub = {}
                if _avail(e, 'lda'):
                    d = e['lda']['strict'] - e['nm']['strict']
                    sub['nm_limiting'] = dict(lda_minus_nm=d, outcome='limiting' if d >= READING_MARGIN else 'not_shown')
                    if _avail(e, 'projection'):
                        d = e['projection']['strict'] - e['lda']['strict']
                        sub['projection_vs_lda'] = dict(
                            projection_minus_lda=d,
                            outcome='projection_outperforms_lda' if d >= READING_MARGIN else 'not_shown')
                    else:
                        sub['projection_vs_lda'] = dict(outcome='not_evaluable', reasons=['projection_null'])
                    if c['kind'] == 'linear':
                        if _avail(e, 'oracle_lda'):
                            d = e['oracle_lda']['strict'] - e['lda']['strict']
                            sub['covariance_estimation'] = dict(
                                oracle_minus_lda=d,
                                outcome='covariance_estimation_limits_lda' if d >= READING_MARGIN else 'not_shown')
                        else:
                            sub['covariance_estimation'] = dict(outcome='not_evaluable', reasons=['oracle_null'])
                else:
                    sub = dict(outcome='not_evaluable', reasons=[f"lda_null:{e['lda']['null']}"])
                rows.append(dict(row, nm=e['nm']['strict'], lda=e['lda']['strict'],
                                 projection=e['projection']['strict'] if _avail(e, 'projection') else None, **sub))
    return rows


def tally(rows, field='outcome'):
    out, reasons = {}, {}
    for r in rows:
        out[r[field]] = out.get(r[field], 0) + 1
        for x in r.get('reasons', []):
            reasons[x] = reasons.get(x, 0) + 1
    return dict(outcomes=dict(sorted(out.items())), not_evaluable_reasons=dict(sorted(reasons.items())))


def summary(configs):
    idx = {c['key']: c for c in configs}
    profile = {}
    for c in configs:
        k = f"{c['kind']}/{c['geometry']}/{c['drive']}"
        p = profile.setdefault(k, dict(configs=0, eligible=0, meets_nm=0, meets_lda=0, failures_nm={},
                                       failures_lda={}))
        p['configs'] += 1
        p['eligible'] += c['eligible']
        p['meets_nm'] += c['meets_profile_nm']
        p['meets_lda'] += c['meets_profile_lda']
        for fname, f in c['families'].items():
            for which in ('nm', 'lda'):
                for t, v in f[f'profile_{which}'].items():
                    if not v:
                        d = p[f'failures_{which}']
                        d[f'{fname}/{t}'] = d.get(f'{fname}/{t}', 0) + 1
    g_rows, a_rows, r_rows = reading_g(idx), reading_a(idx), reading_r(configs)
    r_tally = {}
    for sub in ('nm_limiting', 'projection_vs_lda', 'covariance_estimation'):
        r_tally[sub] = tally([dict(outcome=r[sub]['outcome'], reasons=r[sub].get('reasons', []))
                              for r in r_rows if sub in r])
    r_tally['rows_not_evaluable'] = tally([r for r in r_rows if r.get('outcome') == 'not_evaluable'])
    estimation, prediction, field_tie, spectral = [], [], [], []
    for c in configs:
        if c['kind'] != 'linear':
            continue
        for fname, n_freq in FAMILIES:
            for key in (READING_ABS, READING_REL):
                e = _layer(c, fname, key, c['L'])
                if c['eligible'] and _avail(e, 'lda'):
                    estimation.append(dict(config=c['key'], family=fname, noise=key,
                                           reference=e['reference']['accuracy'], lda=e['lda']['strict'],
                                           gap=e['reference']['accuracy'] - e['lda']['strict']))
            if c['geometry'] == 'constant':
                e = _layer(c, fname, READING_ABS, c['L'])
                g = mode_gains(n_freq, c['L'], c['gamma'], float(c['potential'].split('=')[1]), c['drive'])[-1]
                cf, _ = closed_form_reference(g, c['alpha'], SIGMA_PROFILE)
                row = dict(config=c['key'], family=fname, closed_form=cf,
                           lda=e['lda']['strict'] if _avail(e, 'lda') else None)
                row['flag'] = ('lda_null' if row['lda'] is None else
                               'unexplained' if abs(row['lda'] - cf) > PREDICTION_TOLERANCE else 'within_tolerance')
                prediction.append(row)
            if c['alpha'] == 1:
                bands = []
                for spec in c['families'][fname]['spectrum']:
                    bands.append(None if spec is None else dict(
                        {f'k={k}': spec.get(str(k)) for k in (0, 1, 8)},
                        total=dict(P=sum(v['P'] for v in spec.values()), B=sum(v['B'] for v in spec.values()))))
                spectral.append(dict(config=c['key'], family=fname, layers=bands))
    for drive, pot in DRIVE_POTENTIALS:
        if drive != 'field':
            continue
        for geometry in GEOMETRIES:
            for L in DEPTHS:
                for g in GAMMAS:
                    for a in ALPHAS:
                        cf = idx[config_key(geometry, 'field', pot, L, g, a)]
                        ct = idx[config_key(geometry, 'tie', pot, L, g, a)]
                        for fname, _ in FAMILIES:
                            row = dict(geometry=geometry, potential=Potential(*pot).name, L=L, gamma=g, alpha=a,
                                       family=fname)
                            ef, et = _layer(cf, fname, READING_ABS, L), _layer(ct, fname, READING_ABS, L)
                            if not (cf['eligible'] and ct['eligible']):
                                field_tie.append(_not_evaluable(row, ['ineligible']))
                                continue
                            row['nm_field_minus_tie'] = ef['nm']['strict'] - et['nm']['strict']
                            row['lda_field_minus_tie'] = (ef['lda']['strict'] - et['lda']['strict']
                                                          if _avail(ef, 'lda') and _avail(et, 'lda') else None)
                            field_tie.append(row)
    return dict(configs=len(configs), eligible=sum(c['eligible'] for c in configs),
                meeting_profile_nm=sum(c['meets_profile_nm'] for c in configs),
                meeting_profile_lda=sum(c['meets_profile_lda'] for c in configs),
                stage2_selection=None, profile=profile,
                reading_G=dict(tally=tally(g_rows), rows=g_rows), reading_A=dict(tally=tally(a_rows), rows=a_rows),
                reading_R=dict(tally=r_tally, rows=r_rows), estimation_gap=estimation,
                prediction_check=dict(tally=tally(prediction, 'flag'), rows=prediction),
                field_vs_tie=field_tie, spectral=spectral)


def run_tests():
    names = ['tests.test_layered_module', 'tests.test_layered_module_v2']
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
                environment=(np.__version__, scipy.__version__) == (m['numpy'], m['scipy']),
                v1_results=sha(V1_RESULTS) == m['prior_artifacts'][V1_RESULTS])
    if not all(gate.values()):
        sys.exit(f'contract, environment or v1 results differ: {gate}')
    t0 = time.perf_counter()
    gate['graph'] = graph_gate()
    gate['derivatives'], deriv_worst = derivative_gate()
    gate['attenuation_sanity'], atten = v1.attenuation_gate()
    gate['lda_implementation'], lda_diag = lda_gate()
    nuis = nuisance_unit(SEED_NUISANCE, (len(FAMILIES), CLASSES, DRAWS))
    z_mc = np.random.default_rng(SEED_REFERENCE).standard_normal((MC_DRAWS, 2))
    configs = []
    agg = dict(linear_direct=True, superposition_linear=True, mode_gains=True, reference_procedure=True)
    diag = dict(linear_direct_max_deviation=0.0, mode_gain_max_deviation=0.0, reference_checks=0,
                reference_failures=[])
    for geometry in GEOMETRIES:
        for drive, pot in DRIVE_POTENTIALS:
            for L in DEPTHS:
                for g in GAMMAS:
                    for a in ALPHAS:
                        rec, gs = run_config(nuis, z_mc, geometry, drive, pot, L, g, a)
                        configs.append(rec)
                        agg['linear_direct'] &= gs['linear_direct']
                        agg['superposition_linear'] &= gs['superposition_linear']
                        agg['mode_gains'] &= gs['mode_gains']
                        agg['reference_procedure'] &= gs['reference_ok']
                        for k in ('linear_direct_max_deviation', 'mode_gain_max_deviation'):
                            diag[k] = max(diag[k], gs[k])
                        diag['reference_checks'] += gs['reference_checks']
                        diag['reference_failures'] += [dict(f, config=rec['key']) for f in gs['reference_failures']]
    gate.update(agg)
    gate['v1_replication'], repl = replication_gate(configs)
    seconds = time.perf_counter() - t0
    gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
    tests = run_tests()
    completion = completion_status(gate, tests)
    prov = dict(manifest_sha256=m['sha256'], contract_sha256=m['contract_sha256'],
                code={p: sha(p) for p in ('src/layered_module.py', 'src/layered_module_v2.py',
                                          'experiments/layered_module.py', 'experiments/layered_module_v2.py',
                                          'tests/test_layered_module.py', 'tests/test_layered_module_v2.py')})
    diagnostics = dict(derivative_max_error=deriv_worst, attenuation_rms_n1_n8=atten, lda=lda_diag,
                       v1_replication=repl, **diag)
    dump(out / f'{PREFIX}-results.json', v1.rnd(dict(
        experiment=PREFIX, completion=completion, correctness_gate=gate, gate_diagnostics=diagnostics,
        tests=tests, provenance=prov, summary=summary(configs), configs=configs)))
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds=dict(grid_and_gates=round(seconds, 3)), total=round(time.perf_counter() - started, 3),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_rss_scope='whole process',
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps({k: v for k, v in gate.items()}))


if __name__ == '__main__':
    main()
