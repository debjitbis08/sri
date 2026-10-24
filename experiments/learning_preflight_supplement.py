"""Supplementary preflight: asymptotics, anisotropic nuisance, resilience, identifiability.

Run: .venv/bin/python -m experiments.learning_preflight_supplement
Development calculations only: exact hypothesis-class equivalences, population
references, and Monte Carlo learning curves for spectral (non-neural) estimators.
No neural model is trained and no confirmation data are generated.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.coarse_interior_analysis import A, B, C, N, MID, CORE


BLOCK = N//CORE
SIGNAL_VAR = np.array([8., 6., 4., 2.])   # Cov(c) eigenvalues, per broad coordinate
BUDGETS = (4, 8, 16, 32, 64, 128)
DRAWS = 200
RETRIEVAL_SET = 64
TASKS = ('aligned_constant', 'aligned_profile', 'shifted_profile', 'dense')
NOISE = {'iso_0.25': (0.25, 0.), 'iso_1': (1., 0.), 'iso_4': (4., 0.),
         'aniso_1': (1., 12.)}


# ---------------------------------------------------------------- generators

def basis(task, rng):
    """Unknown 32x4 orthonormal U for one independently generated task."""
    u = np.zeros((N, CORE))
    if task == 'aligned_constant':
        return C.T.copy()
    if task == 'dense':
        return np.linalg.qr(rng.standard_normal((N, CORE)))[0]
    shift = 0 if task == 'aligned_profile' else BLOCK//2
    for col in range(CORE):
        profile = rng.standard_normal(BLOCK)
        sites = (np.arange(BLOCK)+col*BLOCK+shift) % N
        u[sites, col] = profile/np.linalg.norm(profile)
    return u


def nuisance_basis(u, rng, rank=4):
    """Directions orthogonal to the signal carrying anisotropic detail."""
    raw = rng.standard_normal((N, rank))
    raw -= u@(u.T@raw)
    return np.linalg.qr(raw)[0]


def sample_pairs(u, v, noise, tau2, n, rng, distractor=None):
    c = rng.standard_normal((n, CORE))*np.sqrt(SIGNAL_VAR)
    shared = c@u.T
    if distractor is not None:
        vd, var = distractor
        shared = shared+(rng.standard_normal((n, vd.shape[1]))*np.sqrt(var))@vd.T
    views = []
    for _ in range(2):
        d = rng.standard_normal((n, N))*np.sqrt(noise)
        if tau2:
            d += (rng.standard_normal((n, v.shape[1]))*np.sqrt(tau2))@v.T
        views.append(shared+d)
    return c, views[0], views[1]


def detail_cov(v, noise, tau2):
    cov = noise*np.eye(N)
    if tau2:
        cov += tau2*v@v.T
    return cov


# ---------------------------------------------------------------- estimators

def top(mat, k):
    vals, vecs = np.linalg.eigh(mat)
    return vecs[:, ::-1][:, :k]


def cross_cov(h1, h2):
    k = h1.T@h2/len(h1)
    return (k+k.T)/2


def dense_spectral(h1, h2):
    return top(cross_cov(h1, h2), CORE).T


def local_spectral(h1, h2):
    k = cross_cov(h1, h2)
    w = np.zeros((CORE, N))
    for b in range(CORE):
        s = slice(b*BLOCK, (b+1)*BLOCK)
        w[b, s] = top(k[s, s], 1)[:, 0]
    return w


def pooled_pca(h1, h2):
    pooled = np.vstack([h1, h2])
    return top(pooled.T@pooled/len(pooled), CORE).T


ESTIMATORS = dict(dense_spectral=dense_spectral, local_spectral=local_spectral,
                  pooled_pca_dense=pooled_pca, fixed_average=lambda h1, h2: C.copy())


# ---------------------------------------------------------------- metrics

def r2(w, u, cov_detail):
    """Population fraction of broad variance linearly decodable from w h.

    Invariant to any invertible remapping of the core; zero for collapsed w."""
    sc = np.diag(SIGNAL_VAR)
    cov_h = u@sc@u.T+cov_detail
    cross = sc@u.T@w.T
    return float(np.trace(cross@np.linalg.pinv(w@cov_h@w.T, rcond=1e-10)@cross.T)/SIGNAL_VAR.sum())


def subspace_error(w, u):
    """||sin Theta||_F^2 between row space of w and range(u)."""
    q = np.linalg.qr(w.T)[0]
    return float(np.linalg.norm(q@q.T-u@u.T)**2/2)


def retrieval(w, h1_train, h2_train, u, v, noise, tau2, rng):
    """Held-out top-1 view matching; whitening uses training views only."""
    _, t1, t2 = sample_pairs(u, v, noise, tau2, RETRIEVAL_SET, rng)
    train = np.vstack([h1_train, h2_train])@w.T
    train -= train.mean(0)
    cov = train.T@train/len(train)
    cov += 1e-3*np.trace(cov)/CORE*np.eye(CORE)
    white = np.linalg.cholesky(np.linalg.inv(cov))
    e1, e2 = t1@w.T@white, t2@w.T@white
    dist = ((e1[:, None, :]-e2[None, :, :])**2).sum(-1)
    return float(np.mean(np.argmin(dist, 1) == np.arange(RETRIEVAL_SET)))


# ---------------------------------------------------------------- section 1: equivalences

def local_factor(w):
    """Factor a block-local 4x32 encoder as B' A' with 4-site A' and pair B'."""
    a, b = np.zeros((MID, N)), np.zeros((CORE, MID))
    for row in range(MID):
        seg = w[row//2, row*4:(row+1)*4]
        norm = np.linalg.norm(seg)
        a[row, row*4:(row+1)*4] = seg/norm if norm else 0
        b[row//2, row] = norm
    return a, b


def equilibrium(a, b, h, lam=1., kappa=1., eta=0., p=None):
    n, m, k = a.shape[1], a.shape[0], b.shape[0]
    hess = np.zeros((n+m+k, n+m+k))
    hess[:n, :n] = lam*np.eye(n)+kappa*a.T@a
    hess[n:n+m, n:n+m] = kappa*(np.eye(m)+b.T@b)
    hess[n+m:, n+m:] = (kappa+eta)*np.eye(k)
    hess[:n, n:n+m] = -kappa*a.T
    hess[n:n+m, :n] = -kappa*a
    hess[n:n+m, n+m:] = -kappa*b.T
    hess[n+m:, n:n+m] = -kappa*b
    force = np.zeros(n+m+k)
    force[:n] = lam*h
    if p is not None:
        force[n+m:] = eta*p
    s = np.linalg.solve(hess, force)
    return s[:n], s[n:n+m], s[n+m:]


def equivalences(rng):
    out = {}
    # (a) local compact 32->4 class equals 32->8->4 disjoint-local class.
    errs = []
    for _ in range(100):
        w = np.zeros((CORE, N))
        for b in range(CORE):
            w[b, b*BLOCK:(b+1)*BLOCK] = rng.standard_normal(BLOCK)
        a, bb = local_factor(w)
        errs.append(np.max(np.abs(bb@a-w)))
    out['local_compact_equals_hierarchy_max_error'] = float(max(errs))
    out['parameters'] = dict(dense_32_4=N*CORE, local_compact_32_4=N, hierarchy_32_8_4=N+MID,
                             hierarchy_redundant_scale_parameters=MID)
    # (b) settled reciprocal module equals skip-connected feed-forward map,
    #     for arbitrary (non-orthonormal) learned local A, B and any lambda, kappa.
    errs = []
    for _ in range(100):
        a, bb = local_factor(rng.standard_normal((CORE, N))*np.kron(np.eye(CORE), np.ones(BLOCK)))
        a *= rng.uniform(.2, 3, (MID, 1)); bb *= rng.uniform(.2, 3, (CORE, 1))
        h = rng.standard_normal(N)
        lam, kappa = rng.uniform(.1, 5, 2)
        x, z, y = equilibrium(a, bb, h, lam, kappa)
        errs.append(max(np.max(np.abs(x-h)), np.max(np.abs(z-a@h)), np.max(np.abs(y-bb@a@h))))
    out['equilibrium_equals_feedforward_max_error'] = float(max(errs))
    # (c) gradient of a paired-view loss through the equilibrium equals the
    #     feed-forward chain-rule gradient (central finite differences).
    a, bb = local_factor(rng.standard_normal((CORE, N))*np.kron(np.eye(CORE), np.ones(BLOCK)))
    h1, h2 = rng.standard_normal((2, 6, N))
    mask_a, mask_b = a != 0, bb != 0

    def loss_eq(a_, b_):
        ys = [np.array([equilibrium(a_, b_, h)[2] for h in hv]) for hv in (h1, h2)]
        return np.mean(np.sum((ys[0]-ys[1])**2, 1))-np.log(np.linalg.det(ys[0].T@ys[0]/6+np.eye(CORE)))

    y1, y2 = h1@a.T@bb.T, h2@a.T@bb.T
    gy = 2*(y1-y2)/6
    inv = np.linalg.inv(y1.T@y1/6+np.eye(CORE))
    g1, g2 = gy-2*y1@inv/6, -gy
    grad_b = g1.T@(h1@a.T)+g2.T@(h2@a.T)
    grad_a = bb.T@g1.T@h1+bb.T@g2.T@h2
    eps, worst = 1e-6, 0.
    for mat, grad, mask, which in ((a, grad_a, mask_a, 'a'), (bb, grad_b, mask_b, 'b')):
        for idx in zip(*np.nonzero(mask)):
            plus, minus = mat.copy(), mat.copy()
            plus[idx] += eps; minus[idx] -= eps
            if which == 'a':
                fd = (loss_eq(plus, bb)-loss_eq(minus, bb))/(2*eps)
            else:
                fd = (loss_eq(a, plus)-loss_eq(a, minus))/(2*eps)
            worst = max(worst, abs(fd-grad[idx]))
    out['equilibrium_gradient_equals_feedforward_max_error'] = float(worst)
    # (d) with retained context (eta>0) the map is no longer the feed-forward map.
    h, p = rng.standard_normal(N), rng.standard_normal(CORE)
    x, _, y = equilibrium(A, B, h, 1., 1., 1., p)
    out['context_eta1_surface_departure_from_input'] = float(np.linalg.norm(x-h))
    out['context_eta1_core_departure_from_Ch'] = float(np.linalg.norm(y-C@h))
    assert out['local_compact_equals_hierarchy_max_error'] < 1e-12
    assert out['equilibrium_equals_feedforward_max_error'] < 1e-10
    assert out['equilibrium_gradient_equals_feedforward_max_error'] < 1e-6
    return out


# ---------------------------------------------------------------- section 2: population references

def local_ceiling(u, cov_detail, rng, restarts=4):
    """Best population r2 over block-local encoders (numerical optimum)."""
    def neg(params):
        w = np.zeros((CORE, N))
        for b in range(CORE):
            w[b, b*BLOCK:(b+1)*BLOCK] = params[b*BLOCK:(b+1)*BLOCK]
        return -r2(w, u, cov_detail)
    best = max(-minimize(neg, rng.standard_normal(N), method='L-BFGS-B').fun for _ in range(restarts))
    return float(best)


def dense_ceiling(u, cov_detail):
    """Best linear 4-dim encoder: generalized eigenproblem, exact."""
    sc = np.diag(SIGNAL_VAR)
    cov_h = u@sc@u.T+cov_detail
    num = u@sc@sc@u.T
    l = np.linalg.cholesky(cov_h)
    li = np.linalg.inv(l)
    vals = np.linalg.eigvalsh(li@num@li.T)[::-1]
    return float(vals[:CORE].sum()/SIGNAL_VAR.sum())


def asymptotic_subspace_error(n, noise, orth_dims):
    """First-order E||sinTheta||_F^2 for the symmetrized cross-covariance estimator,
    Gaussian c and isotropic Gaussian detail: (1/n) sum_i orth (l_i+s)s/(2 l_i^2)."""
    return float(np.sum(orth_dims*(SIGNAL_VAR+noise)*noise/(2*SIGNAL_VAR**2))/n)


def population(rng):
    out = {}
    for task in TASKS:
        rows = {}
        for key, (noise, tau2) in NOISE.items():
            vals = dict(dense=[], local=[], fixed=[], local_spectral_population=[],
                        dense_spectral_population=[], pca_population=[])
            for _ in range(10 if task != 'aligned_constant' else 1):
                u = basis(task, rng)
                v = nuisance_basis(u, rng)
                cd = detail_cov(v, noise, tau2)
                vals['dense'].append(dense_ceiling(u, cd))
                vals['local'].append(local_ceiling(u, cd, rng))
                vals['fixed'].append(r2(C, u, cd))
                kpop = u@np.diag(SIGNAL_VAR)@u.T   # population cross-covariance
                wl = np.zeros((CORE, N))
                for b in range(CORE):
                    s = slice(b*BLOCK, (b+1)*BLOCK)
                    wl[b, s] = top(kpop[s, s], 1)[:, 0]
                vals['local_spectral_population'].append(r2(wl, u, cd))
                vals['dense_spectral_population'].append(r2(top(kpop, CORE).T, u, cd))
                vals['pca_population'].append(r2(top(kpop+cd, CORE).T, u, cd))
            rows[key] = {k: float(np.mean(x)) for k, x in vals.items()}
        out[task] = rows
    return out


# ---------------------------------------------------------------- section 3: learning curves

def learning_curves(rng):
    out = {}
    for task in TASKS:
        out[task] = {}
        for key, (noise, tau2) in NOISE.items():
            res = {name: {str(n): dict(r2=[], sub=[], ret=[], r2_noise4x=[], r2_new_detail=[])
                          for n in BUDGETS} for name in ESTIMATORS}
            for _ in range(DRAWS):
                u = basis(task, rng)
                v = nuisance_basis(u, rng)
                cd = detail_cov(v, noise, tau2)
                # Changed nuisance: 4x isotropic detail. New detail: extra variance
                # only orthogonal to the true signal (fresh fine variation).
                cd_noise = detail_cov(v, 4*noise, tau2)
                cd_detail = cd+4*noise*(np.eye(N)-u@u.T)
                for n in BUDGETS:
                    _, h1, h2 = sample_pairs(u, v, noise, tau2, n, rng)
                    for name, est in ESTIMATORS.items():
                        w = est(h1, h2)
                        cell = res[name][str(n)]
                        cell['r2'].append(r2(w, u, cd))
                        cell['sub'].append(subspace_error(w, u))
                        cell['ret'].append(retrieval(w, h1, h2, u, v, noise, tau2, rng))
                        cell['r2_noise4x'].append(r2(w, u, cd_noise))
                        cell['r2_new_detail'].append(r2(w, u, cd_detail))
            summary = {}
            for name, per in res.items():
                summary[name] = {}
                for n, cell in per.items():
                    summary[name][n] = {}
                    for metric, xs in cell.items():
                        xs = np.array(xs)
                        summary[name][n][metric] = dict(
                            mean=float(xs.mean()), se=float(xs.std(ddof=1)/np.sqrt(len(xs))))
                summary[name]['auc_r2_vs_log2n'] = float(np.mean(
                    [np.mean(per[str(n)]['r2']) for n in BUDGETS]))
            # Paired difference local - dense on the same training sets.
            paired = {}
            for n in BUDGETS:
                d = np.array(res['local_spectral'][str(n)]['r2'])-np.array(res['dense_spectral'][str(n)]['r2'])
                paired[str(n)] = dict(mean=float(d.mean()), se=float(d.std(ddof=1)/np.sqrt(len(d))))
            summary['paired_local_minus_dense_r2'] = paired
            if tau2 == 0 and task in ('aligned_constant', 'aligned_profile'):
                summary['asymptotic_subspace_error'] = {
                    str(n): dict(dense=asymptotic_subspace_error(n, noise, N-CORE),
                                 local=asymptotic_subspace_error(n, noise, BLOCK-1))
                    for n in BUDGETS}
            out[task][key] = summary
    return out


# ---------------------------------------------------------------- section 4: identifiability and VICReg feasibility

def persistent_distractor(rng):
    """A component shared by both views is indistinguishable from signal under pairing."""
    out = {}
    for var in (0., 1., 4., 10.):
        vals = []
        for _ in range(20):
            u = basis('dense', rng)
            vd = nuisance_basis(u, rng, rank=2)
            kpop = u@np.diag(SIGNAL_VAR)@u.T+var*vd@vd.T
            w = top(kpop, CORE).T
            cd = np.eye(N)+var*vd@vd.T   # distractor is nuisance for the signal probe
            vals.append(r2(w, u, cd))
        out[str(var)] = float(np.mean(vals))
    return out


def vicreg_batch_rank():
    """Per-branch centered batch covariance rank versus the 4-dim embedding."""
    return {str(n): dict(max_rank=min(n-1, CORE), full_rank_possible=bool(n-1 >= CORE))
            for n in BUDGETS}


def threshold_n(curve, target):
    """Interpolate on log2(n); None if never reached within budgets."""
    pts = [(np.log2(int(n)), curve[n]['r2']['mean']) for n in map(str, BUDGETS)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= target:
            return float(2**x0)
        if y1 >= target:
            return float(2**(x0+(target-y0)/(y1-y0)*(x1-x0)))
    return None


def main():
    rng = np.random.default_rng(20261024)
    result = dict(status='development preflight; spectral estimators only; not a confirmation',
                  signal_variances=SIGNAL_VAR.tolist(), budgets=list(BUDGETS), draws=DRAWS,
                  noise_conditions={k: dict(isotropic_variance=a, anisotropic_variance=b,
                                            anisotropic_rank=4) for k, (a, b) in NOISE.items()})
    result['equivalences'] = equivalences(rng)
    print('Equivalences:', result['equivalences'])
    result['population'] = population(rng)
    print('Population ceilings done')
    curves = learning_curves(rng)
    result['learning_curves'] = curves
    thresholds = {}
    for task in TASKS:
        thresholds[task] = {}
        for key in NOISE:
            target = .9*result['population'][task][key]['dense']
            thresholds[task][key] = dict(target_r2=target, **{
                name: threshold_n(curves[task][key][name], target) for name in ESTIMATORS})
    result['examples_to_90pct_of_dense_ceiling'] = thresholds
    result['persistent_distractor_population_r2'] = persistent_distractor(rng)
    result['vicreg_per_branch_covariance_rank'] = vicreg_batch_rank()
    path = Path('reports/single-module-learning-preflight-supplement.json')
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('Wrote', path)
    for task in TASKS:
        for key in NOISE:
            row = curves[task][key]
            pop = result['population'][task][key]
            print(f"{task:17s} {key:9s} ceil dense {pop['dense']:.3f} local {pop['local']:.3f} "
                  f"fixed {pop['fixed']:.3f}")
            for name in ESTIMATORS:
                print('   ', f'{name:17s}', ' '.join(f"{row[name][str(n)]['r2']['mean']:.3f}"
                                                 for n in BUDGETS),
                      '| ret', ' '.join(f"{row[name][str(n)]['ret']['mean']:.2f}" for n in BUDGETS))
            print('    thresholds', thresholds[task][key])
    print('Distractor:', result['persistent_distractor_population_r2'])


if __name__ == '__main__':
    main()
