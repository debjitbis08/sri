"""Relational linear preflight v1: can the weak-prior SRI module represent shared action-conditioned linear dynamics?

Implements docs/relational-linear-preflight-v1-contract.md. Three 8 -> 4 -> 2 modules form a
two-step window (x_t, x_t+1, x_t+2) with per-action inner translation banks at both couplings.
Rendering, nuisance, operators and probe are evaluator-only, except in the privileged P1
capacity searches, which the contract declares. Stages:

P0 generator audit; P1 capacity witness (constructive, privileged multi-horizon search, one-step
diagnostic); P2 binding invariance with frozen banks; P3 exact versus settled inference; P4
direct-reference design audit.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import contextlib
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/relational-linear-preflight-v1-contract.md'
PREFIX = Path(os.environ.get('RLP_PREFIX', str(ROOT / 'reports/relational-linear-preflight-v1')))
DRAFT = ROOT / 'reports/relational-linear-preflight-v1-draft-checks.json'
WORLDS = tuple(int(s) for s in os.environ.get('RLP_WORLDS', ','.join(map(str, range(10)))).split(','))
STEPS = int(os.environ.get('RLP_STEPS', '2000'))  # smoke runs override; the contract fixes 2000
WORKERS = int(os.environ.get('RLP_WORKERS', str(max(1, (os.cpu_count() or 1) - 1))))
LAM, NOISE, NUIS = 1e-3, .05, .25
KAPPAS = (1., 5.)
CLIPS = {'clip2': (2., 3.), 'clip6': (6., 6.)}  # (W and V, T)
FAMILIES = ('A', 'B')
ETAS = (.001, .003, .01)
HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)
FIT_HORIZONS = (1, 2, 8, 32, 128)
BUDGETS = (4, 8, 16, 32, 64, 128, 256)
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))  # action pairs (a_t, a_t+1); pair index 2 * a + b
N_FIT, N_SEL, N_CAL, N_TEST, N_PFIT, N_PSEL, N_SEQ = 256, 128, 128, 256, 256, 128, 8
RESTARTS, ALS_SWEEPS, ALS_INITS, RIDGE = 3, 200, 5, 1e-6
COUNTS = {}


def count(key, n=1):
    COUNTS[key] = COUNTS.get(key, 0) + n


# ---------------------------------------------------------------- generator

def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def refl(a):
    return np.array([[np.cos(2 * a), np.sin(2 * a)], [np.sin(2 * a), -np.cos(2 * a)]])


def rendering(rng):
    while True:
        R = 2 * rng.normal(size=(8, 2)) / np.sqrt(8)
        B = rng.normal(size=(8, 2)) / np.sqrt(8)
        sv = np.linalg.svd(np.hstack([R, B]), compute_uv=False)
        if sv.min() >= .3 and sv.max() <= 3:
            return R, B


def make_world(w):
    rng = np.random.default_rng([w, 0])
    theta, phi = rng.uniform(np.pi / 8, 3 * np.pi / 8), rng.uniform(0, np.pi)
    phi1, delta = rng.uniform(0, np.pi), rng.uniform(np.pi / 8, 3 * np.pi / 8)
    shift = rng.choice((-1, 1)) * rng.uniform(np.pi / 12, np.pi / 6)
    domains = {d: rendering(rng) for d in ('s1', 's2', 't')}
    angle = rng.uniform(0, 2 * np.pi)
    return {'seed': w, 'theta': theta, 'phi': phi, 'phi1': phi1, 'delta': delta, 'shift': shift,
            'u': np.array([np.cos(angle), np.sin(angle)]), 'domains': domains,
            'ops': {'A': (rot(theta), refl(phi)), 'B': (refl(phi1), refl(phi1 + delta)),
                    'broken': (rot(theta + shift), refl(phi))}}


def observe(rng, R, B, z):
    n = rng.normal(scale=np.sqrt(NUIS), size=z.shape[:-1] + (2,))
    e = rng.normal(scale=NOISE, size=z.shape[:-1] + (8,))
    return z @ R.T + n @ B.T + e


def windows(world, ops, domain, n, stream):
    """n windows (x_t, a_t, x_t+1, a_t+1, x_t+2) with probes; z is evaluator-only."""
    rng = np.random.default_rng([world['seed'], stream])
    R, B = world['domains'][domain]
    F = np.stack(ops)
    z0 = rng.normal(size=(n, 2))
    a = rng.integers(0, 2, size=(n, 2))
    z1 = np.einsum('nij,nj->ni', F[a[:, 0]], z0)
    z2 = np.einsum('nij,nj->ni', F[a[:, 1]], z1)
    z = np.stack([z0, z1, z2], 1)
    return {'x': observe(rng, R, B, z), 'a': a, 'y': z @ world['u'], 'z': z}


def gain_matrix(world, domain):
    R, B = world['domains'][domain]
    cov = R @ R.T + NUIS * B @ B.T + NOISE ** 2 * np.eye(8)
    return R.T @ np.linalg.inv(cov)  # E[z0 | x0] = K x0


def sequences(rng, n, horizons, mixed=N_SEQ):
    """Per state: `mixed` random sequences plus (if mixed == N_SEQ and pure) the two pure ones."""
    out = {}
    for L in horizons:
        s = rng.integers(0, 2, size=(n, mixed, L))
        out[L] = s
    return out


def test_set(world, ops, domain, stream, n=N_TEST, horizons=HORIZONS, pure=True):
    rng = np.random.default_rng([world['seed'], stream])
    R, B = world['domains'][domain]
    z0 = rng.normal(size=(n, 2))
    x0 = observe(rng, R, B, z0)
    seqs = {}
    for L in horizons:
        s = rng.integers(0, 2, size=(n, N_SEQ, L))
        if pure:
            s = np.concatenate([s, np.zeros((n, 1, L), int), np.ones((n, 1, L), int)], 1)
        seqs[L] = s
    K = gain_matrix(world, domain)
    zhat = x0 @ K.T
    F = np.stack(ops)
    target, ytarget = {}, {}
    for L, s in seqs.items():
        z = np.repeat(zhat[:, None, :], s.shape[1], 1)
        for k in range(L):
            z = np.einsum('nmij,nmj->nmi', F[s[:, :, k]], z)
        target[L], ytarget[L] = z @ R.T, z @ world['u']
    return {'x0': x0, 'seqs': seqs, 'target': target, 'ytarget': ytarget}


# ---------------------------------------------------------------- module (torch, float64)

def block(W, V, lam=LAM):
    """Hessian of 1/2|x-Wh|^2 + 1/2|h-Vc|^2 + lam/2(|h|^2 + |c|^2) over (x, h, c)."""
    I8, I4, I2 = torch.eye(8), torch.eye(4), torch.eye(2)
    row0 = torch.cat([I8, -W, torch.zeros(8, 2)], 1)
    row1 = torch.cat([-W.T, W.T @ W + (1 + lam) * I4, -V], 1)
    row2 = torch.cat([torch.zeros(2, 8), -V.T, V.T @ V + lam * I2], 1)
    return torch.cat([row0, row1, row2], 0)


def window_q(W, V, T01, T12, kappa, lam=LAM):
    """42 x 42 window Hessian for one action pair."""
    q = torch.block_diag(block(W[0], V[0], lam), block(W[1], V[1], lam), block(W[2], V[2], lam))
    add = torch.zeros(42, 42)
    for (ma, mb), T in (((0, 1), T01), ((1, 2), T12)):
        a, b = 14 * ma + 12, 14 * mb + 12
        add = add + _place(kappa * torch.eye(2), b, b) + _place(kappa * T.T @ T, a, a) \
            - _place(kappa * T, b, a) - _place(kappa * T.T, a, b)
    return q + add


def _place(m, r, c):
    out = torch.zeros(42, 42)
    out[r:r + 2, c:c + 2] = m
    return out


FREE = torch.arange(8, 42)


def encoder(W, V, lam=LAM):
    q = block(W[0], V[0], lam)
    return torch.linalg.solve(q[8:, 8:], -q[8:, :8])[4:6]  # 2 x 8


def operators(p, kappa, lam=LAM):
    """Window maps and effective operators for the four action pairs.

    Returns Phi0 (2x8), M1, M2 (4x8x8), G1, G2 (4x2x2), N1, N2 (4x8x2), the coupled module-0
    core maps C0 (4x2x8) and factorization residuals.
    """
    W, V = p['W'], p['V']
    phi = encoder(W, V, lam)
    qs = torch.stack([window_q(W, V, p['T01'][a], p['T12'][b], kappa, lam) for a, b in PAIRS])
    count('window_factorizations', 4)
    s = torch.linalg.solve(qs[:, 8:, 8:], -qs[:, 8:, :8])  # free activities for unit inputs, (4, 34, 8)
    M1, M2, C0 = s[:, 6:14], s[:, 20:28], s[:, 4:6]
    pinv = phi.T @ torch.linalg.inv(phi @ phi.T)
    out = {'phi': phi, 'M1': M1, 'M2': M2, 'C0': C0}
    for k, M in (('1', M1), ('2', M2)):
        G = phi @ M @ pinv
        out['G' + k], out['N' + k] = G, M @ pinv
        out['res' + k] = torch.linalg.norm(phi @ M - G @ phi, dim=(1, 2)) / torch.linalg.norm(phi @ M, dim=(1, 2))
    return out


def pair_index(seq):
    return 2 * seq[..., 0::2] + seq[..., 1::2]


def rollout(ops, x0, seq):
    """Predicted observations and latent states at horizon L = seq.shape[-1], via the latent form."""
    z = torch.einsum('ij,nj->ni', ops['phi'], x0)
    z = z[:, None, :].expand(-1, seq.shape[1], -1)
    L = seq.shape[-1]
    if L == 1:
        idx = torch.as_tensor(2 * seq[..., 0] + seq[..., 0])  # filler second action = first
        return torch.einsum('nmij,nmj->nmi', ops['N1'][idx], z), torch.einsum('nmij,nmj->nmi', ops['G1'][idx], z)
    pairs = torch.as_tensor(pair_index(seq))
    for k in range(L // 2 - 1):
        z = torch.einsum('nmij,nmj->nmi', ops['G2'][pairs[..., k]], z)
    last = pairs[..., -1]
    return torch.einsum('nmij,nmj->nmi', ops['N2'][last], z), torch.einsum('nmij,nmj->nmi', ops['G2'][last], z)


def horizon_errors(ops, test, horizons, readout=None):
    x0 = torch.as_tensor(test['x0'])
    X, P = {}, {}
    for L in horizons:
        xh, zh = rollout(ops, x0, test['seqs'][L])
        tgt = torch.as_tensor(test['target'][L])
        X[L] = torch.sum((xh - tgt) ** 2) / torch.sum(tgt ** 2)
        if readout is not None:
            yt = torch.as_tensor(test['ytarget'][L])
            P[L] = torch.sum((zh @ readout - yt) ** 2) / torch.sum(yt ** 2)
    return X, P


def window_loss(ops, data):
    x = torch.as_tensor(data['x'])
    idx = torch.as_tensor(2 * data['a'][:, 0] + data['a'][:, 1])
    p1 = torch.einsum('nij,nj->ni', ops['M1'][idx], x[:, 0])
    p2 = torch.einsum('nij,nj->ni', ops['M2'][idx], x[:, 0])
    return (torch.sum((p1 - x[:, 1]) ** 2) + torch.sum((p2 - x[:, 2]) ** 2)) / len(x)


def fit_readout(phi, data):
    """Least-squares probe readout from phi(x) to y over all three window times."""
    x = data['x'].reshape(-1, 8)
    y = data['y'].reshape(-1)
    f = x @ np.asarray(phi).T
    return np.linalg.lstsq(f, y, rcond=None)[0]


def tparams(p, grad=False):
    return {k: torch.tensor(np.asarray(v), requires_grad=grad) for k, v in p.items()}


def nparams(p):
    return {k: v.detach().numpy().copy() for k, v in p.items()}


def project(p, clip):
    cw, ct = CLIPS[clip]
    with torch.no_grad():
        for k in ('W', 'V'):
            p[k].clamp_(-cw, cw)
        for k in ('T01', 'T12'):
            p[k].clamp_(-ct, ct)


def active_clips(p, clip):
    cw, ct = CLIPS[clip]
    return {k: int(np.sum(np.abs(np.asarray(p[k])) >= (cw if k in ('W', 'V') else ct) - 1e-12)) for k in p}


def adam(p, loss_fn, eta, clip, steps=STEPS, frozen=()):
    """Full-batch Adam with clip projection after each step; returns final params and loss trace ends."""
    t = tparams(p, grad=True)
    opt = torch.optim.Adam([v for k, v in t.items() if k not in frozen], lr=eta)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(t)
        if not torch.isfinite(loss):
            count('nonfinite_fits')
            return None, None, float('nan')
        loss.backward()
        opt.step()
        project(t, clip)
        count('adam_steps')
        first = float(loss.detach()) if first is None else first
    final = float(loss_fn(t).detach())
    return nparams(t), first, final


# ---------------------------------------------------------------- P0 exact mismatch

def vec_ops(Fp, F):
    """Stacked Sylvester operator S -> (S F'_a - F_a S)_a on row-major vec(S)."""
    return np.vstack([np.kron(np.eye(2), f.T) - np.kron(g, np.eye(2)) for f, g in zip(Fp, F)])


def mismatch(Fp, F, u=None):
    """Certified lower bound min_S sum|S F' - F S|^2 / sum|F|^2 (|S|_F = 1, or u^T S = u^T)."""
    A = vec_ops(Fp, F)
    norm = sum(float(np.sum(g * g)) for g in F)
    count('mismatch_solves')
    if u is None:
        _, sv, vt = np.linalg.svd(A)
        S = vt[-1].reshape(2, 2)
        bound = sv[-1] ** 2 / norm
    else:
        C = np.array([[u[0], 0, u[1], 0], [0, u[0], 0, u[1]]])
        kkt = np.block([[A.T @ A, C.T], [C, np.zeros((2, 2))]])
        s = np.linalg.lstsq(kkt, np.concatenate([np.zeros(4), u]), rcond=None)[0][:4]
        S = s.reshape(2, 2)
        bound = float(s @ A.T @ A @ s) / norm
    cond = float(np.linalg.cond(S))
    return {'bound': float(bound), 'minimizer_condition': cond if np.isfinite(cond) else 1e300}


def classify(m, threshold):
    if m['bound'] >= threshold:
        return 'detectable'
    if m['bound'] <= 1e-12 and m['minimizer_condition'] <= 1e6:
        return 'equivalent'
    return 'unresolved'


def commutant(F):
    A = vec_ops(F, F)
    sv = np.linalg.svd(A, compute_uv=False)
    return int(4 - np.sum(sv > 1e-10 * sv[0]))


def words(F, n):
    out = [np.eye(2)]
    frontier = [np.eye(2)]
    for _ in range(n):
        frontier = [f @ w for w in frontier for f in F]
        out += frontier
    return out


def p0(world):
    u = world['u']
    out = {'observability': {}, 'rendering_condition': {}, 'commutant': {}, 'scrambles': {}}
    for fam in ('A', 'B', 'broken'):
        F = world['ops'][fam]
        out['observability'][fam] = {
            'per_action': [int(np.linalg.matrix_rank(np.stack([u, F[a].T @ u]))) for a in (0, 1)],
            'mixed_to_4': int(np.linalg.matrix_rank(np.stack([u @ w for w in words(F, 4)])))}
        out['commutant'][fam] = commutant(F)
    out['commutant']['two_rotations'] = commutant([rot(world['theta']), rot(world['theta'] + .7)])
    for d, (R, B) in world['domains'].items():
        out['rendering_condition'][d] = float(np.linalg.cond(np.hstack([R, B])))
    z = np.random.default_rng([world['seed'], 1]).normal(size=(64, 2))
    drift = 0.
    for fam in ('A', 'B'):
        F = world['ops'][fam]
        zz = z.copy()
        for k in range(128):
            zz = zz @ F[k % 2].T
        drift = max(drift, float(np.max(np.abs(np.linalg.norm(zz, axis=1) / np.linalg.norm(z, axis=1) - 1))))
    out['precision_norm_drift_128'] = drift
    rng = np.random.default_rng([world['seed'], 2])
    for fam in ('A', 'B'):
        F = world['ops'][fam]
        sc = {'swap': {'J': mismatch(F[::-1], F), 'J_u': mismatch(F[::-1], F, u)}, 'same_eigenvalues': []}
        for _ in range(10):
            Ps = []
            for _ in (0, 1):
                while True:
                    P = rng.normal(size=(2, 2))
                    if np.linalg.cond(P) <= 10:
                        break
                Ps.append(P)
            Fp = [P @ f @ np.linalg.inv(P) for P, f in zip(Ps, F)]
            sc['same_eigenvalues'].append({'J': mismatch(Fp, F), 'J_u': mismatch(Fp, F, u)})
        out['scrambles'][fam] = sc
    out['scrambles']['broken'] = {'J': mismatch(world['ops']['broken'], world['ops']['A']),
                                  'J_u': mismatch(world['ops']['broken'], world['ops']['A'], u)}
    return out


# ---------------------------------------------------------------- P1 witness

def constructive(world, fam, kappa, clip):
    R, B = world['domains']['s1']
    cw, _ = CLIPS[clip]
    alpha = min(1., cw / np.abs(R).max())
    beta = cw / np.abs(B).max()
    W = np.hstack([alpha * R, beta * B])
    V = np.vstack([np.eye(2), np.zeros((2, 2))]) / alpha
    F = np.stack(world['ops'][fam])
    p = {'W': np.stack([W] * 3), 'V': np.stack([V] * 3), 'T01': F.copy(), 'T12': F.copy()}
    gain = lambda G: np.abs(np.linalg.eigvals(G)).mean(-1)
    with torch.no_grad():
        o = operators(tparams(p), kappa)
        g1 = gain(o['G1'].numpy())  # per pair
        p['T01'] = F / np.array([g1[0:2].mean(), g1[2:4].mean()])[:, None, None]
        o = operators(tparams(p), kappa)
        g2 = gain(o['G2'].numpy())
        p['T12'] = F / np.array([g2[[0, 2]].mean(), g2[[1, 3]].mean()])[:, None, None]
    project_np(p, clip)
    return p, {'alpha': alpha, 'beta': beta}


def project_np(p, clip):
    cw, ct = CLIPS[clip]
    for k in p:
        p[k] = np.clip(p[k], -(cw if k in ('W', 'V') else ct), cw if k in ('W', 'V') else ct)


def report(world, fam, p, kappa, clip, test, readout_data):
    R, B = world['domains']['s1'] if 'R' not in readout_data else readout_data['R']
    with torch.no_grad():
        o = operators(tparams(p), kappa)
        h = fit_readout(o['phi'].numpy(), readout_data['data'])
        X, P = horizon_errors(o, test, HORIZONS, torch.as_tensor(h))
    phi = o['phi'].numpy()
    sig = phi @ R
    sv = np.linalg.svd(sig, compute_uv=False)
    rank = int(np.sum(sv > 1e-8 * sv[0]))
    G1, G2 = o['G1'].numpy(), o['G2'].numpy()
    eig = {k: np.abs(np.linalg.eigvals(G)).tolist() for k, G in (('G1', G1), ('G2', G2))}
    comp = []
    for i, (a, b) in enumerate(PAIRS):
        Gb = G1[2 * b + b]
        comp.append(float(np.linalg.norm(G2[i] - Gb @ G1[i]) / np.linalg.norm(G2[i])))
    M1 = o['M1'].numpy()
    back = [float(np.linalg.norm(M1[2 * a + b] - M1[2 * a + 1 - b]) / np.linalg.norm(M1[2 * a + b])) for a, b in PAIRS]
    C0 = o['C0'].numpy()
    data = readout_data['data']
    ls_gap = []
    for i, (a, b) in enumerate(PAIRS):
        sel = (data['a'][:, 0] == a) & (data['a'][:, 1] == b)
        Mls = np.linalg.lstsq(data['x'][sel, 0], data['x'][sel, 1], rcond=None)[0].T
        ls_gap.append(float(np.linalg.norm(M1[i] - Mls) / np.linalg.norm(Mls)))
    return {
        'X': {str(L): float(v) for L, v in X.items()}, 'P': {str(L): float(v) for L, v in P.items()},
        'eig_moduli': eig, 'eig_mean_pow': {k: {str(L): float(np.mean(v) ** (L if k == 'G1' else L / 2)) for L in HORIZONS}
                                            for k, v in eig.items()},
        'factorization_residual': float(max(o['res1'].max(), o['res2'].max())),
        'composition_consistency': comp, 'back_reaction': back,
        'coupled_encoder_discrepancy': float(max(np.linalg.norm(C0[i] - phi) / np.linalg.norm(phi) for i in range(4))),
        'signal_singular_values': sv.tolist(), 'signal_rank': rank, 'signal_condition': float(sv[0] / sv[-1]),
        'relative_leakage': float(np.linalg.norm(np.linalg.solve(sig, phi @ B), 2)) if rank == 2 else None,
        'active_clips': active_clips(p, clip), 'one_step_ls_gap': ls_gap,
        'decay_penalty': float(sum(np.sum(np.asarray(p[k]) ** 2) for k in ('W', 'V'))),
        'readout': h.tolist()}


def passes_c1(rep):
    return all(rep['X'][str(L)] <= 1e-2 for L in HORIZONS) and rep['factorization_residual'] <= 1e-8


def p1_task(args):
    w, fam, kappa, clip = args
    t0 = time.process_time()
    COUNTS.clear()
    world = make_world(w)
    ops = world['ops'][fam]
    fstream = {'A': 10, 'B': 20}[fam]
    fit = windows(world, ops, 's1', N_FIT, fstream + 1)
    sel = windows(world, ops, 's1', N_SEL, fstream + 2)
    test = test_set(world, ops, 's1', fstream + 3)
    pfit = test_set(world, ops, 's1', fstream + 4, n=N_PFIT, horizons=FIT_HORIZONS, pure=False)
    psel = test_set(world, ops, 's1', fstream + 5, n=N_PSEL, horizons=FIT_HORIZONS, pure=False)
    rd = {'data': fit}
    start, meta = constructive(world, fam, kappa, clip)
    out = {'constructive': report(world, fam, start, kappa, clip, test, rd), 'construction': meta}
    multi = lambda t, d: sum(horizon_errors(operators(t, kappa), d, FIT_HORIZONS)[0].values())
    one = lambda t, d: window_loss(operators(t, kappa), d)
    for name, loss, fitd, seld in (('multi_horizon', multi, pfit, psel), ('one_step', one, fit, sel)):
        cands = {}
        for eta in ETAS:
            p, first, final = adam(start, lambda t: loss(t, fitd), eta, clip)
            if p is None:
                cands[eta] = {'eligible': False}
                continue
            with torch.no_grad():
                s = float(loss(tparams(p), seld))
            cands[eta] = {'eligible': np.isfinite(s), 'selection': s, 'first': first, 'final': final, 'p': p}
        ok = [e for e in ETAS if cands[e]['eligible']]
        if not ok:
            out[name] = {'failed': True}
            continue
        best = min(ok, key=lambda e: (cands[e]['selection'], e))
        out[name] = report(world, fam, cands[best]['p'], kappa, clip, test, rd)
        out[name].update({'eta': best, 'grid_edge': best in (ETAS[0], ETAS[-1]),
                          'selection': {str(e): cands[e].get('selection') for e in ETAS},
                          'objective_first_final': [cands[best]['first'], cands[best]['final']]})
        out[name]['params'] = {k: v.tolist() for k, v in cands[best]['p'].items()}
    out['constructive']['params'] = {k: v.tolist() for k, v in start.items()}
    # P3 on the multi-horizon result (or the constructive witness if the search failed).
    src = out['multi_horizon'].get('params', out['constructive']['params'])
    out['P3'] = p3({k: np.array(v) for k, v in src.items()}, kappa, test)
    out['counts'] = dict(COUNTS)
    out['cpu_seconds'] = time.process_time() - t0
    return (w, fam, kappa, clip), out


# ---------------------------------------------------------------- P3 settled inference

def p3(p, kappa, test, steps=32, cap=1024):
    with torch.no_grad():
        qs = torch.stack([window_q(torch.tensor(p['W']), torch.tensor(p['V']), torch.tensor(p['T01'][a]),
                                   torch.tensor(p['T12'][b]), kappa) for a, b in PAIRS]).numpy()
    H, f = qs[:, 8:, 8:], -qs[:, 8:, :8]
    exact = np.linalg.solve(H, f)
    trans = np.linalg.inv(np.eye(34)[None] + H)
    bias = trans @ f
    y = np.zeros_like(exact)
    err, need = None, None
    for k in range(1, cap + 1):
        y = trans @ y + bias
        e = float(np.max(np.linalg.norm(y - exact, axis=(1, 2)) / np.linalg.norm(exact, axis=(1, 2))))
        if k == steps:
            err, settled = e, y.copy()
        if need is None and e <= 1e-3:
            need = k
        if k >= steps and need is not None:
            break
    count('settle_steps', k)
    M2 = settled[:, 20:28]
    x0 = test['x0']
    X = {}
    for L in (2, 32):
        s = test['seqs'][L]
        pairs = pair_index(s)
        xh = np.repeat(x0[:, None, :], s.shape[1], 1)
        for j in range(L // 2):
            xh = np.einsum('nmij,nmj->nmi', M2[pairs[..., j]], xh)
        X[str(L)] = float(np.sum((xh - test['target'][L]) ** 2) / np.sum(test['target'][L] ** 2))
    return {'relative_activity_error_32': err, 'steps_to_1e-3': need, 'X_settled': X}


# ---------------------------------------------------------------- P2 binding invariance

def p2_task(args):
    (w, fam, kappa, clip), banks, eta, ref_G = args
    t0 = time.process_time()
    COUNTS.clear()
    world = make_world(w)
    out = {}
    fstream = {'A': 10, 'B': 20}[fam]
    for target in ('s2', 't', 'broken'):
        domain = 's2' if target == 's2' else 't'
        ops = world['ops']['broken' if target == 'broken' else fam]
        stream = fstream + {'s2': 30, 't': 40, 'broken': 50}[target]
        n = N_FIT if target == 's2' else N_CAL
        data = windows(world, ops, domain, n, stream + 1)
        test = test_set(world, ops, domain, stream + 3)
        best = None
        for r in range(RESTARTS):
            rng = np.random.default_rng([w, stream, r])
            init = {'W': rng.normal(scale=.1, size=(3, 8, 4)),
                    'V': np.stack([np.vstack([np.eye(2), np.zeros((2, 2))]) / 2] * 3),
                    'T01': banks['T01'], 'T12': banks['T12'], 'h': np.zeros(2)}
            x = torch.as_tensor(data['x'].reshape(-1, 8))
            y = torch.as_tensor(data['y'].reshape(-1))

            def loss(t):
                o = operators(t, kappa)
                probe = torch.sum((x @ o['phi'].T @ t['h'] - y) ** 2) / len(data['x'])
                return window_loss(o, data) + probe
            p, first, final = adam(init, loss, eta, clip, frozen=('T01', 'T12'))
            if p is not None and (best is None or final < best[1]):
                best = (p, final, r)
        if best is None:
            out[target] = {'failed': True}
            continue
        p = best[0]
        h = p.pop('h')
        with torch.no_grad():
            o = operators(tparams(p), kappa)
            X, P = horizon_errors(o, test, HORIZONS, torch.as_tensor(h))
        G = np.concatenate([o['G1'].numpy(), o['G2'].numpy()])
        out[target] = {'restart': best[2], 'training_objective': best[1],
                       'X': {str(L): float(v) for L, v in X.items()}, 'P': {str(L): float(v) for L, v in P.items()},
                       'J_G': mismatch(list(G), list(ref_G)),
                       'eig_moduli': np.abs(np.linalg.eigvals(G)).tolist(),
                       'factorization_residual': float(max(o['res1'].max(), o['res2'].max())),
                       'banks_unchanged': bool(np.array_equal(p['T01'], banks['T01']) and np.array_equal(p['T12'], banks['T12']))}
    out['counts'] = dict(COUNTS)
    out['cpu_seconds'] = time.process_time() - t0
    return (w, fam, kappa, clip), out


# ---------------------------------------------------------------- P4 direct references

def transitions(data):
    x = data['x']
    xs = np.concatenate([x[:, 0], x[:, 1]])
    xn = np.concatenate([x[:, 1], x[:, 2]])
    a = np.concatenate([data['a'][:, 0], data['a'][:, 1]])
    return xs, a, xn, x.reshape(-1, 8), data['y'].reshape(-1)


def latent_objective(m, ev):
    xs, a, xn, xo, yo = ev
    E, F, D, u = m['E'], m['F'], m['D'], m['u']
    v, vn = xs @ E.T, xn @ E.T
    Fv = np.einsum('nij,nj->ni', F[a], v)
    return float(np.sum((xn - Fv @ D.T) ** 2) + np.sum((xs - v @ D.T) ** 2) + np.sum((vn - Fv) ** 2)
                 + np.sum((yo - xo @ E.T @ u) ** 2)
                 + RIDGE * (np.sum(E ** 2) + np.sum(D ** 2) + np.sum(F ** 2) + np.sum(u ** 2)))


def als(ev, init, fixed=()):
    xs, a, xn, xo, yo = ev
    m = {k: v.copy() for k, v in init.items()}
    for _ in range(ALS_SWEEPS):
        count('als_sweeps')
        if 'F' not in fixed:
            v, vn = xs @ m['E'].T, xn @ m['E'].T
            for act in (0, 1):
                s = a == act
                # x' - D F v and E x' - F v, linear in column-major vec(F)
                A1 = np.einsum('ni,jk->njik', v[s], m['D']).reshape(s.sum() * 8, 4)
                A2 = np.einsum('ni,jk->njik', v[s], np.eye(2)).reshape(s.sum() * 2, 4)
                A = np.vstack([A1, A2])
                b = np.concatenate([xn[s].ravel(), vn[s].ravel()])
                vecF = np.linalg.solve(A.T @ A + RIDGE * np.eye(4), A.T @ b)
                m['F'][act] = vecF.reshape(2, 2, order='F')
        # E: all terms are linear in column-major vec(E) (16 unknowns)
        DF = m['D'][None] @ m['F'][a]  # (n, 8, 2)
        rows = [np.einsum('ni,njk->njik', xs, DF).reshape(-1, 16),
                np.einsum('ni,jk->njik', xs, m['D']).reshape(-1, 16),
                (np.einsum('ni,jk->njik', xn, np.eye(2)) - np.einsum('ni,njk->njik', xs, m['F'][a])).reshape(-1, 16),
                np.einsum('ni,k->nik', xo, m['u']).reshape(-1, 16)]
        A = np.vstack(rows)
        b = np.concatenate([xn.ravel(), xs.ravel(), np.zeros(len(xs) * 2), yo])
        m['E'] = np.linalg.solve(A.T @ A + RIDGE * np.eye(16), A.T @ b).reshape(2, 8, order='F')
        v = xs @ m['E'].T
        Fv = np.einsum('nij,nj->ni', m['F'][a], v)
        Z = np.vstack([Fv, v])
        T = np.vstack([xn, xs])
        m['D'] = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(2), Z.T @ T).T
        if 'u' not in fixed:
            f = xo @ m['E'].T
            m['u'] = np.linalg.solve(f.T @ f + RIDGE * np.eye(2), f.T @ yo)
    return m


def rrr_init(ev):
    xs, a, xn, xo, yo = ev
    Bls = np.linalg.lstsq(xs, xn, rcond=None)[0].T
    U = np.linalg.svd(xs @ Bls.T, full_matrices=False)[2][:2]  # top-2 output directions
    return {'E': U @ Bls, 'D': U.T.copy(), 'F': np.stack([np.eye(2)] * 2), 'u': np.zeros(2)}


def fit_latent(ev, seed, fixed=None):
    rng = np.random.default_rng(seed)
    inits = [rrr_init(ev)] + [{'E': rng.normal(scale=1 / np.sqrt(8), size=(2, 8)), 'D': rng.normal(size=(8, 2)),
                               'F': np.stack([np.eye(2)] * 2), 'u': np.zeros(2)} for _ in range(ALS_INITS - 1)]
    best = None
    for init in inits:
        if fixed is not None:
            init.update({k: v.copy() for k, v in fixed.items()})
        m = als(ev, init, fixed=tuple(fixed or ()))
        obj = latent_objective(m, ev)
        if np.isfinite(obj) and (best is None or obj < best[1]):
            best = (m, obj)
    return best[0]


def latent_errors(m, test):
    X = {}
    for L in HORIZONS:
        s = test['seqs'][L]
        z = np.repeat((test['x0'] @ m['E'].T)[:, None, :], s.shape[1], 1)
        for k in range(L):
            z = np.einsum('nmij,nmj->nmi', m['F'][s[:, :, k]], z)
        X[str(L)] = float(np.sum((z @ m['D'].T - test['target'][L]) ** 2) / np.sum(test['target'][L] ** 2))
    return X


def per_action_errors(ev, test, rank=None):
    xs, a, xn, _, _ = ev
    M = []
    for act in (0, 1):
        s = a == act
        Mls = np.linalg.lstsq(xs[s], xn[s], rcond=None)[0].T
        if rank is not None:
            U = np.linalg.svd(xs[s] @ Mls.T, full_matrices=False)[2][:rank]
            Mls = U.T @ U @ Mls
        M.append(Mls)
    M = np.stack(M)
    X = {}
    for L in HORIZONS:
        s = test['seqs'][L]
        x = np.repeat(test['x0'][:, None, :], s.shape[1], 1)
        for k in range(L):
            x = np.einsum('nmij,nmj->nmi', M[s[:, :, k]], x)
        X[str(L)] = float(np.sum((x - test['target'][L]) ** 2) / np.sum(test['target'][L] ** 2))
    return X


def p4_task(args):
    w, fam = args
    t0 = time.process_time()
    COUNTS.clear()
    world = make_world(w)
    ops = world['ops'][fam]
    fstream = {'A': 10, 'B': 20}[fam]
    src = transitions(windows(world, ops, 's1', N_FIT, fstream + 1))
    source = fit_latent(src, [w, fstream, 99])
    cal = windows(world, ops, 't', N_CAL, fstream + 41)
    test = test_set(world, ops, 't', fstream + 43)
    out = {}
    for n in BUDGETS:
        sub = {k: v[:n // 2] for k, v in cal.items()}
        ev = transitions(sub)
        own = fit_latent(ev, [w, fstream, n, 1])
        frozen = fit_latent(ev, [w, fstream, n, 2], fixed={'F': source['F'], 'u': source['u']})
        out[str(n)] = {'target_only': latent_errors(own, test), 'frozen_dynamics': latent_errors(frozen, test),
                       'target_only_F_eig_moduli': np.abs(np.linalg.eigvals(own['F'])).tolist(),
                       'per_action_ls': per_action_errors(ev, test), 'per_action_rrr': per_action_errors(ev, test, 2)}
    out['source_F_eig_moduli'] = np.abs(np.linalg.eigvals(source['F'])).tolist()  # reported diagnostic only
    out['counts'] = dict(COUNTS)
    out['cpu_seconds'] = time.process_time() - t0
    return (w, fam), out


# ---------------------------------------------------------------- verification

def verify():
    from experiments import local_contrastive_consolidation as lcc
    from experiments import local_consolidation as base
    out = {}
    # Energy: module block equals lcc.build at lambda 0.05; FD Hessian of the explicit window energy at 0.001.
    pb = base.initialize(3)
    model = lcc.Model('inner', (0., 0., 0.))
    q = lcc.build(model, pb)
    worst = max(float(np.max(np.abs(block(torch.tensor(pb['W'][m]), torch.tensor(pb['V'][m]), .05).numpy()
                                    - q[14 * m:14 * m + 14, 14 * m:14 * m + 14]))) for m in range(3))
    out['block_vs_lcc_build'] = worst
    rng = np.random.default_rng(5)
    p = {'W': rng.normal(size=(3, 8, 4)) * .5, 'V': rng.normal(size=(3, 4, 2)) * .5,
         'T01': rng.normal(size=(2, 2, 2)), 'T12': rng.normal(size=(2, 2, 2))}

    def energy(s, W, V, T01, T12, k):
        e = 0.
        for m in range(3):
            x, h, c = s[14 * m:14 * m + 8], s[14 * m + 8:14 * m + 12], s[14 * m + 12:14 * m + 14]
            e += .5 * np.sum((x - W[m] @ h) ** 2) + .5 * np.sum((h - V[m] @ c) ** 2) + LAM / 2 * (h @ h + c @ c)
        c0, c1, c2 = s[12:14], s[26:28], s[40:42]
        return e + k / 2 * np.sum((c1 - T01 @ c0) ** 2) + k / 2 * np.sum((c2 - T12 @ c1) ** 2)
    Q = window_q(*(torch.tensor(p[k]) for k in ('W', 'V')), torch.tensor(p['T01'][0]), torch.tensor(p['T12'][1]), 5.).numpy()
    s0 = rng.normal(size=42)
    hstep = 1e-3
    E = lambda s: energy(s, p['W'], p['V'], p['T01'][0], p['T12'][1], 5.)
    Hfd = np.zeros((42, 42))
    I = np.eye(42) * hstep
    for i in range(42):
        for j in range(42):
            Hfd[i, j] = (E(s0 + I[i] + I[j]) - E(s0 + I[i] - I[j]) - E(s0 - I[i] + I[j]) + E(s0 - I[i] - I[j])) / (4 * hstep ** 2)
    out['window_hessian_fd'] = float(np.max(np.abs(Hfd - Q)))
    # Gradients: autograd against central differences on a small multi-horizon objective.
    world = make_world(0)
    test = test_set(world, world['ops']['A'], 's1', 900, n=8, horizons=(1, 2, 8), pure=False)
    small = {k: v * (.5 if k in ('T01', 'T12') else 1) for k, v in p.items()}
    t = tparams(small, grad=True)
    f = lambda t: sum(horizon_errors(operators(t, 1.), test, (1, 2, 8))[0].values())
    f(t).backward()
    worst = 0.
    for k in t:
        flat = small[k].ravel()
        for idx in rng.choice(flat.size, 3, replace=False):
            def shifted(d):
                q = {kk: vv.copy() for kk, vv in small.items()}
                q[k].ravel()[idx] += d
                with torch.no_grad():
                    return float(f(tparams(q)))
            # fourth-order central stencil; the two-point stencil's rounding dominates small components
            h = 1e-3
            fd = (-shifted(2 * h) + 8 * shifted(h) - 8 * shifted(-h) + shifted(-2 * h)) / (12 * h)
            g = float(t[k].grad.numpy().ravel()[idx])
            worst = max(worst, abs(g - fd) / max(abs(fd), 1e-8))
    out['autograd_vs_fd_relative'] = worst
    # Rollout: latent form against repeated observation-space window maps.
    with torch.no_grad():
        o = operators(tparams(small), 1.)
    test = test_set(world, world['ops']['A'], 's1', 901, n=4, horizons=(2, 32, 128), pure=False)
    worst = 0.
    for L in (2, 32, 128):
        xh_lat = rollout(o, torch.as_tensor(test['x0']), test['seqs'][L])[0].numpy()
        pairs = pair_index(test['seqs'][L])
        xh = np.repeat(test['x0'][:, None, :], N_SEQ, 1)
        for j in range(L // 2):
            xh = np.einsum('nmij,nmj->nmi', o['M2'].numpy()[pairs[..., j]], xh)
        worst = max(worst, float(np.max(np.abs(xh - xh_lat)) / max(np.max(np.abs(xh)), 1e-300)))
    out['rollout_latent_vs_observation'] = worst
    # Oracle: K against a Monte-Carlo regression of z on x.
    R, B = world['domains']['s1']
    mc = np.random.default_rng(902)
    z = mc.normal(size=(400000, 2))
    x = observe(mc, R, B, z)
    Kmc = np.linalg.lstsq(x, z, rcond=None)[0].T
    resid = z - x @ Kmc.T
    se = np.sqrt(np.mean(resid ** 2) / (400000 * np.min(np.linalg.eigvalsh(x.T @ x / 400000))))
    out['oracle_mc'] = {'max_abs_difference': float(np.max(np.abs(Kmc - gain_matrix(world, 's1')))), 'standard_error_scale': float(se)}
    # Mismatch on known cases.
    F = world['ops']['A']
    P = np.array([[1., .4], [-.3, 1.2]])
    out['mismatch_basis_change'] = mismatch([P @ f @ np.linalg.inv(P) for f in F], F)
    out['commutant_two_rotations'] = commutant([rot(.5), rot(1.2)])
    Fb = world['ops']['B']
    out['mismatch_family_B_swap'] = mismatch(Fb[::-1], Fb)
    # P4 gauge: rollouts unchanged under E -> SE, F -> S F S^-1, D -> D S^-1.
    test = test_set(world, F, 't', 903, n=16)
    ev = transitions(windows(world, F, 't', 32, 904))
    m = fit_latent(ev, 905)
    S = np.array([[1.3, .2], [-.4, .8]])
    Si = np.linalg.inv(S)
    g = {'E': S @ m['E'], 'F': np.stack([S @ f @ Si for f in m['F']]), 'D': m['D'] @ Si, 'u': Si.T @ m['u']}
    a, b = latent_errors(m, test), latent_errors(g, test)
    out['p4_gauge'] = max(abs(a[k] - b[k]) for k in a)
    # Draft reproduction: the runner's construction on the draft world.
    draft = json.loads(DRAFT.read_text())
    rng = np.random.default_rng(1)
    R = 2 * rng.normal(size=(8, 2)) / np.sqrt(8)
    B = rng.normal(size=(8, 2)) / np.sqrt(8)
    Fd = (rot(.6), refl(.4))
    pd = {'W': np.stack([np.hstack([R, B])] * 3), 'V': np.stack([np.vstack([np.eye(2), np.zeros((2, 2))])] * 3),
          'T01': np.stack(Fd), 'T12': np.stack(Fd)}
    with torch.no_grad():
        o = operators(tparams(pd), 1.)
    names = {0: 'R', 1: 'Q'}
    worst = 0.
    for i, (a_, b_) in enumerate(PAIRS):
        d = draft['window_raw'][names[a_] + names[b_]]
        worst = max(worst, abs(np.abs(np.linalg.eigvals(o['G1'][i].numpy())).mean() - d['gain_one']),
                    abs(np.abs(np.linalg.eigvals(o['G2'][i].numpy())).mean() - d['gain_two']))
    lw = 0.
    for beta, d in draft['leakage'].items():
        W = np.stack([np.hstack([R, float(beta) * B])] * 3)
        with torch.no_grad():
            phi = encoder(torch.tensor(W), torch.tensor(pd['V'])).numpy()
        lw = max(lw, abs(np.linalg.norm(np.linalg.solve(phi @ R, phi @ B), 2) - d['relative_leakage']))
    u = np.array([np.cos(1.1), np.sin(1.1)])
    Fr, Fq = Fd
    sw = max(abs(mismatch([Fq, Fr], [Fr, Fq])['bound'] - draft['scrambles']['A_swap'][0]),
             abs(mismatch([Fq, Fr], [Fr, Fq], u)['bound'] - draft['scrambles']['A_swap_probe'][0]),
             abs(mismatch([refl(.9), refl(.3)], [refl(.3), refl(.9)], u)['bound'] - draft['scrambles']['B_swap_probe'][0]))
    out['draft_reproduction'] = {'window_gains': worst, 'residuals_below_1e-12': bool(max(o['res1'].max(), o['res2'].max()) < 1e-12),
                                 'leakage': lw, 'scramble_bounds': sw}
    checks = {'block_vs_lcc_build': out['block_vs_lcc_build'] <= 1e-12, 'window_hessian_fd': out['window_hessian_fd'] <= 1e-5,
              'autograd_vs_fd': out['autograd_vs_fd_relative'] <= 1e-6, 'rollout': out['rollout_latent_vs_observation'] <= 1e-10,
              'oracle_mc': out['oracle_mc']['max_abs_difference'] <= 4 * out['oracle_mc']['standard_error_scale'],
              'basis_change': out['mismatch_basis_change']['bound'] <= 1e-12 and out['mismatch_basis_change']['minimizer_condition'] <= 1e6,
              'commutant_two_rotations': out['commutant_two_rotations'] == 2,
              'family_B_swap': out['mismatch_family_B_swap']['bound'] <= 1e-12 and abs(out['mismatch_family_B_swap']['minimizer_condition'] - 1) <= 1e-6,
              'p4_gauge': out['p4_gauge'] <= 1e-10,
              'draft': all(v <= 1e-10 for k, v in out['draft_reproduction'].items() if k != 'residuals_below_1e-12')
              and out['draft_reproduction']['residuals_below_1e-12']}
    out['checks'] = checks
    return out


# ---------------------------------------------------------------- criteria

def criteria(res):
    out = {}
    worlds = list(WORLDS)
    # A1
    a1 = {}
    for w in worlds:
        r = res['P0'][str(w)]
        sw = r['scrambles']['A']['swap']
        br = r['scrambles']['broken']
        a1[str(w)] = {
            'observability': all(v == 2 for v in r['observability']['A']['per_action']) and r['observability']['A']['mixed_to_4'] == 2,
            'rendering': all(v <= 10 for v in r['rendering_condition'].values()),
            'commutant': r['commutant']['A'] == 1,
            'swap': classify(sw['J'], .05) == 'detectable' and classify(sw['J_u'], .05) == 'detectable',
            'broken': classify(br['J'], .005) == 'detectable' and classify(br['J_u'], .005) == 'detectable'}
    out['A1'] = {'per_world': a1, 'passes': all(all(v.values()) for v in a1.values()),
                 'family_B_swap': {str(w): [classify(res['P0'][str(w)]['scrambles']['B']['swap'][k], .05) for k in ('J', 'J_u')] for w in worlds}}
    # C1
    c1 = {}
    for fam in FAMILIES:
        for kappa in KAPPAS:
            for clip in CLIPS:
                key = f'{fam}_k{kappa:g}_{clip}'
                per = {}
                for w in worlds:
                    r = res['P1'][key][str(w)]
                    per[str(w)] = {'constructive': passes_c1(r['constructive']),
                                   'multi_horizon': not r['multi_horizon'].get('failed') and passes_c1(r['multi_horizon'])}
                n = sum(v['constructive'] or v['multi_horizon'] for v in per.values())
                c1[key] = {'per_world': per, 'worlds_passing': n, 'witness_found': n >= 9}
    for fam in FAMILIES:
        for kappa in KAPPAS:
            a, b = c1[f'{fam}_k{kappa:g}_clip2'], c1[f'{fam}_k{kappa:g}_clip6']
            a['label'] = 'witness_found' if a['witness_found'] else (
                'witness_found_only_with_relaxed_clips' if b['witness_found'] else 'no_witness_found_by_these_searches')
            b['label'] = 'witness_found' if b['witness_found'] else 'no_witness_found_by_these_searches'
    out['C1'] = c1
    # C2
    c2 = {}
    for key, r in res['P2'].items():
        inv, drift = 0, 0
        for w in worlds:
            rw = r.get(str(w))
            if rw is None:
                continue
            ok = all(not rw[d].get('failed') and rw[d]['J_G']['bound'] <= 1e-3 and rw[d]['J_G']['minimizer_condition'] <= 10 for d in ('s2', 't'))
            inv += ok
            drift += any(not rw[d].get('failed') and rw[d]['J_G']['bound'] >= 1e-2 for d in ('s2', 't'))
        label = 'invariant' if inv >= 9 else ('drifts' if drift >= 2 else 'unresolved')
        c2[key] = {'worlds_invariant': inv, 'worlds_drift': drift, 'label': label}
    out['C2'] = c2
    # C3
    c3 = {}
    for key, r in res['P1'].items():
        n = sum(r[str(w)]['P3']['relative_activity_error_32'] <= 1e-3 for w in worlds)
        c3[key] = {'worlds_agree': n, 'agree': n >= 9}
    out['C3'] = c3
    # C4
    c4 = {}
    for fam in FAMILIES:
        q = []
        table = {}
        for n in BUDGETS:
            for L in HORIZONS:
                own = [res['P4'][f'{fam}'][str(w)][str(n)]['target_only'][str(L)] for w in worlds]
                frz = [res['P4'][f'{fam}'][str(w)][str(n)]['frozen_dynamics'][str(L)] for w in worlds]
                ratio = float(np.mean(own) / np.mean(frz))
                order = int(sum(o > f for o, f in zip(own, frz)))
                table[f'{n}_{L}'] = {'ratio_of_means': ratio, 'worlds_ordered': order}
                if ratio >= 1.5 and order >= 9:
                    q.append([n, L])
        c4[fam] = {'table': table, 'qualifying': q,
                   'label': 'qualifying_gap' if q else 'no_qualifying_gap_found_with_these_references'}
    out['C4'] = c4
    # Outcome (project decision rule)
    witness = [k for k in (f'A_k{k:g}_clip2' for k in KAPPAS) if c1[k]['witness_found']]
    inv = [k for k in witness if c2.get(k, {}).get('label') == 'invariant']
    out['outcome'] = {'A1': out['A1']['passes'], 'A_witness_at_clip2': witness, 'A_invariant': inv,
                      'C4_A': c4['A']['label'],
                      'proceed': bool(out['A1']['passes'] and inv and c4['A']['qualifying'])}
    return out


# ---------------------------------------------------------------- main

def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_params(r):
    return {k: ({kk: vv for kk, vv in v.items() if kk != 'params'} if isinstance(v, dict) else v) for k, v in r.items()}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    phases = {}

    @contextlib.contextmanager
    def phase(name):
        w, c = time.perf_counter(), time.process_time()
        yield
        phases[name] = {'wall_seconds': time.perf_counter() - w, 'parent_cpu_seconds': time.process_time() - c}

    res = {'contract': str(CONTRACT.relative_to(ROOT)), 'worlds': list(WORLDS), 'steps': STEPS}
    worker = {}
    with phase('verification'):
        res['verification'] = verify()
    with phase('P0'):
        res['P0'] = {str(w): p0(make_world(w)) for w in WORLDS}
    ctx = mp.get_context('fork')
    with ctx.Pool(WORKERS) as pool:
        with phase('P1_P3'):
            tasks = [(w, fam, kappa, clip) for fam in FAMILIES for kappa in KAPPAS for clip in CLIPS for w in WORLDS]
            p1 = dict(pool.map(p1_task, tasks, chunksize=1))
        res['P1'] = {}
        for (w, fam, kappa, clip), r in sorted(p1.items()):
            key = f'{fam}_k{kappa:g}_{clip}'
            res['P1'].setdefault(key, {})[str(w)] = r
            worker[f'P1_{key}_{w}'] = {'cpu_seconds': r.pop('cpu_seconds'), 'counts': r.pop('counts')}
        c1 = criteria_partial = None
        # P2 where C1 found a witness for the condition; banks per world as the contract specifies.
        with phase('P2'):
            p2_tasks = []
            for key, per in res['P1'].items():
                found = sum(passes_c1(per[str(w)]['constructive']) or
                            (not per[str(w)]['multi_horizon'].get('failed') and passes_c1(per[str(w)]['multi_horizon']))
                            for w in WORLDS) >= 9
                if not found:
                    continue
                fam, k, clip = key.split('_')
                kappa = float(k[1:])
                for w in WORLDS:
                    r = per[str(w)]
                    use = 'multi_horizon' if (not r['multi_horizon'].get('failed') and passes_c1(r['multi_horizon'])) else 'constructive'
                    banks = {kk: np.array(r[use]['params'][kk]) for kk in ('T01', 'T12')}
                    full = {kk: np.array(v) for kk, v in r[use]['params'].items()}
                    with torch.no_grad():
                        o = operators(tparams(full), kappa)
                    ref = np.concatenate([o['G1'].numpy(), o['G2'].numpy()])
                    eta = r['one_step']['eta'] if not r['one_step'].get('failed') else ETAS[0]
                    p2_tasks.append(((w, fam, kappa, clip), banks, eta, ref))
                    r['P2_banks_from'] = use
            p2 = dict(pool.map(p2_task, p2_tasks, chunksize=1))
        res['P2'] = {}
        for (w, fam, kappa, clip), r in sorted(p2.items()):
            key = f'{fam}_k{kappa:g}_{clip}'
            res['P2'].setdefault(key, {})[str(w)] = r
            worker[f'P2_{key}_{w}'] = {'cpu_seconds': r.pop('cpu_seconds'), 'counts': r.pop('counts')}
        with phase('P4'):
            p4 = dict(pool.map(p4_task, [(w, fam) for fam in FAMILIES for w in WORLDS], chunksize=1))
        res['P4'] = {}
        for (w, fam), r in sorted(p4.items()):
            res['P4'].setdefault(fam, {})[str(w)] = r
            worker[f'P4_{fam}_{w}'] = {'cpu_seconds': r.pop('cpu_seconds'), 'counts': r.pop('counts')}
    with phase('criteria'):
        res['criteria'] = criteria(res)
    # Fitted parameters are kept in a separate file; the main results file omits them.
    params = {key: {w: {arm: r[arm].get('params') for arm in ('constructive', 'multi_horizon', 'one_step')}
                    for w, r in per.items()} for key, per in res['P1'].items()}
    for per in res['P1'].values():
        for w in per:
            per[w] = strip_params(per[w])
    PREFIX.parent.mkdir(exist_ok=True)
    text = json.dumps(res, indent=1, sort_keys=True, default=float) + '\n'
    ptext = json.dumps(params, sort_keys=True, default=float) + '\n'
    outputs = {}
    for suffix, body in (('results', text), ('params', ptext)):
        path = Path(f'{PREFIX}-{suffix}.json')
        data = body.encode()
        outputs[suffix] = {'sha256_uncompressed': hashlib.sha256(data).hexdigest(), 'bytes_uncompressed': len(data)}
        if len(data) > 1_000_000:
            with open(str(path) + '.gz', 'wb') as f:
                with gzip.GzipFile(filename='', mode='wb', fileobj=f, compresslevel=9, mtime=0) as g:
                    g.write(data)
            outputs[suffix]['file'] = rel(path) + '.gz'
        else:
            path.write_bytes(data)
            outputs[suffix]['file'] = rel(path)
    total_worker = sum(v['cpu_seconds'] for v in worker.values())
    agg = {}
    for v in worker.values():
        for k, n in v['counts'].items():
            agg[k] = agg.get(k, 0) + n
    runtime = {
        'contract_sha256': sha(CONTRACT), 'runner_sha256': sha(__file__), 'revision': git('rev-parse', 'HEAD'),
        'dirty': bool(git('status', '--porcelain')), 'command': ' '.join(sys.argv), 'worlds': list(WORLDS), 'steps': STEPS,
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu': platform.processor(), 'cpu_count': os.cpu_count(), 'workers': WORKERS, 'threads_per_worker': 1,
        'dtype': 'float64', 'blas': 'single-threaded (OPENBLAS/OMP/MKL_NUM_THREADS=1)',
        'phases': phases, 'worker_cpu_seconds_total': total_worker, 'worker_tasks': worker, 'counts_total': agg,
        'parent_counts': dict(COUNTS), 'wall_seconds': time.perf_counter() - wall0,
        'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single child',
        'outputs': outputs}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps(runtime, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'verification': res['verification']['checks'], 'criteria_outcome': res['criteria']['outcome'],
                      'wall_seconds': runtime['wall_seconds']}, indent=1, default=float))


if __name__ == '__main__':
    main()
