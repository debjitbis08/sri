"""Draft-stage analytical checks for docs/relational-linear-preflight-v1-contract.md.

Deterministic; run before freezing to support the contract's "what theory predicts"
items. One hand-built world (seed 1, rotation 0.6, reflection axis 0.4):

1. three-module window (x_t, x_t+1, x_t+2) with action banks at both inner couplings:
   factorization of the effective operators through the isolated module-0 encoder,
   per-window gain with T = F and after scalar rescaling, and back-reaction of the
   second action on module 1's prediction;
2. nuisance leakage of the constructed encoder W = [R, beta B] in signal units;
3. commutant dimensions;
4. certified Sylvester lower bounds for scrambles, with and without the probe;
5. objective mismatch: composing the population-optimal one-step map against the oracle.

Writes reports/relational-linear-preflight-v1-draft-checks.json.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/relational-linear-preflight-v1-draft-checks.json'
LAM, KAPPA, NOISE, NUIS = 1e-3, 1.0, .05, .25


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def refl(a):
    return np.array([[np.cos(2 * a), np.sin(2 * a)], [np.sin(2 * a), -np.cos(2 * a)]])


def world():
    rng = np.random.default_rng(1)
    R = 2 * rng.normal(size=(8, 2)) / np.sqrt(8)
    B = rng.normal(size=(8, 2)) / np.sqrt(8)
    return R, B, rot(.6), refl(.4)


def block(W, V):
    """Hessian of 1/2|x-Wh|^2 + 1/2|h-Vc|^2 + lam/2(|h|^2 + |c|^2) over (x, h, c)."""
    q = np.zeros((14, 14))
    x, h, c = slice(0, 8), slice(8, 12), slice(12, 14)
    q[x, x] += np.eye(8)
    q[x, h] -= W
    q[h, x] -= W.T
    q[h, h] += W.T @ W + (1 + LAM) * np.eye(4)
    q[h, c] -= V
    q[c, h] -= V.T
    q[c, c] += V.T @ V + LAM * np.eye(2)
    return q


def window(p, T01, T12):
    q = np.zeros((42, 42))
    for m in range(3):
        q[14 * m:14 * m + 14, 14 * m:14 * m + 14] = block(p['W'][m], p['V'][m])
    for (ma, mb), T in (((0, 1), T01), ((1, 2), T12)):
        a, b = np.arange(14 * ma + 12, 14 * ma + 14), np.arange(14 * mb + 12, 14 * mb + 14)
        q[np.ix_(b, b)] += KAPPA * np.eye(2)
        q[np.ix_(a, a)] += KAPPA * T.T @ T
        q[np.ix_(b, a)] -= KAPPA * T
        q[np.ix_(a, b)] -= KAPPA * T.T
    return q


def clamped(q, obs):
    free = np.setdiff1d(np.arange(len(q)), obs)
    y = np.linalg.solve(q[np.ix_(free, free)], -q[np.ix_(free, obs)])
    s = np.zeros((len(q), len(obs)))
    s[obs], s[free] = np.eye(len(obs)), y
    return s


def encoder(p):
    return clamped(block(p['W'][0], p['V'][0]), np.arange(8))[12:14]


def effective(p, T01, T12):
    """Module-1 and module-2 one-solve predictions and their 2x2 operators in module-0 coordinates."""
    s = clamped(window(p, T01, T12), np.arange(8))
    phi = encoder(p)
    out = {}
    for name, rows in (('one', slice(14, 22)), ('two', slice(28, 36))):
        M = s[rows]
        G = np.linalg.lstsq(phi.T, (phi @ M).T, rcond=None)[0].T
        out[name] = {'M': M, 'G': G, 'residual': np.linalg.norm(phi @ M - G @ phi) / np.linalg.norm(phi @ M)}
    return out


def sylvester(Fp, F, u=None):
    """min over S of sum_a |S F'_a - F_a S|^2 / sum_a |F_a|^2; |S|_F = 1, or u^T S = u^T if u is given.

    The minimum is over all S, singular included, so it is a certified lower bound for invertible S.
    Returns the bound and the condition number of the minimizer.
    """
    rows = [np.kron(np.eye(2), f.T) - np.kron(g, np.eye(2)) for f, g in zip(Fp, F)]  # vec(S) row-major
    A = np.vstack(rows)
    norm = sum(np.sum(g * g) for g in F)
    if u is None:
        _, sv, vt = np.linalg.svd(A)
        S = vt[-1].reshape(2, 2)
        return sv[-1] ** 2 / norm, np.linalg.cond(S)
    C = np.array([[u[0], 0, u[1], 0], [0, u[0], 0, u[1]]])  # u^T S as a linear map of row-major vec(S)
    kkt = np.block([[A.T @ A, C.T], [C, np.zeros((2, 2))]])
    sol = np.linalg.lstsq(kkt, np.concatenate([np.zeros(4), u]), rcond=None)[0]
    s = sol[:4]
    S = s.reshape(2, 2)
    assert np.allclose(u @ S, u)
    return float(s @ A.T @ A @ s) / norm, np.linalg.cond(S)


def commutant(F):
    A = np.vstack([np.kron(np.eye(2), f.T) - np.kron(f, np.eye(2)) for f in F])
    sv = np.linalg.svd(A, compute_uv=False)
    return int(4 - np.sum(sv > 1e-10 * sv[0]))


def main():
    R, B, Fr, Fq = world()
    ops = {'R': Fr, 'Q': Fq}
    out = {}

    # 1. three-module window with the constructed witness.
    base = {'W': [np.hstack([R, B])] * 3, 'V': [np.vstack([np.eye(2), np.zeros((2, 2))])] * 3}
    raw, comp = {}, {}
    for a1 in ops:
        for a2 in ops:
            e = effective(base, ops[a1], ops[a2])
            g1 = np.abs(np.linalg.eigvals(e['one']['G'])).mean()
            # rescale T01 by the window's one-step gain, then T12 by the remaining two-step gain
            e1 = effective(base, ops[a1] / g1, ops[a2])
            g2 = np.abs(np.linalg.eigvals(e1['two']['G'])).mean()
            ec = effective(base, ops[a1] / g1, ops[a2] / g2)
            alt = effective(base, ops[a1], ops['Q' if a2 == 'R' else 'R'])
            raw[a1 + a2] = {
                'residual_one': e['one']['residual'], 'residual_two': e['two']['residual'],
                'gain_one': g1, 'gain_two': float(np.abs(np.linalg.eigvals(e['two']['G'])).mean()),
                'back_reaction_one': float(np.linalg.norm(e['one']['M'] - alt['one']['M']) / np.linalg.norm(e['one']['M']))}
            comp[a1 + a2] = {'gain_two': float(np.abs(np.linalg.eigvals(ec['two']['G'])).mean()),
                             'max_abs_T': float(max(np.abs(ops[a1] / g1).max(), np.abs(ops[a2] / g2).max()))}
    for v in raw.values():
        v['gain_two_pow64'] = v['gain_two'] ** 64  # 64 windows = 128 steps
    for v in comp.values():
        v['gain_two_pow64'] = v['gain_two'] ** 64
    out['window_raw'], out['window_compensated'] = raw, comp

    # 2. leakage in signal units versus nuisance-column scale.
    leak = {}
    for beta in (1, 2, 3, 5):
        p = {'W': [np.hstack([R, beta * B])], 'V': [np.vstack([np.eye(2), np.zeros((2, 2))])]}
        phi = encoder(p)
        sig = phi @ R
        sv = np.linalg.svd(sig, compute_uv=False)
        leak[str(beta)] = {'max_abs_W': float(np.abs(p['W'][0]).max()), 'signal_singular_values': sv.tolist(),
                           'relative_leakage': float(np.linalg.norm(np.linalg.solve(sig, phi @ B), 2))}
    out['leakage'] = leak

    # 3. commutants.
    out['commutant'] = {'R_Q': commutant([Fr, Fq]), 'R': commutant([Fr]), 'R_R2': commutant([Fr, rot(1.3)])}

    # 4. certified scramble bounds.
    u = np.array([np.cos(1.1), np.sin(1.1)])
    Q1, Q2 = refl(.3), refl(.9)
    rng = np.random.default_rng(7)
    P = rng.normal(size=(2, 2))
    out['scrambles'] = {
        'A_swap': sylvester([Fq, Fr], [Fr, Fq]), 'A_swap_probe': sylvester([Fq, Fr], [Fr, Fq], u),
        'B_swap': sylvester([Q2, Q1], [Q1, Q2]), 'B_swap_probe': sylvester([Q2, Q1], [Q1, Q2], u),
        'A_same_eigenvalues': sylvester([P @ Fr @ np.linalg.inv(P), Fq], [Fr, Fq]),
        'A_broken_rotation': sylvester([rot(.9), Fq], [Fr, Fq]),
        'A_basis_change': sylvester([P @ Fr @ np.linalg.inv(P), P @ Fq @ np.linalg.inv(P)], [Fr, Fq])}

    # 5. objective mismatch: K = E[z|x] for x = R z + B n + eps, z ~ N(0, I).
    cov = R @ R.T + NUIS * B @ B.T + NOISE ** 2 * np.eye(8)
    K = R.T @ np.linalg.inv(cov)
    KR = K @ R
    out['objective_mismatch'] = {'KR_singular_values': np.linalg.svd(KR, compute_uv=False).tolist(),
                                 'composed_over_oracle_gain_L128': float(
                                     np.linalg.norm(np.linalg.matrix_power(Fr @ KR, 127) @ Fr @ K @ R)
                                     / np.linalg.norm(np.linalg.matrix_power(Fr, 128) @ K @ R))}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps(out, indent=1, sort_keys=True, default=float))


if __name__ == '__main__':
    main()
