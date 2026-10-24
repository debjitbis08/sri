"""Design-stage calculations and explicitly exploratory pilots, not a scored v3 run.

Run: .venv/bin/python -m experiments.layered_module_design_analysis
No v1/v2 evaluation draws or proposed confirmation seeds are used.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr


N = 32
LAM = 4.0
SIGMA = 0.1
EIG = 2 - 2 * np.cos(2 * np.pi * np.arange(N) / N)
TARGETS = [1, 8, 24, 31]
Q = ((EIG - EIG[1]) * (EIG - EIG[8])) ** 2
TAU = 1 / min(Q[k] for k in range(N) if k not in TARGETS)


def gains(depth, penalty, lam=LAM, centre=False):
    """Mode-by-layer gains; no interior local potential or exterior lateral penalty."""
    result, eigenvalues = [], []
    for k, p in enumerate(penalty):
        h = np.diag([lam + 1] + [2 + p] * (depth - 2) + [1 + p])
        h += np.diag([-1.] * (depth - 1), 1) + np.diag([-1.] * (depth - 1), -1)
        # Eliminating an unanchored centre adds one for non-DC modes only.
        if centre and k != 0:
            h[-1, -1] += 1
        result.append(np.linalg.solve(h, [lam] + [0.] * (depth - 1)))
        eigenvalues.extend(np.linalg.eigvalsh(h))
    return np.array(result).T, [min(eigenvalues), max(eigenvalues)]


def transfer(g):
    """Real circulant matrix with Fourier eigenvalues g."""
    return np.fft.ifft(np.fft.fft(np.eye(N), axis=0) * g[:, None], axis=0).real


def population(g):
    power = np.sum(g * g, axis=1)
    return dict(noise_power_by_layer=power.tolist(), suppression=float(power[-1] / power[0]),
                signal_gains=g[:, [1, 8]].tolist(),
                gaussian_accuracy_alpha1=[float(ndtr(4 * a / np.sqrt(a*a + 2*SIGMA**2))**2)
                                          for a in g[-1, [1, 8]]])


def verify():
    """Independent spatial Hessian check against the Fourier derivation."""
    depth = 4
    eye = np.eye(N)
    c = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    b = (c - EIG[1]*eye) @ (c - EIG[8]*eye)
    h = np.zeros((depth*N, depth*N))
    h[:N, :N] += LAM*eye
    for layer in range(1, depth):
        a, z = slice((layer-1)*N, layer*N), slice(layer*N, (layer+1)*N)
        h[a, a] += eye
        h[z, z] += eye + TAU*(b.T @ b)
        h[a, z] -= eye
        h[z, a] -= eye
    forcing = np.zeros((depth*N, N))
    forcing[:N] = LAM*eye
    direct = np.linalg.solve(h, forcing).reshape(depth, N, N)
    g, ev = gains(depth, TAU*Q)
    spectral = np.array([transfer(row) for row in g])
    error = float(np.max(np.abs(direct-spectral)))
    assert error < 1e-10, error
    assert np.max(np.abs(g[:, TARGETS]-1)) < 1e-12
    assert np.max(g) < 1 + 1e-12 and np.min(g) >= 0
    assert np.all(np.diff(g, axis=0) <= 1e-12)
    eig_error = float(np.max(np.abs(np.linalg.eigvalsh(h)[[0, -1]]-ev)))
    assert eig_error < 1e-8
    disk_h = np.zeros((depth*N+1, depth*N+1))
    disk_h[:-1, :-1] = h
    disk_h[:N, :N] -= (LAM-1)*eye
    disk_h[-N-1:-1, -N-1:-1] += eye
    disk_h[-1, -1] = N
    disk_h[-1, -N-1:-1] = -1
    disk_h[-N-1:-1, -1] = -1
    disk_f = np.zeros((depth*N+1, N))
    disk_f[:N] = eye
    disk_direct = np.linalg.solve(disk_h, disk_f)[:-1].reshape(depth, N, N)
    disk_g, _ = gains(depth, TAU*Q, 1, True)
    disk_error = float(np.max(np.abs(disk_direct-np.array([transfer(row) for row in disk_g]))))
    assert disk_error < 1e-10
    assert np.all(np.diff(disk_g, axis=0) <= 1e-12)
    return dict(spatial_fourier_max_error=error, hessian_extrema_error=eig_error,
                centre_spatial_fourier_max_error=disk_error,
                centre_full_hessian_eigenvalue_range=np.linalg.eigvalsh(disk_h)[[0, -1]].tolist())


def pilot(g, repetitions=200):
    """Frozen analytic candidate, alpha=1; finite-training NM under spherical noise."""
    maps = np.array([transfer(row) for row in g])
    depth = len(g)
    theta = 2*np.pi*np.arange(N)/N
    accuracies, ratios, passes = [], [], []
    for rep in range(repetitions):
        rng = np.random.default_rng([20261029, 0, rep])
        # Same underlying draws across candidate/control/depth evaluations.
        z = rng.standard_normal((2, 4, 50, N))
        z *= 4 / np.linalg.norm(z, axis=-1, keepdims=True)
        family_acc, family_ratio, family_pass = [], [], []
        for family, mode in enumerate((1, 8)):
            signal = np.cos(mode*theta[None, :] + np.arange(4)[:, None]*np.pi/2)
            drives = signal[:, None, :] + z[family]
            states = np.einsum('cdj,lkj->lcdk', drives, maps)
            within = states.var(axis=2).mean(axis=(1, 2))
            total = states.reshape(depth, 200, N).var(axis=1).mean(axis=1)
            noise = np.random.default_rng([20261029, 1, rep, family]).standard_normal((4, 50, N))
            readout = states[-1] + SIGMA*noise
            means = readout[:, :25].mean(axis=1)
            test = readout[:, 25:].reshape(100, N)
            pred = ((test[:, None, :]-means[None, :, :])**2).sum(axis=2).argmin(axis=1)
            acc = float(np.mean(pred == np.repeat(np.arange(4), 25)))
            ratio = float(within[-1]/within[0])
            family_acc.append(acc)
            family_ratio.append(ratio)
            family_pass.append(bool(np.all(total[0] >= total-1e-12) and ratio <= .25 and acc >= .9))
        accuracies.append(family_acc)
        ratios.append(family_ratio)
        passes.append(all(family_pass))
    a, r = np.array(accuracies), np.array(ratios)
    return dict(repetitions=repetitions, full_profile_passes=sum(passes),
                accuracy_min_by_family=a.min(axis=0).tolist(), accuracy_mean_by_family=a.mean(axis=0).tolist(),
                suppression_range=[float(r.min()), float(r.max())])


def main():
    rows, disks = {}, {}
    for depth in (2, 4, 6):
        g, ev = gains(depth, TAU*Q)
        rows[str(depth)] = dict(**population(g), hessian_eigenvalue_range=ev,
                               condition_number=ev[1]/ev[0], exploratory_pilot=pilot(g))
        dg, _ = gains(depth, TAU*Q, 1, True)
        disks[str(depth)] = dict(**population(dg), exploratory_pilot=pilot(dg))
    controls = {}
    wrong_q = ((EIG-EIG[1])*(EIG-EIG[4]))**2
    wrong_tau = 1/min(wrong_q[k] for k in range(N) if k not in (1, 4, 28, 31))
    for name, penalty, lam, centre in (
        ('identity', np.zeros(N), 4., False),
        ('ordinary_smoothing', EIG, 4., False),
        ('wrong_bands_1_4', wrong_tau*wrong_q, 4., False),
        ('weak_tie_ablation', TAU*Q, 1., False),
        ('centre_ablation', TAU*Q, 4., True),
        ('disk_no_lateral', np.zeros(N), 1., True),
        ('disk_ordinary_smoothing', EIG, 1., True),
        ('disk_wrong_bands_1_4', wrong_tau*wrong_q, 1., True),
    ):
        g, _ = gains(4, penalty, lam, centre)
        controls[name] = population(g)
        if name in ('identity', 'ordinary_smoothing', 'wrong_bands_1_4') or name.startswith('disk_'):
            controls[name]['exploratory_pilot'] = pilot(g)
    result = dict(status='analytical design plus exploratory pilots; not confirmatory results',
                  tau=TAU, q_min=1/TAU, max_lateral_eigenvalue=float(max(TAU*Q)),
                  verification=verify(), candidates=rows, controls=controls,
                  centre_candidates=disks,
                  spherical_known_mean_accuracy_lower_bound=float(1-2*np.exp(-4/(.5+SIGMA**2))))
    path = Path('reports/layered-module-design-analysis.json')
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
