"""Deterministic shared-surface analysis; does not access running experiment outputs.

Run: .venv/bin/python -m experiments.shared_surface_analysis
No sampled training/test drives or confirmation data are generated.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path

import numpy as np


N, L = 32, 4
THETA = 2*np.pi*np.arange(N)/N
F = np.column_stack([np.cos(THETA), np.sin(THETA),
                     np.cos(8*THETA), np.sin(8*THETA)])/4
EIG = 2 - 2*np.cos(THETA)


def stiffness(arm):
    eye = np.eye(N)
    c = 2*eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    if arm == 'ordinary':
        return c, EIG
    if arm == 'none':
        return np.zeros_like(c), np.zeros(N)
    bands = (1, 8) if arm == 'selective' else (1, 4)
    a, b = bands
    symbol = ((EIG-EIG[a])*(EIG-EIG[b]))**2
    excluded = {a, b, N-a, N-b}
    tau = 1/min(symbol[k] for k in range(N) if k not in excluded)
    op = (c-EIG[a]*eye) @ (c-EIG[b]*eye)
    return tau*(op.T@op), tau*symbol


def interior(arm, centre):
    k, symbol = stiffness(arm)
    size = N*L + int(centre)
    h = np.zeros((size, size))
    eye = np.eye(N)
    for layer in range(1, L):
        a, b = slice((layer-1)*N, layer*N), slice(layer*N, (layer+1)*N)
        h[a, a] += eye
        h[b, b] += eye+k
        h[a, b] -= eye
        h[b, a] -= eye
    if centre:
        z = slice((L-1)*N, L*N)
        h[z, z] += eye
        h[-1, -1] = N
        h[z, -1] = -1
        h[-1, z] = -1
    s = h[:N, :N] - h[:N, N:] @ np.linalg.solve(h[N:, N:], h[N:, :N])
    s = (s+s.T)/2
    # Independent Fourier recurrence for the terminal load and boundary stiffness.
    load = symbol + (np.array([0.]+[1.]*(N-1)) if centre else 0.)
    for _ in range(L-2):
        load = symbol + load/(1+load)
    effective = load/(1+load)
    spectral = np.fft.ifft(np.fft.fft(np.eye(N), axis=0)*effective[:, None], axis=0).real
    error = float(np.max(np.abs(s-spectral)))
    assert error < 1e-9
    assert np.linalg.eigvalsh(s)[0] > -1e-9
    assert np.linalg.eigvalsh(s)[-1] <= 1+1e-9
    return h, s, effective, error


def prediction(h, s, observed):
    mask = np.zeros(N)
    mask[observed] = 1
    m = np.diag(mask)
    hidden = np.flatnonzero(1-mask)
    a = s+m
    if np.linalg.eigvalsh(a)[0] < 1e-9:
        return dict(status='nonunique', nullity=int(np.sum(np.linalg.eigvalsh(a)<1e-9)))
    t = np.linalg.solve(a, m)
    # Spatial system check with exactly the same boundary-only external coupling.
    full = h.copy()
    full[:N, :N] += m
    forcing = np.zeros((len(h), N))
    forcing[:N] = m
    spatial = np.linalg.solve(full, forcing)[:N]
    assert np.max(np.abs(t-spatial)) < 1e-8
    # E[aa^T]=4I => signal power 1/2 at every boundary position.
    bias = float(4*np.linalg.norm((t@F-F)[hidden])**2/len(hidden))
    noise_factor = float(np.linalg.norm(t[hidden][:, observed])**2/len(hidden))
    results = {}
    for alpha in (0., .5, 1.):
        noise = alpha**2/2*noise_factor
        results[str(alpha)] = dict(clean_bias_mse=bias, nuisance_mse=noise,
                                   nmse_without_readout=(bias+noise)/.5,
                                   nmse_with_sigma_0_1=(bias+noise+.01)/.5)
    return dict(status='unique', predictions=results,
                clean_transfer_max_error=float(np.max(np.abs(t@F-F))),
                hessian_min_eigenvalue=float(np.linalg.eigvalsh(full)[0]),
                spatial_schur_max_error=float(np.max(np.abs(t-spatial))),
                susceptibility_asymmetry=float(np.max(np.abs(np.linalg.inv(a)-np.linalg.inv(a).T))))


def main():
    masks = dict(half=np.arange(16), other_half=np.arange(16,32),
                 even=np.arange(0,32,2), odd=np.arange(1,32,2))
    records = {}
    for label, arm, centre in [('current_selective_disk','selective',True),
                               ('free_selective','selective',False),
                               ('free_ordinary','ordinary',False),
                               ('free_none','none',False),
                               ('free_wrong_bands','wrong',False)]:
        h,s,e,error = interior(arm,centre)
        full_gain = 1/(1+e)
        records[label] = dict(surface_stiffness_spectrum=e.tolist(),
                              surface_gain_spectrum=full_gain.tolist(),
                              effective_stiffness_check_error=error,
                              full_drive_signal_normalized_nuisance_ratio=float(np.sum(full_gain**2)/(32*full_gain[1]**2)),
                              masks={key:prediction(h,s,obs) for key,obs in masks.items()})
    # Check exact clean completion for every contiguous half-mask and its reverse role.
    h,s,_,_ = interior('selective',False)
    rotation_errors=[]
    for offset in range(N):
        obs=(np.arange(16)+offset)%N
        r=prediction(h,s,obs)
        assert r['status']=='unique'
        rotation_errors.append(r['clean_transfer_max_error'])
    assert max(rotation_errors)<1e-9
    # Four-parameter least-squares comparison uses exactly the same observed values.
    obs=masks['half'];hidden=masks['other_half']
    oracle=F[hidden]@np.linalg.pinv(F[obs])
    oracle_noise=float(.5*np.linalg.norm(oracle)**2/len(hidden))
    result=dict(status='deterministic analysis only; no confirmation draws or running outputs accessed',
                records=records,
                observation_gram_eigenvalues={key:np.linalg.eigvalsh(F[obs].T@F[obs]).tolist()
                                             for key,obs in masks.items()},
                all_contiguous_masks_max_clean_error=max(rotation_errors),
                least_squares_alpha1_sigma_0_1_nmse=(oracle_noise+.01)/.5)
    path=Path('reports/shared-surface-analysis.json')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('Wrote',path)
    for label,r in records.items():
        half=r['masks']['half']
        print(label,half.get('predictions',{}).get('1.0',half))
    print('Least-squares NMSE:',result['least_squares_alpha1_sigma_0_1_nmse'])
    print('Max clean completion error:',max(rotation_errors))


if __name__=='__main__':
    main()
