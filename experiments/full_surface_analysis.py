"""Exact full-surface denoising and settling calculations; no sampled data.

Run: .venv/bin/python -m experiments.full_surface_analysis
Uses the separate shared-surface analysis, not any running experiment output.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from experiments.shared_surface_analysis import F, L, N, interior


def target_dynamics(lam):
    """Unit-rate gradient flow in a target mode, no centre, initially zero."""
    h = np.diag([1+lam]+[2.]*(L-2)+[1.])
    h += np.diag([-1.]*(L-1), 1)+np.diag([-1.]*(L-1), -1)
    eigen, vec = np.linalg.eigh(h)
    weights = lam * vec[0]**2/eigen
    assert abs(weights.sum()-1)<1e-11
    def residual(t):
        return float(weights @ np.exp(-eigen*t))
    upper = 1/eigen[0]
    while residual(upper)>.05:
        upper *= 2
    return dict(slowest_target_time_constant=float(1/eigen[0]),
                surface_step_t95=float(brentq(lambda t:residual(t)-.05,0,upper)))


def main():
    rows={}
    for label, centre in [('current_centre',True),('free_terminal',False)]:
        h,s,symbol,error=interior('selective',centre)
        records=[]
        for lam in (.03,.1,.3,1.,3.,10.):
            t=np.linalg.solve(s+lam*np.eye(N),lam*np.eye(N))
            g=lam/(lam+symbol)
            spectral=np.fft.ifft(np.fft.fft(np.eye(N),axis=0)*g[:,None],axis=0).real
            assert np.max(np.abs(t-spectral))<1e-9
            full=h.copy()
            full[:N,:N]+=lam*np.eye(N)
            forcing=np.zeros((len(h),N));forcing[:N]=lam*np.eye(N)
            spatial=np.linalg.solve(full,forcing)[:N]
            assert np.max(np.abs(spatial-t))<1e-9
            # Coefficients have second moment 4I; true input signal power is 1/2.
            bias=float(4*np.linalg.norm(t@F-F)**2/N)
            noise=float(.5*np.linalg.norm(t)**2/N)
            record=dict(lambda_tie=lam,signal_gain=float(g[1]),
                        nuisance_power_fraction=float(np.mean(g*g)),
                        nmse_alpha1_sigma0_1=(bias+noise+.01)/.5,
                        nmse_alpha0_5_sigma0_1=(bias+noise*.25+.01)/.5,
                        clean_bias_mse=bias,
                        full_hessian_min_eigenvalue=float(np.linalg.eigvalsh(full)[0]),
                        surface_spectral_max_error=float(np.max(np.abs(t-spectral))),
                        spatial_surface_max_error=float(np.max(np.abs(spatial-t))))
            if not centre:
                assert bias<1e-18
                record.update(target_dynamics(lam))
            records.append(record)
        rows[label]=records
    targets={1,8,24,31}
    _,_,s,_=interior('selective',False)
    smin=min(s[k] for k in range(N) if k not in targets)
    # Sufficient design condition for total nuisance power fraction <= 0.15,
    # including the unavoidable 1/8 retained in the four target coordinates.
    power_target=.15
    epsilon=np.sqrt((N*power_target-4)/28)
    lam_limit=float(epsilon*smin/(1-epsilon))
    result=dict(status='exact operator and moment calculations; no random draws',
                records=rows,minimum_unwanted_surface_stiffness=float(smin),
                unbiased_projection_nmse_alpha1_sigma0_1=.125+.02,
                linear_mmse_nmse_alpha1_sigma0_1=(4*(4*.5/(4+.5))/32+.01)/.5,
                maximum_lambda_sufficient_for_nuisance_power_0_15=lam_limit,
                target_stiffness_tolerance_for_gain_0_95_at_lambda_0_1=.1/19)
    path=Path('reports/full-surface-analysis.json')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('Wrote',path)
    for label,records in rows.items():
        print(label)
        for r in records:
            print(r['lambda_tie'],r['signal_gain'],r['nuisance_power_fraction'],
                  r['nmse_alpha1_sigma0_1'],r.get('surface_step_t95'))
    print('Sufficient lambda bound:',lam_limit)


if __name__=='__main__':
    main()
