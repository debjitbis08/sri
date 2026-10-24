"""Exact incremental support and residual checks for the coarse hierarchy.

Run: .venv/bin/python -m experiments.incremental_surface_analysis
No timing benchmark, sampled data, or running-experiment outputs.
"""
import os

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import json
from pathlib import Path

import numpy as np

from experiments.coarse_interior_analysis import A,B,C,N,MID,CORE,models


def main():
    systems,q=models(1.,1.,1.)
    _,h,f=systems[0]
    m=float(np.linalg.eigvalsh(h)[0])
    cases={}
    cases['unchanged']=np.zeros(N)
    cases['one_site']=np.eye(N)[0]
    cases['fine_redistribution']=np.eye(N)[0]-np.eye(N)[1]
    cases['middle_redistribution']=np.eye(N)[0]-np.eye(N)[4]
    cases['whole_coarse_block']=np.r_[np.ones(8),np.zeros(24)]
    rows={}
    for label,delta in cases.items():
        forcing=f[:,:N]@delta
        full=np.linalg.solve(h,forcing)
        # Incremental encoder + coarse response + skip-connected surface correction.
        dc=C@delta
        dx=delta-.25*C.T@dc
        dy=.25*dc
        dz=A@delta-.5*B.T@dc
        cached=np.r_[dx,dz,dy]
        assert np.max(np.abs(full-cached))<1e-12
        support=np.flatnonzero(np.abs(cached)>1e-12)
        local=np.zeros_like(full)
        if len(support):
            local[support]=np.linalg.solve(h[np.ix_(support,support)],forcing[support])
        # This verifies a proposed support; it does not claim support discovery is free.
        residual=forcing-h@local
        assert np.max(np.abs(local-full))<1e-12
        assert np.max(np.abs(residual))<1e-12
        rows[label]=dict(changed_inputs=int(np.count_nonzero(delta)),
                        changed_surface=int(np.count_nonzero(np.abs(dx)>1e-12)),
                        changed_middle=int(np.count_nonzero(np.abs(dz)>1e-12)),
                        changed_core=int(np.count_nonzero(np.abs(dy)>1e-12)),
                        total_changed_states=len(support),
                        local_solve_global_residual_norm=float(np.linalg.norm(residual)),
                        cached_forward_difference_error=float(np.max(np.abs(cached-full))))
    # A sparse guess can be wrong: modify only the initially changed surface site.
    forcing=f[:,:N]@cases['one_site']
    exact=np.linalg.solve(h,forcing)
    guess=np.zeros(len(h));guess[0]=forcing[0]/h[0,0]
    residual=forcing-h@guess
    error=float(np.linalg.norm(exact-guess))
    bound=float(np.linalg.norm(residual)/m)
    assert error<=bound
    result=dict(status='exact support and residual analysis; no wall-time speedup claimed',
                total_states=len(h),minimum_hessian_eigenvalue=m,cases=rows,
                one_site_only_guess=dict(error_norm=error,residual_norm=float(np.linalg.norm(residual)),
                                         certified_error_upper_bound=bound),
                assumptions=['fixed coarse context','exact previous equilibrium','batch changes known',
                             'unit parameters','disjoint coarse groups','incremental output allowed'])
    path=Path('reports/incremental-surface-analysis.json')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
