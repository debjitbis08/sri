"""Analytical comparison of coarse interiors; no sampled experiment or running outputs.

Run: .venv/bin/python -m experiments.coarse_interior_analysis
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path

import numpy as np


N, MID, CORE = 32, 8, 4


def block_restriction(n, m):
    """Orthonormal rows: each sums a disjoint block with coefficient 1/sqrt(block size)."""
    block = n//m
    out = np.zeros((m, n))
    for row in range(m):
        out[row, row*block:(row+1)*block] = 1/np.sqrt(block)
    assert np.max(np.abs(out@out.T-np.eye(m))) < 1e-14
    return out


A = block_restriction(N, MID)
B = block_restriction(MID, CORE)
C = B@A
P = C.T@C
Q = np.eye(N)-P


def models(lam, kappa, eta):
    # Hierarchy: lambda/2|x-h|^2 + kappa/2|Ax-z|^2
    #          + kappa/2|Bz-y|^2 + eta/2|y-p|^2.
    h = np.zeros((N+MID+CORE, N+MID+CORE))
    h[:N, :N] = lam*np.eye(N)+kappa*A.T@A
    h[N:N+MID, N:N+MID] = kappa*(np.eye(MID)+B.T@B)
    h[-CORE:, -CORE:] = (kappa+eta)*np.eye(CORE)
    h[:N, N:N+MID] = -kappa*A.T
    h[N:N+MID, :N] = -kappa*A
    h[N:N+MID, -CORE:] = -kappa*B.T
    h[-CORE:, N:N+MID] = -kappa*B
    forcing = np.zeros((len(h), N+CORE))
    forcing[:N, :N] = lam*np.eye(N)
    forcing[-CORE:, N:] = eta*np.eye(CORE)

    # Eliminate z. Two equal springs in series yield kappa/2.
    spring = kappa/2
    hc = np.block([[lam*np.eye(N)+spring*P, -spring*C.T],
                   [-spring*C, (spring+eta)*np.eye(CORE)]])
    fc = np.zeros((N+CORE,N+CORE))
    fc[:N, :N] = lam*np.eye(N)
    fc[-CORE:, N:] = eta*np.eye(CORE)

    effective = kappa*eta/(kappa+2*eta)
    hs = lam*np.eye(N)+effective*P
    fs = np.column_stack([lam*np.eye(N), effective*C.T])
    return [('hierarchy',h,forcing), ('compact',hc,fc), ('surface_only',hs,fs)], effective


def main():
    records={}
    for eta in (0.,1.):
        systems,q = models(1.,1.,eta)
        closed = np.column_stack([Q+P/(1+q), q/(1+q)*C.T])
        rows={}
        for label,h,f in systems:
            transfer=np.linalg.solve(h,f)
            err=float(np.max(np.abs(transfer[:N]-closed)))
            assert err<1e-12
            assert np.max(np.abs(Q@transfer[:N]-np.column_stack([Q,np.zeros((N,CORE))])))<1e-12
            eigen=np.linalg.eigvalsh(h)
            assert eigen[0]>0
            # Distinct scalar off-diagonal Hessian interactions, counting each once.
            edges=int(np.count_nonzero(np.abs(np.triu(h,1))>1e-12))
            rows[label]=dict(states=len(h),off_diagonal_interactions=edges,
                             surface_closed_form_max_error=err,
                             slowest_time_constant=float(1/eigen[0]),
                             hessian_eigenvalue_range=eigen[[0,-1]].tolist())
            if eta==0 and label=='hierarchy':
                assert np.max(np.abs(transfer[N:N+MID,:N]-A))<1e-12
                assert np.max(np.abs(transfer[-CORE:,:N]-C))<1e-12
            if label=='hierarchy':
                hierarchy_core=transfer[-CORE:]
            if label=='compact':
                assert np.max(np.abs(transfer[-CORE:]-hierarchy_core))<1e-12
        records[str(eta)]=dict(effective_surface_stiffness=q,
                               coarse_surface_gain=1/(1+q),
                               detail_surface_gain=1.,models=rows)
    theta=2*np.pi*np.arange(N)/N
    mode_power={}
    for k in (1,8):
        mode_power[str(k)]={}
        for label,v in [('cos',np.cos(k*theta)),('sin',np.sin(k*theta))]:
            mode_power[str(k)][label]=dict(mid_retained_power=float(np.linalg.norm(A@v)**2/np.linalg.norm(v)**2),
                                            core_retained_power=float(np.linalg.norm(C@v)**2/np.linalg.norm(v)**2))
    # Orthogonal decomposition: within-four-site detail (24), between-block detail (4), core (4).
    blocks=[np.eye(N)-A.T@A, A.T@A-P, P]
    for i,proj in enumerate(blocks):
        assert np.max(np.abs(proj@proj-proj))<1e-14
        for j in range(i):
            assert np.max(np.abs(proj@blocks[j]))<1e-14
    assert np.max(np.abs(sum(blocks)-np.eye(N)))<1e-14
    result=dict(status='deterministic matrix and moment analysis; no sampled data',
                models=records,mode_retained_power=mode_power,
                innovation_dimensions=[int(round(np.trace(proj))) for proj in blocks],
                alpha1_noise_variance_physical_averages=dict(surface=.5,mid=.5/4,core=.5/8),
                maximum_total_white_noise_reduction_by_coarse_only_correction=CORE/N,
                eta1_perfect_context_nuisance_fraction=(N-CORE+CORE*.75**2)/N)
    path=Path('reports/coarse-interior-analysis.json')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
