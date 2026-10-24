"""Exact responses to related inputs in the 32 -> 8 -> 4 summary hierarchy.

Run: .venv/bin/python -m experiments.related_input_analysis
No sampled data, fitting, or running-experiment outputs are used.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path

import numpy as np

from experiments.coarse_interior_analysis import A, B, C, N, MID, CORE, models


def main():
    fine=np.zeros(N);fine[:2]=[1,-1];fine/=np.linalg.norm(fine)
    middle=np.zeros(N);middle[:4]=1;middle[4:8]=-1;middle/=np.linalg.norm(middle)
    broad=C.T[:,0]
    deltas=dict(fine=fine,middle=middle,broad=broad)
    base=np.cos(2*np.pi*np.arange(N)/N)+.25*np.cos(8*2*np.pi*np.arange(N)/N)
    context=C@base
    records={}
    for eta in (0.,1.):
        systems,q=models(1.,1.,eta)
        _,h,f=systems[0]
        g=np.linalg.solve(h,f)
        eig,vec=np.linalg.eigh(h)
        baseline=np.linalg.solve(h,f@np.r_[base,context])
        rows={}
        for name,delta in deltas.items():
            expected=g[:,:N]@delta
            updated=np.linalg.solve(h,f@np.r_[base+delta,context])
            assert np.max(np.abs(updated-baseline-expected))<1e-12
            times={}
            for t in (0.,.1,1.,3.,10.):
                remaining=vec@(np.exp(-eig*t)*(vec.T@expected))
                response=expected-remaining
                if name in ('fine','middle'):
                    assert np.max(np.abs(response[-CORE:]))<1e-12
                if name=='fine':
                    assert np.max(np.abs(response[N:]))<1e-12
                times[str(t)]=dict(surface_change_norm=float(np.linalg.norm(response[:N])),
                                   middle_change_norm=float(np.linalg.norm(response[N:N+MID])),
                                   core_change_norm=float(np.linalg.norm(response[-CORE:])),
                                   remaining_full_state_error=float(np.linalg.norm(remaining)))
            rows[name]=dict(surface_delta_norm=float(np.linalg.norm(expected[:N])),
                            middle_delta_norm=float(np.linalg.norm(expected[N:N+MID])),
                            core_delta_norm=float(np.linalg.norm(expected[-CORE:])),transient=times)
            # Evaluate all energy terms at the new equilibrium and compare minimum-energy formula.
            x,z,y=updated[:N],updated[N:N+MID],updated[-CORE:]
            energy=.5*np.sum((x-base-delta)**2)+.5*np.sum((A@x-z)**2)
            energy+=.5*np.sum((B@z-y)**2)+eta/2*np.sum((y-context)**2)
            predicted_energy=q/(2*(1+q))*np.linalg.norm(C@(base+delta)-context)**2
            assert abs(energy-predicted_energy)<1e-12
            rows[name]['minimum_energy']=float(energy)
        records[str(eta)]=rows
    # Explicit context-reuse assumption: previous coarse estimate with variance w per
    # coordinate, independent current nuisance variance v, true broad change squared D.
    m,v,w,gain=CORE,.5,.5,.75
    transfer={}
    for squared_change in (0.,4.,12.,16.):
        mse=gain**2*m*v+(1-gain)**2*(m*w+squared_change)
        transfer[str(squared_change)]=dict(coarse_mse_with_context=mse,
                                          coarse_mse_without_context=m*v,
                                          beneficial=bool(mse<m*v))
    result=dict(status='deterministic related-input response and moment identities; no learned model',
                response=records,
                context_reuse=dict(coarse_dimension=m,new_noise_variance=v,
                                   old_context_noise_variance=w,data_weight=gain,
                                   squared_true_coarse_change_threshold=(1+gain)/(1-gain)*m*v-m*w,
                                   examples=transfer))
    path=Path('reports/related-input-analysis.json')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('Wrote',path)
    for eta,rows in records.items():
        print('Context strength',eta)
        for name,row in rows.items():
            print(name, row['surface_delta_norm'],row['middle_delta_norm'],row['core_delta_norm'])
    print('Context-reuse comparison:',transfer)


if __name__=='__main__':
    main()
