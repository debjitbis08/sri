"""Post-run local-versus-global and exact-inference audit, with no new learning."""
import json
import hashlib
from pathlib import Path
import resource
import sys
import time
from experiments import local_consolidation as base
import numpy as np


def main():
    root=base.ROOT;output=root/'reports/local-consolidation-v1-audit.json'
    runtime_output=root/'reports/local-consolidation-v1-audit-runtime.json'
    if output.exists() or runtime_output.exists():raise RuntimeError('Refusing overwrite')
    wall,cpu=time.perf_counter(),time.process_time()
    source=root/'reports/local-consolidation-v1-results.json'
    science=json.loads(source.read_text());report={};exact_solves=0
    for seed in (0,1,2):
        _,test,_=base.fixture(seed);rows={}
        for arm in ('frozen','energy','support_stability'):
            saved=science['seeds'][str(seed)]['arms'][arm]['final_parameters']
            p={k:np.array(v) for k,v in saved.items()};before=base.digest(p)
            errors=[]
            for module in range(3):
                for coordinate in range(6):
                    mask=np.ones((3,8),bool);mask[module,coordinate]=False
                    state,_,_=base.infer(test,p,mask)
                    errors.append((state[:,module,coordinate]-test[:,module,coordinate])**2)
            local=np.mean(errors,axis=0)
            q=base.build(p);exact_errors=[]
            for module in range(3):
                mask=np.ones((3,8),bool);mask[module]=False
                idx=np.array([m*14+j for m in range(3) for j in range(8) if mask[m,j]])
                free=np.setdiff1d(np.arange(42),idx);state=np.zeros((len(test),42));state[:,idx]=test[:,mask]
                state[:,free]=np.linalg.solve(q[np.ix_(free,free)],-(state[:,idx]@q[np.ix_(free,idx)].T).T).T
                exact_solves+=1
                assert np.max(abs((state@q.T)[:,free]))<1e-10
                exact_errors.append(np.mean((state[:,14*module:14*module+8]-test[:,module])**2,axis=1))
            exact=np.mean(exact_errors,axis=0)
            assert base.digest(p)==before
            rows[arm]={'single_coordinate_rmse_by_structure':[float(np.sqrt(np.mean(local[:12]))),float(np.sqrt(np.mean(local[12:])))],
                'exact_missing_module_rmse_by_structure':[float(np.sqrt(np.mean(exact[:12]))),float(np.sqrt(np.mean(exact[12:])))],
                'finite_missing_module_rmse_by_structure':science['seeds'][str(seed)]['arms'][arm]['evaluations'][-1]['rmse_by_structure']}
        report[str(seed)]=rows
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    timing={'purpose':'post-result diagnosis; no new parameter updates, candidate selection or extra trajectory horizons',
            'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,'cpu_seconds_before_runtime_serialization':time.process_time()-cpu,
            'counts':base.COUNTS,'exact_stationary_solves':exact_solves,'new_learning_proposals':0,
            'complexity':'162 fixed-parameter 32-step inference calls plus 27 O(S^3) stationary solves, S=42',
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_scope':'Linux single process lifetime peak',
            'command':' '.join(sys.argv),'environment_reference':'local-consolidation-v1-runtime.json',
            'environment_note':'same interpreter and machine, single-thread environment set before importing NumPy',
            'hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(base.__file__),source,base.CONTRACT)}}
    runtime_output.write_text(json.dumps(timing,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
