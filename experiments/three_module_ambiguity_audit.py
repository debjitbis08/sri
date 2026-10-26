"""Post-run analytic audit of nonunique stationary solutions; no extra trajectories."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from experiments import three_module_ambiguity as base


def main():
    root=base.ROOT
    result=root/'reports/three-module-ambiguity-v1-audit.json'
    timing=root/'reports/three-module-ambiguity-v1-audit-runtime.json'
    if result.exists() or timing.exists():raise RuntimeError('Refusing overwrite')
    wall,cpu=time.perf_counter(),time.process_time()
    _,_,maps,_,q,arms=base.build()
    source=root/'reports/three-module-ambiguity-v1-results.json'
    science=json.loads(source.read_text())
    report={};svds=0;solves=0
    for arm,a in arms.items():
        rows=[]
        for i,f in enumerate(science['fixtures']):
            idx=np.array(f['indices']);free=np.setdiff1d(np.arange(42),idx)
            b=a[np.ix_(free,free)];rhs=-a[np.ix_(free,idx)]@f['values']
            u,s,vt=np.linalg.svd(b);svds+=1
            rank=int(np.sum(s>1e-10));left_null=u[:,rank:]
            # y(0)=0. Left-null projections of y remain zero because L.T rhs=0.
            assert np.linalg.norm(left_null.T@rhs)<1e-10
            augmented=np.vstack((b,left_null.T))
            target=np.concatenate((rhs,np.zeros(left_null.shape[1])))
            y,_,aug_rank,_=np.linalg.lstsq(augmented,target,rcond=None);solves+=1
            assert aug_rank==len(free)
            assert np.linalg.norm(b@y-rhs)<1e-10
            state=np.zeros(42);state[idx]=f['values'];state[free]=y
            last=science['arms'][arm][i]['trajectory'][-1]
            true=maps[2]@np.array([f['a'],f['b']]) if f['kind']=='complete' else None
            rows.append({'fixture_index':i,'free_nullity':len(free)-rank,
                'invariant_selected_stationary_output':state[28:36].tolist(),
                'last_output_rmse_to_selected_stationary':float(np.sqrt(np.mean((np.array(last['output'])-state[28:36])**2))),
                'selected_stationary_target_rmse':None if true is None else float(np.sqrt(np.mean((state[28:36]-true)**2)))})
        report[arm]={'cases':rows,'complete_stationary_target_rmse':float(np.sqrt(np.mean([r['selected_stationary_target_rmse']**2 for r in rows[:16]]))),
                    'complete_final_to_selected_stationary_rmse':float(np.sqrt(np.mean([r['last_output_rmse_to_selected_stationary']**2 for r in rows[:16]])))}
    result.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    prior=json.loads((root/'reports/three-module-ambiguity-v1-runtime.json').read_text())
    runtime={'purpose':'post-run diagnosis of minimum-norm versus dynamically selected stationary solutions; no tuning or new trajectories',
        'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,'cpu_seconds_before_runtime_serialization':time.process_time()-cpu,
        'svds':svds,'lstsq_solves':solves,'fits':0,'candidates_scored':0,'updates':0,
        'complexity':'112 cases, dense SVD and augmented least squares O(S^3) time, O(S^2) workspace each, S<=42',
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_scope':'Linux single process lifetime peak',
        'command':' '.join(sys.argv),'dtype':'float64','seed_policy':'no randomness',
        'environment_from_main_run':{k:prior[k] for k in ('python','numpy','os','cpu','cpu_limit','memory_limit','blas','thread_environment','threadpools','workers')},
        'environment_note':'same interpreter and machine as main run; no intervening configuration changes',
        'hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(base.__file__),source,base.CONTRACT)}}
    timing.write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k!='cases'} for a,r in report.items()},indent=2))

if __name__=='__main__':main()
