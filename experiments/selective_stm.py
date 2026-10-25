"""Selective short-term state memory; all long-term weights stay frozen."""
from experiments import standing_pattern_hierarchy_chain as base
import hashlib
import io
import contextlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

ROOT=base.ROOT
CONTRACT=ROOT/'docs/selective-stm-v1-contract.md'
RESULT=ROOT/'reports/selective-stm-v1-results.json'
RUNTIME=ROOT/'reports/selective-stm-v1-runtime.json'
SOURCE=ROOT/'reports/standing-pattern-hierarchy-v3-results.json'
ALPHA=.2
COUNTS={}
PHASES={}

def count(k,n=1): COUNTS[k]=COUNTS.get(k,0)+n

@contextlib.contextmanager
def phase(name):
    wall,cpu=time.perf_counter(),time.process_time()
    yield
    PHASES[name]={'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu}

def objective(x,w,m,strength,grad=False):
    value=base.energy(x,w,gradient=grad)
    delta=x[1]-m
    if not grad: return value+strength/2*float(delta@delta)
    val,gx,_=value
    gx[1]+=strength*delta
    return val+strength/2*float(delta@delta),gx,-strength*delta

def step(x,w,m,strength,observed,write):
    val,gx,_=objective(x,w,m,strength,True)
    if observed is not None: gx[0,:8]=0
    for j in range(21):
        xx=x-.1*2.**(-j)*gx
        after=objective(xx,w,m,strength)
        count('trials')
        if np.isfinite(after) and after<=val+1e-12:
            if observed is not None: assert np.array_equal(xx[0,:8],observed)
            mm=(1-ALPHA)*m+ALPHA*xx[1] if write else m
            if write:
                count('trace_writes')
                assert objective(xx,w,mm,strength)<=after+1e-12
            count('activity_updates');count('halvings',j)
            return xx,mm
    raise AssertionError('Line search exhausted')

def main():
    if RESULT.exists() or RUNTIME.exists(): raise RuntimeError('Refusing overwrite')
    wall,cpu=time.perf_counter(),time.process_time()
    with phase('load'):
        prior=json.loads(SOURCE.read_text()); w=np.array(prior['quadratic_weights'])
        frozen=w.tobytes(); fixtures=prior['arms']['quadratic_frozen']['queries']
    with phase('checks'):
        x=np.linspace(-.6,.7,45).reshape(3,15); m=np.linspace(.3,-.2,15)
        _,gx,gm=objective(x,w,m,1.,True); worst=0.
        for arr,g in ((x,gx),(m,gm)):
            for idx in np.ndindex(arr.shape):
                old=arr[idx];arr[idx]=old+1e-6;plus=objective(x,w,m,1.)
                arr[idx]=old-1e-6;minus=objective(x,w,m,1.);arr[idx]=old
                worst=max(worst,abs((plus-minus)/2e-6-g[idx]))
        assert worst<1e-7
        observed=x[0,:8].copy()
        got,_=step(x,w,m,0.,observed,False)
        expected,_=base.update(x,w,observed,base.GRADE,False)
        assert np.array_equal(got,expected)
    arms={}
    for strength in (0.,.1,1.):
        with phase(f'stm_{strength}'):
            episodes=[]
            for fixture in fixtures:
                s=fixture['latent_fixture_value']
                replacement=min(fixtures,key=lambda f:abs(f['latent_fixture_value']+s))
                old=np.array(fixture['target_outer']); new=np.array(replacement['target_outer'])
                x=base.expand(0.,w);m=x[1].copy();scalar=0.;encoded=None
                phases=[]
                protocol=[('encode',1024,fixture,old,True),('recall',1024,None,old,False),
                          ('replace',256,replacement,new,True),('recall_replacement',256,None,new,False)]
                count('episodes')
                for name,budget,cue,target,write in protocol:
                    observed=np.array(cue['input_outer']) if cue else None
                    if observed is not None: x[0,:8]=observed
                    initial_m=m.copy(); rows=[]
                    for t in range(budget+1):
                        if t in (0,1,16,64,256,1024):
                            val,g,_=objective(x,w,m,strength,True)
                            if observed is not None:g[0,:8]=0
                            out=x[-1,:8];direct=base.expand(scalar,w)[-1,:8]
                            rows.append({'step':t,'output':out.tolist(),'mse':float(np.mean((out-target)**2)),
                                         'old_target_mse':float(np.mean((out-old)**2)),
                                         'encoded_output_mse':None if encoded is None else float(np.mean((out-encoded)**2)),
                                         'direct_mse':float(np.mean((direct-target)**2)),
                                         'free_gradient_norm':float(np.linalg.norm(g)),
                                         'trace_change':float(np.linalg.norm(m-initial_m)),'energy':val})
                            count('outer_readouts');count('direct_readouts')
                        if t<budget:
                            x,m=step(x,w,m,strength,observed,write)
                            if write: scalar=(1-ALPHA)*scalar+ALPHA*float(observed.mean());count('scalar_trace_writes')
                    if name=='encode':encoded=x[-1,:8].copy()
                    phases.append({'phase':name,'readouts':rows})
                assert w.tobytes()==frozen
                episodes.append({'initial_fixture_scalar':s,'phases':phases})
            summary={}
            for p in range(4):
                rows=[]
                for k,row in enumerate(episodes[0]['phases'][p]['readouts']):
                    rr=[e['phases'][p]['readouts'][k] for e in episodes]
                    rows.append({'step':row['step'],'rmse':float(np.sqrt(np.mean([r['mse'] for r in rr]))),
                                 'old_target_rmse':float(np.sqrt(np.mean([r['old_target_mse'] for r in rr]))),
                                 'direct_rmse':float(np.sqrt(np.mean([r['direct_mse'] for r in rr]))),
                                 'unsettled':sum(r['free_gradient_norm']>1e-5 for r in rr),
                                 'mean_trace_change':float(np.mean([r['trace_change'] for r in rr]))})
                summary[episodes[0]['phases'][p]['phase']]=rows
            arms[str(strength)]={'summary':summary,'episodes':episodes}
    with phase('scientific_serialization'):
        results={'weights_frozen':True,'gradient_max_error':worst,'lambda_zero_matches_v3_step':True,
                 'alpha':ALPHA,'arms':arms,'counts':COUNTS,'base_counts':base.COUNTS,
                 'scope':'Selective activity trace; no sleep/dream learning or long-term weight updates.'}
        RESULT.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):np.show_config()
    runtime={'hashes':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),Path(base.__file__),CONTRACT,SOURCE,RESULT]},
             'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'dirty_status':subprocess.check_output(['git','status','--short'],text=True),
             'command':'.venv/bin/python -m experiments.selective_stm','seed_policy':'Deterministic frozen fixtures',
             'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'cpu':platform.processor() or platform.machine(),
             'blas':stream.getvalue(),'threads':{k:base.os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
             'processes':1,'workers':0,'dtype':'float64','accelerator':None,'phases':PHASES,
             'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,
             'cpu_seconds_before_runtime_serialization':time.process_time()-cpu,
             'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'rss_convention':'Linux KiB; one process','counts':COUNTS,'base_counts':base.COUNTS,
             'new_fits':0,'selection_tuning_candidates':0,'selection_tuning_seconds':0,
             'independent_heldout_pairs':8,'episodes':24,'cue_presentations':48,
             'deployment_array_bytes':1272,'stm_bytes':120,'direct_scalar_stm_bytes':8,
             'optimizer_bytes':0,'replay_bytes':0,'new_fitted_statistics_bytes':0,
             'historical_training':'Nine pairs; 28 fits in v3; cost in standing-pattern-hierarchy-v3-runtime.json',
             'limitations':{'manual_development_time':None,'allocator_peak':None,'hardware_limits':None,
                            'reason':'Not instrumented; whole-run measurement is not a latency benchmark'},
             'timing_scope':'Includes checks, episode inference, memory writes, comparator and scoring, serialization; excludes runtime JSON serialization'}
    RUNTIME.write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'summary':{k:v['summary'] for k,v in arms.items()},'checks':worst,'counts':COUNTS,
                      'seconds':runtime['wall_seconds_before_runtime_serialization']},indent=2))

if __name__=='__main__':main()
