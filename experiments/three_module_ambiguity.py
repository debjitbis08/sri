"""Three frozen 8→4→2 modules with separately clamped evidence and outer output."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/three-module-ambiguity-v1-contract.md'
PREFIX = ROOT / 'reports/three-module-ambiguity-v1'
ARMS = ('reciprocal', 'forward_only', 'no_middle_to_input', 'no_output_to_middle')
HORIZONS = (0, 1, 4, 16, 64, 256, 1024, 4096)
COUNTS = {'inference_calls': 0, 'updates': 0, 'direct_calls': 0,
          'stationary_solves': 0, 'lstsq_calls': 0, 'svd_calls': 0}
PHASES = {}

@contextlib.contextmanager
def phase(name):
    wall, cpu = time.perf_counter(), time.process_time()
    yield
    PHASES[name] = {'wall_seconds': time.perf_counter()-wall,
                    'cpu_seconds': time.process_time()-cpu}

def least_squares(a, b):
    COUNTS['lstsq_calls'] += 1
    return np.linalg.lstsq(a, b, rcond=None)

def build():
    r = np.array([[1.,0], [0,1], [2**-.5,2**-.5], [2**-.5,-2**-.5]]) / np.sqrt(2)
    had = np.array([[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]]) / 2
    u = np.vstack((np.eye(4), had)) / np.sqrt(2)
    ds = np.stack((u, u, u[::-1]*np.array([1,-1]*4)[:,None]))
    hs = np.stack((r, r[[1,0,2,3]], r))
    local = np.zeros((36,42))
    for m in range(3):
        offset, row = 14*m, 12*m
        local[row:row+8,offset:offset+8] = np.eye(8)
        local[row:row+8,offset+8:offset+12] = -ds[m]
        local[row+8:row+12,offset+8:offset+12] = np.eye(4)
        local[row+8:row+12,offset+12:offset+14] = -hs[m]
    edges=[]
    for left in (0,1):
        e=np.zeros((2,42))
        e[:,14*left+12:14*left+14]=np.eye(2)
        e[:,14*(left+1)+12:14*(left+1)+14]=-np.eye(2)
        edges.append(e)
    residual=np.vstack((local,*edges))
    q=residual.T@residual
    dynamics={}
    for arm in ARMS:
        a=local.T@local
        for j,e in enumerate(edges):
            contribution=e.T@e
            backward=(arm=='reciprocal' or
                      (arm=='no_middle_to_input' and j==1) or
                      (arm=='no_output_to_middle' and j==0))
            if not backward:
                contribution[14*j:14*(j+1),:]=0
            a+=contribution
        dynamics[arm]=a
    maps=np.einsum('mnh,mhd->mnd',ds,hs)
    return ds,hs,maps,residual,q,dynamics

def fixtures():
    out=[]
    vals=(-1.,-.5,.5,1.)
    for a in vals:
        for b in vals:
            out.append({'kind':'complete','a':a,'b':b,'indices':[0,14], 'values':[a/2,b/2]})
    for a in vals:
        out.append({'kind':'missing_context','a':a,'b':None,'indices':[0],'values':[a/2]})
    for b in vals:
        out.append({'kind':'missing_input','a':None,'b':b,'indices':[14],'values':[b/2]})
    for a in (-1.,1.):
        for b in (-1.,1.):
            out.append({'kind':'contradictory','a':a,'b':b,'indices':[0,1,14], 'values':[a/2,b/2,-b/2]})
    return out

def direct(f,maps):
    COUNTS['direct_calls']+=1
    cue_map=np.array([maps[i//14,i%14] for i in f['indices']])
    z,_,rank,_=least_squares(cue_map,np.array(f['values']))
    return maps[2]@z,z,int(rank),float(np.linalg.norm(cue_map@z-f['values']))

def stationary(f,a):
    COUNTS['stationary_solves']+=1
    idx=np.array(f['indices']);free=np.setdiff1d(np.arange(42),idx)
    state=np.zeros(42);state[idx]=f['values']
    state[free]=least_squares(a[np.ix_(free,free)],-a[np.ix_(free,idx)]@state[idx])[0]
    return state,float(np.linalg.norm((a@state)[free]))

def run(f,a,budget,record=False,q=None):
    COUNTS['inference_calls']+=1
    idx=np.array(f['indices']);state=np.zeros(42);state[idx]=f['values']
    rows=[]
    for t in range(budget+1):
        if record and t in HORIZONS:
            gradient=a@state;gradient[idx]=0
            rows.append({'step':t,'output':state[28:36].tolist(),
                         'inner':state.reshape(3,14)[:,12:].tolist(),
                         'update_residual':float(np.linalg.norm(gradient)),
                         'reciprocal_energy':float(.5*state@q@state),
                         'clamp_error':float(np.max(np.abs(state[idx]-f['values'])))})
        if t==budget:break
        gradient=a@state;gradient[idx]=0
        before=float(.5*state@q@state) if record and np.array_equal(a,q) else None
        state-=.1*gradient
        COUNTS['updates']+=1
        if record:
            assert np.array_equal(state[idx],np.array(f['values']))
            if before is not None:assert .5*state@q@state<=before+1e-12
    return rows if record else state[28:36]

def compatible_distance(output,f,maps):
    reference,z,rank,inconsistency=direct(f,maps)
    if inconsistency>1e-10:return None
    if rank==2:return float(np.sqrt(np.mean((output-reference)**2)))
    cue_map=np.array([maps[i//14,i%14] for i in f['indices']])
    COUNTS['svd_calls']+=1
    _,_,vt=np.linalg.svd(cue_map,full_matrices=True)
    directions=maps[2]@vt[rank:].T
    projection=directions@least_squares(directions,np.asarray(output)-reference)[0]
    return float(np.sqrt(np.mean((output-reference-projection)**2)))

def bench(fn):
    fn();wall=[];cpu=[]
    for _ in range(5):
        w,c=time.perf_counter(),time.process_time();fn()
        wall.append((time.perf_counter()-w)/16);cpu.append((time.process_time()-c)/16)
    def summary(x):return {'median':float(np.median(x)),'min':min(x),'max':max(x)}
    return {'warmup_batches':1,'repeats':5,'batch_size':16,
            'wall_seconds_per_query':summary(wall),'cpu_seconds_per_query':summary(cpu)}

def verify(ds,hs,maps,residual,q,dynamics,fs):
    assert ds.shape==(3,8,4) and hs.shape==(3,4,2)
    assert all(np.linalg.matrix_rank(m)==2 for m in maps)
    s=np.linspace(-.7,.9,42);g=q@s;worst=0.
    for i in range(42):
        plus=s.copy();minus=s.copy();plus[i]+=1e-6;minus[i]-=1e-6
        numeric=(.5*np.sum((residual@plus)**2)-.5*np.sum((residual@minus)**2))/2e-6
        worst=max(worst,abs(numeric-g[i]))
    assert worst<1e-8
    assert np.array_equal(q[:14,28:],np.zeros((14,14)))
    assert np.allclose(q,dynamics['reciprocal'])
    for arm,a in dynamics.items():
        for j in (0,1):
            back=(arm=='reciprocal' or (arm=='no_middle_to_input' and j==1) or
                  (arm=='no_output_to_middle' and j==0))
            left=slice(14*j+12,14*j+14);right=slice(14*(j+1)+12,14*(j+1)+14)
            assert np.array_equal(a[left,right],-np.eye(2) if back else np.zeros((2,2)))
            assert np.array_equal(a[right,left],-np.eye(2))
        f=fs[0];initial=np.zeros(42);initial[f['indices']]=f['values']
        grad=a@initial;grad[f['indices']]=0
        assert np.allclose(run(f,a,1), (initial-.1*grad)[28:36])
    # Independent local-residual gradient, not merely another Hessian product.
    independent=np.zeros((3,14));states=s.reshape(3,14)
    for m in range(3):
        x,h,c=states[m,:8],states[m,8:12],states[m,12:]
        e=x-ds[m]@h;v=h-hs[m]@c
        independent[m,:8]=e;independent[m,8:12]=-ds[m].T@e+v
        independent[m,12:]=-hs[m].T@v
    for j in (0,1):
        delta=states[j,12:]-states[j+1,12:]
        independent[j,12:]+=delta;independent[j+1,12:]-=delta
    assert np.allclose(independent.ravel(),g)
    for f in fs:
        output,z,rank,inc=direct(f,maps)
        if f['kind']=='complete':
            assert np.allclose(z,[f['a'],f['b']]) and inc<1e-12
        elif f['kind']=='contradictory':assert inc>0.1
        else:
            assert rank==1 and inc<1e-12
            j=1 if f['kind']=='missing_context' else 0
            assert compatible_distance(output+maps[2,:,j],f,maps)<1e-12
    assert np.all(np.linalg.norm(maps[2],axis=0)>0)
    max_eigen=float(np.linalg.eigvalsh(q).max())
    assert .1*max_eigen<2
    return {'gradient_max_error':worst,'largest_hessian_eigenvalue':max_eigen,
            'checks':'gradients, local matrix equivalence, topology, directional deletions, clamps, descent, maps, cue rank and contradictions'}

def main():
    results=Path(str(PREFIX)+'-results.json');runtime_path=Path(str(PREFIX)+'-runtime.json')
    if results.exists() or runtime_path.exists():raise RuntimeError('Refusing overwrite')
    wall,cpu=time.perf_counter(),time.process_time()
    with phase('construction'):
        ds,hs,maps,residual,q,dynamics=build();fs=fixtures()
        frozen=[v.tobytes() for v in (ds,hs,maps,residual,q,*dynamics.values())]
    with phase('verification'):
        checks=verify(ds,hs,maps,residual,q,dynamics,fs)
    with phase('scientific_inference_and_diagnostics'):
        science={'checks':checks,'fixtures':fs,'arms':{},'direct':[],'summary':{}}
        references=[];optima=[]
        for f in fs:
            out,z,rank,inc=direct(f,maps);references.append(out)
            optimal,res=stationary(f,q);optima.append(optimal)
            science['direct'].append({'output':out.tolist(),'code':z.tolist(),'rank':rank,
                'compatible_set_dimension':2-rank if inc<1e-10 else None,
                'evidence_inconsistency':inc,'reciprocal_optimal_output':optimal[28:36].tolist(),
                'reciprocal_optimal_energy':float(.5*optimal@q@optimal),'optimum_residual':res})
        for arm,a in dynamics.items():
            episodes=[]
            for i,f in enumerate(fs):
                fixed,fixed_residual=stationary(f,a)
                rows=run(f,a,4096,True,q)
                for row in rows:
                    out=np.array(row['output'])
                    row['compatible_set_rmse']=compatible_distance(out,f,maps)
                    row['own_stationary_output_rmse']=float(np.sqrt(np.mean((out-fixed[28:36])**2)))
                    row['reciprocal_optimal_output_rmse']=float(np.sqrt(np.mean((out-optima[i][28:36])**2)))
                episodes.append({'fixture_index':i,'stationary_output':fixed[28:36].tolist(),
                                 'stationary_residual':fixed_residual,'trajectory':rows})
            science['arms'][arm]=episodes
            summary={}
            for kind in ('complete','missing_context','missing_input','contradictory'):
                subset=[e for e,f in zip(episodes,fs) if f['kind']==kind];rows=[]
                for j,t in enumerate(HORIZONS):
                    obs=[e['trajectory'][j] for e in subset]
                    errors=[o['compatible_set_rmse'] for o in obs]
                    rows.append({'step':t,'compatible_set_rmse':None if errors[0] is None else float(np.sqrt(np.mean(np.square(errors)))),
                        'not_converged':sum(o['update_residual']>1e-5 for o in obs),
                        'max_update_residual':max(o['update_residual'] for o in obs),
                        'mean_reciprocal_energy':float(np.mean([o['reciprocal_energy'] for o in obs])),
                        'own_stationary_output_rmse':float(np.sqrt(np.mean([o['own_stationary_output_rmse']**2 for o in obs]))),
                        'reciprocal_optimal_output_rmse':float(np.sqrt(np.mean([o['reciprocal_optimal_output_rmse']**2 for o in obs])))})
                summary[kind]=rows
            # Paired intervention errors measure changed output, not movement alone.
            interventions={}
            for variable in ('a','b'):
                other='b' if variable=='a' else 'a';errors=[];response=[]
                for i,f in enumerate(fs[:16]):
                    for j,g in enumerate(fs[:16]):
                        if i<j and f[other]==g[other] and f[variable]!=g[variable]:
                            delta=np.array(episodes[j]['trajectory'][-1]['output'])-episodes[i]['trajectory'][-1]['output']
                            wanted=references[j]-references[i]
                            errors.append(float(np.mean((delta-wanted)**2)));response.append(float(np.linalg.norm(delta)))
                interventions[variable]={'pairs':len(errors),'delta_rmse':float(np.sqrt(np.mean(errors))),
                                         'minimum_output_change_norm':min(response)}
            summary['cue_interventions']=interventions
            summary['first_complete_quality_horizon']=next((r['step'] for r in summary['complete'] if r['compatible_set_rmse']<=1e-4),None)
            summary['first_complete_convergence_horizon']=next((r['step'] for r in summary['complete'] if r['not_converged']==0),None)
            science['summary'][arm]=summary
        assert frozen==[v.tobytes() for v in (ds,hs,maps,residual,q,*dynamics.values())]
    before_benchmark=COUNTS.copy()
    with phase('benchmarks'):
        complete=fs[:16]
        benchmarks={'direct':bench(lambda:[direct(f,maps) for f in complete])}
        for arm,a in dynamics.items():
            benchmarks[arm]={str(t):bench(lambda:[run(f,a,t) for f in complete]) for t in HORIZONS[1:]}
    with phase('scientific_serialization'):
        results.write_text(json.dumps(science,indent=2,sort_keys=True)+'\n')
    with phase('provenance'):
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):np.show_config()
        try:
            from threadpoolctl import threadpool_info
            threadpools=threadpool_info()
        except ImportError:threadpools=None
        paths=(Path(__file__),CONTRACT)
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    def limit(name):
        p=Path('/sys/fs/cgroup')/name
        return p.read_text().strip() if p.exists() else None
    runtime={'phases':PHASES,'benchmarks':benchmarks,'counts_total':COUNTS,'counts_before_benchmarks':before_benchmark,
        'counts_note':'lstsq calls use SVD internally; svd_calls counts additional explicit nullspace SVDs',
        'unique_fixtures':28,'complete_pairs':16,'scientific_episodes':112,'training_examples':0,'validation_examples':0,
        'fits':0,'optimizer_steps':0,'candidates_scored':0,'selection_tuning_seconds':0,'manual_design_seconds':None,
        'manual_design_reason':'not measured','hashes':hashes,'revision':revision,'dirty_status':dirty,
        'command':' '.join(sys.argv),'seed_policy':'no randomness; exhaustive declared fixtures',
        'python':sys.version,'numpy':np.__version__,'os':platform.platform(),'dtype':'float64',
        'cpu':platform.processor() or Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
        'cpu_count':os.cpu_count(),'affinity_cpu_count':len(os.sched_getaffinity(0)),
        'cpu_limit':limit('cpu.max'),'memory_limit':limit('memory.max'),'blas':buf.getvalue(),
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'threadpools':threadpools,'threadpool_note':None if threadpools is not None else 'threadpoolctl unavailable; actual backend threads unmeasured',
        'workers':1,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'rss_scope':'Linux single process lifetime maximum; includes diagnostics/report arrays',
        'accelerator_peak_bytes':None,'accelerator_reason':'CPU only',
        'storage_bytes':{'local_maps':ds.nbytes+hs.nbytes,'expanded_maps':maps.nbytes,'residual_operator':residual.nbytes,
            'reciprocal_hessian':q.nbytes,'all_four_dynamics':sum(a.nbytes for a in dynamics.values()),
            'activity_per_query':42*8,'minimal_deployed_dense_matrix_and_state':q.nbytes+42*8,
            'all_retained_model_arrays_and_one_state':sum(v.nbytes for v in (ds,hs,maps,residual,q,*dynamics.values()))+42*8,
            'learned_state':0,'optimizer':0,'replay':0,'sufficient_statistics':0,
            'fixture_numeric_payload':sum((len(f['indices'])+len(f['values']))*8 for f in fs)},
        'storage_note':'array/numeric payload excludes Python containers and report retention; peak RSS includes these; gradient temporary 336 bytes plus update temporary',
        'benchmark_scope':'sequential queries; allocations and direct SVD included; scoring and stationary diagnostics excluded; no device transfer or JIT',
        'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,
        'cpu_seconds_before_runtime_serialization':time.process_time()-cpu}
    runtime_path.write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    print(json.dumps({a:{k:v[-1] for k,v in science['summary'][a].items() if isinstance(v,list)} for a in ARMS},indent=2))

if __name__=='__main__':main()
