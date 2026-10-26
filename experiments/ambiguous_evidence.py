"""Frozen distributed evidence: explicit joint-input and directional controls."""
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
CONTRACT = ROOT / 'docs/ambiguous-evidence-v1-contract.md'
PREFIX = ROOT / 'reports/ambiguous-evidence-v1'
D = np.array([[[1,0],[0,1],[2**-.5,2**-.5],[2**-.5,-2**-.5]],
              [[0,1],[-1,0],[2**-.5,-2**-.5],[2**-.5,2**-.5]]])
HORIZONS = (0,1,4,16,64,256,1024)
ARMS = ('isolated','a_to_b','b_to_a','reciprocal')
COUNTS = {'inference_calls':0,'updates':0,'gradient_calls':0}
PHASES = {}

@contextlib.contextmanager
def phase(name):
    w,c = time.perf_counter(),time.process_time()
    yield
    PHASES[name] = {'wall_seconds':time.perf_counter()-w,'cpu_seconds':time.process_time()-c}

def energy(x,c):
    return float(.5*np.sum((x-np.einsum('mnd,md->mn',D,c))**2)+.5*np.sum((c[0]-c[1])**2))

def gradient(x,c,arm):
    COUNTS['gradient_calls'] += 1
    gx = x-np.einsum('mnd,md->mn',D,c)
    gc = -np.einsum('mnd,mn->md',D,gx)
    delta = c[0]-c[1]
    if arm in ('reciprocal','b_to_a'): gc[0] += delta
    if arm in ('reciprocal','a_to_b'): gc[1] -= delta
    return gx,gc

def direct(cues):
    COUNTS['inference_calls'] += 1
    return np.einsum('mnd,d->mn',D,cues)

def run(cues,arm,budget,record=False):
    COUNTS['inference_calls'] += 1
    x,c = np.zeros((2,4)),np.zeros((2,2))
    x[:,0] = cues
    rows=[]
    for t in range(budget+1):
        if record and t in HORIZONS:
            gx,gc=gradient(x,c,arm);gx[:,0]=0
            rows.append({'step':t,'outer':x.tolist(),'inner':c.tolist(),
                         'residual':float(np.sqrt(np.sum(gx**2)+np.sum(gc**2)))})
        if t == budget: break
        gx,gc=gradient(x,c,arm);gx[:,0]=0
        before=energy(x,c) if record and arm=='reciprocal' else None
        x-=.1*gx;c-=.1*gc
        COUNTS['updates'] += 1
        if record:
            assert np.array_equal(x[:,0],cues)
            if before is not None: assert energy(x,c)<=before+1e-12
    return rows if record else x

def benchmark(fn):
    fn() # warmup included in execution counts
    wall=[];cpu=[]
    for _ in range(5):
        w,c=time.perf_counter(),time.process_time();fn()
        wall.append(time.perf_counter()-w);cpu.append(time.process_time()-c)
    return {'repeat_count':5,'batch_size':16,'warmup_batches':1,
            'wall_seconds_per_query':{'median':float(np.median(wall)/16),'min':min(wall)/16,'max':max(wall)/16},
            'cpu_seconds_per_query':{'median':float(np.median(cpu)/16),'min':min(cpu)/16,'max':max(cpu)/16}}

def main():
    paths=[Path(str(PREFIX)+suffix) for suffix in ('-results.json','-runtime.json')]
    if any(p.exists() for p in paths): raise RuntimeError('Refusing overwrite')
    start=time.perf_counter();start_cpu=time.process_time()
    with phase('construction'):
        fixtures=np.array([(a,b) for a in (-1.,-.5,.5,1.) for b in (-1.,-.5,.5,1.)])
        frozen=D.tobytes()
    with phase('verification'):
        x=np.linspace(-.5,.7,8).reshape(2,4);c=np.linspace(-.3,.6,4).reshape(2,2)
        gx,gc=gradient(x,c,'reciprocal');worst=0.
        for arr,g in ((x,gx),(c,gc)):
            for idx in np.ndindex(arr.shape):
                old=arr[idx];arr[idx]=old+1e-6;plus=energy(x,c)
                arr[idx]=old-1e-6;minus=energy(x,c);arr[idx]=old
                worst=max(worst,abs((plus-minus)/2e-6-g[idx]))
        assert worst<1e-8
        for cues in fixtures:
            expected=np.stack([D[m]@cues for m in range(2)])
            assert np.allclose(direct(cues),expected,atol=1e-14)
        for m in range(2):
            first=run(np.array([.5,.5]),'isolated',16)
            altered=np.array([.5,.5]);altered[1-m]=-1
            second=run(altered,'isolated',16)
            assert np.array_equal(first[m],second[m])
        for j in range(2):
            delta=D[:,:,j]
            assert np.all(np.linalg.norm(delta[:,1:],axis=1)>0)
    science={'scope':'supplied linear 4-to-2 modules; no learning or STM','gradient_max_error':worst,'arms':{}}
    with phase('scientific_inference_and_scoring'):
        targets=np.array([direct(cues) for cues in fixtures])
        science['direct']={'missing_rmse':0.,'outputs':targets.tolist()}
        for arm in ARMS:
            trajectories=[run(cues,arm,1024,True) for cues in fixtures]
            summary=[]
            for i,t in enumerate(HORIZONS):
                outputs=np.array([r[i]['outer'] for r in trajectories])
                residuals=[r[i]['residual'] for r in trajectories]
                error=outputs[:,:,1:]-targets[:,:,1:]
                summary.append({'step':t,'missing_rmse':float(np.sqrt(np.mean(error**2))),
                                'per_module_rmse':np.sqrt(np.mean(error**2,axis=(0,2))).tolist(),
                                'not_converged':sum(r>1e-5 for r in residuals),'max_residual':max(residuals)})
            science['arms'][arm]={'summary':summary,'trajectories':trajectories}
        science['cues']=fixtures.tolist()
        assert D.tobytes()==frozen
    scientific_counts=COUNTS.copy()
    with phase('benchmarks'):
        timings={'direct':benchmark(lambda:[direct(q) for q in fixtures])}
        for arm in ARMS:
            timings[arm]={str(t):benchmark(lambda:[run(q,arm,t) for q in fixtures]) for t in HORIZONS[1:]}
    with phase('scientific_serialization'):
        paths[0].write_text(json.dumps(science,indent=2,sort_keys=True)+'\n')
    with phase('provenance'):
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):np.show_config()
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),CONTRACT)}
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    runtime={'phases':PHASES,'benchmarks':timings,'counts_total':COUNTS,'counts_before_benchmark':scientific_counts,
             'fits':0,'candidates_scored':0,'optimizer_steps':0,'selection_tuning_seconds':0,
             'independent_fixture_pairs':16,'validation_pairs':0,'manual_design_cost':None,
             'manual_design_cost_reason':'not measured','hashes':hashes,'revision':revision,'dirty_status':dirty,
             'command':' '.join(sys.argv),'seed_policy':'deterministic exhaustive fixture; no RNG',
             'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
             'cpu':platform.processor() or Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
             'cpu_count':os.cpu_count(),'affinity_cpu_count':len(os.sched_getaffinity(0)),
             'hardware_limits':{'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
                                'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None},
             'blas':buf.getvalue(),'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
             'thread_note':'environment settings; actual backend thread count not measured',
             'workers':1,'dtype':'float64','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'rss_scope':'single process lifetime maximum; Linux KiB','accelerator_memory':None,
             'accelerator_reason':'CPU only','storage_bytes':{'maps':D.nbytes,'activities_per_case':96,'fixtures':fixtures.nbytes,
             'model_and_activity_total':D.nbytes+96,'optimizer':0,'replay':0,'learned_statistics':0},
             'storage_note':'array payload only; Python overhead and retained reporting arrays excluded; peak RSS includes them',
             'timing_scope':'allocations and updates included; scoring excluded from benchmark; no compilation or device transfer',
             'total_wall_seconds_before_runtime_serialization':time.perf_counter()-start,
             'total_cpu_seconds_before_runtime_serialization':time.process_time()-start_cpu}
    paths[1].write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    print(json.dumps({a:science['arms'][a]['summary'][-1] for a in ARMS},indent=2))

if __name__=='__main__':main()
