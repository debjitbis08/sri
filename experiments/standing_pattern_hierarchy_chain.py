"""Three-module stored-pattern hierarchy; see the prospective v3 contract."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/'docs/standing-pattern-hierarchy-v3-contract.md'
RESULT = ROOT/'reports/standing-pattern-hierarchy-v3-results.json'
RUNTIME = ROOT/'reports/standing-pattern-hierarchy-v3-runtime.json'
A = np.array([0,2,4,6,8,10,12]); B = A+1; C = np.arange(8,15)
HORIZONS = (0,1,4,16,64,256,1024,2048)
GRADE = np.array([.05]*8+[.2]*4+[1.]*2+[4.])
COUNTS = {}; PHASES = {}

def count(k, n=1): COUNTS[k] = COUNTS.get(k, 0)+n

@contextlib.contextmanager
def phase(name):
    wall, cpu = time.perf_counter(), time.process_time()
    yield
    PHASES[name] = {'wall_seconds':time.perf_counter()-wall,
                    'cpu_seconds':time.process_time()-cpu}

def features(c): return np.stack((np.ones_like(c), c, c*c), axis=-1)

def expand(core, weights):
    x = np.zeros((3,15)); x[:,14] = core
    for factors in (np.array([6]), np.arange(4,6), np.arange(4)):
        c = x[:,C[factors]]
        d = np.sum(weights[:,factors]*features(c), axis=-1)
        x[:,A[factors]] = c+d; x[:,B[factors]] = c-d
    return x

def energy(x, w, coupling=GRADE, gradient=False):
    count('energy_calls')
    c = x[:,C]; phi = features(c)
    e = c-(x[:,A]+x[:,B])/2
    r = (x[:,A]-x[:,B])/2-np.sum(w*phi, axis=-1)
    strengths = np.zeros(15)+coupling if np.ndim(coupling) else np.array([0.]*14+[coupling])
    diff = x[:-1]-x[1:]
    value = .5*np.sum(e*e+r*r)+.5*np.sum(strengths*diff*diff)+.0025*np.sum(x[:,14]**2)
    if not gradient: return float(value)
    count('gradient_calls')
    gx = np.zeros_like(x)
    for m in range(3):
        np.add.at(gx[m], A, (-e[m]+r[m])/2)
        np.add.at(gx[m], B, (-e[m]-r[m])/2)
        np.add.at(gx[m], C, e[m]-r[m]*(w[m,:,1]+2*w[m,:,2]*c[m]))
    gx[:,14] += .005*x[:,14]
    gx[:-1] += strengths*diff; gx[1:] -= strengths*diff
    return float(value), gx, -r[:,:,None]*phi

def update(x, w, observed, coupling, plastic):
    before, gx, gw = energy(x,w,coupling,True)
    if observed is not None: gx[0,:8] = 0
    for j in range(21):
        scale = 2.**(-j)
        xx = x-.1*scale*gx
        ww = w-.0001*scale*gw if plastic else w
        after = energy(xx,ww,coupling)
        count('line_search_trials')
        if np.isfinite(after) and after <= before+1e-12:
            if observed is not None: assert np.array_equal(xx[0,:8], observed)
            count('updates'); count('backtrack_halvings',j)
            return xx,ww
    raise AssertionError('line search exhausted')

def query(w, observed, coupling=GRADE, plastic=False, steps=2048, record=False):
    count('inference_calls')
    x = expand(0.,w); x[0,:8] = observed
    ww = w.copy(); rows = []
    for t in range(steps+1):
        if record and t in HORIZONS:
            val,gx,_ = energy(x,ww,coupling,True); gx[0,:8] = 0
            rows.append({'step':t, 'outer_output':x[-1,:8].tolist(),
                         'energy':val, 'free_gradient_norm':float(np.linalg.norm(gx)),
                         'weight_drift':float(np.linalg.norm(ww-w))})
            count('outer_readouts')
        if t<steps: x,ww = update(x,ww,observed,coupling,plastic)
    return x[-1,:8], rows

def direct(w, observed):
    count('direct_calls')
    return expand(float(observed.mean()),w)[-1,:8]

def fit(samples, degree):
    w = np.zeros((3,7,3))
    for m in (0,2):
        # Derive all interior coordinates from the observed outer signals.
        x = np.zeros((len(samples),15)); x[:,:8] = samples[:,m,:8]
        for k in range(7): x[:,C[k]] = (x[:,A[k]]+x[:,B[k]])/2
        for k in range(7):
            phi = features(x[:,C[k]])[:,:degree+1]
            target = (x[:,A[k]]-x[:,B[k]])/2
            w[m,k,:degree+1] = np.linalg.lstsq(phi,target,rcond=None)[0]
            count('least_squares_fits'); count('factor_training_rows',len(samples))
    return w

def main():
    if RESULT.exists() or RUNTIME.exists(): raise RuntimeError('Refusing overwrite')
    wall,cpu = time.perf_counter(),time.process_time()
    with phase('construction_and_training'):
        truth = np.zeros((3,7,3))
        for m in (0,2):
            endpoint = m//2
            for l,(offset,n) in enumerate(((0,4),(4,2),(6,1))):
                for j in range(n):
                    truth[m,offset+j] = [.08*(j+1)/n+.03*endpoint,
                                        .12*(-1)**(j+l+endpoint),.35*(-1)**(j+endpoint)]
        train_s = np.linspace(-.8,.8,9); test_s = (train_s[:-1]+train_s[1:])/2
        train = np.array([expand(s,truth) for s in train_s])
        test = np.array([expand(s,truth) for s in test_s])
        quadratic, affine = fit(train,2), fit(train,1)
    with phase('correctness_checks'):
        x = np.linspace(-.7,.8,45).reshape(3,15)
        w = quadratic.copy(); _,gx,gw = energy(x,w,gradient=True)
        worst = 0.
        for arr,g in ((x,gx),(w,gw)):
            for idx in np.ndindex(arr.shape):
                old = arr[idx]; arr[idx] = old+1e-6; plus = energy(x,w)
                arr[idx] = old-1e-6; minus = energy(x,w); arr[idx] = old
                worst = max(worst,abs((plus-minus)/2e-6-g[idx]))
        assert worst<1e-7
        assert np.max(abs(quadratic-truth))<1e-12
        neutral = expand(0.,quadratic)
        _,ng,_ = energy(neutral,quadratic,coupling=0.,gradient=True)
        assert np.linalg.norm(ng)<1e-12
        assert np.max(abs(test[:,:,C]-(test[:,:,A]+test[:,:,B])/2))<1e-12
        old_results = json.loads((ROOT/'reports/standing-pattern-hierarchy-v2-results.json').read_text())
        for example, previous in zip(test, old_results['arms']['quadratic_frozen']['queries']):
            assert np.array_equal(example[0,:8],previous['input_outer'])
            assert np.array_equal(example[-1,:8],previous['target_outer'])
        _,before,_ = energy(x,w,gradient=True)
        changed = x.copy(); changed[-1,0] += 1e-5
        _,after,_ = energy(changed,w,gradient=True)
        assert np.array_equal(before[0],after[0])
        assert abs((after[1,0]-before[1,0])/1e-5+.05)<1e-9
        checks = {'endpoint_fixtures_match_v2':True,'no_direct_endpoint_coupling':True,
                  'gradient_max_error':worst,'coefficient_fit_max_error':float(np.max(abs(quadratic-truth))),
                  'no_evidence_gradient_norm':float(np.linalg.norm(ng)),
                  'no_evidence_output':neutral[-1,:8].tolist(),
                  'haar_pair_gain_constant':1.,'haar_pair_gain_antisymmetric':0.}
    with phase('no_evidence_joint_default'):
        joint_default = neutral.copy()
        for _ in range(2048):
            joint_default,_ = update(joint_default,quadratic,None,GRADE,False)
        _,jg,_ = energy(joint_default,quadratic,gradient=True)
        checks['graded_no_evidence_output'] = joint_default[-1,:8].tolist()
        checks['graded_no_evidence_gradient_norm'] = float(np.linalg.norm(jg))
        checks['graded_no_evidence_cores'] = joint_default[:,14].tolist()
    arms = [('quadratic_frozen',quadratic,GRADE,False),
            ('quadratic_plastic',quadratic,GRADE,True),
            ('affine_frozen',affine,GRADE,False),
            ('zero_detail',np.zeros_like(quadratic),GRADE,False),
            ('uncoupled',quadratic,0.,False),
            ('innermost_only',quadratic,4.,False)]
    results = {}
    for name,w,coupling,plastic in arms:
        with phase(name):
            trials = []
            for s,example in zip(test_s,test):
                _,rows = query(w,example[0,:8],coupling,plastic,record=True)
                for row in rows:
                    row['mse'] = float(np.mean((np.array(row['outer_output'])-example[-1,:8])**2))
                trials.append({'latent_fixture_value':float(s),'input_outer':example[0,:8].tolist(),
                               'target_outer':example[-1,:8].tolist(),'readouts':rows})
            curve = []
            for k,t in enumerate(HORIZONS):
                rr = [q['readouts'][k] for q in trials]
                curve.append({'step':t,'rmse':float(np.sqrt(np.mean([r['mse'] for r in rr]))),
                              'unsettled_queries':sum(r['free_gradient_norm']>1e-5 for r in rr),
                              'mean_weight_drift':float(np.mean([r['weight_drift'] for r in rr]))})
            results[name] = {'curve':curve,'queries':trials}
    with phase('direct_comparators'):
        comparator = {}
        for name,w in [('quadratic',quadratic),('affine',affine),('zero_detail',np.zeros_like(quadratic))]:
            predictions = np.array([direct(w,e[0,:8]) for e in test])
            comparator[name] = {'rmse':float(np.sqrt(np.mean((predictions-test[:,-1,:8])**2)))}
    with phase('timing_benchmarks'):
        benchmarks = {}
        for name,steps in [('direct',0),('quadratic_64',64),('quadratic_2048',2048)]:
            times = []; cpus = []
            for repeat in range(4):
                start,pc = time.perf_counter(),time.process_time()
                for e in test:
                    if steps: query(quadratic,e[0,:8],steps=steps)
                    else: direct(quadratic,e[0,:8])
                elapsed,used = time.perf_counter()-start,time.process_time()-pc
                if repeat: times.append(elapsed); cpus.append(used)
            benchmarks[name] = {'batch_size':8,'warmup_batches':1,'measured_batches':3,
                                'wall_seconds':times,'cpu_seconds':cpus,
                                'median_seconds_per_query':statistics.median(times)/8,
                                'min_batch_seconds':min(times),'max_batch_seconds':max(times)}
    scientific = {'contract':str(CONTRACT.relative_to(ROOT)), 'checks':checks,
                  'topology':[[0,1],[1,2]], 'middle_initialization':'zero detail, no intermediate training targets',
                  'coupling_strengths_outer_to_inner':[.05,.2,1.,4.], 'training_pairs':9,'heldout_pairs':8,'grouping':'supplied nested adjacent pairs',
                  'quadratic_weights':quadratic.tolist(),'affine_weights':affine.tolist(),
                  'zero_output_rmse':float(np.sqrt(np.mean(test[:,-1,:8]**2))),
                  'arms':results,'direct_comparators':comparator,'counts':COUNTS.copy()}
    with phase('scientific_serialization'):
        RESULT.write_text(json.dumps(scientific,indent=2,sort_keys=True)+'\n')
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream): np.show_config()
    def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    runtime = {'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'contract_sha256':hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
               'result_sha256':hashlib.sha256(RESULT.read_bytes()).hexdigest(),
               'revision':git('rev-parse','HEAD'),'git_status':git('status','--short'),
               'command':'.venv/bin/python -m experiments.standing_pattern_hierarchy_chain',
               'seed_policy':'Deterministic grids; no random sampling',
               'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
               'cpu':platform.processor() or platform.machine(),'blas':stream.getvalue(),
               'threads':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
               'workers':0,'processes':1,'accelerator':None,'dtype':'float64',
               'phases':PHASES,'benchmarks':benchmarks,'counts':COUNTS,
               'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,
               'cpu_seconds_before_runtime_serialization':time.process_time()-cpu,
               'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'rss_convention':'Linux KiB; single process',
               'deployment_bytes':{'parameters':quadratic.nbytes,'activities':45*8,
                                   'total_persistent_model_and_activity':quadratic.nbytes+45*8,
                                   'optimizer':0,'replay':0,'sufficient_statistics':0,
                                   'fixed_coupling_bytes':GRADE.nbytes,
                                   'fixed_index_arrays':A.nbytes+B.nbytes+C.nbytes,
                                   'temporary_array_order':'O(Np); candidate weights and state, gradients, features',
                                   'peak_allocator_bytes':None},
               'selection':{'searched_candidates':0,'rejected_candidates':0,'fits':28,
                            'tuning_seconds':0,'reason':'Predeclared arms; no tuning or model selection'},
               'limitations':{'manual_development_seconds':None,'hardware_limits':None,
                              'reason':'Manual time, allocator peak and scheduler limits not instrumented'},
               'timing_scope':'Includes construction, checks, fits, scientific and benchmark queries, scoring and scientific serialization; excludes runtime JSON serialization.'}
    RUNTIME.write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'checks':checks,'curves':{k:v['curve'] for k,v in results.items()},
                      'direct':comparator,'seconds':runtime['wall_seconds_before_runtime_serialization']},indent=2))

if __name__ == '__main__': main()
