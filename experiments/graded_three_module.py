"""Predeclared layer-strength and feedback-gain comparison; no learning."""
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
from experiments import three_module_ambiguity as base

ROOT=base.ROOT
CONTRACT=ROOT/'docs/graded-three-module-v1-contract.md'
PREFIX=ROOT/'reports/graded-three-module-v1'
PHASES={}
COUNTS={'stationary_svd':0,'stationary_lstsq':0,'stability_eigen':0,'energy_eigen':0,
        'gradient_coordinates_checked':0,'configurations_scored':0}

@contextlib.contextmanager
def phase(name):
    w,c=time.perf_counter(),time.process_time()
    yield
    PHASES[name]={'wall_seconds':time.perf_counter()-w,'cpu_seconds':time.process_time()-c}

def grid():
    profiles={'inner_only':(0,0,1),'inward':(.05,.2,1),'uniform':(1,1,1),
              'outward':(1,.2,.05),'outer_only':(1,0,0),'middle_only':(0,1,0)}
    cases=[]
    def add(name,raw,gain=1.,feedback=1.,input_feedback=1.,normalized=True):
        weights=np.array(raw,dtype=float)
        if normalized:weights*=2/(weights@np.array([8,4,2]))
        weights*=gain
        cases.append({'name':name,'weights':weights.tolist(),'output_feedback':feedback,
                      'input_feedback':input_feedback,'layer_budget':float(weights@np.array([8,4,2]))})
    for profile,raw in profiles.items():
        for fb in (0.,.25,1.):add(f'{profile}_b{fb:g}',raw,feedback=fb)
    for gain in (.25,4.):
        for fb in (0.,.25,1.):add(f'inward_gain{gain:g}_b{fb:g}',profiles['inward'],gain,fb)
    for fb in (0.,.25,1.):add(f'inward_raw_b{fb:g}',profiles['inward'],feedback=fb,normalized=False)
    for profile,raw in profiles.items():add(f'{profile}_forward',raw,feedback=0.,input_feedback=0.)
    assert len(cases)==33
    return cases

def build_alignments(ds):
    u=ds[0];ps=[np.eye(4),np.eye(4)[[1,0,2,3]],np.eye(4)]
    os_=[np.eye(8),np.eye(8),np.eye(8)[::-1]*np.array([1,-1]*4)[:,None]]
    return [( (u@p.T@u.T+np.eye(8)-u@u.T)@o.T,p.T,np.eye(2)) for p,o in zip(ps,os_)]

def matrices(spec,local,alignments):
    q=local.T@local;a=q.copy();operators=[]
    for edge in (0,1):
        for layer,(start,width) in enumerate(((0,8),(8,4),(12,2))):
            w=spec['weights'][layer]
            e=np.zeros((width,42))
            e[:,14*edge+start:14*edge+start+width]=alignments[edge][layer]
            e[:,14*(edge+1)+start:14*(edge+1)+start+width]=-alignments[edge+1][layer]
            operators.append((w,e))
            term=w*(e.T@e);q+=term
            feedback=spec['input_feedback'] if edge==0 else spec['output_feedback']
            term[edge*14:(edge+1)*14]*=feedback
            a+=term
    if spec['input_feedback']==spec['output_feedback']==1:a=q.copy()
    return a,q,operators

def selected_stationary(f,a):
    idx=np.array(f['indices']);free=np.setdiff1d(np.arange(42),idx)
    b=a[np.ix_(free,free)];rhs=-a[np.ix_(free,idx)]@f['values']
    u,s,_=np.linalg.svd(b);COUNTS['stationary_svd']+=1
    rank=int(np.sum(s>1e-10));left=u[:,rank:]
    invariant_error=float(np.linalg.norm(left.T@rhs))
    augmented=np.vstack((b,left.T));target=np.concatenate((rhs,np.zeros(left.shape[1])))
    y,_,arank,_=np.linalg.lstsq(augmented,target,rcond=None);COUNTS['stationary_lstsq']+=1
    error=float(np.linalg.norm(augmented@y-target))
    assert invariant_error<1e-9 and error<1e-9 and arank==len(free)
    state=np.zeros(42);state[idx]=f['values'];state[free]=y
    return state,{'nullity':len(free)-rank,'solve_residual':error,'invariant_residual':invariant_error}

def benchmark(fn):
    fn();ws=[];cs=[]
    for _ in range(3):
        w,c=time.perf_counter(),time.process_time();fn()
        ws.append((time.perf_counter()-w)/16);cs.append((time.process_time()-c)/16)
    def stats(v):return {'median':float(np.median(v)),'min':min(v),'max':max(v)}
    return {'warmup_batches':1,'repeats':3,'batch_size':16,'wall_seconds_per_query':stats(ws),'cpu_seconds_per_query':stats(cs)}

def verify(spec,a,q,operators,local,alignments,maps,old,fs):
    # Full-layer transforms must preserve dimensions and agree on every code.
    for transforms in alignments:
        for t in transforms:assert np.allclose(t.T@t,np.eye(len(t)),atol=1e-14)
    for m in range(1,3):
        assert np.allclose(alignments[m][0]@maps[m],alignments[0][0]@maps[0])
    s=np.linspace(-.7,.9,42);g=q@s;worst=0.
    def energy(v):return .5*np.sum((local@v)**2)+sum(.5*w*np.sum((e@v)**2) for w,e in operators)
    for i in range(42):
        plus=s.copy();minus=s.copy();plus[i]+=1e-6;minus[i]-=1e-6
        worst=max(worst,abs((energy(plus)-energy(minus))/2e-6-g[i]))
    COUNTS['gradient_coordinates_checked']+=42
    assert worst<1e-8
    independent=local.T@(local@s)
    for i,(w,e) in enumerate(operators):
        grad=w*e.T@(e@s);edge=i//3
        grad[14*edge:14*(edge+1)]*=spec['input_feedback'] if edge==0 else spec['output_feedback']
        independent+=grad
    assert np.allclose(independent,a@s,atol=1e-13)
    assert np.array_equal(a[:14,28:],np.zeros((14,14)))
    assert np.array_equal(a[28:,:14],np.zeros((14,14)))
    largest=float(np.linalg.eigvalsh(q).max());COUNTS['energy_eigen']+=1
    assert .1*largest<2
    if spec['name']=='inner_only_b1':assert np.array_equal(a,old['reciprocal'])
    if spec['name']=='inner_only_b0':assert np.array_equal(a,old['no_output_to_middle'])
    if spec['name']=='inner_only_forward':assert np.array_equal(a,old['forward_only'])
    stability=[]
    for indices in ((0,14),(0,),(14,),(0,1,14)):
        free=np.setdiff1d(np.arange(42),indices);b=a[np.ix_(free,free)]
        ev=np.linalg.eigvals(np.eye(len(free))-.1*b);COUNTS['stability_eigen']+=1
        stability.append({'clamps':list(indices),'spectral_radius':float(np.max(abs(ev))),
                          'unstable':bool(np.max(abs(ev))>1+1e-8)})
    return {'gradient_max_error':worst,'largest_energy_hessian_eigenvalue':largest,'stability':stability}

def main():
    result=Path(str(PREFIX)+'-results.json');runtime_file=Path(str(PREFIX)+'-runtime.json')
    if result.exists() or runtime_file.exists():raise RuntimeError('Refusing overwrite')
    wall,cpu=time.perf_counter(),time.process_time()
    with phase('construction'):
        ds,hs,maps,residual,_,old=base.build();local=residual[:36].copy();align=build_alignments(ds)
        specs=grid();fs=base.fixtures();models=[matrices(s,local,align) for s in specs]
        frozen=[v.tobytes() for v in (ds,hs,maps,local,*[x for aa in align for x in aa],*[v for a,q,_ in models for v in (a,q)])]
        references=[];projections=[];direct_rows=[]
        for f in fs:
            out,z,rank,inc=base.direct(f,maps);references.append(out)
            if rank==1:
                cue=np.array([maps[i//14,i%14] for i in f['indices']])
                _,_,vt=np.linalg.svd(cue,full_matrices=True);base.COUNTS['svd_calls']+=1
                direction=maps[2]@vt[rank:].T
                projection=direction@base.least_squares(direction,np.eye(8))[0]
            else:projection=np.zeros((8,8))
            projections.append(projection)
            direct_rows.append({'output':out.tolist(),'code':z.tolist(),'rank':rank,'evidence_inconsistency':inc,
                                'compatible_set_dimension':2-rank if inc<1e-10 else None})
    with phase('verification'):
        checks=[verify(s,a,q,ops,local,align,maps,old,fs) for s,(a,q,ops) in zip(specs,models)]
        for s in specs[:18]:assert abs(s['layer_budget']-2)<1e-14
        for m in range(3):assert np.allclose(align[m][1]@hs[m],hs[0])
    science={'specifications':specs,'fixtures':fs,'direct':direct_rows,'configurations':{}}
    with phase('all_candidate_inference_scoring_and_stationary_diagnostics'):
        for spec,(a,q,ops),check in zip(specs,models,checks):
            episodes=[]
            for i,f in enumerate(fs):
                stationary,diagnostic=selected_stationary(f,a)
                rows=base.run(f,a,4096,True,q)
                for row in rows:
                    output=np.array(row['output']);delta=output-references[i]
                    distance=float(np.sqrt(np.mean(((np.eye(8)-projections[i])@delta)**2)))
                    row['compatible_set_rmse']=None if f['kind']=='contradictory' else distance
                    row['selected_stationary_output_rmse']=float(np.sqrt(np.mean((output-stationary[28:36])**2)))
                last=rows[-1]
                # Final local energy is reconstructed from an additional saved-state-free
                # stationary diagnostic only; finite-time row energy is the weighted total.
                episodes.append({'fixture_index':i,'trajectory':rows,'stationary':diagnostic,
                    'selected_stationary_output':stationary[28:36].tolist(),
                    'stationary_local_energy':float(.5*np.sum((local@stationary)**2)),
                    'stationary_weighted_energy':float(.5*stationary@q@stationary),
                    'stationary_target_rmse':None if f['kind']!='complete' else float(np.sqrt(np.mean((stationary[28:36]-references[i])**2)))})
            summaries={}
            for kind in ('complete','missing_context','missing_input','contradictory'):
                subset=[e for e,f in zip(episodes,fs) if f['kind']==kind];summary=[]
                for j,t in enumerate(base.HORIZONS):
                    obs=[e['trajectory'][j] for e in subset]
                    errors=[o['compatible_set_rmse'] for o in obs]
                    summary.append({'step':t,'compatible_set_rmse':None if errors[0] is None else float(np.sqrt(np.mean(np.square(errors)))),
                        'not_converged':sum(o['update_residual']>1e-5 for o in obs),
                        'max_update_residual':max(o['update_residual'] for o in obs),
                        'mean_weighted_reciprocal_energy':float(np.mean([o['reciprocal_energy'] for o in obs])),
                        'selected_stationary_output_rmse':float(np.sqrt(np.mean([o['selected_stationary_output_rmse']**2 for o in obs])))})
                summaries[kind]=summary
            interventions={}
            for variable in ('a','b'):
                other='b' if variable=='a' else 'a';errors=[]
                for i,f in enumerate(fs[:16]):
                    for j,g in enumerate(fs[:16]):
                        if i<j and f[other]==g[other] and f[variable]!=g[variable]:
                            delta=np.array(episodes[j]['trajectory'][-1]['output'])-episodes[i]['trajectory'][-1]['output']
                            errors.append(float(np.mean((delta-(references[j]-references[i]))**2)))
                interventions[variable]={'pairs':len(errors),'delta_rmse':float(np.sqrt(np.mean(errors)))}
            summaries['cue_interventions']=interventions
            summaries['first_quality_horizon']=next((r['step'] for r in summaries['complete'] if r['compatible_set_rmse']<=1e-4),None)
            summaries['first_convergence_horizon']=next((r['step'] for r in summaries['complete'] if r['not_converged']==0),None)
            summaries['stationary_complete_rmse']=float(np.sqrt(np.mean([e['stationary_target_rmse']**2 for e in episodes[:16]])))
            science['configurations'][spec['name']]={'checks':check,'summary':summaries,'episodes':episodes}
            COUNTS['configurations_scored']+=1
        assert frozen==[v.tobytes() for v in (ds,hs,maps,local,*[x for aa in align for x in aa],*[v for a,q,_ in models for v in (a,q)])]
    before_bench=base.COUNTS.copy()
    with phase('benchmarks_all_candidates'):
        complete=fs[:16]
        timings={'direct':benchmark(lambda:[base.direct(f,maps) for f in complete])}
        for spec,(a,q,ops) in zip(specs,models):
            timings[spec['name']]={str(t):benchmark(lambda:[base.run(f,a,t) for f in complete]) for t in (256,1024,4096)}
    with phase('scientific_serialization'):
        result.write_text(json.dumps(science,indent=2,sort_keys=True)+'\n')
    with phase('provenance'):
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):np.show_config()
        try:
            from threadpoolctl import threadpool_info
            pools=threadpool_info()
        except ImportError:pools=None
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(base.__file__),CONTRACT)}
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    def limit(name):
        p=Path('/sys/fs/cgroup')/name
        return p.read_text().strip() if p.exists() else None
    runtime={'phases':PHASES,'benchmarks':timings,'counts_total':base.COUNTS,'counts_before_benchmarks':before_bench,
        'additional_counts':COUNTS,'unique_fixtures':28,'scientific_episodes':33*28,'candidates_scored':33,
        'selection_policy':'all 33 predeclared candidates reported; no selected deployment or held-out confirmation',
        'fits':0,'optimizer_steps':0,'training_examples':0,'validation_examples':0,
        'comparison_cost_scope':'construction, verification, all candidate inference/scoring/diagnostics, benchmarks and serialization; all rejected/unsuccessful candidates included',
        'manual_design_seconds':None,'manual_design_reason':'not measured','hashes':hashes,'revision':revision,'dirty_status':dirty,
        'command':' '.join(sys.argv),'seed_policy':'deterministic; no RNG','python':sys.version,'numpy':np.__version__,
        'os':platform.platform(),'dtype':'float64','cpu':platform.processor() or Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
        'affinity_cpu_count':len(os.sched_getaffinity(0)),'cpu_limit':limit('cpu.max'),'memory_limit':limit('memory.max'),
        'blas':buf.getvalue(),'threadpools':pools,'threadpool_note':None if pools is not None else 'threadpoolctl unavailable; actual threads unmeasured',
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'workers':1,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_scope':'Linux single-process lifetime maximum, including retained reports',
        'accelerator_memory':None,'accelerator_reason':'CPU only',
        'storage_bytes':{'deployed_dense_operator_and_state':42*42*8+42*8,'local_maps':ds.nbytes+hs.nbytes,
            'expanded_maps':maps.nbytes,'all_alignment_maps':sum(v.nbytes for aa in align for v in aa),
            'all_candidate_dynamics_and_energy_matrices':sum(a.nbytes+q.nbytes for a,q,_ in models),
            'all_candidate_edge_operators':sum(e.nbytes for _,_,ops in models for _,e in ops),
            'activity_per_query':42*8,'gradient_per_query':42*8,'learned_state':0,'optimizer':0,'replay':0,
            'fixture_numeric_payload':sum((len(f['indices'])+len(f['values']))*8 for f in fs)},
        'storage_note':'float64 array payload; Python/report containers and SVD/update temporary workspace additional; process peak includes them',
        'benchmark_scope':'allocations/updates included; scoring excluded; direct includes cue SVD; no JIT or device transfer',
        'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,
        'cpu_seconds_before_runtime_serialization':time.process_time()-cpu}
    runtime_file.write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    for name,data in science['configurations'].items():
        s=data['summary'];print(name,s['complete'][-1]['compatible_set_rmse'],s['first_quality_horizon'],s['first_convergence_horizon'])

if __name__=='__main__':main()
