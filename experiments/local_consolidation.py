"""Role-independent module-local replay consolidation; evaluator labels stay private."""
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'docs/local-consolidation-v1-contract.md'
PREFIX=ROOT/'reports/local-consolidation-v1'
ARMS=('frozen','energy','stability','slow_stability','support','support_stability',
      'support_stability_no_inertia','repeated_support')
COUNTS={}
PHASES={}

def count(key,n=1):COUNTS[key]=COUNTS.get(key,0)+n

@contextlib.contextmanager
def phase(name):
    w,c=time.perf_counter(),time.process_time()
    yield
    PHASES[name]={'wall_seconds':time.perf_counter()-w,'cpu_seconds':time.process_time()-c}

def clone(p):return {k:v.copy() for k,v in p.items()}
def digest(p):return b''.join(p[k].tobytes() for k in ('W','V'))

def initialize(seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,1001]))
    return {'W':rng.normal(0,.25,(3,8,4)),'V':rng.normal(0,.25,(3,4,2))}

def build(p):
    count('hessian_constructions')
    l=np.zeros((36,42))
    for m in range(3):
        i,j=m*14,m*12
        l[j:j+8,i:i+8]=np.eye(8);l[j:j+8,i+8:i+12]=-p['W'][m]
        l[j+8:j+12,i+8:i+12]=np.eye(4);l[j+8:j+12,i+12:i+14]=-p['V'][m]
    q=l.T@l
    for m in range(3):
        for i in range(m*14+8,m*14+14):q[i,i]+=.05
    for a,b in ((0,1),(1,2)):
        for j in range(2):
            i,k=a*14+12+j,b*14+12+j
            q[i,i]+=.2;q[k,k]+=.2;q[i,k]-=.2;q[k,i]-=.2
    return q

def infer(data,p,mask=None,slow=False,check=False):
    count('inference_calls');count('inference_examples',len(data));count('activity_updates',32*len(data))
    if mask is None:mask=np.ones((3,8),bool)
    q=build(p);idx=np.array([m*14+j for m in range(3) for j in range(8) if mask[m,j]],dtype=int)
    free=np.setdiff1d(np.arange(42),idx);values=data[:,mask]
    mobility=np.ones(len(free))
    if slow:mobility[np.isin(free%14,(12,13))]=.1
    b=q[np.ix_(free,free)]*mobility[:,None]
    transition=np.linalg.solve(np.eye(len(free))+b,np.eye(len(free)));count('implicit_solves')
    force=(-(values@q[np.ix_(free,idx)].T)*mobility)@transition.T
    y=np.zeros((len(data),len(free)));state=np.zeros((len(data),42));state[:,idx]=values
    previous=None
    for _ in range(32):
        if check:
            state[:,free]=y;before=np.einsum('bi,ij,bj->b',state,q,state)/2
        previous=y;y=y@transition.T+force
        if check:
            state[:,free]=y;after=np.einsum('bi,ij,bj->b',state,q,state)/2
            assert np.all(after<=before+1e-10)
    state[:,free]=y
    last_delta=np.zeros_like(state);last_delta[:,free]=y-previous
    shaped=state.reshape(-1,3,14)
    inner_change=np.mean(np.sum(last_delta.reshape(-1,3,14)[:,:,12:]**2,axis=2),axis=0)
    residual=state@q.T;residual[:,idx]=0
    assert np.array_equal(state[:,idx],values)
    return shaped,inner_change,np.linalg.norm(residual,axis=1)

def proposal(p,states,module,scale):
    x,h,c=states[:,module,:8],states[:,module,8:12],states[:,module,12:]
    ex=x-h@p['W'][module].T;eh=h-c@p['V'][module].T
    delta_w=.2*(ex.T@h/len(states)-.001*p['W'][module])
    delta_v=.2*(eh.T@c/len(states)-.001*p['V'][module])
    candidate=clone(p)
    candidate['W'][module]=np.clip(p['W'][module]+scale*delta_w,-2,2)
    candidate['V'][module]=np.clip(p['V'][module]+scale*delta_v,-2,2)
    count('candidate_proposals')
    return candidate

def diverse_ids(data,ids,module,coordinate,improved):
    retained=[];seen=set();features=np.delete(data[:,module],coordinate,axis=1)
    for i in np.flatnonzero(improved):
        identity=int(ids[i])
        if identity in seen:continue
        if all(np.linalg.norm(features[i]-features[j])>.05 for j in retained):
            retained.append(i);seen.add(identity)
    return {int(ids[i]) for i in retained}

def consolidate(p,proposal_data,validation,validation_ids,arm,support):
    events=[];slow=arm=='slow_stability';supported=arm.startswith('support') or arm=='repeated_support'
    gated=arm not in ('energy','support','frozen')
    repeated=arm=='repeated_support'
    if repeated:
        proposal_data=np.repeat(proposal_data[:1],len(proposal_data),axis=0)
        validation=np.repeat(validation[:1],len(validation),axis=0)
        validation_ids=np.repeat(validation_ids[:1],len(validation_ids))
    for cycle in range(6):
        for module in range(3):
            states,change,_=infer(proposal_data,p,slow=slow);count('proposal_trace_exposures',len(proposal_data))
            gate=1/(1+1000*change[module]) if gated else 1.
            inertia=1/(1+len(support[module])/8) if supported and arm!='support_stability_no_inertia' else 1.
            event={'cycle':cycle,'module':module,'gate':float(gate),'inertia':float(inertia),
                   'support_before':len(support[module]),'accepted':False,'validation_gain':None,'distinct_support':0}
            if arm=='frozen':events.append(event);continue
            candidate=proposal(p,states,module,gate*inertia)
            if supported:
                coordinate=cycle%8;mask=np.ones((3,8),bool);mask[module,coordinate]=False
                incumbent,_,_=infer(validation,p,mask,slow)
                proposed,_,_=infer(validation,candidate,mask,slow)
                count('validation_trace_exposures',2*len(validation));count('candidate_local_checks')
                target=validation[:,module,coordinate]
                old=(incumbent[:,module,coordinate]-target)**2
                new=(proposed[:,module,coordinate]-target)**2
                gain=float(np.mean(old-new));improved=old-new>1e-5
                identities=diverse_ids(validation,validation_ids,module,coordinate,improved)
                accepted=gain>1e-5 and bool(np.all(new<=old+.01)) and len(identities)>=3
                event.update(validation_gain=gain,distinct_support=len(identities),accepted=accepted)
                if accepted:support[module].update(identities)
            else:accepted=True;event['accepted']=True
            if accepted:
                count('accepted_proposals');p=candidate
            else:count('rejected_proposals')
            event['support_after']=len(support[module]);events.append(event)
    return p,events

def fixture(seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,2002]))
    maps=np.stack([np.linalg.qr(rng.normal(size=(8,2)))[0] for _ in range(3)])
    def sample(label,n):
        amplitude=rng.uniform(.8,1.2,(n,1,1));noise=rng.normal(0,.1,(n,3,8))
        for m in range(3):noise[:,m]-=(noise[:,m]@maps[m])@maps[m].T
        return amplitude*maps[:,:,label][None,:,:]+noise
    waking=[sample(label,16) for label in (0,1)]
    testing=np.concatenate([sample(label,12) for label in (0,1)])
    references=np.concatenate([sample(label,8) for label in (0,1)])
    return waking,testing,references

def evaluate(p,test,reference,slow=False):
    frozen=digest(p);errors=[];residuals=[]
    for module in range(3):
        mask=np.ones((3,8),bool);mask[module]=False
        state,_,res=infer(test,p,mask,slow);count('evaluation_reconstruction_examples',len(test))
        errors.append(np.mean((state[:,module,:8]-test[:,module])**2,axis=1));residuals.extend(res.tolist())
    per_example=np.mean(errors,axis=0)
    state,_,res=infer(test,p,slow=slow);code=state[:,:,12:].reshape(len(test),6)
    ref,_,_=infer(reference,p,slow=slow);refcode=ref[:,:,12:].reshape(len(reference),6)
    count('evaluation_probe_examples',len(test)+len(reference))
    centroids=np.stack((refcode[:8].mean(axis=0),refcode[8:].mean(axis=0)))
    prediction=np.argmin(np.sum((code[:,None,:]-centroids[None,:,:])**2,axis=2),axis=1)
    labels=np.repeat([0,1],12) # evaluator only, never passed to consolidation
    centers=np.stack((code[:12].mean(axis=0),code[12:].mean(axis=0)))
    within=float(np.mean(np.sum((code-centers[labels])**2,axis=1)))
    between=float(np.sum((centers[0]-centers[1])**2))
    assert digest(p)==frozen
    return {'rmse_by_structure':[float(np.sqrt(np.mean(per_example[:12]))),float(np.sqrt(np.mean(per_example[12:])))],
            'rmse':float(np.sqrt(np.mean(per_example))),'within_structure_spread':within,'between_centroid_distance_squared':between,
            'within_to_between_ratio':None if between<1e-12 else within/between,
            'centroid_accuracy':float(np.mean(prediction==labels)),'mean_code_squared_norm':float(np.mean(np.sum(code**2,axis=1))),
            'mean_reconstruction_residual':float(np.mean(residuals)),'max_reconstruction_residual':max(residuals),
            'full_observation_residual':float(np.mean(res))}

def direct_predictor(train,test):
    errors=[];parameters=0
    for m in range(3):
        other=[i for i in range(3) if i!=m]
        design=np.column_stack((train[:,other].reshape(len(train),16),np.ones(len(train))))
        targets=train[:,m];test_design=np.column_stack((test[:,other].reshape(len(test),16),np.ones(len(test))))
        weights=np.linalg.solve(design.T@design+.1*np.eye(17),design.T@targets)
        count('direct_fits');count('direct_fit_examples',len(train));count('direct_inference_examples',len(test))
        parameters+=weights.nbytes;errors.append(np.mean((test_design@weights-test[:,m])**2,axis=1))
    e=np.mean(errors,axis=0)
    return {'rmse':float(np.sqrt(np.mean(e))),'rmse_by_structure':[float(np.sqrt(np.mean(e[:12]))),float(np.sqrt(np.mean(e[12:])))],
            'parameter_bytes':parameters}

def verify():
    p=initialize(99);data=np.linspace(-.7,.8,48).reshape(2,3,8)
    states,_,_=infer(data,p,check=True);s=states[0].reshape(42);q=build(p)
    analytic=q@s;worst=0.
    for i in range(42):
        plus=s.copy();minus=s.copy();plus[i]+=1e-6;minus[i]-=1e-6
        worst=max(worst,abs((.5*plus@q@plus-.5*minus@q@minus)/2e-6-analytic[i]))
    assert worst<1e-7
    # Local weight derivative at fixed activities, including parameter regularization.
    candidate=proposal(p,states,1,1.)
    def energy(parameters):
        return float(np.mean(np.einsum('bi,ij,bj->b',states.reshape(2,42),build(parameters),states.reshape(2,42)))/2
                     +.0005*sum(np.sum(v*v) for v in parameters.values()))
    parameter_error=0.
    for name in ('W','V'):
        expected=(p[name][1]-candidate[name][1])/.2
        for idx in np.ndindex(expected.shape):
            plus=clone(p);minus=clone(p);full=(1,*idx)
            plus[name][full]+=1e-6;minus[name][full]-=1e-6
            numeric=(energy(plus)-energy(minus))/2e-6
            parameter_error=max(parameter_error,abs(numeric-expected[idx]))
    assert parameter_error<1e-7
    # A hidden coordinate cannot influence inference via its stored value.
    mask=np.ones((3,8),bool);mask[1,3]=False
    altered=data.copy();altered[:,1,3]+=1000
    a,_,_=infer(data,p,mask);b,_,_=infer(altered,p,mask)
    assert np.array_equal(a,b)
    # Reversing the chain swaps module roles without changing the algorithm.
    rev={k:v[::-1].copy() for k,v in p.items()}
    c,_,_=infer(data[:,::-1],rev)
    assert np.allclose(c[:,::-1],states,atol=1e-12)
    ids=diverse_ids(np.repeat(data[:1],8,axis=0),np.zeros(8,int),0,0,np.ones(8,bool))
    assert len(ids)==1
    support=set(ids);support.update(ids);assert len(support)==1
    return {'activity_gradient_max_error':worst,'parameter_gradient_max_error':parameter_error,
            'hidden_value_leakage_check':True,'chain_role_reversal_check':True,'unique_support_check':True,
            'zero_code_control':{'within_spread':0.,'between_distance':0.,'centroid_accuracy_balanced_ties':.5}}

def main():
    result=Path(str(PREFIX)+'-results.json');runtime=Path(str(PREFIX)+'-runtime.json')
    if result.exists() or runtime.exists():raise RuntimeError('Refusing overwrite')
    wall,cpu=time.perf_counter(),time.process_time()
    with phase('verification'):checks=verify()
    with phase('fixture_construction'):
        datasets={seed:fixture(seed) for seed in (0,1,2)}
    science={'checks':checks,'seeds':{},'scope':'local weight consolidation; coupling fixed; no hidden labels in learning'}
    for seed in (0,1,2):
        waking,test,reference=datasets[seed];initial=initialize(seed)
        seed_result={'arms':{},'direct':[]}
        for arm in ARMS:
            p=clone(initial);support=[set() for _ in range(3)];evaluations=[];events=[];all_waking=[]
            with phase(f'seed{seed}_{arm}_initial_evaluation'):
                evaluations.append(evaluate(p,test,reference,arm=='slow_stability'))
            for stage in range(2):
                with phase(f'seed{seed}_{arm}_stage{stage}_waking'):
                    frozen=digest(p);infer(waking[stage],p,slow=arm=='slow_stability')
                    assert digest(p)==frozen;all_waking.append(waking[stage]);pool=np.concatenate(all_waking)
                    ids=np.arange(len(pool));proposal_data=pool[::2];validation=pool[1::2];validation_ids=ids[1::2]
                    count('waking_episode_exposures',len(waking[stage]))
                with phase(f'seed{seed}_{arm}_stage{stage}_consolidation'):
                    p,stage_events=consolidate(p,proposal_data,validation,validation_ids,arm,support);events.append(stage_events)
                with phase(f'seed{seed}_{arm}_stage{stage}_evaluation'):
                    evaluations.append(evaluate(p,test,reference,arm=='slow_stability'))
            seed_result['arms'][arm]={'evaluations':evaluations,'events':events,'support_counts':[len(x) for x in support],
                'accepted_by_stage':[sum(e['accepted'] for e in stage) for stage in events],
                'parameter_drift':float(np.sqrt(sum(np.sum((p[k]-initial[k])**2) for k in p))),
                'final_parameters':{k:v.tolist() for k,v in p.items()}}
            if arm in ('frozen','repeated_support'):assert digest(p)==digest(initial)
        with phase(f'seed{seed}_direct_comparator'):
            for stage in (0,1):seed_result['direct'].append(direct_predictor(np.concatenate(waking[:stage+1]),test))
        science['seeds'][str(seed)]=seed_result
    with phase('summary'):
        summaries={}
        for arm in ARMS:
            rows=[science['seeds'][str(seed)]['arms'][arm] for seed in (0,1,2)]
            final=[r['evaluations'][2] for r in rows]
            retention=[r['evaluations'][2]['rmse_by_structure'][0]/r['evaluations'][1]['rmse_by_structure'][0] for r in rows]
            summaries[arm]={'mean_final_rmse':float(np.mean([v['rmse'] for v in final])),
                'mean_final_centroid_accuracy':float(np.mean([v['centroid_accuracy'] for v in final])),
                'mean_final_within_between_ratio':float(np.mean([v['within_to_between_ratio'] for v in final])),
                'mean_retention_rmse_ratio':float(np.mean(retention)),
                'retention_pass_seeds':sum(v<=1.1 for v in retention),
                'accepted_proposals':sum(sum(r['accepted_by_stage']) for r in rows),
                'mean_gate':float(np.mean([e['gate'] for r in rows for stage in r['events'] for e in stage])),
                'mean_final_support':float(np.mean([n for r in rows for n in r['support_counts']]))}
        science['summary']=summaries
    with phase('serialization'):
        result.write_text(json.dumps(science,indent=2,sort_keys=True)+'\n')
    with phase('provenance'):
        config=io.StringIO()
        with contextlib.redirect_stdout(config):np.show_config()
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),CONTRACT)}
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    def limit(name):
        p=Path('/sys/fs/cgroup')/name
        return p.read_text().strip() if p.exists() else None
    info={'phases':PHASES,'counts':COUNTS,'hashes':hashes,'revision':revision,'dirty_status':dirty,
          'command':' '.join(sys.argv),'seeds':[0,1,2],'seed_policy':'SeedSequence [seed,1001] initialization; [seed,2002] fixtures',
          'python':sys.version,'numpy':np.__version__,'os':platform.platform(),'blas':config.getvalue(),
          'thread_environment':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
          'backend_threads':None,'backend_threads_reason':'environment fixed before NumPy import; active backend count not measured',
          'workers':1,'cpu':platform.processor() or Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
          'affinity_cpu_count':len(os.sched_getaffinity(0)),'cpu_limit':limit('cpu.max'),'memory_limit':limit('memory.max'),
          'dtype':'float64','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'rss_scope':'Linux single-process lifetime peak; all reporting arrays included',
          'accelerator_memory':None,'accelerator_reason':'CPU only','manual_design_seconds':None,'manual_design_reason':'not measured',
          'independent_waking_examples':3*32,'independent_test_examples':3*24,'evaluator_reference_examples':3*16,
          'hyperparameter_candidates':1,'arms':8,'tuning_seconds':0,'selection_note':'fixed arms, no selected candidate; all rejected local proposals counted as learning costs',
          'storage_bytes_per_learner':{'weights':120*8,'replay':32*3*8*8,'episode_ids':32*8,
             'support_ids_max':3*16*8,'total_persistent_numeric_max':120*8+32*3*8*8+32*8+3*16*8,
             'candidate_weights':120*8,'batch_state_max':32*42*8,'dense_hessian':42*42*8,
             'implicit_matrix_upper_bound':42*42*8,'optimizer_history':0},
          'storage_note':'numeric payload excludes Python containers, interpreter and temporary solve workspace; process peak includes them; labeled evaluation data not learner memory',
          'timing_scope':'whole-run/phase mechanism timings, not warmed latency benchmark; includes allocations and all probes/validation',
          'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,
          'cpu_seconds_before_runtime_serialization':time.process_time()-cpu}
    runtime.write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
    print(json.dumps(science['summary'],indent=2))

if __name__=='__main__':main()
