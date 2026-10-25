"""Continuous local operator, inner diagnostics, and outer-only task readouts.

Run: .venv/bin/python -m experiments.outer_anytime
Contract: docs/outer-anytime-v2-contract.md
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import hashlib
import itertools
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
CONTRACT = ROOT/"docs/outer-anytime-v2-contract.md"
RESULT = ROOT/"reports/outer-anytime-v2-results.json"
RUNTIME = ROOT/"reports/outer-anytime-v2-runtime.json"
SEED = 2026102512
DT, ETA = .1, .005
HORIZONS = (0,1,2,4,8,16,32,64)
PROTOTYPES = np.array([[0.,0.],[.5,1.],[.5,0.],[1.,0.]])
XOR = np.array([0.,1.,1.,0.])
DEFAULT = np.array([0.,0.,1.,1.])
TRAIN = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,1],[1,1,1,0],
                  [0,0,1,1],[1,1,0,0],[0,1,0,1],[1,0,1,0]],dtype=float)
NEW = np.array([[0,0,1,0],[0,1,1,0]],dtype=float)
ALL = np.array(list(itertools.product((0,1),repeat=4)),dtype=float)

class Meter:
    def __init__(self): self.counts = {}
    def add(self,k,v=1): self.counts[k]=self.counts.get(k,0)+v

def klass(b):
    t=sum(b[i]!=b[(i+1)%4] for i in range(4))
    return 0 if t==0 else 3 if t==4 else 1 if int(sum(b))%2 else 2

def basis(a,b):
    return np.stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b),axis=-1)

def local(a,b,theta):
    phi=basis(a,b)
    value=phi@theta
    da=(1-b)*(theta[2]-theta[0])+b*(theta[3]-theta[1])
    db=(1-a)*(theta[1]-theta[0])+a*(theta[3]-theta[2])
    return value,phi,da,db

def direct(x,theta,meter):
    meter.add("direct_calls")
    d,phi,_,_=local(x,np.roll(x,-1),theta)
    p,inner,da,db=local(d[0],d[2],theta)
    r=np.array([d.mean(),p])
    jac=np.stack((phi.mean(axis=0),inner+da*phi[0]+db*phi[2]))
    return d,r,jac

def objective(state,theta,meter,target=None,defaults=False,grad=False):
    meter.add("energy_calls")
    x,d,r=state["x"],state["d"],state["r"]
    f,phi,da,db=local(x,np.roll(x,-1),theta)
    p,psi,pa,pb=local(d[0],d[2],theta)
    e=d-f; eq=r[0]-d.mean(); ep=r[1]-p
    value=.5*float(e@e+eq*eq+ep*ep)
    if target is not None: value+=.5*float(np.sum((r-target)**2))
    if defaults: value+=.025*float(np.sum((x-DEFAULT)**2))+.05*float(np.sum(x*x*(1-x)**2))
    if not grad: return value
    meter.add("joint_gradient_calls")
    gx=-e*da-np.roll(e*db,1)
    if defaults: gx+=.05*(x-DEFAULT)+.1*x*(1-x)*(1-2*x)
    gd=e-eq/4
    gd[0]-=ep*pa; gd[2]-=ep*pb
    gr=np.array([eq,ep])
    if target is not None: gr+=r-target
    gt=-phi.T@e-ep*psi
    return value,{"x":gx,"d":gd,"r":gr},gt

def projected(state,mask,observed):
    out={k:np.clip(v,0,1) for k,v in state.items()}
    out["x"][mask]=observed[mask]
    return out

def step(state,theta,mask,observed,meter,target=None,defaults=False,plastic=True):
    before,gs,gt=objective(state,theta,meter,target,defaults,True)
    for j in range(13):
        a=2.**(-j)
        candidate=projected({k:state[k]-a*DT*gs[k] for k in state},mask,observed)
        tt=np.clip(theta-a*ETA*gt,0,1) if plastic else theta
        after=objective(candidate,tt,meter,target,defaults)
        meter.add("joint_trials")
        if not np.isfinite(after): raise AssertionError("Nonfinite objective")
        if after<=before+1e-12:
            meter.add("joint_steps"); meter.add("joint_halvings",j)
            return candidate,tt
        meter.add("rejected_joint_trials")
    raise AssertionError("Joint line search exhausted")

def direct_step(x,target,theta,meter):
    _,r,jac=direct(x,theta,meter)
    err=r-target; before=.5*float(err@err); gt=jac.T@err
    meter.add("direct_gradient_calls")
    for j in range(13):
        tt=np.clip(theta-2.**(-j)*ETA*gt,0,1)
        _,rr,_=direct(x,tt,meter)
        after=.5*float(np.sum((rr-target)**2)); meter.add("direct_trials")
        if after<=before+1e-12:
            meter.add("direct_steps"); meter.add("direct_halvings",j)
            return tt
        meter.add("rejected_direct_trials")
    raise AssertionError("Direct line search exhausted")

def diagnostic(theta,meter):
    rows=[]
    for b in ALL:
        _,r,_=direct(b,theta,meter)
        scores=2*PROTOTYPES@r-np.sum(PROTOTYPES**2,axis=1)
        rows.append({"outer":b.astype(int).tolist(),"inner":r.tolist(),
                     "diagnostic_class":int(np.argmax(scores)),"true_class":klass(b)})
    train_set={tuple(b.astype(int)) for b in TRAIN}; new_set={tuple(b.astype(int)) for b in NEW}
    subsets={"original_training":[row for row in rows if tuple(row["outer"]) in train_set],
             "A_heldout":[row for row in rows if tuple(row["outer"]) not in train_set],
             "still_unseen_after_B":[row for row in rows if tuple(row["outer"]) not in train_set|new_set]}
    scores={k:{"correct":sum(row["diagnostic_class"]==row["true_class"] for row in v),"total":len(v)} for k,v in subsets.items()}
    return {"theta":theta.tolist(),"correct_inner_diagnostics":sum(x["diagnostic_class"]==x["true_class"] for x in rows),"subset_diagnostics":scores,
            "rows":rows,"role":"Internal diagnostics only, not task outputs."}

def numerical_checks(meter):
    for b in ALL:
        _,r,_=direct(b,XOR,meter)
        assert np.array_equal(r,PROTOTYPES[klass(b)])
        scores=2*PROTOTYPES@r-np.sum(PROTOTYPES**2,axis=1)
        assert scores[klass(b)]-max(np.delete(scores,klass(b)))>=.25
    theta=np.array([.2,.7,.6,.3])
    state={"x":np.array([.2,.4,.6,.8]),"d":np.array([.3,.5,.4,.7]),"r":np.array([.4,.6])}
    target=PROTOTYPES[1]
    _,gs,gt=objective(state,theta,meter,target,True,True)
    worst=0.
    for key,arr in list(state.items())+[("theta",theta)]:
        for i in range(len(arr)):
            old=arr[i]; arr[i]=old+1e-6; plus=objective(state,theta,meter,target,True)
            arr[i]=old-1e-6; minus=objective(state,theta,meter,target,True); arr[i]=old
            expected=gt[i] if key=="theta" else gs[key][i]
            worst=max(worst,abs((plus-minus)/2e-6-expected))
    _,r,jac=direct(state["x"],theta,meter); expected=jac.T@(r-target)
    direct_worst=0.
    for i in range(4):
        old=theta[i]; theta[i]=old+1e-6; plus=.5*np.sum((direct(state["x"],theta,meter)[1]-target)**2)
        theta[i]=old-1e-6; minus=.5*np.sum((direct(state["x"],theta,meter)[1]-target)**2); theta[i]=old
        direct_worst=max(direct_worst,abs((plus-minus)/2e-6-expected[i]))
    assert worst<=2e-7 and direct_worst<=2e-7
    mask=np.array([True,False,True,False]); observed=state["x"].copy()
    old=objective(state,theta,meter,target,True)
    ss,tt=step(state,theta,mask,observed,meter,target,True)
    assert np.array_equal(ss["x"][mask],observed[mask]) and objective(ss,tt,meter,target,True)<=old+1e-12
    return {"joint_error":float(worst),"direct_error":float(direct_worst),"all_16_codes_distinct_by_class":True}

def queries():
    return [(b.copy(),pair) for b in TRAIN for pair in itertools.combinations(range(4),2)]

def oracle(source,pair,meter):
    meter.add("oracle_calls"); meter.add("oracle_candidate_visits",16)
    valid=[b for b in ALL if klass(b)==klass(source) and all(b[i]==source[i] for i in pair)]
    minimum=min(np.count_nonzero(b!=DEFAULT) for b in valid)
    return [b for b in valid if np.count_nonzero(b!=DEFAULT)==minimum]

def initialize_query(theta,source,pair,meter):
    mask=np.array([i in pair for i in range(4)]); x=DEFAULT.copy(); x[mask]=source[mask]
    d,r,_=direct(x,theta,meter)
    return {"x":x,"d":d,"r":r},mask

def sample(state,theta,original,source,mask,valid,meter):
    meter.add("outer_readouts")
    value,gs,_=objective(state,theta,meter,PROTOTYPES[klass(source)],True,True)
    pp=projected({k:state[k]-gs[k] for k in state},mask,source)
    residual=max(float(np.max(np.abs(pp[k]-state[k]))) for k in state)
    rounded=(state["x"]>=.5).astype(float)
    assert np.array_equal(state["x"][mask],source[mask])
    correct=any(np.array_equal(rounded,b) for b in valid)
    return {"outer":state["x"].tolist(),"thresholded":rounded.astype(int).tolist(),
            "oracle_match":correct,"class_valid":klass(rounded)==klass(source),
            "hidden_error":min(float(np.mean((state["x"][~mask]-b[~mask])**2)) for b in valid),
            "hidden_threshold_margin":float(np.min(np.abs(state["x"][~mask]-.5))),
            "activity_residual":residual,"correct_while_unsettled":correct and residual>1e-5,
            "theta_drift":float(np.linalg.norm(theta-original)),"energy":value,
            "inner_diagnostic":state["r"].tolist()}

def battery(theta0,meter):
    rows=[]
    for plastic in (False,True):
        for source,pair in queries():
            valid=oracle(source,pair,meter); theta=theta0.copy()
            state,mask=initialize_query(theta,source,pair,meter)
            reads={"0":sample(state,theta,theta0,source,mask,valid,meter)}
            for t in range(1,65):
                state,theta=step(state,theta,mask,source,meter,PROTOTYPES[klass(source)],True,plastic)
                if t in HORIZONS: reads[str(t)]=sample(state,theta,theta0,source,mask,valid,meter)
            hits=[t for t in HORIZONS if reads[str(t)]["oracle_match"]]
            reversals=sum(reads[str(a)]["oracle_match"] and not reads[str(b)]["oracle_match"] for a,b in zip(HORIZONS,HORIZONS[1:]))
            rows.append({"plastic":plastic,"source_pattern":source.astype(int).tolist(),"observed_indices":list(pair),
                         "first_correct_horizon":min(hits) if hits else None,"reversals":reversals,"readouts":reads})
    curves={}
    for plastic in (False,True):
        subset=[row for row in rows if row["plastic"]==plastic]
        curves["plastic" if plastic else "frozen"]=[{"steps":t,"queries":48,
            "oracle_matches":sum(row["readouts"][str(t)]["oracle_match"] for row in subset),
            "class_valid":sum(row["readouts"][str(t)]["class_valid"] for row in subset),
            "correct_while_unsettled":sum(row["readouts"][str(t)]["correct_while_unsettled"] for row in subset),
            "mean_hidden_error":float(np.mean([row["readouts"][str(t)]["hidden_error"] for row in subset])),
            "mean_theta_drift":float(np.mean([row["readouts"][str(t)]["theta_drift"] for row in subset]))} for t in HORIZONS]
    meter.add("completion_queries",96)
    return {"curves":curves,"rows":rows}

def benchmark(meter):
    result={}
    for budget in ("oracle",1,8,64):
        times=[]; scores=[]
        for repeat in range(4):
            wall,cpu=time.perf_counter(),time.process_time(); correct=0
            for source,pair in queries():
                valid=oracle(source,pair,meter)
                if budget=="oracle":
                    output=valid[0]
                else:
                    state,mask=initialize_query(XOR,source,pair,meter)
                    for _ in range(budget): state,_=step(state,XOR,mask,source,meter,PROTOTYPES[klass(source)],True,False)
                    output=(state["x"]>=.5).astype(float)
                correct+=any(np.array_equal(output,b) for b in valid)
            scores.append(correct)
            if repeat: times.append({"wall_seconds":time.perf_counter()-wall,"process_cpu_seconds":time.process_time()-cpu})
        assert len(set(scores))==1
        result[str(budget)]={"batch_size":48,"correct":scores[0],"warmup_batches":1,"measured_batches":3,
            "samples":times,"summary":{k:{"median":statistics.median(t[k] for t in times),"min":min(t[k] for t in times),"max":max(t[k] for t in times)} for k in times[0]},
            "amortized_wall_seconds_per_query":statistics.median(t["wall_seconds"] for t in times)/48}
    return result

def main():
    if RESULT.exists() or RUNTIME.exists(): raise SystemExit("Refusing to overwrite receipts")
    provenance={"revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "dirty_status":subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),
        "sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CONTRACT,Path(__file__).resolve())},
        "command":".venv/bin/python -m experiments.outer_anytime","root_seed":SEED}
    wall0,cpu0=time.perf_counter(),time.process_time(); phases={}; work={}
    def phase(name,fn):
        meter=Meter(); wall,cpu=time.perf_counter(),time.process_time(); value=fn(meter)
        phases[name]={"wall_seconds":time.perf_counter()-wall,"process_cpu_seconds":time.process_time()-cpu}
        work[name]=meter.counts; return value
    checks=phase("numerical_checks",numerical_checks)
    supplied={"inner_diagnostics":phase("supplied_inner_diagnostics",lambda m:diagnostic(XOR,m)),
              "outer_completion":phase("supplied_outer_completion",lambda m:battery(XOR,m))}
    print(json.dumps({"supplied_outer_curves":supplied["outer_completion"]["curves"]}),flush=True)
    runs=[]
    for replica in range(3):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,replica,1]))
        stream_A=[TRAIN[j].copy() for _ in range(200) for j in rng.permutation(8)]
        stream_B=[NEW[j%2].copy() for j in range(100)]
        initial=np.random.default_rng(np.random.SeedSequence([SEED,replica,0])).uniform(.35,.65,4)
        for arm in ("joint_continuous","direct_gradient"):
            prefix=f"replica_{replica}/{arm}"; theta=initial.copy()
            state={"x":np.full(4,.5),"d":np.full(4,.5),"r":np.full(2,.5)}
            run={"replica":replica,"arm":arm,"checkpoints":{}}
            run["checkpoints"]["initial"]={"inner_diagnostics":phase(prefix+"/initial_diagnostics",lambda m:diagnostic(theta,m))}
            for block,stream in (("A",stream_A),("B",stream_B)):
                def train(meter):
                    nonlocal state,theta
                    for x in stream:
                        meter.add("presentations"); state["x"]=x.copy(); target=PROTOTYPES[klass(x)]
                        for _ in range(20):
                            meter.add("training_updates")
                            if arm=="joint_continuous": state,theta=step(state,theta,np.ones(4,dtype=bool),x,meter,target)
                            else: theta=direct_step(x,target,theta,meter)
                phase(prefix+f"/block_{block}_learning",train)
                checkpoint={"inner_diagnostics":phase(prefix+f"/post_{block}_diagnostics",lambda m:diagnostic(theta,m)),
                            "outer_completion":phase(prefix+f"/post_{block}_outer",lambda m:battery(theta,m))}
                run["checkpoints"]["post_"+block]=checkpoint
                print(json.dumps({"arm":arm,"replica":replica,"block":block,"theta":theta.tolist(),
                    "inner_correct_diagnostic_only":checkpoint["inner_diagnostics"]["correct_inner_diagnostics"],
                    "outer_frozen_curve":[r["oracle_matches"] for r in checkpoint["outer_completion"]["curves"]["frozen"]],
                    "outer_plastic_curve":[r["oracle_matches"] for r in checkpoint["outer_completion"]["curves"]["plastic"]]}),flush=True)
            runs.append(run)
    timing=phase("benchmark",benchmark)
    assert sum(v.get("training_updates",0) for v in work.values())==204000
    assert sum(v.get("completion_queries",0) for v in work.values())==1248
    science={"scope":"Outer-only task output; inner states diagnostic; supplied geometry, learned local interpolation.",
        "horizons":HORIZONS,"checks":checks,"supplied":supplied,"runs":runs,"phase_work_counts":work,
        "benchmark_quality":{k:{"correct":v["correct"],"queries":v["batch_size"]} for k,v in timing.items()},"correctness_checks_passed":True}
    wall,cpu=time.perf_counter(),time.process_time(); RESULT.write_text(json.dumps(science,sort_keys=True,indent=2)+"\n")
    phases["serialization"]={"wall_seconds":time.perf_counter()-wall,"process_cpu_seconds":time.process_time()-cpu}
    elapsed={"wall_seconds":time.perf_counter()-wall0,"process_cpu_seconds":time.process_time()-cpu0}
    cpuinfo=Path("/proc/cpuinfo")
    cpu_name=next((line.split(":",1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")),None) if cpuinfo.exists() else None
    receipt={"provenance":provenance,"environment":{"python":sys.version,"numpy":np.__version__,"numpy_config":np.show_config(mode="dicts"),
        "cpu":cpu_name,"platform":platform.platform(),"threads":{k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        "processes":1,"workers":0,"device":"CPU","dtype":"float64","hardware_limits":None,"hardware_limits_reason":"No quota inventory; no scaling claim."},
        "whole_run":elapsed,"phases":phases,"benchmarks":timing,"phase_work_counts":work,
        "benchmark_scope":"Completion batch timings include semantic oracle scoring on each query, allocation, gradients and line search. Not readout-only latency.",
        "memory":{"peak_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"rss_unit":"KiB",
            "rss_scope":"Parent high-water including all checkpoints/curves and serialization.","learned_parameter_bytes_per_model":32,
            "activity_bytes":80,"fixed_prototype_bytes":64,"optimizer_bytes":0,"replay_bytes":0,
            "temporary_workspace_bytes":None,"temporary_workspace_reason":"Allocator workspace not separately measured.","scientific_artifact_bytes":RESULT.stat().st_size},
        "selection_scope":"All six declared trajectories, supplied control, all curves and benchmark budgets; no retuning or best-seed selection.",
        "manual_development_cost":None,"manual_development_cost_reason":"Not measured.",
        "scientific_sha256":hashlib.sha256(RESULT.read_bytes()).hexdigest()}
    RUNTIME.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"complete":True,"whole_run":elapsed}),flush=True)

if __name__=="__main__": main()
