"""Read-only checkpoint diagnosis; no updates or refits."""

from experiments import continuous_local_learning as model

import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs/continuous-local-learning-v1-diagnosis.md"
RESULT = ROOT / "reports/continuous-local-learning-v1-diagnosis.json"
RUNTIME = ROOT / "reports/continuous-local-learning-v1-diagnosis-runtime.json"


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite diagnostic receipts")
    original_receipt = json.loads(model.RUNTIME.read_text())
    for path, expected in original_receipt["provenance"]["sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(model.RESULT.read_bytes()).hexdigest() == original_receipt["scientific_sha256"]
    paths = [NOTE, Path(__file__).resolve(), Path(model.__file__).resolve(), model.CONTRACT, model.RESULT, model.RUNTIME]
    provenance = {"sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  "revision": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  "dirty_status": subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),
                  "command": ".venv/bin/python -m experiments.continuous_local_learning_diagnosis"}
    wall, cpu = time.perf_counter(), time.process_time()
    original = json.loads(model.RESULT.read_text())
    meter = model.Meter()
    checkpoints, initializations = [], []
    labels = np.array([model.klass(b) for b in model.ALL])
    completion_count = unchanged = 0
    max_hidden_motion = 0.
    baseline_rows = None
    for run in original["runs"]:
        replica, arm = run["replica"], run["arm"]
        rng = np.random.default_rng(np.random.SeedSequence([model.ROOT_SEED, replica, 1]))
        for _ in range(200):
            last_index = rng.permutation(8)[-1]
        last_labels = {"initial": None, "post_A": model.klass(model.TRAIN[last_index]), "post_B": model.klass(model.NEW[-1])}
        for checkpoint, saved in run["checkpoints"].items():
            p = model.initialize(replica) if checkpoint == "initial" else {k:np.array(v) for k,v in saved["parameters"].items()}
            predicted, pooled, decoder = [], [], []
            for b in model.ALL:
                y, (_, z, _, h, _) = model.forward(b, p, meter)
                predicted.append(y); pooled.append(z.mean(axis=0)); decoder.append(h)
            predicted, pooled, decoder = map(np.array, (predicted, pooled, decoder))
            assert np.max(np.abs(predicted-np.array(saved["predictions"]))) <= 1e-12
            local = model.sigmoid(np.array([[0.,0.,1.],[0.,1.,1.],[1.,0.,1.],[1.,1.,1.]]) @ p["W"].T)
            u, v, w = local[0], (local[1]+local[2])/2, local[3]
            odd1, odd2, middle = (u+v)/2, (w+v)/2, (u+w+2*v)/4
            assert np.max(np.abs(pooled[labels==1].mean(axis=0)-pooled[labels==2].mean(axis=0))) <= 1e-12
            a = p["A"] @ np.r_[middle,1.]
            d = p["A"][:,:4] @ (u-w)/4
            gap = (model.sigmoid(a+d)+model.sigmoid(a-d))/2-model.sigmoid(a)
            bound = d*d/(12*np.sqrt(3))
            assert np.all(np.abs(gap) <= bound+1e-12)
            values = [p["B"] @ np.r_[model.sigmoid(p["A"] @ np.r_[t,1.]),1.] for t in (odd1,odd2,middle)]
            margins = [values[0][1]-values[0][2], values[1][1]-values[1][2], values[2][2]-values[2][1]]
            gs = {"actual": {k:np.zeros_like(v) for k,v in p.items()},
                  "mean_target": {k:np.zeros_like(v) for k,v in p.items()}}
            for b in model.TRAIN:
                for kind, target in (("actual", model.onehot(b)), ("mean_target", np.full(4,.25))):
                    _, grad = model.direct_objective(b, target, p, meter, True)
                    for k in p:
                        gs[kind][k] += grad[k]/8
            gradient_diagnosis = {k:{"mean_target_norm":float(np.linalg.norm(gs["mean_target"][k])),
                                     "label_contrast_norm":float(np.linalg.norm(gs["actual"][k]-gs["mean_target"][k])),
                                     "actual_mean_norm":float(np.linalg.norm(gs["actual"][k]))} for k in p}
            last = last_labels[checkpoint]
            checkpoints.append({"replica":replica,"arm":arm,"checkpoint":checkpoint,
                "polarity_norm":float(np.linalg.norm(u-w)),
                "relational_contrast_norm":float(np.linalg.norm((u+w)/2-v)),
                "centered_pooled_singular_values":np.linalg.svd(pooled-pooled.mean(axis=0),compute_uv=False).tolist(),
                "odd_adjacent_mean_gap":{name:float(np.linalg.norm(data[labels==1].mean(axis=0)-data[labels==2].mean(axis=0)))
                                         for name,data in (("pooled",pooled),("decoder",decoder),("output",predicted))},
                "decoder_gap":gap.tolist(),"decoder_gap_bound":bound.tolist(),
                "minimum_correct_odd_adjacent_margin":float(min(margins)),
                "last_presented_label":last,
                "predictions_equal_last_label":int(np.sum(predicted.argmax(axis=1)==last)) if last is not None else None,
                "direct_predictive_gradient_diagnosis":gradient_diagnosis})
            if checkpoint == "initial":
                continue
            if baseline_rows is None:
                baseline_rows = []
            for row in saved["completion"]["rows"]:
                initial = model.DEFAULT.copy()
                mask = np.array([i in row["observed_indices"] for i in range(4)])
                source = np.array(row["source_pattern"],dtype=float)
                initial[mask] = source[mask]
                unchanged += np.array_equal(np.array(row["thresholded"]), initial)
                completion_count += 1
                max_hidden_motion = max(max_hidden_motion, float(np.max(np.abs(np.array(row["final_x"])[~mask]-initial[~mask]))))
                if replica==0 and arm=="joint_continuous" and checkpoint=="post_A" and not row["plastic"]:
                    baseline_rows.append({"class_context":row["class_context"],"observed_indices":row["observed_indices"],
                                          "source_pattern":row["source_pattern"],"default_prediction":initial.astype(int).tolist(),
                                          "class_valid":model.klass(initial)==row["class_context"]})
            for source in model.TRAIN:
                for pair in itertools.combinations(range(4),2):
                    mask=np.array([i in pair for i in range(4)])
                    x=model.DEFAULT.copy(); x[mask]=source[mask]
                    _,(_,z,_,h,_)=model.forward(x,p,meter)
                    state={"x":x,"z":z,"h":h,"y":model.onehot(source)}
                    _, gs0, gp0 = model.joint(state,p,meter,model.DEFAULT,True)
                    hidden=float(np.max(np.abs(gs0["x"][~mask])))
                    assert hidden<=1e-12
                    initializations.append({"arm":arm,"replica":replica,"checkpoint":checkpoint,
                        "source":source.astype(int).tolist(),"observed_indices":list(pair),
                        "hidden_x_gradient_max":hidden,"decoder_activity_gradient_norm":float(np.linalg.norm(gs0["h"])),
                        "data_parameter_gradient_norms":{k:float(np.linalg.norm(gp0[k]-model.DECAY*p[k])) for k in p}})
    assert len(checkpoints)==18 and len(initializations)==576 and completion_count==1152
    assert len(baseline_rows)==48
    assert meter.counts["direct_gradient_calls"]==288 and meter.counts["joint_gradient_calls"]==576
    result={"scope":"Post-run checkpoint diagnosis; no training, updates or inference trajectories.",
            "checkpoint_diagnostics":checkpoints,"completion_initialization_gradients":initializations,
            "default_only_baseline":{"queries":48,"class_valid":sum(row["class_valid"] for row in baseline_rows),"rows":baseline_rows},
            "observed_completion_behavior":{"queries":completion_count,"thresholded_equal_initial":int(unchanged),
                                            "max_hidden_continuous_motion":max_hidden_motion},
            "work_counts":meter.counts,"correctness_checks_passed":True}
    RESULT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    elapsed={"wall_seconds":time.perf_counter()-wall,"process_cpu_seconds":time.process_time()-cpu}
    runtime={"provenance":provenance,"timing":elapsed,
             "scope":"After source verification/provenance through diagnostic serialization; excludes imports/runtime receipt.",
             "environment":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
                            "threads":{k:model.os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                            "cpu":original_receipt["environment"]["cpu"],"numpy_config":original_receipt["environment"]["numpy_config"],
                            "processes":1,"workers":0,"device":"CPU","dtype":"float64"},
             "work_counts":{**meter.counts,"full_svd_calls_16_by_4":18,"parameter_updates":0,"fits":0,"tuning_candidates":0},
             "memory":{"peak_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"rss_unit":"KiB",
                       "rss_scope":"Parent high-water including parsed original artifacts and diagnostic arrays.",
                       "temporary_workspace_bytes":None,"temporary_workspace_reason":"Allocator-level workspace not separately instrumented.",
                       "scientific_artifact_bytes":RESULT.stat().st_size},
             "scientific_sha256":hashlib.sha256(RESULT.read_bytes()).hexdigest()}
    RUNTIME.write_text(json.dumps(runtime,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"checks_passed":True,"default_only_correct":result["default_only_baseline"]["class_valid"],
                      "completion_behavior":result["observed_completion_behavior"],"timing":elapsed}))
    for row in checkpoints:
        if row["checkpoint"]=="post_A":
            print(json.dumps({k:row[k] for k in ("arm","replica","polarity_norm","relational_contrast_norm",
                "odd_adjacent_mean_gap","minimum_correct_odd_adjacent_margin","predictions_equal_last_label","direct_predictive_gradient_diagnosis")}))


if __name__ == "__main__":
    main()
