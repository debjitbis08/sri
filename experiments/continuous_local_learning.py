"""Continuous local adaptation on the declared four-bit development fixture.

Run: .venv/bin/python -m experiments.continuous_local_learning
Contract: docs/continuous-local-learning-v1-contract.md
"""

import os
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import hashlib
import itertools
import json
import platform
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/continuous-local-learning-v1-contract.md"
RESULT = ROOT / "reports/continuous-local-learning-v1-results.json"
RUNTIME = ROOT / "reports/continuous-local-learning-v1-runtime.json"
ROOT_SEED = 2026102511
DECAY, DT, ETA = 1e-4, 0.1, 0.005
TRAIN = np.array([[0,0,0,0], [1,1,1,1], [0,0,0,1], [1,1,1,0],
                  [0,0,1,1], [1,1,0,0], [0,1,0,1], [1,0,1,0]], dtype=float)
NEW = np.array([[0,0,1,0], [0,1,1,0]], dtype=float)
ALL = np.array(list(itertools.product((0, 1), repeat=4)), dtype=float)
DEFAULT = np.array([0.,0.,1.,1.])


class Meter:
    def __init__(self):
        self.counts = {}

    def add(self, key, value=1):
        self.counts[key] = self.counts.get(key, 0) + value

    def peak(self, key, value):
        self.counts[key] = max(self.counts.get(key, 0), float(value))


def klass(x):
    b = [int(v) for v in x]
    t = sum(b[i] != b[(i+1) % 4] for i in range(4))
    return 0 if t == 0 else 3 if t == 4 else 1 if sum(b) % 2 else 2


def onehot(x):
    return np.eye(4)[klass(x)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clone(d):
    return {k: v.copy() for k, v in d.items()}


def initialize(replica):
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, replica, 0]))
    return {k: np.clip(rng.normal(0, scale, shape), -3, 3)
            for k, shape, scale in (("W", (4,3), .5), ("A", (8,5), .5), ("B", (4,9), .2))}


def forward(x, p, meter):
    meter.add("direct_forward_calls")
    phi = np.column_stack((x, np.roll(x, -1), np.ones(4)))
    z = sigmoid(phi @ p["W"].T)
    v = np.r_[z.mean(axis=0), 1.]
    h = sigmoid(p["A"] @ v)
    hb = np.r_[h, 1.]
    y = p["B"] @ hb
    return y, (phi, z, v, h, hb)


def direct_objective(x, target, p, meter, gradients=False):
    y, (phi, z, v, h, hb) = forward(x, p, meter)
    err = y-target
    value = .5*float(err@err) + DECAY/2*sum(float(np.sum(a*a)) for a in p.values())
    if not gradients:
        return value
    meter.add("direct_gradient_calls")
    gh = (p["B"][:,:8].T @ err) * h*(1-h)
    gz = (p["A"][:,:4].T @ gh)[None,:]/4 * z*(1-z)
    gp = {"B": np.outer(err, hb)+DECAY*p["B"],
          "A": np.outer(gh, v)+DECAY*p["A"],
          "W": gz.T @ phi+DECAY*p["W"]}
    return value, gp


def joint(state, p, meter, defaults=None, gradients=False):
    meter.add("joint_energy_calls")
    x, z, h, y = (state[k] for k in ("x", "z", "h", "y"))
    phi = np.column_stack((x, np.roll(x, -1), np.ones(4)))
    f = sigmoid(phi @ p["W"].T)
    v = np.r_[z.mean(axis=0), 1.]
    g = sigmoid(p["A"] @ v)
    hb = np.r_[h, 1.]
    ez, eh, ey = z-f, h-g, y-p["B"] @ hb
    value = .5*sum(float(np.sum(e*e)) for e in (ez, eh, ey))
    value += DECAY/2*sum(float(np.sum(a*a)) for a in p.values())
    if defaults is not None:
        value += .025*float(np.sum((x-defaults)**2)) + .05*float(np.sum(x*x*(1-x)**2))
    if not gradients:
        return value
    meter.add("joint_gradient_calls")
    dz, dh = ez*f*(1-f), eh*g*(1-g)
    gx = -(dz @ p["W"][:,0]) - np.roll(dz @ p["W"][:,1], 1)
    if defaults is not None:
        gx += .05*(x-defaults) + .1*x*(1-x)*(1-2*x)
    gs = {"x": gx, "z": ez-(dh @ p["A"][:,:4])[None,:]/4,
          "h": eh-p["B"][:,:8].T @ ey, "y": ey}
    gp = {"W": -dz.T @ phi+DECAY*p["W"],
          "A": -np.outer(dh, v)+DECAY*p["A"],
          "B": -np.outer(ey, hb)+DECAY*p["B"]}
    return value, gs, gp


def project(state, mask, observed, target):
    for key in ("x", "z", "h"):
        state[key] = np.clip(state[key], 0, 1)
    state["x"][mask] = observed[mask]
    if target is not None:
        state["y"] = target.copy()
    return state


def joint_step(state, p, mask, observed, target, meter, eta=ETA, defaults=None):
    before, gs, gp = joint(state, p, meter, defaults, True)
    for j in range(13):
        scale = 2.**(-j)
        candidate = project({k: state[k]-scale*DT*gs[k] for k in state}, mask, observed, target)
        pp = {k: np.clip(p[k]-scale*eta*gp[k], -3, 3) for k in p} if eta else p
        after = joint(candidate, pp, meter, defaults)
        meter.add("joint_trial_evaluations")
        if not np.isfinite(after):
            raise AssertionError("Nonfinite joint objective")
        if after <= before+1e-12:
            meter.add("accepted_joint_steps")
            meter.add("joint_backtrack_halvings", j)
            meter.peak("maximum_accepted_energy_increase", after-before)
            return candidate, pp
        meter.add("rejected_joint_trials")
    raise AssertionError("Joint line search exhausted")


def direct_step(x, target, p, meter):
    before, gp = direct_objective(x, target, p, meter, True)
    for j in range(13):
        pp = {k: np.clip(p[k]-2.**(-j)*ETA*gp[k], -3, 3) for k in p}
        after = direct_objective(x, target, pp, meter)
        meter.add("direct_trial_evaluations")
        if not np.isfinite(after):
            raise AssertionError("Nonfinite direct objective")
        if after <= before+1e-12:
            meter.add("accepted_direct_steps")
            meter.add("direct_backtrack_halvings", j)
            return pp
        meter.add("rejected_direct_trials")
    raise AssertionError("Direct line search exhausted")


def residuals(state, p, mask, observed, target, meter, defaults=None):
    energy, gs, gp = joint(state, p, meter, defaults, True)
    projected = project({k: state[k]-gs[k] for k in state}, mask, observed, target)
    sr = max(float(np.max(np.abs(projected[k]-state[k]))) for k in state)
    pr = max(float(np.max(np.abs(np.clip(p[k]-gp[k], -3, 3)-p[k]))) for k in p)
    return energy, sr, pr


def classification(p, meter):
    predictions = np.array([forward(x, p, meter)[0] for x in ALL])
    labels = np.array([klass(x) for x in ALL])
    pred = predictions.argmax(axis=1)
    groups = {"all": np.ones(16, dtype=bool),
              "original_training": np.array([any(np.array_equal(x,t) for t in TRAIN) for x in ALL]),
              "block_B_patterns": np.array([any(np.array_equal(x,t) for t in NEW) for x in ALL])}
    groups["A_heldout"] = ~groups["original_training"]
    groups["still_unseen_after_B"] = groups["A_heldout"] & ~groups["block_B_patterns"]
    scores = {k: {"correct": int(np.sum(pred[mask] == labels[mask])), "total": int(mask.sum()),
                  "mean_squared_vector_error": float(np.mean(np.sum((predictions[mask]-np.eye(4)[labels[mask]])**2, axis=1)))}
              for k, mask in groups.items()}
    return {"scores": scores, "predictions": predictions.tolist(),
            "predicted_classes": pred.tolist(), "distinct_predicted_classes": len(set(pred.tolist())),
            "prediction_coordinate_std_mean": float(predictions.std(axis=0).mean()),
            "parameter_l2": float(np.sqrt(sum(np.sum(v*v) for v in p.values())))}


def check_gradients(meter):
    p = initialize(0)
    state = {"x": np.array([.2,.4,.6,.8]), "z": np.linspace(.2,.8,16).reshape(4,4),
             "h": np.linspace(.3,.7,8), "y": np.array([.1,.2,.3,.4])}
    _, gs, gp = joint(state, p, meter, DEFAULT, True)
    worst = 0.
    for obj, gradients in ((state, gs), (p, gp)):
        for key, array in obj.items():
            for idx in np.ndindex(array.shape):
                old = array[idx]
                array[idx] = old+1e-6
                plus = joint(state, p, meter, DEFAULT)
                array[idx] = old-1e-6
                minus = joint(state, p, meter, DEFAULT)
                array[idx] = old
                worst = max(worst, abs((plus-minus)/2e-6-gradients[key][idx]))
    target = np.eye(4)[2]
    _, gradients = direct_objective(state["x"], target, p, meter, True)
    direct_worst = 0.
    for key, array in p.items():
        for idx in np.ndindex(array.shape):
            old = array[idx]
            array[idx] = old+1e-6
            plus = direct_objective(state["x"], target, p, meter)
            array[idx] = old-1e-6
            minus = direct_objective(state["x"], target, p, meter)
            array[idx] = old
            direct_worst = max(direct_worst, abs((plus-minus)/2e-6-gradients[key][idx]))
    assert worst <= 2e-7 and direct_worst <= 2e-7, (worst, direct_worst)
    mask = np.array([True,False,True,False])
    state = project(state, mask, state["x"], target)
    observed = state["x"].copy()
    before = joint(state, p, meter, DEFAULT)
    changed, pp = joint_step(state, p, mask, observed, target, meter, defaults=DEFAULT)
    assert joint(changed, pp, meter, DEFAULT) <= before+1e-12
    assert np.array_equal(changed["x"][mask], observed[mask]) and np.array_equal(changed["y"], target)
    return {"joint_max_absolute_error": float(worst), "direct_max_absolute_error": float(direct_worst)}


def completion_battery(p0, meter):
    rows = []
    before_accuracy = classification(p0, meter)["scores"]["all"]["correct"]
    for plastic in (False, True):
        for pattern in TRAIN:
            for pair in itertools.combinations(range(4), 2):
                mask = np.array([i in pair for i in range(4)])
                target = onehot(pattern)
                meter.add("semantic_oracle_candidate_visits", 16)
                valid = [b for b in ALL if klass(b) == klass(pattern) and np.array_equal(b[mask], pattern[mask])]
                assert valid
                best_distance = min(np.count_nonzero(b != DEFAULT) for b in valid)
                oracle = [b for b in valid if np.count_nonzero(b != DEFAULT) == best_distance]
                x = DEFAULT.copy()
                x[mask] = pattern[mask]
                p = clone(p0)
                _, (_, z, _, h, _) = forward(x, p, meter)
                state = {"x": x, "z": z, "h": h, "y": target.copy()}
                initial = joint(state, p, meter, DEFAULT)
                for _ in range(300):
                    state, p = joint_step(state, p, mask, pattern, target, meter,
                                          eta=ETA if plastic else 0., defaults=DEFAULT)
                final, sr, pr = residuals(state, p, mask, pattern, target, meter, DEFAULT)
                assert final <= initial+300e-12 and np.array_equal(state["x"][mask], pattern[mask])
                rounded = (state["x"] >= .5).astype(float)
                after_accuracy = classification(p, meter)["scores"]["all"]["correct"] if plastic else before_accuracy
                rows.append({"plastic": plastic, "source_pattern": pattern.astype(int).tolist(),
                             "observed_indices": list(pair), "class_context": klass(pattern),
                             "final_x": state["x"].tolist(), "thresholded": rounded.astype(int).tolist(),
                             "class_valid": klass(rounded) == klass(pattern),
                             "nearest_default_oracle_match": any(np.array_equal(rounded,b) for b in oracle),
                             "hidden_mean_squared_error_to_oracle": min(float(np.mean((state["x"][~mask]-b[~mask])**2)) for b in oracle),
                             "initial_energy": initial, "final_energy": final,
                             "activity_residual": sr, "parameter_residual": pr,
                             "activity_settled": sr <= 1e-5,
                             "parameter_l2_drift": float(np.sqrt(sum(np.sum((p[k]-p0[k])**2) for k in p))),
                             "classification_before_probe": before_accuracy,
                             "classification_after_probe": after_accuracy})
    summary = {}
    for plastic in (False, True):
        selected = [r for r in rows if r["plastic"] == plastic]
        summary["plastic" if plastic else "frozen"] = {
            "queries": len(selected), "class_valid": sum(r["class_valid"] for r in selected),
            "oracle_matches": sum(r["nearest_default_oracle_match"] for r in selected),
            "activity_settled": sum(r["activity_settled"] for r in selected),
            "truncated_not_settled": sum(not r["activity_settled"] for r in selected),
            "mean_parameter_l2_drift": float(np.mean([r["parameter_l2_drift"] for r in selected])),
            "queries_with_classification_drop": sum(r["classification_after_probe"] < r["classification_before_probe"] for r in selected),
            "max_activity_residual": max(r["activity_residual"] for r in selected)}
    meter.add("completion_queries", len(rows))
    return {"summary": summary, "rows": rows}


def benchmark(p, meter):
    receipts, science = {}, {}
    mask = np.ones(4, dtype=bool)
    for method in ("direct", "settled"):
        durations, outputs = [], []
        for repeat in range(4):
            wall, cpu = time.perf_counter(), time.process_time()
            batch = []
            for x in ALL:
                expected = forward(x, p, meter)[0]
                if method == "direct":
                    batch.append({"iterations": 0, "settled": True, "max_output_disagreement": 0.})
                    continue
                state = {"x": x.copy(), "z": np.full((4,4), .5),
                         "h": np.full(8, .5), "y": np.zeros(4)}
                settled = False
                for iteration in range(500):
                    state, _ = joint_step(state, p, mask, x, None, meter, eta=0.)
                    _, sr, _ = residuals(state, p, mask, x, None, meter)
                    if sr <= 1e-5:
                        settled = True
                        break
                batch.append({"iterations": iteration+1, "settled": settled,
                              "max_output_disagreement": float(np.max(np.abs(state["y"]-expected)))})
            if repeat:
                durations.append({"wall_seconds": time.perf_counter()-wall,
                                  "process_cpu_seconds": time.process_time()-cpu})
            outputs.append(batch)
        assert all(batch == outputs[0] for batch in outputs)
        science[method] = outputs[0]
        receipts[method] = {"batch_size": 16, "warmup_batches": 1, "measured_batches": 3,
                            "samples": durations,
                            "summary": {k: {"median": statistics.median(d[k] for d in durations),
                                            "min": min(d[k] for d in durations), "max": max(d[k] for d in durations)}
                                        for k in durations[0]},
                            "amortized_wall_seconds_per_query": statistics.median(d["wall_seconds"] for d in durations)/16}
    return science, receipts


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite existing receipts.")
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (CONTRACT, Path(__file__).resolve())},
        "command": ".venv/bin/python -m experiments.continuous_local_learning",
        "seed_policy": {"root": ROOT_SEED, "replicas": [0,1,2], "parameter_stream": 0, "schedule_stream": 1},
    }
    wall0, cpu0 = time.perf_counter(), time.process_time()
    phases, work = {}, {}

    def phase(name, function):
        meter = Meter()
        wall, cpu = time.perf_counter(), time.process_time()
        value = function(meter)
        phases[name] = {"wall_seconds": time.perf_counter()-wall,
                        "process_cpu_seconds": time.process_time()-cpu}
        work[name] = meter.counts
        return value

    checks = phase("numerical_checks", check_gradients)
    print(json.dumps({"phase": "numerical_checks", "result": checks}), flush=True)
    runs, benchmark_timings = [], {}
    for replica in range(3):
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, replica, 1]))
        stream_A = [TRAIN[j].copy() for _ in range(200) for j in rng.permutation(8)]
        stream_B = [NEW[j % 2].copy() for j in range(100)]
        for arm in ("joint_continuous", "direct_gradient"):
            prefix = f"replica_{replica}/{arm}"
            p = initialize(replica)
            state = {"x": np.full(4,.5), "z": np.full((4,4),.5), "h": np.full(8,.5), "y": np.zeros(4)}
            run = {"replica": replica, "arm": arm, "checkpoints": {}}
            run["checkpoints"]["initial"] = phase(prefix+"/initial_classification", lambda m: classification(p,m))
            for block, stream in (("A", stream_A), ("B", stream_B)):
                def train(meter):
                    nonlocal p, state
                    for x in stream:
                        meter.add("presentations")
                        target = onehot(x)
                        state["x"], state["y"] = x.copy(), target.copy()
                        for _ in range(40):
                            meter.add("parameter_update_attempts")
                            if arm == "joint_continuous":
                                state, p = joint_step(state, p, np.ones(4,dtype=bool), x, target, meter)
                            else:
                                p = direct_step(x, target, p, meter)
                    return {"parameters": {k:v.tolist() for k,v in p.items()}}
                parameter_record = phase(prefix+f"/block_{block}_learning", train)
                report = phase(prefix+f"/post_{block}_classification", lambda m: classification(p,m))
                report.update(parameter_record)
                print(json.dumps({"replica": replica, "arm": arm, "checkpoint": block,
                                  "correct": report["scores"]["all"]["correct"]}), flush=True)
                report["completion"] = phase(prefix+f"/post_{block}_completion", lambda m: completion_battery(p,m))
                if replica == 0 and block == "A":
                    bench_science, bench_runtime = phase(prefix+"/inference_benchmark", lambda m: benchmark(p,m))
                    report["inference_benchmark"] = bench_science
                    benchmark_timings[arm] = bench_runtime
                run["checkpoints"]["post_"+block] = report
            runs.append(run)
    updates = sum(v.get("parameter_update_attempts",0) for v in work.values())
    presentations = sum(v.get("presentations",0) for v in work.values())
    probes = sum(v.get("completion_queries",0) for v in work.values())
    assert updates == 408000 and presentations == 10200 and probes == 1152
    science = {"scope": "Supervised simultaneous adaptation on known four-bit fixture; no general-learning claim.",
               "config": {"root_seed": ROOT_SEED, "state_step": DT, "parameter_step": ETA,
                          "weight_decay": DECAY, "parameter_count": 88, "parameter_bytes_float64": 704,
                          "activity_count": 32, "activity_bytes_float64": 256},
               "numerical_checks": checks, "runs": runs, "phase_work_counts": work,
               "correctness_checks_passed": True}
    serial_wall, serial_cpu = time.perf_counter(), time.process_time()
    RESULT.write_text(json.dumps(science, sort_keys=True, indent=2)+"\n")
    phases["scientific_serialization"] = {"wall_seconds": time.perf_counter()-serial_wall,
                                          "process_cpu_seconds": time.process_time()-serial_cpu}
    elapsed = {"wall_seconds": time.perf_counter()-wall0, "process_cpu_seconds": time.process_time()-cpu0}
    cpuinfo = Path("/proc/cpuinfo")
    cpu_name = next((line.split(":",1)[1].strip() for line in cpuinfo.read_text().splitlines()
                     if line.startswith("model name")), None) if cpuinfo.exists() else None
    receipt = {
        "provenance": provenance,
        "environment": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                        "numpy_config": np.show_config(mode="dicts"), "platform": platform.platform(), "cpu": cpu_name,
                        "threads": {v:os.environ[v] for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                        "logical_cpus_visible": os.cpu_count(), "processes": 1, "workers": 0,
                        "device": "CPU", "dtype": "float64", "hardware_limits": None,
                        "hardware_limits_reason": "No quota inventory; no scaling claim."},
        "whole_run": elapsed, "phases": phases, "inference_benchmarks": benchmark_timings,
        "timing_scope": "After imports/provenance through scientific serialization; excludes runtime receipt writing.",
        "phase_work_counts": work,
        "work_summary": {"trajectories": 6, "training_update_attempts": updates, "presentations": presentations,
                         "distinct_labeled_patterns": 10, "completion_queries": probes,
                         "completion_steps": 345600, "selection_candidates": 6,
                         "hyperparameter_tuning": 0, "best_seed_selection": False},
        "memory": {"peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "rss_unit": "KiB" if sys.platform.startswith("linux") else "platform dependent",
                   "rss_scope": "Parent cumulative high-water including checkpoints, diagnostics and serialization.",
                   "learned_parameter_bytes_per_model": 704, "activity_bytes_per_model": 256,
                   "optimizer_state_bytes": 0, "replay_bytes": 0,
                   "saved_A_B_parameter_array_bytes": 6*2*704,
                   "shadow_query_parameter_copy_bytes": 704,
                   "temporary_workspace_bytes": None,
                   "temporary_workspace_reason": "Gradient/candidate arrays allocated; allocator workspace not separately instrumented.",
                   "scientific_artifact_bytes": RESULT.stat().st_size,
                   "accelerator_bytes": None, "accelerator_reason": "Not used."},
        "manual_development_cost": None, "manual_development_cost_reason": "Not measured.",
        "scientific_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    }
    RUNTIME.write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"complete": True, "whole_run": elapsed, "updates": updates, "completion_queries": probes}), flush=True)


if __name__ == "__main__":
    main()
