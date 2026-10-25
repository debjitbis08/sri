"""Finite local circuit, supervised detector identification, and exact return.

Run: python3 -m experiments.local_relation_circuit
Contract: docs/local-relation-circuit-v1-contract.md
"""

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/local-relation-circuit-v1-contract.md"
RESULT = ROOT / "reports/local-relation-circuit-v1-results.json"
RUNTIME = ROOT / "reports/local-relation-circuit-v1-runtime.json"
TRAIN = [((0, 0, 0, 0), 0), ((1, 1, 1, 1), 0),
         ((0, 0, 0, 1), 1), ((1, 1, 1, 0), 1),
         ((0, 0, 1, 1), 2), ((1, 1, 0, 0), 2),
         ((0, 1, 0, 1), 3), ((1, 0, 1, 0), 3)]
CONTROL = {"codes": (6, 8), "decoder": {(0, 0): 0, (0, 4): 0,
           (2, 0): 1, (2, 2): 1, (2, 1): 2, (4, 0): 3}}


class Meter:
    def __init__(self):
        self.counts = {}

    def add(self, name, value=1):
        self.counts[name] = self.counts.get(name, 0) + value

    def peak(self, name, value):
        self.counts[name] = max(self.counts.get(name, 0), value)


def object_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(object_bytes(k, seen)+object_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(object_bytes(v, seen) for v in value)
    return size


def serialized_model(model):
    if model is None:
        return None
    return {"codes": model["codes"],
            "truth_tables_00_01_10_11": [[(code >> k) & 1 for k in range(4)] for code in model["codes"]],
            "decoder": [{"counts": key, "class": value} for key, value in sorted(model["decoder"].items())]}


def target_class(b):
    """Direct analytic reference; not a stored 16-pattern label table."""
    transitions = sum(b[i] != b[(i+1) % len(b)] for i in range(len(b)))
    if transitions == 0:
        return 0
    if transitions == 4:
        return 3
    return 1 if sum(b) % 2 else 2


def gates(a, b, codes, meter):
    meter.add("local_gate_evaluations", len(codes))
    return tuple((code >> (2*a+b)) & 1 for code in codes)


def features(b, codes, meter):
    meter.add("feature_calls")
    result = [0]*len(codes)
    for i, a in enumerate(b):
        for j, value in enumerate(gates(a, b[(i+1) % len(b)], codes, meter)):
            result[j] += value
    return tuple(result)


def predict(b, model, meter):
    return model["decoder"].get(features(b, model["codes"], meter))


def local_return(s, c0, evidence, model, meter):
    """Counter-state chain DP; no complete outer-pattern candidate list."""
    meter.add("local_inference_calls")
    n, k = len(s), len(model["codes"])
    best = None
    for root in (0, 1):
        meter.add("root_candidates")
        if evidence[0] is not None and evidence[0] != root:
            continue
        key = (root,) + (0,)*k
        layers = [{key: (int(root != s[0]), None)}]
        meter.add("profile_entries", 1)
        for pos in range(1, n):
            previous = layers[-1]
            current = {}
            for prevkey in sorted(previous):
                cost, _ = previous[prevkey]
                for b in (0, 1):
                    meter.add("prefix_candidates")
                    if evidence[pos] is not None and evidence[pos] != b:
                        continue
                    delta = gates(prevkey[0], b, model["codes"], meter)
                    newkey = (b,) + tuple(a+d for a, d in zip(prevkey[1:], delta))
                    newcost = cost + int(b != s[pos])
                    if newkey not in current or newcost < current[newkey][0]:
                        current[newkey] = (newcost, prevkey)
            meter.add("profile_entries", len(current))
            meter.peak("peak_frontier_entries", len(previous)+len(current))
            layers.append(current)
        meter.peak("peak_trace_entries", sum(map(len, layers)))
        for key in sorted(layers[-1]):
            meter.add("terminal_candidates")
            delta = gates(key[0], root, model["codes"], meter)
            totals = tuple(a+d for a, d in zip(key[1:], delta))
            c = model["decoder"].get(totals)
            if c is None:
                return {"incomplete_decoder": True, "unknown_counts": totals}
            score = layers[-1][key][0] + 5*(c != c0)
            if best is None or score < best[0]:
                b = [None]*n
                tracekey = key
                for pos in range(n-1, -1, -1):
                    meter.add("traceback_nodes")
                    b[pos] = tracekey[0]
                    tracekey = layers[pos][tracekey][1]
                best = (score, tuple(b), c)
    assert best is not None
    return {"energy": best[0], "outer": best[1], "inner": best[2], "incomplete_decoder": False}


def exhaustive_return(s, c0, evidence, meter, model=None):
    meter.add("reference_inference_calls")
    best, winners = math.inf, set()
    for b in itertools.product((0, 1), repeat=len(s)):
        meter.add("reference_candidate_visits")
        if not all(e is None or b[i] == e for i, e in enumerate(evidence)):
            continue
        meter.add("reference_feasible_scores")
        c = target_class(b) if model is None else predict(b, model, meter)
        assert c is not None
        score = sum(x != y for x, y in zip(b, s)) + 5*(c != c0)
        if score < best:
            best, winners = score, {(b, c)}
        elif score == best:
            winners.add((b, c))
    return best, winners


def learn(width, meter):
    audit, survivors = [], []
    for codes in itertools.product(range(16), repeat=width):
        meter.add("learning_candidates")
        decoder, conflicts = {}, 0
        for b, label in TRAIN:
            meter.add("training_exposures")
            key = features(b, codes, meter)
            if key in decoder and decoder[key] != label:
                conflicts += 1
            else:
                decoder[key] = label
        audit.append({"codes": codes, "conflicts": conflicts, "decoder_entries": len(decoder)})
        if not conflicts:
            survivors.append({"codes": codes, "decoder": decoder})
    return audit, survivors


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite existing receipts.")
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (CONTRACT, Path(__file__).resolve())},
        "command": "python3 -m experiments.local_relation_circuit",
        "seed_policy": "No randomness; declared finite candidate order and split.",
    }
    whole_wall, whole_cpu = time.perf_counter(), time.process_time()
    phases, meters = {}, {}

    def phase(name, function):
        meter = Meter()
        wall, cpu = time.perf_counter(), time.process_time()
        value = function(meter)
        phases[name] = {"wall_seconds": time.perf_counter()-wall,
                        "process_cpu_seconds": time.process_time()-cpu}
        meters[name] = meter.counts
        return value

    states = list(itertools.product((0, 1), repeat=4))
    evidences = list(itertools.product((None, 0, 1), repeat=4))

    def validate(model, meter, learned=False):
        rows, target_matches, own_matches, ties = [], 0, 0, 0
        for s, c0, evidence in itertools.product(states, range(4), evidences):
            target_score, target_set = exhaustive_return(s, c0, evidence, meter)
            own_score, own_set = exhaustive_return(s, c0, evidence, meter, model) if learned else (target_score, target_set)
            result = local_return(s, c0, evidence, model, meter)
            assert not result["incomplete_decoder"]
            assert result["energy"] == own_score and (result["outer"], result["inner"]) in own_set
            assert all(e is None or result["outer"][i] == e for i, e in enumerate(evidence))
            own_matches += 1
            match = result["energy"] == target_score and (result["outer"], result["inner"]) in target_set
            target_matches += match
            ties += len(own_set) > 1
            rows.append({"outer_default": s, "inner_default": c0, "evidence": evidence,
                         "decoded": result, "reference_energy": target_score,
                         "target_match": match, "own_optimum_count": len(own_set)})
        return {"cases": len(rows), "own_reference_matches": own_matches,
                "target_reference_matches": target_matches, "tied_cases": ties, "rows": rows}

    def supplied(meter):
        assert all(predict(b, CONTROL, meter) == target_class(b) for b in states)
        return validate(CONTROL, meter)

    control_result = phase("supplied_circuit_and_return_validation", supplied)
    one_audit, one_survivors = phase("one_detector_training", lambda m: learn(1, m))
    two_audit, two_survivors = phase("two_detector_training", lambda m: learn(2, m))
    # Selection occurs before any held-out evaluation.
    selected = two_survivors[0] if two_survivors else None
    selected_one = one_survivors[0] if one_survivors else None

    def evaluate(meter):
        heldout = [b for b in states if b not in {b for b, _ in TRAIN}]
        diagnostics = []
        for family, survivors in ((1, one_survivors), (2, two_survivors)):
            rows = []
            for model in survivors:
                predictions = {b: predict(b, model, meter) for b in states}
                rows.append({"codes": model["codes"],
                             "all_correct": sum(predictions[b] == target_class(b) for b in states),
                             "heldout_correct": sum(predictions[b] == target_class(b) for b in heldout),
                             "unknown_patterns": [b for b in states if predictions[b] is None],
                             "predictions": [{"outer": b, "predicted": predictions[b], "target": target_class(b)} for b in states]})
            diagnostics.append({"detectors": family, "consistent_candidates": len(survivors), "models": rows})
        return {"heldout": heldout, "families": diagnostics}

    evaluation = phase("post_selection_full_version_space_evaluation", evaluate)
    complete = selected is not None and not evaluation["families"][1]["models"][0]["unknown_patterns"]
    learned_result = phase("learned_circuit_return_validation", lambda m: validate(selected, m, True)) if complete else None

    benchmark_queries = [((0, 0, 1, 1), 2, e) for e in evidences]
    benchmark_functions = {
        "direct_enumeration": lambda s, c, e, m: exhaustive_return(s, c, e, m),
        "supplied_local": lambda s, c, e, m: local_return(s, c, e, CONTROL, m),
    }
    if complete:
        benchmark_functions["learned_local"] = lambda s, c, e, m: local_return(s, c, e, selected, m)
    benchmark_receipts = {}

    def benchmark(meter):
        for name, fn in benchmark_functions.items():
            for args in benchmark_queries:
                fn(*args, meter)
            samples = []
            for _ in range(7):
                wall, cpu = time.perf_counter(), time.process_time()
                for args in benchmark_queries:
                    fn(*args, meter)
                samples.append({"wall_seconds": time.perf_counter()-wall,
                                "process_cpu_seconds": time.process_time()-cpu})
            benchmark_receipts[name] = {"batch_size": 81, "warmup_batches": 1,
                "measured_batches": 7, "samples": samples,
                "batch_summary": {key: {"median": statistics.median(s[key] for s in samples),
                                        "min": min(s[key] for s in samples),
                                        "max": max(s[key] for s in samples)} for key in samples[0]},
                "amortized_wall_seconds_per_query": statistics.median(s["wall_seconds"] for s in samples)/81}

    phase("matched_inference_benchmarks", benchmark)
    assert meters["one_detector_training"]["training_exposures"] == 128
    assert meters["two_detector_training"]["training_exposures"] == 2048
    assert meters["one_detector_training"]["local_gate_evaluations"] == 512
    assert meters["two_detector_training"]["local_gate_evaluations"] == 16384
    hand_examples = {}
    for name, result in (("supplied", control_result), ("learned", learned_result)):
        if result is None:
            continue
        examples = []
        for evidence in ((None, None, None, None), (0, 0, None, None),
                         (1, 1, None, None), (0, 1, 0, 1)):
            row = next(r for r in result["rows"] if r["outer_default"] == (0, 0, 1, 1)
                       and r["inner_default"] == 2 and r["evidence"] == evidence)
            if name == "supplied":
                expected = {(None, None, None, None): ((0, 0, 1, 1), 2),
                            (0, 0, None, None): ((0, 0, 1, 1), 2),
                            (1, 1, None, None): ((1, 1, 0, 0), 2),
                            (0, 1, 0, 1): ((0, 1, 0, 1), 3)}[evidence]
                assert (row["decoded"]["outer"], row["decoded"]["inner"]) == expected
            examples.append(row)
        hand_examples[name] = examples
    science = {
        "scope": "Local deployed detectors and scheduled exact return; supervised finite identification within a rotation-invariant scaffold.",
        "training": [{"outer": b, "class": c} for b, c in TRAIN],
        "supplied_model": serialized_model(CONTROL),
        "learning": {"one_detector_audit": one_audit, "two_detector_audit": two_audit,
                     "selected_one": serialized_model(selected_one), "selected_two": serialized_model(selected),
                     "selected_two_total": complete, "evaluation": evaluation},
        "supplied_validation": control_result, "learned_validation": learned_result,
        "hand_examples": hand_examples, "phase_work_counts": meters,
        "correctness_checks_passed": True,
    }
    serial_wall, serial_cpu = time.perf_counter(), time.process_time()
    RESULT.write_text(json.dumps(science, sort_keys=True, indent=2) + "\n")
    phases["scientific_serialization"] = {"wall_seconds": time.perf_counter()-serial_wall,
                                          "process_cpu_seconds": time.process_time()-serial_cpu}
    elapsed = {"wall_seconds": time.perf_counter()-whole_wall,
               "process_cpu_seconds": time.process_time()-whole_cpu}
    cpuinfo = Path("/proc/cpuinfo")
    cpu_name = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                     if line.startswith("model name")), None) if cpuinfo.exists() else None
    runtime = {
        "provenance": provenance,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "cpu": cpu_name,
                        "logical_cpus_visible": os.cpu_count(),
                        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "processes": 1, "workers": 0, "computation_threads": 1,
                        "libraries": "Python standard library", "dtype": "Exact integer costs/gates; floating timers only",
                        "blas": "not used", "device": "CPU", "hardware_limits": None,
                        "hardware_limits_reason": "No quota inventory; no scaling benchmark."},
        "phases": phases, "whole_run": elapsed, "inference_benchmarks": benchmark_receipts,
        "timing_scope": "After imports/provenance through scientific serialization; excludes runtime receipt writing. Benchmarks include counters, allocation and traceback.",
        "memory": {"peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "rss_unit": "KiB" if sys.platform.startswith("linux") else "platform dependent",
                   "rss_scope": "Parent cumulative high-water, including all candidate audits/reference results and output serialization.",
                   "selected_model_python_object_bytes": object_bytes(selected) if selected else None,
                   "selected_model_json_utf8_bytes": len(json.dumps(serialized_model(selected), sort_keys=True).encode()) if selected else None,
                   "selected_model_absence_reason": None if selected else "No consistent two-detector candidate",
                   "scientific_artifact_bytes": RESULT.stat().st_size,
                   "optimizer_bytes": 0, "inference_replay_bytes": 0, "inference_retained_raw_training_examples": 0,
                   "temporary_workspace_bytes": None,
                   "temporary_workspace_reason": "Entry counts reported; no separate allocator-level workspace measurement.",
                   "accelerator_bytes": None, "accelerator_reason": "Not used."},
        "phase_work_counts": meters,
        "learning_scope": {"independent_labeled_patterns": 8, "heldout_patterns": 8,
                           "candidates_scored": 272, "training_exposures": 2176,
                           "optimizer_steps": 0, "selection": "First training-consistent tuple; no test-based selection"},
        "manual_development_cost": None, "manual_development_cost_reason": "Not measured.",
        "scientific_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    }
    RUNTIME.write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"correctness_passed": True, "one_detector_survivors": len(one_survivors),
                      "two_detector_survivors": len(two_survivors), "selected": serialized_model(selected),
                      "learned_return_target_matches": learned_result["target_reference_matches"] if learned_result else None,
                      "benchmark_amortized_seconds": {k: v["amortized_wall_seconds_per_query"] for k, v in benchmark_receipts.items()},
                      "whole_run": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
