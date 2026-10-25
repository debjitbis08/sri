"""Exhaustive proof illustration; no fitting or random search.

Run: python3 -m experiments.combination_interpretation_analysis
Contract: docs/combination-interpretation-analysis.md, section 8.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/combination-interpretation-analysis.md"
RESULT = ROOT / "reports/combination-interpretation-analysis.json"
RUNTIME = ROOT / "reports/combination-interpretation-analysis-runtime.json"
CONFIG = {"outer_default": [0, 1], "inner_default": 1,
          "compatibility_cost": 4, "inner_default_cost": 2,
          "outer_default_cost": 1}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def object_bytes(value, seen=None):
    """Recursive CPython object footprint, counting shared objects once."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(object_bytes(k, seen) + object_bytes(v, seen)
                     for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(object_bytes(v, seen) for v in value)
    return total


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite existing analytical receipts.")
    provenance = {
        "revision": git("rev-parse", "HEAD"),
        "dirty_status": git("status", "--porcelain"),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (CONTRACT, Path(__file__).resolve())},
        "command": "python3 -m experiments.combination_interpretation_analysis",
        "argv": sys.argv,
        "seed_policy": "No randomness; complete finite enumeration.",
    }
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    counts = {"candidate_states_visited": 0, "feasible_states_scored": 0,
              "energy_evaluations": 0, "local_proposals": 0,
              "accepted_local_updates": 0, "local_sweeps": 0}

    def energy(state):
        counts["energy_evaluations"] += 1
        x1, x2, h = state
        return (CONFIG["compatibility_cost"] * (h != (x1 ^ x2))
                + CONFIG["inner_default_cost"] * (h != CONFIG["inner_default"])
                + CONFIG["outer_default_cost"] * sum(
                    x != s for x, s in zip((x1, x2), CONFIG["outer_default"])))

    states = list(itertools.product((0, 1), repeat=3))
    expected = {
        (None, None): ((0, 1, 1), 0),
        (0, None): ((0, 1, 1), 0),
        (1, None): ((1, 0, 1), 2),
        (None, 0): ((1, 0, 1), 2),
        (None, 1): ((0, 1, 1), 0),
        (0, 0): ((0, 0, 0), 3),
        (0, 1): ((0, 1, 1), 0),
        (1, 0): ((1, 0, 1), 2),
        (1, 1): ((1, 1, 0), 3),
    }
    rows = []
    for evidence in itertools.product((None, 0, 1), repeat=2):
        scored = []
        for state in states:
            counts["candidate_states_visited"] += 1
            if all(e is None or e == state[k] for k, e in enumerate(evidence)):
                counts["feasible_states_scored"] += 1
                scored.append((energy(state), state))
        best = min(e for e, _ in scored)
        winners = [s for e, s in scored if e == best]
        assert len(winners) == 1, (evidence, winners)
        assert (winners[0], best) == expected[evidence]
        rows.append({"evidence": evidence, "state": winners[0], "energy": best,
                     "feasible_count": len(scored), "unique": True})

    def descend(order):
        state = (1, 1, 1)  # x1 clamped to observation; x2,h at defaults.
        score = energy(state)
        trace = [{"state": state, "energy": score}]
        # At most 8 states; strict decreases forbid revisiting any state.
        for _ in range(8):
            counts["local_sweeps"] += 1
            changed = False
            for k in order:
                candidate = list(state)
                candidate[k] ^= 1
                proposal = tuple(candidate)
                counts["local_proposals"] += 1
                trial = energy(proposal)
                if trial < score:
                    assert proposal not in [r["state"] for r in trace]
                    state, score = proposal, trial
                    changed = True
                    counts["accepted_local_updates"] += 1
                    trace.append({"state": state, "energy": score})
            if not changed:
                return trace
        raise AssertionError("Finite strict descent did not terminate")

    h_first = descend((2, 1))
    x2_first = descend((1, 2))
    assert h_first[-1] == {"state": (1, 1, 0), "energy": 3}
    assert x2_first[-1] == {"state": (1, 0, 1), "energy": 2}
    assert energy((1, 0, 0)) == 8  # The trap's other allowed neighbor.
    combinations = [{"message": u, "stored": s, "output": u ^ s}
                    for u, s in itertools.product((0, 1), repeat=2)]
    assert all(row["output"] == row["stored"]
               for row in combinations if row["message"] == 0)
    roundtrips = []
    for q, s in itertools.product((0, 1), repeat=2):
        message = q ^ s
        assert (message ^ s) == q
        roundtrips.append({"descriptor": q, "stored": s, "message": message,
                           "recovered": message ^ s})
    truth = [a ^ b for a, b in itertools.product((0, 1), repeat=2)]
    assert truth[3] != truth[1] + truth[2] - truth[0]
    assert counts["candidate_states_visited"] == 72
    assert counts["feasible_states_scored"] == 32

    science = {
        "scope": "Constructed finite proof illustration; not a learned model or capacity witness.",
        "config": CONFIG, "evidence_results": rows,
        "combination_table": combinations, "message_roundtrips": roundtrips,
        "complete_data_inner_truth_table": truth,
        "local_descent": {"inner_first": h_first, "outer_first": x2_first},
        "work_counts": counts,
        "all_checks_passed": True,
    }
    RESULT.write_text(json.dumps(science, indent=2, sort_keys=True) + "\n")
    elapsed = {"wall_seconds": time.perf_counter() - start_wall,
               "process_cpu_seconds": time.process_time() - start_cpu}
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), None) if cpuinfo.exists() else None
    receipt = {
        "provenance": provenance,
        "environment": {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "cpu": cpu,
            "logical_cpus_visible": os.cpu_count(),
            "cpu_affinity_count": len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else None,
            "libraries": "Python standard library only", "dtype": "Python exact integers/bools",
            "blas": "not used", "device": "CPU", "processes": 1,
            "workers": 0, "computation_threads": 1,
            "hardware_limits": None,
            "hardware_limits_reason": "No cgroup/resource quota inventory; no scaling claim.",
        },
        "timing": {"whole_check_and_scientific_serialization": elapsed},
        "timing_scope": "After imports/provenance; before environment inventory/runtime receipt. One run, not latency benchmark.",
        "memory": {
            "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_unit": "KiB" if sys.platform.startswith("linux") else "platform dependent",
            "rss_scope": "Single parent process cumulative high-water, including interpreter/imports; no worker processes.",
            "configuration_python_object_bytes": object_bytes(CONFIG),
            "configuration_json_utf8_bytes": len(json.dumps(CONFIG, sort_keys=True).encode()),
            "scientific_artifact_bytes": RESULT.stat().st_size,
            "temporary_workspace_bytes": None,
            "temporary_workspace_reason": "Python allocator not instrumented separately from process RSS.",
            "learned_state_bytes": 0, "optimizer_bytes": 0, "replay_bytes": 0,
            "accelerator_bytes": None, "accelerator_reason": "No accelerator used.",
        },
        "work_counts": {**counts, "fits": 0, "tuning_candidates": 0,
                        "independent_empirical_examples": 0, "analytic_evidence_patterns": 9,
                        "optimizer_steps": 0, "joint_inference_calls": 9,
                        "local_inference_calls": 2, "convergence_failures": 0,
                        "local_suboptimal_endpoints": 1},
        "selection_cost_scope": "All 72 joint candidate visits and both local schedules included; no tuning or learning.",
        "development_attempts": {
            "missing_python_alias": 1,
            "failed_assertion_runs": 1,
            "reason": "A hand-calculated neighbor energy was written as 6 instead of 8; corrected assertion and prose, no model change.",
            "wall_seconds": None, "cpu_seconds": None,
            "measurement_reason": "Failed development attempt did not serialize its timers; not included in successful-run timing.",
        },
        "scientific_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    }
    RUNTIME.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_passed": True, "work_counts": counts,
                      "measurement": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
