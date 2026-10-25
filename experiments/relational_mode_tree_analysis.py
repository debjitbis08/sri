"""Exact finite checks for docs/relational-mode-tree-analysis.md.

Run: python3 -m experiments.relational_mode_tree_analysis
"""

from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/relational-mode-tree-analysis.md"
RESULT = ROOT / "reports/relational-mode-tree-analysis.json"
RUNTIME = ROOT / "reports/relational-mode-tree-analysis-runtime.json"
CONFIGS = [
    {"defaults": [0, 1, 1], "leaves": 2, "children": {2: (0, 1)}},
    {"defaults": [0, 1, 1, 0, 1, 1, 0], "leaves": 4,
     "children": {4: (0, 1), 5: (2, 3), 6: (4, 5)}},
]


def object_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(object_bytes(k, seen) + object_bytes(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(object_bytes(v, seen) for v in value)
    return size


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def determinant3(a):
    return (a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
            - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
            + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]))


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite existing analytical receipts.")
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (CONTRACT, Path(__file__).resolve())},
        "command": "python3 -m experiments.relational_mode_tree_analysis",
        "seed_policy": "None: exhaustive integer/rational verification.",
    }
    wall, cpu = time.perf_counter(), time.process_time()
    spectral = []
    for r, tau in itertools.product((-1, 1), (0, 1, 2)):
        low, high = [1, r], [1, -r]
        lap = [[1, -r], [-r, 1]]
        smooth = [[Q(1+tau, 1+2*tau), Q(tau*r, 1+2*tau)],
                  [Q(tau*r, 1+2*tau), Q(1+tau, 1+2*tau)]]
        assert matvec(lap, low) == [0, 0]
        assert matvec(lap, high) == [2*v for v in high]
        assert matvec(smooth, low) == low
        assert matvec(smooth, high) == [Q(v, 1+2*tau) for v in high]
        system = [[int(i == j) + tau*lap[i][j] for j in range(2)] for i in range(2)]
        for j in range(2):
            assert matvec(system, [smooth[i][j] for i in range(2)]) == [int(i == j) for i in range(2)]
        spectral.append({"relation_sign": r, "tau": tau,
                         "high_mode_gain": str(Q(1, 1+2*tau))})
    for a, b, h in itertools.product((0, 1), repeat=3):
        x1, x2, r = (-1)**a, (-1)**b, (-1)**h
        assert 4 * (h != (a ^ b)) == (x1-r*x2)**2
    positive = [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
    frustrated = [[2, 1, -1], [1, 2, -1], [-1, -1, 2]]
    assert determinant3(positive) == 0
    assert determinant3(frustrated) == 4
    # Characteristic roots checked by exact substitution, with trace accounting.
    for matrix, roots in ((positive, (0, 3, 3)), (frustrated, (1, 1, 4))):
        for eigenvalue in set(roots):
            shifted = [[matrix[i][j] - eigenvalue*int(i == j) for j in range(3)] for i in range(3)]
            assert determinant3(shifted) == 0
        assert sum(roots) == sum(matrix[i][i] for i in range(3))

    counts = {"exhaustive_candidate_visits": 0, "exhaustive_feasible_scores": 0,
              "dp_factor_candidates": 0, "dp_profile_entries": 0,
              "traceback_nodes": 0, "inference_cases": 0,
              "decoded_energy_checks": 0}

    def energy(state, config):
        leaves = config["leaves"]
        unary = sum((1 if v < leaves else 2) * (z != config["defaults"][v])
                    for v, z in enumerate(state))
        return unary + sum(4*(state[v] != (state[a] ^ state[b]))
                           for v, (a, b) in config["children"].items())

    def solve(config, evidence):
        profiles, choices = {}, {}
        for v, default in enumerate(config["defaults"]):
            profiles[v] = []
            for h in (0, 1):
                counts["dp_profile_entries"] += 1
                if v < config["leaves"]:
                    value = int(h != default) if evidence[v] in (None, h) else math.inf
                else:
                    l, r = config["children"][v]
                    candidates = []
                    for a, b in itertools.product((0, 1), repeat=2):
                        counts["dp_factor_candidates"] += 1
                        score = 4*(h != (a ^ b)) + profiles[l][a] + profiles[r][b]
                        candidates.append((score, a, b))
                    best, a, b = min(candidates)
                    choices[v, h] = (a, b)
                    value = best + 2*(h != default)
                profiles[v].append(value)
        root = len(config["defaults"])-1
        score, h = min((profiles[root][h], h) for h in (0, 1))
        assert math.isfinite(score)
        assignment = [None]*len(config["defaults"])

        def decode(v, z):
            counts["traceback_nodes"] += 1
            assignment[v] = z
            if v in config["children"]:
                l, r = config["children"][v]
                a, b = choices[v, z]
                decode(l, a)
                decode(r, b)

        decode(root, h)
        return score, tuple(assignment)

    trees = []
    for config in CONFIGS:
        rows = []
        n, leaves = len(config["defaults"]), config["leaves"]
        for evidence in itertools.product((None, 0, 1), repeat=leaves):
            counts["inference_cases"] += 1
            best, winners = math.inf, []
            for state in itertools.product((0, 1), repeat=n):
                counts["exhaustive_candidate_visits"] += 1
                if not all(e is None or state[v] == e for v, e in enumerate(evidence)):
                    continue
                counts["exhaustive_feasible_scores"] += 1
                score = energy(state, config)
                if score < best:
                    best, winners = score, [state]
                elif score == best:
                    winners.append(state)
            score, decoded = solve(config, evidence)
            assert score == best and decoded in winners, (evidence, score, best, decoded)
            counts["decoded_energy_checks"] += 1
            assert energy(decoded, config) == best
            assert all(e is None or decoded[v] == e for v, e in enumerate(evidence))
            if all(e is None for e in evidence):
                assert decoded == tuple(config["defaults"]) and len(winners) == 1
            rows.append({"evidence": evidence, "decoded": decoded, "energy": score,
                         "optimum_count": len(winners)})
        trees.append({"nodes": n, "leaves": leaves, "cases": rows,
                      "ambiguous_optima_cases": sum(row["optimum_count"] > 1 for row in rows)})
    assert counts["exhaustive_candidate_visits"] == 10440
    assert counts["dp_factor_candidates"] == 2016
    trap_case = next(row for row in trees[0]["cases"] if row["evidence"] == (1, None))
    assert trap_case["decoded"] == (1, 0, 1) and trap_case["energy"] == 2
    science = {"scope": "Supplied relation geometry and exact tree inference, not learned capacity.",
               "configs": CONFIGS, "cost_coefficients": {"compatibility": 4, "inner_prior": 2, "leaf_prior": 1},
               "spectral_cases": spectral, "triangle_eigenvalues": {"positive": [0, 3, 3], "frustrated": [1, 1, 4]},
               "trees": trees, "work_counts": counts, "all_checks_passed": True}
    RESULT.write_text(json.dumps(science, sort_keys=True, indent=2) + "\n")
    elapsed = {"wall_seconds": time.perf_counter()-wall, "process_cpu_seconds": time.process_time()-cpu}
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
                        "libraries": "Python standard library only", "dtype": "Exact Python integers/Fraction; infinity sentinel",
                        "blas": "not used", "device": "CPU", "hardware_limits": None,
                        "hardware_limits_reason": "Quota inventory not collected; no scaling benchmark."},
        "timing": {"whole_verification_and_scientific_serialization": elapsed},
        "timing_scope": "After provenance through scientific serialization; excludes startup and runtime receipt; one run, not latency benchmark.",
        "memory": {"peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "rss_unit": "KiB" if sys.platform.startswith("linux") else "platform dependent",
                   "rss_scope": "Single parent cumulative high-water; includes exhaustive reference and interpreter.",
                   "config_python_object_bytes": object_bytes(CONFIGS),
                   "config_json_utf8_bytes": len(json.dumps(CONFIGS, sort_keys=True).encode()),
                   "scientific_artifact_bytes": RESULT.stat().st_size,
                   "learned_state_bytes": 0, "optimizer_bytes": 0, "replay_bytes": 0,
                   "temporary_workspace_bytes": None,
                   "temporary_workspace_reason": "No separate allocator instrumentation; see RSS.",
                   "accelerator_bytes": None, "accelerator_reason": "Not used."},
        "work_counts": {**counts, "fits": 0, "tuning_candidates": 0,
                        "empirical_examples": 0, "optimizer_steps": 0,
                        "spectral_cases": 6, "compatibility_identities": 8,
                        "triangle_checks": 2, "solver_failures": 0},
        "selection_scope": "All DP alternatives and exhaustive candidates included; two prescribed structures, no tuning.",
        "manual_development_cost": None,
        "manual_development_cost_reason": "Not instrumented; this is computation cost for a one-off analytical check.",
        "scientific_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    }
    RUNTIME.write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "counts": counts,
                      "ambiguous_cases": [t["ambiguous_optima_cases"] for t in trees],
                      "elapsed": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
