"""Exact checks for docs/relational-cycle-analysis.md.

Run: python3 -m experiments.relational_cycle_analysis
"""

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
CONTRACT = ROOT / "docs/relational-cycle-analysis.md"
RESULT = ROOT / "reports/relational-cycle-analysis.json"
RUNTIME = ROOT / "reports/relational-cycle-analysis-runtime.json"
CONFIG = {"sizes": [3, 4], "beta": [2, 4, 6], "compatibility": 4,
          "outer_default": 0, "outer_default_cost": 1,
          "preference_menu": "All binary edge preferences",
          "evidence_menu": "All missing/0/1 leaf evidence", "cuts": "All nodes"}


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


def charpoly(matrix):
    """Ascending integer coefficients of det(z I - matrix), by permutations."""
    n = len(matrix)
    result = [0]*(n+1)
    for perm in itertools.permutations(range(n)):
        inversions = sum(perm[i] > perm[j] for i in range(n) for j in range(i+1, n))
        poly = [(-1)**inversions]
        for i, j in enumerate(perm):
            factor = [-matrix[i][j], int(i == j)]
            product = [0]*(len(poly)+1)
            for a, x in enumerate(poly):
                for b, y in enumerate(factor):
                    product[a+b] += x*y
            poly = product
        result = [a+b for a, b in zip(result, poly)]
    return result


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite existing analytical receipts.")
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (CONTRACT, Path(__file__).resolve())},
        "command": "python3 -m experiments.relational_cycle_analysis",
        "seed_policy": "No randomness: complete prescribed enumeration.",
    }
    wall, cpu = time.perf_counter(), time.process_time()
    counts = {"geometry_assignments": 0, "geometry_binary_checks": 0,
              "pair_construction_candidates": 0, "inference_cases": 0,
              "exhaustive_candidate_visits": 0, "exhaustive_feasible_scores": 0,
              "conditioned_chain_calls": 0, "transition_and_closing_candidates": 0,
              "decoded_energy_checks": 0}
    geometry = []
    polynomials = {3: [[0, 9, -6, 1], [-4, 9, -6, 1]],
                   4: [[0, -16, 20, -8, 1], [4, -16, 20, -8, 1]]}
    for n in CONFIG["sizes"]:
        for h in itertools.product((0, 1), repeat=n):
            lap = [[0]*n for _ in range(n)]
            for i, z in enumerate(h):
                j, r = (i+1) % n, (-1)**z
                lap[i][i] += 1
                lap[j][j] += 1
                lap[i][j] -= r
                lap[j][i] -= r
            poly = charpoly(lap)
            parity = sum(h) % 2
            assert poly == polynomials[n][parity]
            counts["geometry_assignments"] += 1
            zeros = 0
            for b in itertools.product((0, 1), repeat=n):
                counts["geometry_binary_checks"] += 1
                mismatch = sum(h[i] != (b[i] ^ b[(i+1) % n]) for i in range(n))
                assert mismatch % 2 == parity
                x = [(-1)**z for z in b]
                quadratic = sum(x[i]*lap[i][j]*x[j] for i in range(n) for j in range(n))
                assert quadratic == 4*mismatch
                zeros += mismatch == 0
            assert zeros == (0 if parity else 2)
            geometry.append({"n": n, "relations": h, "parity": parity,
                             "characteristic_polynomial": poly, "binary_zero_modes": zeros})

    def energy(b, h, t, beta):
        return (sum(b) + beta*sum(x != y for x, y in zip(h, t))
                + 4*sum(h[i] != (b[i] ^ b[(i+1) % len(b)]) for i in range(len(b))))

    def construct(t, beta):
        tables = []
        for preferred in t:
            table = {}
            for a, b in itertools.product((0, 1), repeat=2):
                candidates = []
                for h in (0, 1):
                    counts["pair_construction_candidates"] += 1
                    candidates.append((4*(h != (a ^ b)) + beta*(h != preferred), h))
                best = min(candidates)
                assert best[0] == min(4, beta)*((a ^ b) != preferred)
                table[a, b] = best
            tables.append(table)
        return tables

    def solve(tables, evidence, cut):
        counts["conditioned_chain_calls"] += 1
        n = len(evidence)
        order = [(cut+j) % n for j in range(n)]

        def unary(v, z):
            return z if evidence[v] in (None, z) else math.inf

        winner = None
        for root in (0, 1):
            profile = [unary(cut, z) if z == root else math.inf for z in (0, 1)]
            back = []
            for position in range(1, n):
                edge, node = order[position-1], order[position]
                current, parents = [], []
                for b in (0, 1):
                    candidates = []
                    for a in (0, 1):
                        counts["transition_and_closing_candidates"] += 1
                        candidates.append((profile[a] + tables[edge][a, b][0], a))
                    value, a = min(candidates)
                    current.append(value + unary(node, b))
                    parents.append(a)
                back.append(parents)
                profile = current
            closing = []
            for last in (0, 1):
                counts["transition_and_closing_candidates"] += 1
                closing.append((profile[last] + tables[order[-1]][last, root][0], last))
            score, last = min(closing)
            if not math.isfinite(score):
                continue
            b = [None]*n
            b[order[-1]] = last
            z = last
            for position in range(n-1, 0, -1):
                z = back[position-1][z]
                b[order[position-1]] = z
            assert b[cut] == root
            h = tuple(tables[i][b[i], b[(i+1) % n]][1] for i in range(n))
            candidate = (score, root, last, tuple(b), h)
            if winner is None or candidate[:3] < winner[:3]:
                winner = candidate
        assert winner is not None
        return winner[0], winner[3], winner[4]

    rows, groups = [], []
    for n in CONFIG["sizes"]:
        states = list(itertools.product((0, 1), repeat=n))
        for beta in CONFIG["beta"]:
            summary = {"n": n, "beta": beta, "cases": 0, "tied_cases": 0,
                       "cut_dependent_decodings": 0, "balanced_only": 0,
                       "frustrated_only": 0, "mixed_parity_optima": 0}
            for t in states:
                tables = construct(t, beta)
                for evidence in itertools.product((None, 0, 1), repeat=n):
                    counts["inference_cases"] += 1
                    summary["cases"] += 1
                    best, optima = math.inf, set()
                    for b, h in itertools.product(states, repeat=2):
                        counts["exhaustive_candidate_visits"] += 1
                        if not all(e is None or b[i] == e for i, e in enumerate(evidence)):
                            continue
                        counts["exhaustive_feasible_scores"] += 1
                        score = energy(b, h, t, beta)
                        if score < best:
                            best, optima = score, {(b, h)}
                        elif score == best:
                            optima.add((b, h))
                    parities = sorted({sum(h) % 2 for _, h in optima})
                    if beta == 2:
                        assert parities == [0]
                        assert all(all(h[i] == (b[i] ^ b[(i+1) % n]) for i in range(n)) for b, h in optima)
                    if beta == 6:
                        assert all(h == t for _, h in optima)
                    decoded = []
                    for cut in range(n):
                        score, b, h = solve(tables, evidence, cut)
                        assert score == best and (b, h) in optima, (n, beta, t, evidence, cut)
                        assert all(e is None or b[i] == e for i, e in enumerate(evidence))
                        counts["decoded_energy_checks"] += 1
                        assert energy(b, h, t, beta) == score
                        decoded.append((b, h))
                    summary["tied_cases"] += len(optima) > 1
                    summary["cut_dependent_decodings"] += len(set(decoded)) > 1
                    summary[{(0,): "balanced_only", (1,): "frustrated_only",
                             (0, 1): "mixed_parity_optima"}[tuple(parities)]] += 1
                    rows.append({"n": n, "beta": beta, "preference": t, "evidence": evidence,
                                 "energy": best, "optimum_count": len(optima),
                                 "optimum_relation_parities": parities,
                                 "cut_zero_decoded": decoded[0], "distinct_cut_decodings": len(set(decoded))})
            groups.append(summary)

    expected_counts = {"geometry_assignments": 24, "geometry_binary_checks": 320,
                       "pair_construction_candidates": 2112, "inference_cases": 4536,
                       "exhaustive_candidate_visits": 1036800, "exhaustive_feasible_scores": 208896,
                       "conditioned_chain_calls": 17496, "transition_and_closing_candidates": 474336,
                       "decoded_energy_checks": 17496}
    assert counts == expected_counts, counts

    def lookup(beta, preference, evidence):
        return next(r for r in rows if r["n"] == 3 and r["beta"] == beta
                    and r["preference"] == preference and r["evidence"] == evidence)

    for beta in CONFIG["beta"]:
        row = lookup(beta, (0, 0, 0), (None, None, None))
        assert row["energy"] == 0 and row["optimum_count"] == 1
    for beta, value, multiplicity in ((2, 3, 3), (4, 5, 6), (6, 5, 3)):
        row = lookup(beta, (1, 1, 1), (None, None, None))
        assert row["energy"] == value and row["optimum_count"] == multiplicity
    row = lookup(2, (0, 0, 0), (1, None, None))
    assert row["cut_zero_decoded"] == ((1, 1, 1), (0, 0, 0)) and row["energy"] == 3
    row = lookup(2, (1, 0, 1), (1, None, None))
    assert row["cut_zero_decoded"] == ((1, 0, 0), (1, 0, 1)) and row["energy"] == 1
    science = {"scope": "Supplied small-cycle geometry and exact interpretation; no learned capacity claim.",
               "config": CONFIG, "geometry": geometry, "groups": groups,
               "cases": rows, "work_counts": counts, "all_checks_passed": True}
    RESULT.write_text(json.dumps(science, indent=2, sort_keys=True) + "\n")
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
                        "libraries": "Python standard library", "dtype": "Exact integers; infinity sentinel",
                        "blas": "not used", "device": "CPU", "hardware_limits": None,
                        "hardware_limits_reason": "No quota inventory; no scaling benchmark."},
        "timing": {"whole_check_and_scientific_serialization": elapsed},
        "timing_scope": "After imports/provenance through scientific serialization; excludes startup/runtime receipt. One run, not a latency benchmark.",
        "memory": {"peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "rss_unit": "KiB" if sys.platform.startswith("linux") else "platform dependent",
                   "rss_scope": "Parent cumulative high-water, including exhaustive validation and scientific output.",
                   "configuration_python_object_bytes": object_bytes(CONFIG),
                   "configuration_json_utf8_bytes": len(json.dumps(CONFIG, sort_keys=True).encode()),
                   "scientific_artifact_bytes": RESULT.stat().st_size,
                   "learned_state_bytes": 0, "optimizer_bytes": 0, "replay_bytes": 0,
                   "temporary_workspace_bytes": None,
                   "temporary_workspace_reason": "No allocator-level workspace instrumentation; see process RSS.",
                   "accelerator_bytes": None, "accelerator_reason": "No accelerator used."},
        "work_counts": {**counts, "fits": 0, "tuning_candidates": 0,
                        "empirical_examples": 0, "optimizer_steps": 0, "solver_failures": 0},
        "selection_scope": "Includes all pair-construction, reference, and every-cut alternatives. Regimes declared analytically; none selected by scores.",
        "manual_development_cost": None, "manual_development_cost_reason": "Not measured; one-off computation receipt only.",
        "scientific_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    }
    RUNTIME.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_passed": True, "groups": groups,
                      "work_counts": counts, "elapsed": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
