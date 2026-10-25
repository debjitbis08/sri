"""Exact checks for docs/inner-relation-quotient-analysis.md.

Run: python3 -m experiments.inner_relation_quotient_analysis
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
CONTRACT = ROOT / "docs/inner-relation-quotient-analysis.md"
RESULT = ROOT / "reports/inner-relation-quotient-analysis.json"
RUNTIME = ROOT / "reports/inner-relation-quotient-analysis-runtime.json"
CONFIG = {"outer_bits": 4, "inner_prior": 5, "compatibility": 6,
          "outer_prior": "Hamming", "group": "cyclic rotations and global complement",
          "stored_outer_menu": "all 16", "stored_inner_menu": "all 4",
          "evidence_menu": "all 81 missing/0/1 patterns"}


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


def multiply(a, b):
    columns = list(zip(*b))
    return [[sum(x*y for x, y in zip(row, col)) for col in columns] for row in a]


def shifted(a, scalar):
    return [[value-scalar*int(i == j) for j, value in enumerate(row)] for i, row in enumerate(a)]


def rank(a):
    work = [[Q(x) for x in row] for row in a]
    pivot = 0
    for col in range(len(work[0])):
        choice = next((r for r in range(pivot, len(work)) if work[r][col]), None)
        if choice is None:
            continue
        work[pivot], work[choice] = work[choice], work[pivot]
        scale = work[pivot][col]
        work[pivot] = [x/scale for x in work[pivot]]
        for r in range(pivot+1, len(work)):
            scale = work[r][col]
            work[r] = [x-scale*y for x, y in zip(work[r], work[pivot])]
        pivot += 1
    return pivot


def main():
    if RESULT.exists() or RUNTIME.exists():
        raise SystemExit("Refusing to overwrite existing analytical receipts.")
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (CONTRACT, Path(__file__).resolve())},
        "command": "python3 -m experiments.inner_relation_quotient_analysis",
        "seed_policy": "None: complete integer/rational verification.",
    }
    wall, cpu = time.perf_counter(), time.process_time()
    states = list(itertools.product((0, 1), repeat=4))
    index = {b: i for i, b in enumerate(states)}

    def actions(b):
        return [tuple(v ^ flip for v in b[k:]+b[:k]) for k in range(4) for flip in (0, 1)]

    label = {b: min(actions(b)) for b in states}
    classes = sorted(set(label.values()))
    fibers = {c: [b for b in states if label[b] == c] for c in classes}
    names = dict(zip(classes, ("uniform", "one different", "adjacent pairs", "alternating")))
    assert classes == [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1)]
    assert [len(fibers[c]) for c in classes] == [2, 8, 4, 2]
    assert all(label[b] == label[g] for b in states for g in actions(b))

    def edge_relations(b):
        return tuple(b[i] ^ b[(i+1) % 4] for i in range(4))

    # Separate classifier for the direct energy reference.
    def shape(b):
        transitions = sum(edge_relations(b))
        if transitions == 0:
            return classes[0]
        if transitions == 4:
            return classes[3]
        return classes[1] if sum(b) in (1, 3) else classes[2]

    assert all(shape(b) == label[b] for b in states)
    assert len({edge_relations(b) for b in states}) == 8
    assert label[(0, 0, 1, 1)] == label[(1, 1, 0, 0)] == classes[2]
    modes = [(1, 1, -1, -1), (1, -1, -1, 1)]
    relations = [(0, 1, 0, 1), (1, 0, 1, 0)]
    for v, h in zip(modes, relations):
        assert all(v[i] == (-1)**h[i]*v[(i+1) % 4] for i in range(4))
    assert sum(x*y for x, y in zip(*modes)) == 0

    lap = [[0]*16 for _ in states]
    projection = [[Q(0) for _ in states] for _ in states]
    for b, i in index.items():
        lap[i][i] += 3
        for g in (b[1:]+b[:1], b[-1:]+b[:-1], tuple(1-v for v in b)):
            lap[i][index[g]] -= 1
        for g in actions(b):
            projection[i][index[g]] += Q(1, 8)
    zero = [[0]*16 for _ in states]
    assert lap == list(map(list, zip(*lap)))
    assert projection == list(map(list, zip(*projection)))
    assert multiply(projection, projection) == projection
    assert multiply(lap, projection) == zero
    assert rank(lap) == 12
    poly = lap
    for value in (2, 4, 6):
        poly = multiply(poly, shifted(lap, value))
    assert poly == zero

    def reduce_profile(f):
        return [min(f[index[b]] for b in fibers[c]) for c in classes]

    def lift_profile(profile):
        return [profile[classes.index(label[b])] for b in states]

    for k in range(4):
        profile = [int(j == k) for j in range(4)]
        assert reduce_profile(lift_profile(profile)) == profile
    f, g = list(range(16)), list(range(15, -1, -1))
    assert reduce_profile([min(x, y) for x, y in zip(f, g)]) == [min(x, y) for x, y in zip(reduce_profile(f), reduce_profile(g))]
    assert reduce_profile([x+3 for x in f]) == [x+3 for x in reduce_profile(f)]
    assert lift_profile(reduce_profile(lift_profile(reduce_profile(f)))) == lift_profile(reduce_profile(f))
    counter_f, counter_g = [100]*16, [100]*16
    counter_f[0], counter_f[15] = 0, 2
    counter_g[0], counter_g[15] = 2, 0
    assert reduce_profile([x+y for x, y in zip(counter_f, counter_g)])[0] == 2
    assert reduce_profile(counter_f)[0] + reduce_profile(counter_g)[0] == 0

    counts = {"inference_cases": 0, "joint_reference_visits": 0,
              "joint_reference_feasible_scores": 0, "fiber_outer_visits": 0,
              "fiber_outer_feasible_scores": 0, "inner_profile_candidates": 0}
    summary = {"tied_cases": 0, "multiple_optimal_classes": 0,
               "preferred_class_feasible": 0, "class_changes": 0,
               "coherent_no_evidence_defaults_checked": 0}
    rows = []
    for s in states:
        for c0 in classes:
            for evidence in itertools.product((None, 0, 1), repeat=4):
                counts["inference_cases"] += 1
                best, optima = math.inf, set()
                for b, c in itertools.product(states, classes):
                    counts["joint_reference_visits"] += 1
                    if not all(e is None or b[i] == e for i, e in enumerate(evidence)):
                        continue
                    counts["joint_reference_feasible_scores"] += 1
                    score = sum(x != y for x, y in zip(b, s)) + 5*(c != c0) + 6*(c != shape(b))
                    if score < best:
                        best, optima = score, {(b, c)}
                    elif score == best:
                        optima.add((b, c))
                profiles = {c: (math.inf, None) for c in classes}
                for b in states:
                    counts["fiber_outer_visits"] += 1
                    if not all(e is None or b[i] == e for i, e in enumerate(evidence)):
                        continue
                    counts["fiber_outer_feasible_scores"] += 1
                    c, score = label[b], sum(x != y for x, y in zip(b, s))
                    if score < profiles[c][0]:  # Ordered states supply canonical ties.
                        profiles[c] = (score, b)
                choices = []
                for c in classes:
                    counts["inner_profile_candidates"] += 1
                    choices.append((profiles[c][0] + 5*(c != c0), c))
                score, c = min(choices)
                b = profiles[c][1]
                assert score == best and (b, c) in optima
                assert all(e is None or b[i] == e for i, e in enumerate(evidence))
                assert all(shape(outer) == inner for outer, inner in optima)
                feasible = math.isfinite(profiles[c0][0])
                if feasible:
                    assert all(inner == c0 for _, inner in optima)
                summary["preferred_class_feasible"] += feasible
                summary["class_changes"] += c != c0
                summary["tied_cases"] += len(optima) > 1
                summary["multiple_optimal_classes"] += len({inner for _, inner in optima}) > 1
                if all(e is None for e in evidence) and label[s] == c0:
                    assert optima == {(s, c0)} and best == 0
                    summary["coherent_no_evidence_defaults_checked"] += 1
                if all(e is not None for e in evidence):
                    assert b == evidence and c == label[evidence]
                rows.append({"outer_default": s, "inner_default": c0, "evidence": evidence,
                             "decoded_outer": b, "decoded_inner": c, "energy": score,
                             "optimum_count": len(optima), "preferred_class_feasible": feasible,
                             "optimal_class_count": len({inner for _, inner in optima})})

    assert counts == {"inference_cases": 5184, "joint_reference_visits": 331776,
                      "joint_reference_feasible_scores": 65536, "fiber_outer_visits": 82944,
                      "fiber_outer_feasible_scores": 16384, "inner_profile_candidates": 20736}
    assert summary["coherent_no_evidence_defaults_checked"] == 16
    hand_examples = []
    for evidence, expected in (((None, None, None, None), (0, 0, 1, 1)),
                               ((0, 0, None, None), (0, 0, 1, 1)),
                               ((1, 1, None, None), (1, 1, 0, 0)),
                               ((0, 1, 0, 1), (0, 1, 0, 1))):
        row = next(row for row in rows if row["outer_default"] == (0, 0, 1, 1)
                   and row["inner_default"] == classes[2] and row["evidence"] == evidence)
        assert row["decoded_outer"] == expected and row["decoded_inner"] == label[expected]
        assert row["optimum_count"] == 1
        hand_examples.append(row)
    science = {"scope": "Supplied relation quotient and exact inference; not learned or scalable unit dynamics.",
               "config": CONFIG,
               "classes": [{"label": c, "name": names[c], "outer_patterns": fibers[c],
                            "edge_patterns": sorted({edge_relations(b) for b in fibers[c]})} for c in classes],
               "spectral_checks": {"configuration_dimension": 16, "laplacian_rank": 12,
                                   "retained_dimension": 4, "polynomial_roots": [0, 2, 4, 6],
                                   "projection_identities_passed": True},
               "summary": summary, "hand_examples": hand_examples,
               "cases": rows, "work_counts": counts, "all_checks_passed": True}
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
                        "libraries": "Python standard library", "dtype": "Exact integers/Fraction; infinity sentinel",
                        "processes": 1, "workers": 0, "computation_threads": 1,
                        "blas": "not used", "device": "CPU", "hardware_limits": None,
                        "hardware_limits_reason": "No quota inventory; no scaling benchmark."},
        "timing": {"whole_check_and_scientific_serialization": elapsed},
        "timing_scope": "After imports/provenance through scientific serialization; excludes startup/runtime receipt. One run, not inference benchmark.",
        "memory": {"peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "rss_unit": "KiB" if sys.platform.startswith("linux") else "platform dependent",
                   "rss_scope": "Single parent high-water, including matrix checks, reference and output.",
                   "configuration_python_object_bytes": object_bytes(CONFIG),
                   "configuration_json_utf8_bytes": len(json.dumps(CONFIG, sort_keys=True).encode()),
                   "scientific_artifact_bytes": RESULT.stat().st_size,
                   "learned_state_bytes": 0, "optimizer_bytes": 0, "replay_bytes": 0,
                   "temporary_workspace_bytes": None,
                   "temporary_workspace_reason": "Allocator workspace not separately measured; see RSS.",
                   "accelerator_bytes": None, "accelerator_reason": "Not used."},
        "work_counts": {**counts, "group_actions_checked": 128, "fits": 0,
                        "tuning_candidates": 0, "empirical_examples": 0, "optimizer_steps": 0,
                        "solver_failures": 0},
        "selection_scope": "All declared preferences/evidence and rejected reference/profile alternatives included; no parameter tuning.",
        "manual_development_cost": None, "manual_development_cost_reason": "Not instrumented; computational receipt only.",
        "scientific_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
    }
    RUNTIME.write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "summary": summary,
                      "counts": counts, "elapsed": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
