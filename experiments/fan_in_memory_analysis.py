"""Analytical checks only; no model fitting or experiment selection.

Run: .venv/bin/python -m experiments.fan_in_memory_analysis
See docs/fan-in-memory-analysis.md for assumptions and complexity.
"""
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/fan-in-pilot-v1-fixed-comparison.json.gz'
    report = json.loads(gzip.decompress(source.read_bytes()))
    means = {r['measure']: {k: statistics.mean(v) for k, v in r['values'].items()}
             for r in report['measures']}
    K, H, draws = 1500, 750, 4
    r = 2 ** (-1 / H)
    weights = [r ** age for age in range(K)]
    normalizer = sum(weights)
    mass = {}
    for age in (0, 500, 1000, 1500):
        direct = sum(weights[age:]) / normalizer
        formula = (r ** age - r ** K) / (1 - r ** K)
        assert math.isclose(direct, formula, abs_tol=1e-12)
        mass[str(age)] = {'replay_fraction_old_script': formula,
                          'old_script_episodes_per_update': draws * formula}
    # The implementation's max-one rule makes k=1's drop fraction constant.
    drops = [max(1, round(.3 * (1 + math.cos(math.pi * step / 4500)) / 2))
             for step in range(100, 4501, 100)]
    assert drops == [1] * 45
    # A scalar affine shift preserves all within-script order, but ruins matching.
    counts = list(range(1, 10))
    shifted = [n + 20 for n in counts]
    order = sum((shifted[i] > shifted[j]) == (counts[i] > counts[j])
                for i in range(9) for j in range(9) if i != j) / 72
    matches = sum(min(counts, key=lambda n: abs(n - x)) == y
                  for x, y in zip(shifted, counts)) / 9
    assert order == 1 and matches == 1 / 9
    out = {'source': str(source.relative_to(root)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'fixed_run_means': means, 'fifo_assumptions': 'One fresh write/update; no replay reinsertion; full store; ages 0..K-1',
           'K': K, 'H': H, 'old_script_replay': mass,
           'mean_replay_age': sum(a * w for a, w in enumerate(weights)) / normalizer,
           'stationary_expected_replays_per_inserted_episode': draws * sum(weights) / normalizer,
           'k1_drop_counts': drops, 'scalar_counterexample': {'order': order, 'matching': matches}}
    prefix = root / 'reports/fan-in-memory-analysis'
    prefix.with_suffix('.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    tracked = [Path(__file__), root / 'docs/fan-in-memory-analysis.md',
               root / 'experiments/fan_in_pilot.py', root / 'docs/limited-memory-proposal.md']
    runtime = {'scope': 'Read frozen comparison, aggregate means, verify scalar counterexample and FIFO formulas; no neural training',
               'command': '.venv/bin/python -m experiments.fan_in_memory_analysis',
               'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
               'hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
               'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
               'numeric': 'Python float (double precision)', 'workers': 1, 'seed_policy': 'No random sampling',
               'fits': 0, 'tuning_candidates': 0, 'optimizer_steps': 0,
               'wall_seconds': time.perf_counter() - wall, 'process_cpu_seconds': time.process_time() - cpu,
               'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'measurement_note': 'Whole script through analysis and provenance collection; excludes final runtime JSON write'}
    prefix.with_name(prefix.name + '-runtime.json').write_text(json.dumps(runtime, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'checks': 'passed', 'old_script_replay': mass,
                      'mean_replay_age': out['mean_replay_age'], 'wall_seconds': runtime['wall_seconds']}))


if __name__ == '__main__':
    main()
