"""Single-module learning benchmark v1: development selection, freeze, confirmation.

Usage: .venv/bin/python -m experiments.learning_benchmark --phase {dev,confirm,all} [--workers 8]
Contract: docs/single-module-learning-benchmark-v1-contract.md
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import stats
import torch

from src.learning_benchmark import (ARMS, BUDGETS, CONDITIONS, REFERENCES, REGIMES, SEED,
    encode, evaluation_sets, group_key, parameter_count, probe_r2, task_world, train_ensemble,
    training_pairs)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-learning-benchmark-v1'
DEV_WORLDS = tuple(range(8))
CONFIRM_WORLDS = tuple(range(1000, 1032))
CANDIDATE = 'coarse_local_vicreg'
COMPARATORS = ('dense_linear_vicreg', 'mlp_vicreg', 'conv_vicreg', 'dae_paired')
ESTABLISHED = ('mlp_vicreg', 'conv_vicreg', 'dae_paired')
RESILIENCE_BUDGET, THRESHOLD, MARGIN, ALPHA = 32, .6, .02, .05
SOURCES = ('docs/single-module-learning-benchmark-v1-contract.md', 'src/learning_benchmark.py',
           'experiments/learning_benchmark.py', 'tests/test_learning_benchmark.py', 'requirements.txt')


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(f'{value:.12g}') if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def dump(path, value):
    path.write_text(json.dumps(canonical(value), indent=2, sort_keys=True)+'\n')


def init_seed(world, regime, arm, budget, cfg_index):
    key = [SEED, 3, world, REGIMES.index(regime), list(ARMS).index(arm), BUDGETS.index(budget), cfg_index]
    return int(np.random.SeedSequence(key).generate_state(1)[0] % 2**31)


def world_data(world, regime):
    w = task_world(world)
    return training_pairs(w, regime, world), evaluation_sets(w, regime, world)


def score(code_fn, evals):
    probe, tests = evals
    return probe_r2(code_fn(probe[0]), probe[1], {k: (code_fn(h), c) for k, (h, c) in tests.items()})


def run_job(job):
    """Train one ensemble (cfgs x worlds) for one arm, regime and budget."""
    torch.set_num_threads(1)
    arm, cfg_indices, regime, budget, worlds = job
    grid = ARMS[arm]['grid']
    data = {w: world_data(w, regime) for w in worlds}
    members = [(i, w) for i in cfg_indices for w in worlds]
    h1 = np.stack([data[w][0][0][:budget] for _, w in members])
    h2 = np.stack([data[w][0][1][:budget] for _, w in members])
    params, base, records = train_ensemble(arm, [grid[i] for i, _ in members], h1, h2,
                                           [init_seed(w, regime, arm, budget, i) for i, w in members])
    out = []
    for k, (i, w) in enumerate(members):
        if records[k]['finite']:
            r2 = score(lambda h: encode(params, base, k, h), data[w][1])
            if not all(np.isfinite(v) for v in r2.values()):
                r2 = {c: 0. for c in CONDITIONS}
        else:
            r2 = {c: 0. for c in CONDITIONS}
        out.append(dict(arm=arm, cfg=i, regime=regime, budget=budget, world=w, r2=r2, **records[k]))
    return out


def jobs_for(phase, selection=None):
    jobs = []
    for arm, spec in ARMS.items():
        for regime in REGIMES:
            for budget in BUDGETS:
                if phase == 'dev':
                    groups = {}
                    for i, cfg in enumerate(spec['grid']):
                        groups.setdefault(group_key(cfg), []).append(i)
                    for indices in groups.values():
                        jobs.append((arm, tuple(indices), regime, budget, DEV_WORLDS))
                else:
                    jobs.append((arm, (selection[arm][str(budget)],), regime, budget, CONFIRM_WORLDS))
    # Largest jobs first for load balance.
    cost = lambda j: j[3]*len(j[1])*len(j[4])*(3 if j[0] == 'conv_vicreg' else 1)
    return sorted(jobs, key=cost, reverse=True)


def run_parallel(jobs, workers):
    rows = []
    with Pool(workers) as pool:
        for done, out in enumerate(pool.imap_unordered(run_job, jobs), 1):
            rows.extend(out)
            print(f'  job {done}/{len(jobs)}', flush=True)
    return sorted(rows, key=lambda r: (r['arm'], r['regime'], r['budget'], r['cfg'], r['world']))


def references(worlds):
    rows = []
    for regime in REGIMES:
        for w in worlds:
            (h1, h2), evals = world_data(w, regime)
            for budget in BUDGETS:
                for name, fn in REFERENCES.items():
                    basis = fn(h1[:budget], h2[:budget])
                    rows.append(dict(arm=name, regime=regime, budget=budget, world=w,
                                     r2=score(lambda h: h@basis, evals)))
    return rows


def development(workers, out_dir):
    started = time.perf_counter()
    rows = run_parallel(jobs_for('dev'), workers)
    selection, table = {}, {}
    for arm, spec in ARMS.items():
        selection[arm], table[arm] = {}, {}
        for budget in BUDGETS:
            means = [np.mean([r['r2']['in_distribution'] for r in rows
                              if r['arm'] == arm and r['budget'] == budget and r['cfg'] == i])
                     for i in range(len(spec['grid']))]
            selection[arm][str(budget)] = int(np.argmax(means))
            table[arm][str(budget)] = means
    dump(out_dir/f'{PREFIX}-development.json', dict(
        worlds=DEV_WORLDS, grids={a: s['grid'] for a, s in ARMS.items()},
        mean_in_distribution_r2_by_config=table, selection=selection,
        non_finite_members=sum(not r['finite'] for r in rows), rows=rows,
        references=references(DEV_WORLDS)))
    return selection, time.perf_counter()-started


def freeze(selection, out_dir):
    manifest = dict(
        scope='single-module learning benchmark v1 confirmation',
        seed=SEED, development_worlds=DEV_WORLDS, confirmation_worlds=CONFIRM_WORLDS,
        budgets=BUDGETS, regimes=REGIMES, conditions=CONDITIONS, arms=list(ARMS),
        references=list(REFERENCES),
        selected_configurations={a: {b: ARMS[a]['grid'][i] for b, i in s.items()} for a, s in selection.items()},
        selected_indices=selection,
        parameter_counts={a: {b: parameter_count(a, ARMS[a]['grid'][i]) for b, i in s.items()}
                          for a, s in selection.items()},
        decision_rule=dict(candidate=CANDIDATE, comparators=COMPARATORS, established=ESTABLISHED,
                           margin=MARGIN, alpha=ALPHA, correction='holm', regime='aligned',
                           resilience_budget=RESILIENCE_BUDGET, threshold=THRESHOLD),
        hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        development_report_sha256=hashlib.sha256((out_dir/f'{PREFIX}-development.json').read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__, threads_per_process=1))
    dump(out_dir/f'{PREFIX}-manifest.json', manifest)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.zeros(len(pvalues)), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted


def analyse(rows):
    arms = list(ARMS)+list(REFERENCES)
    get = {(r['arm'], r['regime'], r['budget'], r['world']): r['r2'] for r in rows}
    worlds = CONFIRM_WORLDS

    def lowdata(arm, regime):
        return np.array([np.mean([get[arm, regime, b, w]['in_distribution'] for b in BUDGETS]) for w in worlds])

    def resilience(arm, regime):
        return np.array([np.mean([get[arm, regime, RESILIENCE_BUDGET, w][c] for c in CONDITIONS[1:]])
                         for w in worlds])

    def interval(x):
        se = np.std(x, ddof=1)/np.sqrt(len(x))
        return dict(mean=float(np.mean(x)), ci95=[float(np.mean(x)-1.96*se), float(np.mean(x)+1.96*se)])

    curves = {regime: {arm: {cond: {str(b): interval(np.array([get[arm, regime, b, w][cond] for w in worlds]))
                                    for b in BUDGETS} for cond in CONDITIONS} for arm in arms}
              for regime in REGIMES}
    scores = {regime: {arm: dict(low_data=interval(lowdata(arm, regime)),
                                 resilience=interval(resilience(arm, regime))) for arm in arms}
              for regime in REGIMES}
    thresholds = {regime: {arm: next((b for b in BUDGETS
                                      if curves[regime][arm]['in_distribution'][str(b)]['mean'] >= THRESHOLD),
                                     'not reached by 256') for arm in arms} for regime in REGIMES}
    tests = {}
    for regime in REGIMES:
        tests[regime] = {}
        for name, fn in (('H1_low_data', lowdata), ('H2_resilience', resilience)):
            cand = fn(CANDIDATE, regime)
            diffs = {c: cand-fn(c, regime) for c in COMPARATORS+('local_cca', 'global_cca')}
            primary = [stats.ttest_rel(cand, fn(c, regime)).pvalue for c in COMPARATORS]
            adjusted = holm(np.array(primary))
            family = {}
            for i, c in enumerate(COMPARATORS):
                d = diffs[c]
                label = ('better' if adjusted[i] < ALPHA and d.mean() >= MARGIN else
                         'worse' if adjusted[i] < ALPHA and d.mean() <= -MARGIN else 'no practical difference')
                family[c] = dict(difference=interval(d), p=float(primary[i]), holm_p=float(adjusted[i]),
                                 outcome=label)
            for c in ('local_cca', 'global_cca'):
                family[c] = dict(difference=interval(diffs[c]),
                                 p=float(stats.ttest_rel(cand, fn(c, regime)).pvalue), outcome='descriptive only')
            tests[regime][name] = family
    primary = {h: dict(outcomes={c: tests['aligned'][h][c]['outcome'] for c in COMPARATORS},
                       better_than_all_established=all(tests['aligned'][h][c]['outcome'] == 'better'
                                                       for c in ESTABLISHED))
               for h in ('H1_low_data', 'H2_resilience')}
    return dict(primary=primary, tests=tests, scores=scores, curves=curves,
                first_budget_reaching_threshold=thresholds)


def confirmation(workers, out_dir):
    started = time.perf_counter()
    manifest = json.loads((out_dir/f'{PREFIX}-manifest.json').read_text())
    for name, digest in manifest['hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f'{name} changed after freezing')
    selection = manifest['selected_indices']
    rows = run_parallel(jobs_for('confirm', selection), workers)
    convergence = {arm: dict(non_finite=sum(not r['finite'] for r in rows if r['arm'] == arm),
                             tail_change_over_5pct=sum(r['tail_relative_change'] > .05
                                                       for r in rows if r['arm'] == arm),
                             members=sum(r['arm'] == arm for r in rows)) for arm in ARMS}
    refs = references(CONFIRM_WORLDS)
    dump(out_dir/f'{PREFIX}-results.json', dict(analysis=analyse(rows+refs), convergence=convergence,
                                                 rows=rows, references=refs))
    return time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'confirm', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    runtime = {}
    if args.phase in ('dev', 'all'):
        selection, runtime['development_seconds'] = development(args.workers, args.out_dir)
        freeze(selection, args.out_dir)
        print('Selection frozen:', selection, flush=True)
    if args.phase in ('confirm', 'all'):
        runtime['confirmation_seconds'] = confirmation(args.workers, args.out_dir)
        result = json.loads((args.out_dir/f'{PREFIX}-results.json').read_text())
        print(json.dumps(result['analysis']['primary'], indent=2))
    runtime['workers'] = args.workers
    path = args.out_dir/f'{PREFIX}-runtime.json'
    if path.exists():
        runtime = {**json.loads(path.read_text()), **runtime}
    dump(path, runtime)


if __name__ == '__main__':
    main()
