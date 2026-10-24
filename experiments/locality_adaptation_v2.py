"""Locality adaptation v2: development, manifest freeze, confirmation, timing panel.

Usage (commit the manifest and source between freeze and confirm):
  .venv/bin/python -m unittest tests.test_locality_adaptation_v2 tests.test_locality_adaptation -v
  .venv/bin/python -m experiments.locality_adaptation_v2 --phase dev
  .venv/bin/python -m experiments.locality_adaptation_v2 --phase freeze
  git commit ...
  .venv/bin/python -m experiments.locality_adaptation_v2 --phase confirm
  .venv/bin/python -m experiments.locality_adaptation_v2 --phase timing
Contract: docs/locality-adaptation-v2-contract.md
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np
from scipy import stats

from experiments.locality_adaptation import (canonical, encoded, environment, git, idle_job, infer,
    interval, sha256)
from src.locality_adaptation import (B, BUDGETS, CONDITIONS, D, PHASES, PRIMARY_BUDGETS, RANK,
    REGIMES, Meter, ScoringFailure, local_basis, make_world, noise_covariance, nmse, orthonormal_basis,
    phase_floors, projected_noise, signal_covariance, signal_loss, views, base_stream, fit_global,
    fit_local)
from src.locality_adaptation_v2 import (ARMS, FOLDS, REFIT_ARMS, ROOT_SEED, all_n_candidates,
    fit_arm, fit_cv_refit, fit_holdout_refit, folds)
import src.locality_adaptation as v1


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'locality-adaptation-v2'
DEV_WORLDS, CONFIRM_WORLDS = tuple(range(8)), tuple(range(64))
STAGES = dict(development=0, confirmation=1, timing=2)
CANDIDATE, COMPARATOR = 'cv_refit_cca', 'fixed_cca_all'
CLAIMS = (('H1_recover_shifted_signal', 'half_shifted', 'L', .02),
          ('H2_preserve_aligned_signal', 'aligned', 'L', -.02),
          ('H3_improve_shifted_usefulness', 'half_shifted', 'E', .02),
          ('H4_preserve_aligned_usefulness', 'aligned', 'E', -.02))
CONFIDENCE = 1-.05/len(CLAIMS)
OPTIMAL_TOL = 1e-10
SOURCES = ('docs/locality-adaptation-v2-contract.md', 'src/locality_adaptation_v2.py',
           'experiments/locality_adaptation_v2.py', 'tests/test_locality_adaptation_v2.py', 'requirements.txt')
V1_SOURCES = ('src/locality_adaptation.py', 'experiments/locality_adaptation.py',
              'tests/test_locality_adaptation.py', 'docs/locality-adaptation-v1-contract.md')
TIMING_DIMS, TIMING_REGIMES = (32, 128, 512), ('aligned', 'dense')
TIMING_ARMS = ('cv_refit_cca', 'holdout_refit_cca', 'fixed_cca_all', 'global_cca_all')
TIMING_N, REPEATS, CALLS = 256, 7, 100


def use_v2_seed():
    """v1's generator reads its module ROOT_SEED; v2 swaps in its own root before any draw."""
    v1.ROOT_SEED = ROOT_SEED


# ---------------------------------------------------------------- scientific run

def metrics(basis, s, noises):
    return dict(signal_loss=signal_loss(basis, s),
                projected_noise={c: projected_noise(basis, s, n) for c, n in noises.items()},
                nmse={c: nmse(basis, s, n) for c, n in noises.items()})


def run_world(stage, w, meters, evaluator):
    use_v2_seed()
    world, stream = make_world(stage, w), base_stream(stage, w)
    oracle, rows = {}, []
    for regime in REGIMES:
        h1, h2 = views(world, regime, stream)
        s = signal_covariance(world, regime)
        noises = {c: noise_covariance(world, regime, c) for c in CONDITIONS}
        floors = phase_floors(s)
        best = min(floors)
        optimal = [p for p, f in enumerate(floors) if f-best <= OPTIMAL_TOL]
        oracle_basis = world['rotations'][regime]@world['u']
        oracle[regime] = dict(
            phase_floors=floors, phase0_floor=floors[0], best_phase_floor=best,
            population_optimal_phases=optimal, structural_phase=optimal[0],
            true_shift={'aligned': 0, 'half_shifted': B//2, 'unknown_shift': world['shift']}.get(regime),
            controls=dict(zero_output_nmse={c: 1. for c in CONDITIONS},
                          raw_input_nmse={c: float(np.trace(n)/np.trace(s)) for c, n in noises.items()},
                          oracle_signal_projector=metrics(oracle_basis, s, noises)))
        for n in BUDGETS:
            x, y = h1[:n], h2[:n]
            try:
                with evaluator.time('all_n_diagnostic_fits'):
                    diagnostic = [local_basis(wt, p, D) for p, wt in enumerate(all_n_candidates(x, y, meter=evaluator))]
                all_n = [metrics(c, s, noises) for c in diagnostic]
                all_n_losses = [m['signal_loss'] for m in all_n]
                for arm in ARMS:
                    basis, phase, record = fit_arm(arm, x, y, meter=meters[arm], phase=optimal[0])
                    row = dict(world=w, regime=regime, budget=n, arm=arm, phase=phase, **metrics(basis, s, noises))
                    if record is not None and 'scores' in record:
                        row['selection_scores'] = record['scores']
                    if arm in REFIT_ARMS:
                        row['decomposition'] = dict(best_phase_floor=best, fitting=min(all_n_losses)-best,
                                                    selection=row['signal_loss']-min(all_n_losses))
                    if arm == 'holdout_cca':
                        losses = [signal_loss(c, s) for c in record['candidates']]
                        row['candidate_signal_losses'] = losses
                        row['decomposition'] = dict(best_phase_floor=best, fitting=min(losses)-best,
                                                    selection=losses[phase]-min(losses))
                    if phase is not None:
                        row['selected_population_optimal'] = phase in optimal
                    if not np.all(np.isfinite([row['signal_loss'], *row['nmse'].values()])):
                        raise ScoringFailure('nonfinite metric')
                    rows.append(row)
            except ScoringFailure as error:
                raise ScoringFailure(dict(world=w, regime=regime, budget=n, error=str(error))) from None
            rows.append(dict(world=w, regime=regime, budget=n, arm='evaluator_all_n_candidates',
                             signal_losses=all_n_losses,
                             projected_noise=[m['projected_noise']['in_distribution'] for m in all_n],
                             ceiling_signal_loss=min(all_n_losses)))
    return dict(world=w, unknown_shift=world['shift'], oracle=oracle), rows


def run_stage(stage, worlds):
    meters, evaluator = {arm: Meter() for arm in ARMS}, Meter()
    worlds_out, rows, failure = [], [], None
    for w in worlds:
        try:
            info, world_rows = run_world(stage, w, meters, evaluator)
        except ScoringFailure as error:
            failure = error.args[0]
            break
        worlds_out.append(info)
        rows.extend(world_rows)
    return dict(status='incomplete' if failure else 'complete', failure=failure,
                worlds=worlds_out, rows=rows), meters, evaluator


# ---------------------------------------------------------------- analysis

def analyse(rows, worlds):
    get = {(r['world'], r['regime'], r['arm'], r['budget']): r for r in rows}
    measures = dict(L=lambda r: r['signal_loss'], E=lambda r: r['nmse']['in_distribution'],
                    E_texture_16=lambda r: r['nmse']['texture_16'],
                    E_extra_noise=lambda r: r['nmse']['extra_noise'],
                    projected_noise=lambda r: r['projected_noise']['in_distribution'])

    def score(regime, arm, measure):
        return np.array([np.mean([measures[measure](get[w, regime, arm, n]) for n in PRIMARY_BUDGETS])
                         for w in worlds])

    def ceiling(regime):
        return np.array([np.mean([get[w, regime, 'evaluator_all_n_candidates', n]['ceiling_signal_loss']
                                  for n in PRIMARY_BUDGETS]) for w in worlds])

    q = stats.t.ppf(CONFIDENCE, len(worlds)-1)
    primary = {}
    for name, regime, measure, threshold in CLAIMS:
        d = score(regime, COMPARATOR, measure)-score(regime, CANDIDATE, measure)
        se = float(np.std(d, ddof=1)/np.sqrt(len(d)))
        bound = float(d.mean()) if se == 0 else float(d.mean()-q*se)
        primary[name] = dict(regime=regime, score=measure, threshold=threshold, mean_improvement=float(d.mean()),
                             se=se, t_quantile=float(q), confidence=CONFIDENCE, lower_bound=bound,
                             zero_variance=se == 0, passed=bool(bound > threshold), per_world_improvement=d)

    scores = {regime: {arm: {m: interval(score(regime, arm, m)) for m in measures} for arm in ARMS}
              for regime in REGIMES}
    contrasts = {regime: {
        'cv_refit_minus_holdout_refit': {m: interval(score(regime, 'cv_refit_cca', m)
                                                     - score(regime, 'holdout_refit_cca', m)) for m in measures},
        'holdout_refit_minus_holdout': {m: interval(score(regime, 'holdout_refit_cca', m)
                                                    - score(regime, 'holdout_cca', m)) for m in measures},
        'candidate_minus_global_cca_all': {m: interval(score(regime, CANDIDATE, m)
                                                       - score(regime, 'global_cca_all', m)) for m in measures},
        'candidate_minus_oracle_structural_phase': {m: interval(score(regime, CANDIDATE, m)
                                                                - score(regime, 'oracle_structural_phase_cca_all', m))
                                                    for m in measures},
        'candidate_L_minus_finite_sample_ceiling': interval(score(regime, CANDIDATE, 'L')-ceiling(regime)),
        'holdout_refit_L_minus_finite_sample_ceiling': interval(score(regime, 'holdout_refit_cca', 'L')-ceiling(regime)),
    } for regime in REGIMES}
    stress = {regime: {arm: {c: interval(score(regime, arm, f'E_{c}')-score(regime, arm, 'E'))
                             for c in ('texture_16', 'extra_noise')} for arm in ARMS} for regime in REGIMES}
    curves = {regime: {arm: {m: {str(n): interval([measures[m](get[w, regime, arm, n]) for w in worlds])
                                 for n in BUDGETS} for m in ('L', 'E', 'projected_noise')}
                       for arm in ARMS} for regime in REGIMES}
    decomposition, selection = {}, {}
    for arm in (*REFIT_ARMS, 'holdout_cca'):
        decomposition[arm], selection[arm] = {}, {}
        for regime in REGIMES:
            decomposition[arm][regime], selection[arm][regime] = {}, {}
            for n in BUDGETS:
                rs = [get[w, regime, arm, n] for w in worlds]
                decomposition[arm][regime][str(n)] = {t: interval([r['decomposition'][t] for r in rs])
                                                      for t in ('best_phase_floor', 'fitting', 'selection')}
                selection[arm][regime][str(n)] = dict(
                    phase_counts=dict(sorted(Counter(r['phase'] for r in rs).items())),
                    fraction_population_optimal=float(np.mean([r['selected_population_optimal'] for r in rs])))
    return dict(primary=primary, overall_pass=all(p['passed'] for p in primary.values()),
                primary_scores=scores, contrasts=contrasts, stress_increase=stress, learning_curves=curves,
                signal_loss_decomposition=decomposition, selection=selection,
                notes=['Primary bounds are one-sided paired Student-t bounds over worlds; world-level '
                       'differences need not be Gaussian, so coverage is a finite-sample approximation.',
                       'Contrasts are named first-minus-second in lower-is-better scores: negative favours '
                       'the first; descriptive 95% t intervals, no claims.',
                       'The decomposition is in signal loss L only; reconstruction is diagnosed through L and '
                       'projected noise, never by substituting E.',
                       'oracle_structural_phase_cca_all is a structural-phase reference, not a ceiling; the '
                       'finite-sample L ceiling is the minimum over the eight all-n fits.'])


def oracle_summary(worlds_out):
    out = {}
    for regime in REGIMES:
        infos = [w['oracle'][regime] for w in worlds_out]
        out[regime] = dict(
            phase0_floor=interval([i['phase0_floor'] for i in infos]),
            best_phase_floor=interval([i['best_phase_floor'] for i in infos]),
            controls={k: {c: float(np.mean([i['controls'][k][c] for i in infos])) for c in CONDITIONS}
                      for k in ('zero_output_nmse', 'raw_input_nmse')},
            oracle_signal_projector_nmse={c: float(np.mean([i['controls']['oracle_signal_projector']['nmse'][c]
                                                            for i in infos])) for c in CONDITIONS})
    return out


# ---------------------------------------------------------------- computation records

def meter_record(meters):
    return {arm: dict(seconds=dict(m.seconds), counts=dict(m.counts)) for arm, m in meters.items()}


def fold_counts():
    return {str(n): [int((folds(n) == f).sum()) for f in range(FOLDS)] for n in BUDGETS}


def complexity_record(d=D, b=B, k=PHASES, f=FOLDS):
    rank = d//b
    return dict(
        orders=dict(cv_refit_cca=f'{f}x{k} local fits on 3n/4 + 1 refit on n; O({k} n d) scoring',
                    holdout_refit_cca=f'{k} local fits on 3n/4 + 1 refit on n; O({k} n/4 d) scoring',
                    local_fit='O(n d b + d b^2)', global_cca_fit='O(n d^2 + d^3)',
                    local_inference='O(d)', global_inference='O(d k)', optimizer_state=0, tuning_trials=0,
                    solver='dense direct Cholesky + symmetric eigh (LAPACK syevd)'),
        bytes=dict(raw_pairs_per_budget={str(n): 2*n*d*8 for n in BUDGETS},
                   statistics_per_block=2*b*b*8, statistics_one_candidate=2*d*b*8,
                   statistics_all_cv_candidates_if_held=f*k*2*d*b*8, global_statistics=2*d*d*8,
                   retained_fold_candidate_weights=d*8, cv_scores=k*8,
                   deployed_local_weights=d*8, deployed_local_metadata=3*8, deployed_global_basis=d*rank*8,
                   eigh_workspace_estimate=dict(
                       local=(1+6*b+2*b*b)*8+(3+5*b)*4, global_=(1+6*d+2*d*d)*8+(3+5*d)*4,
                       label='analytical LAPACK dsyevd lwork/liwork estimate, not measured')),
        storage_note='one block K/T pair live at a time; each fold candidate is discarded after scoring, '
                     'so at most one candidate (d weights) is retained during selection')


def record_runtime(out_dir, phase, record):
    path = out_dir/f'{PREFIX}-runtime.json'
    runtime = json.loads(path.read_text()) if path.exists() else {}
    runtime[phase] = dict(record, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          rss_scope='cumulative parent-process high-water RSS (Linux KiB), not per-arm memory',
                          dirty_paths=git('status', '--porcelain').splitlines(),
                          revision=git('rev-parse', 'HEAD'))
    runtime.update(environment=environment(), sources={n: sha256(ROOT/n) for n in SOURCES+V1_SOURCES},
                   command=' '.join([Path(sys.executable).name, '-m', 'experiments.locality_adaptation_v2',
                                     *sys.argv[1:]]),
                   seed_policy=f'SeedSequence([{ROOT_SEED}, stage, world, part]); timing appends d',
                   complexity=complexity_record(),
                   flops='not measured; not inferred from wall time or parameter counts')
    path.write_text(json.dumps(canonical(runtime), indent=2, sort_keys=True)+'\n')


# ---------------------------------------------------------------- phases

def development(out_dir):
    started, cpu = time.perf_counter(), time.process_time()
    first, meters, evaluator = run_stage(STAGES['development'], DEV_WORLDS)
    second, _, _ = run_stage(STAGES['development'], DEV_WORLDS)
    identical = encoded(first) == encoded(second)
    report = dict(first, stage=STAGES['development'], root_seed=ROOT_SEED, fold_counts=fold_counts(),
                  repeated_run_byte_identical=identical,
                  descriptive=analyse(first['rows'], DEV_WORLDS) if first['status'] == 'complete' else None,
                  oracle_summary=oracle_summary(first['worlds']),
                  note='development is for implementation and correctness only; no threshold here is a gate')
    (out_dir/f'{PREFIX}-development.json').write_text(encoded(report))
    record_runtime(out_dir, 'development', dict(
        wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu,
        scope='two complete development passes (determinism check) and JSON write; fit timing from pass 1',
        fit_stage_timing=meter_record(meters), evaluator_timing=meter_record({'evaluator': evaluator}),
        solver_failures=int(first['failure'] is not None)))
    print(json.dumps(dict(status=first['status'], failure=first['failure'],
                          repeated_run_byte_identical=identical), indent=2))
    if first['status'] != 'complete' or not identical:
        raise SystemExit('development gate failed')


def run_tests():
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_locality_adaptation_v2',
                            'tests.test_locality_adaptation'], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise SystemExit(tests.stderr)
    lines = [line for line in tests.stderr.strip().splitlines() if line.strip() and set(line) != {'-'}]
    return ' / '.join(lines[-2:])


def freeze(out_dir):
    dev = json.loads((out_dir/f'{PREFIX}-development.json').read_text())
    if dev['status'] != 'complete' or not dev['repeated_run_byte_identical']:
        raise SystemExit('development gate not satisfied')
    v1_manifest = json.loads((out_dir/'locality-adaptation-v1-manifest.json').read_text())
    for name in V1_SOURCES:
        if sha256(ROOT/name) != v1_manifest['hashes'][name]:
            raise SystemExit(f'v1 source {name} differs from the v1 manifest')
    manifest = dict(
        scope='locality adaptation v2 confirmation', contract='docs/locality-adaptation-v2-contract.md',
        hashes={name: sha256(ROOT/name) for name in SOURCES},
        v1_hashes={name: sha256(ROOT/name) for name in V1_SOURCES},
        v1_manifest_sha256=sha256(out_dir/'locality-adaptation-v1-manifest.json'),
        development_report_sha256=sha256(out_dir/f'{PREFIX}-development.json'),
        revision_at_freeze=git('rev-parse', 'HEAD'),
        correctness_tests=run_tests(),
        seeds=dict(root=ROOT_SEED, stages=STAGES, entropy='[root, stage, world, part] (+ [d] for timing)',
                   parts={0: 'template coefficients, per block 3 signal then 5 texture normals',
                          1: 'd x d Gaussian for QR', 2: 'unknown shift integers(1, b)',
                          3: 'c (256,k), g1 (256,k), g2 (256,k), e1 (256,d), e2 (256,d)'},
                   loop_order='world > regime > budget > evaluator all-n fits > arm'),
        development_worlds=DEV_WORLDS, confirmation_worlds=CONFIRM_WORLDS,
        dimensions=dict(d=D, k=RANK, b=B, phases=PHASES, folds=FOLDS), budgets=BUDGETS,
        primary_budgets=PRIMARY_BUDGETS, regimes=REGIMES, conditions=CONDITIONS, fold_counts=fold_counts(),
        arms={'cv_refit_cca': '4 folds x 8 phases on 3n/4, CV score select, refit selected phase on all n',
              'fixed_cca_all': 'all n, phase 0', 'global_cca_all': 'all n, global rank k',
              'holdout_refit_cca': '8 phases on fold-0 complement, fold-0 score select, refit on all n',
              'holdout_cca': 'v1 adaptive_cca: 8 phases on fold-0 complement, fold-0 select, no refit',
              'oracle_structural_phase_cca_all': 'all n at the evaluator smallest population-optimal '
                                                 'phase; structural reference, not a learner or ceiling'},
        conventions=dict(estimators='v1 module unchanged (ridge, ties, orientation, projectors, score tie)',
                         fold='one-based index i mod 4', reuse='none across folds, arms or budgets',
                         evaluator='eight all-n phase fits per (world, regime, budget), diagnostic only'),
        decision_rule=dict(candidate=CANDIDATE, comparator=COMPARATOR, claims=CLAIMS, confidence=CONFIDENCE,
                           test='one-sided paired Student-t lower bound over 64 worlds; all four required'),
        environment=environment())
    (out_dir/f'{PREFIX}-manifest.json').write_text(encoded(manifest))
    print('Manifest written. Commit it with the source before --phase confirm.')


def check_frozen(out_dir):
    path = out_dir/f'{PREFIX}-manifest.json'
    manifest = json.loads(path.read_text())
    for name, digest in {**manifest['hashes'], **manifest['v1_hashes']}.items():
        if sha256(ROOT/name) != digest:
            raise SystemExit(f'{name} changed after freezing')
    tracked = [str(path.relative_to(ROOT)), *manifest['hashes'], *manifest['v1_hashes']]
    if git('status', '--porcelain', '--', *tracked):
        raise SystemExit('manifest and frozen sources must be committed before confirmation')
    return manifest


def confirmation(out_dir):
    manifest = check_frozen(out_dir)
    started, cpu = time.perf_counter(), time.process_time()
    result, meters, evaluator = run_stage(STAGES['confirmation'], CONFIRM_WORLDS)
    fit_seconds = time.perf_counter()-started
    analysis = analyse(result['rows'], CONFIRM_WORLDS) if result['status'] == 'complete' else None
    report = dict(result, stage=STAGES['confirmation'], root_seed=ROOT_SEED,
                  manifest_sha256=sha256(out_dir/f'{PREFIX}-manifest.json'),
                  analysis=analysis, oracle_summary=oracle_summary(result['worlds']))
    (out_dir/f'{PREFIX}-results.json').write_text(encoded(report))
    record_runtime(out_dir, 'confirmation', dict(
        wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu,
        generation_fit_metric_seconds=fit_seconds,
        scope='single process: generation, oracle diagnostics, evaluator fits, candidate fits, metrics, '
              'analysis and JSON write',
        fit_stage_timing=meter_record(meters), evaluator_timing=meter_record({'evaluator': evaluator}),
        solver_failures=int(result['failure'] is not None),
        independent_pairs=dict(per_world_regime=max(BUDGETS), per_budget={str(n): n for n in BUDGETS},
                               worlds=len(CONFIRM_WORLDS), regimes=len(REGIMES)),
        frozen_revision=manifest['revision_at_freeze']))
    if result['status'] != 'complete':
        raise SystemExit(f"scoring stopped: {result['failure']}")
    print(json.dumps({k: {f: v[f] for f in ('mean_improvement', 'lower_bound', 'threshold', 'passed')}
                      for k, v in analysis['primary'].items()}, indent=2))
    print('overall pass:', analysis['overall_pass'])


# ---------------------------------------------------------------- timing panel

def timing_fit(arm, x, y, b, meter):
    """Deployable state only: (kind, phase, weights or basis)."""
    if arm == 'cv_refit_cca':
        chosen, w, _ = fit_cv_refit(x, y, b, meter)
        return 'local', chosen, w
    if arm == 'holdout_refit_cca':
        chosen, w, _ = fit_holdout_refit(x, y, b, meter)
        return 'local', chosen, w
    if arm == 'fixed_cca_all':
        return 'local', 0, fit_local(x, y, 0, 'cca', b, meter)
    return 'global', None, orthonormal_basis(fit_global(x, y, 'cca', x.shape[1]//b, meter))


def timing_job(config):
    use_v2_seed()
    d, regime, arm = config
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    world = make_world(STAGES['timing'], 0, d=d, extra=(d,))
    h1, h2 = views(world, regime, base_stream(STAGES['timing'], 0, d=d, n=TIMING_N, extra=(d,)))
    fits = []
    for repeat in range(REPEATS+1):
        meter = Meter()
        wall, cpu = time.perf_counter(), time.process_time()
        model = timing_fit(arm, h1, h2, B, meter)
        record = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                      stage_seconds=dict(meter.seconds))
        if repeat:
            fits.append(record)
        else:
            warmup = record
    counts = dict(meter.counts)
    inference = {}
    for name, h in (('single_input', h1[0]), ('batch_256', h1)):
        infer(model, h)
        blocks = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            for _ in range(CALLS):
                infer(model, h)
            blocks.append((time.perf_counter()-start)/CALLS)
        inference[name] = dict(per_call_seconds=blocks, median=float(np.median(blocks)),
                               iqr=[float(np.percentile(blocks, 25)), float(np.percentile(blocks, 75))])
    tracemalloc.start()
    timing_fit(arm, h1, h2, B, Meter())
    peak_fit = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    infer(model, h1)
    peak_infer = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    kind, phase, w = model
    walls = [f['wall_seconds'] for f in fits]
    return dict(d=d, k=d//B, b=B, n=TIMING_N, regime=regime, arm=arm, warmup=warmup, repeats=fits,
                fit_median_seconds=float(np.median(walls)),
                fit_iqr_seconds=[float(np.percentile(walls, 25)), float(np.percentile(walls, 75))],
                work_per_fit=counts, selected_phase=phase, inference=inference,
                persistent_bytes=dict(weights=int(w.nbytes), metadata=3*8 if kind == 'local' else 0),
                peak_tracemalloc_bytes=dict(fit=peak_fit, batch_256_inference=peak_infer,
                                            scope='NumPy/Python allocations only; LAPACK workspace excluded'),
                rss_kib=dict(after_imports_and_spawn=baseline,
                             peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             scope='fresh spawned worker high-water RSS, Linux KiB; includes interpreter'),
                complexity=complexity_record(d))


def timing(out_dir):
    check_frozen(out_dir)
    started, cpu = time.perf_counter(), time.process_time()
    context = multiprocessing.get_context('spawn')
    configs = [(d, regime, arm) for d in TIMING_DIMS for regime in TIMING_REGIMES for arm in TIMING_ARMS]
    records = []
    for config in configs:
        with context.Pool(1) as pool:
            records.append(pool.apply(timing_job, (config,)))
        print('  timed', config, flush=True)
    with context.Pool(1) as pool:
        idle = pool.apply(idle_job, (None,))
    for r in records:
        r['amortized_seconds'] = {str(q): r['fit_median_seconds']+q*r['inference']['single_input']['median']
                                  for q in (1, 100, 10000)}
    record_runtime(out_dir, 'timing_panel', dict(
        wall_seconds=time.perf_counter()-started, parent_cpu_seconds=time.process_time()-cpu,
        worker_cpu_note='per-fit worker CPU seconds are recorded per repeat; parent CPU excludes workers',
        idle_worker_rss_kib=idle, records=records, solver_failures=0,
        inference_calls=len(records)*2*(1+REPEATS*CALLS),
        protocol=dict(stage=STAGES['timing'], world=0, warmup_fits=1, repeats=REPEATS,
                      inference_blocks=REPEATS, calls_per_block=CALLS, inference_warmup_calls=1,
                      inference_scope='encode + reconstruct; output allocation included; generation excluded',
                      amortization='fit median + Q x single-input inference median; estimate, not measured',
                      note='k = d/8 changes with d: not fixed-rank scaling; stage-2 data never scored')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'freeze', 'confirm', 'timing'), required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    use_v2_seed()
    dict(dev=development, freeze=freeze, confirm=confirmation, timing=timing)[args.phase](args.out_dir)


if __name__ == '__main__':
    main()
