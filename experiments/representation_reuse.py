"""Representation reuse v1: development, manifest freeze, confirmation, timing panel.

Usage (commit the manifest and source between freeze and confirm):
  .venv/bin/python -m unittest tests.test_representation_reuse -v
  .venv/bin/python -m experiments.representation_reuse --phase dev
  .venv/bin/python -m experiments.representation_reuse --phase freeze
  git commit ...
  .venv/bin/python -m experiments.representation_reuse --phase confirm
  .venv/bin/python -m experiments.representation_reuse --phase timing
Contract: docs/representation-reuse-v1-contract.md
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

from experiments.locality_adaptation import canonical, environment, git, idle_job, infer, interval, sha256
from src.locality_adaptation import B, D, PHASES, RANK, Meter, ScoringFailure, phase_floors
from src.locality_adaptation_v2 import FOLDS
from src.representation_reuse import (ANGLES, ARMS, CHECKPOINTS, CONDITIONS, FRESH_ARMS,
    GLOBAL_ACQUISITION_ARMS, LOCAL_ACQUISITION_ARMS, PRIMARY_SOURCE, PRIMARY_TARGET, ROOT_SEED,
    SOURCE_ARMS, SOURCE_BUDGETS, STREAM, TARGET_BUDGETS, acquire_global, acquire_local, check_projector,
    dense_basis, drifted_signal, make_learner, make_world, metrics, noise_covariances, observe,
    signal_covariance, stream)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'representation-reuse-v1'
DEV_WORLDS, CONFIRM_WORLDS = tuple(range(8)), tuple(range(64))
STAGES = dict(development=0, confirmation=1, timing=2)
# (name, angle, measure, first, second, margin): D = measure(first) - measure(second), positive favours reuse
CLAIMS = (('H1_stable_reuse_preserves_signal', 0, 'L', 'fresh_local_cv', 'frozen_local', .02),
          ('H2_stable_reuse_reconstructs_better', 0, 'E', 'fresh_local_cv', 'frozen_local', .02),
          ('H3_updated_weights_beat_frozen', 30, 'E', 'frozen_local', 'phase_reuse_target', .02),
          ('H4_reused_grouping_noninferior', 30, 'E', 'fresh_local_cv', 'phase_reuse_target', -.02))
CONFIDENCE = 1-.05/len(CLAIMS)
FROZEN = ('frozen_local', 'frozen_global')
SOURCES = ('docs/representation-reuse-v1-contract.md', 'src/representation_reuse.py',
           'experiments/representation_reuse.py', 'tests/test_representation_reuse.py', 'requirements.txt')
V1_SOURCES = ('src/locality_adaptation.py', 'experiments/locality_adaptation.py',
              'tests/test_locality_adaptation.py', 'docs/locality-adaptation-v1-contract.md')
V2_SOURCES = ('docs/locality-adaptation-v2-contract.md', 'src/locality_adaptation_v2.py',
              'experiments/locality_adaptation_v2.py', 'tests/test_locality_adaptation_v2.py')
TESTS = ('tests.test_representation_reuse', 'tests.test_locality_adaptation_v2', 'tests.test_locality_adaptation')
TIMING_DIMS, TIMING_ANGLES, TIMING_ENDPOINTS = (32, 128, 512), (0, 30), (32, 256)
REPEATS, CALLS, QUERIES, EPISODES = 7, 100, (1, 100, 10000), (1, 10, 100)
COLUMNS = ('world', 'angle', 'source_budget', 'target_budget', 'arm', 'defined', 'phase', 'L', 'E',
           'projected_noise', 'source_L', 'source_E', 'forgetting_L', 'forgetting_E')


def encoded(value, compact=False):
    if compact:
        return json.dumps(canonical(value), sort_keys=True, separators=(',', ':'))+'\n'
    return json.dumps(canonical(value), indent=2, sort_keys=True)+'\n'


# ---------------------------------------------------------------- scientific run

def evaluate(basis, s_target, s_source, noises, baseline=None):
    target, source = metrics(basis, s_target, noises), metrics(basis, s_source, noises)
    row = dict(L=target['signal_loss'], E=target['nmse'], projected_noise=target['projected_noise'],
               source_L=source['signal_loss'], source_E=source['nmse'])
    if baseline is not None:
        row.update(forgetting_L=row['source_L']-baseline['signal_loss'],
                   forgetting_E={c: row['source_E'][c]-baseline['nmse'][c] for c in CONDITIONS})
    values = [row['L'], row['source_L'], *row['E'].values(), *row['source_E'].values()]
    if not np.all(np.isfinite(values)):
        raise ScoringFailure('nonfinite metric')
    return row


def scorer(s_target, s_source, noises, evaluator, baseline=None):
    def score(basis):
        with evaluator.time('metrics'):
            return evaluate(basis, s_target, s_source, noises, baseline)
    return score


def branch_rows(arm, learner, h1, h2, base, evaluate_basis, d):
    """One adaptation branch: m=0 row, then one target pair at a time, solving only at checkpoints."""
    rows = []
    def emit(m):
        basis = dense_basis(learner.deploy(), d)
        check_projector(basis, d//B)
        rows.append(dict(base, arm=arm, target_budget=m, defined=True, phase=learner.deploy()[1],
                         **evaluate_basis(basis)))
    if learner.defined_at_zero:
        emit(0)
    else:
        rows.append(dict(base, arm=arm, target_budget=0, defined=False, phase=None, reason='no target pairs'))
    for i in range(STREAM):
        if arm not in FROZEN:
            learner.observe(h1[i], h2[i])
        if i+1 in CHECKPOINTS:
            if arm not in FROZEN:
                learner.checkpoint()
            emit(i+1)
    return rows


def run_world(stage, w, meters, evaluator):
    world = make_world(stage, w)
    d, k = world['d'], world['d']//B
    xs, ys = observe(world, world['u'], stream(stage, w, 3))
    target_draws = stream(stage, w, 4)
    noises = noise_covariances(world)
    s_source = signal_covariance(world, world['u'])
    sources, info = {}, dict(world=w, true_phase=world['phase'], source={}, angles={})
    for s in SOURCE_BUDGETS:
        local = acquire_local(xs[:s], ys[:s], meter=meters['source_local'])
        glob = acquire_global(xs[:s], ys[:s], meter=meters['source_global'])
        local_basis_ = dense_basis(('local', local['phase'], local['weights']), d)
        sources[s] = dict(local=local, glob=glob,
                          baseline=dict(local=metrics(local_basis_, s_source, noises),
                                        glob=metrics(glob['basis'], s_source, noises)))
        info['source'][str(s)] = dict(phase=local['phase'], phase_correct=local['phase'] == world['phase'],
                                      cv_scores=local['scores'],
                                      local_source_risk=sources[s]['baseline']['local'],
                                      global_source_risk=sources[s]['baseline']['glob'])
    rows = []
    for theta in ANGLES:
        signal = drifted_signal(world, theta)
        h1, h2 = observe(world, signal, target_draws)
        s_target = signal_covariance(world, signal)
        with evaluator.time('oracle_diagnostics'):
            floors = phase_floors(s_target)
            oracle = metrics(world['rotation']@signal, s_target, noises)
            source_oracle = metrics(world['rotation']@world['u'], s_target, noises)
        info['angles'][str(theta)] = dict(
            target_phase_floors=floors, true_phase_floor=floors[world['phase']],
            floor_at_source_phase={str(s): floors[sources[s]['local']['phase']] for s in SOURCE_BUDGETS},
            controls=dict(zero_output_E={c: 1. for c in CONDITIONS},
                          raw_input_E={c: float(np.trace(n)/k) for c, n in noises.items()},
                          oracle_target_projector=oracle, frozen_oracle_source_projector=source_oracle))
        if theta == 0:
            info['source_phase_floors'] = floors
        try:
            for s in SOURCE_BUDGETS:
                src = sources[s]
                for arm in SOURCE_ARMS:
                    baseline = src['baseline']['local' if arm in LOCAL_ACQUISITION_ARMS else 'glob']
                    learner = make_learner(arm, src['local'], src['glob'], d, meters[arm])
                    rows += branch_rows(arm, learner, h1, h2, dict(world=w, angle=theta, source_budget=s),
                                        scorer(s_target, s_source, noises, evaluator, baseline), d)
            for arm in FRESH_ARMS:
                learner = make_learner(arm, None, None, d, meters[arm])
                rows += branch_rows(arm, learner, h1, h2, dict(world=w, angle=theta, source_budget=None),
                                    scorer(s_target, s_source, noises, evaluator), d)
        except ScoringFailure as error:
            raise ScoringFailure(dict(world=w, angle=theta, error=str(error))) from None
    return info, rows


def run_stage(stage, worlds):
    meters = {name: Meter() for name in ('source_local', 'source_global', *ARMS)}
    evaluator = Meter()
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


def table(rows):
    """Columnar rows; noise-dependent values are listed in CONDITIONS order."""
    def cell(row, name):
        value = row.get(name)
        return [value[c] for c in CONDITIONS] if isinstance(value, dict) else value
    return dict(columns=COLUMNS, conditions=CONDITIONS, values=[[cell(r, c) for c in COLUMNS] for r in rows])


# ---------------------------------------------------------------- analysis

MEASURES = dict(L=lambda r: r['L'], E=lambda r: r['E']['in_distribution'],
                E_low_noise=lambda r: r['E']['low_noise'], E_texture_16=lambda r: r['E']['texture_16'],
                projected_noise=lambda r: r['projected_noise']['in_distribution'],
                source_L=lambda r: r['source_L'], source_E=lambda r: r['source_E']['in_distribution'],
                forgetting_L=lambda r: r.get('forgetting_L'),
                forgetting_E=lambda r: (r.get('forgetting_E') or {}).get('in_distribution'))
CONTRASTS = (('phase_reuse_target', 'frozen_local'), ('phase_reuse_target', 'fresh_local_cv'),
             ('frozen_local', 'fresh_local_cv'), ('pooled_local', 'phase_reuse_target'),
             ('pooled_local', 'frozen_local'), ('pooled_global', 'phase_reuse_target'),
             ('pooled_global', 'pooled_local'), ('frozen_global', 'frozen_local'),
             ('fresh_global', 'fresh_local_cv'), ('pooled_global', 'fresh_global'))


def analyse(rows, worlds_out):
    worlds = [i['world'] for i in worlds_out]
    get = {(r['world'], r['angle'], r['source_budget'], r['arm'], r['target_budget']): r
           for r in rows if r['defined']}
    key_s = lambda arm, s: None if arm in FRESH_ARMS else s

    def window(angle, arm, measure, s=PRIMARY_SOURCE):
        return np.array([np.mean([MEASURES[measure](get[w, angle, key_s(arm, s), arm, m]) for m in PRIMARY_TARGET])
                         for w in worlds])

    q = stats.t.ppf(CONFIDENCE, len(worlds)-1)
    primary = {}
    for name, angle, measure, first, second, margin in CLAIMS:
        diff = window(angle, first, measure)-window(angle, second, measure)
        se = float(np.std(diff, ddof=1)/np.sqrt(len(diff)))
        bound = float(diff.mean()) if se == 0 else float(diff.mean()-q*se)
        primary[name] = dict(angle=angle, score=measure, difference=f'{measure}({first}) - {measure}({second})',
                             margin=margin, mean_difference=float(diff.mean()), se=se, t_quantile=float(q),
                             confidence=CONFIDENCE, lower_bound=bound, zero_variance=se == 0,
                             passed=bool(bound > margin), per_world_difference=diff)

    measures_for = lambda arm: [m for m in MEASURES if not (m.startswith('forgetting') and arm in FRESH_ARMS)]
    window_scores = {str(s): {str(a): {arm: {m: interval(window(a, arm, m, s)) for m in measures_for(arm)}
                                       for arm in ARMS} for a in ANGLES} for s in SOURCE_BUDGETS}
    contrasts = {str(s): {str(a): {f'{x}_vs_{y}': {m: interval(window(a, y, m, s)-window(a, x, m, s))
                                                   for m in ('L', 'E', 'source_E')}
                                   for x, y in CONTRASTS} for a in ANGLES} for s in SOURCE_BUDGETS}
    curves = {str(s): {str(a): {arm: {m: {str(n): interval([MEASURES[m](get[w, a, key_s(arm, s), arm, n])
                                                             for w in worlds])
                                          for n in TARGET_BUDGETS if (worlds[0], a, key_s(arm, s), arm, n) in get}
                                      for m in ('L', 'E', 'projected_noise', 'source_E')}
                                for arm in ARMS} for a in ANGLES} for s in SOURCE_BUDGETS}
    source_selection = {str(s): dict(
        fraction_correct=float(np.mean([i['source'][str(s)]['phase_correct'] for i in worlds_out])))
        for s in SOURCE_BUDGETS}
    truth = {i['world']: i['true_phase'] for i in worlds_out}
    target_selection = {str(a): {str(n): float(np.mean([get[w, a, None, 'fresh_local_cv', n]['phase'] == truth[w]
                                                        for w in worlds])) for n in CHECKPOINTS} for a in ANGLES}
    return dict(primary=primary, overall_pass=all(p['passed'] for p in primary.values()),
                window_scores=window_scores, contrasts=contrasts, learning_curves=curves,
                source_phase_selection=source_selection,
                fresh_local_cv_fraction_true_phase=target_selection,
                notes=['Window scores are per-world means over m = 16, 32, 64; primary claims use s = 256.',
                       'Contrasts `x_vs_y` are measure(y) - measure(x): positive favours x. Descriptive 95% '
                       't intervals only; no claim may be rescued by a contrast.',
                       'Angles share target draws and are not independent replicates.',
                       'Source-arm curves at m are trained on s + m pairs; fresh arms on m pairs only.',
                       'One-sided paired Student-t bounds over worlds; coverage is approximate.'])


def oracle_summary(worlds_out):
    out = {}
    for a in ANGLES:
        infos = [i['angles'][str(a)] for i in worlds_out]
        controls = [i['controls'] for i in infos]
        out[str(a)] = dict(
            true_phase_floor=interval([i['true_phase_floor'] for i in infos]),
            floor_at_source_phase={str(s): interval([i['floor_at_source_phase'][str(s)] for i in infos])
                                   for s in SOURCE_BUDGETS},
            raw_input_E={c: float(np.mean([x['raw_input_E'][c] for x in controls])) for c in CONDITIONS},
            oracle_target_projector_E={c: float(np.mean([x['oracle_target_projector']['nmse'][c]
                                                         for x in controls])) for c in CONDITIONS},
            frozen_oracle_source_projector=dict(
                L=float(np.mean([x['frozen_oracle_source_projector']['signal_loss'] for x in controls])),
                E={c: float(np.mean([x['frozen_oracle_source_projector']['nmse'][c] for x in controls]))
                   for c in CONDITIONS}))
    return out


# ---------------------------------------------------------------- computation records

def local_acquisition_counts(s):
    fits = FOLDS*PHASES
    return dict(local_fits=fits+1, refits=1, candidate_scores=fits, local_solves=1, source_pairs=s,
                fit_row_exposures=fits*(s-s//FOLDS)+s, validation_row_exposures=PHASES*s)


def global_acquisition_counts(s):
    return dict(global_fits=1, global_solves=1, source_pairs=s, fit_row_exposures=s)


def cv_counts(m):
    return dict(local_fits=FOLDS*PHASES+1, candidate_scores=FOLDS*PHASES, fit_row_exposures=FOLDS*PHASES*(m-m//FOLDS)+m,
                validation_row_exposures=PHASES*m, cv_runs=1)


def charged_counts(arm, s, m, path='trajectory'):
    """Standalone per-branch charge: source history once, then target work through checkpoint m."""
    out = Counter()
    if arm in LOCAL_ACQUISITION_ARMS:
        out.update(local_acquisition_counts(s))
    if arm in GLOBAL_ACQUISITION_ARMS:
        out.update(global_acquisition_counts(s))
    if arm in FROZEN or m == 0:
        return dict(out, target_pairs=0)
    solved = [c for c in CHECKPOINTS if c <= m] if path == 'trajectory' else [m]
    out['target_pairs'] = m
    if arm == 'fresh_local_cv':
        for c in solved:
            out.update(cv_counts(c))
        out['target_pair_exposures_in_fits'] = out['fit_row_exposures']
    else:
        out['checkpoint_solves'] = len(solved)
    return dict(out)


def charged_record():
    return {arm: {str(s) if arm in SOURCE_ARMS else 'none': {
        path: {str(m): charged_counts(arm, s, m, path) for m in TARGET_BUDGETS}
        for path in ('trajectory', 'isolated_endpoint')} for s in (SOURCE_BUDGETS if arm in SOURCE_ARMS else (0,))}
        for arm in ARMS}


def state_bytes_record(d=D, s=PRIMARY_SOURCE):
    """Actual array bytes held by each learner at transition, after 256 updates, and deployed."""
    world = make_world(STAGES['development'], 0)
    xs, ys = observe(world, world['u'], stream(0, 0, 3))
    h1, h2 = observe(world, drifted_signal(world, 30), stream(0, 0, 4))
    local, glob = acquire_local(xs[:s], ys[:s]), acquire_global(xs[:s], ys[:s])
    out = {}
    for arm in ARMS:
        learner = make_learner(arm, local, glob, d)
        transition = learner.state_bytes()
        if arm not in FROZEN:
            for i in range(STREAM):
                learner.observe(h1[i], h2[i])
            learner.checkpoint()
        kind, phase, w = learner.deploy()
        out[arm] = dict(transition=transition, after_256_updates=learner.state_bytes(),
                        deployed=dict(weights=int(w.nbytes), metadata=3*8 if kind == 'local' else 0),
                        acquisition_workspace=dict(
                            raw_source_pairs=2*s*d*8 if arm in SOURCE_ARMS else 0,
                            local_block_sums=int(local['c'].nbytes+local['a'].nbytes)
                            if arm in LOCAL_ACQUISITION_ARMS else 0,
                            global_full_sums=int(glob['c'].nbytes+glob['a'].nbytes)
                            if arm in GLOBAL_ACQUISITION_ARMS else 0,
                            note='raw source pairs are released at the transition; no learner holds them'))
    return out


def complexity_record(d=D, b=B):
    k = d//b
    return dict(
        orders=dict(local_source_acquisition=f'{FOLDS}x{PHASES} local fits on 3s/4 + 1 refit: '
                                             'O(FK(3s/4)db + FKdb^2 + sdb + db^2) + O(Ksd) scoring',
                    global_source_acquisition='O(s d^2 + d^3)', local_target_update='O(db) per pair',
                    local_checkpoint_solve='O(db^2)', global_target_update='O(d^2) per pair',
                    global_checkpoint_solve='O(d^3)', fresh_local_cv='one v2 CV+refit run per checkpoint on m',
                    local_inference='O(d)', global_inference='O(dk)', optimizer_state=0, tuning_trials=0,
                    gradient_steps=0, solver='dense direct Cholesky + symmetric eigh (LAPACK syevd)'),
        bytes=dict(local_block_sums=2*k*b*b*8, global_full_sums=2*d*d*8, deployed_local=d*8+3*8,
                   deployed_global=d*k*8, raw_pairs_per_q='2 q d 8',
                   eigh_workspace_estimate=dict(local=(1+6*b+2*b*b)*8+(3+5*b)*4, global_=(1+6*d+2*d*d)*8+(3+5*d)*4,
                                                label='analytical LAPACK dsyevd lwork/liwork estimate, not measured')))


def meter_record(meters):
    return {name: dict(seconds=dict(m.seconds), counts=dict(m.counts)) for name, m in meters.items()}


def record_runtime(out_dir, phase, record):
    path = out_dir/f'{PREFIX}-runtime.json'
    runtime = json.loads(path.read_text()) if path.exists() else {}
    runtime[phase] = dict(record, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          rss_scope='cumulative parent-process high-water RSS (Linux KiB), not per-arm memory',
                          dirty_paths=git('status', '--porcelain').splitlines(), revision=git('rev-parse', 'HEAD'))
    runtime.update(environment=environment(),
                   sources={n: sha256(ROOT/n) for n in SOURCES+V1_SOURCES+V2_SOURCES},
                   command=' '.join([Path(sys.executable).name, '-m', 'experiments.representation_reuse',
                                     *sys.argv[1:]]),
                   seed_policy=f'SeedSequence([{ROOT_SEED}, stage, world, part]); timing appends d',
                   complexity=complexity_record(), charged_counts_per_branch=charged_record(),
                   state_bytes_d32=state_bytes_record(),
                   flops='not measured; not inferred from wall time or parameter counts')
    path.write_text(encoded(runtime))


def stage_record(result, meters, evaluator, worlds, started, cpu):
    return dict(wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu,
                executed_timing_and_counts=meter_record(meters),
                execution_note='source_local/source_global meters are actual shared executions (one per world '
                               'and source budget); each source arm is charged the full standalone acquisition '
                               'in charged_counts_per_branch. Fresh arms execute once per world and angle.',
                evaluator_timing=meter_record({'evaluator': evaluator}),
                solver_failures=int(result['failure'] is not None),
                independent_pairs=dict(source_per_world=STREAM, target_per_world=STREAM, worlds=len(worlds),
                                       note='target draws are shared across angles'))


# ---------------------------------------------------------------- phases

def development(out_dir):
    started, cpu = time.perf_counter(), time.process_time()
    first, meters, evaluator = run_stage(STAGES['development'], DEV_WORLDS)
    second, _, _ = run_stage(STAGES['development'], DEV_WORLDS)
    identical = encoded(first) == encoded(second)
    complete = first['status'] == 'complete'
    report = dict(status=first['status'], failure=first['failure'], worlds=first['worlds'],
                  rows=table(first['rows']), stage=STAGES['development'], root_seed=ROOT_SEED,
                  repeated_run_byte_identical=identical,
                  descriptive=analyse(first['rows'], first['worlds']) if complete else None,
                  oracle_summary=oracle_summary(first['worlds']) if complete else None,
                  note='development is for implementation and correctness only; no threshold here is a gate')
    (out_dir/f'{PREFIX}-development.json').write_text(encoded(report, compact=True))
    record_runtime(out_dir, 'development', dict(
        stage_record(first, meters, evaluator, DEV_WORLDS, started, cpu),
        scope='two complete development passes (determinism check) and JSON write; meters from pass 1'))
    print(json.dumps(dict(status=first['status'], failure=first['failure'],
                          repeated_run_byte_identical=identical), indent=2))
    if not complete or not identical:
        raise SystemExit('development gate failed')


def run_tests():
    tests = subprocess.run([sys.executable, '-m', 'unittest', *TESTS], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise SystemExit(tests.stderr)
    lines = [line for line in tests.stderr.strip().splitlines() if line.strip() and set(line) != {'-'}]
    return ' / '.join(lines[-2:])


def check_dependencies(out_dir):
    v1 = json.loads((out_dir/'locality-adaptation-v1-manifest.json').read_text())
    v2 = json.loads((out_dir/'locality-adaptation-v2-manifest.json').read_text())
    for name in V1_SOURCES:
        if sha256(ROOT/name) != v1['hashes'][name]:
            raise SystemExit(f'v1 source {name} differs from the v1 manifest')
    for name in V2_SOURCES:
        if sha256(ROOT/name) != v2['hashes'][name]:
            raise SystemExit(f'v2 source {name} differs from the v2 manifest')
    return dict(v1_manifest_sha256=sha256(out_dir/'locality-adaptation-v1-manifest.json'),
                v2_manifest_sha256=sha256(out_dir/'locality-adaptation-v2-manifest.json'))


def freeze(out_dir):
    dev = json.loads((out_dir/f'{PREFIX}-development.json').read_text())
    if dev['status'] != 'complete' or not dev['repeated_run_byte_identical']:
        raise SystemExit('development gate not satisfied')
    manifest = dict(
        scope='representation reuse v1 confirmation', contract='docs/representation-reuse-v1-contract.md',
        hashes={name: sha256(ROOT/name) for name in SOURCES},
        dependency_hashes={name: sha256(ROOT/name) for name in V1_SOURCES+V2_SOURCES},
        **check_dependencies(out_dir),
        development_report_sha256=sha256(out_dir/f'{PREFIX}-development.json'),
        revision_at_freeze=git('rev-parse', 'HEAD'), correctness_tests=run_tests(), test_modules=TESTS,
        seeds=dict(root=ROOT_SEED, stages=STAGES, entropy='[root, stage, world, part] (+ [d] for timing)',
                   parts={0: 'templates: per block 3 signal then 5 texture normals (v1 recipe)',
                          1: 'reserved, never drawn', 2: 'source phase integers(0, 8)',
                          3: 'source stream c (256,k), g1 (256,k), g2 (256,k), e1 (256,d), e2 (256,d)',
                          4: 'target stream, same shapes and order', 5: 'drift directions (k, b) Gaussian'},
                   loop_order='world > source acquisition per s > angle > s > source arm > pair > '
                              'checkpoint; then fresh arms per angle'),
        development_worlds=DEV_WORLDS, confirmation_worlds=CONFIRM_WORLDS,
        dimensions=dict(d=D, k=RANK, b=B, phases=PHASES, folds=FOLDS), angles=ANGLES,
        source_budgets=SOURCE_BUDGETS, target_budgets=TARGET_BUDGETS, primary_source=PRIMARY_SOURCE,
        primary_target=PRIMARY_TARGET, conditions=CONDITIONS, arms=ARMS,
        decision_rule=dict(claims=CLAIMS, confidence=CONFIDENCE,
                           test='one-sided paired Student-t lower bound over 64 worlds; all four required'),
        environment=environment())
    (out_dir/f'{PREFIX}-manifest.json').write_text(encoded(manifest))
    print('Manifest written. Commit it with the source before --phase confirm.')


def check_frozen(out_dir):
    path = out_dir/f'{PREFIX}-manifest.json'
    manifest = json.loads(path.read_text())
    check_dependencies(out_dir)
    for name, digest in {**manifest['hashes'], **manifest['dependency_hashes']}.items():
        if sha256(ROOT/name) != digest:
            raise SystemExit(f'{name} changed after freezing')
    tracked = [str(path.relative_to(ROOT)), *manifest['hashes'], *manifest['dependency_hashes']]
    if git('status', '--porcelain', '--', *tracked):
        raise SystemExit('manifest and frozen sources must be committed before confirmation')
    return manifest


def confirmation(out_dir):
    manifest = check_frozen(out_dir)
    started, cpu = time.perf_counter(), time.process_time()
    result, meters, evaluator = run_stage(STAGES['confirmation'], CONFIRM_WORLDS)
    complete = result['status'] == 'complete'
    analysis = analyse(result['rows'], result['worlds']) if complete else None
    report = dict(status=result['status'], failure=result['failure'], worlds=result['worlds'],
                  rows=table(result['rows']), stage=STAGES['confirmation'], root_seed=ROOT_SEED,
                  manifest_sha256=sha256(out_dir/f'{PREFIX}-manifest.json'), analysis=analysis,
                  oracle_summary=oracle_summary(result['worlds']) if complete else None)
    (out_dir/f'{PREFIX}-results.json').write_text(encoded(report, compact=True))
    record_runtime(out_dir, 'confirmation', dict(
        stage_record(result, meters, evaluator, CONFIRM_WORLDS, started, cpu),
        scope='single process: generation, source acquisition, adaptation branches, evaluator, analysis, write',
        frozen_revision=manifest['revision_at_freeze']))
    if not complete:
        raise SystemExit(f"scoring stopped: {result['failure']}")
    print(json.dumps({k: {f: v[f] for f in ('mean_difference', 'lower_bound', 'margin', 'passed')}
                      for k, v in analysis['primary'].items()}, indent=2))
    print('overall pass:', analysis['overall_pass'])


# ---------------------------------------------------------------- timing panel

def timed(fn):
    wall, cpu = time.perf_counter(), time.process_time()
    out = fn()
    return out, dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)


def summary(records):
    walls = [r['wall_seconds'] for r in records]
    return dict(median=float(np.median(walls)), iqr=[float(np.percentile(walls, 25)), float(np.percentile(walls, 75))])


def repeat(fn):
    """One discarded warm-up then REPEATS fresh executions on identical data."""
    _, warmup = timed(fn)
    runs = [timed(fn) for _ in range(REPEATS)]
    return runs[-1][0], dict(warmup=warmup, repeats=[r for _, r in runs], **summary([r for _, r in runs]))


def timing_job(config):
    d, theta, arm = config
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stage, extra = STAGES['timing'], (d,)
    world = make_world(stage, 0, d=d, extra=extra)
    xs, ys = observe(world, world['u'], stream(stage, 0, 3, d=d, extra=extra))
    h1, h2 = observe(world, drifted_signal(world, theta), stream(stage, 0, 4, d=d, extra=extra))
    s = PRIMARY_SOURCE

    def acquire():
        meter = Meter()
        local = acquire_local(xs[:s], ys[:s], meter=meter) if arm in LOCAL_ACQUISITION_ARMS else None
        glob = acquire_global(xs[:s], ys[:s], meter=meter) if arm in GLOBAL_ACQUISITION_ARMS else None
        return local, glob, meter

    if arm in SOURCE_ARMS:
        (local, glob, acquisition_meter), acquisition = repeat(acquire)
        acquisition['work'] = dict(acquisition_meter.counts)
    else:
        local = glob = None
        acquisition = dict(median=0., note='fresh arms have no source acquisition', work={})

    def trajectory():
        meter = Meter()
        learner, marks, start = make_learner(arm, local, glob, d, meter), {}, time.perf_counter()
        for i in range(STREAM):
            if arm not in FROZEN:
                learner.observe(h1[i], h2[i])
            if i+1 in CHECKPOINTS:
                if arm not in FROZEN:
                    learner.checkpoint()
                learner.deploy()
                marks[str(i+1)] = time.perf_counter()-start
        return learner, marks, meter

    def endpoint(m):
        def run():
            meter = Meter()
            learner = make_learner(arm, local, glob, d, meter)
            if arm not in FROZEN:
                for i in range(m):
                    learner.observe(h1[i], h2[i])
                learner.checkpoint()
            return learner, meter
        return run

    _, warm = timed(trajectory)
    runs = [trajectory() for _ in range(REPEATS)]
    traj = dict(warmup_wall_seconds=warm['wall_seconds'],
                cumulative_seconds_at_checkpoint={c: [r[1][c] for r in runs] for c in runs[0][1]},
                median_cumulative_seconds={c: float(np.median([r[1][c] for r in runs])) for c in runs[0][1]},
                work=dict(runs[-1][2].counts))
    endpoints = {}
    for m in TIMING_ENDPOINTS:
        (learner_m, meter_m), record = repeat(endpoint(m))
        endpoints[str(m)] = dict(record, work=dict(meter_m.counts), state_bytes=learner_m.state_bytes())
    model = learner_m.deploy()
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
    if arm in SOURCE_ARMS:
        acquire()
    peak_acquisition = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    trajectory()
    peak_trajectory = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    infer(model, h1)
    peak_infer = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    kind, _, w = model
    return dict(d=d, k=d//B, b=B, s=s, theta=theta, arm=arm, acquisition=acquisition, trajectory=traj,
                endpoints=endpoints, inference=inference,
                deployed_bytes=dict(weights=int(w.nbytes), metadata=3*8 if kind == 'local' else 0),
                transition_state_bytes=make_learner(arm, local, glob, d).state_bytes(),
                peak_tracemalloc_bytes=dict(acquisition=peak_acquisition if arm in SOURCE_ARMS else None,
                                            trajectory=peak_trajectory, batch_256_inference=peak_infer,
                                            scope='NumPy/Python allocations in a separate untimed pass; '
                                                  'LAPACK workspace excluded; tracemalloc includes live inputs'),
                rss_kib=dict(after_imports_and_spawn=baseline, peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             scope='fresh spawned worker high-water RSS, Linux KiB; includes interpreter'),
                complexity=complexity_record(d))


def amortization(r):
    acquisition, single = r['acquisition']['median'], r['inference']['single_input']['median']
    fresh = r['arm'] in FRESH_ARMS
    out = {}
    for m in TIMING_ENDPOINTS:
        adaptation = r['endpoints'][str(m)]['median']
        out[str(m)] = dict(
            total_seconds={str(q): acquisition+adaptation+q*single for q in QUERIES},
            repeated_episodes={str(e): {str(q): acquisition+e*(adaptation+q*single) for q in QUERIES}
                               for e in EPISODES},
            total_pairs={str(e): e*m if fresh else PRIMARY_SOURCE+e*m for e in EPISODES})
    return out


def timing(out_dir):
    check_frozen(out_dir)
    started, cpu = time.perf_counter(), time.process_time()
    context = multiprocessing.get_context('spawn')
    configs = [(d, theta, arm) for d in TIMING_DIMS for theta in TIMING_ANGLES for arm in ARMS]
    records = []
    for config in configs:
        with context.Pool(1) as pool:
            records.append(pool.apply(timing_job, (config,)))
        print('  timed', config, flush=True)
    with context.Pool(1) as pool:
        idle = pool.apply(idle_job, (None,))
    for r in records:
        r['amortized_estimate_seconds'] = amortization(r)
    record_runtime(out_dir, 'timing_panel', dict(
        wall_seconds=time.perf_counter()-started, parent_cpu_seconds=time.process_time()-cpu,
        worker_cpu_note='per-execution worker CPU seconds are recorded per repeat; parent CPU excludes workers',
        idle_worker_rss_kib=idle, records=records, solver_failures=0,
        inference_calls=len(records)*2*(1+REPEATS*CALLS),
        protocol=dict(stage=STAGES['timing'], world=0, s=PRIMARY_SOURCE, warmup_executions=1, repeats=REPEATS,
                      trajectory='source checkpoint copy, 256 single-pair updates, solves at the 7 checkpoints; '
                                 'cumulative seconds read at each checkpoint',
                      endpoints='separate fresh executions for m in (32, 256): copy, m updates, one solve',
                      inference_blocks=REPEATS, calls_per_block=CALLS, inference_warmup_calls=1,
                      inference_scope='encode + reconstruct; output allocation included; generation excluded',
                      amortization='acquisition median + R x (endpoint median + Q x single-input median); '
                                   'arithmetic estimate from isolated endpoints, not measured episodes',
                      note='k = d/8 changes with d: not fixed-rank scaling; stage-2 data never scored')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'freeze', 'confirm', 'timing'), required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    dict(dev=development, freeze=freeze, confirm=confirmation, timing=timing)[args.phase](args.out_dir)


if __name__ == '__main__':
    main()
