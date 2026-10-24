"""Single-module plasticity v1: development, manifest freeze, confirmation, timing panel.

Usage (commit the manifest and source between freeze and confirm):
  .venv/bin/python -m unittest tests.test_single_module_plasticity -v
  .venv/bin/python -m experiments.single_module_plasticity --phase dev
  .venv/bin/python -m experiments.single_module_plasticity --phase freeze
  git commit ...
  .venv/bin/python -m experiments.single_module_plasticity --phase confirm
  .venv/bin/python -m experiments.single_module_plasticity --phase timing
Contract: docs/single-module-plasticity-v1-contract.md
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import copy
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

from experiments.locality_adaptation import canonical, environment, git, idle_job, interval, sha256
from src.locality_adaptation import B, D, PHASES, ScoringFailure, groups
from src.locality_adaptation_v2 import FOLDS
from src.single_module_plasticity import (ADAPTING_ARMS, ARMS, B_CHECKPOINTS, BLOCKS, BRANCHES, DIRECT_ARMS,
    DRIFT_DEGREES, ETA, LAYERED_ARMS, M, PAIRS_PER_BATCH, PRIMARY_WINDOW, PROFILES, RANK, RETURN_CHECKPOINTS,
    RINGS, ROOT_SEED, SCHUR_INPUTS, SECTORS, STREAM, UPDATES_PER_BATCH, WIDTHS, Meter2, Module, acquire,
    branch_signal, clean, copy_params, direct_reference, make_world, observe, parameter_distance,
    projector_output, schur_check, score, sector_indices, stream, warm_up)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-plasticity-v1'
DEV_WORLDS, CONFIRM_WORLDS = tuple(range(8)), tuple(range(64))
STAGES = dict(development=0, confirmation=1, timing=2)
CANDIDATE = 'graded_matched'
# (name, measure, first, margin): D = mean window measure(first) - measure(candidate); positive favours C
CLAIMS = (('H1_matching_beats_mean_routing_on_target', 'E_B', 'graded_mean', .01),
          ('H2_matching_beats_shifted_routing_on_target', 'E_B', 'graded_shifted', .01),
          ('H3_slower_inner_learning_preserves_source', 'E_A', 'uniform_matched', .01),
          ('H4_target_adaptation_noninferior_to_uniform', 'E_B', 'uniform_matched', -.02))
CONFIDENCE = 1-.05/len(CLAIMS)
PRIMARY_BRANCH = 'regional'
SOURCES = ('docs/single-module-plasticity-v1-contract.md', 'src/single_module_plasticity.py',
           'experiments/single_module_plasticity.py', 'tests/test_single_module_plasticity.py', 'requirements.txt')
DEPENDENCIES = dict(
    v1=('src/locality_adaptation.py', 'experiments/locality_adaptation.py', 'tests/test_locality_adaptation.py',
        'docs/locality-adaptation-v1-contract.md'),
    v2=('docs/locality-adaptation-v2-contract.md', 'src/locality_adaptation_v2.py',
        'experiments/locality_adaptation_v2.py', 'tests/test_locality_adaptation_v2.py'),
    reuse=('docs/representation-reuse-v1-contract.md', 'src/representation_reuse.py',
           'experiments/representation_reuse.py', 'tests/test_representation_reuse.py'))
MANIFESTS = dict(v1='locality-adaptation-v1-manifest.json', v2='locality-adaptation-v2-manifest.json',
                 reuse='representation-reuse-v1-manifest.json')
TESTS = ('tests.test_single_module_plasticity', 'tests.test_representation_reuse',
         'tests.test_locality_adaptation_v2', 'tests.test_locality_adaptation')
SETS = ('A', 'B')
COLUMNS = ('world', 'branch', 'episode', 'm', 'arm', 'defined', 'E_A', 'E_B', 'distortion_A', 'distortion_B',
           'sector_E_A', 'sector_E_B', 'gate_mean_A', 'gate_mean_B', 'gate_error_corr_B', 'param_change',
           'update_norm', 'schur_max_rel')
REPEATS, CALLS, QUERIES, TIMING_BATCH = 7, 20, (1, 100, 10000), 32


def encoded(value, compact=False):
    if compact:
        return json.dumps(canonical(value), sort_keys=True, separators=(',', ':'))+'\n'
    return json.dumps(canonical(value), indent=2, sort_keys=True)+'\n'


# ---------------------------------------------------------------- evaluator

def evaluation_sets(world, signal_b, ev_a, ev_b):
    """Noisy first views and clean signals; evaluator-only, never given to training or routing."""
    return dict(A=(observe(world, world['u'], ev_a)[0], clean(world, world['u'], ev_a)),
                B=(observe(world, signal_b, ev_b)[0], clean(world, signal_b, ev_b)))


def evaluate_layered(module, sets, phase, evaluator):
    """Fresh two-pass inference on each noisy and clean input; scoring uses evaluator truth only."""
    view = Module(module.p, None, module.policy, evaluator)
    row, worst = {}, 0.
    for name, (x, s) in sets.items():
        with evaluator.time('inference'):
            out, info = view.infer(x)
            out_clean, _ = view.infer(s)
        evaluator.counts['evaluator_inputs'] += 2*len(x)
        with evaluator.time('metrics'):
            noisy = score(out, s, phase)
            row[f'E_{name}'], row[f'sector_E_{name}'] = noisy['E'], noisy['sector_E']
            row[f'distortion_{name}'] = score(out_clean, s, phase)['E']
            row[f'gate_mean_{name}'] = info['g'].mean(0)
            if name == 'B':
                err = np.stack([np.sum((out[:, g]-s[:, g])**2, 1) for g in groups(module.p['phase'], D, B)], 1)
                row['gate_error_corr_B'] = float(np.corrcoef(info['g'][:, 0].ravel(), err.ravel())[0, 1])
        with evaluator.time('schur_reference'):
            worst = max(worst, schur_check(view, x[:SCHUR_INPUTS]))
    row['schur_max_rel'] = worst
    return row


def evaluate_direct(learner, sets, phase, evaluator):
    row = {}
    for name, (x, s) in sets.items():
        with evaluator.time('inference'):
            out, out_clean = projector_output(learner, x), projector_output(learner, s)
        evaluator.counts['evaluator_inputs'] += 2*len(x)
        noisy = score(out, s, phase)
        row[f'E_{name}'], row[f'sector_E_{name}'] = noisy['E'], noisy['sector_E']
        row[f'distortion_{name}'] = score(out_clean, s, phase)['E']
    return row


def controls(world, sets, signal_b):
    """Zero output, raw input and oracle signal projection on the same draws (expected 1, 6, 0.25)."""
    out = {}
    for name, signal in (('A', world['u']), ('B', signal_b)):
        x, s = sets[name]
        oracle = world['rotation']@signal
        out[name] = dict(zero=score(np.zeros_like(s), s, world['phase'])['E'],
                         raw=score(x, s, world['phase'])['E'],
                         oracle_projection=score(x@oracle@oracle.T, s, world['phase'])['E'])
    return out


# ---------------------------------------------------------------- scientific run

def layered_branch(arm, p_source, data, sets, world, meters, evaluator, base):
    profile, policy = LAYERED_ARMS[arm]
    module = Module(copy_params(p_source), profile, policy, meters[arm])
    rows, norms = [], np.zeros(RINGS)

    def emit(episode, m):
        row = evaluate_layered(module, sets, world['phase'], evaluator)
        rows.append(dict(base, episode=episode, m=m, arm=arm, defined=True, **row,
                         param_change=parameter_distance(module.p, p_source), update_norm=norms.copy()))

    emit('B', 0)
    for episode, (h1, h2), marks in (('B', data['B'], B_CHECKPOINTS), ('return', data['return'], RETURN_CHECKPOINTS)):
        for i in range(0, STREAM, PAIRS_PER_BATCH):
            if profile is not None:
                with meters[arm].time('learning'):
                    _, changes = module.learn_batch(h1[i:i+PAIRS_PER_BATCH], h2[i:i+PAIRS_PER_BATCH])
                norms += changes
            if i+PAIRS_PER_BATCH in marks:
                emit(episode, i+PAIRS_PER_BATCH)
    return rows


def direct_branch(arm, local, data, sets, world, meters, evaluator, base):
    learner, rows = direct_reference(arm, local, meters[arm]), []
    if learner.defined_at_zero:
        rows.append(dict(base, episode='B', m=0, arm=arm, defined=True,
                         **evaluate_direct(learner, sets, world['phase'], evaluator)))
    else:
        rows.append(dict(base, episode='B', m=0, arm=arm, defined=False, reason='no target pairs'))
    for episode, (h1, h2), marks in (('B', data['B'], B_CHECKPOINTS), ('return', data['return'], RETURN_CHECKPOINTS)):
        for i in range(STREAM):
            learner.observe(h1[i], h2[i])
            if i+1 in marks:
                learner.checkpoint()
                rows.append(dict(base, episode=episode, m=i+1, arm=arm, defined=True,
                                 **evaluate_direct(learner, sets, world['phase'], evaluator)))
    return rows


def run_world(stage, w, meters, evaluator):
    world = make_world(stage, w)
    xs, ys = observe(world, world['u'], stream(stage, w, 3))
    local, p_init = acquire(xs, ys, meters['source_acquisition'])
    with meters['source_warm_up'].time('warm_up'):
        p_source, losses = warm_up(p_init, xs, ys, meters['source_warm_up'])
    del xs, ys
    draws_b, draws_r, ev_a, ev_b = (stream(stage, w, part) for part in (4, 7, 8, 9))
    info = dict(world=w, true_phase=world['phase'], acquired_phase=local['phase'],
                phase_correct=local['phase'] == world['phase'], jstar=world['jstar'], cv_scores=local['scores'],
                warm_up_loss=dict(first_batch=float(np.mean(losses[:UPDATES_PER_BATCH])),
                                  last_batch=float(np.mean(losses[-UPDATES_PER_BATCH:]))),
                branches={})
    sets_a = evaluation_sets(world, world['u'], ev_a, ev_b)
    with evaluator.time('pre_warm_up_diagnostic'):
        info['pre_warm_up_E_A_matched'] = evaluate_layered(Module(copy_params(p_init)), dict(A=sets_a['A']),
                                                           world['phase'], evaluator)['E_A']
    rows = []
    for branch in BRANCHES:
        signal_b = branch_signal(world, branch)
        data = dict(B=observe(world, signal_b, draws_b), **{'return': observe(world, world['u'], draws_r)})
        sets = evaluation_sets(world, signal_b, ev_a, ev_b)
        with evaluator.time('controls'):
            info['branches'][branch] = dict(controls=controls(world, sets, signal_b))
        base = dict(world=w, branch=branch)
        try:
            for arm in LAYERED_ARMS:
                rows += layered_branch(arm, p_source, data, sets, world, meters, evaluator, base)
            for arm in DIRECT_ARMS:
                rows += direct_branch(arm, local, data, sets, world, meters, evaluator, base)
        except ScoringFailure as error:
            raise ScoringFailure(dict(world=w, branch=branch, error=str(error))) from None
    return info, rows


def new_meters():
    return {name: Meter2() for name in ('source_acquisition', 'source_warm_up', *ARMS)}


def run_stage(stage, worlds, progress=False):
    meters, evaluator = new_meters(), Meter2()
    worlds_out, rows, failure = [], [], None
    for w in worlds:
        try:
            info, world_rows = run_world(stage, w, meters, evaluator)
        except ScoringFailure as error:
            failure = error.args[0]
            break
        worlds_out.append(info)
        rows.extend(world_rows)
        if progress:
            print(f'  world {w} done', flush=True)
    return dict(status='incomplete' if failure else 'complete', failure=failure,
                worlds=worlds_out, rows=rows), meters, evaluator


def table(rows):
    return dict(columns=COLUMNS, values=[[r.get(c) for c in COLUMNS] for r in rows])


# ---------------------------------------------------------------- analysis

def analyse(rows, worlds_out):
    worlds = [i['world'] for i in worlds_out]
    get = {(r['world'], r['branch'], r['episode'], r['arm'], r['m']): r for r in rows if r['defined']}

    def window(branch, arm, measure, episode='B', marks=PRIMARY_WINDOW):
        return np.array([np.mean([get[w, branch, episode, arm, m][measure] for m in marks]) for w in worlds])

    def forgetting(branch, arm, episode='B', marks=PRIMARY_WINDOW):
        return window(branch, arm, 'E_A', episode, marks)-np.array([get[w, branch, 'B', arm, 0]['E_A']
                                                                     for w in worlds])

    q = stats.t.ppf(CONFIDENCE, len(worlds)-1)
    primary = {}
    for name, measure, first, margin in CLAIMS:
        diff = window(PRIMARY_BRANCH, first, measure)-window(PRIMARY_BRANCH, CANDIDATE, measure)
        se = float(np.std(diff, ddof=1)/np.sqrt(len(diff)))
        bound = float(diff.mean()) if se == 0 else float(diff.mean()-q*se)
        primary[name] = dict(branch=PRIMARY_BRANCH, window=PRIMARY_WINDOW,
                             difference=f'mean {measure}({first}) - mean {measure}({CANDIDATE})', margin=margin,
                             mean_difference=float(diff.mean()), se=se, t_quantile=float(q), confidence=CONFIDENCE,
                             lower_bound=bound, zero_variance=se == 0, passed=bool(bound > margin),
                             per_world_difference=diff)

    measures = ('E_A', 'E_B', 'distortion_A', 'distortion_B')
    at_zero = lambda b, arm: (worlds[0], b, 'B', arm, 0) in get
    window_scores = {b: {arm: dict({m: interval(window(b, arm, m)) for m in measures},
                                   forgetting_A=interval(forgetting(b, arm)) if at_zero(b, arm) else None)
                         for arm in ARMS} for b in BRANCHES}
    source_checkpoint = {b: {arm: interval([get[w, b, 'B', arm, 0]['E_A'] for w in worlds])
                             for arm in ARMS if (worlds[0], b, 'B', arm, 0) in get} for b in BRANCHES}
    return_scores = {b: {arm: dict(E_A=interval(window(b, arm, 'E_A', 'return', RETURN_CHECKPOINTS)),
                                   E_B=interval(window(b, arm, 'E_B', 'return', RETURN_CHECKPOINTS)),
                                   final_E_A=interval([get[w, b, 'return', arm, 256]['E_A'] for w in worlds]),
                                   final_E_B=interval([get[w, b, 'return', arm, 256]['E_B'] for w in worlds]),
                                   final_forgetting_A=interval(forgetting(b, arm, 'return', (256,)))
                                   if at_zero(b, arm) else None)
                         for arm in ARMS} for b in BRANCHES}
    contrasts = {b: {f'{arm}_minus_{CANDIDATE}': {m: interval(window(b, arm, m)-window(b, CANDIDATE, m))
                                                  for m in ('E_A', 'E_B')}
                     for arm in ARMS if arm != CANDIDATE} for b in BRANCHES}
    contrasts_return = {b: {f'{arm}_minus_{CANDIDATE}': {
        m: interval(window(b, arm, m, 'return', RETURN_CHECKPOINTS)-window(b, CANDIDATE, m, 'return', RETURN_CHECKPOINTS))
        for m in ('E_A', 'E_B')} for arm in ARMS if arm != CANDIDATE} for b in BRANCHES}
    marks = [('B', m) for m in B_CHECKPOINTS]+[('return', m) for m in RETURN_CHECKPOINTS]
    curves = {b: {arm: {f'{e}_{m}': {x: interval([get[w, b, e, arm, m][x] for w in worlds]) for x in ('E_A', 'E_B')}
                        for e, m in marks if (worlds[0], b, e, arm, m) in get} for arm in ARMS} for b in BRANCHES}
    ring_changes = {b: {arm: {f'{e}_{m}': dict(
        param_change=[interval([get[w, b, e, arm, m]['param_change'][l] for w in worlds]) for l in range(RINGS)],
        update_norm=[interval([get[w, b, e, arm, m]['update_norm'][l] for w in worlds]) for l in range(RINGS)])
        for e, m in (('B', 256), ('return', 256))} for arm in LAYERED_ARMS} for b in BRANCHES}
    by_world = {i['world']: i for i in worlds_out}
    correct = [w for w in worlds if by_world[w]['phase_correct']]

    def gate_contrast(arm, e, m, name):
        """g at the drifted sector minus mean g at the other sectors, per ring (worlds with p_hat = p)."""
        vals = []
        for w in correct:
            g, j = np.asarray(get[w, PRIMARY_BRANCH, e, arm, m][f'gate_mean_{name}']), by_world[w]['jstar']
            vals.append(g[:, j]-np.delete(g, j, 1).mean(1))
        return [interval(np.array(vals)[:, l]) for l in range(RINGS)] if len(vals) > 1 else None

    gates = {arm: {f'{e}_{m}': dict(drifted_minus_other_B=gate_contrast(arm, e, m, 'B'),
                                    drifted_minus_other_A=gate_contrast(arm, e, m, 'A'),
                                    outer_gate_error_corr_B=interval([get[w, PRIMARY_BRANCH, e, arm, m]['gate_error_corr_B']
                                                                      for w in worlds]))
                   for e, m in marks} for arm in LAYERED_ARMS}
    sector_regional = {arm: {f'{e}_{m}': dict(
        drifted_sector_E_B=interval([get[w, PRIMARY_BRANCH, e, arm, m]['sector_E_B'][by_world[w]['jstar']]
                                     for w in worlds]),
        other_sectors_E_B=interval([np.mean(np.delete(get[w, PRIMARY_BRANCH, e, arm, m]['sector_E_B'],
                                                      by_world[w]['jstar'])) for w in worlds]))
        for e, m in marks if (worlds[0], PRIMARY_BRANCH, e, arm, m) in get} for arm in ARMS}
    return dict(primary=primary, overall_pass=all(p['passed'] for p in primary.values()),
                window_scores=window_scores, source_checkpoint_E_A=source_checkpoint, return_scores=return_scores,
                contrasts_window=contrasts, contrasts_return=contrasts_return, curves=curves,
                ring_changes=ring_changes, gate_diagnostics=gates, regional_sector_errors=sector_regional,
                source=dict(fraction_phase_correct=float(np.mean([by_world[w]['phase_correct'] for w in worlds])),
                            pre_warm_up_E_A_matched=interval([by_world[w]['pre_warm_up_E_A_matched'] for w in worlds]),
                            warm_up_loss_first_batch=interval([by_world[w]['warm_up_loss']['first_batch'] for w in worlds]),
                            warm_up_loss_last_batch=interval([by_world[w]['warm_up_loss']['last_batch'] for w in worlds])),
                controls={b: {s: {c: float(np.mean([by_world[w]['branches'][b]['controls'][s][c] for w in worlds]))
                                  for c in ('zero', 'raw', 'oracle_projection')} for s in SETS} for b in BRANCHES},
                notes=['Window = per-world mean over B checkpoints m = 16, 32, 64, 128, 256.',
                       'Contrasts `x_minus_graded_matched` are measure(x) - measure(C): positive favours C. '
                       'Descriptive 95% t intervals only; no claim may be rescued by a contrast.',
                       'Branches share B and A-return draws and evaluator draws; they are not independent replicates.',
                       'Gate contrasts use the learner sector label j*; restricted to worlds with acquired phase = true.',
                       'phase_target_cca is undefined at B m=0; direct references never reset at the return.',
                       'One-sided paired Student-t bounds over worlds; coverage is approximate.'])


# ---------------------------------------------------------------- computation records

def charged_counts(arm):
    """Standalone charge per arm and branch: source acquisition and warm-up once, then adaptation."""
    n_dir = 2*PAIRS_PER_BATCH
    source = dict(source_pairs=STREAM, local_fits=FOLDS*PHASES+1, candidate_scores=FOLDS*PHASES,
                  basis_completions=RANK)
    warm = dict(warm_up_updates=STREAM//PAIRS_PER_BATCH*UPDATES_PER_BATCH,
                fixed_route_factorizations=STREAM//PAIRS_PER_BATCH*UPDATES_PER_BATCH,
                warm_up_second_solves=STREAM//PAIRS_PER_BATCH*UPDATES_PER_BATCH*n_dir,
                warm_up_adjoint_solves=STREAM//PAIRS_PER_BATCH*UPDATES_PER_BATCH*n_dir,
                source_pair_exposures=STREAM*UPDATES_PER_BATCH)
    if arm in DIRECT_ARMS:
        return dict(source, target_pairs_per_episode=STREAM, checkpoint_solves=len(B_CHECKPOINTS)-1+len(RETURN_CHECKPOINTS),
                    note='no layered warm-up; pooled deploys its source fit at m=0')
    out = dict(source, **warm)
    if arm in ADAPTING_ARMS:
        updates = 2*STREAM//PAIRS_PER_BATCH*UPDATES_PER_BATCH
        out.update(adaptation_updates=updates, pilot_factorizations=updates, pilot_solves=updates*n_dir,
                   route_evaluations=updates*n_dir, second_factorizations=updates*n_dir,
                   second_solves=updates*n_dir, adjoint_solves=updates*n_dir, retracted_blocks=updates*BLOCKS,
                   target_pairs=2*STREAM, target_pair_exposures=2*STREAM*UPDATES_PER_BATCH)
    return out


def complexity_record():
    return dict(
        dimensions=dict(M=M, d=D, J=RANK, widths=WIDTHS, sector_widths=SECTORS, stored_parameters=328,
                        q=B, batch_pairs=PAIRS_PER_BATCH, directions_per_update=2*PAIRS_PER_BATCH, K=PHASES),
        orders=dict(source_acquisition='v2 CV + refit (33 local fits) + O(J q^3) basis completion',
                    hessian_assembly='dense in this implementation: O(16 M^2) per parameter state plus '
                                     'O(n 16 M^2) per routed batch of n inputs (dense tensordot)',
                    factorization='dense Cholesky O(M^3) per routed input; pilot factor O(M^3) per parameter state',
                    solves='O(M^2) per pilot, second and adjoint solve',
                    gradient='O(n sum_l J n_l n_(l+1)) block contractions; retraction O(sum_l J n_l n_(l+1)^2)',
                    routed_update='O(M^3 + 2B (M^3 + 3 M^2) + 16 M^2 (2B+1) + retractions)',
                    fixed_route_update='O(M^3 + 2B 2 M^2): one shared factor for all directions and adjoints',
                    inference='two passes: O(M^2) pilot solve with shared factor + O(M^3) second factor and solve',
                    schur_reference='diagnostic only: O(M^3) LU solves on 8 inputs per set and checkpoint'),
        bytes=dict(persistent_parameters=328*8, phase_metadata=8, optimizer_state=0,
                   current_minibatch=2*PAIRS_PER_BATCH*D*8, curvature_cache=BLOCKS*M*M*8,
                   routed_batch_hessians_per_input=M*M*8, direct_reference_state='2 J q^2 float64 sums + count + d weights',
                   note='ring topology is immutable index metadata, reported separately'),
        flops='not measured; not inferred from wall time or parameter counts')


def state_bytes_record():
    world = make_world(STAGES['development'], 0)
    xs, ys = observe(world, world['u'], stream(0, 0, 3))
    local, params = acquire(xs, ys)
    persistent = sum(m.nbytes for m in params['maps'])+params['v'].nbytes
    out = dict(layered=dict(maps_and_terminal=int(persistent), phase=8,
                            topology_indices=int(sum(i.nbytes for i in sector_indices(0)))))
    for arm in DIRECT_ARMS:
        learner = direct_reference(arm, local)
        out[arm] = dict(transition=learner.state_bytes())
        for i in range(STREAM):
            learner.observe(xs[i], ys[i])
        learner.checkpoint()
        out[arm]['after_256_pairs'] = learner.state_bytes()
    return out


def meter_record(meters):
    return {name: dict(seconds=dict(m.seconds), counts=dict(m.counts), stats=m.stats) for name, m in meters.items()}


def record_runtime(out_dir, phase, record):
    path = out_dir/f'{PREFIX}-runtime.json'
    runtime = json.loads(path.read_text()) if path.exists() else {}
    runtime[phase] = dict(record, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          rss_scope='cumulative parent-process high-water RSS (Linux KiB), not per-arm memory',
                          dirty_paths=git('status', '--porcelain').splitlines(), revision=git('rev-parse', 'HEAD'))
    runtime.update(environment=environment(),
                   sources={n: sha256(ROOT/n) for n in SOURCES+sum(DEPENDENCIES.values(), ())},
                   command=' '.join([Path(sys.executable).name, '-m', 'experiments.single_module_plasticity',
                                     *sys.argv[1:]]),
                   seed_policy=f'SeedSequence([{ROOT_SEED}, stage, world, part])',
                   complexity=complexity_record(), charged_counts_per_arm_and_branch={a: charged_counts(a) for a in ARMS},
                   state_bytes=state_bytes_record())
    path.write_text(encoded(runtime))


def stage_record(result, meters, evaluator, worlds, started, cpu):
    return dict(wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu,
                executed_timing_and_counts=meter_record(meters), evaluator=meter_record({'evaluator': evaluator}),
                execution_note='source acquisition and warm-up execute once per world and are shared by every '
                               'arm and branch; each standalone arm is charged them in charged_counts_per_arm_and_branch. '
                               'Arm meters count adaptation only; evaluator inference, controls and Schur references '
                               'are in the evaluator meter.',
                solver_failures=int(result['failure'] is not None),
                independent_pairs=dict(source_per_world=STREAM, b_per_world=STREAM, return_per_world=STREAM,
                                       evaluator_per_set=STREAM, worlds=len(worlds),
                                       note='B, return and evaluator draws are shared across branches; '
                                            'evaluator second views are generated but unused'))


# ---------------------------------------------------------------- phases

def development(out_dir):
    started, cpu = time.perf_counter(), time.process_time()
    first, meters, evaluator = run_stage(STAGES['development'], DEV_WORLDS, progress=True)
    second, _, _ = run_stage(STAGES['development'], DEV_WORLDS)
    identical = encoded(first) == encoded(second)
    complete = first['status'] == 'complete'
    report = dict(status=first['status'], failure=first['failure'], worlds=first['worlds'],
                  rows=table(first['rows']), stage=STAGES['development'], root_seed=ROOT_SEED,
                  repeated_run_byte_identical=identical,
                  descriptive=analyse(first['rows'], first['worlds']) if complete else None,
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
    out = {}
    for name, files in DEPENDENCIES.items():
        manifest = json.loads((out_dir/MANIFESTS[name]).read_text())
        frozen = {**manifest.get('hashes', {}), **manifest.get('dependency_hashes', {})}
        for f in files:
            if sha256(ROOT/f) != frozen[f]:
                raise SystemExit(f'{name} source {f} differs from its manifest')
        out[f'{name}_manifest_sha256'] = sha256(out_dir/MANIFESTS[name])
    return out


def freeze(out_dir):
    dev = json.loads((out_dir/f'{PREFIX}-development.json').read_text())
    if dev['status'] != 'complete' or not dev['repeated_run_byte_identical']:
        raise SystemExit('development gate not satisfied')
    manifest = dict(
        scope='single-module plasticity v1 confirmation', contract='docs/single-module-plasticity-v1-contract.md',
        hashes={name: sha256(ROOT/name) for name in SOURCES},
        dependency_hashes={name: sha256(ROOT/name) for name in sum(DEPENDENCIES.values(), ())},
        **check_dependencies(out_dir),
        development_report_sha256=sha256(out_dir/f'{PREFIX}-development.json'),
        revision_at_freeze=git('rev-parse', 'HEAD'), correctness_tests=run_tests(), test_modules=TESTS,
        seeds=dict(root=ROOT_SEED, stages=STAGES, entropy='[root, stage, world, part]',
                   parts={0: 'templates: per block 3 signal then 5 texture normals (v1 recipe)',
                          1: 'reserved, never drawn', 2: 'source phase integers(0, 8)',
                          3: 'source stream c (256,k), g1 (256,k), g2 (256,k), e1 (256,d), e2 (256,d)',
                          4: 'B stream, same shapes and order', 5: 'drift directions (k, b) Gaussian',
                          6: 'regional drifted sector integers(0, 4)', 7: 'A-return stream',
                          8: 'evaluator A (first view and clean signal used)', 9: 'evaluator B'},
                   loop_order='world > source acquisition > warm-up > branch > layered arms > direct arms > '
                              'B batches > return batches; evaluation at checkpoints'),
        development_worlds=DEV_WORLDS, confirmation_worlds=CONFIRM_WORLDS,
        dimensions=dict(d=D, k=RANK, b=B, widths=WIDTHS, sectors=SECTORS, M=M, phases=PHASES, folds=FOLDS),
        learning=dict(eta=ETA, profiles=PROFILES, pairs_per_batch=PAIRS_PER_BATCH, updates_per_batch=UPDATES_PER_BATCH,
                      drift_degrees=DRIFT_DEGREES),
        arms=LAYERED_ARMS, direct_arms=DIRECT_ARMS, branches=BRANCHES, b_checkpoints=B_CHECKPOINTS,
        return_checkpoints=RETURN_CHECKPOINTS, primary_window=PRIMARY_WINDOW,
        decision_rule=dict(claims=CLAIMS, candidate=CANDIDATE, branch=PRIMARY_BRANCH, confidence=CONFIDENCE,
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
    result, meters, evaluator = run_stage(STAGES['confirmation'], CONFIRM_WORLDS, progress=True)
    complete = result['status'] == 'complete'
    analysis = analyse(result['rows'], result['worlds']) if complete else None
    report = dict(status=result['status'], failure=result['failure'], worlds=result['worlds'],
                  rows=table(result['rows']), stage=STAGES['confirmation'], root_seed=ROOT_SEED,
                  manifest_sha256=sha256(out_dir/f'{PREFIX}-manifest.json'), analysis=analysis)
    (out_dir/f'{PREFIX}-results.json').write_text(encoded(report, compact=True))
    record_runtime(out_dir, 'confirmation', dict(
        stage_record(result, meters, evaluator, CONFIRM_WORLDS, started, cpu),
        scope='single process: generation, acquisition, warm-up, branches, evaluator, analysis, write',
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


def summary(values):
    return dict(median=float(np.median(values)), iqr=[float(np.percentile(values, 25)), float(np.percentile(values, 75))])


def repeat(fn):
    """One discarded warm-up then REPEATS fresh executions on identical frozen inputs."""
    _, warmup = timed(fn)
    runs = [timed(fn) for _ in range(REPEATS)]
    return runs[-1][0], dict(warmup=warmup, repeats=[r for _, r in runs],
                             **summary([r['wall_seconds'] for _, r in runs]))


def inference_blocks(fn, x):
    fn(x)
    blocks = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(CALLS):
            fn(x)
        blocks.append((time.perf_counter()-start)/CALLS)
    return dict(per_call_seconds=blocks, **summary(blocks))


def timing_job(config):
    arm, point = config
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stage = STAGES['timing']
    world = make_world(stage, 0)
    xs, ys = observe(world, world['u'], stream(stage, 0, 3))
    h1, h2 = observe(world, branch_signal(world, 'regional'), stream(stage, 0, 4))
    x_eval = observe(world, branch_signal(world, 'regional'), stream(stage, 0, 9))[0][:TIMING_BATCH]
    (local, p_init), acquisition = repeat(lambda: acquire(xs, ys))
    record = dict(arm=arm, checkpoint=point, acquisition=acquisition)
    seen = 0 if point == 'post_source' else 32
    batch = slice(seen, seen+PAIRS_PER_BATCH)
    if arm in LAYERED_ARMS:
        (p_source, _), record['warm_up'] = repeat(lambda: warm_up(p_init, xs, ys))
        profile, policy = LAYERED_ARMS[arm]
        module = Module(copy_params(p_source), profile, policy)
        if profile is not None:
            for i in range(0, seen, PAIRS_PER_BATCH):
                module.learn_batch(h1[i:i+PAIRS_PER_BATCH], h2[i:i+PAIRS_PER_BATCH])
            x, t = np.concatenate([h1[batch], h2[batch]]), np.concatenate([h2[batch], h1[batch]])
            meters = []

            def one_update():
                copy_ = Module(copy_params(module.p), profile, policy)
                copy_.update(x, t)
                meters.append(copy_.meter)
            _, record['one_update'] = repeat(one_update)
            record['one_update']['components_seconds_per_repeat'] = [dict(m.seconds) for m in meters[1:]]
            record['one_update']['work'] = dict(meters[-1].counts)
            record['one_update']['scope'] = ('parameter copy + pilot factor + pilot solves + routes + second '
                                             'factors/solves + adjoint solves + gradient + retraction')
        else:
            record['one_update'] = dict(median=0., note='frozen arm: no update cost')
        infer = lambda z: Module(module.p, None, policy).infer(z)
        record['inference'] = dict(single_input=inference_blocks(infer, x_eval[:1]),
                                   batch_32=inference_blocks(infer, x_eval),
                                   scope='two passes incl. pilot factor construction (fresh per call: no cached '
                                         'factor across calls), routes, second factors and solves; allocation '
                                         'included, generation excluded')
        tracemalloc.start()
        if profile is not None:
            Module(copy_params(module.p), profile, policy).update(x, t)
        peak_update = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        infer(x_eval)
        peak_infer = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        learner = direct_reference(arm, local)
        for i in range(seen):
            learner.observe(h1[i], h2[i])
        if seen or learner.defined_at_zero:
            if seen:
                learner.checkpoint()

        def one_update():
            c = copy.deepcopy(learner)
            for i in range(batch.start, batch.stop):
                c.observe(h1[i], h2[i])
            c.checkpoint()
            return c
        trained, record['one_update'] = repeat(one_update)
        record['one_update']['scope'] = 'deep copy + 8 pair statistic updates + one checkpoint solve'
        infer = lambda z: projector_output(trained, z)
        record['inference'] = dict(single_input=inference_blocks(infer, x_eval[:1]),
                                   batch_32=inference_blocks(infer, x_eval),
                                   scope='dense projector from deployed weights + apply; allocation included')
        tracemalloc.start()
        one_update()
        peak_update = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        infer(x_eval)
        peak_infer = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    record['peak_tracemalloc_bytes'] = dict(update=peak_update, batch_32_inference=peak_infer,
                                            scope='NumPy/Python allocations in a separate untimed pass; '
                                                  'LAPACK workspace excluded')
    record['rss_kib'] = dict(after_imports_and_spawn=baseline, peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             scope='fresh spawned worker high-water RSS, Linux KiB; includes interpreter')
    return record


def amortization(r):
    """Arithmetic estimates: acquisition (+ warm-up) + updates x median update + Q x single inference."""
    single = r['inference']['single_input']['median']
    base = r['acquisition']['median']+r.get('warm_up', {}).get('median', 0.)
    out = {}
    for name, pairs in (('final_B', STREAM), ('final_return', 2*STREAM)):
        if r['arm'] in LAYERED_ARMS:
            updates = pairs//PAIRS_PER_BATCH*UPDATES_PER_BATCH if LAYERED_ARMS[r['arm']][0] else 0
            training = base+updates*r['one_update']['median']
        else:
            training = base+pairs//PAIRS_PER_BATCH*r['one_update']['median']
        out[name] = {str(q): training+q*single for q in QUERIES}
    return out


def timing(out_dir):
    check_frozen(out_dir)
    started, cpu = time.perf_counter(), time.process_time()
    context = multiprocessing.get_context('spawn')
    configs = [(arm, point) for point in ('post_source', 'post_32_B') for arm in ARMS]
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
        protocol=dict(stage=STAGES['timing'], world=0, branch='regional', checkpoints=('post_source', 'post_32_B'),
                      warmup_executions=1, repeats=REPEATS, inference_blocks=REPEATS, calls_per_block=CALLS,
                      batch=TIMING_BATCH,
                      amortization='acquisition (+ warm-up) + updates x median one-update + Q x single-input '
                                   'median; direct references: per-batch statistic update + solve per batch, an '
                                   'overestimate of checkpoint-only solves; arithmetic estimates, not measured',
                      note='one-size instrumentation at M=80; no scaling or speed claim; stage-2 data never scored')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'freeze', 'confirm', 'timing'), required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    dict(dev=development, freeze=freeze, confirm=confirmation, timing=timing)[args.phase](args.out_dir)


if __name__ == '__main__':
    main()
