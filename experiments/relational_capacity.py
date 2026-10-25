"""Relational capacity Stage B v1: bounded privileged capacity search over S2, P0 and D1(mu)."""
import os
import time

START, CPU_START = time.perf_counter(), time.process_time()
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import torch

from experiments.locality_adaptation import canonical, encoded, environment, git, sha256
from src import abstraction_preflight as pre
from src import masked_capacity as c
from src import relational_capacity as rc

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'relational-capacity-stage-b-v1'
SOURCES = (f'docs/{PREFIX}-contract.md', 'src/relational_capacity.py',
           'experiments/relational_capacity.py', 'tests/test_relational_capacity.py', 'requirements.txt')
PARENT_MANIFEST = 'reports/single-module-masked-capacity-v1-manifest.json'
FROZEN_INPUTS = ('reports/single-module-masked-capacity-v1-audit.json',
                 'docs/relational-capacity-stage-b-decisions.md',
                 'docs/relational-capacity-and-composition-analysis.md',
                 'docs/relational-composition-weak-coupling-bound.md',
                 'docs/single-module-masked-capacity-obstruction.md')
TESTS = ('tests.test_masked_capacity', 'tests.test_relational_capacity')
DISCONNECTED = ('With no port the output module has a zero right-hand side; its SPD equilibrium is zero '
                'and every scene loss is exactly 1. Stated analytically, not computed.')


def machine():
    return dict(environment(), torch=torch.__version__, torch_build=torch.__config__.show(),
                torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def dependencies():
    m = json.loads((ROOT/PARENT_MANIFEST).read_text())
    deps = dict(m['sources'], **m['dependencies'])
    deps[PARENT_MANIFEST] = sha256(ROOT/PARENT_MANIFEST)
    for name in FROZEN_INPUTS:
        deps[name] = sha256(ROOT/name)
    for name, digest in deps.items():
        if sha256(ROOT/name) != digest:
            raise RuntimeError(f'frozen dependency changed: {name}')
    return deps


def freeze(out, runtime):
    path = out/f'{PREFIX}-manifest.json'
    if path.exists():
        raise RuntimeError('no silent manifest overwrite')
    manifest = dict(status='frozen_before_tests_and_fits', sources={s: sha256(ROOT/s) for s in SOURCES},
        dependencies=dependencies(), revision=git('rev-parse', 'HEAD'), environment=machine(),
        root_seed=rc.ROOT_SEED, worlds=rc.WORLDS, starts=rc.STARTS, step_sizes=rc.RATES, mus=rc.MUS,
        arms=[dict(configuration=a, mus=m) for a, m in rc.ARMS], checkpoints=rc.CHECKPOINTS,
        maximum_updates_per_candidate=rc.UPDATES, maximum_candidates=40, maximum_total_updates=40000,
        objective='exact maximum per-scene hidden loss (torch.amax tie semantics)',
        selection='lexicographic (worst, mean, update index); across candidates (worst, mean, enumeration)',
        task=dict(scenes=72, fit_scope='privileged capacity fitting on all 72 targets'),
        tests=TESTS, repeat_scope='two serialized-candidate verifications; one search trajectory')
    path.write_text(encoded(manifest))
    runtime.update(status='complete', manifest_sha256=sha256(path))


def frozen(out):
    path = out/f'{PREFIX}-manifest.json'
    m = json.loads(path.read_text())
    dependencies()
    for name, digest in dict(m['sources'], **m['dependencies']).items():
        if sha256(ROOT/name) != digest:
            raise RuntimeError(f'manifest mismatch: {name}')
        if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != (ROOT/name).read_bytes():
            raise RuntimeError(f'frozen input uncommitted: {name}')
    if subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT) != path.read_bytes():
        raise RuntimeError('manifest uncommitted')
    return sha256(path)


def tests(out, runtime):
    digest = frozen(out)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    command = [sys.executable, '-m', 'unittest', *TESTS, '-v']
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime.update(status='complete' if proc.returncode == 0 else 'failed', manifest_sha256=digest,
        test_command=command, test_output=proc.stdout+proc.stderr, returncode=proc.returncode,
        subprocess_wall_seconds=time.perf_counter()-start,
        subprocess_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        subprocess_peak_rss_kib=after.ru_maxrss, rss_scope='RUSAGE_CHILDREN high-water, Linux KiB',
        operation_counts=None, operation_counts_reason='test fixture operations not instrumented; entire test cost timed')
    print(proc.stdout+proc.stderr, flush=True)


def require_previous(out, phase, digest):
    old = json.loads((out/f'{PREFIX}-runtime.json').read_text())
    if old.get(phase, {}).get('status') != 'complete' or old[phase]['manifest_sha256'] != digest:
        raise RuntimeError(f'successful {phase} for this manifest required')


def exclusion_ledger():
    threshold = rc.weak_coupling_threshold()
    return dict(
        S1=dict(status='excluded_before_budget', basis='docs/single-module-masked-capacity-obstruction.md'),
        fixed_route_compositions=dict(status='excluded_before_budget',
                                      basis='affine floor (8-2sqrt2)/21 on the q=1 subset'),
        weak_coupling=dict(basis='docs/relational-composition-weak-coupling-bound.md', mu_threshold=threshold,
                           excluded={str(mu): bool(mu < threshold) for mu in rc.MUS}),
        open_bounds=['sector-aligned communication bound', 'rank-two route-insensitivity bound'],
        disconnected_output=DISCONNECTED)


def optimizer_bytes(optimizer):
    entries, total = [], 0
    for i, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value):
                size = value.numel()*value.element_size()
                total += size
                entries.append(dict(parameter_tensor=i, field=key, dtype=str(value.dtype), bytes=size))
    return dict(total_bytes=total, entries=entries)


def as_arrays(saved):
    out = []
    for p in canonical(saved):
        q = dict(maps=[np.asarray(d, dtype=np.float64) for d in p['maps']])
        if 'v' in p:
            q['v'] = np.asarray(p['v'], dtype=np.float64)
        out.append(q)
    return out


def verify(saved, arm, mu, x, matrices, meter):
    """Serialize to 12 significant digits; independent NumPy/Schur inference twice, byte-identical."""
    params = as_arrays(saved)
    rc.check_constraints(params)
    records = []
    observed = torch.tensor(x[:, :32]*pre.MASK)
    for _ in range(2):
        prediction, fixed, checks = rc.numpy_prediction(params, arm, mu, x, meter)
        with torch.no_grad():
            routed, pilot = rc.Composition(arm, mu, params, meter)(observed)
        difference = float(max(np.abs(prediction-routed.numpy()).max(), np.abs(fixed-pilot.numpy()).max()))
        pre.require(difference <= 1e-9, 'serialized Torch/NumPy mismatch')
        records.append(dict(prediction=c.prediction_record(prediction, x, matrices),
                            fixed_route_control=c.prediction_record(fixed, x, matrices)['all'],
                            screen=rc.route_screen(prediction, fixed, x[:, 16:]),
                            checks=checks, torch_numpy_max_abs=difference))
    pre.require(encoded(records[0]) == encoded(records[1]), 'saved candidate verification not byte-identical')
    return dict(parameters=params, byte_identical=True, **records[0])


def checked_screen(prediction, fixed, x):
    screen = rc.route_screen(prediction, fixed, x[:, 16:])
    # The fixed-route output is linear in the observations, so it cannot beat the affine floor.
    pre.require(screen['fixed_route_rms_error'] >= np.sqrt(2*rc.AFFINE_FLOOR)-1e-9,
                'fixed-route predictor below the affine floor')
    return screen


def fit_candidate(record, timed, x, matrices):
    meter = pre.Meter()
    arm, mu = record['configuration'], record['mu']
    params = rc.initial_parameters(arm, record['world'], record['start'])
    record.update(initial_parameters=params,
                  initial_parameters_sha256=hashlib.sha256(encoded(params).encode()).hexdigest(),
                  stored_floats=rc.stored_floats(params))
    model = rc.Composition(arm, mu, params, meter)
    optimizer = torch.optim.Adam(model.parameters(), lr=record['step_size'], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    observed = torch.tensor(x[:, :32]*pre.MASK, dtype=torch.float64)
    target = torch.tensor(x[:, 16:], dtype=torch.float64)
    observation_uses = sum(1 for s in rc.LAYOUTS[arm] for a in s['anchors'] if a > 0)
    best, best_key, best_parameters, verified_step = None, None, None, None
    start, cpu = time.perf_counter(), time.process_time()
    record['checkpoints'], timed['checkpoint_costs'] = [], []
    try:
        for step in range(rc.UPDATES+1):
            record['last_completed_updates'] = step
            if step in rc.CHECKPOINTS:
                with meter.phase('checkpoint_inference'):
                    with torch.no_grad():
                        routed, fixed = model(observed)
                    prediction, fixed = routed.numpy(), fixed.numpy()
                    summary = c.prediction_record(prediction, x, matrices)
                    screen = checked_screen(prediction, fixed, x)
                    meter.counts['screen_evaluations'] += 1
                record['checkpoints'].append(dict(step=step, screen=screen,
                    **{k: summary[k] for k in ('all', 'train', 'held_out')}))
                timed['checkpoint_costs'].append(dict(step=step, cumulative_wall_seconds=time.perf_counter()-start,
                    cumulative_process_cpu_seconds=time.process_time()-cpu, counts=dict(meter.counts)))
                key = rc.selection_key(summary, step)
                if best_key is None or key < best_key:
                    best, best_key, best_parameters = dict(step=step, **summary), key, model.numpy_parameters()
                print(f"{arm} mu={mu} world={record['world']} start={record['start']} lr={record['step_size']} "
                      f"step={step} mean={summary['all']['mean_loss']:.6f} worst={summary['all']['worst_loss']:.6f} "
                      f"d={screen['d']:.4f}", flush=True)
                if summary['all']['all_pass']:
                    with meter.phase('independent_verification'):
                        checked = verify(model.numpy_parameters(), arm, mu, x, matrices, meter)
                    pre.require(checked['prediction']['all']['all_pass'], 'witness lost after serialization')
                    pre.require(checked['screen']['d'] >= checked['screen']['parameter_specific_threshold']-1e-9,
                                'verified witness violates the necessary route-sensitivity inequality')
                    best, best_parameters = dict(step=step, **summary), model.numpy_parameters()
                    record['verification'], verified_step = checked, step
                    break
            if step == rc.UPDATES:
                break
            with meter.phase('optimization'):
                optimizer.zero_grad(set_to_none=True)
                objective, losses = model.objective(observed, target)
                pre.require(bool(torch.isfinite(losses).all()), 'nonfinite capacity loss')
                objective.backward()
                meter.counts['backward_passes'] += 1
                model.project_gradients()
                optimizer.step()
                model.retract()
                meter.counts.update(optimizer_updates=1, fitting_scene_exposures=72,
                                    local_observation_uses=72*observation_uses)
        record.update(best_step=best['step'], best_checkpoint=best, parameters_at_exit=model.numpy_parameters(),
                      min_gate_denominator=model.stats.get('min_gate_denominator'),
                      small_gate_denominator_batches=meter.counts['small_gate_denominator_batches'])
        if verified_step is None:
            with meter.phase('independent_verification'):
                record['verification'] = verify(best_parameters, arm, mu, x, matrices, meter)
        record['is_witness'] = record['verification']['prediction']['all']['all_pass']
        record['status'] = 'witness' if record['is_witness'] else 'budget_exhausted'
    except Exception as error:
        record.update(status='failed', error=str(error), parameters_at_exit=model.numpy_parameters())
        raise
    finally:
        timed.update(wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
            counts=dict(meter.counts), phase_wall_seconds=dict(meter.seconds), phase_cpu_seconds=dict(meter.cpu_seconds),
            phase_scope='disjoint components of this candidate; candidate totals are contained in the search command',
            optimizer_storage=optimizer_bytes(optimizer),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def search(out, runtime):
    digest = frozen(out)
    require_previous(out, 'tests', digest)
    result = dict(status='running', manifest_sha256=digest, exclusions=exclusion_ledger(), groups=[], candidates=[])
    runtime['candidates'] = []
    path = out/f'{PREFIX}-results.json'
    _, _, clean, _ = c.task()
    try:
        for arm, mus in rc.ARMS:
            for world in rc.WORLDS:
                matrices = c.rendering(world)
                x = pre.render(clean, matrices)
                found = False
                for mu in mus:
                    for start in rc.STARTS:
                        for rate in rc.RATES:
                            rec = dict(configuration=arm, mu=mu, world=world, start=start, step_size=rate,
                                       enumeration_index=len(result['candidates']))
                            result['candidates'].append(rec)
                            if found:
                                rec.update(status='unexecuted_after_witness', is_witness=False)
                                continue
                            timed = {k: rec[k] for k in ('configuration', 'mu', 'world', 'start', 'step_size')}
                            runtime['candidates'].append(timed)
                            fit_candidate(rec, timed, x, matrices)
                            found = rec['is_witness']
                            path.write_text(encoded(result))
                group = [r for r in result['candidates']
                         if r['configuration'] == arm and r['world'] == world and 'verification' in r]
                winner = min(group, key=lambda r: (r['verification']['prediction']['all']['worst_loss'],
                                                   r['verification']['prediction']['all']['mean_loss'],
                                                   r['enumeration_index']))
                result['groups'].append(dict(configuration=arm, world=world, renderers=matrices,
                    classification='constructive_witness' if found else 'unresolved_after_budget',
                    selected_candidate=dict(enumeration_index=winner['enumeration_index'], mu=winner['mu'],
                                            start=winner['start'], step_size=winner['step_size'],
                                            step=winner['best_step']),
                    best_verified_scores=winner['verification']['prediction']['all'],
                    best_verified_screen=winner['verification']['screen'],
                    fixed_route_control=winner['verification']['fixed_route_control']))
        result['status'] = 'complete'
        path.write_text(encoded(result))
    except Exception as error:
        result.update(status='incomplete', failure=dict(type=type(error).__name__, message=str(error)))
        path.write_text(encoded(result))
        raise
    runtime.update(status='complete', manifest_sha256=digest, results_sha256=sha256(path),
        entire_optimizer_trajectory_repeated=False,
        repeated_scope='two independent inference verifications of each serialized selected checkpoint',
        memory=dict(parameter_bytes=dict(S2=2560, P0=5248, D1=5248),
                    retained_training_input_target_bytes_per_rendering=27648,
                    curvature_tensor_bytes=dict(S2=614400, P0=1638400, D1=1638400),
                    d1_batch_hessian_tensor_bytes=14745600,
                    native_autograd_workspace_peak_bytes=None,
                    unavailable_reason='native Torch/BLAS temporary peaks not separately instrumented; '
                                       'process RSS and retained optimizer tensors recorded'),
        flops=None, flops_reason='not instrumented; dense forward/backward orders are in the contract',
        backward_primitive_solve_counts=None,
        backward_primitive_solve_counts_reason='autograd internals not instrumented; backward invocations counted')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=('freeze', 'tests', 'search'), required=True)
    args = parser.parse_args()
    out = ROOT/'reports'
    path = out/f'{PREFIX}-runtime.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if args.phase in previous:
        raise RuntimeError('phase already recorded; preserve all previous work')
    record = dict(command=[sys.executable, '-m', 'experiments.relational_capacity', *sys.argv[1:]],
                  revision=git('rev-parse', 'HEAD'), dirty_paths_at_start=git('status', '--porcelain').splitlines())
    try:
        dict(freeze=freeze, tests=tests, search=search)[args.phase](out, record)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
    record.update(command_wall_seconds=time.perf_counter()-START,
        parent_cpu_seconds=time.process_time()-CPU_START,
        parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Linux KiB cumulative process high-water; candidate values are not isolated peaks')
    previous.update(environment=machine(), contract_sha256=sha256(ROOT/SOURCES[0]))
    previous[args.phase] = record
    path.write_text(encoded(previous))
    print(encoded({k: v for k, v in record.items() if k in ('status', 'error')}), flush=True)
    if record['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
