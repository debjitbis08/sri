"""Run the predeclared graded-cellular mechanism test, without an architecture claim."""
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import time
import tracemalloc

from src.graded_cellular import (
    Model, VARIANTS, ReleasedFlow, RECOVERY_TOLERANCE, classify, mean,
    pulse, stationary_modes,
)


ROOT = Path(__file__).resolve().parents[1]
GAMMAS = (0., .25, 1., 2., 4.)
AMPLITUDES = (.25, 1., 4.)


def model_case(model):
    modes = stationary_modes(model)
    engine = ReleasedFlow(model)
    basins = []
    for initial in product((-1., 0., 1.), repeat=4):
        trajectory = engine.run(initial)
        mode = classify(model, trajectory, modes)
        basins.append(dict(initial=initial, final=trajectory['final'],
            status=trajectory['status'], residual=trajectory['gradient_residual'], mode=mode))
    basin_counts = Counter(row['mode']['kind'] for row in basins)
    pulse_results = []
    for evidence in (-1, 1):
        driven = pulse(model, evidence)
        released = engine.run(driven['state'])
        refined_driven = pulse(model, evidence, dt=1/32)
        refined_engine = ReleasedFlow(model, dt=1/32)
        refined = refined_engine.run(refined_driven['state'])
        discrepancy = max(abs(a-b) for a, b in zip(released['final'], refined['final']))
        pulse_results.append(dict(driven=driven, released=released,
            released_mode=classify(model, released, modes),
            refinement=dict(final=refined['final'], status=refined['status'],
                max_difference=discrepancy, within_tolerance=discrepancy <= RECOVERY_TOLERANCE,
                work=refined_engine.work())))
    disturbances = []
    for index, mode in enumerate(modes):
        if mode['kind'] not in ('isolated_minimum', 'minimum_manifold', 'flat'):
            continue
        source = mode['state']
        for coordinate, magnitude, direction in product(range(4), AMPLITUDES, (-1, 1)):
            delta = direction*magnitude
            initial = list(source)
            initial[coordinate] += delta
            trajectory = engine.run(tuple(initial))
            output_error = abs(trajectory['output']-source[2])
            full_error = max(abs(a-b) for a, b in zip(trajectory['final'], source))
            converged = trajectory['status'] == 'converged_numerically'
            disturbances.append(dict(source_index=index, source_kind=mode['kind'],
                coordinate=coordinate, delta=delta, final=trajectory['final'],
                status=trajectory['status'], residual=trajectory['gradient_residual'],
                output_error=output_error, full_error=full_error,
                restores_output=(output_error <= RECOVERY_TOLERANCE) if converged else None,
                restores_full_state=(full_error <= RECOVERY_TOLERANCE) if converged else None,
                mode=classify(model, trajectory, modes)))
    recovery_counts = {}
    for magnitude in AMPLITUDES:
        selected = [r for r in disturbances if abs(r['delta']) == magnitude]
        recovery_counts[str(magnitude)] = dict(total=len(selected),
            restored=sum(r['restores_output'] is True for r in selected),
            changed=sum(r['restores_output'] is False for r in selected),
            unresolved=sum(r['restores_output'] is None for r in selected))
    outputs = [row['released']['output'] for row in pulse_results]
    pulse_converged = all(row['released']['status'] == 'converged_numerically' for row in pulse_results)
    return dict(model=asdict(model), stationary_modes=modes, basins=dict(
            sample_count=len(basins), kind_counts=dict(basin_counts), samples=basins),
        pulses=pulse_results, disturbances=disturbances, recovery_counts=recovery_counts,
        isolated_stable_modes=sum(mode['kind'] == 'isolated_minimum' for mode in modes),
        pulse_distinction=(abs(outputs[0]-outputs[1]) > RECOVERY_TOLERANCE) if pulse_converged else None,
        numerical_audit=dict(
            max_energy_increase=max((r['max_energy_increase'] for r in engine.cache.values()), default=0.),
            all_pulse_refinements_agree=all(p['refinement']['within_tolerance'] for p in pulse_results)),
        work=engine.work())


def sensitivity_probes():
    context = []
    for epsilon in (1/64, 1/1024):
        pair = []
        for gamma in (2-epsilon, 2+epsilon):
            model = Model('combined', True, 1., 0., gamma, 1.)
            modes = stationary_modes(model)
            pair.append(dict(gamma=gamma, modes=modes,
                isolated_modes=sum(m['kind'] == 'isolated_minimum' for m in modes)))
        context.append(dict(epsilon=epsilon, comparison=pair,
                            claim='Stationary-branch sensitivity only, not a finite-time switching certificate'))
    face = []
    for f in (4/9-1/1024, 4/9+1/1024):
        modes = stationary_modes(Model('face_probe', True, 1., f, 0., 1.))
        face.append(dict(face=f, modes=modes,
                         isolated_modes=sum(m['kind'] == 'isolated_minimum' for m in modes)))
    return dict(context_saddle_node=context, face_bifurcation=face)


def summarize(rows):
    summary = {}
    for name, *_ in VARIANTS:
        subset = [row for row in rows if row['model']['name'] == name]
        small = [trial for row in subset for trial in row['disturbances'] if abs(trial['delta']) == .25]
        summary[name] = dict(configurations=len(subset),
            two_isolated_modes=sum(row['isolated_stable_modes'] == 2 for row in subset),
            distinct_pulses=sum(row['pulse_distinction'] is True for row in subset),
            collapsed_pulses=sum(row['pulse_distinction'] is False for row in subset),
            unresolved_pulses=sum(row['pulse_distinction'] is None for row in subset),
            small_disturbances=dict(total=len(small), restored=sum(t['restores_output'] is True for t in small),
                changed=sum(t['restores_output'] is False for t in small),
                unresolved=sum(t['restores_output'] is None for t in small)))
    comparisons = []
    by_key = {(r['model']['name'], r['model']['gamma'], r['model']['context']): r for r in rows}
    for name, *_ in VARIANTS:
        for gamma in GAMMAS:
            left, right = (by_key[name, gamma, c] for c in (-1., 1.))
            for index, e in enumerate((-1, 1)):
                a, b = left['pulses'][index]['released'], right['pulses'][index]['released']
                resolved = a['status'] == b['status'] == 'converged_numerically'
                comparisons.append(dict(variant=name, gamma=gamma, evidence=e,
                    outputs=(a['output'], b['output']),
                    changes_output=(abs(a['output']-b['output']) > RECOVERY_TOLERANCE) if resolved else None,
                    changes_sign=(a['output']*b['output'] < 0 and
                        min(abs(a['output']), abs(b['output'])) > RECOVERY_TOLERANCE) if resolved else None))
    omission = []
    for row in rows:
        model = row['model']
        if not model['gamma']:
            continue
        reference = by_key[model['name'], 0., model['context']]
        for index, e in enumerate((-1, 1)):
            a, b = row['pulses'][index]['released'], reference['pulses'][index]['released']
            omission.append(dict(variant=model['name'], gamma=model['gamma'], context=model['context'], evidence=e,
                finite_horizon_output_difference=abs(a['output']-b['output']),
                coupled_status=a['status'], omitted_status=b['status']))
    return dict(variants=summary, context_comparisons=comparisons, context_omissions=omission)


def certificate(progress=None):
    rows = []
    for name, cellular, beta, face in VARIANTS:
        for gamma, context in product(GAMMAS, (-1., 1.)):
            rows.append(model_case(Model(name, cellular, beta, face, gamma, context)))
        if progress:
            progress(name)
    files = ('docs/graded-cellular-v1-contract.md', 'src/graded_cellular.py', 'experiments/graded_cellular.py')
    return dict(specification='Graded cellular mechanism v1', status='complete',
        trajectory_evidence='Numerical RK4 with analytic contrast evolution; local stability analytic from Hessian',
        runtime=dict(python=platform.python_version(), implementation=platform.python_implementation()),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        bounds=dict(configurations=60, basin_starts_per_configuration=81,
            pulse_duration=16, release_duration=128, dt=1/16, refinement_dt=1/32,
            gradient_tolerance=1e-7, recovery_tolerance=RECOVERY_TOLERANCE),
        scope_discount=dict(
            architecture_grade='not_evaluated',
            implemented=['one cellular cochain', 'hand-designed polynomial preference', 'scalar dynamic context',
                         'single graded mode/context coupling', 'separate input/output coordinates'],
            unimplemented=['graded layer connections', 'multiple layers per module',
                'structured input/output/core/satellite modules', 'autonomous selective core mediation',
                'learned modes', 'STM', 'dreaming', 'abstraction', 'higher-arity module relations',
                'autonomous context activation'],
            rule='Credit only the implemented mechanism; omitted features are untested, not failed or assumed successful'),
        cases=rows, summary=summarize(rows), probes=sensitivity_probes())


def render(report):
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'


if __name__ == '__main__':
    started = time.perf_counter()
    report = certificate(progress=lambda name: print(name, 'complete', flush=True))
    elapsed = time.perf_counter()-started
    path = ROOT / 'reports' / 'graded-cellular-v1.json'
    path.write_text(render(report))
    # A representative memory sample, not a misleading full-run allocation claim.
    tracemalloc.start()
    sample = ReleasedFlow(Model('combined', True, 1., 0., 1., 1.))
    sample.run((1., 0., -1., 0.))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    (ROOT/'reports'/'graded-cellular-v1-runtime.json').write_text(render(dict(
        runtime=report['runtime'], seconds_full_certificate=elapsed,
        peak_python_bytes_single_representative_trajectory=peak,
        note='Memory sample excludes full report; wall time includes basin/certificate work, not a scaling benchmark')))
    print(path)
