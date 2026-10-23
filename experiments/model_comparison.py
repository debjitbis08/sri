"""Bounded four-family comparison: python -m experiments.model_comparison."""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import time
import tracemalloc

from src.model_comparison import (
    Work, D1, additive_disturbances, algebra_response, algebra_system,
    cell_boundary, cell_driven, cell_matrix, cell_withdrawal_limit,
    contraction, difference, dominance, eigendiagnostic, iterate, matmul,
    matvec, norm, rank, relation_minima, relation_response, replacements,
    residual_bound, solve, spectral_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
GRID = (F(0), F(1, 4), F(1, 2), F(1), F(2))
STEPS = 128
TOLERANCE = F(1, 1024)


def grade(outcome, evidence, detail):
    return dict(outcome='mixed' if outcome == 'partial' else outcome,
                evidence=evidence, detail=detail)


def spectral_cases():
    rows = []
    for g in GRID:
        q = spectral_matrix(g)
        eigen = eigendiagnostic(q)
        for e in (-1, 1):
            work = Work()
            b = (F(e), F(0), F(0), F(0))
            equilibrium = solve(q, b, work=work)
            base = solve(spectral_matrix(0), b)
            ratio = contraction(q, F(1, 16))
            trials = []
            for coord, delta, damaged in additive_disturbances(equilibrium):
                final = iterate(q, b, damaged, F(1, 16), STEPS)
                error = norm(difference(final, equilibrium))
                bound = ratio ** STEPS
                trials.append(dict(coordinate=coord, delta=delta, error=error,
                                   bound=bound, certified=error <= bound,
                                   within_tolerance=error <= TOLERANCE))
            withdrawn = iterate(q, (F(0),)*4, equilibrium, F(1, 16), STEPS)
            cutoffs = []
            for cutoff in (1, 2, 4, 8):
                # Float spectral construction, exact posterior rational check.
                components = [0.] * 4
                retained = 0
                for value, vector in zip(eigen['eigenvalues'], eigen['eigenvectors']):
                    if value <= cutoff + 1e-12:
                        coefficient = float(sum(x*y for x, y in zip(b, vector))) / value
                        components = [a + coefficient*v for a, v in zip(components, vector)]
                        retained += 1
                approximate = tuple(F(str(round(v, 12))) for v in components)
                actual = norm(difference(approximate, equilibrium))
                bound = residual_bound(q, b, approximate)
                cutoffs.append(dict(cutoff=cutoff, retained=retained, error=actual,
                                    residual_bound=bound, certified=actual <= bound))
            omitted_error = norm(difference(base, equilibrium))
            omission_bound = residual_bound(q, b, base)
            rows.append(dict(status='complete', response_status='exact_equilibrium',
                continuation_status='certified_error_bound',
                g=g, evidence=e, equilibrium=equilibrium, output=equilibrium[2],
                context_changes_output=equilibrium[2] != base[2],
                core_disabled_output=solve(spectral_matrix(g, core=False), b)[2],
                spectrum=eigen, dominance=dominance(q), contraction=ratio,
                recovery=trials, withdrawal_after_128=withdrawn, withdrawal_limit=(0,)*4,
                withdrawal_bound=ratio**STEPS * norm(equilibrium),
                spectral_cutoffs=cutoffs, omitted_context_error=omitted_error,
                omitted_context_bound=omission_bound, omission_certified=omitted_error <= omission_bound,
                construction_work=asdict(work), stored_matrix_entries=16))
    return dict(cases=rows, profile=dict(
        distinction=grade('pass', 'exact bounded check', 'Two different outputs at every declared g.'),
        context=grade('pass', 'exact bounded check', 'Positive context coupling changes output; g=0 control does not.'),
        recovery=grade('pass', 'contraction proof plus exact checks', 'Driven recovery; simultaneous step, delta=1 perturbations. No all-fair claim.'),
        persistence=grade('fail', 'positive-definite operator proof', 'After withdrawal every configuration converges to zero.'),
        selective_core_mediation=grade('unresolved', 'uncoupled control only', 'Core coupling affects response; selective compatibility creation is not tested.'),
        grading_nesting=grade('partial', 'grid and certified cutoff checks', 'Coupling and spectral cutoffs tested; no architectural layer advantage.'),
        topology=grade('not_claimed', 'not tested', 'No cell-incidence comparison in this candidate.'),
        approximation=grade('pass', 'diagonal-dominance bound', 'Exact posterior residual bound; not a universal pruning rule.'),
        learning=grade('unresolved', 'untested', 'No structural update law.')))


def algebra_cases():
    rows = []
    for context, e in product((0, 1), (1, 2)):
        work = Work()
        response = algebra_response(e, context, work=work)
        starts = [algebra_response(e, context, start) for start in product(range(3), repeat=4)]
        driven_trials, released_trials = [], []
        for coord, value, damaged in replacements(response, range(3)):
            driven = algebra_response(e, context, damaged)
            released = algebra_response(None, context, damaged)
            driven_trials.append(dict(coordinate=coord, value=value, restored=driven == response))
            released_trials.append(dict(coordinate=coord, value=value,
                after=released, restores_output=released[2] == response[2]))
        rows.append(dict(status='complete', response_status='exact_selected_configuration',
            evidence=e, context=context, response=response, output=response[2],
            initialized_cases=len(starts), unique_from_all_starts=len(set(starts)) == 1,
            core_disabled=algebra_response(e, context, core=False),
            driven_recovery=driven_trials, withdrawal=algebra_response(None, context, response),
            withdrawal_recovery=released_trials, construction_work=asdict(work)))
    base_rows = ((-1, 1, 1, 0), (0, 1, 2, 1))
    filtration = [4-rank(base_rows, 3),
                  4-rank(algebra_system(None, 0)[0], 3),
                  4-rank(algebra_system(1, 0)[0], 3)]
    return dict(cases=rows, direction_dimensions=filtration, profile=dict(
        distinction=grade('pass', 'exhaustive finite checks', 'Unique different outputs for the two evidence labels in both contexts.'),
        context=grade('pass', 'exact finite checks', 'S=0 versus S=1 changes O under unchanged evidence.'),
        recovery=grade('pass', 'exhaustive finite checks', 'Driven one-step reconstruction, including all 81 starts.'),
        persistence=grade('pass', 'idempotent projection', 'Declared selector retains free O after withdrawal; this is a latch, not collective restoration.'),
        withdrawal_recovery=grade('fail', 'explicit finite counterexamples', 'Changing free O changes the retained response.'),
        grading_nesting=grade('partial', 'rank calculation', 'Affine filtration dimensions 2,1,0; field coefficients are not ordered strengths.'),
        selective_core_mediation=grade('unresolved', 'equation ablation only', 'A constraint system, not demonstrated enabling mediation.'),
        topology=grade('not_claimed', 'not tested', 'No cell topology in this model.'),
        approximation=grade('not_claimed', 'not specified', 'No weak graded coupling semantics.'),
        learning=grade('unresolved', 'untested', 'No structural update law.')))


def relational_cases():
    rows = []
    for g, context, e in product(GRID[1:], (0, 1), (0, 1)):
        work = Work()
        minima, best = relation_minima(e, context, g, work=work)
        response = relation_response(e, context, g)
        starts = [relation_response(e, context, g, state) for state in product((0, 1), repeat=4)]
        driven, released = [], []
        for source in minima:
            for coord, value, damaged in replacements(source, (0, 1)):
                driven_after = relation_response(e, context, g, damaged)
                released_after = relation_response(None, context, g, damaged)
                driven.append(dict(source=source, coordinate=coord, value=value,
                    after=driven_after, restores_output=driven_after[2] == source[2]))
                released.append(dict(source=source, coordinate=coord, value=value,
                    after=released_after, restores_output=released_after[2] == source[2]))
        outputs = sorted({state[2] for state in minima})
        rows.append(dict(status='complete', response_status='exact_minimum_set',
            g=g, context=context, evidence=e, minima=minima, best_cost=best,
            selected=response, output_set=outputs, unique_output=len(outputs) == 1,
            outputs_from_all_starts=sorted({state[2] for state in starts}),
            core_disabled=relation_response(e, context, g, core=False),
            driven_recovery=driven, withdrawal=relation_response(None, context, g, response),
            withdrawal_recovery=released, construction_work=asdict(work)))
    full = tuple(state for state in product((0, 1), repeat=4) if not (state[0]^state[1]^state[2]^state[3]))
    projections = [{tuple(v for j, v in enumerate(state) if j != omitted) for state in full}
                   for omitted in range(4)]
    reconstructed = [state for state in product((0, 1), repeat=4) if all(
        tuple(v for j, v in enumerate(state) if j != omitted) in projections[omitted]
        for omitted in range(4))]
    epsilon = F(1, 1024)
    exact = relation_response(0, 1, F(2), scale=epsilon)
    pruned = relation_response(0, 1, F(0), scale=epsilon)
    return dict(cases=rows, higher_arity=dict(allowed=len(full),
        projection_sizes=list(map(len, projections)), reconstructed=len(reconstructed),
        false_positives=[state for state in reconstructed if state not in full]),
        weak_term_counterexample=dict(scale=epsilon, dropped_weight=2*epsilon,
            exact=exact, pruned=pruned, output_error=abs(exact[2]-pruned[2])),
        profile=dict(
        distinction=grade('partial', 'exhaustive bounded check', 'Unique distinctions except g=1, context=1; ties expose both outputs.'),
        context=grade('partial', 'exhaustive bounded check', 'g<1 changes unique outputs; g>1 preserves output while core configuration changes.'),
        recovery=grade('partial', 'exhaustive bounded check', 'Unique minima reconstruct; g=1/context=1 admits competing responses.'),
        persistence=grade('pass', 'finite checks and selection rule', 'Keep-current-minimum continuation preserves selected responses, including ambiguous ones.'),
        withdrawal_recovery=grade('fail', 'finite counterexamples', 'Retention does not guarantee reconstruction of every prior response.'),
        grading_nesting=grade('partial', 'exact weight sweep', 'Preference threshold and tie detected; nesting not specified.'),
        selective_core_mediation=grade('unresolved', 'parity ablation only', 'Parity restricts compatibility; no enabled-combination core mechanism.'),
        higher_arity=grade('pass', 'projection counterexample', 'All three-coordinate projections lose the parity restriction.'),
        topology=grade('not_claimed', 'not tested', 'Arity is not cell topology.'),
        approximation=grade('fail', 'explicit counterexample', 'Small omitted cost alone cannot bound categorical response change without a margin.'),
        learning=grade('unresolved', 'untested', 'No structural update law.')))


def cellular_cases():
    rows = []
    for attached, g, e in product((False, True), GRID, (-1, 1)):
        work = Work()
        matrix = cell_matrix(g, attached)
        eq = cell_driven(g, attached, e, work)
        q = tuple(tuple(row[1:]) for row in matrix[1:])
        forcing = tuple(-row[0]*e for row in matrix[1:])
        ratio = contraction(q, F(1, 8))
        trials = []
        for coord, delta, damaged in additive_disturbances(eq):
            final = (F(e),) + iterate(q, forcing, damaged[1:], F(1, 8), STEPS)
            error = norm(difference(final, eq))
            limit = cell_withdrawal_limit(g, attached, damaged)
            trials.append(dict(coordinate=coord, delta=delta, driven_error=error,
                driven_bound=ratio**STEPS, certified=error <= ratio**STEPS,
                withdrawal_limit=limit,
                restores_released_output=limit[2] == cell_withdrawal_limit(g, attached, eq)[2]))
        released = iterate(matrix, (F(0),)*3, eq, F(1, 8), STEPS)
        limit = cell_withdrawal_limit(g, attached, eq)
        # Symmetry splits mean and zero-sum sectors; certificate uses their rates.
        mean = sum(eq)/3
        cyclic = (mean,)*3
        orthogonal = difference(eq, cyclic)
        mean_error = (abs(1-3*g/F(8))**STEPS * abs(mean)
                      if attached and g > 0 else F(0))
        withdrawal_bound = mean_error + F(5, 8)**STEPS * norm(orthogonal)
        approximate = (F(e),)*3  # omitting the face contribution
        approximation_error = norm(difference(approximate, eq))
        bound = residual_bound(q, forcing, approximate[1:])
        d2 = cell_boundary(attached)
        rows.append(dict(status='complete', response_status='exact_equilibrium',
            continuation_status='certified_error_bound',
            attached=attached, g=g, evidence=e, equilibrium=eq, output=eq[2],
            d1_d2=matmul(D1, d2), betti1=3-rank(D1)-rank(d2), betti2=1-rank(d2),
            harmonic_dimension=3-rank(matrix), contraction=ratio, recovery=trials,
            withdrawal_after_128=released, withdrawal_limit=limit,
            withdrawal_bound=withdrawal_bound,
            withdrawal_certified=norm(difference(released, limit)) <= withdrawal_bound,
            omitted_face_error=approximation_error, omitted_face_bound=bound,
            omission_certified=approximation_error <= bound, construction_work=asdict(work),
            stored_cell_counts=(3, 3, 1)))
    return dict(cases=rows, profile=dict(
        distinction=grade('partial', 'exact bounded checks', 'Driven outputs differ except disk at g=1, where both outputs are zero.'),
        context=grade('pass', 'matched cell-count comparison', 'Changing 2-cell attachment changes positive-weight responses; this is structural context.'),
        recovery=grade('pass', 'contraction proof and exact checks', 'Driven reconstruction, with input re-clamped; released harmonic modes are not protected.'),
        persistence=grade('partial', 'exact kernel and limit', 'Constant attachment retains a harmonic response; positive-weight disk erases it.'),
        withdrawal_recovery=grade('fail', 'explicit harmonic counterexamples', 'A single edge perturbation changes the conserved mean by delta/3.'),
        grading_nesting=grade('partial', 'exact weights and cell filtration', 'Cellular nesting and weight sensitivity; no abstraction hierarchy or layer advantage.'),
        topology=grade('pass', 'exact boundary/rank checks', 'Same cell counts and graph skeleton, different attachment and first homology. At g=0 operator kernel is not a topology certificate.'),
        selective_core_mediation=grade('unresolved', 'not tested', 'A cochain model, not a structurally specialized mediating core.'),
        approximation=grade('partial', 'residual bound and limiting counterexample', 'Driven omission bound holds; dropping any positive face weight can change infinite-time persistence.'),
        learning=grade('unresolved', 'untested', 'No structural update law.')))


RUNNERS = {'spectral': spectral_cases, 'finite_algebra': algebra_cases,
           'graded_relation': relational_cases, 'cellular': cellular_cases}


def jsonable(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def render(value):
    return json.dumps(jsonable(value), indent=2, sort_keys=True) + '\n'


def certificate(measure=False):
    families, receipt = {}, {}
    for name, runner in RUNNERS.items():
        if measure:
            tracemalloc.start()
        started = time.perf_counter()
        families[name] = runner()
        if measure:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            receipt[name] = dict(seconds_including_instrumentation=time.perf_counter()-started,
                                 peak_python_bytes=peak)
    paths = ('docs/comparison-v1-contract.md', 'src/model_comparison.py',
             'experiments/model_comparison.py')
    report = dict(specification='Comparative modeling v1', status='complete',
        runtime=dict(python=platform.python_version(), implementation=platform.python_implementation()),
        source_sha256={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
        bounds=dict(spectral_cases=10, algebra_cases=4, relational_cases=16, cellular_cases=20,
                    continuation_steps=STEPS, tolerance=TOLERANCE), families=families,
        cost_units='Work.arithmetic counts RREF multiplication/subtraction operations only; '
                   'Work.candidates counts inspected tuples. Neither is total CPU work.',
        scope='Four bounded representatives, no family-wide ranking or selected SRI architecture')
    return report, receipt


if __name__ == '__main__':
    report, receipt = certificate(measure=True)
    path = ROOT / 'reports' / 'model-comparison-v1.json'
    path.write_text(render(report))
    (ROOT / 'reports' / 'model-comparison-v1-runtime.json').write_text(render(dict(
        runtime=report['runtime'], measurements=receipt,
        note='Includes certificate construction and allocation tracing; not a cross-family inference benchmark')))
    print(path)
    for name, family in report['families'].items():
        print(name, len(family['cases']), 'cases')
