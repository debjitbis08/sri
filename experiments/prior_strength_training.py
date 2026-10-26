"""Prior-strength training v1: does training with weaker activity priors keep the repaired completion?

Implements docs/prior-strength-training-v1-contract.md. Learners see only observed
outer activities and episode IDs; labels, maps, shared directions, target means and
the decorrelated sets are evaluator only.

The activity-prior Hessian lambda enters through Q. Every evaluation helper reused
from earlier studies builds Q with lcc.build, so `priors(lam)` rebinds lcc.build to
a lambda-aware version for the duration of a condition; the original is restored on
exit and verified to be identical at lambda = 0.05.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
from experiments import local_consolidation as base
from experiments import local_contrastive_consolidation as lcc
from experiments import clamped_energy_ablation as cea
from experiments import correlated_nuisance as cn
from experiments import inner_channel_diagnostic as icd
from experiments import energy_scaling_analysis as esa
from experiments import prior_intervention_analysis as pia

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/prior-strength-training-v1-contract.md'
ICD_RESULTS = ROOT / 'reports/inner-channel-diagnostic-v1-results.json.gz'
PREFIX = Path(os.environ.get('PST_PREFIX', str(ROOT / 'reports/prior-strength-training-v1')))
C = int(os.environ.get('PST_CYCLES', '300'))  # smoke runs override; the contract fixes 300 per stage
ETA_GRID = (.001, .003, .01, .03, .1, .3)
PILOT_SEEDS = (0, 1, 2)
SEEDS = tuple(int(s) for s in os.environ.get('PST_SEEDS', '43,44,45,46,47,48,49,50,51,52').split(','))
REGIMES = ('independent', 'weak')
BUDGETS = (('2', 2.), ('10', 10.))
PRIORS = (('0.05', .05), ('0.005', .005), ('0.001', .001))
ARMS = {'E': dict(family='inner', exact=False, decay=True, clip=True),
        'E_exact': dict(family='inner', exact=True, decay=True, clip=True),
        'E_noclip': dict(family='inner', exact=False, decay=True, clip=False),
        'E_nodecay': dict(family='inner', exact=False, decay=False, clip=True),
        'E_free': dict(family='inner', exact=False, decay=False, clip=False),
        'U': dict(family='uniform', exact=False, decay=True, clip=True)}
CHECKPOINT = 50
GRAD_FLOOR, GRAD_TOL = 1e-8, 1e-6
FLAG_EIG, FLAG_COND = 1e-6, 1e8
SEED_COUNT = cea.SEED_COUNT
ORIGINAL_BUILD = lcc.build
COUNTS = {}
PHASES = {}
CPU = {}


def count(key, n=1):
    COUNTS[key] = COUNTS.get(key, 0) + n


@contextlib.contextmanager
def phase(name):
    w, c = time.perf_counter(), time.process_time()
    yield
    PHASES[name] = {'wall_seconds': time.perf_counter() - w, 'cpu_seconds': time.process_time() - c}


def build_with(model, p, lam):
    """ORIGINAL_BUILD with both activity-prior Hessians set to lam (pia.prior_q's adjustment)."""
    q = ORIGINAL_BUILD(model, p)
    if lam != pia.BASE:
        for m in range(3):
            for layer in (1, 2):
                idx = model.sl(m, layer)
                q[np.ix_(idx, idx)] += (lam - pia.BASE) * np.eye(len(idx))
    return q


@contextlib.contextmanager
def priors(lam):
    lcc.build = lambda model, p: build_with(model, p, lam)
    try:
        yield
    finally:
        lcc.build = ORIGINAL_BUILD


def model_for(arm, b):
    return cn.model_for(ARMS[arm]['family'], b)


def name_of(regime, arm, lname, bname):
    return f'{regime}_{arm}_{lname}_{bname}'


def conditions():
    return [(name_of(r, a, ln, bn), r, a, ln, lam, bn, b) for r in REGIMES for a in ARMS for ln, lam in PRIORS
            for bn, b in BUDGETS]


def top_singular(p):
    return {k: [float(np.linalg.svd(p[k][m] if k in ('W', 'V') else p[k], compute_uv=False)[0])
                for m in (range(3) if k in ('W', 'V') else [0])] for k in p}


def evaluate(model, p, d, train_data, lam):
    """Under priors(lam): the diagnostic's evaluation plus signal gain, conditioning and singular values."""
    out = icd.evaluate(model, p, d, train_data)
    extra = pia.measure(model, lcc.build(model, p), d)
    out['signal_gain'] = {k: extra[f'signal_gain_{k}'] for k in cn.SETS}
    out['conditioning'] = {k: extra[k] for k in ('max_masked_free_block_condition', 'min_eigenvalue_Q')}
    out['top_singular_values'] = top_singular(p)
    return out


def run(model, p, stages, eta, cycles, arm, lam, d=None, settle_check=False, label=None):
    """Clamped-energy Adam under priors(lam); the arm sets exact settling, decay and clips."""
    cfg = ARMS[arm]
    keys = list(model.keys)
    theta = np.concatenate([p[k].ravel() for k in keys])
    initial = theta.copy()
    bound = np.concatenate([np.full(p[k].size, 3. if k.startswith('T') else 2.) for k in keys])
    wd = np.concatenate([np.full(p[k].size, lcc.DECAY if k in ('W', 'V') and cfg['decay'] else 0.) for k in keys])
    slices, i = {}, 0
    for k in keys:
        slices[k] = slice(i, i + p[k].size)
        i += p[k].size
    groups = {}
    for k in keys:
        groups.setdefault(cea.group(k), []).append(np.arange(slices[k].start, slices[k].stop))
    groups = {g: np.concatenate(v) for g, v in groups.items()}
    current = lambda: {k: theta[slices[k]].reshape(p[k].shape) for k in keys}
    first, second, step = np.zeros_like(theta), np.zeros_like(theta), 0
    clips = {g: 0 for g in groups}
    check = {'max': 0., 'argmax': None, 'stage_max': [0., 0.], 'updates_above_tol': 0}
    evaluations, curves, cpu_curve = [], [], []
    cumulative, train_cpu, train_wall, diverged = 0, 0., 0., False
    with priors(lam):
        if d is not None:
            evaluations.append(evaluate(model, current(), d, stages[0][1], lam))
        for index, (waking, data) in enumerate(stages):
            frozen = lcc.digest(current())
            lcc.settle(lcc.build(model, current()), model.mask([]), waking.reshape(len(waking), 24))
            count('waking_episode_exposures', len(waking))
            assert lcc.digest(current()) == frozen
            flat = data.reshape(len(data), 24)
            M = model.mask([])
            for cycle in range(cycles[index]):
                c0, w0 = time.process_time(), time.perf_counter()
                params = current()
                q = lcc.build(model, params)
                if cfg['exact']:
                    g = lcc.param_grad(model, params, lcc.assemble(model, M, flat, lcc.exact_free(q, M, flat)))
                else:
                    g = lcc.energy_grad(model, params, q, flat)
                count('training_trace_exposures', len(flat))
                gradient = np.concatenate([g[k].ravel() for k in keys]) + wd * theta
                step += 1
                first = lcc.ADAM[0] * first + (1 - lcc.ADAM[0]) * gradient
                second = lcc.ADAM[1] * second + (1 - lcc.ADAM[1]) * gradient ** 2
                update = eta * (first / (1 - lcc.ADAM[0] ** step)) / (
                    np.sqrt(second / (1 - lcc.ADAM[1] ** step)) + lcc.ADAM[2])
                proposed = theta - update
                new = np.clip(proposed, -bound, bound) if cfg['clip'] else proposed
                for name, idx in groups.items():
                    clips[name] += int(np.sum(new[idx] != proposed[idx]))
                count('adam_steps')
                train_cpu += time.process_time() - c0
                train_wall += time.perf_counter() - w0
                if settle_check:  # measurement only; does not change the update
                    gs = lcc.param_grad(model, params, lcc.assemble(model, M, flat, lcc.exact_free(q, M, flat)))
                    a = np.concatenate([g[k].ravel() for k in keys])
                    b = np.concatenate([gs[k].ravel() for k in keys])
                    r = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), GRAD_FLOOR))
                    if r > check['max']:
                        check['max'], check['argmax'] = r, cumulative
                    check['stage_max'][index] = max(check['stage_max'][index], r)
                    check['updates_above_tol'] += int(r > GRAD_TOL)
                    count('settle_check_exact_solves')
                theta = new
                cumulative += 1
                if not np.all(np.isfinite(theta)):
                    diverged = True
                    break
                if d is not None and (cycle + 1) % CHECKPOINT == 0:
                    point = cn.curve_point(model, current(), d, data)
                    point.update(stage=index, stage_cycle=cycle + 1, cumulative_cycles=cumulative,
                                 displacement_from_init=float(np.linalg.norm(theta - initial)))
                    curves.append(point)
                    cpu_curve.append([cumulative, train_cpu, train_wall])
            if diverged:
                break
            if d is not None:
                evaluations.append(evaluate(model, current(), d, data, lam))
    final = current()
    if label is not None:
        CPU[label] = {'training_cpu_seconds': train_cpu, 'training_wall_seconds': train_wall,
                      'columns': ['cumulative_cycles', 'training_cpu_seconds', 'training_wall_seconds'],
                      'rows': cpu_curve}
    record = {'diverged': diverged, 'evaluations': evaluations, 'curves': curves, 'clip_activations': clips,
              'cycles': cumulative, 'displacement_from_init': float(np.linalg.norm(theta - initial)),
              'final_parameters': {k: v.tolist() for k, v in final.items()}}
    if settle_check:
        record['settle_check'] = check
    return final, record


def cross_evaluate(model, p, d, own):
    out = {}
    for lname, lam in PRIORS:
        m = pia.measure(model, build_with(model, p, lam), d)
        out[lname] = m
        if lname == own:
            out['own_matches_record'] = m
    return out


def verify():
    out = {'carried_over_from_inner_channel_diagnostic': icd.verify()}
    rng = np.random.default_rng(31)
    # The prior-modified Q equals lcc.build at 0.05 and the explicit energy's Hessian at weaker priors.
    model = model_for('E', 2.)
    p = lcc.initialize(model, 99)
    for k in p:
        p[k] = p[k] + rng.normal(0, .1, p[k].shape)
    assert np.array_equal(build_with(model, p, .05), ORIGINAL_BUILD(model, p))
    worst = 0.
    for lam in (.005, .001):
        q = build_with(model, p, lam)
        s = rng.normal(size=model.S)
        split = lambda v: ([v[model.sl(m, 0)] for m in range(3)], [v[model.sl(m, 1)] for m in range(3)],
                           [v[model.sl(m, 2)] for m in range(3)])
        E = lambda v: esa.energy(*split(v), p['W'], p['V'], [p['T0_2'], p['T1_2']], model.kappa[2],
                                 (1., 1., lam, lam, 1.))
        for i in range(model.S):
            e = np.zeros(model.S)
            e[i] = 1e-6
            worst = max(worst, abs((E(s + e) - E(s - e)) / 2e-6 - (q @ s)[i]))
    out['prior_q_finite_difference'] = worst
    assert worst < 1e-6
    # priors() rebinds and restores lcc.build.
    with priors(.005):
        assert np.array_equal(lcc.build(model, p), build_with(model, p, .005))
    assert lcc.build is ORIGINAL_BUILD
    # The 32-step gradient approaches the exact-equilibrium gradient as the step count grows.
    d = cn.fixture(99)['weak']
    flat = np.concatenate(d['waking']).reshape(32, 24)
    q, M = build_with(model, p, .005), model.mask([])
    ge = lcc.param_grad(model, p, lcc.assemble(model, M, flat, lcc.exact_free(q, M, flat)))
    gl = lcc.param_grad(model, p, lcc.assemble(model, M, flat, lcc.settle(q, M, flat, steps=4000)[0]))
    out['long_settle_gradient_gap'] = float(max(np.max(abs(ge[k] - gl[k])) for k in ge))
    assert out['long_settle_gradient_gap'] < 1e-10
    # Clips and decay follow the arm table.
    start = lcc.initialize(model, 99)
    start['W'] = np.full_like(start['W'], 2.5)
    data = np.concatenate(d['waking'])
    for arm in ('E', 'E_noclip', 'E_nodecay', 'E_free'):
        final, _ = run(model, {k: v.copy() for k, v in start.items()}, [(d['waking'][0], data)], 1e-6, [1], arm, .05)
        out[f'{arm}_W_after_one_step'] = float(np.max(abs(final['W'])))
        assert (out[f'{arm}_W_after_one_step'] <= 2) == ARMS[arm]['clip']
    for arm in ARMS:
        assert ARMS[arm]['decay'] == (arm in ('E', 'E_exact', 'E_noclip', 'U'))
    # Q is positive definite at initialization for every prior, arm family and budget.
    out['min_initial_eigenvalue'] = min(float(np.linalg.eigvalsh(build_with(model_for(a, b), lcc.initialize(
        model_for(a, b), s), lam))[0]) for a in ('E', 'U') for _, b in BUDGETS for _, lam in PRIORS for s in (0, 43))
    assert out['min_initial_eigenvalue'] > 0
    # Reproduction: seed 33 weak / E / 10 at lambda 0.05 and eta 0.1 matches the diagnostic.
    old = json.loads(gzip.decompress(ICD_RESULTS.read_bytes()))['seeds']['33']['arms']['weak_E_10']
    d = cn.fixture(33)['weak']
    model = model_for('E', 10.)
    _, record = run(model, lcc.initialize(model, 33), cea.stages_for(d['waking']), .1, [300, 300], 'E', .05, d)
    now, then = record['evaluations'][-1]['nuisance']['exact'], old['evaluations'][-1]['nuisance']['exact']
    out['reproduction_inner_channel_diagnostic_seed33_weak_E_10'] = {
        w: abs(now[w]['combined']['avoidable_mse'] - then[w]['combined']['avoidable_mse']) for w in cn.SETS}
    assert max(out['reproduction_inner_channel_diagnostic_seed33_weak_E_10'].values()) < 1e-12
    return out


def pilot(datasets):
    rows, selected = {}, {}
    for name, regime, arm, lname, lam, bname, b in conditions():
        rows[name] = []
        for eta in ETA_GRID:
            scores, bad = [], False
            for seed in PILOT_SEEDS:
                d = datasets[seed][regime]
                model = model_for(arm, b)
                final, record = run(model, lcc.initialize(model, seed), cea.stages_for(d['waking'], 0), eta, [C, C],
                                    arm, lam)
                odd = np.concatenate([d['waking'][0][1::2], d['waking'][1][1::2]]).reshape(-1, 24)
                s = None
                if not record['diverged']:
                    with priors(lam):
                        s = cn.masked_mse_exact(model, final, odd)
                bad |= s is None or not np.isfinite(s)
                scores.append(s)
            rows[name].append({'eta': eta, 'scores_by_seed': scores, 'disqualified': bad,
                               'mean_score': None if bad else float(np.mean(scores))})
        eligible = [r for r in rows[name] if not r['disqualified']]
        if eligible:
            best = min(eligible, key=lambda r: (r['mean_score'], r['eta']))
            selected[name] = {'eta': best['eta'], 'at_grid_edge': best['eta'] in (ETA_GRID[0], ETA_GRID[-1]),
                              'failed': False}
        else:
            selected[name] = {'eta': None, 'at_grid_edge': False, 'failed': True}
    return {'grids': rows, 'selected': selected,
            'score': 'exact-equilibrium observed masked MSE under the condition\'s own prior, on odd-index waking '
                     'traces after stage B; training on even-index traces'}


def criteria(seeds, selection):
    order = sorted(seeds, key=int)
    failed = lambda n: selection['selected'][n]['failed'] or any(seeds[s]['conditions'][n]['diverged'] for s in order)
    last = lambda s, n: seeds[s]['conditions'][n]['evaluations'][-1]
    Xdec = lambda n: [last(s, n)['nuisance']['exact']['decorrelated']['combined']['avoidable_mse'] for s in order]
    Xin = lambda n: [last(s, n)['nuisance']['exact']['in_distribution']['combined']['avoidable_mse'] for s in order]
    gam = lambda n: [last(s, n)['nuisance']['exact']['gain']['gamma'] for s in order]
    sgain = lambda n: [last(s, n)['signal_gain']['decorrelated'] for s in order]
    cross = lambda n, ln: [seeds[s]['conditions'][n]['cross_evaluation'][ln]['X_decorrelated'] for s in order]
    flagged = lambda n: any(last(s, n)['conditioning']['min_eigenvalue_Q'] <= FLAG_EIG
                            or last(s, n)['conditioning']['max_masked_free_block_condition'] >= FLAG_COND for s in order)
    out = {k: {} for k in ('N1', 'summary', 'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7')}

    def compare(x1, x2, names):
        """x1, x2: thunks, evaluated only when no named condition failed."""
        if any(failed(n) for n in names):
            return {'status': 'failed_condition', 'conclusion_blocked': True}
        r = cea.compare(x1(), x2())
        r['conclusion_blocked'] = any(flagged(n) for n in names)
        return r

    for name, regime, arm, lname, lam, bname, b in conditions():
        if failed(name):
            out['summary'][name] = {'failed': True}
            continue
        ev = [[e['nuisance']['exact']['in_distribution'] for e in seeds[s]['conditions'][name]['evaluations']] for s in order]
        b_falls = sum(e[2]['B']['avoidable_mse'] < e[1]['B']['avoidable_mse'] for e in ev)
        a_ratio = [(e[2]['A']['avoidable_mse'] + cea.DELTA) / (e[1]['A']['avoidable_mse'] + cea.DELTA) for e in ev]
        frozen = [seeds[s]['frozen'][f'{regime}_{ARMS[arm]["family"]}_{lname}_{bname}']['nuisance']['exact']
                  ['in_distribution']['combined']['avoidable_mse'] for s in order]
        to_frozen = (np.mean(Xin(name)) + cea.DELTA) / (np.mean(frozen) + cea.DELTA)
        a_ok = sum(r <= 1.5 for r in a_ratio)
        out['N1'][name] = {'B_falls_seeds': int(b_falls), 'A_retained_seeds': int(a_ok),
                           'ratio_to_own_frozen': float(to_frozen),
                           'learns': bool(b_falls >= SEED_COUNT and a_ok >= SEED_COUNT and to_frozen <= .1)}
        g = gam(name)
        cond = [last(s, name)['conditioning'] for s in order]
        out['summary'][name] = {
            'mean_X_in': float(np.mean(Xin(name))), 'mean_X_dec': float(np.mean(Xdec(name))),
            'mean_gamma': None if g[0] is None else float(np.mean(g)),
            'category': cn.category(g) if regime == 'weak' else 'undefined',
            'mean_signal_gain': float(np.mean(sgain(name))),
            'mean_signal_error_dec': float(np.mean([last(s, name)['nuisance']['exact']['decorrelated']['combined']
                                                    ['signal_error_mse'] for s in order])),
            'min_eigenvalue_Q': float(min(c['min_eigenvalue_Q'] for c in cond)),
            'max_condition': float(max(c['max_masked_free_block_condition'] for c in cond)),
            'mean_top_W': float(np.mean([np.mean(last(s, name)['top_singular_values']['W']) for s in order])),
            'mean_top_V': float(np.mean([np.mean(last(s, name)['top_singular_values']['V']) for s in order])),
            'mean_top_T': float(np.mean([np.mean([v[0] for k, v in last(s, name)['top_singular_values'].items()
                                                  if k.startswith('T')]) for s in order])),
            'clip_activations': int(sum(sum(seeds[s]['conditions'][name]['clip_activations'].values()) for s in order)),
            'flagged': flagged(name), 'eta': selection['selected'][name]['eta']}
        if arm in ('E', 'U'):
            out['summary'][name]['cross_evaluation_mean_X_dec'] = {ln: float(np.mean(cross(name, ln))) for ln, _ in PRIORS}
        if 'settle_check' in seeds[order[0]]['conditions'][name]:
            sc = [seeds[s]['conditions'][name]['settle_check'] for s in order]
            out['summary'][name]['settle_check'] = {'max': max(c['max'] for c in sc),
                                                    'stage_max': [max(c['stage_max'][i] for c in sc) for i in (0, 1)],
                                                    'mean_updates_above_tol': float(np.mean([c['updates_above_tol'] for c in sc]))}
    cat = lambda n: out['summary'][n].get('category', 'failed')
    refs = {}
    for regime in REGIMES:
        refs[regime] = {}
        for comp in ('uncentered_rank_2', 'ridge_intercept'):
            rows = [seeds[s]['comparators'][regime][comp][-1] for s in order]
            refs[regime][comp] = {'X_dec': [r['decorrelated']['combined']['avoidable_mse'] for r in rows],
                                  'mean_X_dec': float(np.mean([r['decorrelated']['combined']['avoidable_mse'] for r in rows]))}
    out['summary']['comparators'] = refs
    for bname, _ in BUDGETS:
        n = lambda arm, ln: name_of('weak', arm, ln, bname)
        base_e = n('E', '0.05')
        if not failed(base_e):
            out['P0'][bname] = {'category': cat(base_e), 'mean_signal_gain': out['summary'][base_e]['mean_signal_gain'],
                                'mean_X_dec': out['summary'][base_e]['mean_X_dec']}
        for ln, _ in PRIORS:
            e = n('E', ln)
            key = f'{ln}_{bname}'
            if ln != '0.05':
                r = compare(lambda: Xdec(base_e), lambda: Xdec(e), [base_e, e])
                if r['status'] == 'first_clearly_worse' and not r['conclusion_blocked']:
                    r['conclusion'] = ('weaker_prior_training_improves' if cat(e) == 'filters'
                                       else 'improves_but_loses_filtering')
                else:
                    r['conclusion'] = 'no_improvement_found'
                r['E_category'] = cat(e)
                out['P1'][key] = r
                if not failed(base_e) and not failed(e):
                    r2 = compare(lambda: cross(base_e, ln), lambda: Xdec(e), [base_e, e])
                    r2['conclusion'] = {'first_clearly_worse': 'training_improves_on_inference_only_repair',
                                        'second_clearly_worse': 'training_erodes_repair',
                                        'comparable': 'training_preserves_repair'}.get(r2['status'], 'unresolved')
                    if r2.get('conclusion_blocked'):
                        r2['conclusion'] = 'no_conclusion_flagged'
                    out['P2'][key] = r2
            u = n('U', ln)
            r3 = compare(lambda: Xdec(u), lambda: Xdec(e), [u, e])
            ok = r3.get('status') == 'first_clearly_worse' and not r3['conclusion_blocked'] and cat(e) == 'filters'
            r3['conclusion'] = 'inner_learner_beats_surface_coupling' if ok else 'no_such_result'
            if not failed(e):
                r3['vs_U_0.05'] = cea.compare(Xdec(n('U', '0.05')), Xdec(e)) if not failed(n('U', '0.05')) else None
                r3['vs_uncentered_rank_2'] = cea.compare(refs['weak']['uncentered_rank_2']['X_dec'], Xdec(e))
            out['P3'][key] = r3
            if not failed(e):
                sg = sgain(e)
                out['P4'][key] = {'mean_signal_gain': float(np.mean(sg)), 'seeds_at_least_0.9': int(sum(x >= .9 for x in sg)),
                                  'restored': bool(np.mean(sg) >= .95 and sum(x >= .9 for x in sg) >= SEED_COUNT)}
            p5 = {}
            for arm in ('E_noclip', 'E_nodecay', 'E_free'):
                a = n(arm, ln)
                r5 = compare(lambda: Xdec(e), lambda: Xdec(a), [e, a])
                r5['conclusion'] = {'first_clearly_worse': 'removing_helps_under_this_protocol',
                                    'second_clearly_worse': 'removing_does_worse'}.get(r5['status'], 'no_difference_established')
                if r5.get('conclusion_blocked'):
                    r5['conclusion'] = 'no_conclusion_flagged_or_failed'
                r5['arm_category'] = cat(a)
                p5[arm] = r5
            out['P5'][key] = p5
            x = n('E_exact', ln)
            r6 = compare(lambda: Xdec(e), lambda: Xdec(x), [e, x])
            r6['conclusion'] = {'first_clearly_worse': 'exact_training_helps',
                                'second_clearly_worse': 'exact_training_does_worse',
                                'comparable': 'no_effect_detected'}.get(r6['status'], 'unresolved')
            if r6.get('conclusion_blocked'):
                r6['conclusion'] = 'no_conclusion_flagged_or_failed'
            r6['E_settle_check'] = out['summary'].get(e, {}).get('settle_check')
            out['P6'][key] = r6
    out['P7'] = {n: {k: out['summary'][n][k] for k in ('min_eigenvalue_Q', 'max_condition', 'mean_top_W', 'mean_top_V',
                                                        'mean_top_T', 'flagged')}
                 for n in out['summary'] if n != 'comparators' and not out['summary'][n].get('failed')}
    return out


def main():
    paths = [Path(str(PREFIX) + s) for s in ('-results.json', '-runtime.json')]
    if any(f.exists() or Path(str(f) + '.gz').exists() for f in paths):
        raise RuntimeError('Refusing overwrite')
    result, runtime = paths
    wall, cpu = time.perf_counter(), time.process_time()
    with phase('verification'):
        checks = verify()
    with phase('fixture_construction'):
        datasets = {seed: {r: cn.fixture(seed)[r] for r in REGIMES} for seed in PILOT_SEEDS + SEEDS}
    with phase('selection_pilot'):
        selection = pilot(datasets)
    science = {'checks': checks, 'selection': selection, 'cycles_per_stage': C, 'seeds': {},
               'arms': ARMS, 'priors': dict(PRIORS),
               'scope': 'clamped-energy block-local learning with global settling; labels, maps, shared directions, '
                        'decorrelated sets and target means evaluator only'}
    for seed in SEEDS:
        row = {'frozen': {}, 'conditions': {}, 'comparators': {}}
        initial = lcc.digest(base.initialize(seed))
        for regime in REGIMES:
            d = datasets[seed][regime]
            with phase(f'seed{seed}_{regime}_frozen_and_comparators'):
                for fam in ('inner', 'uniform'):
                    for lname, lam in PRIORS:
                        for bname, b in BUDGETS:
                            model = cn.model_for(fam, b)
                            with priors(lam):
                                row['frozen'][f'{regime}_{fam}_{lname}_{bname}'] = evaluate(
                                    model, lcc.initialize(model, seed), d, d['waking'][0], lam)
                comps = icd.comparators(d)
                row['comparators'][regime] = {k: comps[k] for k in ('uncentered_rank_2', 'ridge_intercept')}
        for name, regime, arm, lname, lam, bname, b in conditions():
            sel = selection['selected'][name]
            if sel['failed']:
                row['conditions'][name] = {'diverged': True, 'failed_in_pilot': True}
                continue
            d = datasets[seed][regime]
            model = model_for(arm, b)
            p = lcc.initialize(model, seed)
            assert lcc.digest({k: p[k] for k in ('W', 'V')}) == initial
            with phase(f'seed{seed}_{name}'):
                final, record = run(model, p, cea.stages_for(d['waking']), sel['eta'], [C, C], arm, lam, d,
                                    settle_check=not ARMS[arm]['exact'], label=f'seed{seed}_{name}')
                if arm in ('E', 'U') and not record['diverged']:
                    record['cross_evaluation'] = cross_evaluate(model, final, d, lname)
                    own = record['cross_evaluation'].pop('own_matches_record')
                    assert abs(own['X_decorrelated'] - record['evaluations'][-1]['nuisance']['exact']['decorrelated']
                               ['combined']['avoidable_mse']) < 1e-12
            record.update(arm=arm, regime=regime, prior=lam, budget=b, family=ARMS[arm]['family'], eta=sel['eta'])
            row['conditions'][name] = record
        science['seeds'][str(seed)] = row
    with phase('criteria'):
        science['criteria'] = criteria(science['seeds'], selection)
    with phase('serialization'):
        result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
    with phase('provenance'):
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in (Path(__file__), Path(icd.__file__), Path(cn.__file__), Path(pia.__file__),
                            Path(esa.__file__), Path(lcc.__file__), Path(base.__file__), Path(cea.__file__),
                            CONTRACT, ICD_RESULTS)}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)

    def limit(name):
        p = Path('/sys/fs/cgroup') / name
        return p.read_text().strip() if p.exists() else None
    counts = dict(COUNTS)
    for prefix, mod in (('icd', icd), ('cn', cn), ('ablation', cea), ('lcc', lcc), ('v1_module', base)):
        counts.update({f'{prefix}_{k}': v for k, v in mod.COUNTS.items()})
    storage = {'parameters_inner': 128 * 8, 'parameters_uniform': 288 * 8, 'adam': '16P + 8',
               'replay': 32 * 24 * 8, 'dense_q': 42 * 42 * 8, 'clamped_settle_workspace': (18 * 18 + 32 * 18) * 8,
               'settle_check_accumulators': 48}
    info = {'phases': PHASES, 'counts': counts, 'training_cpu': CPU, 'hashes': hashes, 'revision': revision,
            'dirty_status': dirty, 'command': ' '.join(sys.argv), 'pilot_seeds': list(PILOT_SEEDS),
            'confirmatory_seeds': list(SEEDS), 'cycles_per_stage': C,
            'seed_policy': 'v1 SeedSequence [seed,1001] initialization; [seed,2002] v1 fixture; [seed,3003] shared nuisance',
            'python': sys.version, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
            'os': platform.platform(), 'blas': config.getvalue(),
            'thread_environment': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'workers': 1, 'cpu': platform.processor() or Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
            'affinity_cpu_count': len(os.sched_getaffinity(0)), 'cpu_limit': limit('cpu.max'),
            'memory_limit': limit('memory.max'), 'dtype': 'float64',
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'rss_scope': 'Linux single-process lifetime peak; all reporting arrays included',
            'accelerator_memory': None, 'accelerator_reason': 'CPU only',
            'manual_design_seconds': None, 'manual_design_reason': 'not measured',
            'storage_bytes_per_learner': storage,
            'timing_scope': 'whole-run/phase timings including evaluation; training CPU excludes evaluation and the '
                            'settle check; not a latency benchmark',
            'wall_seconds_before_runtime_serialization': time.perf_counter() - wall,
            'cpu_seconds_before_runtime_serialization': time.process_time() - cpu}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: science['criteria'][k] for k in ('P0', 'P1', 'P2', 'P3', 'P4')}, indent=1))


if __name__ == '__main__':
    main()
