"""Inner-channel diagnostic v1: can inner coupling keep its filtering and recover the signal?

Implements docs/inner-channel-diagnostic-v1-contract.md. Arms E and U (clamped
energy) and M (masked completion) see only observed outer activities and
episode IDs. Arms S and S' are evaluator-assisted capacity fits to the
signal-only mean and are not learners. Labels, maps, shared directions, target
means and the decorrelated sets are otherwise evaluator only.
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
from experiments import correlated_nuisance_posthoc as cnp

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/inner-channel-diagnostic-v1-contract.md'
CN_RESULTS = ROOT / 'reports/correlated-nuisance-v1-results.json.gz'
PREFIX = Path(os.environ.get('ICD_PREFIX', str(ROOT / 'reports/inner-channel-diagnostic-v1')))
C = int(os.environ.get('ICD_CYCLES', '300'))  # smoke runs override; the contract fixes 300 per stage
CS = int(os.environ.get('ICD_FIT_CYCLES', '3000'))  # the contract fixes 3,000 for S and S'
ETA_GRID = (.001, .003, .01, .03, .1)
PILOT_SEEDS = (0, 1, 2)
SEEDS = tuple(int(s) for s in os.environ.get('ICD_SEEDS', '33,34,35,36,37,38,39,40,41,42').split(','))
REGIMES = ('independent', 'weak')
BUDGETS = (('2', 2.), ('10', 10.))
ARMS = {'E': ('inner', 'energy'), 'U': ('uniform', 'energy'), 'M': ('inner', 'masked'),
        'S': ('inner', 'signal'), 'S_prime': ('inner', 'signal')}
LEARNERS, FITS = ('E', 'U', 'M'), ('S', 'S_prime')
CHECKPOINT, FIT_CHECKPOINT = 50, 250
GRAD_FLOOR, SETTLE_TOL, GRAD_TOL = 1e-8, 1e-8, 1e-6
SEED_COUNT = cea.SEED_COUNT
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


def model_for(arm, b):
    return cn.model_for(ARMS[arm][0], b)


def masked_grad(model, p, q, flat, target):
    """Adjoint gradient of the exact-equilibrium masked loss against `target` (n,24); lcc.exact_grad's form."""
    g, loss = lcc.zero_grad(p), 0.
    for m in range(3):
        M = model.module_mask(m)
        _, _, free, hid, tcols = M
        y = lcc.exact_free(q, M, flat)
        residual = np.zeros_like(y)
        residual[:, hid] = y[:, hid] - target[:, tcols]
        adjoint = np.linalg.solve(q[np.ix_(free, free)], residual.T).T
        count('adjoint_solves')
        gp = lcc.param_grad(model, p, lcc.assemble(model, M, flat, y - adjoint))
        gm = lcc.param_grad(model, p, lcc.assemble(model, M, flat, y + adjoint))
        for k in g:
            g[k] += (gp[k] - gm[k]) / 2 / 3
        loss += .5 * np.mean(np.sum(residual[:, hid] ** 2, axis=1)) / 3
    return g, loss


def masked_loss(model, p, flat, target):
    q, loss = lcc.build(model, p), 0.
    for m in range(3):
        M = model.module_mask(m)
        y = lcc.exact_free(q, M, flat)
        loss += .5 * np.mean(np.sum((y[:, M[3]] - target[:, M[4]]) ** 2, axis=1)) / 3
    return float(loss)


def clamped_energy(model, p, flat):
    q, M = lcc.build(model, p), model.mask([])
    s = lcc.assemble(model, M, flat, lcc.exact_free(q, M, flat))
    return float(np.mean(np.einsum('bi,ij,bj->b', s, q, s)) / 2)


def signal_target(data, d):
    return cea.noiseless(data, d['maps']).reshape(len(data), 24)


def objectives(model, p, data, d):
    flat = data.reshape(len(data), 24)
    return {'clamped_energy': clamped_energy(model, p, flat), 'masked_observed': masked_loss(model, p, flat, flat),
            'masked_signal': masked_loss(model, p, flat, signal_target(data, d))}


def shrinkage(model, p, x):
    """Regression of each hidden module's masked inner code on its fully observed inner code (exact)."""
    q, flat = lcc.build(model, p), x.reshape(len(x), 24)
    full = lcc.assemble(model, model.mask([]), flat, lcc.exact_free(q, model.mask([]), flat))
    codes = []
    for h in range(3):
        M = model.module_mask(h)
        masked = lcc.assemble(model, M, flat, lcc.exact_free(q, M, flat))
        codes.append((masked[:, model.sl(h, 2)], full[:, model.sl(h, 2)]))
    return shrinkage_stats(codes)


def shrinkage_stats(codes):
    num = sum(float(np.sum(cm * cf)) for cm, cf in codes)
    den = sum(float(np.sum(cf ** 2)) for _, cf in codes)
    rho = num / den
    residual = sum(float(np.sum((cm - rho * cf) ** 2)) for cm, cf in codes) / den
    return {'rho': rho, 'residual_fraction': residual}


def inside_bounds(p):
    return bool(all(np.max(abs(v)) <= (3. if k.startswith('T') else 2.) for k, v in p.items()))


def evaluate(model, p, d, train_data):
    out = cn.evaluate(model, p, d, train_data)
    out['gain_split'] = None if d['s'] == 0 else {inf: cnp.split_gain(cn.completion_maps(model, p, inf), d)
                                                  for inf in ('exact', '32_step')}
    out['shrinkage'] = shrinkage(model, p, d['test'])
    out['objectives'] = objectives(model, p, np.concatenate(d['waking']), d)
    out['inside_clip_bounds'] = inside_bounds(p)
    return out


def curve_point(model, p, d, train_data, target):
    point = cn.curve_point(model, p, d, train_data)
    if target is not None:
        point['training_objective'] = masked_loss(model, p, train_data.reshape(len(train_data), 24), target)
    return point


def run(model, p, stages, eta, cycles, objective, d=None, clip=True, decay=True, settle_check=False,
        checkpoint=CHECKPOINT, label=None):
    """Adam on one objective. stages: list of (waking or None, training data). objective: energy|masked|signal."""
    keys = list(model.keys)
    theta = np.concatenate([p[k].ravel() for k in keys])
    initial = theta.copy()
    bound = np.concatenate([np.full(p[k].size, 3. if k.startswith('T') else 2.) for k in keys]) if clip else None
    wd = np.concatenate([np.full(p[k].size, lcc.DECAY if k in ('W', 'V') and decay else 0.) for k in keys])
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
    per_cycle = {g: [] for g in list(groups) + ['all']}
    settle_series = {'distance': [], 'gradient_discrepancy': []}
    work = {'example_steps': 0, 'weighted_operations_estimate': 0}
    evaluations, curves, cpu_curve = [], [], []
    cumulative, train_cpu, train_wall, diverged, last_loss = 0, 0., 0., False, None
    if d is not None:
        evaluations.append(evaluate(model, current(), d, stages[0][1]))
    for index, (waking, data) in enumerate(stages):
        if waking is not None:
            frozen = lcc.digest(current())
            lcc.settle(lcc.build(model, current()), model.mask([]), waking.reshape(len(waking), 24))
            count('waking_episode_exposures', len(waking))
            assert lcc.digest(current()) == frozen
        flat = data.reshape(len(data), 24)
        target = flat if objective == 'masked' else signal_target(data, d) if objective == 'signal' else None
        for cycle in range(cycles[index]):
            c0, w0 = time.process_time(), time.perf_counter()
            params = current()
            q = lcc.build(model, params)
            if objective == 'energy':
                g = lcc.energy_grad(model, params, q, flat, work=work)
            else:
                g, last_loss = masked_grad(model, params, q, flat, target)
            count('training_trace_exposures', len(flat))
            gradient = np.concatenate([g[k].ravel() for k in keys]) + wd * theta
            step += 1
            first = lcc.ADAM[0] * first + (1 - lcc.ADAM[0]) * gradient
            second = lcc.ADAM[1] * second + (1 - lcc.ADAM[1]) * gradient ** 2
            update = eta * (first / (1 - lcc.ADAM[0] ** step)) / (
                np.sqrt(second / (1 - lcc.ADAM[1] ** step)) + lcc.ADAM[2])
            proposed = theta - update
            new = np.clip(proposed, -bound, bound) if clip else proposed
            for name, idx in groups.items():
                clips[name] += int(np.sum(new[idx] != proposed[idx]))
            delta = new - theta
            count('adam_steps')
            train_cpu += time.process_time() - c0
            train_wall += time.perf_counter() - w0
            if settle_check:  # measurement only; does not change the update
                M = model.mask([])
                y, _ = lcc.settle(q, M, flat)
                ys = lcc.exact_free(q, M, flat)
                gs = lcc.param_grad(model, params, lcc.assemble(model, M, flat, ys))
                a = np.concatenate([g[k].ravel() for k in keys])
                b = np.concatenate([gs[k].ravel() for k in keys])
                settle_series['distance'].append(float(np.max(np.linalg.norm(y - ys, axis=1))))
                settle_series['gradient_discrepancy'].append(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), GRAD_FLOOR)))
                count('settle_check_exact_solves')
            theta = new
            norms = {name: float(np.linalg.norm(delta[idx])) for name, idx in groups.items()}
            norms['all'] = float(np.linalg.norm(delta))
            for name, value in norms.items():
                per_cycle[name].append(value)
            cumulative += 1
            if not np.all(np.isfinite(theta)) or (last_loss is not None and not np.isfinite(last_loss)):
                diverged = True
                break
            if d is not None and (cycle + 1) % checkpoint == 0:
                point = curve_point(model, current(), d, data, target)
                point.update(stage=index, stage_cycle=cycle + 1, cumulative_cycles=cumulative,
                             displacement_from_init=float(np.linalg.norm(theta - initial)))
                curves.append(point)
                cpu_curve.append([cumulative, train_cpu, train_wall])
        if diverged:
            break
        if d is not None:
            evaluations.append(evaluate(model, current(), d, data))
    final = current()
    if label is not None:
        CPU[label] = {'training_cpu_seconds': train_cpu, 'training_wall_seconds': train_wall,
                      'columns': ['cumulative_cycles', 'training_cpu_seconds', 'training_wall_seconds'],
                      'rows': cpu_curve}
    record = {'diverged': diverged, 'evaluations': evaluations, 'curves': curves, 'clip_activations': clips,
              'cycles': cumulative, 'work': work, 'clip': clip, 'decay': decay,
              'displacement_from_init': float(np.linalg.norm(theta - initial)),
              'final_parameters': {k: v.tolist() for k, v in final.items()}}
    if settle_check:
        for k, series in settle_series.items():
            record[f'settle_check_max_{k}'] = max(series) if series else None
            record[f'settle_check_argmax_{k}'] = int(np.argmax(series)) if series else None
    return final, record, {'movement': per_cycle, 'settle_check': settle_series if settle_check else None}


def learner_stages(waking, subset=None):
    return cea.stages_for(waking, subset)


def fit(arm, model, starts, d, eta, cycles, label_prefix=None):
    """Capacity fit from several starts; the lower final S objective on training traces is the fit."""
    data = np.concatenate(d['waking'])
    rows, movement = {}, {}
    for name, start in starts.items():
        _, record, mv = run(model, {k: v.copy() for k, v in start.items()}, [(None, data)], eta, [cycles], 'signal',
                            d, clip=arm == 'S', decay=arm == 'S', checkpoint=FIT_CHECKPOINT,
                            label=None if label_prefix is None else f'{label_prefix}_{name}')
        final = {k: np.array(v) for k, v in record['final_parameters'].items()}
        objective = None if record['diverged'] else masked_loss(model, final, data.reshape(len(data), 24),
                                                                signal_target(data, d))
        record['final_training_objective'] = objective
        record['disqualified'] = objective is None or not np.isfinite(objective)
        rows[name], movement[name] = record, mv
    eligible = [n for n in rows if not rows[n]['disqualified']]
    chosen = min(eligible, key=lambda n: (rows[n]['final_training_objective'], n)) if eligible else None
    return {'starts': rows, 'chosen_start': chosen, 'failed': chosen is None}, movement


def no_intercept_fit(train_set, rank=None):
    fits = []
    for m in range(3):
        other = [i for i in range(3) if i != m]
        X = train_set[:, other].reshape(len(train_set), 16)
        B = np.linalg.solve(X.T @ X + .1 * np.eye(16), X.T @ train_set[:, m])
        if rank is not None:
            V2 = np.linalg.svd(X @ B)[2][:rank].T
            B = B @ V2 @ V2.T
        fits.append(B)
        count('comparator_fits')
    return fits


def no_intercept_predict(fits, x):
    return np.stack([x[:, [i for i in range(3) if i != m]].reshape(len(x), 16) @ B for m, B in enumerate(fits)])


def comparators(d):
    rows = {k: [] for k in ('ridge_no_intercept', 'uncentered_rank_2', 'ridge_intercept', 'centered_rank_2_recentred')}
    for stage in (0, 1):
        tr = np.concatenate(d['waking'][:stage + 1])
        for name in rows:
            if name in ('ridge_no_intercept', 'uncentered_rank_2'):
                f = no_intercept_fit(tr, 2 if name == 'uncentered_rank_2' else None)
                rows[name].append(cn.score(lambda x: no_intercept_predict(f, x), [B.T for B in f], d))
            else:
                f = cn.linear_fit(tr) if name == 'ridge_intercept' else cnp.recentred(tr)
                rows[name].append(cn.score(lambda x: cn.linear_predict(f, x), [w[:16].T for w in f], d))
    return rows


def finite_difference_masked(rng):
    model = model_for('E', 10.)
    p = lcc.initialize(model, 99)
    for k in p:
        p[k] = p[k] + rng.normal(0, .1, p[k].shape)
    d = cn.fixture(99)['weak']
    data = np.concatenate(d['waking'])[:6]
    flat = data.reshape(len(data), 24)
    worst = 0.
    for target in (flat, signal_target(data, d)):
        g, _ = masked_grad(model, p, lcc.build(model, p), flat, target)
        for k in model.keys:
            for idx in np.ndindex(p[k].shape):
                plus = {a: v.copy() for a, v in p.items()}
                minus = {a: v.copy() for a, v in p.items()}
                plus[k][idx] += 1e-6
                minus[k][idx] -= 1e-6
                numeric = (masked_loss(model, plus, flat, target) - masked_loss(model, minus, flat, target)) / 2e-6
                worst = max(worst, abs(numeric - g[k][idx]))
    return worst


def verify():
    out = {'carried_over_from_correlated_nuisance': cn.verify()}
    rng = np.random.default_rng(21)
    out['masked_gradient_finite_difference'] = finite_difference_masked(rng)
    assert out['masked_gradient_finite_difference'] < 1e-6
    # With observed targets the masked gradient is lcc.exact_grad, bit for bit.
    model = model_for('E', 2.)
    p = lcc.initialize(model, 99)
    d = cn.fixture(99)['weak']
    flat = np.concatenate(d['waking']).reshape(32, 24)
    q = lcc.build(model, p)
    g1, l1 = masked_grad(model, p, q, flat, flat)
    g2, l2 = lcc.exact_grad(model, p, q, flat)
    out['masked_grad_equals_exact_grad'] = bool(l1 == l2 and all(np.array_equal(g1[k], g2[k]) for k in g1))
    assert out['masked_grad_equals_exact_grad']
    # S clips and decays; S' does neither.
    start = lcc.initialize(model, 99)
    start['W'] = np.full_like(start['W'], 2.5)
    fs, _, _ = run(model, {k: v.copy() for k, v in start.items()}, [(None, np.concatenate(d['waking']))], 1e-4, [1],
                   'signal', d, clip=True, decay=True)
    fp, _, _ = run(model, {k: v.copy() for k, v in start.items()}, [(None, np.concatenate(d['waking']))], 1e-4, [1],
                   'signal', d, clip=False, decay=False)
    out['S_clips'], out['S_prime_unclipped'] = float(np.max(abs(fs['W']))), float(np.max(abs(fp['W'])))
    assert out['S_clips'] <= 2 and out['S_prime_unclipped'] > 2
    # Uncentered comparator: rank <= 2, equals no-intercept ridge on exact rank-2 linear targets.
    tr = np.concatenate(d['waking'])
    out['uncentered_rank_2_slope_ranks'] = [int(np.linalg.matrix_rank(B, tol=1e-10)) for B in no_intercept_fit(tr, 2)]
    assert all(r <= 2 for r in out['uncentered_rank_2_slope_ranks'])
    X = rng.normal(size=(32, 2, 8))
    synthetic = np.concatenate([(X.reshape(32, 16) @ rng.normal(size=(16, 2)) @ rng.normal(size=(2, 8)))[:, None], X],
                               axis=1)
    out['uncentered_equals_ridge_on_rank2'] = float(np.max(abs(no_intercept_fit(synthetic)[0]
                                                               - no_intercept_fit(synthetic, 2)[0])))
    assert out['uncentered_equals_ridge_on_rank2'] < 1e-10
    # Shrinkage statistics on equal codes.
    c = rng.normal(size=(5, 2))
    out['shrinkage_equal_codes'] = shrinkage_stats([(c, c), (2 * c, 2 * c)])
    assert abs(out['shrinkage_equal_codes']['rho'] - 1) < 1e-12 and out['shrinkage_equal_codes']['residual_fraction'] < 1e-24
    # Reproduction: seed 23 weak / inner / 10, arm E at eta 0.1, reproduces correlated nuisance v1.
    old = json.loads(gzip.decompress(CN_RESULTS.read_bytes()))['seeds']['23']['conditions']['weak_inner_10']
    d = cn.fixture(23)['weak']
    model = model_for('E', 10.)
    _, record, _ = run(model, lcc.initialize(model, 23), learner_stages(d['waking']), .1, [300, 300], 'energy', d)
    now, then = record['evaluations'][-1]['nuisance']['exact'], old['evaluations'][-1]['nuisance']['exact']
    out['reproduction_correlated_nuisance_seed23_weak_inner_10'] = {
        w: abs(now[w]['combined']['avoidable_mse'] - then[w]['combined']['avoidable_mse']) for w in cn.SETS}
    assert max(out['reproduction_correlated_nuisance_seed23_weak_inner_10'].values()) < 1e-12
    return out


def settings():
    return [(r, bn, b) for r in REGIMES for bn, b in BUDGETS]


def pilot(datasets):
    rows, selected = {}, {}
    for regime, bname, b in settings():
        for arm in ARMS:
            name = f'{regime}_{arm}_{bname}'
            rows[name] = []
            for eta in ETA_GRID:
                scores, bad = [], False
                for seed in PILOT_SEEDS:
                    d = datasets[seed][regime]
                    model = model_for(arm, b)
                    p = lcc.initialize(model, seed)
                    if arm in LEARNERS:
                        final, record, _ = run(model, p, learner_stages(d['waking'], 0), eta, [C, C], ARMS[arm][1])
                        odd = np.concatenate([d['waking'][0][1::2], d['waking'][1][1::2]]).reshape(-1, 24)
                        s = None if record['diverged'] else cn.masked_mse_exact(model, final, odd)
                    else:
                        data = np.concatenate(d['waking'])
                        final, record, _ = run(model, p, [(None, data)], eta, [CS], 'signal', d,
                                               clip=arm == 'S', decay=arm == 'S', checkpoint=10 ** 9)
                        s = None if record['diverged'] else masked_loss(model, final, data.reshape(32, 24),
                                                                        signal_target(data, d))
                    bad |= s is None or not np.isfinite(s)
                    scores.append(s)
                rows[name].append({'eta': eta, 'scores_by_seed': scores, 'disqualified': bad,
                                   'mean_score': None if bad else float(np.mean(scores))})
            best = min((r for r in rows[name] if not r['disqualified']), key=lambda r: (r['mean_score'], r['eta']))
            selected[name] = {'eta': best['eta'], 'at_grid_edge': best['eta'] in (ETA_GRID[0], ETA_GRID[-1])}
    return {'grids': rows, 'selected': selected,
            'score': {'learners': 'exact-equilibrium observed masked MSE on odd-index waking traces after stage B; '
                                  'training on even-index traces',
                      'fits': 'final S objective on all 32 pilot waking traces, from the shared initialization'}}


def final_eval(record, arm):
    if arm in FITS:
        return record['starts'][record['chosen_start']]['evaluations'][-1] if not record['failed'] else None
    return record['evaluations'][-1]


def criteria(seeds):
    order = sorted(seeds, key=int)
    get = lambda s, name: final_eval(seeds[s]['arms'][name], name.split('_')[1] if '_prime' not in name else 'S_prime')
    nuis = lambda s, name: get(s, name)['nuisance']['exact']
    Xdec = lambda name: [nuis(s, name)['decorrelated']['combined']['avoidable_mse'] for s in order]
    Xin = lambda name: [nuis(s, name)['in_distribution']['combined']['avoidable_mse'] for s in order]
    gam = lambda name: [nuis(s, name)['gain']['gamma'] for s in order]
    out = {k: {} for k in ('N1', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'summary')}
    for regime, bname, _ in settings():
        for arm in LEARNERS:
            name = f'{regime}_{arm}_{bname}'
            ev = [[e['nuisance']['exact']['in_distribution'] for e in seeds[s]['arms'][name]['evaluations']] for s in order]
            b_falls = sum(e[2]['B']['avoidable_mse'] < e[1]['B']['avoidable_mse'] for e in ev)
            a_ratio = [(e[2]['A']['avoidable_mse'] + cea.DELTA) / (e[1]['A']['avoidable_mse'] + cea.DELTA) for e in ev]
            fam = ARMS[arm][0]
            frozen = [seeds[s]['frozen'][f'{regime}_{fam}_{bname}']['nuisance']['exact']['in_distribution']
                      ['combined']['avoidable_mse'] for s in order]
            to_frozen = (np.mean(Xin(name)) + cea.DELTA) / (np.mean(frozen) + cea.DELTA)
            a_ok = sum(r <= 1.5 for r in a_ratio)
            out['N1'][name] = {'B_falls_seeds': int(b_falls), 'A_retained_seeds': int(a_ok),
                               'A_ratio_by_seed': [float(r) for r in a_ratio], 'ratio_to_own_frozen': float(to_frozen),
                               'learns': bool(b_falls >= SEED_COUNT and a_ok >= SEED_COUNT and to_frozen <= .1)}
        for arm in ARMS:
            name = f'{regime}_{arm}_{bname}'
            g = gam(name)
            out['summary'][name] = {
                'mean_X_in': float(np.mean(Xin(name))), 'mean_X_dec': float(np.mean(Xdec(name))),
                'mean_gamma': None if g[0] is None else float(np.mean(g)),
                'category': cn.category(g) if regime == 'weak' else 'undefined',
                'mean_signal_error_dec': float(np.mean([nuis(s, name)['decorrelated']['combined']['signal_error_mse'] for s in order])),
                'mean_rho': float(np.mean([get(s, name)['shrinkage']['rho'] for s in order])),
                'objectives': {k: float(np.mean([get(s, name)['objectives'][k] for s in order]))
                               for k in ('clamped_energy', 'masked_observed', 'masked_signal')}}
            if arm in FITS:
                out['summary'][name]['chosen_starts'] = [seeds[s]['arms'][name]['chosen_start'] for s in order]
        refs = {}
        for comp in ('ridge_no_intercept', 'uncentered_rank_2', 'ridge_intercept', 'centered_rank_2_recentred'):
            rows = [seeds[s]['comparators'][regime][comp][-1] for s in order]
            gs = [r['gain']['gamma'] for r in rows]
            refs[comp] = {'X_dec': [r['decorrelated']['combined']['avoidable_mse'] for r in rows],
                          'mean_X_in': float(np.mean([r['in_distribution']['combined']['avoidable_mse'] for r in rows])),
                          'mean_gamma': None if gs[0] is None else float(np.mean(gs)),
                          'category': cn.category(gs) if regime == 'weak' else 'undefined'}
            refs[comp]['mean_X_dec'] = float(np.mean(refs[comp]['X_dec']))
        out['summary'][f'{regime}_comparators_{bname}'] = refs
        if regime != 'weak':
            continue
        n = lambda arm: f'weak_{arm}_{bname}'
        cat = lambda arm: out['summary'][n(arm)]['category']
        learns = {a: out['N1'][n(a)]['learns'] for a in LEARNERS}
        out['D0'][bname] = {'E_category': cat('E'), 'U_category': cat('U'),
                            'X_dec_E_vs_U': cea.compare(Xdec(n('E')), Xdec(n('U'))), 'learns': learns}
        d1a = cea.compare(Xdec(n('E')), Xdec(n('S')))
        d1a['conclusion'] = 'parameters_can_do_better_than_learner' if d1a['status'] == 'first_clearly_worse' else 'no_improvement_found'
        d1b = cea.compare(Xdec(n('U')), Xdec(n('S')))
        if cat('S') != 'filters':
            d1b['conclusion'] = 'no_conclusion_S_does_not_filter'
        else:
            d1b['conclusion'] = ('inner_architecture_can_beat_surface_coupling' if d1b['status'] == 'first_clearly_worse'
                                 else 'no_such_solution_found')
        failed = {a: sum(seeds[s]['arms'][n(a)]['failed'] for s in order) for a in FITS}
        out['D1'][bname] = {'D1a_E_vs_S': d1a, 'D1b_U_vs_S': d1b, 'S_category': cat('S'), 'failed_fits': failed,
                            'learns': learns}
        d2 = cea.compare(Xdec(n('S')), Xdec(n('S_prime')))
        inside = sum(get(s, n('S_prime'))['inside_clip_bounds'] for s in order)
        if d2['status'] == 'first_clearly_worse':
            d2['conclusion'] = 'removing_clips_and_decay_helps_fit'
            d2['feasible_witness'] = bool(inside >= SEED_COUNT)
        else:
            d2['conclusion'], d2['feasible_witness'] = 'no_improvement_found', False
        d2['S_prime_inside_bounds_seeds'] = int(inside)
        out['D2'][bname] = d2
        d3 = cea.compare(Xdec(n('E')), Xdec(n('M')))
        d3['conclusion'] = {'first_clearly_worse': 'masked_completion_helps_under_this_protocol',
                            'second_clearly_worse': 'masked_completion_does_worse_under_this_protocol'}.get(
            d3['status'], 'no_difference_established')
        d3['M_category'], d3['learns'] = cat('M'), learns
        out['D3'][bname] = d3
        u = refs['uncentered_rank_2']
        d5 = cea.compare(Xdec(n('U')), u['X_dec'])
        d5['conclusion'] = ('achieves_both' if u['category'] == 'filters' and d5['status'] == 'first_clearly_worse'
                            else 'not_both')
        d5.update(uncentered_category=u['category'], uncentered_mean_gamma=u['mean_gamma'],
                  uncentered_mean_X_dec=u['mean_X_dec'], centered=refs['centered_rank_2_recentred'])
        out['D5'][bname] = d5
        rho = {a: [get(s, n(a))['shrinkage']['rho'] for s in order] for a in ('E', 'M', 'S', 'S_prime')}
        e_low, s_low = sum(r < .9 for r in rho['E']), sum(r < .9 for r in rho['S'])
        out['D6'][bname] = {'rho_by_seed': rho, 'E_below_0.9_seeds': int(e_low), 'S_below_0.9_seeds': int(s_low),
                            'shrinkage_hypothesis_consistent': bool(e_low >= SEED_COUNT and s_low < SEED_COUNT)}
        positives = [k for k, ok in (('D1a', d1a['conclusion'] != 'no_improvement_found'),
                                     ('D1b', d1b['conclusion'] == 'inner_architecture_can_beat_surface_coupling'),
                                     ('D2', d2['conclusion'] != 'no_improvement_found'),
                                     ('D3', d3['conclusion'] == 'masked_completion_helps_under_this_protocol'),
                                     ('D5', d5['conclusion'] == 'achieves_both')) if ok]
        out['D4'][bname] = {'constructive_improvements': positives,
                            'conclusion': 'cause_unresolved' if not positives else 'improvements_found'}
    dist = [seeds[s]['arms'][f'{r}_{a}_{bn}']['settle_check_max_distance'] for s in order for r, bn, _ in settings()
            for a in ('E', 'U')]
    grad = [seeds[s]['arms'][f'{r}_{a}_{bn}']['settle_check_max_gradient_discrepancy'] for s in order
            for r, bn, _ in settings() for a in ('E', 'U')]
    out['D7'] = {'max_distance': max(dist), 'max_gradient_discrepancy': max(grad),
                 'exact_for_practical_purposes': bool(max(dist) <= SETTLE_TOL and max(grad) <= GRAD_TOL),
                 'scope': 'every update of every confirmatory E and U run'}
    return out


def main():
    paths = [Path(str(PREFIX) + s) for s in ('-results.json', '-runtime.json', '-movement.json')]
    if any(f.exists() or Path(str(f) + '.gz').exists() for f in paths):
        raise RuntimeError('Refusing overwrite')
    result, runtime, movement_path = paths
    wall, cpu = time.perf_counter(), time.process_time()
    with phase('verification'):
        checks = verify()
    with phase('fixture_construction'):
        datasets = {seed: {r: cn.fixture(seed)[r] for r in REGIMES} for seed in PILOT_SEEDS + SEEDS}
    with phase('selection_pilot'):
        selection = pilot(datasets)
    science = {'checks': checks, 'selection': selection, 'cycles_per_stage': C, 'fit_cycles': CS, 'seeds': {},
               'scope': 'E, U, M: block-local or adjoint learning on observed data; S, S_prime: evaluator-assisted '
                        'capacity fits to mu_sig, not learners'}
    movement = {}
    for seed in SEEDS:
        row = {'frozen': {}, 'arms': {}, 'comparators': {}}
        movement[str(seed)] = {}
        initial = lcc.digest(base.initialize(seed))
        for regime in REGIMES:
            d = datasets[seed][regime]
            with phase(f'seed{seed}_{regime}_frozen_and_comparators'):
                for fam in ('inner', 'uniform'):
                    for bname, b in BUDGETS:
                        model = cn.model_for(fam, b)
                        row['frozen'][f'{regime}_{fam}_{bname}'] = evaluate(model, lcc.initialize(model, seed), d,
                                                                            d['waking'][0])
                row['comparators'][regime] = comparators(d)
            for bname, b in BUDGETS:
                for arm in ARMS:
                    name = f'{regime}_{arm}_{bname}'
                    model = model_for(arm, b)
                    p = lcc.initialize(model, seed)
                    assert lcc.digest({k: p[k] for k in ('W', 'V')}) == initial
                    eta = selection['selected'][name]['eta']
                    with phase(f'seed{seed}_{name}'):
                        if arm in LEARNERS:
                            _, record, mv = run(model, p, learner_stages(d['waking']), eta, [C, C], ARMS[arm][1], d,
                                                settle_check=ARMS[arm][1] == 'energy', label=f'seed{seed}_{name}')
                        else:
                            e_final = {k: np.array(v) for k, v in row['arms'][f'{regime}_E_{bname}']['final_parameters'].items()}
                            record, mv = fit(arm, model, {'shared_init': p, 'E_final': e_final}, d, eta, CS,
                                             label_prefix=f'seed{seed}_{name}')
                    record.update(arm=arm, regime=regime, budget=b, family=ARMS[arm][0], eta=eta)
                    row['arms'][name] = record
                    movement[str(seed)][name] = mv
        science['seeds'][str(seed)] = row
    with phase('criteria'):
        science['criteria'] = criteria(science['seeds'])
    with phase('serialization'):
        movement_bytes = (json.dumps(movement, sort_keys=True, separators=(',', ':')) + '\n').encode()
        science['movement_file'] = {'path': movement_path.name + '.gz', 'compression': 'gzip -n -9',
                                    'uncompressed_sha256': hashlib.sha256(movement_bytes).hexdigest()}
        result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
        movement_path.write_bytes(movement_bytes)
    with phase('provenance'):
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in (Path(__file__), Path(cn.__file__), Path(cnp.__file__), Path(lcc.__file__),
                            Path(base.__file__), Path(cea.__file__), CONTRACT, CN_RESULTS)}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)

    def limit(name):
        p = Path('/sys/fs/cgroup') / name
        return p.read_text().strip() if p.exists() else None
    counts = dict(COUNTS)
    for prefix, mod in (('cn', cn), ('ablation', cea), ('lcc', lcc), ('v1_module', base)):
        counts.update({f'{prefix}_{k}': v for k, v in mod.COUNTS.items()})
    storage = {'parameters_inner': 128 * 8, 'parameters_uniform': 288 * 8, 'adam': '16P + 8',
               'training_traces': 32 * 24 * 8, 'dense_q': 42 * 42 * 8, 'masked_free_block': 26 * 26 * 8,
               'equilibrium_and_adjoint_states': 2 * 32 * 26 * 8, 'clamped_settle_workspace': (18 * 18 + 32 * 18) * 8,
               'second_start': '24P + 8'}
    info = {'phases': PHASES, 'counts': counts, 'training_cpu': CPU, 'hashes': hashes, 'revision': revision,
            'dirty_status': dirty, 'command': ' '.join(sys.argv), 'pilot_seeds': list(PILOT_SEEDS),
            'confirmatory_seeds': list(SEEDS), 'cycles_per_stage': C, 'fit_cycles': CS,
            'seed_policy': 'v1 SeedSequence [seed,1001] initialization; [seed,2002] v1 fixture; [seed,3003] shared '
                           'nuisance',
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
    print(json.dumps({k: science['criteria'][k] for k in ('D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D7')}, indent=1))


if __name__ == '__main__':
    main()
