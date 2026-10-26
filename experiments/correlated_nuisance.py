"""Correlated nuisance v1: which coupling topologies transmit non-persistent surface correlations.

Implements docs/correlated-nuisance-v1-contract.md. Learners see only observed
outer activities and episode IDs; structure labels, pattern maps, shared-nuisance
directions, the decorrelated test set and all target means are evaluator only.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import contextlib
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
from scipy.linalg import block_diag
from experiments import local_consolidation as base
from experiments import local_contrastive_consolidation as lcc
from experiments import clamped_energy_ablation as cea
from experiments.local_contrastive_floor_analysis import ridge_predictions

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/correlated-nuisance-v1-contract.md'
V1_RESULTS = ROOT / 'reports/local-contrastive-consolidation-v1-results.json'
PREFIX = Path(os.environ.get('CN_PREFIX', str(ROOT / 'reports/correlated-nuisance-v1')))
C = int(os.environ.get('CN_CYCLES', '300'))  # smoke runs override; the contract fixes 300
ETA_GRID = (.001, .003, .01, .03, .1)
PILOT_SEEDS = (0, 1, 2)
SEEDS = tuple(int(s) for s in os.environ.get('CN_SEEDS', '23,24,25,26,27,28,29,30,31,32').split(','))
REGIMES = {'independent': (1., 0.), 'weak': (np.sqrt(.1), np.sqrt(.027)), 'strong': (np.sqrt(.1), 1.)}
FAMILIES = ('uniform', 'inner', 'outer', 'outer_matched')
BUDGETS = (('2', 2.), ('10', 10.))
TRANSMITS, FILTERS, GAMMA_DELTA = .5, .2, .01
SEED_COUNT = cea.SEED_COUNT
CHECKPOINT = 50
ORACLE_TABLE = {'weak': (.0066, .0034), 'strong': (.250, .125)}
SETS = ('in_distribution', 'decorrelated')
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


def kappa(family, b):
    return {'uniform': (b / 14, b / 14, b / 14), 'inner': (0., 0., b / 2), 'outer': (b / 8, 0., 0.),
            'outer_matched': (b / 14, 0., 0.)}[family]


def model_for(family, b):
    return lcc.Model(family, kappa(family, b))  # every family here is hierarchical (8 -> 4 -> 2)


def conditions():
    return [(f'{r}_{f}_{bn}', r, f, bn, b) for r in REGIMES for f in FAMILIES for bn, b in BUDGETS]


def complement(M):
    return np.linalg.svd(M, full_matrices=True)[0][:, 2:]


def shared_directions(seed, maps):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3003]))
    g = rng.normal(size=(3, 8, 2))
    U = np.stack([np.linalg.qr(g[m] - maps[m] @ (maps[m].T @ g[m]))[0] for m in range(3)])
    return rng, U


def fixture(seed):
    """All regimes of one seed. v1's fixture supplies amplitudes, labels and e; [seed,3003] the shared latents."""
    waking, testing, references = base.fixture(seed)
    maps = lcc.private_maps(seed)
    rng, U = shared_directions(seed, maps)
    z_waking = [rng.normal(size=(16, 2)), rng.normal(size=(16, 2))]
    z_test = np.concatenate([rng.normal(size=(12, 2)), rng.normal(size=(12, 2))])
    z_reference = np.concatenate([rng.normal(size=(8, 2)), rng.normal(size=(8, 2))])
    z_dec = rng.normal(size=(24, 3, 2))
    shared = lambda z: np.einsum('mik,nk->nmi', U, z)
    out = {}
    for regime, (c, s) in REGIMES.items():
        d = {'maps': maps, 'U': U, 'c': c, 's': s}
        if s == 0:
            d.update(waking=waking, test=testing, decorrelated=testing, reference=references)
        else:
            def scaled(data):
                signal = cea.noiseless(data, maps)
                return signal + c * (data - signal)
            base_test = scaled(testing)
            d.update(waking=[scaled(w) + s * shared(z) for w, z in zip(waking, z_waking)],
                     test=base_test + s * shared(z_test),
                     decorrelated=base_test + s * np.einsum('mik,nmk->nmi', U, z_dec),
                     reference=scaled(references) + s * shared(z_reference))
            d['base_test'] = base_test
        out[regime] = d
    return out


def targets(x, d):
    """mu_sig and mu_full, (n,3,8); mu_full[:, h] conditions on the other two modules (evaluator only)."""
    sig = cea.noiseless(x, d['maps'])
    full = sig.copy()
    c, s = d['c'], d['s']
    if s > 0:
        proj = np.einsum('mik,nmi->nmk', d['U'], x)
        k = s / (.01 * c * c + 2 * s * s)
        for h in range(3):
            zhat = k * sum(proj[:, o] for o in range(3) if o != h)
            full[:, h] += s * zhat @ d['U'][h].T
    return sig, full


def predictions(model, p, x, inference):
    q, flat, out = lcc.build(model, p), x.reshape(len(x), 24), []
    for h in range(3):
        M = model.module_mask(h)
        y = lcc.exact_free(q, M, flat) if inference == 'exact' else lcc.settle(q, M, flat)[0]
        out.append(y[:, M[3]])
    return np.stack(out)


def completion_maps(model, p, inference):
    """A_h (8x16): masked completion is linear in the observed outer activities (no bias terms)."""
    q, out = lcc.build(model, p), []
    for h in range(3):
        M = model.module_mask(h)
        basis = np.zeros((16, 24))
        basis[np.arange(16), M[1]] = 1
        y = lcc.exact_free(q, M, basis) if inference == 'exact' else lcc.settle(q, M, basis)[0]
        out.append(y[:, M[3]].T)
    return out


def oracle_scale(d):
    return d['s'] ** 2 / (.01 * d['c'] ** 2 + 2 * d['s'] ** 2)


def gain(A, d):
    """Nuisance gain gamma and companion norms of the completion maps restricted to observed nuisance."""
    U, maps = d['U'], d['maps']
    shared = comp = leak = 0.
    for h in range(3):
        o = [m for m in range(3) if m != h]
        G = A[h] @ block_diag(U[o[0]], U[o[1]])
        shared += float(np.sum(G ** 2))
        comp += float(np.sum((A[h] @ block_diag(complement(maps[o[0]]), complement(maps[o[1]]))) ** 2))
        leak += float(np.sum((maps[h] @ maps[h].T @ G) ** 2))
    star = 12 * oracle_scale(d) ** 2  # sum over h of ||k U_h [I I]||_F^2
    return {'gamma': float(np.sqrt(shared / star)) if d['s'] > 0 else None,
            'shared_gain_norm': float(np.sqrt(shared)), 'complement_gain_norm': float(np.sqrt(comp)),
            'pattern_span_leak_norm': float(np.sqrt(leak))}


def set_metrics(pred, x, mean, sig, maps, splits):
    """pred (3,n,8); x, mean, sig (n,3,8). Per coordinate, averaged over hidden modules."""
    t, mu, sg = (np.transpose(a, (1, 0, 2)) for a in (x, mean, sig))
    P = np.einsum('mij,mkj->mik', maps, maps)
    rows = np.stack([np.mean((pred - t) ** 2, axis=(0, 2)), np.mean((t - mu) ** 2, axis=(0, 2)),
                     np.mean((pred - mu) ** 2, axis=(0, 2)), -2 * np.mean((pred - mu) * (t - mu), axis=(0, 2)),
                     np.mean(np.einsum('mij,mnj->mni', P, pred - sg) ** 2, axis=(0, 2))])
    out = {}
    for split, idx in splits.items():
        o, f, a, c, e = (float(np.mean(rows[i, idx])) for i in range(5))
        assert abs(o - f - a - c) < 1e-12
        out[split] = {'observed_mse': o, 'floor_mse': f, 'noise_subtracted_mse': o - f, 'avoidable_mse': a,
                      'cross_term': c, 'signal_error_mse': e}
    return out


def tau(pred, x, d):
    sig, full = targets(x, d)
    dev, ref = pred - np.transpose(sig, (1, 0, 2)), np.transpose(full - sig, (1, 0, 2))
    return float(np.sum(dev * ref) / np.sum(ref ** 2))


def score(predict, A, d):
    out = {}
    for name in SETS:
        x = d['test'] if name == 'in_distribution' else d['decorrelated']
        sig, full = targets(x, d)
        pred = predict(x)
        out[name] = set_metrics(pred, x, full if name == 'in_distribution' else sig, sig, d['maps'], cea.TEST_SPLITS)
        if name == 'in_distribution':
            out['tau'] = tau(pred, x, d) if d['s'] > 0 else None
    out['gain'] = gain(A, d)
    count('nuisance_scored_examples', 3 * 48)
    return out


def oracles(d):
    out = {}
    for name in SETS:
        x = d['test'] if name == 'in_distribution' else d['decorrelated']
        sig, full = targets(x, d)
        out[name] = {'signal_oracle': float(np.mean((sig - full) ** 2)) if name == 'in_distribution' else 0.,
                     'full_oracle': 0. if name == 'in_distribution' else float(np.mean((full - sig) ** 2))}
    return out


def outer_gains(p, d):
    out = {}
    for a in (0, 1):
        key = f'T{a}_0'
        if key not in p:
            continue
        M, U = d['maps'][a], d['U'][a]
        R = np.linalg.svd(np.hstack([M, U]), full_matrices=True)[0][:, 4:]
        T = p[key]
        out[key] = {name: np.linalg.svd(T @ B, compute_uv=False).tolist()
                    for name, B in (('pattern_span', M), ('shared_nuisance', U), ('remaining_complement', R))}
    return out


def evaluate(model, p, d, replay):
    out = lcc.evaluate(model, p, d['test'], d['reference'], d['maps'])
    out['nuisance'] = {inf: score(lambda x: predictions(model, p, x, inf), completion_maps(model, p, inf), d)
                       for inf in ('exact', '32_step')}
    sig, full = targets(replay, d)
    splits = cea.train_splits(len(replay))
    out['train'] = {inf: set_metrics(predictions(model, p, replay, inf), replay, full, sig, d['maps'], splits)
                    for inf in ('exact', '32_step')}
    out['generalization_gap'] = {inf: {k: out['nuisance'][inf]['in_distribution']['combined'][k]
                                       - out['train'][inf]['combined'][k] for k in ('observed_mse', 'avoidable_mse')}
                                 for inf in ('exact', '32_step')}
    out['outer_gains'] = outer_gains(p, d)
    return out


def curve_point(model, p, d, replay):
    nuis = score(lambda x: predictions(model, p, x, 'exact'), completion_maps(model, p, 'exact'), d)
    sig, full = targets(replay, d)
    tr = set_metrics(predictions(model, p, replay, 'exact'), replay, full, sig, d['maps'], {'combined': np.arange(len(replay))})
    return {'X_in': nuis['in_distribution']['combined']['avoidable_mse'],
            'X_dec': nuis['decorrelated']['combined']['avoidable_mse'], 'gamma': nuis['gain']['gamma'],
            'train_X_in': tr['combined']['avoidable_mse']}


def train(model, p, stages, eta, cycles, d=None, label=None):
    """Clamped-energy Adam training with free translations (the ablation's mode L)."""
    keys = list(model.keys)
    theta = np.concatenate([p[k].ravel() for k in keys])
    initial = theta.copy()
    bound = np.concatenate([np.full(p[k].size, 3. if k.startswith('T') else 2.) for k in keys])
    decay = np.concatenate([np.full(p[k].size, lcc.DECAY if k in ('W', 'V') else 0.) for k in keys])
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
    path = {g: 0. for g in per_cycle}
    work = {'example_steps': 0, 'weighted_operations_estimate': 0}
    evaluations, curves, cpu_curve = [], [], []
    cumulative, train_cpu, train_wall, diverged = 0, 0., 0., False
    if d is not None:
        evaluations.append(evaluate(model, current(), d, stages[0][1]))
    for index, (waking, replay) in enumerate(stages):
        frozen = lcc.digest(current())
        lcc.settle(lcc.build(model, current()), model.mask([]), waking.reshape(len(waking), 24))
        count('waking_episode_exposures', len(waking))
        assert lcc.digest(current()) == frozen
        flat = replay.reshape(len(replay), 24)
        for cycle in range(cycles[index]):
            c0, w0 = time.process_time(), time.perf_counter()
            params = current()
            q = lcc.build(model, params)
            g = lcc.energy_grad(model, params, q, flat, work=work)
            count('replay_trace_exposures', len(flat))
            gradient = np.concatenate([g[k].ravel() for k in keys]) + decay * theta
            step += 1
            first = lcc.ADAM[0] * first + (1 - lcc.ADAM[0]) * gradient
            second = lcc.ADAM[1] * second + (1 - lcc.ADAM[1]) * gradient ** 2
            update = eta * (first / (1 - lcc.ADAM[0] ** step)) / (
                np.sqrt(second / (1 - lcc.ADAM[1] ** step)) + lcc.ADAM[2])
            proposed = theta - update
            new = np.clip(proposed, -bound, bound)
            for name, idx in groups.items():
                clips[name] += int(np.sum(new[idx] != proposed[idx]))
            delta = new - theta
            theta = new
            count('adam_steps')
            norms = {name: float(np.linalg.norm(delta[idx])) for name, idx in groups.items()}
            norms['all'] = float(np.linalg.norm(delta))
            for name, value in norms.items():
                per_cycle[name].append(value)
                path[name] += value
            cumulative += 1
            train_cpu += time.process_time() - c0
            train_wall += time.perf_counter() - w0
            if not np.all(np.isfinite(theta)):
                diverged = True
                break
            if d is not None and (cycle + 1) % CHECKPOINT == 0:
                point = curve_point(model, current(), d, replay)
                point.update(stage=index, stage_cycle=cycle + 1, cumulative_cycles=cumulative,
                             displacement_from_init=float(np.linalg.norm(theta - initial)))
                curves.append(point)
                cpu_curve.append([cumulative, train_cpu, train_wall])
        if diverged:
            break
        if d is not None:
            evaluations.append(evaluate(model, current(), d, replay))
    final = current()
    if label is not None:
        CPU[label] = {'training_cpu_seconds': train_cpu, 'training_wall_seconds': train_wall,
                      'columns': ['cumulative_cycles', 'training_cpu_seconds', 'training_wall_seconds'],
                      'rows': cpu_curve}
    displacement = {name: float(np.linalg.norm((theta - initial)[idx])) for name, idx in groups.items()}
    displacement['all'] = float(np.linalg.norm(theta - initial))
    record = {'diverged': diverged, 'evaluations': evaluations, 'curves': curves, 'clip_activations': clips,
              'cycles': cumulative, 'work': work, 'displacement_from_init': displacement, 'path_length': path,
              'final_parameters': {k: v.tolist() for k, v in final.items()}}
    return final, record, per_cycle


def linear_fit(train_set, rank=None):
    """v1's direct predictor (ridge 0.1 with intercept); rank=2 gives reduced-rank regression."""
    fits = []
    for m in range(3):
        other = [i for i in range(3) if i != m]
        design = np.column_stack((train_set[:, other].reshape(len(train_set), 16), np.ones(len(train_set))))
        weights = np.linalg.solve(design.T @ design + .1 * np.eye(17), design.T @ train_set[:, m])
        if rank is not None:
            fitted = design @ weights
            V2 = np.linalg.svd(fitted - fitted.mean(axis=0))[2][:rank].T
            weights = np.vstack([weights[:16] @ V2 @ V2.T, weights[16:]])
        fits.append(weights)
        count('comparator_fits')
    return fits


def linear_predict(fits, x):
    out = []
    for m, w in enumerate(fits):
        other = [i for i in range(3) if i != m]
        out.append(np.column_stack((x[:, other].reshape(len(x), 16), np.ones(len(x)))) @ w)
    return np.stack(out)


def comparators(d):
    rows = {'ridge': [], 'reduced_rank_2': []}
    for stage in (0, 1):
        train_set = np.concatenate(d['waking'][:stage + 1])
        for name, rank in (('ridge', None), ('reduced_rank_2', 2)):
            fits = linear_fit(train_set, rank)
            rows[name].append(score(lambda x: linear_predict(fits, x), [w[:16].T for w in fits], d))
    return rows


def masked_mse_exact(model, p, flat):
    """Pilot score: exact-equilibrium whole-module masked squared error per coordinate."""
    q, errors = lcc.build(model, p), []
    for m in range(3):
        M = model.module_mask(m)
        errors.append(np.mean((lcc.exact_free(q, M, flat)[:, M[3]] - flat[:, M[4]]) ** 2))
    return float(np.mean(errors))


def finite_differences(model, rng):
    p = lcc.initialize(model, 99)
    for k in p:
        p[k] = p[k] + rng.normal(0, .1, p[k].shape)
    states = rng.normal(size=(5, model.S))
    g = lcc.param_grad(model, p, states)
    energy = lambda r: float(np.mean(np.einsum('bi,ij,bj->b', states, lcc.build(model, r), states)) / 2)
    worst = 0.
    for k in model.keys:
        for idx in np.ndindex(p[k].shape):
            plus = {a: v.copy() for a, v in p.items()}
            minus = {a: v.copy() for a, v in p.items()}
            plus[k][idx] += 1e-6
            minus[k][idx] -= 1e-6
            worst = max(worst, abs((energy(plus) - energy(minus)) / 2e-6 - g[k][idx]))
    return worst


def monte_carlo(maps, U, n=200000):
    rng = np.random.default_rng(np.random.SeedSequence([0, 4004]))
    out = {}
    for regime in ('weak', 'strong'):
        c, s = REGIMES[regime]
        d = {'maps': maps, 'U': U, 'c': c, 's': s}
        e = rng.normal(0, .1, (n, 3, 8))
        for m in range(3):
            e[:, m] -= (e[:, m] @ maps[m]) @ maps[m].T
        z, zd = rng.normal(size=(n, 2)), rng.normal(size=(n, 3, 2))
        x_in = c * e + s * np.einsum('mik,nk->nmi', U, z)
        x_dec = c * e + s * np.einsum('mik,nmk->nmi', U, zd)
        sig, full = targets(x_in, d)
        sig_d, full_d = targets(x_dec, d)
        out[regime] = {'signal_oracle_in': float(np.mean((sig - full) ** 2)),
                       'full_oracle_dec': float(np.mean((full_d - sig_d) ** 2))}
        for value, table in zip(out[regime].values(), ORACLE_TABLE[regime]):
            assert abs(value / table - 1) < .02, (regime, value, table)
    return out


def verify():
    out = {'carried_over_from_ablation': cea.verify()}
    rng = np.random.default_rng(11)
    out['finite_difference_outer_families'] = max(finite_differences(model_for(f, 10.), rng)
                                                   for f in ('outer', 'outer_matched'))
    assert out['finite_difference_outer_families'] < 1e-6
    # The independent regime is v1's fixture byte for byte.
    for seed in (0, 3, 23):
        v1, d = base.fixture(seed), fixture(seed)['independent']
        assert all(np.array_equal(a, b) for a, b in zip(v1[0], d['waking']))
        assert np.array_equal(v1[1], d['test']) and np.array_equal(v1[2], d['reference'])
    out['independent_regime_is_v1_fixture'] = [0, 3, 23]
    data = fixture(3)
    d = data['weak']
    maps, U = d['maps'], d['U']
    out['U_orthonormality_error'] = float(max(np.max(abs(U[m].T @ U[m] - np.eye(2))) for m in range(3)))
    out['U_pattern_overlap'] = float(max(np.max(abs(maps[m].T @ U[m])) for m in range(3)))
    assert out['U_orthonormality_error'] < 1e-12 and out['U_pattern_overlap'] < 1e-12
    # Both test sets share one base array; recomputing it independently is bit-identical.
    for regime in ('weak', 'strong'):
        again = fixture(3)[regime]
        assert np.array_equal(again['base_test'], data[regime]['base_test'])
        off = lambda x: x - np.einsum('mik,mjk,nmj->nmi', U, U, x)  # remove each module's shared directions
        worst = float(np.max(abs(off(data[regime]['test']) - off(data[regime]['decorrelated']))))
        out[f'decorrelated_vs_in_distribution_outside_shared_span_{regime}'] = worst
        assert worst < 1e-14
    out['decorrelated_shares_base_array'] = True
    # mu_full equals the generic Gaussian conditional mean of the hidden nuisance.
    worst = 0.
    for regime in ('weak', 'strong'):
        d = data[regime]
        c, s = d['c'], d['s']
        Us = np.concatenate(list(U))
        cov = block_diag(*[.01 * c * c * (np.eye(8) - maps[m] @ maps[m].T) for m in range(3)]) + s * s * Us @ Us.T
        x = d['test']
        sig, full = targets(x, d)
        noise = (x - sig).reshape(len(x), 24)
        for h in range(3):
            hi = np.arange(8 * h, 8 * h + 8)
            oi = np.setdiff1d(np.arange(24), hi)
            cond = noise[:, oi] @ (cov[np.ix_(hi, oi)] @ np.linalg.pinv(cov[np.ix_(oi, oi)], rcond=1e-10)).T
            worst = max(worst, float(np.max(abs(sig[:, h] + cond - full[:, h]))))
    out['mu_full_vs_generic_conditional'] = worst
    assert worst < 1e-10
    out['oracle_monte_carlo'] = monte_carlo(maps, U)
    # Signal component equals the generator's amplitude x pattern.
    out['mu_sig_vs_generator'] = float(np.max(abs(cea.noiseless(data['strong']['test'], maps)
                                                   - cea.generator_means(3, 'test'))))
    assert out['mu_sig_vs_generator'] < 1e-12
    # Completion maps reproduce predictions; gamma is 1 for the oracle map and rotation invariant.
    d = data['weak']
    worst = 0.
    for family in FAMILIES:
        model = model_for(family, 2.)
        p = lcc.initialize(model, 99)
        for k in p:
            p[k] = p[k] + rng.normal(0, .1, p[k].shape)
        x = rng.normal(size=(7, 3, 8))
        for inf in ('exact', '32_step'):
            A, pred = completion_maps(model, p, inf), predictions(model, p, x, inf)
            for h in range(3):
                o = [m for m in range(3) if m != h]
                worst = max(worst, float(np.max(abs(x[:, o].reshape(7, 16) @ A[h].T - pred[h]))))
    out['completion_map_error'] = worst
    assert worst < 1e-10
    k = oracle_scale(d)
    star = []
    for h in range(3):
        o = [m for m in range(3) if m != h]
        G = k * U[h] @ np.hstack([np.eye(2), np.eye(2)])  # 8x4, mapped back to 8x16 through U_o^T
        star.append(G @ block_diag(U[o[0]], U[o[1]]).T)
    theta = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.array([[0, -1], [1, 0]])
    rotated = [U[h] @ R @ U[h].T @ star[h] for h in range(3)]
    out['oracle_gamma'] = gain(star, d)['gamma']
    out['rotated_oracle_gamma'] = gain(rotated, d)['gamma']
    assert abs(out['oracle_gamma'] - 1) < 1e-12 and abs(out['rotated_oracle_gamma'] - 1) < 1e-12
    # tau: the full oracle gives 1, the signal oracle 0.
    sig, full = targets(d['test'], d)
    out['tau_full_oracle'] = tau(np.transpose(full, (1, 0, 2)), d['test'], d)
    out['tau_signal_oracle'] = tau(np.transpose(sig, (1, 0, 2)), d['test'], d)
    assert abs(out['tau_full_oracle'] - 1) < 1e-12 and abs(out['tau_signal_oracle']) < 1e-12
    # Comparators: ridge equals the earlier fit; reduced-rank slope has rank 2 and equals ridge on rank-2 targets.
    train_set = np.concatenate(d['waking'])
    out['ridge_vs_floor_analysis'] = float(np.max(abs(linear_predict(linear_fit(train_set), d['test'])
                                                      - ridge_predictions(train_set, d['test']))))
    assert out['ridge_vs_floor_analysis'] < 1e-12
    ranks = [int(np.linalg.matrix_rank(w[:16], tol=1e-10)) for w in linear_fit(train_set, 2)]
    out['reduced_rank_slope_ranks'] = ranks
    assert all(r <= 2 for r in ranks)
    X = rng.normal(size=(32, 2, 8))
    Cmap = rng.normal(size=(16, 2)) @ rng.normal(size=(2, 8))
    synthetic = np.concatenate([(X.reshape(32, 16) @ Cmap)[:, None], X], axis=1)  # module 0 is rank-2 in 1, 2
    r, rr = linear_fit(synthetic), linear_fit(synthetic, 2)
    out['reduced_rank_equals_ridge_on_rank2_targets'] = float(np.max(abs(r[0] - rr[0])))
    assert out['reduced_rank_equals_ridge_on_rank2_targets'] < 1e-10
    # Reproduction: independent / uniform / 2 at eta 0.03 on seed 3 reproduces v1 arm 2a.
    v1 = json.loads(V1_RESULTS.read_text())['seeds']['3']['arms']['2a_clamped_energy_C']
    d = data['independent']
    model = model_for('uniform', 2.)
    final, _, _ = train(model, lcc.initialize(model, 3), cea.stages_for(d['waking']), .03, [300, 300])
    now, then = lcc.quick_rmse(model, final, d['test']), v1['evaluations'][-1]['whole_module_rmse_32_step']
    out['reproduction_v1_arm_2a_seed3'] = {k: abs(now[k] - then[k]) for k in ('A', 'B', 'combined')}
    assert max(out['reproduction_v1_arm_2a_seed3'].values()) < 1e-12
    return out


def pilot(datasets):
    rows, selected = {}, {}
    for name, regime, family, _, b in conditions():
        rows[name] = []
        for eta in ETA_GRID:
            scores, diverged = [], False
            for seed in PILOT_SEEDS:
                waking = datasets[seed][regime]['waking']
                model = model_for(family, b)
                final, record, _ = train(model, lcc.initialize(model, seed), cea.stages_for(waking, 0), eta, [C, C])
                odd = np.concatenate([waking[0][1::2], waking[1][1::2]]).reshape(-1, 24)
                s = None if record['diverged'] else masked_mse_exact(model, final, odd)
                diverged |= s is None or not np.isfinite(s)
                scores.append(s)
            rows[name].append({'eta': eta, 'scores_by_seed': scores, 'disqualified': diverged,
                               'mean_score': None if diverged else float(np.mean(scores))})
        best = min((r for r in rows[name] if not r['disqualified']), key=lambda r: (r['mean_score'], r['eta']))
        selected[name] = {'eta': best['eta'], 'at_grid_edge': best['eta'] in (ETA_GRID[0], ETA_GRID[-1])}
    return {'grids': rows, 'selected': selected,
            'score': 'mean over pilot seeds of exact-equilibrium whole-module masked squared error per coordinate '
                     'on odd-index waking traces after stage B; training on even-index traces only; '
                     'pilot test sets and decorrelated sets never scored'}


def category(gammas):
    if any(g is None for g in gammas):
        return 'undefined'
    if sum(g >= TRANSMITS for g in gammas) >= SEED_COUNT:
        return 'transmits'
    if sum(g <= FILTERS for g in gammas) >= SEED_COUNT:
        return 'filters'
    return 'unresolved'


def gamma_compare(g1, g2):
    g1, g2 = np.asarray(g1), np.asarray(g2)
    ratio = (g1.mean() + GAMMA_DELTA) / (g2.mean() + GAMMA_DELTA)
    lower, higher = int(np.sum(g1 < g2)), int(np.sum(g1 > g2))
    if ratio <= 1 / cea.WORSE and lower >= SEED_COUNT:
        status = 'reduces'
    elif ratio >= cea.WORSE and higher >= SEED_COUNT:
        status = 'increases'
    elif 1 / cea.COMPARABLE <= ratio <= cea.COMPARABLE:
        status = 'no_effect'
    else:
        status = 'unresolved'
    return {'ratio_first_to_second': float(ratio), 'seeds_first_lower': lower, 'seeds_first_higher': higher,
            'status': status}


def criteria(seeds):
    order = sorted(seeds, key=int)
    last = lambda s, n: seeds[s]['conditions'][n]['evaluations'][-1]['nuisance']['exact']
    X = lambda n, which: [last(s, n)[which]['combined']['avoidable_mse'] for s in order]
    Xin, Xdec = (lambda n: X(n, 'in_distribution')), (lambda n: X(n, 'decorrelated'))
    signal = lambda n, which: float(np.mean([last(s, n)[which]['combined']['signal_error_mse'] for s in order]))
    gam = lambda n: [last(s, n)['gain']['gamma'] for s in order]
    out = {k: {} for k in ('N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'R1', 'R2')}
    for name, regime, family, bname, _ in conditions():
        ev = [[e['nuisance']['exact']['in_distribution'] for e in seeds[s]['conditions'][name]['evaluations']]
              for s in order]
        b_falls = sum(e[2]['B']['avoidable_mse'] < e[1]['B']['avoidable_mse'] for e in ev)
        a_ratio = [(e[2]['A']['avoidable_mse'] + cea.DELTA) / (e[1]['A']['avoidable_mse'] + cea.DELTA) for e in ev]
        frozen = [seeds[s]['frozen'][f'{regime}_{family}_{bname}']['nuisance']['exact']['in_distribution']
                  ['combined']['avoidable_mse'] for s in order]
        to_frozen = (np.mean(Xin(name)) + cea.DELTA) / (np.mean(frozen) + cea.DELTA)
        a_ok = sum(r <= 1.5 for r in a_ratio)
        out['N1'][name] = {'B_falls_seeds': int(b_falls), 'A_retained_seeds': int(a_ok),
                           'A_ratio_by_seed': [float(r) for r in a_ratio], 'ratio_to_own_frozen': float(to_frozen),
                           'learns': bool(b_falls >= SEED_COUNT and a_ok >= SEED_COUNT and to_frozen <= .1)}
        g = gam(name)
        out['R1'][name] = {'mean_X_in': float(np.mean(Xin(name))), 'mean_X_dec': float(np.mean(Xdec(name))),
                           'mean_gamma': None if g[0] is None else float(np.mean(g)),
                           'mean_tau': None if regime == 'independent' else float(np.mean([last(s, name)['tau'] for s in order])),
                           'signal_error_in': signal(name, 'in_distribution'),
                           'signal_error_dec': signal(name, 'decorrelated')}
        if regime != 'independent':
            out['N3'][name] = {'category': category(g), 'gamma_by_seed': g}
    learns = lambda n: out['N1'][n]['learns']
    references = {}
    for regime in REGIMES:
        ref = {}
        for comp in ('ridge', 'reduced_rank_2'):
            rows = [seeds[s]['comparators'][regime][comp][-1] for s in order]
            gs = [r['gain']['gamma'] for r in rows]
            ref[comp] = {'mean_X_in': float(np.mean([r['in_distribution']['combined']['avoidable_mse'] for r in rows])),
                         'mean_X_dec': float(np.mean([r['decorrelated']['combined']['avoidable_mse'] for r in rows])),
                         'mean_gamma': None if gs[0] is None else float(np.mean(gs)),
                         'category': category(gs) if regime != 'independent' else 'undefined', 'gamma_by_seed': gs}
        for o in ('signal_oracle', 'full_oracle'):
            ref[o] = {w: float(np.mean([seeds[s]['oracles'][regime][w][o] for s in order])) for w in SETS}
        references[regime] = ref
    out['R1']['references'] = references
    for bname, _ in BUDGETS:
        n = lambda r, f: f'{r}_{f}_{bname}'
        out['N2'][bname] = {f'{a}_vs_{b}': cea.compare(Xin(n('independent', a)), Xin(n('independent', b)))
                            for i, a in enumerate(FAMILIES) for b in FAMILIES[i + 1:]}
        inner = n('weak', 'inner')
        inner_filters = out['N3'][inner]['category'] == 'filters'
        pairs = {}
        for other in ('uniform', 'outer'):
            o = n('weak', other)
            dec = cea.compare(Xdec(o), Xdec(inner))
            cost = cea.compare(Xin(inner), Xin(o))
            if dec['status'] == 'first_clearly_worse':
                verdict = 'advantage'
            elif dec['status'] == 'second_clearly_worse' or (dec['status'] == 'comparable'
                                                              and cost['status'] == 'first_clearly_worse'):
                verdict = 'dominated'
            elif dec['status'] == 'comparable':
                verdict = 'neutral'
            else:
                verdict = 'unresolved'
            pairs[other] = {'verdict': verdict, 'X_dec_other_vs_inner': dec, 'X_in_inner_vs_other': cost,
                            'both_learn': learns(inner) and learns(o)}
        v = [pairs[o]['verdict'] for o in ('uniform', 'outer')]
        if 'advantage' in v and 'dominated' in v:
            agg = 'mixed'
        elif 'dominated' in v:
            agg = 'counts_against_mechanism_on_this_fixture'
        elif v == ['advantage', 'advantage']:
            agg = 'supports_mechanism' if inner_filters else 'advantage_not_explained_by_filtering'
        elif sorted(v) == ['advantage', 'neutral']:
            agg = 'partial_support' if inner_filters else 'advantage_not_explained_by_filtering'
        else:
            agg = 'unresolved'
        out['N4'][bname] = {'pairs': pairs, 'inner_N3': out['N3'][inner]['category'], 'aggregate': agg,
                            'inner_learns': learns(inner)}
        weak, strong = out['N3'][inner]['category'], out['N3'][n('strong', 'inner')]['category']
        n5 = {('filters', 'transmits'): 'filtering_depends_on_regime', ('filters', 'filters'): 'filtering_persists'}
        out['N5'][bname] = {'weak': weak, 'strong': strong,
                            'conclusion': 'no_filtering_to_explain' if weak == 'transmits' else n5.get((weak, strong), 'unresolved')}
        u, om, ofull = n('weak', 'uniform'), n('weak', 'outer_matched'), n('weak', 'outer')
        out['N6'][bname] = {
            'gamma_uniform_vs_outer_matched': gamma_compare(gam(u), gam(om)),
            'X_dec_uniform_vs_outer_matched': cea.compare(Xdec(u), Xdec(om)),
            'X_in_uniform_vs_outer_matched': cea.compare(Xin(u), Xin(om)),
            'signal_error_dec': {'uniform': signal(u, 'decorrelated'), 'outer_matched': signal(om, 'decorrelated')},
            'both_learn': learns(u) and learns(om),
            'fixed_budget_uniform_vs_outer': {'gamma': gamma_compare(gam(u), gam(ofull)),
                                              'X_dec': cea.compare(Xdec(u), Xdec(ofull)),
                                              'X_in': cea.compare(Xin(u), Xin(ofull))}}
        out['N6'][bname]['conclusion'] = {'reduces': 'deeper_coupling_reduces_transmission',
                                          'increases': 'deeper_coupling_increases_transmission',
                                          'no_effect': 'no_effect_on_transmission'}.get(
            out['N6'][bname]['gamma_uniform_vs_outer_matched']['status'], 'unresolved')
        for regime in ('weak', 'strong'):
            i = n(regime, 'inner')
            rrr = references[regime]['reduced_rank_2']
            out['R2'][f'{regime}_{bname}'] = {'inner_mean_gamma': out['R1'][i]['mean_gamma'],
                                              'inner_mean_X_dec': out['R1'][i]['mean_X_dec'],
                                              'rrr_mean_gamma': rrr['mean_gamma'], 'rrr_mean_X_dec': rrr['mean_X_dec'],
                                              'inner_category': out['N3'][i]['category'], 'rrr_category': rrr['category'],
                                              'same_category': out['N3'][i]['category'] == rrr['category']}
    out['near_zero_delta'], out['gamma_delta'] = cea.DELTA, GAMMA_DELTA
    return out


def main():
    result = Path(str(PREFIX) + '-results.json')
    runtime = Path(str(PREFIX) + '-runtime.json')
    movement_path = Path(str(PREFIX) + '-movement.json')
    if any(f.exists() for f in (result, runtime, movement_path, Path(str(movement_path) + '.gz'))):
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    with phase('verification'):
        checks = verify()
    with phase('fixture_construction'):
        datasets = {}
        for seed in PILOT_SEEDS + SEEDS:
            datasets[seed] = fixture(seed)
            for d in datasets[seed].values():
                lcc.check_maps(seed, d['waking'], d['maps'])
    with phase('selection_pilot'):
        selection = pilot(datasets)
    science = {'checks': checks, 'selection': selection, 'cycles_per_stage': C, 'seeds': {},
               'regimes': {r: {'c': c, 's': s} for r, (c, s) in REGIMES.items()},
               'kappa': {f'{f}_{bn}': list(kappa(f, b)) for f in FAMILIES for bn, b in BUDGETS},
               'scope': 'clamped-energy block-local learning signals with global settling; labels, maps, shared '
                        'directions, decorrelated sets and target means evaluator only'}
    movement = {}
    for seed in SEEDS:
        row = {'frozen': {}, 'conditions': {}, 'comparators': {}, 'oracles': {}}
        with phase(f'seed{seed}_frozen_and_references'):
            for regime, d in datasets[seed].items():
                for family in FAMILIES:
                    for bname, b in BUDGETS:
                        model = model_for(family, b)
                        row['frozen'][f'{regime}_{family}_{bname}'] = evaluate(model, lcc.initialize(model, seed), d,
                                                                               d['waking'][0])
                row['comparators'][regime] = comparators(d)
                row['oracles'][regime] = oracles(d)
        movement[str(seed)] = {}
        initial = lcc.digest(base.initialize(seed))
        for name, regime, family, _, b in conditions():
            d = datasets[seed][regime]
            model = model_for(family, b)
            p = lcc.initialize(model, seed)
            assert lcc.digest({k: p[k] for k in ('W', 'V')}) == initial
            eta = selection['selected'][name]['eta']
            with phase(f'seed{seed}_{name}'):
                _, record, per_cycle = train(model, p, cea.stages_for(d['waking']), eta, [C, C], d,
                                             label=f'seed{seed}_{name}')
            record.update(regime=regime, family=family, budget=b, kappa=list(kappa(family, b)), eta=eta)
            row['conditions'][name] = record
            movement[str(seed)][name] = per_cycle
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
                  for f in (Path(__file__), Path(lcc.__file__), Path(base.__file__), Path(cea.__file__), CONTRACT,
                            V1_RESULTS)}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)

    def limit(name):
        p = Path('/sys/fs/cgroup') / name
        return p.read_text().strip() if p.exists() else None
    counts = dict(COUNTS)
    counts.update({f'ablation_{k}': v for k, v in cea.COUNTS.items()})
    counts.update({f'lcc_{k}': v for k, v in lcc.COUNTS.items()})
    counts.update({f'v1_module_{k}': v for k, v in base.COUNTS.items()})
    storage = {'parameters_uniform': 288 * 8, 'parameters_inner': 128 * 8, 'parameters_outer': 248 * 8,
               'adam_moments': '2 x parameters', 'adam_step_counter': 8, 'replay': 32 * 24 * 8,
               'episode_ids': 32 * 8, 'activity_batch': 32 * 42 * 8, 'dense_q': 42 * 42 * 8,
               'implicit_matrix': 42 * 42 * 8}
    info = {'phases': PHASES, 'counts': counts, 'training_cpu': CPU, 'hashes': hashes, 'revision': revision,
            'dirty_status': dirty, 'command': ' '.join(sys.argv), 'pilot_seeds': list(PILOT_SEEDS),
            'confirmatory_seeds': list(SEEDS), 'cycles_per_stage': C,
            'seed_policy': 'v1 SeedSequence [seed,1001] initialization; [seed,2002] v1 fixture; [seed,3003] shared '
                           'nuisance; [0,4004] verification Monte Carlo',
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
            'timing_scope': 'whole-run/phase mechanism timings including evaluation; training CPU excludes evaluation; not a latency benchmark',
            'wall_seconds_before_runtime_serialization': time.perf_counter() - wall,
            'cpu_seconds_before_runtime_serialization': time.process_time() - cpu}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'selection': selection['selected'], 'criteria': {k: v for k, v in science['criteria'].items()
                                                                       if k in ('N3', 'N4', 'N5', 'N6', 'R2')}}, indent=1))


if __name__ == '__main__':
    main()
