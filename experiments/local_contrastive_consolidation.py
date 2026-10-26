"""Two-phase local consolidation with learned graded translations.

Implements docs/local-contrastive-consolidation-v1-contract.md. Learners see only
observed outer activities and episode IDs; structure labels and private pattern
maps are used by the evaluator only.
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
from scipy.linalg import subspace_angles
from experiments import local_consolidation as base

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/local-contrastive-consolidation-v1-contract.md'
PREFIX = Path(os.environ.get('LCC_PREFIX', str(ROOT / 'reports/local-contrastive-consolidation-v1')))
C = int(os.environ.get('LCC_CYCLES', '300'))  # smoke runs override; the contract fixes 300
STEPS = 32
BETA = .01
DECAY = 1e-3
ADAM = (.9, .999, 1e-8)
CHECKPOINT = 50
ETA_GRID = (.003, .01, .03, .1)
LAMBDA_GRID = (1, 10, 100)
PILOT_SEEDS = (0, 1, 2)
SEEDS = tuple(int(s) for s in os.environ.get('LCC_SEEDS', '3,4,5,6,7,8,9,10,11,12').split(','))
FAMILIES = {'uniform': (1 / 7, 1 / 7, 1 / 7), 'inner': (0, 0, 1.), 'no_outer': (0, 1 / 3, 1 / 3),
            'outer_only': (1 / 4, 0, 0)}
LAYER_NAMES = ('outer', 'middle', 'inner')
COUNTS = {}
PHASES = {}
CPU_CURVES = {}


def count(key, n=1):
    COUNTS[key] = COUNTS.get(key, 0) + n


@contextlib.contextmanager
def phase(name):
    w, c = time.perf_counter(), time.process_time()
    yield
    PHASES[name] = {'wall_seconds': time.perf_counter() - w, 'cpu_seconds': time.process_time() - c}


class Model:
    """Layer layout and coupling for one family; kappa is per-unit strength by layer."""

    def __init__(self, family, kappa=None):
        self.family = family
        self.hier = family != 'outer_only'
        self.kappa = FAMILIES[family] if kappa is None else kappa
        self.width = 14 if self.hier else 8
        self.S = 3 * self.width
        self.layers = ((0, 8), (8, 12), (12, 14)) if self.hier else ((0, 8),)
        self.coupled = [l for l in range(len(self.layers)) if self.kappa[l] > 0]
        self.keys = (['W', 'V'] if self.hier else []) + [f'T{a}_{l}' for a in (0, 1) for l in self.coupled]
        self.cache = {}

    def sl(self, m, l):
        lo, hi = self.layers[l]
        return np.arange(m * self.width + lo, m * self.width + hi)

    def mask(self, hidden):
        """Observed outer coordinates are all except `hidden` [(module, coordinate)]."""
        key = tuple(hidden)
        if key not in self.cache:
            observed = [(m, j) for m in range(3) for j in range(8) if (m, j) not in hidden]
            obs = np.array([m * self.width + j for m, j in observed], int)
            cols = np.array([m * 8 + j for m, j in observed], int)
            free = np.setdiff1d(np.arange(self.S), obs)
            hid = np.array([np.searchsorted(free, m * self.width + j) for m, j in hidden], int)
            tcols = np.array([m * 8 + j for m, j in hidden], int)
            self.cache[key] = (obs, cols, free, hid, tcols)
        return self.cache[key]

    def module_mask(self, m):
        return self.mask([(m, j) for j in range(8)])

    def groups(self):
        out = {}
        for k in self.keys:
            name = k if k in ('W', 'V') else 'T_' + LAYER_NAMES[int(k.split('_')[1])]
            out.setdefault(name, []).append(k)
        return out


def initialize(model, seed):
    p = {}
    if model.hier:
        p.update(base.initialize(seed))
    for k in model.keys:
        if k.startswith('T'):
            n = len(model.sl(0, int(k.split('_')[1])))
            p[k] = np.eye(n)
    return p


def digest(p):
    return hashlib.sha256(b''.join(p[k].tobytes() for k in sorted(p))).hexdigest()


def vec(model, p):
    return np.concatenate([p[k].ravel() for k in model.keys])


def unvec(model, theta, template):
    out, i = {}, 0
    for k in model.keys:
        n = template[k].size
        out[k] = theta[i:i + n].reshape(template[k].shape)
        i += n
    return out


def build(model, p):
    count('q_constructions')
    q = np.zeros((model.S, model.S))
    if model.hier:
        for m in range(3):
            x, h, c = model.sl(m, 0), model.sl(m, 1), model.sl(m, 2)
            W, V = p['W'][m], p['V'][m]
            q[np.ix_(x, x)] += np.eye(8)
            q[np.ix_(x, h)] -= W
            q[np.ix_(h, x)] -= W.T
            q[np.ix_(h, h)] += W.T @ W + np.eye(4) + .05 * np.eye(4)
            q[np.ix_(h, c)] -= V
            q[np.ix_(c, h)] -= V.T
            q[np.ix_(c, c)] += V.T @ V + .05 * np.eye(2)
    else:
        q += .05 * np.eye(model.S)
    for a in (0, 1):
        for l in model.coupled:
            k, T = model.kappa[l], p[f'T{a}_{l}']
            ia, ib = model.sl(a, l), model.sl(a + 1, l)
            q[np.ix_(ib, ib)] += k * np.eye(len(ib))
            q[np.ix_(ia, ia)] += k * T.T @ T
            q[np.ix_(ib, ia)] -= k * T
            q[np.ix_(ia, ib)] -= k * T.T
    return q


def param_grad(model, p, s):
    """Mean over examples of dE/dtheta at states s (B,S); each block reads only its own layers."""
    count('energy_derivatives')
    n, g = len(s), {}
    if model.hier:
        g['W'], g['V'] = np.zeros_like(p['W']), np.zeros_like(p['V'])
        for m in range(3):
            x, h, c = s[:, model.sl(m, 0)], s[:, model.sl(m, 1)], s[:, model.sl(m, 2)]
            g['W'][m] = -(x - h @ p['W'][m].T).T @ h / n
            g['V'][m] = -(h - c @ p['V'][m].T).T @ c / n
    for a in (0, 1):
        for l in model.coupled:
            key = f'T{a}_{l}'
            sa, sb = s[:, model.sl(a, l)], s[:, model.sl(a + 1, l)]
            g[key] = -model.kappa[l] * (sb - sa @ p[key].T).T @ sa / n
    return g


def assemble(model, M, flat, y):
    obs, cols, free, _, _ = M
    s = np.zeros((len(flat), model.S))
    s[:, obs] = flat[:, cols]
    s[:, free] = y
    assert np.array_equal(s[:, obs], flat[:, cols])
    return s


def settle(q, M, flat, steps=STEPS, y0=None, sign=0, work=None, check=False, beta=BETA):
    """Implicit Euler (step 1) on the clamped energy, optionally with a +/-beta nudge."""
    obs, cols, free, hid, tcols = M
    vals = flat[:, cols]
    hessian = q[np.ix_(free, free)].copy()
    force = -(vals @ q[np.ix_(free, obs)].T)
    if sign:
        hessian[hid, hid] += sign * beta
        force[:, hid] += sign * beta * flat[:, tcols]
    transition = np.linalg.solve(np.eye(len(free)) + hessian, np.eye(len(free)))
    count('implicit_factorizations')
    bias = force @ transition.T
    y = np.zeros((len(flat), len(free))) if y0 is None else y0.copy()
    previous = y
    for _ in range(steps):
        if check:
            before = .5 * np.einsum('bi,ij,bj->b', y, hessian, y) - np.sum(y * force, axis=1)
        previous, y = y, y @ transition.T + bias
        if check:
            after = .5 * np.einsum('bi,ij,bj->b', y, hessian, y) - np.sum(y * force, axis=1)
            assert np.all(after <= before + 1e-10)
    count('settles')
    count('activity_example_steps', steps * len(flat))
    if work is not None:
        work['example_steps'] += steps * len(flat)
        work['weighted_operations_estimate'] += steps * len(flat) * len(free) ** 2
    return y, y - previous


def exact_free(q, M, flat):
    obs, cols, free, _, _ = M
    count('exact_solves')
    return np.linalg.solve(q[np.ix_(free, free)], -(q[np.ix_(free, obs)] @ flat[:, cols].T)).T


def zero_grad(p):
    return {k: np.zeros_like(v) for k, v in p.items()}


def two_phase_grad(model, p, q, flat, steps=STEPS, work=None, beta=BETA):
    g, skips = zero_grad(p), 0
    for m in range(3):
        M = model.module_mask(m)
        _, _, free, hid, _ = M
        y0, _ = settle(q, M, flat, steps, work=work)
        negative = q[np.ix_(free, free)].copy()
        negative[hid, hid] -= beta
        count('pd_checks', 2)
        if np.linalg.eigvalsh(np.eye(len(free)) + negative)[0] <= 0 or np.linalg.eigvalsh(negative)[0] <= 0:
            skips += 1
            count('pd_skips')
            continue
        yp, _ = settle(q, M, flat, steps, y0, +1, work=work, beta=beta)
        ym, _ = settle(q, M, flat, steps, y0, -1, work=work, beta=beta)
        gp = param_grad(model, p, assemble(model, M, flat, yp))
        gm = param_grad(model, p, assemble(model, M, flat, ym))
        for k in g:
            g[k] += (gp[k] - gm[k]) / (2 * beta) / 3
    return g, skips


def energy_grad(model, p, q, flat, work=None):
    M = model.mask([])
    y, _ = settle(q, M, flat, work=work)
    return param_grad(model, p, assemble(model, M, flat, y))


def exact_grad(model, p, q, flat):
    """Adjoint gradient of the exact-equilibrium masked loss; exact for a quadratic energy."""
    g, loss = zero_grad(p), 0.
    for m in range(3):
        M = model.module_mask(m)
        _, _, free, hid, tcols = M
        y = exact_free(q, M, flat)
        residual = np.zeros_like(y)
        residual[:, hid] = y[:, hid] - flat[:, tcols]
        adjoint = np.linalg.solve(q[np.ix_(free, free)], residual.T).T
        count('adjoint_solves')
        gp = param_grad(model, p, assemble(model, M, flat, y - adjoint))
        gm = param_grad(model, p, assemble(model, M, flat, y + adjoint))
        for k in g:
            g[k] += (gp[k] - gm[k]) / 2 / 3
        loss += .5 * np.mean(np.sum(residual[:, hid] ** 2, axis=1)) / 3
    return g, loss


def exact_loss(model, p, flat):
    q, loss = build(model, p), 0.
    for m in range(3):
        M = model.module_mask(m)
        y = exact_free(q, M, flat)
        loss += .5 * np.mean(np.sum((y[:, M[3]] - flat[:, M[4]]) ** 2, axis=1)) / 3
    return loss


def masked_mse(model, p, flat):
    """Pilot selection score: 32-step whole-module masked squared error per coordinate."""
    q, errors = build(model, p), []
    for m in range(3):
        M = model.module_mask(m)
        y, _ = settle(q, M, flat)
        errors.append(np.mean((y[:, M[3]] - flat[:, M[4]]) ** 2))
    return float(np.mean(errors))


def compare(model, g, gs):
    out = {}
    for name, keys in list(model.groups().items()) + [('all', model.keys)]:
        a = np.concatenate([g[k].ravel() for k in keys])
        b = np.concatenate([gs[k].ravel() for k in keys])
        nb, na = float(np.linalg.norm(b)), float(np.linalg.norm(a))
        near = nb < 1e-10
        out[name] = {'estimate_norm': na, 'exact_norm': nb, 'near_zero': near,
                     'cosine': None if near or na == 0 else float(a @ b / (na * nb)),
                     'relative_error': None if near else float(np.linalg.norm(a - b) / nb),
                     'norm_ratio': None if near else na / nb}
    return out


def fidelity(model, p, flat):
    q = build(model, p)
    g, skips = two_phase_grad(model, p, q, flat)
    gs, _ = exact_grad(model, p, q, flat)
    out = compare(model, g, gs)
    out['pd_skips'] = skips
    return out


def rmse_split(per_example):
    return {'combined': float(np.sqrt(np.mean(per_example))), 'A': float(np.sqrt(np.mean(per_example[:12]))),
            'B': float(np.sqrt(np.mean(per_example[12:])))}


def quick_rmse(model, p, test):
    q, flat, errors = build(model, p), test.reshape(len(test), 24), []
    for m in range(3):
        M = model.module_mask(m)
        y, _ = settle(q, M, flat)
        errors.append(np.mean((y[:, M[3]] - flat[:, M[4]]) ** 2, axis=1))
    count('evaluation_reconstruction_examples', 3 * len(test))
    return rmse_split(np.mean(errors, axis=0))


def evaluate(model, p, test, reference, maps):
    before = digest(p)
    q, flat = build(model, p), test.reshape(len(test), 24)
    e32, eex, last_step, distance = [], [], [], []
    for m in range(3):
        M = model.module_mask(m)
        y, delta = settle(q, M, flat)
        ystar = exact_free(q, M, flat)
        e32.append(np.mean((y[:, M[3]] - flat[:, M[4]]) ** 2, axis=1))
        eex.append(np.mean((ystar[:, M[3]] - flat[:, M[4]]) ** 2, axis=1))
        last_step.append(float(np.mean(np.linalg.norm(delta, axis=1))))
        distance.append(float(np.mean(np.linalg.norm(y - ystar, axis=1))))
    single = []
    for m in range(3):
        for j in range(8):
            M = model.mask([(m, j)])
            y, _ = settle(q, M, flat)
            single.append((y[:, M[3][0]] - flat[:, m * 8 + j]) ** 2)
    count('evaluation_reconstruction_examples', 27 * len(test))
    full = model.mask([])
    states = assemble(model, full, flat, settle(q, full, flat)[0])
    result = {'whole_module_rmse_32_step': rmse_split(np.mean(e32, axis=0)),
              'whole_module_rmse_exact': rmse_split(np.mean(eex, axis=0)),
              'single_coordinate_rmse_32_step': rmse_split(np.mean(single, axis=0)),
              'free_phase_last_step_norm_by_mask': last_step,
              'free_phase_distance_to_exact_by_mask': distance, 'abstraction': None, 'translations': {}}
    if model.hier:
        rflat = reference.reshape(len(reference), 24)
        rstates = assemble(model, full, rflat, settle(q, full, rflat)[0])
        count('evaluation_probe_examples', len(test) + len(reference))
        inner = np.concatenate([model.sl(m, 2) for m in range(3)])
        code, refcode = states[:, inner], rstates[:, inner]
        labels = np.repeat([0, 1], 12)  # evaluator only, never passed to learning
        centroids = np.stack((refcode[:8].mean(axis=0), refcode[8:].mean(axis=0)))
        prediction = np.argmin(np.sum((code[:, None] - centroids[None]) ** 2, axis=2), axis=1)
        centers = np.stack((code[:12].mean(axis=0), code[12:].mean(axis=0)))
        within = float(np.mean(np.sum((code - centers[labels]) ** 2, axis=1)))
        between = float(np.sum((centers[0] - centers[1]) ** 2))
        result['abstraction'] = {'within_structure_spread': within, 'between_centroid_distance_squared': between,
                                 'within_to_between_ratio': None if between < 1e-12 else within / between,
                                 'centroid_accuracy': float(np.mean(prediction == labels)),
                                 'mean_code_squared_norm': float(np.mean(np.sum(code ** 2, axis=1)))}
    for key in model.keys:
        if not key.startswith('T'):
            continue
        a, l = int(key[1]), int(key.split('_')[1])
        T, k = p[key], model.kappa[l]
        sa, sb = states[:, model.sl(a, l)], states[:, model.sl(a + 1, l)]
        row = {'layer': LAYER_NAMES[l], 'edge': [a, a + 1],
               'singular_values': np.linalg.svd(T, compute_uv=False).tolist(),
               'source_curvature_eigenvalues': np.linalg.eigvalsh(k * T.T @ T).tolist(), 'target_curvature': k,
               'functional_relative_residual': float(np.linalg.norm(sa @ T.T - sb) / np.linalg.norm(sb))}
        if l == 0:  # evaluator-only private maps; supported pattern subspace only
            row['pattern_relative_error'] = float(np.linalg.norm(T @ maps[a] - maps[a + 1]) / np.linalg.norm(maps[a + 1]))
            row['pattern_principal_angles'] = subspace_angles(T @ maps[a], maps[a + 1]).tolist()
        result['translations'][key] = row
    assert digest(p) == before
    return result


def train(model, p, stages, rule, eta, cycles, lam=None, uniform=None, evaluation=None, fidelity_checks=False,
          label=None):
    """stages: list of (waking, replay, replay_ids). Waking only runs frozen inference."""
    template, theta = p, vec(model, p)
    initial = theta.copy()
    bound = np.concatenate([np.full(p[k].size, 3. if k.startswith('T') else 2.) for k in model.keys])
    decay = np.concatenate([np.full(p[k].size, DECAY if k in ('W', 'V') else 0.) for k in model.keys])
    group_slices, i = {}, 0
    for k in model.keys:
        name = k if k in ('W', 'V') else 'T_' + LAYER_NAMES[int(k.split('_')[1])]
        group_slices.setdefault(name, []).append(np.arange(i, i + p[k].size))
        i += p[k].size
    group_slices = {k: np.concatenate(v) for k, v in group_slices.items()}
    first, second, step = np.zeros_like(theta), np.zeros_like(theta), 0
    omega, importance = np.zeros_like(theta), np.zeros_like(theta)
    work = {'example_steps': 0, 'weighted_operations_estimate': 0}
    evaluations, curves, fidelities, cpu_curve = [], [], [], []
    per_cycle = {name: [] for name in list(group_slices) + ['all']}
    path = {name: 0. for name in per_cycle}
    stage_b = {}
    clips = skips = masks = cumulative = 0
    train_cpu = train_wall = 0.
    diverged = False
    current = lambda: unvec(model, theta, template)
    if evaluation:
        evaluations.append(evaluate(model, current(), *evaluation))
    if fidelity_checks:
        fidelities.append(fidelity(model, current(), stages[0][1].reshape(-1, 24)))
    scales = []
    for index, (waking, replay, ids) in enumerate(stages):
        frozen = digest(current())
        settle(build(model, current()), model.mask([]), waking.reshape(len(waking), 24))
        count('waking_episode_exposures', len(waking))
        assert digest(current()) == frozen
        start = theta.copy()
        stage_path = {name: 0. for name in per_cycle}
        scale = np.ones_like(theta)
        if lam is not None:
            scale = 1 / (1 + lam * importance)
        if uniform is not None and index == 1:
            scale = np.full_like(theta, uniform)
        scales.append(float(np.mean(scale)))
        flat = replay.reshape(len(replay), 24)
        for cycle in range(cycles[index]):
            c0, w0 = time.process_time(), time.perf_counter()
            params = current()
            q = build(model, params)
            if rule == 'two_phase':
                g, skipped = two_phase_grad(model, params, q, flat, work=work)
                skips += skipped
                masks += 3
            elif rule == 'energy':
                g = energy_grad(model, params, q, flat, work=work)
            else:
                g, _ = exact_grad(model, params, q, flat)
                work['exact_solve_cycles'] = work.get('exact_solve_cycles', 0) + 1
            count('replay_trace_exposures', len(flat))
            raw = vec(model, g)
            gradient = raw + decay * theta
            step += 1
            first = ADAM[0] * first + (1 - ADAM[0]) * gradient
            second = ADAM[1] * second + (1 - ADAM[1]) * gradient ** 2
            update = eta * (first / (1 - ADAM[0] ** step)) / (np.sqrt(second / (1 - ADAM[1] ** step)) + ADAM[2])
            proposed = theta - scale * update
            new = np.clip(proposed, -bound, bound)
            clips += int(np.sum(new != proposed))
            delta = new - theta
            theta = new
            count('adam_steps')
            if lam is not None:
                omega += np.maximum(0., -raw * delta)
            norms = {name: float(np.linalg.norm(delta[s])) for name, s in group_slices.items()}
            norms['all'] = float(np.linalg.norm(delta))
            for name, value in norms.items():
                per_cycle[name].append(value)
                path[name] += value
                stage_path[name] += value
            cumulative += 1
            train_cpu += time.process_time() - c0
            train_wall += time.perf_counter() - w0
            if not np.all(np.isfinite(theta)):
                diverged = True
                break
            if (cycle + 1) % CHECKPOINT == 0:
                entry = {'stage': index, 'stage_cycle': cycle + 1, 'cumulative_cycles': cumulative,
                         'example_steps': work['example_steps'],
                         'weighted_operations_estimate': work['weighted_operations_estimate'],
                         'displacement_from_init': float(np.linalg.norm(theta - initial))}
                if evaluation:
                    entry['whole_module_rmse_32_step'] = quick_rmse(model, current(), evaluation[0])
                curves.append(entry)
                cpu_curve.append([cumulative, train_cpu, train_wall])
        if diverged:
            break
        if lam is not None:
            importance += omega / ((theta - start) ** 2 + 1e-3)
            omega[:] = 0
        if index == 1:
            stage_b = {'displacement': {n: float(np.linalg.norm((theta - start)[s])) for n, s in group_slices.items()},
                       'path_length': {n: stage_path[n] for n in group_slices}}
            stage_b['displacement']['all'] = float(np.linalg.norm(theta - start))
            stage_b['path_length']['all'] = stage_path['all']
        if evaluation:
            evaluations.append(evaluate(model, current(), *evaluation))
        if fidelity_checks:
            fidelities.append(fidelity(model, current(), flat))
    if label is not None:
        CPU_CURVES[label] = {'columns': ['cumulative_cycles', 'training_cpu_seconds', 'training_wall_seconds'],
                             'rows': cpu_curve, 'training_cpu_seconds': train_cpu, 'training_wall_seconds': train_wall}
    final = current()
    displacement = {n: float(np.linalg.norm((theta - initial)[s])) for n, s in group_slices.items()}
    displacement['all'] = float(np.linalg.norm(theta - initial))
    record = {'diverged': diverged, 'evaluations': evaluations, 'curves': curves, 'fidelity': fidelities,
              'clip_activations': clips, 'pd_skips': skips, 'two_phase_masks': masks, 'cycles': cumulative,
              'work': work, 'displacement_from_init': displacement, 'path_length': path,
              'stage_b_movement': stage_b, 'mean_step_scale_by_stage': scales,
              'final_parameters': {k: v.tolist() for k, v in final.items()}}
    if lam is not None:
        record['importance_scale_mean_at_end'] = float(np.mean(1 / (1 + lam * importance)))
    return final, record, per_cycle


def private_maps(seed):
    """Re-derive the fixture's private maps (its first RNG draws); evaluator only."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2002]))
    return np.stack([np.linalg.qr(rng.normal(size=(8, 2)))[0] for _ in range(3)])


def check_maps(seed, waking, maps):
    for label, data in enumerate(waking):
        coords = np.einsum('bmi,mik->bmk', data, maps)
        assert np.max(abs(coords[:, :, 1 - label])) < 1e-12
        assert np.all((coords[:, :, label] > .8 - 1e-12) & (coords[:, :, label] < 1.2 + 1e-12))


def verify():
    waking, _, _ = base.fixture(99)
    data = np.concatenate([waking[0][:4], waking[1][:4]])
    flat = data.reshape(8, 24)
    out = {}
    # The energy reduces to v1's when coupling is inner-only, identity, strength 0.2.
    v1 = Model('v1', (0, 0, .2))
    p = initialize(v1, 99)
    assert np.allclose(build(v1, p), base.build({'W': p['W'], 'V': p['V']}), atol=1e-12)
    rng = np.random.default_rng(99)
    worst_activity = worst_param = worst_exact = 0.
    for family in ('uniform', 'outer_only'):
        model = Model(family)
        p = initialize(model, 99)
        for k in p:
            if k.startswith('T'):
                p[k] = p[k] + rng.normal(0, .1, p[k].shape)
        q = build(model, p)
        s = rng.normal(size=(model.S,))
        for i in range(model.S):
            e = np.zeros(model.S)
            e[i] = 1e-6
            numeric = (.5 * (s + e) @ q @ (s + e) - .5 * (s - e) @ q @ (s - e)) / 2e-6
            worst_activity = max(worst_activity, abs(numeric - (q @ s)[i]))
        states = rng.normal(size=(5, model.S))
        g = param_grad(model, p, states)
        gs, _ = exact_grad(model, p, q, flat)
        for k in model.keys:
            for idx in np.ndindex(p[k].shape):
                plus = {a: b.copy() for a, b in p.items()}
                minus = {a: b.copy() for a, b in p.items()}
                plus[k][idx] += 1e-6
                minus[k][idx] -= 1e-6
                energy = lambda r: float(np.mean(np.einsum('bi,ij,bj->b', states, build(model, r), states)) / 2)
                worst_param = max(worst_param, abs((energy(plus) - energy(minus)) / 2e-6 - g[k][idx]))
                numeric = (exact_loss(model, plus, flat) - exact_loss(model, minus, flat)) / 2e-6
                worst_exact = max(worst_exact, abs(numeric - gs[k][idx]))
        # Implementation check: long settling with a tiny nudge recovers the exact gradient.
        # The contracted beta's long-settle and 32-step deviations are reported, not asserted.
        tiny, _ = two_phase_grad(model, p, q, flat, steps=3000, beta=1e-5)
        out[f'{family}_long_settle_beta_1e-5_fidelity'] = compare(model, tiny, gs)['all']
        assert out[f'{family}_long_settle_beta_1e-5_fidelity']['relative_error'] < 1e-6
        long, _ = two_phase_grad(model, p, q, flat, steps=3000)
        out[f'{family}_long_settle_fidelity'] = compare(model, long, gs)['all']
        out[f'{family}_32_step_fidelity'] = compare(model, two_phase_grad(model, p, q, flat)[0], gs)['all']
        # Implicit descent in free and nudged phases, exact clamps, and no hidden-target leakage.
        M = model.module_mask(1)
        y0, _ = settle(q, M, flat, check=True)
        settle(q, M, flat, y0=y0, sign=1, check=True)
        settle(q, M, flat, y0=y0, sign=-1, check=True)
        altered = flat.copy()
        altered[:, M[4]] += 1000
        assert np.array_equal(settle(q, M, altered)[0], y0)
    assert worst_activity < 1e-6 and worst_param < 1e-7 and worst_exact < 1e-6
    out.update(activity_gradient_max_error=worst_activity, parameter_gradient_max_error=worst_param,
               exact_loss_gradient_max_error=worst_exact, v1_energy_equivalence=True,
               implicit_descent_check=True, hidden_target_leakage_check=True)
    # Chain reversal with identity translations: reversed modules give reversed states.
    model = Model('uniform')
    p = initialize(model, 99)
    q = build(model, p)
    M = model.mask([])
    states = assemble(model, M, flat, settle(q, M, flat)[0]).reshape(8, 3, 14)
    rev = {'W': p['W'][::-1].copy(), 'V': p['V'][::-1].copy(), **{k: v for k, v in p.items() if k.startswith('T')}}
    rflat = data[:, ::-1].reshape(8, 24)
    rstates = assemble(model, M, rflat, settle(build(model, rev), M, rflat)[0]).reshape(8, 3, 14)
    assert np.allclose(rstates[:, ::-1], states, atol=1e-12)
    out['chain_reversal_identity_check'] = True
    out['chain_reversal_learned_translation'] = 'not tested; learned T breaks exact reversal symmetry'
    return out


def stages_for(waking, replay, subset=None):
    """replay 'AB' or 'B' for stage B. subset selects even/odd indices for the pilot."""
    a, b = waking
    ids_a, ids_b = np.arange(16), np.arange(16, 32)
    if subset is not None:
        a, b, ids_a, ids_b = a[subset::2], b[subset::2], ids_a[subset::2], ids_b[subset::2]
    second = (np.concatenate([a, b]), np.concatenate([ids_a, ids_b])) if replay == 'AB' else (b, ids_b)
    if replay == 'B':
        assert np.all(second[1] >= 16)
    return [(waking[0], a, ids_a), (waking[1], *second)]


def pilot(datasets):
    rows, selected = {}, {}
    uniform = Model('uniform')
    for family, rule in (('two_phase', 'two_phase'), ('clamped_energy', 'energy'), ('exact_reference', 'exact')):
        rows[family] = []
        for eta in ETA_GRID:
            scores, skips, masks, diverged = [], 0, 0, False
            for seed in PILOT_SEEDS:
                waking = datasets[seed][0]
                final, record, _ = train(uniform, initialize(uniform, seed), stages_for(waking, 'AB', 0), rule, eta,
                                         [C, C])
                diverged |= record['diverged']
                skips += record['pd_skips']
                masks += record['two_phase_masks']
                odd = np.concatenate([waking[0][1::2], waking[1][1::2]]).reshape(-1, 24)
                scores.append(None if record['diverged'] else masked_mse(uniform, final, odd))
            disqualified = diverged or (rule == 'two_phase' and skips > .01 * masks)
            rows[family].append({'eta': eta, 'scores_by_seed': scores, 'pd_skips': skips, 'masks': masks,
                                 'diverged': diverged, 'disqualified': disqualified,
                                 'mean_score': None if diverged else float(np.mean(scores))})
        eligible = [r for r in rows[family] if not r['disqualified']]
        best = min(eligible, key=lambda r: (r['mean_score'], r['eta']))
        selected[family] = {'eta': best['eta'], 'at_grid_edge': best['eta'] in (ETA_GRID[0], ETA_GRID[-1])}
    rows['inertia_lambda'] = []
    eta = selected['two_phase']['eta']
    for lam in LAMBDA_GRID:
        scores, diverged = [], False
        for seed in PILOT_SEEDS:
            waking = datasets[seed][0]
            final, record, _ = train(uniform, initialize(uniform, seed), stages_for(waking, 'B', 0), 'two_phase', eta,
                                     [C, C], lam=lam)
            diverged |= record['diverged']
            odd = np.concatenate([waking[0][1::2], waking[1][1::2]]).reshape(-1, 24)
            scores.append(None if record['diverged'] else masked_mse(uniform, final, odd))
        rows['inertia_lambda'].append({'lambda': lam, 'scores_by_seed': scores, 'diverged': diverged,
                                       'mean_score': None if diverged else float(np.mean(scores))})
    best = min([r for r in rows['inertia_lambda'] if not r['diverged']], key=lambda r: (r['mean_score'], r['lambda']))
    selected['inertia_lambda'] = {'lambda': best['lambda'],
                                  'at_grid_edge': best['lambda'] in (LAMBDA_GRID[0], LAMBDA_GRID[-1])}
    return {'grids': rows, 'selected': selected,
            'score': 'mean over pilot seeds of 32-step whole-module masked squared error per coordinate on odd-index waking traces after stage B; training on even-index traces only; pilot test sets never scored'}


def arms(selected):
    eta = {k: v['eta'] for k, v in selected.items() if k != 'inertia_lambda'}
    lam = selected['inertia_lambda']['lambda']
    return [('2a_clamped_energy_C', 'uniform', 'energy', eta['clamped_energy'], [C, C], 'AB', {}),
            ('2b_clamped_energy_9C', 'uniform', 'energy', eta['clamped_energy'], [9 * C, 9 * C], 'AB', {}),
            ('3_two_phase_uniform', 'uniform', 'two_phase', eta['two_phase'], [C, C], 'AB', {}),
            ('4_two_phase_inner', 'inner', 'two_phase', eta['two_phase'], [C, C], 'AB', {}),
            ('5_two_phase_no_outer', 'no_outer', 'two_phase', eta['two_phase'], [C, C], 'AB', {}),
            ('5b_two_phase_outer_only', 'outer_only', 'two_phase', eta['two_phase'], [C, C], 'AB', {}),
            ('6_exact_reference', 'uniform', 'exact', eta['exact_reference'], [C, C], 'AB', {}),
            ('7_two_phase_B_only', 'uniform', 'two_phase', eta['two_phase'], [C, C], 'B', {}),
            ('8_two_phase_B_only_inertia', 'uniform', 'two_phase', eta['two_phase'], [C, C], 'B', {'lam': lam}),
            ('9_two_phase_B_only_uniform_slowing', 'uniform', 'two_phase', eta['two_phase'], [C, C], 'B',
             {'uniform_from': '8_two_phase_B_only_inertia'})]


def criteria(seeds):
    final = lambda arm, metric='whole_module_rmse_32_step': [seeds[s]['arms'][arm]['evaluations'][-1][metric]
                                                              for s in seeds]
    comb = lambda arm, metric='whole_module_rmse_32_step': np.array([r['combined'] for r in final(arm, metric)])
    a3 = '3_two_phase_uniform'
    after_a = [seeds[s]['arms'][a3]['evaluations'][1]['whole_module_rmse_32_step'] for s in seeds]
    after_b = final(a3)
    frozen = np.array([seeds[s]['frozen']['whole_module_rmse_32_step']['combined'] for s in seeds])
    out = {}
    acquisition = sum(b['B'] < a['B'] for a, b in zip(after_a, after_b))
    retention = sum(b['A'] / a['A'] <= 1.1 for a, b in zip(after_a, after_b))
    out['P1'] = {'B_improved_seeds': int(acquisition), 'A_retained_seeds': int(retention),
                 'mean_final_combined': float(comb(a3).mean()), 'mean_frozen_combined': float(frozen.mean()),
                 'pass': bool(acquisition >= 9 and retention >= 9 and comb(a3).mean() <= .5 * frozen.mean())}
    beats = {arm: int(np.sum(comb(a3) < comb(arm))) for arm in ('2a_clamped_energy_C', '2b_clamped_energy_9C')}
    out['P2'] = {'seeds_arm3_better': beats, 'pass': bool(min(beats.values()) >= 9)}
    ratios = {m: float(comb(a3, m).mean() / comb('6_exact_reference', m).mean())
              for m in ('whole_module_rmse_32_step', 'whole_module_rmse_exact')}
    out['P3'] = {'mean_ratio_arm3_to_arm6': ratios, 'pass': bool(max(ratios.values()) <= 1.15)}
    r5 = float(comb('5_two_phase_no_outer').mean() / comb(a3).mean())
    r5b = float(comb('5b_two_phase_outer_only').mean() / comb(a3).mean())
    out['S1'] = {'mean_ratio_arm4_to_arm3': float(comb('4_two_phase_inner').mean() / comb(a3).mean()),
                 'mean_ratio_arm5_to_arm3': r5, 'mean_ratio_arm5b_to_arm3': r5b,
                 'outer_connections_help_this_configuration': r5 >= 1.25,
                 'shortcut_suffices_here': r5b <= 1.1, 'regularized_shortcut_underperforms': r5b >= 1.25}
    ratio = {}
    for arm in ('7_two_phase_B_only', '8_two_phase_B_only_inertia', '9_two_phase_B_only_uniform_slowing'):
        ev = [seeds[s]['arms'][arm]['evaluations'] for s in seeds]
        ratio[arm] = np.array([e[2]['whole_module_rmse_32_step']['A'] / e[1]['whole_module_rmse_32_step']['A']
                               for e in ev])
    b8 = np.array([r['B'] for r in final('8_two_phase_B_only_inertia')])
    b9 = np.array([r['B'] for r in final('9_two_phase_B_only_uniform_slowing')])
    flags = []
    for s in seeds:
        m8 = seeds[s]['arms']['8_two_phase_B_only_inertia']['stage_b_movement']
        m9 = seeds[s]['arms']['9_two_phase_B_only_uniform_slowing']['stage_b_movement']
        flagged = False
        for key in ('displacement', 'path_length'):
            x8, x9 = m8[key]['all'], m9[key]['all']
            flagged |= (abs(x9 - x8) > 1e-6) if x8 < 1e-6 else (abs(x9 - x8) / x8 > .2)
        flags.append(bool(flagged))
    better = (ratio['8_two_phase_B_only_inertia'] < ratio['7_two_phase_B_only']) & \
             (ratio['8_two_phase_B_only_inertia'] < ratio['9_two_phase_B_only_uniform_slowing'])
    keep = ~np.array(flags)

    def s2(mask):
        return {'seeds': int(mask.sum()), 'seeds_arm8_lower_A_ratio_than_7_and_9': int(np.sum(better & mask)),
                'mean_B_ratio_arm8_to_arm9': None if not mask.any() else float(b8[mask].mean() / b9[mask].mean())}
    primary = s2(np.ones(len(flags), bool))
    out['S2'] = {'mean_A_ratio': {k: float(v.mean()) for k, v in ratio.items()}, 'primary_all_seeds': primary,
                 'pass': bool(primary['seeds_arm8_lower_A_ratio_than_7_and_9'] >= 8
                              and primary['mean_B_ratio_arm8_to_arm9'] <= 1.1),
                 'movement_unmatched_flags': flags, 'sensitivity_unflagged_seeds': s2(keep),
                 'B_condition_interpretation': 'mean final B RMSE over the seeds considered'}
    ridge = np.array([seeds[s]['ridge'][1]['rmse'] for s in seeds])
    out['S3'] = {'mean_ratio_arm3_to_ridge': float(comb(a3).mean() / ridge.mean()),
                 'match': bool(comb(a3).mean() <= 1.1 * ridge.mean())}
    m1 = []
    for s in seeds:
        fid = seeds[s]['arms'][a3]['fidelity'][0]
        ok = all(v['near_zero'] or (v['cosine'] >= .95 and v['relative_error'] <= .3 and .8 <= v['norm_ratio'] <= 1.25)
                 for k, v in fid.items() if k not in ('all', 'pd_skips'))
        m1.append(bool(ok))
    out['M1'] = {'seeds_passing': int(sum(m1)), 'by_seed': m1, 'pass': all(m1),
                 'scope': 'arm 3 at initialization; continuous values for all arms and checkpoints are in fidelity'}
    return out


def main():
    result = Path(str(PREFIX) + '-results.json')
    runtime = Path(str(PREFIX) + '-runtime.json')
    movement_path = Path(str(PREFIX) + '-movement.json')
    if result.exists() or runtime.exists() or movement_path.exists():
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    with phase('verification'):
        checks = verify()
    with phase('fixture_construction'):
        datasets = {}
        for seed in PILOT_SEEDS + SEEDS:
            waking, test, reference = base.fixture(seed)
            maps = private_maps(seed)
            check_maps(seed, waking, maps)
            datasets[seed] = (waking, test, reference, maps)
    with phase('selection_pilot'):
        selection = pilot(datasets)
    science = {'checks': checks, 'selection': selection, 'cycles_per_stage': C, 'seeds': {},
               'scope': 'two-phase block-local learning signals with global settling; labels and maps evaluator only'}
    movement = {}
    for seed in SEEDS:
        waking, test, reference, maps = datasets[seed]
        evaluation = (test, reference, maps)
        uniform = Model('uniform')
        with phase(f'seed{seed}_1_frozen_evaluation'):
            row = {'frozen': evaluate(uniform, initialize(uniform, seed), *evaluation), 'arms': {}, 'ridge': []}
        movement[str(seed)] = {}
        initial_digest = digest({k: v for k, v in initialize(uniform, seed).items() if k in ('W', 'V')})
        for name, family, rule, eta, cycles, replay, extra in arms(selection['selected']):
            model = Model(family)
            p = initialize(model, seed)
            if model.hier:
                assert digest({k: v for k, v in p.items() if k in ('W', 'V')}) == initial_digest
            uniform_scale = None
            if 'uniform_from' in extra:
                uniform_scale = row['arms'][extra['uniform_from']]['mean_step_scale_by_stage'][1]
            with phase(f'seed{seed}_{name}'):
                _, record, per_cycle = train(model, p, stages_for(waking, replay), rule, eta, cycles,
                                             lam=extra.get('lam'), uniform=uniform_scale, evaluation=evaluation,
                                             fidelity_checks=rule == 'two_phase', label=f'seed{seed}_{name}')
            record.update(family=family, rule=rule, eta=eta, replay_stage_b=replay, uniform_scale=uniform_scale,
                          inertia_lambda=extra.get('lam'))
            row['arms'][name] = record
            movement[str(seed)][name] = per_cycle
        # Arms 3, 7, 8 and 9 share stage A exactly (no inertia before stage B).
        shared = [row['arms'][a]['evaluations'][1]['whole_module_rmse_32_step'] for a in
                  ('3_two_phase_uniform', '7_two_phase_B_only', '8_two_phase_B_only_inertia',
                   '9_two_phase_B_only_uniform_slowing')]
        assert all(s == shared[0] for s in shared)
        with phase(f'seed{seed}_10_ridge'):
            for stage in (0, 1):
                row['ridge'].append(base.direct_predictor(np.concatenate(waking[:stage + 1]), test))
        science['seeds'][str(seed)] = row
    with phase('criteria'):
        science['criteria'] = criteria(science['seeds'])
    with phase('serialization'):
        result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
        movement_path.write_text(json.dumps(movement, sort_keys=True, separators=(',', ':')) + '\n')
    with phase('provenance'):
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__), Path(base.__file__), CONTRACT)}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)

    def limit(name):
        p = Path('/sys/fs/cgroup') / name
        return p.read_text().strip() if p.exists() else None
    counts = dict(COUNTS)
    counts.update({f'v1_module_{k}': v for k, v in base.COUNTS.items()})
    storage = {'parameters_uniform': 288 * 8, 'adam_moments_uniform': 2 * 288 * 8, 'adam_step_counter': 8,
               'replay': 32 * 24 * 8, 'episode_ids': 32 * 8, 'inertia_state_uniform': 3 * 288 * 8,
               'activity_batches_max': 3 * 32 * 42 * 8, 'dense_q': 42 * 42 * 8, 'implicit_matrices_peak': 2 * 42 * 42 * 8}
    info = {'phases': PHASES, 'counts': counts, 'training_cpu_curves': CPU_CURVES, 'hashes': hashes,
            'revision': revision, 'dirty_status': dirty, 'command': ' '.join(sys.argv),
            'pilot_seeds': list(PILOT_SEEDS), 'confirmatory_seeds': list(SEEDS), 'cycles_per_stage': C,
            'seed_policy': 'v1 SeedSequence [seed,1001] initialization; [seed,2002] fixtures',
            'python': sys.version, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
            'os': platform.platform(), 'blas': config.getvalue(),
            'thread_environment': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'backend_threads': None, 'backend_threads_reason': 'environment fixed before NumPy import; active count not measured',
            'workers': 1, 'cpu': platform.processor() or Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
            'affinity_cpu_count': len(os.sched_getaffinity(0)), 'cpu_limit': limit('cpu.max'),
            'memory_limit': limit('memory.max'), 'dtype': 'float64',
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'rss_scope': 'Linux single-process lifetime peak; all reporting arrays included',
            'accelerator_memory': None, 'accelerator_reason': 'CPU only',
            'manual_design_seconds': None, 'manual_design_reason': 'not measured; exploratory probes have no cost receipts',
            'storage_bytes_per_learner': storage,
            'timing_scope': 'whole-run/phase mechanism timings including evaluation and diagnostics; training CPU curves exclude evaluation; not a warmed latency benchmark',
            'wall_seconds_before_runtime_serialization': time.perf_counter() - wall,
            'cpu_seconds_before_runtime_serialization': time.process_time() - cpu}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'selection': selection['selected'], 'criteria': science['criteria']}, indent=2))


if __name__ == '__main__':
    main()
