"""Finite task and independent masked-energy checks; no module training."""
from collections import Counter
from contextlib import contextmanager
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

from src import single_module_plasticity as v1

ROOT_SEED = 2026102502
TIE_TOL = 1e-12
VISIBLE = np.arange(16)
HIDDEN = np.arange(16, 32)
MASK = np.r_[np.ones(16), np.zeros(16)]
AFFINE_FLOOR = (8 - 2*np.sqrt(2))/21


class Meter:
    def __init__(self):
        self.counts = Counter()
        self.seconds = Counter()
        self.cpu_seconds = Counter()

    @contextmanager
    def phase(self, name):
        start, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter()-start
            self.cpu_seconds[name] += time.process_time()-cpu


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scenes():
    labels = np.array([(r, t) for r in (1, 2, 3) for t in range(8)])
    angles = (labels[:, 1, None] + labels[:, 0, None]*np.arange(4)) % 8
    clean = np.zeros((24, 4, 8))
    clean[:, :, 0] = np.cos(angles*np.pi/4)
    clean[:, :, 1] = np.sin(angles*np.pi/4)
    test = labels[:, 1] % 3 == labels[:, 0]-1
    return labels, angles, clean, test


def renderers(world):
    if world is None:
        return np.repeat(np.eye(8)[None], 4, axis=0)
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, world, 0]))
    return np.stack([v1.qf(rng.standard_normal((8, 8))) for _ in range(4)])


def render(clean, matrices):
    return np.einsum('jab,njb->nja', matrices, clean).reshape(len(clean), 32)


def random_params(rng):
    maps = [np.stack([v1.qf(rng.standard_normal((s, t))) for _ in range(4)])
            for s, t in zip(v1.SECTORS[:-1], v1.SECTORS[1:])]
    v = rng.standard_normal((4, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = dict(phase=0, maps=maps, v=v)
    v1.check_constraints(p)
    return p


def params(world, part):
    return random_params(np.random.default_rng(np.random.SeedSequence([ROOT_SEED, world, part])))


def tied_argmin(distances):
    """Last axis is in the predeclared lexicographic/codebook order."""
    return np.argmax(distances <= distances.min(axis=-1, keepdims=True)+TIE_TOL, axis=-1)


def decode(values, matrices, sectors):
    base = np.zeros((8, 8))
    base[:, 0], base[:, 1] = np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)
    choices = []
    for k, sector in enumerate(sectors):
        codebook = base@matrices[sector].T
        distances = np.sum((values[:, k*8:(k+1)*8, None]-codebook.T[None])**2, axis=1)
        choices.append(tied_argmin(distances))
    return np.stack(choices, axis=1)


def summarize(prediction, target, labels, angles, matrices, subset):
    loss = np.sum((prediction-target)**2, axis=1)/2
    decoded = decode(prediction, matrices, (2, 3))
    correct = decoded == angles[:, 2:]
    per_rule = {}
    for r in (1, 2, 3):
        pick = subset & (labels[:, 0] == r)
        per_rule[str(r)] = dict(n=int(pick.sum()), mean_loss=float(loss[pick].mean()),
                               worst_loss=float(loss[pick].max()),
                               accuracy_by_part=correct[pick].mean(axis=0),
                               joint_accuracy=float(correct[pick].all(axis=1).mean()))
    return dict(n=int(subset.sum()), scene_mean_loss=float(loss[subset].mean()),
                macro_rule_loss=float(np.mean([v['mean_loss'] for v in per_rule.values()])),
                worst_loss=float(loss[subset].max()), accuracy_by_part=correct[subset].mean(axis=0),
                joint_accuracy=float(correct[subset].all(axis=1).mean()), per_rule=per_rule)


def prediction_record(prediction, x, matrices):
    labels, angles, _, test = scenes()
    return dict(prediction_hidden=prediction,
                per_scene_loss=np.sum((prediction-x[:, HIDDEN])**2, axis=1)/2,
                decoded_hidden=decode(prediction, matrices, (2, 3)),
                scores={name: summarize(prediction, x[:, HIDDEN], labels, angles, matrices, pick)
                        for name, pick in [('train', ~test), ('held_out', test), ('all', np.ones(24, bool))]})


def task_audit():
    labels, angles, clean, test = scenes()
    matched = []
    for j in range(4):
        for a in range(24):
            for b in range(a+1, 24):
                if angles[a, j] == angles[b, j] and labels[a, 0] != labels[b, 0]:
                    matched.append(dict(sector=j, scenes=[a, b], held_out=[bool(test[a]), bool(test[b])]))
    counts = {}
    for name, subset in [('train', ~test), ('held_out', test), ('all', np.ones(24, bool))]:
        counts[name] = {str(r): [np.bincount(angles[subset & (labels[:, 0] == r), j], minlength=8)
                                for j in range(4)] for r in (1, 2, 3)}
    visible_keys = [(int(a), int(b)) for a, b in angles[:, :2]]
    require(len(set(visible_keys)) == 24, 'visible evidence is ambiguous')
    require(all(np.any((labels[:, 1] == t) & ~test) for t in range(8)), 'unseen orientation factor')
    return dict(labels=labels, part_orientations=angles, clean_identity=clean.reshape(24, 32),
                train_scene_indices=np.flatnonzero(~test), held_out_scene_indices=np.flatnonzero(test),
                part_orientation_counts=counts, orientation_matched_pairs=matched,
                oracle_identifiable=True, visible_squared_norm=2, hidden_squared_norm=2)


def affine_fit(x, y, meter):
    design = np.column_stack([x, np.ones(len(x))])
    weights, _, rank, singular = np.linalg.lstsq(design, y, rcond=1e-12)
    meter.counts['svd_affine_fits'] += 1
    return weights, dict(rank=int(rank), singular_values=singular, persistent_bytes=weights.nbytes)


def comparator_audit(x, matrices, meter):
    labels, angles, _, test = scenes()
    train, y = ~test, x[:, HIDDEN]
    records = {}
    mean = y[train].mean(axis=0)
    meter.counts['constant_mean_fits'] += 1
    records['constant'] = dict(prediction_record(np.broadcast_to(mean, y.shape), x, matrices),
                               persistent_bytes=mean.nbytes)
    for name, cols, fit in [('first_part_affine', np.arange(8), train),
                            ('two_part_affine', VISIBLE, train),
                            ('all_scene_affine_capacity_oracle', VISIBLE, np.ones(24, bool))]:
        weights, metadata = affine_fit(x[fit][:, cols], y[fit], meter)
        pred = np.column_stack([x[:, cols], np.ones(24)])@weights
        records[name] = dict(prediction_record(pred, x, matrices), fit=metadata,
                             weights=weights, training_scene_indices=np.flatnonzero(fit))
    distances = np.sum((x[:, None, VISIBLE]-x[train][None, :, VISIBLE])**2, axis=-1)
    chosen = np.flatnonzero(train)[tied_argmin(distances)]
    records['nearest_training_input'] = dict(prediction_record(y[chosen], x, matrices),
        chosen_training_scene=chosen, persistent_bytes=x[train][:, VISIBLE].nbytes+y[train].nbytes)
    meter.counts['lookup_reference_builds'] += 1
    meter.counts['nearest_neighbor_distance_evaluations'] += 24*int(train.sum())
    decoded_visible = decode(x[:, VISIBLE], matrices, (0, 1))
    oracle_r = (decoded_visible[:, 1]-decoded_visible[:, 0]) % 8
    require(np.array_equal(oracle_r, labels[:, 0]), 'relation oracle failed')
    oracle_angles = (decoded_visible[:, 0, None] + oracle_r[:, None]*np.array([2, 3])) % 8
    oracle_clean = np.zeros((24, 4, 8))
    oracle_clean[:, 2:, 0], oracle_clean[:, 2:, 1] = np.cos(oracle_angles*np.pi/4), np.sin(oracle_angles*np.pi/4)
    oracle = render(oracle_clean, matrices)[:, HIDDEN]
    records['rule_oracle_evaluator_only'] = prediction_record(oracle, x, matrices)
    require(np.max(np.abs(oracle-y)) < 1e-10, 'rule oracle is not exact')
    actual = records['all_scene_affine_capacity_oracle']['scores']['all']['scene_mean_loss']
    require(abs(actual-AFFINE_FLOOR) < 1e-10, 'affine residual differs from analytic expression')
    meter.counts['comparator_scored_scenes'] += 6*24
    return records


def masked_hessian(curvatures, a, mask):
    h = v1.hessians(curvatures, a[None])[0].copy()
    h[np.arange(32), np.arange(32)] -= v1.ANCHOR*(1-mask)
    return h


def masked_energy(state, observed, p, a, mask):
    # v1.energy expands residuals directly, independently of the Hessian code.
    return (v1.energy(state, observed, p, a)
            - v1.ANCHOR/2*np.sum((1-mask)*(state[:32]-observed)**2))


def solve_checked(p, curvatures, a, mask, observed, meter):
    h = masked_hessian(curvatures, a, mask)
    require(np.max(np.abs(h-h.T)) < 1e-12, 'Hessian symmetry')
    rhs = np.zeros(80)
    rhs[:32] = v1.ANCHOR*mask*observed
    l = cho_factor(h, lower=True, check_finite=True)
    meter.counts['cholesky_80'] += 1
    state = cho_solve(l, rhs)
    meter.counts['cholesky_vector_solves_80'] += 1
    residual = np.linalg.norm(h@state-rhs)/max(1., np.linalg.norm(rhs))
    require(residual <= 1e-10, 'full solve residual')
    eigen = np.linalg.eigvalsh(h)
    meter.counts['eigenspectra_80'] += 1
    require(eigen[0] > 0, 'Hessian not positive definite')

    inner = lu_factor(h[32:, 32:])
    t = lu_solve(inner, h[32:, :32])
    schur = h[:32, :32]-h[:32, 32:]@t
    outer = lu_factor(schur)
    x0 = lu_solve(outer, rhs[:32])
    reference = np.r_[x0, -t@x0]
    schur_error = np.linalg.norm(reference-state)/max(1., np.linalg.norm(state))
    require(schur_error <= 1e-9, 'full/Schur disagreement')
    meter.counts.update(lu_factor_48=1, lu_factor_32=1, lu_matrix_solve_48_rhs32=1, lu_vector_solve_32=1)

    anchor = np.diag(v1.ANCHOR*mask)
    structural_schur = schur-anchor
    s_eigen = np.linalg.eigvalsh(structural_schur)
    meter.counts['eigenspectra_structural_schur_32'] += 1
    require(s_eigen[0] >= -1e-10, 'structural Schur is not PSD')
    full_rhs_map = np.vstack([anchor, np.zeros((48, 32))])
    full_map = cho_solve(l, full_rhs_map)[:32]
    schur_map = lu_solve(outer, anchor)
    meter.counts.update(cholesky_matrix_solve_80_rhs32=1, lu_matrix_solve_32_rhs32=1)
    map_error = np.linalg.norm(full_map-schur_map)/max(1., np.linalg.norm(full_map))
    require(map_error <= 1e-9, 'full/Schur map disagreement')

    energy_error = 0.
    for vector in (state, np.linspace(-.7, .9, 80)):
        direct = masked_energy(vector, observed, p, a, mask)
        quadratic = vector@h@vector/2-rhs@vector + v1.ANCHOR*np.sum(mask*observed**2)/2
        energy_error = max(energy_error, abs(direct-quadratic)/max(1., abs(direct)))
    require(energy_error < 1e-10, 'residual-energy expansion mismatch')
    meter.counts['residual_energy_expansions'] += 2
    metrics = dict(relative_residual=residual, schur_relative_error=schur_error,
        map_relative_error=map_error, energy_relative_error=energy_error,
        hessian_min_eigenvalue=eigen[0], hessian_max_eigenvalue=eigen[-1],
        hessian_condition=eigen[-1]/eigen[0], structural_schur_min_eigenvalue=s_eigen[0])
    if np.all(mask == 1):
        eig_f = np.linalg.eigvalsh(full_map)
        cross = np.linalg.svd(full_map[np.ix_(HIDDEN, VISIBLE)], compute_uv=False)[0]
        meter.counts.update(eigenspectra_outer_map_32=1, transfer_svd_16=1)
        require(np.max(np.abs(full_map-full_map.T)) < 1e-9, 'outer map symmetry')
        require(eig_f[0] >= -1e-9 and eig_f[-1] <= 1+1e-9 and cross <= .5+1e-9,
                'original-interface contraction bound')
        metrics.update(map_min_eigenvalue=eig_f[0], map_max_eigenvalue=eig_f[-1], transfer_operator_norm=cross)
    meter.counts['checked_equilibrium_passes'] += 1
    return state, metrics


def fixture_audit(p, x, matrices, interface, policy, meter, context):
    mask = np.ones(32) if interface == 'zero_filled' else MASK
    curvatures = v1.block_curvatures(p)
    meter.counts['curvature_assemblies'] += 1
    predictions, checks, g_extrema = [], [], []
    for i, scene in enumerate(x):
        context.update(scene=i, pass_name='pilot' if policy == 'matched' else 'fixed')
        observed = scene*MASK
        a = np.ones((4, 4))
        state, metrics = solve_checked(p, curvatures, a, mask, observed, meter)
        checks.append(metrics)
        if policy == 'matched':
            g = v1.gates(p, state[None])[0]
            require(np.all((g >= .1) & (g <= 1)), 'gate bounds')
            a, _ = v1.routes(g[None], 'matched')
            g_extrema.append([g.min(), g.max()])
            meter.counts['route_evaluations'] += 1
            context['pass_name'] = 'second'
            state, metrics = solve_checked(p, curvatures, a[0], mask, observed, meter)
            checks.append(metrics)
        prediction = state[HIDDEN]
        if interface == 'zero_filled':
            require(np.sum((prediction-scene[HIDDEN])**2)/2 >= .25-1e-9, 'clean loss floor')
        predictions.append(prediction)
    rec = prediction_record(np.array(predictions), x, matrices)
    rec['checks'] = {key: dict(min=min(c[key] for c in checks), max=max(c[key] for c in checks))
                     for key in checks[0]}
    rec['gate_range'] = [np.min(g_extrema), np.max(g_extrema)] if g_extrema else None
    rec['is_capacity_witness'] = bool(rec['scores']['all']['worst_loss'] <= .10
                                      and rec['scores']['all']['joint_accuracy'] == 1.)
    meter.counts['module_scored_scenes'] += len(x)
    return rec
