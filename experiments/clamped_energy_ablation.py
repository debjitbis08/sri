"""Clamped-energy ablation: fixed, rotation-only and free translations across coupling.

Implements docs/clamped-energy-ablation-v1-contract.md. Learners see only
observed outer activities and episode IDs; structure labels, private pattern
maps and the noiseless conditional mean are used by the evaluator only.
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
from experiments import local_consolidation as base
from experiments import local_contrastive_consolidation as lcc
from experiments.local_contrastive_floor_analysis import ridge_predictions

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/clamped-energy-ablation-v1-contract.md'
V1_RESULTS = ROOT / 'reports/local-contrastive-consolidation-v1-results.json'
PREFIX = Path(os.environ.get('CEA_PREFIX', str(ROOT / 'reports/clamped-energy-ablation-v1')))
C = int(os.environ.get('CEA_CYCLES', '300'))  # smoke runs override; the contract fixes 300
ETA_GRID = (.003, .01, .03, .1)
PILOT_SEEDS = (0, 1, 2)
SEEDS = tuple(int(s) for s in os.environ.get('CEA_SEEDS', '13,14,15,16,17,18,19,20,21,22').split(','))
FAMILIES = ('uniform', 'inner')
BUDGETS = (('0.4', .4), ('2', 2.), ('10', 10.))
MODES = ('F', 'O', 'L')
DELTA = 1e-5
COMPARABLE, WORSE, SEED_COUNT = 1.25, 1.5, 9
CHECKPOINT = 50
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
    return (b / 14, b / 14, b / 14) if family == 'uniform' else (0., 0., b / 2)


def conditions():
    return [(f'{f}_{m}_{name}', f, m, name, b) for f in FAMILIES for name, b in BUDGETS for m in MODES]


def model_for(family, b):
    return lcc.Model(family, kappa(family, b))


def learnable(model, mode):
    return [k for k in model.keys if mode != 'F' or not k.startswith('T')]


def group(k):
    return k if k in ('W', 'V') else 'T_' + lcc.LAYER_NAMES[int(k.split('_')[1])]


def noiseless(data, maps):
    """Noiseless conditional mean: projection onto each module's pattern span (evaluator only)."""
    return np.einsum('mij,mkj,nmk->nmi', maps, maps, data)


def avoidable(model, p, data, maps, splits):
    """Observed, floor, noise-subtracted, avoidable MSE and cross term, 32-step and exact, per split."""
    q, flat, mean = lcc.build(model, p), data.reshape(len(data), 24), noiseless(data, maps)
    terms = {'32_step': [], 'exact': []}
    for m in range(3):
        M = model.module_mask(m)
        y32, _ = lcc.settle(q, M, flat)
        yex = lcc.exact_free(q, M, flat)
        for name, y in (('32_step', y32[:, M[3]]), ('exact', yex[:, M[3]])):
            pred, target, mu = y, data[:, m], mean[:, m]
            terms[name].append(np.stack([np.mean((pred - target) ** 2, axis=1), np.mean((target - mu) ** 2, axis=1),
                                         np.mean((pred - mu) ** 2, axis=1),
                                         -2 * np.mean((pred - mu) * (target - mu), axis=1)]))
    count('avoidable_evaluation_examples', 3 * len(data))
    out = {}
    for name, rows in terms.items():
        per = np.mean(rows, axis=0)  # (4, n): averaged over masks
        out[name] = {}
        for split, idx in splits.items():
            o, f, a, c = (float(np.mean(per[i, idx])) for i in range(4))
            assert abs(o - f - a - c) < 1e-12
            out[name][split] = {'observed_mse': o, 'floor_mse': f, 'noise_subtracted_mse': o - f,
                                'avoidable_mse': a, 'cross_term': c}
    return out


TEST_SPLITS = {'combined': np.arange(24), 'A': np.arange(12), 'B': np.arange(12, 24)}


def train_splits(n):
    return {'combined': np.arange(n)} if n == 16 else {'combined': np.arange(32), 'A': np.arange(16),
                                                       'B': np.arange(16, 32)}


def train(model, mode, p, stages, eta, cycles, evaluation=None, label=None):
    """Clamped-energy Adam training; stages: list of (waking, replay). F keeps T fixed; O projects T."""
    keys = learnable(model, mode)
    fixed = {k: v.copy() for k, v in p.items() if k not in keys}
    fixed_digest = lcc.digest(fixed) if fixed else None
    sizes = [p[k].size for k in keys]
    theta = np.concatenate([p[k].ravel() for k in keys])
    initial = theta.copy()
    bound = np.concatenate([np.full(p[k].size, 3. if k.startswith('T') else 2.) for k in keys])
    decay = np.concatenate([np.full(p[k].size, lcc.DECAY if k in ('W', 'V') else 0.) for k in keys])
    slices, i = {}, 0
    for k, n in zip(keys, sizes):
        slices[k] = slice(i, i + n)
        i += n
    groups = {}
    for k in keys:
        groups.setdefault(group(k), []).append(np.arange(slices[k].start, slices[k].stop))
    groups = {g: np.concatenate(v) for g, v in groups.items()}

    def current():
        out = {k: v.copy() for k, v in fixed.items()}
        for k in keys:
            out[k] = theta[slices[k]].reshape(p[k].shape)
        return out

    first, second, step = np.zeros_like(theta), np.zeros_like(theta), 0
    clips = {g: 0 for g in groups}
    per_cycle = {g: [] for g in list(groups) + ['all']}
    path = {g: 0. for g in per_cycle}
    work = {'example_steps': 0, 'weighted_operations_estimate': 0}
    evaluations, curves, cpu_curve = [], [], []
    projections, orth_worst, cumulative, train_cpu, train_wall, diverged = 0, 0., 0, 0., 0., False
    if evaluation:
        evaluations.append(evaluate(model, current(), evaluation, stages[0][1]))
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
            if mode == 'O':
                for k in keys:
                    if k.startswith('T'):
                        u, _, vt = np.linalg.svd(new[slices[k]].reshape(p[k].shape))
                        T = u @ vt
                        orth_worst = max(orth_worst, float(np.max(abs(T.T @ T - np.eye(len(T))))))
                        assert orth_worst < 1e-10
                        new[slices[k]] = T.ravel()
                        projections += 1
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
            if evaluation and (cycle + 1) % CHECKPOINT == 0:
                test, _, _, maps = evaluation
                now = current()
                curves.append({'stage': index, 'stage_cycle': cycle + 1, 'cumulative_cycles': cumulative,
                               'test': avoidable(model, now, test, maps, TEST_SPLITS)['32_step']['combined'],
                               'train': avoidable(model, now, replay, maps, train_splits(len(replay)))['32_step']['combined'],
                               'displacement_from_init': float(np.linalg.norm(theta - initial))})
                cpu_curve.append([cumulative, train_cpu, train_wall])
        if diverged:
            break
        if evaluation:
            evaluations.append(evaluate(model, current(), evaluation, replay))
    final = current()
    if fixed:
        assert lcc.digest({k: final[k] for k in fixed}) == fixed_digest
    count('orthogonal_projections', projections)
    if label is not None:
        CPU[label] = {'training_cpu_seconds': train_cpu, 'training_wall_seconds': train_wall,
                      'columns': ['cumulative_cycles', 'training_cpu_seconds', 'training_wall_seconds'],
                      'rows': cpu_curve}
    displacement = {name: float(np.linalg.norm((theta - initial)[idx])) for name, idx in groups.items()}
    displacement['all'] = float(np.linalg.norm(theta - initial))
    record = {'diverged': diverged, 'evaluations': evaluations, 'curves': curves, 'clip_activations': clips,
              'orthogonal_projections': projections, 'max_orthogonality_error': orth_worst,
              'cycles': cumulative, 'work': work, 'displacement_from_init': displacement, 'path_length': path,
              'final_parameters': {k: v.tolist() for k, v in final.items()}}
    return final, record, per_cycle


def evaluate(model, p, evaluation, replay):
    test, reference, _, maps = evaluation
    out = lcc.evaluate(model, p, test, reference, maps)
    out['avoidable_test'] = avoidable(model, p, test, maps, TEST_SPLITS)
    out['avoidable_train'] = avoidable(model, p, replay, maps, train_splits(len(replay)))
    gap = {}
    for split in out['avoidable_train']['32_step']:
        tr, te = out['avoidable_train']['32_step'][split], out['avoidable_test']['32_step'][split]
        gap[split] = {k: te[k] - tr[k] for k in ('observed_mse', 'avoidable_mse')}
    out['generalization_gap_32_step'] = gap
    return out


def stages_for(waking, subset=None):
    a, b = waking
    if subset is not None:
        a, b = a[subset::2], b[subset::2]
    return [(waking[0], a), (waking[1], np.concatenate([a, b]))]


def generator_means(seed, data_index):
    """Replay the fixture's RNG to get amplitude x pattern for the test set (verification only)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2002]))
    maps = np.stack([np.linalg.qr(rng.normal(size=(8, 2)))[0] for _ in range(3)])
    blocks = []
    for label, n in ((0, 16), (1, 16), (0, 12), (1, 12)):
        amplitude = rng.uniform(.8, 1.2, (n, 1, 1))
        rng.normal(0, .1, (n, 3, 8))
        blocks.append(amplitude * maps[:, :, label][None])
    return np.concatenate(blocks[2:]) if data_index == 'test' else None


def verify():
    out = {}
    # inner / F / 0.4 is v1's coupling and energy.
    model = model_for('inner', .4)
    p = lcc.initialize(model, 99)
    rng = np.random.default_rng(7)
    W, V = rng.normal(0, .5, p['W'].shape), rng.normal(0, .5, p['V'].shape)
    p.update(W=W, V=V)
    out['inner_F_0.4_matches_v1_build'] = float(np.max(abs(lcc.build(model, p) - base.build({'W': W, 'V': V}))))
    assert out['inner_F_0.4_matches_v1_build'] < 1e-12
    # Finite differences for activity and parameter energy gradients, including T, E = s'Qs/2.
    worst_activity = worst_param = 0.
    for family, b in (('uniform', 10.), ('inner', .4)):
        model = model_for(family, b)
        p = lcc.initialize(model, 99)
        for k in p:
            p[k] = p[k] + rng.normal(0, .1, p[k].shape)
        q, s = lcc.build(model, p), rng.normal(size=(model.S,))
        for i in range(model.S):
            e = np.zeros(model.S)
            e[i] = 1e-6
            numeric = (.5 * (s + e) @ q @ (s + e) - .5 * (s - e) @ q @ (s - e)) / 2e-6
            worst_activity = max(worst_activity, abs(numeric - (q @ s)[i]))
        states = rng.normal(size=(5, model.S))
        g = lcc.param_grad(model, p, states)
        energy = lambda r: float(np.mean(np.einsum('bi,ij,bj->b', states, lcc.build(model, r), states)) / 2)
        for k in model.keys:
            for idx in np.ndindex(p[k].shape):
                plus = {a: v.copy() for a, v in p.items()}
                minus = {a: v.copy() for a, v in p.items()}
                plus[k][idx] += 1e-6
                minus[k][idx] -= 1e-6
                worst_param = max(worst_param, abs((energy(plus) - energy(minus)) / 2e-6 - g[k][idx]))
    out['finite_difference_activity'], out['finite_difference_parameter'] = worst_activity, worst_param
    assert worst_activity < 1e-6 and worst_param < 1e-6
    # Exact clamps (asserted in lcc.assemble) and implicit descent in clamped and masked settles.
    waking, test, _ = base.fixture(99)
    flat = np.concatenate(waking).reshape(32, 24)
    q = lcc.build(model_for('uniform', 2.), lcc.initialize(model_for('uniform', 2.), 99))
    for M in (model_for('uniform', 2.).mask([]), model_for('uniform', 2.).module_mask(1)):
        lcc.settle(q, M, flat, check=True)
    out['implicit_descent'] = True
    # Avoidable-error decomposition: mu is the generator's noiseless value.
    maps = lcc.private_maps(99)
    out['noiseless_mean_vs_generator'] = float(np.max(abs(noiseless(test, maps) - generator_means(99, 'test'))))
    assert out['noiseless_mean_vs_generator'] < 1e-12
    # F keeps T byte-identical, O stays orthogonal (asserted inside train), on a short run.
    for mode in ('F', 'O'):
        model = model_for('uniform', 2.)
        _, record, _ = train(model, mode, lcc.initialize(model, 99), stages_for(waking), .03, [5, 5])
        out[f'{mode}_short_run'] = {'max_orthogonality_error': record['max_orthogonality_error'],
                                    'projections': record['orthogonal_projections']}
    out['F_translations_byte_identical'] = True
    # Reproduction: uniform / L / 2 at eta 0.03 on seed 3 reproduces local contrastive v1 arm 2a.
    v1 = json.loads(V1_RESULTS.read_text())['seeds']['3']['arms']['2a_clamped_energy_C']
    waking, test, _ = base.fixture(3)
    model = model_for('uniform', 2.)
    final, _, _ = train(model, 'L', lcc.initialize(model, 3), stages_for(waking), .03, [300, 300])
    now = lcc.quick_rmse(model, final, test)
    then = v1['evaluations'][-1]['whole_module_rmse_32_step']
    out['reproduction_v1_arm_2a_seed3'] = {k: abs(now[k] - then[k]) for k in ('A', 'B', 'combined')}
    assert max(out['reproduction_v1_arm_2a_seed3'].values()) < 1e-12
    return out


def pilot(datasets):
    rows, selected = {}, {}
    for name, family, mode, _, b in conditions():
        rows[name] = []
        for eta in ETA_GRID:
            scores, diverged = [], False
            for seed in PILOT_SEEDS:
                waking = datasets[seed][0]
                model = model_for(family, b)
                final, record, _ = train(model, mode, lcc.initialize(model, seed), stages_for(waking, 0), eta, [C, C])
                diverged |= record['diverged']
                odd = np.concatenate([waking[0][1::2], waking[1][1::2]]).reshape(-1, 24)
                score = None if record['diverged'] else lcc.masked_mse(model, final, odd)
                diverged |= score is None or not np.isfinite(score)
                scores.append(score)
            rows[name].append({'eta': eta, 'scores_by_seed': scores, 'disqualified': diverged,
                               'mean_score': None if diverged else float(np.mean(scores))})
        eligible = [r for r in rows[name] if not r['disqualified']]
        best = min(eligible, key=lambda r: (r['mean_score'], r['eta']))
        selected[name] = {'eta': best['eta'], 'at_grid_edge': best['eta'] in (ETA_GRID[0], ETA_GRID[-1])}
    return {'grids': rows, 'selected': selected,
            'score': 'mean over pilot seeds of 32-step whole-module masked squared error per coordinate on '
                     'odd-index waking traces after stage B; training on even-index traces only; pilot test sets never scored'}


def compare(x1, x2):
    """Per-seed avoidable errors of conditions 1 and 2 -> comparison status under the contract."""
    x1, x2 = np.asarray(x1), np.asarray(x2)
    ratio = (x1.mean() + DELTA) / (x2.mean() + DELTA)
    worse1, worse2 = int(np.sum(x1 > x2)), int(np.sum(x2 > x1))
    if max(ratio, 1 / ratio) <= COMPARABLE:
        status = 'comparable'
    elif ratio >= WORSE and worse1 >= SEED_COUNT:
        status = 'first_clearly_worse'
    elif 1 / ratio >= WORSE and worse2 >= SEED_COUNT:
        status = 'second_clearly_worse'
    else:
        status = 'unresolved'
    return {'ratio_first_to_second': float(ratio), 'seeds_first_worse': worse1, 'seeds_second_worse': worse2,
            'status': status}


def criteria(seeds):
    order = list(seeds)
    av = lambda ev, split='combined': ev['avoidable_test']['32_step'][split]['avoidable_mse']
    X = lambda c: [av(seeds[s]['conditions'][c]['evaluations'][-1]) for s in order]
    out = {'A1': {}, 'A2': {}, 'A3': {}, 'A4': {}, 'R1': {}}
    for name, family, mode, bname, _ in conditions():
        ev = [seeds[s]['conditions'][name]['evaluations'] for s in order]
        b_falls = sum(av(e[2], 'B') < av(e[1], 'B') for e in ev)
        a_ratio = [(av(e[2], 'A') + DELTA) / (av(e[1], 'A') + DELTA) for e in ev]
        frozen = [av(seeds[s]['frozen'][f'{family}_{bname}']) for s in order]
        to_frozen = (np.mean(X(name)) + DELTA) / (np.mean(frozen) + DELTA)
        a_ok = sum(r <= 1.5 for r in a_ratio)
        out['A1'][name] = {'B_falls_seeds': int(b_falls), 'A_retained_seeds': int(a_ok),
                           'A_ratio_by_seed': [float(r) for r in a_ratio], 'ratio_to_own_frozen': float(to_frozen),
                           'mean_avoidable': float(np.mean(X(name))), 'mean_frozen_avoidable': float(np.mean(frozen)),
                           'learns': bool(b_falls >= SEED_COUNT and a_ok >= SEED_COUNT and to_frozen <= .1)}
        ridge = [seeds[s]['ridge'][-1]['avoidable']['combined']['avoidable_mse'] for s in order]
        out['R1'][name] = compare(X(name), ridge)
    learns = lambda c: out['A1'][c]['learns']
    for family in FAMILIES:
        F, O, L = (f'{family}_{m}_2' for m in MODES)
        a2 = compare(X(F), X(L))
        a2['conclusion'] = {'first_clearly_worse': 'decisive', 'comparable': 'not_decisive'}.get(a2['status'], 'unresolved')
        out['A2'][family] = a2
        a3 = compare(X(O), X(L))
        if a3['status'] == 'comparable':
            a3['conclusion'] = 'isometric_suffices' if learns(O) and learns(L) else 'no_conclusion_not_both_learn'
        else:
            a3['conclusion'] = 'non_isometric_helps' if a3['status'] == 'first_clearly_worse' else 'unresolved'
        a3['O_vs_F'] = compare(X(O), X(F))
        out['A3'][family] = a3
        for mode in MODES:
            mid = f'{family}_{mode}_2'
            weak, strong = compare(X(f'{family}_{mode}_0.4'), X(mid)), compare(X(f'{family}_{mode}_10'), X(mid))
            weak['conclusion'] = 'weak_coupling_limits' if weak['status'] == 'first_clearly_worse' else 'none'
            strong['conclusion'] = 'strong_coupling_limits' if strong['status'] == 'first_clearly_worse' else 'none'
            out['A4'][f'{family}_{mode}'] = {'b0.4_vs_b2': weak, 'b10_vs_b2': strong}
    if learns('inner_F_0.4'):
        a5 = 'v1_coupling_can_support_learning_under_this_protocol'
    elif learns('inner_L_0.4') or learns('inner_F_2'):
        a5 = 'v1_coupling_limiting_under_this_protocol'
    else:
        a5 = 'unresolved'
    out['A5'] = {'conclusion': a5, 'inner_F_0.4_learns': learns('inner_F_0.4'),
                 'inner_L_0.4_learns': learns('inner_L_0.4'), 'inner_F_2_learns': learns('inner_F_2')}
    out['near_zero_delta'] = DELTA
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
            maps = lcc.private_maps(seed)
            lcc.check_maps(seed, waking, maps)
            datasets[seed] = (waking, test, reference, maps)
    with phase('selection_pilot'):
        selection = pilot(datasets)
    science = {'checks': checks, 'selection': selection, 'cycles_per_stage': C, 'seeds': {},
               'scope': 'clamped-energy block-local learning signals with global settling; labels, maps and noiseless means evaluator only'}
    movement = {}
    for seed in SEEDS:
        waking, test, reference, maps = datasets[seed]
        evaluation = (test, reference, None, maps)
        row = {'frozen': {}, 'conditions': {}, 'ridge': []}
        with phase(f'seed{seed}_frozen_evaluation'):
            for family in FAMILIES:
                for bname, b in BUDGETS:
                    model = model_for(family, b)
                    row['frozen'][f'{family}_{bname}'] = evaluate(model, lcc.initialize(model, seed), evaluation,
                                                                  waking[0])
        movement[str(seed)] = {}
        initial = lcc.digest({k: v for k, v in base.initialize(seed).items()})
        for name, family, mode, _, b in conditions():
            model = model_for(family, b)
            p = lcc.initialize(model, seed)
            assert lcc.digest({k: p[k] for k in ('W', 'V')}) == initial
            eta = selection['selected'][name]['eta']
            with phase(f'seed{seed}_{name}'):
                _, record, per_cycle = train(model, mode, p, stages_for(waking), eta, [C, C], evaluation,
                                             label=f'seed{seed}_{name}')
            record.update(family=family, mode=mode, budget=b, kappa=list(kappa(family, b)), eta=eta)
            row['conditions'][name] = record
            movement[str(seed)][name] = per_cycle
        with phase(f'seed{seed}_ridge'):
            for stage in (0, 1):
                train_set = np.concatenate(waking[:stage + 1])
                entry = base.direct_predictor(train_set, test)
                pred, mean = ridge_predictions(train_set, test), noiseless(test, maps)
                target = np.transpose(test, (1, 0, 2))
                mu = np.transpose(mean, (1, 0, 2))
                entry['avoidable'] = {}
                for split, idx in TEST_SPLITS.items():
                    entry['avoidable'][split] = {
                        'observed_mse': float(np.mean((pred[:, idx] - target[:, idx]) ** 2)),
                        'floor_mse': float(np.mean((target[:, idx] - mu[:, idx]) ** 2)),
                        'avoidable_mse': float(np.mean((pred[:, idx] - mu[:, idx]) ** 2))}
                row['ridge'].append(entry)
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
        hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in (Path(__file__), Path(lcc.__file__), Path(base.__file__), CONTRACT, V1_RESULTS)}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)

    def limit(name):
        p = Path('/sys/fs/cgroup') / name
        return p.read_text().strip() if p.exists() else None
    counts = dict(COUNTS)
    counts.update({f'lcc_{k}': v for k, v in lcc.COUNTS.items()})
    counts.update({f'v1_module_{k}': v for k, v in base.COUNTS.items()})
    storage = {'parameters_F': 120 * 8, 'parameters_uniform_O_L': 288 * 8, 'parameters_inner_O_L': 128 * 8,
               'adam_moments': '2 x parameters', 'adam_step_counter': 8, 'replay': 32 * 24 * 8,
               'episode_ids': 32 * 8, 'activity_batch': 32 * 42 * 8, 'dense_q': 42 * 42 * 8,
               'implicit_matrix': 42 * 42 * 8}
    info = {'phases': PHASES, 'counts': counts, 'training_cpu': CPU, 'hashes': hashes, 'revision': revision,
            'dirty_status': dirty, 'command': ' '.join(sys.argv), 'pilot_seeds': list(PILOT_SEEDS),
            'confirmatory_seeds': list(SEEDS), 'cycles_per_stage': C,
            'seed_policy': 'v1 SeedSequence [seed,1001] initialization; [seed,2002] fixtures',
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
    print(json.dumps({'selection': selection['selected'], 'criteria': science['criteria']}, indent=1))


if __name__ == '__main__':
    main()
