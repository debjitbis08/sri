"""Weak-prior topology v1: correlated-nuisance topology comparison at activity priors 0.05 and 0.001.

Implements docs/weak-prior-topology-v1-contract.md. Learners see only observed outer
activities and episode IDs; labels, maps, shared directions, target means and the
decorrelated sets are evaluator only.

Training reuses prior_strength_training.run with its arm 'E' configuration (32-step
settles, weight decay, clips) for every family; the arm name selects only that
regularization, while the model passed in sets the family.
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
from experiments import prior_strength_training as pst

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/weak-prior-topology-v1-contract.md'
PST_RESULTS = ROOT / 'reports/prior-strength-training-v1-results.json.gz'
PREFIX = Path(os.environ.get('WPT_PREFIX', str(ROOT / 'reports/weak-prior-topology-v1')))
C = int(os.environ.get('WPT_CYCLES', '300'))  # smoke runs override; the contract fixes 300 per stage
ETA_GRID = pst.ETA_GRID
PILOT_SEEDS = (0, 1, 2)
SEEDS = tuple(int(s) for s in os.environ.get('WPT_SEEDS', '53,54,55,56,57,58,59,60,61,62').split(','))
REGIMES = ('independent', 'weak', 'strong')
FAMILIES = ('uniform', 'inner', 'outer', 'outer_matched')
BUDGETS = (('2', 2.), ('10', 10.))
PRIORS = (('0.05', .05), ('0.001', .001))
SEED_COUNT = cea.SEED_COUNT
PHASES = {}


@contextlib.contextmanager
def phase(name):
    w, c = time.perf_counter(), time.process_time()
    yield
    PHASES[name] = {'wall_seconds': time.perf_counter() - w, 'cpu_seconds': time.process_time() - c}


def name_of(regime, family, lname, bname):
    return f'{regime}_{family}_{lname}_{bname}'


def conditions():
    return [(name_of(r, f, ln, bn), r, f, ln, lam, bn, b) for r in REGIMES for f in FAMILIES for ln, lam in PRIORS
            for bn, b in BUDGETS]


def verify():
    out = {'carried_over_from_prior_strength_training': pst.verify()}
    model = cn.model_for('outer', 2.)
    p = lcc.initialize(model, 99)
    assert np.array_equal(pst.build_with(model, p, .05), pst.ORIGINAL_BUILD(model, p))
    out['min_initial_eigenvalue'] = min(float(np.linalg.eigvalsh(pst.build_with(cn.model_for(f, b), lcc.initialize(
        cn.model_for(f, b), s), lam))[0]) for f in FAMILIES for _, b in BUDGETS for _, lam in PRIORS for s in (0, 53))
    assert out['min_initial_eigenvalue'] > 0
    old = json.loads(gzip.decompress(PST_RESULTS.read_bytes()))['seeds']['43']['conditions']['weak_E_0.001_10']
    d = cn.fixture(43)['weak']
    model = cn.model_for('inner', 10.)
    _, record = pst.run(model, lcc.initialize(model, 43), cea.stages_for(d['waking']), .1, [300, 300], 'E', .001, d)
    now, then = record['evaluations'][-1]['nuisance']['exact'], old['evaluations'][-1]['nuisance']['exact']
    out['reproduction_prior_strength_seed43_weak_E_0.001_10'] = {
        w: abs(now[w]['combined']['avoidable_mse'] - then[w]['combined']['avoidable_mse']) for w in cn.SETS}
    assert max(out['reproduction_prior_strength_seed43_weak_E_0.001_10'].values()) < 1e-12
    return out


def pilot(datasets):
    rows, selected = {}, {}
    for name, regime, family, lname, lam, bname, b in conditions():
        rows[name] = []
        for eta in ETA_GRID:
            scores, bad = [], False
            for seed in PILOT_SEEDS:
                d = datasets[seed][regime]
                model = cn.model_for(family, b)
                final, record = pst.run(model, lcc.initialize(model, seed), cea.stages_for(d['waking'], 0), eta,
                                        [C, C], 'E', lam)
                odd = np.concatenate([d['waking'][0][1::2], d['waking'][1][1::2]]).reshape(-1, 24)
                s = None
                if not record['diverged']:
                    with pst.priors(lam):
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
            'score': 'exact-equilibrium observed masked MSE under the condition\'s own prior on odd-index waking traces '
                     'after stage B; training on even-index traces'}


def criteria(seeds, selection):
    order = sorted(seeds, key=int)
    failed = lambda n: selection['selected'][n]['failed'] or any(seeds[s]['conditions'][n]['diverged'] for s in order)
    last = lambda s, n: seeds[s]['conditions'][n]['evaluations'][-1]['nuisance']['exact']
    X = lambda n, w: [last(s, n)[w]['combined']['avoidable_mse'] for s in order]
    Xin, Xdec = (lambda n: X(n, 'in_distribution')), (lambda n: X(n, 'decorrelated'))
    gam = lambda n: [last(s, n)['gain']['gamma'] for s in order]
    out = {k: {} for k in ('N1', 'summary', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6')}
    for name, regime, family, lname, lam, bname, b in conditions():
        if failed(name):
            out['summary'][name] = {'failed': True}
            out['N1'][name] = {'learns': False, 'failed': True}
            continue
        ev = [[e['nuisance']['exact']['in_distribution'] for e in seeds[s]['conditions'][name]['evaluations']] for s in order]
        fz = [seeds[s]['frozen'][name]['nuisance']['exact']['in_distribution'] for s in order]
        b_falls = sum(e[2]['B']['avoidable_mse'] < e[1]['B']['avoidable_mse'] for e in ev)
        a_abs = sum(e[2]['A']['avoidable_mse'] <= .05 * f['A']['avoidable_mse'] for e, f in zip(ev, fz))
        a_ratio = sum((e[2]['A']['avoidable_mse'] + cea.DELTA) / (e[1]['A']['avoidable_mse'] + cea.DELTA) <= 1.5 for e in ev)
        to_frozen = (np.mean(Xin(name)) + cea.DELTA) / (np.mean([f['combined']['avoidable_mse'] for f in fz]) + cea.DELTA)
        out['N1'][name] = {'B_falls_seeds': int(b_falls), 'A_retained_absolute_seeds': int(a_abs),
                           'A_retained_ratio_form_seeds': int(a_ratio), 'ratio_to_own_frozen': float(to_frozen),
                           'learns': bool(b_falls >= SEED_COUNT and a_abs >= SEED_COUNT and to_frozen <= .1)}
        g = gam(name)
        out['summary'][name] = {
            'eta': selection['selected'][name]['eta'], 'mean_X_in': float(np.mean(Xin(name))),
            'mean_X_dec': float(np.mean(Xdec(name))), 'mean_gamma': None if g[0] is None else float(np.mean(g)),
            'category': cn.category(g) if regime != 'independent' else 'undefined',
            'mean_signal_gain': float(np.mean([seeds[s]['conditions'][name]['evaluations'][-1]['signal_gain']['decorrelated']
                                               for s in order])),
            'mean_signal_error_dec': float(np.mean([last(s, name)['decorrelated']['combined']['signal_error_mse'] for s in order]))}
    cat = lambda n: out['summary'][n].get('category', 'failed')
    learns = lambda n: out['N1'][n]['learns']
    refs = {}
    for regime in REGIMES:
        refs[regime] = {}
        for comp in ('uncentered_rank_2', 'ridge_intercept'):
            rows = [seeds[s]['comparators'][regime][comp][-1] for s in order]
            refs[regime][comp] = {w: float(np.mean([r[w]['combined']['avoidable_mse'] for r in rows])) for w in cn.SETS}
            gs = [r['gain']['gamma'] for r in rows]
            refs[regime][comp]['mean_gamma'] = None if gs[0] is None else float(np.mean(gs))
    out['summary']['comparators'] = refs

    def cmp(a, b, f):
        return {'status': 'failed_condition'} if failed(a) or failed(b) else cea.compare(f(a), f(b))

    for lname, _ in PRIORS:
        for bname, _ in BUDGETS:
            key = f'{lname}_{bname}'
            n = lambda r, f: name_of(r, f, lname, bname)
            out['T1'][key] = {f'{a}_vs_{b}': cmp(n('independent', a), n('independent', b), Xin)
                              for i, a in enumerate(FAMILIES) for b in FAMILIES[i + 1:]}
            out['T2'][key] = {f'{r}_{f}': cat(n(r, f)) for r in ('weak', 'strong') for f in FAMILIES}
            inner = n('weak', 'inner')
            pairs = {}
            for other in ('uniform', 'outer'):
                o = n('weak', other)
                dec, cost = cmp(o, inner, Xdec), cmp(inner, o, Xin)
                if dec['status'] == 'first_clearly_worse':
                    v = 'advantage'
                elif dec['status'] == 'second_clearly_worse' or (dec['status'] == 'comparable'
                                                                 and cost['status'] == 'first_clearly_worse'):
                    v = 'dominated'
                elif dec['status'] == 'comparable':
                    v = 'neutral'
                else:
                    v = 'unresolved'
                pairs[other] = {'verdict': v, 'X_dec_other_vs_inner': dec, 'X_in_inner_vs_other': cost,
                                'learns': {'inner': learns(inner), other: learns(o)}}
            vs = [pairs[o]['verdict'] for o in ('uniform', 'outer')]
            filt = cat(inner) == 'filters'
            if 'advantage' in vs and 'dominated' in vs:
                agg = 'mixed'
            elif 'dominated' in vs:
                agg = 'counts_against'
            elif vs == ['advantage', 'advantage']:
                agg = 'supports_mechanism' if filt else 'advantage_not_explained_by_filtering'
            elif sorted(vs) == ['advantage', 'neutral']:
                agg = 'partial_support' if filt else 'advantage_not_explained_by_filtering'
            else:
                agg = 'unresolved'
            out['T3'][key] = {'pairs': pairs, 'inner_category': cat(inner), 'aggregate': agg}
            w, s_ = cat(n('weak', 'inner')), cat(n('strong', 'inner'))
            both = learns(n('weak', 'inner')) and learns(n('strong', 'inner'))
            if both and (w, s_) == ('filters', 'transmits'):
                t4 = 'filtering_depends_on_regime'
            elif both and (w, s_) == ('filters', 'filters'):
                t4 = 'filtering_persists'
            else:
                t4 = 'descriptive_only'
            out['T4'][key] = {'weak': w, 'strong': s_, 'both_learn': both, 'conclusion': t4}
            u, om = n('weak', 'uniform'), n('weak', 'outer_matched')
            if failed(u) or failed(om):
                t5 = {'status': 'failed_condition'}
            else:
                t5 = cn.gamma_compare(gam(u), gam(om))
            out['T5'][key] = {'gamma_uniform_vs_outer_matched': t5, 'X_dec': cmp(u, om, Xdec), 'X_in': cmp(u, om, Xin),
                              'signal_error_dec': {f: out['summary'][x].get('mean_signal_error_dec')
                                                   for f, x in (('uniform', u), ('outer_matched', om))},
                              'learns': {'uniform': learns(u), 'outer_matched': learns(om)}}
    for bname, _ in BUDGETS:
        a, b = f'0.05_{bname}', f'0.001_{bname}'
        out['T6'][bname] = {
            'T1': {k: [out['T1'][a][k]['status'], out['T1'][b][k]['status']] for k in out['T1'][a]},
            'T2': {k: [out['T2'][a][k], out['T2'][b][k]] for k in out['T2'][a]},
            'T3': [out['T3'][a]['aggregate'], out['T3'][b]['aggregate']],
            'T4': [out['T4'][a]['conclusion'], out['T4'][b]['conclusion']],
            'T5': [out['T5'][a]['gamma_uniform_vs_outer_matched']['status'],
                   out['T5'][b]['gamma_uniform_vs_outer_matched']['status']]}
        out['T6'][bname]['changed'] = {
            'T1': [k for k, v in out['T6'][bname]['T1'].items() if v[0] != v[1]],
            'T2': [k for k, v in out['T6'][bname]['T2'].items() if v[0] != v[1]],
            'T3': out['T6'][bname]['T3'][0] != out['T6'][bname]['T3'][1],
            'T4': out['T6'][bname]['T4'][0] != out['T6'][bname]['T4'][1],
            'T5': out['T6'][bname]['T5'][0] != out['T6'][bname]['T5'][1]}
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
        datasets = {seed: cn.fixture(seed) for seed in PILOT_SEEDS + SEEDS}
    with phase('selection_pilot'):
        selection = pilot(datasets)
    science = {'checks': checks, 'selection': selection, 'cycles_per_stage': C, 'seeds': {}, 'priors': dict(PRIORS),
               'scope': 'clamped-energy block-local learning with global settling; labels, maps, shared directions, '
                        'decorrelated sets and target means evaluator only'}
    for seed in SEEDS:
        row = {'frozen': {}, 'conditions': {}, 'comparators': {}}
        initial = lcc.digest(base.initialize(seed))
        for regime in REGIMES:
            d = datasets[seed][regime]
            with phase(f'seed{seed}_{regime}_frozen_and_comparators'):
                comps = icd.comparators(d)
                row['comparators'][regime] = {k: comps[k] for k in ('uncentered_rank_2', 'ridge_intercept')}
        for name, regime, family, lname, lam, bname, b in conditions():
            d = datasets[seed][regime]
            model = cn.model_for(family, b)
            with pst.priors(lam):
                row['frozen'][name] = pst.evaluate(model, lcc.initialize(model, seed), d, d['waking'][0], lam)
            sel = selection['selected'][name]
            if sel['failed']:
                row['conditions'][name] = {'diverged': True, 'failed_in_pilot': True}
                continue
            p = lcc.initialize(model, seed)
            assert lcc.digest({k: p[k] for k in ('W', 'V')}) == initial
            with phase(f'seed{seed}_{name}'):
                _, record = pst.run(model, p, cea.stages_for(d['waking']), sel['eta'], [C, C], 'E', lam, d,
                                    settle_check=True, label=f'seed{seed}_{name}')
            record.update(regime=regime, family=family, prior=lam, budget=b, eta=sel['eta'])
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
                  for f in (Path(__file__), Path(pst.__file__), Path(icd.__file__), Path(cn.__file__),
                            Path(lcc.__file__), Path(base.__file__), Path(cea.__file__), CONTRACT, PST_RESULTS)}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)

    def limit(name):
        p = Path('/sys/fs/cgroup') / name
        return p.read_text().strip() if p.exists() else None
    counts = dict(pst.COUNTS)
    for prefix, mod in (('icd', icd), ('cn', cn), ('ablation', cea), ('lcc', lcc), ('v1_module', base)):
        counts.update({f'{prefix}_{k}': v for k, v in mod.COUNTS.items()})
    info = {'phases': PHASES, 'counts': counts, 'training_cpu': pst.CPU, 'hashes': hashes, 'revision': revision,
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
            'storage_bytes_per_learner': {'parameters': '8P (P = 128 inner, 248 outer, 288 uniform)',
                                          'adam': '16P + 8', 'replay': 32 * 24 * 8, 'dense_q': 42 * 42 * 8},
            'timing_scope': 'whole-run/phase timings including evaluation; training CPU excludes evaluation and the '
                            'settle check; not a latency benchmark',
            'wall_seconds_before_runtime_serialization': time.perf_counter() - wall,
            'cpu_seconds_before_runtime_serialization': time.process_time() - cpu}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: science['criteria'][k] for k in ('T2', 'T4', 'T6')}, indent=1))


if __name__ == '__main__':
    main()
