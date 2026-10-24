"""Analytical checks and retrospective v2 diagnostics; never confirmation."""
import os

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from src import locality_adaptation as v1
from src.locality_adaptation_v2 import ROOT_SEED, folds, score_sum

ROOT = Path(__file__).resolve().parents[1]
INPUT = 'reports/locality-adaptation-v2-results.json'


def moments(a, s, noise):
    """Unnormalized single-pair score-difference variances, fixed A."""
    t = s + noise
    signal = np.trace(a @ t @ a @ t) + np.trace(a @ s @ a @ s)
    risk = .5*np.trace(a @ (t+s) @ a @ (t+s)) + 4.5*np.trace(a @ noise @ a @ noise)
    return float(signal), float(risk)


def wick_variance(h, covariance):
    """Explicit centered Gaussian fourth moment, independent of trace formula."""
    size = len(h)
    return sum(h[i, j]*h[l, m]*(covariance[i, l]*covariance[j, m]
               + covariance[i, m]*covariance[j, l])
               for i in range(size) for j in range(size)
               for l in range(size) for m in range(size))


def identities():
    z = np.array([[1., 2.], [0., 1.], [2., -1.], [1., 0.]])
    q, _ = np.linalg.qr(z)
    p = q @ q.T
    q2, _ = np.linalg.qr(np.roll(z, 1, axis=0))
    other = q2 @ q2.T
    s = z @ z.T / 8
    c = np.array([[1., .2, 0., .1], [0., 1., .3, 0.],
                  [.2, 0., 1., .4], [0., .1, 0., 1.]])
    noise = c @ c.T
    t = s + noise
    cov = np.block([[t, s], [s, t]])
    records = []
    for label, a in [('projector', p), ('difference', p-other), ('zero', p-p)]:
        zero = np.zeros_like(a)
        hs = np.block([[zero, a/2], [a/2, zero]])
        hr = np.block([[a/2, -a], [-a, a/2]])
        vs, vr = moments(a, s, noise)
        checks = [abs(wick_variance(hs, cov)-vs), abs(wick_variance(hr, cov)-vr),
                  abs(np.trace(hs@cov)-np.trace(a@s)),
                  abs(np.trace(hr@cov)-np.trace(a@(noise-s))),
                  abs(vr-vs-4*np.trace(a@noise@a@noise))]
        assert max(checks) < 1e-10
        assert vr >= vs-1e-10
        records.append(dict(case=label, signal_variance=vs, risk_variance=vr,
                            max_identity_error=max(checks)))
    return records


def explore(artifact, meter):
    lookup = {(r['world'], r['budget'], r['arm']): r for r in artifact['rows']
              if r['regime'] == 'aligned'}
    rows = []
    old_seed = v1.ROOT_SEED
    try:
        v1.ROOT_SEED = ROOT_SEED
        for world_index in range(64):
            world = v1.make_world(1, world_index)
            xfull, yfull = v1.views(world, 'aligned', v1.base_stream(1, world_index))
            s = v1.signal_covariance(world, 'aligned')
            noise = v1.noise_covariance(world, 'aligned')
            for n in (16, 32):
                x, y = xfull[:n], yfull[:n]
                fold_l, fold_e, total = np.zeros(8), np.zeros(8), np.zeros(8)
                fold_variances = []
                for f in range(4):
                    train, held = folds(n) != f, folds(n) == f
                    projectors = []
                    for phase in range(8):
                        weights = v1.fit_local(x[train], y[train], phase, 'cca', meter=meter)
                        with meter.time('validation'):
                            total[phase] += score_sum(x[held], y[held], phase, weights)
                        meter.counts['candidate_scores'] += 1
                        meter.counts['validation_row_exposures'] += int(held.sum())
                        basis = v1.local_basis(weights, phase, 32)
                        projectors.append(basis @ basis.T)
                        fold_l[phase] += v1.signal_loss(basis, s)/4
                        fold_e[phase] += v1.nmse(basis, s, noise)/4
                    # Hypothetical fresh held-out pairs, conditional on this fold.
                    # NOT an estimate of CV variance (cross-fold covariance omitted).
                    fold_variances.extend(moments(projectors[p]-projectors[0], s, noise)
                                          for p in range(1, 8))
                actual = lookup[world_index, n, 'cv_refit_cca']
                assert np.allclose(total/n, actual['selection_scores'], atol=1e-10, rtol=1e-10)
                selected = v1.select(total/n)
                assert selected == actual['phase']
                all_n = lookup[world_index, n, 'evaluator_all_n_candidates']
                loss = np.asarray(all_n['signal_losses'])
                risk = loss + np.asarray(all_n['projected_noise'])
                assert abs(loss[selected]-actual['signal_loss']) < 1e-10
                assert abs(risk[selected]-actual['nmse']['in_distribution']) < 1e-10
                fold_best = v1.select(-fold_l)
                fold_risk_best = v1.select(-fold_e)
                best = v1.select(-loss)
                best_risk = v1.select(-risk)
                # Exact signed decomposition: the first term need not be positive.
                sampling = loss[selected]-loss[fold_best]
                mismatch = loss[fold_best]-loss[best]
                assert abs(sampling+mismatch-(loss[selected]-loss[best])) < 1e-12
                rows.append(dict(world=world_index, budget=n, selected=selected,
                    oracle_fold_signal_phase=fold_best, oracle_fold_risk_phase=fold_risk_best,
                    oracle_all_n_signal_phase=best, oracle_all_n_risk_phase=best_risk,
                    mean_fold_signal_losses=fold_l.tolist(), mean_fold_risks=fold_e.tolist(),
                    all_n_signal_losses=loss.tolist(), all_n_risks=risk.tolist(),
                    signed_selection_L=float(sampling), refit_mismatch_L=float(mismatch),
                    actual_excess_L=float(loss[selected]-loss[0]),
                    actual_excess_E=float(risk[selected]-risk[0]),
                    oracle_fold_signal_excess_E=float(risk[fold_best]-risk[0]),
                    oracle_fold_risk_excess_E=float(risk[fold_risk_best]-risk[0]),
                    fresh_pair_signal_variance_mean=float(np.mean(np.array(fold_variances)[:, 0])),
                    fresh_pair_risk_variance_mean=float(np.mean(np.array(fold_variances)[:, 1]))))
    finally:
        v1.ROOT_SEED = old_seed
    summary = {}
    for n in (16, 32):
        subset = [r for r in rows if r['budget'] == n]
        means = {key: float(np.mean([r[key] for r in subset])) for key in (
            'signed_selection_L', 'refit_mismatch_L', 'actual_excess_L', 'actual_excess_E',
            'oracle_fold_signal_excess_E', 'oracle_fold_risk_excess_E',
            'fresh_pair_signal_variance_mean', 'fresh_pair_risk_variance_mean')}
        means.update(actual_vs_fold_disagreements=sum(r['selected'] != r['oracle_fold_signal_phase'] for r in subset),
                     fold_vs_refit_disagreements=sum(r['oracle_fold_signal_phase'] != r['oracle_all_n_signal_phase'] for r in subset),
                     signal_vs_risk_fold_disagreements=sum(r['oracle_fold_signal_phase'] != r['oracle_fold_risk_phase'] for r in subset))
        summary[str(n)] = means
    return dict(summary=summary, rows=rows)


def main():
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    phases = {}
    def timed(name, fn):
        wall, cpu = time.perf_counter(), time.process_time()
        out = fn()
        phases[name] = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
        return out
    artifact = timed('load_historical_artifact', lambda: json.loads((ROOT/INPUT).read_text()))
    assert artifact['status'] == 'complete' and artifact['root_seed'] == ROOT_SEED
    meter = v1.Meter()
    result = dict(status='complete', scope='analytical identities and retrospective aligned v2 diagnostics; no new confirmation',
                  identities=timed('analytical_checks', identities),
                  exploratory=timed('replay_diagnostics', lambda: explore(artifact, meter)))
    result['work'] = dict(meter.counts, world_streams_replayed=64, distinct_pairs_used=2048,
        generated_pairs=16384, all_n_fits_reused=1024, new_all_n_fits=0, tuning_trials=0,
        optimizer_steps=0, inference_calls=0, gaussian_fourth_moment_terms=6*8**4)
    output = ROOT/'reports/phase-selection-analysis.json'
    timed('scientific_write', lambda: output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n'))
    sources = ['experiments/phase_selection_analysis.py', 'docs/phase-selection-analysis.md',
               'docs/computation-reporting.md', 'src/locality_adaptation.py',
               'src/locality_adaptation_v2.py', INPUT, 'reports/locality-adaptation-v2-manifest.json']
    runtime = dict(scope='one process, excludes imports and runtime receipt construction/write',
        phases=phases, wall_seconds=time.perf_counter()-start_wall, cpu_seconds=time.process_time()-start_cpu,
        fitting_stage_wall_seconds=dict(meter.seconds), work=result['work'],
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Linux whole-process high-water mark including imports and historical JSON; not per fit',
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, os=platform.platform(),
        cpu=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), 'unknown'),
        dtype='float64', workers=1, numpy_build=np.__config__.show(mode='dicts'),
        threads={key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_paths=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines(),
        sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        seed_policy='replay v2 root 2026102408 stage 1 worlds 0..63; no new random worlds',
        command='.venv/bin/python -m experiments.phase_selection_analysis',
        flops=None, flops_unavailable_reason='not measured; theoretical orders in study document',
        peak_temporary_bytes=None, peak_temporary_unavailable_reason='RSS only; includes Python/BLAS, no isolated allocation measurement')
    (ROOT/'reports/phase-selection-analysis-runtime.json').write_text(json.dumps(runtime, sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(summary=result['exploratory']['summary'], work=result['work'],
                          wall_seconds=runtime['wall_seconds']), indent=2))


if __name__ == '__main__':
    main()
