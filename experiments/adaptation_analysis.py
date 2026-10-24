"""Deterministic locality and transfer witnesses; no fitting or confirmation data.

Run: .venv/bin/python -m experiments.adaptation_analysis
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def analyze():
    d, k, b = 32, 4, 8
    aligned = np.zeros((d, k))
    for j in range(k):
        aligned[j*b:(j+1)*b, j] = 1/np.sqrt(b)
    hadamard = np.ones((1, 1))
    while len(hadamard) < d:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    worlds = {'aligned': aligned, 'shifted': np.roll(aligned, 4, axis=0),
              'dense_witness': hadamard[:, :k]/np.sqrt(d)}
    locality = {}
    eigensolves = 0
    for name, u in worlds.items():
        assert np.allclose(u.T@u, np.eye(k), atol=1e-12)
        s = u@u.T
        losses = []
        for phase in range(b):
            retained = 0.
            for j in range(k):
                sites = (np.arange(j*b, (j+1)*b)+phase) % d
                retained += np.linalg.eigvalsh(s[np.ix_(sites, sites)])[-1]
                eigensolves += 1
            losses.append(float(1-retained/k))
        locality[name] = {'phase_losses': losses, 'best_phase_loss': min(losses)}
    assert abs(locality['aligned']['phase_losses'][0]) < 1e-12
    assert abs(locality['shifted']['phase_losses'][0]-.5) < 1e-12
    assert abs(locality['shifted']['best_phase_loss']) < 1e-12
    assert np.allclose(locality['dense_witness']['phase_losses'], .75)

    # Independent fourth-moment sum for the Gaussian bilinear score variance.
    t = np.diag([1., 2., 3., 4.])
    s = np.diag([.2, .4, .6, .8])
    q = np.array([[1., 0.], [1., 1.], [0., 1.], [1., -1.]])
    q, _ = np.linalg.qr(q)
    p = q@q.T
    wick = sum(p[i,j]*p[l,m]*(t[i,l]*t[j,m]+s[i,m]*s[l,j])
               for i in range(4) for j in range(4)
               for l in range(4) for m in range(4))
    trace_variance = np.trace(p@t@p@t)+np.trace(p@s@p@s)
    assert abs(wick-trace_variance) < 1e-12

    # Surface risk from a full linear error map versus the principal-angle law.
    p = np.diag([1.]*k+[0.]*(d-k))
    transfers = []
    for noise in (.01, .25):
        for degrees in (0, 10, 30, 60, 90):
            angle = np.deg2rad(degrees)
            u = np.zeros((d, k))
            u[:k] = np.cos(angle)*np.eye(k)
            u[k:2*k] = np.sin(angle)*np.eye(k)
            error_map = np.column_stack(((p-np.eye(d))@u, np.sqrt(noise)*p))
            risk = np.linalg.norm(error_map, 'fro')**2
            predicted = k*np.sin(angle)**2+k*noise
            assert abs(risk-predicted) < 1e-12
            transfers.append(dict(noise_variance=noise, degrees=degrees,
                                  projected_risk=float(risk), raw_risk=d*noise,
                                  signal_loss=float(k*np.sin(angle)**2)))

    # Full covariance of [old clean, new clean, old noise, new noise].
    blends = []
    for rho in (0., .5, 1.):
        for a in (0., .25, .75, 1.):
            tau2, w, v = 1., .2, .5
            cov = np.diag([tau2, tau2, w, v])
            cov[0,1] = cov[1,0] = rho*tau2
            old_error = np.array([a, -a, a, 1-a])
            zero_error = np.array([0., -a, 0., 1-a])
            old_risk = old_error@cov@old_error
            zero_risk = zero_error@cov@zero_error
            difference = a*a*((1-2*rho)*tau2+w)
            assert abs(old_risk-zero_risk-difference) < 1e-12
            blends.append(dict(rho=rho, old_weight=a, old_risk=float(old_risk),
                               zero_shrinkage_risk=float(zero_risk)))
    return dict(status='complete', scope='constructed analytical witnesses; no training',
                locality=locality, gaussian_score_variance=float(trace_variance),
                transfer=transfers, activation_reuse=blends,
                low_noise_boundary_degrees=float(np.rad2deg(np.arcsin(np.sqrt(.07)))),
                work=dict(block_eigendecompositions=eigensolves, block_dimension=b,
                          gaussian_fourth_moment_terms=4**4, transfer_cases=len(transfers),
                          activation_cases=len(blends), fitted_models=0,
                          sampled_examples=0, tuning_trials=0),
                checks='all assertions passed')


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    result = analyze()
    root = Path(__file__).resolve().parents[1]
    outputs = root/'reports'
    scientific = outputs/'adaptation-analysis.json'
    scientific.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    sources = ['experiments/adaptation_analysis.py', 'docs/locality-adaptation-analysis.md',
               'docs/broad-structure-transfer-analysis.md', 'docs/computation-reporting.md']
    runtime = dict(
        scope='single-process analytical calculation and scientific JSON write; excludes imports',
        wall_seconds=time.perf_counter()-wall, process_cpu_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope='whole process high-water RSS, including imports; Linux KiB',
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        cpu=platform.processor() or 'unspecified; see platform', dtype='float64', workers=1,
        thread_environment={name: os.environ[name] for name in
                            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        numpy_build=np.__config__.show(mode='dicts'),
        revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
        worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'], cwd=root)),
        command='.venv/bin/python -m experiments.adaptation_analysis',
        seed_policy='none; explicit matrices only',
        sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        work=result['work'])
    (outputs/'adaptation-analysis-runtime.json').write_text(
        json.dumps(runtime, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'checks': result['checks'], 'work': result['work'],
                      'wall_seconds': runtime['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
