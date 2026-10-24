"""Constructed layered-reuse witnesses; no learning or confirmation draws."""
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
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = (32, 24, 16, 8)
OFFSETS = (0, 32, 56, 72, 80)
SLICES = tuple(slice(OFFSETS[i], OFFSETS[i+1]) for i in range(4))


def fixture():
    sites = np.arange(8)+.5
    q = np.column_stack([np.cos(np.pi*j*sites/8) for j in range(8)])
    q[:, 0] /= np.sqrt(8)
    q[:, 1:] *= np.sqrt(2/8)
    maps = [np.zeros((32, 24)), np.zeros((24, 16)), np.zeros((16, 8))]
    signal, drift, texture = (np.zeros((32, 4)) for _ in range(3))
    fullq = np.zeros((32, 32))
    for sector in range(4):
        fullq[8*sector:8*sector+8, 8*sector:8*sector+8] = q
        signal[8*sector:8*sector+8, sector] = q[:, 0]
        drift[8*sector:8*sector+8, sector] = q[:, 6]
        texture[8*sector:8*sector+8, sector] = q[:, 7]
        maps[0][8*sector:8*sector+8, 6*sector:6*sector+6] = q[:, :6]
        maps[1][6*sector:6*sector+6, 4*sector:4*sector+4] = np.eye(6, 4)
        maps[2][4*sector:4*sector+4, 2*sector:2*sector+2] = np.eye(4, 2)
    terminal = np.diag([0., 1.]*4)
    return maps, terminal, signal, drift, texture, fullq


def hessian(maps, terminal, lam, centre):
    factors = []
    anchor = np.zeros((32, 81))
    anchor[:, :32] = np.sqrt(lam)*np.eye(32)
    factors.append(anchor)
    for i, d in enumerate(maps):
        factor = np.zeros((SIZES[i], 81))
        factor[:, SLICES[i]] = np.eye(SIZES[i])
        factor[:, SLICES[i+1]] = -d
        factors.append(factor)
    penalty = np.zeros((8, 81))
    penalty[:, SLICES[3]] = terminal  # this fixture's terminal penalty is a projector
    factors.append(penalty)
    a = np.ones(8)/np.sqrt(8)
    if centre == 'summary':
        factor = np.zeros((1, 81))
        factor[0, SLICES[3]] = -a
        factor[0, -1] = 1
    else:
        factor = np.zeros((8, 81))
        factor[:, SLICES[3]] = np.eye(8)
        factor[:, -1] = -a
    factors.append(factor)
    return sum(f.T@f for f in factors)


def independent_energy(x, maps, terminal, lam, centre):
    value = lam*np.dot(x[:32], x[:32])
    for i, d in enumerate(maps):
        residual = x[SLICES[i]]-d@x[SLICES[i+1]]
        value += np.dot(residual, residual)
    last = x[SLICES[3]]
    value += last@terminal@last
    a = np.ones(8)/np.sqrt(8)
    if centre == 'summary':
        value += (x[-1]-a@last)**2
    else:
        residual = last-a*x[-1]
        value += residual@residual
    return value/2


def solve_map(h, lam, work):
    rhs = np.zeros((81, 32))
    rhs[:32] = lam*np.eye(32)
    work['dense_solves_81_rhs32'] += 1
    return np.linalg.solve(h, rhs)


def checks():
    maps, penalty, u, w, v, q = fixture()
    work = Counter()
    s, noise = u@u.T, 4*v@v.T+.25*np.eye(32)
    result = []
    for lam in (1., .1):
        for centre in ('summary', 'original_spring'):
            h = hessian(maps, penalty, lam, centre)
            # Independent quadratic polarization, using a separate scalar energy.
            eye = np.eye(81)
            energies = np.array([independent_energy(e, maps, penalty, lam, centre) for e in eye])
            independent = np.diag(2*energies)
            for i in range(81):
                for j in range(i):
                    independent[i, j] = independent[j, i] = (
                        independent_energy(eye[i]+eye[j], maps, penalty, lam, centre)
                        - energies[i]-energies[j])
            work['scalar_energy_evaluations'] += 81+81*80//2
            assert np.max(np.abs(h-independent)) < 1e-12
            eig = np.linalg.eigvalsh(h)
            work['eigensolves_81'] += 1
            assert eig[0] > 0
            transfer = solve_map(h, lam, work)
            # Directly eliminate all interior variables, independently of full solve.
            schur = h[:32, :32]-h[:32, 32:]@np.linalg.solve(h[32:, 32:], h[32:, :32])
            direct = np.linalg.solve(schur, lam*np.eye(32))
            work['dense_solves_49_rhs32'] += 1
            work['dense_solves_32_rhs32'] += 1
            assert np.max(np.abs(transfer[:32]-direct)) < 1e-12
            g = transfer[:32]
            residual = g-np.eye(32)
            risk = (np.trace(residual@s@residual.T)+np.trace(g@noise@g.T))/4
            error_map = np.column_stack((residual@u, 2*g@v, .5*g))
            assert abs(risk-np.linalg.norm(error_map, 'fro')**2/4) < 1e-12
            work['independent_error_map_checks'] += 1
            signal_error = np.linalg.norm(residual@u, 'fro')**2/4
            if centre == 'summary':
                js = [0, 4, 3, 3, 2, 2, 1, 1]*4
                gains = np.array([1 if j == 0 else lam*j/(1+lam*j) for j in js])
                assert np.max(np.abs(g-q@np.diag(gains)@q.T)) < 1e-12
                assert np.max(np.abs(g@u-u)) < 1e-12
                # This different readout is explicitly a control, not the outer state.
                terminal_signal = np.eye(8)-penalty
                decoded = maps[0]@maps[1]@maps[2]@terminal_signal@transfer[SLICES[3]]
                assert np.max(np.abs(decoded-s)) < 1e-12
            mobility = np.repeat([1., .5, .1, .01, .01], [32, 24, 16, 8, 1])
            scaled = np.sqrt(mobility)[:, None]*h*np.sqrt(mobility)[None, :]
            rates = np.linalg.eigvalsh(scaled)
            work['eigensolves_81'] += 1
            assert rates[0] > 0
            result.append(dict(lambda_outer=lam, centre=centre,
                min_hessian_eigenvalue=float(eig[0]), condition_number=float(eig[-1]/eig[0]),
                exterior_signal_error=float(signal_error), exterior_reconstruction_E=float(risk),
                schur_error=float(np.max(np.abs(transfer[:32]-direct))),
                uniform_relaxation_slowest_rate=float(eig[0]),
                graded_relaxation_slowest_rate=float(rates[0]),
                signal_mode_gains=np.linalg.eigvalsh(u.T@g@u).tolist()))

    base_h = hessian(maps, penalty, .1, 'summary')
    base_transfer = solve_map(base_h, .1, work)
    transports = []
    for degrees in (0, 30, 90):
        theta = np.deg2rad(degrees)
        rotation = np.eye(32)+(np.cos(theta)-1)*(u@u.T+w@w.T)+np.sin(theta)*(w@u.T-u@w.T)
        assert np.max(np.abs(rotation.T@rotation-np.eye(32))) < 1e-12
        rotated_maps = [rotation@maps[0], *maps[1:]]
        transport = solve_map(hessian(rotated_maps, penalty, .1, 'summary'), .1, work)
        expected = base_transfer.copy()
        expected[:32] = rotation@expected[:32]
        equivariance_error = float(np.max(np.abs(transport@rotation-expected)))
        assert equivariance_error < 1e-12
        current = rotation@u
        source_loss = 1-np.trace(current@current.T@s)/4
        assert abs(source_loss-np.sin(theta)**2) < 1e-12
        union_rank = int(np.linalg.matrix_rank(np.column_stack((u, current)), tol=1e-10))
        assert union_rank == (4 if degrees == 0 else 8)
        transports.append(dict(degrees=degrees, all_state_transport_error=equivariance_error,
                               target_subspace_source_loss=float(source_loss), source_target_union_rank=union_rank))
    rank3_loss = 1-np.trace(u[:, :3]@u[:, :3].T@s)/4
    assert abs(rank3_loss-.25) < 1e-12
    # An arbitrary latent rotation is observationally invisible to paired moments.
    o = np.diag([-1., 1., 1., 1.])  # even a within-sector sign is unidentifiable
    assert np.max(np.abs((u@o)@(u@o).T-s)) < 1e-12

    ema = []
    for eta in (.5, .1, .01):
        mean, variance = 0., 0.
        for t in range(1, 10001):
            mean = (1-eta)*mean+eta*.001*t
            variance = (1-eta)**2*variance+eta**2
        lag = mean-.001*10000
        assert abs(lag+(1-eta)*.001/eta) < 1e-10
        assert abs(variance-eta/(2-eta)) < 1e-12
        ema.append(dict(eta=eta, steady_drift_lag=lag, stationary_noise_variance=variance,
                        illustrative_mse=lag**2+variance))
    work.update(ema_moment_steps=30000, fitted_models=0, sampled_examples=0,
                tuning_trials=0, candidate_selections=0, iterative_inference_steps=0,
                rank_checks=3)
    return dict(status='complete', scope='constructed analytical checks only',
                equilibrium=result, transport=transports, ema=ema,
                rank3_unit_signal_loss=rank3_loss, work=dict(work))


def diagram():
    # Code-native geometry, not a generated bitmap or an empirical figure.
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="660" viewBox="0 0 1000 660">',
             '<rect width="1000" height="660" fill="#fff"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172b4d}.title{font-size:23px;font-weight:bold}.label{font-size:17px}.small{font-size:14px}</style>',
             '<text x="30" y="36" class="title">Circular layered module with learned reciprocal maps</text>',
             '<text x="30" y="61" class="small">Analytical candidate: geometry retained; coupling and centre rules explicitly revised</text>']
    colours = ['#127c9c', '#238b69', '#7562a8', '#ae6343']
    for idx, (radius, count, colour) in enumerate(zip((240, 180, 120, 60), SIZES, colours)):
        parts.append(f'<circle cx="300" cy="340" r="{radius}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for j in range(count):
            angle = 2*np.pi*j/count
            x, y = 300+radius*np.cos(angle), 340+radius*np.sin(angle)
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{colour}"/>')
        y = 140+idx*90
        label = ['32 outer units: observed + reconstructed interface',
                 '24 units: learned local coordinates',
                 '16 units: narrower retained coordinates',
                 '8 inner units: capacity for persistent modes'][idx]
        parts.append(f'<text x="570" y="{y}" class="label">{label}</text>')
    for left, right in ((60,120), (120,180), (180,240)):
        parts.append(f'<path d="M {300+left+8} 340 H {300+right-8}" stroke="#172b4d" stroke-width="2"/>')
        parts.append(f'<text x="{300+(left+right)/2-8}" y="332" class="label">↔</text>')
    parts += ['<circle cx="300" cy="340" r="7" fill="#172b4d"/>',
              '<text x="570" y="442" class="small">Centre: scalar summary, not a mandatory bottleneck</text>',
              '<text x="570" y="490" class="label">Persistent learning: fast outside, slower inward</text>',
              '<text x="570" y="516" class="small">Hypothesis; rates do not guarantee useful retention.</text>',
              '<text x="570" y="552" class="label">Inference: all adjacent rings influence each other</text>',
              '<text x="570" y="578" class="small">Local maps are shared in both directions.</text>',
              '<text x="30" y="625" class="small">Input h stays external. Output is the settled outer field x₀. Links shown schematically, not one-to-one wires.</text>', '</svg>']
    return '\n'.join(parts)+'\n'


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    phase_wall, phase_cpu = wall, cpu
    first = checks()
    first_times = dict(wall_seconds=time.perf_counter()-phase_wall, cpu_seconds=time.process_time()-phase_cpu)
    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
    second = checks()
    second_times = dict(wall_seconds=time.perf_counter()-phase_wall, cpu_seconds=time.process_time()-phase_cpu)
    scientific = json.dumps(first, sort_keys=True, indent=2)+'\n'
    assert scientific == json.dumps(second, sort_keys=True, indent=2)+'\n'
    (ROOT/'reports/layered-reuse-analysis.json').write_text(scientific)
    (ROOT/'reports/layered-reuse-architecture.svg').write_text(diagram())
    sources = ['experiments/layered_reuse_analysis.py', 'docs/layered-reuse-analysis.md',
               'docs/computation-reporting.md', 'docs/layered-module-v1-contract.md',
               'docs/layered-module-v2-results.md', 'docs/layered-module-design-analysis.md',
               'docs/representation-reuse-v1-results.md', 'docs/research-program.md']
    runtime = dict(scope='two complete analytical passes and artifact writes; excludes imports and receipt write',
        wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
        phases=dict(first_pass=first_times, reproducibility_pass=second_times), byte_identical=True,
        work_per_pass=first['work'], total_work={k:2*v for k,v in first['work'].items()},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='whole-process Linux KiB including imports; not per configuration',
        python=sys.version, numpy=np.__version__, os=platform.platform(),
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), 'unknown'),
        numpy_build=np.__config__.show(mode='dicts'), dtype='float64', workers=1,
        threads={key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        command='.venv/bin/python -m experiments.layered_reuse_analysis', seed_policy='none; explicit matrices',
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_paths=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        scientific_sha256=hashlib.sha256(scientific.encode()).hexdigest(),
        flops=None, flops_reason='not measured; complexity in analysis document',
        temporary_bytes=None, temporary_bytes_reason='RSS only; no isolated allocation tracking')
    (ROOT/'reports/layered-reuse-analysis-runtime.json').write_text(json.dumps(runtime,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(equilibrium=first['equilibrium'],transport=first['transport'],ema=first['ema'],
                         wall_seconds=runtime['wall_seconds'],byte_identical=True),indent=2))


if __name__ == '__main__':
    main()
