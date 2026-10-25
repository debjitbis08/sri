"""Bounded algebra checks; no fits, capacity search, or learned parameters."""
import os
import time

START, CPU_START = time.perf_counter(), time.process_time()
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
import scipy

from src import abstraction_preflight as pre
from src import single_module_plasticity as v1

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'relational-composition-analysis-v1'
COUNTS = Counter()


def eig(a):
    COUNTS[f'eigvalsh_{len(a)}'] += 1
    return np.linalg.eigvalsh(a)


def solve(a, b):
    COUNTS[f'dense_solve_{len(a)}_rhs_columns_{1 if b.ndim == 1 else b.shape[1]}'] += 1
    return np.linalg.solve(a, b)


def rank(a):
    COUNTS[f'svd_{a.shape[0]}_by_{a.shape[1]}'] += 1
    return int(np.sum(np.linalg.svd(a, compute_uv=False) > 1e-9))


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def cycle(n):
    incidence = np.eye(n)-np.roll(np.eye(n), 1, axis=1)
    return incidence.T@incidence


def module_checks():
    records = []
    lateral = v1.H_BASE.copy()
    lateral[np.arange(32), np.arange(32)] -= v1.ANCHOR
    anchor = np.zeros((80, 80))
    anchor[np.arange(16), np.arange(16)] = v1.ANCHOR
    port = np.eye(80)[72:80]
    coupling = np.concatenate([port, -port], axis=1)
    for seed in (2026102504, 2026102505):
        p = pre.random_params(np.random.default_rng(seed))
        blocks = v1.block_curvatures(p)
        for retained in (1, 2):
            radial = blocks.sum(axis=0) if retained == 1 else blocks[:12].sum(axis=0)
            radial_eigs = eig(radial)
            nullity = int(np.sum(abs(radial_eigs) < 1e-9))
            assert nullity == 4*retained
            local = radial+lateral
            masked = local+anchor
            low = float(eig(masked)[0])
            assert low > 1e-10
            # Complete terminal coupling, with evidence only in the first module.
            joint = np.zeros((160, 160))
            joint[:80, :80], joint[80:, 80:] = masked, local
            joint += .1*coupling.T@coupling
            joint_eigs = eig(joint)
            assert joint_eigs[0] > 1e-10
            forcing = np.zeros((160, 16))
            forcing[np.arange(16), np.arange(16)] = v1.ANCHOR
            full = solve(joint, forcing)
            a, b, d = joint[:80, :80], joint[:80, 80:], joint[80:, 80:]
            di_bt = solve(d, b.T)
            x = solve(a-b@di_bt, forcing[:80])
            z = -di_bt@x
            discrepancy = float(np.max(abs(z-full[80:])))
            assert discrepancy < 1e-9
            assert rank(full[80:]) <= rank(b) == 8
            records.append(dict(seed=seed, retained=retained, radial_nullity=nullity,
                masked_min_eigenvalue=low, coupled_min_eigenvalue=float(joint_eigs[0]),
                coupled_condition_number=float(joint_eigs[-1]/joint_eigs[0]),
                port_cross_rank=8, output_transfer_rank=rank(full[80:]),
                schur_discrepancy=discrepancy,
                relative_solve_residual=float(np.linalg.norm(joint@full-forcing)/np.linalg.norm(forcing))))
    return records


def passive_checks():
    p = np.array([[1., 2., 0.], [0., 1., 1.]])
    q = np.array([[1., .5], [0., 1.]])
    g = np.diag([.3, .7])
    mu = .4
    delta = mu*p.T@p-mu**2*p.T@q@solve(g+mu*q.T@q, q.T@p)
    eigenvalues = eig(delta)
    assert eigenvalues[0] > -1e-12
    x = np.array([1., -2., 3.])
    z = solve(g+mu*q.T@q, mu*q.T@p@x)
    energy = .5*z@g@z+.5*mu*np.linalg.norm(p@x-q@z)**2
    assert abs(energy-.5*x@delta@x) < 1e-12
    follower = mu*p.T@p-mu**2*p.T@q@solve(mu*q.T@q, q.T@p)
    assert np.max(abs(follower)) < 1e-12
    return dict(delta_eigenvalues=eigenvalues, minimized_energy=energy,
                schur_energy=.5*x@delta@x, unconstrained_follower_max_stiffness=np.max(abs(follower)))


def relation_checks():
    records, subspaces, penalties = [], [], []
    for r in (1, 2, 3):
        t = rotation(r*np.pi/4)
        w = np.concatenate([np.linalg.matrix_power(t, j) for j in range(4)])
        subspaces.append(w)
        chain = np.zeros((6, 8))
        for j in range(3):
            chain[2*j:2*j+2, 2*j:2*j+2] = -t
            chain[2*j:2*j+2, 2*j+2:2*j+4] = np.eye(2)
        assert np.max(abs(chain@w)) < 1e-12
        penalties.append(chain.T@chain)
        closing = np.zeros((2, 8))
        closing[:, :2], closing[:, 6:] = np.eye(2), -t
        closed = np.concatenate([chain, closing])
        good_close = closing.copy()
        good_close[:, 6:] = -np.linalg.matrix_power(t, -3)
        corrected = np.concatenate([chain, good_close])
        closed_nullity = 8-rank(closed)
        assert closed_nullity == (2 if r == 2 else 0)
        assert 8-rank(corrected) == 2
        assert np.max(abs(corrected@w)) < 1e-12
        records.append(dict(relation=r, uniform_cycle_nullity=closed_nullity,
            corrected_cycle_nullity=2, open_chain_nullity=8-rank(chain),
            cycle_product=np.linalg.matrix_power(t, 4),
            uniform_closing_energy_for_unit_first_part=.5*np.linalg.norm(closed@w[:, 0])**2))
    span_rank = rank(np.concatenate(subspaces, axis=1))
    intersection_nullity = 8-rank(sum(penalties))
    assert span_rank == 6 and intersection_nullity == 0
    # Explicit invalid mixture: its first part is unit, its second has norm < 1.
    mixture = .5*(subspaces[0][:, 0]+subspaces[1][:, 0])
    assert np.linalg.norm(mixture[2:4]) < .99*np.linalg.norm(mixture[:2])
    return dict(cycles=records, combined_span_dimension=span_rank,
        positive_penalty_intersection_nullity=intersection_nullity,
        mixture_part_norms=np.linalg.norm(mixture.reshape(4, 2), axis=1))


def geometric_checks():
    u = np.kron(np.eye(4), rotation(.37))
    scalar, vector = cycle(8), np.kron(cycle(4), np.eye(2))
    scalar_commutator = np.linalg.norm(scalar@u-u@scalar)
    vector_commutator = np.linalg.norm(vector@u-u@vector)
    assert scalar_commutator > .1 and vector_commutator < 1e-12
    gi, gj = rotation(.2), rotation(-.7)
    t = rotation(.8)
    x, y = np.array([1., 2.]), np.array([-1., .5])
    original = np.linalg.norm(y-t@x)**2
    changed = np.linalg.norm(gj@y-(gj@t@gi.T)@gi@x)**2
    assert abs(original-changed) < 1e-12
    return dict(scalar_cycle_rotation_commutator_norm=scalar_commutator,
        vector_cycle_rotation_commutator_norm=vector_commutator,
        original_transport_residual_squared=original,
        changed_frame_transport_residual_squared=changed)


def encode(x):
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if isinstance(x, np.ndarray):
        return encode(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(format(x, '.12g'))
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scientific_path = args.output_dir/f'{PREFIX}.json'
    runtime_path = args.output_dir/f'{PREFIX}-runtime.json'
    if scientific_path.exists() or runtime_path.exists():
        raise FileExistsError('Refusing to overwrite analytical receipts')
    phase_wall, phase_cpu, results = {}, {}, {}
    for name, fn in [('module', module_checks), ('passive', passive_checks),
                     ('relations', relation_checks), ('geometry', geometric_checks)]:
        wall, cpu = time.perf_counter(), time.process_time()
        results[name] = fn()
        phase_wall[name], phase_cpu[name] = time.perf_counter()-wall, time.process_time()-cpu
    results.update(status='all_checks_passed', scope='finite algebra fixtures, no fitting',
                   operation_counts=dict(COUNTS), fitted_candidates=0, optimizer_steps=0)
    scientific_path.write_text(json.dumps(encode(results), indent=2, sort_keys=True)+'\n')
    files = [Path(__file__), ROOT/'docs/relational-capacity-and-composition-analysis.md',
             ROOT/'docs/computation-reporting.md', *sorted((ROOT/'src').glob('*.py'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    runtime = dict(command=[sys.executable, *sys.argv], source_hashes=hashes,
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), machine=platform.machine(),
        cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')), 'unavailable'),
        dtype='float64', numpy_configuration=config.getvalue(),
        thread_environment={k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        worker_processes=0, phase_wall_seconds=phase_wall, phase_cpu_seconds=phase_cpu,
        wall_seconds=time.perf_counter()-START, process_cpu_seconds=time.process_time()-CPU_START,
        timing_scope='whole script including imports and provenance, excluding final runtime serialization',
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_rss_unit='KiB on Linux',
        peak_rss_scope='single Python process', dense_joint_hessian_bytes=160*160*8,
        learned_state_bytes=0, optimizer_bytes=0, replay_bytes=0,
        native_workspace_bytes=None, native_workspace_reason='LAPACK internal workspace not instrumented',
        flops=None, flops_reason='primitive FLOPs not instrumented; asymptotic costs in analysis',
        selection_cost_seconds=0, selection_cost_reason='no selection or fitting performed',
        operation_counts=dict(COUNTS),
        scientific_sha256=hashlib.sha256(scientific_path.read_bytes()).hexdigest())
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=results['status'], scientific=str(scientific_path),
                         wall_seconds=runtime['wall_seconds'], counts=dict(COUNTS))))


if __name__ == '__main__':
    main()
