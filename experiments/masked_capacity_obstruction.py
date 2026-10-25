"""Post-search proof receipt, using only saved capacity parameters and scenes."""
import os
import time

START, CPU_START = time.perf_counter(), time.process_time()
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np

from experiments.locality_adaptation import encoded, environment, git, sha256
from src import abstraction_preflight as pre
from src import single_module_plasticity as v1

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-masked-capacity-v1'
A_MIN = .325
BOUND = np.sqrt(240*v1.ANCHOR*v1.LATERAL)/A_MIN
FLOOR = (1-BOUND)**2/2


def geometry():
    labels = np.array([(r, q, t) for r in (1, 2, 3) for q in range(3) for t in range(8)])
    q = np.array([1/3, 1., 3.])[labels[:, 1]]
    a = np.sqrt(2/(1+q*q))
    amplitudes = np.stack([a, q*a, a, q*a], axis=1)
    angle = (labels[:, 2, None]+labels[:, 0, None]*np.arange(4)) % 8
    clean = np.zeros((72, 4, 8))
    clean[:, :, 0], clean[:, :, 1] = amplitudes*np.cos(angle*np.pi/4), amplitudes*np.sin(angle*np.pi/4)
    return labels, clean


def laplacian():
    l = np.zeros((80, 80))
    for offset, width in zip(v1.OFFSETS, v1.WIDTHS):
        for i in range(width):
            a, b = offset+i, offset+(i+1) % width
            l[a, a] += 1
            l[b, b] += 1
            l[a, b] -= 1
            l[b, a] -= 1
    expected = v1.H_BASE.copy()
    expected[np.arange(32), np.arange(32)] -= v1.ANCHOR
    pre.require(np.max(abs(expected-v1.LATERAL*l)) < 1e-14, 'source Laplacian mismatch')
    pre.require(np.linalg.eigvalsh(l)[-1] <= 4+1e-12, 'Laplacian spectral bound')
    return l


def check_state(p, x, labels, meter):
    p = dict(phase=0, maps=[np.array(m) for m in p['maps']], v=np.array(p['v']))
    v1.check_constraints(p)
    k, l = v1.block_curvatures(p), laplacian()
    idx = v1.sector_indices(0)
    projections, groups = [], []
    for j in (2, 3):
        w = p['maps'][0][j]@p['maps'][1][j]@p['maps'][2][j]@p['v'][j]
        w = w/np.linalg.norm(w)
        projections.append(np.eye(8)-np.outer(w, w))
        groups.append(np.concatenate([indices[j] for indices in idx]))
    records = []
    for i, scene in enumerate(x):
        h = scene*pre.MASK
        pilot, _ = pre.solve_checked(p, k, np.ones((4, 4)), pre.MASK, h, meter)
        a, _ = v1.routes(v1.gates(p, pilot[None]), 'matched')
        a = a[0]
        state, _ = pre.solve_checked(p, k, a, pre.MASK, h, meter)
        pre.require(a.min() >= A_MIN-1e-12, 'gate lower bound')
        energy = pre.masked_energy(state, h, p, a, pre.MASK)
        pre.require(energy <= v1.ANCHOR+1e-9, 'energy larger than zero-state comparator')
        force = -v1.LATERAL*l@state
        pre.require(np.linalg.norm(force) <= np.sqrt(8*v1.ANCHOR*v1.LATERAL)+1e-9, 'lateral force bound')
        prediction_perp, target_perp, norms = [], [], []
        for j, proj, group in zip((2, 3), projections, groups):
            radial = sum(a[ring, j]*k[4*ring+j][np.ix_(group, group)] for ring in range(4))
            eigen, vec = np.linalg.eigh(radial)
            positive = eigen > 1e-10
            pre.require(int(positive.sum()) == 19, 'radial nullspace dimension')
            pseudo = (vec[:, positive]/eigen[positive])@vec[:, positive].T
            response = proj@pseudo[:8]
            opnorm = float(np.linalg.svd(response, compute_uv=False)[0])
            pre.require(opnorm <= np.sqrt(30)/A_MIN+1e-8, 'radial operator bound')
            yp = proj@state[idx[0][j]]
            pre.require(np.linalg.norm(yp-response@force[group]) < 1e-8, 'radial response identity')
            prediction_perp.append(yp)
            target_perp.append(proj@scene[idx[0][j]])
            norms.append(opnorm)
            meter.counts.update(radial_eigendecompositions_20=1, response_svd_8_by_20=1)
        pred_norm = float(np.linalg.norm(prediction_perp))
        target_norm = float(np.linalg.norm(target_perp))
        loss = float(np.sum((state[16:32]-scene[16:])**2)/2)
        lower = max(0., target_norm-BOUND)**2/2
        pre.require(pred_norm <= BOUND+1e-8, 'combined perpendicular output bound')
        pre.require(loss >= lower-1e-8, 'scene loss inequality')
        records.append(dict(scene=i, energy=energy, force_norm=np.linalg.norm(force),
            radial_response_operator_norms=norms, prediction_perp_norm=pred_norm,
            target_perp_norm=target_norm, actual_loss=loss, derived_scene_loss_floor=lower))
    combinations = []
    for r in (1, 2, 3):
        for q in range(3):
            chosen = np.flatnonzero((labels[:, 0] == r) & (labels[:, 1] == q))
            average = float(np.mean([records[i]['target_perp_norm']**2 for i in chosen]))
            worst = float(max(records[i]['actual_loss'] for i in chosen))
            pre.require(average >= 1-1e-9, 'target perpendicular average bound')
            pre.require(worst >= FLOOR-1e-8, 'combination worst-case lower bound')
            combinations.append(dict(relation=r, ratio_index=q, target_perp_squared_mean=average, worst_loss=worst))
    return dict(scenes=records, combinations=combinations)


def main():
    path = ROOT/'reports'/f'{PREFIX}-obstruction.json'
    runtime_path = ROOT/'reports'/f'{PREFIX}-obstruction-runtime.json'
    if path.exists() or runtime_path.exists():
        raise RuntimeError('preserve existing proof receipt')
    inputs = [Path(__file__).resolve(), ROOT/'docs/single-module-masked-capacity-obstruction.md',
              ROOT/'reports'/f'{PREFIX}-results.json', ROOT/'reports'/f'{PREFIX}-audit.json',
              ROOT/'src/abstraction_preflight.py', ROOT/'src/single_module_plasticity.py',
              ROOT/'experiments/locality_adaptation.py']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    source = json.loads(inputs[2].read_text())
    audit = json.loads(inputs[3].read_text())
    pre.require(source['status'] == 'complete', 'search must be complete')
    labels, clean = geometry()
    rendered = {w['world']: pre.render(clean, np.array(w['renderers'])) for w in audit['worlds']}
    # Numerical hand check of the anchored-chain constant, separate from model fixtures.
    chain = np.zeros((4, 4))
    for i in range(3):
        edge = np.eye(4)[i]-np.eye(4)[i+1]
        chain += A_MIN*np.outer(edge, edge)
    chain[3, 3] += A_MIN
    np.testing.assert_allclose(np.linalg.inv(chain)[0], np.array([4, 3, 2, 1])/A_MIN, atol=1e-12)
    result = dict(status='running', input_hashes=hashes, source_revision=git('rev-parse', 'HEAD'),
        scope='post-search proof audit; no optimization, new observations or task selection',
        a_min=A_MIN, perpendicular_norm_bound=BOUND, worst_scene_loss_lower_bound=FLOOR,
        classification='proved_obstruction_for_declared_masked_module', records=[])
    meter = pre.Meter()
    seen = set()
    try:
        for candidate in source['candidates']:
            if 'verification' not in candidate:
                continue
            for name, p in [('selected', candidate['verification']['parameters']), ('last', candidate['parameters_at_exit'])]:
                digest = hashlib.sha256(encoded(p).encode()).hexdigest()
                if (candidate['world'], digest) in seen:
                    continue
                seen.add((candidate['world'], digest))
                rec = dict(world=candidate['world'], start=candidate['start'], step_size=candidate['step_size'],
                           checkpoint=name, parameters_sha256=digest)
                rec.update(check_state(p, rendered[candidate['world']], labels, meter))
                result['records'].append(rec)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
    path.write_text(encoded(result))
    runtime = dict(status=result['status'], input_hashes=hashes, environment=environment(),
        command=[sys.executable, '-m', 'experiments.masked_capacity_obstruction'],
        revision=git('rev-parse', 'HEAD'), dirty_paths=git('status', '--porcelain').splitlines(),
        wall_seconds=time.perf_counter()-START, process_cpu_seconds=time.process_time()-CPU_START,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Linux KiB cumulative process high-water',
        counts=dict(meter.counts), unique_parameter_states=len(seen), new_training_updates=0,
        complexity='O(states * 72 * M^3) time for full/Schur and spectral checks; O(M^2) solver workspace plus report arrays',
        native_workspace_peak_bytes=None, unavailable_reason='native LAPACK workspace not independently instrumented',
        receipt_sha256=sha256(path))
    runtime_path.write_text(encoded(runtime))
    print(encoded(dict(status=result['status'], states=len(seen), worst_scene_loss_lower_bound=FLOOR)))
    if result['status'] != 'complete':
        print(result['failure'])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
