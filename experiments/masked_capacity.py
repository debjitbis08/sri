"""Frozen task audit and bounded, privileged masked-module capacity search."""
import os
import time

START, CPU_START = time.perf_counter(), time.process_time()
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import torch

from experiments.locality_adaptation import canonical, encoded, environment, git, sha256
from src import abstraction_preflight as pre
from src import masked_capacity as c
from src import single_module_plasticity as v1

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-masked-capacity-v1'
SOURCES = (f'docs/{PREFIX}-contract.md', 'src/masked_capacity.py',
           'experiments/masked_capacity.py', 'tests/test_masked_capacity.py', 'requirements.txt')
PARENT_MANIFEST = 'reports/single-module-abstraction-preflight-v1-manifest.json'
TESTS = ('tests.test_masked_capacity', 'tests.test_abstraction_preflight')


def machine():
    return dict(environment(), torch=torch.__version__, torch_build=torch.__config__.show(),
                torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def dependencies():
    m = json.loads((ROOT/PARENT_MANIFEST).read_text())
    deps = dict(m['sources'], **m['dependencies'])
    deps[PARENT_MANIFEST] = sha256(ROOT/PARENT_MANIFEST)
    for name, digest in deps.items():
        if sha256(ROOT/name) != digest:
            raise RuntimeError(f'frozen dependency changed: {name}')
    return deps


def freeze(out, runtime):
    path = out/f'{PREFIX}-manifest.json'
    if path.exists():
        raise RuntimeError('no silent manifest overwrite')
    manifest = dict(status='frozen_before_tests_audits_and_fits', sources={s: sha256(ROOT/s) for s in SOURCES},
        dependencies=dependencies(), revision=git('rev-parse', 'HEAD'), environment=machine(),
        root_seed=c.ROOT_SEED, worlds=[0, 1], starts=[0, 1], step_sizes=c.RATES,
        maximum_updates_per_candidate=1000, maximum_total_updates=8000, checkpoints=c.CHECKPOINTS,
        task=dict(scenes=72, train_scenes=48, held_out_scenes=24, ratios=c.RATIOS,
                  fit_scope='capacity uses all 72 targets, including nominal holdout'),
        tests=TESTS, setup_runtime_sha256=sha256(out/f'{PREFIX}-setup-runtime.json'),
        repeat_scope='two task audits and two serialized-candidate verifications; one search trajectory')
    path.write_text(encoded(manifest))
    runtime.update(status='complete', manifest_sha256=sha256(path))


def frozen(out):
    path = out/f'{PREFIX}-manifest.json'
    m = json.loads(path.read_text())
    dependencies()
    for name, digest in dict(m['sources'], **m['dependencies']).items():
        if sha256(ROOT/name) != digest:
            raise RuntimeError(f'manifest mismatch: {name}')
        if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != (ROOT/name).read_bytes():
            raise RuntimeError(f'frozen input uncommitted: {name}')
    if subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT) != path.read_bytes():
        raise RuntimeError('manifest uncommitted')
    return sha256(path)


def tests(out, runtime):
    digest = frozen(out)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    command = [sys.executable, '-m', 'unittest', *TESTS, '-v']
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime.update(status='complete' if proc.returncode == 0 else 'failed', manifest_sha256=digest,
        test_command=command, test_output=proc.stdout+proc.stderr, returncode=proc.returncode,
        subprocess_wall_seconds=time.perf_counter()-start,
        subprocess_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        subprocess_peak_rss_kib=after.ru_maxrss,
        operation_counts=None, operation_counts_reason='test fixture operations not instrumented; entire test cost timed')
    print(proc.stdout+proc.stderr, flush=True)


def require_previous(out, phase, digest):
    old = json.loads((out/f'{PREFIX}-runtime.json').read_text())
    if old.get(phase, {}).get('status') != 'complete' or old[phase]['manifest_sha256'] != digest:
        raise RuntimeError(f'successful {phase} for this manifest required')


def audit(out, runtime):
    digest = frozen(out)
    require_previous(out, 'tests', digest)
    copies = []
    timings = []
    for repeat in range(2):
        meter = pre.Meter()
        start, cpu = time.perf_counter(), time.process_time()
        labels, angles, clean, held = c.task()
        result = dict(status='running', manifest_sha256=digest, labels=labels, angles=angles,
            train_indices=np.flatnonzero(~held), held_out_indices=np.flatnonzero(held),
            factor_counts={name: {f'{r},{q}': int(np.sum(pick & (labels[:, 0] == r) & (labels[:, 1] == q)))
                          for r in (1, 2, 3) for q in range(3)}
                          for name, pick in [('train', ~held), ('held_out', held)]}, worlds=[])
        try:
            for world in (0, 1):
                result['worlds'].append(c.audit_world(world, meter))
            result.update(status='complete', narrow_split_gate=all(w['narrow_split_gate'] for w in result['worlds']))
        except Exception as error:
            result.update(status='incomplete', failure=str(error))
            (out/f'{PREFIX}-audit.json').write_text(encoded(result))
            raise
        finally:
            timings.append(dict(repeat=repeat, wall_seconds=time.perf_counter()-start,
                                 process_cpu_seconds=time.process_time()-cpu, counts=dict(meter.counts)))
            runtime['repeats'] = timings
        copies.append(encoded(result))
    pre.require(copies[0] == copies[1], 'task audit not byte-identical')
    path = out/f'{PREFIX}-audit.json'
    path.write_text(copies[0])
    runtime.update(status='complete', manifest_sha256=digest, audit_sha256=sha256(path), byte_identical=True)


def optimizer_bytes(optimizer):
    entries = []
    total = 0
    for i, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value):
                size = value.numel()*value.element_size()
                total += size
                entries.append(dict(parameter_tensor=i, field=key, dtype=str(value.dtype), bytes=size))
    return dict(total_bytes=total, entries=entries)


def verify(saved, x, matrices, counts):
    p = c.as_numpy_parameters(canonical(saved))
    v1.check_constraints(p)
    records = []
    for _ in range(2):
        prediction, checks = c.numpy_prediction(p, x, counts)
        with torch.no_grad():
            model = c.MaskedModule(p, counts)
            tensor = torch.tensor(x*pre.MASK, dtype=torch.float64)
            expected = model(tensor).numpy()[:, 16:]
        difference = float(np.max(np.abs(prediction-expected)))
        pre.require(difference <= 1e-9, 'serialized Torch/NumPy mismatch')
        records.append(dict(prediction=c.prediction_record(prediction, x, matrices),
                            checks=checks, torch_numpy_max_abs=difference))
    pre.require(encoded(records[0]) == encoded(records[1]), 'saved candidate verification not byte-identical')
    return dict(parameters=p, byte_identical=True, **records[0])


def fit_candidate(record, timed, x, matrices):
    meter = pre.Meter()
    model = c.MaskedModule(c.initial_parameters(record['world'], record['start']), meter)
    optimizer = torch.optim.Adam(model.parameters(), lr=record['step_size'],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0.)
    observed = torch.tensor(x*pre.MASK, dtype=torch.float64)
    target = torch.tensor(x[:, 16:], dtype=torch.float64)
    best, best_parameters, verified_step = None, None, None
    start, cpu = time.perf_counter(), time.process_time()
    record['checkpoints'], timed['checkpoint_costs'] = [], []
    try:
        for step in range(1001):
            record['last_completed_updates'] = step
            if step in c.CHECKPOINTS:
                with meter.phase('checkpoint_inference'):
                    with torch.no_grad():
                        pred = model(observed).numpy()[:, 16:]
                    summary = c.prediction_record(pred, x, matrices)
                record['checkpoints'].append(dict(step=step, **{k: summary[k] for k in ('all', 'train', 'held_out')}))
                timed['checkpoint_costs'].append(dict(step=step, wall_seconds=time.perf_counter()-start,
                    process_cpu_seconds=time.process_time()-cpu, counts=dict(meter.counts)))
                if best is None or summary['all']['worst_loss'] < best['all']['worst_loss']-1e-12:
                    best = dict(step=step, **summary)
                    best_parameters = model.numpy_parameters()
                print(f"world={record['world']} start={record['start']} lr={record['step_size']} "
                      f"step={step} mean={summary['all']['mean_loss']:.6f} worst={summary['all']['worst_loss']:.6f}", flush=True)
                if summary['all']['all_pass']:
                    with meter.phase('independent_verification'):
                        checked = verify(model.numpy_parameters(), x, matrices, meter)
                    pre.require(checked['prediction']['all']['all_pass'], 'witness lost after serialization')
                    best, best_parameters = dict(step=step, **summary), model.numpy_parameters()
                    record['verification'] = checked
                    verified_step = step
                    break
            if step == 1000:
                break
            with meter.phase('optimization'):
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(observed, target)
                pre.require(bool(torch.isfinite(loss)), 'nonfinite capacity loss')
                loss.backward()
                meter.counts['backward_passes'] += 1
                model.project_gradients()
                optimizer.step()
                model.retract()
                meter.counts.update(optimizer_updates=1, fitting_scene_exposures=72)
        record.update(best_step=best['step'], best_checkpoint=best,
                      parameters_at_exit=model.numpy_parameters())
        if verified_step is None:
            with meter.phase('independent_verification'):
                record['verification'] = verify(best_parameters, x, matrices, meter)
        record['is_witness'] = record['verification']['prediction']['all']['all_pass']
        record['status'] = 'witness' if record['is_witness'] else 'budget_exhausted'
    except Exception as error:
        record.update(status='failed', error=str(error), parameters_at_exit=model.numpy_parameters())
        raise
    finally:
        timed.update(wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
            counts=dict(meter.counts), phase_wall_seconds=dict(meter.seconds), phase_cpu_seconds=dict(meter.cpu_seconds),
            optimizer_storage=optimizer_bytes(optimizer),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def search(out, runtime):
    digest = frozen(out)
    require_previous(out, 'audit', digest)
    audits = json.loads((out/f'{PREFIX}-audit.json').read_text())
    result = dict(status='running', manifest_sha256=digest, audit_sha256=sha256(out/f'{PREFIX}-audit.json'),
                  narrow_split_gate=audits['narrow_split_gate'], worlds=[], candidates=[])
    runtime['candidates'] = []
    path = out/f'{PREFIX}-results.json'
    _, _, clean, _ = c.task()
    try:
        for world in (0, 1):
            matrices = c.rendering(world)
            x = pre.render(clean, matrices)
            found = False
            for start in (0, 1):
                for rate in c.RATES:
                    rec = dict(world=world, start=start, step_size=rate)
                    result['candidates'].append(rec)
                    if found:
                        rec.update(status='unexecuted_after_witness', is_witness=False)
                        continue
                    timed = dict(world=world, start=start, step_size=rate)
                    runtime['candidates'].append(timed)
                    fit_candidate(rec, timed, x, matrices)
                    found = rec['is_witness']
                    path.write_text(encoded(result))
            executed = [r for r in result['candidates'] if r['world'] == world and 'verification' in r]
            winner = min(executed, key=lambda r: r['verification']['prediction']['all']['worst_loss'])
            result['worlds'].append(dict(world=world, renderers=matrices,
                capacity='constructive_witness' if found else 'unresolved_after_budget',
                selected_candidate=dict(start=winner['start'], step_size=winner['step_size'], step=winner['best_step']),
                best_verified_scores=winner['verification']['prediction']['all']))
        result['status'] = 'complete'
        path.write_text(encoded(result))
    except Exception as error:
        result.update(status='incomplete', failure=dict(type=type(error).__name__, message=str(error)))
        path.write_text(encoded(result))
        raise
    runtime.update(status='complete', manifest_sha256=digest, results_sha256=sha256(path),
        entire_optimizer_trajectory_repeated=False,
        repeated_scope='two task audits; two independent inference verifications of each serialized selected candidate',
        memory=dict(parameter_bytes=2624, retained_training_input_target_bytes_per_rendering=27648,
                    curvature_tensor_bytes=819200, batch_hessian_tensor_bytes=3686400,
                    native_autograd_workspace_peak_bytes=None,
                    unavailable_reason='native Torch/BLAS temporary peaks not separately instrumented; process RSS and retained optimizer tensors recorded'),
        flops=None, flops_reason='not instrumented; dense forward/backward orders are in contract',
        backward_primitive_solve_counts=None,
        backward_primitive_solve_counts_reason='autograd internals not instrumented; backward invocations counted')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=('freeze', 'tests', 'audit', 'search'), required=True)
    args = parser.parse_args()
    out = ROOT/'reports'
    path = out/f'{PREFIX}-runtime.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if args.phase in previous:
        raise RuntimeError('phase already recorded; preserve all previous work')
    record = dict(command=[sys.executable, '-m', 'experiments.masked_capacity', *sys.argv[1:]],
                  revision=git('rev-parse', 'HEAD'), dirty_paths_at_start=git('status', '--porcelain').splitlines())
    try:
        dict(freeze=freeze, tests=tests, audit=audit, search=search)[args.phase](out, record)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
    record.update(command_wall_seconds=time.perf_counter()-START,
        parent_cpu_seconds=time.process_time()-CPU_START,
        parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Linux KiB cumulative process high-water; candidate values are not isolated peaks')
    previous.update(environment=machine(), contract_sha256=sha256(ROOT/SOURCES[0]))
    previous[args.phase] = record
    path.write_text(encoded(previous))
    print(encoded({k: v for k, v in record.items() if k in ('status', 'error', 'byte_identical')}), flush=True)
    if record['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
