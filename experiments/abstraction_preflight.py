"""Freeze, test, and execute the contracted analytical preflight twice."""
import os
import time

START, CPU_START = time.perf_counter(), time.process_time()
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np

from experiments.locality_adaptation import encoded, environment, git, sha256
from src import abstraction_preflight as study

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-abstraction-preflight-v1'
SOURCES = (f'docs/{PREFIX}-contract.md', 'src/abstraction_preflight.py',
           'experiments/abstraction_preflight.py', 'tests/test_abstraction_preflight.py', 'requirements.txt')
V1_MANIFEST = 'reports/single-module-plasticity-v1-manifest.json'
TESTS = ('tests.test_abstraction_preflight', 'tests.test_single_module_plasticity')


def dependencies():
    old = json.loads((ROOT/V1_MANIFEST).read_text())
    hashes = dict(old['hashes'], **old['dependency_hashes'])
    for name, digest in hashes.items():
        if sha256(ROOT/name) != digest:
            raise RuntimeError(f'frozen v1 dependency changed: {name}')
    return dict(hashes, **{V1_MANIFEST: sha256(ROOT/V1_MANIFEST)})


def freeze(out):
    path = out/f'{PREFIX}-manifest.json'
    if path.exists():
        raise RuntimeError('manifest exists; no silent refreeze')
    manifest = dict(contract=SOURCES[0], sources={s: sha256(ROOT/s) for s in SOURCES},
        dependencies=dependencies(), revision_before_execution=git('rev-parse', 'HEAD'),
        environment=environment(), root_seed=study.ROOT_SEED,
        rng='SeedSequence([root, world, part]); worlds 0..3; rendering part 0; parameters parts 1..4',
        scenes=24, renderings=5, parameter_fixtures=16, repetitions=2, test_modules=TESTS,
        training_updates=0, tuning_executions=0,
        analytical_full_affine_loss='(8 - 2 sqrt(2))/21',
        numerical_ties='absolute 1e-12; lexicographic scenes, smallest orientation',
        serialization='12 significant digits, sorted keys; no timing in science',
        status='frozen_before_tests_and_finite_execution')
    path.write_text(encoded(manifest))
    return dict(status='complete', manifest_sha256=sha256(path))


def check_frozen(out):
    path = out/f'{PREFIX}-manifest.json'
    m = json.loads(path.read_text())
    dependencies()
    for name, digest in dict(m['sources'], **m['dependencies']).items():
        if sha256(ROOT/name) != digest:
            raise RuntimeError(f'manifest mismatch: {name}')
        committed = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT)
        if committed != (ROOT/name).read_bytes():
            raise RuntimeError(f'frozen input is not committed: {name}')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT)
    if committed != path.read_bytes():
        raise RuntimeError('manifest is not committed')
    return sha256(path)


def test_phase(out):
    digest = check_frozen(out)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    p = subprocess.run([sys.executable, '-m', 'unittest', *TESTS, '-v'], cwd=ROOT,
                       capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(status='complete' if p.returncode == 0 else 'failed', returncode=p.returncode,
        manifest_sha256=digest, command=[sys.executable, '-m', 'unittest', *TESTS, '-v'],
        wall_seconds=time.perf_counter()-start,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        child_peak_rss_kib=after.ru_maxrss, output=p.stdout+p.stderr,
        count_scope='unit tests include additional small fixtures; not part of contracted sweep counts')


def scientific_run(result, meter):
    result.update(status='running', failure=None, task={}, comparators=[], fixtures=[], module_checks=[])
    with meter.phase('enumeration'):
        result['task'] = study.task_audit()
        meter.counts['unique_abstract_scenes'] += 24
        labels, angles, clean, test = study.scenes()
    rendered = {}
    for world in [None, 0, 1, 2, 3]:
        with meter.phase('enumeration'):
            matrices = study.renderers(world)
            x = study.render(clean, matrices)
            rendered[world] = (matrices, x)
            meter.counts['rendered_scenes'] += 24
            if world is not None:
                meter.counts['renderer_qr_8'] += 4
        with meter.phase('comparator_fitting_and_scoring'):
            result['comparators'].append(dict(world=world, renderers=matrices,
                arms=study.comparator_audit(x, matrices, meter)))
    for world in range(4):
        matrices, x = rendered[world]
        for part in range(1, 5):
            with meter.phase('fixture_generation'):
                p = study.params(world, part)
                result['fixtures'].append(dict(world=world, part=part, parameters=p))
                meter.counts.update(parameter_fixtures=1, parameter_qr_blocks=12, unit_normalizations=4)
            for interface in ('zero_filled', 'masked'):
                for policy in ('fixed', 'matched'):
                    context = dict(world=world, part=part, interface=interface, policy=policy)
                    result['active_fixture'] = context
                    with meter.phase('solver_checks_and_scoring'):
                        record = study.fixture_audit(p, x, matrices, interface, policy, meter, context)
                    result['module_checks'].append(dict(world=world, part=part, interface=interface,
                                                       policy=policy, **record))
    result.pop('active_fixture')
    masked_witnesses = [dict(world=r['world'], part=r['part'], policy=r['policy'])
                        for r in result['module_checks'] if r['interface'] == 'masked' and r['is_capacity_witness']]
    result['analytical'] = dict(zero_filled_clean_loss_lower_bound=.25,
        all_scene_affine_mean_loss=study.AFFINE_FLOOR,
        all_scene_affine_mean_loss_exact='(8 - 2 sqrt(2))/21',
        masked_uniqueness='proved with positive lateral cycles and at least one observed outer coordinate',
        fixed_route_capacity='proved_obstruction', zero_filled_capacity='proved_obstruction',
        masked_matched_capacity='constructive_witness' if masked_witnesses else 'unresolved',
        masked_witnesses=masked_witnesses)
    result['status'] = 'complete'


def run_phase(out):
    digest = check_frozen(out)
    prior = json.loads((out/f'{PREFIX}-runtime.json').read_text())
    if prior.get('tests', {}).get('status') != 'complete' or prior['tests']['manifest_sha256'] != digest:
        raise RuntimeError('successful tests for this manifest required before finite execution')
    timings, previous = [], None
    result = {}
    for repeat in range(2):
        meter = study.Meter()
        result = dict(manifest_sha256=digest, root_seed=study.ROOT_SEED)
        start, cpu = time.perf_counter(), time.process_time()
        try:
            scientific_run(result, meter)
            serialized = encoded(result)
            if previous is not None and previous != serialized:
                raise RuntimeError('scientific repetition is not byte-identical')
        except Exception as e:
            result.update(status='incomplete', failure=dict(type=type(e).__name__, message=str(e)))
            (out/f'{PREFIX}-results.json').write_text(encoded(result))
            timings.append(dict(repeat=repeat, wall_seconds=time.perf_counter()-start,
                process_cpu_seconds=time.process_time()-cpu, counts=dict(meter.counts),
                phase_wall_seconds=dict(meter.seconds), phase_cpu_seconds=dict(meter.cpu_seconds)))
            return dict(status='failed', failure=result['failure'], repeats=timings)
        timings.append(dict(repeat=repeat, wall_seconds=time.perf_counter()-start,
            process_cpu_seconds=time.process_time()-cpu, counts=dict(meter.counts),
            phase_wall_seconds=dict(meter.seconds), phase_cpu_seconds=dict(meter.cpu_seconds)))
        previous = serialized
        print(f'Finite execution {repeat+1}/2 complete', flush=True)
    (out/f'{PREFIX}-results.json').write_text(previous)
    return dict(status='complete', byte_identical_repetition=True, manifest_sha256=digest,
        results_sha256=sha256(out/f'{PREFIX}-results.json'), repeats=timings,
        training_updates=0, tuning_executions=0,
        memory=dict(parameter_bytes=2624, phase_metadata_bytes=8, mask_bytes=study.MASK.nbytes,
            curvature_cache_bytes=819200, one_dense_hessian_bytes=51200,
            scene_arrays_all_renderings_bytes=5*24*32*8,
            solver_workspace_peak_bytes=None,
            solver_workspace_peak_reason='NumPy/SciPy native temporary peaks not separately instrumented; process RSS recorded',
            result_file_bytes=(out/f'{PREFIX}-results.json').stat().st_size),
        flops=None, flops_reason='not instrumented; complexity orders in contract',
        timing_scope='repeat total includes generation, fitting, numerical checks, scoring and serialization; phase timers are disjoint and contained in total')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', required=True, choices=('freeze', 'tests', 'run'))
    args = parser.parse_args()
    out = ROOT/'reports'
    path = out/f'{PREFIX}-runtime.json'
    runtime = json.loads(path.read_text()) if path.exists() else {}
    if args.phase in runtime:
        raise RuntimeError(f'{args.phase} already recorded; do not silently overwrite previous work')
    started_dirty = git('status', '--porcelain').splitlines()
    try:
        record = dict(freeze=freeze, tests=test_phase, run=run_phase)[args.phase](out)
    except Exception as e:
        record = dict(status='failed', error_type=type(e).__name__, error=str(e))
    record.update(command=[sys.executable, '-m', 'experiments.abstraction_preflight', *sys.argv[1:]],
        revision=git('rev-parse', 'HEAD'), dirty_paths_at_start=started_dirty,
        command_wall_seconds=time.perf_counter()-START,
        parent_process_cpu_seconds=time.process_time()-CPU_START,
        parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Linux KiB, cumulative process high-water RSS; tests child reported separately')
    runtime.update(environment=environment(), contract_sha256=sha256(ROOT/SOURCES[0]))
    runtime[args.phase] = record
    path.write_text(encoded(runtime))
    print(encoded({k: v for k, v in record.items() if k in ('status', 'error', 'failure', 'byte_identical_repetition')}))
    if record['status'] != 'complete':
        if 'output' in record:
            print(record['output'])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
