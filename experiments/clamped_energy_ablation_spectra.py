"""Post-hoc translation spectra for clamped-energy ablation v1.

Reads the committed results JSON and reports, for every learned-translation
condition, the singular values of each final translation and the outer
translations' gains on the source module's pattern span versus its nuisance
complement (evaluator-only maps). The frozen artifacts are not modified.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
from experiments import local_contrastive_consolidation as lcc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/clamped-energy-ablation-v1-results.json'
PREFIX = ROOT / 'reports/clamped-energy-ablation-v1-spectra'


def stats(rows):
    a = np.array(rows)
    return {'mean_sorted': a.mean(axis=0).tolist(), 'min': float(a.min()), 'max': float(a.max()),
            'fraction_below_0.5': float(np.mean(a < .5))}


def main():
    result, runtime = Path(str(PREFIX) + '-results.json'), Path(str(PREFIX) + '-runtime.json')
    if result.exists() or runtime.exists():
        raise RuntimeError('Refusing overwrite')
    wall, cpu = time.perf_counter(), time.process_time()
    source = json.loads(SOURCE.read_text())
    seeds = sorted(source['seeds'], key=int)
    names = [n for n in source['seeds'][seeds[0]]['conditions'] if '_F_' not in n]
    out = {}
    for name in names:
        spectra, pattern, nuisance = {}, [], []
        for seed in seeds:
            params = source['seeds'][seed]['conditions'][name]['final_parameters']
            maps = lcc.private_maps(int(seed))
            for key, value in params.items():
                if not key.startswith('T'):
                    continue
                T = np.array(value)
                layer = lcc.LAYER_NAMES[int(key.split('_')[1])]
                spectra.setdefault(layer, []).append(np.linalg.svd(T, compute_uv=False))
                if layer == 'outer':
                    M = maps[int(key[1])]  # source module's orthonormal pattern columns
                    N = np.linalg.svd(M, full_matrices=True)[0][:, 2:]  # orthonormal nuisance complement
                    pattern.append(np.linalg.svd(T @ M, compute_uv=False))
                    nuisance.append(np.linalg.svd(T @ N, compute_uv=False))
        out[name] = {'singular_values_by_layer': {layer: stats(rows) for layer, rows in spectra.items()}}
        if pattern:
            out[name]['outer_gain_on_pattern_span'] = stats(pattern)
            out[name]['outer_gain_on_nuisance_complement'] = stats(nuisance)
    science = {'source': str(SOURCE.relative_to(ROOT)), 'seeds': seeds, 'conditions': out,
               'definitions': {'singular_values': 'per translation (both edges), sorted descending, pooled over seeds',
                               'outer_gain': 'singular values of T M and T N, where M spans the source module pattern '
                                             'columns and N its orthonormal complement; basis-invariant'}}
    result.write_text(json.dumps(science, indent=1, sort_keys=True) + '\n')
    files = (Path(__file__), Path(lcc.__file__), SOURCE, result)
    info = {'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'command': ' '.join(sys.argv), 'python': sys.version, 'numpy': np.__version__,
            'os': platform.platform(), 'dtype': 'float64',
            'complexity': 'one SVD of at most 8x8 per translation, seed and condition, plus two 8x2/8x6 products',
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'timing_scope': 'whole run including JSON parsing; not a latency benchmark'}
    runtime.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
