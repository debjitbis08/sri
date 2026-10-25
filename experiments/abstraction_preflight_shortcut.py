"""Post-run algebraic receipt for the declared split's sign-reversal shortcut.

Uses only the frozen preflight's stored scenes/renderers. No new task, fit,
parameter search, or change to the prespecified comparator table.
"""
import os
import time

START, CPU_START = time.perf_counter(), time.process_time()
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import json
from pathlib import Path
import resource
import sys

import numpy as np

from experiments.locality_adaptation import encoded, environment, git, sha256

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'single-module-abstraction-preflight-v1'


def main():
    source = ROOT/'reports'/f'{PREFIX}-results.json'
    target = ROOT/'reports'/f'{PREFIX}-shortcut-receipt.json'
    runtime_path = ROOT/'reports'/f'{PREFIX}-shortcut-runtime.json'
    if target.exists() or runtime_path.exists():
        raise RuntimeError('receipt already exists; no silent overwrite')
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in
              (source, Path(__file__).resolve(), ROOT/'experiments/locality_adaptation.py')}
    provenance = dict(hashes_before_execution=hashes, revision=git('rev-parse', 'HEAD'),
        dirty_paths=git('status', '--porcelain').splitlines(),
        scope='post-run deterministic proof receipt; no new samples, fitted predictor or tuning; main results unchanged')
    result = json.loads(source.read_text())
    labels = np.array(result['task']['labels'])
    clean = np.array(result['task']['clean_identity']).reshape(24, 4, 8)
    train = np.array(result['task']['train_scene_indices'])
    held = np.array(result['task']['held_out_scene_indices'])
    lookup = {tuple(pair): i for i, pair in enumerate(labels)}
    pairs = []
    for i in held:
        r, t = labels[i]
        opposite = lookup[int(r), int((t+4) % 8)]
        assert opposite in train
        assert np.max(np.abs(clean[i]+clean[opposite])) < 1e-10
        pairs.append(dict(held_out_scene=int(i), training_scene=opposite,
                          relation=int(r), held_out_t=int(t), training_t=int((t+4) % 8)))
    assert len({(int(labels[i, 0]), int(labels[i, 1] % 4)) for i in train}) == 12

    receipts = []
    for entry in result['comparators']:
        matrices = np.array(entry['renderers'])
        x = np.einsum('jab,njb->nja', matrices, clean).reshape(24, 32)
        # Predictor uses only training observations/targets and the supplied
        # odd-symmetry prior. It never uses r, t or held-out targets to choose.
        candidates = np.stack([x[train], -x[train]], axis=1).reshape(-1, 32)
        squared_distances = np.sum((x[held, None, :16]-candidates[None, :, :16])**2, axis=-1)
        selected = np.argmax(squared_distances <= squared_distances.min(axis=1, keepdims=True)+1e-12, axis=1)
        predictions = candidates[selected, 16:]
        loss = np.sum((predictions-x[held, 16:])**2, axis=1)/2
        assert squared_distances.min(axis=1).max() < 1e-20
        assert loss.max() < 1e-20
        receipts.append(dict(world=entry['world'], chosen_training_scenes=train[selected//2],
            chosen_sign=np.where(selected % 2 == 0, 1, -1),
            max_visible_squared_distance=float(squared_distances.min(axis=1).max()),
            per_scene_hidden_loss=loss, worst_hidden_loss=float(loss.max()),
            stored_lookup_bytes=x[train].nbytes))
    scientific = dict(status='complete', input_hashes=hashes,
        finding='every held-out scene is the negative of a training scene with the same relation',
        total_sign_orbits=12, training_sign_orbits=12, held_out_scenes=8,
        antipodal_pairs=pairs, signed_lookup_receipts=receipts,
        interpretation='the declared split cannot distinguish relational learning from memorization plus odd symmetry')
    target.write_text(encoded(scientific))
    runtime = dict(status='complete', provenance=provenance, environment=environment(),
        command=[sys.executable, '-m', 'experiments.abstraction_preflight_shortcut'],
        counts=dict(renderings=5, held_out_queries=40, candidates_per_query=32,
                    squared_distances=1280, fitted_models=0, new_scenes=0, training_updates=0,
                    tuning_executions=0),
        complexity='O(R N_test (2 N_train) d) time; O(N_train d) lookup plus O(N_test N_train) distance workspace',
        wall_seconds=time.perf_counter()-START, process_cpu_seconds=time.process_time()-CPU_START,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope='Linux KiB, cumulative single-process high-water RSS',
        receipt_sha256=sha256(target))
    runtime_path.write_text(encoded(runtime))
    print(encoded(dict(status='complete', held_out_scenes=8, training_sign_orbits=12,
                       worst_hidden_loss=max(r['worst_hidden_loss'] for r in receipts))))


if __name__ == '__main__':
    main()
