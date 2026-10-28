"""Reuse cost on the number systems v1 pilot checkpoints (docs/number-systems-v1-pilot-results.md, Stage A).

Reuse benefit = cost to learn a new representation without source experience - cost with it, both to the
same accuracy and retention requirements. The new task is few-shot: a held-out representation with all
numbers labelled but only k training examples per number, plus within-representation relations among them.
Held-out accuracy and source retention are scored every EVERY updates up to MAX_STEPS.

The held-out representation always trains a replica of its medium satellite. Its training is then identical
to training the shared satellite (the original receives no gradient from it), and one run gives both
retention readings: sources through the replica (shared interface) and through the original (separate
interface, which isolates change in the core and output).

Usage: .venv/bin/python -m experiments.number_systems_reuse_cost pilot
"""
import copy
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

from experiments import number_systems as X

nc, ncore, ns, net = X.nc, X.ncore, X.ns, X.net
ROOT = X.ROOT
PREFIX = ROOT / 'reports' / f'{X.NAME}-reuse-cost'
SOURCES = [('C14', 1000), ('C14', 3000), ('I14', 1000), ('I14', 3000), (None, 0)]
MODES = ('interface', 'whole')
SHOTS = (2, 8, 32) if not X.SMOKE else (2,)
MAX_STEPS, EVERY = (1000, 25) if not X.SMOKE else (20, 10)
HELD_EVAL, SOURCE_EVAL = (100, 50) if not X.SMOKE else (10, 5)  # test images per number
CRITERION, RETAIN = 0.8, 0.9  # held-out accuracy over all numbers; source labelled accuracy relative to before


def subset(r, per, key):
    y = X.DATA[(r, 'eval')][1]
    return np.concatenate([nc.stream('reuse_eval', key, r, int(c)).permutation(np.flatnonzero(y == c))[:per] for c in X.numbers(r)])


def held_accuracy(st, y, codes):
    acc = X.label_acc(st, y, codes)
    return float(np.mean(list(acc.values()))), {str(k): v for k, v in acc.items()}


def source_retention(enc, sri, sources, labelled, groups, codes, med, shared):
    """Mean over sources of accuracy on their own labelled numbers; shared routes this medium through the replica."""
    saved = dict(X.MEDIUM)
    try:
        if shared:
            for r in sources:
                if saved[r] == med:
                    X.MEDIUM[r] = 'replica'
        st = X.encode(enc, sri, groups, 'eval')
    finally:
        X.MEDIUM.clear(); X.MEDIUM.update(saved)
    out = {}
    for r in sources:
        acc = X.label_acc(st[r], X.DATA[(r, 'eval')][1][groups[r]], codes)
        out[r] = float(np.mean([acc[n] for n in labelled[r]]))
    return float(np.mean(list(out.values()))), out


def first_reaching(curve, ok):
    return next((p['step'] for p in curve if ok(p)), None)


def run(job):
    source, step, held, mode, k = job
    t0, c0 = time.perf_counter(), time.process_time()
    enc, sri = X.make_model(X.SEED)
    sources, labelled = [], {}
    if source is not None:
        ck = torch.load(X.RUNS / f'{source}-{step}.pt', weights_only=False)
        enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
        sri.load_state_dict(ck['sri'])
        sources, labelled = ck['sources'], {r: tuple(v) for r, v in ck['labelled'].items()}
    codes = ncore.label_codes(net.SIZES[0])
    med = X.MEDIUM[held]
    enc['replica'] = copy.deepcopy(enc[med])
    sri.add_satellite('replica')
    with torch.no_grad():
        for p, q in zip(sri.satellite_parameters('replica'), sri.satellite_parameters(med)):
            p.copy_(q)
    held_idx = subset(held, HELD_EVAL, 'held')
    y_held = X.DATA[(held, 'eval')][1][held_idx]
    src_groups = {r: subset(r, SOURCE_EVAL, 'source') for r in sources}
    train_y = X.DATA[(held, 'train')][1]
    shots = {c: nc.stream('reuse_shots', held, int(c)).permutation(np.flatnonzero(train_y == c))[:k] for c in X.numbers(held)}
    curve, eval_cpu = [], 0.

    def evaluate(n):
        nonlocal eval_cpu
        e0 = time.process_time()
        for e in enc.values():
            e.eval()
        saved = dict(X.MEDIUM)
        X.MEDIUM[held] = 'replica'
        try:
            with torch.no_grad():
                acc, per = held_accuracy(X.encode(enc, sri, {held: held_idx}, 'eval')[held], y_held, codes)
                point = {'step': n, 'held_accuracy': acc, 'held_by_number': per}
                if sources:
                    point['retention_shared'], _ = source_retention(enc, sri, sources, labelled, src_groups, codes, med, True)
                    point['retention_separate'], _ = source_retention(enc, sri, sources, labelled, src_groups, codes, med, False)
        finally:
            X.MEDIUM.clear(); X.MEDIUM.update(saved)
        for e in enc.values():
            e.train()
        eval_cpu += time.process_time() - e0
        point['train_cpu_seconds'] = time.process_time() - c0 - eval_cpu
        curve.append(point)

    params = [*enc['replica'].parameters(), *sri.satellite_parameters('replica')] if mode == 'interface' else \
        [p for e in enc.values() for p in e.parameters()] + list(sri.parameters())
    opt = torch.optim.Adam(params, X.LR)
    order = {held: {n: n for n in X.numbers(held)}}
    X.MEDIUM[held] = 'replica'
    try:
        evaluate(0)
        losses, encodings = X.train_steps(enc, sri, opt, [held], {held: shots}, {held: tuple(X.numbers(held))}, order, MAX_STEPS,
                                          nc.stream('reuse_adapt', X.SEED, str(source), step, held, mode, k), codes,
                                          lambda n: evaluate(n) if n % EVERY == 0 else None)
    finally:
        X.MEDIUM[held] = med
    before = curve[0]
    reach = {'accuracy_only': first_reaching(curve, lambda p: p['held_accuracy'] >= CRITERION)}
    if sources:
        for kind in ('shared', 'separate'):
            reach[f'with_retention_{kind}'] = first_reaching(
                curve, lambda p: p['held_accuracy'] >= CRITERION and p[f'retention_{kind}'] >= RETAIN * before[f'retention_{kind}'])
    cost = {kname: None if s is None else {'updates': s, 'training_encodings': s * encodings // MAX_STEPS,
                                           'train_cpu_seconds': next(p['train_cpu_seconds'] for p in curve if p['step'] == s)}
            for kname, s in reach.items()}
    return (str(source), step, held, mode, k), {
        'examples': k * len(X.numbers(held)), 'curve': curve, 'loss_every_100': losses,
        'steps_to_criterion': reach, 'cost_to_criterion': cost}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'eval_cpu_seconds': eval_cpu,
        'updates': MAX_STEPS, 'training_encodings': encodings, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    jobs = [(s, c, h, m, k) for s, c in SOURCES for h in X.HELD_OUT for m in MODES for k in SHOTS]
    jobs.sort(key=lambda j: -j[4])  # largest shots first
    done = []
    t = time.perf_counter()
    with mp.get_context('fork').Pool(X.WORKERS) as pool:
        for item in pool.imap_unordered(run, jobs):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(jobs)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['adaptation_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    res = {'source_report': f'reports/{X.NAME}-results.json', 'status': 'Stage A measurement on pilot checkpoints: not headline evidence',
           'config': {'sources': SOURCES, 'modes': MODES, 'shots': SHOTS, 'max_steps': MAX_STEPS, 'every': EVERY,
                      'held_eval_per_number': HELD_EVAL, 'source_eval_per_number': SOURCE_EVAL, 'criterion': CRITERION,
                      'retain': RETAIN, 'lr': X.LR, 'batch': X.BATCH, 'pairs': X.PAIRS, 'held_out': X.HELD_OUT,
                      'retention': 'mean over sources of label accuracy on their own labelled numbers, relative to step 0'},
           'checkpoint_sha256': {f'{s}/{c}': nc.sha256(X.RUNS / f'{s}-{c}.pt') for s, c in SOURCES if s},
           'results': {key(k): r for k, r, _ in sorted(done, key=lambda d: key(d[0]))}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'experiment_sha256': nc.sha256(X.__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NSY_')},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'threads_per_worker': 1,
        'phases': phases, 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'worker_eval_cpu_seconds_total': sum(r['eval_cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
