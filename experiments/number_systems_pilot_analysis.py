"""Post-hoc diagnostics of the number systems v1 pilot (docs/number-systems-v1-pilot-results.md).

Three checks from the external review, on the saved source checkpoints:

1. Cross-representation number probe by number. Each test number of a probe pair is either labelled in
   both representations (a shared anchor), in the probe's training representation only, in the test
   representation only, or in neither. Transfer at numbers labelled in neither cannot come from the label
   pull and has to propagate through the shared relational structure.
2. Fixed versus refitted readouts across adaptation. A pooled source number probe is fitted on half of the
   pre-adaptation source codes and kept fixed; a second probe is refitted on post-adaptation codes. Both are
   tested on the other half. A drop under the fixed probe only means changed coordinates; a drop under both
   means lost information.
3. Interface copy. In 'whole-copy' the held-out representation adapts a copy of its medium satellite while
   the sources keep the original, so source retention measures change in the core alone. Interface-only
   adaptation of a copy would train the held-out representation exactly as 'interface' does and leave the
   sources untouched by construction, so it is not run.

Usage: .venv/bin/python -m experiments.number_systems_pilot_analysis pilot
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
PREFIX = ROOT / 'reports' / f'{X.NAME}-analysis'
RUNS = ('C14', 'I14')
VARIANTS = ('zero-shot', 'interface', 'whole', 'whole-copy')
MARKS = ('dice', 'dots', 'tally', 'banana', 'egyptian', 'roman')  # mark/collection systems, as in the pilot report


def load(name, step):
    enc, sri = X.make_model(X.SEED)
    ck = torch.load(X.RUNS / f'{name}-{step}.pt', weights_only=False)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    sri.load_state_dict(ck['sri'])
    return enc, sri, ck['sources'], {r: tuple(v) for r, v in ck['labelled'].items()}


def fit_probe(x, y, k, seed, steps=300):
    """The pilot's linear probe (X.probe), returned as a fixed predictor."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    lin = torch.nn.Linear(x.shape[1], k)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * .01)
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin((x - mu) / sd), y).backward()
        opt.step()
    return lambda z: lin((z - mu) / sd).argmax(1).detach().numpy()


# ---------------------------------------------------------------- 1. probe transfer by number

def category(n, labelled, a, b):
    return {(True, True): 'both', (True, False): 'train_only', (False, True): 'test_only',
            (False, False): 'neither'}[(n in labelled[a], n in labelled[b])]


def by_number(job):
    name, step = job
    t0, c0 = time.perf_counter(), time.process_time()
    enc, sri, sources, labelled = load(name, step)
    enc['image'].eval(); enc['sequence'].eval()
    st = X.eval_codes(enc, sri, sources)
    pairs = {}
    for a in sources:
        ya = torch.as_tensor(X.DATA[(a, 'eval')][1])
        for b in sources:
            if a == b:
                continue
            yb = X.DATA[(b, 'eval')][1]
            hit = X.probe(st[a]['core'], ya, st[b]['core'], torch.as_tensor(yb), 10, 1)  # the pilot's call
            shared = sorted(set(X.numbers(a)) & set(X.numbers(b)))
            pairs[f'{a}->{b}'] = {'mean': float(hit.mean()),
                                  'by_number': {str(n): float(hit[yb == n].mean()) for n in shared},
                                  'category': {str(n): category(n, labelled, a, b) for n in shared}}

    def summary(keep):
        out = {}
        for cat in ('both', 'train_only', 'test_only', 'neither'):
            v = [p['by_number'][n] for k, p in pairs.items() if keep(*k.split('->')) for n, c in p['category'].items() if c == cat]
            out[cat] = {'mean': float(np.mean(v)) if v else None, 'cells': len(v)}
        return out
    res = {'pairs': pairs, 'summary': {
        'all': summary(lambda a, b: True),
        'not_both_marks': summary(lambda a, b: not (a in MARKS and b in MARKS)),
        'both_marks': summary(lambda a, b: a in MARKS and b in MARKS),
        'cross_medium': summary(lambda a, b: X.MEDIUM[a] != X.MEDIUM[b])}}
    return ('by_number', name, step), res, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                            'probes': len(pairs), 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


# ---------------------------------------------------------------- 2-3. adaptation with fixed and refitted readouts

def halves(r):
    y = X.DATA[(r, 'eval')][1]
    p = nc.stream('analysis_halves', r).permutation(len(y))
    return np.sort(p[:len(y) // 2]), np.sort(p[len(y) // 2:])


def pooled(st, sources, part):
    return (torch.cat([st[r]['core'][part[r]] for r in sources]),
            np.concatenate([X.DATA[(r, 'eval')][1][part[r]] for r in sources]))


def adapt_variant(job):
    name, step, held, variant = job
    t0, c0 = time.perf_counter(), time.process_time()
    enc, sri, sources, labelled = load(name, step)
    codes = ncore.label_codes(net.SIZES[0])
    med = X.MEDIUM[held]
    A = {r: halves(r)[0] for r in sources}
    B = {r: halves(r)[1] for r in sources}
    enc['image'].eval(); enc['sequence'].eval()
    with torch.no_grad():
        st0 = X.eval_codes(enc, sri, sources)
    xa, ya = pooled(st0, sources, A)
    fixed = fit_probe(xa, torch.as_tensor(ya), 10, 3)
    enc['image'].train(); enc['sequence'].train()
    saved = dict(X.MEDIUM)
    try:
        if variant == 'whole-copy':
            enc['replica'] = copy.deepcopy(enc[med])
            sri.add_satellite('replica')
            with torch.no_grad():
                for p, q in zip(sri.satellite_parameters('replica'), sri.satellite_parameters(med)):
                    p.copy_(q)
            X.MEDIUM[held] = 'replica'
        losses = []
        if variant != 'zero-shot':
            base = variant.removesuffix('-copy')
            params = [*enc[med].parameters(), *sri.satellite_parameters(med)] if base == 'interface' else \
                [p for e in enc.values() for p in e.parameters()] + list(sri.parameters())
            opt = torch.optim.Adam(params, X.LR)
            idx = {held: {c: np.flatnonzero(X.DATA[(held, 'train')][1] == c) for c in X.numbers(held)}}
            order = {held: {n: n for n in X.numbers(held)}}
            # Same data stream as the pilot's run of the base mode, so variants differ only in the interface.
            losses, _ = X.train_steps(enc, sri, opt, [held], idx, {held: X.ADAPT_LABELS}, order, X.ADAPT,
                                      nc.stream('adapt', X.SEED, name, step, held, base), codes)
        for e in enc.values():
            e.eval()
        with torch.no_grad():
            st = X.eval_codes(enc, sri, [held] + sources)
    finally:
        X.MEDIUM.clear(); X.MEDIUM.update(saved)
    xa1, ya1 = pooled(st, sources, A)
    refit = fit_probe(xa1, torch.as_tensor(ya1), 10, 3)
    xb0, yb = pooled(st0, sources, B)
    xb1, _ = pooled(st, sources, B)
    with torch.no_grad():
        hits = {'before': fixed(xb0) == yb, 'fixed': fixed(xb1) == yb, 'refit': refit(xb1) == yb}
    src = np.concatenate([[r] * len(B[r]) for r in sources])
    lab = np.array([n in labelled[r] for r, n in zip(src, yb)])
    probe_res = {k: {'mean': float(h.mean()), 'source_labelled': float(h[lab].mean()), 'source_unlabelled': float(h[~lab].mean()),
                     'per_source': {r: float(h[src == r].mean()) for r in sources}} for k, h in hits.items()}
    readout = {k: float(np.mean([np.mean(list(X.label_acc(s[r], X.DATA[(r, 'eval')][1], codes).values())) for r in sources]))
               for k, s in (('before', st0), ('after', st))}
    y = X.DATA[(held, 'eval')][1]
    acc = X.label_acc(st[held], y, codes)
    with torch.no_grad():
        on_held = fixed(st[held]['core']) == y
    res = {'loss_every_100': losses,
           'held_out': {'labelled': float(np.mean([acc[n] for n in X.ADAPT_LABELS])),
                        'interpolation_3_4': float(np.mean([acc[3], acc[4]])),
                        'extrapolation_6_9': float(np.mean([acc[n] for n in (6, 7, 8, 9)])),
                        'fixed_source_probe': {str(c): float(on_held[y == c].mean()) for c in X.numbers(held)},
                        'fixed_source_probe_unlabelled': float(on_held[~np.isin(y, X.ADAPT_LABELS)].mean())},
           'source_number_probe': probe_res, 'source_label_readout': readout}
    return ('adapt', name, step, held, variant), res, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'steps': 0 if variant == 'zero-shot' else X.ADAPT, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(job):
    return by_number(job[1:]) if job[0] == 'by_number' else adapt_variant(job[1:])


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    jobs = [('by_number', n, c) for n in RUNS for c in X.CKPTS] + \
           [('adapt', n, c, h, v) for n in RUNS for c in X.CKPTS for h in X.HELD_OUT for v in VARIANTS]
    done = []
    t = time.perf_counter()
    with mp.get_context('fork').Pool(X.WORKERS) as pool:
        for item in pool.imap_unordered(run, jobs):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(jobs)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['analysis_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    res = {'source_report': f'reports/{X.NAME}-results.json', 'status': 'post-hoc diagnostics of a pilot: not headline evidence',
           'config': {'runs': RUNS, 'checkpoints': X.CKPTS, 'variants': VARIANTS, 'marks': MARKS, 'adapt_steps': X.ADAPT,
                      'probe_halves': 'eval pool split in half per representation by stream analysis_halves'},
           'checkpoint_sha256': {f'{n}/{c}': nc.sha256(X.RUNS / f'{n}-{c}.pt') for n in RUNS for c in X.CKPTS},
           'results': {key(k): r for k, r, _ in sorted(done, key=lambda d: key(d[0]))}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json',
        'runner_sha256': nc.sha256(__file__), 'experiment_sha256': nc.sha256(X.__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NSY_')},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'threads_per_worker': 1,
        'phases': phases, 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
