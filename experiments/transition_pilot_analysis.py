"""Transition network pilot analysis (docs/transition-pilot-v1-spec.md): standard network with relation heads (stage 1),
generic comparator only (generic relations v1), and generic comparator plus transition network, paired by seed.

The generic measures (judgment revision, next-judgment coherence) are recomputed from end-of-training checkpoints for
all three networks with the same procedures. Composition is reported for the transition network only. Outcome labels
compare the transition network with each of the other two, using the audit's proposed margins.

Usage: .venv/bin/python -m experiments.transition_pilot_analysis pilot
"""
import gzip
import json
import sys
import time

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import standard_baseline_compare as C
from experiments import generic_relations as G
from experiments import transition_pilot as TP

ROOT = X.ROOT
SEEDS = C.SEEDS
NETS = {'heads': ('number-construction-v1-standard-part-d-pilot', SB.make_model),
        'generic': ('generic-relations-v1-pilot', G.make_model),
        'transition': (TP.NAME, TP.make_model)}
MAIN = 'D-varied+rehearsal'


def fresh(net, seed):
    name, make = NETS[net]
    ck = torch.load(ROOT / 'runs' / name / f'{MAIN}-seed{seed}.pt', weights_only=False)
    enc, model = make(seed)
    model.load_state_dict(ck['sri'])
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    out = G.generic_extras(enc, model, ck['withheld'])
    out['coherence'] = G.generic_coherence(enc, model)
    return out


def dig(d, path):
    for p in path:
        d = d[p]
    return d


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    X.prepare(); P.prepare(); D.prepare()
    R = {n: C.load(v[0]) for n, v in NETS.items()}
    F = {n: {s: fresh(n, s) for s in SEEDS} for n in NETS}
    iso = lambda key, block=-1: lambda n, arm, s: R[n][f'{arm}/seed{s}']['after_block'][block]['isolated'][key]
    sen = lambda *p: lambda n, arm, s: dig(R[n][f'{arm}/seed{s}']['after_block'][-1]['senses'], p)
    fr = lambda *p: lambda n, arm, s: dig(F[n][s], p)
    measures = [
        ('Chinese withheld, own block', MAIN, iso('chinese/withheld', 1), 'acc'),
        ('Devanagari withheld, own block', MAIN, iso('devanagari_digits/withheld', 2), 'acc'),
        ('Chinese paired, own block', MAIN, iso('chinese/paired', 1), 'acc'),
        ('Chinese paired, end (rehearsal)', 'D-rehearsal', iso('chinese/paired'), 'acc'),
        ('English, end', MAIN, iso('english/paired'), 'acc'),
        ('taps → collection (taught)', MAIN, sen('taps_to_collection'), 'acc'),
        ('taps → Chinese form, withheld', MAIN, sen('cross_sense', 'taps_to_form/chinese/withheld'), 'acc'),
        ('taps → Devanagari form, withheld', MAIN, sen('cross_sense', 'taps_to_form/devanagari_digits/withheld'), 'acc'),
        ('form → taps, Chinese withheld', MAIN, sen('cross_sense', 'form_to_taps/chinese/withheld'), 'acc'),
        ('Comparison, Chinese forms (untaught)', MAIN, sen('comparison', 'forms/chinese'), 'acc'),
        ('Comparison, Devanagari forms (untaught)', MAIN, sen('comparison', 'forms/devanagari_digits'), 'acc'),
        ('Comparison, English forms (untaught)', MAIN, sen('comparison', 'forms/english'), 'acc'),
        ('Comparison, collections', MAIN, sen('comparison', 'collections'), 'acc'),
        ('Succession pairwise, collections', MAIN, fr('succession_pairwise', 'collections'), 'acc'),
        ('Judgment revision: isolated, withheld', MAIN, fr('context/withheld', 'isolated'), 'acc'),
        ('Judgment revision: revised, withheld', MAIN, fr('context/withheld', 'two_sided_revised'), 'acc'),
        ('Judgment revision: swap follows context', MAIN, fr('context/withheld', 'two_sided_swap_follows_context'), 'acc'),
        ('Next-judgment coherence AUROC, Chinese', MAIN, fr('coherence', 'forms/chinese'), 'auroc'),
        ('Next-judgment coherence AUROC, English', MAIN, fr('coherence', 'forms/english'), 'auroc'),
        ('Thai, 4 shots, after 200 updates', MAIN, lambda n, arm, s: R[n][f'{arm}/seed{s}']['transfer']['200'], 'acc'),
    ]
    rows = []
    for name, arm, get, kind in measures:
        v = {n: [get(n, arm, s) for s in SEEDS] for n in NETS}
        rows.append({'measure': name, 'arm': arm, 'kind': kind, **v,
                     'vs_heads': C.classify(np.subtract(v['transition'], v['heads']), kind),
                     'vs_generic': C.classify(np.subtract(v['transition'], v['generic']), kind)})
    comp = {k: [R['transition'][f'{MAIN}/seed{s}']['after_block'][-1]['senses']['composition'][k] for s in SEEDS]
            for k in R['transition'][f'{MAIN}/seed{SEEDS[0]}']['after_block'][-1]['senses']['composition']}
    stored = max(abs(R['transition'][f'{MAIN}/seed{s}']['after_block'][-1]['context/withheld']['two_sided_revised'] - F['transition'][s]['context/withheld']['two_sided_revised']) for s in SEEDS)
    out = {'seeds': SEEDS, 'margins': C.MARGIN, 'networks': {n: v[0] for n, v in NETS.items()}, 'measures': rows,
           'composition': comp, 'recomputation_max_abs_error': stored}
    with gzip.open(ROOT / 'reports' / 'transition-pilot-v1-comparison.json.gz', 'wt') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    (ROOT / 'reports' / 'transition-pilot-v1-comparison-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing; results in transition-pilot-v1-comparison.json.gz', 'module_sha256': X.nc.sha256(__file__),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')), 'command': ' '.join([sys.executable, *sys.argv]),
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}, indent=1) + '\n')
    f = lambda v: f'{np.mean(v):.2f} ({", ".join(f"{x:.2f}" for x in v)})'
    print('| Measure | Heads | Generic | Generic + transition | vs heads | vs generic |\n| --- | --- | --- | --- | --- | --- |')
    for r in rows:
        print(f"| {r['measure']} | {np.mean(r['heads']):.2f} | {np.mean(r['generic']):.2f} | {f(r['transition'])} | {r['vs_heads']} | {r['vs_generic']} |")
    print('\nComposition (transition network, D-varied+rehearsal):')
    for k, v in comp.items():
        if k.endswith('/same') or k.endswith('/distance'):
            print(k, f(v))
    print('recomputation error', stored)


if __name__ == '__main__':
    main()
