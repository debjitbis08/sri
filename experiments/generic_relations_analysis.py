"""Generic relations v1 pilot analysis (docs/generic-relations-v1-spec.md): the generic comparator against the standard
network with relation heads, paired by seed (100-102) on the Part D arms.

Measures from the runners' results (same evaluation code for both), plus the judgment-based revision and coherence of
experiments.generic_relations recomputed from the end-of-training checkpoints of both networks, so both are scored by
the same procedures. Outcome labels use the audit's proposed margins (experiments.standard_baseline_compare).

Usage: .venv/bin/python -m experiments.generic_relations_analysis pilot [standard|sri]
With 'sri', the comparison is SRI with relation heads (the Part D pilot) against SRI with the generic comparator.
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

ROOT = X.ROOT
SEEDS = C.SEEDS
if G.ARCH == 'sri':
    REPORT = {'heads': 'number-construction-v1-part-d-pilot', 'generic': G.NAME}
    MAKE = {'heads': X.make_model, 'generic': G.make_sri_model}
else:
    REPORT = {'heads': 'number-construction-v1-standard-part-d-pilot', 'generic': G.NAME}
    MAKE = {'heads': SB.make_model, 'generic': G.make_model}
RUNS = {a: ROOT / 'runs' / n for a, n in REPORT.items()}
SUFFIX = '-sri' if G.ARCH == 'sri' else ''
MAIN = 'D-varied+rehearsal'


def fresh(arch, arm, seed):
    ck = torch.load(RUNS[arch] / f'{arm}-seed{seed}.pt', weights_only=False)
    enc, model = MAKE[arch](seed)
    model.load_state_dict(ck['sri'])
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    out = G.generic_extras(enc, model, ck['withheld'])
    out['coherence'] = G.generic_coherence(enc, model)
    return out


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    X.prepare(); P.prepare(); D.prepare()
    R = {a: C.load(n) for a, n in REPORT.items()}
    F = {a: {s: fresh(a, MAIN, s) for s in SEEDS} for a in R}
    iso = lambda key, block=-1: lambda a, arm, s: R[a][f'{arm}/seed{s}']['after_block'][block]['isolated'][key]
    sen = lambda *p: lambda a, arm, s: _dig(R[a][f'{arm}/seed{s}']['after_block'][-1]['senses'], p)
    fr = lambda *p: lambda a, arm, s: _dig(F[a][s], p)
    measures = [
        ('Chinese withheld, own block', MAIN, iso('chinese/withheld', 1), 'acc'),
        ('Devanagari withheld, own block', MAIN, iso('devanagari_digits/withheld', 2), 'acc'),
        ('Chinese paired, end (rehearsal)', 'D-rehearsal', iso('chinese/paired'), 'acc'),
        ('Chinese paired, end (no rehearsal)', 'D-base', iso('chinese/paired'), 'acc'),
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
        ('Succession pairwise, Chinese forms', MAIN, fr('succession_pairwise', 'chinese'), 'acc'),
        ('Judgment revision: isolated, withheld', MAIN, fr('context/withheld', 'isolated'), 'acc'),
        ('Judgment revision: revised, withheld', MAIN, fr('context/withheld', 'two_sided_revised'), 'acc'),
        ('Judgment revision: swap follows context', MAIN, fr('context/withheld', 'two_sided_swap_follows_context'), 'acc'),
        ('Next-judgment coherence AUROC, collections', MAIN, fr('coherence', 'collections'), 'auroc'),
        ('Next-judgment coherence AUROC, Chinese', MAIN, fr('coherence', 'forms/chinese'), 'auroc'),
        ('Next-judgment coherence AUROC, Devanagari', MAIN, fr('coherence', 'forms/devanagari_digits'), 'auroc'),
        ('Next-judgment coherence AUROC, English', MAIN, fr('coherence', 'forms/english'), 'auroc'),
        ('Next-judgment coherence AUROC, taps', MAIN, fr('coherence', 'taps'), 'auroc'),
        ('Thai, 4 shots, after 200 updates', MAIN, lambda a, arm, s: R[a][f'{arm}/seed{s}']['transfer']['200'], 'acc'),
    ]
    rows = []
    for name, arm, get, kind in measures:
        st = [get('heads', arm, s) for s in SEEDS]; ge = [get('generic', arm, s) for s in SEEDS]
        d = [b - a for a, b in zip(st, ge)]
        rows.append({'measure': name, 'arm': arm, 'kind': kind, 'heads': st, 'generic': ge, 'diff_mean': float(np.mean(d)), 'outcome': C.classify(d, kind)})
    stored = max(abs(R['generic'][f'{MAIN}/seed{s}']['after_block'][-1]['context/withheld']['two_sided_revised'] - F['generic'][s]['context/withheld']['two_sided_revised']) for s in SEEDS)
    out = {'architecture': G.ARCH, 'reports': REPORT, 'seeds': SEEDS, 'margins': C.MARGIN, 'measures': rows, 'recomputation_max_abs_error': stored}
    with gzip.open(ROOT / 'reports' / f'generic-relations-v1-comparison{SUFFIX}.json.gz', 'wt') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    (ROOT / 'reports' / f'generic-relations-v1-comparison{SUFFIX}-runtime.json').write_text(json.dumps({
        'scope': f'nondeterministic timing; results in generic-relations-v1-comparison{SUFFIX}.json.gz', 'module_sha256': X.nc.sha256(__file__),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')), 'command': ' '.join([sys.executable, *sys.argv]),
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}, indent=1) + '\n')
    f = lambda v: f'{np.mean(v):.2f} ({", ".join(f"{x:.2f}" for x in v)})'
    print(f'| Measure | {G.ARCH} with heads | {G.ARCH} generic | Δ | Outcome |\n| --- | --- | --- | --- | --- |')
    for r in rows:
        print(f"| {r['measure']} | {f(r['heads'])} | {f(r['generic'])} | {r['diff_mean']:+.2f} | {r['outcome']} |")
    print('recomputation error', stored)


def _dig(d, path):
    for p in path:
        d = d[p]
    return d


if __name__ == '__main__':
    main()
