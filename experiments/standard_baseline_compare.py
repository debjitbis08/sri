"""Architecture audit stage 1: paired comparison of the standard network with the SRI pilots (docs/architecture-audit.md).

Reads the frozen SRI pilot reports and the standard-network reports written by experiments.standard_baseline, and
compares each headline measure seed by seed. Margins are the audit's proposed values (not yet decided): 0.05 for
accuracies, 0.03 for AUROCs. Outcome per measure: equivalent (every paired difference within the margin), better or
worse (mean difference beyond the margin with every seed in the same direction), otherwise inconclusive.

Usage: .venv/bin/python -m experiments.standard_baseline_compare
"""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (100, 101, 102)
MARGIN = {'acc': .05, 'auroc': .03}
REPORTS = {
    'part-d': ('number-construction-v1-part-d-pilot', 'number-construction-v1-standard-part-d-pilot'),
    'multistep': ('number-construction-v1-multistep-pilot', 'number-construction-v1-standard-multistep-pilot'),
    'composition-m': ('number-construction-v1-composition-pilot-multistep-pilot',
                      'number-construction-v1-composition-pilot-standard-multistep-pilot'),
    'projection-m': ('number-construction-v1-projection-pilot-multistep-pilot',
                     'number-construction-v1-projection-pilot-standard-multistep-pilot'),
    'imagery': ('number-construction-v1-imagery-pilot', 'number-construction-v1-standard-imagery-pilot'),
}
END, OWN = -1, {'english': 0, 'chinese': 1, 'devanagari_digits': 2}


def load(name):
    with gzip.open(ROOT / 'reports' / f'{name}-results.json.gz', 'rt') as f:
        return json.load(f)['results']


def iso(key, block=END):
    return lambda r: r['after_block'][block]['isolated'][key]


def senses(*path):
    def get(r):
        d = r['after_block'][END]['senses']
        for p in path:
            d = d[p]
        return d
    return get


def end(*path):
    def get(r):
        d = r['after_block'][END]
        for p in path:
            d = d[p]
        return d
    return get


# (result, measure, report, arm, getter, kind)
MEASURES = [
    ('1 withheld forms', 'Chinese withheld, own block', 'part-d', 'D-varied+rehearsal', iso('chinese/withheld', 1), 'acc'),
    ('1 withheld forms', 'Devanagari withheld, own block', 'part-d', 'D-varied+rehearsal', iso('devanagari_digits/withheld', 2), 'acc'),
    ('1 withheld forms', 'Devanagari withheld, own block (multi-step)', 'multistep', 'M-multistep', iso('devanagari_digits/withheld', 2), 'acc'),
    ('2 revision', 'Two-sided revision, withheld, end', 'part-d', 'D-varied+rehearsal', end('context/withheld', 'two_sided_revised'), 'acc'),
    ('2 revision', 'Isolated, same items', 'part-d', 'D-varied+rehearsal', end('context/withheld', 'isolated'), 'acc'),
    ('2 revision', 'Swap follows context', 'part-d', 'D-varied+rehearsal', end('context/withheld', 'two_sided_swap_follows_context'), 'acc'),
    ('3 rehearsal', 'English, end', 'part-d', 'D-rehearsal', iso('english/paired'), 'acc'),
    ('3 rehearsal', 'Chinese paired, end', 'part-d', 'D-rehearsal', iso('chinese/paired'), 'acc'),
    ('3 rehearsal', 'Chinese paired, end (no rehearsal)', 'part-d', 'D-base', iso('chinese/paired'), 'acc'),
    ('4 varied contexts', 'Chinese withheld, end (varied)', 'part-d', 'D-varied+rehearsal', iso('chinese/withheld'), 'acc'),
    ('4 varied contexts', 'Chinese withheld, end (repeated)', 'part-d', 'D-repeated+rehearsal', iso('chinese/withheld'), 'acc'),
    ('4 varied contexts', 'Devanagari withheld, own block (repeated)', 'part-d', 'D-repeated+rehearsal', iso('devanagari_digits/withheld', 2), 'acc'),
    ('5 untaught composition', 'taps → Chinese form, paired', 'part-d', 'D-varied+rehearsal', senses('cross_sense', 'taps_to_form/chinese/paired'), 'acc'),
    ('5 untaught composition', 'taps → Chinese form, withheld', 'part-d', 'D-varied+rehearsal', senses('cross_sense', 'taps_to_form/chinese/withheld'), 'acc'),
    ('5 untaught composition', 'taps → Devanagari form, withheld', 'part-d', 'D-varied+rehearsal', senses('cross_sense', 'taps_to_form/devanagari_digits/withheld'), 'acc'),
    ('5 untaught composition', 'form → taps, Chinese withheld', 'part-d', 'D-varied+rehearsal', senses('cross_sense', 'form_to_taps/chinese/withheld'), 'acc'),
    ('5 untaught composition', 'Comparison, English forms', 'part-d', 'D-varied+rehearsal', senses('comparison', 'forms/english'), 'acc'),
    ('5 untaught composition', 'Comparison, Chinese forms', 'part-d', 'D-varied+rehearsal', senses('comparison', 'forms/chinese'), 'acc'),
    ('5 untaught composition', 'Comparison, Devanagari forms', 'part-d', 'D-varied+rehearsal', senses('comparison', 'forms/devanagari_digits'), 'acc'),
    ('6 violations', 'Coherence AUROC, collections', 'part-d', 'D-varied+rehearsal', senses('coherence_auroc_successor_residual', 'collections'), 'auroc'),
    ('6 violations', 'Coherence AUROC, Chinese', 'part-d', 'D-varied+rehearsal', senses('coherence_auroc_successor_residual', 'forms/chinese'), 'auroc'),
    ('6 violations', 'Coherence AUROC, Devanagari', 'part-d', 'D-varied+rehearsal', senses('coherence_auroc_successor_residual', 'forms/devanagari_digits'), 'auroc'),
    ('6 violations', 'Coherence AUROC, English', 'part-d', 'D-varied+rehearsal', senses('coherence_auroc_successor_residual', 'forms/english'), 'auroc'),
    ('7 imagery', 'Collection prototype → image, nearest count', 'imagery', 'D-varied+rehearsal', lambda r: r['probe']['collections/image']['nn_top1_count'], 'acc'),
    ('7 imagery', 'Taps prototype → image, nearest count', 'imagery', 'D-varied+rehearsal', lambda r: r['probe']['taps/image']['nn_top1_count'], 'acc'),
    ('7 imagery', 'Chinese prototype → taps, nearest count', 'imagery', 'D-varied+rehearsal', lambda r: r['probe']['forms/chinese/sequence']['nn_top1_count'], 'acc'),
    ('7 imagery', 'Imagined successor → image, nearest count', 'imagery', 'D-varied+rehearsal', lambda r: r['probe']['imagined_successor/image']['nn_top1_count'], 'acc'),
    ('7 imagery', 'Gap → image, nearest count', 'imagery', 'D-varied+rehearsal', lambda r: r['probe']['gap/image']['nn_top1_count'], 'acc'),
    ('7 imagery', 'Random state → image (control)', 'imagery', 'D-varied+rehearsal', lambda r: r['probe']['random/image']['nn_top1_count'], 'acc'),
    ('8 composition', 'Forward strict crossing k=2, control', 'composition-m', 'M-control', lambda r: r['composed/add/critical/neither/2'], 'acc'),
    ('8 composition', 'Forward strict crossing k=3, control', 'composition-m', 'M-control', lambda r: r['composed/add/critical/neither/3'], 'acc'),
    ('8 composition', 'Forward strict crossing k=2, multi-step', 'composition-m', 'M-multistep', lambda r: r['composed/add/critical/neither/2'], 'acc'),
    ('8 composition', 'Forward strict crossing k=3, multi-step', 'composition-m', 'M-multistep', lambda r: r['composed/add/critical/neither/3'], 'acc'),
    ('8 composition', 'Shown result k=3, multi-step', 'composition-m', 'M-multistep', lambda r: r['shown/add/critical/neither/3'], 'acc'),
    ('8 composition', 'Advance ratio k=1, collections, control', 'projection-m', 'M-control', lambda r: r['kinds']['collections']['add/1']['advance_ratio'], 'ratio'),
    ('8 composition', 'Advance ratio k=1, collections, multi-step', 'projection-m', 'M-multistep', lambda r: r['kinds']['collections']['add/1']['advance_ratio'], 'ratio'),
]

# effects within each architecture: (name, measure index of the treated arm, measure index of the comparison arm)
EFFECTS = [
    ('Rehearsal: Chinese paired, end (rehearsal − base)', 7, 8),
    ('Varied − repeated: Chinese withheld, end', 9, 10),
    ('Revision − isolated: withheld, end', 3, 4),
    ('Multi-step − control: strict crossing k=3', 32, 30),
    ('Multi-step − control: advance ratio k=1', 35, 34),
]


def classify(diffs, kind):
    if kind not in MARGIN:
        return 'descriptive'
    d, m = np.array(diffs), MARGIN[kind]
    if np.all(np.abs(d) <= m):
        return 'equivalent'
    if abs(d.mean()) > m and (np.all(d > 0) or np.all(d < 0)):
        return 'better' if d.mean() > 0 else 'worse'
    return 'inconclusive'


def main():
    data = {k: (load(a), load(b)) for k, (a, b) in REPORTS.items()}
    rows, values = [], []
    for res, name, rep, arm, get, kind in MEASURES:
        sri = [get(data[rep][0][f'{arm}/seed{s}']) for s in SEEDS]
        std = [get(data[rep][1][f'{arm}/seed{s}']) for s in SEEDS]
        values.append((sri, std))
        diffs = [b - a for a, b in zip(sri, std)]
        rows.append({'result': res, 'measure': name, 'arm': arm, 'kind': kind, 'sri': sri, 'standard': std,
                     'diff_mean': float(np.mean(diffs)), 'outcome': classify(diffs, kind)})
    effects = []
    for name, i, j in EFFECTS:
        e = {a: [values[i][a][s] - values[j][a][s] for s in range(len(SEEDS))] for a in (0, 1)}
        effects.append({'effect': name, 'sri': e[0], 'standard': e[1]})
    out = {'margins': MARGIN, 'seeds': SEEDS, 'reports': REPORTS, 'measures': rows, 'effects': effects}
    path = ROOT / 'reports' / 'architecture-audit-stage1-comparison.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    f = lambda v: f'{np.mean(v):.2f} ({", ".join(f"{x:.2f}" for x in v)})'
    print('| Result | Measure | SRI | Standard | Δ | Outcome |\n| --- | --- | --- | --- | --- | --- |')
    for r in rows:
        print(f"| {r['result']} | {r['measure']} | {f(r['sri'])} | {f(r['standard'])} | {r['diff_mean']:+.2f} | {r['outcome']} |")
    print('\n| Effect | SRI | Standard |\n| --- | --- | --- |')
    for e in effects:
        print(f"| {e['effect']} | {f(e['sri'])} | {f(e['standard'])} |")


if __name__ == '__main__':
    main()
