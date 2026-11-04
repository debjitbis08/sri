"""Tables for the shared output pilot (docs/shared-output-v1-contract.md): reads N1-N3 per arm and the pre-declared pilot criteria.

Usage: .venv/bin/python -m experiments.shared_output_summary [report ...]   (default reports/shared-output-v1-pilot.json)
Several reports are merged (later ones add arms). With the K arms present, their criteria (docs/shared-output-v1-k-arms.md) are printed too.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('gp', 'gp-grounded', 'common', 'common-raw', 'common-k-cand', 'common-k-shared')
GROUNDED = {'gp-grounded', 'common'}


def mode(arm):
    return 'hard' if arm in GROUNDED else 'direct'


def load(*paths):
    res = {}
    for path in paths:
        res.update(json.loads(Path(path).read_text())['results'])
    seeds = sorted({int(k.split('seed')[1]) for k in res})
    return res, seeds


def get(res, arm, seed, point, f):
    r = res.get(f'{arm}/seed{seed}')
    if r is None:
        return np.nan
    try:
        return float(f(r['snapshots'][point], arm))
    except (KeyError, TypeError):
        return np.nan


def cont(kind, hist, field='recall', m=None):
    def f(s, arm):
        c = s['shared_output']['continuation']
        return np.mean([c[f'{m or mode(arm)}/{kind}/{hist}/{d}'][field] for d in ('up', 'down')])
    return f


def cont_dir(kind, hist, d, field='recall', m=None):
    return lambda s, arm: s['shared_output']['continuation'][f'{m or mode(arm)}/{kind}/{hist}/{d}'][field]


READS = {
    'Name identification (counts 2–8)': lambda s, a: s['shared_output']['identification']['names_counts_2_8'],
    'Name identified and accepted': lambda s, a: s['shared_output']['identification']['names_identified_and_accepted'],
    'Never-taught symbols accepted': lambda s, a: s['shared_output']['identification']['unseen_accepted'],
    'N1 name, collection history, up': cont_dir('name', 'collection', 'up'),
    'N1 name, collection history, down': cont_dir('name', 'collection', 'down'),
    'N1 mean of directions': cont('name', 'collection'),
    'N1 without abstention': cont('name', 'collection', 'recall_no_abstention'),
    'N1 given correct identification': cont('name', 'collection', 'recall_given_identified'),
    'N1 soft grounding (grounded arms)': lambda s, a: cont('name', 'collection', m='soft')(s, a) if a in GROUNDED else np.nan,
    'N1 external top-participant grounding (raw arms)': lambda s, a: cont('name', 'collection', m='external-hard')(s, a) if a not in GROUNDED else np.nan,
    'Name, taps history': cont('name', 'taps'),
    'Name, name history': cont('name', 'name'),
    'Familiar (collection, collection)': cont('collection', 'collection'),
    'Familiar (collection, taps)': cont('taps', 'collection'),
    'Familiar (collection, Devanagari)': cont('devanagari', 'collection'),
    'N2 names: removed set abstains': cont('name', 'collection', 'removed_set_abstains'),
    'N2 names: reverse accepted': cont('name', 'collection', 'accept_reverse'),
    'N2 names: repeat accepted': cont('name', 'collection', 'accept_repeat'),
    'N2 names: rest accepted': cont('name', 'collection', 'accept_rest'),
    'N2 names: correct accepted': cont('name', 'collection', 'accept_correct'),
    'N2 names: AUROC against reverse and repeat': cont('name', 'collection', 'auroc_reverse_repeat'),
    'N2 familiar: removed set abstains': cont('collection', 'collection', 'removed_set_abstains'),
    'Collection geometry, between / within': lambda s, a: s['shared_output']['collection_geometry']['ratio'],
}
PRIOR = {
    'English matching, paired': lambda s, a: s['four_way']['full']['matching']['english/paired']['all'],
    'Chinese matching, paired': lambda s, a: s['four_way']['full']['matching']['chinese/paired']['all'],
    'Devanagari matching, paired': lambda s, a: s['four_way']['full']['matching']['devanagari_digits/paired']['all'],
    'Devanagari matching, withheld': lambda s, a: s['four_way']['full']['matching']['devanagari_digits/withheld']['all'],
    'Taps → Devanagari, withheld': lambda s, a: s['four_way']['full']['transfer8']['taps_to_devanagari_withheld'],
    'Comparison, Devanagari': lambda s, a: s['four_way']['full']['transfer8']['comparison_devanagari'],
    'Comparison, collections': lambda s, a: s['evaluate']['senses']['comparison']['collections'],
    'Composition, collections up 1 (distance)': lambda s, a: s['evaluate']['senses']['composition']['collections/up/1/distance'],
    'Composition, collections down 1 (distance)': lambda s, a: s['evaluate']['senses']['composition']['collections/down/1/distance'],
    'Composition, collections up 2 (distance)': lambda s, a: s['evaluate']['senses']['composition']['collections/up/2/distance'],
    'Taps → collection': lambda s, a: s['evaluate']['senses']['taps_to_collection'] if not isinstance(s['evaluate']['senses']['taps_to_collection'], dict)
    else np.mean(list(s['evaluate']['senses']['taps_to_collection'].values())),
}


def fmt(v):
    return '—' if v is None or np.isnan(v) else f'{v:.2f}'


def table(res, seeds, reads, point, arms):
    lines = ['| Read | ' + ' | '.join(f'`{a}`' for a in arms) + ' |', '|---|' + '---|' * len(arms)]
    for name, f in reads.items():
        cells = []
        for a in arms:
            v = np.array([get(res, a, s, point, f) for s in seeds])
            cells.append('—' if np.all(np.isnan(v)) else f'{np.nanmean(v):.2f} ({np.nanmin(v):.2f}–{np.nanmax(v):.2f})')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def paired(res, seeds, f, a, b, point):
    return np.array([get(res, a, s, point, f) - get(res, b, s, point, f) for s in seeds])


def criteria(res, seeds):
    out = []
    ok = True
    for a in ARMS:
        ident = np.nanmean([get(res, a, s, 'block3', READS['Name identification (counts 2–8)']) for s in seeds])
        fam = np.nanmean([get(res, a, s, 'delay', READS['Familiar (collection, collection)']) for s in seeds])
        out.append(f'- Gate `{a}`: name identification at block 3 {fmt(ident)} (≥ 0.80), familiar continuation at delay {fmt(fam)} (≥ 0.60)'
                   f' → {"pass" if ident >= .8 and fam >= .6 else "fail"}')
    d = paired(res, seeds, READS['N1 mean of directions'], 'common', 'gp', 'delay')
    t = np.nanmean(d) >= .10 and (d > 0).sum() >= 3
    out.append(f'- Transfer, `common` − `gp` at delay: {fmt(np.nanmean(d))} paired (seeds {", ".join(fmt(x) for x in d)}) → {"pass" if t else "fail"}')
    fa = -paired(res, seeds, READS['N2 names: removed set abstains'], 'common', 'gp', 'delay')
    r = np.nanmean(fa) <= .05
    out.append(f'- Rejection, removed-set false acceptance `common` − `gp`: {fmt(np.nanmean(fa))} (≤ +0.05) → {"pass" if r else "fail"}')
    prior_ok = True
    worst = []
    for name, f in {**{k: READS[k] for k in ('Familiar (collection, collection)', 'Familiar (collection, taps)', 'Familiar (collection, Devanagari)')}, **PRIOR}.items():
        dd = np.nanmean(paired(res, seeds, f, 'common', 'gp', 'delay'))
        if dd < -.05:
            prior_ok = False
            worst.append(f'{name} {fmt(dd)}')
    e = np.nanmean(paired(res, seeds, PRIOR['English matching, paired'], 'common', 'gp', 'relearn'))
    if e < -.05:
        prior_ok = False
        worst.append(f'English after relearning {fmt(e)}')
    out.append(f'- Prior abilities within 0.05 of `gp`: {"pass" if prior_ok else "fail: " + "; ".join(worst)}')
    for (a, b, what) in (('gp-grounded', 'gp', 'grounding with the existing readers'), ('common', 'gp-grounded', 'one shared read'), ('common-raw', 'gp', 'sharing alone')):
        dd = paired(res, seeds, READS['N1 mean of directions'], a, b, 'delay')
        out.append(f'- Attribution, {what} (`{a}` − `{b}`): N1 {fmt(np.nanmean(dd))} (seeds {", ".join(fmt(x) for x in dd)})')
    return '\n'.join(out)


def k_criteria(res, seeds):
    """Pre-declared criteria of the K arms, at delay."""
    out = []
    n1 = READS['N1 mean of directions']
    for a in ('common-k-cand', 'common-k-shared'):
        ident = np.nanmean([get(res, a, s, 'block3', READS['Name identification (counts 2–8)']) for s in seeds])
        fam = np.nanmean([get(res, a, s, 'delay', READS['Familiar (collection, collection)']) for s in seeds])
        fam_raw = np.nanmean([get(res, 'common-raw', s, 'delay', READS['Familiar (collection, collection)']) for s in seeds])
        out.append(f'- Gate `{a}`: identification at block 3 {fmt(ident)} (≥ 0.80, else read N1 given identification), familiar continuation at delay '
                   f'{fmt(fam)} (≥ 0.60, and ≥ `common-raw` − 0.10 = {fmt(fam_raw - .1)}) → {"pass" if ident >= .8 and fam >= .6 and fam >= fam_raw - .1 else "fail"}')
    for a, b in (('common-k-shared', 'common-raw'), ('common-k-shared', 'gp'), ('common-k-cand', 'common-raw'), ('common-k-shared', 'common-k-cand')):
        d = paired(res, seeds, n1, a, b, 'delay')
        verdict = f' → {"pass" if np.nanmean(d) >= .10 and (d > 0).sum() >= 3 else "fail"}' if (a, b) == ('common-k-shared', 'common-raw') else ''
        out.append(f'- N1 `{a}` − `{b}` at delay: {fmt(np.nanmean(d))} (seeds {", ".join(fmt(x) for x in d)}){verdict}')
    fa = -paired(res, seeds, READS['N2 names: removed set abstains'], 'common-k-shared', 'gp', 'delay')
    out.append(f'- Rejection, removed-set false acceptance `common-k-shared` − `gp`: {fmt(np.nanmean(fa))} (≤ +0.05) → {"pass" if np.nanmean(fa) <= .05 else "fail"}')
    worst = []
    for name, f in {**{k: READS[k] for k in ('Familiar (collection, collection)', 'Familiar (collection, taps)', 'Familiar (collection, Devanagari)')}, **PRIOR}.items():
        dd = np.nanmean(paired(res, seeds, f, 'common-k-shared', 'gp', 'delay'))
        if dd < -.05:
            worst.append(f'{name} {fmt(dd)}')
    e = np.nanmean(paired(res, seeds, PRIOR['English matching, paired'], 'common-k-shared', 'gp', 'relearn'))
    if e < -.05:
        worst.append(f'English after relearning {fmt(e)}')
    out.append(f'- Prior abilities of `common-k-shared` within 0.05 of `gp`: {"pass" if not worst else "fail: " + "; ".join(worst)}')
    for a in ('common-k-cand', 'common-k-shared'):
        for point in ('block3', 'delay'):
            v = {sp: np.nanmean([get(res, a, s, point, lambda r, _, sp=sp: r['shared_output']['substitutability'][sp]['names_nearest_own_count_2_8']) for s in seeds])
                 for sp in ('core', 'key')}
            out.append(f'- Substitutability `{a}` at {point}, names nearest own count (2–8): core {fmt(v["core"])}, K space {fmt(v["key"])}')
    return '\n'.join(out)


def main():
    warnings.simplefilter('ignore', RuntimeWarning)  # arms or reads missing from a report give empty means
    paths = sys.argv[1:] or [ROOT / 'reports' / 'shared-output-v1-pilot.json']
    res, seeds = load(*paths)
    arms = [a for a in ARMS if any(f'{a}/seed{s}' in res for s in seeds)]
    print(f'Seeds {seeds}; mean (range over seeds). Grounded arms read with hard grounding, raw arms directly.\n')
    for point in ('block3', 'delay', 'relearn'):
        print(f'### {point}\n\n{table(res, seeds, READS, point, arms)}\n\n{table(res, seeds, PRIOR, point, arms)}\n')
    print('### Pilot criteria\n\n' + criteria(res, seeds))
    if any(f'common-k-shared/seed{s}' in res for s in seeds):
        print('\n### K-arm criteria\n\n' + k_criteria(res, seeds))


if __name__ == '__main__':
    main()
