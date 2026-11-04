"""Tables for experiments.sense_memory_replay (docs/sense-memory-replay-v1-design.md).
Usage: python -m experiments.sense_memory_replay_summary reports/sense-memory-replay-v1-pilot.json[.gz] [reports/sense-core-v1-calibrated.json.gz]
       python -m experiments.sense_memory_replay_summary reports/sense-memory-replay-v1-combined.json[.gz]   (memory and core together)
The second report, step 8's calibrated core, is the reference `no-replay` must reproduce.
"""
import sys

import numpy as np

from experiments import sense_core_summary as SS

POINTS = ('block1', 'block2', 'block3', 'delay', 'relearn')
READS = {'r3': 'taps', 'r2': 'dots', 'r4': 'comparison'}
RETENTION = ('senses taps_to_collection', 'senses comparison/collections', 'senses comparison/taps', 'matching devanagari_digits/paired',
             'matching devanagari_digits/withheld', 'transfer chinese_withheld_matching', 'transfer devanagari_withheld_matching',
             'transfer taps_to_chinese_withheld', 'transfer taps_to_devanagari_withheld', 'transfer chinese_to_taps_withheld',
             'transfer comparison_chinese', 'transfer comparison_devanagari', 'transfer comparison_english')


def cell(xs):
    return SS.cell(xs)


def signed(xs):
    xs = [x for x in xs if x is not None]
    return '—' if not xs else f'{np.mean(xs):+.3f} ({min(xs):+.3f} to {max(xs):+.3f})'


def measures(snap):
    m = SS.measures(snap)
    s = snap['evaluate']['senses']
    m['senses taps_to_collection'] = s['taps_to_collection']
    for k, v in s['comparison'].items():
        if isinstance(v, (int, float)):
            m[f'senses comparison/{k}'] = v
    return m


def main(path, ref=None):
    d = SS.load(path)
    R, arms, seeds = d['results'], d['arms'], d['seeds']
    run = lambda a, s: R.get(f'{a}/seed{s}')

    print('## memory and buffer')
    for s in seeds:
        r = run(arms[0], s)
        if r:
            fm = r['smr']['final_memory']
            print(f'seed {s}: {fm["writes"]} writes -> {fm["units"]} units ({fm["units"] / fm["writes"]:.2f} per write, {fm["units_per_fact"]:.1f} per fact); '
                  f'memory {fm["total"] / 1e6:.2f} MB total; buffer capacity {r["smr"]["buffer_capacity"]} pairs '
                  f'({"binds" if r["smr"]["buffer_capacity"] < fm["writes"] else "holds every write"})')
    for a in arms:
        for s in seeds[:1]:
            r = run(a, s)
            if r:
                print(a, {p: (b['memory_units'], b['writes'], round(b['memory']['total'] / 1e6, 2), b['buffer_items']) for p, b in r['smr']['bytes'].items()})

    print('\n## acquisition, memory removed: fresh renderings of each set through the core alone; mean (range) over seeds')
    for name in ('A', 'B'):
        for k, label in READS.items():
            print(f'\n### set {name}, {label}')
            print(f'{"arm":28s} ' + ' '.join(f'{p:>18s}' for p in POINTS))
            for a in arms:
                vals = {p: [run(a, s)['snapshots'][p]['smr'][f'{name}|core-only'][k] for s in seeds if run(a, s)] for p in POINTS}
                print(f'{a:28s} ' + ' '.join(f'{cell(vals[p]):>18s}' for p in POINTS))

    print('\n## with memory present (secondary): stage 1 end-to-end read at tau')
    for name in ('A', 'B'):
        for k, label in READS.items():
            row = []
            for a in arms:
                row.append(f'{a}: ' + ' '.join(f'{p} {cell([run(a, s)["snapshots"][p]["smr"].get(f"{name}|with-memory", {}).get(k) for s in seeds if run(a, s)])}' for p in ('block3', 'relearn')))
            print(f'set {name} {label}: ' + ' | '.join(row))

    print('\n## retention, memory removed: paired difference from no-replay (mean, range over seeds)')
    base = {s: {p: measures(run('no-replay', s)['snapshots'][p]) for p in POINTS} for s in seeds if run('no-replay', s)}
    for p in ('block3', 'delay', 'relearn'):
        print(f'\n### {p}')
        print(f'{"measure":44s} {"no-replay":>18s} ' + ' '.join(f'{a:>26s}' for a in arms[1:]))
        for m in RETENTION:
            nr = [base[s][p].get(m) for s in base]
            diffs = []
            for a in arms[1:]:
                xs = [measures(run(a, s)['snapshots'][p]).get(m) - base[s][p].get(m) for s in base if run(a, s)]
                diffs.append(signed(xs))
            print(f'{m:44s} {cell(nr):>18s} ' + ' '.join(f'{x:>26s}' for x in diffs))

    print('\n## slot chosen by the core: accuracy against the true (majority) count, by phase and set; n replays')
    for a in arms[1:]:
        acc = {}
        for s in seeds:
            r = run(a, s)
            if r:
                for k, v in r['smr']['slot_accuracy'].items():
                    acc.setdefault(k, []).append(v['accuracy'])
        print(a, {k: cell(v) for k, v in sorted(acc.items())})
        for s in seeds[:1]:
            r = run(a, s)
            if r:
                sm = r['smr']
                print(f'   seed {s}: replays {sm["replays"]}, from units holding more than one fact {sm["mixed_fact_replays"]}, more than one count {sm["mixed_count_replays"]}; '
                      f'lambda {np.mean(sm["lambda_every_100"]):.3f}, data matching items {np.mean(sm["data_match_items_every_100"]):.0f}')

    print('\n## exposures: share of replays per count (min-max over counts), and per-fact coefficient of variation')
    for a in arms[1:]:
        cv, lo, hi = [], [], []
        for s in seeds:
            r = run(a, s)
            if r:
                c = np.array(r['smr']['exposure_per_count'], float)
                f = np.array(r['smr']['exposure_per_fact'], float)
                lo.append(c.min() / c.sum()); hi.append(c.max() / c.sum()); cv.append(f.std() / f.mean())
        print(f'{a:28s} per count {cell(lo)} to {cell(hi)}; per fact CV {cell(cv)}')

    if ref:
        L = SS.load(ref)['results']
        same = n = 0
        for s in seeds:
            mine, other = run('no-replay', s), L.get(f'joined-mlp-calibrated/seed{s}')
            if not mine or not other:
                continue
            for p in ('block1', 'block2', 'block3', 'delay', 'relearn'):
                a, b = SS.measures(mine['snapshots'][p]), SS.measures(other['snapshots'][p])
                for k in b:
                    n += 1
                    same += abs(a.get(k, np.nan) - b[k]) < 1e-9
        print(f'\nreference: no-replay equals step 8 joined-mlp-calibrated in {same} of {n} measures')




def combined(path):
    """Tables for experiments.sense_memory_replay_combined: memory and core together, by arm, set and checkpoint."""
    d = SS.load(path)
    R, arms, seeds, points = d['results'], d['arms'], d['seeds'], d['points']
    reads = (('r3', 'symbol -> taps'), ('r3_taps_to_symbol', 'taps -> symbol'), ('r2', 'symbol -> dots'), ('r4', 'comparison'))
    for name, label in (('A', 'old (set A, written in block 2)'), ('B', 'new (set B, written in block 3)')):
        print(f'\n## {label}: retrieval {cell([R[f"{arms[0]}/seed{s}"][points[0]][name]["R0_retrieval"] for s in seeds])} (the same memory in every arm)')
        for mode in ('end_to_end', 'bypass', 'core_only'):
            for k, rl in reads:
                print(f'{mode:11s} {rl:15s} ' + ' | '.join(f'{a}: ' + ' '.join(cell([R[f"{a}/seed{s}"][p][name][mode][k] for s in seeds]) for p in points) for a in arms))
        for k, rl in reads[:1] + reads[2:]:
            xs = {a: [(R[f"{a}/seed{s}"][p][name]['retrieved'][k]['end_to_end'] or 0) - (R[f"{a}/seed{s}"][p][name]['retrieved'][k]['bypass'] or 0) for s in seeds for p in points] for a in arms}
            print(f'retrieved, end to end - bypass, {rl}: ' + ' | '.join(f'{a} {signed(v)}' for a, v in xs.items()))
    print('\n## the core\'s own cross-sense reading, both directions (no memory)')
    for k in ('collections_to_taps', 'taps_to_collections'):
        print(f'{k:22s} ' + ' | '.join(f'{a}: ' + ' '.join(cell([R[f"{a}/seed{s}"][p]['core_own'][k] for s in seeds]) for p in points) for a in arms))
    print('\n## paired against no-replay, end to end (mean over seeds)')
    for name in ('A', 'B'):
        for k, rl in reads:
            print(f'{name} {rl:15s} ' + ' | '.join(f'{a}: ' + ' '.join(signed([R[f"{a}/seed{s}"][p][name]["end_to_end"][k] - R[f"no-replay/seed{s}"][p][name]["end_to_end"][k] for s in seeds]) for p in points) for a in arms[1:]))
    for k in ('collections_to_taps', 'taps_to_collections'):
        print(f'own {k:22s} ' + ' | '.join(f'{a}: ' + ' '.join(signed([R[f"{a}/seed{s}"][p]['core_own'][k] - R[f"no-replay/seed{s}"][p]['core_own'][k] for s in seeds]) for p in points) for a in arms[1:]))


if __name__ == '__main__':
    (combined if 'combined' in sys.argv[1] else main)(*sys.argv[1:])
