"""Run learned compatibility v1 over the declared relations, splits, orders, rules and core sizes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from src.compat_learning import (
    CONTROLS, RELATIONS, STAR_RULES, components, evaluate, prefix_max_components, splits, train,
)


ROOT = Path(__file__).resolve().parents[1]
CORE_SIZES = (2, 3, 4, 8)
ORDERS = 20
METRICS = ('heldout_recall', 'seen_recall', 'precision', 'invalid_responses', 'exact_cues',
           'single_cues_unique', 'multi_cues_multiple', 'cues_needing_dynamic_core')


def orders(relation_name, family, index, train_pairs):
    for o in range(ORDERS):
        order = list(train_pairs)
        random.Random(f'{relation_name}/{family}/{index}/{o}').shuffle(order)
        yield order


def state_key(learner):
    return (learner.kind, learner.n, tuple((frozenset(a), frozenset(b))
                                           for a, b in zip(learner.A, learner.B)),
            frozenset(learner.pairs), frozenset((a, b) for a, b, _ in learner.triples))


class Aggregate:
    def __init__(self):
        self.sums, self.counts = defaultdict(float), defaultdict(int)
        self.runs = self.perfect = self.forced_runs = self.merges = 0
        self.modes_used = 0

    def add(self, row, learner):
        self.runs += 1
        for m in METRICS:
            if row[m] is not None:
                self.sums[m] += row[m]; self.counts[m] += 1
        self.perfect += row['heldout_recall'] == 1 and row['precision'] == 1
        self.forced_runs += learner.forced > 0
        self.merges += learner.merges
        self.modes_used += sum(1 for a, b in zip(learner.A, learner.B) if a or b)

    def report(self):
        return dict(runs=self.runs,
                    means={m: round(self.sums[m]/self.counts[m], 6) for m in METRICS if self.counts[m]},
                    fraction_perfect_generalization=round(self.perfect/self.runs, 6),
                    runs_with_forced_reuse=self.forced_runs, total_merges=self.merges,
                    mean_core_modes_used=round(self.modes_used/self.runs, 6))


def render():
    agg = defaultdict(Aggregate)
    checks = defaultdict(int)
    split_counts = {}
    cache = {}
    for rname, relation in RELATIONS.items():
        families = splits(relation)
        split_counts[rname] = {f: dict(admissible=len(h), inadmissible=bad)
                               for f, (h, bad) in families.items()}
        for family, (held_sets, _) in families.items():
            for index, held in enumerate(held_sets):
                train_pairs = sorted(relation - held)
                comps = components(train_pairs)
                for o, order in enumerate(orders(rname, family, index, train_pairs)):
                    prefix_max = prefix_max_components(order)
                    learned = {}
                    for kind in STAR_RULES + (CONTROLS if o == 0 else ()):
                        for n in (CORE_SIZES if kind in STAR_RULES else (8,)):
                            learner = train(kind, n, order)
                            key = (state_key(learner), rname, held)
                            if key not in cache:
                                cache[key] = evaluate(learner, relation, held)
                            row = cache[key]
                            learned[(kind, n)] = (learner, row)
                            agg[(rname, family, kind, n if kind in STAR_RULES else None)].add(row, learner)
                            checks['response_violations'] += row['response_proposition_violations']
                            checks['runs'] += 1
                            if kind in CONTROLS:
                                checks['control_runs'] += 1
                                checks['control_heldout_recovered'] += row['heldout_recall'] > 0
                            if kind == 'merge':
                                got = {(frozenset(a), frozenset(b)) for a, b in zip(learner.A, learner.B) if a or b}
                                if n >= len(comps):  # condition as originally declared
                                    checks['merge_declared_condition_runs'] += 1
                                    checks['merge_declared_condition_mismatch'] += got != comps
                                if n >= prefix_max:  # corrected condition
                                    checks['merge_corrected_condition_runs'] += 1
                                    checks['merge_corrected_condition_mismatch'] += got != comps
                                    checks['merge_corrected_condition_forced'] += learner.forced > 0
                    for n in CORE_SIZES:
                        jr, mr = learned[('join', n)][1]['resp'], learned[('merge', n)][1]['resp']
                        forced = learned[('join', n)][0].forced or learned[('merge', n)][0].forced
                        bad = any(not jr[k] <= mr[k] for k in jr)
                        checks['join_subset_checks'] += 1
                        checks['join_not_subset'] += bad
                        checks['join_subset_checks_without_forced_reuse'] += not forced
                        checks['join_not_subset_without_forced_reuse'] += bad and not forced
    files = ['src/compat_learning.py', 'experiments/learned_compatibility.py',
             'docs/learned-compatibility-v1-contract.md']
    return dict(
        experiment='learned-compatibility-v1',
        contract='docs/learned-compatibility-v1-contract.md',
        relations={k: sorted(map(list, v)) for k, v in RELATIONS.items()},
        relation_components={k: len(components(v)) for k, v in RELATIONS.items()},
        grid=dict(core_sizes=CORE_SIZES, orders=ORDERS),
        splits=split_counts,
        proposition_checks=dict(checks),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        results=[dict(relation=r, split=f, learner=k, n=n, **a.report())
                 for (r, f, k, n), a in sorted(agg.items(), key=str)],
    )


def main():
    start = time.perf_counter()
    report = render()
    elapsed = time.perf_counter() - start
    (ROOT/'reports/learned-compatibility-v1.json').write_text(json.dumps(report, indent=1) + '\n')
    (ROOT/'reports/learned-compatibility-v1-runtime.json').write_text(json.dumps(dict(
        seconds=round(elapsed, 3), python=platform.python_version(), machine=platform.machine(),
        runs=report['proposition_checks']['runs']), indent=1) + '\n')
    print(report['proposition_checks'], round(elapsed, 1), 's')


if __name__ == '__main__':
    main()
