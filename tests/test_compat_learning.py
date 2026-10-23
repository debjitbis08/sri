import hashlib
import json
from pathlib import Path
import unittest

from src.compat_learning import (
    RELATIONS, attractor_responses, components, evaluate, predicted_responses,
    prefix_max_components, splits, train,
)


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = RELATIONS['blocks']


class LearnedCompatibilityTests(unittest.TestCase):
    def test_controls_never_recover_held_out_pairs(self):
        held = frozenset([(0, 1), (3, 4)])
        for kind in ('memorize', 'ternary', 'enumerate'):
            row = evaluate(train(kind, 8, sorted(BLOCKS - held)), BLOCKS, held)
            self.assertEqual(row['heldout_recall'], 0)

    def test_merge_generalizes_blocks_with_unique_and_multiple_responses(self):
        held = frozenset([(0, 1)])
        learner = train('merge', 8, sorted(BLOCKS - held))
        row = evaluate(learner, BLOCKS, held)
        self.assertEqual((row['heldout_recall'], row['precision'], row['exact_cues']), (1, 1, 1))
        self.assertEqual(attractor_responses(learner, 5, 'A'), {5})
        self.assertEqual(attractor_responses(learner, 0, 'A'), {0, 1, 2})

    def test_merge_learns_components_when_core_covers_every_prefix(self):
        order = sorted(BLOCKS - {(0, 1)})
        learner = train('merge', prefix_max_components(order), order)
        learned = {(frozenset(a), frozenset(b)) for a, b in zip(learner.A, learner.B) if a or b}
        self.assertEqual(learned, components(order))

    def test_originally_declared_merge_condition_has_a_counterexample(self):
        # Final graph has 3 components = n, but a prefix had 4, forcing a wrong merge.
        order = [(0, 0), (1, 1), (3, 3), (5, 5), (0, 1), (1, 0), (0, 2), (2, 0), (2, 2),
                 (3, 4), (4, 3), (4, 4), (1, 2)]
        learner = train('merge', 3, order)
        self.assertEqual((len(components(order)), prefix_max_components(order), learner.forced), (3, 4, 1))
        self.assertIn(5, attractor_responses(learner, 0, 'A'))

    def test_join_containment_does_not_imply_higher_precision(self):
        # Review counterexample: join's responses lie inside merge's, yet its precision is lower.
        from experiments.learned_compatibility import orders
        rel = RELATIONS['overlap']
        held = splits(rel)['leave_one_out'][0][13]
        order = list(orders('overlap', 'leave_one_out', 13, sorted(rel - held)))[9]
        join, merge = train('join', 8, order), train('merge', 8, order)
        self.assertEqual((held, join.forced, merge.forced), (frozenset([(3, 4)]), 0, 0))
        for side, used in (('A', range(6)), ('B', range(5))):
            for c in used:
                self.assertLessEqual(attractor_responses(join, c, side), attractor_responses(merge, c, side))
        jr, mr = evaluate(join, rel, held), evaluate(merge, rel, held)
        self.assertAlmostEqual(jr['precision'], 19/29)
        self.assertAlmostEqual(mr['precision'], 20/30)
        self.assertEqual((jr['heldout_recall'], mr['heldout_recall']), (0, 1))

    def test_merge_overgeneralizes_overlap_to_its_bounding_rectangle(self):
        rel = RELATIONS['overlap']
        row = evaluate(train('merge', 8, sorted(rel)), rel, frozenset())
        self.assertAlmostEqual(row['precision'], 20/30)

    def test_response_proposition_scope_excludes_untrained_cues(self):
        learner = train('join', 8, sorted(RELATIONS['overlap']))
        self.assertEqual(predicted_responses(learner, 5, 'B'), set())
        self.assertEqual(attractor_responses(learner, 5, 'B'), set(range(6)))

    def test_half_splits_are_distinct_and_admissible(self):
        held, _ = splits(BLOCKS)['half']
        self.assertEqual(len(set(held)), 100)

    def test_frozen_report_propositions(self):
        report = json.loads((ROOT/'reports/learned-compatibility-v1.json').read_text())
        for path, digest in report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest, path)
        c = report['proposition_checks']
        self.assertEqual((c['response_violations'], c['control_heldout_recovered'],
                          c['merge_corrected_condition_mismatch'],
                          c['join_not_subset_without_forced_reuse']), (0, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()
