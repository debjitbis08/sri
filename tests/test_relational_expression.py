"""Semantics checks for src/relational_expression on an explicit finite truth table (stage 1 of
docs/relational-expression-core-design.md). The table is an oracle fixture built from labelled counts; it checks the
implementation's semantics and makes no learning or perceptual claim."""
import unittest

from src.relational_expression import (Domain, Ent, Inverse, Not, Rel, Table, Var, atom, conj, exists, precision_recall,
                                       query, rank, truth)

# Oracle fixture only: opaque entities, two renderings of some quantities, two kinds. The counts generate the table and
# are never visible to the evaluator.
ORACLE_COUNTS = {'d1': 1, 'd2a': 2, 'd2b': 2, 'd3': 3, 'd4': 4, 't2': 2, 't3': 3}
E = {k: Ent(k) for k in ORACLE_COUNTS}
ALL = Domain(tuple(E.values()))
REL = {'same_quantity': lambda a, b: a == b, 'successor': lambda a, b: b == a + 1, 'more_than': lambda a, b: a > b}
TABLE = Table({(r, (E[x], E[y])): f(ORACLE_COUNTS[x], ORACLE_COUNTS[y]) for r, f in REL.items() for x in E for y in E},
              closed=REL)
DEFS = {'predecessor': Inverse('successor'), 'less_than': Inverse('more_than')}
X, Y, Z, R, W = Var('x'), Var('y'), Var('z'), Var('r'), Var('w')


def ents(*names):
    return {(E[n],) for n in names}


def logit_scorer(rel, args):
    """A soft scorer from the same table: +1 true, -1 false."""
    return 1.0 if TABLE(rel, args) else -1.0


class TruthAndFalsehood(unittest.TestCase):
    def test_true_and_false_statements_are_both_expressions(self):
        self.assertEqual(truth(atom('same_quantity', E['d2a'], E['d2b']), TABLE)[0], True)
        self.assertEqual(truth(atom('same_quantity', E['d2a'], E['d3']), TABLE)[0], False)
        self.assertEqual(truth(Not(atom('same_quantity', E['d2a'], E['d3'])), TABLE)[0], True)

    def test_missing_fact_is_unknown_unless_relation_is_closed(self):
        open_table = Table({('named', (E['d1'], E['d2a'])): True})
        self.assertIsNone(truth(atom('named', E['d2a'], E['d3']), open_table)[0])
        self.assertIsNone(truth(Not(atom('named', E['d2a'], E['d3'])), open_table)[0])
        self.assertEqual(truth(conj(atom('named', E['d1'], E['d2a']), atom('named', E['d2a'], E['d3'])), open_table)[0], None)
        closed = Table({}, closed={'named'})
        self.assertEqual(truth(atom('named', E['d2a'], E['d3']), closed)[0], False)


class Binding(unittest.TestCase):
    def test_multiple_answers_across_renderings_and_kinds(self):
        a = query(atom('same_quantity', E['d2a'], X), (X,), {'x': ALL}, TABLE)
        self.assertEqual(set(a.true), ents('d2a', 'd2b', 't2'))
        self.assertEqual(precision_recall(a.true, ents('d2a', 'd2b', 't2')), (1.0, 1.0))
        self.assertEqual(a.atom_evaluations, len(E))

    def test_argument_reversal_uses_the_same_relation(self):
        fwd = query(atom('successor', E['d2b'], X), (X,), {'x': ALL}, TABLE)
        rev = query(atom('successor', X, E['d3']), (X,), {'x': ALL}, TABLE)
        self.assertEqual(set(fwd.true), ents('d3', 't3'))
        self.assertEqual(set(rev.true), ents('d2a', 'd2b', 't2'))
        self.assertEqual(rev.supplied_definitions_used, frozenset())

    def test_repeated_variable_has_one_binding(self):
        self.assertEqual(query(atom('successor', X, X), (X,), {'x': ALL}, TABLE).true, [])
        self.assertEqual(len(query(atom('same_quantity', X, X), (X,), {'x': ALL}, TABLE).true), len(E))

    def test_unrequested_free_variable_is_an_error(self):
        with self.assertRaises(ValueError):
            query(atom('successor', X, Y), (X,), {'x': ALL, 'y': ALL}, TABLE)

    def test_relation_identification_returns_every_holding_relation_and_records_supplied_ones(self):
        rels = Domain(tuple(Rel(n) for n in ('same_quantity', 'successor', 'predecessor', 'more_than', 'less_than')))
        a = query(atom(R, E['d2a'], E['d3']), (R,), {'r': rels}, TABLE, DEFS)
        self.assertEqual(set(a.true), {(Rel('successor'),), (Rel('less_than'),)})
        self.assertEqual(a.supplied_definitions_used, frozenset({'predecessor', 'less_than'}))


class Composition(unittest.TestCase):
    CHAIN = exists((Y,), conj(atom('successor', E['d1'], Y), atom('successor', Y, Z)))

    def test_two_step_composition_returns_endpoints_and_all_witnesses(self):
        a = query(self.CHAIN, (Z,), {'z': ALL, 'y': ALL}, TABLE)
        self.assertEqual(set(a.true), ents('d3', 't3'))
        for b in a.true:
            self.assertEqual(set(a.witnesses[b]), ents('d2a', 'd2b', 't2'))

    def test_renaming_the_bound_variable_does_not_change_meaning(self):
        renamed = exists((W,), conj(atom('successor', E['d1'], W), atom('successor', W, Z)))
        a = query(self.CHAIN, (Z,), {'z': ALL, 'y': ALL}, TABLE)
        b = query(renamed, (Z,), {'z': ALL, 'w': ALL}, TABLE)
        self.assertEqual((a.true, a.false, a.witnesses), (b.true, b.false, b.witnesses))

    def test_inner_binding_shadows_outer(self):
        inner = exists((X,), atom('successor', E['d1'], X))  # x bound here, unrelated to the requested x
        a = query(conj(atom('same_quantity', X, E['d4']), inner), (X,), {'x': ALL}, TABLE)
        self.assertEqual(set(a.true), ents('d4'))

    def test_inferred_direction_continuation_with_supplied_predecessor(self):
        rels = Domain((Rel('successor'), Rel('predecessor')))
        cont = lambda prev, cur: exists((R,), conj(atom(R, prev, cur), atom(R, cur, X)))
        down = query(cont(E['d3'], E['t2']), (X,), {'x': ALL, 'r': rels}, TABLE, DEFS)
        self.assertEqual(set(down.true), ents('d1'))
        self.assertEqual(down.witnesses[(E['d1'],)], [(Rel('predecessor'),)])
        self.assertIn('predecessor', down.supplied_definitions_used)
        up = query(cont(E['d2a'], E['t3']), (X,), {'x': ALL, 'r': rels}, TABLE, DEFS)
        self.assertEqual(set(up.true), ents('d4'))
        self.assertEqual(up.witnesses[(E['d4'],)], [(Rel('successor'),)])


class Domains(unittest.TestCase):
    def test_out_of_range_continuation_is_false_only_on_a_complete_domain(self):
        rels = Domain((Rel('successor'), Rel('predecessor')))
        f = exists((R,), conj(atom(R, E['d2a'], E['d1']), atom(R, E['d1'], X)))
        done = query(f, (X,), {'x': ALL, 'r': rels}, TABLE, DEFS)
        self.assertEqual((done.true, done.unknown), ([], []))
        partial = query(f, (X,), {'x': ALL, 'r': Domain(rels.items, complete=False)}, TABLE, DEFS)
        self.assertEqual((partial.true, partial.false), ([], []))
        self.assertEqual(len(partial.unknown), len(E))

    def test_empty_domains(self):
        f = exists((Y,), atom('successor', E['d1'], Y))
        self.assertEqual(truth(f, TABLE, domains={'y': Domain(())})[0], False)
        self.assertIsNone(truth(f, TABLE, domains={'y': Domain((), complete=False)})[0])
        a = query(atom('successor', E['d1'], X), (X,), {'x': Domain(())}, TABLE)
        self.assertEqual((a.true, a.false, a.unknown), ([], [], []))
        self.assertEqual(precision_recall(a.true, ents('d2a')), (None, 0.0))

    def test_incomplete_requested_domain_is_reported(self):
        a = query(atom('successor', E['d1'], X), (X,), {'x': Domain((E['d3'],), complete=False)}, TABLE)
        self.assertEqual((a.false, a.domain_complete), ([(E['d3'],)], False))


class Soft(unittest.TestCase):
    def test_max_min_reproduces_exact_answers_on_a_two_valued_scorer(self):
        f = Composition.CHAIN
        r = rank(f, (Z,), {'z': ALL, 'y': ALL}, logit_scorer)
        top = {b for b, s in r.ranked if s > 0}
        self.assertEqual(top, set(query(f, (Z,), {'z': ALL, 'y': ALL}, TABLE).true))
        self.assertIn(r.witnesses[(E['d3'],)], ents('d2a', 'd2b', 't2'))
        self.assertEqual(r.atom_evaluations, len(E) ** 2)  # the first hop's atoms recur in the second: d1 is in y's domain

    def test_graded_scores_rank_by_weakest_link_and_record_argmax_witness(self):
        g = {('successor', (E['d1'], E['d2a'])): 3.0, ('successor', (E['d1'], E['d2b'])): 1.0,
             ('successor', (E['d2a'], E['d3'])): 0.5, ('successor', (E['d2b'], E['d3'])): 2.0}
        scorer = lambda rel, args: g.get((rel, args), -5.0)
        dom = Domain((E['d2a'], E['d2b']))
        r = rank(Composition.CHAIN, (Z,), {'z': Domain((E['d3'], E['d4'])), 'y': dom}, scorer)
        self.assertEqual(r.ranked[0], ((E['d3'],), 1.0))  # max(min(3, .5), min(1, 2))
        self.assertEqual(r.witnesses[(E['d3'],)], (E['d2b'],))

    def test_inverse_definition_swaps_arguments_for_the_scorer(self):
        seen = []
        rank(atom('predecessor', E['d3'], X), (X,), {'x': Domain((E['d2a'],))}, lambda rel, a: seen.append((rel, a)) or 0.0, DEFS)
        self.assertEqual(seen, [('successor', (E['d2a'], E['d3']))])


if __name__ == '__main__':
    unittest.main()
