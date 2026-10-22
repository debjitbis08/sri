"""Exact matched-ablation search: python -m experiments.recurrence_necessity."""
from itertools import product
import json
from pathlib import Path
from src.recurrent import (Network, SCHEDULES, EDGE_MASKS, INPUTS, INTERNAL_STATES,
                           FULL, orbit, attractors, recurrence_test,
                           stable_projection)


FIELDS = ('cases', 'full_fixed', 'isolated_observations_stable', 'contextual',
          'recurrence_essential', 'essential_with_all_acyclic_observations_stable',
          'essential_from_zero')


def empty_counts():
    return dict.fromkeys(FIELDS, 0)


def add_case(counts, result, initial):
    counts['cases'] += 1
    for key in ('full_fixed', 'isolated_observations_stable', 'contextual',
                'recurrence_essential'):
        counts[key] += int(result[key])
    if result['recurrence_essential']:
        counts['essential_with_all_acyclic_observations_stable'] += int(
            result['all_acyclic_observations_stable'])
        counts['essential_from_zero'] += int(initial[2:] == (0, 0, 0))


def describe(network, initial):
    controls = []
    for edges in EDGE_MASKS:
        if any(present and not allowed for present, allowed in zip(edges, network.edges)):
            continue
        control = Network(network.rules, edges, network.schedule)
        result = orbit(control, initial)
        controls.append(dict(edges=edges, orbit=result,
                             satellite_response=stable_projection(result, (3, 4))))
    return dict(rules=network.rules, edges=network.edges, schedule=network.schedule,
                initial=initial, criterion=recurrence_test(network, initial),
                orbit=orbit(network, initial),
                basins=attractors(network, initial[:2]), edge_ablations=controls)


def certificate():
    full_counts = empty_counts()
    by_edges = [empty_counts() for _ in range(5)]
    full_witnesses = []
    minimum = None
    first_settled_control_witness = None
    rows = []
    for rules in product(('or', 'xor'), repeat=3):
        for schedule in SCHEDULES:
            row = empty_counts()
            for edges in EDGE_MASKS:
                network = Network(rules, edges, schedule)
                for evidence in INPUTS:
                    for internal in INTERNAL_STATES:
                        initial = evidence + internal
                        result = recurrence_test(network, initial)
                        add_case(by_edges[sum(edges)], result, initial)
                        if result['recurrence_essential']:
                            if (first_settled_control_witness is None and
                                    result['all_acyclic_observations_stable']):
                                first_settled_control_witness = describe(network, initial)
                            if minimum is None or sum(edges) < sum(minimum['edges']):
                                minimum = dict(rules=rules, edges=edges,
                                               schedule=schedule, initial=initial)
                        if edges == FULL:
                            add_case(full_counts, result, initial)
                            add_case(row, result, initial)
                            if result['recurrence_essential']:
                                full_witnesses.append(dict(rules=rules, schedule=schedule,
                                                           initial=initial,
                                                           response=result['satellite_response']))
            rows.append(dict(rules=rules, schedule=schedule, **row))
    representative = describe(Network(('or', 'or', 'xor')), (0, 1, 0, 0, 0))
    # A weak whole-control settling criterion falsely credits hidden-core cycles.
    projection_counterexample = describe(Network(('xor', 'or', 'xor')), (0, 0, 0, 0, 1))
    return dict(specification='OP-02B recurrence necessity under edge deletion v0.1',
                status='complete', bounds=dict(rule_triples=8, schedules=list(SCHEDULES),
                    evidence_pairs=4, internal_initial_states=8, edge_masks=16,
                    candidate_cases=28672, full_star_cases=1792),
                full_star_counts=full_counts,
                counts_by_edge_count=by_edges, full_star_enumeration=rows,
                full_star_witnesses=full_witnesses,
                minimum_edge_witness=(describe(Network(minimum['rules'], minimum['edges'],
                                                       minimum['schedule']), minimum['initial'])
                                      if minimum else None),
                representative=representative,
                settled_control_witness=first_settled_control_witness,
                projection_counterexample=projection_counterexample)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'reports' / 'recurrence-necessity.json'
    out.write_text(json.dumps(certificate(), indent=2, sort_keys=True) + '\n')
    print(out)
