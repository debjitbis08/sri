"""Exhaustive recurrent-star certificate: python -m experiments.recurrent."""
from collections import Counter
from itertools import product
import json
from pathlib import Path
from src.recurrent import (Network, SCHEDULES, EDGE_MASKS, INPUTS, INTERNAL_STATES,
                           FULL, UNCOUPLED, A_ONLY, B_ONLY, FEED_FORWARD,
                           orbit, attractors, stable_context)


def certificate():
    rows = []
    first_cycle = None
    first_schedule_difference = None
    for rules in product(('or', 'xor'), repeat=3):
        reference = {}
        for schedule in SCHEDULES:
            periods = Counter()
            attractor_periods = Counter()
            contextual_a = contextual_b = 0
            max_transitions = 0
            for edges in EDGE_MASKS:
                network = Network(rules, edges, schedule)
                for evidence in INPUTS:
                    for basin in attractors(network, evidence):
                        attractor_periods[basin['period']] += 1
                    for internal in INTERNAL_STATES:
                        initial = evidence + internal
                        result = orbit(network, initial)
                        periods[result['period']] += 1
                        max_transitions = max(max_transitions, result['transitions'])
                        if first_cycle is None and result['period'] > 1 and edges == FULL:
                            first_cycle = dict(rules=rules, schedule=schedule,
                                               initial=initial, edges=edges, orbit=result)
                        if edges != FULL:
                            continue
                        isolated_a = orbit(Network(rules, A_ONLY, schedule), initial)
                        isolated_b = orbit(Network(rules, B_ONLY, schedule), initial)
                        contextual_a += stable_context(result, isolated_a, 3)
                        contextual_b += stable_context(result, isolated_b, 4)
                        # Fixed-point differences avoid conflating cycle phase with context.
                        if schedule == 'sync':
                            reference[initial] = result
                        elif first_schedule_difference is None:
                            other = reference[initial]
                            if (result['period'] != other['period'] or
                                (result['period'] == 1 and result['cycle'] != other['cycle'])):
                                first_schedule_difference = dict(
                                    rules=rules, edges=edges, initial=initial,
                                    synchronous=other, schedule=schedule, sequential=result)
            rows.append(dict(rules=rules, schedule=schedule, maps=64, trajectories=512,
                             trajectory_period_counts=dict(sorted(periods.items())),
                             attractor_period_counts=dict(sorted(attractor_periods.items())),
                             max_transitions=max_transitions,
                             full_star_stable_context_A=contextual_a,
                             full_star_stable_context_B=contextual_b))
    initial = (0, 1, 0, 0, 0)
    controls = {}
    for name, edges in (('coupled', FULL), ('A_only', A_ONLY), ('B_only', B_ONLY),
                        ('uncoupled', UNCOUPLED), ('feed_forward', FEED_FORWARD)):
        network = Network(edges=edges)
        controls[name] = dict(edges=edges, orbit=orbit(network, initial),
                              attractors=attractors(network, initial[:2]))
    return dict(specification='OP-02B Boolean recurrent star v0.1', status='complete',
                bounds=dict(rule_triples=8, edge_masks=16, schedules=list(SCHEDULES),
                            evidence_pairs=4, internal_initial_states=8,
                            maps=3584, trajectories=28672),
                enumeration=rows,
                stable_witness=dict(rules=('or',) * 3, schedule='sync',
                                    initial=initial, controls=controls,
                                    edge_ablations=[dict(edges=edges, orbit=orbit(Network(edges=edges), initial))
                                                    for edges in EDGE_MASKS]),
                cycle_counterexample=first_cycle,
                schedule_counterexample=first_schedule_difference)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'reports' / 'recurrent.json'
    out.write_text(json.dumps(certificate(), indent=2, sort_keys=True) + '\n')
    print(out)
