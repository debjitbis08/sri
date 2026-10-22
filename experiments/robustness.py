"""OP-03 exact robustness certificate: python -m experiments.robustness."""
from itertools import product
import json
from pathlib import Path
from src.recurrent import (Network, FULL, EDGE_MASKS, INPUTS, INTERNAL_STATES,
                           SCHEDULES, orbit, attractors)
from src.robustness import fair_graph, fair_outcomes, guaranteed_response, fair_lasso


def analyze_case(network, evidence, cache):
    analysis = cache[network.rules, network.edges, evidence]
    fixed = [s for s in INTERNAL_STATES if all(t == s for t in analysis['graph'][s])]
    basin_rows = []
    for target in fixed:
        response = target[1:]
        recovery = []
        for bit in range(3):
            disturbed = tuple(value ^ (i == bit) for i, value in enumerate(target))
            recovery.append(dict(flipped_node=bit, disturbed=disturbed,
                same_state_all_fair=guaranteed_response(analysis, disturbed, target, (0, 1, 2)),
                same_response_all_fair=guaranteed_response(analysis, disturbed, response)))
        deletions = []
        for edge in range(4):
            changed = tuple(enabled and i != edge for i, enabled in enumerate(network.edges))
            modified = cache[network.rules, changed, evidence]
            deletions.append(dict(deleted_edge=edge, edges=changed,
                from_established_state_all_fair=guaranteed_response(modified, target, response),
                from_zero_all_fair=guaranteed_response(modified, (0, 0, 0), response),
                target_remains_fixed=all(s == target for s in modified['graph'][target])))
        basin_rows.append(dict(state=target, satellite_response=response,
            possible_state_basin=[s for s in INTERNAL_STATES
                                  if target in analysis['reachable'][s]],
            guaranteed_state_basin=[s for s in INTERNAL_STATES
                if guaranteed_response(analysis, s, target, (0, 1, 2))],
            guaranteed_response_basin=[s for s in INTERNAL_STATES
                if guaranteed_response(analysis, s, response)],
            single_bit_recovery=recovery, single_edge_deletions=deletions))
    initial_rows = []
    for internal in INTERNAL_STATES:
        outcomes = fair_outcomes(analysis, internal)
        observations = {s[1:] for c in outcomes for s in c}
        initial_rows.append(dict(initial=internal,
            reachable_fair_components=outcomes,
            all_fair_settle=all(len(c) == 1 for c in outcomes),
            unique_stable_response=(next(iter(observations)) if len(observations) == 1 else None)))
    return dict(rules=network.rules, evidence=evidence, edges=network.edges,
                synchronous_basins=attractors(network, evidence), fixed_points=basin_rows,
                initial_states=initial_rows,
                zero_start_by_fixed_schedule=[dict(schedule=schedule,
                    orbit=orbit(Network(network.rules, network.edges, schedule), evidence + (0, 0, 0)))
                    for schedule in SCHEDULES])


def certificate():
    cache = {}
    for rules in product(('or', 'xor'), repeat=3):
        for edges in EDGE_MASKS:
            for evidence in INPUTS:
                cache[rules, edges, evidence] = fair_graph(Network(rules, edges), evidence)
    rows = [analyze_case(Network(rules), evidence, cache)
            for rules in product(('or', 'xor'), repeat=3) for evidence in INPUTS]
    counts = dict(full_star_graphs=32, initial_cases=256, all_fair_settle=0,
                  unique_stable_response=0, fixed_points=0,
                  recover_all_single_bit_response=0,
                  preserve_all_single_deletions_from_established=0)
    for row in rows:
        counts['all_fair_settle'] += sum(s['all_fair_settle'] for s in row['initial_states'])
        counts['unique_stable_response'] += sum(s['unique_stable_response'] is not None
                                                for s in row['initial_states'])
        counts['fixed_points'] += len(row['fixed_points'])
        counts['recover_all_single_bit_response'] += sum(
            all(p['same_response_all_fair'] for p in f['single_bit_recovery'])
            for f in row['fixed_points'])
        counts['preserve_all_single_deletions_from_established'] += sum(
            all(d['from_established_state_all_fair'] for d in f['single_edge_deletions'])
            for f in row['fixed_points'])
    # Include a reproducible fair nonsettling lasso, not a finite schedule sample.
    counterexample = None
    for row in rows:
        analysis = cache[row['rules'], FULL, row['evidence']]
        for component in analysis['fair_components']:
            if len(component) > 1:
                counterexample = dict(rules=row['rules'], evidence=row['evidence'], edges=FULL,
                    component=component, lasso=fair_lasso(analysis, component[0], component))
                break
        if counterexample:
            break
    witness_analysis = cache[('or', 'or', 'xor'), FULL, (0, 1)]
    witness_component = next(c for c in witness_analysis['fair_components']
                             if (0, 0, 0) in c)
    missed_by_seven = dict(rules=('or', 'or', 'xor'), evidence=(0, 1), edges=FULL,
        lasso=fair_lasso(witness_analysis, (0, 0, 0), witness_component),
        short_repeating_schedule=(2, 2, 0, 1))
    distinguishability = []
    for rules in product(('or', 'xor'), repeat=3):
        selected = [r for r in rows if r['rules'] == rules]
        responses = [r['initial_states'][0]['unique_stable_response'] for r in selected]
        distinguishability.append(dict(rules=rules, evidence_order=INPUTS,
            zero_start_all_fair_responses=responses,
            distinct_certified_responses=len({r for r in responses if r is not None})))
    return dict(specification='OP-03 fair asynchronous robustness v0.1', status='complete',
                bounds=dict(rule_triples=8, edge_masks=16, evidence_pairs=4,
                    asynchronous_graphs=512, internal_states_per_graph=8,
                    labeled_transitions_per_graph=24,
                    detail_scope='32 full-star graphs; single-edge deletions use cached graphs'),
                fairness='each of k, a, b selected infinitely often; one node per step; no delay bound',
                summary=counts, full_star_cases=rows,
                zero_start_distinguishability=distinguishability,
                fair_nonsettling_counterexample=counterexample,
                seven_schedule_counterexample=missed_by_seven)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'reports' / 'robustness.json'
    out.write_text(json.dumps(certificate(), indent=2, sort_keys=True) + '\n')
    print(out)
