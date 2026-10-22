"""Bounded OP-03B contract: python -m experiments.response_family."""
from itertools import combinations, product
import json
from pathlib import Path
from src.recurrent import EDGE_MASKS, FULL, INPUTS
from src.response_family import REGIMES, assess_input, assess_pair

GATES = ('stable', 'distinguishable', 'contextual', 'recoverable', 'structurally_tolerant')


def certificate():
    regimes = []
    monotone = []
    for regime in REGIMES:
        rows = []
        counts = dict(candidates=0, **dict.fromkeys(GATES, 0))
        full_counts = counts.copy()
        # Count this obstruction independently of contextual/structural filters.
        distinguishable_recoverable = 0
        contextual_structurally_tolerant = 0
        recoverable_inputs = []
        for rules in product(('or', 'xor'), repeat=3):
            for edges in EDGE_MASKS:
                inputs = [assess_input(rules, edges, evidence, regime) for evidence in INPUTS]
                recoverable_inputs.extend(case['response'] for case in inputs if case['recoverable'])
                families = [assess_pair(left, right) for left, right in combinations(inputs, 2)]
                for family in families:
                    for counter in (counts, full_counts) if edges == FULL else (counts,):
                        counter['candidates'] += 1
                        passed = True
                        for gate in GATES:
                            passed = passed and family['gates'][gate]
                            counter[gate] += int(passed)
                    contextual_structurally_tolerant += int(all(family['gates'][gate]
                        for gate in ('stable', 'distinguishable', 'contextual', 'structurally_tolerant')))
                    distinguishable_recoverable += int(family['gates']['distinguishable']
                                                      and family['gates']['recoverable'])
                rows.append(dict(rules=rules, edges=edges, inputs=inputs, families=families))
                if rules == ('or',) * 3 and regime == 'all_fair':
                    monotone.append(dict(edges=edges,
                        responses=[case['response'] for case in inputs],
                        D=len({case['response'] for case in inputs}),
                        R=sum(case['recoverable'] for case in inputs)))
        regimes.append(dict(regime=regime, successive_gate_counts=counts,
                            full_star_successive_gate_counts=full_counts,
                            distinguishable_and_recoverable=distinguishable_recoverable,
                            stable_distinguishable_contextual_structurally_tolerant=contextual_structurally_tolerant,
                            recoverable_input_cases=len(recoverable_inputs),
                            recoverable_response_inventory=sorted(set(recoverable_inputs)),
                            networks=rows))
    strict_comparisons = []
    for left, right in product(monotone, repeat=2):
        if left['edges'] != right['edges'] and all(not a or b for a, b in zip(left['edges'], right['edges'])):
            strict_comparisons.append(dict(before=left['edges'], after=right['edges'],
                delta_R=right['R'] - left['R'], delta_D=right['D'] - left['D']))
    return dict(specification='OP-03B two-input response family v0.1', status='complete',
                contract='docs/op-03b-contract.md',
                bounds=dict(rule_triples=8, edge_masks=16, evidence_pairs=6,
                            candidate_families_per_regime=768, regimes=REGIMES,
                            initial_internal_state=(0, 0, 0)),
                regimes=regimes, all_OR_tradeoff=dict(networks=monotone,
                    strict_edge_additions=strict_comparisons))


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'reports' / 'response-family.json'
    out.write_text(json.dumps(certificate(), indent=2, sort_keys=True) + '\n')
    print(out)
