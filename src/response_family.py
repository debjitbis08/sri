"""OP-03B's fixed response-family contract; no search over new local rules."""
from functools import lru_cache
from src.recurrent import Network, SCHEDULES, A_ONLY, B_ONLY
from src.robustness import fair_graph, fair_outcomes, update_node

REGIMES = ('all_fair', 'fixed_sweeps')
ZERO = (0, 0, 0)


@lru_cache(maxsize=None)
def _fair_analysis(rules, edges, evidence):
    return fair_graph(Network(rules, edges), evidence)


@lru_cache(maxsize=None)
def recurrent_states(rules, edges, evidence, initial, regime):
    """All possible recurrent internal states under the declared schedule policy."""
    if regime == 'all_fair':
        components = fair_outcomes(_fair_analysis(rules, edges, evidence), initial)
        states = {s for component in components for s in component}
    elif regime == 'fixed_sweeps':
        states = set()
        network = Network(rules, edges)
        for schedule in SCHEDULES[1:]:
            # Observe atomic phases, not just sweep boundaries. A fixed-order
            # execution has at most 8 states x 3 schedule phases before repeating.
            seen, path = {}, []
            state, phase = initial, 0
            while (state, phase) not in seen:
                seen[state, phase] = len(path)
                path.append(state)
                node = 'kab'.index(schedule[phase])
                state = update_node(network, evidence, state, node)
                phase = (phase + 1) % 3
            states.update(path[seen[state, phase]:])
    else:
        raise ValueError('unknown schedule regime')
    return tuple(sorted(states))


def observation(states, components=(1, 2)):
    values = {tuple(s[i] for i in components) for s in states}
    return next(iter(values)) if len(values) == 1 else None


def assess_input(rules, edges, evidence, regime):
    states = recurrent_states(rules, edges, evidence, ZERO, regime)
    target = observation(states)
    isolated = []
    for mask, component in ((A_ONLY, 1), (B_ONLY, 2)):
        isolated_edges = tuple(present and allowed for present, allowed in zip(edges, mask))
        isolated.append(observation(recurrent_states(rules, isolated_edges, evidence, ZERO, regime),
                                    (component,)))
    independent = ((isolated[0][0], isolated[1][0])
                   if all(value is not None for value in isolated) else None)
    result = dict(evidence=evidence, response=target, recurrent_states=states,
                  independent_response=independent,
                  contextual=target is not None and independent is not None and target != independent,
                  recoverable=False, first_recovery_failure=None, preserving_deletions=[])
    if target is None:
        return result
    for state in states:
        for bit in range(3):
            disturbed = tuple(value ^ (i == bit) for i, value in enumerate(state))
            after = recurrent_states(rules, edges, evidence, disturbed, regime)
            if observation(after) != target and result['first_recovery_failure'] is None:
                result['first_recovery_failure'] = dict(state=state, flipped_node=bit,
                    disturbed=disturbed, recurrent_states=after, response=observation(after))
    result['recoverable'] = result['first_recovery_failure'] is None
    for edge, present in enumerate(edges):
        if not present:
            continue
        modified = tuple(enabled and i != edge for i, enabled in enumerate(edges))
        if all(observation(recurrent_states(rules, modified, evidence, initial, regime)) == target
               for initial in (ZERO,) + states):
            result['preserving_deletions'].append(edge)
    return result


def assess_pair(left, right):
    """Return each gate independently; the report also records successive gates."""
    stable = left['response'] is not None and right['response'] is not None
    distinguishable = stable and left['response'] != right['response']
    contextual = (stable and left['independent_response'] is not None
                  and right['independent_response'] is not None
                  and (left['contextual'] or right['contextual']))
    recoverable = stable and left['recoverable'] and right['recoverable']
    common = sorted(set(left['preserving_deletions']) & set(right['preserving_deletions']))
    gates = dict(stable=stable, distinguishable=distinguishable, contextual=contextual,
                 recoverable=recoverable, structurally_tolerant=stable and bool(common))
    return dict(inputs=(left['evidence'], right['evidence']), gates=gates,
                common_preserving_deletions=common, passes=all(gates.values()))
