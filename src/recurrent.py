"""OP-02B: Boolean recurrent star, persistent evidence, exact orbit analysis."""
from dataclasses import dataclass
from itertools import permutations, product

# State order: (x_A, x_B, k, a, b). Evidence bits never update.
SCHEDULES = ('sync',) + tuple(''.join(order) for order in permutations('kab'))
EDGE_MASKS = tuple(product((False, True), repeat=4))
INTERNAL_STATES = tuple(product((0, 1), repeat=3))
INPUTS = tuple(product((0, 1), repeat=2))
FULL = (True,) * 4
UNCOUPLED = (False,) * 4
A_ONLY = (True, False, True, False)
B_ONLY = (False, True, False, True)
FEED_FORWARD = (True, True, False, False)


def combine(rule, values):
    return int(any(values)) if rule == 'or' else sum(values) % 2


@dataclass(frozen=True)
class Network:
    # Rule order k, a, b; edge order A->K, B->K, K->A, K->B.
    rules: tuple[str, str, str] = ('or', 'or', 'or')
    edges: tuple[bool, bool, bool, bool] = FULL
    schedule: str = 'sync'

    def __post_init__(self):
        if len(self.rules) != 3 or any(r not in ('or', 'xor') for r in self.rules):
            raise ValueError('expected three OR/XOR rules')
        if len(self.edges) != 4 or any(type(e) is not bool for e in self.edges):
            raise ValueError('expected four Boolean edge indicators')
        if self.schedule not in SCHEDULES:
            raise ValueError('unknown schedule')

    def step(self, state):
        if len(state) != 5 or any(type(s) is not int or s not in (0, 1) for s in state):
            raise ValueError('expected five Boolean state indices')
        values = list(state)

        def update(node, source):
            x, y, k, a, b = source
            ak, bk, ka, kb = self.edges
            terms = {'k': (k, a if ak else 0, b if bk else 0),
                     'a': (a, k if ka else 0, x),
                     'b': (b, k if kb else 0, y)}
            index = 'kab'.index(node)
            values[index + 2] = combine(self.rules[index], terms[node])

        if self.schedule == 'sync':
            for node in 'kab':
                update(node, state)
        else:
            for node in self.schedule:
                update(node, values)
        return tuple(values)


def orbit(network, initial):
    """Stop on the first exact repeat, never on an assumed relaxation budget.

    Fixed evidence leaves eight internal states. At most eight transitions
    suffice to find a repeat. A cycle of length >1 is complete but not settled.
    """
    seen, path = {}, []
    state = tuple(initial)
    while state not in seen:
        seen[state] = len(path)
        path.append(state)
        state = network.step(state)
    start = seen[state]
    return dict(status='complete', transient=path[:start], cycle=path[start:],
                period=len(path) - start, transitions=len(path))


def canonical_cycle(cycle):
    cycle = tuple(cycle)
    return min(cycle[i:] + cycle[:i] for i in range(len(cycle)))


def attractors(network, evidence):
    """Enumerate the entire eight-state functional graph, with basin sizes."""
    basins = {}
    for internal in INTERNAL_STATES:
        result = orbit(network, tuple(evidence) + internal)
        key = canonical_cycle(result['cycle'])
        basins[key] = basins.get(key, 0) + 1
    return [dict(cycle=key, period=len(key), basin_size=basins[key])
            for key in sorted(basins)]


def stable_context(joint, isolated, component):
    """Compare settled component states; cycles are not treated as fixed outputs."""
    return (joint['period'] == isolated['period'] == 1
            and joint['cycle'][0][component] != isolated['cycle'][0][component])


def stable_projection(result, components):
    """Return the constant eventual observation, or None if it oscillates.

    Hidden coordinates may cycle. Inspect every phase instead of mistaking a
    single sampled phase for a settled component response.
    """
    observations = {tuple(state[i] for i in components) for state in result['cycle']}
    return next(iter(observations)) if len(observations) == 1 else None


def acyclic(edges):
    """No inter-node directed cycle in this star; local state retention remains."""
    return not (edges[0] and edges[2]) and not (edges[1] and edges[3])


def submasks(edges):
    return tuple(mask for mask in EDGE_MASKS
                 if all(not present or allowed for present, allowed in zip(mask, edges)))


def recurrence_test(network, initial):
    """Scoped necessity of inter-node recurrence for a stable satellite pair.

    Hold rules, evidence, initialization and schedule fixed. Require a full-state
    fixed point and stable isolated A/B observations, with contextual change.
    Then reject if ANY acyclic edge deletion has the same stable satellite pair.
    This tests edge deletions, not all possible acyclic implementations.
    """
    joint = orbit(network, initial)
    result = dict(full_fixed=joint['period'] == 1,
                  isolated_observations_stable=False, contextual=False,
                  recurrence_essential=False, all_acyclic_observations_stable=False)
    if not result['full_fixed']:
        return result
    isolated = []
    for mask, component in ((A_ONLY, 3), (B_ONLY, 4)):
        edges = tuple(a and b for a, b in zip(network.edges, mask))
        control = orbit(Network(network.rules, edges, network.schedule), initial)
        isolated.append(stable_projection(control, (component,)))
    if any(value is None for value in isolated):
        return result
    result['isolated_observations_stable'] = True
    pair = stable_projection(joint, (3, 4))
    independent = (isolated[0][0], isolated[1][0])
    result['contextual'] = pair != independent
    if not result['contextual']:
        return result
    controls = []
    for edges in submasks(network.edges):
        if acyclic(edges):
            control = orbit(Network(network.rules, edges, network.schedule), initial)
            controls.append(dict(edges=edges, period=control['period'],
                                 satellite_response=stable_projection(control, (3, 4))))
    matches = [c['edges'] for c in controls if c['satellite_response'] == pair]
    result.update(satellite_response=pair, independent_response=independent,
                  acyclic_controls=controls, matching_acyclic_masks=matches,
                  recurrence_essential=not matches,
                  all_acyclic_observations_stable=all(c['satellite_response'] is not None
                                                     for c in controls))
    return result
