"""Exact fair single-node scheduling analysis for the finite Boolean star."""
from collections import deque
from src.recurrent import INTERNAL_STATES, combine


def update_node(network, evidence, internal, node):
    """One asynchronous update; evidence remains clamped, including on no-ops."""
    if node not in range(3):
        raise ValueError('node must be 0 (core), 1 (A), or 2 (B)')
    x, y = evidence
    k, a, b = internal
    ak, bk, ka, kb = network.edges
    terms = ((k, a if ak else 0, b if bk else 0),
             (a, k if ka else 0, x), (b, k if kb else 0, y))
    result = list(internal)
    result[node] = combine(network.rules[node], terms[node])
    return tuple(result)


def fair_graph(network, evidence):
    """Enumerate maximal SCCs supporting an infinite node-fair execution.

    Fairness means each node is selected infinitely often (unbounded delays).
    A strongly connected component supports such an execution iff it has an
    internal edge bearing each of the three labels. Non-bottom SCCs matter.
    """
    graph = {s: tuple(update_node(network, evidence, s, n) for n in range(3))
             for s in INTERNAL_STATES}
    reachable = {}
    for start in INTERNAL_STATES:
        visited, pending = {start}, [start]
        while pending:
            for target in graph[pending.pop()]:
                if target not in visited:
                    visited.add(target)
                    pending.append(target)
        reachable[start] = visited
    remaining = set(INTERNAL_STATES)
    fair_components = []
    while remaining:
        start = min(remaining)
        component = {s for s in reachable[start] if start in reachable[s]}
        remaining -= component
        labels = {n for s in component for n, t in enumerate(graph[s]) if t in component}
        if labels == {0, 1, 2}:
            fair_components.append(tuple(sorted(component)))
    fair_components.sort()
    return dict(graph=graph, reachable=reachable, fair_components=fair_components)


def fair_outcomes(analysis, initial):
    return [component for component in analysis['fair_components']
            if component[0] in analysis['reachable'][initial]]


def guaranteed_response(analysis, initial, response, components=(1, 2)):
    """Every fair execution eventually has exactly the requested projection."""
    return all(all(tuple(s[i] for i in components) == tuple(response) for s in component)
               for component in fair_outcomes(analysis, initial))


def path(graph, source, target, allowed=None):
    """Shortest labeled path, with deterministic tie-breaking."""
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        state, steps = queue.popleft()
        if state == target:
            return steps
        for node, nxt in enumerate(graph[state]):
            if nxt not in seen and (allowed is None or nxt in allowed):
                seen.add(nxt)
                queue.append((nxt, steps + [dict(node=node, state=nxt)]))
    raise ValueError('no path')


def fair_lasso(analysis, initial, component):
    """A finite prefix and repeating fair tour visiting every component state."""
    graph = analysis['graph']
    start = component[0]
    prefix = path(graph, initial, start)
    current, loop = start, []

    def visit(target):
        nonlocal current
        steps = path(graph, current, target, set(component))
        loop.extend(steps)
        current = target

    for state in component:
        visit(state)
    for node in range(3):
        source = next(s for s in component if graph[s][node] in component)
        visit(source)
        current = graph[source][node]
        loop.append(dict(node=node, state=current))
    visit(start)
    return dict(initial=initial, prefix=prefix, loop_start=start, loop=loop)
