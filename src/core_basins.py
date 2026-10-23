"""Basin contract for the graded core interface: exact all-schedule and random-schedule basins."""
from fractions import Fraction
from itertools import product

from src.core_graded import GradedSystem, encode, popcount
from src.core_mediation import energy as mode_energy, moves


def tie_free(sigma, lam):
    """Proposition P-tie: no flip can be energy-neutral when 64*lambda/sigma is never integral."""
    sat, core = (lam, lam) if isinstance(lam, int) else lam
    return all((64*l) % sigma for l in (sat, core))


def predicted_terminal_prototypes(cls, tables, sigma, lam):
    """Proposition P-terminal: prototype x is terminal iff no mode move has sigma*D < -4*lambda_M."""
    sat, core = (lam, lam) if isinstance(lam, int) else lam
    barrier = (sat, sat, core)
    out = []
    for x in product((0, 1), repeat=3):
        e = mode_energy(cls, tables, x)
        blocked = True
        for y in moves(x, 2):
            m = next(i for i in range(3) if x[i] != y[i])
            if sigma*(mode_energy(cls, tables, y) - e) < -4*barrier[m]:
                blocked = False
        if blocked:
            out.append(x)
    return out


def basins(system, targets):
    """Exact DP over the strict-descent DAG, in increasing energy order."""
    states = range(1 << system.free_units)
    order = sorted(states, key=system.energy)
    reach, prob = {}, {}
    for s in order:
        nxt = system.successors(s)
        if not nxt:
            r = system.readouts(s)
            label = r if (tuple(r) in targets and s == encode(r)) else ('spurious', r)
            reach[s] = frozenset([label])
            prob[s] = {label: Fraction(1)}
            continue
        reach[s] = frozenset().union(*(reach[t] for t in nxt))
        p = {}
        w = Fraction(1, len(nxt))
        for t in nxt:
            for label, v in prob[t].items():
                p[label] = p.get(label, 0) + w*v
        prob[s] = p
    return reach, prob


def summarize(system, targets, reach, prob):
    terminals = [s for s in reach if not system.successors(s)]
    off_proto = [s for s in terminals if any(popcount(w) not in (0, 4) for w in system.words(s))]
    proto_terminals = sorted(tuple(system.readouts(s)) for s in terminals if s not in off_proto)
    n = len(reach)
    committed = sum(1 for s in reach if len(reach[s]) == 1 and next(iter(reach[s])) in targets)
    mean_target = sum(sum(v for k, v in prob[s].items() if k in targets) for s in reach) / n
    radius = {}
    for x in targets:
        s0 = encode(x)
        rows, certain = [], 0
        for r in range(1, 7):
            ring = [s for s in reach if popcount(s ^ s0) == r]
            ps = [prob[s].get(x, Fraction(0)) for s in ring]
            sure = all(reach[s] == frozenset([x]) for s in ring)
            if sure and certain == r-1:
                certain = r
            rows.append(dict(r=r, states=len(ring), min_return=float(min(ps)),
                             mean_return=float(sum(ps)/len(ps)), all_schedule_return=sure))
        radius[str(x)] = dict(certain_radius=certain, rings=rows)
    return dict(
        terminals=len(terminals), off_prototype_terminals=len(off_proto),
        prototype_terminals=[list(x) for x in proto_terminals],
        spurious_prototype_terminals=[list(x) for x in proto_terminals if x not in targets],
        committed_to_target_all_schedules=committed,
        mean_target_probability_uniform_start=float(mean_target),
        radius=radius,
    )


def analyze(cls, tables, targets, sigma, lam):
    system = GradedSystem(cls, tables, sigma, lam)
    reach, prob = basins(system, targets)
    out = summarize(system, targets, reach, prob)
    predicted = predicted_terminal_prototypes(cls, tables, sigma, lam)
    out['tie_free'] = tie_free(sigma, lam)
    out['predicted_prototype_terminals'] = [list(x) for x in predicted]
    out['propositions_confirmed'] = (
        sorted(map(tuple, out['prototype_terminals'])) == sorted(predicted)
        and (not out['tie_free'] or out['off_prototype_terminals'] == 0))
    return out
