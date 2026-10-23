"""Shared input interpretation v1: v1 dynamics plus an internal input coupling and the NF control.

See docs/shared-input-mediation-v1-contract.md. State, domains, evidence, queries and the
strict-descent update rule are those of learned reciprocal participation v1 (imported, not
modified). Coefficients w = (w_1A, w_2A, w_1B, w_2B, w_KA, w_KB, w_12); family S has w_12 = 0.

    E_C = E_v1 - w_12 * i1 * i2
    E_I = 2((i1-e1)^2 + (i2-e2)^2) - w_12 * i1 * i2      (input-local energy)

NF ('satellite feedback removed'): an input update is permitted when it strictly lowers E_I;
K, A and B updates use their full E_C differences. This removes A/B -> I influence only.
"""
from collections import deque
from dataclasses import dataclass
import itertools

from src.reciprocal_participation import DOMAINS, QUERIES, TERNARY, Query, signature


COORDS = ('i1', 'i2', 'k', 'a', 'b')
I1, I2, K, A, B = range(5)
COEFFS = ('w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12')
VALUES = (-3, 0, 3)
N_VECTORS = 3 ** 7
ZERO = (0,) * 7


def vector(index):
    """Lexicographic in coefficient order with -3 < 0 < 3."""
    if type(index) is not int or not 0 <= index < N_VECTORS:
        raise ValueError('coefficient index out of range')
    return tuple(VALUES[(index // 3 ** (6 - j)) % 3] for j in range(7))


def check(w):
    if len(w) != 7 or any(type(v) is not int or v not in VALUES for v in w):
        raise ValueError('w must be seven coefficients in {-3, 0, 3}')
    return tuple(w)


def energy(z, x, w):
    i1, i2, k, a, b = z
    w1a, w2a, w1b, w2b, wka, wkb, w12 = w
    return (2 * ((i1 - x.e1) ** 2 + (i2 - x.e2) ** 2) + 2 * (a * a + b * b) + k * k
            - w1a * i1 * a - w2a * i2 * a - w1b * i1 * b - w2b * i2 * b - wka * k * a - wkb * k * b
            - w12 * i1 * i2)


def input_energy(z, x, w):
    return 2 * ((z[I1] - x.e1) ** 2 + (z[I2] - x.e2) ** 2) - w[6] * z[I1] * z[I2]


def satellite_field(z, w, coord):
    """Field from satellites on an input coordinate, or the full field on K, A, B."""
    i1, i2, k, a, b = z
    w1a, w2a, w1b, w2b, wka, wkb, _ = w
    return (w1a * a + w1b * b, w2a * a + w2b * b, wka * a + wkb * b,
            w1a * i1 + w2a * i2 + wka * k, w1b * i1 + w2b * i2 + wkb * k)[coord]


def delta(z, x, w, coord, new, nf=False):
    """Change used to decide an update: E_C difference, or E_I difference for inputs under NF."""
    old = z[coord]
    if coord < 2:
        e = (x.e1, x.e2)[coord]
        internal = w[6] * z[1 - coord]
        d = 2 * ((new - e) ** 2 - (old - e) ** 2) - internal * (new - old)
        return d if nf else d - satellite_field(z, w, coord) * (new - old)
    scale = 1 if coord == K else 2
    return scale * (new * new - old * old) - satellite_field(z, w, coord) * (new - old)


@dataclass(frozen=True)
class Result:
    query: Query
    w: tuple
    start: tuple
    frozen: tuple
    nf: bool
    status: str
    reachable: object
    terminals: object
    certified_terminals: frozenset
    Y: object
    edges: int
    inspections: int
    acyclic: object
    parent: dict

    def projection(self, coord):
        return frozenset(t[coord] for t in self.terminals)

    def shortest_path(self, terminal):
        path, node = [], terminal
        while self.parent[node] is not None:
            prev, c = self.parent[node]
            path.append(dict(coord=COORDS[c], source=list(prev), target=list(node)))
            node = prev
        return path[::-1]


def start_state(x, clamps=()):
    """Canonical start with the named coordinates replaced; clamps is ((coord, value), ...)."""
    z = list(x.start())
    for c, v in clamps:
        z[c] = v
    return tuple(z)


def explore(x, w, start=None, frozen=(), nf=False, cap=None):
    """Breadth-first enumeration of every permitted single-coordinate update (v1 traversal order)."""
    if not isinstance(x, Query):
        raise TypeError('x must be a Query')
    w = check(w)
    start = x.start() if start is None else tuple(start)
    if len(start) != 5 or any(start[c] not in x.allowed(c) for c in range(5)):
        raise ValueError('start state outside the local domains')
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError('cap must be a non-negative int or None')
    movable = [c for c in range(5) if c not in frozen]
    parent, succ = {start: None}, {}
    frontier = deque([start])
    work = edges = 0
    certified = []
    truncated = cap == 0
    while frontier and not truncated:
        z = frontier.popleft()
        out = []
        for c in movable:
            for v in x.allowed(c):
                if v == z[c]:
                    continue
                if cap is not None and work >= cap:
                    truncated = True
                    break
                work += 1
                if delta(z, x, w, c, v, nf) < 0:
                    nz = z[:c] + (v,) + z[c + 1:]
                    out.append(nz)
                    edges += 1
                    if nz not in parent:
                        parent[nz] = (z, c)
                        frontier.append(nz)
            if truncated:
                break
        if truncated:
            break
        succ[z] = out
        if not out:
            certified.append(z)
    common = dict(query=x, w=w, start=start, frozen=tuple(sorted(frozen)), nf=nf,
                  certified_terminals=frozenset(certified), edges=edges, inspections=work, parent=parent)
    if truncated:
        return Result(status='truncated', reachable=None, terminals=None, Y=None, acyclic=None, **common)
    terminals = frozenset(certified)
    if not terminals:
        raise AssertionError('complete finite run without a terminal: implementation error')
    return Result(status='complete', reachable=frozenset(parent), terminals=terminals,
                  Y=frozenset((t[A], t[B]) for t in terminals), acyclic=is_acyclic(start, succ), **common)


def is_acyclic(start, succ):
    color = dict.fromkeys(succ, 0)
    stack = [(start, iter(succ[start]))]
    color[start] = 1
    while stack:
        z, it = stack[-1]
        n = next(it, None)
        if n is None:
            color[z] = 2
            stack.pop()
        elif color[n] == 1:
            return False
        elif color[n] == 0:
            color[n] = 1
            stack.append((n, iter(succ[n])))
    return True


def zeroed(w, *names):
    return tuple(0 if COEFFS[j] in names else v for j, v in enumerate(w))


def core_disconnected(w):
    return zeroed(w, 'w_KA', 'w_KB')


def independent_product(x, w):
    """I fixed at evidence, core couplings removed, A and B evolved separately; Cartesian product."""
    wd = core_disconnected(w)
    a_set = explore(x, wd, frozen=(I1, I2, K, B)).projection(A)
    b_set = explore(x, wd, frozen=(I1, I2, K, A)).projection(B)
    return frozenset(itertools.product(sorted(a_set), sorted(b_set)))


def legacy_input_feedback_removed(x, w):
    """v1's control: inputs see only the evidence penalty (no internal coupling). Regression only."""
    if w[6]:
        raise ValueError('legacy v1 control is defined only for w_12 = 0')
    return explore(x, w, nf=True).Y


def controls(x, w):
    out = dict(
        full=explore(x, w).Y,
        uncoupled=explore(x, ZERO).Y,
        core_disconnected=explore(x, core_disconnected(w)).Y,
        input_frozen=explore(x, w, frozen=(I1, I2)).Y,
        satellite_feedback_removed_NF=explore(x, w, nf=True).Y,
        independent_product=independent_product(x, w),
    )
    for kv in TERNARY:
        out[f'core_fixed_{kv}'] = explore(x, w, start=start_state(x, ((K, kv),)), frozen=(K,)).Y
    return out


def decreases(z, x, w, coord, v):
    return delta(z, x, w, coord, v) < 0


def influence_witnesses(x, w, states, targets, neighbors):
    found = []
    for z in sorted(states):
        for c in neighbors:
            for cv in x.allowed(c):
                if cv <= z[c]:
                    continue
                z2 = z[:c] + (cv,) + z[c + 1:]
                if z2 not in states:
                    continue
                for t in targets:
                    for v in x.allowed(t):
                        if v != z[t] and decreases(z, x, w, t, v) != decreases(z2, x, w, t, v):
                            found.append(dict(state=z, neighbor_state=z2, changed=COORDS[c], target=COORDS[t],
                                              proposed=v, decreases_in_state=decreases(z, x, w, t, v),
                                              decreases_in_neighbor_state=decreases(z2, x, w, t, v)))
    return found


def all_states(x):
    return frozenset(itertools.product(*(x.allowed(c) for c in range(5))))


def clamp_sets(x, w, clamped):
    other = B if clamped == A else A
    out = {}
    for v in x.allowed(clamped):
        r = explore(x, w, start=start_state(x, ((clamped, v),)), frozen=(clamped,))
        out[v] = r.projection(other)
    return out


__all__ = ['A', 'B', 'COEFFS', 'COORDS', 'DOMAINS', 'I1', 'I2', 'K', 'N_VECTORS', 'QUERIES', 'Query', 'TERNARY',
           'ZERO', 'all_states', 'check', 'clamp_sets', 'controls', 'core_disconnected', 'delta', 'energy',
           'explore', 'independent_product', 'influence_witnesses', 'input_energy', 'legacy_input_feedback_removed',
           'satellite_field', 'signature', 'start_state', 'vector', 'zeroed']
