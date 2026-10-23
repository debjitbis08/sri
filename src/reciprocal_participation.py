"""Learned reciprocal participation v1: exact strict-descent dynamics, controls and learners.

See docs/learned-reciprocal-participation-v1-contract.md. Four modules: input I = (i1, i2),
core K = k, satellites A = a and B = b, all ternary; satellites have immutable domains.
State z = (i1, i2, k, a, b). Six signed pairwise couplings w in {-3, 0, 3}^6 enter one integer
energy. An atomic update changes one coordinate to another allowed value when the energy
strictly decreases; all such updates are kept (no scheduler choice). Learner APIs receive
queries and feedback sets only.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import itertools
import random


SCHEMA_VERSION = 'learned-reciprocal-participation-v1'
COORDS = ('i1', 'i2', 'k', 'a', 'b')
TERNARY = (-1, 0, 1)
DOMAINS = ((-1, 0, 1), (0, 1), (-1, 0))          # D0, D1, D2
COEFFS = ('w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB')
VALUES = (-3, 0, 3)
N_HYPOTHESES = 3 ** 6
ZERO = (0, 0, 0, 0, 0, 0)
TEACHERS = {
    'T0': (3, 0, 0, -3, 0, 0),
    'T1': (3, 0, 0, -3, 3, -3),
    'T2': (3, 0, 0, -3, 3, 3),
    'T3': (0, 3, -3, 0, -3, 3),
}


@dataclass(frozen=True)
class Query:
    """Immutable evidence: (e1, e2) and satellite domain indices."""
    e1: int
    e2: int
    da: int
    db: int

    def __post_init__(self):
        for name in ('e1', 'e2', 'da', 'db'):
            if type(getattr(self, name)) is not int:
                raise TypeError(f'{name} must be int')
        if self.e1 not in TERNARY or self.e2 not in TERNARY:
            raise ValueError('evidence must be in {-1, 0, 1}')
        if not 0 <= self.da < 3 or not 0 <= self.db < 3:
            raise ValueError('domain index out of range')

    def allowed(self, coord):
        return DOMAINS[self.da] if coord == 3 else DOMAINS[self.db] if coord == 4 else TERNARY

    def start(self):
        return (self.e1, self.e2, 0, 0, 0)

    def to_json(self):
        return dict(e=[self.e1, self.e2], domain_A=f'D{self.da}', domain_B=f'D{self.db}')


QUERIES = tuple(Query(e1, e2, da, db) for e1 in TERNARY for e2 in TERNARY for da in range(3) for db in range(3))
QUERY_INDEX = {q: i for i, q in enumerate(QUERIES)}


def hypothesis(index):
    """Lexicographic in coefficient order with -3 < 0 < 3."""
    if type(index) is not int or not 0 <= index < N_HYPOTHESES:
        raise ValueError('hypothesis index out of range')
    return tuple(VALUES[(index // 3 ** (5 - j)) % 3] for j in range(6))


def hypothesis_index(w):
    check_weights(w)
    return sum(VALUES.index(v) * 3 ** (5 - j) for j, v in enumerate(w))


def check_weights(w):
    if len(w) != 6 or any(type(v) is not int or v not in VALUES for v in w):
        raise ValueError('w must be six coefficients in {-3, 0, 3}')


def energy(z, x, w):
    i1, i2, k, a, b = z
    w1a, w2a, w1b, w2b, wka, wkb = w
    return (2 * ((i1 - x.e1) ** 2 + (i2 - x.e2) ** 2) + 2 * (a * a + b * b) + k * k
            - w1a * i1 * a - w2a * i2 * a - w1b * i1 * b - w2b * i2 * b - wka * k * a - wkb * k * b)


def field(z, w, coord):
    """Linear coupling field on one coordinate (the negative coefficient of that coordinate)."""
    i1, i2, k, a, b = z
    w1a, w2a, w1b, w2b, wka, wkb = w
    return (w1a * a + w1b * b, w2a * a + w2b * b, wka * a + wkb * b,
            w1a * i1 + w2a * i2 + wka * k, w1b * i1 + w2b * i2 + wkb * k)[coord]


def delta(z, x, w, coord, new, input_feedback=True):
    """Energy change of setting one coordinate; with input_feedback=False an I move sees only
    its evidence penalty (the 'input feedback removed' control)."""
    old = z[coord]
    if coord < 2:
        e = (x.e1, x.e2)[coord]
        d = 2 * ((new - e) ** 2 - (old - e) ** 2)
        return d - field(z, w, coord) * (new - old) if input_feedback else d
    scale = 1 if coord == 2 else 2
    return scale * (new * new - old * old) - field(z, w, coord) * (new - old)


# ---------------------------------------------------------------- exact exploration

@dataclass(frozen=True)
class Result:
    query: Query
    w: tuple
    start: tuple
    frozen: tuple                 # coordinates that never move
    input_feedback: bool
    status: str                   # 'complete' | 'truncated'
    reachable: object             # frozenset of states when complete
    terminals: object             # frozenset of terminal states when complete
    certified_terminals: frozenset  # terminals discovered and fully examined (also when truncated)
    Y: object                     # frozenset of (a, b) when complete, else None
    edges: int
    inspections: int
    max_frontier: int
    acyclic: object               # True/False when complete
    parent: dict                  # BFS parent map: state -> (previous state, coord) or None
    longest_path: object

    @property
    def signatures(self):
        return None if self.terminals is None else frozenset(signature(t[3], t[4]) for t in self.terminals)

    def shortest_path(self, terminal):
        path, node = [], terminal
        while self.parent[node] is not None:
            prev, c = self.parent[node]
            path.append(dict(coord=COORDS[c], source=prev, target=node))
            node = prev
        return path[::-1]


def signature(a, b):
    return ('A' if a else '') + ('B' if b else '')


def explore(x, w, start=None, frozen=(), input_feedback=True, cap=None):
    """Breadth-first enumeration of every strictly decreasing single-coordinate update.

    One inspection examines one (coordinate, proposed value) at a dequeued state, in the
    declared i1, i2, k, a, b and -1, 0, 1 order. Exhaustion is checked before each
    inspection; a zero cap is truncated; finishing exactly at the cap is complete.
    """
    if not isinstance(x, Query):
        raise TypeError('x must be a Query')
    check_weights(w)
    w = tuple(w)
    start = x.start() if start is None else tuple(start)
    if len(start) != 5 or any(start[c] not in x.allowed(c) for c in range(5)):
        raise ValueError('start state outside the local domains')
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError('cap must be a non-negative int or None')
    movable = [c for c in range(5) if c not in frozen]
    parent = {start: None}
    succ = {}
    frontier = deque([start])
    max_frontier, work, edges = 1, 0, 0
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
                if delta(z, x, w, c, v, input_feedback) < 0:
                    nz = z[:c] + (v,) + z[c + 1:]
                    out.append(nz)
                    edges += 1
                    if nz not in parent:
                        parent[nz] = (z, c)
                        frontier.append(nz)
                        max_frontier = max(max_frontier, len(frontier))
            if truncated:
                break
        if truncated:
            break
        succ[z] = out
        if not out:
            certified.append(z)
    common = dict(query=x, w=w, start=start, frozen=tuple(frozen), input_feedback=input_feedback,
                  certified_terminals=frozenset(certified), edges=edges, inspections=work,
                  max_frontier=max_frontier, parent=parent)
    if truncated:
        return Result(status='truncated', reachable=None, terminals=None, Y=None, acyclic=None,
                      longest_path=None, **common)
    acyclic, longest = longest_paths(start, succ)
    terminals = frozenset(certified)
    return Result(status='complete', reachable=frozenset(parent), terminals=terminals,
                  Y=frozenset((t[3], t[4]) for t in terminals), acyclic=acyclic,
                  longest_path=longest, **common)


def longest_paths(start, succ):
    """Cycle check and longest path length by iterative DFS over the reachable graph."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(succ, WHITE)
    depth = {}
    stack = [(start, iter(succ[start]))]
    color[start] = GREY
    while stack:
        z, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            depth[z] = max((depth[n] + 1 for n in succ[z]), default=0)
            color[z] = BLACK
            stack.pop()
        elif color[nxt] == GREY:
            return False, None
        elif color[nxt] == WHITE:
            color[nxt] = GREY
            stack.append((nxt, iter(succ[nxt])))
    return True, depth[start]


@lru_cache(maxsize=None)
def outputs(x, w):
    """Pure memo of the canonical-start response set Y_w(x) and its inspection count."""
    r = explore(x, w)
    return r.Y, r.inspections


def core_disconnected(w):
    return w[:4] + (0, 0)


# ---------------------------------------------------------------- frozen split (label-free)

AXIS_CONTEXTS = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
MIXED_CONTEXTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SPLIT_SEEDS = range(5)
ORDER_SEEDS = range(5)
CHECKPOINTS = (0, 5, 10, 15)


def make_split(seed):
    """Query IDs only; no teacher or response is consulted."""
    train = []
    for ci, e in enumerate(AXIS_CONTEXTS):
        ids = [i for i, q in enumerate(QUERIES) if (q.e1, q.e2) == e]
        random.Random(1000 * seed + ci).shuffle(ids)
        train += ids[:3]
    within = [i for i, q in enumerate(QUERIES) if (q.e1, q.e2) in AXIS_CONTEXTS and i not in set(train)]
    mixed = [i for i, q in enumerate(QUERIES) if (q.e1, q.e2) in MIXED_CONTEXTS]
    orders = []
    for o in ORDER_SEEDS:
        order = sorted(train)
        random.Random(10000 * seed + o).shuffle(order)
        orders.append(order)
    return dict(seed=seed, train=sorted(train), within_context=within, mixed_evidence=mixed, orders=orders)


# ---------------------------------------------------------------- learners (no teacher access)

ALL_BITS = (1 << N_HYPOTHESES) - 1


class VersionSpace:
    """Exact 729-bit version space; full feedback keeps Y_w(x) == f, positive-only keeps f <= Y_w(x)."""
    def __init__(self, positive_only=False):
        self.positive_only = positive_only
        self.bits = ALL_BITS
        self.failed = False
        self.evaluations = self.inspections = 0

    def survivors(self):
        return [h for h in range(N_HYPOTHESES) if self.bits >> h & 1]

    def update(self, x, feedback):
        if not isinstance(x, Query):
            raise TypeError('x must be a Query')
        feedback = frozenset(feedback)
        if self.failed:
            return 'inconsistent_feedback'
        nxt = 0
        for h in self.survivors():
            y, work = outputs(x, hypothesis(h))
            self.evaluations += 1
            self.inspections += work
            if (feedback <= y) if self.positive_only else (y == feedback):
                nxt |= 1 << h
        if not nxt:
            self.failed = True          # V is not reset; no approximate model is chosen
            return 'inconsistent_feedback'
        self.bits = nxt
        return 'updated'

    def predict(self, x):
        """(certain, possible, certified set or None). Uncertain when survivors disagree."""
        sets = {outputs(x, hypothesis(h))[0] for h in self.survivors()}
        certain = frozenset.intersection(*sets)
        possible = frozenset.union(*sets)
        return certain, possible, next(iter(sets)) if len(sets) == 1 else None

    def serialize(self):
        return dict(schema=SCHEMA_VERSION, hypotheses=N_HYPOTHESES, failed=self.failed,
                    rule='positive_only' if self.positive_only else 'full_feedback',
                    membership=f'{self.bits:0183x}')


BLOCKS = ((0, 1), (2, 3), (4, 5))


class BlockLearner:
    """One six-coefficient vector, edited one block at a time on global set discrepancy.

    Local in parameter-edit support, not in credit information: the loss is the symmetric
    difference between the whole predicted and supplied output sets, computed by exact inference.
    One pass, at most one block update per example, no replay or stored examples.
    """
    def __init__(self):
        self.w = ZERO
        self.evaluations = self.inspections = self.commits = self.coefficients_changed = 0

    def _loss(self, x, w, feedback):
        y, work = outputs(x, w)
        self.evaluations += 1
        self.inspections += work
        return len(y ^ feedback)

    def neighbors(self):
        for block in BLOCKS:
            for vals in itertools.product(VALUES, repeat=2):
                cand = list(self.w)
                for j, v in zip(block, vals):
                    cand[j] = v
                cand = tuple(cand)
                if cand != self.w:
                    yield cand

    def update(self, x, feedback):
        feedback = frozenset(feedback)
        current = self._loss(x, self.w, feedback)
        if current == 0:
            return 'unchanged'
        best = min((self._loss(x, c, feedback), sum(u != v for u, v in zip(c, self.w)), c)
                   for c in self.neighbors())
        if best[0] < current:
            self.commits += 1
            self.coefficients_changed += best[1]
            self.w = best[2]
            return 'updated'
        return 'unchanged'

    def predict(self, x):
        return outputs(x, self.w)[0]


class ExactMemory:
    """Stored sets for seen queries; 'unknown' (None) otherwise."""
    def __init__(self):
        self.table = {}

    def update(self, x, feedback):
        self.table[x] = frozenset(feedback)
        return 'updated'

    def predict(self, x):
        return self.table.get(x)


# ---------------------------------------------------------------- phase A controls and diagnostics

I1, I2, K, A, B = range(5)


def independent_product(x, w):
    """I fixed at evidence, core couplings removed, A and B evolved separately; Cartesian product."""
    wd = core_disconnected(w)
    a_set = {t[A] for t in explore(x, wd, frozen=(I1, I2, K, B)).terminals}
    b_set = {t[B] for t in explore(x, wd, frozen=(I1, I2, K, A)).terminals}
    return frozenset(itertools.product(sorted(a_set), sorted(b_set)))


def controls(x, w):
    """Response sets of the declared controls, same evidence and domains."""
    out = dict(
        full=explore(x, w).Y,
        uncoupled=explore(x, ZERO).Y,
        core_disconnected=explore(x, core_disconnected(w)).Y,
        input_frozen=explore(x, w, frozen=(I1, I2)).Y,
        input_feedback_removed=explore(x, w, input_feedback=False).Y,
        independent_product=independent_product(x, w),
    )
    for kv in TERNARY:   # initialization intervention: k starts at and stays at kv
        out[f'core_fixed_{kv}'] = explore(x, w, start=(x.e1, x.e2, kv, 0, 0), frozen=(K,)).Y
    return out


def decreases(z, x, w, coord, v):
    return delta(z, x, w, coord, v) < 0


def influence_witnesses(x, w, states, targets, neighbors):
    """Pairs of states (both in `states`) differing only in one neighbor coordinate, for which
    some move of a target coordinate decreases energy in one state and not in the other."""
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
    """Clamp one satellite at each value of its domain (initialized there, never moved);
    return {value: terminal mode set of the other satellite} with I and K free."""
    other = B if clamped == A else A
    out = {}
    for v in x.allowed(clamped):
        start = [x.e1, x.e2, 0, 0, 0]
        start[clamped] = v
        r = explore(x, w, start=tuple(start), frozen=(clamped,))
        out[v] = frozenset(t[other] for t in r.terminals)
    return out
