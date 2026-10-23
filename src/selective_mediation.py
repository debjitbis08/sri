"""Selective satellite mediation v1: exact gated repair, version-space learning and controls.

See docs/selective-satellite-mediation-v1-contract.md. Two satellites A, B with modes 0..3,
immutable allowed domains, an external two-bit context, and four XOR primitives whose gates
come from a fixed vocabulary. A hypothesis assigns one gate to each primitive (256 in all).
Learner APIs receive queries and feedback sets only; teachers live with the evaluator.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import random


SCHEMA_VERSION = 'selective-satellite-mediation-v1'
MODES = (0, 1, 2, 3)
DOMAINS = (frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3}),
           frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 1, 2, 3}))
CONTEXTS = ((0, 0), (0, 1), (1, 0), (1, 1))   # (c0, c1); index order 00, 01, 10, 11
CONTEXT_NAMES = ('00', '01', '10', '11')
GATES = ('never', 'c0', 'c1', 'always')
PRIMITIVES = (('A_low', 'A', 1), ('A_high', 'A', 2), ('B_low', 'B', 1), ('B_high', 'B', 2))
N_HYPOTHESES = len(GATES) ** len(PRIMITIVES)
ALL_HYPOTHESES = (1 << N_HYPOTHESES) - 1


@dataclass(frozen=True)
class Query:
    """Immutable evidence: domain indices, starting modes and context index."""
    domain_a: int
    domain_b: int
    a0: int
    b0: int
    context: int

    def __post_init__(self):
        for name in ('domain_a', 'domain_b', 'a0', 'b0', 'context'):
            v = getattr(self, name)
            if type(v) is not int:
                raise TypeError(f'{name} must be int, got {type(v).__name__}')
        if not 0 <= self.domain_a < len(DOMAINS) or not 0 <= self.domain_b < len(DOMAINS):
            raise ValueError('domain index out of range')
        if not 0 <= self.context < len(CONTEXTS):
            raise ValueError('context index out of range')
        if self.a0 not in DOMAINS[self.domain_a] or self.b0 not in DOMAINS[self.domain_b]:
            raise ValueError('starting mode outside its allowed domain')

    @property
    def c(self):
        return CONTEXTS[self.context]

    def key(self):
        return (self.context, self.domain_a, self.domain_b, self.a0, self.b0)

    def to_json(self):
        return dict(domain_a=sorted(DOMAINS[self.domain_a]), domain_b=sorted(DOMAINS[self.domain_b]),
                    a0=self.a0, b0=self.b0, context=CONTEXT_NAMES[self.context])


def all_queries():
    """The 576 queries sorted by (context, domain_A, domain_B, a0, b0)."""
    return tuple(Query(da, db, a0, b0, c)
                 for c in range(len(CONTEXTS))
                 for da in range(len(DOMAINS)) for db in range(len(DOMAINS))
                 for a0 in sorted(DOMAINS[da]) for b0 in sorted(DOMAINS[db]))


QUERIES = all_queries()
QUERY_INDEX = {q: i for i, q in enumerate(QUERIES)}


def hypothesis_gates(h):
    """Gate indices in primitive order; h = g0*64 + g1*16 + g2*4 + g3 (lexicographic)."""
    if type(h) is not int or not 0 <= h < N_HYPOTHESES:
        raise ValueError('hypothesis id out of range')
    return tuple((h >> (2 * (3 - i))) & 3 for i in range(4))


def hypothesis_id(gates):
    """Inverse of hypothesis_gates; accepts gate names or indices."""
    idx = [GATES.index(g) if isinstance(g, str) else g for g in gates]
    if len(idx) != 4 or any(type(g) is not int or not 0 <= g < 4 for g in idx):
        raise ValueError('a hypothesis assigns one of four gates to each of four primitives')
    return idx[0] * 64 + idx[1] * 16 + idx[2] * 4 + idx[3]


def gate_open(gate, c):
    return (False, c[0] == 1, c[1] == 1, True)[gate]


TEACHERS = {
    'T0': hypothesis_id(('c0', 'never', 'c1', 'never')),
    'T1': hypothesis_id(('c0', 'c1', 'c1', 'c0')),
    'T2': hypothesis_id(('always', 'never', 'never', 'c0')),
    'T3': hypothesis_id(('never', 'never', 'never', 'never')),
}
NO_MEDIATION = hypothesis_id(('never',) * 4)
PERMISSIVE = hypothesis_id(('always',) * 4)


# ---------------------------------------------------------------- exact inference

@dataclass(frozen=True)
class Step:
    primitive: str
    gate: str
    source: tuple
    target: tuple


@dataclass(frozen=True)
class Inference:
    """Result of exact realization. `response` is set only when complete."""
    query: Query
    hypothesis: int
    status: str                     # 'complete' | 'truncated'
    feasible: object                # True/False when complete, None when truncated
    response: object                # frozenset of t when complete, else None
    partial_response: object        # successful t discovered before truncation, else None
    representative: object          # smallest t, only when complete and feasible
    path: tuple                     # Steps of the first BFS shortest path to (t, t)
    visited: int
    edge_checks: int
    max_frontier: int
    invalid_attempts: tuple         # (state, primitive) whose gate was open but target left the domain

    def certificate(self):
        """Transient core certificate: selected hypothesis and the primitive used at each step."""
        return dict(hypothesis=self.hypothesis, primitives=[s.primitive for s in self.path])


def successor(query, state, primitive_index):
    """Destination of one primitive, ignoring gates and domains."""
    _, side, mask = PRIMITIVES[primitive_index]
    a, b = state
    return (a ^ mask, b) if side == 'A' else (a, b ^ mask)


def locally_valid(query, state):
    return state[0] in DOMAINS[query.domain_a] and state[1] in DOMAINS[query.domain_b]


def infer(query, hypothesis, cap=None):
    """Breadth-first realization from (a0, b0) in canonical primitive order.

    One unit of work examines one primitive at a dequeued state. Exhaustion is checked
    before each examination; a zero cap is truncated; work ending exactly at the cap is
    complete. Successful states are expanded too, so the response set is complete.
    """
    if not isinstance(query, Query):
        raise TypeError('query must be a Query')
    gates = hypothesis_gates(hypothesis)
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError('cap must be a non-negative int or None')
    start = (query.a0, query.b0)
    parent = {start: None}
    frontier = deque([start])
    max_frontier = 1
    work = 0
    invalid = []
    truncated = cap == 0
    while frontier and not truncated:
        state = frontier.popleft()
        for i in range(len(PRIMITIVES)):
            if cap is not None and work >= cap:
                truncated = True
                break
            work += 1
            if not gate_open(gates[i], query.c):
                continue
            nxt = successor(query, state, i)
            if not locally_valid(query, nxt):
                invalid.append((state, PRIMITIVES[i][0]))
                continue
            if nxt not in parent:
                parent[nxt] = (state, i)
                frontier.append(nxt)
                max_frontier = max(max_frontier, len(frontier))
    found = frozenset(s[0] for s in parent if s[0] == s[1])
    common = dict(query=query, hypothesis=hypothesis, visited=len(parent), edge_checks=work,
                  max_frontier=max_frontier, invalid_attempts=tuple(invalid))
    if truncated:
        return Inference(status='truncated', feasible=None, response=None, partial_response=found,
                         representative=None, path=(), **common)
    rep = min(found) if found else None
    path = []
    if rep is not None:
        node = (rep, rep)
        while parent[node] is not None:
            prev, i = parent[node]
            path.append(Step(PRIMITIVES[i][0], GATES[gates[i]], prev, node))
            node = prev
        path.reverse()
    return Inference(status='complete', feasible=bool(found), response=found, partial_response=None,
                     representative=rep, path=tuple(path), **common)


@lru_cache(maxsize=None)
def realize(query, hypothesis):
    """Pure uncapped inference, memoized. Callers account work from the returned counts."""
    return infer(query, hypothesis)


def response(query, hypothesis):
    return realize(query, hypothesis).response


def verify_path(query, result):
    """Violations in a committed result: domain, gate, step and evidence-mutation counts."""
    gates = hypothesis_gates(result.hypothesis)
    names = [p[0] for p in PRIMITIVES]
    domain = gate = step = 0
    state = (query.a0, query.b0)
    for s in result.path:
        i = names.index(s.primitive)
        domain += not locally_valid(query, s.target)
        gate += not gate_open(gates[i], query.c) or s.gate != GATES[gates[i]]
        step += s.source != state or s.target != successor(query, state, i)
        state = s.target
    if result.representative is not None:
        step += state != (result.representative, result.representative)
    return dict(domain=domain, gate=gate, step=step, evidence=int(result.query != query))


def direct_response(query, gates):
    """Core-free control: execute gates directly by closure, with no path or certificate object."""
    da, db, c = DOMAINS[query.domain_a], DOMAINS[query.domain_b], CONTEXTS[query.context]
    a_moves = [m for g, (_, side, m) in zip(gates, PRIMITIVES) if side == 'A' and gate_open(g, c)]
    b_moves = [m for g, (_, side, m) in zip(gates, PRIMITIVES) if side == 'B' and gate_open(g, c)]
    seen = {(query.a0, query.b0)}
    changed = True
    while changed:
        changed = False
        for a, b in list(seen):
            for nxt in [(a ^ m, b) for m in a_moves] + [(a, b ^ m) for m in b_moves]:
                if nxt[0] in da and nxt[1] in db and nxt not in seen:
                    seen.add(nxt)
                    changed = True
    return frozenset(a for a, b in seen if a == b)


direct_response_cached = lru_cache(maxsize=None)(direct_response)


# ---------------------------------------------------------------- incompatibility classes

def conflict_class(query, teacher_response):
    """Exclusive classes, tested in the declared order. Needs the teacher: evaluator only."""
    if query.a0 == query.b0:
        return 'already_compatible'
    if teacher_response:
        return 'resolvable'
    if not DOMAINS[query.domain_a] & DOMAINS[query.domain_b]:
        return 'unresolved_disjoint_domain'
    if not response(query, PERMISSIVE):
        return 'unresolved_all_enabled'
    return 'unresolved_teacher_permission'


CLASSES = ('already_compatible', 'resolvable', 'unresolved_disjoint_domain',
           'unresolved_all_enabled', 'unresolved_teacher_permission')


# ---------------------------------------------------------------- frozen split (label-free)

SPLIT_SEEDS = range(5)
ORDER_SEEDS = range(5)
TRAIN_PER_CONTEXT = 16
TRAIN_CONTEXTS = (0, 1, 2)
CHECKPOINTS = (0, 12, 24, 48)


def make_split(seed):
    """Query indices only; no teacher, response or label is consulted."""
    chosen = []
    for c in TRAIN_CONTEXTS:
        indices = [i for i, q in enumerate(QUERIES) if q.context == c]
        random.Random(1000 * seed + c).shuffle(indices)
        chosen += indices[:TRAIN_PER_CONTEXT]
    train = sorted(chosen, key=lambda i: QUERIES[i].key())
    within = [i for i, q in enumerate(QUERIES) if q.context in TRAIN_CONTEXTS and i not in set(train)]
    unseen = [i for i, q in enumerate(QUERIES) if q.context == 3]
    orders = []
    for o in ORDER_SEEDS:
        order = list(train)
        random.Random(10000 * seed + o).shuffle(order)
        orders.append(order)
    return dict(seed=seed, train=train, within_context=within, unseen_context=unseen, orders=orders)


# ---------------------------------------------------------------- learners (no teacher access)

class VersionSpaceLearner:
    """Exact finite version space over the 256 gate hypotheses.

    Persistent state is a 256-bit membership integer plus schema metadata; no past
    query/label records are kept. Full feedback keeps h when R_h(x) == feedback; the
    positive-only ablation keeps h when feedback is a subset of R_h(x) and ignores empty
    feedback. The executable model is the smallest surviving id. Executed responses carry
    a transient core certificate (selected id and primitive per step).
    """
    def __init__(self, positive_only=False):
        self.positive_only = positive_only
        self.bits = ALL_HYPOTHESES
        self.failed = False
        self.updates = self.hypothesis_evaluations = self.update_edge_checks = 0
        self.inferences = self.inference_edge_checks = 0

    def survivors(self):
        return [h for h in range(N_HYPOTHESES) if self.bits >> h & 1]

    @property
    def selected(self):
        return (self.bits & -self.bits).bit_length() - 1 if self.bits else None

    def _keeps(self, query, h, feedback):
        r = realize(query, h)
        self.hypothesis_evaluations += 1
        self.update_edge_checks += r.edge_checks
        return feedback <= r.response if self.positive_only else r.response == feedback

    def update(self, query, feedback):
        """Returns 'updated', 'unchanged' or 'inconsistent_feedback' (no model is committed)."""
        if not isinstance(query, Query):
            raise TypeError('query must be a Query')
        feedback = frozenset(feedback)
        if not feedback <= set(MODES):
            raise ValueError('feedback must be a set of modes')
        if self.failed:
            return 'inconsistent_feedback'
        self.updates += 1
        if self.positive_only and not feedback:
            return 'unchanged'
        nxt = 0
        for h in self.survivors():
            if self._keeps(query, h, feedback):
                nxt |= 1 << h
        if not nxt:
            self.failed = True              # V is not reset and no replacement model is committed
            return 'inconsistent_feedback'
        changed = nxt != self.bits
        self.bits = nxt
        return 'updated' if changed else 'unchanged'

    def predict(self, query):
        """Executed response of the selected model, with its certificate path."""
        if self.failed:
            return None
        r = realize(query, self.selected)
        self.inferences += 1
        self.inference_edge_checks += r.edge_checks
        return r

    def serialize(self):
        return dict(schema=SCHEMA_VERSION, hypotheses=N_HYPOTHESES,
                    rule='positive_only' if self.positive_only else 'full_feedback',
                    failed=self.failed, membership=f'{self.bits:064x}')


class CoreFreeLearner:
    """Same version space and selection, executed by direct closure with no certificate object."""
    def __init__(self):
        self.bits = ALL_HYPOTHESES
        self.failed = False

    @property
    def selected(self):
        return (self.bits & -self.bits).bit_length() - 1 if self.bits else None

    def update(self, query, feedback):
        if self.failed:
            return 'inconsistent_feedback'
        feedback = frozenset(feedback)
        nxt = 0
        for h in range(N_HYPOTHESES):
            if self.bits >> h & 1 and direct_response_cached(query, hypothesis_gates(h)) == feedback:
                nxt |= 1 << h
        if not nxt:
            self.failed = True
            return 'inconsistent_feedback'
        self.bits = nxt
        return 'updated'

    def predict(self, query):
        return None if self.failed else direct_response_cached(query, hypothesis_gates(self.selected))


class ExactQueryMemory:
    """Stores feedback keyed by the entire query; unseen queries get the no-mediation response."""
    def __init__(self):
        self.table = {}

    def update(self, query, feedback):
        self.table[query] = frozenset(feedback)
        return 'updated'

    def predict(self, query):
        if query in self.table:
            return self.table[query]
        return frozenset({query.a0}) if query.a0 == query.b0 else frozenset()
