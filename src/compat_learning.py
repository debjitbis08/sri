"""Learned compatibility v1: mode-level core learning with set-valued responses."""
from itertools import combinations
import random


M = 6
STAR_RULES = ('join', 'merge')
HALF_SPLITS, HALF_SEARCH = 100, 200_000
CONTROLS = ('memorize', 'ternary', 'enumerate')


def rectangles(*blocks):
    return frozenset((a, b) for A, B in blocks for a in A for b in B)


def random_relation(seed=0, size=14):
    rng = random.Random(seed)
    cells = [(a, b) for a in range(M) for b in range(M)]
    while True:
        rel = frozenset(rng.sample(cells, size))
        if {a for a, _ in rel} == set(range(M)) and {b for _, b in rel} == set(range(M)):
            return rel


RELATIONS = {
    'blocks': rectangles(((0, 1, 2), (0, 1, 2)), ((3, 4), (3, 4)), ((5,), (5,))),
    'overlap': rectangles(((0, 1, 2, 3), (0, 1)), ((2, 3, 4, 5), (2, 3, 4))),
    'random': random_relation(),
}


class Learner:
    """Learned supports. Star rules hold per-core-mode rectangles; controls hold pair tables."""
    def __init__(self, kind, n):
        self.kind, self.n = kind, n
        self.A = [set() for _ in range(n)]
        self.B = [set() for _ in range(n)]
        self.pairs, self.triples = set(), set()
        self.merges = self.forced = 0

    def star_energy(self, a, b, k):
        return -(a in self.A[k]) - (b in self.B[k])

    def choose(self, a, b):
        energies = [self.star_energy(a, b, k) for k in range(self.n)]
        best = min(energies)
        if best == 0:
            unused = [k for k in range(self.n) if not self.A[k] and not self.B[k]]
            if not unused:
                self.forced += 1
            return unused[0] if unused else 0
        return energies.index(best)

    def present(self, a, b):
        if self.kind == 'memorize':
            self.pairs.add((a, b))
            return
        if self.kind == 'enumerate':
            k = len(self.triples)
            self.triples.add((a, b, k))
            return
        if self.kind == 'merge':
            ka = {k for k in range(self.n) if a in self.A[k]}
            kb = {k for k in range(self.n) if b in self.B[k]}
            if ka and kb and not ka & kb:
                target = min(ka)
                for k in sorted(kb):
                    self.A[target] |= self.A[k]; self.B[target] |= self.B[k]
                    self.A[k], self.B[k] = set(), set()
                    self.merges += 1
        k = self.choose(a, b)
        if self.kind == 'ternary':
            self.triples.add((a, b, k))
        self.A[k].add(a); self.B[k].add(b)  # ternary also tracks modes used, for choice only

    def core_modes(self):
        if self.kind == 'memorize':
            return 1
        if self.kind == 'enumerate':
            return max(1, len(self.triples))
        return self.n

    def energy(self, a, b, k):
        if self.kind in STAR_RULES:
            return self.star_energy(a, b, k)
        if self.kind == 'memorize':
            return -((a, b) in self.pairs)
        return -((a, b, k) in self.triples)


def attractor_responses(learner, cue, side, clamp=None):
    """Project sinks of the non-increasing whole-module move graph onto the free satellite."""
    ks = [clamp] if clamp is not None else list(range(learner.core_modes()))
    states = [(x, k) for x in range(M) for k in ks]

    def e(state):
        x, k = state
        return learner.energy(cue, x, k) if side == 'A' else learner.energy(x, cue, k)

    def neighbours(state):
        x, k = state
        out = [(y, k) for y in range(M) if y != x]
        out += [(x, j) for j in ks if j != k]
        return [t for t in out if e(t) <= e(state)]

    # Sink strongly connected components (Kosaraju) of the non-increasing move graph.
    graph = {s: neighbours(s) for s in states}
    order, seen = [], set()
    for root in states:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(graph[root]))]
        while stack:
            node, it = stack[-1]
            nxt = next((t for t in it if t not in seen), None)
            if nxt is None:
                order.append(node); stack.pop()
            else:
                seen.add(nxt); stack.append((nxt, iter(graph[nxt])))
    reverse = {s: [] for s in states}
    for s, outs in graph.items():
        for t in outs:
            reverse[t].append(s)
    comp = {}
    for root in reversed(order):
        if root in comp:
            continue
        comp[root] = root
        stack = [root]
        while stack:
            for t in reverse[stack.pop()]:
                if t not in comp:
                    comp[t] = root; stack.append(t)
    leaves = {c for c in set(comp.values())
              if all(comp[t] == c for s in states if comp[s] == c for t in graph[s])}
    sinks = [s for s in states if comp[s] in leaves]
    return {x for x, _ in sinks}


def predicted_responses(learner, cue, side):
    """Proposition P-response."""
    if learner.kind in STAR_RULES:
        own, other = (learner.A, learner.B) if side == 'A' else (learner.B, learner.A)
        return set().union(*[other[k] for k in range(learner.n) if cue in own[k]])
    pairs = learner.pairs if learner.kind == 'memorize' else {(a, b) for a, b, _ in learner.triples}
    return {b for a, b in pairs if a == cue} if side == 'A' else {a for a, b in pairs if b == cue}


def components(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(('A', a))] = find(('B', b))
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return {(frozenset(x for s, x in g if s == 'A'), frozenset(x for s, x in g if s == 'B'))
            for g in groups.values()}


def splits(relation):
    pairs = sorted(relation)
    used_a, used_b = {a for a, _ in pairs}, {b for _, b in pairs}

    def admissible(held):
        train = [p for p in pairs if p not in held]
        return {a for a, _ in train} == used_a and {b for _, b in train} == used_b
    families = {
        'leave_one_out': [frozenset([p]) for p in pairs],
        'leave_two_out': [frozenset(c) for c in combinations(pairs, 2)],
    }
    out = {name: ([h for h in held if admissible(h)], sum(not admissible(h) for h in held))
           for name, held in families.items()}
    half, rejected = [], 0
    for seed in range(HALF_SEARCH):
        h = frozenset(random.Random(seed).sample(pairs, len(pairs)//2))
        if not admissible(h):
            rejected += 1
        elif h not in half:
            half.append(h)
            if len(half) == HALF_SPLITS:
                break
    out['half'] = (half, rejected)
    return out


def cues(relation):
    """Modes the relation uses; untrained modes are not cues (amendment after first run)."""
    return {'A': sorted({a for a, _ in relation}), 'B': sorted({b for _, b in relation})}


def evaluate(learner, relation, held):
    used = cues(relation)
    valid = {side: {c: ({b for a, b in relation if a == c} if side == 'A' else
                        {a for a, b in relation if b == c}) for c in used[side]} for side in 'AB'}
    resp, violations = {}, 0
    for side in 'AB':
        for c in used[side]:
            r = attractor_responses(learner, c, side)
            violations += r != predicted_responses(learner, c, side)
            resp[(side, c)] = r
    responses = [(s, c, x) for (s, c), r in resp.items() for x in r]
    correct = sum(x in valid[s][c] for s, c, x in responses)
    recovered = sum(b in resp[('A', a)] and a in resp[('B', b)] for a, b in held)
    single = [(s, c) for s in 'AB' for c in used[s] if len(valid[s][c]) == 1]
    multi = [(s, c) for s in 'AB' for c in used[s] if len(valid[s][c]) > 1]
    participation = 0
    if learner.kind in STAR_RULES:
        for c in used['A']:
            full = resp[('A', c)]
            if all(attractor_responses(learner, c, 'A', clamp=k) < full
                   for k in range(learner.n)):
                participation += 1
    return dict(
        heldout_recall=recovered/len(held) if held else None,
        seen_recall=sum(b in resp[('A', a)] for a, b in relation - held)/len(relation - held),
        precision=correct/len(responses) if responses else 1.0,
        invalid_responses=len(responses) - correct,
        exact_cues=sum(resp[k] == valid[k[0]][k[1]] for k in resp)/len(resp),
        single_cues_unique=sum(len(resp[k]) == 1 for k in single)/len(single) if single else None,
        multi_cues_multiple=sum(len(resp[k]) > 1 for k in multi)/len(multi) if multi else None,
        cues_needing_dynamic_core=participation,
        response_proposition_violations=violations,
        resp={k: frozenset(v) for k, v in resp.items()},
    )


def prefix_max_components(order):
    """Largest component count over training prefixes: merge needs this many core modes."""
    return max(len(components(order[:i+1])) for i in range(len(order)))


def train(kind, n, order):
    learner = Learner(kind, n)
    for a, b in order:
        learner.present(a, b)
    return learner
