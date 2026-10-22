"""Finite feed-forward interference against a set-union independent control."""
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Algebra:
    """State 0 is the distinguished quiescent identity; labels are categorical."""

    table: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        n = len(self.table)
        if not n or any(len(row) != n for row in self.table):
            raise ValueError("table must be nonempty and square")
        if any(type(x) is not int or not 0 <= x < n for row in self.table for x in row):
            raise ValueError("table entries must be state indices")
        if any(self.table[0][x] != x or self.table[x][0] != x for x in range(n)):
            raise ValueError("state 0 must be a two-sided identity")

    def commutative(self):
        return all(self.table[a][b] == self.table[b][a]
                   for a, b in product(range(len(self.table)), repeat=2))

    def associative(self):
        return all(self.table[self.table[a][b]][c] == self.table[a][self.table[b][c]]
                   for a, b, c in product(range(len(self.table)), repeat=3))


def observe(state):
    """The complete nonquiescent core-state signature, without numeric scoring."""
    return frozenset() if state == 0 else frozenset({state})


def response(algebra, left, right, edges=(True, True)):
    """Two clamped satellites feed a stateless core; missing links transmit 0."""
    n = len(algebra.table)
    if any(type(x) is not int or not 0 <= x < n for x in (left, right)):
        raise ValueError("unknown satellite state")
    if len(edges) != 2 or any(type(edge) is not bool for edge in edges):
        raise ValueError("expected two Boolean edge indicators")
    a, b = left if edges[0] else 0, right if edges[1] else 0
    joint = observe(algebra.table[a][b])
    independent = observe(a) | observe(b)
    return joint, independent


XOR = Algebra(((0, 1), (1, 0)))
SATURATING = Algebra(((0, 1, 2), (1, 2, 2), (2, 2, 2)))
OR = Algebra(((0, 1), (1, 1)))


def commutative_identity_tables(size):
    """Every labeled commutative table with fixed identity, exactly once."""
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    cells = [(a, b) for a in range(1, size) for b in range(a, size)]
    for values in product(range(size), repeat=len(cells)):
        table = [[0] * size for _ in range(size)]
        for x in range(size):
            table[0][x] = table[x][0] = x
        for (a, b), value in zip(cells, values):
            table[a][b] = table[b][a] = value
        yield Algebra(tuple(tuple(row) for row in table))
