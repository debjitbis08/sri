"""Minimal finite monotone resonance laboratory. Not a model of intelligence."""
from dataclasses import dataclass, field
from typing import Dict, Tuple, FrozenSet
import random

@dataclass(frozen=True)
class Node:
    module: str
    layer: int  # 0 = interior
    plasticity: int = 0  # metadata for future experimental learning rules

@dataclass
class Model:
    nodes: Dict[int, Node]
    # Undirected links with bounded integer coupling (0..255).
    links: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def link(self, a: int, b: int, strength: int) -> None:
        assert a != b and a in self.nodes and b in self.nodes
        assert 0 <= strength <= 255
        self.links[tuple(sorted((a, b)))] = strength

    def neighbors(self, node: int, threshold: int):
        for (a, b), w in sorted(self.links.items()):
            if w >= threshold:
                if a == node: yield b
                elif b == node: yield a

@dataclass(frozen=True)
class Result:
    active: FrozenSet[int]
    rounds: int
    traversals: int


def settle(model: Model, seeds, threshold: int = 1, max_rounds: int | None = None) -> Result:
    """Least fixed point of inflationary reachability; deliberately simple baseline.

    With max_rounds=None, returns complete closure. With a cap, returns a partial
    approximation (it must never be described as a converged result).
    """
    assert 0 <= threshold <= 255
    active = set(seeds)
    assert active <= model.nodes.keys()
    frontier = set(active)
    rounds = traversals = 0
    while frontier and (max_rounds is None or rounds < max_rounds):
        new = set()
        for a in sorted(frontier):
            for b in model.neighbors(a, threshold):
                traversals += 1
                if b not in active: new.add(b)
        active.update(new)
        frontier = new
        rounds += 1
    return Result(frozenset(active), rounds, traversals)


def dream_seed(model: Model, rng: random.Random) -> int:
    """Endogenous seed sampled from the current structure, no replay buffer."""
    ids = sorted(model.nodes)
    if not ids: raise ValueError('empty model')
    return rng.choice(ids)


def example() -> Model:
    nodes = {0: Node('input', 0), 1: Node('core', 0),
             2: Node('math', 0), 3: Node('math', 1),
             4: Node('spatial', 0), 5: Node('spatial', 1),
             6: Node('stm', 0, 10), 7: Node('stm', 1, 10)}
    m = Model(nodes)
    for a, b, w in [(0,6,220),(6,7,205),(6,1,240),(1,2,225),
                     (2,3,80),(1,4,215),(4,5,70),(3,5,35)]:
        m.link(a,b,w)
    return m
