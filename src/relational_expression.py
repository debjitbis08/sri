"""Finite relational expressions and their evaluation (docs/relational-expression-core-design.md, stage 1).

Terms are entity references, relation references and named variables. Formulas are atoms (a relation, or a relation
variable, applied to a tuple of argument terms of any arity), conjunctions, negations and existentials over declared
finite domains. A query asks for bindings of named free variables; every other free variable is an error.

Two evaluators share this syntax and differ only in semantics, which are declared, not interchangeable:

  exact   three-valued (Kleene) truth: True, False or None (unknown). An atom's truth comes from an interpretation; a
          missing fact is unknown unless its relation is declared closed. Exists is False only when every assignment
          is False and every domain it ranges over is declared complete; an incomplete domain never yields False.
  soft    real-valued scores from a scorer (e.g. a comparator logit). And is min, Exists is max over the domain, Not is
          negation: max-min semantics, one fuzzy choice among several, chosen before any perceptual read. Scores are
          not probabilities and are not comparable across relations unless that is separately established.

Supplied algebra is explicit: a relation may be defined as the inverse of another (inverse(R)(x, ..., y) := R(y, ..., x),
argument order reversed). Every query records which defined relations it used, so results can separate supplied
algebra from learned behaviour. Nothing here identifies quantities: entities are opaque references whose meaning is
whatever the interpretation or scorer gives them.

Cost (per query, C one atom evaluation, cached per distinct atom within the query): requested variables with domain
sizes N_1..N_V and existential variables with sizes M_1..M_U cost at most prod(N) * prod(M) * A atom evaluations for A
atoms; distinct-atom caching bounds stored scores by the number of distinct ground atoms. One missing slot is O(N C);
a two-step chain from a fixed start with one witness variable and N endpoints evaluates at most N + N^2 distinct atoms,
O(N^2 C). Enumeration is exponential in the number of variables; domains must stay finite and small.
"""
from dataclasses import dataclass, field
from itertools import product


@dataclass(frozen=True)
class Ent:
    name: str


@dataclass(frozen=True)
class Rel:
    name: str


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Atom:
    rel: object  # Rel or Var
    args: tuple


@dataclass(frozen=True)
class And:
    parts: tuple


@dataclass(frozen=True)
class Not:
    part: object


@dataclass(frozen=True)
class Exists:
    vars: tuple  # of Var
    body: object


@dataclass(frozen=True)
class Inverse:
    """Supplied definition: the defined relation holds of args when `of` holds of the reversed args."""
    of: str


@dataclass(frozen=True)
class Domain:
    items: tuple
    complete: bool = True  # False: the world may hold answers outside these items


def atom(rel, *args):
    return Atom(rel if isinstance(rel, (Rel, Var)) else Rel(rel), tuple(args))


def conj(*parts):
    return And(tuple(parts))


def exists(vars, body):
    return Exists(tuple(vars), body)


def free_vars(f, bound=frozenset()):
    if isinstance(f, Atom):
        terms = (f.rel, *f.args)
        return {t for t in terms if isinstance(t, Var) and t not in bound}
    if isinstance(f, And):
        return set().union(*(free_vars(p, bound) for p in f.parts))
    if isinstance(f, Not):
        return free_vars(f.part, bound)
    if isinstance(f, Exists):
        return free_vars(f.body, bound | set(f.vars))
    raise TypeError(f'not a formula: {f!r}')


class _Run:
    """Shared machinery: term resolution, supplied definitions, per-query atom cache and counts."""

    def __init__(self, definitions, domains):
        self.definitions, self.domains = dict(definitions or {}), domains
        self.cache, self.used = {}, set()
        self.atom_evaluations = 0

    def ground(self, f, env):
        rel = env[f.rel] if isinstance(f.rel, Var) else f.rel
        if not isinstance(rel, Rel):
            raise TypeError(f'relation variable bound to a non-relation: {rel!r}')
        args = tuple(env[a] if isinstance(a, Var) else a for a in f.args)
        if any(isinstance(a, (Var, Rel)) for a in args):
            raise TypeError(f'argument is not an entity: {args!r}')
        name = rel.name
        while name in self.definitions:  # supplied algebra, recorded
            d = self.definitions[name]
            self.used.add(name)
            if isinstance(d, Inverse):
                name, args = d.of, args[::-1]
            else:
                raise TypeError(f'unknown definition {d!r}')
        return name, args

    def atom_value(self, f, env):
        key = self.ground(f, env)
        if key not in self.cache:
            self.atom_evaluations += 1
            self.cache[key] = self.base(*key)
        return self.cache[key]

    def domain(self, v):
        if v.name not in self.domains:
            raise KeyError(f'no domain declared for {v.name}')
        return self.domains[v.name]

    def assignments(self, vars, env):
        doms = [self.domain(v) for v in vars]
        for values in product(*(d.items for d in doms)):
            yield {**env, **dict(zip(vars, values))}  # inner bindings shadow outer ones


class _Exact(_Run):
    def __init__(self, interpretation, definitions, domains):
        super().__init__(definitions, domains)
        self.base = interpretation

    def value(self, f, env):
        if isinstance(f, Atom):
            return self.atom_value(f, env)
        if isinstance(f, And):
            vals = [self.value(p, env) for p in f.parts]
            return False if False in vals else (True if all(v is True for v in vals) else None)
        if isinstance(f, Not):
            v = self.value(f.part, env)
            return None if v is None else not v
        if isinstance(f, Exists):
            vals = [self.value(f.body, e) for e in self.assignments(f.vars, env)]
            if True in vals:
                return True
            complete = all(self.domain(v).complete for v in f.vars)
            return False if complete and all(v is False for v in vals) else None
        raise TypeError(f'not a formula: {f!r}')

    def witnesses(self, f, env):
        if not isinstance(f, Exists):
            return []
        return [tuple(e[v] for v in f.vars) for e in self.assignments(f.vars, env) if self.value(f.body, e) is True]


class _Soft(_Run):
    def __init__(self, scorer, definitions, domains):
        super().__init__(definitions, domains)
        self.base = scorer

    def value(self, f, env):
        if isinstance(f, Atom):
            return float(self.atom_value(f, env))
        if isinstance(f, And):
            return min(self.value(p, env) for p in f.parts)
        if isinstance(f, Not):
            return -self.value(f.part, env)
        if isinstance(f, Exists):
            vals = [self.value(f.body, e) for e in self.assignments(f.vars, env)]
            return max(vals) if vals else float('-inf')  # an empty domain supports nothing; not evidence of falsehood
        raise TypeError(f'not a formula: {f!r}')

    def witness(self, f, env):
        if not isinstance(f, Exists):
            return None
        best = max(self.assignments(f.vars, env), key=lambda e: self.value(f.body, e), default=None)
        return None if best is None else tuple(best[v] for v in f.vars)


@dataclass
class Answers:
    requested: tuple
    true: list = field(default_factory=list)
    false: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    witnesses: dict = field(default_factory=dict)
    domain_complete: bool = True
    supplied_definitions_used: frozenset = frozenset()
    atom_evaluations: int = 0


@dataclass
class Ranking:
    requested: tuple
    ranked: list = field(default_factory=list)  # (binding, score), highest first; ties keep domain order
    witnesses: dict = field(default_factory=dict)
    supplied_definitions_used: frozenset = frozenset()
    atom_evaluations: int = 0


def _check(formula, requested):
    extra = free_vars(formula) - set(requested)
    if extra:
        raise ValueError(f'free variables not requested: {sorted(v.name for v in extra)}')


def query(formula, requested, domains, interpretation, definitions=None):
    """Exact three-valued answers: each binding of the requested variables is true, false or unknown."""
    requested = tuple(requested)
    _check(formula, requested)
    run = _Exact(interpretation, definitions, domains)
    out = Answers(requested, domain_complete=all(run.domain(v).complete for v in requested))
    for env in run.assignments(requested, {}):
        b = tuple(env[v] for v in requested)
        v = run.value(formula, env)
        (out.true if v is True else out.false if v is False else out.unknown).append(b)
        if v is True and isinstance(formula, Exists):
            out.witnesses[b] = run.witnesses(formula, env)
    out.supplied_definitions_used, out.atom_evaluations = frozenset(run.used), run.atom_evaluations
    return out


def truth(formula, interpretation, definitions=None, domains=None):
    """Exact truth of a closed formula (True, False or None), with the supplied definitions it used."""
    a = query(formula, (), domains or {}, interpretation, definitions)
    return (True if a.true else False if a.false else None), a.supplied_definitions_used


def rank(formula, requested, domains, scorer, definitions=None):
    """Soft (max-min) scores for every binding of the requested variables, highest first, with argmax witnesses."""
    requested = tuple(requested)
    _check(formula, requested)
    run = _Soft(scorer, definitions, domains)
    out = Ranking(requested)
    scored = []
    for env in run.assignments(requested, {}):
        b = tuple(env[v] for v in requested)
        scored.append((b, run.value(formula, env)))
        if isinstance(formula, Exists):
            out.witnesses[b] = run.witness(formula, env)
    out.ranked = sorted(scored, key=lambda x: -x[1])  # stable
    out.supplied_definitions_used, out.atom_evaluations = frozenset(run.used), run.atom_evaluations
    return out


def precision_recall(predicted, positives):
    """Multi-answer metric over binding sets. With no predictions, precision is undefined (None)."""
    predicted, positives = set(predicted), set(positives)
    hit = len(predicted & positives)
    return (hit / len(predicted) if predicted else None), (hit / len(positives) if positives else None)


class Table:
    """A finite truth table: facts maps (relation, args) to True or False. Missing facts are unknown unless the
    relation is in `closed`, where they are False."""

    def __init__(self, facts, closed=()):
        self.facts, self.closed = dict(facts), set(closed)

    def __call__(self, rel, args):
        if (rel, args) in self.facts:
            return self.facts[(rel, args)]
        return False if rel in self.closed else None
