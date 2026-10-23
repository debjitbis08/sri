"""Illustrative collective role binding; not the OP-04 architecture.

The builder searches bindings, then commits one whole immutable configuration.
Its recursion is an implementation detail, not a sequence of module updates.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Satellite:
    name: str
    role: str
    sites: tuple[str, ...]
    direct_site: str

    def __post_init__(self):
        if not self.name or not self.role or not self.sites:
            raise ValueError('satellite requires a name, role and nonempty site domain')
        if any(not isinstance(site, str) or not site for site in self.sites):
            raise ValueError('sites must be nonempty strings')
        if len(set(self.sites)) != len(self.sites) or self.direct_site not in self.sites:
            raise ValueError('sites must be distinct and contain the direct binding')


@dataclass(frozen=True)
class Structure:
    satellites: tuple[Satellite, ...]

    def __post_init__(self):
        if len({s.name for s in self.satellites}) != len(self.satellites):
            raise ValueError('satellite names must be unique')


@dataclass(frozen=True)
class InputModule:
    """Immutable external evidence: which satellite roles must participate."""
    active: tuple[str, ...]


@dataclass(frozen=True)
class SatelliteConfiguration:
    name: str
    role: str
    site: str | None


@dataclass(frozen=True)
class CoreConfiguration:
    # Scoped names prevent equal local role labels from becoming one global role.
    bindings: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class OutputModule:
    # Independent observation interface, constrained to agree with satellite state.
    bindings: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class Configuration:
    input: InputModule
    core: CoreConfiguration
    satellites: tuple[SatelliteConfiguration, ...]
    output: OutputModule
    # Endogenous participation condition; separate from immutable input evidence.
    context: tuple[str, ...] = ()


@dataclass(frozen=True)
class Work:
    candidate_extensions: int
    collision_rejections: int
    max_partial_bindings: int
    terminal_assignments: int


@dataclass(frozen=True)
class Result:
    status: str  # complete or truncated, independent of feasibility
    feasible: bool | None
    configuration: Configuration | None
    work: Work


def _active(structure, evidence, context=()):
    names = evidence.active + context
    if len(set(names)) != len(names):
        raise ValueError('input contains duplicate participants')
    inventory = {s.name: s for s in structure.satellites}
    if any(name not in inventory for name in names):
        raise ValueError('input names an unknown satellite')
    return tuple(inventory[name] for name in sorted(names))


def initial_configuration(structure, evidence, context=()):
    _active(structure, evidence, context)
    return Configuration(evidence, CoreConfiguration(), tuple(
        SatelliteConfiguration(s.name, s.role, None) for s in structure.satellites),
        OutputModule(), context)


def valid_response(structure, state, mediated):
    """Check whole-configuration agreement, not only the observed binding list."""
    active = _active(structure, state.input, state.context)
    specs = {s.name: s for s in structure.satellites}
    if len(state.satellites) != len(specs):
        return False
    configs = {s.name: s for s in state.satellites}
    if set(configs) != set(specs):
        return False
    selected = set(state.input.active + state.context)
    bindings = []
    for name, spec in specs.items():
        config = configs[name]
        if config.role != spec.role:
            return False
        if name not in selected:
            if config.site is not None:
                return False
        elif config.site not in spec.sites:
            return False
        elif not mediated and config.site != spec.direct_site:
            return False
    for spec in active:
        bindings.append((spec.name, spec.role, configs[spec.name].site))
    expected = tuple(bindings)
    if len({site for _, _, site in expected}) != len(expected):
        return False
    return (state.output.bindings == expected and
            state.core.bindings == (expected if mediated else ()))


def realize(structure, before, mediated=True, max_extensions=None, context=None):
    """Reconstruct a response using a declared deterministic joint selector.

    Supports one initial or previously valid response, not arbitrary damage.
    A cap returns unknown without committing a partial response. Old bindings
    are not privileged: the selection is lexicographic by name, then site.
    An explicit context intervention changes endogenous satellite participation
    while keeping external evidence and the existing configuration intact until
    the joint result is committed.
    """
    if type(mediated) is not bool:
        raise ValueError('mediated must be Boolean')
    if max_extensions is not None and (type(max_extensions) is not int or max_extensions < 0):
        raise ValueError('max_extensions must be a nonnegative integer or None')
    _active(structure, before.input, before.context)
    if before != initial_configuration(structure, before.input, before.context) and not any(
            valid_response(structure, before, mode) for mode in (False, True)):
        raise ValueError('expected an initial or previously valid configuration')
    participation = before.context if context is None else context
    active = _active(structure, before.input, participation)
    extensions = collisions = depth = terminals = 0
    chosen = []
    used = set()

    class BudgetReached(Exception):
        pass

    def search(index):
        nonlocal extensions, collisions, depth, terminals
        if index == len(active):
            terminals += 1
            return tuple(chosen)
        spec = active[index]
        domain = sorted(spec.sites) if mediated else (spec.direct_site,)
        for site in domain:
            if max_extensions is not None and extensions >= max_extensions:
                raise BudgetReached
            extensions += 1
            if site in used:
                collisions += 1
                continue
            used.add(site)
            chosen.append((spec.name, spec.role, site))
            depth = max(depth, len(chosen))
            answer = search(index + 1)
            chosen.pop()
            used.remove(site)
            if answer is not None:
                return answer
        return None

    try:
        binding = search(0)
    except BudgetReached:
        return Result('truncated', None, None, Work(extensions, collisions, depth, terminals))
    work = Work(extensions, collisions, depth, terminals)
    if binding is None:
        return Result('complete', False, None, work)
    sites = {name: site for name, _, site in binding}
    after = Configuration(before.input,
        CoreConfiguration(binding if mediated else ()),
        tuple(SatelliteConfiguration(s.name, s.role, sites.get(s.name))
              for s in structure.satellites), OutputModule(binding), participation)
    if not valid_response(structure, after, mediated):
        raise AssertionError('selector constructed an invalid collective response')
    return Result('complete', True, after, work)
