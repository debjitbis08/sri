"""Fixed worked examples only: python -m experiments.mediation."""
from dataclasses import asdict
import json
from pathlib import Path
import platform

from src.mediation import InputModule, Satellite, Structure, initial_configuration, realize


def example(sites=('a', 'b'), third_sites=None):
    return Structure((Satellite('A', 'local-role', sites, sites[0]),
                      Satellite('B', 'local-role', sites, sites[0]),
                      Satellite('S', 'local-role', third_sites or sites, sites[0])))


def certificate():
    pair = example()
    context = example(('a', 'b', 'c'), ('a',))
    cases = []
    for name, structure, participation, mediated, budget in (
        ('direct_pair_collision', pair, (), False, None),
        ('mediated_pair', pair, (), True, None),
        ('retained_capacity_conflict', pair, ('S',), True, None),
        ('context_inactive', context, (), True, None),
        ('context_active', context, ('S',), True, None),
        ('budget_unknown', pair, (), True, 0),
    ):
        before = initial_configuration(structure, InputModule(('A', 'B')), participation)
        intervention = None
        if name == 'context_active':
            before = realize(structure, initial_configuration(
                structure, InputModule(('A', 'B')))).configuration
            intervention = participation
        result = realize(structure, before, mediated, budget, context=intervention)
        cases.append(dict(name=name, structure=asdict(structure), before=asdict(before),
                          mediated=mediated, context_intervention=intervention,
                          max_extensions=budget, result=asdict(result)))
    return dict(specification='Collective role-binding illustration v0.1',
                status='complete', runtime=dict(implementation=platform.python_implementation(),
                                               python=platform.python_version()),
                scope='Six fixed examples; no OP-04 search, learning, or recovery claim',
                selection='First injective assignment in sorted participant/site order',
                work_units='Candidate site extensions; not total CPU instructions', cases=cases)


def render(report):
    return json.dumps(report, indent=2, sort_keys=True) + '\n'


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'reports' / 'mediation-example.json'
    path.write_text(render(certificate()))
    print(path)
