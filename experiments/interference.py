"""Exhaustive OP-02 certificate: python -m experiments.interference."""
import json
from itertools import product
from pathlib import Path
from src.interference import commutative_identity_tables, response


def certificate():
    rows = []
    first = None
    for size in range(1, 4):
        tables = monoids = cases = witnesses = 0
        by_edges = [0, 0, 0]
        for algebra in commutative_identity_tables(size):
            tables += 1
            if not algebra.associative():
                continue
            monoids += 1
            for edges in product((False, True), repeat=2):
                for left, right in product(range(size), repeat=2):
                    cases += 1
                    joint, independent = response(algebra, left, right, edges)
                    if joint != independent:
                        witnesses += 1
                        by_edges[sum(edges)] += 1
                        if first is None:
                            first = dict(states=size, nodes=3, edges=list(edges),
                                         table=algebra.table, inputs=[left, right],
                                         joint=sorted(joint), independent=sorted(independent))
        rows.append(dict(states=size, tables=tables, monoids=monoids,
                         cases=cases, witnesses=witnesses, witnesses_by_edge_count=by_edges))
    return dict(specification="OP-02 feed-forward commutative monoids v0.1",
                status="complete", bounds=dict(states=[1, 2, 3], nodes=3,
                                                edge_masks=4),
                enumeration=rows, smallest_witness=first)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'reports' / 'interference.json'
    out.write_text(json.dumps(certificate(), indent=2, sort_keys=True) + '\n')
    print(out.read_text())
