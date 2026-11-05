"""Everyday relations: does a new code fitted from 'is a bird' alone come out 'not a fish'? Post-hoc on v1 checkpoints
(user question 2026-11-05, not in a contract). Fits: v1 (true facts only, weight decay) and prior-only (true facts, MAP
under the Gaussian prior of the trained codes, no false facts). Reads is-a against every class, and the class properties."""
import time

import numpy as np
import torch

from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from src import everyday_world as W1


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    world = W1.build()
    specs = W1.fast_mapping_specs()
    out = {}
    for arm in R1.ARMS:
        for seed in R1.SEEDS:
            model = R1.Net(arm, len(world.entities), len(world.relations), torch.Generator().manual_seed(0))
            model.load_state_dict(torch.load(R1.RUNS / f'{arm}-real-seed{seed}.pt'))
            prior = V2.gaussian_prior(model.E)
            for name, cls in (('dax', 'bird'), ('fep', 'fish')):
                given, read, ks = specs[name]
                isa = [('is-a', name, c, c == cls) for c in W1.CLASSES]
                props = [x for x in read if x[0] != 'is-a']
                for fit, pr in (('v1', None), ('prior-only', prior)):
                    for k in (1, 4):
                        rec = V2.read_fit(model, world, name, given, isa + props, [], k, seed, pr, cls)
                        out[f'{arm}/{seed}/{name}/{fit}/k{k}'] = {
                            'other_classes_rejected': rec['by_relation']['is-a']['tnr'],
                            'own_class_accepted': rec['by_relation']['is-a']['tpr'],
                            'read_bal_acc_all': rec['read']['bal_acc']}
                        out[f'{arm}/{seed}/{name}/{fit}/k{k}']['by_relation'] = rec['by_relation']
    rt = {'scope': 'whole-script timing', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    R1.save_json(R1.ROOT / 'reports' / 'everyday-relations-exclusivity.json', out)
    R1.save_json(R1.ROOT / 'reports' / 'everyday-relations-exclusivity-runtime.json', rt)
    print('arm      fit         k  other classes rejected (dax, fep)   own class   properties (has/can/lives-in/eats)')
    for arm in R1.ARMS:
        for fit in ('v1', 'prior-only'):
            for k in (1, 4):
                g = lambda n, key: np.mean([out[f'{arm}/{s}/{n}/{fit}/k{k}'][key] for s in R1.SEEDS])
                pr = lambda n: np.mean([np.mean([v['bal_acc'] if v['bal_acc'] is not None else v['acc']
                                                 for rel, v in out[f'{arm}/{s}/{n}/{fit}/k{k}']['by_relation'].items()
                                                 if rel != 'is-a']) for s in R1.SEEDS])
                print(f'{arm:8s} {fit:10s} {k}   {g("dax", "other_classes_rejected"):.2f}, {g("fep", "other_classes_rejected"):.2f}'
                      f'                         {g("dax", "own_class_accepted"):.2f}, {g("fep", "own_class_accepted"):.2f}'
                      f'   {pr("dax"):.2f}, {pr("fep"):.2f}')
    print(rt)




if __name__ == '__main__':
    main()
