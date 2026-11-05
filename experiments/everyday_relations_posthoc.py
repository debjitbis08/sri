"""Everyday relations v1, post-hoc diagnostics (not in the contract): T3a (penguin's trained exception and other birds' flying, robin
included as a held-out T1 animal), T4 accuracy by size-rank distance, and whether the
T5/T6 failures are thresholds (held-out true facts ranked above false ones but below 0) or ranking. Reads checkpoints."""
import json
import time

import numpy as np
import torch

from experiments import everyday_relations as R
from src import everyday_world as W


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    world = W.build()
    out = {}
    for arm in R.ARMS:
        recs = []
        for seed in R.SEEDS:
            m = R.Net(arm, len(world.entities), len(world.relations), torch.Generator().manual_seed(0))
            m.load_state_dict(torch.load(R.RUNS / f'{arm}-real-seed{seed}.pt'))
            s = R.scores(m, world)
            rec = {}
            ri = W.RELATIONS.index('bigger-than')
            mt = world.test == 'T4'
            dist = np.array([abs(W.SIZE.index(world.entities[h]) - W.SIZE.index(world.entities[t]))
                             for h, t in zip(world.h[mt], world.t[mt])])
            rec['T4_by_distance'] = {int(d): float(((s[mt][dist == d] > 0) == (world.y[mt][dist == d] > 0)).mean())
                                     for d in np.unique(dist)}
            rec['T4_frac_true'] = float((s[mt] > 0).mean())
            tr = (world.r == ri) & (world.test == '')
            ci = W.RELATIONS.index('can')
            fly = (world.r == ci) & (world.t == world.index('fly'))
            birds = [a for a in W.ANIMALS if W.ANIMALS[a][0] == 'bird' and a != 'penguin']
            rec['T3a_penguin_fly_logit'] = float(s[fly & (world.h == world.index('penguin'))][0])
            rec['T3a_other_birds_fly_acc'] = float(np.mean([(s[fly & (world.h == world.index(a))][0] > 0) == W.ANIMALS[a][2].split().count('fly')
                                                            for a in birds]))
            rec['bigger_trained_acc'] = float(((s[tr] > 0) == (world.y[tr] > 0)).mean())
            for tst in ('T5', 'T6', 'T7'):
                mm = world.test == tst
                pos = mm & (world.y == 1)
                rel = np.isin(world.r, np.unique(world.r[mm]))
                rec[tst] = {'heldout_true_mean_logit': float(s[pos].mean()),
                            'trained_true_mean_logit': float(s[rel & (world.test == '') & (world.y == 1)].mean()),
                            'trained_false_mean_logit': float(s[rel & (world.test == '') & (world.y == 0)].mean()),
                            'heldout_false_mean_logit': float(s[mm & (world.y == 0)].mean()) if (mm & (world.y == 0)).any() else None}
            recs.append(rec)
        out[arm] = recs
    rt = {'scope': 'whole-script timing', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    R.save_json(R.ROOT / 'reports' / 'everyday-relations-v1-posthoc.json', out)
    R.save_json(R.ROOT / 'reports' / 'everyday-relations-v1-posthoc-runtime.json', rt)
    for arm, recs in out.items():
        d = {k: round(np.mean([r['T4_by_distance'][k] for r in recs]), 2) for k in recs[0]['T4_by_distance']}
        print(arm, 'T4 by dist', d, 'frac judged true', round(np.mean([r['T4_frac_true'] for r in recs]), 2),
              'trained', round(np.mean([r['bigger_trained_acc'] for r in recs]), 2))
        print('    T3a penguin fly logit', [round(r['T3a_penguin_fly_logit'], 1) for r in recs], 'other birds fly acc',
              round(np.mean([r['T3a_other_birds_fly_acc'] for r in recs]), 2))
        for tst in ('T5', 'T6', 'T7'):
            print('   ', tst, {k: round(np.mean([r[tst][k] for r in recs]), 1) for k in recs[0][tst] if recs[0][tst][k] is not None})
    print(rt)


if __name__ == '__main__':
    main()
