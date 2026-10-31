"""Derived-relations reading of retention remedies v1 (contract: "Connecting to what is already stored"), computed in a separate pass because the main runner
did not record it. Arm `mixed`, seeds 350-359. Each task recomputes the deterministic end-of-block-4 state and the two cache destinations, checks that their
accuracies equal the main study's records exactly, reruns `ctx` to its end, and measures, on pairs of marked symbols with different counts (first evaluation
rendering of each, as in the pilot), the accuracy of "which count is larger":
  teacher        the comparison head (relation head 2) on the teacher's codes of the two symbols
  ctx_end        the same on the host's codes after 2,000 context updates
  ninefold       the nine-way destination's answers compared as numbers
  anchored       the comparison head on the anchored destination's predicted codes
  anchored_answers  the anchored destination's own nearest-canonical answers compared as numbers (what its individual recall predicts)
A connection benefit is read as `anchored` exceeding `anchored_answers` (the relation works better through the shared code than the answers alone allow).

Usage: NFI_DEVICE=cuda NSY_WORKERS=5 /data/venvs/sri-cuda/bin/python -m experiments.retention_remedies_derived
"""
import gzip
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from experiments import fan_in_pilot_analysis as AN
from experiments import retention_remedies as RR

SEEDS = list(range(350, 360))
ARM = 'mixed'
OUT = RR.X.ROOT / 'reports' / 'retention-remedies-v1-derived.json'
RUNTIME = RR.X.ROOT / 'reports' / 'retention-remedies-v1-derived-runtime.json'


def task(seed):
    torch.set_num_threads(RR.AP.FI.THREADS)
    RR.AP.DEV.install()
    t0 = time.perf_counter()
    pre = RR.prepare(ARM, seed)
    ref_enc, ref_model = pre['ref']
    ctx, M = pre['ctx'], pre['M']
    static = RR.static_branches(ref_enc, ref_model, ctx, seed, pre['sets'])
    canon = RR.codes(ref_enc, ref_model, ctx['canon_imgs'])
    tr = RR.X.DATA[('pairsym', 'train')][0].reshape(-1, RR.R, 28, 28)
    cues = RR.codes(ref_enc, ref_model, np.concatenate([tr[ctx['row'][cp]][list(ctx['buf'][cp])] for cp in M]))
    with torch.no_grad():
        soft = torch.softmax(RR.scores(ref_model, cues, canon), 1)
    heads = {tag: RR.distill(cues, soft, anchored, canon, seed, f'cache-{anchored}')[0] for tag, anchored in (('ninefold', False), ('anchored', True))}
    rec, end = RR.run_branch('ctx', RR.HOST['ctx'], pre['b4'], ARM, seed, ctx, pre['withheld'], keep_end=True)
    e_enc, e_model, _, _ = RR.AP.thaw(end, ARM, seed)
    ev = ctx['ev_M'][:, 0]
    counts = torch.tensor([M[cp] for cp in M])
    pairs = [(i, j) for i in range(len(M)) for j in range(len(M)) if counts[i] != counts[j]]
    iu, ju = torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])
    gt = (counts[iu] > counts[ju]).float()
    acc = lambda logit: float(((logit > 0).float() == gt).float().mean())
    num = lambda a: float(((a[iu] > a[ju]).float() == gt).float().mean())
    q = RR.codes(ref_enc, ref_model, ev)
    qe = RR.codes(e_enc, e_model, ev)
    with torch.no_grad():
        out = {'n_pairs': len(pairs),
               'teacher': acc(ref_model.relation_logits(q[iu], q[ju], 2)),
               'ctx_end': acc(e_model.relation_logits(qe[iu], qe[ju], 2)),
               'ninefold': num(RR.answer(heads['ninefold'], q, canon, False)),
               'anchored': acc(ref_model.relation_logits(heads['anchored'](q)[iu], heads['anchored'](q)[ju], 2)),
               'anchored_answers': num(RR.answer(heads['anchored'], q, canon, True))}
    main = json.load(gzip.open(RR.X.ROOT / 'reports' / 'retention-remedies-v1-results.json.gz', 'rt'))['results'][f'{ARM}/seed{seed}']
    out['matches_main_record'] = all(static[k]['acc_M'] == main['static'][k]['acc_M'] and static[k]['hash_after'] == main['static'][k]['hash_after']
                                     for k in ('consolidate-cache', 'consolidate-cache-anchor')) and rec['curve'][-1] == main['branches']['ctx']['curve'][-1]
    return seed, out, time.perf_counter() - t0


def main():
    t0 = time.perf_counter()
    RR.LM.install()
    RR.X.prepare(); RR.P.prepare(); RR.D.prepare()
    with mp.get_context('fork').Pool(int(os.environ.get('NSY_WORKERS', '5'))) as pool:
        res = {s: (o, w) for s, o, w in pool.imap_unordered(task, SEEDS)}
    per = {str(s): res[s][0] for s in SEEDS}
    keys = ('teacher', 'ctx_end', 'ninefold', 'anchored', 'anchored_answers')
    vec = lambda k: np.array([per[str(s)][k] for s in SEEDS])
    summary = {'means': {k: float(vec(k).mean()) for k in keys}, 'all_match_main_record': all(v['matches_main_record'] for v in per.values()),
               'contrasts': {f'anchored - {k}': {'mean_ci': AN.interval(vec('anchored') - vec(k))}
                             for k in ('anchored_answers', 'ninefold', 'teacher', 'ctx_end')}}
    OUT.write_text(json.dumps({'status': 'supplement: computed in a separate pass (the main runner omitted this reading); deterministic recomputation checked against the main records',
                               'arm': ARM, 'seeds': SEEDS, 'per_seed': per, 'summary': summary}, indent=1, sort_keys=True, default=float) + '\n')
    RUNTIME.write_text(json.dumps({'wall_seconds': time.perf_counter() - t0, 'task_seconds': {str(s): res[s][1] for s in SEEDS},
                                   'scope': 'nondeterministic timing'}, indent=1) + '\n')
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
