"""Pilot for retention remedies v1 (docs/retention-remedies-v1-contract.md, "Decisions"): fixes the free hyperparameters on seed 299, arm `mixed`
(never a scored seed), checks the derived-relations reading and times every branch. Run on the GPU:

    NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.retention_remedies_pilot

Selection rules (written before running):
  distillation  grid over head width {16, 32, 64} x steps {100, 300, 1000} x weight decay {0, 1e-3, 1e-2} (learning rate 0.01 fixed); score = agreement with the
                teacher's canonical arg-max on the held-out training renderings of M (those not in the K = 4 cache), for the nine-way head and, separately,
                the anchored head (nearest canonical code); ties go to fewer steps, then the narrower head. No evaluation renderings are used for selection.
  DER weight    {0.1, 1, 10}: the value with the highest recall on M after 2,000 updates (ties to the smaller weight). One seed and 18 pairs make this noisy; it is
                a choice of a working value, not an estimate.
  amendment     the first pilot picked grid corners (largest head, most steps; smallest DER weight), so both grids were widened (heads to 128, steps to 3000,
                DER weights down to 0.01) and the pilot rerun; the rules are unchanged. A selection at an edge of the widened grid is reported as such.
  fixed         buffer cadence (every 10th update, 16 items: the earlier rehearse-10 precedent), K = 4, random and other cue counts (= |M| K).
Outputs reports/retention-remedies-v1-pilot.json (deterministic content: selections, grids, feasibility) and -pilot-runtime.json (timing).
"""
import itertools
import json
import time

import numpy as np
import torch

from experiments import retention_remedies as RR

SEED, ARM = 299, 'mixed'
GRID = list(itertools.product((16, 32, 64, 128), (100, 300, 1000, 3000), (0., 1e-3, 1e-2)))  # widened after a first pilot picked grid corners (amendment)
DER_GRID = (0.01, 0.03, 0.1, 1., 10.)  # extended downward after the first pilot picked the smallest value (amendment)
OUT = RR.X.ROOT / 'reports' / 'retention-remedies-v1-pilot.json'
RUNTIME = RR.X.ROOT / 'reports' / 'retention-remedies-v1-pilot-runtime.json'


def validation(pre):
    """Held-out training renderings of M: codes and the teacher's canonical arg-max."""
    ref_enc, ref_model = pre['ref']
    ctx, M = pre['ctx'], pre['M']
    tr = RR.X.DATA[('pairsym', 'train')][0].reshape(-1, RR.R, 28, 28)
    imgs, truth = [], []
    for cp in M:
        keep = [j for j in range(RR.R) if j not in ctx['buf'][cp]]
        imgs.append(tr[ctx['row'][cp]][keep]); truth += [M[cp]] * len(keep)
    q = RR.codes(ref_enc, ref_model, np.concatenate(imgs))
    canon = RR.codes(ref_enc, ref_model, ctx['canon_imgs'])
    with torch.no_grad():
        teacher = RR.scores(ref_model, q, canon).argmax(1) + 1
    return q, canon, teacher, torch.tensor(truth)


def cue_cache(pre):
    ref_enc, ref_model = pre['ref']
    ctx, M = pre['ctx'], pre['M']
    tr = RR.X.DATA[('pairsym', 'train')][0].reshape(-1, RR.R, 28, 28)
    return RR.codes(ref_enc, ref_model, np.concatenate([tr[ctx['row'][cp]][list(ctx['buf'][cp])] for cp in M]))


def select_distillation(pre):
    ref_enc, ref_model = pre['ref']
    cues = cue_cache(pre)
    q, canon, teacher, truth = validation(pre)
    with torch.no_grad():
        soft = torch.softmax(RR.scores(ref_model, cues, canon), 1)
    out = {}
    for anchored in (False, True):
        rows = []
        for h, steps, wd in GRID:
            RR.HP[anchored] = (h, steps, wd)
            head, loss = RR.distill(cues, soft, anchored, canon, SEED, f'pilot-{anchored}')
            pred = RR.answer(head, q, canon, anchored)
            rows.append({'head': h, 'steps': steps, 'wd': wd, 'final_loss': loss, 'val_agreement': float((pred == teacher).float().mean()),
                         'val_accuracy': float((pred == truth).float().mean())})
        best = sorted(rows, key=lambda r: (-r['val_agreement'], r['steps'], r['head']))[0]
        out['anchored' if anchored else 'ninefold'] = {'grid': rows, 'selected': {k: best[k] for k in ('head', 'steps', 'wd')}, 'selected_row': best}
    out['teacher_validation_accuracy'] = float((teacher == truth).float().mean())
    return out


def derived(pre, sel):
    """Feasibility of the derived-relations reading: comparing which of two marked symbols has the larger count, using (a) the teacher's own codes of the symbols,
    (b) the nine-way destination's answers, (c) the anchored destination's predicted codes through the comparison head (head 2), on evaluation renderings."""
    ref_enc, ref_model = pre['ref']
    ctx, M = pre['ctx'], pre['M']
    canon = RR.codes(ref_enc, ref_model, ctx['canon_imgs'])
    ev = ctx['ev_M'][:, 0]
    q = RR.codes(ref_enc, ref_model, ev)
    counts = torch.tensor([M[cp] for cp in M])
    cues, (_, _, teacher, _) = cue_cache(pre), validation(pre)
    with torch.no_grad():
        soft = torch.softmax(RR.scores(ref_model, cues, canon), 1)
    out = {}
    pairs = [(i, j) for i in range(len(M)) for j in range(len(M)) if counts[i] != counts[j]]
    iu, ju = torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])
    gt = (counts[iu] > counts[ju]).float()

    def acc(logit):
        return float(((logit > 0).float() == gt).float().mean())
    with torch.no_grad():
        out['n_pairs'] = len(pairs)
        out['collections_sanity'] = float(np.mean([float((ref_model.relation_logits(canon[a:a + 1], canon[b:b + 1], 2)[0] > 0) == (a > b)) for a in range(9) for b in range(9) if a != b]))
        out['teacher_symbol_codes'] = acc(ref_model.relation_logits(q[iu], q[ju], 2))
        for tag, anchored in (('ninefold', False), ('anchored', True)):
            RR.HP[anchored] = tuple(sel[tag]['selected'][k] for k in ('head', 'steps', 'wd'))
            with torch.enable_grad():
                head, _ = RR.distill(cues, soft, anchored, canon, SEED, f'derived-{tag}')
            if anchored:
                pred = head(q)
                out[tag] = acc(ref_model.relation_logits(pred[iu], pred[ju], 2))
            else:
                a = RR.answer(head, q, canon, False)
                out[tag] = float(((a[iu] > a[ju]).float() == gt).float().mean())
        out['chance'] = 0.5
    return out


def main():
    t0 = time.perf_counter()
    RR.LM.install()
    RR.X.prepare()  # the datasets and renderings, as the runner's main does before it forks workers
    RR.P.prepare()
    RR.D.prepare()
    RR.torch.set_num_threads(RR.AP.FI.THREADS)
    RR.AP.DEV.install()
    pre = RR.prepare(ARM, SEED)
    t_prep = time.perf_counter() - t0
    sel = select_distillation(pre)
    t_dist = time.perf_counter() - t0 - t_prep
    der_rows = []
    for a in DER_GRID:
        RR.HOST['der-M']['alpha'] = a
        t1 = time.perf_counter()
        rec = RR.run_branch('der-M', RR.HOST['der-M'], pre['b4'], ARM, SEED, pre['ctx'], pre['withheld'])
        der_rows.append({'alpha': a, 'M_at_2000': rec['curve'][-1]['M'], 'U_at_2000': rec['curve'][-1]['U'], 'penalty_mean': rec.get('replay_penalty_mean'),
                         'B_at_end': rec['new_B'][-1]['B'], 'wall_seconds': time.perf_counter() - t1})
    best = sorted(der_rows, key=lambda r: (-r['M_at_2000'], r['alpha']))[0]
    RR.HOST['der-M']['alpha'] = best['alpha']
    timing = {r['alpha']: r['wall_seconds'] for r in der_rows}
    branches = {}
    for name, spec in RR.HOST.items():
        if ARM in spec.get('arms', (ARM,)) and name != 'der-M':
            t1 = time.perf_counter()
            rec = RR.run_branch(name, spec, pre['b4'], ARM, SEED, pre['ctx'], pre['withheld'])
            branches[name] = {'M': [c['M'] for c in rec['curve']], 'U': [c['U'] for c in rec['curve']], 'B_end': rec['new_B'][-1]['B'],
                              'path_unchanged': rec.get('path_unchanged'), 'wall_seconds': time.perf_counter() - t1}
    t1 = time.perf_counter()
    deriv = derived(pre, sel)
    for anchored, tag in ((False, 'ninefold'), (True, 'anchored')):
        RR.HP[anchored] = tuple(sel[tag]['selected'][k] for k in ('head', 'steps', 'wd'))
    static = RR.static_branches(*pre['ref'], pre['ctx'], SEED, pre['sets'])
    result = {'seed': SEED, 'arm': ARM, 'N': RR.N, 'L': RR.L, 'distillation': sel, 'der': {'grid': der_rows, 'selected_alpha': best['alpha']},
              'reference_recall_at_0': pre['curve4'][-1], 'teacher_M_original': RR.recall_MU(*pre['ref'], pre['ctx'], SEED), 'branches': branches, 'derived_relations': deriv, 'static': static}
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + '\n')
    RUNTIME.write_text(json.dumps({'wall_seconds_total': time.perf_counter() - t0, 'prepare_seconds': t_prep, 'distillation_grid_seconds': t_dist,
                                   'der_branch_seconds': timing, 'branch_seconds': {k: v['wall_seconds'] for k, v in branches.items()}, 'device': str(RR.AP.DEV.ACTIVE),
                                   'scope': 'nondeterministic timing; the pilot JSON holds the selections and feasibility only'}, indent=1) + '\n')
    print('selected', {k: sel[k]['selected'] for k in ('ninefold', 'anchored')}, 'der alpha', best['alpha'])
    print('DER sweep', [(r['alpha'], round(r['M_at_2000'], 3)) for r in der_rows])
    print('branches M@2000', {k: round(v['M'][-1], 3) for k, v in branches.items()}, 'ctx M@0', round(branches['ctx']['M'][0], 3))
    print('derived', deriv)


if __name__ == '__main__':
    main()
