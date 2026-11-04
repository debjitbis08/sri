"""From update 10 of the form-candidate stop trajectory, stop form rows' new sequence-prediction gradients into the predictor P as well.
(docs/relearning-predictor-stop-v1.md; proposal in docs/relearning-margins-stop-v1.md)

From the captured pre-relearning state per seed, 20 relearning updates on identical batches and draws:
  unchanged            the original relearning
  form-cand-stop       ordinary stop-gradient of form rows' sequence gradient through the candidate collections, all 20 updates
  form-cand-stop+P     as form-cand-stop for updates 0-9 (identical to it); from update 10 form rows' predictions are also computed with P's parameters
                       detached, keeping their gradient into the codes they are made from. Adam's state at update 10 is shared with form-cand-stop and
                       kept, so earlier form-driven contributions in its first moment keep decaying into later steps: this stops new gradients only.
Context rows' sequence gradients and every other loss are unchanged.
Reads at 0, 10, 15, 20: experiments.relearning_branches.Reads (interpreter, memory present with actual retrieval, English matching) and
sequence_pilot.composition. For the two stop arms at 15 and 20, composition is also read with
  current codes + P as at update 10      (gains from representations adapting to the existing predictor)
  codes as at update 10 + current P      (gains from the predictor's change)
where codes are the image head, the taps encoder and the core projections.
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_predictor_stop
"""
import copy
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_mechanism as RM
from experiments import relearning_remedies as RR
sys.argv = _argv
import torch
import torch.nn.functional as F

RB, SC, S0, P, D, X, DEV, LM, SP, TP = RM.RB, RM.SC, RM.S0, RM.P, RM.D, RM.X, RM.DEV, RM.LM, RM.SP, RM.TP
NAME = 'relearning-predictor-stop-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
STEPS, BRANCH, READS = 20, 10, (0, 10, 15, 20)
ARMS = ('unchanged', 'form-cand-stop', 'form-cand-stop+P')


def sequence_loss_pstop(model, material, c, cand):
    """The sequence loss with form rows' candidates detached and form rows' predictions made with P's parameters detached."""
    seqs, n_form = material[3], material[4]
    rows, form = [], []
    for e, steps in enumerate(seqs):
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next))
                    form.append(e < n_form)
    prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
    cnt = torch.tensor([r[3] for r in rows]) - 1
    Cs = cand[None].expand(len(rows), -1, -1).clone()
    Cs[torch.arange(len(rows)), cnt] = c[tgt]
    fm = torch.tensor(form)
    Cs = torch.where(fm[:, None, None], Cs.detach(), Cs)
    x = torch.cat([c[prev], c[cur]], 1)
    pred = model.P(x)
    frozen = torch.func.functional_call(model.P, {k: v.detach() for k, v in model.P.named_parameters()}, (x,))
    pred = torch.where(fm[:, None], frozen, pred)
    return F.cross_entropy(TP.predict_logits(model, pred, Cs), cnt)


def code_modules(enc, model):
    return {'image': enc['image'], 'sequence': enc['sequence'], 'proj': model.proj}


def snapshot(mods):
    return {k: copy.deepcopy(m.state_dict()) for k, m in mods.items()}


def swapped(enc, model, codes=None, predictor=None):
    """Composition read with some parts temporarily set to their update-10 states."""
    mods = code_modules(enc, model)
    keep_codes = snapshot(mods) if codes else None
    keep_p = copy.deepcopy(model.P.state_dict()) if predictor else None
    try:
        if codes:
            for k, m in mods.items():
                m.load_state_dict(codes[k])
        if predictor:
            model.P.load_state_dict(predictor)
        enc['image'].eval(); enc['sequence'].eval()
        return SP.composition(enc, model)
    finally:
        if codes:
            for k, m in mods.items():
                m.load_state_dict(keep_codes[k])
        if predictor:
            model.P.load_state_dict(keep_p)
        enc['image'].train(); enc['sequence'].train()


def arm(name, box, reads):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    params = [p_ for g_ in opt.param_groups for p_ in g_['params']]
    withheld = box['withheld']
    track, losses, at10 = {}, [], None
    for t in range(STEPS + 1):
        if t == BRANCH:
            at10 = {'codes': snapshot(code_modules(enc, model)), 'P': copy.deepcopy(model.P.state_dict())}
        if t in READS:
            track[t] = RR.read_all(reads, enc, model)
            if name != 'unchanged' and t > BRANCH:
                track[t]['composition_current_codes_P_at_10'] = swapped(enc, model, predictor=at10['P'])
                track[t]['composition_codes_at_10_current_P'] = swapped(enc, model, codes=at10['codes'])
        if t == STEPS:
            break
        eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
        eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
        material = (*SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps'), len(eps))
        LM.CURRENT['step'] += 1
        L, c, cand, _ = RM.forward(enc, model, material, st['rng'], True, None if name == 'unchanged' else 'cand')
        if name == 'form-cand-stop+P' and t >= BRANCH:
            L['sequence'] = sequence_loss_pstop(model, material, c, cand)
        loss = L['match'] + L['next'] + L['compare'] + L['sequence']
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        losses.append({k: float(v.detach()) for k, v in L.items()})
    return track, losses


def task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    RB.RUNS = RUNS
    box = RB.capture(seed)
    out = {'delay_state_max_abs_difference_from_step8': RB.state_difference(
        box['weights'], torch.load(SC.RUNS / RB.ENC / 'checkpoints' / f'sparse48-seed{seed}-delay.pt', weights_only=False, map_location='cpu'))}
    out['capture_seconds'] = time.perf_counter() - t0
    reads = RB.Reads(seed, box['withheld'])
    for a in ARMS:
        b0 = time.perf_counter()
        track, losses = arm(a, box, reads)
        out[a] = {'track': track, 'losses': losses, 'seconds': time.perf_counter() - b0}
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    print('seed', seed, f'{rt["wall_seconds"]:.0f} s', flush=True)
    return seed, out, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RB_WORKERS', 3))) as pool:
        res = pool.map(task, RB.SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'diagnostic: form rows new gradients into the predictor stopped from update 10 on the form-candidate stop trajectory',
                                    'seeds': RB.SEEDS, 'arms': ARMS, 'steps': STEPS, 'branch': BRANCH, 'reads_at': READS, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning predictor stop', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
