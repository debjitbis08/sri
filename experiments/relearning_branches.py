"""Can the core keep a learned relation through new learning? Relearning branched from one exact pre-relearning state (docs/relearning-branches-v1.md).

Each seed of step 8's calibrated core is rerun exactly as limited-memory main up to the first relearning update. The run is stopped there, and its full
state captured: weights, Adam's state, every random stream, the rehearsal buffer, the first relearning batch and the step counter. From that state,
three 200-update branches each change one thing:
  as-run          relearning as in the original run; must reproduce the original relearning checkpoint
  no-forms        the delay's material continued instead: context episodes (taps with collections), no forms
  frozen-image    relearning as run, with the image projection (shared by collections and forms) not updated; collection codes can still move
                  through the core
Every 10 updates, on identical pairs and candidates:
  the `same` relation's symmetry, pair discrimination, ranking in both directions (raw and candidate-centred) and geometry (same_relation_diagnosis)
  collections -> taps on fresh and stored dot values in both comparator orders, and taps -> collections (collections_taps_diagnosis)
  English matching, for reacquisition of the relearned script
  memory and core together (stage 2's memory): symbol -> taps, taps -> symbol, dots and comparison, end to end and on the stored value
Explicit reverse-direction training is deferred (user, 2026-11-04).
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_branches
"""
import copy
import os
import pickle
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import same_relation_diagnosis as SRD
sys.argv = _argv
import numpy as np
import torch

CTD, SMR = SRD.CTD, SRD.CTD.SMR
SMC, SC, S0, P, D, X, nc, DEV, DUR = SMR.SMC, SMR.SC, SMR.S0, SMR.P, SMR.D, SMR.X, SMR.nc, SMR.DEV, SMR.DUR
LM, SP = SC.LM, SC.LM.SP
ENC = 'joined-mlp-calibrated'
SEEDS = tuple(int(s) for s in os.environ.get('RB_SEEDS', '300,301,302,303').split(','))
BRANCHES = ('as-run', 'no-forms', 'frozen-image')
EVERY = 10
NAME = 'relearning-branches-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME


class Captured(Exception):
    pass


def capture(seed):
    """Rerun limited-memory main for this seed and stop at the first relearning update, returning everything that update depends on."""
    LM.RUNS = RUNS / f'core-seed{seed}'
    X.make_model = lambda s: DEV.build(lambda: SC.make_model(s, 'sparse48', ENC))
    orig = LM.rehearsal_batch
    box = {}

    def stop(eps, buffer, reh_rng):
        f = sys._getframe(1).f_locals  # run_main's locals, at the first relearning batch
        box.update({'eps0': copy.deepcopy(eps), 'buffer': copy.deepcopy(buffer), 'reh_rng': copy.deepcopy(reh_rng),
                    **{k: copy.deepcopy(f[k]) for k in ('rng', 'srng', 'crng', 'drng', 'withheld')},
                    'enc': f['enc'], 'model': f['model'], 'opt_state': copy.deepcopy(f['opt'].state_dict()),
                    'weights': {'image': copy.deepcopy(f['enc']['image'].state_dict()), 'sequence': copy.deepcopy(f['enc']['sequence'].state_dict()),
                                'sri': copy.deepcopy(f['model'].state_dict())},
                    'current': dict(LM.CURRENT), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        raise Captured()
    LM.rehearsal_batch = stop
    try:
        LM.run_main(('sparse48', seed))
    except Captured:
        pass
    finally:
        LM.rehearsal_batch = orig
    return box


def restore(box):
    enc, model = box['enc'], box['model']
    enc['image'].load_state_dict(box['weights']['image']); enc['sequence'].load_state_dict(box['weights']['sequence']); model.load_state_dict(box['weights']['sri'])
    params = [*enc['image'].parameters(), *enc['sequence'].parameters(), *model.parameters()]  # LM.setup's order
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(params, P.LR)
    opt.load_state_dict(copy.deepcopy(box['opt_state']))
    LM.CURRENT.clear(); LM.CURRENT.update(box['current'])
    torch.set_rng_state(box['torch_rng'])
    if box['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(box['cuda_rng'])
    streams = {k: copy.deepcopy(box[k]) for k in ('rng', 'srng', 'crng', 'drng', 'reh_rng')}
    return enc, model, opt, streams, copy.deepcopy(box['buffer']), copy.deepcopy(box['eps0'])


class Reads:
    """Everything read at each evaluation point, from fixed inputs."""
    def __init__(self, seed, withheld):
        self.seed, self.withheld = seed, withheld
        prep = pickle.loads((SMR.RUNS / f'prep-seed{seed}.pkl').read_bytes())
        mem = SMR.LiveMemory(prep)
        mem.open_run(0); mem.write_upto('A', len(mem.order['A'])); mem.open_run(1); mem.write_upto('B', len(mem.order['B']))
        keys = mem.units()
        self.values = np.stack([mem.pair(k)[1] for k in keys]).astype(np.float32)
        self.vcounts = np.array([mem.describe(k)[1] for k in keys])
        self.ctd_draws = CTD.fixed_draws(seed, len(self.values))
        self.y, self.srd_draws = SRD.make_draws(seed)
        own = {}
        for key, ws in mem.log.items():
            for i in ws:
                f = prep['writes'][i]['fact']
                own.setdefault(f, {}).setdefault(key, 0)
                own[f][key] += 1
        own = {f: max(u, key=lambda k: (u[k], -k[0], -k[1])) for f, u in own.items()}
        self.sets = {}
        for name in ('A', 'B'):
            counts = np.repeat([prep['sets'][name][cp] for cp in prep['sets'][name]], SMR.QUERIES)
            facts = (0 if name == 'A' else SMR.N_SET) + np.repeat(np.arange(SMR.N_SET), SMR.QUERIES)
            r = mem.km.read(mem.split(prep['query'][name]['codes']))
            ok = r['score'] >= SMR.TAU
            held = [mem.log[(int(c), int(u))] for c, u in zip(r['centre'], r['unit'])]
            fact = np.array([int(np.bincount([prep['writes'][i]['fact'] for i in h]).argmax()) for h in held])
            byp = np.stack([mem.km.parts[own[f][0]].u[own[f][1]] / mem.km.parts[own[f][0]].n[own[f][1]] for f in facts]).astype(np.float32)
            self.sets[name] = {'counts': counts, 'value': r['value'].astype(np.float32), 'ok': ok, 'R0': ok & (fact == facts), 'bypass': byp,
                               'draws': SMC.draws(seed, name, len(counts), counts)}

    @torch.no_grad()
    def __call__(self, enc, model):
        enc['image'].eval(); enc['sequence'].eval()
        out = {'relation': SRD.measure(enc, model, self.y, self.srd_draws)}
        ctd, _ = CTD.read(enc, model, self.values, self.vcounts, self.ctd_draws)
        out['collections_taps'] = {k: ctd[k] for k in ('collections_to_taps', 'collections_to_taps_swapped_order', 'taps_to_collections')}
        host = SMC.Host.__new__(SMC.Host)
        host.enc, host.model, host.frozen, host.cache = enc, model, True, {}
        mp = {}
        for name, st in self.sets.items():
            e2e, _ = SMC.score(host, host.raw(st['value']), st['ok'], st['counts'], st['draws'])
            bp, _ = SMC.score(host, host.raw(st['bypass']), np.ones(len(st['ok']), bool), st['counts'], st['draws'])
            mp[name] = {'end_to_end': {k: float(v.mean()) for k, v in e2e.items()}, 'bypass': {k: float(v.mean()) for k, v in bp.items()}}
        out['memory_present'] = mp
        enc['image'].train(); enc['sequence'].train()
        m = LM.matching(enc, model, self.withheld)  # toggles modes itself
        out['english_matching'] = m['english/paired']['all']
        out['matching'] = {k: v['all'] for k, v in m.items()}
        return out


def branch(name, box, reads):
    enc, model, opt, st, buffer, eps0 = restore(box)
    if name == 'frozen-image':
        for p_ in enc['image'].parameters():
            p_.requires_grad_(False)  # no gradient: Adam leaves these weights and their state untouched
    track, losses = {}, []
    withheld = box['withheld']
    for t in range(LM.RELEARN + 1):
        if t % EVERY == 0:
            track[t] = reads(enc, model)
        if t == LM.RELEARN:
            break
        if name == 'no-forms':
            material = SP.batch_material([], D.context_episodes(st['drng'], D.N_CONTEXT), 'taps')
        else:
            eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
            material = SP.batch_material(LM.rehearsal_batch(eps, buffer, st['reh_rng']), D.context_episodes(st['crng'], D.N_CONTEXT), 'taps')
        losses.append(LM.train_update(enc, model, opt, material, st['rng'], True))
    final = {'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': model.state_dict()}
    return track, losses, final


def state_difference(a, b):
    worst = 0.
    for part in ('image', 'sequence', 'sri'):
        for k, v in a[part].items():
            if torch.is_tensor(v) and v.is_floating_point():
                worst = max(worst, float((v.detach().cpu() - b[part][k].detach().cpu()).abs().max()))
    return worst


def task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    box = capture(seed)
    t_cap = time.perf_counter() - t0
    delay_ck = torch.load(SC.RUNS / ENC / 'checkpoints' / f'sparse48-seed{seed}-delay.pt', weights_only=False, map_location='cpu')
    delay_check = state_difference(box['weights'], delay_ck)
    reads = Reads(seed, box['withheld'])
    out = {'delay_state_max_abs_difference_from_step8': delay_check}
    for name in BRANCHES:
        b0 = time.perf_counter()
        track, losses, final = branch(name, box, reads)
        out[name] = {'track': track, 'loss_every_update': losses, 'seconds': time.perf_counter() - b0}
        if name == 'as-run':
            ref = torch.load(SC.RUNS / ENC / 'checkpoints' / f'sparse48-seed{seed}-relearn.pt', weights_only=False, map_location='cpu')
            out[name]['relearn_state_max_abs_difference_from_step8'] = state_difference(final, ref)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'capture_seconds': t_cap,
          'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if DEV.ACTIVE else None}
    print('seed', seed, 'delay check', delay_check, 'relearn check', out['as-run']['relearn_state_max_abs_difference_from_step8'], f'{rt["wall_seconds"]:.0f} s', flush=True)
    return seed, out, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RB_WORKERS', 3))) as pool:
        res = pool.map(task, SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'diagnostic continuations from one captured pre-relearning state per seed; no reverse-direction training',
                                    'encoder': ENC, 'seeds': SEEDS, 'branches': BRANCHES, 'every': EVERY, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning branches', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
