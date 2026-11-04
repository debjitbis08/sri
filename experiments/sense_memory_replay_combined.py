"""Stage 2's checkpoints read as the combined system: memory and core together, at block 3, the delay and relearning, in every arm
(docs/sense-memory-replay-v1-pilot.md, section "Memory and core together").

Memory holds both sets (the same in every arm: it never depends on the core). For each set, through each arm's core:
  end to end    retrieval at tau, the recalled raw value through the core
  bypass        the fact's own stored value (its unit holding most of its writes), no address selection
  core-only     memory disabled: the symbol rendering through the core (a diagnostic)
Cross-sense directions are kept apart: symbol -> taps (the recalled value against nine taps) and taps -> symbol (a taps sequence against nine
symbols' recalled values); for the core's own knowledge, collections -> taps and taps -> collections.
Usage: SMR_WORKERS=4 /data/venvs/sri-cuda/bin/python -m experiments.sense_memory_replay_combined pilot
"""
import pickle
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import sense_memory_replay as SMR
sys.argv = _argv
import numpy as np
import torch

SMC, SC, S0, P, D, nc, DUR = SMR.SMC, SMR.SC, SMR.S0, SMR.P, SMR.D, SMR.nc, SMR.DUR
POINTS = ('block3', 'delay', 'relearn')
REPORT = SMR.REPORT


def core_own(host, seed):
    """The core's own cross-sense reading, both directions, no memory: a fresh collection against nine taps, and a taps sequence against nine collections."""
    rng = nc.stream('smrc_own', seed)
    E, T = P.EVAL_PER, len(D.TAPS[('eval', 1)])
    counts = np.repeat(np.arange(1, 10), 30)
    n = len(counts)
    order = np.stack([rng.permutation(9) for _ in range(n)])
    qi, cj, tj = rng.integers(E, size=n), rng.integers(E, size=(n, 9)), rng.integers(T, size=(n, 9))
    qt = rng.integers(T, size=n)
    with torch.no_grad():
        Qc = torch.stack([host.dots(int(c))[i] for c, i in zip(counts, qi)])
        Qt = torch.stack([host.taps(int(c))[i] for c, i in zip(counts, qt)])
        tc = torch.stack([torch.stack([host.taps(int(k) + 1)[tj[i, k]] for k in order[i]]) for i in range(n)])
        cc = torch.stack([torch.stack([host.dots(int(k) + 1)[cj[i, k]] for k in order[i]]) for i in range(n)])
        c2t = order[np.arange(n), D.pick(host.model, Qc, tc)] + 1 == counts
        t2c = order[np.arange(n), D.pick(host.model, Qt, cc)] + 1 == counts
    return {'collections_to_taps': float(c2t.mean()), 'taps_to_collections': float(t2c.mean())}


def task(t):
    arm, seed = t
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(SMC.FI.THREADS)
    prep = pickle.loads((SMR.RUNS / f'prep-seed{seed}.pkl').read_bytes())
    mem = SMR.LiveMemory(prep)
    mem.open_run(0)
    mem.write_upto('A', len(mem.order['A']))
    mem.open_run(1)
    mem.write_upto('B', len(mem.order['B']))
    for g in mem.km.gates:  # gates were fitted, and pickled, on the GPU; these reads run on the CPU
        g.layer = g.layer.cpu()
    own = {}  # fact -> its own unit: the one holding most of its writes
    for key, ws in mem.log.items():
        for i in ws:
            f = prep['writes'][i]['fact']
            own.setdefault(f, {}).setdefault(key, 0)
            own[f][key] += 1
    own = {f: max(u, key=lambda k: (u[k], -k[0], -k[1])) for f, u in own.items()}

    enc, model = SC.make_model(seed, 'sparse48', SR_ENCODER)
    for k in ('A', 'B', 'unseen'):
        x = prep['query'][k]['images'].reshape(-1, 28, 28)
        enc['image'].add_codes([SC.image_key(np.ascontiguousarray(r, dtype=np.float32)) for r in x], prep['query'][k]['codes'])
    out = {}
    for point in POINTS:
        ck = torch.load(SMR.RUNS / arm / SR_ENCODER / 'checkpoints' / f'sparse48-seed{seed}-{point}.pt', weights_only=False, map_location='cpu')
        enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); model.load_state_dict(ck['sri'])
        for m in (enc['image'], enc['sequence'], model):
            m.eval()
        host = SMC.Host.__new__(SMC.Host)
        host.enc, host.model, host.frozen, host.cache = enc, model, True, {}
        res = {'core_own': core_own(host, seed)}
        for name in ('A', 'B'):
            counts = np.repeat([prep['sets'][name][cp] for cp in prep['sets'][name]], SMR.QUERIES)
            facts = (0 if name == 'A' else SMR.N_SET) + np.repeat(np.arange(SMR.N_SET), SMR.QUERIES)
            dr = SMC.draws(seed, name, len(counts), counts)
            r = mem.km.read(mem.split(prep['query'][name]['codes']))
            ok = r['score'] >= SMR.TAU
            held = [mem.log[(int(c), int(u))] for c, u in zip(r['centre'], r['unit'])]
            fact = np.array([int(np.bincount([prep['writes'][i]['fact'] for i in h]).argmax()) for h in held])
            R0 = ok & (fact == facts)
            ones = np.ones(len(ok), bool)
            byp = np.stack([prep_value(mem, own[f]) for f in facts])
            with torch.no_grad():
                e2e, _ = SMC.score(host, host.raw(r['value']), ok, counts, dr)
                bp, _ = SMC.score(host, host.raw(byp), ones, counts, dr)
                co, _ = SMC.score(host, host.images(prep['query'][name]['images'].reshape(-1, 28, 28)), ones, counts, dr)
            given = {'r2': R0, 'r3': R0, 'r4': R0 & R0[dr['partner']]}
            res[name] = {'R0_retrieval': float(R0.mean()), 'answered': float(ok.mean()),
                         'end_to_end': {k: float(v.mean()) for k, v in e2e.items()},
                         'bypass': {k: float(v.mean()) for k, v in bp.items()},
                         'core_only': {k: float(v.mean()) for k, v in co.items()},
                         'retrieved': {k: {'share': float(g.mean()), 'end_to_end': float(e2e[k][g].mean()) if g.any() else None,
                                           'bypass': float(bp[k][g].mean()) if g.any() else None} for k, g in given.items()}}
        out[point] = res
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    return f'{arm}/seed{seed}', out, rt


def prep_value(mem, key):
    j, u = key
    m = mem.km.parts[j]
    return m.u[u] / m.n[u]


SR_ENCODER = SMR.ENCODER


def main():
    t0 = time.perf_counter()
    S0.prepare()
    SC.LM.install()
    tasks = [(a, s) for a in SMR.ARMS for s in SMR.SEEDS]
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(__import__('os').environ.get('SMR_WORKERS', 4))) as pool:
        res = pool.map(task, tasks, chunksize=1)
    S0.save_json(f'{REPORT}-combined.json', {'status': 'stage 2 checkpoints read as the combined system (memory present), with bypass and memory-disabled diagnostics',
                                             'arms': SMR.ARMS, 'seeds': SMR.SEEDS, 'points': POINTS, 'tau': SMR.TAU, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-combined-runtime.json', S0.runtime({'phase': 'combined reads', 'wall_seconds': time.perf_counter() - t0,
                                                                'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
