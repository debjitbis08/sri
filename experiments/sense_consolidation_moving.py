"""Everything moving at once (docs/sense-consolidation-v1-pilot.md, Everything moving): keys from the unlabelled encoder as it keeps learning, rebuilt and consolidated; values kept as sensory traces and read through the host as it is then, while the host also keeps learning; A then B and B then A.
Each kept experience is a cue image and, for that cue, the first value image paired with it. A unit's meaning is the mean host code of its experiences' value images at the time. Stored value images are by default ones no host trained on. The distinction test compares those current codes. True symbols and counts only score merges and reads.

Stages (argv[2]):
  hosts     GPU: sparse48 through blocks 4-6 teaching B then A, keeping weights; A then B hosts come from the sensory trace pilot
  learning  GPU: memory on the learning encoder, read at the end of blocks 4, 5, 6 through that block's host

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.sense_consolidation_moving pilot hosts
"""
import sys
import time

import numpy as np
import torch

from experiments import arbitrary_pairings as AP
from experiments import durable as DUR
from experiments import limited_memory as LM
from experiments import sense_consolidation as SC
from experiments import sense_consolidation_evidence as EV
from experiments import sense_consolidation_traces as T
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_learning_senses as LS
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import shared_core_traces as TR
from experiments import shared_core_unseen_traces as UT
from experiments import translation_stable_keys as TK

nc, P, D, SP = S0.nc, S0.P, S0.D, S0.SP
NAME = 'sense-consolidation-v1-moving'
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
N, E, R = S1.N, S0.EVAL_RENDERS, S0.TRAIN_RENDERS
BLOCK, BATCH = LS.BLOCK, LS.BATCH
BYTES_PER_EXPERIENCE = 2 * SC.TRACE_BYTES  # a cue image and a value image


def ck_path(seed, b, order):
    return TR.ck_path(seed, b) if order == 'AB' else RUNS / 'checkpoints' / f'seed{seed}-BA-block{b}.pt'


def fit_ba(task):
    """As the sensory trace pilot's host, with the two sets exchanged. Each set keeps its own pair stream, so the pairs match memory's in this order."""
    seed, = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    withheld = D.withheld_counts(seed)
    sets, row, _ = AP.build_sets(seed)
    enc, model, opt = LM.setup('sparse48', seed)
    LM.CURRENT['end'] = None
    rng = nc.stream('partA_train', seed, 'A-order')
    for lang in P.LANGS:
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    hashes = {'3': S0.state_hash(enc, model)}
    for b, tag in ((4, 'B'), (5, 'A'), (6, None)):
        crng = nc.stream('ap_context', seed, b)
        prng = nc.stream('ap_pairs', seed, {'A': 4, 'B': 5}[tag], tag) if tag else None
        for _ in range(AP.L):
            items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
            if tag:
                AP.pair_items(items, match, sets[tag], row, prng)
            LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True)
        hashes[str(b)] = S0.state_hash(enc, model)
        LM.save_atomic(LM.weights(enc, model, withheld, 'sparse48', seed), ck_path(seed, b, 'BA'))
    return f'seed{seed}', {'hashes': hashes, 'block3_matches_ab_host': hashes['3'] == __import__('json').load(open(S0.X.ROOT / 'reports' / f'{TR.NAME}-hosts.json'))[f'seed{seed}']['hashes']['3'], 'updates': 3 * D.STEPS + 3 * AP.L}, S0.timing(t0, c0)


def hosts():
    t0 = time.perf_counter()
    S0.prepare()
    AP.POOL.update(S0.PS.survivors())
    res = S0.pool_map(DUR.durable(fit_ba, RUNS / 'done-hosts', 'seed{0}'), [(s,) for s in S0.SEEDS])
    S0.save_json(f'{REPORT}-hosts.json', {k: r for k, r, _ in res})
    S0.save_json(f'{REPORT}-hosts-runtime.json', S0.runtime({'phase': 'B then A host runs, 12,000 updates each', 'wall_seconds': time.perf_counter() - t0, 'fits': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))
    print({k: r['block3_matches_ab_host'] for k, r, _ in res}, flush=True)


def load_code(seed, b, order):
    ck = torch.load(ck_path(seed, b, order), weights_only=False, map_location='cpu')
    enc, model, _ = LM.setup('sparse48', seed)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    for m in (*enc.values(), model):
        m.requires_grad_(False)
    return lambda x: model.proj['image'](enc['image'](x))


def unit_values(tu, tr, h, first_coll, counts, table):
    """Each unit's value: the mean code under table of its kept experiences' value images. table (collections, 9, d); tr: cue images; first_coll: the value image's collection, per cue image. Pure, so testable."""
    tu, tr = np.asarray(tu, np.int64), np.asarray(tr, np.int64)
    rows = table[first_coll[tr] % len(table), counts[tr // R] - 1]
    total = np.zeros((h, table.shape[-1]), np.float32)
    np.add.at(total, tu, rows)
    return total / np.maximum(np.bincount(tu, minlength=h), 1)[:, None]


def arms():
    gate = dict(gate=EV.GATE, radius=0., traces='spread', keep=8)
    cover = dict(evidence='cover', phi=.25, merge=.6, veto=True, **gate)
    out = {'frozen-keys': dict(H=2048, frozen=True, every=None, rebuilds=False, keep=8), 'neither': dict(H=2048, every=None, rebuilds=False, keep=8), 'rebuild': dict(H=2048, keep=8), 'rebuild-H512': dict(H=512, keep=8),
           'cover-0.25-distinct': dict(H=2048, **cover), 'cover-0.25-distinct-H512': dict(H=512, **cover),
           'address-0.8-distinct': dict(H=2048, merge=.8, veto=True, **gate), 'address-0.7-distinct': dict(H=2048, merge=.7, veto=True, **gate), 'address-0.7-distinct-H512': dict(H=512, merge=.7, veto=True, **gate)}
    return out


def learn(task):
    seed, lr, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    LS.TAU = SC.TAU
    from experiments import shared_core_value as V
    c = S1.load(seed)
    model, start = SM.encoder('unlabelled', seed), SM.encoder('unlabelled', seed).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr)
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    rng = nc.stream('sm_learning', seed, str(lr))
    first, second = SC.ORDERS[order]
    streams = {t: S1.stream(t, seed, c) for t in 'AB'}
    active = {k: T.Arm(**v) for k, v in arms().items()}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        start_cues, start_queries = np_(enc(start, cue)), np_(enc(start, query))
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    unseen_images = UT.source_images(seed)['new-radius']
    cand_images = {t: V.candidates(N, E, seed, t) for t in 'AB'}
    first_coll = np.full(len(cue), -1, np.int64)
    counts = c['counts']
    reads, gates, zero = {}, {}, np.zeros(S1.CORE, np.float32)

    def host_view(b):
        code = load_code(seed, b, order)
        fn = lambda x: S0.batched(code, x).cpu().numpy().astype(np.float32)
        with torch.no_grad():
            trained = TR.table(code)
            unseen = np.stack([fn(unseen_images[:, k]) for k in range(9)], 1)
            view = dict(c)
            for t in 'AB':
                view[f'cand_{t}'] = fn(cand_images[t][1]).reshape(N * E, 9, -1)
        view['protos'] = trained.mean(0)
        return trained, unseen, view

    def ceiling(view, tag):
        off = 0 if tag == 'A' else N
        truth = counts[np.repeat(np.arange(off, off + N), E)]
        return S1.decode(view['protos'][truth - 1], np.ones(len(truth), bool), truth, view[f'cand_{tag}'], view[f'order_{tag}'], view['protos'])

    def read_point(b, tags, trained, unseen, view):
        model.eval()
        with torch.no_grad():
            q = np_(enc(model, query))
        reads[str(b)] = {'ceiling': {t: ceiling(view, t) for t in tags}}
        for name, arm in active.items():
            mem = arm.mem
            Zq = S1.unit(start_queries if arm.frozen else q)
            h = mem.h
            out = {'units': h, 'forced': mem.forced, 'experiences': len(arm.tu), 'bytes': 4 * h * (32 + 2) + len(arm.tu) * BYTES_PER_EXPERIENCE, 'unknown_coverage': float(mem.read(Zq[2 * N * E:3 * N * E])[1].mean())}
            for label, table, v in (('unseen', unseen, view), ('training', trained, view), ('stale-copy', c['coll_train'], view), ('frozen-host', c['coll_train'], c)):
                mem.u[:h] = unit_values(arm.tu, arm.tr, h, first_coll, counts, table) * mem.n[:h, None]
                out[label] = S1.read_sets(mem, Zq, v, tags)
            reads[str(b)][name] = out

    step = 0
    for b, tag in ((4, first), (5, second), (6, None)):
        s = streams[tag] if tag else None
        for u in range(BLOCK):
            base = torch.from_numpy(np.ascontiguousarray(bg[rng.integers(n_cls, size=TK.CLASSES), rng.integers(n_r, size=TK.CLASSES)], dtype=np.float32))
            x = torch.stack([SRL.augment_with(base, rng, SRL.AUGMENTS['full']), SRL.augment_with(base, rng, SRL.AUGMENTS['full'])], 1).reshape(-1, 28, 28)
            model.train()
            z = model(x)
            loss = SRL.info_nce(z @ z.T / SRL.T, partner)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            model.eval()
            if s is not None:
                part = s[u * BATCH:(u + 1) * BATCH]
                flat = part[:, 0] * R + part[:, 1]
                with torch.no_grad():
                    live = np_(model(torch.from_numpy(cue[flat])))
                for i in range(len(part)):
                    if first_coll[flat[i]] < 0:
                        first_coll[flat[i]] = part[i, 2]
                    for arm in active.values():
                        arm.write(start_cues[flat[i]] if arm.frozen else live[i], zero, flat[i])
        trained, unseen, view = host_view(b)
        with torch.no_grad():
            table = np_(enc(model, cue))
        for arm in active.values():
            if arm.every:
                h = arm.mem.h
                arm.mem.u[:h] = unit_values(arm.tu, arm.tr, h, first_coll, counts, unseen) * arm.mem.n[:h, None]  # what the distinction test compares
                arm.tick(start_cues if arm.frozen else table, counts)
        read_point(b, first if b == 4 else 'AB', trained, unseen, view)
    res = {'reads': reads, 'events': {k: a.events for k, a in active.items()}, 'seed': seed, 'lr': lr, 'order': order}
    rt = S0.timing(t0, c0)
    rt['tick_seconds'] = {k: a.seconds for k, a in active.items()}
    return f'{seed}/{lr}/{order}', res, rt


def learning():
    t0 = time.perf_counter()
    S0.prepare()
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done', '{0}-{1}-{2}'), [(s, lr, o) for s in S0.SEEDS for lr in SC.LRS for o in SC.ORDERS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, both orders; keys, values and host all learning; keys and host learn separately; values are stored images read through the current host; true symbols and counts only score merges', 'arms': arms(), 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder training with memory writes, reads through hosts', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'hosts': hosts, 'learning': learning}[sys.argv[2]]()
