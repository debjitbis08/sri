"""Step 3 of the sense-level relational pilot (docs/sense-relational-v1-pilot.md): local memory addressed by the unlabelled contrastive encoder, with A read after B. Values and reading stay on block-3 host codes, so differences come from addresses.

Stages (argv[2]):
  codes     GPU: both encoders' codes of the pair and font material
  grid      CPU: H x tau on frozen keys in both orders, selected by the contract's rule per encoder; reads on a second font at each selected setting
  learning  GPU: at the selected setting and at H 2048, the encoder keeps learning from augmentations while memory is taught, with and without rebuilding addresses from cue traces

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.sense_memory pilot codes
"""
import io
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import sense_relational as SRL
from experiments import shared_core_learning_senses as LS
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import streaming_retrieval as SR
from experiments import translation_stable_keys as TK
from src.sparse_local_memory import Memory

nc, X, DEV = S0.nc, S0.X, S0.DEV
NAME = 'sense-memory-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
ENCODERS = ('unlabelled', 'rendering-pairs')
S1.HS, S1.KS, S1.TAUS = (128, 256, 512, 1024, 2048), (1,), (.5, .6, .7, .75, .8, .85)  # 0.75, 0.8 and H above 512 were added after none was eligible on the first grid
WORKERS = int(os.environ.get('NSY_WORKERS', 6))
N, E, R = S1.N, S0.EVAL_RENDERS, S0.TRAIN_RENDERS
BLOCK, BATCH = LS.BLOCK, LS.BATCH
LEARN_ARMS = {'frozen': (0, None), 'no-rebuild': (0, None), 'rebuild-8-every-2000': (8, 2000), 'rebuild-all-every-250': (None, 250)}
LRS = (1e-3, 1e-2)


def encoder(name, seed):
    device = DEV.NAME if DEV.ACTIVE else 'cpu'
    if name == 'rendering-pairs':
        return SR.key_encoder(seed).to(device)
    m = TK.build('base', 32, 30_000 + seed)
    m.load_state_dict(torch.load(SRL.RUNS / 'encoders' / f'full-seed{seed}.pt', map_location='cpu'))
    return m.to(device)


def codes_path(name, seed):
    return RUNS / 'codes' / f'{name}-seed{seed}.npz'


def make_codes():
    S0.set_up_task('sparse48')
    for seed in S0.SEEDS:
        train, ev, _ = S0.test_material(seed)
        ftrain, fsame, fcross = SRL.font_cache(seed)
        for name in ENCODERS:
            m = encoder(name, seed).eval().requires_grad_(False)
            fn = lambda x: S0.batched(lambda b: m(b), x.reshape(-1, 28, 28)).cpu().numpy().astype(np.float32)
            with torch.no_grad():
                out = {'key_train': fn(train[:3 * N]), 'key_eval': fn(ev[:3 * N]), 'f_train': fn(ftrain).reshape(len(ftrain), R, -1), 'f_same': fn(fsame).reshape(len(ftrain), E, -1), 'f_cross': fn(fcross).reshape(len(ftrain), E, -1)}
            buf = io.BytesIO()
            np.savez(buf, **out)
            DUR.atomic_write(codes_path(name, seed), buf.getvalue())


def use_encoder(name, seed):
    """S1's runners read keys from their code cache: this process addresses by the named encoder."""
    S1.CODES.pop(seed, None)
    c = dict(S1.load(seed))
    z = np.load(codes_path(name, seed))
    c.update(key_train=z['key_train'], key_eval=z['key_eval'], f_train=z['f_train'], f_same=z['f_same'], f_cross=z['f_cross'])
    S1.CODES[seed] = c
    return c


def grid_task(task):
    name, seed, H, k, tau = task
    use_encoder(name, seed)
    key, res, rt = S1.one_grid((seed, 'sense-only', H, k, tau))
    return f'{name}/{key}', res, rt


def font_read(name, seed, H, tau):
    """90 symbols with a second font; counts assigned in turn. Cue traces in the symbol's own font; reads there and in the other font. Memory level, abstentions wrong."""
    c = use_encoder(name, seed)
    S = len(c['f_train'])
    counts = np.repeat(np.arange(1, 10), S // 9 + 1)[:S]
    rng = nc.stream('sm_fonts', seed)
    mem = Memory(H, 1, tau, 32, S1.CORE)
    for _ in range(S1.STEPS * BATCH):
        i, r, col = int(rng.integers(S)), int(rng.integers(R)), int(rng.integers(len(c['coll_train'])))
        mem.write(S1.unit(c['f_train'][i, r]), c['coll_train'][col, counts[i] - 1])
    truth = np.repeat(counts, E)
    out = {'units': mem.h, 'forced': mem.forced}
    for field in ('f_same', 'f_cross'):
        content, ok = mem.read(S1.unit(c[field].reshape(S * E, -1)))
        pick = (S1.unit(content) @ S1.unit(c['protos']).T).argmax(1) + 1
        out[field] = {'accuracy': float(((pick == truth) & ok).mean()), 'coverage': float(ok.mean())}
    return out


def run_grid():
    t0 = time.perf_counter()
    tasks = [(n, s, H, k, t) for n in ENCODERS for s in S0.SEEDS for H, k, t in itertools.product(S1.HS, S1.KS, S1.TAUS)]
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = list(pool.imap_unordered(DUR.durable(grid_task, RUNS / 'done-grid', '{0}-{1}-{2}-{3}-{4}'), tasks))
    out = {}
    for n in ENCODERS:
        grid = {k[len(n) + 1:]: r for k, r, _ in res if k.startswith(n + '/')}
        stores = {}
        for s in S0.SEEDS:
            use_encoder(n, s)
            stores[f'{s}/sense-only'] = S1.explicit_store(s, 'sense-only')
        sel = S1.select('sense-only', grid, stores, S0.SEEDS)
        best = sel['selected']
        fonts = {str(s): font_read(n, s, best['H'], best['tau']) for s in S0.SEEDS} if best else None
        out[n] = {'selection': sel, 'stores': stores, 'fonts': fonts, 'grid': grid}
    S0.save_json(f'{REPORT}-grid.json', {'status': 'exploratory, pilot seeds; tau grid widened to 0.6 because this encoder has lower same-symbol similarity', 'results': out})
    S0.save_json(f'{REPORT}-grid-runtime.json', S0.runtime({'phase': 'CPU grid', 'device': 'cpu', 'workers': WORKERS, 'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)), 'writes': int(sum(t['writes'] for _, _, t in res))}))
    print({n: (out[n]['selection']['selected'] or {}).get('H') and (out[n]['selection']['selected']['H'], out[n]['selection']['selected']['tau']) for n in ENCODERS}, flush=True)


def learn(task):
    seed, lr, H, tau = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    LS.TAU = tau
    c = S1.load(seed)
    model, start = encoder('unlabelled', seed), encoder('unlabelled', seed).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr)
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    rng = nc.stream('sm_learning', seed, str(lr))
    streams = {'A': S1.stream('A', seed, c), 'B': S1.stream('B', seed, c)}
    arms = {k: LS.Arm(*v, H) for k, v in LEARN_ARMS.items()}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)
    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    start_cues, start_queries = np_(enc(start, cue)), np_(enc(start, query))
    partner = SRL.pair_labels()
    n_cls, n_r = bg.shape[:2]
    reads, drift = {}, {}

    def read_point(b, tags):
        model.eval()
        with torch.no_grad():
            q, cc = np_(enc(model, query)), np_(enc(model, cue))
        reads[str(b)] = {}
        for name, arm in arms.items():
            Zq = start_queries if name == 'frozen' else q
            out = S1.read_sets(arm.mem, S1.unit(Zq), c, tags)
            out['units'], out['forced'] = arm.mem.h, arm.mem.forced
            out['unknown_coverage'] = float(arm.mem.read(S1.unit(Zq[2 * N * E:3 * N * E]))[1].mean())
            reads[str(b)][name] = out
        idn = S0.address_reading(torch.from_numpy(cc).reshape(2 * N, R, -1), torch.from_numpy(q[:2 * N * E]).reshape(2 * N, E, -1), c['counts'])
        drift[str(b)] = {'query_cosine_with_start': float((q * start_queries).sum(1).mean()), 'identity_90': idn['identity'], 'same_symbol_similarity': idn['same_symbol_similarity'], 'other_symbol_similarity': idn['other_symbol_similarity']}

    step = 0
    for b, tag in ((4, 'A'), (5, 'B'), (6, None)):
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
                v = c['coll_train'][part[:, 2], c['counts'][part[:, 0]] - 1]
                for i in range(len(part)):
                    for name, arm in arms.items():
                        arm.write(start_cues[flat[i]] if name == 'frozen' else live[i], v[i], flat[i])
            if step % LS.TICK == 0:
                due = [a for a in arms.values() if a.every and step % a.every == 0]
                if due:
                    with torch.no_grad():
                        table = np_(enc(model, cue))
                    for a in due:
                        a.rebuild(table)
        read_point(b, 'A' if b == 4 else 'AB')
    return f'{seed}/{lr}/{H}', {'reads': reads, 'drift': drift, 'seed': seed, 'lr': lr, 'H': H, 'tau': tau}, S0.timing(t0, c0)


def learning():
    t0 = time.perf_counter()
    sel = json.load(open(f'{REPORT}-grid.json'))['results']['unlabelled']['selection']['selected']
    assert sel, 'no eligible setting on frozen unlabelled keys'
    tasks = [(s, lr, H, sel['tau']) for s in S0.SEEDS for lr in LRS for H in sorted({sel['H'], 2048})]
    res = S0.pool_map(DUR.durable(learn, RUNS / 'done-learning', '{0}-{1}-{2}'), tasks)
    S0.save_json(f'{REPORT}-learning.json', {'status': 'exploratory, pilot seeds, A then B; the encoder keeps learning from augmentations, no symbol knowledge; values and reading on block-3 host codes', 'selected': {k: sel[k] for k in ('H', 'k', 'tau')}, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-learning-runtime.json', S0.runtime({'phase': 'encoder training with memory writes', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'codes': make_codes, 'grid': run_grid, 'learning': learning}[sys.argv[2]]()
