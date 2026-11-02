"""Shared-core sparse memory v1: unseen traces. The sensory trace pilot's traces were collections the host trained on. Here stored traces are images it never saw, at increasing distance from training:
new arrangements in the training rendering, the evaluation rendering, count formats never taught (tally, tree, banana), and a script with no quantity to read, as a negative control (thai_digits). Same hosts, memory and writes.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_unseen_traces pilot
"""
import json
import time

import numpy as np
import torch

from experiments import shared_core_moving_values as MV
from experiments import shared_core_stage1_pilot as S1
from experiments import shared_core_traces as TR
from src.sparse_local_memory import Memory

S0, nc, X, P, ns, DEV = TR.S0, TR.nc, TR.X, TR.P, TR.S0.ns, TR.DEV
NAME = 'shared-core-sparse-memory-v1-unseen-traces'
REPORT = X.ROOT / 'reports' / NAME
H, K, TAU, PER = MV.H, MV.K, MV.TAU, 300
FORMATS = ('tally', 'tree', 'banana', 'thai_digits')
HOWS = ('exemplar-1', f'exemplar-{TR.EXEMPLARS}')


def source_images(seed):
    """Per source, (PER, 9, 28, 28): images no host trained on. Trace row (collection, count) maps to image (collection mod PER, count)."""
    out = {}
    for mode in ('radius', 'area'):
        out[f'new-{mode}'] = np.stack([np.stack([ns.render_dots(n, nc.stream('sc_unseen', seed, mode, n, j), mode=mode) for n in ns.NUMBERS]) for j in range(PER)]).astype(np.float32)
    for name in FORMATS:
        x, y = X.DATA[(name, 'eval')]
        out[name] = np.stack([x[y == n][:PER] for n in ns.NUMBERS], 1).astype(np.float32)
    return out


def one_seed(seed):
    c = S1.load(seed)
    d = MV.CACHE / f'sparse48-seed{seed}-full-N45'
    ev = {b: dict(np.load(d / f'eval-{b}.npz')) for b in MV.BLOCKS}
    streams = {'A': S1.stream('A', seed, c), 'B': S1.stream('B', seed, c)}
    mems, writes, units = {}, [], []
    mem = Memory(H, K, TAU, 32, S1.CORE)
    for b, tag in ((4, 'A'), (5, 'B')):
        z = np.load(d / f'train-{b}.npz')
        for zq, v in zip(S1.unit(z['q_frozen']), z['t_live']):
            units.append(int(mem.write(zq, v)[0]))
        s = streams[tag]
        writes.append(np.stack([s[:, 2], c['counts'][s[:, 0]] - 1], 1))
        snap = Memory(H, K, TAU, 32, S1.CORE)
        snap.a, snap.n, snap.h, snap.u = mem.a.copy(), mem.n.copy(), mem.h, mem.u.copy()
        mems[b] = snap
    mems[6] = mems[5]
    rows, units = np.concatenate(writes), np.array(units)
    images = source_images(seed)
    res = {'after': {}, 'single_trace_to_count': {}, 'image_range': {k: [float(v.min()), float(v.max())] for k, v in images.items()}}
    for b in (4, 5, 6):
        _, _, code = TR.load_host(seed, b)
        protos_table = TR.table(code)
        protos = protos_table.mean(0)
        n_w = len(streams['A']) if b == 4 else len(rows)
        tables = {'training-images': protos_table}
        for name, img in images.items():
            tables[name] = torch.stack([S0.batched(code, img[:, k]) for k in range(9)], 1).cpu().numpy().astype(np.float32)
        row = {}
        for tag in ('A',) if b == 4 else ('A', 'B'):
            s = MV.eval_set(ev[b], tag)
            out = {}
            for name, T in tables.items():
                r = rows[:n_w].copy()
                r[:, 0] %= len(T)
                out[name] = {how: TR.read_with(mems[b], TR.unit_values(units[:n_w], r, T, how), s['q'], s['cand'], s['counts'], s['truth'], protos) for how in HOWS}
            row[tag] = out
        res['after'][str(b)] = row
        counts = np.arange(1, 10)
        res['single_trace_to_count'][str(b)] = {name: float((np.linalg.norm(T[:, :, None] - protos[None, None], axis=-1).argmin(-1) + 1 == counts).mean()) for name, T in tables.items()}
        res.setdefault('mean_trace_to_count', {})[str(b)] = {name: float((np.linalg.norm(T.mean(0)[:, None] - protos[None], axis=-1).argmin(-1) + 1 == counts).mean()) for name, T in tables.items()}
    return res


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    S0.prepare()
    S0.set_up_task('sparse48')
    with torch.no_grad():
        out = {str(s): one_seed(s) for s in S0.SEEDS}
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, A then B; hosts, memory, keys and writes as the sensory trace pilot; only the stored trace images differ', 'results': out})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'reads at checkpoints', 'wall_seconds': time.perf_counter() - t0, 'parent_cpu_seconds': time.process_time() - c0}))


if __name__ == '__main__':
    main()
