"""Shared-core sparse memory v1: a sensory encoder that keeps learning. The frozen encoder continues its own training on background symbols while memory is taught (no host), so addresses drift. Do they go stale, and can stored cue traces rebuild them?
Values and reading stay on block-3 host codes, so every difference here is from addresses.

Time: 6,000 encoder updates, 2,000 per block: A in the first, B in the second, nothing in the third; 16 writes per update, with the encoder as it is at that update.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 SC0_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_learning_senses pilot
"""
import json
import time

import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import shared_core_moving_values as MV
from experiments import shared_core_stage0 as S0
from experiments import shared_core_stage1_pilot as S1
from experiments import streaming_retrieval as SR
from experiments import translation_stable_keys as TK
from src.sparse_local_memory import Memory

nc, X = S0.nc, S0.X
NAME = 'shared-core-sparse-memory-v1-learning-senses'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
K, TAU = MV.K, MV.TAU
HS = (128, 512, 2048)  # 128 is the setting locked on stable keys; the larger two were added after H 128 filled, so they are post hoc
N, E, R = S1.N, S0.EVAL_RENDERS, S0.TRAIN_RENDERS
BLOCK, BATCH, TICK = 2000, 16, 250
LRS = (1e-3, 1e-2)
# arm: (traces kept per unit, or None for all; updates between rebuilds, or None)
ARMS = {'frozen': (0, None), 'no-rebuild': (0, None), 'rebuild-1-every-250': (1, 250), 'rebuild-8-every-250': (8, 250), 'rebuild-8-every-2000': (8, 2000), 'rebuild-all-every-250': (None, 250)}


class Arm:
    def __init__(self, keep, every, H):
        self.H = H
        self.mem = Memory(H, K, TAU, 32, S1.CORE)
        self.keep, self.every = keep, every
        self.seen = np.zeros(H, np.int64)
        self.tu, self.tr = [], []  # traces: unit, and flat index into the cue table

    def write(self, z, v, flat):
        i = int(self.mem.write(z, v)[0])
        if self.keep != 0 and (self.keep is None or self.seen[i] < self.keep):
            self.tu.append(i); self.tr.append(flat)
            self.seen[i] += 1

    def rebuild(self, table):
        """Each unit's address becomes the mean direction of its kept traces under the current encoder, weighted as its write count."""
        if not self.tu:
            return
        tu, tr = np.array(self.tu), np.array(self.tr)
        total = np.zeros((self.H, 32), np.float32)
        np.add.at(total, tu, table[tr])
        have = np.bincount(tu, minlength=self.H)
        ok = have > 0
        self.mem.c[ok] = total[ok] / have[ok, None] * self.mem.n[ok, None]
        self.mem.a[ok] = S1.unit(self.mem.c[ok])


def run(task):
    seed, lr, H = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    c = S1.load(seed)
    device = DEV.NAME if DEV.ACTIVE else 'cpu'
    model = SR.key_encoder(seed).to(device)
    start = SR.key_encoder(seed).to(device).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr)
    bg = S0.background_images(seed)
    train, ev, _ = S0.test_material(seed)
    cue_images = np.ascontiguousarray(train[:2 * N].reshape(-1, 28, 28), dtype=np.float32)
    query_images = np.ascontiguousarray(ev[:3 * N].reshape(-1, 28, 28), dtype=np.float32)
    rng = nc.stream('sc_learning_senses', seed, str(lr))
    streams = {'A': S1.stream('A', seed, c), 'B': S1.stream('B', seed, c)}
    arms = {k: Arm(*v, H) for k, v in ARMS.items()}
    enc = lambda m, x: S0.batched(lambda b: m(b), x)

    def to_np(t):
        return t.detach().cpu().numpy().astype(np.float32)

    start_cues, start_queries = to_np(enc(start, cue_images)), to_np(enc(start, query_images))
    labels = torch.from_numpy(np.repeat(np.arange(TK.CLASSES), TK.VIEWS))
    reads, drift = {}, {}
    n_cls, n_r = bg.shape[:2]

    def read_point(b, tags):
        model.eval()
        with torch.no_grad():
            q, cues = to_np(enc(model, query_images)), to_np(enc(model, cue_images))
        reads[str(b)] = {}
        for name, arm in arms.items():
            Zq = start_queries if name == 'frozen' else q
            out = S1.read_sets(arm.mem, Zq, c, tags)
            out['units'], out['forced'] = arm.mem.h, arm.mem.forced
            out['unknown_coverage'] = float(arm.mem.read(Zq[2 * N * E:3 * N * E])[1].mean())
            reads[str(b)][name] = out
        counts = c['counts']
        identity = S0.address_reading(torch.from_numpy(cues).reshape(2 * N, R, -1), torch.from_numpy(q[:2 * N * E]).reshape(2 * N, E, -1), counts)
        drift[str(b)] = {'query_cosine_with_start': float((q * start_queries).sum(1).mean()), 'identity_90': identity['identity'], 'same_symbol_similarity': identity['same_symbol_similarity'], 'other_symbol_similarity': identity['other_symbol_similarity']}

    step = 0
    losses = []
    for b, tag in ((4, 'A'), (5, 'B'), (6, None)):
        s = streams[tag] if tag else None
        for u in range(BLOCK):
            cls = rng.choice(n_cls, TK.CLASSES, replace=False)
            view = np.stack([rng.choice(n_r, TK.VIEWS, replace=False) for _ in cls])
            x = bg[cls[:, None], view].reshape(-1, 28, 28)
            model.train()
            loss = TK.supcon(model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
            step += 1
            model.eval()
            if s is not None:
                part = s[u * BATCH:(u + 1) * BATCH]
                flat = part[:, 0] * R + part[:, 1]
                with torch.no_grad():
                    z = to_np(model(torch.from_numpy(cue_images[flat])))
                v = c['coll_train'][part[:, 2], c['counts'][part[:, 0]] - 1]
                for i in range(len(part)):
                    for name, arm in arms.items():
                        arm.write(start_cues[flat[i]] if name == 'frozen' else z[i], v[i], flat[i])
            if step % TICK == 0:
                due = [a for a in arms.values() if a.every and step % a.every == 0]
                if due:
                    with torch.no_grad():
                        table = to_np(enc(model, cue_images))
                    for a in due:
                        a.rebuild(table)
        read_point(b, 'A' if b == 4 else 'AB')
    res = {'reads': reads, 'drift': drift, 'loss_first_200': float(np.mean(losses[:200])), 'loss_last_200': float(np.mean(losses[-200:])), 'seed': seed, 'lr': lr, 'H': H}
    return f'{seed}/{lr}/{H}', res, S0.timing(t0, c0)


def main():
    t0 = time.perf_counter()
    S0.prepare()
    for s in S0.SEEDS:
        S0.test_material(s)
    res = S0.pool_map(DUR.durable(run, RUNS / 'done', '{0}-{1}-{2}'), [(s, lr, H) for s in S0.SEEDS for lr in LRS for H in HS])
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory, pilot seeds, A then B; encoder keeps its labelled background training, so this tests drift and not identity without labels; values and reading on block-3 host codes', 'setting': {'H': HS, 'k': K, 'tau': TAU}, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'encoder training with memory writes', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
