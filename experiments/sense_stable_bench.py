"""Costs for docs/sense-stable-v1-pilot.md, corrected: occupied and allocated state from the pilot's unit counts, and isolated batch-one retrieval latency.
Latency is on synthetic codes of the pilot's sizes (two models of 384 dims, the pilot's units per centre), the first query of each read excluded; CPU, one thread.
Usage: /data/venvs/sri-cuda/bin/python -m experiments.sense_stable_bench reports/sense-stable-v1-pilot.json.gz
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src import senses as SN
from src.sparse_local_memory import Memory

D, PER_CENTRE, SINGLE = 384, 2048, 8192
PER_UNIT = 4 * 3 * (2 * D) + 8  # address sum, address, content sum (float32); count (int64). Scoring tags are diagnostics and excluded.


def state(report):
    d = json.loads(gzip.decompress(Path(report).read_bytes()))
    out = {}
    for run, r in d['results'].items():
        u = r['units']
        k = len(u['kept+frozen'])
        out[run] = {'online': {'occupied_MB': u['online'] * PER_UNIT / 2 ** 20, 'allocated_MB': SINGLE * PER_UNIT / 2 ** 20},
                    'kept+frozen': {'occupied_MB': sum(u['kept+frozen']) * PER_UNIT / 2 ** 20, 'allocated_MB': k * PER_CENTRE * PER_UNIT / 2 ** 20},
                    'centres_and_gates_kB': (k * 2 * D * 4 + k * 770 * 4) / 2 ** 10, 'kept_gate_codes_MB': r['state_bytes']['kept+frozen']['kept_codes'] / 2 ** 20}
    return out


def latency(units_per_centre, repeats=2000, seed=0):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    models = ('dinov2-s', 'trocr-s')
    gate, _ = SN.Gate.train([SN.unit(rng.normal(size=(256, D)).astype(np.float32)) for _ in models],
                            [np.stack([SN.unit(rng.normal(size=(256, D)).astype(np.float32))] * 2) for _ in models], 0, steps=5, batch=64)
    km = SN.KeptMemory(models, lambda: Memory(PER_CENTRE, 1, 2., 2 * D, 2 * D))  # tau above 1: every write opens a unit
    single = Memory(SINGLE, 1, 2., 2 * D, 2 * D)
    for h in units_per_centre:
        c = {n: rng.normal(size=D).astype(np.float32) for n in models}
        j = km.open_centre(c, gate)
        f = {n: (c[n] + rng.normal(size=(h, D))).astype(np.float32) for n in models}
        km.write(f, f, np.full(h, j))
        for z in km.key(f, j):
            single.write(z, z)
    qs = [{n: rng.normal(size=(1, D)).astype(np.float32) for n in models} for _ in range(repeats)]
    C = {n: np.stack([c[n] for c in km.centres]) for n in models}
    t = []
    for q in qs:
        t0 = time.perf_counter()
        km.read(q)
        t.append(time.perf_counter() - t0)
    kept = float(np.median(t[1:]))
    t = []
    for q in qs:
        t0 = time.perf_counter()
        a = (SN.unit(q['dinov2-s']) @ SN.unit(C['dinov2-s']).T).argmax(1)
        w = gate([SN.unit(q[n]) for n in models])
        z = SN.join([SN.centred(q[n], C[n][a]) for n in models], [w[:, 0], w[:, 1]])
        S = z @ single.a[:single.h].T
        i = S.argmax(1)
        single.u[i] / single.n[i, None]
        t.append(time.perf_counter() - t0)
    return {'kept+frozen_ms': 1e3 * kept, 'online_ms': 1e3 * float(np.median(t[1:])), 'units': int(sum(units_per_centre)), 'centres': len(units_per_centre)}


if __name__ == '__main__':
    st = state(sys.argv[1])
    print(json.dumps(st, indent=1))
    d = json.loads(gzip.decompress(Path(sys.argv[1]).read_bytes()))
    print(json.dumps({run: latency(r['units']['kept+frozen']) for run, r in d['results'].items()}, indent=1))
