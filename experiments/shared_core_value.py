"""Shared-core sparse memory v1: the value diagnostic, on stage 0's locked sparse48 checkpoints. No memory learns here. Separates what limits an answer:
the address (oracle, or the explicit store on real keys), the stored value (accumulated from the real teaching stream, or the right count's mean) and the decoder.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_value pilot
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import shared_core_stage0 as S0
from experiments import streaming_retrieval as SR

nc, X, DEV = S0.nc, S0.X, S0.DEV
NAME = 'shared-core-sparse-memory-v1-value' + ('-smoke' if S0.SMOKE else '')
REPORT = X.ROOT / 'reports' / NAME
N = S0.LOADS[0] // 2
STEPS, BATCH = (20, 16) if S0.SMOKE else (2000, 16)


def host(seed):
    S0.set_up_task('sparse48')
    ck = torch.load(S0.checkpoint_path('sparse48', seed), weights_only=False, map_location='cpu')
    enc, model, _ = LM.setup('sparse48', seed)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    for m in (*enc.values(), model):
        m.requires_grad_(False)
    return enc, model


def writes(cps, row, counts, tag, seed):
    """The arbitrary pairings sampler: the same draws as AP.pair_items, as (symbol row, rendering, collection index)."""
    rng = nc.stream('ap_pairs', seed, {'A': 4, 'B': 5}[tag], tag)
    out = []
    for _ in range(STEPS * BATCH):
        cp = cps[int(rng.integers(len(cps)))]
        out.append((row[cp], int(rng.integers(S0.TRAIN_RENDERS)), int(rng.integers(len(P.DOTS['train'])))))
    return np.array(out)


def candidates(n, E, seed, tag):
    rng = nc.stream('ap_eval_candidates', seed, tag)
    order = np.stack([rng.permutation(9) for _ in range(n * E)])
    j = rng.integers(P.EVAL_PER, size=n * E)
    return order, np.stack([P.DOTS['succ'][int(k) + 1][j[t]] for t in range(n * E) for k in order[t]])


def decoders(content, cand, order, truth, model, centre, protos):
    """content (q, d), cand (q, 9, d), order (q, 9), truth 1..9. Accuracy and off-by-one by decoder; none is told the answer."""
    out = {}
    pos = lambda s: order[np.arange(len(order)), s.argmax(1).cpu().numpy()] + 1
    picks = {'cosine': pos((F.normalize(content, dim=-1)[:, None] * F.normalize(cand, dim=-1)).sum(-1)),
             'cosine_centred': pos((F.normalize(content - centre, dim=-1)[:, None] * F.normalize(cand - centre, dim=-1)).sum(-1)),
             'euclidean': pos(-(content[:, None] - cand).pow(2).sum(-1)), 'head': pos(S0.G.pair(model, content, cand, 1)),
             'count_prototypes_no_candidates': (F.normalize(content, dim=-1) @ F.normalize(protos, dim=-1).T).argmax(1).cpu().numpy() + 1}
    for k, p in picks.items():
        out[k] = {'accuracy': float((p == truth).mean()), 'within_one': float((np.abs(p - truth) <= 1).mean())}
    return out


@torch.no_grad()
def one_seed(seed):
    enc, model = host(seed)
    code = lambda x: model.proj['image'](enc['image'](x))
    key = SR.key_encoder(seed).to(DEV.NAME if DEV.ACTIVE else 'cpu').requires_grad_(False)
    train, ev, _ = S0.test_material(seed)
    sets = S0.PS.draw_sets(seed, N, S0.PS.survivors())
    A, B = list(sets['A']), list(sets['B'])
    cps = A + B
    row = {c: k for k, c in enumerate(cps)}
    counts = np.array([{**sets['A'], **sets['B']}[c] for c in cps])
    S, R, E = len(cps), S0.TRAIN_RENDERS, S0.EVAL_RENDERS
    dots = P.DOTS['train']
    coll = torch.stack([S0.batched(code, dots[:, k]) for k in range(9)], 1)
    protos, centre = coll.mean(0), coll.reshape(-1, coll.shape[-1]).mean(0)
    w = np.concatenate([writes(A, row, counts, 'A', seed), writes(B, row, counts, 'B', seed)])
    i_sym, i_ren, i_col = (torch.as_tensor(w[:, j]) for j in range(3))
    v = coll[i_col, torch.as_tensor(counts[w[:, 0]] - 1)]
    stored = torch.zeros(S, v.shape[1]).index_add_(0, i_sym, v) / torch.bincount(i_sym, minlength=S)[:, None]
    keys = {'core': (lambda x: code(x)), 'frozen': (lambda x: key(x))}
    res = {'unique_symbols': S, 'exposures': len(w), 'distinct_exposures_per_symbol_min': int(torch.bincount(i_sym).min())}
    for tag, sub, off in (('A', A, 0), ('B', B, len(A))):
        n = len(sub)
        order, imgs = candidates(n, E, seed, tag)
        cand = S0.batched(code, imgs).reshape(n * E, 9, -1)
        rows = torch.arange(off, off + n).repeat_interleave(E)
        truth = counts[rows.cpu().numpy()]
        out = {'perfect_contents': decoders(protos[torch.as_tensor(truth - 1)], cand, order, truth, model, centre, protos),
               'oracle_address_accumulated': decoders(stored[rows], cand, order, truth, model, centre, protos)}
        for name, f in keys.items():
            t = S0.batched(f, train.reshape(-1, 28, 28)).reshape(len(train), R, -1)[:S]
            q = S0.batched(f, ev[off:off + n].reshape(-1, 28, 28))
            proto = F.normalize(F.normalize(t, dim=-1).mean(1), dim=-1)
            near = (F.normalize(q, dim=-1) @ proto.T).argmax(1)
            out[f'store_on_{name}_keys_accumulated'] = decoders(stored[near], cand, order, truth, model, centre, protos)
            out[f'store_on_{name}_keys_label'] = float((counts[near.cpu().numpy()] == truth).mean())
        res[tag] = out
    # the candidates themselves: single evaluation collections against the nine count prototypes
    succ = torch.stack([S0.batched(code, P.DOTS['succ'][k + 1]) for k in range(9)])
    near = (F.normalize(succ, dim=-1).reshape(9 * P.EVAL_PER, -1) @ F.normalize(protos, dim=-1).T).argmax(1).cpu().numpy() + 1
    t = np.repeat(np.arange(1, 10), P.EVAL_PER)
    res['single_collection_to_count'] = {'accuracy': float((near == t).mean()), 'within_one': float((np.abs(near - t) <= 1).mean()), 'by_count': [float((near[t == k] == k).mean()) for k in range(1, 10)]}
    return res


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    S0.prepare()
    out = {str(s): one_seed(s) for s in S0.SEEDS}
    S0.save_json(f'{REPORT}.json', {'status': 'exploratory diagnostic on pilot seeds; hosts frozen at block 3, so live values are block-3 codes', 'results': out})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'value diagnostic', 'wall_seconds': time.perf_counter() - t0, 'parent_cpu_seconds': time.process_time() - c0}))
    print('done', flush=True)


if __name__ == '__main__':
    main()
