"""Post hoc controls for the translation-memory pilot (docs/translation-memory-v1-pilot.md, "Controls").

1. Oracle source codes (diagnostic): every word gets an independent random sparse source pattern; the writes (order, the
   host's teacher answers, target codes) and the retrieval rule are those of the pilot. Exact and corrupted cues.
   Asks whether the memory mechanism works when identity and stability are supplied.
2. Historical keys: the pilot's own consolidation (same buffer renderings, write points, map, separation and teacher
   answers), with every retrieval method on identical queries:
     flat-store   nearest dense mapped key (cosine)      flat-sparse  largest overlap with a separated key
     assoc-1step  one-step associative read              assoc-16     recurrent read, 16 steps
   Queries: the written keys themselves (exact), evaluation renderings encoded at the word's own write point (no drift
   since the write), and evaluation renderings at points 6 and 7 (after all writes; after the final context block).
   Scored against the written teacher answer (agreement) and against the true translation (correct), by write batch.

Usage: .venv/bin/python -m experiments.translation_memory_controls
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from experiments import limited_memory as LM
from experiments import number_systems as X
from experiments import translation_memory as TM
from experiments import translation_memory_pass as MP
from experiments import translation_store as TS

nc = X.nc
SEEDS, LAM = (299, 298), 0.01
GRID = MP.GRID
CORRUPT = (0., .2, .4, .6)


def corrupt(S, frac, rng):
    out = S.clone()
    k = int(S[0].sum())
    j = int(round(frac * k))
    for i in range(len(out)):
        on, off = out[i].nonzero()[:, 0], (out[i] == 0).nonzero()[:, 0]
        out[i, on[torch.from_numpy(rng.choice(len(on), j, replace=False))]] = 0
        out[i, off[torch.from_numpy(rng.choice(len(off), j, replace=False))]] = 1
    return out


def consolidate(arm, seed):
    """The pilot's consolidation, keeping each word's written keys, teacher answer and write point (measurement copies),
    and the map after each point."""
    host = MP.Host(arm, seed, 'main')
    mat = host.mat
    m = TS.ReferenceMap(MP.D_CODE, LAM)
    rec, maps, encs = {}, {}, {}
    for point in range(1, TM.BATCHES + 2):
        enc, model = host.load(point)
        A = MP.encode(enc, model, mat['anchor_imgs'])
        if point == 1:
            m.set_reference(A)
        elif point <= TM.BATCHES:
            m.fit(A)
        maps[point] = (m.A.clone(), m.mu, m.s2)
        tcodes = MP.encode(enc, model, mat['tgt_train'][:, 0])
        ev = MP.encode(enc, model, mat['src_eval']).reshape(TM.N_WORDS, TM.R_EVAL, -1)
        encs[point] = m(ev.reshape(-1, MP.D_CODE)).reshape(TM.N_WORDS, TM.R_EVAL, -1)
        if point > TM.BATCHES:
            continue
        buf = host.points[point - 1]['buffer']
        new = [e for e in buf if e['word'] not in rec]
        if not new:
            continue
        x = MP.encode(enc, model, np.stack([mat['src_train'][e['word'], e['render']] for e in new]))
        ans = MP.logits_against_targets(model, x, tcodes).argmax(1)
        xr = m(x)
        for e, k, a in zip(new, xr, ans.tolist()):
            rec[e['word']] = {'key': k, 'teacher': a, 'point': point}
    return mat, rec, maps, encs


def oracle(seed, teacher_order):
    """Control 1 for every grid configuration: random independent source patterns, the pilot's writes and teachers."""
    words = [w for w, _ in teacher_order]
    tg = torch.tensor([a for _, a in teacher_order])
    out = {}
    for m, k in GRID:
        S = TS.codebook(len(words), m, k, nc.stream('tm_oracle', seed, m, k))
        cb = TS.codebook(TM.N_TGT, m, k, nc.stream('tm_codebook', seed, m, k))
        mem = TS.Associative(m, m, k, k, nc.stream('tm_assoc', seed, m, k))
        mem.write(S, cb[tg])
        rng = nc.stream('tm_oracle_corrupt', seed, m, k)
        r = {}
        for frac in CORRUPT:
            res, info = mem.retrieve(corrupt(S, frac, rng))
            r[str(frac)] = {**{str(t): float((TS.target_scores(tp, cb).argmax(1) == tg).float().mean()) for t, tp in res.items()},
                            'converged': float(info['converged'].float().mean()), 'cycle': float(info['cycle'].float().mean())}
        out[f'{m},{k}'] = r
    return out


def historical(seed, mat, rec, maps, encs, cfgs):
    """Control 2: retrieval methods on identical queries from the pilot's own keys."""
    words = sorted(rec)
    truth = torch.tensor([mat['tgt'][w] for w in words])
    teach = torch.tensor([rec[w]['teacher'] for w in words])
    batch = torch.tensor([w // TM.WORDS for w in words])
    A6, mu, s2 = maps[TM.BATCHES]
    zkey = lambda xr: (xr.float() - mu.float()) / float(np.sqrt(s2))
    keys = torch.stack([rec[w]['key'] for w in words])
    dense = F.normalize(zkey(keys), dim=1)
    queries = {'exact': keys[:, None],
               'write_point': torch.stack([encs[rec[w]['point']][w] for w in words]),
               'point6': torch.stack([encs[TM.BATCHES][w] for w in words]),
               'point7': torch.stack([encs[TM.BATCHES + 1][w] for w in words])}
    out = {}
    for m, k in cfgs:
        sep = TS.Separation(MP.D_CODE, m, k, nc.stream('tm_separation', seed, m, k))
        cb = TS.codebook(TM.N_TGT, m, k, nc.stream('tm_codebook', seed, m, k))
        mem = TS.Associative(m, m, k, k, nc.stream('tm_assoc', seed, m, k))
        S = sep(keys, mu, s2)
        mem.write(S, cb[teach])
        r = {}
        for qname, q in queries.items():
            R = q.shape[1]
            qf = q.reshape(-1, MP.D_CODE)
            qs = sep(qf, mu, s2)
            rep = lambda t: t.repeat_interleave(R)
            ans = {'flat-store': teach[(F.normalize(zkey(qf), dim=1) @ dense.T).argmax(1)],
                   'flat-sparse': teach[(qs @ S.T + torch.arange(len(S)) * 1e-6).argmax(1)],
                   'assoc-1step': TS.target_scores(mem.one_step(qs), cb).argmax(1)}
            res, info = mem.retrieve(qs)
            ans['assoc-16'] = TS.target_scores(res[16], cb).argmax(1)
            r[qname] = {name: {'agree_teacher': {str(b + 1): float((a == rep(teach))[rep(batch) == b].float().mean()) for b in range(TM.BATCHES)},
                               'correct': {str(b + 1): float((a == rep(truth))[rep(batch) == b].float().mean()) for b in range(TM.BATCHES)}}
                        for name, a in ans.items()}
            r[qname]['assoc_cycle'] = float(info['cycle'].float().mean())
        out[f'{m},{k}'] = r
    out['teacher_correct'] = {str(b + 1): float((teach == truth)[batch == b].float().mean()) for b in range(TM.BATCHES)}
    return out


def main():
    torch.set_num_threads(8)
    LM.install()
    out = {}
    for seed in SEEDS:
        for arm in TM.ARMS:
            mat, rec, maps, encs = consolidate(arm, seed)
            order = sorted(rec, key=lambda w: (rec[w]['point'], w))
            out[f'{arm}/seed{seed}'] = {'oracle': oracle(seed, [(w, rec[w]['teacher']) for w in order]),
                                         'historical': historical(seed, mat, rec, maps, encs, ((1024, 41), (1024, 20), (256, 5)))}
            print('done', arm, seed, flush=True)
    path = X.ROOT / 'reports' / 'translation-memory-v1-pilot-controls.json'
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
