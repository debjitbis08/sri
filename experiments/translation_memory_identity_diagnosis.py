"""Post hoc decomposition of the identity host's perception-gate failure (docs/translation-memory-v1-pilot.md).

Read-only on the saved states. Recognition = the nearest key's word, by cosine of centred keys, from evaluation renderings:
  within_batch   point p's encoder, unmapped: batch p's single-rendering keys only (20-way), no other batch, no drift
  same_encoder   point p's encoder, unmapped: fresh single-rendering keys of all 120 words (the buffered renderings; a
                 measurement, since the system no longer holds old renderings), 120-way: no drift, no map
  prototype      the same with prototypes of eight training renderings
  seen_queries   same_encoder with other training renderings as queries (does the failure generalize to seen images?)
  historical     the keys as written at each word's write point, 120-way, for each map setting (`none` or lambda), queries
                 at the write point and at points 6 and 7
The historical setting with lambda 0.01 is the gate's own reading.

Usage: TM_IDENTITY=1 .venv/bin/python -m experiments.translation_memory_identity_diagnosis
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

SEEDS = (299, 298)
MAPS = (None, 0.001, 0.01, 0.1, 1.)
N, R = TM.N_WORDS, TM.R_EVAL


def recognize(queries, keys, mu, ids=None):
    """queries (n, r, d), keys (m, d); ids (m,) word of each key, by default row order. Per query word: accuracy (n,)."""
    ids = torch.arange(len(keys)) if ids is None else ids
    q = F.normalize(queries.reshape(-1, queries.shape[-1]).float() - mu.float(), dim=1)
    near = ids[(q @ F.normalize(keys.float() - mu.float(), dim=1).T).argmax(1)]
    truth = (torch.arange(N) if len(queries) == N else ids).repeat_interleave(queries.shape[1])
    return (near == truth).float().reshape(len(queries), -1).mean(1)


def by_batch(acc):
    return [float(acc[TM.WORDS * b:TM.WORDS * (b + 1)].mean()) for b in range(TM.BATCHES)]


def analyse(arm, seed):
    host = MP.Host(arm, seed, 'main')
    mat = host.mat
    render = {e['word']: e['render'] for e in host.events if e['kind'] == 'insert'}
    fresh_imgs = np.stack([mat['src_train'][w, render[w]] for w in range(N)])
    seen = np.stack([mat['src_train'][w, [(render[w] + 1 + i) % TM.R_TRAIN for i in range(R)]] for w in range(N)])
    keys_at, ev_at = {}, {}
    out = {'within_batch': [], 'same_encoder': {}, 'prototype': {}, 'seen_queries': {}, 'anchor_residual': {}}
    maps = {lam: TS.ReferenceMap(MP.D_CODE, lam or 1.) for lam in MAPS}
    mu = None
    hist = {lam: {} for lam in MAPS}
    queries = {lam: {} for lam in MAPS}
    for point in range(1, TM.BATCHES + 2):
        enc, model = host.load(point)
        code = lambda x: MP.encode_keys(enc, model, x)
        A, ev, fresh = code(mat['anchor_imgs']), code(mat['src_eval']).reshape(N, R, -1), code(fresh_imgs)
        if point == 1:
            mu = A.mean(0)
            for m in maps.values():
                m.set_reference(A)
        elif point <= TM.BATCHES:
            for lam, m in maps.items():
                if lam is not None:
                    m.fit(A)
        if point <= TM.BATCHES:
            b = TM.batch_words(point - 1)
            out['within_batch'].append(float(recognize(ev[b], fresh[b], mu).mean()))
        if point >= TM.BATCHES:
            tag = str(point)
            out['same_encoder'][tag] = by_batch(recognize(ev, fresh, mu))
            proto = code(mat['src_train'][:, :8]).reshape(N, 8, -1).mean(1)
            out['prototype'][tag] = by_batch(recognize(ev, proto, mu))
            out['seen_queries'][tag] = by_batch(recognize(code(seen).reshape(N, R, -1), fresh, mu))
        for lam, m in maps.items():
            apply = (lambda z: z.float()) if lam is None else m
            if point <= TM.BATCHES:
                for w in TM.batch_words(point - 1):
                    hist[lam][w] = apply(fresh[w][None])[0]
            queries[lam][point] = apply(ev.reshape(-1, MP.D_CODE)).reshape(N, R, -1)
            out['anchor_residual'].setdefault(str(lam), {})[str(point)] = float((apply(A).double() - maps[lam].ref).pow(2).mean() / maps[lam].s2)
    out['historical'] = {}
    for lam in MAPS:
        K = torch.stack([hist[lam][w] for w in range(N)])
        write_q = torch.stack([queries[lam][w // TM.WORDS + 1][w] for w in range(N)])
        out['historical'][str(lam)] = {'write_point': by_batch(recognize(write_q, K, mu)),
                                       '6': by_batch(recognize(queries[lam][TM.BATCHES], K, mu)),
                                       '7': by_batch(recognize(queries[lam][TM.BATCHES + 1], K, mu))}
    return out


def main():
    torch.set_num_threads(8)
    LM.install()
    TM.install_key_head()
    out = {f'{a}/seed{s}': analyse(a, s) for s in SEEDS for a in TM.ARMS}
    path = X.ROOT / 'reports' / ('translation-memory-v1-identity-diagnosis.json' if TM.IDENTITY else 'translation-memory-v1-pilot-decomposition.json')
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    mean = lambda f: round(float(np.mean([np.mean(f(v)) for v in out.values()])), 3)
    print('within_batch 20-way', mean(lambda v: v['within_batch']))
    for t in ('6', '7'):
        print(f'point {t} same encoder 120-way', mean(lambda v: v['same_encoder'][t]), 'prototype', mean(lambda v: v['prototype'][t]), 'seen', mean(lambda v: v['seen_queries'][t]))
        print('   by batch same encoder', np.round(np.mean([v['same_encoder'][t] for v in out.values()], 0), 2).tolist())
    for lam in MAPS:
        print('historical', lam, {q: mean(lambda v: v['historical'][str(lam)][q]) for q in ('write_point', '6', '7')})


if __name__ == '__main__':
    main()
