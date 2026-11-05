"""Taught truth v1, post-hoc on checkpoints (not in the contract): when the kind -> individual link fails (S3b held-out
anonymous individuals, S3 named animals), what is answered instead, and where does U rank?"""
import json
import time

import numpy as np
import torch

from experiments import taught_truth as TT
from src import everyday_truth as T


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    st, _ = T.build()
    out = {}
    for arm in ('pred', 'pred-flat'):
        rec = {}
        for seed in TT.SEEDS:
            m = TT.Pred(arm, len(st.entities), len(st.relations), torch.Generator().manual_seed(0))
            m.load_state_dict(torch.load(TT.RUNS / f'{arm}-seed{seed}.pt'))
            for tag, rows in (('S3b', st.test == 'S3b'), ('fit-anon', (st.kind == 'anon') & (st.test == ''))):
                lg = TT.predict_values(m, st, rows)
                pred = np.array(T.VALUES)[lg.argmax(1)]
                rank_u = (lg > lg[:, [T.VALUES.index('U')]]).sum(1) + 1
                for q in ('always', 'usually', 'sometimes', 'never'):
                    mm = st.sub[rows] == q
                    d = rec.setdefault(f'{tag}/{q}', {'pred': {}, 'u_rank': []})
                    for v in T.VALUES:
                        d['pred'][v] = d['pred'].get(v, 0) + int((pred[mm] == v).sum())
                    d['u_rank'] += rank_u[mm].tolist()
        for k, d in rec.items():
            n = sum(d['pred'].values())
            out[f'{arm}/{k}'] = {'pred_share': {v: round(c / n, 2) for v, c in d['pred'].items() if c},
                                 'u_rank_mean': round(float(np.mean(d['u_rank'])), 2), 'n': n}
    rt = {'scope': 'whole-script timing', 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    TT.V2.save_json(TT.ROOT / 'reports' / 'taught-truth-v1-posthoc.json', out)
    TT.V2.save_json(TT.ROOT / 'reports' / 'taught-truth-v1-posthoc-runtime.json', rt)
    for k, v in out.items():
        print(k.ljust(28), v)
    print(rt)


if __name__ == '__main__':
    main()
