"""Post hoc diagnosis of the translation-memory pilot's storage-gate failure (docs/translation-memory-v1-pilot.md).

On the saved host state at the end of batch 6 (seed 299, `main`), for each arm: how well each candidate key representation
identifies the 120 source words (1-nearest-neighbor word identity of evaluation renderings against prototypes of 8
training renderings), how much of the prototypes' variance lies in the top three directions, and how distinct the
separated patterns are (1,024 units, 20 active): each word's largest overlap with another word, and an evaluation
rendering's overlap with its own word's pattern.

Usage: .venv/bin/python -m experiments.translation_memory_diagnosis
"""
import json

import numpy as np
import torch

from experiments import limited_memory as LM
from experiments import number_systems as X
from experiments import translation_memory_pass as MP

M, K, SEED, POINT = 1024, 20, 299, 6


def separation_stats(proto, ev, transform):
    pw, ew = transform(proto), transform(ev)
    ident = float((torch.cdist(ew, pw).argmin(1) == torch.arange(len(proto)).repeat_interleave(len(ev) // len(proto))).float().mean())
    R = torch.from_numpy(X.nc.stream('tm_diagnosis', M, pw.shape[1]).standard_normal((M, pw.shape[1])))
    top = lambda z: torch.zeros(len(z), M, dtype=torch.float64).scatter_(1, (z @ R.T).topk(K, 1).indices, 1.)
    sp, se = top(pw), top(ew)
    ov = sp @ sp.T / K
    ov.fill_diagonal_(0)
    own = float(((se.reshape(len(proto), -1, M) * sp[:, None]).sum(-1) / K).mean())
    return {'identity_1nn': ident, 'nearest_other_overlap_mean': float(ov.max(1).values.mean()), 'nearest_other_overlap_max': float(ov.max()),
            'eval_vs_own_overlap': own}


def main():
    torch.set_num_threads(4)
    LM.install()
    out = {}
    for arm in ('mixed', 'sparse48'):
        h = MP.Host(arm, SEED, 'main')
        mat = h.mat
        enc, model = h.load(POINT)
        code = lambda x: MP.encode(enc, model, x)
        with torch.no_grad():
            feat = lambda x: enc['image'](torch.from_numpy(np.ascontiguousarray(x.reshape(-1, 28, 28), dtype=np.float32))).double()
        rec = {}
        for name, f in (('code16', code), ('encoder64', feat)):
            tr, ev = f(mat['src_train'][:, :8]), f(mat['src_eval'])
            proto = tr.reshape(120, 8, -1).mean(1)
            eig = torch.linalg.eigvalsh(torch.cov(proto.T))
            mu, s = proto.mean(0), proto.std()
            rec[name] = {'top3_variance_share': float(eig[-3:].sum() / eig.sum()), 'centred': separation_stats(proto, ev, lambda z: (z - mu) / s)}
            if name == 'code16':
                A = code(mat['anchor_imgs'])
                for wname, stats in (('anchor_whitened', A), ('prototype_whitened_oracle', proto)):
                    C = torch.cov(stats.T) + 1e-3 * torch.eye(16, dtype=torch.float64)
                    L = torch.linalg.cholesky(torch.linalg.inv(C))
                    m0 = stats.mean(0)
                    rec[name][wname] = separation_stats(proto, ev, lambda z: (z - m0) @ L)
        out[arm] = rec
    path = X.ROOT / 'reports' / 'translation-memory-v1-pilot-diagnosis.json'
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
