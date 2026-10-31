"""Post hoc diagnosis for streaming retrieval v1, after the frozen rule was applied. Not part of the decision.

1. Settling: input-driven dynamics under a constant exact cue, at the published time step, with no stream at all. Does the readout find the
   cued word? On real keys at width 32, their sign patterns at width 1,024, and independent random patterns.
2. Rejection: can the frames separate familiar from unfamiliar? Single-frame and smoothed AUROC of the top similarity, and the
   effect of removing the stream's mean query, a common component that corruption may add.

Usage: .venv/bin/python -m experiments.streaming_retrieval_diagnosis diagnose
"""
import json
import time

import numpy as np

from experiments import streaming_retrieval as SR
from src import streaming_retrieval as sr

nc = SR.nc
SEEDS = SR.CONFIRM_SEEDS
WORDS = 40
GAINS = (1, 4, 16, 64)


def spectrum(keys):
    gram = keys @ keys.T
    off = gram[~np.eye(len(keys), dtype=bool)]
    ev = np.linalg.eigvalsh(gram).clip(0)
    return {'mean_abs_cosine': float(np.abs(off).mean()), 'p99_abs_cosine': float(np.percentile(np.abs(off), 99)), 'effective_rank': float(ev.sum() ** 2 / (ev ** 2).sum())}


def settle(pat, sim, gain, sigma, seed, tag):
    """pat (P, N) with squared norm N; sim (P, P): the cue's similarities. 200 frames of 10 steps at dt = 0.01, as in the paper."""
    ok, top, second = [], [], []
    for w in range(WORDS):
        rng = nc.stream('sr_settle', seed, tag, gain, sigma, w)
        m = final_overlaps(sim[w], pat, gain, sigma, rng)
        ok.append(m.argmax() == w)
        s = np.sort(m)
        top.append(s[-1])
        second.append(s[-2])
    return {'correct': float(np.mean(ok)), 'top_overlap': float(np.mean(top)), 'runner_up_overlap': float(np.mean(second))}


def final_overlaps(sim, pat, gain, sigma, rng):
    N = pat.shape[1]
    x = rng.standard_normal(N, dtype=np.float32).astype(pat.dtype)
    a = gain * sim ** 2
    dt = .01
    for _ in range(2000):
        m = pat @ np.tanh(5 * x) / N
        x = x + dt * (pat.T @ (a * m) - x)
        if sigma:
            x = x + sigma * np.sqrt(dt) * rng.standard_normal(N, dtype=np.float32)
    return np.abs(pat @ np.tanh(5 * x)) / N


def auroc(pos, neg):
    order = np.argsort(np.r_[pos, neg])
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def masks(sch):
    T = len(sch['truth'])
    fam, unf = np.zeros(T, bool), np.zeros(T, bool)
    for a, b, w in sch['segments']:
        (unf if w == sr.REJECT else fam)[a + sr.GRACE:b] = True
    for start, length, _ in sch['bursts']:
        fam[start:start + length + sr.AFTER_CLEAR] = False
    return fam[:T], unf[:T]


def rejection(seed, cond):
    q, keys, sch = SR.load_stream(seed, cond, SR.PRIMARY_LOAD)
    fam, unf = masks(sch)
    out = {}
    centred = q - q[:2000].mean(0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    k = keys - keys.mean(0)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    for name, S in (('raw', q @ keys.T), ('centred', centred @ k.T)):
        top = S.max(1)
        smooth = sr.ema(S, 1 / 8)[1]
        out[name] = {'auroc_single_frame': auroc(top[fam], top[unf]), 'auroc_smoothed': auroc(smooth[fam], smooth[unf]), 'top_familiar': float(top[fam].mean()), 'top_unfamiliar': float(top[unf].mean()), 'mean_similarity': float(S.mean())}
    out['mean_query_norm'] = float(np.linalg.norm(q[:2000].mean(0)))
    return out


def task(seed):
    t0 = time.process_time()
    _, keys, _ = SR.load_stream(seed, SR.PRIMARY, SR.PRIMARY_LOAD)
    _, oracle, _ = SR.load_stream(seed, 'oracle', SR.PRIMARY_LOAD)
    proj = nc.stream('sr_project', seed).standard_normal((SR.ORACLE_N, keys.shape[1])).astype(np.float32)
    wide = sr.expand(keys, proj)[1]
    kinds = {'real keys, width 32': (sr.to_patterns(keys), keys @ keys.T),
             'sign patterns of real keys, width 1,024': (wide, sr.expand(keys, proj)[0] @ wide.T / np.sqrt(SR.ORACLE_N)),
             'independent patterns, width 1,024': (sr.to_patterns(oracle), oracle @ oracle.T)}
    out = {'spectrum': {'real keys': spectrum(keys), 'sign patterns': spectrum(wide / np.sqrt(SR.ORACLE_N)), 'independent patterns': spectrum(oracle)}, 'settling': {}}
    for tag, (pat, sim) in kinds.items():
        out['settling'][tag] = {f'gain {g}, sigma {s}': settle(pat, sim, g, s, seed, tag) for g in GAINS for s in (0, 2)}
    out['rejection'] = {c: rejection(seed, c) for c in ('clean', 'degraded', 'held')}
    return seed, out, time.process_time() - t0


def mean_tree(items):
    if isinstance(items[0], dict):
        return {k: mean_tree([i[k] for i in items]) for k in items[0]}
    return float(np.mean(items))


def main():
    t0 = time.perf_counter()
    done = SR.run_all(task, list(SEEDS[:4]) + [])
    res = mean_tree([d[1] for d in sorted(done, key=lambda z: z[0])])
    rej = SR.run_all(rej_task, list(SEEDS))
    res['rejection'] = mean_tree([r for r in rej])
    SR.save_json(f'{SR.REPORT}-diagnosis.json', {'settling_seeds': list(SEEDS[:4]), 'rejection_seeds': list(SEEDS), 'words_per_setting': WORDS, **res})
    SR.save_json(f'{SR.REPORT}-diagnosis-runtime.json', SR.runtime({'wall_seconds': time.perf_counter() - t0, 'settling_worker_cpu_seconds': sum(d[2] for d in done)}))
    print(json.dumps(res, indent=1))


def rej_task(seed):
    return {c: rejection(seed, c) for c in ('clean', 'degraded', 'held')}


if __name__ == '__main__':
    main()
