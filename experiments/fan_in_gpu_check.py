"""GPU against CPU check for the fan-in runner (docs/gpu-timing-fan-in-v1.md): the same RigL configuration and seeds
230-239, CPU run from the reset stage, GPU runs tagged -gputiming (230), -gputiming3 (231-233), -gputiming6 (234-239).

Throughput from the GPU runtime files; per-measure comparison of the last block's evaluation (mean over seeds, Welch z with
10 seeds each; measures are correlated, so the count of |z| > 2 is descriptive).

Usage: .venv/bin/python -m experiments.fan_in_gpu_check
"""
import gzip
import json

import numpy as np

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
CFG, SEEDS = 'read-q16-untied-rigl', range(230, 240)
TAGS = ('gputiming', 'gputiming3', 'gputiming6')


def results(suffix):
    return json.load(gzip.open(ROOT / 'reports' / f'fan-in-pilot-v1-pilot-reset{suffix}-results.json.gz', 'rt'))['results']


def flat(d, p=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + k + '/')
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield p + k, float(v)


def table(R):
    rows = [dict(flat(R[f'{CFG}/seed{s}']['after_block'][-1])) for s in SEEDS]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def main():
    for tag, w in zip(TAGS, (1, 3, 6)):
        rt = json.loads((ROOT / 'reports' / f'fan-in-pilot-v1-pilot-reset-{tag}-runtime.json').read_text())
        n = len(rt['tasks'])
        print(f'{w} GPU process(es): {n} run(s), runs wall {rt["phases"]["runs_wall_seconds"]:.0f} s, per run '
              f'{sorted({round(t["wall_seconds"]) for t in rt["tasks"].values()})} s, {rt["phases"]["runs_wall_seconds"] / n:.0f} s effective, '
              f'peak GPU {max(t["gpu_peak_allocated_bytes"] for t in rt["tasks"].values()) / 1e6:.0f} MB')
    cpu = json.loads((ROOT / 'reports' / 'fan-in-pilot-v1-pilot-reset-runtime.json').read_text())
    print(f'CPU reference (reset stage, {cpu["workers"]} workers x {cpu["threads_per_worker"]} threads): {cpu["tasks"][CFG + "/seed230"]["wall_seconds"]:.0f} s per run, '
          f'{cpu["phases"]["runs_wall_seconds"] / len(cpu["tasks"]):.0f} s effective')
    G = {}
    for tag in TAGS:
        G.update(results('-' + tag))
    tg, tc = table(G), table(results(''))
    z = {k: (tg[k].mean() - tc[k].mean()) / np.sqrt(tg[k].var(ddof=1) / 10 + tc[k].var(ddof=1) / 10)
         for k in tg if k in tc and tg[k].std() + tc[k].std() > 0}
    big = {k: v for k, v in z.items() if abs(v) > 2}
    print(f'{len(z)} measures: |z| > 2 in {len(big)} (about {0.05 * len(z):.0f} by chance), mean z {np.mean(list(z.values())):.2f}')
    for k, v in big.items():
        print(f'  {k}: CPU {tc[k].mean():.2f}, GPU {tg[k].mean():.2f}, z {v:.1f}')


if __name__ == '__main__':
    main()
