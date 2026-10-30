"""CPU vs GPU timing of a training step shaped like the sri sequence/fan-in pilot (standalone; imports nothing from the repo).

Shapes copied from src/number_core.py (ImageEncoder, MorseEncoder: two stride-2 convs, 1568|1024 -> 128 -> 64) and
src/generic_net.py (proj 64 -> 16, comparator g and predictor P: 64-wide two-hidden-layer MLPs). One step = encode a batch of
B images and B morse signals, project to the 16-dim codes, run g on code pairs and P on code pairs, MSE/BCE-style loss,
backward, Adam. float32, synthetic data (timing only; no scientific output).

Usage:
  bench.py single  --device cpu|cuda --threads N --batch B [--steps S]
  bench.py workers --device cpu|cuda --threads N --batch B --workers W [--steps S]   (W concurrent processes; aggregate rate)
Prints one JSON line.
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import time

import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 16, 3, 2, 1), nn.Conv2d(16, 32, 3, 2, 1)
        self.fc, self.out = nn.Linear(32 * 49, 128), nn.Linear(128, out)

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x[:, None]))))
        return self.out(F.relu(self.fc(h.flatten(1))))


class MorseEncoder(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(1, 16, 5, 2, 2), nn.Conv1d(16, 32, 5, 2, 2)
        self.fc, self.out = nn.Linear(32 * 32, 128), nn.Linear(128, out)

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x[:, None]))))
        return self.out(F.relu(self.fc(h.flatten(1))))


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.img, self.mor = ImageEncoder(), MorseEncoder()
        self.proj = nn.ModuleDict({'image': nn.Linear(64, 16), 'sequence': nn.Linear(64, 16)})
        self.token = nn.Embedding(4, 8)
        self.g, self.P = mlp(2 * 16 + 8, 1), mlp(2 * 16, 16)


def step_fn(model, opt, device, B, gen):
    xi = torch.rand(B, 28, 28, device=device, generator=gen)
    xm = torch.rand(B, 128, device=device, generator=gen)
    y = torch.rand(B, device=device, generator=gen).round()
    ci, cm = model.proj['image'](model.img(xi)), model.proj['sequence'](model.mor(xm))
    tok = model.token.weight[1].expand(B, -1)
    logit = model.g(torch.cat([ci, cm, tok], 1)).squeeze(1)
    pred = model.P(torch.cat([ci, cm], 1))
    loss = F.binary_cross_entropy_with_logits(logit, y) + F.mse_loss(pred, ci.detach())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def run_steps(device, B, steps, warm=20):
    torch.manual_seed(0)
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(0)
    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sync = (lambda: torch.cuda.synchronize()) if device == 'cuda' else (lambda: None)
    for _ in range(warm):
        step_fn(model, opt, dev, B, gen)
    sync()
    times = []
    for _ in range(steps):
        t = time.perf_counter()
        step_fn(model, opt, dev, B, gen)
        sync()
        times.append(time.perf_counter() - t)
    return times


def summarize(times):
    q = statistics.quantiles(times, n=4)
    return {'ms_median': 1e3 * statistics.median(times), 'ms_q1': 1e3 * q[0], 'ms_q3': 1e3 * q[2], 'steps_per_s': 1 / statistics.mean(times)}


def worker(args, barrier, out):
    torch.set_num_threads(args.threads)
    barrier.wait()
    t0 = time.perf_counter()
    times = run_steps(args.device, args.batch, args.steps)
    out.put((sum(times), time.perf_counter() - t0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['single', 'workers'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--batch', type=int, default=270)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    rec = {'mode': a.mode, 'device': a.device, 'threads': a.threads, 'batch': a.batch, 'steps': a.steps, 'torch': torch.__version__}
    if a.mode == 'single':
        torch.set_num_threads(a.threads)
        rec.update(summarize(run_steps(a.device, a.batch, a.steps)))
    else:
        ctx = mp.get_context('spawn')
        barrier, out = ctx.Barrier(a.workers), ctx.Queue()
        procs = [ctx.Process(target=worker, args=(a, barrier, out)) for _ in range(a.workers)]
        t0 = time.perf_counter()
        for pr in procs:
            pr.start()
        res = [out.get() for _ in procs]
        for pr in procs:
            pr.join()
        # aggregate rate: total steps over the slowest worker's timed loop
        slowest = max(r[0] for r in res)
        rec.update({'workers': a.workers, 'aggregate_steps_per_s': a.workers * a.steps / slowest,
                    'per_worker_ms': 1e3 * slowest / a.steps})
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
