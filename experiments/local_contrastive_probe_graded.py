"""Exploratory design probe: inner-only versus graded (all-layer) symmetric coupling
with learned translations, trained by the same two-phase local rule as
local_contrastive_probe. Same scope and caveats; no cost receipt, no artifacts.
Run: .venv/bin/python -m experiments.local_contrastive_probe_graded"""
import numpy as np
import torch
from experiments import local_contrastive_probe as probe
from experiments import local_consolidation as base

LAYERS = ((0, 8), (8, 12), (12, 14))  # outer, middle, inner slices


def make_energy(kappa):
    """kappa: per-unit coupling strength for (outer, middle, inner); T[edge][layer] translations."""
    def energy_terms(s, W, V, T):
        x, h, c = s[:, :, :8], s[:, :, 8:12], s[:, :, 12:]
        ex = x - torch.einsum('mij,bmj->bmi', W, h)
        eh = h - torch.einsum('mij,bmj->bmi', V, c)
        e = .5 * (ex ** 2).sum((1, 2)) + .5 * (eh ** 2).sum((1, 2)) + .025 * (h ** 2).sum((1, 2)) + .025 * (c ** 2).sum((1, 2))
        for a in range(2):
            for layer, (lo, hi) in enumerate(LAYERS):
                if kappa[layer]:
                    d = s[:, a + 1, lo:hi] - s[:, a, lo:hi] @ T[a][layer].T
                    e = e + .5 * kappa[layer] * (d ** 2).sum(1)
        return e
    return energy_terms


def init(seed):
    p = base.initialize(seed)
    return [torch.tensor(p['W'], requires_grad=True), torch.tensor(p['V'], requires_grad=True)] + \
        [torch.eye(n).requires_grad_() for _ in range(2) for n in (8, 4, 2)]


def unflat(ts):
    return [ts[0], ts[1], [ts[2:5], ts[5:8]]]


def hessian(E, leaves):
    z = torch.zeros(1, 42, requires_grad=True)
    g = torch.autograd.grad(E(z.view(1, 3, 14), *unflat(leaves)).sum(), z, create_graph=True)[0][0]
    return torch.stack([torch.autograd.grad(g[i], z, retain_graph=True)[0][0] for i in range(42)]).detach()


def eqprop_grad(E, leaves, data, beta=1e-3):
    Q = hessian(E, leaves)
    with torch.no_grad():
        phases = [(probe.solve(Q, data, m), probe.solve(Q, data, m, data[:, m], beta)) for m in range(3)]
    grads = [torch.zeros_like(p) for p in leaves]
    for s0, sb in phases:
        g0 = torch.autograd.grad(E(s0, *unflat(leaves)).mean(), leaves, allow_unused=True, materialize_grads=True)
        gb = torch.autograd.grad(E(sb, *unflat(leaves)).mean(), leaves, allow_unused=True, materialize_grads=True)
        for i in range(len(leaves)):
            grads[i] += (gb[i] - g0[i]) / beta / 3
    return grads


def rmse(E, leaves, test):
    Q = hessian(E, leaves)
    with torch.no_grad():
        e = torch.stack([((probe.solve(Q, test, m)[:, m, :8] - test[:, m]) ** 2).mean(1) for m in range(3)]).mean(0)
    return [float(e.mean().sqrt()), float(e[:12].mean().sqrt()), float(e[12:].mean().sqrt())]


def run(kappa, seed, stages, test, steps=300, lr=.02):
    E = make_energy(kappa)
    leaves = init(seed)
    opt = torch.optim.Adam(leaves, lr=lr)
    before = rmse(E, leaves, test)
    for data in stages:
        for _ in range(steps):
            for p, g in zip(leaves, eqprop_grad(E, leaves, data)):
                p.grad = g
            opt.step()
    return before, rmse(E, leaves, test)


def main():
    inward = np.array([.05, .2, 1]) * 2 / (8 * .05 + 4 * .2 + 2)
    configs = {'inner_only_v1_strength': (0, 0, .2),
               'inner_only_budget2': (0, 0, 1.),
               'uniform_per_unit_budget2': (1 / 7, 1 / 7, 1 / 7),
               'inward_graded_budget2': tuple(inward)}
    out = {}
    for seed in (0, 1, 2):
        waking, test, _ = base.fixture(seed)
        A, B = (torch.tensor(w) for w in waking)
        test = torch.tensor(test)
        for name, kappa in configs.items():
            before, after = run(kappa, seed, [A, torch.cat([A, B])], test)
            out.setdefault(name + ' (frozen)', []).append(before)
            out.setdefault(name, []).append(after)
            print(seed, name, np.round(before, 4), '->', np.round(after, 4), flush=True)
    print('\nmean over seeds [combined, A, B]:')
    for name, rows in out.items():
        print(f'  {name:36s}', np.round(np.mean(rows, axis=0), 4))


if __name__ == '__main__':
    main()
