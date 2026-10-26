"""Exploratory design probe, not a registered experiment: can the local-consolidation
energy architecture reach the direct predictor when trained on whole-module
masked reconstruction, and does a local equilibrium-propagation rule track it?

Informed docs/local-contrastive-consolidation-v1-contract.md. Uses seeds 0-2 and
their v1 test sets, Adam, exact equilibrium solves; no cost receipt, writes no
artifacts. Run: .venv/bin/python -m experiments.local_contrastive_probe"""
import numpy as np
import torch
from experiments import local_consolidation as base

torch.set_default_dtype(torch.float64)
K = 0.2  # fixed chain coupling strength, as in v1


def energy_terms(s, W, V, T):
    """Per-example energy for states s (B,3,14). T[e] maps inner code of module e to e+1."""
    x, h, c = s[:, :, :8], s[:, :, 8:12], s[:, :, 12:]
    ex = x - torch.einsum('mij,bmj->bmi', W, h)
    eh = h - torch.einsum('mij,bmj->bmi', V, c)
    e = .5 * (ex ** 2).sum((1, 2)) + .5 * (eh ** 2).sum((1, 2)) + .025 * (h ** 2).sum((1, 2)) + .025 * (c ** 2).sum((1, 2))
    for a in range(2):
        d = c[:, a + 1] - c[:, a] @ T[a].T
        e = e + .5 * K * (d ** 2).sum(1)
    return e


def hessian(W, V, T):
    # Energy is quadratic in s: build Q by autograd on a basis (42 dims, cheap).
    s = torch.zeros(1, 42, requires_grad=True)
    g = torch.autograd.grad(energy_terms(s.view(1, 3, 14), W, V, T).sum(), s, create_graph=True)[0][0]
    return torch.stack([torch.autograd.grad(g[i], s, retain_graph=True, create_graph=True)[0][0] for i in range(42)])


def solve(Q, data, module, nudge=None, beta=0.):
    """Exact equilibrium with module's outer surface hidden; optional nudge toward target."""
    obs = [m * 14 + j for m in range(3) for j in range(8) if m != module]
    free = [i for i in range(42) if i not in obs]
    vals = data[:, [m for m in range(3) if m != module]].reshape(len(data), 16)
    A = Q[free][:, free].clone(); rhs = -(vals @ Q[free][:, obs].T)
    if beta:
        hid = [free.index(module * 14 + j) for j in range(8)]
        A[hid, hid] += beta; rhs[:, hid] += beta * nudge
    y = torch.linalg.solve(A, rhs.T).T
    s = torch.zeros(len(data), 42); s[:, obs] = vals; s[:, free] = y
    return s.view(-1, 3, 14)


def loss(params, data):
    Q = hessian(*params)
    return sum(((solve(Q, data, m)[:, m, :8] - data[:, m]) ** 2).mean() for m in range(3)) / 3


def eqprop_grad(params, data, beta=1e-3):
    """Local two-phase estimate: dE/dtheta at nudged minus free equilibrium, over beta.
    dE/dW_m touches only module m's x,h; dE/dT touches only the two inner codes."""
    Q = hessian(*params).detach()
    with torch.no_grad():
        phases = [(solve(Q, data, m), solve(Q, data, m, data[:, m], beta)) for m in range(3)]
    grads = [torch.zeros_like(p) for p in params]
    ps = [p if p.requires_grad else p.detach().requires_grad_() for p in params]
    for s0, sb in phases:
        g0 = torch.autograd.grad(energy_terms(s0, *ps).mean(), ps)
        gb = torch.autograd.grad(energy_terms(sb, *ps).mean(), ps)
        for i in range(3):
            grads[i] += (gb[i] - g0[i]) / beta / 3
    return grads


def init(seed, learn_T):
    p = base.initialize(seed)
    W = torch.tensor(p['W'], requires_grad=True); V = torch.tensor(p['V'], requires_grad=True)
    T = torch.eye(2).repeat(2, 1, 1).requires_grad_(learn_T)
    return [W, V, T]


def rmse_by_structure(params, test):
    Q = hessian(*params).detach()
    with torch.no_grad():
        e = torch.stack([((solve(Q, test, m)[:, m, :8] - test[:, m]) ** 2).mean(1) for m in range(3)]).mean(0)
    return [float(e.mean().sqrt()), float(e[:12].mean().sqrt()), float(e[12:].mean().sqrt())]


def train(params, stages, rule, steps, lr):
    learnable = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(learnable, lr=lr)
    for data in stages:
        for _ in range(steps):
            opt.zero_grad()
            if rule == 'exact':
                loss(params, data).backward()
            else:
                for p, g in zip(params, eqprop_grad(params, data)):
                    if p.requires_grad: p.grad = g
            opt.step()
    return params


def main():
    rows = {}
    for seed in (0, 1, 2):
        waking, test, _ = base.fixture(seed)
        A, B = (torch.tensor(w) for w in waking); test = torch.tensor(test)
        both = torch.cat([A, B])
        if seed == 0:  # gradient agreement check at initialization
            p = init(seed, True)
            exact = torch.autograd.grad(loss(p, both), p)
            est = eqprop_grad(p, both)
            cos = [float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)) for a, b in zip(exact, est)]
            print('eqprop vs exact gradient cosine (W,V,T):', [round(c, 6) for c in cos])
        r = {'frozen': rmse_by_structure(init(seed, False), test)}
        for name, learn_T, rule, stages in [
            ('exact_fixedT_joint', False, 'exact', [both]),
            ('exact_learnT_joint', True, 'exact', [both]),
            ('eqprop_fixedT_seq', False, 'eqprop', [A, both]),
            ('eqprop_learnT_seq', True, 'eqprop', [A, both]),
            ('eqprop_learnT_seq_Bonly', True, 'eqprop', [A, B]),
        ]:
            r[name] = rmse_by_structure(train(init(seed, learn_T), stages, rule, 300, .02), test)
        r['direct'] = base.direct_predictor(np.concatenate(waking), test.numpy())['rmse']
        rows[seed] = r
        print(seed, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in r.items()})
    names = rows[0].keys()
    print('\nmean over seeds [combined, A, B]:')
    for n in names:
        v = np.mean([rows[s][n] for s in rows], axis=0)
        print(f'  {n:26s}', np.round(v, 4))


if __name__ == '__main__':
    main()
