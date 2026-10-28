"""Multi-representation number core network (docs/number-systems-v1-spec.md, draft).

One satellite per medium: a front end (image CNN or sequence 1-D CNN, from src/number_core.py)
and one SRI input module, coupled to a shared core at corresponding layers through learned
translations; the core couples to an output module with fixed label codes. Every representation
of a medium goes through that medium's satellite, including held-out ones. Each example clamps
its satellite's outer layer and solves the energy over that module's hidden layers, the core and
the output.
"""
import torch
from torch import nn
import torch.nn.functional as F

from src import number_core as ncore

SIZES = ncore.WIDTHS['wide']


class MultiSRI(nn.Module):
    def __init__(self, media, generator, sizes=SIZES, kappa=ncore.TOPOLOGIES['uniform']):
        super().__init__()
        self.sizes, self.state, self.kappa, self.gen = tuple(sizes), sum(sizes), kappa, generator
        o, m, i = self.sizes
        self.W, self.V, self.T = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        for k in ('core', 'output'):
            self._module(k)
        for l, n in enumerate(self.sizes):
            self.T[f'core-output-{l}'] = nn.Parameter(torch.eye(n))
        for r in media:
            self.add_satellite(r)
        self.A, self.t = nn.Parameter(torch.eye(i)), nn.Parameter(torch.zeros(i))
        self.larger = nn.Parameter(torch.randn(i, generator=generator) * .1)
        self.judge_bias, self.judge_scale = nn.Parameter(torch.ones(3)), nn.Parameter(torch.zeros(3))

    def _module(self, k):
        o, m, i = self.sizes
        self.W[k] = nn.Parameter(torch.randn(o, m, generator=self.gen) * .25)
        self.V[k] = nn.Parameter(torch.randn(m, i, generator=self.gen) * .25)

    def add_satellite(self, r):
        self._module(r)
        for l, n in enumerate(self.sizes):
            self.T[f'{r}-core-{l}'] = nn.Parameter(torch.eye(n))

    def satellite_parameters(self, r):
        return [self.W[r], self.V[r]] + [self.T[f'{r}-core-{l}'] for l in range(3)]

    def S(self, c):
        return c @ self.A.T + self.t

    def hessian(self, r):
        order, STATE = (r, 'core', 'output'), self.state
        Q = torch.zeros(3 * STATE, 3 * STATE)
        o, m, i = self.sizes
        for j, k in enumerate(order):
            b = j * STATE
            x, h, c = slice(b, b + o), slice(b + o, b + o + m), slice(b + o + m, b + STATE)
            W, V = self.W[k], self.V[k]
            Q[x, x] += torch.eye(o) * (1 + (ncore.LAMBDA if k == 'core' else 0))
            Q[x, h] -= W; Q[h, x] -= W.T
            Q[h, h] += W.T @ W + (1 + ncore.LAMBDA) * torch.eye(m)
            Q[h, c] -= V; Q[c, h] -= V.T
            Q[c, c] += V.T @ V + ncore.LAMBDA * torch.eye(i)
        for a, b, key in ((r, 'core', f'{r}-core'), ('core', 'output', 'core-output')):
            ja, jb = order.index(a), order.index(b)
            off = 0
            for l, n in enumerate(self.sizes):
                T, k = self.T[f'{key}-{l}'], self.kappa[l]
                sa, sb = slice(ja * STATE + off, ja * STATE + off + n), slice(jb * STATE + off, jb * STATE + off + n)
                Q[sb, sb] += k * torch.eye(n)
                Q[sa, sa] += k * T.T @ T
                Q[sb, sa] -= k * T
                Q[sa, sb] -= k * T.T
                off += n
        return Q

    def equilibrium_map(self, r):
        Q = self.hessian(r)
        o = self.sizes[0]
        L = torch.linalg.cholesky(Q[o:, o:])
        return -torch.cholesky_solve(Q[o:, :o], L), Q

    def split(self, y):
        o, m, i = self.sizes
        core = y[:, m + i:m + i + self.state]
        return {'core': core, 'core_inner': core[:, o + m:], 'output_outer': y[:, m + i + self.state:][:, :o]}

    def relation_logits(self, ca, cb, head):
        """head 0: next (B = A + 1), 1: same, 2: larger (A > B)."""
        if head == 2:
            return self.judge_bias[2] + F.softplus(self.judge_scale[2]) * ((ca - cb) @ self.larger)
        d = (cb - (self.S(ca) if head == 0 else ca)).pow(2).sum(1)
        return self.judge_bias[head] - F.softplus(self.judge_scale[head]) * d
