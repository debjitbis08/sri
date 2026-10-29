"""Standard shared-embedding network for the architecture audit (docs/architecture-audit.md, stage 1).

The number line's network without the SRI modules or energy: each sense's front end feeds one linear projection
(nn.Linear, standard initialization) into the shared 16-dim code. The relation heads are those of MultiSRI, with the
same parameters and initialization: successor S(c) = A c + t, the distance-based next/same judge and the linear larger
head. Revision is regularized least squares in code space with a scalar evidence weight; there is no energy.

StandardSolver stands in for the SRI solvers in the number-construction runners: it returns a joint-state-shaped
vector whose core-inner slice holds the code and whose other entries are zero, so the runners' training and evaluation
code is reused unchanged.
"""
import torch
from torch import nn
import torch.nn.functional as F

from src import number_systems_net as net

SIZES = net.SIZES
EVIDENCE_WEIGHT = 1.0  # scalar weight on ||c - c0||^2 in revision, against mu = 1 on each relational term


class StandardNet(nn.Module):
    def __init__(self, media, generator, sizes=SIZES):
        super().__init__()
        self.sizes, self.state = tuple(sizes), sum(sizes)
        o, m, i = self.sizes
        self.proj = nn.ModuleDict({r: nn.Linear(o, i) for r in media})
        self.A, self.t = nn.Parameter(torch.eye(i)), nn.Parameter(torch.zeros(i))
        self.larger = nn.Parameter(torch.randn(i, generator=generator) * .1)
        self.judge_bias, self.judge_scale = nn.Parameter(torch.ones(3)), nn.Parameter(torch.zeros(3))

    S = net.MultiSRI.S
    relation_logits = net.MultiSRI.relation_logits
    split = net.MultiSRI.split

    def inner_slice(self):
        o, m, i = self.sizes
        start = m + i + o + m  # same layout as MultiSRI's joint state without the clamped outer layer
        return slice(start, start + i)


class StandardSolver:
    """The SRI solvers' interface (solve, revise, energy) over a code computed by one linear projection."""

    def __init__(self, model, medium):
        self.model, self.medium = model, medium
        self.inner = model.inner_slice()
        self.width = 3 * model.state - model.sizes[0]

    def code(self, f):
        return self.model.proj[self.medium](f)

    def embed(self, c):
        n = len(c)
        return torch.cat([c.new_zeros(n, self.inner.start), c, c.new_zeros(n, self.width - self.inner.stop)], 1)

    def solve(self, f, kind='none', prev=None):
        if kind != 'none':
            raise ValueError('persistent-state solves are not defined for the standard network')
        return self.embed(self.code(f))

    def energy(self, f, y):
        return torch.zeros(len(f))  # no energy: the energy confidence feature is undefined

    def revise(self, f, mu, A, t, prev, nxt, key=None):
        """argmin_c w/2 ||c - c0||^2 + mu/2 ||c - (A prev + t)||^2 + mu/2 ||nxt - (A c + t)||^2 (rows without context: None)."""
        c0 = self.code(f)
        n, d = c0.shape
        has_p = torch.tensor([p is not None for p in prev]) if isinstance(prev, list) else torch.ones(n, dtype=torch.bool)
        has_n = torch.tensor([q is not None for q in nxt]) if isinstance(nxt, list) else torch.ones(n, dtype=torch.bool)
        P = torch.stack([p if p is not None else torch.zeros(d) for p in prev]) if isinstance(prev, list) else prev
        N = torch.stack([q if q is not None else torch.zeros(d) for q in nxt]) if isinstance(nxt, list) else nxt
        out = c0.clone()
        for hp in (False, True):
            for hn in (False, True):
                rows = (has_p == hp) & (has_n == hn)
                if not rows.any():
                    continue
                K = EVIDENCE_WEIGHT * torch.eye(d) + (mu * torch.eye(d) if hp else 0) + (mu * A.T @ A if hn else 0)
                b = EVIDENCE_WEIGHT * c0[rows]
                if hp:
                    b = b + mu * (P[rows] @ A.T + t)
                if hn:
                    b = b + mu * ((N[rows] - t) @ A)
                out[rows] = torch.linalg.solve(K, b.T).T
        return self.embed(out)


def total_energy(sv, f, y, mu, A, t, cm, h):
    """The revision objective at y (the standard counterpart of number_construction_b1_ambiguous.total_energy)."""
    c, c0 = y[:, sv.inner], sv.code(f)
    rel = c - (cm @ A.T + t) if h > 0 else cm - (c @ A.T + t)
    return .5 * EVIDENCE_WEIGHT * (c - c0).pow(2).sum(1) + .5 * mu * rel.pow(2).sum(1)


class LinearDecoder:
    """Imagery for the standard network: ridge regression from codes to front-end features, fitted on training inputs.
    A baseline for the SRI outward solve, not its equivalent (docs/architecture-audit.md, fact 4)."""

    def __init__(self, codes, feats, ridge=1e-3):
        X = torch.cat([codes, torch.ones(len(codes), 1)], 1).double()
        G = X.T @ X + ridge * len(X) * torch.eye(X.shape[1], dtype=torch.float64)
        self.W = torch.linalg.solve(G, X.T @ feats.double()).float()

    def __call__(self, c):
        return torch.cat([c, torch.ones(len(c), 1)], 1) @ self.W
