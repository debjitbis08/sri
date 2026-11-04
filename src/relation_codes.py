"""Relation codes pilot (docs/relation-codes-check-v1.md, part C): object encoder plus relation codes in one or two spaces.

Arms
  count, count-ctx   forms taught by `same` against collection-count means; collections by offset relations
  one                relations as offsets r_d in object space: score -|c_b - c_a - r_d|^2 (TransE)
  two                relation space q = G([c_a, c_b]); score -|q - r_d|^2
  two-ctx            as two, on codes context-normalized within each kind
  two-frozen         as two, G and r_d frozen after block 1
Relations are differences d = y - x in -8..8 (17 codes).
"""
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ('count', 'count-ctx', 'one', 'two', 'two-ctx', 'two-frozen')
CTX = {'count-ctx', 'two-ctx'}
PAIR = {'two', 'two-ctx', 'two-frozen'}
COUNT = {'count', 'count-ctx'}
D = 16
N = 9
ND = 2 * N - 1


def context_norm(c, groups):
    """Standardize rows of c within each group label (one kind = one context)."""
    out = torch.empty_like(c)
    for g in torch.unique(groups):
        m = groups == g
        x = c[m]
        out[m] = (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-5)
    return out


class RelationNet(nn.Module):
    def __init__(self, arm, width, generator):
        super().__init__()
        self.arm = arm
        self.E = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, D))
        self.r = nn.Parameter(torch.randn(ND, D, generator=generator) * .1)
        self.log_scale = nn.Parameter(torch.zeros(()))
        if arm in PAIR:
            self.G = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU(), nn.Linear(64, D))

    def codes(self, x, groups):
        c = self.E(x)
        return context_norm(c, groups) if self.arm in CTX else c

    def rel_logits(self, ca, cb):
        """(n, 17) logits over d for aligned pairs."""
        q = self.G(torch.cat([ca, cb], 1)) if self.arm in PAIR else cb - ca
        return -F.softplus(self.log_scale + 1.0) * (q[:, None] - self.r[None]).pow(2).sum(2)

    def freeze_structure(self):
        if self.arm == 'two-frozen':
            for p in (*self.G.parameters(), self.r):
                p.requires_grad_(False)


def all_pairs(ca, ya, cb, yb):
    i = torch.arange(len(ca)).repeat_interleave(len(cb))
    j = torch.arange(len(cb)).repeat(len(ca))
    return ca[i], cb[j], (yb[j] - ya[i]) + N - 1


def loss(model, c_coll, y_coll, c_form, y_form):
    """Collections: relations among collections. Forms: count arms match to collection-count means; relation arms learn relations
    to every collection and among the forms."""
    a, b, d = all_pairs(c_coll, y_coll, c_coll, y_coll)
    L = F.cross_entropy(model.rel_logits(a, b), d)
    if c_form is None:
        return L
    if model.arm in COUNT:
        mu = torch.stack([c_coll[y_coll == n].mean(0) for n in range(1, N + 1)])
        lg = -F.softplus(model.log_scale + 1.0) * (c_form[:, None] - mu[None]).pow(2).sum(2)
        return L + F.cross_entropy(lg, y_form - 1)
    terms = []
    for (ca, ya, cb, yb) in ((c_form, y_form, c_coll, y_coll), (c_coll, y_coll, c_form, y_form), (c_form, y_form, c_form, y_form)):
        a, b, d = all_pairs(ca, ya, cb, yb)
        terms.append(F.cross_entropy(model.rel_logits(a, b), d))
    return L + sum(terms) / len(terms)


@torch.no_grad()
def relational_identity(model, c, mu):
    """argmax_n sum_m log p(d = m - n | c, mu_m), mu: (9, D) collection-count means."""
    n = len(c)
    lp = torch.log_softmax(model.rel_logits(c.repeat_interleave(N, 0), mu.repeat(n, 1)), 1).reshape(n, N, ND)
    cand = torch.arange(1, N + 1)
    m = torch.arange(1, N + 1)
    idx = (m[None, :] - cand[:, None]) + N - 1  # (cand, m)
    score = torch.stack([lp[:, torch.arange(N), idx[k]].sum(1) for k in range(N)], 1)
    return score.argmax(1) + 1
