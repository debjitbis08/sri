"""Consistency critic with learned probe selection v1 (docs/consistency-critic-v1.md).

The critic sees only operator outputs, never item codes. A query is an operator q on an ordered pair (a, b). The evidence is
a set of probe units. Each unit gives four binary slots (value +-1 where present, 0 where absent) plus a mask:
  same-pair unit for operator r:      [r(a,b), r(b,a), -, -]   (the query slot r(a,b) is masked when r == q)
  context unit for operator r and m:  [r(a,m), r(m,a), r(b,m), r(m,b)]
The critic is masked conditional prediction: from any subset of units it predicts the query's teaching label. It is a
DeepSets network (token MLP, mean pool, head). It contains no rule; the dependencies are learned from collection examples.

The selector scores units from the query alone, before their values are computed. Scoring is query-key matching with learned
projections: one query vector per (operator, unit type) from [c_a, c_b, q], dotted with a key K(c_m) for context units (K is
precomputable per memory item) or a learned key per operator for same-pair units, plus a bias per (q, r, type).
"""
import torch
from torch import nn
import torch.nn.functional as F

SLOTS, H, DK = 4, 64, 16


class Critic(nn.Module):
    def __init__(self, n_ops):
        super().__init__()
        self.er, self.et, self.eq = nn.Embedding(n_ops, 8), nn.Embedding(2, 4), nn.Embedding(n_ops, 8)
        self.tok = nn.Sequential(nn.Linear(2 * SLOTS + 20, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(H + n_ops + 1, H), nn.ReLU(), nn.Linear(H, 1))
        self.n_ops = n_ops

    def forward(self, q, vals, mask, r, typ, present):
        """q (Q,), vals/mask (Q,U,4), r/typ (U,) or (Q,U), present (Q,U) bool -> logit (Q,)."""
        Q, U = present.shape
        r, typ = r.expand(Q, U), typ.expand(Q, U)
        x = torch.cat([vals * mask, mask, self.er(r), self.et(typ), self.eq(q)[:, None].expand(Q, U, -1)], 2)
        h = self.tok(x) * present[..., None]
        n = present.sum(1, keepdim=True).float()
        pooled = h.sum(1) / n.clamp_min(1)
        return self.head(torch.cat([pooled, F.one_hot(q, self.n_ops).float(), n / U], 1))[:, 0]


class Selector(nn.Module):
    def __init__(self, n_ops, d_code):
        super().__init__()
        self.n_ops = n_ops
        self.Q = nn.Sequential(nn.Linear(2 * d_code + n_ops, H), nn.ReLU(), nn.Linear(H, n_ops * 2 * DK))
        self.K = nn.Sequential(nn.Linear(d_code, H), nn.ReLU(), nn.Linear(H, DK))
        self.k_same = nn.Parameter(torch.randn(n_ops, DK) * .1)
        self.bias = nn.Parameter(torch.zeros(n_ops, n_ops, 2))

    def forward(self, q, ca, cb, cm, r, typ):
        """q (Q,), ca/cb (Q,d), cm (Q,K,d) context codes; r/typ (U,) unit operator and type; units ordered as
        [same-pair r = 0..R-1, then context (r, m) with r major, m minor] -> scores (Q,U)."""
        Qn = len(q)
        qv = self.Q(torch.cat([ca, cb, F.one_hot(q, self.n_ops).float()], 1)).view(Qn, self.n_ops, 2, DK)
        same = (qv[:, :, 0] * self.k_same[None]).sum(2)  # (Q,R)
        km = self.K(cm)  # (Q,K,DK)
        ctx = torch.einsum('qrd,qkd->qrk', qv[:, :, 1], km).reshape(Qn, -1)  # (Q, R*K), r major
        return torch.cat([same, ctx], 1) + self.bias[q][:, r, typ]
