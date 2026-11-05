"""Equality as substitutability v1 (docs/equality-substitution-v1.md): the Part C network plus two operators learned on
collections only, and a substitution (Leibniz) loss for cross-language equality pairs.

Arms
  count             Part C `count`
  count-enbuf       + English buffer replayed with count matching (control for eq-en)
  eq-en             + English buffer replayed only in substitution pairs with the current script
  two-ctx           Part C `two-ctx`
  two-ctx-enbuf     + English buffer replayed with labelled relations (control for two-ctx-eq-en)
  two-ctx-eq-en     + English buffer replayed only in substitution pairs
  count-enzhbuf     count-enbuf + Chinese buffer replayed in block 3 with count matching (control for eq-en-zh)
  eq-en-zh          eq-en + Chinese buffer replayed in block 3 only in substitution pairs with Devanagari
Operators (block 3, collections only): `more` (b > a) and `sum10` (a + b = 10), one shared readout with learned operator codes.
"""
import torch
from torch import nn
import torch.nn.functional as F

from src import relation_codes as RC

ARMS = ('count', 'count-enbuf', 'eq-en', 'two-ctx', 'two-ctx-enbuf', 'two-ctx-eq-en', 'count-enzhbuf', 'eq-en-zh')
OPS = ('more', 'sum10')
N, D = RC.N, RC.D
OPD = 8


def base_arm(arm):
    return 'two-ctx' if arm.startswith('two-ctx') else 'count'


def en_mode(arm):
    """How the English buffer is used: None, 'label' or 'eq'."""
    return 'eq' if 'eq-en' in arm else ('label' if 'enbuf' in arm or 'enzhbuf' in arm else None)


def zh_mode(arm):
    return 'eq' if arm == 'eq-en-zh' else ('label' if arm == 'count-enzhbuf' else None)


def op_targets(ya, yb):
    return torch.stack([(yb > ya).float(), ((ya + yb) == 10).float()], 1)


class EqNet(RC.RelationNet):
    def __init__(self, arm, width, generator):
        super().__init__(base_arm(arm), width, generator)
        self.variant = arm
        self.op_code = nn.Parameter(torch.randn(len(OPS), OPD, generator=generator) * .1)
        fin = D if self.arm in RC.PAIR else 2 * D
        self.S = nn.Sequential(nn.Linear(fin + OPD, 64), nn.ReLU(), nn.Linear(64, 1))

    def op_logits(self, ca, cb):
        """(n, len(OPS)) logits: one readout, operator given by its code."""
        f = self.G(torch.cat([ca, cb], 1)) if self.arm in RC.PAIR else torch.cat([ca, cb], 1)
        return torch.stack([self.S(torch.cat([f, self.op_code[k].expand(len(f), -1)], 1))[:, 0] for k in range(len(OPS))], 1)


def match_loss(model, c_coll, y_coll, c, y):
    """Count matching of items c to collection-count means (count arms)."""
    mu = torch.stack([c_coll[y_coll == n].mean(0) for n in range(1, N + 1)])
    lg = -F.softplus(model.log_scale + 1.0) * (c[:, None] - mu[None]).pow(2).sum(2)
    return F.cross_entropy(lg, y - 1)


def labelled_rel_loss(model, c_coll, y_coll, c, y):
    """Labelled relations of items c to every collection, both directions (relation arms)."""
    terms = []
    for ca, ya, cb, yb in ((c, y, c_coll, y_coll), (c_coll, y_coll, c, y)):
        a, b, d = RC.all_pairs(ca, ya, cb, yb)
        terms.append(F.cross_entropy(model.rel_logits(a, b), d))
    return sum(terms) / 2


def label_loss(model, c_coll, y_coll, c, y):
    return labelled_rel_loss(model, c_coll, y_coll, c, y) if model.arm in RC.PAIR else match_loss(model, c_coll, y_coll, c, y)


def _sym_kl(la, lb):
    pa, pb = la.log_softmax(-1), lb.log_softmax(-1)
    return .5 * ((pa.exp() * (pa - pb)).sum(-1) + (pb.exp() * (pb - pa)).sum(-1)).mean()


def substitution_loss(model, c_coll, ca, cb, with_ops):
    """Leibniz: for equal pairs (ca[i], cb[i]), every operator present gives the same output with every collection, in both
    argument positions. No count label is used here; equality of the pair is the only supervision."""
    n, k = len(ca), len(c_coll)
    A, B = ca.repeat_interleave(k, 0), cb.repeat_interleave(k, 0)
    M = c_coll.repeat(n, 1)
    L = _sym_kl(model.rel_logits(A, M), model.rel_logits(B, M)) + _sym_kl(model.rel_logits(M, A), model.rel_logits(M, B))
    if with_ops:
        two = lambda o: torch.stack([F.logsigmoid(-o), F.logsigmoid(o)], -1)
        L = L + _sym_kl(two(model.op_logits(A, M)), two(model.op_logits(B, M))) + _sym_kl(two(model.op_logits(M, A)), two(model.op_logits(M, B)))
    return L


def op_loss(model, c_coll, y_coll):
    a, b, _ = RC.all_pairs(c_coll, y_coll, c_coll, y_coll)
    i = torch.arange(len(c_coll)).repeat_interleave(len(c_coll))
    j = torch.arange(len(c_coll)).repeat(len(c_coll))
    t = op_targets(y_coll[i], y_coll[j])
    pos = t.mean(0).clamp_min(1e-3)
    return F.binary_cross_entropy_with_logits(model.op_logits(a, b), t, pos_weight=(1 - pos) / pos)


def same_count_pairs(ya, yb):
    """All index pairs (i, j) with ya[i] == yb[j]."""
    return (ya[:, None] == yb[None, :]).nonzero(as_tuple=True)
