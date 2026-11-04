"""Common readout for the shared output study (docs/shared-output-v1-contract.md).

The readout converts a GenericSequenceNet (g, P, tokens, built and masked as sparse48) in place, so initial weights, RigL masks and every existing read
that calls relation_logits or predict stay as they are.

Arms:
  gp           g for pair relations, P with a distance logit for continuation (plus a learned bias b_c for acceptance); raw codes in every slot
  gp-grounded  the same readers; every relational slot (next, more, continuation) takes a grounded state
  common       one query encoder Q and one scorer s for every read, identification included; grounded slots
  common-raw   Q and s on raw codes (ablation)
For the common arms Q is g with its last layer 64 -> 16 (input [slot 1, slot 2, token], g's layout) and s is P with its last layer 64 -> 1 (input
[q, y], P's layout), so the masked first layers and the reader parameter count are unchanged.

Participants M: raw frozen codes of familiar collections, a buffer, re-read through the current projection and core. Identification is the `same` read
of an item against M; ground(x) = softmax(identification logits) @ M (or the top participant's state when `hard`). Gradient rule: the identification
weights are detached, so relational losses reach M's states but never identification. `same` reads raw items in every arm.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src import generic_net as gen

ARMS = ('gp', 'gp-grounded', 'common', 'common-raw')
GROUNDED = {'gp-grounded', 'common'}
COMMON = {'common', 'common-raw'}
SAME, NEXT, MORE = 1, 0, 2


class Readout(gen.GenericSequenceNet):
    """Use `convert`; not constructed directly."""

    # ---------------------------------------------------------------- pieces
    def _q(self, a, b, tok):
        return self.g(torch.cat([a, b, tok.expand(len(a), -1)], 1))

    def _s(self, q, y):
        return self.P(torch.cat([q, y], 1)).squeeze(1)

    def _pair(self, a, b, head):
        """Raw pair logit for aligned rows."""
        if self.arm in COMMON:
            return self._s(self._q(a, self.empty.expand(len(a), -1), self.token.weight[head]), b)
        return gen.GenericNet.relation_logits(self, a, b, head)

    # ---------------------------------------------------------------- participants and identification
    def participants(self):
        if self._M is None:
            with torch.no_grad():
                self._M = self.encode_participants(None)
        return self._M

    def encode_participants(self, solver):
        """Core codes of M's raw frozen codes through the current projection and core (with gradient when called with the update's solver)."""
        from experiments import number_construction_part_a as P  # the runner's solver; imported late so src stays importable alone
        solver = solver or P.Coupled(self)
        return solver.solve(self._enc['image'].from_codes(self.M_raw.cpu().numpy()))[:, P.INNER]

    def identify(self, x, M=None):
        """Identification logits (n, N) of items x against participants M, raw `same` reads."""
        M = self.participants() if M is None else M
        n, k, d = len(x), len(M), x.shape[1]
        if self.arm in COMMON:
            q = self._q(x, self.empty.expand(n, -1), self.token.weight[SAME])
            return self._s(q[:, None].expand(-1, k, -1).reshape(-1, d), M[None].expand(n, -1, -1).reshape(-1, d)).reshape(n, k)
        return gen.GenericNet.relation_logits(self, x[:, None].expand(-1, k, -1).reshape(-1, d), M[None].expand(n, -1, -1).reshape(-1, d), SAME).reshape(n, k)

    def ground(self, x):
        """The participant state x is identified with. Identification weights are detached (gradient rule)."""
        M = self.participants()
        u, inv = torch.unique(x.detach(), dim=0, return_inverse=True)
        lg = self.identify(u, M).detach()
        w = F.one_hot(lg.argmax(1), len(M)).to(M.dtype) if self.hard else lg.softmax(1)
        return (w @ M)[inv]

    # ---------------------------------------------------------------- reads used by training and every existing evaluation
    def relation_logits(self, ca, cb, head):
        if head != SAME and self.arm in GROUNDED:
            ca, cb = self.ground(ca), self.ground(cb)
        return self._pair(ca, cb, head)

    def cont_logits(self, prev, cur, cand):
        """Continuation logits (n, k) of candidates cand (n, k, d) after the history (prev, cur)."""
        n, k, d = cand.shape
        if self.arm in GROUNDED:
            prev, cur = self.ground(prev), self.ground(cur)
            cand = self.ground(cand.reshape(-1, d)).reshape(n, k, d)
        if self.arm in COMMON:
            q = self._q(prev, cur, self.cont)
            return self._s(q[:, None].expand(-1, k, -1).reshape(-1, d), cand.reshape(-1, d)).reshape(n, k)
        pred = self.P(torch.cat([prev, cur], 1))
        return -self.scale.exp() * (pred[:, None] - cand).pow(2).sum(-1) + self.b_c

    def predict(self, prev, cur):
        """The next item's state: P's generated code (gp arms), or the recalled participant (common arms), so composition reads roll either."""
        if self.arm in COMMON:
            M = self.participants()
            lg = self.cont_logits(prev, cur, M[None].expand(len(prev), -1, -1))
            w = F.one_hot(lg.argmax(1), len(M)).to(M.dtype) if self.hard else lg.softmax(1)
            return w @ M
        if self.arm in GROUNDED:
            prev, cur = self.ground(prev), self.ground(cur)
        return self.P(torch.cat([prev, cur], 1))


def convert(model, arm, enc, raw_participants, generator_seed):
    """Turn a sparse48 GenericSequenceNet into the arm's readout in place. New parameters draw from their own seeded generator, so the network's
    other initial weights are those of every arm."""
    assert arm in ARMS
    model.__class__ = Readout
    model.arm, model.hard, model._M = arm, False, None
    object.__setattr__(model, '_enc', enc)  # a plain reference, not a submodule
    model.register_buffer('M_raw', torch.as_tensor(raw_participants, dtype=torch.float32).clone())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(generator_seed)
        if arm in COMMON:
            h, i = model.g[-1].in_features, model.sizes[2]
            model.g[-1] = nn.Linear(h, i)  # Q: 64 -> 16
            model.P[-1] = nn.Linear(h, 1)  # s: 64 -> 1
            model.cont = nn.Parameter(torch.randn(gen.TOKEN))  # the continuation token, drawn like the relation tokens
            model.empty = nn.Parameter(torch.zeros(i))  # the empty slot
        else:
            model.b_c = nn.Parameter(torch.zeros(()))
    return model


def acceptance(logits, target):
    """Binary cross-entropy on ranked logits (n, k): the target column positive, the k-1 others negative, weighted so both classes count equally."""
    y = F.one_hot(target, logits.shape[1]).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(float(logits.shape[1] - 1), device=logits.device))


def reader_parameters(model):
    """Reader parameters: g/Q, P/s, tokens and the arm's additions (not scale, which only gp's distance logit uses)."""
    n = sum(p.numel() for p in (*model.g.parameters(), *model.P.parameters(), model.token.weight))
    for name in ('cont', 'empty', 'b_c'):
        if hasattr(model, name):
            n += getattr(model, name).numel()
    if model.arm not in COMMON:
        n += model.scale.numel()
    return int(n)


def threshold(pos, neg):
    """The threshold maximizing balanced accuracy (accept when logit >= t) over positive and negative logits; ties take the lowest such t."""
    import numpy as np
    pos, neg = np.sort(np.asarray(pos, np.float64)), np.sort(np.asarray(neg, np.float64))
    cand = np.unique(np.concatenate([pos, neg, [np.inf]]))
    tpr = 1 - np.searchsorted(pos, cand, 'left') / len(pos)
    tnr = np.searchsorted(neg, cand, 'left') / len(neg)
    bal = np.round((tpr + tnr) / 2, 12)  # so exact ties are ties
    j = int(np.argmax(bal))
    return float(cand[j]), float(bal[j])
