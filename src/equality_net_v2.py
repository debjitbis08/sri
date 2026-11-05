"""Equality as a learned relation v2 (docs/equality-relations-v2.md).

Equality is one operator among others, learned from examples with positives and negatives; no property (symmetry,
substitution) is built into a loss. Operators and their true properties:
  eq     a = b        symmetric, reflexive
  more   b > a        antisymmetric, irreflexive
  sum10  a + b = 10   symmetric; introduced in block 3 on collections only
One readout S([pair feature, operator code]); pair feature G([c_a, c_b]) (two-space base) or [c_a, c_b] (count base).

Arms
  two-ctx-enbuf     English buffer replayed with labelled relations to collections (control)
  rel-en            English buffer in eq/more pairs with the current script, current script first only
  rel-en-both       as rel-en, both orders (upper bound for order generalization)
  two-ctx-enzhbuf   control for rel-en-zh: both buffers with labelled relations
  rel-en-zh         rel-en + Chinese buffer in eq/more pairs with Devanagari (Devanagari first only), block 3
  count-enbuf       count base, English buffer with count matching (control)
  count-rel-en      count base, as rel-en
"""
import torch
from torch import nn
import torch.nn.functional as F

from src import equality_net as EQ
from src import relation_codes as RC

ARMS = ('two-ctx-enbuf', 'rel-en', 'rel-en-both', 'two-ctx-enzhbuf', 'rel-en-zh', 'count-enbuf', 'count-rel-en')
OPS = ('eq', 'more', 'sum10')
SYMMETRIC = {'eq': True, 'more': False, 'sum10': True}
CROSS_OPS = (0, 1)  # operators trained on cross-script pairs: eq, more
N, D, OPD = RC.N, RC.D, 8


def base_arm(arm):
    return 'count' if arm.startswith('count') else 'two-ctx'


def buffer_modes(arm):
    """(English mode, Chinese mode); modes: None, 'label', 'one' (current script first), 'both'."""
    if arm in ('two-ctx-enbuf', 'count-enbuf'):
        return 'label', None
    if arm == 'two-ctx-enzhbuf':
        return 'label', 'label'
    if arm == 'rel-en-both':
        return 'both', None
    if arm == 'rel-en-zh':
        return 'one', 'one'
    return 'one', None


def op_targets(ya, yb):
    return torch.stack([(ya == yb).float(), (yb > ya).float(), ((ya + yb) == 10).float()], 1)


class RelNet(RC.RelationNet):
    def __init__(self, arm, width, generator):
        super().__init__(base_arm(arm), width, generator)
        self.variant = arm
        self.op_code = nn.Parameter(torch.randn(len(OPS), OPD, generator=generator) * .1)
        fin = D if self.arm in RC.PAIR else 2 * D
        self.S = nn.Sequential(nn.Linear(fin + OPD, 64), nn.ReLU(), nn.Linear(64, 1))

    def op_logits(self, ca, cb):
        f = self.G(torch.cat([ca, cb], 1)) if self.arm in RC.PAIR else torch.cat([ca, cb], 1)
        return torch.stack([self.S(torch.cat([f, self.op_code[k].expand(len(f), -1)], 1))[:, 0] for k in range(len(OPS))], 1)


def op_loss(model, ca, ya, cb, yb, ops):
    """Balanced BCE for the listed operators over all ordered pairs (ca[i], cb[j]); positives and negatives both used."""
    a, b, _ = RC.all_pairs(ca, ya, cb, yb)
    i = torch.arange(len(ca)).repeat_interleave(len(cb))
    j = torch.arange(len(cb)).repeat(len(ca))
    t = op_targets(ya[i], yb[j])[:, ops]
    pos = t.mean(0).clamp(1e-3, 1 - 1e-3)
    return F.binary_cross_entropy_with_logits(model.op_logits(a, b)[:, ops], t, pos_weight=(1 - pos) / pos)


label_loss = EQ.label_loss
