"""Standard network without relation-specific heads (docs/generic-relations-v1-spec.md).

Perception is the standard network's: each sense's front end feeds one linear projection into the shared 16-dim code.
The relation heads (affine successor, distance judge, linear order vector) are replaced by one generic comparator: a
two-layer MLP g(a, b, r) on two codes and a learned relation token r. 'same' (matching), 'next' and 'more' are tokens,
not parameters with their own form. Every answer is read from g's output; nothing is computed from code distances, and
there is no transition map to iterate.

relation_logits keeps MultiSRI's signature (head 0 next, 1 same, 2 more), so the runners' training and evaluation code
that calls it works unchanged. Calls to S, A, t or larger fail: those mechanisms no longer exist.
"""
import torch
from torch import nn

from src import number_systems_net as net
from src import standard_net as std

SIZES = std.SIZES
RELATIONS = ('next', 'same', 'more')  # index = MultiSRI head index
TOKEN, HIDDEN = 8, 64


class GenericNet(nn.Module):
    def __init__(self, media, generator=None, sizes=SIZES):
        super().__init__()
        self.sizes, self.state = tuple(sizes), sum(sizes)
        o, m, i = self.sizes
        self.proj = nn.ModuleDict({r: nn.Linear(o, i) for r in media})
        self.token = nn.Embedding(len(RELATIONS), TOKEN)
        self.g = nn.Sequential(nn.Linear(2 * i + TOKEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    split = std.StandardNet.split
    inner_slice = std.StandardNet.inner_slice

    def relation_logits(self, ca, cb, head):
        r = self.token.weight[head].expand(len(ca), -1)
        return self.g(torch.cat([ca, cb, r], 1)).squeeze(1)


class GenericSRI(net.MultiSRI):
    """The SRI network with its relation heads replaced by the same generic comparator on the core-inner code.

    The SRI modules, translations and energy are MultiSRI's, initialized identically (the heads' parameters are still
    created, so the generator draws match, but are unused and receive no gradient). The comparator draws its initial
    values from the global generator after the encoders.
    """

    def __init__(self, media, generator, sizes=SIZES):
        super().__init__(media, generator, sizes)
        i = self.sizes[2]
        self.token = nn.Embedding(len(RELATIONS), TOKEN)
        self.g = nn.Sequential(nn.Linear(2 * i + TOKEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    relation_logits = GenericNet.relation_logits

    def S(self, c):
        raise AttributeError('GenericSRI has no successor map')


ACTIONS = ('add', 'remove')


class GenericTransitionNet(GenericNet):
    """GenericNet plus one action-conditioned transition network T(c, action): a two-layer MLP on a code and a learned
    action token, returning a predicted code. Trained to predict the next item's code (a world model); the actions are
    tokens, not parameters with their own form. `scale` is the learned inverse temperature of the prediction loss."""

    def __init__(self, media, generator=None, sizes=SIZES):
        super().__init__(media, generator, sizes)
        i = self.sizes[2]
        self.action = nn.Embedding(len(ACTIONS), TOKEN)
        self.T = nn.Sequential(nn.Linear(i + TOKEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, i))
        self.scale = nn.Parameter(torch.zeros(()))

    def transition(self, c, action):
        return self.T(torch.cat([c, self.action.weight[ACTIONS.index(action)].expand(len(c), -1)], 1))
