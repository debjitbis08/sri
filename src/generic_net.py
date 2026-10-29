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


class GenericSequenceNet(GenericNet):
    """GenericNet plus one next-item predictor P(c_prev, c_cur): a two-layer MLP on the codes of the two most recent items
    of a sequence, returning the predicted code of the next item. No action or relation tokens: the direction of a sequence
    is inferred from its history. `scale` is the learned inverse temperature of the prediction loss."""

    def __init__(self, media, generator=None, sizes=SIZES):
        super().__init__(media, generator, sizes)
        i = self.sizes[2]
        self.P = nn.Sequential(nn.Linear(2 * i, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, i))
        self.scale = nn.Parameter(torch.zeros(()))

    def predict(self, prev, cur):
        return self.P(torch.cat([prev, cur], 1))


ROLES = ('query', 'candidate', 'first', 'second')
THINK_HIDDEN, THINK_STEPS = 32, 4


class ThinkingSequenceNet(GenericSequenceNet):
    """GenericSequenceNet whose answers come from a recurrent relational loop (Palm, Paquet and Winther 2018) instead of
    the one-pass comparator g (kept as a parameter only so the other parameters start as in the sequence pilot).

    A question is a set of nodes: for 'same' (matching), the query and its 9 candidates; for 'next' and 'more', the two
    items. Each node starts from its code, its role and the question token. For K steps every node sums messages
    f(h_i, h_j) from the others and updates its state with a GRU cell; an answer logit is read from every node at every
    step (candidates for matching, the second item otherwise). K is the thinking effort (`steps`, settable at test).

    relation_logits keeps the pairwise signature used by the runners: for 'same', rows come in groups of 9 with one
    repeated query, and each group is answered as one set. relation_steps returns the logits of every step.
    """

    def __init__(self, media, generator=None, sizes=SIZES):
        super().__init__(media, generator, sizes)
        i, H = self.sizes[2], THINK_HIDDEN
        self.role = nn.Embedding(len(ROLES), TOKEN)
        self.inp = nn.Linear(i + 2 * TOKEN, H)
        self.msg = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, H))
        self.cell = nn.GRUCell(2 * H, H)
        self.out = nn.Linear(H, 1)
        self.steps = THINK_STEPS

    def think(self, codes, roles, head):
        """codes (B, N, d), roles (N,) role indices -> list over steps of (B, N) logits."""
        B, N, _ = codes.shape
        x = self.inp(torch.cat([codes, self.role.weight[roles][None].expand(B, -1, -1),
                                self.token.weight[head][None, None].expand(B, N, -1)], -1))
        h, outs = x, []
        off = 1 - torch.eye(N)[None, :, :, None]
        for _ in range(self.steps):
            pair = torch.cat([h[:, :, None].expand(-1, -1, N, -1), h[:, None].expand(-1, N, -1, -1)], -1)
            m = (self.msg(pair) * off).sum(2)
            h = self.cell(torch.cat([x, m], -1).reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1)
            outs.append(self.out(h).squeeze(-1))
        return outs

    def relation_steps(self, ca, cb, head):
        d = ca.shape[1]
        if head == RELATIONS.index('same'):
            if len(ca) % 9:
                raise ValueError('matching expects groups of 9 candidates per query')
            q = ca.reshape(-1, 9, d)
            if not torch.equal(q, q[:, :1].expand_as(q)):
                raise ValueError('matching expects one repeated query per group of 9')
            codes = torch.cat([q[:, :1], cb.reshape(-1, 9, d)], 1)
            roles = torch.tensor([ROLES.index('query')] + [ROLES.index('candidate')] * 9)
            return [o[:, 1:].reshape(-1) for o in self.think(codes, roles, head)]
        roles = torch.tensor([ROLES.index('first'), ROLES.index('second')])
        return [o[:, 1] for o in self.think(torch.stack([ca, cb], 1), roles, head)]

    def relation_logits(self, ca, cb, head):
        return self.relation_steps(ca, cb, head)[-1]
