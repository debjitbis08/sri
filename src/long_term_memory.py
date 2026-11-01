"""Long-term memory v1 (docs/long-term-memory-v1-contract.md): a memory network beside sparse readers.

score(q, c) = reader(q, c) + s <M(W_K key(q)), W_V value(c)>. Memory trains on the readers' errors, from detached inputs, and cannot reach the host.
Everything here runs on cached arrays, in its own process, so nothing in the host depends on it.
"""
import math

import torch
import torch.nn.functional as F

D = 32
BUDGET = 64 * 1024
CHARGED_COPIES = 3     # weights, momentum, accumulated gradient: charged for every setting, so a setting cannot change capacity
MAX_IN = 32
CLIP = 5.
FORMS = ('linear', 'mlp', 'k-sparse')


def width(form):
    """Hidden width, set once per form at the largest input width. A matrix has no width."""
    floats = BUDGET // 4 // CHARGED_COPIES - 2 * MAX_IN * D - 1
    if form == 'linear':
        return None
    return (floats - D) // (2 * D + 1)      # D -> H -> D with biases: (D + 1) H + (H + 1) D


def param_floats(form, d_k, d_v):
    w = (d_k + d_v) * D + 1
    if form == 'linear':
        return w + D * D
    H = width(form)
    return w + (D + 1) * H + (H + 1) * D


class Memory:
    def __init__(self, form, d_k, d_v, projections='learned', eta=.01, beta=.9, C=16, p=.05, seed=0):
        assert form in FORMS and projections in ('learned', 'fixed')
        g = torch.Generator().manual_seed(seed)
        self.form, self.eta, self.beta, self.C, self.p = form, eta, beta, C, p
        self.fixed = projections == 'fixed'
        self.W_K = torch.randn(d_k, D, generator=g) / math.sqrt(d_k)
        self.W_V = torch.randn(d_v, D, generator=g) / math.sqrt(d_v)
        self.s = torch.zeros(())
        if form == 'linear':
            self.M = [torch.randn(D, D, generator=g) / math.sqrt(D)]
        else:
            H = width(form)
            self.M = [torch.randn(D, H, generator=g) / math.sqrt(D), torch.zeros(H), torch.randn(H, D, generator=g) / math.sqrt(H), torch.zeros(D)]
        self.params = [*self.M, self.s] + ([] if self.fixed else [self.W_K, self.W_V])
        for t in self.params:
            t.requires_grad_(True)
        self.S = [torch.zeros_like(t) for t in self.params]
        self.acc = [torch.zeros_like(t) for t in self.params]
        self.t = 0

    def recall(self, k):
        if self.form == 'linear':
            return k @ self.M[0]
        h = torch.tanh(k @ self.M[0] + self.M[1])
        if self.form == 'k-sparse':
            keep = max(1, int(round(self.p * h.shape[-1])))
            top = h.abs().topk(keep, dim=-1).indices
            mask = torch.zeros_like(h).scatter(-1, top, 1.)
            h = h * mask
        return h @ self.M[2] + self.M[3]

    def term(self, q, C):
        """s <M(W_K q), W_V c>, for q (n, d_k) and C (n, 9, d_v)."""
        return self.s * ((self.recall(q @ self.W_K)[:, None] * (C @ self.W_V)).sum(-1))

    def update(self, q, C, reader_logits, target):
        """One host update's items. The reader's logits are constants; every C updates, S = beta^C S - eta C g, on g the clipped mean gradient."""
        loss = F.cross_entropy(reader_logits + self.term(q, C), target)
        grads = torch.autograd.grad(loss, self.params)
        for a, g in zip(self.acc, grads):
            a += g
        self.t += 1
        if self.t % self.C == 0:
            mean = [a / self.C for a in self.acc]
            norm = torch.sqrt(sum((m * m).sum() for m in mean))
            scale = min(1., CLIP / (float(norm) + 1e-12))
            with torch.no_grad():
                for i, (th, m) in enumerate(zip(self.params, mean)):
                    self.S[i] = self.beta ** self.C * self.S[i] - self.eta * self.C * scale * m
                    th += self.S[i]
                    self.acc[i] = torch.zeros_like(th)
        return float(loss.detach())

    @torch.no_grad()
    def combined(self, q, C, reader_logits):
        return reader_logits + self.term(q, C)

    def state_hash(self):
        import hashlib
        h = hashlib.sha256()
        for t in [*self.params, self.W_K, self.W_V]:
            h.update(t.detach().numpy().tobytes())
        return h.hexdigest()


@torch.no_grad()
def accuracy(scores, counts, truth):
    """Pick the top candidate; counts (n, 9) names each candidate's count."""
    picks = counts.gather(1, scores.argmax(1, keepdim=True))[:, 0]
    return float((picks == truth).float().mean())


def nearest_store(keys, values, queries):
    """`flat-store`, an explicit diagnostic: the stored value of the nearest stored key."""
    return values[(queries @ keys.T).argmax(1)]
