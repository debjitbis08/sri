"""Memory components for translation memory v1 (docs/translation-memory-v1-draft.md, "System under test").

Pure tensor code with no host or task dependence, so every rule here can be checked in isolation
(experiments/translation_store_checks.py):
  ReferenceMap   identity-centred ridge map from current codes to the reference space, fitted on anchors only
  Separation     fixed random expansion of a centred, scaled reference code, k-winners-take-all
  Associative    binary clipped Hebbian weights over [source; target], synchronous k-winners-take-all settling,
                 checkpoints 1, 2, 4, 8, 16 (checkpoint 1's target part is the one-step control)
  FlatSparse     an explicit slot per write keyed by the same separated pattern (storage control)
  FlatStore      an explicit slot per write keyed by the mapped dense code (system comparison)
  Gate           one logistic unit over five confidence scalars
  SentenceRule   a learned 3 x 3 position mixing of routed target patterns

Every memory returns, per query, a score in [0, 1] for each target index (normalized overlap or clipped cosine of the
best entry with that value), so gate, router and sentence rule treat all memories alike.
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINTS = (1, 2, 4, 8, 16)
TEMPERATURE = 10.  # memory and sentence scores -> probabilities: softmax(TEMPERATURE * score), scores in [0, 1]


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            continue
        h.update(np.ascontiguousarray(t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)).tobytes())
    return h.hexdigest()


def generator(rng):
    """A torch generator seeded from a numpy stream (keeps every draw inside the project's named streams)."""
    return torch.Generator().manual_seed(int(rng.integers(2 ** 62)))


# ---------------------------------------------------------------- reference coordinates

class ReferenceMap:
    """x_ref = A [x; 1], A of shape (d, d + 1) fitted by

        min_A ||A X1 - Y||_F^2 / (K s^2) + lam ||A - [I 0]||_F^2 / d,

    X1 the current anchor codes with a ones row, Y their stored reference codes (both columns per anchor). The bias column
    is regularized toward 0 by the same term (an affine extension of the draft's d x d map, so a uniform shift of the
    code can be corrected). Before the first consolidation's reference is set the map is the identity."""

    def __init__(self, d, lam):
        self.d, self.lam = d, lam
        self.A = torch.cat([torch.eye(d), torch.zeros(d, 1)], 1)
        self.ref, self.s2, self.mu = None, None, None

    def set_reference(self, anchors):
        """anchors (K, d) at the first consolidation: stored codes, the code scale s^2 and the centre used by separation."""
        self.ref = anchors.detach().double().clone()
        self.s2 = float(self.ref.pow(2).mean())
        self.mu = self.ref.mean(0)

    @staticmethod
    def solve(X, Y, lam, s2):
        """X, Y (K, d) -> A (d, d + 1), closed form."""
        K, d = X.shape
        X1 = torch.cat([X.double(), torch.ones(K, 1, dtype=torch.float64)], 1)  # (K, d+1)
        I0 = torch.cat([torch.eye(d, dtype=torch.float64), torch.zeros(d, 1, dtype=torch.float64)], 1)
        a, b = 1. / (K * s2), lam / d
        lhs = a * X1.T @ X1 + b * torch.eye(d + 1, dtype=torch.float64)  # (d+1, d+1)
        rhs = a * Y.double().T @ X1 + b * I0  # (d, d+1)
        return torch.linalg.solve(lhs, rhs.T).T

    def fit(self, anchors_now):
        self.A = self.solve(anchors_now, self.ref, self.lam, self.s2).float()

    def residual(self, anchors_now):
        """Mean squared anchor error after the map, in units of s^2."""
        return float((self(anchors_now).double() - self.ref).pow(2).mean() / self.s2)

    def __call__(self, x):
        return x.float() @ self.A[:, :-1].T + self.A[:, -1]


def heldout_anchor_error(X, Y, symbols, lam, s2, folds):
    """Symbol-level cross-validation: anchors of one symbol never straddle fitting and validation. X, Y (K, d); symbols (K,)
    symbol index per anchor; folds: list of symbol-index sets held out in turn. Mean squared error in units of s^2."""
    errs = []
    for held in folds:
        te = np.isin(symbols, list(held))
        A = ReferenceMap.solve(X[~te], Y[~te], lam, s2)
        pred = torch.cat([X[te].double(), torch.ones(int(te.sum()), 1, dtype=torch.float64)], 1) @ A.T
        errs.append(float((pred - Y[te].double()).pow(2).mean() / s2))
    return float(np.mean(errs))


# ---------------------------------------------------------------- separation and k-winners-take-all

def kwta(h, k, priority):
    """Top-k units per row of h (n, m); ties broken by a fixed per-unit priority in [0, 0.5) (h is integer-valued in
    the associative memory, so the priority never reorders unequal sums). Returns a 0/1 float tensor."""
    idx = (h + priority).topk(k, 1).indices
    return torch.zeros_like(h).scatter_(1, idx, 1.)


def priorities(m, rng):
    return torch.from_numpy(rng.permutation(m).astype(np.float32) / (2. * m))


class Separation:
    """Fixed random expansion: pattern = top-k_s of R z, z = (x_ref - mu) / s (mu, s from the reference anchors), R (M_s, d)
    standard normal from a seeded stream (storable as the seed alone)."""

    def __init__(self, d, m_s, k_s, rng):
        self.R = torch.from_numpy(rng.standard_normal((m_s, d)).astype(np.float32))
        self.k = k_s
        self.prio = priorities(m_s, rng)

    def __call__(self, x_ref, mu, s2):
        z = (x_ref.float() - mu.float()) / float(np.sqrt(s2))
        return kwta(z @ self.R.T, self.k, self.prio)


def codebook(n_targets, m_t, k_t, rng):
    """One fixed sparse pattern (k_t of m_t units) per target index."""
    cb = torch.zeros(n_targets, m_t)
    for t in range(n_targets):
        cb[t, rng.choice(m_t, k_t, replace=False)] = 1.
    return cb


# ---------------------------------------------------------------- associative memory

class Associative:
    """Binary clipped Hebbian weights over the joint pattern [source (m_s); target (m_t)], no self-connections.

    Retrieval: x_0 = [cue; 0]; each step, synchronously, h = W x and the new state keeps the top k_s source units and the top
    k_t target units of h (ties by a fixed seeded priority). The source part is free after the start. A state equal to the
    previous one is a fixed point (converged); a state equal to an earlier, non-adjacent one is a cycle (flagged
    non-converged). Every read runs to the last checkpoint; the target part at checkpoint 1 is the one-step control."""

    def __init__(self, m_s, m_t, k_s, k_t, rng, device='cpu'):
        self.m_s, self.m_t, self.k_s, self.k_t = m_s, m_t, k_s, k_t
        self.M = m_s + m_t
        self.W = torch.zeros(self.M, self.M, device=device)
        self.prio_s, self.prio_t = priorities(m_s, rng).to(device), priorities(m_t, rng).to(device)
        self.writes = 0

    def write(self, s, t):
        """s (n, m_s), t (n, m_t) 0/1 patterns; W_ij = 1 for co-active i != j."""
        p = torch.cat([s, t], 1).to(self.W)
        self.W = ((self.W + p.T @ p) > 0).float()
        self.W.fill_diagonal_(0.)
        self.writes += len(p)

    def step(self, x):
        h = x @ self.W.T
        return torch.cat([kwta(h[:, :self.m_s], self.k_s, self.prio_s), kwta(h[:, self.m_s:], self.k_t, self.prio_t)], 1)

    def retrieve(self, cue, checkpoints=CHECKPOINTS):
        """cue (n, m_s) -> {checkpoint: target part (n, m_t)}, plus per query: steps to a fixed point (or the cap), whether
        it converged, whether it entered a cycle, and the work counts (dense operations; connections from active units)."""
        x = torch.cat([cue.to(self.W), torch.zeros(len(cue), self.m_t, device=self.W.device)], 1)
        n, cap = len(cue), max(checkpoints)
        hist, out = [x], {}
        fixed_at = torch.full((n,), -1, dtype=torch.long)
        cycle = torch.zeros(n, dtype=torch.bool)
        nnz_col = self.W.sum(0)  # outgoing nonzero connections of each unit (W symmetric)
        active_conn = torch.zeros(n)
        for t in range(1, cap + 1):
            active_conn += (hist[-1] @ nnz_col).cpu()
            x = self.step(hist[-1])
            same_prev = (x == hist[-1]).all(1).cpu()
            fixed_at[(fixed_at < 0) & same_prev] = t - 1
            if t >= 2:
                for old in hist[:-1]:
                    cycle |= ((x == old).all(1).cpu() & (fixed_at < 0))
            hist.append(x)
            if t in checkpoints:
                out[t] = x[:, self.m_s:].clone()
        converged = fixed_at >= 0
        steps = torch.where(converged, fixed_at.clamp(min=1), torch.full_like(fixed_at, cap))
        return out, {'steps': steps, 'converged': converged, 'cycle': cycle & ~converged,
                     'dense_ops': cap * self.M * self.M, 'active_connections': active_conn}

    def one_step(self, cue):
        """Target part = top k_t of W_ts cue (identical to checkpoint 1's target part)."""
        h = cue.to(self.W) @ self.W[self.m_s:, :self.m_s].T
        return kwta(h, self.k_t, self.prio_t)

    def nbytes(self):
        return self.M * self.M / 8  # one bit per weight


class OneStep:
    """The source-to-target part alone: binary clipped W_ts (m_t, m_s), one-step read, target part = top k_t of W_ts s. Draws its
    tie-break priorities in Associative's order, so its reads equal Associative.one_step on the same writes. No recurrence, so
    a tolerant read here is associative retrieval, not recurrent pattern completion."""

    def __init__(self, m_s, m_t, k_s, k_t, rng, device='cpu'):
        self.m_s, self.m_t, self.k_s, self.k_t = m_s, m_t, k_s, k_t
        self.W = torch.zeros(m_t, m_s, device=device)
        priorities(m_s, rng)
        self.prio_t = priorities(m_t, rng).to(device)
        self.writes = 0

    def write(self, s, t):
        self.W = ((self.W + t.to(self.W).T @ s.to(self.W)) > 0).float()
        self.writes += len(s)

    def read(self, cue):
        return kwta(cue.to(self.W) @ self.W.T, self.k_t, self.prio_t)

    def active_connections(self, cue):
        """Per query: nonzero weights leaving the cue's active units (the work if the implementation follows them)."""
        return cue.to(self.W) @ self.W.sum(0)

    def dense_ops(self):
        return self.m_s * self.m_t

    def nbytes(self):
        return self.m_s * self.m_t / 8  # one bit per weight


def target_scores(tpart, cb):
    """Normalized overlap of retrieved target parts (n, m_t) with every codebook pattern (T, m_t): (n, T) in [0, 1]."""
    return (tpart.to(cb) @ cb.T) / cb.sum(1)[None]


# ---------------------------------------------------------------- explicit stores

class _Slots:
    """Shared slot bookkeeping: capacity n_slots; allocation to a free slot; when full, evict the least-used slot (usage
    counts reads won during training phases, never evaluation), ties to the oldest."""

    def __init__(self, n_slots, n_targets):
        self.n_slots, self.n_targets = n_slots, n_targets
        self.keys, self.values, self.used, self.born = [], [], [], []
        self.clock, self.evictions, self.refreshes, self.overwrites = 0, 0, 0, 0

    def _place(self, key, value, dup):
        self.clock += 1
        if dup is not None:
            if self.values[dup] == value:
                self.refreshes += 1
            else:
                self.overwrites += 1
            self.keys[dup], self.values[dup], self.born[dup] = key, value, self.clock
            return
        if len(self.keys) >= self.n_slots:
            j = min(range(len(self.keys)), key=lambda i: (self.used[i], self.born[i]))
            del self.keys[j], self.values[j], self.used[j], self.born[j]
            self.evictions += 1
        self.keys.append(key); self.values.append(value); self.used.append(0); self.born.append(self.clock)

    def per_target(self, sim):
        """sim (n, slots) -> (n, T): best similarity among slots holding each target (0 where none), and the winning slot."""
        n = len(sim)
        out = torch.zeros(n, self.n_targets)
        if not self.keys:
            return out, torch.full((n,), -1, dtype=torch.long)
        vals = torch.tensor(self.values)
        out = out.scatter_reduce(1, vals[None].expand(n, -1), sim.clamp(min=0).float(), 'amax', include_self=True)
        return out, sim.argmax(1)

    def count_use(self, winners):
        for j in winners.tolist():
            if j >= 0:
                self.used[j] += 1


class FlatSparse(_Slots):
    """Key = the separated source pattern; read by largest overlap / k_s over all slots (exhaustive). A write whose pattern
    equals an existing key exactly refreshes that slot (same value) or overwrites it (different value)."""

    def __init__(self, n_slots, n_targets, k_s):
        super().__init__(n_slots, n_targets)
        self.k = k_s

    def write(self, s, values):
        for key, v in zip(s, values):
            dup = None
            if self.keys:
                ov = torch.stack(self.keys) @ key
                j = int(ov.argmax())
                if float(ov[j]) >= self.k:
                    dup = j
            self._place(key.clone(), int(v), dup)

    def read(self, s):
        sim = (s @ torch.stack(self.keys).T) / self.k if self.keys else torch.zeros(len(s), 0)
        return self.per_target(sim)

    def nbytes(self):
        return len(self.keys) * (self.k * 2 + 2)  # k_s 16-bit unit indices and a 16-bit value per slot


class FlatStore(_Slots):
    """Key = the mapped, centred, scaled dense code; read by cosine similarity (clipped at 0) over all slots. A write whose
    key has cosine >= 1 - 1e-6 with an existing key is a duplicate."""

    def write(self, z, values):
        for key, v in zip(z, values):
            key = F.normalize(key.float(), dim=0)
            dup = None
            if self.keys:
                cs = torch.stack(self.keys) @ key
                j = int(cs.argmax())
                if float(cs[j]) >= 1 - 1e-6:
                    dup = j
            self._place(key, int(v), dup)

    def read(self, z):
        sim = F.normalize(z.float(), dim=1) @ torch.stack(self.keys).T if self.keys else torch.zeros(len(z), 0)
        return self.per_target(sim)

    def nbytes(self):
        return len(self.keys) * (self.keys[0].numel() * 4 + 2) if self.keys else 0


# ---------------------------------------------------------------- confidence, gate, router

def entropy01(p):
    return -(p.clamp_min(1e-12).log() * p).sum(1) / np.log(p.shape[1])


def features(p_host, mem_scores, converged):
    """Five scalars per query: host max softmax probability, host normalized entropy, memory best score, memory margin
    (best minus second), convergence (1 for explicit stores)."""
    top2 = mem_scores.topk(2, 1).values
    return torch.stack([p_host.max(1).values, entropy01(p_host), top2[:, 0], top2[:, 0] - top2[:, 1], converged.float()], 1)


def p_memory(mem_scores):
    return F.softmax(TEMPERATURE * mem_scores, 1)


class Gate(torch.nn.Module):
    """w = sigmoid(a . f + b): the memory's weight in p = w p_mem + (1 - w) p_host."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 1)
        torch.nn.init.zeros_(self.lin.weight); torch.nn.init.zeros_(self.lin.bias)

    def forward(self, f):
        return torch.sigmoid(self.lin(f))[:, 0]

    def fit(self, f, p_mem, p_host, labels, steps, lr):
        """Gate parameters only: minimize -log p_mix[label] over the buffered entries (full batch)."""
        opt = torch.optim.Adam(self.parameters(), lr)
        for _ in range(steps):
            w = self(f)[:, None]
            p = w * p_mem + (1 - w) * p_host
            loss = -p[torch.arange(len(labels)), labels].clamp_min(1e-12).log().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def answer(self, f, p_mem, p_host):
        w = self(f)
        return (w[:, None] * p_mem + (1 - w[:, None]) * p_host).argmax(1), w


ROUTER_THRESHOLDS = ('host', 0., .25, .5, .75, 1.)


def route(threshold, p_host, mem_scores):
    """Confidence router: the memory answers when its best score reaches the threshold; 'host' always uses the host."""
    host = p_host.argmax(1)
    if threshold == 'host':
        return host
    best = mem_scores.max(1)
    return torch.where(best.values >= threshold, best.indices, host)


# ---------------------------------------------------------------- sentences

class SentenceRule(torch.nn.Module):
    """out_j = sum_i softmax(theta)_ji r_i over three routed target patterns r_i; answer by codebook overlap."""

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(3, 3))

    def forward(self, r):  # r (n, 3, m_t) -> (n, 3, m_t)
        return torch.einsum('ji,nim->njm', F.softmax(self.theta, 1), r)

    def logits(self, r, cb):
        return TEMPERATURE * (self(r) @ cb.T) / cb.sum(1)

    def fit(self, r, targets, cb, steps, lr):
        opt = torch.optim.Adam(self.parameters(), lr)
        for _ in range(steps):
            loss = F.cross_entropy(self.logits(r, cb).reshape(-1, len(cb)), targets.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.detach())


def routed_patterns(w, mem_tpart, p_host, cb):
    """w (n,) memory weight; mem_tpart (n, m_t) reinstated target part; host contribution = its softmax-weighted codebook."""
    return w[:, None] * mem_tpart.to(cb) + (1 - w[:, None]) * (p_host @ cb)
