"""Surprise lifecycle v1 (docs/surprise-lifecycle-v1-contract.md): the world's schedule, every arm, and the budget arithmetic.

An arm sees only cue keys and outcome keys, in stream order, through step(c, o). The outcome table, kinds and cause ids stay with the
scheduler and the evaluator; only `OracleLife` is told them, through its own method. Nothing here depends on torch.
"""
import numpy as np

D_C, D_O = 96, 32
READER_BYTES, KNOWN_BYTES = D_C * D_O * 4, 24 * (D_C + D_O) * 4
TRACE, SHADOW, SLOT, EVENT, SKETCH_ENTRY = 560, 528, 512, 516, 8
KIB = 1024

N_KNOWN_RULES, N_NEW_RULES, N_BINDING_FAMILIES, N_BACKGROUND = 8, 4, 4, 32
N_FAMILIES = N_KNOWN_RULES + N_NEW_RULES + N_BINDING_FAMILIES + N_BACKGROUND
N_OUTCOMES, N_ITEMS, N_AUDIT_ITEMS, N_KNOWN_BINDINGS = 32, 300, 100, 24
N_SYMBOLS = N_FAMILIES + N_OUTCOMES + N_ITEMS + N_AUDIT_ITEMS
QUERY_RATE = 0.1
KINDS = ('known-rule', 'known-binding', 'rule', 'rule-noise', 'binding', 'exception', 'rare', 'noise-repeat', 'noise-burst', 'noise-once', 'query')
K = {k: i for i, k in enumerate(KINDS)}


# ---------------------------------------------------------------- the world

def symbol_ranges():
    """Indices into a seed's permuted symbol table."""
    a = N_KNOWN_RULES
    b = a + N_NEW_RULES
    c = b + N_BINDING_FAMILIES
    d = c + N_BACKGROUND
    e = d + N_OUTCOMES
    f = e + N_ITEMS
    return {'known': np.arange(0, a), 'new': np.arange(a, b), 'binding': np.arange(b, c), 'background': np.arange(c, d),
            'outcomes': np.arange(d, e), 'items': np.arange(e, f), 'audit_items': np.arange(f, f + N_AUDIT_ITEMS)}


def _pair(rng, items):
    i, j = rng.choice(items, 2, replace=False)
    return int(i), int(j)


def make_tables(rng, scale=1):
    """The seed's fixed knowledge and causes, identical at every gap. Cues are (family, item, item). scale multiplies every cause but
    `rule`, whose families are limited, for the load probe."""
    r = symbol_ranges()
    out = rng.permutation(r['outcomes'])
    known_out = out[:N_KNOWN_RULES]
    new_out = out[N_KNOWN_RULES:N_KNOWN_RULES + N_NEW_RULES]
    t = {'known_rules': [(int(f), int(o)) for f, o in zip(r['known'], known_out)], 'new_rules': [(int(f), int(o)) for f, o in zip(r['new'], new_out)]}
    used = set()

    def fresh(family):
        while True:
            cue = (int(family), *_pair(rng, r['items']))
            if cue not in used:
                used.add(cue)
                return cue
    t['known_bindings'] = [(fresh(rng.choice(r['known'])), int(rng.choice(r['outcomes']))) for _ in range(N_KNOWN_BINDINGS)]
    bf = r['binding']
    t['bindings'] = [(fresh(rng.choice(bf)), int(rng.choice(r['outcomes']))) for _ in range(12 * scale)]
    t['rares'] = [(fresh(rng.choice(bf)), int(rng.choice(r['outcomes']))) for _ in range(12 * scale)]
    t['noise_repeat'] = [fresh(rng.choice(bf)) for _ in range(8 * scale)]
    t['noise_burst'] = [fresh(rng.choice(bf)) for _ in range(4 * scale)]
    t['exceptions'] = []
    for _ in range(6 * scale):
        i = int(rng.integers(N_KNOWN_RULES))
        other = int(rng.choice([o for o in r['outcomes'] if o != known_out[i]]))
        t['exceptions'].append((fresh(r['known'][i]), other))
    t['reserved'] = used
    return t


def make_world(rng, tables, g, q, occurrences=8):
    """One stream of 10g events (more if occurrences exceed 8). Arrays: cue triples, outcome (-1 for a query), kind, cause."""
    r = symbol_ranges()
    n_periods = max(8, occurrences)
    L = (n_periods + 2) * g
    fam, it1, it2 = np.zeros(L, np.int32), np.zeros(L, np.int32), np.zeros(L, np.int32)
    outcome, kind, cause = np.full(L, -1, np.int32), np.full(L, -1, np.int8), np.full(L, -1, np.int32)
    free = np.ones(L, bool)
    cause_names = []

    def place(want):
        """The free position nearest after `want`; wrapping is never needed since the tail is free."""
        t = int(np.clip(want, 0, L - 1))
        while t < L and not free[t]:
            t += 1
        if t == L:
            t = int(np.flatnonzero(free)[0])
        free[t] = False
        return t

    def periodic(n):
        phase = rng.uniform(0, g)
        return [phase + i * g + rng.uniform(-.1, .1) * g for i in range(n)]

    def put(t, cue, o, k, c):
        fam[t], it1[t], it2[t] = cue
        outcome[t], kind[t], cause[t] = o, K[k], c

    def new_cause(name):
        cause_names.append(name)
        return len(cause_names) - 1

    for f, o in tables['new_rules']:
        c = new_cause(('rule', f))
        times = periodic(occurrences)
        for t0 in times:
            put(place(t0), (f, *_pair(rng, r['items'])), o, 'rule', c)
        for _ in range(2):
            wrong = int(rng.choice([x for x in r['outcomes'] if x != o]))
            put(place(rng.uniform(times[0], times[-1])), (f, *_pair(rng, r['items'])), wrong, 'rule-noise', c)
    for name, group, k, n in (('binding', tables['bindings'], 'binding', occurrences), ('exception', tables['exceptions'], 'exception', occurrences),
                              ('rare', tables['rares'], 'rare', 2)):
        for cue, o in group:
            c = new_cause((name, cue))
            for t0 in periodic(n):
                put(place(t0), cue, o, k, c)
    for cue in tables['noise_repeat']:
        c = new_cause(('noise-repeat', cue))
        for t0 in periodic(occurrences):
            put(place(t0), cue, int(rng.choice(r['outcomes'])), 'noise-repeat', c)
    for cue in tables['noise_burst']:
        c = new_cause(('noise-burst', cue))
        for t0 in periodic(4):
            o = int(rng.choice(r['outcomes']))
            for dt in np.sort(rng.choice(32, 6, replace=False)):
                put(place(t0 + dt), cue, o, 'noise-burst', c)

    # background families become active at the same uniform range as rule families start, so a new family does not announce a rule
    active_at = {int(f): 0. for f in r['known']}
    active_at.update({int(f): 0. for f in r['binding']})
    active_at.update({int(f): float(rng.uniform(0, g)) for f in r['background']})
    families = np.array(sorted(active_at))
    times = np.array([active_at[f] for f in families])

    idx = np.flatnonzero(free)
    rng.shuffle(idx)
    n_once = min(int(round(q * L)), len(idx))
    n_query = min(int(round(QUERY_RATE * L)), len(idx) - n_once)
    once, query, known = np.sort(idx[:n_once]), np.sort(idx[n_once:n_once + n_query]), idx[n_once + n_query:]

    seen = set(tables['reserved'])
    for t in once:
        pick = families[times <= t]
        for _ in range(1000):
            cue = (int(rng.choice(pick)), *_pair(rng, r['items']))
            if cue not in seen:
                seen.add(cue)
                break
        else:
            raise RuntimeError('one-off cues exhausted')
        put(int(t), cue, int(rng.choice(r['outcomes'])), 'noise-once', -1)
    for t in known:
        if rng.random() < .75:
            f, o = tables['known_rules'][rng.integers(N_KNOWN_RULES)]
            put(int(t), (f, *_pair(rng, r['items'])), o, 'known-rule', -1)
        else:
            cue, o = tables['known_bindings'][rng.integers(N_KNOWN_BINDINGS)]
            put(int(t), cue, o, 'known-binding', -1)
    # queries repeat a cue already shown; a query before anything else is shown takes a known binding
    for t in query:
        shown = np.flatnonzero((kind[:t] >= 0) & (kind[:t] != K['query']))
        if len(shown):
            s = shown[rng.integers(len(shown))]
            cue = (int(fam[s]), int(it1[s]), int(it2[s]))
        else:
            cue = tables['known_bindings'][0][0]
        put(int(t), cue, -1, 'query', -1)
    assert (kind >= 0).all()
    render = rng.integers(0, 1 << 15, (L, 4)).astype(np.int32)
    return {'fam': fam, 'it1': it1, 'it2': it2, 'outcome': outcome, 'kind': kind, 'cause': cause, 'render': render, 'L': L, 'g': g, 'q': q, 'causes': cause_names}


def join_cues(a, b, c):
    """Three unit keys into one unit cue: the dot product of two cues is the mean of the three cosines."""
    return np.concatenate([a, b, c], axis=-1) / np.sqrt(3.)


# ---------------------------------------------------------------- budgets

def allocation(B_kib, S=0, sketch=False):
    """Traces, slot cap and sketch entries for a lifecycle arm, or None-free numbers the caller checks with `feasible`."""
    B = B_kib * KIB
    z = (B // 2) // SKETCH_ENTRY if sketch else 0
    pool = B - z * SKETCH_ENTRY - S * SHADOW
    return {'pool': pool, 'traces': pool // TRACE, 'slot_cap': (pool // 2) // SLOT, 'sketch': z, 'shadows': S}


def feasible(B_kib, S, m, sketch=False):
    a = allocation(B_kib, S, sketch)
    return a['traces'] >= S * m + 4 * m and a['slot_cap'] >= 12


def fifo_capacity(B_kib):
    return B_kib * KIB // EVENT


def mlp_params(H, layers):
    return D_C * H + H + D_O * H + D_O if layers == 1 else D_C * H + H + H * H + H + D_O * H + D_O


def mlp_width(B_kib, layers, momentum=False, window=0):
    """The largest hidden width whose weights, momentum and window fit in the total budget."""
    floats = (B_kib * KIB + READER_BYTES + KNOWN_BYTES) // 4 - window * (D_C + D_O)
    H = 1
    while mlp_params(H + 1, layers) * (2 if momentum else 1) <= floats:
        H += 1
    return H


# ---------------------------------------------------------------- common pieces

def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def fit_reader(C, O, lam=1.):
    """Ridge in closed form. R maps a cue on the right of a row vector: prediction = c @ R."""
    A = C.T @ C + lam * np.eye(C.shape[1], dtype=C.dtype)
    return np.linalg.solve(A, C.T @ O).astype(np.float32)


class Arm:
    """Every arm: step(c, o) with o None for a query, and answers(C), which must not change state."""
    name = 'arm'

    def __init__(self):
        self.ops = 0
        self.events = 0

    def step(self, c, o):
        raise NotImplementedError

    def answers(self, C):
        raise NotImplementedError

    def persistent_bytes(self):
        raise NotImplementedError


class ReaderArm(Arm):
    """A linear reader, known-binding slots and a threshold on a cue match. `frozen` still answers every event."""

    def __init__(self, R, slot_c, slot_o, tau, theta, name='frozen'):
        super().__init__()
        self.name = name
        self.R = R.copy()
        self.known_c, self.known_o = slot_c, slot_o
        self.tau, self.theta = tau, theta

    def hit(self, p, o):
        return float(p @ o) >= (1 - self.theta) * max(float(np.linalg.norm(p) * np.linalg.norm(o)), 1e-12)

    def extra_store(self):
        return None, None

    def _read(self, c):
        """Answer for one cue; `extra_store` lists stored events newest last, and the newest match wins over known bindings."""
        sc, so = self.extra_store()
        self.ops += self.known_c.shape[0] * D_C + D_C * D_O
        if sc is not None and len(sc):
            self.ops += len(sc) * D_C
            s = sc @ c
            w = np.flatnonzero(s >= self.tau)
            if len(w):
                return so[w[-1]]
        s = self.known_c @ c
        i = int(s.argmax())
        if s[i] >= self.tau:
            return self.known_o[i]
        return c @ self.R

    def answers(self, C):
        out = C @ self.R
        s = C @ self.known_c.T
        best, val = s.argmax(1), s.max(1)
        use = val >= self.tau
        out[use] = self.known_o[best[use]]
        sc, so = self.extra_store()
        if sc is not None and len(sc):
            s = C @ sc.T >= self.tau
            found = s.any(1)
            newest = s.shape[1] - 1 - s[:, ::-1].argmax(1)
            out[found] = so[newest[found]]
        return out

    def step(self, c, o):
        self.events += 1
        self.ops += self.known_c.shape[0] * D_C + D_C * D_O

    def persistent_bytes(self):
        return {'reader': READER_BYTES, 'known': KNOWN_BYTES}


class Slow(ReaderArm):
    def __init__(self, R, sc, so, tau, theta, eta):
        super().__init__(R, sc, so, tau, theta, 'slow')
        self.eta = eta

    def step(self, c, o):
        self.events += 1
        self.ops += self.known_c.shape[0] * D_C + D_C * D_O       # every arm answers every event
        if o is None:
            return
        self.R += self.eta * np.outer(c, o - c @ self.R)
        self.ops += 2 * D_C * D_O


class StoreArm(ReaderArm):
    """A ring of events answering lookups, newest last."""

    def __init__(self, R, sc, so, tau, theta, name, capacity):
        super().__init__(R, sc, so, tau, theta, name)
        self.cap = capacity
        self.c = np.zeros((capacity, D_C), np.float32)
        self.o = np.zeros((capacity, D_O), np.float32)
        self.n = 0
        self.head = 0

    def extra_store(self):
        if self.n < self.cap:
            return self.c[:self.n], self.o[:self.n]
        order = np.r_[self.head:self.cap, 0:self.head]
        return self.c[order], self.o[order]

    def push(self, c, o):
        self.c[self.head], self.o[self.head] = c, o
        self.head = (self.head + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def persistent_bytes(self):
        return {**super().persistent_bytes(), 'events': self.cap * EVENT}


class SurpriseFifo(StoreArm):
    def __init__(self, R, sc, so, tau, theta, eta, capacity):
        super().__init__(R, sc, so, tau, theta, 'surprise-fifo', capacity)
        self.eta = eta

    def step(self, c, o):
        self.events += 1
        p = self._read(c)
        if o is None or self.hit(p, o):
            return
        w = 1 - float(p @ o) / max(float(np.linalg.norm(p) * np.linalg.norm(o)), 1e-12)
        self.R += self.eta * w * np.outer(c, o - c @ self.R)
        self.ops += 2 * D_C * D_O
        self.push(c, o)


class Replay(StoreArm):
    def __init__(self, R, sc, so, tau, theta, eta, r, capacity, rng, reservoir):
        super().__init__(R, sc, so, tau, theta, 'reservoir-replay' if reservoir else 'fifo-replay', capacity)
        self.eta, self.r, self.rng, self.reservoir = eta, r, rng, reservoir
        self.seen = 0

    def _store(self, c, o):
        self.seen += 1
        if not self.reservoir:
            self.push(c, o)
        elif self.n < self.cap:
            self.push(c, o)
        else:
            j = int(self.rng.integers(self.seen))  # Vitter's Algorithm R
            if j < self.cap:
                self.c[j], self.o[j] = c, o

    def extra_store(self):
        if self.reservoir:
            return self.c[:self.n], self.o[:self.n]
        return super().extra_store()

    def step(self, c, o):
        self.events += 1
        self.ops += (self.known_c.shape[0] + self.n) * D_C + D_C * D_O
        if o is None:
            return
        self.R += self.eta * np.outer(c, o - c @ self.R)
        self.ops += 2 * D_C * D_O
        self._store(c, o)
        for _ in range(self.r):
            i = int(self.rng.integers(self.n))
            x, y = self.c[i], self.o[i]
            self.R += self.eta * np.outer(x, y - x @ self.R)
            self.ops += 2 * D_C * D_O


class SleepPeriodic(StoreArm):
    """Every P events, the store's residuals go into the reader with no selection, and the store clears. With rank set, into a new frozen expert."""

    def __init__(self, R, sc, so, tau, theta, lam, P, capacity, rank=None):
        super().__init__(R, sc, so, tau, theta, 'sleep-grow' if rank else 'sleep-periodic', capacity)
        self.lam, self.P, self.rank = lam, P, rank
        self.experts = 0
        self.since = 0

    def _sleep(self):
        C, O = self.c[:self.n], self.o[:self.n]
        E = O - C @ self.R
        A = C.T @ C + self.lam * np.eye(D_C, dtype=C.dtype)
        delta = np.linalg.solve(A, C.T @ E).astype(np.float32)
        self.ops += self.n * D_C * D_C + D_C ** 3
        if self.rank:
            u, s, vt = np.linalg.svd(delta, full_matrices=False)
            delta = ((u[:, :self.rank] * s[:self.rank]) @ vt[:self.rank]).astype(np.float32)  # kept merged for speed; bytes count the factors
            self.experts += 1
            self.ops += D_C * D_O * D_O
        self.R += delta
        self.n = self.head = 0
        self.since = 0

    def step(self, c, o):
        self.events += 1
        self.ops += (self.known_c.shape[0] + self.n) * D_C + D_C * D_O
        if o is None:
            return
        self.push(c, o)
        self.since += 1
        if self.since >= self.P:
            self._sleep()

    def persistent_bytes(self):
        out = super().persistent_bytes()
        if self.rank:
            out['experts'] = self.experts * self.rank * (D_C + D_O) * 4
        return out


# ---------------------------------------------------------------- neural memory, inspired by Titans and ATLAS

class MLP:
    def __init__(self, H, layers, rng):
        sizes = [D_C] + [H] * layers + [D_O]
        self.W = [(rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    def params(self):
        return self.W + self.b

    def set(self, values):
        n = len(self.W)
        self.W, self.b = list(values[:n]), list(values[n:])

    def forward(self, X):
        for w, b in zip(self.W[:-1], self.b[:-1]):
            X = np.tanh(X @ w + b)
        return X @ self.W[-1] + self.b[-1]

    def grad(self, X, Y):
        """Gradient of the mean squared error over a batch; returned in the order of params()."""
        hs, ps = [X], []
        h = X
        for w, b in zip(self.W[:-1], self.b[:-1]):
            a = h @ w + b
            h = np.tanh(a)
            hs.append(h)
        out = h @ self.W[-1] + self.b[-1]
        d = 2 * (out - Y) / len(X)
        gw, gb = [None] * len(self.W), [None] * len(self.b)
        gw[-1] = hs[-1].T @ d
        gb[-1] = d.sum(0)
        for i in range(len(self.W) - 2, -1, -1):
            d = (d @ self.W[i + 1].T) * (1 - hs[i + 1] ** 2)
            gw[i] = hs[i].T @ d
            gb[i] = d.sum(0)
        return gw + gb

    def count(self):
        return sum(p.size for p in self.params())


def fit_mlp(net, C, O, rng, lr=1e-3, batch=64, max_epochs=400, patience=20):
    """The contract's old-knowledge fit: Adam, 10% held out, best epoch restored. Adam's state is discarded with this function."""
    order = rng.permutation(len(C))
    split = len(C) // 10
    va, tr = order[:split], order[split:]
    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    best, best_state, wait, t, epochs = np.inf, None, 0, 0, 0
    for epochs in range(1, max_epochs + 1):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            t += 1
            gs = net.grad(C[b], O[b])
            new = []
            for j, (p, g) in enumerate(zip(net.params(), gs)):
                m[j] = .9 * m[j] + .1 * g
                v[j] = .999 * v[j] + .001 * g * g
                new.append(p - lr * (m[j] / (1 - .9 ** t)) / (np.sqrt(v[j] / (1 - .999 ** t)) + 1e-8))
            net.set([x.astype(np.float32) for x in new])
        loss = float(((net.forward(C[va]) - O[va]) ** 2).sum(1).mean())
        if loss < best - 1e-5:
            best, best_state, wait = loss, [p.copy() for p in net.params()], 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.set(best_state)
    return {'epochs': epochs, 'val_loss': best}


class NeuralMemory(Arm):
    """M_t = (1 - alpha) M_{t-1} + S_t, S_t = beta S_{t-1} - theta grad, with constants for what the papers make functions of the input.
    window = 1 is a Titans-style step; window > 1 is one step on the last c events, Omega-style with gamma = 1."""

    def __init__(self, net, alpha, beta, lr, window, name):
        super().__init__()
        self.name, self.net, self.alpha, self.beta, self.lr, self.window = name, net, alpha, beta, lr, window
        self.S = [np.zeros_like(p) for p in net.params()]
        self.wc = np.zeros((window, D_C), np.float32)
        self.wo = np.zeros((window, D_O), np.float32)
        self.n = self.head = 0

    def step(self, c, o):
        self.events += 1
        self.ops += self.net.count()
        if o is None:
            return
        self.wc[self.head], self.wo[self.head] = c, o
        self.head = (self.head + 1) % self.window
        self.n = min(self.n + 1, self.window)
        gs = self.net.grad(self.wc[:self.n], self.wo[:self.n])
        self.ops += 2 * self.n * self.net.count()
        new = []
        for j, (p, g) in enumerate(zip(self.net.params(), gs)):
            self.S[j] = self.beta * self.S[j] - self.lr * g.astype(np.float32)
            new.append((1 - self.alpha) * p + self.S[j])
        self.net.set(new)

    def answers(self, C):
        return self.net.forward(C)

    def persistent_bytes(self):
        out = {'weights': self.net.count() * 4}
        if self.beta:
            out['momentum'] = self.net.count() * 4
        if self.window > 1:
            out['window'] = self.window * (D_C + D_O) * 4
        return out


# ---------------------------------------------------------------- the lifecycle

class Group:
    __slots__ = ('id', 'kind', 'hits', 'misses', 'episodes', 'last', 'tries', 'shadow', 'born', 'norm')

    def __init__(self, gid, kind, t):
        self.id, self.kind, self.born = gid, kind, t
        self.hits = self.misses = self.episodes = self.tries = 0
        self.last = t
        self.shadow = None
        self.norm = None     # length of the members' mean unit cue, recomputed when membership changes


class Shadow:
    __slots__ = ('id', 'group', 'ebar', 'chat', 'norm2', 'fit', 'wins', 'failures', 'damage', 'rhits', 'last_win', 'born')


class Life(ReaderArm):
    """The mechanism in the order the contract gives it. Traces live in fixed arrays; a group's cue and outcome are computed from its
    members, so nothing beyond the charged counters is stored for it."""

    def __init__(self, R, sc, so, cfg, B_kib, name='life', sketch=False, group_by='residual', validate=True, self_count=False, seed_rng=None):
        super().__init__(R, sc, so, cfg['tau'], cfg['theta'], name)
        self.cfg, self.B_kib = cfg, B_kib
        self.group_by, self.validate, self.self_count = group_by, validate, self_count
        a = allocation(B_kib, cfg['S'], sketch)
        self.pool, self.slot_cap, self.Z = a['pool'], a['slot_cap'], a['sketch']
        self.cap = a['traces']
        self.tc = np.zeros((self.cap + 1, D_C), np.float32)
        self.te = np.zeros((self.cap + 1, D_O), np.float32)  # the residual under the current R, stored instead of the outcome: the same bytes, o = e + c R
        self.alive = np.zeros(self.cap + 1, bool)
        self.age = np.zeros(self.cap + 1, np.int64)
        self.same = np.full(self.cap + 1, -1, np.int32)
        self.res = np.full(self.cap + 1, -1, np.int32)
        self.pinned = np.zeros(self.cap + 1, bool)
        self.used = np.zeros(self.cap + 1, bool)     # has fitted a shadow: consumed
        self.familiar = np.zeros(self.cap + 1, bool)
        self.fits_of = {}                             # trace serial -> shadows fitted, kept only for the tests' invariant
        self.serial = np.zeros(self.cap + 1, np.int64)
        self.next_serial = 0
        self.groups = {}
        self.shadows = []
        self.next_gid = self.next_sid = 0
        self.new_c = np.zeros((self.slot_cap, D_C), np.float32)
        self.new_o = np.zeros((self.slot_cap, D_O), np.float32)
        self.new_used = np.zeros(self.slot_cap, np.int64)
        self.n_new = 0
        self.t = 0
        self.log = {'proposed': 0, 'retired': 0, 'displaced': 0, 'promoted': 0, 'slots': 0, 'dropped_noise': 0, 'evicted': 0, 'refits': 0, 'sketch_matches': 0}
        if self.Z:
            gen = seed_rng if seed_rng is not None else np.random.default_rng(0)
            self.pc = gen.standard_normal((32, D_C)).astype(np.float32)
            self.pe = gen.standard_normal((32, D_O)).astype(np.float32)
            self.sk_c = np.zeros(self.Z, np.uint32)
            self.sk_e = np.zeros(self.Z, np.uint32)
            self.sk_n = self.sk_head = 0

    # ---- reading

    def trace_budget_now(self):
        return (self.pool - self.n_new * SLOT) // TRACE

    def _reader_or_slots(self, c):
        self.ops += (self.known_c.shape[0] + self.n_new) * D_C + D_C * D_O
        if self.n_new:
            s = self.new_c[:self.n_new] @ c
            i = int(s.argmax())
            if s[i] >= self.tau:
                self.new_used[i] = self.t
                return self.new_o[i]
        s = self.known_c @ c
        i = int(s.argmax())
        if s[i] >= self.tau:
            return self.known_o[i]
        return c @ self.R

    def answers(self, C):
        out = C @ self.R
        s = C @ self.known_c.T
        use = s.max(1) >= self.tau
        out[use] = self.known_o[s.argmax(1)[use]]
        if self.n_new:
            s = C @ self.new_c[:self.n_new].T
            use = s.max(1) >= self.tau
            out[use] = self.new_o[:self.n_new][s.argmax(1)[use]]
        return out

    # ---- groups

    def _members(self, gid, kind):
        return np.flatnonzero(self.alive & ((self.same if kind == 'same' else self.res) == gid))

    def _same_cue(self, gid):
        return unit(self.tc[self._members(gid, 'same')].mean(0))

    def _outcomes(self, idx):
        self.ops += len(idx) * D_C * D_O
        return self.te[idx] + self.tc[idx] @ self.R

    def _same_outcome(self, gid):
        return self._outcomes(self._members(gid, 'same')).mean(0)

    def _delete_if_empty(self, gid):
        g = self.groups.get(gid)
        if g is not None and not len(self._members(gid, g.kind)):
            del self.groups[gid]

    def _drop_trace(self, i):
        """A pinned trace is never here. It leaves both groups, whose counts stay until they hold nothing."""
        sg, rg = int(self.same[i]), int(self.res[i])
        self.alive[i] = False
        self.same[i] = self.res[i] = -1
        self.used[i] = self.familiar[i] = False
        self._delete_if_empty(sg)
        self._delete_if_empty(rg)
        if sg in self.groups:
            self.groups[sg].norm = None

    def _drop_same_group(self, gid):
        """Its traces go, except pinned ones, which only leave this group and stay until their shadow leaves."""
        for i in self._members(gid, 'same'):
            if self.pinned[i]:
                self.same[i] = -1
            else:
                self._drop_trace(i)
        self.groups.pop(gid, None)

    # ---- the event

    def _eligible_same(self, c):
        """Same-cue groups by cue alone: cos(c, mean member) is the mean similarity over the mean's length. Also returns the best match."""
        idx = np.flatnonzero(self.alive & (self.same >= 0))
        self.ops += len(idx) * D_C
        if not len(idx):
            return [], None
        ids, inv = np.unique(self.same[idx], return_inverse=True)
        sums = np.bincount(inv, weights=self.tc[idx] @ c, minlength=len(ids))
        cnt = np.bincount(inv, minlength=len(ids))
        norms = np.empty(len(ids))
        for j, gid in enumerate(ids):
            g = self.groups[int(gid)]
            if g.norm is None:
                g.norm = float(np.linalg.norm(self.tc[idx[inv == j]].mean(0)))
            norms[j] = g.norm
        cos = sums / (cnt * np.maximum(norms, 1e-12))
        out = [self.groups[int(ids[j])] for j in np.flatnonzero(cos >= self.tau)]
        best = self.groups[int(ids[int(cos.argmax())])] if len(out) else None
        return out, best

    def _score(self, c, o, Rc, eligible):
        """Step 2. With o None (self_count only), the arm's own answer counts as confirmation: the failure mode under test."""
        t, E = self.t, self.cfg['E']
        for g in eligible:
            new_episode = t - g.last >= E
            if new_episode:
                g.episodes += 1
                g.last = t
            if o is None:
                if new_episode:
                    g.hits += 1
            elif self.hit(self._same_outcome(g.id), o):
                if new_episode:
                    g.hits += 1
            else:
                g.misses += 1
        if not self.shadows or not self.validate:
            return
        r_hit = o is not None and self.hit(Rc, o)
        for s in self.shadows:
            d = float(c @ s.chat) / s.norm2
            p = Rc + d * s.ebar
            self.ops += D_C + 2 * D_O
            eligible_shadow = d * np.sqrt(s.norm2) >= self.cfg['tau_s'] and not (self.tc[s.fit] @ c >= self.tau).any()
            if o is None:
                if eligible_shadow and t - s.last_win >= E:
                    s.wins += 1
                    s.last_win = t
                continue
            p_hit = self.hit(p, o)
            if eligible_shadow:
                if not p_hit:
                    s.failures += 1
                elif not r_hit and t - s.last_win >= E:
                    s.wins += 1
                    s.last_win = t
            if r_hit:
                s.rhits += 1
                if not p_hit:
                    s.damage += 1

    def _write_slot(self, cue, outcome):
        if self.n_new >= self.slot_cap:
            i = int(self.new_used[:self.n_new].argmin())
        else:
            i = self.n_new
            self.n_new += 1
        self.new_c[i], self.new_o[i], self.new_used[i] = cue, outcome, self.t
        self.log['slots'] += 1

    def _retire(self, s, reason='retired'):
        for i in s.fit:
            self.pinned[i] = False
            self.used[i] = True
        g = self.groups.get(s.group)
        if g is not None:
            g.shadow = None
        self.shadows.remove(s)
        self.log[reason] += 1

    def _decide(self, eligible):
        """Step 3. Only groups scored this event can have newly met a same-cue threshold, so only they are examined."""
        c = self.cfg
        for gid in sorted(g.id for g in eligible):
            g = self.groups.get(gid)
            if g is None:
                continue
            if self.validate and g.misses >= c['h_n']:
                self._drop_same_group(gid)
                self.log['dropped_noise'] += 1
            elif (g.hits >= c['h_b'] and g.misses == 0) if self.validate else g.episodes >= c['h_b']:
                self._write_slot(self._same_cue(gid), self._same_outcome(gid))
                self._drop_same_group(gid)
        if not self.validate:
            return self._count_promote()
        for s in sorted(self.shadows, key=lambda s: s.born):
            if s.failures >= c['h_f'] or (s.rhits >= 64 and s.damage > c['delta'] * s.rhits):
                self._retire(s)
        ready = [s for s in self.shadows if s.wins >= c['h_r'] and s.damage <= c['delta'] * s.rhits]
        if ready:
            s = sorted(ready, key=lambda s: (-s.wins, s.born))[0]
            self._promote(s)
            return True
        return False

    def _apply(self, ebar, chat, norm2):
        self.R = self.R + np.outer(chat, ebar) / norm2
        self.te -= np.outer(self.tc @ chat / norm2, ebar)       # keeps every o = e + c R unchanged
        self.ops += D_C * D_O + (self.cap + 1) * (D_C + D_O)

    def _promote(self, s):
        self._apply(s.ebar, s.chat, s.norm2)
        gid = s.group
        for i in s.fit:
            self.pinned[i] = False
        self.shadows.remove(s)
        for i in self._members(gid, 'res'):
            if self.pinned[i]:
                self.res[i] = -1
            else:
                self._drop_trace(i)
        self.groups.pop(gid, None)
        self.log['promoted'] += 1

    def _count_promote(self):
        """`count`: a residual group promotes on support alone, with no validation, once it could have proposed."""
        c = self.cfg
        for g in sorted((g for g in self.groups.values() if g.kind == 'res'), key=lambda g: g.born):
            if g.episodes < c['h_r']:
                continue
            fit = self._fit_set(g.id, False)
            if fit is None:
                continue
            ebar, chat, norm2 = self._fit(fit)
            if np.sqrt(norm2) < c['kappa']:
                continue
            self._apply(ebar, chat, norm2)
            for i in self._members(g.id, 'res'):
                self._drop_trace(i)
            self.groups.pop(g.id, None)
            self.log['promoted'] += 1
            return True
        return False

    def _after_reader_change(self):
        """Step 4."""
        live = np.flatnonzero(self.alive)
        if len(live):
            P = self.tc[live] @ self.R
            self.ops += len(live) * D_C * D_O
            for i, p in zip(live, P):
                if self.res[i] >= 0 and self.hit(p, self.te[i] + p) and not self.pinned[i]:
                    rg = int(self.res[i])
                    self.res[i] = -1
                    self._delete_if_empty(rg)
        for s in list(self.shadows):
            keep = [i for i in s.fit if self.alive[i] and not self.hit(self.tc[i] @ self.R, self.te[i] + self.tc[i] @ self.R)]
            for i in s.fit:
                self.pinned[i] = False
                if i not in keep:
                    self.used[i] = True
                    rg = int(self.res[i])
                    self.res[i] = -1
                    self._delete_if_empty(rg)
            top = self._fit_set(s.group, True, keep)
            if top is None:
                s.fit = []
                self._retire(s)
                continue
            s.fit = top
            s.ebar, s.chat, s.norm2 = self._fit(top)
            for i in top:
                if not self.used[i]:
                    self.fits_of.setdefault(int(self.serial[i]), []).append(s.id)
                self.pinned[i] = True
                self.used[i] = True
            s.wins = s.failures = s.damage = s.rhits = 0
            s.last_win = self.t - self.cfg['E']
            self.log['refits'] += 1

    def _fit(self, idx):
        chat = self.tc[idx].mean(0)
        self.ops += len(idx) * (D_C + D_O)
        return self.te[idx].mean(0), chat, max(float(chat @ chat), 1e-12)

    def _fit_set(self, gid, refit, keep=()):
        """The oldest m unused members from distinct episodes, after any traces kept for a refit of the same shadow."""
        m, E = self.cfg['m'], self.cfg['E']
        chosen = list(keep)
        members = self._members(gid, 'res')
        if len(members) + len(chosen) < m:
            return None
        for i in members[np.argsort(self.age[members])]:
            if len(chosen) >= m:
                break
            if i in chosen or self.used[i]:
                continue
            if all(abs(int(self.age[i]) - int(self.age[j])) >= E for j in chosen):
                chosen.append(int(i))
        return chosen if len(chosen) >= m else None

    def _propose(self):
        """Step 7: one proposal per event at most, oldest group first."""
        c = self.cfg
        idx = self.res[self.alive & ~self.used & (self.res >= 0)]
        if not len(idx):
            return
        ids, cnt = np.unique(idx, return_counts=True)
        big = [self.groups[int(i)] for i, n in zip(ids, cnt) if n >= c['m']]
        for g in sorted((g for g in big if g.shadow is None), key=lambda g: g.born):
            fit = self._fit_set(g.id, False)
            if fit is None:
                continue
            ebar, chat, norm2 = self._fit(fit)
            if np.sqrt(norm2) < c['kappa']:
                continue
            if len(self.shadows) >= c['S']:
                self._retire(sorted(self.shadows, key=lambda s: (s.wins, s.born))[0], 'displaced')
            s = Shadow()
            s.id, s.group, s.ebar, s.chat, s.norm2, s.fit = self.next_sid, g.id, ebar, chat, norm2, fit
            s.wins = s.failures = s.damage = s.rhits = 0
            s.last_win, s.born = self.t - c['E'], self.t
            self.next_sid += 1
            self.shadows.append(s)
            g.shadow = s.id
            g.tries += 1
            for i in fit:
                self.pinned[i] = self.used[i] = True
                self.fits_of.setdefault(int(self.serial[i]), []).append(s.id)
            self.log['proposed'] += 1
            return

    def _sketch(self, c, e):
        cue = int(((self.pc @ c) > 0).astype(np.uint32) @ (1 << np.arange(32, dtype=np.uint32)))
        res = int(((self.pe @ e) > 0).astype(np.uint32) @ (1 << np.arange(32, dtype=np.uint32)))
        match = False
        if self.sk_n:
            rho = self.cfg['rho']
            a = np.bitwise_count(self.sk_c[:self.sk_n] ^ cue)
            b = np.bitwise_count(self.sk_e[:self.sk_n] ^ res)
            match = bool((a <= rho).any() or (b <= rho).any())
            self.ops += 2 * self.sk_n
        self.sk_c[self.sk_head], self.sk_e[self.sk_head] = cue, res
        self.sk_head = (self.sk_head + 1) % self.Z
        self.sk_n = min(self.sk_n + 1, self.Z)
        if match:
            self.log['sketch_matches'] += 1
        return match

    def _evict_to_budget(self):
        budget = self.trace_budget_now()
        while int(self.alive.sum()) > budget:
            i = self._eviction_choice()
            if i is None:
                raise RuntimeError('every trace is pinned: this configuration should have been rejected')
            self._drop_trace(i)
            self.log['evicted'] += 1

    def _support(self, i):
        out = 0
        for gid in (int(self.same[i]), int(self.res[i])):
            g = self.groups.get(gid)
            if g is not None:
                out = max(out, g.episodes)
        return out

    def _eviction_choice(self):
        idx = np.flatnonzero(self.alive & ~self.pinned)
        if not len(idx):
            return None
        return min((int(i) for i in idx), key=lambda i: (self._support(i), bool(self.familiar[i]), int(self.age[i])))

    def _group_for(self, c, e):
        """Step 6's two memberships. A trace with no match founds its own singleton group, which holds no record until it has two members."""
        same = self._eligible_same(c)[1]
        idx = np.flatnonzero(self.alive & (self.res >= 0))
        if not len(idx):
            return same, None
        ids, inv = np.unique(self.res[idx], return_inverse=True)
        if self.group_by == 'residual':
            X, q, floor = self.te[idx], unit(e), self.cfg['tau_e']
            self.ops += len(idx) * D_O
        else:
            X, q, floor = self.tc[idx], c, self.cfg['tau_g']
            self.ops += len(idx) * D_C
        sums = np.zeros((len(ids), X.shape[1]), np.float32)
        np.add.at(sums, inv, X)
        cos = unit(sums) @ q
        j = int(cos.argmax())
        return same, self.groups[int(ids[j])] if cos[j] >= floor else None

    def _keep(self, c, o, e):
        i = int(np.flatnonzero(~self.alive)[0])
        same, res = self._group_for(c, e)
        self.tc[i], self.te[i], self.alive[i], self.age[i] = c, e, True, self.t
        self.pinned[i] = self.used[i] = False
        self.serial[i] = self.next_serial
        self.next_serial += 1
        self.familiar[i] = self._sketch(c, e) if self.Z else False
        if same is None:
            same = Group(self.next_gid, 'same', self.t)
            self.next_gid += 1
            self.groups[same.id] = same
        elif self.t - same.last >= self.cfg['E']:
            same.episodes += 1
            same.last = self.t
        same.norm = None
        self.same[i] = same.id
        if res is None:
            res = Group(self.next_gid, 'res', self.t)
            self.next_gid += 1
            self.groups[res.id] = res
        elif self.t - res.last >= self.cfg['E']:
            res.episodes += 1
            res.last = self.t
        self.res[i] = res.id

    def step(self, c, o):
        self.t += 1
        self.events += 1
        if o is None and not self.self_count:
            self.ops += (self.known_c.shape[0] + self.n_new) * D_C + D_C * D_O
            return
        eligible = self._eligible_same(c)[0]                       # step 1: from the cue alone
        Rc = c @ self.R
        self._score(c, o, Rc, eligible)                            # step 2
        changed = self._decide(eligible)                           # step 3
        if changed:
            self._after_reader_change()                            # step 4
        if o is None:
            return
        if self.hit(self._reader_or_slots(c), o):                  # step 5
            self._evict_to_budget()
            return
        self._keep(c, o, o - c @ self.R)                           # step 6
        if self.validate:
            self._propose()                                        # step 7
        self._evict_to_budget()                                    # step 8

    def persistent_bytes(self):
        return {'reader': READER_BYTES, 'known': KNOWN_BYTES, 'traces': int(self.alive.sum()) * TRACE, 'new_slots': self.n_new * SLOT,
                'shadow_places': self.cfg['S'] * SHADOW, 'sketch': self.Z * SKETCH_ENTRY}


class OracleLife(ReaderArm):
    """`life-oracle`: hidden cause identity replaces grouping and routing, with the same stores and fits. Noise is never kept, so it shows
    how much room perfect sorting has. Rules apply after m traces; bindings write a slot after h_b later occurrences."""

    def __init__(self, R, sc, so, cfg, B_kib):
        super().__init__(R, sc, so, cfg['tau'], cfg['theta'], 'life-oracle')
        self.cfg = cfg
        a = allocation(B_kib, cfg['S'], False)
        self.slot_cap = a['slot_cap']
        self.new_c = np.zeros((self.slot_cap, D_C), np.float32)
        self.new_o = np.zeros((self.slot_cap, D_O), np.float32)
        self.n_new = 0
        self.pending = {}

    def told(self, c, o, kind, cause):
        self.events += 1
        if o is None:
            return
        if kind == K['rule']:
            cs = self.pending.setdefault(cause, [])
            cs.append((c, o))
            if len(cs) == self.cfg['m']:
                C = np.stack([x for x, _ in cs])
                O = np.stack([y for _, y in cs])
                E = O - C @ self.R
                chat = C.mean(0)
                self.R = self.R + np.outer(chat, E.mean(0)) / max(float(chat @ chat), 1e-12)
        elif kind in (K['binding'], K['exception'], K['rare']):
            cs = self.pending.setdefault(cause, [])
            cs.append((c, o))
            if len(cs) == self.cfg['h_b'] + 1 and self.n_new < self.slot_cap:
                self.new_c[self.n_new] = unit(np.stack([x for x, _ in cs]).mean(0))
                self.new_o[self.n_new] = np.stack([y for _, y in cs]).mean(0)
                self.n_new += 1

    def answers(self, C):
        out = super().answers(C)
        if self.n_new:
            s = C @ self.new_c[:self.n_new].T
            use = s.max(1) >= self.tau
            out[use] = self.new_o[:self.n_new][s.argmax(1)[use]]
        return out

    def persistent_bytes(self):
        return {'reader': READER_BYTES, 'known': KNOWN_BYTES, 'new_slots': self.n_new * SLOT}


# ---------------------------------------------------------------- readings

def noise_capture(answers_of, keys_decode, targets_by_kind, derange):
    """Mean over kinds of P(answer = the cue's last outcome) minus P(answer = another cue's last outcome), over a fixed derangement.
    answers_of[kind]: decoded outcome per cue (cues, renderings). targets_by_kind[kind]: outcome per cue."""
    parts = {}
    for kind, a in answers_of.items():
        t = targets_by_kind[kind]
        d = derange[kind]
        parts[kind] = float((a == t[:, None]).mean() - (a == t[d][:, None]).mean())
    return float(np.mean(list(parts.values()))), parts


def derangement(rng, n):
    if n < 2:
        return np.zeros(n, int)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p
