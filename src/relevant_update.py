"""Relevant update v1 (docs/relevant-update-v1-contract.md), stage 1: update rules, handed sets of experiences as keys only.

A rule sees a set as (C, O), k cue keys and k outcome keys, and answers cues. Trials, contaminants and outcome tables stay with the caller.
"""
import numpy as np

from src import surprise_lifecycle as sl

D_C, D_O = sl.D_C, sl.D_O
SLOT_TAU = .8
ENTRY = (D_C + D_O) * 4


def unit(x):
    return sl.unit(x)


def dual_ridge(C, E, lam):
    """argmin_D sum ||e_i - c_i D||^2 + lam ||D||^2, by the k x k dual. Returned with shape (D_C, D_O), prediction c @ D."""
    G = C @ C.T + lam * np.eye(len(C), dtype=C.dtype)
    return (C.T @ np.linalg.solve(G, E)).astype(np.float32)


class LinearRule:
    """A reader R, known slots and an update written into R."""

    def __init__(self, base, cfg):
        self.R = base['R'].copy()
        self.sc, self.so = base['slot_c'], base['slot_o']
        self.cfg = cfg
        self.ops = 0

    def answers(self, C):
        out = C @ self.R
        s = C @ self.sc.T
        use = s.max(1) >= SLOT_TAU
        out[use] = self.so[s.argmax(1)[use]]
        return out

    def residual(self, C, O):
        self.ops += len(C) * D_C * D_O
        return O - C @ self.R

    def apply(self, D):
        self.R = self.R + D
        self.ops += D_C * D_O

    def extra_bytes(self, L, k=None):
        return 0

    def begin_trial(self):
        pass


class RankOne(LinearRule):
    def update(self, C, O):
        E = self.residual(C, O)
        chat, ebar = C.mean(0), E.mean(0)
        self.ops += len(C) * (D_C + D_O)
        self.apply(np.outer(chat, ebar) / max(float(chat @ chat), 1e-12))


class Ridge(LinearRule):
    def solve(self, C, E):
        n = len(C)
        self.ops += n * n * D_C + n ** 3 + n * D_C * D_O
        return dual_ridge(C, E, self.cfg['lam'])

    def update(self, C, O):
        self.apply(self.solve(C, self.residual(C, O)))


class RidgeCycled(Ridge):
    """k passes over the set, a single-example ridge step for each experience in turn: the matched control for `ridge`."""

    def update(self, C, O):
        for _ in range(len(C)):
            for i in range(len(C)):
                c, o = C[i:i + 1], O[i:i + 1]
                self.apply(self.solve(c, self.residual(c, o)))


class RidgeRehearse(Ridge):
    """A fixed buffer of old events, built once. Each pulls the update towards keeping or restoring its own answer."""

    def __init__(self, base, cfg):
        super().__init__(base, cfg)
        idx = even_indices(base['rule_labels'], cfg['r'], base['buffer_order'])
        self.bc, self.bo = base['rule_C'][idx], base['rule_O'][idx]

    def fit_with(self, C, E, bc, bo):
        sw = np.sqrt(self.cfg['w'])
        Eb = self.residual(bc, bo)
        return self.solve(np.vstack([C, sw * bc]), np.vstack([E, sw * Eb]))

    def update(self, C, O):
        self.apply(self.fit_with(C, self.residual(C, O), self.bc, self.bo))

    def extra_bytes(self, L, k=None):
        return len(self.bc) * ENTRY


class RidgeKeep(RidgeRehearse):
    """As `ridge-rehearse`, but every new set is kept too and all of it is refit on each update, under the current R. Its bytes grow with what it has seen."""

    def __init__(self, base, cfg):
        super().__init__(base, cfg)
        self.kc, self.ko = np.zeros((0, D_C), np.float32), np.zeros((0, D_O), np.float32)

    def update(self, C, O):
        self.kc, self.ko = np.vstack([self.kc, C]), np.vstack([self.ko, O])
        sw = np.sqrt(self.cfg['w'])
        E = self.residual(self.kc, self.ko)
        Eb = self.residual(self.bc, self.bo)
        self.apply(self.solve(np.vstack([self.kc, sw * self.bc]), np.vstack([E, sw * Eb])))

    def extra_bytes(self, L, k=None):
        return (len(self.bc) + len(self.kc)) * ENTRY


class RidgeInterfered(RidgeRehearse):
    """Of a fixed buffer of B old events, the r a tentative `ridge` update damages most enter the fit; ties to the lower index."""

    def __init__(self, base, cfg):
        super().__init__(base, {**cfg, 'r': cfg['B']})
        self.r = cfg['r']

    def update(self, C, O):
        E = self.residual(C, O)
        D0 = self.solve(C, E)
        before = ((self.bo - self.bc @ self.R) ** 2).sum(1)
        after = ((self.bo - self.bc @ (self.R + D0)) ** 2).sum(1)
        self.ops += 2 * len(self.bc) * D_C * D_O
        top = np.argsort(-(after - before), kind='stable')[:self.r]
        self.apply(self.fit_with(C, E, self.bc[top], self.bo[top]))


class RidgeOrthogonal(Ridge):
    """The update's cue side projected away from a fixed basis of old cue directions. Only the basis is kept."""

    def __init__(self, base, cfg):
        super().__init__(base, cfg)
        self.U = base['eigvecs'][:, :cfg['q']]

    def update(self, C, O):
        D = self.solve(C, self.residual(C, O))
        self.ops += 2 * self.U.shape[1] * D_C * D_O
        self.apply(D - self.U @ (self.U.T @ D))

    def extra_bytes(self, L, k=None):
        return self.U.shape[1] * D_C * 4


class Expert(LinearRule):
    """Local gated memory: R stays; a cue close enough to a stored mean cue adds that expert's mean residual. Highest cosine wins, the newest on a tie."""

    def __init__(self, base, cfg):
        super().__init__(base, cfg)
        self.xc, self.xe = [], []

    def update(self, C, O):
        E = self.residual(C, O)
        self.ops += len(C) * (D_C + D_O)
        self.xc.append(unit(C.mean(0)))
        self.xe.append(E.mean(0))

    def answers(self, C):
        out = C @ self.R
        if self.xc:
            X, Y = np.stack(self.xc), np.stack(self.xe)
            s = C @ X.T
            best = s.shape[1] - 1 - s[:, ::-1].argmax(1)
            use = s.max(1) >= self.cfg['tau_x']
            out[use] += Y[best[use]]
        s = C @ self.sc.T
        use = s.max(1) >= SLOT_TAU
        out[use] = self.so[s.argmax(1)[use]]
        return out

    def extra_bytes(self, L, k=None):
        return L * (k or 1) * ENTRY


def even_indices(labels, n, order):
    """n events with equal numbers per known rule, in a fixed drawn order."""
    classes = np.unique(labels)
    per = n // len(classes)
    return np.concatenate([order[labels[order] == c][:per] for c in classes])


class NeuralRule:
    """An MLP memory. S persists across a trial and is zeroed at its start. alpha is decay: (1 - alpha) M."""

    def __init__(self, net, cfg):
        self.net, self.cfg = net, cfg
        self.alpha, self.lr = cfg['alpha'], cfg['eta']
        self.beta = cfg.get('beta', 0.)
        self.ops = 0
        self.S = None

    def answers(self, C):
        return self.net.forward(C)

    def extra_bytes(self, L, k=None):
        return 0

    def begin_trial(self):
        self.S = [np.zeros_like(p) for p in self.net.params()]

    def step(self, C, O, lr, decay):
        self.ops += 2 * len(C) * self.net.count()
        g = self.net.grad(C, O)
        new = []
        for j, (p, gg) in enumerate(zip(self.net.params(), g)):
            self.S[j] = self.beta * self.S[j] - lr * gg.astype(np.float32)
            new.append(decay * p + self.S[j])
        self.net.set(new)


class Titans(NeuralRule):
    def update(self, C, O):
        for i in range(len(C)):
            self.step(C[i:i + 1], O[i:i + 1], self.lr, 1 - self.alpha)


class Omega(NeuralRule):
    """On the arrival of a set's j-th experience, one step on the mean of the j so far."""

    def update(self, C, O):
        for j in range(1, len(C) + 1):
            self.step(C[:j], O[:j], self.lr, 1 - self.alpha)


class OmegaSeq(NeuralRule):
    """The matched control: where `omega` takes one step on j experiences at rate eta, j single steps at eta/j, on the same j, with the same total decay."""

    def update(self, C, O):
        for j in range(1, len(C) + 1):
            for i in range(j):
                self.step(C[i:i + 1], O[i:i + 1], self.lr / j, (1 - self.alpha) ** (1 / j))


# ---------------------------------------------------------------- trials

def rule_outcomes(rng, tables):
    """A fixed outcome per rule family, uniform over the outcomes known rules do not use."""
    r = sl.symbol_ranges()
    known = {o for _, o in tables['known_rules']}
    free = np.array([o for o in r['outcomes'] if o not in known])
    return {int(f): int(rng.choice(free)) for f in np.r_[r['new'], r['background']]}


def make_trial(rng, table, L, k, p=0., kind=None):
    """Symbol indices for one trial: (L, k) families, items and outcomes, plus the families the trial leaves unused."""
    r = sl.symbol_ranges()
    fams = np.array(sorted(table))
    order = rng.permutation(fams)
    used, unused = order[:L], order[L:]
    F = np.repeat(used[:, None], k, 1)
    O = np.array([[table[int(f)]] * k for f in used])
    pairs = np.array([[rng.choice(r['items'], 2, replace=False) for _ in range(k)] for _ in range(L)])
    n_bad = int(round(p * k))
    contaminated = np.zeros((L, k), bool)
    if n_bad:
        assert kind in ('unrelated', 'same-outcome', 'same-family')
        for a in range(L):
            for b in rng.choice(k, n_bad, replace=False):
                contaminated[a, b] = True
                if kind != 'same-family':
                    F[a, b] = rng.choice(r['binding'])
                if kind == 'unrelated':
                    O[a, b] = rng.choice(r['outcomes'])
                elif kind == 'same-family':
                    O[a, b] = rng.choice([x for x in r['outcomes'] if x != table[int(used[a])]])
    render = rng.integers(0, 1 << 15, (L, k, 4))
    return {'families': used, 'unused': unused, 'F': F, 'I': pairs[..., 0], 'J': pairs[..., 1], 'O': O, 'render': render, 'contaminated': contaminated}
