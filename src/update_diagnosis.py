"""Local update diagnosis v1: neighbourhoods, rule variants and evaluator-only update diagnostics.

See docs/local-update-diagnosis-v1-contract.md. The battery, IB supervision, projected simulation
service and v1 block learner are imported unchanged. Variants change exactly one property of the v1
rule: neighbourhood (SCOPE), updates per experience (REPEAT) or selection with stored earlier
feedback (REPLAY, a diagnostic comparator). Classification and reachability functions here are
pure and receive evaluator data only; nothing in them is called by a learner.
"""
from collections import deque
import hashlib
import itertools

from src.mechanism_learning import BLOCKS, N_VECTORS, hypothesis_index, vector


VALUES = (-3, 0, 3)
ALL_VECTORS = [vector(h) for h in range(N_VECTORS)]
VARIANTS = ('B0', 'SCOPE', 'REPEAT', 'REPLAY')


def _block_values(block):
    return list(itertools.product(VALUES, repeat=len(block)))


def n1(w):
    """The v1 neighbourhood: 26 single-block edits, in v1 enumeration order."""
    out = []
    for block in BLOCKS:
        for vals in _block_values(block):
            cand = list(w)
            for j, v in zip(block, vals):
                cand[j] = v
            if tuple(cand) != w:
                out.append(tuple(cand))
    return out


def n2(w):
    """All 240 edits changing exactly two blocks (each changed block takes a different value)."""
    out = []
    for b1, b2 in itertools.combinations(BLOCKS, 2):
        for v1 in _block_values(b1):
            if v1 == tuple(w[j] for j in b1):
                continue
            for v2 in _block_values(b2):
                if v2 == tuple(w[j] for j in b2):
                    continue
                cand = list(w)
                for j, v in zip(b1, v1):
                    cand[j] = v
                for j, v in zip(b2, v2):
                    cand[j] = v
                out.append(tuple(cand))
    return out


def blocks_differ(w, u):
    return sum(any(w[j] != u[j] for j in block) for block in BLOCKS)


def changed(w, u):
    return sum(a != b for a, b in zip(w, u))


def class_distance(w, members):
    """d(w): blocks differing from the nearer member of the teacher class."""
    return min(blocks_differ(w, m) for m in members)


# ---------------------------------------------------------------- learners

class VariantLearner:
    """The v1 block learner with exactly one property changed, per `variant`.

    Learner-facing inputs are only (query, feedback map) and the simulation service. `history`
    exists only for REPLAY and holds the feedback maps it was given.
    """
    def __init__(self, sim, variant):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.sim, self.variant = sim, variant
        self.w = (0,) * 7
        self.history = []
        self.evaluations = self.history_predictions = 0
        self.candidate_losses = self.decisions = self.commits = self.coefficients_changed = 0

    def loss(self, w, x, feedback):
        total = 0
        for key, f in feedback.items():
            total += len(self.sim.predict(w, x, key) ^ f)
            self.evaluations += 1
        return total

    def past_loss(self, w):
        total = 0
        for x, fb in self.history:
            for key, f in fb.items():
                total += len(self.sim.predict(w, x, key) ^ f)
                self.history_predictions += 1
        return total

    def neighbours(self):
        return n1(self.w) + (n2(self.w) if self.variant == 'SCOPE' else [])

    def decide(self, x, feedback):
        """One update decision. Returns (committed, before, after)."""
        before = self.w
        self.decisions += 1
        cands = [self.w] + self.neighbours()
        self.candidate_losses += len(cands)
        losses = [self.loss(c, x, feedback) for c in cands]
        current = losses[0]
        if self.variant == 'REPLAY':
            past = [self.past_loss(c) for c in cands]
            eligible = [(past[i], losses[i], changed(c, before), c) for i, c in enumerate(cands) if losses[i] < current]
            if not eligible:
                return False, before, before
            best = min(eligible)[3]
        else:
            best_key = min((losses[i], changed(c, before), c) for i, c in enumerate(cands))
            if best_key[0] >= current:
                return False, before, before
            best = best_key[2]
        self.w = best
        self.commits += 1
        self.coefficients_changed += changed(best, before)
        return True, before, best

    def update(self, x, feedback):
        """One experience: one decision, or (REPEAT) decisions until none commits."""
        trace = []
        while True:
            committed, before, after = self.decide(x, feedback)
            trace.append(dict(before=hypothesis_index(before), after=hypothesis_index(after), committed=committed))
            if not committed or self.variant != 'REPEAT':
                break
        if self.variant == 'REPLAY':
            self.history.append((x, feedback))
        return trace


# ---------------------------------------------------------------- pure diagnostics (evaluator data)

def b0_choice(w, loss):
    """The v1 decision at w under loss(u): the committed vector, or None."""
    best = min((loss(c), changed(c, w), c) for c in [w] + n1(w))
    return best[2] if best[0] < loss(w) else None


def classify(w, loss, broken, neighbourhood, chosen, scope_n2=False):
    """E0/E1a/E1b/E2a/E2b/E2c for one decision at state w.

    loss(u): current-example loss; broken(u): previously correct items that u gets wrong;
    neighbourhood: the decision's allowed edits; chosen: the committed vector or None.
    `scope_n2` marks a decision whose neighbourhood already contains N2 (SCOPE).
    """
    l0 = loss(w)
    if l0 == 0:
        return dict(cls='E0', l0=0)
    improving = [u for u in neighbourhood if loss(u) < l0]
    if not improving:
        if not scope_n2 and any(loss(u) < l0 for u in n2(w)):
            return dict(cls='E1a', l0=l0)
        lower = [u for u in ALL_VECTORS if loss(u) < l0]
        if not lower:
            raise AssertionError('no lower-loss vector although the clean teacher has zero loss')
        return dict(cls='E1b', l0=l0, min_blocks_to_lower=min(blocks_differ(w, u) for u in lower))
    if chosen is None:
        raise AssertionError('an improving edit exists but none was committed')
    b = broken(chosen)
    rec = dict(l0=l0, chosen_loss=loss(chosen), chosen_breaks=b, chosen_exact=loss(chosen) == 0)
    if b == 0:
        return dict(rec, cls='E2a')
    preserving = [u for u in improving if broken(u) == 0]
    if preserving:
        best = min(loss(u) for u in preserving)
        return dict(rec, cls='E2b', best_preserving_loss=best,
                    e2b_kind='tie_selection' if best == loss(chosen) else 'tradeoff')
    return dict(rec, cls='E2c')


def proximity(w, loss, members):
    """Section 4.2 proximity diagnostics at w (outside the teacher class)."""
    d0, l0 = class_distance(w, members), loss(w)

    def kind(u):
        lu = loss(u)
        return 'improving' if lu < l0 else 'neutral' if lu == l0 else 'worse'
    out = dict(d=d0)
    for name, cands, need in (('N1', n1(w), 1), ('N2', n2(w), 2)):
        toward = [u for u in cands if class_distance(u, members) <= d0 - need]
        counts = {'improving': 0, 'neutral': 0, 'worse': 0}
        for u in toward:
            counts[kind(u)] += 1
        best = min(toward, key=lambda u: (loss(u), changed(u, w), u)) if toward else None
        out[name] = dict(edits=len(toward), counts=counts, best=None if best is None else kind(best))
    return out


def reverse_reachable(losses, adjacency, targets, strict):
    """Exact reverse BFS: every state with a path to `targets` whose edges never increase loss
    (strictly decrease if `strict`). Returns {state: next hop toward the targets} (targets map to None)."""
    nxt = {t: None for t in targets}
    q = deque(targets)
    while q:
        u = q.popleft()
        for w in adjacency[u]:
            if w in nxt:
                continue
            if (losses[w] > losses[u]) if strict else (losses[w] >= losses[u]):
                nxt[w] = u
                q.append(w)
    return nxt


def witness(nxt, h):
    path = [h]
    while nxt[path[-1]] is not None:
        path.append(nxt[path[-1]])
    return path


def set_hash(states):
    return hashlib.sha256(','.join(map(str, sorted(states))).encode()).hexdigest()


def repetition_probe(w, loss, broken_items):
    """Repeat the v1 decision on one fixed example until no strict improvement.

    broken_items(u) returns the set of previously correct items u gets wrong. Returns the
    commits, the intermediate states, and items lost at any intermediate commit.
    """
    states, lost_any = [w], set()
    cur = w
    while True:
        nxt = b0_choice(cur, loss)
        if nxt is None:
            break
        cur = nxt
        states.append(cur)
        lost_any |= broken_items(cur)
    return dict(commits=len(states) - 1, decisions=len(states), candidate_losses=27 * len(states),
                states=[hypothesis_index(s) for s in states], terminal=cur,
                lost_at_any_commit=sorted(lost_any), wrong_at_end=sorted(broken_items(cur)))


__all__ = ['VARIANTS', 'VariantLearner', 'b0_choice', 'blocks_differ', 'changed', 'class_distance', 'classify',
           'n1', 'n2', 'proximity', 'repetition_probe', 'reverse_reachable', 'set_hash', 'witness']
