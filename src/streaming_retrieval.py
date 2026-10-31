"""Streaming retrieval v1 (docs/streaming-retrieval-v1-contract.md): schedules, retrieval methods and scores.

Every method reads a stream of unit-norm queries against P stored unit-norm keys, one frame at a time, and outputs a decision
per frame: a stored index, or REJECT. Nothing here looks ahead, and nothing depends on torch.
"""
import numpy as np

REJECT = -1
MIN_DWELL, MAX_DWELL = 32, 64
P_UNFAMILIAR, P_BURST = 0.2, 0.5
BURST_LENGTHS = (1, 2, 4)
BURST_KINDS = ('intrude', 'flash', 'corrupt')
ONSET_CLEAR, AFTER_CLEAR = 16, 8  # a burst starts at least 16 frames into a dwell and ends at least 8 before its end
STABLE = 3  # frames a decision must hold to count as reached
GRACE = 16  # frames after an onset excluded from the settled-state readings


# ---------------------------------------------------------------- schedules

def make_schedule(rng, n_words, n_unfamiliar, frames):
    """truth[t]: the dwell's word, or REJECT in an unfamiliar dwell. shown[t]: the symbol on screen, word i as i, unfamiliar j as
    n_words + j. corrupt[t]: burst-level image corruption. Transient bursts never change truth."""
    truth, shown, corrupt, segments, bursts = [], [], [], [], []
    t, prev = 0, None
    while t < frames:
        dwell = int(rng.integers(MIN_DWELL, MAX_DWELL + 1))
        if prev is not None and prev != REJECT and rng.random() < P_UNFAMILIAR:
            word, sym = REJECT, n_words + int(rng.integers(n_unfamiliar))
        else:
            word = int(rng.integers(n_words))
            while word == prev:
                word = int(rng.integers(n_words))
            sym = word
        seg_shown, seg_corrupt = np.full(dwell, sym), np.zeros(dwell, bool)
        if word != REJECT and rng.random() < P_BURST:
            length = int(rng.choice(BURST_LENGTHS))
            kind = str(rng.choice(BURST_KINDS))
            start = int(rng.integers(ONSET_CLEAR, dwell - length - AFTER_CLEAR + 1))
            if kind == 'intrude':
                other = int(rng.integers(n_words - 1))
                seg_shown[start:start + length] = other + (other >= word)
            elif kind == 'flash':
                seg_shown[start:start + length] = n_words + int(rng.integers(n_unfamiliar))
            else:
                seg_corrupt[start:start + length] = True
            bursts.append((t + start, length, kind))
        truth.append(np.full(dwell, word))
        shown.append(seg_shown)
        corrupt.append(seg_corrupt)
        segments.append((t, t + dwell, word))
        t += dwell
        prev = word
    return {'truth': np.concatenate(truth), 'shown': np.concatenate(shown), 'corrupt': np.concatenate(corrupt),
            'segments': segments, 'bursts': bursts}


# ---------------------------------------------------------------- methods: each returns (index, score) or decisions

def threshold(index, score, tau):
    return np.where(score >= tau, index, REJECT)


def nearest(S):
    return S.argmax(1), S.max(1)


def ema(S, a):
    """Exponential smoothing of the similarities. Linear, so it equals nearest-key on the smoothed query: its state can be D floats."""
    out = np.empty_like(S)
    s = S[0].copy()
    for t, row in enumerate(S):
        s += a * (row - s)
        out[t] = s
    return out.argmax(1), out.max(1)


def hysteresis(S, a, tau, delta):
    """EMA with a held decision: a held word yields only to another word ahead by delta, or falls below tau - delta."""
    y = np.empty(len(S), int)
    s = S[0].copy()
    cur = REJECT
    for t, row in enumerate(S):
        s += a * (row - s)
        best = int(s.argmax())
        if cur == REJECT:
            if s[best] >= tau:
                cur = best
        elif s[cur] < tau - delta:
            cur = best if s[best] >= tau else REJECT
        elif s[best] - s[cur] > delta:
            cur = best
        y[t] = cur
    return y


def hmm_filter(S, kappa, hazard, tau):
    """Forward filter over P words plus an unfamiliar state: von Mises-Fisher evidence kappa * similarity, constant evidence
    kappa * tau for the unfamiliar state, and a uniform switch at rate hazard. Reference, not one of the mechanisms compared."""
    P = S.shape[1]
    y = np.empty(len(S), int)
    logp = np.full(P + 1, -np.log(P + 1), dtype=S.dtype)
    floor = np.log(hazard / (P + 1))
    keep = np.log1p(-hazard)
    for t, row in enumerate(S):
        logp = np.logaddexp(logp + keep, floor)
        logp[:P] += kappa * row
        logp[P] += kappa * tau
        logp -= logp.max()
        best = int(logp.argmax())
        y[t] = REJECT if best == P else best
    return y


def idp(sim, patterns, gain, tau, steps, sigma, rng):
    """Input-driven Hopfield dynamics, after Betteti et al. 2025, Eqs. 7-8 and 19: dx = (-x + W(u) psi(x)) dt + sigma dB, with
    W(u) = (1/N) sum_mu alpha_mu xi xi^T, alpha_mu = gain * (similarity_mu)^2 and psi = tanh(5x), integrated by Euler-Maruyama with
    `steps` steps over a frame of duration tau. patterns: (P, N) with squared norm N. The dense matrix is never formed.
    Returns (index, score): the stored pattern with the largest absolute overlap m_mu = xi^T psi(x) / N, and that overlap."""
    T, P = sim.shape
    N = patterns.shape[1]
    dt = tau / steps
    noise = sigma * np.sqrt(dt)
    x = rng.standard_normal(N, dtype=np.float32).astype(patterns.dtype)
    alpha = gain * sim ** 2
    index, score = np.empty(T, int), np.empty(T, patterns.dtype)
    for t in range(T):
        a = alpha[t]
        for _ in range(steps):
            m = patterns @ np.tanh(5 * x) / N
            x = x + dt * (patterns.T @ (a * m) - x)
            if noise:
                x = x + noise * rng.standard_normal(N, dtype=np.float32)
        m = np.abs(patterns @ np.tanh(5 * x)) / N
        index[t] = m.argmax()
        score[t] = m[index[t]]
    return index, score


def to_patterns(keys):
    """Unit keys as patterns with squared norm equal to their width, the normalization of Eq. 8."""
    return keys * np.sqrt(keys.shape[1])


def expand(keys_or_queries, projection):
    """Unit-norm image under a fixed Gaussian projection (a query) and its sign pattern (a stored key)."""
    z = keys_or_queries @ projection.T
    return z / np.linalg.norm(z, axis=-1, keepdims=True), np.sign(z)


# ---------------------------------------------------------------- multiplications per frame (P keys of width D; N the dynamics width)

def ops(method, P, D, steps=0, N=None):
    read = P * D
    if method == 'nearest':
        return read
    if method == 'ema':
        return read + P
    if method == 'hysteresis':
        return read + P
    if method == 'hmm':
        return read + 3 * P
    if method == 'idp':
        N = N or D
        project = 0 if N == D else N * D
        cos = P * N if N != D else read
        return project + cos + P + steps * (2 * P * N + P + 2 * N) + P * N
    raise ValueError(method)


def state_floats(method, P, D, N=None):
    return {'nearest': 0, 'ema': D, 'hysteresis': D + 1, 'hmm': P + 1, 'idp': N or D}[method]


# ---------------------------------------------------------------- scores

def _reached(hit):
    """Index of the first run of STABLE true values in a boolean array, or None."""
    if len(hit) < STABLE:
        return None
    runs = np.convolve(hit.astype(int), np.ones(STABLE, int), 'valid')
    where = np.flatnonzero(runs == STABLE)
    return int(where[0]) if len(where) else None


def score(y, sch):
    """The contract's readings for one decision sequence."""
    y = np.asarray(y)
    truth = sch['truth']
    T = min(len(y), len(truth))
    y, truth = y[:T], truth[:T]
    changed = np.r_[False, y[1:] != y[:-1]]
    wrong = changed & (y != REJECT) & (y != truth)

    window = np.zeros(T, bool)
    captures = {k: [] for k in BURST_KINDS}
    captures_len = {n: [] for n in BURST_LENGTHS}
    for start, length, kind in sch['bursts']:
        end = min(start + length + AFTER_CLEAR, T)
        if start >= T:
            continue
        window[start:end] = True
        hit = bool(wrong[start:end].any())
        captures[kind].append(hit)
        captures_len[length].append(hit)
    wrong_familiar, delays, misses, reject_delays, reject_miss = [], [], [], [], []
    settled_familiar, settled_unfamiliar = np.zeros(T, bool), np.zeros(T, bool)
    for a, b, word in sch['segments']:
        if a >= T:
            break
        b = min(b, T)
        if word == REJECT:
            settled_unfamiliar[a + GRACE:b] = True
            d = _reached(y[a:b] == REJECT)
            reject_delays.append(b - a if d is None else d)
            reject_miss.append(d is None)
        else:
            settled_familiar[a + GRACE:b] = True
            if a > 0:
                d = _reached(y[a:b] == word)
                delays.append(b - a if d is None else d)
                misses.append(d is None)
    calm = settled_familiar & ~window
    return {
        'wrong_per_1000': 1000 * float(wrong.sum()) / T,
        'wrong_outside_bursts_per_1000': 1000 * float((wrong & ~window).sum()) / T,
        'capture_by_kind': {k: float(np.mean(v)) if v else None for k, v in captures.items()},
        'capture_by_length': {str(k): float(np.mean(v)) if v else None for k, v in captures_len.items()},
        'delay_mean': float(np.mean(delays)),
        'delay_median': float(np.median(delays)),
        'delay_miss': float(np.mean(misses)),
        'accept_unfamiliar': float((y[settled_unfamiliar] != REJECT).mean()),
        'reject_delay_mean': float(np.mean(reject_delays)),
        'reject_miss': float(np.mean(reject_miss)),
        'dropout': float((y[calm] == REJECT).mean()),
        'settled_correct': float((y[calm] == truth[calm]).mean()),
        'frame_correct': float((y == truth).mean()),
        'frames': int(T), 'changes': len(delays), 'unfamiliar_dwells': len(reject_delays), 'bursts': len(sch['bursts']),
    }
