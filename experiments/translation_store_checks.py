"""Checks of the translation-memory components on synthetic data (no host, no rendering).

Usage: .venv/bin/python -m experiments.translation_store_checks
"""
import numpy as np
import torch

from experiments import translation_store as TS


def check_map():
    rng = np.random.default_rng(0)
    d, K = 16, 64
    Y = torch.from_numpy(rng.standard_normal((K, d)))
    A_true = torch.eye(d, dtype=torch.float64) + 0.2 * torch.from_numpy(rng.standard_normal((d, d)))
    b_true = torch.from_numpy(rng.standard_normal(d)) * 0.3
    X = torch.linalg.solve(A_true, (Y - b_true).T).T  # Y = A X + b exactly
    m = TS.ReferenceMap(d, 1e-6)
    m.set_reference(Y)
    assert m.residual(X) > 0.01  # identity before fitting
    m.fit(X)
    assert m.residual(X) < 1e-6, m.residual(X)
    big = TS.ReferenceMap(d, 1e6); big.set_reference(Y); big.fit(X)
    assert torch.allclose(big.A[:, :-1], torch.eye(d), atol=1e-3)  # heavy regularization stays at the identity
    syms = np.repeat(np.arange(32), 2)
    folds = [set(range(i, 32, 4)) for i in range(4)]
    e_small = TS.heldout_anchor_error(X, Y, syms, 1e-6, m.s2, folds)
    e_big = TS.heldout_anchor_error(X, Y, syms, 1e6, m.s2, folds)
    assert e_small < 1e-6 < e_big, (e_small, e_big)
    print(f'map: exact affine recovered; held-out error {e_small:.1e} (small lambda) against {e_big:.2f} (identity)')


def check_associative():
    rng = np.random.default_rng(1)
    m_s = m_t = 512
    k_s = k_t = 10
    mem = TS.Associative(m_s, m_t, k_s, k_t, rng)
    n = 120
    S = TS.codebook(n, m_s, k_s, rng)
    cb = TS.codebook(150, m_t, k_t, rng)
    tgt = torch.from_numpy(rng.permutation(150)[:n])
    mem.write(S, cb[tgt])
    assert mem.W.diagonal().abs().sum() == 0 and torch.equal(mem.W, mem.W.T)
    out, info = mem.retrieve(S)
    assert torch.equal(out[1], mem.one_step(S))
    for t, tp in out.items():
        acc = float((TS.target_scores(tp, cb).argmax(1) == tgt).float().mean())
        assert acc == 1.0, (t, acc)
    # corrupted cues: replace 4 of the 10 active source units
    noisy = S.clone()
    for i in range(n):
        on, off = noisy[i].nonzero()[:, 0], (noisy[i] == 0).nonzero()[:, 0]
        noisy[i, on[rng.choice(len(on), 4, replace=False)]] = 0
        noisy[i, off[rng.choice(len(off), 4, replace=False)]] = 1
    out, info = mem.retrieve(noisy)
    accs = {t: float((TS.target_scores(tp, cb).argmax(1) == tgt).float().mean()) for t, tp in out.items()}
    print(f'associative: exact cues 1.00 at every checkpoint; 40% corrupted cues {accs}; converged {float(info["converged"].float().mean()):.2f}, '
          f'cycles {float(info["cycle"].float().mean()):.2f}, mean steps {float(info["steps"].float().mean()):.1f}')


def check_flat():
    rng = np.random.default_rng(2)
    S = TS.codebook(5, 64, 4, rng)
    fs = TS.FlatSparse(4, 10, 4)
    fs.write(S[:3], [0, 1, 2])
    fs.write(S[:1], [0]); assert fs.refreshes == 1
    fs.write(S[1:2], [5]); assert fs.overwrites == 1 and fs.values[1] == 5
    sc, win = fs.read(S[:3])
    assert sc[0, 0] == 1 and sc[1, 5] == 1 and int(win[2]) == 2
    fs.count_use(torch.tensor([0, 1, 2]))
    fs.write(S[3:4], [3]); fs.write(S[4:5], [4])  # capacity 4: the fifth key evicts the least used (slot 3, unused)
    assert fs.evictions == 1 and 3 not in fs.values and 4 in fs.values
    z = torch.from_numpy(rng.standard_normal((3, 16))).float()
    st = TS.FlatStore(10, 10)
    st.write(z, [7, 8, 9])
    sc, _ = st.read(z)
    assert torch.equal(sc.argmax(1), torch.tensor([7, 8, 9]))
    print('flat stores: duplicate refresh, overwrite, least-used eviction and exhaustive reads behave as specified')


def check_gate_and_sentences():
    rng = np.random.default_rng(3)
    T, n = 20, 200
    labels = torch.from_numpy(rng.integers(T, size=n))
    old = torch.arange(n) < n // 2  # old items: host wrong and unsure, memory right and confident
    p_host = torch.full((n, T), 0.5 / (T - 1))
    wrong = (labels + 1) % T
    p_host[torch.arange(n), torch.where(old, wrong, labels)] = 0.5
    p_host[old] = torch.full((T,), 1. / T)
    mem = torch.zeros(n, T)
    mem[torch.arange(n), torch.where(old, labels, wrong)] = torch.where(old, 1., .2)
    f = TS.features(p_host, mem, torch.ones(n))
    g = TS.Gate()
    g.fit(f, TS.p_memory(mem), p_host, labels, 300, 0.1)
    ans, w = g.answer(f, TS.p_memory(mem), p_host)
    assert float((ans == labels).float().mean()) == 1.0 and float(w[old].mean()) > 0.5 > float(w[~old].mean())
    cb = TS.codebook(T, 64, 4, rng)
    words = torch.from_numpy(rng.integers(T, size=(64, 3)))
    rule = TS.SentenceRule()
    rule.fit(cb[words], words.flip(1), cb, 200, 0.1)
    assert torch.equal(TS.SentenceRule.logits(rule, cb[words], cb).argmax(2), words.flip(1))
    print('gate: routes old items to memory and new to host; sentence rule learns reversal')


if __name__ == '__main__':
    torch.manual_seed(0)
    check_map()
    check_associative()
    check_flat()
    check_gate_and_sentences()
