# Conflict loop v1: confusion triggers thinking in the logic route, with a learned arbiter

Status: **contract frozen 2026-11-06 on the user's instruction to draft and run** (no separate review round). Frozen
after a smoke run that only checked the runner.

## User direction (2026-11-06)

After [dual-route truth v1](dual-route-truth-v1-results.md):
- keep separate codes for the two routes;
- add a learned arbiter;
- "Can we use conflict or confusion to trigger more loops in the logic route?"

## Why

- **Separate codes keep intuition as good as the built-in unit** (grandparent 1.00, size 0.97).
- **The two-scalar logic route ranks unseen chain facts well** (AUROC 0.93–0.98) but stays cautious: low support,
  not T.
- **Graded disagreement flags intuition's errors** (AUROC 0.74–0.77).
- **No fixed rule combines** intuition's chains with logic's unknown.
- **The loop of [multi-step reader v1](multistep-reader-v1-results.md) never stepped,** because one pass fitted every
  training statement.

## Design

### Routes

- **Intuition:** its own codes; one pass gives the verdict u; trained on all trained individual T/F statements, as in
  dual-route v1.
- **Logic:** its own codes. Step 0 is the one-pass support (s⁺, s⁻) of dual-route v1, with the same taught parts: kind
  quantities and fill-the-tail through a code head, connectives as an operator on support pairs. **Logic also has a
  loop.**

### The loop (logic route only)

```
conflict d0 = |σ(u) − s⁺_0| + |(1 − σ(u)) − s⁻_0|            (u detached: logic does not move intuition)
if d0 > τ (0.5):
    state from [c_a, c_b, r];  m_0 = c_a
    for t = 1..3:
        ρ_t  = softmax(W_ρ state) · R_L                         choose a relation to follow
        m_t  = rescale(code head(m_{t−1}, ρ_t, MASK, tail))     PRODUCE the middle thing (Option 2)
        ρ'_t = softmax(W_ρ' state) · R_L
        s_t  = logic support(m_t, ρ'_t, c_b)
        d_t  = disagreement of s_t with σ(u)
        λ_t  = σ(w_h · [state, d_t])                            stop? (sees the current disagreement)
        state = GRU(state, [m_t, ρ_t])
    output support = PonderNet mixture of s_0 … s_3
else:
    output support = s_0
```

The step structure is the one accepted in multi-step reader v1. Nothing tells the loop which relation to follow, what
middle thing to produce, or how many steps to take. The trigger is the only new rule: steps are spent only on
conflicts.

### Pressure: facts logic must work out

The chain and higher-order relations are:
- `grandparent-of`, `ancestor-of`, `kind-of`;
- `bigger-than`, `smaller-than`;
- `named-can`, `named-has`, `named-kind`;
- `same-colour-as`.

**Half of their trained statements** (T and F, drawn per seed) are **withheld from logic's direct training**. Intuition
still learns them. Their base relations (`parent-of`, `is-a`, `subkind-of`, `just-bigger-than`, `names`, `can`, `has`,
`colour-of`) are always told.

| Row | Logic's loss |
|---|---|
| Told, no conflict | support at step 0 |
| Told, conflict | the loop's mixture over t = 0…3 |
| Withheld, conflict | the loop's mixture over t = 1…3 only; step 0 is never trained on it |
| Withheld, no conflict | none (logic already agrees with intuition on its own) |

So a withheld fact can be fitted only by stepping, and only when logic and intuition disagree.

### Arbiter

A small network: 7 inputs → 16 → 3 (T, F, U). Its inputs:
- σ(u);
- logic's final s⁺ and s⁻;
- d0 and the final disagreement;
- the expected number of steps;
- whether the loop was triggered.

**It is trained on the validation statements only** (5% of trainable individual statements, T/F/U), which neither
route was trained on: Adam, 500 steps, class-balanced cross-entropy. So it learns when to trust which route from cases
like new questions.

## Arms

| Arm | Loop | Withheld half | Arbiter |
|---|---|---|---|
| `arbiter` | none (logic is one-pass) | no | yes |
| `conflict-loop` | yes | **yes** | yes |
| `conflict-loop-told` | yes | no (everything told) | yes |

The reference is dual-route v1's `dual-separate` with fixed rules. Seeds 450–453 (the same splits), 12 runs.
Statements as in multi-step reader v1. Training: full batch, Adam lr 3e-3, weight decay 1e-4, 3,000 steps;
τ = 0.5, T = 3, the PonderNet geometric prior (p = 0.5, weight 0.01).

## Reads

- **Does the loop work?**
  - Trigger rate in training and test.
  - Withheld facts: logic's step-0 answer against its answer after the loop (are they derived?).
  - Expected steps by chain length.
  - The first produced code against the true middle thing (word → its animal; anonymous individual → its class;
    grandparent → a shared child).
- **Answers:** on every T/F test, the intuition verdict, logic at step 0, logic after the loop, and the arbiter.
  Balanced accuracy (unknown is wrong) and AUROC.
- **Unknown:** S4a, P7, "sometimes" on new individuals (S3b), the dax, and the U rate on inferable facts, for logic and
  the arbiter.
- **Conflict:** d0 as an error signal for intuition, as in dual-route v1.

## Predictions

1. **[multi-step reader v1; derived]** `conflict-loop-told` rarely triggers in training (< 5% of rows) and gains
   nothing from its loop. `conflict-loop` triggers on most withheld T facts and fits them after the loop (≥ 0.8,
   against step 0 ≤ 0.6).
2. **[derived]** On held-out chains, `conflict-loop`'s logic after the loop beats its step 0 by ≥ 0.10. The first
   produced code is the true middle thing ≥ 0.5 on the named relations.
3. **[derived]** The arbiter keeps intuition's chain accuracy within 0.05 and says unknown on ≥ 0.5 of held-out unknown
   colours, with ≤ 0.05 unknown on inferable facts. This is the combination no fixed rule achieved.
4. **[taught truth v1]** "Sometimes → unknown" for new individuals improves only through the arbiter (logic alone
   stays ≤ 0.3).
5. **[thinking-loop pilot]** Risk: intuition is unaffected (separate codes); logic's inheritance may drop (≥ 0.05).

## Prior work

As in [dual-route truth v1](dual-route-truth-v1.md) and [multi-step reader v1](multistep-reader-v1.md). Added:
- **Adaptive computation triggered by difficulty or uncertainty:** ACT and PonderNet halt on learned signals
  **[familiar]**. Conflict monitoring in cognitive control (Botvinick et al. 2001) proposes that detected response
  conflict recruits more control **[familiar]**. Here the trigger is the disagreement between two routes.
- **Learning to defer / mixtures of experts:** an arbiter choosing between predictors is a gating network (Jacobs et
  al. 1991) or a learned deferral rule **[familiar]**. It is trained here on held-out statements, as a calibration set.
- **Withholding as pressure:** the program's principle "constraints are pressure" (local). This is a curriculum that
  makes some answers reachable only by steps.

## Scale

- Steps run only on conflicts: about 10–15% of questions in dual-route v1. Inference cost is one pass per route, plus
  ≤ 3 loop steps on that fraction.
- The arbiter is 7 → 16 → 3, negligible.

## Computation (declared)

- Like dual-route v1 (about 16 CPU min per run at 6 workers), plus loop steps on triggered rows. Estimate 20–30 CPU min
  per run; 12 runs on 6 workers ≈ 1 h wall.
- Progress logged every 250 steps.
- Stored: reports; checkpoints in `runs/` (not tracked). No tuning: τ, T, the withheld fraction and the arbiter's size
  are fixed here.

## Commands

```
EV_WORKERS=6 .venv/bin/python -m experiments.conflict_loop main      # rerun resumes
.venv/bin/python -m experiments.conflict_loop summary
```

## Implementation (recorded at freezing)

- **Runner:** `experiments/conflict_loop.py`. It reuses dual-route v1's `Dual` (separate codes) for the routes, the
  logic support and the code head.
- **The trigger** compares logic's step-0 support with intuition's σ(u) (no gradient into intuition). In training it is
  recomputed every step, so it fires on many rows early, while both routes are untrained, and on few later. The log
  records the count every 250 steps.
- **Withheld rows with a conflict** are trained on the halting mixture over steps 1–3 (step 0 excluded); told rows with
  a conflict on 0–3. The KL to the geometric prior is taken over the steps used.
- **The arbiter** is trained after the routes, on validation individual statements (T, F and U), with the routes frozen.
- **Parameters:** 107,017 (all arms carry the loop and arbiter modules; `arbiter` does not use the loop).
- **Smoke run** (60 steps, 3 runs, 3 workers): 42 s wall. Early-training triggering is heavy (11,498 rows at step 60
  in `conflict-loop-told`). Projected at most about 30 min per run; 12 runs on 6 workers ≈ 1–1.5 h.
