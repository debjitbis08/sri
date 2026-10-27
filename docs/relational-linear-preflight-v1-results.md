# Relational linear preflight v1: results

Results for the frozen [contract](relational-linear-preflight-v1-contract.md),
run 2026-10-26 at revision e268ba1. First deliverable of the
[relational abstraction proposal](relational-abstraction-learning-proposal.md).

Files:

- [Runner](../experiments/relational_linear_preflight.py).
- Results: `reports/relational-linear-preflight-v1-results.json.gz`
  (gzip -n -9; 1.38 MB uncompressed, SHA-256 a028e3f9…c68faa9).
- Fitted P1 parameters:
  [`reports/relational-linear-preflight-v1-params.json`](../reports/relational-linear-preflight-v1-params.json)
  (SHA-256 2f151dba…e80601).
- [Runtime](../reports/relational-linear-preflight-v1-runtime.json).

No movement file. Reproduce with
`.venv/bin/python -m experiments.relational_linear_preflight` (about 72
minutes on 11 workers).

The runtime file marks the tree dirty. The only change was an unrelated
untracked notes file created during the run; the runner and contract hashes
match the committed revision.

## Summary

| Criterion | Outcome |
| --- | --- |
| A1 generator | **passes** in 10/10 worlds |
| C1 representability | **witness found** for both families, both κ and both clip settings (10/10 worlds each) |
| C2 binding invariance | **invariant** at the frozen threshold in every condition (10/10 worlds, J_G ≤ 2.7e-4); the residual drift is not negligible at L = 128 (see C2) |
| C3 inference | **does not agree** in any condition (0/10 worlds); not a gate |
| C4 design audit | **qualifying gap** for both families, concentrated at small budgets; see the caveats |
| Outcome | **proceed** to the learning/transfer contract, by the project decision rule |

- **The existing three-module chain, with per-action banks, can represent
  long-horizon action-conditioned prediction on this fixture.** It does so
  at the learner's ±2 clips. At L = 128 the privileged multi-horizon search
  reaches a median avoidable error of 1.3e-6 to 3.2e-5 across the ±2-clip
  conditions, with no clips active. The ±2 values are 3.1e-5 (κ = 1) and
  1.3e-6 (κ = 5) for family A, and 3.2e-5 and 5.3e-6 for family B.
- **Freezing the banks approximately freezes the effective dynamics.**
  Refitting the bindings for a new source domain or the target leaves the
  eight effective operators close to the source's up to one shared basis
  change (J_G ≤ 2.7e-4, with a well-conditioned alignment). The remaining
  drift is not negligible at long horizons. A post-hoc check found that
  the worst eigenvalue-modulus gap is 0.011-0.019 (median by condition, up to
  0.044). Compounded to L = 128, that is an amplitude factor of e^1 to e^2 at
  the median (see C2).
- **Fitting on observed one-step and window errors does not give
  long-horizon accuracy.** This holds for both the one-step diagnostic and
  the P2 binding fits, though both use the same architecture. Median
  avoidable error at L = 128 is 0.07-0.18, against 1e-6 to 3e-5 for the
  multi-horizon search. The learning contract has to address this objective
  mismatch. In the P2 fits, encoder/decoder error and the residual operator
  drift above have not been separated.
- **32-step settling falls far short at prior 0.001.** Prediction-time
  settling with the P1 capacity parameters was tested. After 32 steps the
  relative activity error is 0.41-0.55 (median by condition). The settled
  32-step rollout has avoidable error 0.97-1.0. Reaching relative error 1e-3
  takes 175-898 steps across all conditions. Training-time clamped settles
  were not measured.

## A1 generator audit (P0)

All 10 worlds pass:

- **Observability:** rank 2 for each action and for mixed sequences.
- **Rendering conditioning:** at most 8.9.
- **Precision:** norm drift over 128 steps is at most 6.4e-15.
- **Commutants:** dimension 1 for family A, family B and the broken
  target, and 2 for the two-rotation control.

Certified mismatch bounds:

| Scramble | J | J_u (with the probe) | Threshold |
| --- | --- | --- | --- |
| Family A swap | min 0.10 | min 0.44 | 0.05: detectable in 10/10 |
| Broken target | min 0.032 | min 0.061 | 0.005: detectable in 10/10 |
| Family A, same-eigenvalue replacement (100 draws) | quartiles 0.018 / 0.095 / 0.28 | — | 92% ≥ 0.005 by J, 100% by J_u |
| Family B swap | equivalent in 10/10 | detectable in 9/10 | world 8 unresolved (J_u = 0.049) |

The family B result is the one the [draft checks](relational-linear-preflight-v1-contract.md#prior-work)
predicted. Without the probe, a basis change undoes the swap exactly. With
the probe it is detectable in every world but one, where the bound falls
just under the threshold.

## C1 capacity witness (P1)

Median over worlds of X_L (avoidable error), for family A, κ = 1, ±2 clips:

| Arm | L = 1 | 2 | 8 | 32 | 128 | max at L = 128 | Leakage ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constructive witness | 1.1e-3 | 7.1e-4 | 8.3e-4 | 9.2e-4 | 2.1e-3 | 1.6e-2 | 0.096 |
| Multi-horizon search | 1.0e-7 | 1.2e-6 | 1.3e-6 | 4.6e-6 | 3.1e-5 | 3.9e-4 | 0.012 |
| One-step diagnostic | 7.7e-4 | 6.9e-4 | 1.6e-3 | 8.5e-3 | 0.12 | 49 | 0.020 |

The other conditions are similar:

- **κ = 5:** the multi-horizon median at L = 128 is 1.3e-6.
- **±6 clips:** the constructive witness improves (leakage 0.017). The
  multi-horizon search reaches 1.7e-6 to 7.3e-6 across the four ±6
  conditions; 3.6e-6 for family A at κ = 1.
- **Family B:** the constructive median at L = 128 is 4.1e-3, and the
  multi-horizon median 3.2e-5.

Points to note:

- **The constructive witness alone fails in worlds 2 and 5** at every
  setting: X₁₂₈ is 0.010-0.026, just over the 1e-2 threshold. The
  multi-horizon search passes in all 10 worlds everywhere, so C1 rests on
  the search in those two worlds.
- **The draft's clip prediction was specific to its construction.** The
  draft predicted leakage-dominated error near 5e-3 at ±2 clips. The
  constructive encoder W = [R, βB] did leak (ρ ≈ 0.10). The search, however,
  found an encoder with ρ ≈ 0.012 inside the clips, with no clip active and
  a decay penalty unchanged at 77. The tension was a property of that
  construction, not of the architecture.
- **Diagnostics:**
  - factorization residuals ≤ 1.8e-15 in every arm;
  - effective-operator moduli at 1 in the constructive and search arms;
  - back-reaction of the second action ≤ 4e-5;
  - composition consistency (G² against G¹·G¹): median 0.0028;
  - coupled-encoder discrepancy: median 0.007.
- **The objective mismatch is large.** The one-step diagnostic's error
  grows with horizon. At L = 128 it is about 4,000 times the search's
  median, and in the worst world X₁₂₈ = 49: one-step fitting can produce
  rollouts that grow without bound. This is the
  composition-of-one-step-optima effect the review predicted.
- **Learning rates:** the multi-horizon search selected η = 0.001 in most
  worlds and 0.003 in the rest. 0.001 is the grid's lower edge; the grid was
  not extended. The one-step diagnostic's selections span the whole grid.

## C2 binding invariance (P2)

The banks were frozen from the multi-horizon search in every world, since
it passed C1 everywhere. The bank tensors were verified unchanged after
every fit.

| Condition (family A, κ = 1, ±2) | J_G max | J_G median | Alignment condition, max | X₁ median | X₁₂₈ median |
| --- | --- | --- | --- | --- | --- |
| Source s2 | 1.1e-4 | 4.4e-5 | 2.1 | 9.0e-4 | 0.078 |
| Target t | 2.2e-4 | 9.4e-5 | 1.7 | 1.2e-3 | 0.15 |
| Broken target | 0.13 | 0.046 | 5.7 | 2.1e-3 | 0.85 |

- **Every condition is invariant** in 10/10 worlds at the frozen
  threshold. J_G is a squared-residual bound, and passing it does not mean
  the drift is negligible at L = 128.
- **The source of the long-horizon error is not established.** The binding
  fits use observed window and probe errors only, so their rollouts carry
  the same objective mismatch as the one-step diagnostic. Within the J_G
  tolerance, however, the effective operators' spectra still move.
  - Example: in world 0, the target fit has J_G = 1.3e-4, yet one two-step
    operator has eigenvalue modulus 0.986, about 0.40 after 64 repetitions.
  - A post-hoc check (not in the contract) compared each fit's eight
    operators with the frozen source's
    ([script](../experiments/relational_linear_preflight_posthoc.py), which
    reads only the stored results). Results are in the table below.
  - Both encoder/decoder error and operator drift can therefore contribute.
    Separating them needs an ablation: roll out each fit's encoder and
    decoder with the source operators transported by the J_G alignment, and
    compare. That ablation needs the P2 fitted parameters, which were not
    saved, so it is not done here.

| Post-hoc, by target (all 8 conditions) | Worst modulus gap, median by condition | Worst log-amplitude gap at L = 128, median by condition | X₁₂₈ median by condition |
| --- | --- | --- | --- |
| Source s2 | 0.012-0.019 | 1.1-2.0 | 0.071-0.16 |
| Target t | 0.011-0.018 | 1.3-1.6 | 0.12-0.16 |

The gap is the largest over operators and eigenvalues of
|log(|λ_fit| / |λ_source|)| × (128 for G¹, 64 for G²). It is an upper
envelope for one eigen-direction, not a rollout decomposition. Family B
fits, whose X₁₂₈ is larger for s2, also have larger gaps (about 2 against
1.1-1.4 for family A).
- **Frozen banks cannot absorb broken structure.** Against the
  broken-structure target the refit leaves J_G at about 0.05 and the error
  reaches 0.85 at L = 128. For family B, P2's broken target is family A's
  broken pair (a different family), and its errors are larger still (X₁₂₈
  ≈ 1).

## C3 inference (P3)

- **32 steps is far too few.** After 32 implicit-Euler steps the relative
  activity error has median 0.41-0.55 by condition, and the settled 32-step rollout has
  avoidable error 0.97-1.0, so it predicts essentially nothing.
- **Steps needed to reach relative error 1e-3:**
  - family A: 194-572 at ±2 clips, 287-898 at ±6;
  - family B: 175-611 at ±2 clips, 329-851 at ±6.

P3 tests prediction-time settling with module 0 clamped and the remaining
activities free, using the P1 capacity parameters. It does not measure the
convergence of training's clamped states, which have more units clamped.
The result is consistent with the
[prior-strength study](prior-strength-training-v1-results.md)'s concern about
training settles: at prior 0.001, prediction-time inference on this window
needs hundreds of steps. It does not show that a learner using 32-step
training settles has badly unconverged states.
Whether that is explained by the smallest free-Hessian eigenvalue is the
hypothesis in the [GDL notes](geometric-deep-learning-notes.md); it is not
tested here.

## C4 direct-reference design audit (P4)

The qualifying (n, L) sets, with ratio ≥ 1.5 and ordering in ≥ 9/10
worlds:

- **Family A:**
  - n = 4 and 8: every horizon;
  - n = 16: L ∈ {2, 32, 64, 128};
  - n = 32: L ∈ {32, 64, 128};
  - n = 64: L ∈ {64, 128}.
- **Family B:**
  - n = 4 and 8: every horizon;
  - n = 16: L = 128;
  - n = 32: L ∈ {64, 128}.

Median X_L over worlds, family A:

| n | Target-only, L = 1 / 8 / 128 | Frozen-dynamics, L = 1 / 8 / 128 |
| --- | --- | --- |
| 4 | 0.37 / 1.05 / 1.3e8 | 0.060 / 0.071 / 0.39 |
| 16 | 0.019 / 0.19 / 1.6e3 | 0.014 / 0.019 / 0.37 |
| 64 | 0.0056 / 0.027 / 11 | 0.0044 / 0.010 / 0.37 |
| 256 | 0.0029 / 0.011 / 0.48 | 0.0032 / 0.0085 / 0.37 |

Two caveats limit how far C4 can be used:

- **The ratios of means are inflated by divergent target-only fits.** At
  small budgets the target-only operators often have |eig| > 1 (for
  example 1.1-2.5 at n = 4), so long-horizon errors overflow. The criterion
  is met as frozen, and the per-world ordering is robust, but the ratio
  magnitudes (up to infinity) mean little. The medians above are the
  better summary.
- **The frozen-dynamics reference is capped by attenuation.** Its source F̂
  has eigenvalue moduli 0.85-0.998 across worlds, which is the
  errors-in-variables shrinkage noted in the smoke test. Its X₁₂₈ therefore
  plateaus at about 0.37 whatever the budget. At n ≥ 128, target-only
  overtakes it at most horizons (ratios 0.36-1.15).

So the gap is real at small budgets. At large budgets and long horizons it
says more about this reference's bias than about what transfer can achieve.
The SRI witness (median X₁₂₈ of 1.3e-6 to 3.2e-5 with exact source dynamics) shows how much
better frozen dynamics can do.

## Computation

| Phase | Wall | Worker CPU |
| --- | --- | --- |
| Verification and P0 | 1 s | (parent) |
| P1 and P3 (80 tasks) | 50 min | 31,618 s (median 414 s per task) |
| P2 (80 tasks) | 21 min | 13,784 s (median 187 s per task) |
| P4 (20 tasks) | 30 s | 284 s |
| **Total** | **72 min** | **45,687 s (12.7 CPU-hours)** |

- **Counts:** 2.4 million Adam steps, 9.6 million window factorizations,
  300,000 ALS sweeps and 37,160 settling steps.
- **Peak RSS:** 334 MB in the largest worker.
- **Environment:** float64 on CPU (PyTorch 2.14.1+cpu, NumPy 2.5.4), 11
  workers with one thread each.
- **Against the estimate:** the contract expected "tens of minutes on one
  core". The measured cost is about 20-40 times that, dominated by the P1
  searches and P2 restarts. Each Adam step's cost was timed only for the
  multi-horizon objective in isolation (30 ms); the other searches and the
  P2 fits were not timed in advance.

## Implementation notes

- **Gradient check.** The runner's gradient verification uses a
  fourth-order central stencil (h = 1e-3); its relative error is 2.6e-9.
  The two-point stencil at h = 1e-6 had rounding error up to 1.9e-5 on
  components with small gradients. Both are central differences, and the
  contract's tolerance (1e-6) is met.
- **Extra diagnostics.** P4 additionally reports the latent references'
  operator eigenvalue moduli, which explain the attenuation above. The
  criteria are unchanged.
- **Constructive scaling.** The constructive witness scales R by
  α = min(1, clip / max|R|) and V by 1/α. In this run α = 1 in every world.

## Implications for the learning/transfer contract

1. **Representability is not the bottleneck; the objective, and possibly
   residual operator drift, is.** Every observed-data fit (the one-step
   diagnostic and the P2 binding fits) reaches X₁₂₈ ≈ 0.1. The
   clamped-energy learner sees only windows, so it should be expected to do
   no better. The contract needs:
   - a multi-step or simulation-error control, as the proposal already
     plans;
   - consideration of a declared gain or orthogonality arm (see the
     [GDL notes](geometric-deep-learning-notes.md) on forward vanishing);
   - the multi-horizon search's error as the capacity ceiling to report
     against;
   - fitted binding parameters saved, and the operator-substitution
     ablation from C2, to separate encoder/decoder error from drift;
   - a spectral or multi-horizon check of dynamics transfer alongside J_G,
     because a passing J_G still allows consequential drift.
2. **Use exact inference as primary, and cost iterative inference
   separately.** At prior 0.001 prediction-time settling needs hundreds of
   steps. Training-time clamped settles were not measured. Any learner that
   uses iterative settles should report their convergence directly, rather
   than assume 32 steps suffice.
3. **Budgets and horizons.** Use n ∈ {4, 8, 16, 32} transitions with
   horizons up to 128, where the direct references show a gap. Report
   target-only medians as well as ratios, because target-only fits can
   diverge.
4. **The direct references need a better transfer reference.** The latent
   frozen-dynamics reference is biased by errors-in-variables attenuation.
   A post-hoc, bias-corrected estimator (for example instrumental-variable
   or total least squares fitting of the consistency term) should be
   considered as an additional comparator. It would be declared in the new
   contract; this result is not changed.
5. **Scrambles.** The family A swap and same-eigenvalue draws certified as
   detectable are available. Family B's swap is detectable only through the
   probe.
