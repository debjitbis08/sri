# Correlated nuisance v1: does inner coupling filter non-persistent surface correlations?

Prospective contract, 2026-10-26. Frozen before execution. Follows the
[clamped-energy ablation](clamped-energy-ablation-v1-results.md) and tests
fixture 2 of the [inner-coupling research direction](inner-coupling-hypothesis.md).

In the ablation, uniform coupling beat inner-only coupling by about 4× in
avoidable error. The project hypothesis is that surface coupling wins there
because the fixture's only shared structure is a direct linear relation
between surfaces. This study adds nuisance that is correlated across modules
during training and then breaks at test. It asks which coupling topologies
transmit that correlation between modules, and whether a module coupled only
at its inner layer is more robust when the correlation stops holding.

The modules stay linear, so this is not a test of abstraction. It tests one
mechanism the hypothesis relies on: that a narrow inner channel carries the
structure modules share and leaves out surface correlations that do not
persist.

## Design provenance and contamination

No probes of learning informed this design. The oracle values quoted below
come from the fixture alone, by derivation and a Monte Carlo check of the
derivation (200,000 samples, no learner), recorded under Prior work.

In the independent regime the fixture is bit-identical to v1's fixture, whose
results on seeds 0-22 are published. Seeds 0-2 are the pilot again.
Confirmatory claims use new seeds 23-32. Seed 3 is used once, only for the
reproduction check in Verification.

## Prior work

Reviewed under [literature review for contracts](literature-review.md).
Shared survey and sources: [prior work](prior-work.md). Tags: **[read]**,
**[abstract]**, **[familiar]**, **[derived]**, **[analogy]**, as defined
there.

- **Related methods.**
  - Inner-only coupling makes every cross-module prediction pass through a
    2-dimensional channel per module. In a linear model, that resembles
    reduced-rank regression (Izenman 1975) [familiar; analogy]. Reduced-rank
    regression under squared loss, solved exactly, keeps the directions of
    largest fitted variance [familiar]. A rank constraint alone does not
    imply that: it depends on the objective and on optimization succeeding.
    This contract does not establish that local clamped-energy learning
    implements reduced-rank regression, so a reduced-rank regression
    comparator is included to test the analogy directly.
  - Breaking a training-time correlation at test is a spurious-correlation
    shift, the setting of invariant risk minimization (Arjovsky et al. 2019)
    and anchor regression (Rothenhäusler et al. 2021) [familiar]. Anchor
    regularization has been added to reduced-rank regression for
    out-of-distribution robustness ([Durand et al., AISTATS 2025](https://proceedings.mlr.press/v258/durand25a.html))
    [abstract].
  - Uniform versus inner coupling resembles early versus late fusion in
    multimodal learning: combining raw features versus combining deep
    representations ([Baltrušaitis et al. 2017](https://arxiv.org/pdf/1705.09406))
    [abstract; analogy]. Surveys report trade-offs, not a general ranking.
- **What theory already predicts.**
  - *Trade-off [derived].* With correlated nuisance, the in-distribution
    Bayes predictor of a hidden module includes the part of its nuisance
    predictable from the observed modules. When the correlation breaks, the
    Bayes predictor is the signal alone. No predictor reaches zero avoidable
    error on both test sets. The two oracles below mark the ends of the
    trade-off.
  - *Variance ranking (hypothesis, motivated by the reduced-rank analogy).*
    If the inner channel behaves like exact reduced-rank regression, it keeps
    signal over nuisance only while the signal's directions carry more
    predictable power. Per pattern coordinate, the signal's uncentered second
    moment is E[a²]/2 ≈ 0.51; its centered variance is 0.51 − 0.25 ≈ 0.26.
    The energy has no bias terms, so if the analogy holds, the uncentered
    moment is the relevant one for the energy model; the comparator's
    intercept makes the centered variance relevant for it [derived]. The
    shared nuisance has zero mean, with variance per direction 0.027 (weak)
    and 1.0 (strong). By either measure, signal dominates in the weak regime
    and nuisance in the strong regime. The hypothesis is therefore that
    inner coupling filters nuisance in the weak regime and not in the strong
    one. This is not derived for the clamped-energy objective. The strong
    regime tests it and is not expected to favour inner coupling.
  - *Quantitative contest [derived from the ablation].* In the ablation, at
    exact equilibrium, uniform L/2 had avoidable error 0.0010 and inner L/10
    had 0.0043. In the weak regime, a predictor that transmits all
    predictable nuisance pays 0.0034 on the decorrelated test. If those
    figures carry over, uniform and inner coupling would be close on that
    test. The direction of the result is largely predictable; its size is not.
  - *Prior results on linear spurious correlation [abstract].* For ridge
    regression, spurious correlation learned and in-distribution loss trade
    off, and the regularization that minimizes test loss lies where the
    learned spurious correlation is increasing
    ([Bombari & Mondelli 2025](https://arxiv.org/abs/2502.01347)).
    Alignment objectives fail when nuisance is strongly correlated across
    views ([Kamai et al. 2026](https://arxiv.org/abs/2606.11190)) [abstract].
    Both concern estimators or objectives, not this local energy rule.
- **Known failure modes.** A low-rank channel captures nuisance when nuisance
  dominates (strong regime; included on purpose). Inner-only learning was
  slow in the ablation and selected an edge learning rate. This design
  extends the grid downward and makes exact-equilibrium error primary.
- **Baselines.**
  - Ridge on all waking traces, which can exploit the correlation in
    distribution.
  - Rank-2 reduced-rank regression, the non-local estimator the inner
    channel is compared with by analogy.
  - Two evaluator-only oracles. They establish attainable reference errors
    at the two ends of the trade-off; they do not test the analogy.
  - Frozen baselines per coupling.
- **What this adds.** An internal test of whether this architecture's
  topology controls what is shared between modules, under a shift the
  learner never sees. These searches found no study of the question for
  local energy learning across coupled modules, which is not a novelty claim.

## Fixture

v1's `fixture(seed)` supplies the pattern maps M_m (8×2, orthonormal),
amplitudes, labels and the independent nuisance e_m. It is called unchanged,
with stream SeedSequence([seed, 2002]). A second stream,
SeedSequence([seed, 3003]), supplies:

1. **Shared-nuisance directions.** Draw g = normal(size=(3, 8, 2)). For each
   module, remove its pattern span, g_m ← g_m − M_m M_mᵀ g_m, and set
   U_m = Q from the QR factorization of g_m. U_m is 8×2, orthonormal and
   orthogonal to M_m.
2. **Shared latents.** In v1's sampling order (waking 16+16, test 12+12,
   references 8+8), draw z = normal(size=(n, 2)) for each block.
3. **Decorrelated latents** for the decorrelated test set. After everything
   else, draw z_dec = normal(size=(24, 3, 2)), independently per module.

Observation of module m:

    x_m = a · M_m[:, label] + c · e_m + s · U_m z

Here e_m is v1's projected noise, with standard deviation 0.1 before
projection. Three regimes:

| Regime | c | s | Shared variance per direction | Total nuisance variance per module |
| --- | ---: | ---: | ---: | ---: |
| independent | 1 | 0 | 0 | 0.060 (v1) |
| weak | √0.1 | √0.027 | 0.027 | 0.060 |
| strong | √0.1 | 1 | 1.0 | 2.006 |

The independent regime is v1's fixture bit for bit. All regimes share the
same amplitudes, labels and e draws within a seed, so regime comparisons use
common random numbers.

**Test sets.**

- **In-distribution:** v1's 24 test examples, with the same shared z for all
  three modules.
- **Decorrelated:** the same amplitudes, labels and e, with z replaced by
  z_dec. Each module keeps its own marginal distribution; only the dependence
  between modules is removed.

The learner never sees the decorrelated set, and the pilot never scores it.

**Evaluator-only targets.** For hidden module h with observed modules o:

- signal mean μ_sig = a · M_h[:, label], the projection of x_h onto its
  pattern span;
- in-distribution mean μ_full = μ_sig + s U_h ẑ, where
  ẑ = s Σ_o U_oᵀ x_o / (0.01 c² + 2 s²). Because U_o is orthogonal to M_o,
  U_oᵀ x_o = s z + c U_oᵀ e_o. This is the exact Gaussian conditional mean
  of the shared latent [derived].

On the decorrelated set, z_h is independent of the observed modules, so the
Bayes mean is μ_sig.

**Oracle references** [derived; checked by Monte Carlo], per coordinate:

| Regime | Signal oracle (μ_sig), in-distribution | Full oracle (μ_full), decorrelated |
| --- | ---: | ---: |
| weak | 0.0066 | 0.0034 |
| strong | 0.250 | 0.125 |

Each oracle scores 0 on its own test set. Both are computed exactly on every
seed's test sets and reported.

## Architecture, energy and rule

Unchanged from the ablation: three 8 → 4 → 2 linear modules in a chain, the
same energy, clamped-energy rule, Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8) with
persistent moments, weight decay 1e-3 and clips ±2 on W and V, C = 300 cycles
per stage, simultaneous updates. Stage A presents structure 0, stage B
structure 1. Replay is A in stage A and A+B in stage B. Training settles use
32 implicit Euler steps.

Translations are learned freely (the ablation's mode L): initialized to I,
entries clipped to ±3, no weight decay. Fixed and rotation-only translations
are not run here.

**Comparators**, fitted per regime and seed on all waking traces at each
stage, and scored like the learners:

- **Ridge:** v1's `direct_predictor` (ridge 0.1, with intercept), as before.
- **Reduced-rank regression, rank 2:** start from the same ridge fit for each
  hidden module. Take the top two right singular vectors V₂ of the centered
  fitted values on the training traces, and replace the slope matrix B with
  B V₂ V₂ᵀ. The intercept is unchanged. The output rank of the non-constant
  part is then at most 2, matching the inner channel.

## Factors

24 conditions:

| Factor | Levels |
| --- | --- |
| Regime | independent, weak, strong |
| Topology family | uniform (κ = b/14 per layer), inner (κ_inner = b/2), outer (κ_outer = b/8), outer-matched (κ_outer = b/14) |
| Budget level | b ∈ {2, 10} |

The budget b is defined as in the ablation: b = 8κ_outer + 4κ_middle +
2κ_inner. Uniform, inner and outer spend the full budget b; outer-matched
does not.

- **outer** is new. It isolates the surface channel, which the ablation
  could not. Uniform against outer is a **fixed-budget topology comparison**:
  it adds middle and inner coupling and also lowers surface strength from
  b/8 to b/14, so it does not isolate either change.
- **outer-matched** has uniform's surface strength without its middle and
  inner coupling. Its budget is 8b/14. Uniform against outer-matched holds
  surface coupling fixed and isolates the effect of adding deeper coupling.

## Pilot selection (seeds 0-2 only)

For each of the 24 conditions, select η ∈ {0.001, 0.003, 0.01, 0.03, 0.1} on
seeds 0-2:

- train on even-index waking traces;
- score observed whole-module masked MSE on odd-index waking traces after
  stage B, **using exact-equilibrium inference**, the primary endpoint. The
  ablation's pilot scored 32-step inference. Using exact inference here keeps
  learning-rate selection from optimizing the settling-speed confound the
  primary endpoint removes;
- ties go to the smaller η;
- non-finite values disqualify an η.

The pilot score uses only learner-accessible, in-distribution data. Selection
therefore favours in-distribution fit, as a deployed learner's would. Grid-edge
selections are reported as limitations, with no extension. The full grid is
reported, and all pilot runs are charged to a named selection phase.

## Evaluations

At initialization, after each stage and every 50 cycles:

- **Primary inference:** exact equilibrium of whole-module masked completion.
  The hidden module is rotated over all three. **32-step inference is
  secondary.** The ablation showed that 32-step error mixes model quality
  with settling speed, which differs between topologies.
- **X_in:** avoidable MSE per coordinate against μ_full on the
  in-distribution set.
- **X_dec:** avoidable MSE per coordinate against μ_sig on the decorrelated
  set.
- **Nuisance gain γ (the filtering measure).** Masked completion is linear
  in the observed outer activities, for exact equilibrium and for 32 steps
  from a zero state, so ŷ_h = A_h x_o, with A_h an 8×16 map computed from
  the model's Q; the energy has no bias terms. For ridge and reduced-rank
  regression, A_h is the slope part of the fit. Restrict it to the shared-nuisance
  directions of the observed modules: G_h = A_h [U_o₁ ⊕ U_o₂], an 8×4 map.
  The full oracle's map is G*_h = (s² / (0.01c² + 2s²)) U_h [I₂ I₂]. Then

      γ = sqrt( Σ_h ‖G_h‖²_F / Σ_h ‖G*_h‖²_F ).

  γ measures how strongly predictions respond to observed shared nuisance,
  in any output direction. It cannot be hidden by rotation, sign or
  cancellation across modules or examples. γ = 1 matches the oracle's
  magnitude and γ = 0 ignores the shared nuisance. Also reported:
  - unnormalized ‖G_h‖_F, defined in every regime;
  - the same norm on each observed module's full 6-dimensional nuisance
    complement;
  - the part of G_h landing in the hidden module's pattern span, where
    nuisance corrupts the signal estimate.
- **Alignment τ (diagnostic only).** On the in-distribution set,
  τ = Σ ⟨ŷ − μ_sig, s U_h ẑ⟩ / Σ ‖s U_h ẑ‖², summed over examples and hidden
  modules. τ is 1 for the full oracle and 0 for the signal oracle, but it
  can be near 0 for a predictor that transmits nuisance in a rotated
  direction. Undefined in the independent regime.
- **Signal error:** ‖P_h (ŷ − μ_sig)‖² on each set, where P_h projects onto
  the pattern span. This separates signal loss from nuisance leakage.
- Observed RMSE, the sample floor, noise-subtracted MSE and cross terms, as
  in the ablation, on both sets.
- Training-set error and the generalization gap; translation singular values
  and outer gains on the pattern span, U_m and the remaining complement;
  parameter movement; settle convergence; clip activations.

## Criteria (confirmatory seeds 23-32, declared before execution)

All criteria use exact-equilibrium avoidable MSE. Ratios are
(X₁ + δ) / (X₂ + δ) with δ = 1e-5, and the decision rules are the
ablation's:

- **comparable:** larger ratio ≤ 1.25;
- **clearly worse:** ratio ≥ 1.5 and worse in ≥ 9/10 seeds;
- otherwise **unresolved**.

Each criterion is evaluated at each budget separately.

- **N1 learning.** As the ablation's A1, applied to X_in:
  - B's error falls during stage B in ≥ 9/10 seeds;
  - A's after-B / after-A ratio is ≤ 1.5 in ≥ 9/10 seeds;
  - final X_in is ≤ 0.1× the condition's own frozen X_in.

  Comparisons involving a condition that does not learn are reported but
  support no conclusion.
- **N2 control (independent regime).** Report uniform, outer,
  outer-matched and inner against each other. This is the ablation's comparison on fresh seeds, now
  at exact equilibrium and with the layers isolated. Descriptive.
- **N3 transmission (weak and strong), per condition, on γ.**
  - **Transmits:** γ ≥ 0.5 in ≥ 9/10 seeds.
  - **Filters:** γ ≤ 0.2 in ≥ 9/10 seeds.
  - Otherwise **unresolved**. Unresolved is its own outcome and is never
    counted as either of the others.

  τ is reported beside γ as an alignment diagnostic and decides nothing.
- **N4 robustness, the main test (weak regime).** Two pairwise verdicts on
  final X_dec: inner against uniform, and inner against outer. Each pair is:
  - **advantage** if the other family is clearly worse than inner on X_dec;
  - **dominated** if inner is clearly worse on X_dec, or the two are
    comparable on X_dec while the other is clearly better on X_in;
  - **neutral** if comparable on X_dec and not dominated;
  - otherwise **unresolved**.

  Aggregation, applied in this order:
  1. **Mixed** if one pair is an advantage and the other is dominated. No
     conclusion; both verdicts are reported.
  2. **Counts against the mechanism on this fixture** if at least one pair
     is dominated and neither is an advantage.
  3. **Supports the mechanism** if both pairs are advantages and inner
     filters under N3.
  4. **Partial support** if one pair is an advantage, the other is neutral,
     and inner filters under N3.
  5. **Advantage not explained by filtering** if 3 or 4 holds except that
     inner does not filter under N3. No support is claimed.
  6. Otherwise **unresolved**.

  X_in for the same pairs is reported alongside, as the in-distribution cost.
- **N5 does filtering depend on the regime (inner family, strong against
  weak).** N3 gives inner one of three outcomes per regime.
  - **Filtering depends on the regime, as the variance-ranking hypothesis
    expects:** inner filters in the weak regime and transmits in the strong
    regime.
  - **Filtering persists:** inner filters in both regimes. Then filtering
    in this model is not explained by variance ranking.
  - **No filtering to explain:** inner transmits in the weak regime.
  - Any other combination, including either regime unresolved, is
    **unresolved**.

  X_in and X_dec are reported for all families.
- **N6 adding deeper coupling at fixed surface strength (weak regime).**
  Compare uniform with outer-matched, which share κ_outer = b/14.
  - On γ, with ratio γ_uniform / γ_outer-matched computed as
    (γ₁ + 0.01) / (γ₂ + 0.01):
    - **deeper coupling reduces transmission** if the ratio is ≤ 1/1.5 and
      uniform's γ is lower in ≥ 9/10 seeds;
    - **deeper coupling increases transmission** if the ratio is ≥ 1.5 and
      uniform's γ is higher in ≥ 9/10 seeds;
    - **no effect on transmission** if the ratio is between 1/1.25 and
      1.25;
    - otherwise unresolved.
  - X_dec and X_in are compared with the usual rules and reported. A lower
    X_dec alone is not read as lower transmission, because it can come from
    better signal reconstruction; the signal-error split is reported.
  - Uniform against outer (full budget) is reported with the same measures
    as a fixed-budget topology comparison, with no conclusion about either
    added layers or surface strength.
- **R1 references.** For every condition, report the pair (X_in, X_dec)
  and γ beside ridge's, reduced-rank regression's and the two oracles'.
  Descriptive.
- **R2 reduced-rank analogy (weak and strong).** Report inner's γ and
  X_dec beside rank-2 reduced-rank regression's, and whether both fall in
  the same N3 category in each regime. Descriptive: agreement is consistent
  with the analogy, and disagreement shows that the inner channel does not
  behave like reduced-rank regression here.

These are screening rules on ten seeds, not significance tests. Partial
passes are reported as such. There are no post-result extensions of
regimes, cycles, grids, budgets or seeds.

**Reading against the hypothesis.** Support under N4, which requires inner
filtering under N3, supports the hypothesis's mechanism in linear modules: a narrow inner
channel shares what persists and drops surface correlations that do not. It
does not show that inner layers share deeper or more abstract structure,
which needs nonlinear modules (fixture 1 of the research direction). A
"counts against" outcome on N4 counts against the mechanism only for this fixture's
signal-to-nuisance ratio and these budgets.

## Computation

Per learner, as in the ablation: one 18-dimensional factorization and 32
batched steps per cycle; an O(BP) energy derivative; an O(P) Adam step.

| Family | Translation parameters | P |
| --- | ---: | ---: |
| uniform | 168 | 288 |
| inner | 8 | 128 |
| outer | 128 | 248 |
| outer-matched | 128 | 248 |

Exact-equilibrium evaluation: three masked solves per set. Oracles and τ:
O(n) 8×2 products per evaluation. γ: the completion map A_h from one
factorization of the free block per hidden module, O(S³), then an 8×16×4
product. Reduced-rank regression: one 8-column SVD per fit beyond ridge.

Executed totals:

- confirmatory runs: 24 conditions × 10 seeds × 600 cycles;
- pilot: 24 × 5 × 3 = 360 trajectories, scored at exact equilibrium;
- 8 frozen baselines, ridge and reduced-rank regression per regime and
  seed.

The ablation took 191 s wall for 18 conditions with a 216-trajectory pilot.
This study is expected to take about 5 minutes. Report executed counts, not
these estimates.

Follow [computation reporting](computation-reporting.md):

- a deterministic scientific JSON (`reports/correlated-nuisance-v1-results.json`);
- a separate runtime JSON with hashes, revision, dirty status, versions and
  thread settings;
- per-phase wall and CPU time, with the pilot charged to selection;
- peak RSS, and counts of settles, factorizations, exact solves and clip
  activations;
- per-cycle movement in a separate deterministic file, compressed after the
  run with `gzip -n -9` if large, with its uncompressed SHA-256 recorded.

## Verification

- The independent regime reproduces v1's `fixture(seed)` byte for byte, on
  seeds 0, 3 and 23.
- **Reproduction:** on seed 3, independent / uniform / b = 2 at η = 0.03
  reproduces local contrastive v1 arm 2a's final per-structure 32-step RMSE
  to 1e-12.
- U_m is orthonormal and orthogonal to M_m, to 1e-12.
- The decorrelated set equals the in-distribution set outside the s U_m z
  term, bit for bit.
- μ_full matches a generic Gaussian conditional (pseudo-inverse of the
  stacked observed-nuisance covariance) to 1e-10 on one seed. On 200,000
  fresh samples, the oracle values match the table to within 2%.
- γ: the map A_h reproduces exact-equilibrium and 32-step predictions to
  1e-10 on random inputs. Applied to the full oracle's map, γ = 1 exactly;
  a random 90° rotation of G* within the nuisance span leaves γ unchanged.
- Reduced-rank regression: the fitted slope has rank 2, and when the targets are an exact
  rank-2 linear function of the inputs it equals ridge to 1e-10.
- On the in-distribution set the full oracle gives τ = 1 and the signal
  oracle τ = 0, both exactly.
- Every check carried over from the ablation: finite-difference gradients,
  exact clamps, waking and evaluation leave parameters unchanged, identical W
  and V initialization across conditions, and the avoidable-error identities.

Out of scope:

- nonlinear modules;
- fixed and rotation-only translations;
- learned per-layer coupling strength;
- nuisance that changes between stages;
- scaling of the surface dimension or sample count;
- other learning rules.
