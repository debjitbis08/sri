# Locality adaptation v2 — select the partition phase by cross-validation, then refit on every pair

**Status:** prospective contract, 2026-10-24, committed before
implementation; no implementation, development run or confirmation draw
existed when it was committed. It incorporates one review round (failure
wording, oracle-arm naming, qualitative predictions with fold dependence,
L-only decomposition, storage orders) made before that commit. After correctness gates, freeze
the source/environment manifest and commit it before generating
confirmation worlds. Any substantive change requires a dated amendment
before confirmation, or a new version after confirmation starts.

Follows [locality adaptation v1](locality-adaptation-v1-results.md), whose
frozen contract, code and artifacts remain unchanged. v1 recovered translated
structure (H1, H3 pass) but failed aligned noninferiority (H2 −0.058, H4
−0.071 lower bounds against −0.02). About two-thirds of the aligned
signal-loss deficit (0.033 of 0.049) was the cost of withholding a quarter of
the pairs. v1 forbade refitting after selection, so that cost was built in.
This is a methodological change, so v2 uses fresh confirmation seed space.

## 1. Question, predictions, and scope

If the selected phase is refit on all n pairs, does phase adaptation keep
v1's shifted recovery while becoming noninferior to fixed locality on
aligned worlds, at equal independent pairs and with all selection fits
counted?

Refitting removes the withholding cost. When selection picks phase 0, the
refit model is **exactly** `fixed_cca_all`. On aligned worlds the remaining
cost therefore comes only from selecting another phase. Predictions are
qualitative:

- v1's aligned selection regret was concentrated at n=16 and 32 (0.065 and
  0.022, falling to about zero by n=128). Those values were measured among
  ¾-data candidates. Refitting on all n changes candidate losses and may
  change their ordering. v1's primary SE belongs to a different contrast.
  These numbers motivate the experiment but neither predict nor bound
  post-refit regret or its uncertainty. Holdout selection followed by refit
  could plausibly land near the margin. It is a diagnostic arm, not the
  candidate.
- The primary candidate scores every pair once by four-fold
  cross-validation. This is expected to reduce selection error relative to
  a single n/4 holdout, mostly at small budgets. The folds' training sets
  overlap, so the n out-of-fold scores are dependent and are **not** n
  independent validation observations. Fold candidates are fitted on 3n/4
  pairs, so the gain is not guaranteed. Failure of H2/H4 remains a plausible
  outcome, especially from n=16.
- Translated-world recovery should match or exceed v1's, because the
  deployed fit uses all n pairs. Dense rotations keep a positive best-phase
  floor. Global CCA remains the stronger dense-regime choice.
- Cross-validated selection costs about four times v1's bank (33 local fits
  rather than 8) and gives no deployment advantage. Total cost may favour
  global CCA even where quality favours adaptation.

Unchanged exclusions from v1: no neural training, reciprocal dynamics,
retained activation, online plasticity, learned arbitrary routing, overlapping
memberships, subspace drift or real-data claim. Fresh combinations mean
independent coefficients in the same world.

## 2. Worlds, regimes and permitted information

Identical to v1 section 2: d=32, k=4, b=8, the same template recipe,
unconditioned Gaussian coefficients, the same four regimes (aligned,
half-shifted, unknown shift in 1–7, dense Gaussian QR), view model
`h = R(U c + 2 V g + 0.5 e)`, one length-256 base stream per world shared
across regimes, nested budget prefixes `n={4,8,16,32,64,128,256}`, zero
population mean supplied, no empirical centering. Estimators receive only
observed pairs, dimensions and declared phases.

The v2 generator calls v1's `src/locality_adaptation.py` generator functions
unchanged. Only the root seed differs (section 6).

## 3. Candidate, arms and estimators

Local and global ridge CCA, eigenspace tie handling, orientation, span
projectors, validation score `h1^T P h2` and score tie rule are exactly v1
section 3. They are imported from v1's frozen module, which must not be
edited. The v2 manifest records its hash.

**Folds.** The pair with one-based index i belongs to fold `i mod 4`. Fold 0
is exactly v1's validation split. Folds are nested across budgets, and each
fold holds n/4 pairs (one pair at n=4).

**Cross-validated score.** For each phase p and fold f, fit phase p on the
pairs outside f. Each pair then receives one out-of-fold score
`h1^T P_{p,-f(i)} h2`, and

```
cv_score_p = (1/n) * sum_i h1_i^T P_{p,-f(i)} h2_i.
```

Select the maximum, with v1's tie rule (within `1e-12 max(1,|max|)`; the
smallest phase wins). Then **refit the selected phase on all n pairs**. That
refit is the deployed model. Each term is unbiased for `tr(P_{p,-f} S)`
conditional on its fold candidate. The CV score estimates the quality of
fold candidates, not of the refit, and it is never reported as performance.
All reported quality uses exact population moments of the deployed
projector, so selection does not bias the evaluation.

| Arm | Fits | Selection | Role |
| --- | --- | --- | --- |
| `cv_refit_cca` | 4 folds × 8 phases on 3n/4, + 1 refit on n | CV score | **sole primary candidate** |
| `fixed_cca_all` | phase 0 on n | none | primary comparator (as v1) |
| `global_cca_all` | global rank k on n | none | mandatory strong reference |
| `holdout_refit_cca` | 8 phases on fold-0 complement, + 1 refit on n | fold-0 score | isolates refit from CV |
| `holdout_cca` | 8 phases on fold-0 complement | fold-0 score, no refit | v1 `adaptive_cca` replication |
| `oracle_structural_phase_cca_all` | phase on n | evaluator's smallest population-optimal phase | **oracle structural-phase reference, not a learner** |

`oracle_structural_phase_cca_all` fits, on all n pairs, the phase with the
best representational floor. It equals `fixed_cca_all` on aligned worlds by
construction. It is **not** a finite-sample ceiling or bound: on a given
sample another phase can yield a better fitted projector, and the
signal-optimal phase need not minimize reconstruction error. It is never
eligible for any claim. The finite-sample signal-loss ceiling for phase
selection is the minimum true L among the eight all-n fitted candidates
(section 4). Spectral arms are dropped. v1 showed they
admit far more nuisance (projected noise 0.67–1.13 versus 0.26–0.34) without
changing the primary picture.

Every arm fits from scratch at each budget, with independent block
statistics per phase and fold. No fit is reused across folds, arms or
budgets, even where identical (e.g. the fold-0 candidates of `cv_refit_cca`
and `holdout_cca`). No computational work is credited free. There are no
epochs, gradient updates, augmentations or tuned hyperparameters. The fold
count (4), phase bank (8), ridge (0.1), split rule and margins are fixed by
declaration and are not tuned on development or confirmation.

## 4. Evaluator-only ceilings and failure diagnosis

Report v1's per-phase population floors, phase-0 floor and best-phase floor.
For refit arms, the evaluator also fits all eight phases on all n pairs.
These are diagnostic fits, excluded from candidate cost and reported
separately. The exact decomposition, in **signal loss L only**, is then

```
L(selected refit)
  = best-phase population floor of L
  + (min L over the eight all-n phase fits - best-phase floor)
  + (L(selected refit) - min L over the eight all-n phase fits).
```

All three terms are nonnegative, since the selected refit is one of the
eight all-n fits. They separate representation, full-data fitting and
selection regret. The middle minimum is the finite-sample signal-loss
ceiling for phase selection. For `holdout_cca`, use v1's L decomposition
over its own 3n/4 candidates.

As in v1, do not transfer this decomposition to reconstruction error E: its
optimal projector also depends on nuisance, so neither the floor nor the
nonnegativity applies. Diagnose reconstruction through L and projected noise
`tr(PN)/k` reported separately, for the selected model and for all eight
all-n candidates. Report the selected phase, the population-optimal set, and
the fraction of worlds selecting an optimal phase per budget and regime.
Exact phase identification is not a success criterion when phases tie.

## 5. Metrics

Unchanged from v1 section 5:

- Signal loss is `L(P) = 1 - tr(PS)/k`.
- In-distribution reconstruction is
  `E(P) = [tr((I-P)S) + tr(PN)]/k`, with `N = R(4VV^T + 0.25I)R^T`.
- Projected noise is reported separately.
- Controls are zero output (1), raw input (6) and the oracle signal
  projector (0.25).
- The texture-16 and extra-noise conditions are descriptive stress tests.

Primary scores are arithmetic means over n in `{16,32,64,128,256}` within
each world and regime. Budgets 4 and 8 are reported diagnostics, and no
budget may be dropped after seeing results.

## 6. Replication, randomness, and primary decisions

Root seed `2026102408` (v1 used `2026102407`). The entropy layout is v1's
`[root, stage, w, part]`, with the same part assignments, shapes and loop
order. Stages: 0 development (w=0..7, correctness only), 1 confirmation
(w=0..63), 2 timing (`[root,2,0,part,d]`). There is no seed substitution,
optional stopping, or addition of worlds after inspecting a bound, and no
development result is pooled into confirmation.

For each world compute `D = fixed_cca_all - cv_refit_cca` in each
lower-is-better primary score. All four claims are jointly required, with
thresholds unchanged from v1 so the two versions remain comparable:

| Claim | Regime / score | Required population improvement |
| --- | --- | --- |
| H1: recover shifted signal | half-shifted / L | greater than 0.02 |
| H2: preserve aligned signal | aligned / L | greater than −0.02 (noninferiority) |
| H3: improve shifted usefulness | half-shifted / E | greater than 0.02 |
| H4: preserve aligned usefulness | aligned / E | greater than −0.02 (noninferiority) |

Use one-sided paired Student-t lower bounds over the 64 world differences,
each at confidence `1-0.05/4 = 0.9875`. Require each bound strictly above
its threshold. With zero sample variance, use the constant difference as the
bound and report the degeneracy. The finite-sample t approximation is
disclosed as in v1. **Overall pass requires all four.** Non-significance is
inconclusive, not equivalence. Report each bound even if another fails.

No best-arm substitution: `holdout_refit_cca` and `holdout_cca` cannot rescue
or replace the candidate's outcome, and the oracle arm is never a candidate.

Descriptive only (paired means and 95% t intervals, no claims):

- `cv_refit_cca − holdout_refit_cca` (value of cross-validation);
- `holdout_refit_cca − holdout_cca` (value of refitting);
- the candidate versus `global_cca_all` and
  `oracle_structural_phase_cca_all`, and its L versus the finite-sample
  ceiling (min L over the eight all-n fits);
- the unknown-shift and dense regimes, learning curves, and stress
  conditions.

`holdout_cca` replicates v1's `adaptive_cca` on new seeds. Its v1 and v2
means may be compared descriptively, but they are not paired. A pass
establishes noninferior adaptation under this generator, not superiority to
global CCA.

## 7. Computation

With `b=d/k`, K=8 phases and F=4 folds:

| Arm | Fitting time | Selection time |
| --- | --- | --- |
| `cv_refit_cca` | `O(F K (3n/4) d b + F K d b²)` fold fits + `O(n d b + d b²)` refit | `O(K n d)` scoring |
| `holdout_refit_cca` | `O(K (3n/4) d b + K d b²)` + refit | `O(K (n/4) d)` |
| `global_cca_all` | `O(n d² + d³)` | none |

Statistics storage is `O(b²)` per block, `O(db)` across all blocks of one
candidate, or `O(FKdb)` if every candidate's statistics are held at once.
Candidate weights take `O(FKd)` if retained. The implementation discards
each fold candidate after scoring and records retained bytes. Deployed
state and inference are identical in form to v1: d weights plus phase
metadata, O(d) work, and the same NumPy inference path. The evaluator's
diagnostic all-n fits are timed and counted under a separate `evaluator`
heading, never as candidate cost.

Requirements carried over from v1 section 7:
- one process and one BLAS thread, float64;
- stage timers for covariance, factorization, validation/selection and refit;
- counters for executed fits, candidate scores, fit-row and validation-row
  exposures, factorization sizes, solver failures and inference calls;
- null values with a reason where unavailable; no FLOPs inferred from time.

The descriptive timing panel uses stage-2 world 0, d ∈ {32,128,512}, b=8,
k=d/8, n=256, and the aligned and dense regimes. Its arms are
`cv_refit_cca`, `holdout_refit_cca`, `fixed_cca_all` and `global_cca_all`.
It keeps v1's protocol: one warm-up and seven fresh fits, and inference
warm-up plus seven blocks of 100 calls for single inputs and 256-batches.
Each configuration runs in a fresh spawned process, with an idle-worker RSS
baseline, a separate untimed `tracemalloc` pass, and an analytical LAPACK
workspace estimate. Amortization is reported for Q in {1,100,10000} as an
estimate. It is not a speed hypothesis.

Two receipt corrections relative to v1:
- the manifest records the full test summary (count and status), not only
  the final status line;
- runtime records list the exact dirty paths, so outputs being written are
  distinguishable from modified sources.

## 8. Correctness gates and freeze

Before any confirmation data are generated, require:

1. v1's frozen sources (`src/locality_adaptation.py`,
   `experiments/locality_adaptation.py`, `tests/test_locality_adaptation.py`)
   match v1's manifest hashes, and v1's test suite passes unchanged.
2. Folds partition every budget's pairs into four disjoint, nested sets of
   n/4. Each pair is validated exactly once. Fold 0 equals v1's validation
   split.
3. An independent per-pair bilinear computation of each CV score agrees with
   the code-product implementation within 1e-10 relative/absolute scale.
   The phase tie rule is tested.
4. When phase 0 is selected, the `cv_refit_cca` and `holdout_refit_cca`
   deployed weights are bitwise identical to `fixed_cca_all` on the same
   pairs. When phase p is selected, they are bitwise identical to a direct
   phase-p fit on all n.
5. Fold candidates depend only on their training folds, so altering a
   validation fold leaves those candidates unchanged. No fitter accepts
   evaluator truth (signature check). Altering evaluator truth cannot change
   any selection or fitted weights.
6. The L decomposition terms are nonnegative to 1e-9 and sum correctly. The
   oracle structural-phase arm equals `fixed_cca_all` on aligned worlds.
7. Full pipeline is finite at every budget, regime and arm, including n=4
   (one pair per fold, a singular unregularized covariance). There is no
   rank repair, ridge increase or silent fallback.
8. Repeated development run produces byte-identical scientific JSON in the
   pinned environment (timing excluded). All new tests pass.
9. Manifest records this contract's hash, v2 and v1 source/test hashes,
   pinned Python/NumPy/SciPy/BLAS, seeds, arm definitions, fold counts and
   the full test summary. Commit manifest and source before the first
   confirmation read. The runner refuses to confirm otherwise.

No quality threshold on development worlds is a gate. Poor development
performance is reported, not used to tune the design.

## 9. Failures, stopping, and artifacts

As v1 section 9:
- A nonfinite result, nonpositive regularized covariance, rank failure or
  correctness violation stops scoring. Record the affected world, arm and
  budget, and mark the run incomplete.
- An implementation-only fix must preserve this contract, rerun all worlds
  and be disclosed.
- A methodological change requires a new contract and fresh seeds.
- Resource-limited runs cannot satisfy primary claims.

Planned artifacts (not yet produced):

- `src/locality_adaptation_v2.py` (folds, CV selection, refit, diagnostics;
  imports v1's module unchanged), `experiments/locality_adaptation_v2.py`,
  `tests/test_locality_adaptation_v2.py`;
- `reports/locality-adaptation-v2-manifest.json`,
  `-development.json`, `-results.json`, `-runtime.json`;
- `docs/locality-adaptation-v2-results.md`.

On completion, classify representation, full-data fitting, selection,
usefulness and computation outcomes separately. Failure of H2/H4 would mean
this four-fold CV-and-refit procedure did not establish noninferiority at
the declared margin. It would not establish a lower bound on the cost of
phase adaptation in general: another selector, objective or level of
precision could behave differently. Concentrated small-budget regret would
identify where this procedure loses performance. Neither outcome licenses a
larger fold count, a different margin or budget-specific selection inside
this study. A pass supports moving to source-to-target reuse and
subspace drift ([transfer analysis](broad-structure-transfer-analysis.md),
step 2) with `cv_refit_cca` as the local reference.
