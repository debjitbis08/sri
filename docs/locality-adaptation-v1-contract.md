# Locality adaptation v1 — learning partition boundaries at counted cost

**Status:** prospective contract, 2026-10-24; no implementation or scored run
under this contract yet. Commit this contract before implementation. After
correctness gates, freeze the source/environment manifest and commit it before
generating confirmation worlds. Any substantive change requires a dated
amendment before confirmation, or a new version after confirmation starts.

**Pre-implementation review, 2026-10-24.** Before any implementation, the
contract was edited only to remove ambiguities: per-block ridge trace,
spectral tie handling, absence of covariance reuse in the bank, the score
entering each primary claim, the timing seed, memory-measurement scope, the
artifact layout, and exact descriptive control values. No threshold, arm,
budget, regime, seed or decision rule changed.

Follows the [locality analysis](locality-adaptation-analysis.md),
[transfer analysis](broad-structure-transfer-analysis.md), and
[computation protocol](computation-reporting.md). Historical experiments and
their frozen artifacts remain unchanged.

## 1. Question, predictions, and scope

Can selection among eight fixed partition phases reduce misaligned-locality
error while preserving aligned performance, after paying for validation
examples and all candidate fits? Does the improvement preserve useful broad
structure in independent fresh combinations, as measured by reconstruction?

Predictions, not assumptions of success:

- A phase bank contains the true supports for aligned and translated worlds.
  It can remove their population approximation floor. Finite-sample fitting
  and selection can still fail, particularly at small budgets.
- Dense rotations generally leave a positive best-phase floor. Phase selection
  is not expected to remove arbitrary nonlocal mixing.
- Deployed adaptive and fixed local codes have the same size and arithmetic
  order, but adaptation costs more to fit and select. Global CCA may remain
  the stronger quality or total-cost choice.

The candidate is an analytical paired-view estimator. No neural training,
reciprocal dynamics, retained activation, online plasticity, learned arbitrary
routing, or broad claim about real-world data is tested. Source-to-target
learning and subspace drift remain a separate prospective experiment. Fresh
combinations here mean independent coefficients in the same world, not
combinations deliberately excluded from training.

## 2. Worlds and permitted information

Primary dimensions are `d=32`, `k=4`, block size `b=8`. For each independent
world, construct orthonormal U and V exactly by the template recipe in
benchmark v1: in each block a normalized random combination of orthonormal
DCT-II modes 0–2 gives one signal column; modes 3–7 give one texture column.
The three and five coefficients respectively are independent standard normals.
The estimator receives neither these modes nor U, V, the regime, nor the
true partition phase. The zero population mean is supplied to all methods;
no empirical centering is performed.

Use four paired regimes from the same world:

| Regime | Orthogonal transform R | Role |
| --- | --- | --- |
| aligned | identity | primary preservation test |
| half-shifted | circular shift by four sites | primary adaptation test |
| unknown-shift | circular shift by one uniformly drawn integer in 1–7 | secondary boundary-generalization check |
| dense | Q from the QR factorization Q·T_r of a d×d standard Gaussian matrix, with column signs fixed so diag(T_r) > 0 | secondary limit case |

Rotate both signal and nuisance. For every independent training pair draw

```
c ~ N(0,I_k)
h1 = R(U c + 2 V g1 + 0.5 e1)
h2 = R(U c + 2 V g2 + 0.5 e2)
g1,g2 ~ N(0,I_k); e1,e2 ~ N(0,I_d), all independent of c and each other.
```

**Declared change from benchmark v1:** coefficients are unconditioned
Gaussians. This removes the sign-correlation shift between fitting and
evaluation and makes the population objective and validation-score moments
exactly the ones in the analysis. This is a new experiment, not a rerun of
v1's R² benchmark. U and V retain the unknown smooth-template/texture recipe;
no clean targets or labelled probe examples are used in fitting or selection.

For each world, draw one length-256 pair stream and transform those same
draws into each regime. Budgets `n={4,8,16,32,64,128,256}` are nested prefixes.
This pairs the regimes without pretending they are independent replicates.

## 3. Candidate and estimators

Phase p in `{0,...,7}` defines k disjoint cyclic groups
`G_p,j = {(p+8j+t) mod d : t=0,...,7}`. Every local code has one feature
per group. Its deployment must retain p, not the true phase.

From fitting pairs X,Y with m rows define

```
K = (X^T Y + Y^T X)/(2m)
T = (X^T X + Y^T Y)/(2m).
```

**Ridge CCA reference convention:** in each allowed block solve the symmetric
generalized eigenproblem `(K, T + r I)`, where
`r=0.1 tr(T)/block_dimension`. Choose the largest algebraic eigenvalue's
direction, normalized in Euclidean norm. Here T and K are the block's own
submatrices and block_dimension=b, so `r=0.1 tr(T_gg)/b`. Global CCA takes
the top k such directions using global T and `r=0.1 tr(T)/d`. This matches v1's pooled-covariance
ridge-CCA convention; it is not claimed to be every definition of CCA.
Use Cholesky whitening and symmetric eigensolvers, float64. Spectral arms
use the largest algebraic eigenvectors of K with no whitening.

For numerical eigenspace ties, group adjacent sorted eigenvalues when their
gap is at most `1e-12 * max(1, spectral_radius)`. Within each group construct
a canonical basis by projecting coordinate axes in index order into the
eigenspace and applying Gram–Schmidt, skipping residual norms below 1e-10.
Use this basis to fill a cutoff through a tied group. Apply this convention
in whitened coordinates for CCA and in raw coordinates for spectral arms.
After back-transformation and normalization, orient each resulting weight
vector so its first largest-magnitude coordinate is positive. Record the tolerance convention;
do not claim invariance across unpinned numerical environments.

Convert each fitted feature matrix W to its orthogonal span projector
`P=W(W^T W)^{-1}W^T`, implemented through reduced QR, not a numerical
inverse. Local rows have disjoint support and are already orthonormal after
normalization. Global CCA vectors need not be orthogonal. Rank k is required.
Inference is encoding by an orthonormal basis B and reconstruction `B B^T h`;
it is not a fitted decoder or a labelled probe.

### Phase selection and accounting

Within the budget-n prefix, pairs whose one-based index is divisible by four
are validation; the rest are fitting. Thus `n_fit=3n/4`, `n_val=n/4`, with
nested fitting and validation streams even at n=4. Fit all eight phase codes
on fitting pairs only. For each fitted projector compute

```
score_p = mean_validation(h1^T P_p h2).
```

Select its maximum. Scores within `1e-12 * max(1, abs(max_score))` of the
maximum tie; choose the smallest phase. **Do not refit after selection.**
This keeps the selected predictor exactly the one independently validated
and makes the cost of withholding examples visible. No ridge, allocation,
phase bank, metric, or selection rule is tuned on development or confirmation.

The score is unbiased for `tr(P_p S)` conditional on the fitted candidates,
where `S=R U U^T R^T`. It measures signal retention, not nuisance rejection.
Reconstruction is a separate gate below; a better score alone is insufficient
to call the structure more useful. The eight scores are correlated; they are
not eight independent observations.

| Arm | Fit pairs | Selection | Role |
| --- | --- | --- | --- |
| `adaptive_cca` | fitting split for each phase | validation score above | sole primary candidate |
| `fixed_cca_all` | all n, phase 0 | none | primary fixed-locality comparator; receives full data budget |
| `global_cca_all` | all n, global rank k | none | mandatory strong reference |
| `fixed_cca_fit` | fitting split, phase 0 | none | isolates withholding cost |
| `global_cca_fit` | fitting split, global rank k | none | diagnoses data-split cost against global reference |
| `adaptive_spectral` | fitting split for each phase | same validation rule | estimator/objective diagnostic |
| `fixed_spectral_all` | all n, phase 0 | none | spectral locality control |
| `global_spectral_all` | all n, global rank k | none | spectral global control |

There are no epochs, gradient updates, augmentations, or development
hyperparameter searches. Reusing examples across eight fits counts as eight
fit exposures, not eight times as many independent examples. At each n,
fit from scratch; caching across budgets is forbidden in the primary cost
measurement. The reference implementation computes every phase's block
statistics independently from the fitting rows, with no reuse between phases
or arms; this matches the complexity stated in section 7. Any later reuse
requires disclosure of its construction, retained storage and implementation.
No computational work may be credited free because another arm computed it.

## 4. Evaluator-only ceilings and failure diagnosis

For every world/regime compute the top eigenvalue sum of each diagonal
block of S. Report each phase's optimal signal retention, the phase-0 floor,
and the best-phase floor. Global rank-k signal floor is zero. These are
oracle structural ceilings, never inputs to fitting or selection.

Also retain all eight fitted candidates' true signal losses, accessible only
to the evaluator. At each budget report the exact decomposition

```
selected signal loss
  = best-phase population floor
  + (best fitted candidate loss - best-phase population floor)
  + (selected loss - best fitted candidate loss).
```

All three terms are nonnegative up to numerical tolerance. They separate
representation, fitting, and selection failures. Do not transfer this
decomposition to reconstruction risk: its optimal projector also depends
on nuisance. Record selected phase and population-optimal phases, but exact
phase identification is not a success criterion when phases tie.

## 5. Metrics: signal preservation and usefulness

For each fitted P, use exact population second moments on **fresh** signals.
No finite test sample or labelled probe is needed. Ground truth is evaluator
information only. Primary signal loss is

```
L(P) = tr((I-P) S)/k = 1-tr(P S)/k.
```

Primary reconstruction NMSE uses in-distribution noise
`N=R(4 V V^T + 0.25 I)R^T`:

```
E(P) = E||P h-s||² / E||s||²
     = [tr((I-P)S) + tr(P N)]/k.
```

Report bias/signal loss L and projected-noise contribution separately. A
projector capturing nuisance can preserve signal better yet reconstruct
worse. Fixed rank k prevents zero-code collapse from passing as a learned
code. Include zero-output NMSE=1 and raw-input NMSE=`tr(N)/k` (6 in
distribution) as descriptive controls, plus oracle signal projector P=S for
reconstruction (NMSE 0.25 in distribution, since U is orthogonal to V). The oracle
signal projector is not claimed to be the optimal shrinkage estimator.

Primary low-data scores are arithmetic means over n in `{16,32,64,128,256}`,
computed within each world and regime. Call them mean log-spaced-budget
scores, not an exact trapezoidal area. Budgets 4 and 8 remain reported
diagnostics; no budget may be dropped after seeing results.

Secondary reconstruction conditions use the same fitted P with:

- texture variance 16 instead of 4;
- additional noise covariance `2 R(I-UU^T-VV^T)R^T`.

The latter increases variance in already noisy directions from 0.25 to 2.25;
it does not introduce entirely unseen directions. For each condition report
absolute NMSE and its increase from in-distribution separately. These are
descriptive stress tests, not additional routes to primary success.

## 6. Replication, randomness, and primary decisions

Root seed `2026102407`. Use NumPy `default_rng(SeedSequence([root,stage,w,part]))`:
stage 0 for development, 1 for confirmation, 2 for timing; w is world index.
Part 0 draws signal/texture template coefficients, interleaved by block;
part 1 draws the dense Gaussian matrix for QR; part 2 draws unknown phase;
part 3 draws c, g1, g2, e1, e2 arrays in that order. Freeze shapes and loop
order in the manifest. For timing, append d to the SeedSequence entropy:
`[root,2,0,part,d]`.

Development worlds `w=0..7` are for implementation and correctness checks
only. Confirmation uses `w=0..63` with stage 1: 64 independent worlds, one
training stream per world. All arms, budgets and regimes share a world.
No development result is pooled into confirmation. No seed substitution,
optional stopping, or additional worlds after inspecting a confidence bound.

For each world compute improvement `D = fixed_cca_all - adaptive_cca` in
each lower-is-better score. Each claim uses the primary mean log-spaced-budget
score of section 5 (n in 16–256), computed per world before differencing.
Four primary claims are jointly required:

| Claim | Regime / score | Required population improvement |
| --- | --- | --- |
| H1: recover shifted signal | half-shifted / L | greater than 0.02 |
| H2: preserve aligned signal | aligned / L | greater than -0.02 (noninferiority) |
| H3: improve shifted usefulness | half-shifted / E | greater than 0.02 |
| H4: preserve aligned usefulness | aligned / E | greater than -0.02 (noninferiority) |

Use one-sided paired Student-t lower bounds over the 64 world differences,
each at confidence `1-0.05/4=0.9875`: `mean(D)-t_63(0.9875)*SE(D)`.
Require each bound strictly above its table threshold. This Bonferroni
family controls the four declared claims under the t-test assumptions;
world-level differences need not be exactly Gaussian, so disclose the
finite-sample approximation. With zero sample variance use the constant
difference as the bound and report the degeneracy.

**Overall pass requires all four.** Non-significance is inconclusive, not
equivalence. Report each bound and outcome even if another fails. No best-arm
substitution: an adaptive-spectral win cannot rescue adaptive-CCA's result.

Report descriptive paired means and 95% t intervals against global CCA,
the split controls, all learning curves, and the unknown-shift/dense regimes.
These do not license an unregistered superiority claim. A pass establishes
adaptation versus fixed locality under this generator, not superiority to
global CCA. If global CCA is preferable in quality/cost, state that explicitly.

## 7. Computation: required measurements and scaling

Record complexity with `b=d/k`, K=8 and actual n_fit/n_val. With independent
full block covariance construction and dense factorization, local CCA fitting
costs `O(n_fit d b + d b²)`; the adaptive bank costs K times that plus
`O(K n_val d)` validation. Global CCA costs `O(n d²+d³)`. Marginal and
cross-covariance statistics require `O(db)` local or `O(d²)` global storage;
the bank requires `O(Kdb)` concurrently or reduced workspace with rereading.
Count storage of raw pairs, all candidate bases, and eigensolver workspace
separately. Solvers are full dense direct factorizations, not iterative
settling. Optimizer state/updates and tuning trials are zero by design.

Deployed local basis storage is d float64 weights plus phase/support metadata;
global is dk weights. Local encode/reconstruct work is O(d), global O(dk).
Report actual allocated bytes, including metadata, and peak temporary bytes
where measurable; asymptotic notation is not a memory measurement. Raw pairs,
validation data and rejected bases are not needed for deployed inference.

Scientific computation uses one process and one BLAS thread, float64, to make
per-arm cost attribution straightforward. Time covariance construction,
factorization, validation/selection, and inference separately. Keep generation,
oracle diagnostics, metric evaluation and report writing in separate phases.
Record RSS in isolated worker processes for standalone per-arm memory where
needed; do not label a cumulative process maximum as per-arm memory. The
timing panel runs each (d, regime, arm) in a fresh spawned process and also
records an idle worker's RSS baseline. Peak temporary NumPy bytes come from
`tracemalloc` in a separate untimed pass; LAPACK-internal workspace is not
visible to it and is reported as an analytical estimate, labelled as such.

After the manifest freeze, a separate **descriptive timing panel** uses
stage-2 world 0, dimensions `{32,128,512}`, b=8, k=d/8, n=256. Run the aligned
and dense regimes for adaptive CCA, fixed CCA all, and global CCA all. The
DCT and per-block generator generalize unchanged. This changes k with d;
do not call it fixed-rank scaling. Stage-2 data never enter statistical scores.

For each timed configuration, perform one discarded warm-up and seven
independent fresh-fit repetitions on identical data. Record every time,
median and interquartile range. Inference measures encoding plus reconstruction
on (a) one input and (b) a batch of 256; use one warm-up batch and seven blocks
of 100 calls, report per-call timing, with allocation included and data
generation excluded. Warm-up/repetition costs remain part of total study cost.
No solver failure or long run may be silently omitted.

Required runtime fields follow the computation protocol: wall and CPU seconds,
RSS and scope, CPU/OS/BLAS/libraries/thread counts, source hashes, dirty state,
executed fits, candidate scores, independent pairs, exposures, factorization
sizes, solver failures and inference calls. Mark unavailable quantities null
with a reason. Do not infer FLOPs from wall time or parameter count.

Report quality versus total pair count and measured fit-plus-selection cost,
inference latency and persistent bytes. Do not claim equal-compute superiority
from the equal-example test. For Q in `{1,100,10000}`, also report
`fit+selection+Q*single_input_inference` using measured times and the winning
phase only for deployment. These are amortization estimates, not measured
end-to-end runs. Timing-panel scaling is descriptive, not a speed hypothesis.

## 8. Correctness gates and freeze

Before any confirmation data are generated, require:

1. U,V orthonormal and mutually orthogonal; rotations orthogonal; shared
   base draws and nested pair/split indices verified. Fitters accept only
   observed pairs, dimensions and declared phase, not evaluator truth.
2. Local supports disjoint for every phase; rank k and projector symmetry,
   idempotence and trace k within 1e-9. Global CCA projector must be invariant
   to nonsingular changes of basis in W within 1e-9.
3. Reproduce the analysis witnesses: fixed half-shift floor 0.5, best-phase
   floor zero, Hadamard witness floor 0.75 at every phase, tolerance 1e-10.
4. Independent bilinear validation calculation agrees with projected-code
   products within 1e-10 relative/absolute scale; phase tie rule tested.
   Altering validation data cannot change fitted candidate weights. Altering
   evaluator truth cannot change any selected phase or fitted weights.
5. Independent full linear error-map calculation agrees with trace NMSE
   for known projectors and all three noise conditions within 1e-10.
   Loss decomposition is nonnegative to 1e-9 and sums correctly.
6. CCA residual and factorization checks pass; fixtures cover singular
   unregularized covariance, tied eigenvalues and n=4. Record the specified
   eigenspace tie handling in the implementation manifest. No rank repair,
   ridge increase or fallback may silently change the declared method.
7. Repeated development run produces byte-identical scientific JSON in the
   pinned environment. Timing is excluded from that comparison. Relevant
   existing estimator tests and all new correctness tests pass.
8. Manifest records this contract hash, implementation/test hashes, pinned
   NumPy/SciPy/Python/BLAS environment, seeds, exact arm definitions and split
   counts. Commit manifest and source before the first confirmation read.

No quality threshold on development worlds is a gate; poor development
performance is reported rather than used to tune this frozen design.

## 9. Failures, stopping, and artifacts

A nonfinite result, nonpositive regularized covariance, rank failure or
correctness violation stops scoring. Record the affected world/arm/budget and
mark the run incomplete; never replace it with zero loss or remove the world.
If discovered after confirmation starts, record which outcomes were accessed.
An implementation-only fix must preserve this contract and rerun all worlds,
with the correction disclosed. A methodological change needs a new contract
and fresh confirmation seed space. Resource-limited runs are truncated and
cannot satisfy primary claims.

Planned artifacts (not yet produced):

- `src/locality_adaptation.py` (estimators, generator, metrics),
  `experiments/locality_adaptation.py` (runner) and
  `tests/test_locality_adaptation.py` (correctness gates);
- `reports/locality-adaptation-v1-manifest.json`;
- `reports/locality-adaptation-v1-development.json`;
- `reports/locality-adaptation-v1-results.json`, including per-world metrics,
  all phase scores/floors, selections, decomposition and primary intervals;
- `reports/locality-adaptation-v1-runtime.json`, including timing-panel records;
- `docs/locality-adaptation-v1-results.md`, interpretation and scope ledger.

On completion, classify representation, fitting, selection, usefulness and
computation outcomes separately. A dense-regime floor supports investigating
overlapping/adaptive memberships or a global fallback; selection regret
supports improving selection within the same class. Neither authorizes an
expanded rescue search inside this study. Preservation under subsequent
learning, retained context and real-data utility remain untested.
