# Single-module plasticity v1 — energy inference and familiarity-routed ring learning

**Status:** draft prospective contract, 2026-10-24. No implementation or run
exists under this contract. Commit the reviewed contract before implementation;
commit tested sources, development artifacts and manifest before confirmation.
Substantive changes after confirmation starts require a new contract and seeds.

Follows the [current single-module design](single-module-plasticity-design.md),
[layered analysis](layered-reuse-analysis.md),
[reuse results](representation-reuse-v1-results.md), and
[computation protocol](computation-reporting.md). Earlier sources, results and
analytical receipts stay unchanged. This study is not a continuation of the
reuse experiment's statistical claims.

## 1. Scope and questions

Study exactly one circular module with four rings and one outer input/output
interface. Test whether local familiarity can modulate both reciprocal
influence and plasticity, and whether slower learning inward improves source
retention without unacceptable target-adaptation cost.

No multiple modules, satellite/core system, inter-module routing, singleton
centre, explicit context IDs, learned drift detector, topology learning,
dreaming, activation memory across observations, or retained raw historical
examples. Inference states reset for every observation. Persistent maps and
acquired phase are the memory being tested.

This is a test of rate and route arrangements within a fixed layered model.
It does not establish that layering outperforms all flat models, that a depth
profile is universally optimal, or that implicit differentiation is a local
biological learning rule. Direct equivalent inference is mandatory.

## 2. Geometry, parameters and energy

Ring widths n=(32,24,16,8); indices l=0,1,2,3 outward to inward. Units lie
in circular order; every ring has wraparound lateral neighbors. Four sectors
contain (8,6,4,2) coordinates per ring. The exterior sector origin is acquired
from source pairs (section 5); inner sectors follow those four sector labels.
This is not the same integer shift applied to unequal-width rings.

Learn block-supported maps D_l,j of sizes 8×6,6×4,4×2 for l=0,1,2 and
j=0..3. Every D_l,j has orthonormal columns. The terminal ring has one learned
unit vector v_j in R² per sector, with Q_j=I-v_j v_jᵀ. Thus the terminal
penalty leaves four directions unpenalized in total; it does not receive the
true signal subspace. The 320 map entries and 8 terminal-vector entries are
stored even though orthonormality reduces degrees of freedom. No biases,
decoder head or learnable coupling strengths.

For a route field a_l,j>0, use

```
E_a(x;h,Theta) = 0.1/2 ||x_0-h||²
 + (1/2) sum_(l=0)^2 sum_j a_l,j ||x_l,j-D_l,j x_(l+1),j||²
 + (1/2) sum_j a_3,j ||Q_j x_3,j||²
 + 0.001/2 sum_(l=0)^3 sum_i (x_l,i-x_l,(i+1 mod n_l))².
```

The lateral term connects sectors within each ring; sectors are not separate
modules. Unlike the zero-lateral analytical witness, this fixed term may bias
signal modes. Its coefficient and the exterior tie 0.1 are declared here,
not inherited as proven optimal settings. The actual output is x_0*, never a
projection of selected interior coordinates. No centre term is present.

For fixed routes, full column rank and the boundary anchor imply a strictly
positive-definite Hessian H. Positive lateral/terminal terms preserve this.
Inference minimizes this energy by a float64 Cholesky direct solve on all
80 activity coordinates. Structure and routes remain fixed during that solve.
Use no iterative optimizer in the primary implementation. The condition number,
relative residual and symmetry are checked; no diagonal repair is allowed.
The direct solve represents the reciprocal equilibrium, not evidence that
iterative dynamics are necessary or fast.

## 3. Familiarity routes: two passes on one observation

For each observation h separately, first solve with all a_l,j=1 to obtain a
pilot state z. Let epsilon=1e-12. Define

```
r_l,j = ||z_l,j-D_l,j z_(l+1),j||² /
        (||z_l,j||²+||D_l,j z_(l+1),j||²+epsilon),  l=0,1,2
r_3,j = ||Q_j z_3,j||² / (||z_3,j||²+epsilon)
g_l,j = 0.1 + 0.9 exp(-r_l,j/0.25)
a_l,j = 0.25 + 0.75 g_l,j
b_l,j = g_l,j / mean_j(g_l,j).
```

g is a compatibility heuristic, not a calibrated probability or an oracle
novelty measure. Matching residual zero gives g=1; every gate stays positive.
Inference uses a in a second solve. Learning uses b, which allocates the
same mean plasticity multiplier to each ring across its sectors. Thus the
test changes spatial allocation, not just the average amount of learning.
It does not promise a globally smaller update on an unfamiliar observation.

Routes depend only on h and current structure. In paired-view training, the
route for predicting h2 from h1 must not read h2, or conversely. A route is
computed once per observation per optimizer update; do not iteratively revise
it inside the second solve. Both pilot and second solve count as inference
work. No pruning or zeroing of poorly matched sectors.

For source acquisition warm-up only, a=b=1, with no pilot needed. For controls
defined below, transform g before computing a,b. No route formula is tuned.

## 4. Learning objective and exact update

For a minibatch of B=8 observed pairs, compute both directions of prediction:

```
J(Theta | fixed routes) = (1/(4 d B)) sum_i
    [||f_Theta,a(h1_i)-h2_i||² + ||f_Theta,a(h2_i)-h1_i||²], d=32.
```

f is the settled outer field. Differentiate through the **second equilibrium
solve**, holding pilot states and their routes fixed (stop-gradient). With
H x=b_input, the derivative is obtained by the adjoint solve
Hᵀ u=partial J/partial x and the residual derivative, or an independently
checked automatic-differentiation equivalent. Gates and pilot solutions are
not differentiated. This is a frozen-route surrogate update; monotonic descent
of the recomputed two-pass loss is not claimed. The first-view prediction
still excludes its target. For fixed parameters and fresh paired draws,
independent paired noise supplies the expected clean-risk objective up to a
constant. That identity neither applies as an unbiased estimate to reused
training errors after fitting nor turns the surrogate derivative into the
full gradient through routing.

Average the exact gradients of J over its 2B directions. For each parameter
block, compute b_bar_l,j as the mean of its 2B route multipliers. For a matrix
block D, project its unscaled gradient G onto its Stiefel tangent:
`G_tan = G - D sym(DᵀG)`. For v use `G_tan=(I-vvᵀ)G`.
Clip each projected block to Frobenius norm at most 1; this is clipping, not
unit-normalizing every nonzero gradient. With base step eta=0.1, update by
`D_new=qf(D-eta*d_l*b_bar_l,j*G_tan)` and
`v_new=normalize(v-eta*d_3*b_bar_3,j*G_tan)`.
Thus matching allocates the current minibatch's update budget, while the
predictive gradient supplies its direction. Positive block multipliers
preserve a descent direction for the fixed-route objective before retraction;
the fixed finite step and subsequent route recomputation need not decrease it.
qf is thin QR with R diagonal made positive. This retraction is identity on
an already orthonormal block up to numeric tolerance. Do not independently
reorient columns afterward. Zero/nonfinite normalization or loss of rank
stops the run; no restart. No momentum, weight decay, Adam state, line search
or learning-rate schedule.

Candidate depth multipliers are (1,1/2,1/4,1/8), rescaled to arithmetic mean
one across the four parameter groups: (32/15,16/15,8/15,4/15). Matching the
mean multiplier does not match actual update norm or FLOPs; report both.
Inference mobility is not changed with depth.

Process chronological minibatches of eight pairs. Perform exactly four
optimizer updates on the current minibatch, recomputing routes each time,
then discard it. No older raw pairs are accessible. The four exposures are
not four independent observations. Parameter gradients, route buffers and
activities are transient. Only learned parameters and phase survive a batch.

## 5. Source acquisition and initialization

Use s=256 source pairs. Acquire a phase and one signal weight per exterior
sector with frozen v2 four-fold CV and all-s refit (33 local fits), counting
all selection work. The paired directions are estimates, not evaluator truth.
For each normalized weight w, make a full orthonormal basis by taking w as
column one, then projecting coordinate axes in order into its orthogonal
complement with two-pass modified Gram-Schmidt; skip norms <=1e-10. Require
eight columns, without replacing w. Initialize D_0 with the first six columns,
D_1 with the first four coordinate axes in R⁶, D_2 with the first two in R⁴,
and v=(1,0). No extra source eigenvectors or clean modes enter initialization.

Apply one chronological source pass of the minibatch procedure (32 batches,
128 optimizer updates), with uniform depth multipliers and a=b=1. This shared
source procedure initializes every layered arm identically. It is not selected
for a development score. The initialized source checkpoint is copied once per
arm/branch; all source acquisition costs are charged to each standalone arm.
Discard source pairs and all source covariance buffers before target learning.
The immutable checkpoint is held by the experiment harness for independent
branches, not by the deployed learner for recall or reset within an episode.

## 6. Streams: stable, regional drift, and return

Use d=32,k=4,b=8, source U,V and orthogonal local drift directions W from
the representation-reuse generator recipe. One random circular source phase
in {0,...,7}; unconditioned Gaussian coefficients, independent view nuisances,
`h_r=R(U_current c+2V g_r+0.5e_r)` with c shared within a pair. Source and
later streams are independent, zero population mean supplied, no centering.

After source acquisition, run three separate branches from the same checkpoint:

- **Stable:** B has U_B=U in all sectors.
- **Regional drift (primary):** one uniformly drawn sector j* changes to
  cos(60°)u_j*+sin(60°)w_j*; all other source directions stay fixed.
- **Global drift:** all four sectors change by 60° (descriptive).

Each branch presents 256 B pairs followed immediately by 256 fresh A-return
pairs with the original U. No episode label, j*, angle or boundary/reset
signal reaches a learner. The minibatch counter runs continuously, with
no special action at the return. Source and target phases stay fixed; this
study does not test changes to the support pattern.

Share B coefficient/noise draws across the three branches, and likewise the
A-return draws, to pair comparisons. Every branch is scored separately, not
treated as an independent world. Checkpoints after m={0,8,16,32,64,128,256}
pairs of B and after m={8,16,32,64,128,256} return pairs. The primary window
is m={16,32,64,128,256} in B. No budget is removed after observing results.

## 7. Arms and the comparisons they support

| Arm | Depth multipliers | g used for inference and learning | Role |
| --- | --- | --- | --- |
| `graded_matched` | candidate decreasing profile | computed at its actual sector | sole primary candidate |
| `uniform_matched` | all one | actual sector | isolates depth-rate profile |
| `reversed_matched` | reversed candidate profile | actual sector | tests rate ordering; descriptive |
| `graded_mean` | decreasing | each ring's mean g repeated at every sector | removes within-ring matching |
| `graded_shifted` | decreasing | g_l,(j+1 mod 4) applied to sector j | retains gate distribution but misassigns it |
| `frozen_layered` | no updates after source | matched, recomputed per observation | persistent-structure control |
| `phase_target_cca` | no layered learning | none | direct target-statistic reference |
| `pooled_local_cca` | no layered learning | none | history-retaining direct reference |

All layered arms use their own current pilot states; g is never borrowed from
the candidate. For shifted gates the same permutation applies at every ring
and every observation. For mean gates b=1 exactly. These interventions jointly
change inference and plasticity; a pass supports that combined mechanism,
not separate causal claims for each component.

Direct references use the acquired source phase and frozen v1 CCA. The
phase-target reference starts empty after source and accumulates **all** B
and subsequent A-return pairs, without boundary resets. Pooled local starts
with source sums and adds the same chronological pairs. Solve at each nonzero
checkpoint. At B m=0 phase-target is undefined (null with reason); pooled
deploys its source fit. They do not use the layered source warm-up. Count
their true acquisition, statistics state and solves; unequal training work
does not imply equal-compute superiority. Neither is a rescue primary arm.

For every layered arm, implement a Schur-complement/direct-map reference for
the pilot and second fixed-route solve and compare full reconstructed states.
This is an equivalent implementation, not an independently trained extra arm.
The two-pass overall map is input dependent and generally nonlinear; do not
replace it by a single constant linear matrix.

## 8. Evaluation: fresh draws, not projector identities

Within each world generate independent evaluator-only sets of 256 noisy
observations and clean signals for source A and B. The B base draws are shared
across branch transformations. The A set also evaluates A-return; no evaluator
data are given to training or routing. Reuse the same sets at all checkpoints
and across arms for paired comparisons. An evaluator draw is a fresh sample
relative to training, not a new independent replicate each time it is reused.

Primary E is empirical mean `||f(h)-s_clean||²/k` on the indicated set.
Also report clean-input response distortion `||f(s_clean)-s_clean||²/k`
using a fresh inference and routes for that clean input. This distortion is
a separate diagnostic, **not** the bias term in an additive signal/noise
decomposition of nonlinear f. Do not use `1-tr(PS)/k`, a constant-projector
noise term, or exact-population language for these routed scores.

At each checkpoint evaluate both A and B distributions, with true regional
drift locations used only for diagnostic per-sector errors. Source forgetting
is E_A(current)−E_A(source checkpoint) for that arm's own inference policy.
All arms also report absolute E_A, since routing can change the source
checkpoint response even with identical parameters. Return evaluation tracks
recovery on A and any loss on B. No selected checkpoint or best-so-far result.

Report zero output, raw input and oracle signal projection on the same draws,
with their expected in-distribution E values 1,6,0.25 as checks, not finite-set
equalities. Oracle projectors never initialize or guide learners. Report
per-ring parameter changes, per-example gate distributions and correlations
with evaluator-only drift/error as descriptive diagnostics. A gate's formula
alone is not evidence of correct familiarity recognition.

## 9. Primary decisions

For each of 64 independent confirmation worlds, compute mean E_B and mean
E_A over the five primary B checkpoints, regional-drift branch only. Denote
the candidate C. Positive differences below favor C:

| Claim | World-level difference | Required population mean |
| --- | --- | --- |
| H1 matching improves target reconstruction over mean routing | mean E_B(graded_mean) − mean E_B(C) | > 0.01 |
| H2 matching improves target reconstruction over misplaced routing | mean E_B(graded_shifted) − mean E_B(C) | > 0.01 |
| H3 slower inner learning preserves source reconstruction | mean E_A(uniform_matched) − mean E_A(C) | > 0.01 |
| H4 source preservation does not excessively harm target adaptation | mean E_B(uniform_matched) − mean E_B(C) | > −0.02 |

H3 compares absolute source risk to avoid rewarding an arm for a worse
initial response; source checkpoint risks must also be reported. Candidate
and uniform matched share the same initial parameters and routing policy.
An overall pass is a relative mechanism result. Absolute errors, frozen and
direct references determine whether the resulting module is practically useful;
beating weak controls does not establish usefulness by itself.

Use paired one-sided Student-t bounds over 64 world differences:
`mean(D)-t_63(0.9875)*SE(D)`. Each must strictly exceed its margin; all four
are required for overall pass. Report all estimates and bounds independently
of outcome. Zero variance gives the constant difference with degeneracy
disclosed. Coverage is approximate; world-level sampling includes finite
evaluation-set randomness. No test points, checkpoints, sectors or branches
are treated as independent replicates.

Reversed rates, stable/global drift, return-to-A, direct controls, clean-input
distortion and all other contrasts are descriptive (world-paired means and
95% t intervals). No arm substitution, threshold relaxation, extra worlds,
or alternate routing after observing a bound. Failure of H4 is failure to
establish noninferiority, not proof of excessive inferiority.

## 10. Seeds, development and numerical gates

Root 2026102410; NumPy default_rng(SeedSequence([root,stage,w,part])). Stage
0 development w=0..7; stage 1 confirmation w=0..63; stage 2 timing w=0.
Part 0 U,V uses the frozen interleaved template recipe; part 1 reserved;
part 2 phase via integers(0,8); part 3 length-256 source training; part 4
length-256 B training; part 5 one (4,8) Gaussian drift array projected off
U,V; part 6 j* via integers(0,4); part 7 length-256 A-return training;
parts 8,9 length-256 evaluator A,B arrays. Each stream draws c,g1,g2,e1,e2
in that order with shapes (256,4),(256,4),(256,4),(256,32),(256,32).
Evaluator uses its first noisy view and clean signal; unused second-view
draws preserve the layout and are explicitly counted as generated, not fitted.
No shuffling or stochastic optimization randomness. Freeze array order.

Development is correctness-only. All hyperparameters, including four updates
per minibatch and eta=0.1, are fixed by this draft; poor development quality
is disclosed, not used to choose a rate or gate. Any later design revision
needs a dated pre-confirmation amendment; after confirmation it needs new seeds.

Before freeze require:

1. Frozen generator/CCA/CV dependencies match original manifests; unchanged
   relevant suites and all new tests pass. Record complete counts/status.
2. Supports, circular lateral neighbors, sector indexing and QR initialization
   match hand cases; inferred phase matches frozen v2. Every Stiefel and
   terminal constraint holds to 1e-9 after updates.
3. Independent residual-energy expansion, analytic gradient and Hessian agree
   within 1e-8 on nondegenerate fixtures; positive Hessian and relative solve
   residual `||Hx-b||₂/max(1,||b||₂) <=1e-10`. Schur and full solves agree
   in relative/absolute 1e-9 scale for all rings and each pass.
4. Fixed-route implicit gradients agree with central finite differences at
   step 1e-6 to 1e-5 relative/absolute scale, holding gates fixed in both.
   Separate fixtures check route recomputation, averaged b preconditioning,
   clipping, retraction and equal/reversed rate profiles. Assemble terminal
   curvature as QᵀQ when checking off-manifold finite differences, rather
   than assuming Q²=Q for a perturbed, non-unit v. Do not incorrectly
   compare stop-gradient derivatives with a recomputed-route finite difference.
5. Gate bounds, mean-one b and permutation invariance of their multiset hold.
   Changing a prediction target cannot change that prediction's routes or
   output before a learning update. Evaluator truth cannot change any update.
6. Verify exact stream isolation, nested checkpoints, continuous return
   processing, no boundary reset, no source/raw replay beyond current batch,
   and no influence between independent branches. Count actual vs charged work.
7. Linear fixed-route trace risk matches an independent full error map on
   constructed cases; the routed evaluator instead agrees with explicit
   sample-wise errors. Clean-input diagnostics are not summed with noise.
8. Every arm completes a full development trajectory with finite values.
   Two development executions produce byte-identical scientific JSON in the
   pinned environment; runtime is separate. No quality threshold is a gate.

Commit contract, sources, tests, development report and manifest (hashes of all
dependencies, source/test files, environment, RNG, arm/metric definitions and
full test summary) before first confirmation read. Refuse confirmation if a
frozen source is dirty, uncommitted or mismatched. Output-only dirtiness is
listed by exact path. Use one process, one BLAS thread, pinned float64
Python/NumPy/SciPy; if autodiff is used, pin and record it too.

## 11. Computation contract

Let M=80,d=32,J=4, P=328 stored floating parameters, q=8, B=8 and K=8 phase
candidates. Source CV retains v2's fitting/selection orders and 33 local fits;
basis completion costs O(Jq³). Source warm-up has 128 gradient updates. Each
branch has 32 B batches and 32 return batches, hence 256 gradient updates
per adapting layered arm, 128 per episode. Each update predicts 16 directions;
each uses a pilot and second solve, and one adjoint solve. Source warm-up
omits the pilot (routes fixed). Count every repetition and actual reuse.

Hessian construction from residual factors is O(sum_l J n_l,sector
* n_(l+1),sector² + M) for sparse block terms/lateral edges, at most O(M³)
in a dense generic implementation. The reference Cholesky is O(M³) time,
O(M²) workspace; a solve with cached factor is O(M²). The pilot factor can
be reused until parameters change. Route-dependent second factors generally
cannot be reused between inputs; reuse each for its adjoint. Building block
derivatives and retractions adds O(P+sum_l J n_l,sector*n_(l+1),sector²).
Charge implementation reality if it uses a more expensive dense derivative.

Per routed learning update the dominant reference cost is
O(M³ + 2B(M³+M²) + retractions); source fixed-route updates can share one
factor across observations and adjoints. Evaluation also requires both passes
and per-input second factorizations. Report its substantial cost separately
from learning. No efficient-local or depth-speed claim follows from sparse
parameters when using a dense solver. Schur references are diagnostic costs,
not credited free to candidates or charged as their learning work.

Persistent learner state is 328 float64 values, phase/index metadata and no
optimizer momentum: 2,624 parameter bytes plus metadata at d=32. Immutable
fixed ring topology is reported separately. Current minibatch is 2Bd values;
states/routes/gradients and solve caches are temporary, as is source CV data.
Retain only current-batch samples until its fourth update. Direct references
retain O(db) statistics and count, plus O(d) basis. Harness checkpoint clones,
evaluation sets and report arrays count in whole-study memory but are not
learner memory. Report allocated bytes, not merely parameter counts.

Runtime JSON includes source/environment hashes, revision, command, exact
dirty paths, platform/CPU/BLAS/threads, wall and process CPU per phase,
whole-process RSS units/scope, executed/charged acquisitions, candidate scores,
unique pairs and repeated exposures, pilot/second/adjoint solves, factorizations,
parameter updates/retractions, route evaluations, evaluator calls and failures.
Unavailable FLOPs/workspace quantities are null with reasons. No unreported
tuning, gradient steps, rejected runs or solver retries.

Descriptive timing uses stage-2 world 0 at M=80 only: post-source checkpoint
and post-32-B-pair checkpoint for regional drift. All layered policies plus
both direct references; one discarded warm-up and seven fresh repetitions of
one update, one-input inference and a 32-input batch. Freeze input batch and
parameters across repetitions; use copies, not continued training. Frozen
arms report no update cost. Inference repeats use seven blocks of 20 calls,
allocation included, generation excluded. Time pilot, route, second solve,
adjoint and retraction separately; include factor-cache construction and state
copy scope explicitly. Report raw times, medians/IQR and isolated worker RSS
with idle baseline. This is one-size instrumentation, not a scaling claim.

Report quality versus cumulative source+adaptation cost, independent pairs,
inference latency and persistent bytes. At final B and return checkpoints,
report measured training cost plus Q×measured inference for Q={1,100,10000}
as arithmetic estimates. No multiple-module or repeated-episode amortization
model is added to this study. All timing warm-ups and repeats count as study
cost, distinct from a standalone learner's charged trajectory.

## 12. Failures, artifacts and interpretation

Nonfinite values, failed positive definiteness/rank, constraint or solve
residual violation, or correctness failure stop scoring. Record world, arm,
branch, batch, update, pass and solver diagnostics; mark incomplete. Do not
repair a covariance, increase regularization, replace a world or omit a failed
observation. Resource-limited runs are truncated and cannot satisfy claims.
Implementation-only fixes preserve the method, disclose accessed outcomes,
refreeze and rerun all confirmation worlds; methodological fixes need new seeds.

Planned files: `src/single_module_plasticity.py`,
`experiments/single_module_plasticity.py`, `tests/test_single_module_plasticity.py`;
`reports/single-module-plasticity-v1-{manifest,development,results,runtime}.json`;
`docs/single-module-plasticity-v1-results.md`.

Classify compatibility recognition, target adaptation, source retention,
return recovery, ring-rate ordering, computation and absolute usefulness
separately. Small parameter changes are not proof of retained function. A
pass supports this single-module, combined routing/plasticity policy under
the declared stream. No automatic transition to multiple modules follows;
any such transition requires a separate reason and contract.
