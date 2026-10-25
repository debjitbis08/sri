# Single-module mechanism diagnostics v1

**Status:** prospective contract, 2026-10-25. No implementation, selection,
or execution under this contract has occurred. Commit this document before
development; commit tested sources, selected settings and a manifest before
generating the diagnostic worlds. This is a mechanism study with descriptive
estimates, not a new attempt to pass the plasticity-v1 hypotheses.

Follows the [plasticity study](single-module-plasticity-v1-results.md), its
[contract](single-module-plasticity-v1-contract.md), and the
[computation protocol](computation-reporting.md). The linked
[abstraction preflight](single-module-abstraction-preflight-v1-contract.md)
addresses a different question. Neither study establishes abstraction from
depth, small parameter changes, or successful denoising alone.

## 1. Questions and evidence already seen

1. Does a drifted sector receive a smaller plasticity multiplier despite
   larger prediction error?
2. Does that allocation produce smaller actual updates, and does removing
   it improve adaptation at the same checkpoint?
3. Does source forgetting accompany outer-map changes while inner maps
   remain relatively stable? What happens when their updates are disabled?

Plasticity-v1 confirmation results have already been read. They motivated
these questions and cannot be used as fresh confirmatory evidence. Its saved
gate means are evaluator statistics, not training multipliers. In general
`mean(g / mean_sector(g))` cannot be recovered from `mean(g)`. Its saved
parameter-change and update norms aggregate sectors within each ring.

The reported absence of forgetting applies to regional branch means. In
the existing global branch, graded-matched window forgetting is 0.003850,
with descriptive 95% interval [0.002276, 0.005425]. This motivates including
global drift without replacing the regional primary branch of the old study.
Do not rewrite or pool the old scientific artifacts.

## 2. Fixed substrate and scope

Keep plasticity-v1 geometry, energy coefficients, generator, source phase
acquisition, constraints, exact Cholesky inference, gates and implicit
fixed-route gradients. Dimensions are d=32, M=80, four rings and four sectors,
with 328 stored float64 parameters. Source phase acquisition still uses eight
candidates, four folds and 33 local fits. No phase is reacquired after drift.

Extensions are explicit: a bounded source-training selection, independent
choices of inference and plasticity routes, explicit zero learning rates for
frozen parameter groups, and additional diagnostics. Implement these in new
files, importing frozen functions where possible; do not patch frozen v1 code
or change its globals. No new optimizer, architecture, gate formula or loss.

Use one process, one BLAS thread and the repository's pinned NumPy/SciPy
environment. No neural framework is required.

## 3. Fresh worlds and source-learning selection

Root `2026102501`; RNG keys `[root, stage, world, part]`. Stage 0 uses eight
development worlds, w=0..7. Stage 1 uses 32 diagnostic worlds, w=0..31.
Reuse the v1 part meanings 0..9 with the following explicit sizes:

| Part | Draws |
| --- | --- |
| 3 | 4,096 source pairs, nested prefixes |
| 4, 7 | 256 B pairs and 256 A-return pairs respectively |
| 8, 9 | 256 evaluator A and B pairs respectively |
| 10 | 512 independent source-validation pairs |
| 11 | 24 fresh probe B pairs: three consecutive batches of eight |

Each stream draws c,g1,g2,e1,e2 in that order, with the v1 shapes generalized
to its declared length. Regenerate branch observations from the same base
draws. No branch or arm is an independent replicate. Parts 8/9 supply clean
evaluation targets only to the evaluator; part 10 supplies noisy paired
validation only. Part 11 never enters the continuing learner trajectory.

Acquire phase and initialize parameters from the first 256 source pairs.
For each source step size eta in {0.03,0.1,0.3}, start from the same acquired
parameters and train chronologically on all 4,096 pairs, four updates per
minibatch of eight, with uniform depth rates and `a=b=1`. Save checkpoints
after 256, 1,024 and 4,096 pairs: 128, 512 and 2,048 updates. The first 256
pairs are deliberately reused for acquisition and warm-up, as in v1;
count this exposure, not additional independent examples. No replay thereafter.

There are nine candidate configurations, but three shared-prefix training
runs per development world. Select one (eta, source-prefix length) globally
by the lowest mean bidirectional fixed-route paired loss on part 10 across
the eight worlds. Within 1e-12, prefer fewer updates, then smaller eta.
Record all nine scores, including poor candidates. Development clean-risk
curves may be reported but cannot change this selection or the budget.

Freeze the selected setting before stage 1. Diagnostic worlds train only
that source prefix at that eta. Report source clean risk at initialization
and all reached checkpoints, under fixed, matched and ring-mean inference.
Also report frozen acquired CCA and an acquired-phase CCA refit using the
same source prefix; the latter's extra fit and retained statistics are charged.
Source-validation pairs are selection data, never learner updates.

Selection at the largest budget, a falling last segment, or poor absolute
source risk must be disclosed. A finite-budget plateau is not proof of a
global optimum. The diagnostic study proceeds even when source learning is
weak, but cannot claim to settle retention in a well-trained representation.

## 4. Adaptation arms

Every arm starts from the same selected source parameters. Use all three
v1 branches: stable, regional and global. Each has 256 B pairs followed by
256 A-return pairs, four updates per batch of eight, base eta=0.1 throughout
adaptation. This eta is not selected. No boundary reset or source replay.

For each input compute raw pilot gates g. Define matched inference
`a_M=0.25+0.75g`, mean inference
`a_mean=0.25+0.75 mean_sector(g)` broadcast within a ring, and matched
plasticity `b_M=g/mean_sector(g)`. These choices are independent: changing
inference policy does not change which raw pilot gates define b.

| Arm | Inference a | Plasticity b | Outer-to-inner rate profile |
| --- | --- | --- | --- |
| route_MM | a_M | b_M | (1,1,1,1) |
| route_MU | a_M | 1 | (1,1,1,1) |
| route_meanM | a_mean | b_M | (1,1,1,1) |
| route_meanU | a_mean | 1 | (1,1,1,1) |
| retain_uniform | 1 | 1 | (1,1,1,1) |
| retain_slow_inner | 1 | 1 | (1,1/2,1/4,1/8) |
| retain_outer_only | 1 | 1 | (1,0,0,0) |
| retain_inner_only | 1 | 1 | (0,1,1,1) |
| frozen | 1 | 1 | (0,0,0,0) |

Do not mean-normalize the retention profiles: equal outer rates are the
control. Total update norms can differ and must be reported. Zero-rate
blocks are copied exactly, with no QR or renormalization. Gradients may
still be computed for diagnostics; count that executed work.

Checkpoint B at m={0,8,16,32,64,128,256}; return at
m={8,16,32,64,128,256}. Scores, clean-input distortion and oracle/zero/raw
controls retain v1 definitions. All forgetting is relative to that arm's own
source checkpoint and inference policy. Report absolute source risk too.

## 5. Per-update diagnostics and controlled one-step probes

Before each actual update, log per (ring, sector): mean raw g, actual b_bar,
Euclidean-gradient norm, tangent-gradient norm, clipped-gradient norm,
pre-retraction step norm and actual post-retraction change norm. Save sector
paired prediction error before updating, averaged over the 16 directions.
Prediction error is the mean squared Euclidean residual in the sector's
eight output coordinates; it is not a clean-risk estimate after fitting.
Log clipping incidence, effective eta*d*b_bar, and constraint/solve diagnostics.

Q1 contrasts the true drifted sector with the mean of the other three,
separately for b_bar and error. Q2 does the same for gradients, intended steps
and actual changes. Evaluate sectors using true phase only in the evaluator;
learner-sector summaries conditional on correct acquired phase report their
denominator. Keep misacquired worlds in absolute risks and main arm contrasts.

At route_MM B checkpoints m={0,32,256}, clone the same parameters into two
one-step probes. Use the corresponding part-11 minibatch, its raw pilot gates
and the same matched inference a for both. Compute a single common gradient.
Apply either actual matched b_bar or b=1, with the same eta and depth profile.
The common starting state and gradient isolate update allocation at that step.
Measure per-sector applied changes and before/after clean A/B risks on the
shared evaluator arrays. Discard both clones afterward. No probe result can
affect training, checkpoint selection or configuration choice.

This estimates a local intervention, not the effect of a complete alternative
learning trajectory. The full route_MM versus route_MU and route_meanM versus
route_meanU trajectories provide that complementary evidence. Comparing
inference policies changes the predictive gradient as well as the output;
do not label that contrast a pure plasticity effect.

## 6. Retention and interpretation

For Q3 report source-risk change against each ring's displacement from its
source parameters and cumulative applied path length. Specify Frobenius norms
over each entire ring group; also retain per-sector values. Coordinate motion
is not a measure of information lost, and compensating map changes can alter
parameter distance without comparable functional change.

Use the retention arms for interventions. Compare slow-inner versus uniform
at the same outer rate; outer-only versus frozen; inner-only versus frozen;
and the full model versus each restricted learner. Always show target gain
beside source cost. These interventions change the optimization trajectory
and total learning effort; they do not constitute an additive decomposition
of forgetting, or establish where an abstract relation is stored.

For all trajectory contrasts, report window means over B checkpoints
{16,32,64,128,256}, final B, and final return. Regional and global results
are separate. Negative source-risk changes are improvement, not forgetting.
If retention cannot be tested because risk improves, report that limitation.

## 7. Analysis, correctness and stopping

For trajectory and intervention contrasts, reduce to one value per world
first, then report paired means and two-sided 95% Student-t intervals over
32 worlds. For update-allocation summaries, average the updates within each
world's B episode before constructing intervals. Show per-ring summaries.
Intervals are descriptive and not simultaneous; there is no overall pass,
no selected contrast, and no counting examples, updates or rings as replicates.
Associations within trajectories are descriptive, not causal evidence.

Before freezing require: frozen dependency hashes; exact v1 equivalence on
fixtures for supported settings; independent a/b-policy tests; finite-
difference checks with routes held fixed; correct before/after sector logs;
unchanged zero-rate blocks; no probe mutation or evaluator leakage; exact
stream isolation; direct and Schur solve agreement at v1 tolerances; and two
byte-identical small scientific fixture executions. Record test counts/status.

Apply v1 residual, rank, finite-value and constraint tolerances. Failure stops
the affected run and marks the study incomplete; do not omit worlds, repair
the Hessian, replace seeds or select another configuration. No additional
tuning after stage 1 begins. Method changes need a new dated contract and
new diagnostic seeds; all previously inspected outcomes must be disclosed.

## 8. Complexity and measured computation

Let L=4,J=4,M=80,d=32,P=328,B=8, and T be optimizer updates. With K=8 phase
candidates and F=4 folds, source selection uses KF+1=33 local fits. Each
local fit on n pairs costs O(n d q + J q^3), q=8, including block statistics
and dense factorizations; validation costs O(n d q) per candidate/fold as an
upper bound. Account for actual fold sizes, repeated exposures, refits and
basis completion O(J q^3). No selected configuration is charged only its
winning fit when reporting whole-study selection cost.

Dense curvature construction costs O(LJ M^2) per parameter state; routed
assembly O(2B LJ M^2) per update. A routed update costs
O(M^3 + 2B(M^3+M^2) + 2B LJ M^2 + retractions), with one pilot factor,
2B second factors, and 2B pilot, second and adjoint solves each. Fixed-route
updates share a factor and cost O(M^3+2B M^2+LJ M^2+retractions).
Retractions cost O(sum_l J s_l s_(l+1)^2). Count the executed implementation,
including gradients for frozen groups. Inference on N inputs costs
O(M^3+N(M^3+M^2+LJ M^2)) routed or O(M^3+N M^2) fixed-route.

Selection executes 8*3*2,048=49,152 source updates, plus nine validations
per world and acquisitions. Stage 1 executes 32*T_source source updates,
32*3*8*256=196,608 adaptation updates and 32*3*3*2=576 one-step probe
applications. Shared probe gradients, source acquisitions, cached factors
and branch clones must be distinguished from independently charged learner
work. All evaluator and Schur diagnostics are counted separately.

Persistent parameters occupy 2,624 bytes plus phase/index metadata; optimizer
state is zero. Current minibatch occupies 4,096 bytes. Dense curvature cache
occupies LJ*M^2*8=819,200 bytes; factor/state workspace is temporary and must
be measured from actual allocations (including simultaneous per-input
Hessians/factors). Source acquisition may transiently retain its 256 pairs;
the training learner retains only its current batch. Harness source arrays,
validation/evaluation sets, checkpoints, probe clones and diagnostic logs are
whole-study storage, not learner state. Logs scale as O(W A C T L J Z), where
Z is the number of recorded scalar diagnostics; report their actual bytes.

Write deterministic scientific JSON separately from runtime JSON. Runtime
records hashes, revisions, dirty paths, commands, environment/thread limits,
wall/process CPU, peak RSS with units/scope, fits, candidate scores, pairs,
exposures, updates, route evaluations, factors, solves, probes and failures.
Timers are nested: do not add a learning total to its contained stage times.
Include rejected development configurations and correctness/reproducibility
runs in total cost. Unavailable quantities are null with reasons.

Report risk versus independent pairs, measured training/selection time and
persistent bytes. Inference totals divided by input count are explicitly
batched evaluator throughput, not standalone latency. This study makes no
speed or scaling claim and requires no separate timing sweep. Historical
unavailable costs are disclosed rather than retrofitted.

## 9. Deliverables and next decision

Use new source/runner/tests and report prefix
`single-module-mechanism-diagnostics-v1` for manifest, development, results
and runtime JSON; place the human-readable result in docs/. Keep README
focused on the project.

The result must answer Q1, Q2 and Q3 separately, including unresolved cases.
It can motivate a subsequent learning-rule contract. Abstraction claims
require the relational task and interface feasibility addressed by the linked
preflight, followed by a separately frozen learning/transfer experiment.
