# Representation reuse v1 — counted acquisition, frozen reuse and controlled drift

**Status:** reviewed prospective contract, 2026-10-24. No implementation,
development run or confirmation draw exists under this contract. Commit the
reviewed contract before implementation; commit the source/environment manifest
after correctness gates and before confirmation. Review clarifications are
listed in section 10; none changes a hypothesis, margin, budget or arm.

Follows [v2 results](locality-adaptation-v2-results.md),
[phase-selection analysis](phase-selection-analysis.md), and
[transfer analysis](broad-structure-transfer-analysis.md). Apply
[computation reporting](computation-reporting.md). Frozen historical sources
and artifacts remain unchanged.

**Explicit research-sequence revision:** v2 failed its overall gate. This
study proceeds despite the earlier pass-contingent transfer plan. It neither
rescues v2 nor assumes its n=16 acquisition deficit is solved. The question
changes from learning a partition anew at every budget to using structure
learned from a separately counted source history. The retrospective analysis
motivates this design but supplies no confirmation worlds.

## 1. Questions and scope

1. With an unchanged target subspace and independent fresh coefficients,
   does a frozen source representation improve on learning from few target
   pairs alone?
2. When signal directions drift inside a stable partition, does retaining
   the source-selected phase while refitting target weights improve on
   freezing everything, without losing much to fresh target phase selection?
3. Descriptively, how do these policies compare with local/global methods
   retaining all source and target sufficient statistics, and what happens
   to source reconstruction after adapting?

The first comparison grants the reuse method additional, explicitly paid
history. It measures target acquisition savings, not equal-total-example
superiority. The history-matched comparisons are mandatory and can show that
simple covariance retention explains any advantage.

No previous activation is retained. Source and target coefficients are
independent. Coefficient correlation, changing partition boundaries, dense
mixing, online tracking of a continuous trajectory, neural plasticity and
reciprocal dynamics are excluded. This narrows the broader transfer plan to
representation reuse; a later activation-memory study must cross temporal
correlation with drift and include memoryless shrinkage controls.

## 2. Worlds and observed streams

Use d=32, k=4, b=8 and eight cyclic partition phases. Source template U and
texture V follow v1's DCT-II recipe: one normalized signal combination of
modes 0–2 and one texture combination of modes 3–7 in each block. Draw a
source partition phase p uniformly from {0,...,7}; unlike v1's aligned
condition, phase 0 is not privileged. Apply the corresponding circular shift
R_p to all source and target components. Learners receive neither p nor the
regime, U, V, drift direction, or drift angle.

In each unshifted block j, draw an independent standard Gaussian b-vector a_j
and define

```
w_j = normalize((I - u_j u_jᵀ - v_j v_jᵀ) a_j)
U_theta[:,j] = cos(theta) U[:,j] + sin(theta) W[:,j].
```

Here u_j, v_j, w_j are the b-site restrictions of column j to unshifted block
j, and the projection is block-local. All columns have disjoint supports; W is
orthonormal and orthogonal to U,V.
Angles theta={0,15,30,60,90} degrees rotate every signal direction equally.
The source/target signal subspaces have the declared principal angles and
remain orthogonal to the same nuisance V. A zero/nonfinite residual norm
stops the run; no redraw. The correct source partition continues to contain
the target signal. Thus this is weight drift with a stable grouping, not a
claim about adapting boundaries or arbitrary mixing.

For source observations, U_current=U; for targets, U_current=U_theta:

```
h1 = R_p (U_current c + 2 V g1 + 0.5 e1)
h2 = R_p (U_current c + 2 V g2 + 0.5 e2).
```

All coefficients are unconditioned standard Gaussians; c is shared within
a pair, and g1,g2,e1,e2 are independent of c and each other. Source and target
streams are independent. Zero population mean is supplied; no empirical
centering. Generate length-256 source and target streams separately. Share
the target coefficients and noise across angles to pair contrasts; angles
are not independent replicates.

Source budgets s={16,64,256} are nested prefixes. Target budgets
m={0,4,8,16,32,64,128,256} are nested prefixes of the separate target stream.
Each angle is an independent adaptation branch starting from the same source
checkpoint, not a sequential drift trajectory. There is no target-to-source
or cross-angle update. Primary s=256; s=16 and 64 are descriptive acquisition
sensitivity checks that retain the known small-source-budget risk.

## 3. Estimators, source acquisition and retained state

Import frozen v1 CCA and v2 four-fold CV/refit conventions without modifying
their modules: local rank one per block, global rank k, ridge
0.1 tr(T)/block_dimension, Cholesky whitening and full symmetric eigh,
canonical ties/orientation, orthogonal span reconstruction, float64.
Numerical eigenspace and score tolerances are exactly v1/v2. There is no
iteration limit because solves are direct, and no fallback or ridge increase.

For each source budget, acquire a local phase p_hat and source weights with
`cv_refit_cca`: four folds, eight phases, one all-s refit (33 local fits).
Every pair belongs to fold i mod 4 for one-based index i. Select by mean
out-of-fold h1ᵀPh2, then refit on all s. There is no new selection rule.
Acquire the global source reference by one all-s global CCA fit.
Implementation: selection calls v2 `cv_scores` and v1 `select` unchanged; the
refit and the global fit build the sums below once and solve them with v1
`cca_top`, so the sums are exposed without extra construction. A gate checks
bitwise equality with v2 `fit_cv_refit` and v1 `fit_global` on identical
observations.

For sums of observed pairs define

```
C = sum_i (x_i y_iᵀ + y_i x_iᵀ)/2
A = sum_i (x_i x_iᵀ + y_i y_iᵀ)/2.
```

Local versions retain only blocks at p_hat. A fit on q observations uses
K=C/q and T=A/q with the unchanged ridge rule. Source and target sums add
exactly for a pooled fit; no forgetting factor is selected. These statistics
are learned history and count as retained state. No raw source examples are
retained after acquisition. CV may temporarily retain observed pairs during
the acquisition/target fitting phase; count this workspace separately.

| Arm | Target action | State available after source acquisition |
| --- | --- | --- |
| `frozen_local` | none | source phase and source local basis |
| `phase_reuse_target` | fit local weights on target pairs at p_hat | source phase only; source basis is released |
| `pooled_local` | fit at p_hat from source+target block sums | phase, source block sums and count |
| `frozen_global` | none | source global basis |
| `pooled_global` | fit from source+target full sums | source full sums and count |
| `fresh_local_cv` | v2 CV and refit using target pairs only | none |
| `fresh_global` | global CCA on target pairs only | none |

The first three arms pay the full local acquisition cost, including the
source refit even if its basis is subsequently released. Pooled-global and
frozen-global pay source global acquisition; pooled-global additionally
retains the sums. If sums are not exposed by the source fitter, their extra
construction must be timed and counted. The two fresh arms pay no source
acquisition and receive only m pairs.

At m=0, only frozen and pooled arms are defined; pooled arms deploy their
source fit. Target-only arms report null with reason "no target pairs".
No undefined arm is replaced with an oracle or source weights. Primary
contrasts use m>=16 and are unaffected by these nulls.

For pooled arms and `phase_reuse_target`, update target sums once per arriving
pair and solve at the seven nonzero checkpoints. No intermediate solves.
Each branch starts with zero target sums. Fresh-global likewise updates
target sums; fresh-local CV reruns from scratch at each checkpoint on the
target prefix, so it retains raw target pairs as declared workspace. Fresh
arms do not depend on s; they execute once per world and angle and are
compared with every source budget. This
declared reuse is part of the implementation, with retained bytes and all
executed work counted. Report both cumulative stream cost through each
checkpoint and isolated endpoint cost from a source checkpoint. Neither
may be mislabeled as the other. No tuning of weights, folds or update rates.

The experiment harness may share identical source acquisitions and identical
fresh target fits across comparisons; charge the complete standalone cost
to every policy using them and record actual execution counts separately.
Cross-policy caching must never make an arm appear free. A source acquisition
is charged once per standalone branch, not again at each nested checkpoint.

## 4. Evaluation and controls

Evaluate deployed projectors with exact population moments on independent
fresh observations, never fitting or selecting with evaluator truth:

```
S_theta = R_p U_theta U_thetaᵀ R_pᵀ
N = R_p (4 V Vᵀ + 0.25 I) R_pᵀ
L_theta(P) = 1 - tr(P S_theta)/k
E_theta(P) = [tr((I-P) S_theta) + tr(PN)]/k.
```

Report target L, E and projected noise separately. Unqualified E, including
in section 5, uses N; section 4's stress noises are reported separately. Also report source L_0
and E_0 after every target update. Source forgetting is post-update source
risk minus the corresponding source-acquired local/global risk, not minus
an oracle. For phase-reuse this baseline is its released source local model,
retained only by the evaluator; the learner cannot access it. Fresh arms
report absolute source risk but have no pre-adaptation forgetting score.
Source risk and forgetting are reported under each evaluation noise.

Evaluator controls: zero output E=1, raw input E=6, oracle target signal
projector E=0.25, and the frozen oracle source projector with
L=sin²(theta), E=sin²(theta)+0.25. Report phase-bank structural floors for
source and target, including the floor at p_hat. Oracle quantities are
diagnostics only; they are never candidate arms or learned-performance bounds.

The low-noise condition N_low=R_p(4VVᵀ+0.01I)R_pᵀ and texture-16 condition
N_texture=R_p(16VVᵀ+0.25I)R_pᵀ are descriptive evaluation stresses using
unchanged fitted projectors. They are not new training conditions or primary
claims; recompute raw/oracle controls for each. Nuisance changes are thus
kept separate from the primary signal-drift experiment.

Report per-angle learning curves and quality against target pairs, total
pairs, cumulative training/selection time, inference latency and retained
bytes. Include every arm, even if global or pooled controls are preferable.
Do not infer equal-compute superiority from a conditional transfer gain.

## 5. Primary hypotheses and interpretation

Within each independent world, take the arithmetic mean of each metric over
m={16,32,64}, always at s=256. These target acquisition budgets are fixed
before implementation. No angle or budget may be selected from results.
Differences below are oriented so positive values favor the named reuse policy.

| Claim | Angle | World-level difference | Required mean |
| --- | --- | --- | --- |
| H1 stable reuse preserves signal better | 0° | L(fresh_local_cv) − L(frozen_local) | > 0.02 |
| H2 stable reuse reconstructs better | 0° | E(fresh_local_cv) − E(frozen_local) | > 0.02 |
| H3 updating reused grouping beats freezing weights | 30° | E(frozen_local) − E(phase_reuse_target) | > 0.02 |
| H4 reused grouping is noninferior to target reselection | 30° | E(fresh_local_cv) − E(phase_reuse_target) | > −0.02 |

H1/H2 establish only benefit from additional paid source history. H3/H4
test a specific weight-update policy under known-to-the-evaluator stable
grouping; they do not establish general drift robustness or preservation of
source performance. The latter is reported as a tradeoff, not presumed.
The two policies are named prospectively; no choosing the better policy after
seeing a world or angle. There is no claim of an autonomous drift detector.

Use 64 independent confirmation worlds, paired differences per world, and
one-sided Student-t lower bounds at 98.75% confidence:
`mean(D) - t_63(0.9875) SE(D)`. Each bound must be strictly above its margin.
All four required for overall `pass`; otherwise overall `fail`, with every
claim reported. Finite-sample t coverage is approximate. Zero sample variance
uses the constant difference as the bound, with degeneracy disclosed.
Failure to establish noninferiority is inconclusive, not proof of inferiority.

Paired means and 95% t intervals are descriptive for all other comparisons,
including global controls, pooled history-matched controls, source retention,
other angles and source budgets. No claim may be rescued by another arm.
A pooled/global win must be stated plainly. No universal adaptation lower
bound, neural-mechanism claim or retained-activation claim can follow.

## 6. Randomness, development and freeze

Root seed 2026102409. Use NumPy default_rng(SeedSequence([root,stage,w,part])).
Stages: 0 development, w=0..7; 1 confirmation, w=0..63; 2 timing, w=0.
Parts: 0 source U,V template coefficients with v1's interleaved block order;
1 reserved unused; 2 source phase via integers(0,8); 3 source stream;
4 target stream; 5 drift directions. Stream arrays are drawn in order
c,g1,g2,e1,e2 with shapes (256,k),(256,k),(256,k),(256,d),(256,d).
Drift draws use one (k,b) Gaussian array in block order before projection.
Timing appends d: [root,2,0,part,d]. Freeze shapes and loop order in manifest.
If wrappers call frozen v1 generator utilities, verify direct seed equivalence
and scope any root override so no historical module file is changed. v1
`make_world` is not called to build worlds, because it also draws part 1 and
draws part 2 with integers(1,b); the part-0 recipe is reimplemented and a gate
checks it against `make_world`'s U,V under a scoped root override.

Development is correctness-only, with no quality gate or tuning. Do not pool
development with confirmation. No extra worlds, seed replacement, optional
stopping or retrospective margin changes. Commit the contract first; after
correctness, commit implementation, tests, development report and manifest.
Confirmation refuses uncommitted/modified frozen sources or hash mismatches.
Any methodological change after confirmation starts needs a new contract and
fresh seed space. An implementation-only correction must preserve the method,
disclose accessed results, refreeze and rerun all confirmation worlds.

## 7. Computation and timing

Let K=8 phases, F=4 folds, b=d/k. Construction of local block statistics on
q pairs costs O(qdb), global O(qd²). Local direct solves cost O(db²), global
O(d³). Local source CV costs O(FK(3s/4)db+FKdb²+sdb+db²), plus O(Ksd)
selection; fresh target CV substitutes m for s. All rejected candidates and
refits count. Frozen reuse has no target fitting/selection work.

For locked-phase and pooled-local policies, target statistic updates cost
O(db) per pair and each checkpoint solve O(db²). Global updates cost O(d²)
per pair and checkpoint solves O(d³). For seven checkpoints sum all solves
and charge the source phase once. Isolated endpoint costs are O(mdb+db²)
local or O(md²+d³) global, plus acquisition where applicable. No iterative
solver or optimizer state; tuning trials and gradient steps are zero.

Local deployed basis is d float64 weights plus phase metadata; global is dk
weights. Inference encoding+reconstruction costs O(d) and O(dk), respectively.
Two local statistic matrices use O(db) storage, global O(d²), plus sample
counts. Separate source/target sums and checkpoint copies, if kept, count in
full. Frozen policies retain basis only; updating policies retain their sums
and current deployed basis. CV raw-pair workspace is O(qd), with O(db) local
statistics and discarded fold candidates; do not retain discarded sources
under a "no replay" label. Record actual bytes at acquisition, transition,
update and deployment, including metadata and all live buffers. Eigensolver
workspace is additional. Full raw replay is unnecessary for additive sums.

Follow the complete computation protocol in a separate runtime JSON:
wall/CPU by generation, source fitting, selection, statistic construction,
target updates/solves, evaluator, inference and writing; revision/hash and
exact dirty paths; interpreter/libraries/BLAS, CPU/OS, float64, one process,
one BLAS thread; RSS units and scope; fits, candidate scores, pair exposures,
independent source/target pairs, matrix dimensions, checkpoint solves, solver
failures and inference calls. Shared execution and per-policy charged costs
must have distinct fields. Unknown values are null with reasons, not zeros.

Descriptive timing panel: stage 2, d={32,128,512}, b=8, k=d/8, s=256,
m checkpoints as above, theta={0,30}; all seven arms. One discarded warm-up
and seven fresh independent execution repetitions on identical data. Time
both full checkpoint trajectory and separate endpoint m=32 and 256 paths;
do not derive one from another. Fresh spawned process per configuration,
idle-worker RSS baseline, and separate untimed allocation-tracing pass.
Report analytical LAPACK workspace estimates separately from measured memory.
Inference: single input and batch 256, one warm-up and seven blocks of 100
calls, allocation included, generation excluded. Report median, IQR and raw
times. All warm-ups/repetitions count in total study cost. No speed hypothesis.

Report `acquisition + adaptation + Q * inference` for Q={1,100,10000}.
For descriptive repeated reuse, R={1,10,100} independent target episodes
gives `acquisition + R*(adaptation + Q*inference)`, with the source checkpoint
reset each episode; fresh policies reacquire on target each episode. These
are arithmetic estimates using isolated endpoint times, not measured repeated
episodes or quality results. Also report total pairs s+Rm (reuse) versus Rm
(fresh); extra history is never hidden. No unsupported extrapolated speed win.

## 8. Correctness gates

Before confirmation require:

1. Frozen v1/v2 dependencies match their original manifest hashes and their
   unchanged test suites pass. Pin Python/NumPy/SciPy/BLAS versions and record
   full test counts and status in the new manifest.
2. U,V,W orthonormal and mutually orthogonal; shifted supports valid; exact
   source/target principal angles; theta=0 identity and theta=90 orthogonality.
   Structural best-phase floors are zero to 1e-9 at all angles.
3. Independent source/target RNG namespaces, shared target draws across
   angles, nested prefixes, and branch isolation verified against direct draws.
   Every CV pair is validated exactly once; source selection matches v2 on
   identical observations. Repeated calls cannot inherit another root seed.
4. At theta=0 the oracle source risk is 0.25; at every angle its signal loss
   is sin²(theta). Independent full error-map and trace metrics agree within
   1e-10 on constructed witnesses, including both nuisance stresses.
5. Incremental sums match direct batch sums on source+target concatenation.
   Local/global pooled fitted projectors and population risks match direct
   batch references within 1e-9 on nondegenerate fixtures. Tied cases follow
   the canonical convention and compare selected subspaces with appropriate
   declared tolerance. Direct-reference verification costs count as evaluator
   work; direct raw samples are test inputs, not deployed learner state.
6. Phase-reuse weights/projector match direct target-only fitting at p_hat;
   frozen weights never change with target input; target-only methods do not
   depend on source pairs. Pooled source history is never silently discarded.
7. Altering evaluator truth or source-retention evaluation cannot change
   selection, target updates or weights. Fitters accept observations,
   sufficient statistics and declared phase, not world truth. No source raw
   rows remain in a learner after its acquisition transition.
8. All defined arm/budget combinations are finite, including m=4; rank k,
   projector symmetry/idempotence and trace k hold to 1e-9. Undefined m=0
   arms are explicit nulls. Nonpositive regularized covariance, rank failure
   or silent fallback fails a gate.
9. Cost counters match closed-form fit/exposure/checkpoint counts on a small
   fixture, distinguish source/target unique pairs from repeated exposures,
   and charge source history once per branch. Account for raw source release.
10. Two complete development executions produce byte-identical scientific
    JSON. Timing is separate. Freeze source/test/contract/dependency hashes,
    environment, RNG layout, arms and decisions before any confirmation read.

## 9. Stopping and artifacts

Any correctness violation, solver/rank/nonfinite failure stops scoring and
records the world, angle, source/target budget and arm. Mark incomplete;
never drop or substitute a world. Resource-limited execution cannot satisfy
primary claims. No additional angle, update weight, method or source budget
may be introduced to rescue an outcome.

Planned artifacts:

- `src/representation_reuse.py`, `experiments/representation_reuse.py`,
  `tests/test_representation_reuse.py`;
- `reports/representation-reuse-v1-manifest.json`,
  `reports/representation-reuse-v1-development.json`,
  `reports/representation-reuse-v1-results.json`,
  `reports/representation-reuse-v1-runtime.json`;
- `docs/representation-reuse-v1-results.md`.

The results must distinguish source acquisition, stable reuse, target weight
adaptation, source forgetting, historical-statistic retention and computation.
Any benefit explained by extra history, supplied stable grouping or direct
covariance updates is attributed to that mechanism. A pass establishes these
declared transfer properties under this generator; it does not justify a
retained-context or reciprocal-dynamics claim.

## 10. Review record (before implementation)

Clarifications made in review, 2026-10-24: block-local construction of W;
how source acquisition exposes sums while reusing v1/v2 code; mandatory
incremental target sums for fresh-global; fresh arms executed independently
of s; in-distribution N for unqualified E; per-noise source forgetting; and
why v1 `make_world` is not called.

Expectation recorded before any implementation or draw: v2's aligned curves
(selection error concentrated at n<=16, fitting loss falling from about 0.29
at n=16 to 0.02 at n=256) make H1, H2 and H4 likely to pass by wide margins.
H3 is the informative primary comparison. A pass of H1/H2/H4 should be read
as confirming the expected value of paid source history and a correct source
phase, not as a surprising transfer result.
