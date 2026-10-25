# Single-module abstraction preflight v1: revise the interface and the split

**Completion:** complete, 2026-10-25. The analytical preflight, finite task
audit, comparator fits and all declared numerical checks finished. All 45
tests passed (10 new and 35 unchanged plasticity-v1 tests). Two complete
scientific executions were byte-identical. No circular-module training,
hyperparameter selection, parameter search or confirmation experiment ran.

**Decision:** do not launch the proposed abstraction learning experiment
unchanged. Zero-filled completion has a proven output limitation. The explicit
observation mask gives a well-defined equilibrium, but masked routed capacity
remains unresolved. Independently, the declared held-out split can be solved
exactly by memorization plus a sign symmetry already built into the module.

[Contract](single-module-abstraction-preflight-v1-contract.md) ·
[manifest](../reports/single-module-abstraction-preflight-v1-manifest.json) ·
[scientific results](../reports/single-module-abstraction-preflight-v1-results.json) ·
[runtime](../reports/single-module-abstraction-preflight-v1-runtime.json) ·
[shortcut receipt](../reports/single-module-abstraction-preflight-v1-shortcut-receipt.json) ·
[shortcut runtime](../reports/single-module-abstraction-preflight-v1-shortcut-runtime.json)

## 1. Findings and their limits

| Question | Finding |
| --- | --- |
| Does visible evidence determine the answer? | Yes, within the declared generator: two adjacent orientations identify the angular step and the missing parts. |
| Does the current zero-filled interface admit the 0.10 completion target? | No. Every scene has normalized missing-part error at least 0.25, for any allowed parameters and realized routes. |
| Is the explicit-mask equilibrium unique? | Yes, for positive lateral coupling, positive radial routes, full-column-rank maps and at least one observed outer coordinate. |
| Can any fixed affine map meet the target on all scenes? | No. Its optimal all-scene mean error is (8−2√2)/21 = 0.246265. Fixed-route module inference is included in this class. |
| Can the masked module with input-dependent routes meet the target? | Unresolved. No random fixture is a witness; that is not a proof of inability. |
| Does this holdout exclude memorization? | No. All eight held-out scenes are sign reversals of training scenes. Signed template lookup completes them exactly. |
| Is abstraction or a learned inner relation demonstrated? | No learner was trained, and the split would not isolate that interpretation. |

Identifiability here is conditional on the specified generative rule family.
It does not mean finite training observations uniquely select that family
among every possible extension of the training data.

## 2. Proof of the zero-filled output limitation

For any fixed second-pass route a, separate the structural Hessian K_a from
the outer anchor alpha=0.1. The structural energy is a sum of nonnegative
weighted squared residuals, so K_a is positive semidefinite. Its interior
block K_II is positive definite: fixing x_0=0 and setting the energy to zero
forces D_0 x_1=0, then x_1=0 by full column rank, and successively x_2=x_3=0.

Eliminating the interior yields

```
S_a = K_00 - K_0I K_II^(-1) K_I0 >= 0
x_0 = F_a h
F_a = alpha (alpha I + S_a)^(-1)
```

F_a is symmetric, with every eigenvalue between zero and one. Let P_H and
P_V project onto the disjoint missing and visible coordinates. Since
P_H I P_V=0,

```
||P_H F_a P_V||_2
  = ||P_H (F_a - I/2) P_V||_2
  <= 1/2.
```

The clean visible and missing signals each have norm sqrt(2). Feeding zero
in the missing coordinates therefore limits the missing output to norm
sqrt(2)/2. The reverse triangle inequality gives

```
||prediction_H - target_H||^2 / 2 >= 1/4.
```

This argument is pointwise: input-dependent routes select a different F_a,
but each selected matrix satisfies the same inequality. It does not assume
the complete routed predictor is linear, or claim that it is globally
nonexpansive between different inputs. It also does not claim that the bound
is attainable by the actual sector-supported architecture.

Output rescaling, a decoder, or changing the observation term would change
the tested model. Direction-only accuracy would conceal this amplitude
limitation. No training sweep was used to challenge a proven obstruction.

## 3. What an observation mask fixes, and what it leaves open

Replacing the data term by alpha/2 sum_i m_i(x_0,i−h_i)^2 gives

```
x_0 = (alpha M_obs + S_a)^(-1) alpha M_obs h,
M_obs = diag(m).
```

This inverse exists under the stated conditions. If the total quadratic form
is zero, positive cycle-Laplacian penalties force each ring to be constant.
Any observed outer coordinate forces the outer constant to zero. Positive
radial penalties and full-column-rank maps then force all inner states to
zero. Thus the full Hessian is positive definite; its outer Schur complement
is positive definite too.

The resulting input/output matrix generally is not symmetric and is not the
previous alpha(alpha I+S_a)^(-1) contraction. The universal off-block 1/2
bound therefore does not carry over. This removes that particular proof of
failure; it does not prove useful completion is representable.

The masked fixture Hessians were more poorly conditioned: maximum condition
number about 2,534 versus about 151 with the original interface. All solves
still met the prescribed residual tolerances. Positive lateral coupling is
part of the uniqueness proof; cutting links during a future causal test must
be checked afresh rather than repaired silently.

Learnable radial maps still act within sectors, and cross-sector lateral
couplings remain fixed. The existence of a graph path does not establish
that these parameters can express the desired conditional relation. The
remaining masked, input-dependent-route capacity question is **unresolved**.

## 4. Exact affine capacity calculation and comparator audit

Represent each two-coordinate orientation as the complex unit u. The three
angular steps are z in {exp(i*pi/4), i, exp(3i*pi/4)}. Visible parts are
(u,zu), and missing parts are (z^2 u,z^3 u).

Averaging over all eight rotations of u makes intercept and conjugate-frequency
components orthogonal to the target. The best real affine predictor therefore
reduces to approximating each z^k by a_k+b_k z on the three z values. Two
distinct rules can be interpolated by this affine expression, which explains
why merely choosing two relations would not demand nonlinear inference.

For three rules define

```
w = (z_2-z_3, z_3-z_1, z_1-z_2).
sum(w) = sum(w*z) = 0
||w||^2 = 6 - 2 sqrt(2).
```

The residual mean for z^2 is
`|sum(w*z^2)|^2 / (3 ||w||^2) = (10−6√2)/21`.
For z^3 the numerator is multiplied by `|z_1+z_2+z_3|^2 = 3+2√2`,
giving `(6+2√2)/21`. Averaging the two missing-part errors gives
`(8−2√2)/21 = 0.246265375012`. An error <=0.10 on every scene would imply
a mean <=0.10, so no affine map meets the capacity target. This applies to
any fixed-route module, including with an explicit observation mask; it does
not apply to the nonlinear matched-route predictor.

The finite SVD calculation agrees with this expression within 1e-10. All
five renderings give the same comparator scores to reported precision, as
expected under independent orthogonal changes of coordinates within parts.

The table uses macro-rule held-out loss; accuracy is the fraction of the
eight held-out scenes with both missing orientations correct. Losses also
retain per-scene, per-rule and worst-case values in the scientific JSON.

| Predictor | Held-out loss | Worst held-out loss | Both orientations correct |
| --- | --- | --- | --- |
| Training-mean constant | 1.0190 | 1.0840 | 0/8 |
| First-visible-part affine | 0.9894 | 1.2554 | 0/8 |
| Two-visible-part affine | 0.2980 | 0.5841 | 0/8 |
| Nearest training input | 2.3928 | 2.7071 | 0/8 |
| Rule oracle, evaluator-only | 0 | 0 | 8/8 |
| All-scene affine capacity oracle, fitted to held-out targets too | 0.2463 | 0.4659 | 3/8 |

The last row is a privileged capacity calculation, not a valid held-out
learned baseline. Zero output has loss exactly one on this clean fixture.
The trained two-part affine predictor gets the first missing orientation
right on all eight held-out scenes while getting the second right on none;
scoring both prevents that partial success from being called completion.

## 5. The split has a complete sign-reversal shortcut

The declared split withholds a scene when `t mod 3 == r−1`. For every such
scene, the same r with orientation `(t+4) mod 8` is in training:

| Relation r | Held-out t | Training antipode t |
| --- | --- | --- |
| 1 | 0, 3, 6 | 4, 7, 2 |
| 2 | 1, 4, 7 | 5, 0, 3 |
| 3 | 2, 5 | 6, 1 |

Adding four rotates every orientation by pi, so both visible inputs and
hidden targets are negated. All 12 classes of scenes identified up to a
global sign already occur in training. Consequently, a lookup table that
stores each training pair (h,y), and returns −y when queried with −h,
solves all eight held-out examples. It does not need relation labels or a
learned modular-completion rule.

This shortcut matches an existing symmetry of the candidate itself. For a
fixed mask and parameters, the pilot state is linear in h. Every residual
ratio defining g uses squared norms, so g(−h)=g(h). The same second-pass
Hessian is then used with a negated right-hand side, giving f(−h)=−f(h).
No new symmetry needs to be learned to extend memorized training responses
to these held-out scenes.

More strongly, every held-out scene's squared completion error equals the
error of its training antipode for any odd predictor. This split introduces
no new clean functional constraint on this module. The saved module errors
indeed agree for all eight pairs in all 64 fixture/interface/policy records
at the reported precision.

A separate, explicitly post-run deterministic receipt verifies the signed
lookup using only training inputs and targets for selection, across all five
stored renderings. Maximum hidden loss was 2.48e-32, numerical zero. It uses
no new observations, fitting, tuning or parameter search. It supplements the
analytical split audit and does not replace the frozen comparator table.

The failure of ordinary nearest-neighbor lookup therefore does not exclude
memorization under the candidate's built-in symmetry. A future split must
hold out whole sign classes, and must audit other supplied transformation
symmetries too. That is a necessary safeguard, not a sufficient abstraction
criterion by itself. This study retains the original split and records the
limitation rather than selecting an easier or harder replacement afterward.

## 6. Numerical fixtures and correctness

There were 16 independently generated constrained parameter fixtures, four
in each rendered world. Every fixture was evaluated on all 24 scenes under
both interfaces and both fixed and matched routing. No fixture was optimized.

| Interface | Route | Mean all-scene loss over fixtures | Best fixture's worst-scene loss |
| --- | --- | --- | --- |
| Original zero-filled | Fixed | 1.000081 | 1.000242 |
| Original zero-filled | Matched | 1.000080 | 1.000045 |
| Explicit mask | Fixed | 1.001985 | 1.002504 |
| Explicit mask | Matched | 1.001951 | 1.000055 |

None is a capacity witness. Errors near the zero-output control describe
these untrained fixtures only; they cannot prove that training or other
allowed parameters would fail under the masked routed interface.

Across each complete execution's 2,304 checked equilibrium passes:

- maximum relative solve residual: 6.67e-16;
- maximum full/Schur state disagreement: 1.40e-14;
- maximum full/Schur map disagreement: 1.10e-14;
- maximum independent energy-expansion discrepancy: 5.19e-16;
- no failed positive definiteness, constraints, gate bounds or contraction
  inequalities, and no repaired matrices, restarts or omitted fixtures.

Tests additionally check the original implementation agreement, energy
polarization, explicit tie handling, masked-input independence, odd symmetry,
single-observed-coordinate positivity, the split and the two-rule affine
control. The complete 45-test transcript is in the runtime JSON.

## 7. Provenance, computation and reproduction

Contract commit `c920897` preceded implementation. Implementation and manifest
commit `e6ec5bb` preceded tests and finite execution. Frozen plasticity-v1
dependencies matched their manifests. The shortcut-check source was committed
as `4cb6e0b` before its separate execution, after the main results were known;
its hashes and that disclosure are recorded in its receipt/runtime.

The two full executions produced byte-identical scientific JSON. Timing is
kept separate. The mechanism-diagnostics contract was present but untracked
during execution and was not executed; exact dirty paths are recorded.

On the recorded Ryzen 5 3600 CPU, NumPy 2.5.4, SciPy 1.18.1, float64, one
process and one BLAS thread:

| Phase | Wall seconds | CPU seconds | Peak RSS KiB, process scope |
| --- | --- | --- | --- |
| Freeze command, including imports | 0.58 | 0.57 | 98,332 parent |
| Test subprocess, including imports | 5.42 | 5.42 | 158,260 child |
| Full finite execution 1, including serialization | 2.88 | 2.88 | see run-process maximum |
| Full finite execution 2, including serialization | 2.78 | 2.78 | see run-process maximum |
| Complete run command, containing both executions | 6.29 | 6.24 | 116,268 parent |
| Supplementary sign-shortcut receipt | 0.59 | 0.59 | 104,308 parent |

The tests command's total wall time was 6.04 s including its parent overhead;
its parent CPU time was 0.58 s in addition to child CPU. These are nested
scopes, not additive rows. They are not inference latency or whole-job peak
memory measurements. The supplementary receipt performed 40 held-out queries
against 32 signed training candidates, totaling 1,280 squared distances, with
no new scenes or fits; its timing is in a separate runtime JSON.

Per complete finite execution: 24 abstract scenes, 120 rendered scenes,
15 SVD affine fits (10 legitimate training fits and 5 all-scene capacity
fits), 5 constant means, 5 lookup tables, 720 comparator-scene scores,
1,536 module-scene scores, 768 route evaluations and 2,304 equilibrium checks.
Each check includes one full Cholesky, independent interior and outer Schur
LU factorizations, state and map solves, spectral diagnostics and two direct
energy expansions. The full counts are in runtime JSON; both executions
are charged. Test fixtures are additional correctness work, measured by
their subprocess time and reported separately from sweep operation counts.
There were zero training updates and zero tuning executions.

The contract derives the construction, fitting, solve and storage orders.
The implementation uses dense O(M^3) factorizations and eigenspectra, with
O(M^2) matrix workspace; it makes no sparse-solver or scaling claim. Stored
module parameters occupy 2,624 bytes plus metadata, the dense curvature
cache 819,200 bytes, and one 80x80 matrix 51,200 bytes. The full result file
is 2,114,693 bytes of harness output. Native peak solver workspace and FLOPs
are unavailable with reasons; neither is inferred from parameter counts.

To reproduce the archived computation, use an isolated checkout of `4cb6e0b`
with the same pinned interpreter. Keep that revision's frozen manifest and
initial freeze-runtime file, then run `python -m experiments.abstraction_preflight
--phase tests` and `python -m experiments.abstraction_preflight --phase run`.
Run `python -m experiments.abstraction_preflight_shortcut` afterward for the
supplementary receipt. Do not re-freeze the existing manifest. The runners
refuse changed/uncommitted frozen inputs or overwriting already recorded
phases/artifacts, protecting the archived outputs in the working checkout.

## 8. Recommended next decision

First revise the relational split so unseen examples are not supplied by
the candidate's known sign symmetry, and audit any other trivial template
extensions. Preserve relation-changing counterexamples and amplitude-sensitive
completion error. Do not treat a probe or classification-only success as a
substitute for the required predictive transfer.

For the model, use an explicit observation mask in a new prospective design
if missing-part completion remains the task. Fixed-route inference is ruled
out at the present capacity target. Before training a large routed learner,
require either a constructive capacity witness or a separately contracted,
bounded capacity search. Such a search may use evaluator knowledge only as
an explicitly privileged feasibility check; its parameters must not initialize
the later learner. Failure of that bounded search would remain inconclusive.

The plasticity and forgetting diagnostics can proceed under their own contract.
They address update allocation and retention and cannot resolve these task
and representational issues by themselves.
