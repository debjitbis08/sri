# Layered selectivity v1 — proposed confirmation of an analytically designed disk

**Status:** prospective; frozen for implementation on 2026-10-24. The manifest
records the hashes of this contract, the implementation, the seeds and the
environment. No section 3 draw had been generated when it was frozen.

**Review revisions (2026-10-24, before freezing and before any confirmation
draw).** The draft was reviewed against an implementation and its
pre-confirmation gates. Pure-signal settlings and one smoke replicate with a
non-declared seed (`1`) were used. Thresholds, factors, seeds, the
primary configuration and the success rule are unchanged.

1. **Derivative-gate tolerance (section 6).** The draft's elementwise
   tolerance (`1e-5` absolute plus `1e-5` relative) fails for a correct
   implementation, from rounding alone. For standard-normal states the
   energy reaches about `1.3e5` (selective, `L=4`) and `4.6e6` (wrong
   bands). A central difference with step `1e-6` then carries absolute
   error up to about `1e-3` on small gradient entries. The gate now scales
   by the largest analytic entry. It adds an exact check: the term-by-term
   energy must equal the quadratic form of the assembled Hessian.
2. **Control predictions (section 5).** The draft said ordinary smoothing
   loses F-high accuracy. That holds at `L=4` (population 0.3775), not at
   `L=2` (0.9955). The no-lateral control at `L=2` has population
   suppression 0.3008 and is expected to fail T2. Section 5 now tabulates
   population values for all 16 configurations.
3. **Wrong-band stiffness (section 2).** The same coefficient rule gives
   `tau=1973.30` and a largest lateral stiffness of `3.61e5`. That is 34
   times the selective arm. The two arms are matched on the smallest
   unwanted-mode stiffness (one), not the largest. At `L=4` its F-high
   interior gain is `1.4e-13`. `Vsig(L)` is then below the `1e-12` floor and
   `J_n` is unavailable.
4. **Unavailable `J_n` (section 4)** counts as T4 not met, with its reason
   recorded. The floor applies to `P(1)`, `Vsig(1)` and `Vsig(L)`.
5. **Index conventions (sections 3 and 6).** Layer `l` is 1-based. `family` is
   0 for F-low and 1 for F-high. `arm_index` is 0-based in section 2 order.
   Gate states are drawn before gate drives.
6. **Interval and outcome labels (section 4).** The binomial interval is
   two-sided Clopper–Pearson. The primary outcome is one of three declared
   labels.
7. **Execution.** Units of (configuration, replicate) run in parallel
   processes, with one BLAS thread each. Results do not depend on the number
   of workers. Every gate that needs no section 3 draw runs first, including
   the pure-signal settlings. Confirmation draws are generated only if all of
   those gates pass.
8. **Smaller clarifications.** Primary decoding calls v1's `nearest_mean`.
   The centre is reported separately and enters no criterion. The pure-signal
   gate uses each arm's own tridiagonal gains.

This follows [the analytical design study](layered-module-design-analysis.md).
That study includes disclosed exploratory pilots. This is a fresh experiment,
not an extension or retuning of layered-module v2.

## 1. Question and primary claim

Can a stable reciprocal disk with a finite-range lateral penalty meet the
original T1–T3 profile for both known signal families under fixed readout
noise, while improving nuisance relative to signal rather than merely
shrinking both?

Success would establish a constructed, task-informed selective filter.
It would not establish learning, abstraction, temporal retention, or readiness
for Stage 2. No Stage 2 configuration is selected by this experiment.

## 2. Fixed architecture and controls

Every configuration has `L` rings of 32 units and one centre. Radial edges
are `j <-> j`, with unit strength, and all last-ring units connect to the
centre with unit strength. There is a unit-strength exterior tie, no exterior
lateral penalty, and no scalar local potential. Energy is

```
E = 1/2 ||u_1-h||² + 1/2 sum_{l=2}^L ||u_l-u_{l-1}||²
  + 1/2 sum_{l=2}^L u_l^T K u_l + 1/2 sum_j (u_{L,j}-c)².
```

Let `C` be the 32-cycle Laplacian and `e_k=2(1-cos(2 pi k/32))`.
Four arms differ only in `K`:

1. **Selective:** `K=tau B^T B`, `B=(C-e_1 I)(C-e_8 I)`,
   `tau=1/min_{k not in {1,8,24,31}} [(e_k-e_1)(e_k-e_8)]²`.
2. **No lateral penalty:** `K=0`. Control for attenuation without selectivity.
3. **Ordinary smoothing:** `K=C`. Control for the original type of lateral
   operator, matched on exterior, radial, and centre rules. This is not an
   exact v2 replication because its exterior has no lateral smoothing.
4. **Wrong bands:** substitute modes 1 and 4 into arm 1 and recompute `tau`
   by the same smallest-positive-eigenvalue rule, excluding `{1,4,28,31}`.
   This controls for knowing which high-frequency distinction matters.

The coefficient rule gives every unwanted mode an interior stiffness of at
least one in both band arms. It does not match their largest stiffness:

| Arm | `tau` | Largest lateral stiffness | Hessian condition, `L=2` / `L=4` |
| --- | --- | --- | --- |
| Selective | 169.2824 | 10,626.9 | 10,802 / 27,831 |
| No lateral penalty | — | 0 | 88.5 / 277.6 |
| Ordinary smoothing | — | 4 | 88.5 / 277.6 |
| Wrong bands | 1973.3008 | 361,001.9 | 366,870 / 945,124 |

The wrong-band control therefore differs from the selective arm in
stiffness as well as in its nullspace. Its F-high loss comes from mode 8
receiving stiffness of at least one. It is not attributed to stiffness.
The centre is reported with the same measures as v1 and enters no
criterion.

Depths are `{2,4}` and nuisance amplitudes are `{0.5,1}`: 16 configurations.
**Primary configuration:** selective arm, `L=4`, `alpha=1`. Depth 2 and
`alpha=0.5` are declared secondary comparisons, not fallback selections.
There is no parameter search, nonlinear arm, or relative-noise primary.

## 3. Fresh paired data and decoding

Use the v1 signal classes and orthonormal real Fourier nuisance basis.
For replicate `r=0,...,19`, draw standard normal Fourier coefficients from
`default_rng([20261030,0,r])`, shape `(2,4,50,32)` in C order, and rescale each
drive's nuisance to RMS `alpha/sqrt(2)`. Families are ordered F-low then
F-high. Use the same underlying draws across all arms, depths, and amplitudes.
Draws 0–24 train and 25–49 test, separately per class and family.

Readout noise is always absolute `sigma=0.1`. For each replicate, family,
and layer `l`, draw `(4,50,32)` standard normal entries from
`default_rng([20261030,1,r,l,family])`. Here `l` is 1-based and `family` is 0
for F-low and 1 for F-high. Share these across arms, depths where the layer
exists, and nuisance amplitudes. No relative-noise results enter the success
rule.

Primary decoding is unweighted nearest class mean on **all 32 coordinates**,
with means fitted only on certified noisy training drives and lowest-index
tie breaking. It calls v1's `nearest_mean` function unchanged. Use v1's
strict and secondary accuracy conventions. No fitted covariance, Fourier
projection, or population means enter primary decoding.

This gives `16 * 20 * 400 = 128,000` canonical settling evaluations.
Also settle four pure-signal drives per family and configuration (128
evaluations; they can be reused across replicates). They provide the signal
power ratio only, not classifier means. No exploratory seed is reused.

Units of (configuration, replicate) may run in parallel processes, each with
one BLAS thread. Each settling is deterministic in its own process, and
results are assembled in the declared order, so they do not depend on the
worker count. Pure-signal settlings and every gate in section 6 that needs no
section 3 draw run first. Confirmation draws are generated only if those
gates pass.

## 4. Measures and acceptance rule

Report v1 `S(l)`, `P(l)`, suppression ratio, and strict/secondary NM accuracy.
The original profile is unchanged: each family separately must satisfy
T1 `S(1)>=S(l)` for every layer, T2 `P(L)/P(1)<=0.25`, and T3 interior strict
accuracy at least 0.9. As in v1, every canonical settling in both families
must certify for that configuration/replicate to be eligible.

For each family also compute pure-signal between-class power

```
Vsig(l) = mean_{c,j} (u_l(hsig_c)_j - mean_d u_l(hsig_d)_j)²
J_n = [P(L)/P(1)] / [Vsig(L)/Vsig(1)].
```

`J_n` is unavailable in either of two cases. One is that a required
pure-signal settling fails certification. The other is that `P(1)`,
`Vsig(1)` or `Vsig(L)` is below `1e-12`. The reason is recorded. Such a case
cannot support selectivity and counts as T4 not met. The new criterion is
**T4: `J_n<=0.5` in each family**. Report T1–T3 and T4 separately, preserving
comparability with v1/v2.

**Primary success:** at least 18 of the 20 replicates of the single primary
configuration are eligible and meet all four criteria in both families.
This is a prespecified engineering reliability criterion, not a p-value or
a claim that the underlying success probability exceeds 90% with confidence.
Report the two-sided Clopper–Pearson 95% interval for the replicate success
proportion, all replicate values, and the count of failures by criterion.
The primary outcome takes exactly one of three labels:

- `success`: the primary success rule holds;
- `profile_pass_without_demonstrated_selectivity`: the rule fails, but at
  least 18 replicates are eligible and meet T1–T3 in both families;
- `not_met`: otherwise.

The same counts are reported for every other configuration. They are
secondary comparisons and carry no label.

Report matched control differences in accuracy and `J_n`, with no selection
among replicates. To describe an observed mechanism, require that the primary
selective arm meets its success rule; describe control outcomes as measured,
without declaring that a control failed when it merely misses T4. In
particular, the no-lateral arm is expected to pass much of the old profile.

The following are diagnostic only: full Fourier gain spectra, theoretical
population `P` and `S`, Gaussian accuracy, the spherical known-mean lower
bound, Hessian conditioning, optimizer iterations, and runtime. Report
input-relative nuisance and signal powers as well as exterior-relative
ratios so that exterior filtering is visible.

## 5. Predictions declared before confirmation

For the primary selective arm, population values are:

- signal gains `(0.8,0.6,0.4,0.2)` for both families;
- `P(L)/P(1)=0.0165802463`;
- `J_1=J_8=0.2652839410`;
- Gaussian-reference accuracy `0.9989094623` at `sigma=0.1`, `alpha=1`;
- true-distribution population-NM accuracy at least `0.9903441`.

The last two describe known means, not a guarantee for fitted NM. The
disclosed pilot suggests that fitted NM should also pass comfortably.

The no-lateral control has the same signal gains, but population nuisance
ratio `0.1074856046` and `J_n=1.7197696737`. It can pass the old profile
without satisfying T4. No failure of a predicted control outcome
will be repaired by changing coefficients, noise, or thresholds.

Population values for every configuration follow. They are computed from the
tridiagonal gains, with no draws. `g(L)/g(1)` is the target-mode signal gain
ratio. Accuracy is the Gaussian reference at `sigma=0.1`, for `alpha=0.5` and
`alpha=1`.

| Arm | `L` | `P(L)/P(1)` | F-low `g(L)/g(1)` | F-low `J_1` | F-low accuracy | F-high `g(L)/g(1)` | F-high `J_8` | F-high accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Selective | 2 | 0.0620 | 0.5 | 0.248 | 1.0000 / 0.9998 | 0.5 | 0.248 | 1.0000 / 0.9998 |
| Selective | 4 | 0.0166 | 0.25 | 0.265 | 1.0000 / 0.9989 | 0.25 | 0.265 | 1.0000 / 0.9989 |
| No lateral | 2 | 0.3008 | 0.5 | 1.203 | 1.0000 / 0.9998 | 0.5 | 1.203 | 1.0000 / 0.9998 |
| No lateral | 4 | 0.1075 | 0.25 | 1.720 | 1.0000 / 0.9989 | 0.25 | 1.720 | 1.0000 / 0.9989 |
| Ordinary smoothing | 2 | 0.1779 | 0.491 | 0.739 | 1.0000 / 0.9998 | 0.25 | 2.847 | 0.9997 / 0.9955 |
| Ordinary smoothing | 4 | 0.0890 | 0.228 | 1.719 | 1.0000 / 0.9982 | 0.0179 | 279.2 | 0.3778 / 0.3775 |
| Wrong bands | 2 | 0.0628 | 0.5 | 0.251 | 1.0000 / 0.9998 | 6.6e-5 | 1.4e7 | 0.2504 / 0.2504 |
| Wrong bands | 4 | 0.0165 | 0.25 | 0.264 | 1.0000 / 0.9989 | 2.9e-13 | unavailable | 0.2500 / 0.2500 |

Hence, in population terms:

- **No lateral penalty** passes T1–T3 at `L=4` but fails T4. At `L=2` it
  also fails T2.
- **Ordinary smoothing** at `L=2` keeps F-high accuracy and passes T1–T3
  but fails T4 in both families. At `L=4` it also loses F-high accuracy.
- **Wrong bands** behaves like the selective arm for F-low, including T4,
  and loses F-high accuracy at both depths. At `L=4` its F-high `Vsig(L)`
  is about `1e-26`, so `J_8` is unavailable.

These are population expectations. Finite-sample outcomes are reported as
measured.

## 6. Implementation and correctness gates

Implement the spatial energy and its analytic gradient/Hessian. Settle each
drive from zero with the v1 L-BFGS-B and Newton-polish conventions; apply the
same gradient/eigenvalue certification thresholds. Use an SPD direct solve
of the full spatial Hessian as an independent reference, not as an
undisclosed replacement for optimizer results. Report any certification
failures without changing solver settings after scoring.

Before confirmation, require:

- spatial Hessian exact symmetry and positive definiteness in every
  configuration;
- the largest circular distance of an interior lateral coupling is four for
  the two band arms, one for ordinary smoothing, and zero without a lateral
  penalty; the declared radial/centre graph holds (ring blocks `-I` between
  consecutive rings only, no exterior lateral coupling, centre joined to
  ring `L` only);
- analytic gradient and Hessian versus central finite differences at seeded
  states (step `1e-6`). The largest absolute discrepancy must be at most
  `1e-5 max(1, max |analytic entry|)`, separately for the gradient and the
  Hessian. Also, the term-by-term energy must equal
  `1/2 u^T H u - h^T u_1 + 1/2 ||h||²` within `1e-12 max(1,|E|)`.
- spatial direct-solve transfer and independent tridiagonal Fourier gains
  agreeing within `1e-9` absolute, and the centre equal to the ring-`L`
  mean to the same tolerance;
- every settled canonical state within `1e-7` infinity norm of the direct
  solution; every pure-signal response equal, on each ring, to that arm's
  tridiagonal gain times the drive, within `1e-9`;
- monotone inward mode gains and no gain above one, up to `1e-10` numerical
  tolerance; no exact finite-sample T1 equality comparisons are rounded
  before evaluation;
- v1's `nearest_mean` agreeing exactly with an independent vectorized
  implementation on fixed synthetic arrays, including an exact tie
  (`default_rng([20261030,3])`);
- no use of test drives in classifier means or pure-signal scaling;
- pinned environment, one BLAS thread, recorded hashes, and byte-identical
  results regeneration in a separate directory.

Gate random states use `default_rng([20261030,2,arm_index,L])`, with 0-based
arm indices in section 2 order. Derivative checks use a standard-normal
state over all units, including the centre, drawn before a standard-normal
drive. Maximum absolute and scaled discrepancies are recorded.
Freeze code and contract hashes before generating section 3 draws. Runtime
metadata lives in a separate file. A failed numerical gate is reported as
an implementation/numerical failure; it does not authorize parameter tuning.

## 7. Interpretation and stopping rule

Run the declared confirmation once, report every configuration/replicate,
and stop. The analysis and pilot remain visibly exploratory and separate
from confirmation. No extension of depth, noise, bands, or coefficients
after inspecting results is part of this experiment.

If T1–T3 pass but T4 fails, report a profile pass without demonstrated
selectivity. If all pass, the result supports the existence of a stable
finite-range, task-informed filter that preserves both declared bands while
rejecting irrelevant variation. It does not establish that relevance can be
discovered without supervision or that the construction generalizes to
unseen bands. The next analytical question would be coefficient tolerance,
less-stiff implementations, or learning the relevant subspace—not an
automatic move to coupled modules.
