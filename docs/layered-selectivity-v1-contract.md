# Layered selectivity v1 — proposed confirmation of an analytically designed disk

**Status:** draft proposal, 2026-10-24. Not frozen or run. Freeze the contract,
implementation, seeds, and environment before confirmation data are generated.

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
`default_rng([20261030,1,r,l,family])`. Share these across arms, depths where
the layer exists, and nuisance amplitudes. No relative-noise results enter
the success rule.

Primary decoding is unweighted nearest class mean on **all 32 coordinates**,
with means fitted only on certified noisy training drives and lowest-index
tie breaking. Use v1's strict and secondary accuracy conventions. No fitted
covariance, Fourier projection, or population means enter primary decoding.

This gives `16 * 20 * 400 = 128,000` canonical settling evaluations.
Also settle four pure-signal drives per family and configuration (128
evaluations; they can be reused across replicates). They provide the signal
power ratio only, not classifier means. No exploratory seed is reused.

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

If a required pure-signal settling fails certification or a denominator is
below `1e-12`, `J_n` is unavailable. Such a case cannot support selectivity.
The new criterion is **T4: `J_n<=0.5` in each family**. Report T1–T3 and T4
separately, preserving comparability with v1/v2.

**Primary success:** at least 18 of the 20 replicates of the single primary
configuration are eligible and meet all four criteria in both families.
This is a prespecified engineering reliability criterion, not a p-value or
a claim that the underlying success probability exceeds 90% with confidence.
Report the exact binomial 95% interval for the replicate success proportion,
all replicate values, and the count of failures by criterion.

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
without satisfying T4. Ordinary smoothing and the wrong-band control are
expected to lose F-high accuracy. No failure of a predicted control outcome
will be repaired by changing coefficients, noise, or thresholds.

## 6. Implementation and correctness gates

Implement the spatial energy and its analytic gradient/Hessian. Settle each
drive from zero with the v1 L-BFGS-B and Newton-polish conventions; apply the
same gradient/eigenvalue certification thresholds. Use an SPD direct solve
of the full spatial Hessian as an independent reference, not as an
undisclosed replacement for optimizer results. Report any certification
failures without changing solver settings after scoring.

Before confirmation, require:

- spatial Hessian symmetry and positive definiteness in every configuration;
- interior spatial couplings at circular distances at most four for the
  selective arms, and the declared radial/centre graph;
- analytic gradient and Hessian versus central finite differences at seeded
  states (step `1e-6`, relative tolerance `1e-5`, absolute tolerance `1e-5`);
- spatial direct-solve transfer and independent tridiagonal Fourier gains
  agreeing within `1e-9` absolute;
- every settled canonical state within `1e-7` infinity norm of the direct
  solution, and every pure target response matching the declared gains
  within `1e-9`;
- monotone inward mode gains and no gain above one, up to `1e-10` numerical
  tolerance; no exact finite-sample T1 equality comparisons are rounded
  before evaluation;
- NM implementation agreeing with v1 on fixed synthetic arrays;
- no use of test drives in classifier means or pure-signal scaling;
- pinned environment, one BLAS thread, recorded hashes, and byte-identical
  results regeneration in a separate directory.

Gate random states use `default_rng([20261030,2,arm_index,L])`, with arm
indices in section 2 order. Derivative checks should use standard-normal
states and drives, with maximum absolute and relative discrepancies recorded.
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
