# Clamped-energy ablation v1: learned outer translations are decisive; weak coupling fails

Completed 2026-10-26 at revision 80ca54e, run once as contracted, with no
extensions. [Contract](clamped-energy-ablation-v1-contract.md) ·
[runner](../experiments/clamped_energy_ablation.py) ·
[scientific results](../reports/clamped-energy-ablation-v1-results.json) ·
[per-cycle movement, gzip](../reports/clamped-energy-ablation-v1-movement.json.gz) ·
[runtime](../reports/clamped-energy-ablation-v1-runtime.json)

## Outcome

The question was which change rescued module-local learning between
[local consolidation v1](local-consolidation-v1-results.md) and
[local contrastive v1](local-contrastive-consolidation-v1-results.md). With the
clamped-energy rule, Adam, update count and replay held fixed:

- **Learning the outer translation is the decisive ingredient.** With fixed
  identity translations, uniform coupling fails at every budget: avoidable
  error stays at 81-86% of frozen, at both 32-step and exact-equilibrium
  inference. Learned translations at the same coupling reduce it to about 1%
  of frozen. Inner-only coupling, which has no outer translation, learns only
  at the strongest budget, and even then has 5× the error of learned uniform
  coupling.
- **Rotation-only alignment gets most of the way.** Uniform coupling with
  orthogonal translations learns and beats ridge. Its avoidable error is 1.31×
  that of free translations, which falls in the unresolved band between
  "comparable" and "clearly worse", although free translations are better in
  10/10 seeds.
- **v1's weak budget is too weak.** At budget 0.4, no condition learns under
  this protocol, whatever the translation mode or topology. Part of this is
  slow 32-step settling, but error stays clearly worse at exact equilibrium
  too.
- **The gauge prediction holds.** In the inner family, rotation-only and fixed
  translations are indistinguishable (ratio 0.997). There were no W or V clip
  activations in that family, so the clipping caveat did not bind.

The reference condition uniform / L / 2, which is local contrastive v1's arm 2a
on fresh seeds, again gives the lowest error: avoidable MSE 0.00111, against
0.00112 on v1's seeds.

| Criterion | Result |
| --- | --- |
| A1 learning | 6/18 conditions learn: uniform O and L at b = 2 and 10; inner F, O and L at b = 10 |
| A2 learned translations, uniform | **decisive**: F/L ratio 97, L better 10/10 |
| A2 learned translations, inner | decisive on the declared 32-step metric (1.61, 10/10); see the qualification below |
| A3 isometric vs free, uniform | unresolved: O/L 1.31, L better 10/10 |
| A3 isometric vs free, inner | "non-isometric helps" on the declared metric (1.60); same qualification. O vs F comparable (0.997) |
| A4 weak coupling (b = 0.4 vs 2) | limits learning in 5 of 6 family × mode cells; the exception is uniform F, where every budget fails |
| A4 strong coupling (b = 10 vs 2) | limits only uniform O (1.80); unresolved for uniform L (1.31); b = 10 is clearly better in the inner family |
| A5 bridge to v1 | unresolved under the declared rule; inner / F / 0.4 does not learn, and neither do inner / L / 0.4 or inner / F / 2 |
| R1 against ridge | uniform L/2, L/10 and O/2 beat ridge clearly; O/10 comparable; all others clearly worse |

**Qualification for the inner family at b = 2.** The declared criteria use
32-step inference. In the inner family at b = 2, 32 steps leave the state far
from equilibrium: mean distance to exact is 0.19 for F and O, against 0.086 for
L. Free translations learn slightly larger inner singular values (mean 1.13),
which raises effective coupling and speeds settling. At exact equilibrium, F
and L are comparable, with F slightly better (ratio 0.86). So the inner-family
A2 and A3 results reflect faster settling under learned strength, not a better
equilibrium model. The uniform-family results hold under both inference modes.

## Main comparison

Mean over confirmatory seeds 13-22. X is avoidable MSE per coordinate against
the noiseless mean; observed RMSE includes the sample floor (mean 0.0892 on
these seeds). "Learns" is A1.

| Condition | X, 32-step | X, exact | Observed RMSE | X / own frozen | Learns |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform L 2 | **0.00111** | 0.00099 | 0.0948 | 0.009 | yes |
| uniform O 2 | 0.00145 | 0.00132 | 0.0966 | 0.011 | yes |
| uniform L 10 | 0.00145 | 0.00145 | 0.0965 | 0.010 | yes |
| uniform O 10 | 0.00263 | 0.00263 | 0.1024 | 0.017 | yes |
| ridge (all waking traces) | 0.00246 | | 0.1025 | | |
| inner L 10 | 0.00571 | 0.00430 | 0.1168 | 0.045 | yes |
| inner O 10 | 0.00573 | 0.00437 | 0.1169 | 0.045 | yes |
| inner F 10 | 0.00576 | 0.00440 | 0.1171 | 0.045 | yes |
| inner L 2 | 0.01302 | 0.00783 | 0.1448 | 0.102 | no |
| inner O 2 | 0.02087 | 0.00675 | 0.1697 | 0.164 | no |
| inner F 2 | 0.02094 | 0.00677 | 0.1699 | 0.165 | no |
| uniform L 0.4 | 0.02262 | 0.00681 | 0.1748 | 0.177 | no |
| uniform O 0.4 | 0.02370 | 0.00743 | 0.1778 | 0.185 | no |
| inner L 0.4 | 0.04989 | 0.02770 | 0.2403 | 0.391 | no |
| inner O 0.4 | 0.05069 | 0.03121 | 0.2411 | 0.398 | no |
| inner F 0.4 | 0.05199 | 0.03283 | 0.2437 | 0.408 | no |
| uniform F 2 | 0.10806 | 0.10804 | 0.3387 | 0.828 | no |
| uniform F 0.4 | 0.11013 | 0.10773 | 0.3424 | 0.861 | no |
| uniform F 10 | 0.12356 | 0.12356 | 0.3604 | 0.810 | no |

Frozen avoidable error is about 0.127 to 0.131, except uniform b = 10 at 0.152,
where the stronger identity outer coupling makes initial predictions worse.
Every condition improves B and retains A: B error falls in at least 8/10
seeds, and A's after-B/after-A ratio is at most 1.5 in 10/10 seeds
everywhere. The six learning conditions are those that also reach X <= 0.1×
their own frozen error.

## Reading

**Outer translations.** Fixed identity coupling at the outer layer is not
merely uninformative. It ties each module's hidden surface to a neighbour's
unrelated coordinates, and learning W and V cannot undo it, because x is
observed. That was the contract's stated hypothesis, and the uniform-F result
is consistent with it, at every budget and at exact equilibrium. Removing outer
coupling altogether (the inner family) avoids the harm but loses the most
useful channel: even at b = 10, inner-only coupling has 4.4× the exact-inference
error of uniform L/2 (10/10 seeds).

**Alignment versus non-isometric scaling.** In the uniform family, rotation-only
translations learn and come within 1.31× (exact: 1.34×) of free translations.
Free translations change outer singular values only slightly (mean largest
1.06 at b = 2), so the remaining gap is a small non-isometric effect. A3 does
not identify whether that effect is strength, anisotropy or nuisance
suppression.

**Coupling strength.** Weak coupling hurts twice over:

- 32-step settling has not converged at b = 0.4 (distance to exact about 0.2);
- even at exact equilibrium, b = 0.4 is clearly worse than b = 2: 3.5-6.9× in
  the learned-translation and inner cells, 10/10 seeds.

Stronger coupling helps the inner family (b = 10 better than b = 2, both
inference modes). In the uniform family it costs rotation-only translations
(1.80×, exact 1.98×) and is unresolved for free ones (1.31×). That is
consistent with free translations compensating for strength, but this study
does not test that.

**Bridge to v1 (A5).** The declared rule leaves A5 unresolved, because none of
its three named conditions learns. Outside the declared rule, inner / F / 10
learns (X 0.045× frozen). So identity inner-only coupling can support learning
under this protocol, at 25 times v1's budget. Together with A4, the data
suggest that v1's coupling budget was too weak for this protocol. That is not a
declared conclusion, and v1's other differences remain untested.

**Gauge.** Inner O and F agree to within 0.4% in mean avoidable error at b = 2
and b = 10, and within 2.5% at b = 0.4. W and V clipped zero times in the inner family, so the unconstrained
equivalence applied, and Adam's lack of rotation-equivariance made no
measurable difference here.

## Exploratory observations (not criteria)

- **Generalization gap.** Avoidable error on the replayed traces is close to
  test error in every condition. The observed-MSE gap is 0 to +0.0039,
  largest in the best uniform conditions. The objective question local
  contrastive v1 raised is not addressed by this rule-fixed design.
- **Learning curves.** Test avoidable error is flat or rises slightly over the
  last 100 cycles in most conditions, for example inner F 0.4 from 0.0500 to
  0.0520. More cycles at the selected η would not obviously help.
- **Clipping.** W and V clip only in the uniform family (up to 381 activations
  per run on average), never in the inner family.

## Pilot selection and limitations

The pilot used seeds 0-2, trained on even-index waking traces and scored
observed MSE on odd-index traces after stage B. All 18 × 4 η values were
retained and none was disqualified. **12 of 18 selected values lie at a grid
edge**:
- 0.003 (smallest) for all inner conditions except F/2 and O/2, which chose
  0.01, and for uniform F/10;
- 0.1 (largest) for uniform F/0.4, F/2, L/0.4 and O/10.

The grid was not extended, as declared. The inner conditions' preference for
the smallest η suggests their error may be lower still at smaller η. That would
not change the uniform-family conclusions, but it limits the inner-family and
A5 results.

The primary metric scores 32-step predictions, so in weakly coupled conditions
it mixes learned-model quality with settling speed. Exact-equilibrium values
are reported alongside, and the qualification above shows where they diverge.

## Verification and determinism

All checks passed before the pilot:
- uniform / L / 2 at η = 0.03 on seed 3 reproduces local contrastive v1 arm
  2a's final A, B and combined RMSE with zero difference;
- inner / F / 0.4 builds exactly v1's Q (difference 0);
- finite differences: activity 9.4e-9, parameters including T 8.7e-9;
- implicit descent in clamped and masked settles;
- the noiseless mean matches the generator's amplitude × pattern to 4.4e-16;
- F translations stay byte-identical, and O translations stay orthogonal to
  2.4e-15 in the short verification run and below 1e-10, asserted, after every
  projection in the full run;
- identical W, V initialization across conditions.

The full run was repeated to a scratch location from the same commit. The
scientific JSON and movement file were byte-identical. Before the runner was
committed, one pair of 2-cycle smoke runs differed somewhere in the results
file; the files were deleted before the difference was located. Six further
smoke runs and the full repeat were all byte-identical. Smoke outputs were
inspected only for checks and file structure; with 2 cycles they carry no
information about results.

## Implementation interpretations

These choices were made in the runner where the contract left room. They are
recorded here and were not tuned.
- Clip activations are counted after the Adam step and before orthogonal
  projection. For O, T is clipped to ±3 and then projected.
- Frozen baselines use T = I. Their "training" error uses the stage-A replay
  set.
- A1's "B error falls during stage B" compares test B avoidable error after
  stage A and after stage B.
- Ridge's avoidable error uses its stage-B fit on all waking traces.
- The pilot score is observed masked MSE, as in local contrastive v1. Avoidable
  error uses evaluator-only maps and is never used for selection.

## Computation

Whole run: 191.1 s wall, peak RSS 125 MB on one thread. This includes:
- pilot selection: 95.2 s, 216 trajectories;
- verification: 0.95 s, including a 600-cycle reproduction run;
- all evaluations.

Training CPU summed over 180 confirmatory trajectories is 81.7 s:
- inner F 0.36 s per trajectory;
- uniform F and L 0.48-0.50 s;
- uniform O 0.61 s, where the extra cost is 316,860 SVD projections over the
  whole run.

The repeat run ran concurrently on another core, so these timings may be
slightly inflated.

Executed counts:

| Count | Value |
| --- | ---: |
| Adam steps | 238,220 |
| Settles (each with one implicit factorization) | 273,631 |
| Activity example-steps | 159.7 M |
| Exact solves | 18,360 |
| Replay trace exposures | 4.16 M |

Storage per learner, float64:
- parameters: 960 B (F), 2,304 B (uniform O/L) or 1,024 B (inner O/L);
- Adam moments: twice the parameters, plus a step counter;
- replay: 6,144 B;
- episode IDs: 256 B.

Raw traces are retained.

Artifacts: the results JSON is 6.7 MB, committed uncompressed. The movement
file was compressed after the run with `gzip -n -9`, from 9.8 MB to 4.2 MB;
its uncompressed SHA-256 is
`75ec779141943606dbbbaf8b4b62344931062cc1dd46582c6f672077a7820dec`. Artifact
sizes keep growing, so the storage question deferred earlier is becoming
pressing. The runtime JSON's `dirty_status` lists the two output files, which
are written before the provenance step; the source tree was clean at 80ca54e.

## Next questions

These are not results:
- **Settling versus learning.** Weak coupling is partly a settling-speed
  problem. A follow-up should evaluate at exact equilibrium, or settle to a
  tolerance with step counts reported, before attributing differences to
  learned structure.
- **Learning-rate edges.** The inner family wants smaller η than the grid
  allowed.
- **The non-isometric gap.** Uniform O vs L (1.31×) is small but consistent
  across seeds. Constraining T to rotation × scalar would separate overall
  strength from anisotropy.
- **Harder fixtures.** In this linear fixture the best rule already sits within
  about 1% of frozen avoidable error. A correlated-nuisance or nonlinear
  fixture, as noted in [prior work](prior-work.md), is needed before further
  mechanism comparisons can discriminate.
