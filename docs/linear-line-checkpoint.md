# Linear-line checkpoint

2026-10-26. Checkpoint for the continuous linear three-module line, written
before the move to
[relational abstraction learning](relational-abstraction-learning-proposal.md).
It summarizes; each claim's evidence and caveats are in the linked results.

## Setting

Three 8 → 4 → 2 linear modules in a role-free 0 ↔ 1 ↔ 2 chain. They are
coupled by learned translations at one or more layers, with a quadratic
energy, and trained by local clamped-energy learning with replay. Fixtures:
v1's two-structure data, and its correlated-nuisance extension, with a test
set on which cross-module correlations break. Errors are measured against
exact oracles (avoidable error).

## Established on these fixtures

1. **Local learning works, and the clamped-energy rule beats the two-phase
   (EP) rule** ([local contrastive v1](local-contrastive-consolidation-v1-results.md),
   [floor analysis](local-contrastive-consolidation-v1-floor-analysis.md)).
2. **Learned translations are decisive under uniform coupling**
   ([clamped-energy ablation](clamped-energy-ablation-v1-results.md)).
   Fixed identity coupling fails at every budget, and free translations
   contract nuisance directions.
3. **What a topology transmits is a property of the topology**
   ([correlated nuisance v1](correlated-nuisance-v1-results.md),
   [weak-prior topology](weak-prior-topology-v1-results.md)), at both priors
   tested:
   - surface coupling passes on non-persistent cross-module correlations;
   - a narrow inner channel filters them when they are weaker than the
     signal;
   - adding inner coupling to surface coupling does not gate the surface.
4. **Inner coupling's earlier deficit was a gain loss from the activity
   priors** ([inner-channel diagnostic](inner-channel-diagnostic-v1-results.md),
   [energy scaling analysis](energy-scaling-analysis.md)):
   - for the constrained arms, signal error is almost entirely scalar
     under-gain;
   - a scalar chain model predicts it;
   - lowering the priors at fixed parameters repairs it.
5. **Training with weak priors keeps the repair**
   ([prior-strength training](prior-strength-training-v1-results.md),
   [weak-prior topology](weak-prior-topology-v1-results.md)). The inner
   learner then:
   - filters;
   - recovers its signal;
   - beats every surface-coupled family on the decorrelated test (2.4-7.1×);
   - matches uncentered rank-2 regression, the best non-local reference.

   Rankings of inner against surface coupling made at prior 0.05 do not
   hold at prior 0.001.
6. **Clipping and exact training made no detectable difference; weight
   decay helps filtering**, under this protocol.

## Not established

- A learning inner channel in the strong regime, where nuisance outranks
  signal. The variance-ranking prediction (inner transmits) is only
  descriptive.
- Separate or learned layer priors, learned per-layer coupling, and gating.
- Anything about abstraction. Linear modules cannot hold structure their
  surfaces do not linearly contain. The filtering result is about a
  low-rank channel, not about deeper patterns.
- Generality beyond one fixture family, one chain topology and these
  budgets.

## Methods to carry forward

- Measure against exact or constructed oracles where possible, and declare
  criteria on avoidable error.
- Frame diagnostic criteria constructively: a positive result exhibits
  parameters or a change that does better. Otherwise report "no
  improvement found", and leave the cause unresolved.
- Declare regularization, priors and their strengths as design choices.
  They changed rankings here.
- Define retention on absolute error, not on a ratio to an
  already-low value.
- Keep learning criteria as reported outcomes, not gates on diagnostic
  comparisons.
- Report the measures that explain the error (signal gain, nuisance gain),
  not only the error.
- Compress large result files before their first commit, and keep no
  per-cycle movement files.

## Bearing on what comes next

The [relational proposal](relational-abstraction-learning-proposal.md)
starts from a small explicit tabular mechanism, not from these modules. What
transfers from this line is mainly method, not mechanism.

The [inner-coupling hypothesis](inner-coupling-hypothesis.md) remains open
for nonlinear modules and non-quadratic energies, planned separately. There,
prior strength and regularization should be declared up front.
