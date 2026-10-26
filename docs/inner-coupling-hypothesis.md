# Research direction: inner coupling and shared structure

Research-direction note, 2026-10-26. It records a project hypothesis and how
it could be tested. It is not a contract and has no results.

## Hypothesis

In the [clamped-energy ablation](clamped-energy-ablation-v1-results.md),
uniform coupling (all three layers) beat inner-only coupling by roughly 4× in
avoidable error. The project hypothesis is that this reflects the fixture,
not a general property:

- outer (surface) coupling does well only because, in this data, one module's
  surface predicts another's through a single fixed linear map;
- on complex data, where modules share deeper structure but their surfaces
  differ or carry misleading correlations, inner-layer coupling becomes
  important, because it lets modules share patterns at a level their surfaces
  do not expose.

This is hard to establish with small experiments. The aim here is to find
small fixtures whose outcomes would count for or against it.

## Why current evidence does not test it

- **The fixture favours surface coupling by construction.** The best
  predictor of a hidden module is a direct linear map from the observed
  surfaces, which a learned outer translation represents directly. Nothing
  deeper is shared.
- **Inner-only results are partly pessimistic.** 32-step settling had not
  converged at weak coupling, and most inner conditions selected the smallest
  learning rate on the grid.
- **The layers were not isolated.** Uniform and inner families differ in both
  outer and middle coupling.
- **The modules are linear.** A stack of linear maps is one linear map, so
  the inner layer is a 2-dimensional linear projection of the surface, not a
  more abstract feature. In linear modules, inner coupling can act as a
  low-rank channel between modules, which may filter what is shared, but it
  cannot share structure the surface does not linearly contain. A full test
  of the hypothesis needs nonlinear modules.

## Candidate fixtures and what would count against the hypothesis

| # | Fixture | Supports the hypothesis | Counts against it |
| --- | --- | --- | --- |
| 1 | Shared low-dimensional latent seen through private nonlinear maps, with nonlinear modules | Inner coupling recovers the shared latent and predicts hidden modules better than outer coupling | Outer coupling matches inner, or only learned outer translations help |
| 2 | Correlated cross-module nuisance that does not persist (linear modules suffice) | Inner coupling transmits less nuisance and is more robust when the correlation breaks, at an acceptable in-distribution cost | Uniform or outer coupling is as robust, or the in-distribution cost outweighs the robustness gain |
| 3 | Grow surface dimension, shrink sample count | Outer coupling, with d² translation parameters per edge, degrades before inner coupling | Outer coupling keeps its advantage as d grows and n shrinks |
| 4 | Shared and private latent factors | Inner coupling carries shared factors without leaking private ones | Outer coupling separates them equally well |

Order:

1. Correlated nuisance (fixture 2): no architecture change. Contract:
   [correlated nuisance v1](correlated-nuisance-v1-contract.md).
   [Results](correlated-nuisance-v1-results.md): inner coupling filters the
   correlation but loses too much signal to win overall. Adding inner
   coupling to surface coupling does not gate what the surface passes.
   Learned or gated per-layer coupling is the natural next step before
   nonlinear modules.
   [Inner-channel diagnostic v1](inner-channel-diagnostic-v1-results.md):
   a rank-2 channel can keep the filtering and the signal (uncentered
   rank-2 regression does). So can this energy architecture: the
   successful solutions found used large weights outside the learner's
   bounds, found without clips or decay. Within the learner's constraints
   no improvement was found, which does not show none exists.
   [Energy scaling analysis](energy-scaling-analysis.md): the constrained
   arms' signal loss is a scalar gain loss caused by the fixed activity
   priors, which the clamped-energy learner does not compensate for.
   [Prior-strength training v1](prior-strength-training-v1-results.md):
   trained with weaker activity priors, the inner learner keeps its
   filtering, recovers the signal and beats uniform coupling under
   decorrelation (1.9-5.3×). In linear modules, the mechanism the
   hypothesis relies on now gives a measured advantage on this fixture.
   Earlier topology comparisons were all made at prior 0.05 and need
   re-reading.
2. Scaling sweep (fixture 3): cheap, and gives a trend relevant to scale.
3. Nonlinear modules (fixture 1): the real test, but the learning rule,
   settling and Gaussian-based floor analysis all change.

In linear modules, a positive result on fixture 2 shows only that a low-rank
inner channel filters non-persistent surface correlations. That is a known
property of rank constraints, and it is evidence for the hypothesis's
mechanism, not for abstraction.

## Prior work to review

Each contract gets its own prior-work section. Shared findings go in
[prior work](prior-work.md). Relevant areas: early, intermediate and late
fusion in multimodal learning; reduced-rank regression and information
bottlenecks; spurious correlations and invariance under distribution shift;
alignment versus cross-prediction under correlated nuisance (Kamai et al.
2026); shared and private latents in multi-view models.
