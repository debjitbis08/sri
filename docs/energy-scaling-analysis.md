# Energy scaling analysis: why the inner channel under-reproduces the signal

Post-hoc analysis, 2026-10-26. It is not part of any frozen contract, has no
criteria, and changes no reported outcome. It follows the
[inner-channel diagnostic](inner-channel-diagnostic-v1-results.md), where the
learner, masked completion and the constrained capacity fit stayed close,
while a fit without clips or weight decay did much better using large
weights. The question here is analytical: how do W, V and the translations
jointly change the effective activity penalties and coupling, and what
actually limits the signal?

[Script](../experiments/energy_scaling_analysis.py) ·
[results](../reports/energy-scaling-analysis-results.json) ·
[runtime](../reports/energy-scaling-analysis-runtime.json). Reproduce with
`.venv/bin/python -m experiments.energy_scaling_analysis` (under 1 s). It reads
the diagnostic's committed results.

## Summary

- **The signal loss is a gain loss.** For the learner, masked completion and
  the constrained fit, 97-100% of the signal error is a single scalar
  under-gain: completions reproduce the signal at 0.78-0.86 of its size.
- **A scalar chain model predicts that gain** from each module's top
  singular values of W, V and the inner translations, to within 0.006.
- **The fixed activity priors cause it.** With the priors removed, the
  chain's gain for the learner's parameters is 1.000. The middle-layer prior
  costs more than the inner one.
- **The learner does not compensate.** Its inner translations keep unit gain
  (τ = 1.00), so nothing in its chain amplifies to offset the shrinkage.
  Translations have no weight decay in this design, so decay cannot be the
  reason.
- **Scaling weights is not a neutral change.** An exact identity shows that
  larger weights are equivalent to weakening every deeper energy term,
  coupling included, relative to outer reconstruction. Under weight decay,
  the large-weight route the unconstrained fit took would cost 50-100× more
  than it gains.

## 1. Weight scaling reweights the energy

Inner-family energy, with λ = 0.05 (the Hessian of the 0.025 activity
priors):

    E = Σ_m [½‖x_m − W_m h_m‖² + ½‖h_m − V_m c_m‖² + (λ/2)(‖h_m‖² + ‖c_m‖²)]
      + Σ_a (κ/2)‖c_{a+1} − T_a c_a‖²

**Identity [derived; checked numerically to 2e-16].** Scale W by s and V by
t, and the activities by h → h/s, c → c/(st). The energy becomes

    ½‖x − W h‖² + (1/s²)·½‖h − V c‖² + (λ/s²)·½‖h‖² + (λ/(st)²)·½‖c‖²
      + (κ/(st)²)·½‖c_{a+1} − T c_a‖²

So a model with weights (sW, tV) is the original model with every deeper
term reweighted:

- **Relative to outer reconstruction**, all middle-layer terms weaken by
  1/s², and all inner-layer terms, coupling included, by 1/(st)².
- **Within each level the ratios are invariant.** At the middle layer,
  error : prior = 1 : λ. At the inner layer, prior : coupling = λ : κ.
- So larger weights do not pose the same inference problem: they make outer
  evidence dominate the deeper layers. The middle-layer shrinkage inside a
  hidden module, 1/(1 + λ), does not change under any such scaling. It can
  only be offset by gain elsewhere in the chain.

## 2. A hidden module's effective prior

For a hidden module, the free outer layer settles to x = W h, at no cost.
Minimizing its middle layer out leaves an effective prior on its inner code
[derived; checked against the model's Q to 6e-15]:

    λ I + (λ / (1 + λ)) VᵀV

Growing V therefore raises the hidden code's effective prior, which works
against the coupling that is supposed to set it.

## 3. Closed-form gain of a two-module chain

Take one observed module o and one hidden module h, one unit per layer
(scalar weights w, v and translation τ), with the signal along those units.
The completion gain x_h / x_o is [derived; checked against direct solves to
3e-13]:

    g = (w_h v_h τ / (w_o v_o))                      balance
        × 1 / (1 + λ)                                 hidden middle-layer prior
        × κ / (κ + P)                                 hidden coupling against its prior
        × v_o² / (v_o² + Q)                           observed inner level
        × w_o² / (w_o² + λ + Q / (v_o² + Q))           observed outer level

with P = λ + λ v_h² / (1 + λ), κ_eff = κP / (κ + P) and Q = λ + κ_eff τ².

- With no priors (λ = 0), g equals the balance term. With balanced weights,
  completion is exact.
- Every other factor is a prior competing with a precision, and each is
  below 1.
- The levers interact:
  - raising v shrinks Q / v² but raises P;
  - raising τ raises the balance but also Q;
  - raising w removes only the outer factor, the smallest loss.

  Offsetting the priors therefore needs balance above 1, that is,
  amplification along the chain.

At the learner's typical values (w = 2.5, v = 1.41, τ = 1):

| κ | Hidden middle prior | Hidden coupling vs prior | Observed inner | Observed outer | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (b = 2) | 0.952 | 0.874 | 0.919 | 0.979 | 0.748 |
| 5 (b = 10) | 0.952 | 0.972 | 0.913 | 0.978 | 0.826 |

The three-module chain, which the learner actually uses, has two observed
neighbours or a longer path, and gives 0.78 and 0.86.

## 4. The learned solutions

Weak regime, mean over seeds 33-42. "Measured" is the exact completion's
gain on the signal (decorrelated test). "Chain" is the three-module scalar
chain built from each module's top singular values. The independent regime
gives the same picture within 0.01 for the constrained arms.

| Arm, b | Measured | Chain | Scalar share of signal error | w | v | τ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E 2 | 0.781 | 0.783 | 0.98 | 2.50 | 1.41 | 1.00 |
| E 10 | 0.856 | 0.859 | 0.99 | 2.50 | 1.41 | 1.00 |
| M 2 | 0.789 | 0.793 | 0.97 | 1.66 | 1.28 | 1.36 |
| M 10 | 0.833 | 0.837 | 0.99 | 1.54 | 1.38 | 1.12 |
| S 2 | 0.791 | 0.791 | 0.97 | 1.64 | 1.28 | 1.34 |
| S 10 | 0.835 | 0.835 | 1.00 | 1.50 | 1.38 | 1.10 |
| S′ 2 | 0.957 | 0.962 | 0.20 | 24.97 | 2.72 | 3.28 |
| S′ 10 | 0.960 | 0.979 | 0.25 | 20.42 | 3.25 | 2.58 |

What the chain says would happen if one thing changed, at the learned
values:

| Arm, b | As learned | No middle prior | No inner prior | No priors | W × 10 | κ × 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E 2 | 0.783 | 0.917 | 0.848 | 1.000 | 0.796 | 0.881 |
| E 10 | 0.859 | 0.949 | 0.903 | 1.000 | 0.873 | 0.882 |
| M 10 | 0.837 | 0.939 | 0.890 | 1.001 | 0.876 | 0.855 |
| S 10 | 0.835 | 0.938 | 0.889 | 1.001 | 0.876 | 0.853 |

Readings:

- **For the constrained arms the signal error is almost entirely gain.** The
  chain matches the measured gain to within 0.006. Signal loss is shrinkage,
  not distortion or missing capacity in direction.
- **The priors are the source.** Without them, the learner's chain is
  exact. Removing the middle-layer prior alone recovers more than removing
  the inner one. Larger outer weights barely help.
- **The arms compensate differently but end up close.** The learner keeps
  τ at 1.00 with W at about 2.5. Masked completion and the constrained fit
  learn some amplification (τ 1.1-1.4), but their W is smaller (about 1.5),
  which costs gain elsewhere. That fits the diagnostic's finding that the
  three stayed close.
- **The unconstrained fit escapes through large weights and amplification**
  (w about 20-25, τ about 2.6-3.3). Its remaining error is mostly not scalar
  (share 0.20-0.25), and the chain is less accurate for it: the rank-1
  caricature misses its anisotropy.
- **Weight decay prices that route out.** At the unconstrained fit's weights,
  the decay penalty ½·10⁻³(‖W‖² + ‖V‖²) would be 1.48 (b = 2) and 0.96
  (b = 10). That compares with S's objective of 0.023 and 0.014, and its own
  decay of 0.013 and 0.012. This shows the large-weight route is not
  available at comparable cost under decay. It does not show that no
  low-weight solution with higher gain exists.

## 5. Why the learner keeps unit translation gain

[Derived, structural] The clamped-energy rule trains with every outer layer
clamped. No module is hidden at training time, so the hidden-module
shrinkage factors in section 3, 1/(1 + λ) and κ/(κ + P), never occur in its
training states. Its translations are fitted between codes that are all
inferred from data, at similar scales.

[Hypothesis] That is why τ settles at 1.00: the objective never sees the
gain lost at completion time. It is consistent with the measurement but not
tested. Masked completion and the capacity fit do see completion-time
states, and they do learn τ above 1.

## 6. Implications for the follow-up

- **Activity priors are the most direct lever.** In the chain model,
  shrinking λ raises gain directly, and at λ = 0 gain equals the balance.
  Growing weights reaches a similar end indirectly, at a large decay cost,
  and changes the inference problem (section 1). A prior-scale factor,
  for example λ ∈ {0.05, 0.005}, belongs in the follow-up beside the
  decay, clip and exact-training arms. The priors also keep the energy
  positive definite and regularize, so smaller λ may hurt filtering or
  conditioning. Nuisance gain must be measured alongside.
- **A sharp prediction for the decay arm.** Translations are not decayed, so
  decay cannot explain the learner's τ = 1.00. Removing decay alone should
  help masked completion and the capacity fit more than the learner, unless
  something drives the learner's balance above 1.
- **Exact training** remains an open, separate question (diagnostic D7).
- **Report signal gain** as a primary diagnostic beside signal error and
  nuisance gain: in this model it explains the constrained arms' signal
  error almost completely.

## Limitations

- The chain model is rank-1. It uses each module's top singular values and
  assumes the signal path aligns with them. That is accurate for the
  constrained arms and less so for the unconstrained fit.
- These are post-hoc readings of one study's fitted parameters, on one
  fixture.
- Section 5's causal claim is a hypothesis.

## Computation

Three identity checks on 20 to 50 random cases, then, for each of 16 arm
settings and 10 seeds, three exact masked solves on 24 test examples,
18 scalar-chain solves of at most 9 dimensions and one masked loss.
0.66 s wall on one thread. Hashes, revision and versions are in the runtime
JSON. The runtime `dirty_status` lists only the results file, which is
written before the provenance step.
