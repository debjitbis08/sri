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
- **In the scalar chain, the fixed activity priors cause it.** With the
  priors removed, the chain's gain for the learner's parameters is 1.000,
  and the middle-layer prior costs more than the inner one. That
  counterfactual holds for the scalar approximation; section 7 tests it in
  the full model.
- **The learner does not compensate.** Its inner translations keep unit gain
  (τ = 1.00), so nothing in its chain amplifies to offset the shrinkage.
  Translations have no direct decay penalty in this design. Decay still
  shapes W, V and the inferred codes, and through them the translation
  gradients, so it is not ruled out.
- **The full model confirms it at the learned parameters** (section 7).
  Lowering both priors to 0.005, without retraining, raises the learner's
  gain to 0.97-0.98 and cuts its decorrelated error 7-14×, while filtering
  holds (γ about 0.05).
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
- **In the scalar chain, the priors are the source.** Without them, the
  learner's chain is exact. Removing the middle-layer prior alone recovers
  more than removing the inner one. Larger outer weights barely help. This
  counterfactual is established for the rank-1 approximation only: matching
  the baseline gain closely does not guarantee matching responses to
  interventions, and top singular values omit alignment and direction.
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
- **A hypothesis for the decay arm.** Translations have no direct decay
  penalty, but decay acts on them indirectly through W, V and the codes.
  The hypothesis is that removing decay alone helps masked completion and
  the capacity fit more than the learner, unless something drives the
  learner's balance above 1.
- **Exact training** remains an open, separate question (diagnostic D7).
- **Report signal gain** as a primary diagnostic beside signal error and
  nuisance gain: in this model it explains the constrained arms' signal
  error almost completely.

## 7. Full-model test: lower the priors at the learned parameters

[Script](../experiments/prior_intervention_analysis.py) · results
(`reports/prior-intervention-analysis-results.json.gz`, gzip -n -9, 1.7 MB
uncompressed, SHA-256 2ff9d4fe…14d0ed) ·
[runtime](../reports/prior-intervention-analysis-runtime.json). Reproduce
with `.venv/bin/python -m experiments.prior_intervention_analysis` (about 7 s).

This tests the scalar chain's counterfactual in the full quadratic model.
Each arm's final learned parameters are held fixed; nothing is retrained.
Only the activity-prior Hessians in Q change: the middle-layer prior, the
inner-layer prior, or both, from the trained value 0.05 down to 0.001.
Values stay positive, so the systems stay non-singular. With the priors
unchanged, the script reproduces the diagnostic's recorded errors exactly
(difference 0).

Weak regime, mean over seeds 33-42; each cell is signal gain / X_dec / γ:

| Arm, b | Trained priors (0.05) | Middle 0.005 | Inner 0.005 | Both 0.005 | Both 0.001 |
| --- | --- | --- | --- | --- | --- |
| E 2 | 0.781 / 0.0064 / 0.042 | 0.899 / 0.0016 / 0.048 | 0.839 / 0.0036 / 0.045 | 0.971 / 0.00046 / 0.052 | 0.992 / 0.00037 / 0.053 |
| E 10 | 0.856 / 0.0029 / 0.043 | 0.937 / 0.0008 / 0.047 | 0.896 / 0.0017 / 0.045 | 0.982 / 0.00040 / 0.049 | 0.994 / 0.00037 / 0.050 |
| M 10 | 0.833 / 0.0038 / 0.042 | 0.923 / 0.0011 / 0.047 | 0.882 / 0.0021 / 0.045 | 0.980 / 0.00044 / 0.050 | 0.995 / 0.00040 / 0.050 |
| S 10 | 0.835 / 0.0035 / 0.006 | 0.927 / 0.0007 / 0.007 | 0.883 / 0.0018 / 0.007 | 0.982 / 0.00008 / 0.007 | 0.997 / 0.00004 / 0.008 |
| S′ 10 | 0.960 / 0.0010 / 0.129 | 1.051 / 0.0012 / 0.144 | 0.982 / 0.0008 / 0.132 | 1.075 / 0.0017 / 0.148 | 1.087 / 0.0019 / 0.150 |
| U 2 | 0.944 / 0.0018 / 0.62 | 0.976 / 0.0015 / 0.63 | 0.960 / 0.0016 / 0.62 | 0.993 / 0.0015 / 0.63 | 0.998 / 0.0015 / 0.63 |
| U 10 | 0.987 / 0.0037 / 1.05 | 0.994 / 0.0037 / 1.05 | 0.991 / 0.0037 / 1.05 | 0.998 / 0.0037 / 1.05 | 0.999 / 0.0037 / 1.05 |

M 2 and S 2 behave like M 10 and S 10 (both priors at 0.005: X_dec 0.00082
and 0.00046).

Readings:

- **Lowering the priors repairs completion at the existing parameters.** For
  the learner, with both priors at 0.005, gain rises to 0.97-0.98 and X_dec
  falls 7-14× (to 0.00040-0.00046), in 10/10 seeds at each budget. Filtering
  survives: γ rises only from about 0.04 to 0.05 (at most 0.076 in any
  seed). The learner's own parameters then beat uniform coupling on the
  decorrelated test by 4-9×, and come within about 1.2× of uncentered
  rank-2 regression (0.00037).
- **The scalar chain's counterfactual holds in the full model.** It predicted
  0.917 (no middle prior), 0.848 (no inner prior) and 1.000 (neither) for
  the learner at b = 2. The full model at 0.001 gives 0.911, 0.845 and 0.992.
  The middle-layer prior is the larger cost here too.
- **The constrained capacity fit improves most.** With both priors at 0.001,
  S reaches X_dec 0.00004 at b = 10. Its parameters, fitted under the
  trained priors, are nearly exact once the priors are lowered.
- **The unconstrained fit gets worse.** Its weights already compensate for
  the priors (balance above 1), so lowering the priors makes it overshoot
  (gain 1.05-1.09) and raises its error.
- **Uniform coupling barely changes.** Its decorrelated error is dominated by
  transmitted nuisance, which the priors do not touch.
- **The independent regime agrees.** For the learner, X falls from 0.0065 to
  0.00056 (b = 2) and from 0.0030 to 0.00050 (b = 10) with both priors at
  0.005.
- **Conditioning stays acceptable at these values.** The largest masked
  free-block condition number for the learner rises from 202 to 244 (b = 2).
  The smallest eigenvalue of Q falls with the priors, from 0.0096 to
  0.00098 at 0.005 and 0.0002 at 0.001. The unconstrained fit's systems were
  already ill-conditioned (condition number about 8×10⁵).

This answers the first of two questions: at the learned parameters, weaker
priors repair completion and keep the filtering, so the inference mechanism
is established in the full model. The second question is still open:
whether training with weaker priors learns parameters that keep both the
improvement and the filtering. Training would adapt W, V and T to the new
priors, and might, like the unconstrained fit, end up compensating in ways
that undo the gain. That is a learning experiment.

## Limitations

- The chain model is rank-1. It uses each module's top singular values and
  assumes the signal path aligns with them. That is accurate for the
  constrained arms and less so for the unconstrained fit.
- These are post-hoc readings of one study's fitted parameters, on one
  fixture.
- Section 5's causal claim is a hypothesis.
- Section 7 changes the priors at inference only. It shows what the learned
  parameters can do under weaker priors, not what training with weaker
  priors would learn.

## Computation

Three identity checks on 20 to 50 random cases, then, for each of 16 arm
settings and 10 seeds, three exact masked solves on 24 test examples,
18 scalar-chain solves of at most 9 dimensions and one masked loss.
0.66 s wall on one thread. Hashes, revision and versions are in the runtime
JSON. The runtime `dirty_status` lists only the results file, which is
written before the provenance step.
