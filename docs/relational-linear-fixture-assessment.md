# Relational linear fixture: obligations and positioning

2026-10-27. Analytical assessment, not a contract or experiment. No new runs.
It places the linear fixture of the
[relational proposal](relational-abstraction-learning-proposal.md) and its
[preflight](relational-linear-preflight-v1-results.md) against the program's
unresolved obligations and against established methods.

## Two separate questions

1. **Program progress:** which unresolved obligation of the
   [research program](research-program.md) does a step discharge, and what
   existing work lets it be discharged cheaply?
2. **Originality:** does the resulting mechanism merit a claim beyond prior
   work?

Neither answer settles the other. A step can advance the program using known
methods, provided its result is labelled as establishing a capability within
SRI, not as a new general result. The
[originality review](relational-originality-review.md) asked only the
second question before any further work. The reviewer has since said that is
too restrictive for an intermediate milestone.

**Status:** on this fixture, no original advantage has been established.
That does not show that none can survive, and it does not establish either
closing the line or continuing it unchanged.

## What the fixture tests and what it supplies

The program's relational aims are to discover, preserve and reuse
relational structure across experiences. The relevant obligations are
P4–P6 in the [research program](research-program.md): structural
preservation, endogenous generation, and abstraction with transfer.

### Supplied in advance

The proposal and contract disclose all of these:

- Action identity is consistent across domains.
- The action interface is a bank of per-action inner translations. This is
  externally selected relation structure, not autonomous relation
  selection.
- The experimenter chooses which parameters are shared (the banks) and which
  are domain-specific (the bindings), and when to freeze them.
- The generator is linear-Gaussian, and the target abstraction is a joint
  similarity class of linear operators.
- The three modules are assigned to window positions x_t, x_{t+1} and
  x_{t+2}.
- The preflight's capacity searches use privileged targets.

### Tested by the preflight

These are feasibility results, not learning results:

- **Representability:** the chain represents the dynamics at the learner's
  clips.
- **Binding refits:** with frozen banks, refitted bindings keep the
  effective operators within the invariance threshold. Post-hoc spectra show
  drift that could matter at L = 128. How much of the measured error that
  drift causes has not been isolated.
- **Objective mismatch:** observed-data fits suffer from it.
- **Settling cost:** prediction-time settling needs hundreds of steps at
  prior 0.001.

### Untested

- **Discovery (P6):** learning the banks from observed source experience
  with the local rule (proposal experiment 2).
- **Reuse (P6):** whether source structure reduces the target evidence
  needed, against target-only and transfer comparators (proposal
  experiment 3).
- **Preservation (P4):** whether source responses are retained after target
  adaptation or fine-tuning. Nothing here measures it.
- **Endogenous generation (P5):** no-history dreaming. The proposal defers
  it.
- **Discovering what is shared:** learning which structure is shared,
  rather than being told, and learning action correspondence. The proposal
  lists both as later conditions.

## Established methods to borrow or use as controls

| Subproblem | Established method | What it offers | Caveats for SRI's constraints |
| --- | --- | --- | --- |
| Long-horizon objective | Simulation-error training (Venkatraman et al.); tPC-RTRL ([Potter and Rhodes 2026](https://arxiv.org/abs/2602.18131)) [read, main text] | A multi-step objective. A local-in-space-and-time rule that recovers the full-sequence (BPTT) gradient under stated assumptions | That exactness is for a single-layer, element-wise recurrence. The multilayer extension is layer-local and approximate (§5.5). The storage figures cover credit assignment only (§5.2), and hardware advantages are argued, not measured (§5.4). SRI uses coupled modules in an untied window, and the mapping onto this method would need specifying. |
| Teacher forcing | tPC's inferred-state propagation (Potter and Rhodes, Appendix F) | Shows that conditioning training on corrected states removes the pressure to learn open-loop dynamics | Consistent with the preflight's objective mismatch; not tested on SRI's window. |
| Replay-free learning | Recursive least squares; online EM ([Cappé and Moulines 2009](https://arxiv.org/abs/0712.4273)) [abstract] | RLS equals the batch solution exactly for fully observed linear regression. Online EM converges asymptotically to stationary points for latent models with independent observations | RLS fits the equation-error regression, which attenuates under observation noise. With latent states, the expected statistics depend on the current parameters. Online EM is not exactly the batch estimate, and its state-space versions were not reviewed. Its locality is not SRI's. |
| Transfer with frozen dynamics | Shared latent dynamics with a refitted observation map ([Sun et al. 2022](https://arxiv.org/abs/2201.00248), Theorem 8 under its assumptions) [read]; frozen-transition latent fits; a bias-corrected (IV or TLS) latent reference | Comparators and a transfer reference that removes errors-in-variables attenuation | Maximum likelihood is asymptotically efficient under correct specification, not best at every finite budget. At budgets of 4–32 transitions, the ranking of estimators is an empirical question. |
| Missing observations | Kalman filtering and smoothing; tPC test-time inference (Potter and Rhodes §4.5) | Conditioning on any observation mask | With known parameters, SRI's exact completion is the conditional mean of its own Gaussian. In use, its quality depends on parameters estimated from finite data, and so does the comparator's. |
| Shared structure | Multi-task linear representation learning (Du et al.; Tripuraneni et al.); joint similarity tests ([prior work](prior-work.md#linear-action-conditioned-dynamics-and-transfer)) | Estimators, and the preflight's mismatch certificates | These target single capabilities, not the combination with local rules and no replay store. |
| Limits of local rules | RFLO and truncated BPTT in linear RNNs ([Williams et al. 2026](https://proceedings.mlr.press/v306/williams26a.html)) [read, main text] | RFLO's weight changes have rank at most the output dimension, and it converges more slowly | Whether SRI's clamped-energy rule has a similar restriction is an open diagnostic question. |

## What the literature settles and what it leaves open

- **Settled:** each single capability above has an established method on
  this kind of fixture. A result here establishes that capability within
  the SRI architecture. It should not be presented as a new general result.
- **Open:** whether SRI's combination achieves these capabilities under its
  own constraints. Those are local rules, coupled role-free modules, no
  replay store, and CPU-bounded cost. Comparing single capabilities with
  separate methods does not show the combination is redundant. Finite-budget
  rankings against maximum-likelihood references are empirical.

## Options for the next step

These are for decision, not frozen.

1. **Discharge discovery, reuse and preservation with borrowed methods.**
   - Learn the banks from observed source experience with the local rule.
     Add a declared multi-step objective borrowed from simulation-error or
     tPC-RTRL-style credit, and disclose its locality.
   - Then refit the bindings on the target.
   - Measure:
     - how much less target evidence is needed (P6);
     - source retention after adaptation and after fine-tuning (P4).
   - Controls:
     - target-only arms;
     - the frozen-dynamics latent reference and a bias-corrected variant;
     - multi-step predictive training;
     - replay.
   - Save the fitted bindings and run the operator-substitution ablation
     from the preflight results.
   - Label the result as a capability established within SRI.
2. **Stop supplying what is shared.** Keep the linear fixture, but make the
   learner find which structure is shared across domains, or the action
   correspondence. This tests discovery more directly. It needs a new
   identifiability audit.

Option 1 is closer to the proposal as written. Option 2 removes more of
the structure the fixture currently supplies. Either needs its own contract
with a prior-work section.
