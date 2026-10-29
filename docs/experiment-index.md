# Experiment index

Running list of experiments, with contracts, results and reproduction commands.
Shared prior-work survey: `prior-work.md`; contract review practice: `literature-review.md`.
Research directions: `inner-coupling-hypothesis.md` (whether inner coupling matters on complex data).
Image study: [number concept network v1 contract](number-concept-network-v1-contract.md)
(frozen 2026-10-28; draft history in the [specification](number-concept-network-v1-spec.md)),
digit/dice succession and one-anchor transfer.
Implementation: `src/number_concept.py`, `experiments/number_concept_network.py`
(`pilot` | `confirmation`; confirmation refuses to run before the frozen contract
exists), tests in `tests/test_number_concept.py`. MNIST goes in gitignored
`data/mnist/` (checksums in code); checkpoints in gitignored `runs/`.
[Pilot results](number-concept-network-v1-pilot-results.md): pipeline and stability
checks pass; not headline evidence.
[Confirmation results](number-concept-network-v1-results.md): digit gates pass for
both variants; one-anchor transfer fails for both (A mean 0.39 rotation, 0.40 MLP,
seed-bimodal). Dice succession is learned internally, but the anchor does not
propagate along an approximately recurrent orbit. Results hold for the v1 jittered
dice generator. Reproduce with the confirmation command (about 98 minutes on 11 workers).
[Alignment analysis](number-concept-network-v1-alignment-analysis.md) (post hoc, 38 s):
correlated latent shape in untied frames. For rotation the offset is the predicted
eight-angle phase symmetry, which the single-image anchor fails to fix
(class-level phase correction 0.83). `.venv/bin/python experiments/number_concept_alignment_analysis.py`.
[Cross-relation pilot](number-cross-relation-pilot.md) (exploratory, no contract):
shared encoder, joint training, true/false *same* and *next* judgments, with
cross-modal equivalents for a few counts only. With pairs {1, 3, 5}, a judge that
reads both relations off one code space with a shared successor aligns unpaired
dice 2, 4 and 6 exactly in 3/3 seeds. Without succession, or with a free judge,
the same pairs give 0.00 and about 0.3 respectively. Pairs {1, 2, 3} extrapolate to
4–6 at 0.78. `.venv/bin/python experiments/number_cross_relation_pilot.py` (about 33 minutes on 11 workers).
[Number core network v1](number-core-network-v1-spec.md) (draft, three pilots;
[results](number-core-network-v1-pilot-results.md)): image (MNIST, dice) and Morse
modules coupled to a shared SRI core and a fixed-code label module, with exact
equilibria. Labels on unlabelled counts stay at 0.00 under sparse labels; only the
equivalence-loss arm at wide size aligns them in the core. The input kind is decodable
from the core at 0.999. A tally test labelled at 1, 2 and 5 puts every other count in
those bins. `src/number_core.py`, tests in `tests/test_number_core.py`;
`.venv/bin/python experiments/number_core_network.py pilot|pilot2|tally` (6–8 minutes each).
[Number systems v1](number-systems-v1-spec.md) (draft, pilot;
[results](number-systems-v1-pilot-results.md)): 18 number representations after
Lockhart's *Arithmetic* (`src/number_systems.py`, tests in `tests/test_number_systems.py`),
one satellite per medium, complementary sparse labels, compatible versus private
relation orders, held-out tree, german, thai_digits and hand.
- Compatible order aligns numbers labelled in neither representation in the
  cross-representation probe (0.26 against 0.15), including Morse ↔ images.
- Labels and held-out transfer stay at chance.
- Post-hoc diagnostics (fixed versus refitted probes, an interface copy) show
  adaptation mostly moves the sources' coordinates through the shared interface.
- Stage A (reuse cost): source experience raises few-shot accuracy, but private orders
  benefit equally, and a shared interface halves source retention.

Commands: `.venv/bin/python experiments/number_systems.py pilot` (25 minutes),
`-m experiments.number_systems_pilot_analysis pilot` (12 minutes), and
`-m experiments.number_systems_reuse_cost pilot` (67 minutes on 11 workers). Results
gzip-compressed where large.
[Number systems meta v1](number-systems-meta-v1-spec.md) (draft, pilot;
[results](number-systems-meta-v1-pilot-results.md)): first-order MAML episodes with a
preservation term inside source training, against ordinary training at an equal
encoding budget, with three development representations.
- Meta adapts somewhat faster than both ordinary arms, including the data-matched one.
- It does not protect the sources (Morse loses two thirds of its accuracy when Hand
  adapts).
- Its gain does not depend on the shared order.

Development-only diagnostics:
- **Gradient diagnostic:** the first-order gradient points almost unrelated to the
  gradient through adaptation.
- **Episode audit:** every inner procedure that learns also forgets.
- **Gradient audit** through a working inner procedure (Adam 1e-3, 25 updates): neither
  direction makes new learning cheaper in single steps.

Commands, each `NSY_WORKERS=6 .venv/bin/python -m experiments.<name> pilot`:
- `number_systems_meta` (26 minutes);
- `number_systems_meta_gradient` (15 minutes);
- `number_systems_meta_episode` (7 minutes);
- `number_systems_meta_audit` (27 minutes).
[Number construction v1](number-construction-v1-spec.md) (draft): counting episodes
grounded in collections (Part A) and confidence-controlled relational revision (Part B),
with collection matching as the answer.
- [B1 results](number-construction-v1-b1-results.md), on frozen pilot checkpoints: with a
  compatible order, revision through the learned successor lifts collection matching from
  0.21–0.32 to 0.62–0.72. Private orders and a perturbed successor remove the gain.
- A swapped-item control shows the answer comes from the grounded context through the
  successor. That is genuine relational inference, but the item's own evidence is not
  combined with it, and nothing is learned from it. Gating has nothing to select.
- B1-ambiguous: with one grounded neighbour and its side unknown, the item decides, but
  relational hypothesis selection (side right 0.62–0.67) does no better than restricted
  recognition (0.59–0.67).

Commands, each under a minute:
- `NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_b1 pilot`;
- `NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_b1_ambiguous pilot`.

[Part A results](number-construction-v1-part-a-results.md), pilots 1–4, three seeds for
pilots 3–4: counting episodes over persistent dot collections, languages in blocks.
- Forms never directly paired with collections acquire quantity meaning through counting
  transitions alone (0.31–0.46 against 0.40–0.54 for paired forms, chance 0.11).
- Episode order gives no advantage over shuffled evidence.
- Persistent state between steps never trained stably.
- Pseudo-label fast mapping from the core's inferences adds nothing.
- Agreement between instinct and relational inference predicts correctness (0.56–0.62
  against 0.06–0.10).
- Blocked teaching forgets earlier languages.

Command: `NCA_SEED=<seed> NSY_WORKERS=5 .venv/bin/python -m experiments.number_construction_part_a pilot|pilot2|pilot3|pilot4`
(6–35 minutes).

Confidence analysis: agreement can be anticipated by a cheap relational consistency check
(AUROC 0.66–0.96) but not by recognition-only signals, and gating on it does not beat
random gating. `NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_confidence pilot3`
(under a minute).

[Part D results](number-construction-v1-part-d-results.md), three seeds: senses, varied
contexts, rehearsal and an imagery probe.
- Rehearsal removes most forgetting (Chinese at the end 0.20 → 0.53).
- At matched exposure, varied contexts (taps through a second sense, plus comparison)
  beat repeated ones on acquisition and retention by 0.08–0.17.
- Untaught compositions work: taps → form 0.76–0.91 (withheld forms included), and
  "which is more" learned on collections transfers to symbols (0.77–0.92).
- The learned successor flags incoherent counting runs, without separate anomaly
  training, wherever the concept is retained (0.94–0.97).
- Imagery: clamped core states, including the state inferred between two neighbours,
  project back toward count-specific sensory features in both senses (nearest real
  inputs). Decoded "dreams" are blurred and not shown to preserve count.

Commands: `NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_part_d pilot`
(25 minutes), then `NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_imagery pilot`
(13 minutes).
New experiments are added here, not to the README. Entries carried over from the
README's original layout list keep their wording. Earlier studies that list never
covered (for example the layered-module, locality-adaptation and relational
analyses) are not yet indexed here; see the files in this directory.

- `mediation-example.md`: worked collective role-binding example, its controls and limits; reproduce with `python -m experiments.mediation`.
- `comparison-v1-results.md`: four-family modeling comparison, evidence grades and limitations; reproduce with `python -m experiments.model_comparison`.
- `core-mediation-v1-contract.md`: declared contract for discrete core-mediated compatibility, realizability and certified truncation.
- `core-mediation-v1-results.md`: phase B realizability results (pairwise star versus ternary versus agreement); reproduce with `.venv/bin/python -m experiments.core_mediation` (about 18 minutes).
- `core-mediation-v1-phase-c-results.md`: structured internals, recovery, input/output; robust codes make readout-only coupling inert. Reproduce with `.venv/bin/python -m experiments.core_structured` after phase B.
- `core-graded-v1-contract.md`, `core-graded-v1-results.md`: graded unit-level coupling; one worked example where recovery and input-driven core participation coexist. Reproduce with `.venv/bin/python -m experiments.core_graded`.
- `core-graded-basins-contract.md`, `core-graded-basins-results.md`: exact basins, tie and terminal propositions, and the joint-success window; reproduce with `.venv/bin/python -m experiments.core_graded_basins`.
- `learned-compatibility-v1-contract.md`, `learned-compatibility-v1-results.md`: learned core compatibility, generalization to unseen combinations, set-valued responses; reproduce with `.venv/bin/python -m experiments.learned_compatibility`.
- `selective-satellite-mediation-v1-contract.md`, `selective-satellite-mediation-v1-results.md`: learned context-gated repair under immutable domains; exact version space, transfer to an unseen context, matched controls (the positive-only ablation and selection prior are reported limits). Reproduce with `.venv/bin/python -m experiments.selective_mediation`.
- `learned-reciprocal-participation-v1-contract.md`, `learned-reciprocal-participation-v1-results.md`: signed-energy input/core/satellite dynamics; phase-A gate fails (input revision never changes responses for the declared teachers), so no learning was run. Reproduce with `.venv/bin/python -m experiments.reciprocal_participation`.
- `shared-input-mediation-v1-contract.md`, `shared-input-mediation-v1-results.md`: causal chain A → input → B through a shared or internally coupled input; 2 of 6 declared teachers (S1, C2) pass the full gate, mechanism only, no learning. Reproduce with `.venv/bin/python -m experiments.shared_input_mediation`.
- `learned-input-mediation-v1-contract.md`, `learned-input-mediation-v1-results.md`: learning couplings from outcomes and checking the learned mechanism on a held-out intervention battery; exact learners identify S1/C2 mechanisms; for a negative control, outputs alone leave the mechanism uncertain (false-chain survivors beside the correct ones). Reproduce with `.venv/bin/python -m experiments.mechanism_learning`.
- `local-learning-supervision-v1-contract.md`, `local-learning-supervision-v1-results.md`: fixed local rule under four supervision conditions; clamps with external readout or 12 extra mixed-input queries suffice for exact identification, but the local learner reaches the mechanism reliably only with internal readouts. Reproduce with `.venv/bin/python -m experiments.local_supervision`.
- `local-update-diagnosis-v1-contract.md`, `local-update-diagnosis-v1-results.md`: why the block learner fails under IB supervision (mixed causes: inexact single commits, preservation conflicts, restricted edits, plateau barriers) and three single-property variants; none suffices for both teachers, and history-aware selection (a replay comparator) suffices for C2 only. Reproduce with `.venv/bin/python -m experiments.update_diagnosis`.
- `core-abstraction-v1-contract.md`, `core-abstraction-v1-results.md`: no qualifying core for S1/C2 under isolated-response preservation after assembly; isolated acquisition also leaves core identity unobserved. Reproduce with `.venv/bin/python -m experiments.core_abstraction`.
- `core-abstraction-v2-contract.md`, `core-abstraction-v2-results.md`: with preservation judged in context and 15 connected axis observations, an exact constraint selector picks the teacher's private blocks and core for S1/C2/T1 in every run with full mixed transfer; donor-dependent selection holds for S1 in 4/10 runs and C2 in 0/10 (each C2 satellite's own evidence pins the core), so the minimal target fails. Reproduce with `.venv/bin/python -m experiments.core_abstraction_v2`.
- `shared-core-abstraction-v3-contract.md`, `shared-core-abstraction-v3-results.md`: complementary shared-core evidence. Over all subsets of the axis training pool, matched or unmatched, no teacher yields two informative but individually insufficient satellite views: each teacher's reachable core masks form a chain, so intersection never adds resolution. Complete exact negative. Reproduce with `.venv/bin/python -m experiments.shared_core_abstraction_v3` (about 25 s).
- `graded-cellular-v1-results.md`: nonlinear graded-cellular follow-up, recovery results and explicit scope discount; reproduce with `python -m experiments.graded_cellular`.
- `three-module-ambiguity-v1-contract.md`, `three-module-ambiguity-v1-results.md`: three 8→4→2 modules reconstructing outputs from distributed partial evidence; input ↔ intermediate → output is the strongest tested configuration, full reciprocity misses the finite-budget thresholds. Reproduce with `.venv/bin/python -m experiments.three_module_ambiguity`.
- `graded-three-module-v1-contract.md`, `graded-three-module-v1-results.md`: frozen sensitivity study of layer-graded coupling and output-feedback gain with supplied translations; uniform per-unit coupling and low output feedback give the lowest final error. Reproduce with `.venv/bin/python -m experiments.graded_three_module`.
- `local-consolidation-v1-contract.md`, `local-consolidation-v1-results.md`: module-local proposal/check/support consolidation with stability gate and inertia; second-structure reconstruction does not improve, the gate adds nothing, discrimination is already trivial. Reproduce with `.venv/bin/python -m experiments.local_consolidation`.
- `local-contrastive-consolidation-v1-contract.md`, `local-contrastive-consolidation-v1-results.md`: learnable graded inter-module translations in three 8→4→2 modules; local learning acquires and (with replay) retains both structures; the two-phase rule reaches nearly the exact-gradient reference's final error, but a clamped-energy control learns better and cheaper and is the working baseline; importance inertia is not shown to beat uniform slowing. Reproduce with `.venv/bin/python -m experiments.local_contrastive_consolidation` (about 4 minutes); the movement artifact is stored gzip-compressed. Post-hoc `local-contrastive-consolidation-v1-floor-analysis.md` measures avoidable error against the noiseless conditional mean; the floor is about 0.086 RMSE, and two-phase has 2.1× clamped energy's avoidable error. Reproduce with `.venv/bin/python -m experiments.local_contrastive_floor_analysis` (under 1 s).
- `clamped-energy-ablation-v1-contract.md`, `clamped-energy-ablation-v1-results.md`: fixed, rotation-only and free inter-module translations × coupling budget × topology, under the clamped-energy rule; learning translations is decisive under uniform coupling (fixed identity fails at every budget; the design varies all coupled layers together, so the outer layer is not isolated), rotation-only alignment gets within 1.31× of free translations, whose spectra contract nuisance directions, v1's 0.4 budget is too weak for any condition to learn, and inner-family fixed and rotation-only translations match as the gauge argument predicts. Criteria use avoidable error against the noiseless mean. Reproduce with `.venv/bin/python -m experiments.clamped_energy_ablation` (about 3 minutes); the movement artifact is stored gzip-compressed. Post-hoc translation spectra: `.venv/bin/python -m experiments.clamped_energy_ablation_spectra`.
- `correlated-nuisance-v1-contract.md`, `correlated-nuisance-v1-results.md`: nuisance correlated across modules in training and broken at test. Inner-only coupling filters it (nuisance gain about 0.05) while all surface-coupled families transmit it, but the narrow inner channel loses so much signal that it is worse even on the decorrelated test at b = 2 (counts against the mechanism on this fixture) and better only at b = 10 in a condition that fails retention; when nuisance dominates, inner coupling transmits too, but those conditions do not learn (descriptive only); adding deeper coupling to surface coupling changes nothing; outer-only coupling matches uniform on v1's data. Reproduce with `.venv/bin/python -m experiments.correlated_nuisance` (about 4.5 minutes); post-hoc recentred reduced-rank comparator and gain split: `.venv/bin/python -m experiments.correlated_nuisance_posthoc`.
- `inner-channel-diagnostic-v1-contract.md`, `inner-channel-diagnostic-v1-results.md`: whether inner coupling can keep its filtering and recover the signal. Uncentered rank-2 regression does both (decorrelated error 4.8-9.7× below uniform coupling), so rank restriction is not the barrier; an evaluator-assisted fit without clips or weight decay also does, using large weights, while within the learner's constraints neither a capacity fit nor masked completion beat the clamped-energy learner; training settles are not exact, with the largest discrepancy during stage A. Reproduce with `.venv/bin/python -m experiments.inner_channel_diagnostic` (about 24 minutes); outputs stored gzip-compressed.
- `energy-scaling-analysis.md` (post hoc, after the inner-channel diagnostic): weight scaling exactly reweights every deeper energy term relative to outer reconstruction; the constrained inner arms' signal error is 97-100% a scalar gain loss (0.78-0.86) that a scalar chain model predicts from learned singular values, caused by the fixed activity priors; the clamped-energy learner keeps unit translation gain and so does not compensate. Lowering both priors to 0.005 at the learner's fixed parameters raises gain to 0.97-0.98 and cuts decorrelated error 7-14× with filtering intact. Reproduce with `.venv/bin/python -m experiments.energy_scaling_analysis` (under 1 s) and `.venv/bin/python -m experiments.prior_intervention_analysis` (about 7 s).
- `prior-strength-training-v1-contract.md`, `prior-strength-training-v1-results.md`: training the clamped-energy inner learner with weaker activity priors (0.005, 0.001) keeps the repaired completion and the filtering: decorrelated error falls 4.2-9.4×, signal gain is restored to 0.97-0.99, and the inner learner beats uniform coupling under decorrelation by 1.9-5.3× while matching uncentered rank-2 regression. On v1's own data the uniform-over-inner ranking reverses at weak priors. Clipping and exact training made no detectable difference; removing decay cost filtering. Reproduce with `.venv/bin/python -m experiments.prior_strength_training` (about 16 minutes); results stored gzip-compressed.
- `weak-prior-topology-v1-contract.md`, `weak-prior-topology-v1-results.md`: checkpoint repeat of the correlated-nuisance topology comparison at priors 0.05 and 0.001 on fresh seeds. Transmission and filtering categories are properties of the topologies (identical at both priors); every ranking involving inner coupling changes with the prior, and at 0.001 the inner family beats uniform and outer coupling on the decorrelated test 2.4-7.1× and matches uncentered rank-2 regression. Strong-regime inner coupling still does not learn. Reproduce with `.venv/bin/python -m experiments.weak_prior_topology` (about 11 minutes); results stored gzip-compressed. Summary of the line: `linear-line-checkpoint.md`.
- `relational-linear-preflight-v1-contract.md`, `relational-linear-preflight-v1-results.md`: generator, identifiability and capacity preflight for the relational proposal's linear fixture, with the three-module weak-prior SRI as a two-step window and per-action inner translation banks. The generator passes; a capacity witness exists at the learner's ±2 clips (privileged multi-horizon search, median X₁₂₈ 1.3e-6 to 3.2e-5); binding refits with frozen banks pass the invariance threshold (J_G ≤ 2.7e-4), but post-hoc spectra show drift that could matter at L = 128 (its share of the measured error is not isolated); observed-data fits reach only X₁₂₈ ≈ 0.1 (objective mismatch); 32-step prediction-time settling is far from converged (175-898 steps needed); the direct references show a transfer gap at small budgets, with the frozen latent reference capped by attenuation. Outcome by the frozen rule: proceed. A later [assessment](relational-linear-fixture-assessment.md) found no original advantage established on this fixture. It does not show that none can survive, and does not decide between continuing and closing the line. It maps the fixture to the program obligations it leaves untested. Reproduce with `.venv/bin/python -m experiments.relational_linear_preflight` (about 72 minutes on 11 workers); results stored gzip-compressed; draft checks: `.venv/bin/python -m experiments.relational_linear_preflight_checks`.
