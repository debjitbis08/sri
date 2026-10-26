# Learning transferable relational abstractions

2026-10-26. Discussion proposal, not a frozen contract, implementation, or result.
Revised the same day, after the [linear-line checkpoint](linear-line-checkpoint.md),
to make the existing SRI module the primary candidate.

Source: the user's [original research-direction note](relational-abstraction-original-note.md),
preserved verbatim separately. This document develops an experimental proposal
and assessment of that direction.

## Assessment and claim

The useful next question is whether a learned reduction of distinctions preserves
behavior and improves transfer. A unique equilibrium for fixed parameters does
not rule out abstraction in learned parameters. Conversely, low energy, stable
codes, compression, or a sleep schedule do not establish abstraction. Prediction
can train structural representations; success would distinguish reusable
behavioral structure from memorized reconstruction, not refute optimization.

In the original note's driving-to-painting example, a person drives 1,000 km,
experiencing movement, anticipation, fatigue, uncertainty, and eventual release.
They later express similar feelings in an abstract painting without depicting
the journey. The proposed shared pattern is **Tension → Persistence →
Transformation → Release**: relational organization transferred across different
sensory experiences. This motivates the question but supplies neither a
ground-truth equivalence nor an objective test of shared meaning. Do not encode
these four named emotional stages in the first generator.

Operational hypothesis: after source experience, a learned abstract transition
model reduces target evidence needed to predict previously unseen compositions,
relative to replay, compression, and ordinary predictive model transfer. It must
also distinguish structures with different consequences.

Use the current continuous linear SRI module as the primary architectural
candidate. The [prior-strength study](prior-strength-training-v1-results.md)
provides the starting configuration: three 8 → 4 → 2 modules, learned inner
translations, clamped-energy learning, weak activity priors, and weight decay.
At prior 0.001 it preserved signal while filtering nuisance and matched the
uncentered rank-2 regression reference on the decorrelated fixture; the
[weak-prior topology study](weak-prior-topology-v1-results.md) repeated this on
fresh seeds. This is
evidence of useful learned filtering, not yet relational abstraction or a
universal best configuration. Carry the existing clips and decay into the
baseline; declare any change and select learning rates on development data.

The existing [quotient example](inner-relation-quotient-analysis-results.md)
supplies its equivalences. Learning them remains the new obligation. The
[Stage B study](relational-capacity-stage-b-v1-results.md) found no capacity
witness for a different fixture and tested mechanisms; it does not exclude the
newer linear learner from this study. First audit this learner's capacity on
the proposed task, then separate learning from transfer. Keep an explicit
quotient learner as a reference, rather than requiring it to succeed before
testing SRI. The linear line established filtering of a low-rank channel on one
fixture family, not composition of dynamics; nothing here assumes it extends to
action-conditioned transitions.

## Mathematical target and identifiability

For domain i, learn a state map phi_i, action correspondence psi_i, abstract
transition T_bar, and an observable-outcome decoder h_bar. Require on held-out
transitions:

    phi_i(T_i(s,a)) ≈ T_bar(phi_i(s), psi_i(a))
    h_i(s) ≈ h_bar(phi_i(s))

Test their composition on held-out action sequences. If a domain is partially
observed, s must be a history or inferred state; a raw observation need not be
Markov. Begin with fully observed rendered states to avoid conflating that problem.

Composition alone admits a constant phi and a one-state model. Outcome
preservation prevents this when some action sequences have distinct observed
consequences. Define equivalence relative to a declared family of observable
queries, not all conceivable future meanings. A task requiring discarded
information should fail; measure that boundary explicitly.

Unpaired arbitrary state/action symbols cannot in general be aligned uniquely.
Supply a measured target calibration budget, including observations of outcomes,
to every arm. Score only identifiable queries or predictions invariant under
remaining symmetries. Do not resolve ties with hidden state IDs. Zero-calibration
transfer is a diagnostic, not a required success. Both state and action
correspondence must be learned or explicitly disclosed as supplied.

In the linear fixture below these objects are defined operationally, by the
exact equilibrium solve, not by individual learned matrices; see
[the operational definition](#operational-state-transition-and-rollout). They
need agree with the generator only up to one invertible change of latent basis
shared by all actions. With action identity supplied, psi_i is the identity and
is disclosed as such.

## First fixture: shared linear relationships across domains

Start with a low-dimensional latent state z and two bounded linear operators
F_a whose compositions depend on action order. Domain i observes

    x_i = R_i z + B_i n,       z_next = F_a z,

where R_i and B_i are domain-specific renderings and n is nuisance independent
of the latent dynamics. Start with latent dimension two to fit the existing
inner width; use eight-dimensional observations and audit separation of signal
and nuisance. Fit only from observed transitions, never from z, F_a, or the
rendering matrices. Those remain evaluator-only. A visible outcome probe is a
nonconstant function of z that must be declared, and its observations counted
as ordinary environment evidence. Audit observability across action sequences:
the probes must distinguish the latent directions that transfer queries need.

Use independent renderings in source and target domains, multiple operator
families, varied initial states, and nuisance shifts. Train on short trajectories;
test unseen initial states, action sequences, and longer compositions. Include
noncommuting operators, broken-structure targets, and nuisance that becomes
correlated across observations. Keep the transition law exactly linear: do not
normalize states during or after a trajectory. Per-step normalization makes the
dynamics nonlinear, and whole-trajectory scaling makes the initial-state scale
depend on the later action sequence. Bound trajectories by construction
instead, for example norm-preserving operators with a fixed maximum horizon.
In two dimensions rotations commute, so a noncommuting bounded pair needs, for
example, a rotation and a reflection. Audit the declared horizon for overflow,
collapse and conditioning before fitting. A fixture whose transition law
includes normalization is a separate, nonlinear fixture; failure on it is not
evidence about the linear learner.

Initially supply action identity consistently across domains and disclose that
choice. Learn target observation bindings from a measured calibration budget.
Independent action renaming and learned action correspondence are a later
condition. The first study therefore tests transfer across observation
representations, not discovery of entirely unfamiliar action meanings.

In this fixture an abstraction can be a continuous map that removes nuisance
while preserving action effects; it need not be a discrete partition. Recovery
of the generator's particular latent coordinates is unnecessary: invertible
changes of latent basis are equivalent. Evaluate behavioral preservation and
transfer, not agreement with arbitrary coordinate labels. Linear reduced-rank
system identification and predictive transfer are particularly strong controls
here; a learned shared subspace alone is not the full success criterion.

## Operational state, transition and rollout

Equilibrium activities can depend on the selected action, the observation
mask, and whatever else is clamped, and a learned translation is not by itself
the transition the full solve implements. Fix the following before freezing.

- **State encoder.** phi_i(x) is the exact-equilibrium inner activity with only
  the current observation clamped, under one declared mask that has no action,
  next-state or outcome input. It must be computable from the current
  observation alone.
- **Common coordinates.** All actions act on that same inner activity space.
  An action interface whose operators live in different coordinates must
  either declare the maps between them or fail this requirement.
- **Effective transition.** G_{a,i} is the linear map from phi_i(x) to the
  inner activity of the predicted next state, extracted from the exact solve
  with action a selected. Because exact inference is linear for fixed
  parameters and masks, G_{a,i} can be computed as a matrix and compared
  directly; do not substitute the learned translation for it.
- **Rollout.** An L-step prediction is h_bar(G_{a_L,i} ··· G_{a_1,i} phi_i(x_0)),
  or the declared equivalent that decodes and re-encodes, given only x_0 and
  the action sequence. No intermediate ground-truth observation may be clamped
  in a mixed-action rollout.

Freezing shared translations while fitting target bindings does not by itself
guarantee frozen effective dynamics, since the bindings enter the solve. The
capacity preflight must therefore verify that, after the target binding fit,
the target's effective operators equal the source's up to a single basis
change shared by all actions, at a declared tolerance. Report any drift
separately from prediction error. If bindings can change G, transfer of
"frozen" structure has not been isolated, and the action interface must be
redesigned before learning is tested.

## Later fixture: finite relational quotients

Use small deterministic controlled transition systems: initially six latent
states, two actions, and two nuisance variants per state. Each observable state
gets an arbitrary token; no token reveals latent state or nuisance membership.
Actions change the latent state according to a hidden transition table, while
nuisance changes without influencing latent transitions or observable outcomes.
A nonconstant binary outcome is observed during interaction; it is ordinary
environment feedback, not a latent-class or cross-domain correspondence label.

Construct several source environments with independent token/action renamings
and nuisance dynamics. Target environments receive fresh renamings and sparse
calibration transitions. Shared structure is built into the benchmark, but the
learner receives neither its table nor equivalence labels. Use more than one
structural family so that always assuming a single graph cannot pass. The first
test concerns representation transfer within these finite families; genuinely
new structure families and new action semantics are later tests.

Audit the generator before fitting: the six proposed states must actually be
behaviorally distinguishable by outcome sequences, nuisance duplicates must be
equivalent, and selected calibration/query pairs must have the advertised
identifiability. Include a noncommuting action pair so order matters. Include
broken-structure targets with matched easy statistics but different compositions.
Failure of these audits requires redesign before confirmation, not exclusion of
unfavorable test worlds after observing a learner's score.

Source training exposes short trajectories; evaluation uses unseen sequences
and longer compositions. Target calibration withholds selected edges as well as
paths, separating inference of missing relationships from merely executing a
fully observed transition table. All queried observable tokens must have been
introduced; unseen token recognition is a separate perception problem.

This finite fixture tests a broader reduction of distinctions after the linear
experiment. Arbitrary token-to-class assignments are not presumed realizable by
the current linear module. Run a separate capacity audit before using this
fixture to assess its learning. Do not silently supply a nonlinear encoder or
the correct partition to make it pass.

## Primary mechanism: existing SRI, then a declared consolidation extension

First reuse the module's equations, clamped-energy rule, learned translations,
weak priors, and weight decay. Source waking experience supplies observed
transition pairs and outcomes; replay is the established consolidation baseline.
The temporal task is new: before freezing, specify which ports receive current
state, next state, and outcomes, their masks, and every adapter. The original
three-module completion fixture does not already implement action dynamics.

For fixed parameters and an observation mask, exact quadratic inference is
linear in clamped values. Multiple arbitrary action operators cannot generally
be implemented merely by appending an action indicator to that same fixed
linear map. Start the capacity preflight with one action-conditioned relation
at a time. For multi-action composition, explicitly specify and cost a bank of
action-indexed learned translations or another action interface. Such a bank
reuses the module internals but adds externally selected relation structure;
it is an extension, not evidence of autonomous relation selection. Do not
attribute a supplied action selector's capability to the unchanged module.

Use exact equilibrium for the capacity reference and primary evaluation, and
the operational encoder, effective transitions and rollout defined above.
Compare iterative inference to it at a declared tolerance; the
[prior-strength study](prior-strength-training-v1-results.md) found large
training-settle discrepancies despite similar final task errors.
A fixed iteration count alone is not evidence of convergence on the new task.

After establishing representability, test the existing learning rule on
independent source experience. Freeze shared source relationships for transfer
and fit only declared target observation bindings from calibration, checking
that the effective operators stay fixed as defined above; report a
fine-tuning variant separately. Initially measure whether ordinary SRI learning
already preserves composition and transfers. Success need not require inventing
a new sleep rule.

Then compare ordinary replay against a proposed structural consolidation rule
within the same architecture. A candidate is to propose changes to shared
translations and retain those improving held-out multi-step behavioral
prediction subject to one-step accuracy and retention constraints. This adds
a selection criterion; it is not already supplied by clamped-energy learning.
Its proposal distribution, locality, acceptance thresholds, validation use,
and cost must be fixed in a new contract. Any global selection or gradient
must be disclosed rather than described as purely local learning. Compare to
ordinary multi-step predictive training using the same evidence and budget.

Recombinations can propose consistency tests but cannot count as independent
evidence that an unobserved relationship is real. Replay-based prototypes do
not satisfy the no-history dreaming commitment in foundations.md. A no-history
implementation requires its own later test; the initial study isolates learned
structure and transfer.

## Explicit quotient reference for the finite fixture

Wake builds empirical transition/outcome counts from observations. Sleep searches
for a smaller state partition and shared abstract transition table across source
environments. Begin with singleton blocks; propose pairwise merges, refit the
abstract table, and retain merges that shorten an explicitly encoded model plus
data description while preserving prediction on separate waking episodes.
Do not treat missing transitions as evidence of equality. Penalize unexplained
outcomes and transition conflicts; retain uncertainty on unseen entries.

This reference is an MDL/state-merging, homomorphism-inspired learner, not a new
established algorithm or a local SRI rule. Fix the coding scheme, merge order,
tie-breaking, candidate cap, predictive tolerance, validation use, and handling
of uncertainty before freezing. Never insert the true six-state partition or
select its size using evaluation labels. Start with exact tiny-table alignment
as a transparent reference; count every rejected correspondence candidate.

Freeze the source abstract table at transfer. Fit only target state/action
bindings from calibration; compare against the target-only arms and transfer of
the uncompressed source model. Separately report a fine-tuning variant if needed.
This reference retains counts and, if validation episodes are stored, history;
the recombination and no-history limits above apply to it too.

## Experiments and controls

1. **Correctness and capacity.** Audit observability and alignment, and verify
   composition against a generator oracle. Fit the actual SRI candidate with
   privileged targets on development worlds to test representability; check
   held-out states and compositions, exact versus iterative inference, and a
   direct linear equivalent for each fixed action/mask. Verify that target
   binding fits leave the effective operators unchanged up to a shared basis
   change, and that mixed-action rollouts use only x_0 and the actions. Keep
   these privileged fits out of all learning initializations and confirmation. Failure to fit
   is unresolved unless an analytical capacity bound rules the task out.
   For the finite fixture, also exhaustively check tiny supplied quotients,
   invalid merges, and ambiguity. Ordinary learners must never access hidden IDs,
   generator matrices/tables, or held-out outcomes.
2. **Learned abstraction.** Train SRI from observed experience and measure
   nuisance invariance, discrimination, held-out behavioral prediction,
   composition error, and retained model size. Compare constant representations,
   uncompressed models, and random matched-width projections. For finite
   quotients, additionally evaluate partitions against evaluator-only behavioral
   equivalence. Compression without behavioral accuracy is failure.
3. **Cross-domain transfer.** Compare target accuracy curves at calibration
   budgets such as 0, 4, 8, 16, and 32 transitions. Choose the actual budgets
   after generator audits, before confirmation. Primary endpoint: paired
   difference in normalized quality-versus-budget area on held-out compositions:
   use declared normalized prediction error for continuous outcomes and accuracy
   for finite outcomes, with the normalization fixed before confirmation.
   Report it separately against the target-only arms and against the
   strongest matched transfer comparator.
4. **Mechanism and boundary.** As a supporting diagnostic, scramble the
   learned transitions while retaining code size: for SRI, permute the action
   bank or replace each effective operator with a random one of matched
   eigenvalues; for the finite reference, permute abstract table entries.
   Apply comparable interventions to the predictive controls, and declare
   whether target bindings are refitted afterward. Before scoring, audit
   whether each intervention changes observable behavior modulo the allowed
   symmetries: permuting two operators is undone by a basis change when one
   exists that conjugates each into the other while preserving the probes.
   Scrambling shows reliance on learned dynamics, which any predictive model
   has; it does not by itself show relational abstraction. Remove composition
   constraints; test broken structure;
   and compare batched sleep against interleaved execution of the same rule at
   equal exposures and work. Retention/adaptation and dream proposals are later
   extensions, not prerequisites for this first transfer test.

Required SRI arms are wake-only learning, ordinary replay, compression-only
consolidation, and proposed structural consolidation, with capacity and action
interfaces matched. Compression-only consolidation replays the same experience
under a declared compression penalty (for example on activity rank or
weight description length), without the
multi-step selection criterion; it separates the effect of compression from
that of structural selection. Ordinary SRI replay is the starting baseline, not
a presumed failure. Also include direct linear prediction, reduced-rank predictive transfer,
and multi-step predictive training. On every fixture, include target-only SRI
and a target-only direct linear or reduced-rank predictor, trained on exactly
the same target calibration evidence and with no source experience. Include the tabular state-minimization/model-
matching reference on the finite fixture. These distinguish architectural benefit
from the benefit of shared structure or a different training objective.

Use the same observed evidence and target calibration policy. Include extra
wake updates as a compute control. Report parameter count, retained history,
total bytes, exposures, selection cost, and quality at matched compute; equal
latent dimension is insufficient. The direct reference may itself be the best
method. That supports the capability, not an SRI-specific advantage.

## Decision rules and statistics to freeze

Separate claims of compression, composition preservation, transfer, and benefit
from scheduling. A benefit from abstraction does not imply a benefit from sleep.
Predeclare a behavioral-error ceiling (or finite-task accuracy floor) and a
practically meaningful transfer margin on
development worlds, before independent confirmation. Use independent generated
worlds as replication units, pair arms on worlds, and report paired confidence
intervals; paths within a world are not independent replicates. Reserve distinct
development and confirmation seeds and budget equal tuning opportunities.

Keep two comparisons separate. First, whether source experience helps:
transfer arms against the target-only arms on the same calibration evidence,
which tests the operational hypothesis directly. Second, whether structural
consolidation helps: the proposed rule against ordinary replay, compression-only
consolidation and multi-step predictive training at matched evidence and
budget. Only the second can support a consolidation-specific benefit.

Evidence for the mechanism requires accurate noncollapsed abstractions, an
advantage over target-only learning, and a transfer advantage over the
strongest matched comparator. Loss of the advantage under scrambling is
supporting evidence only, and only for interventions shown to change behavior
modulo allowed symmetries. On broken structure,
report negative transfer and uncertainty rather than forcing shared agreement.
If the oracle fails, repair the benchmark. If SRI capacity is ruled out, state
the obstruction and propose a separately tested extension. If a capacity witness
exists but ordinary learning fails, diagnose learning before transfer. If source
learning succeeds but transfer fails, diagnose alignment and reuse. If ordinary
predictive transfer matches the candidate, the proposed structural consolidation
has no demonstrated additional benefit. A successful unchanged SRI baseline is
itself a useful result, even if the additional sleep rule adds nothing.

## Computation obligations

For SRI, let d be the number of free activities, P learned parameters, H action
operators, U training updates, J tuning configurations, T settling iterations,
and F the work of one implemented settling update. A dense exact solve costs
O(d³) factorization time, O(d²) solve time per right-hand side, and O(d²)
workspace; factorization reuse is valid only while parameters and masks remain
fixed. Iterative settling costs O(TF), with actual F derived from the chosen
block structure. Count local weight updates, model assembly, and validation
separately. Total fitting includes all J configurations and U updates, not just
the selected run. An action bank stores its separate translations; full model
copies could multiply parameter and optimizer storage by H and must be reported.

Extracting each effective operator G_{a,i} costs one factorization plus one
solve per inner dimension for each action and mask; repeat it after every
binding fit that the invariance check covers. An L-step latent rollout with
extracted operators costs O(L k²) for inner width k; a rollout through repeated
solves repeats the selected inference procedure L times. Include
target adapter fitting, capacity searches, failed proposals, repeated solves,
multi-step validation, and action-alignment search in the total. Report learned
parameters, optimizer state, traces, adapters, and peak workspace separately.
Use measured costs for any comparison with direct linear rollout. Exact and
iterative algorithms are separate cost profiles, even if their outputs agree.

For the finite reference:

Let N be observed state tokens, A actions, E transition records, K abstract
states, C scored partition candidates, V validation records, and M scored
cross-domain correspondences. A simple sparse count builder takes O(E) time
and O(min(E,N²A)+N) storage. A full rescan per partition candidate costs
O(C(E+V+N+KA)); sparse abstract edge storage is O(min(E,K²A)). Correspondence
search by rescoring each mapping costs O(M(E+V+N+KA)); exhaustive permutations
can make M factorial. This is acceptable only for a tiny reference and must
not be called scalable. Derive the actual implementation's bounds at freeze.

Store partition assignments, correspondence maps, count tables, abstract model,
validation/replay, and temporary candidate workspace separately. Deterministic
table rollout costs O(L) for length L once bindings are resolved; uncertain
multi-hypothesis rollout has additional cost that must be counted explicitly.

Follow computation-reporting.md: separate scientific and runtime JSON; record
hashes, environment, wall/CPU time by phase, peak RSS, dtype, retained bytes,
all fits/candidates/exposures, oracle audits, development selection, failures,
confirmation, and deployment timings. No measurements exist for this proposal.
Historical runs lacking comparable cost fields must be labeled as such.

## Prior work and next deliverable

See the relational-abstraction entry in [prior work](prior-work.md). Relevant
established ideas are transition-system quotients, MDP homomorphisms,
bisimulation, predictive state abstraction, MDL/state merging, and structural
generalization in TEM. The [linear-dynamics entry](prior-work.md#linear-action-conditioned-dynamics-and-transfer)
covers system identification up to similarity, joint learning of shared
dynamics, linear representation transfer, action-conditioned linear models,
transfer across observation spaces, and multi-step objectives. It predicts,
without running SRI, that a rotation-and-reflection pair makes target binding
identifiable up to scale and makes operator swaps detectable; that the source
benefit over a target-only shared-encoder latent model should be modest at
large budgets; and that it should concentrate at small budgets and long
horizons, where norm-preserving operators accumulate estimation error. The
preflight should fix budgets and horizons from these direct-reference
predictions, and add the target-only shared-encoder latent model and
simulation-error multi-step training as references. The specific structural
consolidation objective has not yet been reviewed. This draft does not
establish novelty. Exact preservation is a mathematical property; discovery
from incomplete experience and useful transfer are the empirical obligations.

The next deliverable is a generator/identifiability and SRI-capacity preflight,
starting with the weak-prior linear module; its frozen contract is
[relational linear preflight v1](relational-linear-preflight-v1-contract.md). Freeze its action interface and
cost accounting before execution, followed by a learning/transfer contract with
sample counts, thresholds, seeds and tuning budget. Keep capacity, learning,
transfer, and sleep-specific outcomes distinct. This proposal makes no claim
about subjective meaning, artistic creativity, or biological sleep.
