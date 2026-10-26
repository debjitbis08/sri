# Learning transferable relational abstractions

2026-10-26. Discussion proposal, not a frozen contract, implementation, or result.

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

The existing [quotient example](inner-relation-quotient-analysis-results.md)
supplies its equivalences. Learning them is the new obligation. The
[Stage B study](relational-capacity-stage-b-v1-results.md) found no capacity
witness for its tested mechanisms. Start with a small explicit algorithm; do
not presume the current SRI module can implement it. A successful tabular
mechanism would justify a later implementation study, not validate SRI itself.

## Mathematical target and identifiability

For domain i, learn a state map phi_i, action correspondence psi_i, abstract
transition T_bar, and an observable-outcome decoder h_bar. Require on held-out
transitions:

    phi_i(T_i(s,a)) ≈ T_bar(phi_i(s), psi_i(a))
    h_i(s) ≈ h_bar(phi_i(s))

Test their composition on held-out action sequences. If a domain is partially
observed, s must be a history or inferred state; a raw observation need not be
Markov. Begin with observable finite states to avoid conflating that problem.

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

## Proposed first fixture

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

## Minimal executable mechanism

Wake builds empirical transition/outcome counts from observations. Sleep searches
for a smaller state partition and shared abstract transition table across source
environments. Begin with singleton blocks; propose pairwise merges, refit the
abstract table, and retain merges that shorten an explicitly encoded model plus
data description while preserving prediction on separate waking episodes.
Do not treat missing transitions as evidence of equality. Penalize unexplained
outcomes and transition conflicts; retain uncertainty on unseen entries.

This is an MDL/state-merging, homomorphism-inspired learner, not a new established
algorithm and not yet a local SRI rule. Fix the coding scheme, merge order,
tie-breaking, candidate cap, predictive tolerance, validation use, and handling
of uncertainty before freezing. Never insert the true six-state partition or
select its size using evaluation labels. Start with exact tiny-table alignment
as a transparent reference; count every rejected correspondence candidate.

Freeze the source abstract table at transfer. Fit only target state/action
bindings from calibration; compare against target-only learning and transfer of
the uncompressed source model. Separately report a fine-tuning variant if needed.
Recombinations can propose consistency tests but cannot count as independent
evidence that an unobserved transition is real.

This prototype retains counts and, if validation episodes are stored, history.
It does not satisfy the no-history dreaming commitment in foundations.md.
That is an explicit scope limit of the mechanism experiment; a no-history
implementation requires its own later test. Replay remains a legitimate control.

## Experiments and controls

1. **Correctness and feasibility.** Exhaustively verify supplied tiny quotients,
   composition, invalid-merge rejection, action order, and ambiguity handling.
   A hidden-structure oracle must solve identifiable held-out queries. Learners
   must be unable to access evaluator IDs, splits, seeds, or generator tables.
2. **Learned abstraction.** Test learned partitions against evaluator-only
   behavioral equivalence, held-out outcome accuracy, and retained model size.
   Constant codes, identity partitions, and random matched-size partitions are
   diagnostics. Compression without behavioral accuracy is failure.
3. **Cross-domain transfer.** Compare target accuracy curves at calibration
   budgets such as 0, 4, 8, 16, and 32 transitions. Choose the actual budgets
   after generator audits, before confirmation. Primary endpoint: paired
   difference in normalized accuracy-versus-budget area on held-out compositions.
4. **Mechanism and boundary.** Scramble the learned abstract transitions while
   retaining code size; remove composition constraints; test broken structure;
   and compare batched sleep against interleaved execution of the same rule at
   equal exposures and work. Retention/adaptation and dream proposals are later
   extensions, not prerequisites for this first transfer test.

Required learning arms are wake-only prediction, prediction with ordinary
replay, prediction with a compression bottleneck, and relational abstraction.
Also include ordinary predictive latent/model transfer and a direct tabular
state-minimization/model-matching reference. The four proposed arms alone cannot
show that explicit relational abstraction improves on predictive transfer.

Use the same observed evidence and target calibration policy. Include extra
wake updates as a compute control. Report parameter count, retained history,
total bytes, exposures, selection cost, and quality at matched compute; equal
latent dimension is insufficient. The direct reference may itself be the best
method. That supports the capability, not an SRI-specific advantage.

## Decision rules and statistics to freeze

Separate claims of compression, composition preservation, transfer, and benefit
from scheduling. A benefit from abstraction does not imply a benefit from sleep.
Predeclare an accuracy floor and practically meaningful transfer margin on
development worlds, before independent confirmation. Use independent generated
worlds as replication units, pair arms on worlds, and report paired confidence
intervals; paths within a world are not independent replicates. Reserve distinct
development and confirmation seeds and budget equal tuning opportunities.

Evidence for the mechanism requires accurate noncollapsed abstractions, a
transfer advantage over the strongest matched comparator, and loss of that
advantage when the learned relationships are scrambled. On broken structure,
report negative transfer and uncertainty rather than forcing shared agreement.
If the oracle fails, repair the benchmark. If the oracle passes but learning
fails, diagnose discovery/alignment. If ordinary predictive transfer matches
the candidate, explicit quotient learning has no demonstrated additional benefit.

## Computation obligations

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
generalization in TEM. Exact preservation is a mathematical property; discovery
from incomplete experience and useful transfer are the empirical obligations.

The next deliverable is a generator/identifiability preflight and a frozen
contract specifying the remaining algorithmic choices, sample counts, thresholds,
seeds and tuning budget. Only then run confirmation. This proposal makes no
claim about subjective meaning, artistic creativity, or biological sleep.
