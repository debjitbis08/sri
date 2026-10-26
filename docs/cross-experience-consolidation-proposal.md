# Cross-experience consolidation: proposed next learning study

2026-10-26. Design proposal, not an implemented or validated learning mechanism.
Run the frozen distributed-evidence study independently before selecting a learning
implementation. The user distinguishes activity inference, selective immediate
memory, and long-term relationship consolidation during sleep/dream periods.

## What should govern change

Activity compatibility and relationship usefulness are distinct. An energy can
organize present activities without defining which relationships deserve to
persist. Multiplying a local energy gradient by a stability gate changes when
that same update occurs; it does not provide independent evidence of usefulness.
A wrong stable interpretation can reinforce itself. Temporal stability within a
single episode is also different from persistence across diverse experiences.

We should test stability as a permissive signal, and cross-experience predictive
support as evidence for retaining a proposed relationship. This is a hypothesis,
not a claim that stability identifies abstraction or that global task labels
are needed to assess local predictions. A relationship may need to weaken or
change despite stable activity; incompatible evidence must remain able to revise it.

## Concrete candidate to formalize before implementation

During waking freeze long-term relationships. Maintain bounded temporary traces
of locally observed activities, observation masks, episode identifiers, and cue
contexts. Distinguish observed evidence from values inferred by the model. An
inferred value or self-generated dream is not an independent confirming example.
Trace collection must not receive hidden pattern identities or evaluation targets.

During consolidation propose small local relationship changes from one subset of
traces. On disjoint waking episodes, mask an actually observed local coordinate,
reactivate from the remaining evidence, and compare the proposed relationship's
prediction of that coordinate with the incumbent's prediction. The hidden value
must be excluded from activity initialization, inference, gating, and proposal
construction for that comparison. This is a local predictive test with observed
values as self-supervision, not externally supplied pattern labels or a global
backpropagated task loss. It still introduces a learning criterion, whose role
must be stated explicitly.

Permit consolidation only when support spans multiple distinct observed contexts,
not just repeated replays of one trace. Use inner stability as an optional gate.
Require improvement on proposal-independent traces and limit degradation on
retained earlier traces. Keep a temporary candidate until there is enough support;
otherwise retain the incumbent. Candidate buffers, counters, validation traces,
rejected updates, and repeated inference all incur storage and computation costs.

Context diversity must be computed from observed local information, never hidden
pattern labels. Distance in raw inputs may count nuisance variation as diversity;
therefore explicitly audit this failure mode rather than assuming an effective
sample count. Dream recombinations are proposals or consistency probes, not proof
that a generated combination occurs in the world. Include incompatible synthetic
recombinations to test rejection instead of rewarding universal agreement.

## Evaluation and controls

Use two hidden structures with several outer variations each, sequential exposure
to the first and then the second, plus unseen variations reserved for evaluation.
Only the evaluator receives structure identities. Initial relationships must not
already encode those identities; initial capacity and any supplied relationships
must be disclosed. First verify that the local observed evidence makes the
structures identifiable: no learning rule can recover a distinction absent from
its observations. Freeze the generator, splits and hyperparameters before runs.

Measure separately: invariance across variations, discrimination between
structures, masked prediction/reconstruction on unseen variations, retention of
the first structure, and adaptation to genuinely changed relationships. Never
use these evaluation scores to select updates or choose hyperparameters on test
examples. Include constant-code and very slow-code diagnostics; apparent inner
stability alone cannot count as success.

Required matched controls:

- Frozen long-term weights and identical trace collection.
- Ungated local energy-gradient learning during consolidation.
- Within-episode stability-gated local energy-gradient learning.
- Cross-experience predictive-support learning without the stability gate.
- Cross-experience predictive-support learning with the stability gate.
- Slower/anchored inner activities without support-based consolidation, with a
  declared validation-only matching procedure and all its selection costs.
- Repetition of one trace versus varied independent traces at equal replay work.
- A direct local predictor/replay comparator using the same observed evidence,
  trace budget and proposal-validation split.

The paired gated/ungated support arms isolate the gate; comparing gated energy
learning with support-based learning alone would confound two changes. Match
proposal opportunities and report accepted-update counts, since fewer updates
can mimic protection from forgetting. Use identity-shuffled evaluation labels
only as a diagnostic; never provide identities to the learner.

## Computation contract required before execution

Let P denote relationship parameters, S activity values, B retained traces,
q observed scalars per trace, T inference updates, F one update's local work,
C candidate updates and V validation trace evaluations per candidate. Waking
inference costs O(TF) per episode; trace retention O(Bq). Candidate evaluation
costs O(CVTF), in addition to proposal construction, trace selection and any
stability statistics. Persistent state is O(P+S+Bq), plus candidate copies and
support bookkeeping; detail those terms once the algorithm is fixed. Slowing
activity may increase T, and validation may dominate total learning cost.

Before implementation fix dimensions, trace capacity/replacement, proposals,
gate parameters, validation partition, support/diversity criteria, acceptance
thresholds, seeds, schedules, sample counts, tuning grid and failure criteria.
Record prediction probes, rejected candidates, replay exposures and effective
independent episodes separately. Report quality versus total selection/training
cost, deployment latency and retained bytes, including raw traces. Use separate
deterministic scientific and nondeterministic runtime artifacts as specified in
[computation reporting](computation-reporting.md). Historical least-squares fits
are not sleep learning; historical missing cost data must remain explicit.

This proposal deliberately makes no learning result claim. The next design
obligation is an identifiable generator and a fully specified local candidate
rule; timescale separation by itself does not supply either.

## User refinement: accumulated knowledge should produce inertia

The intended early learning behavior is greater plasticity with little retained
knowledge, followed by increasing resistance to changing relationships supported
across diverse experiences. This inertia is a desired consequence of accumulated
support, not automatically a defect to eliminate. The human-development analogy
motivates this design choice; it is not a biological explanation or an empirical
claim established by this project. Mechanisms for maintaining high adaptability
after extensive consolidation are deferred.

Prefer relationship-specific inertia to a global clock: repeatedly supported
relationships become harder to revise, while unsupported or new relationships
remain relatively plastic. A possible implementation keeps a local support
quantity that reduces the size or acceptance rate of future changes. Its precise
rule remains to be specified. Repeated dreams of the same episode must not
increase independent support indefinitely, and mere settling must not count as
support. Stable wrong beliefs remain a possible failure mode to measure.

Add a prospective inertia evaluation: vary the amount and diversity of supporting
waking evidence before presenting a fixed contradictory experience sequence.
Measure relationship drift, old-pattern retention, and the exposures/work needed
to acquire the changed relationship. Compare fresh and consolidated relationships
under the same new evidence. Slower revision with stronger prior support is an
intended result, provided that support also improves retention or prediction.
Uniformly slowing every update can imitate the revision curve but does not by
itself demonstrate useful knowledge accumulation. Keep the matched slow/anchored
control for that distinction.

Report adaptation and forgetting separately rather than requiring maximum
adaptability as an early success condition. Include a matched elapsed-time and
repeated-single-trace control to distinguish support-driven inertia from a
hard-coded age schedule or replay count. Do not optimize away the measured
inertia in this first study; later work may address how to preserve reasonable
learning ability after extensive consolidation.

## Connection strengths and direction gains as candidate learned relationships

The three-module ablations motivate including inter-module coupling in the
candidate parameter set, not only each module's reconstruction weights. Separate
three choices: the layer's coupling strength, the direction's feedback gain, and
the translation between the connected representations. The graded-connection
study varies strengths/gains while supplying translations; it is a sensitivity
study, not a learned selection of connections.

Future consolidation can propose local changes to these parameters and retain
them only with predictive support across independent observed contexts, using the
same support-driven inertia as other relationships. Turning an edge off trivially
reduces its contribution to compatibility energy, so minimizing that energy over
connection strengths is not evidence that the resulting disconnection is useful.
Likewise, a large strength can enforce agreement without improving prediction.
Assess cue-conditioned reconstruction and retention, not agreement alone.

A parameter found unhelpful in a task where the output has no independent evidence
may become useful when that module has a relevant prior or other observations.
Do not freeze the present topology as a universal architectural conclusion. Keep
strengths nonnegative in the initial candidate family, account for direction-
dependent dynamics that no longer descend a common energy, and verify update
stability after proposed changes. Exact bounds, update rules and support criteria
remain obligations of the prospective learning contract; no coupling has been
learned in these frozen studies.

## First implemented test (2026-10-26)

The [local consolidation v1 experiment](local-consolidation-v1-results.md) implements
one bounded module-local proposal/check/support rule, the reviewer's stability
gate, sequential two-structure exposure, retention and unseen reconstruction
checks, and matched slow-inner/no-gate/no-inertia/repeated-trace controls. It does
not validate the proposed learning mechanism: second-structure cross-module
reconstruction fails to improve, the stability gate supplies no additional benefit,
and initial random codes already discriminate the easy structures. Local predictive
support and accumulated inertia are implemented, but their global usefulness is
not established. All observations and rejected proposals are costed. Learned
coupling, more informative support tests, trace recombination and autonomous
consolidation remain open. See the results rather than treating this proposal as
a validated specification.
