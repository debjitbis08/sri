# Core abstraction v2 — context-consistent preservation and relational evidence

**Status:** prospective draft for review and implementation handoff. Not frozen,
implemented, or scored. The v1 contract, code, and reports remain unchanged.

## 1. Why v1 failed and what changes

[v1](core-abstraction-v1-results.md) required isolated waking responses to remain
unchanged after assembly, while scoring transfer against a connected teacher
whose responses differ from those isolated ones. On the true blocks, S1's own
core loses 3–6 of those waking targets and C2's loses 5–8, despite making every
mixed marginal correct. Preserving cores yield no transfer on those assemblies.
This is a failure of the declared conjunction, not evidence that dreaming fails
or that a larger core is needed. S1 also exhibits assembly interference before
the core changes in 9 of 10 runs.

The v1 acquisition protocol removes both core links during every observation.
Its state therefore contains no information distinguishing the nine possible
cores. Repetition cannot recover the missing distinction. The information
certificates additionally show cases requiring incompatible core choices from
identical acquired states.

V2 makes two explicit changes, evaluated separately:

1. Judge preservation in the context where the behavior was acquired. Keep
   isolated behavior under isolated tests; judge coupled correctness against
   coupled targets. Report the old cross-context invariance criterion as a
   diagnostic, without requiring it to pass.
2. Supply a bounded set of connected waking observations, so satellite structure
   can encode evidence about relationships before consolidation. Compare against
   the unchanged isolated-information arm. These are additional external labels,
   not dreams or free internally generated knowledge.

Keep the S1/C2 substrate and nine-core family. Do not add core capacity or change
the old local learner. Preservation-signal, plateau-move, larger-edit, and dream
discovery studies remain deferred.

## 2. Question and limited claim

Can complementary, axis-only connected observations, encoded in finite learned
constraints, select a core that improves an untrained mixed response in one
satellite while retaining its established behaviors? Does information supplied
to the other satellite matter to that selection?

Distinguish an evaluator-found core, a core selected from permitted observations,
and an endogenous consolidation mechanism. V2 contains the first two only.
Its exact constraint representation is an information reference, not a proposed
SRI plasticity law. A pass establishes acquired reusable coupling in this finite
family, not deeper abstraction, compression, or a solution to forgetting.

Recovering the teacher's core is not disqualifying: the teacher defines the
target behavior. Conversely, reproducing its coefficients is insufficient.
Require held-out reuse and donor-information controls. Report teacher-core
recovery separately from any non-teacher solutions; do not reward a wrong core
merely because it compensates for wrong private blocks.

## 3. Fixed setting

Import unchanged dynamics, complete reachable-terminal sets, queries, domain
constraints, canonical starts, and controls from v1. Use S1/C2 as primaries and
S2/T1 as specificity controls. The latter are not assumed negative for this
new transfer task. Retain all five splits, both alternating A/B assignments,
and five frozen orders. Do not treat order-independent final states as samples.

Retain `p_A`, `p_B`, `c=(w_KA,w_KB)`, `r=w_12`, and `c0=(0,0)`. Only `c` changes
at installation. Input infrastructure `r` remains supplied, as in v1. All 36
mixed queries are withheld from both satellites; all 30 within queries remain
secondary held-out tests. Neither selector may inspect them.

## 4. Acquisition and experimental arms

First reproduce v1's isolated acquisition, obtaining nine-bit masks `V_A,V_B`
from 15 total addressed-satellite observations. Record the same uncertainty and
assembly-interference results. No new private-block fitting rule is introduced.

Before any connected observation, choose an executable private assembly
`p*=(min V_A,min V_B)` using the frozen lexicographic block order. Freeze both
private blocks thereafter. This deliberately explicit prior can be wrong; do
not switch to another assembly after learning its performance.

In addition to that executable branch, evaluate every `p in V_A × V_B` as a
reference branch with private blocks fixed throughout. Report true-private-block
branches as evaluator-only ceilings, never as the actual learner. Do not pool
the best branch per task and call that a learned result.

Run three arms from identical isolated masks and private branches:

| Arm | Further observation budget | Purpose |
| --- | --- | --- |
| I: isolated only | None | Original information limit under revised scoring. |
| IR: isolated repetition | Repeat the original 15 isolated observations once | Same extra label count as C, without relational evidence. |
| C: connected evidence | 15 connected marginals on the same assigned training queries | Supply information about the relationship. |

In C, each query supplies only the assigned satellite's complete marginal under
the unchanged, fully connected teacher. A receives its 8 assigned marginals and
B its 7 (swapped in the other condition). No joint sets, hidden input readouts,
core values, intervention labels, or partner marginals are supplied. The query
IDs are those already assigned without labels. No new mixed inputs are added.
Total external observations: I has 15; IR and C each have 30. Count target-set
cardinalities and simulations separately; these arms are not compute-matched.

## 5. Finite structural reference and selector

For each fixed private branch `p`, each satellite has a nine-bit core constraint
mask `C_s(p)`, initially all cores. A connected observation `(x,f_s)` updates
only its addressed satellite's mask:

`C_s(p) <- {c in C_s(p) : M_s((p,c,r),x) = f_s}`.

Here `M_s` is the full reachable-terminal marginal, with no chosen trajectory.
Discard the observation after updating. I and IR leave these core masks full;
IR repeats the original private filtering and verifies its idempotence.

Consolidation computes `C_AB(p)=C_A(p) ∩ C_B(p)`. If nonempty, install its
lexicographically first core; ties cannot use held-out performance, teacher
identity, or private-block truth. If empty, retain `c0` for defined output and
mark the branch `inconsistent_private_assembly`, which cannot pass. Never reset
the mask, edit private blocks, or silently omit these failures from denominators.
The true-private branch must retain the teacher core; the preselected executable
branch need not. A selected core of `c0` is valid but gives no installation gain.

Persisting the masks requires no raw episode list. However, their semantics
reference both satellites' private blocks and full-system simulations. This is
a global finite hypothesis representation, not evidence of local structural
learning or a scalable memory mechanism. For one executable branch the two
masks use 18 logical bits plus executable coefficients; all-branch reference
tables use up to `2*81*9` bits. Account for runtime caches separately.

At inference, use only the installed coefficient vector and the existing
dynamics. Test serialization/reload after deleting feedback records and masks.
It must reproduce all installed-model outputs. Subsequent learning is out of
scope, so this does not establish a durable preservation mechanism.

## 6. Stage A: context-consistent feasibility

Before scoring the selector, enumerate all nine cores for each fixed branch,
with held-out data confined to the evaluator. Preserve the old v1 scores too.

Use separate quantities, never a pooled retention number:

- **Isolated retention:** accuracy on original isolated observations in their
  original environments. This follows from fixed private blocks and is a
  regression gate, not evidence that consolidation actively protects memory.
- **Coupled preservation:** let `P0` be all `(satellite, training-query)` pairs
  whose full coupled marginal at `c0` is teacher-correct. Require every member
  of `P0` to remain correct after the core change. Record its denominator per
  satellite; a nonvacuous full pass requires at least one member for each.
- **Connected acquisition:** exact accuracy on the 15 supplied connected
  observations. A consistent selected core must reproduce them. Keep this
  separate from preservation of behavior already correct before installation.
- **Cross-context change:** the v1 isolated-label-versus-connected-output
  comparison, reported as context sensitivity rather than automatically called
  forgetting. The old failed target remains a valid archived result.

Transfer requires a strictly positive net mixed-marginal gain for at least one
recipient relative to the same private branch at `c0`, at least one newly exact
mixed query, and no mixed-marginal regressions in either satellite. Record joint
set accuracy and all within-partition gains/losses separately. No held-out
quantity influences the selector or changes the train/test assignment.

These feasibility conditions may be known to be attainable on true blocks
with the teacher core under the revised target; disclose that analytical
expectation. The unknown is whether the permitted evidence and fixed selector
support useful, donor-dependent transfer without private-block oracle access.

## 7. Stage B: acquired transfer and controls

Score the declared selector independently of the feasibility ceiling. Require
isolated retention, coupled preservation, connected acquisition, and the mixed
gain conditions above. Evaluate all gains before selecting a readable witness.

For one gain `(recipient s, query x)`, require:

1. Core removal and unmodified core lose the gain. Both single core-link cuts
   also lose it, on that same query with the same private blocks.
2. Recipient-only consolidation, using `min C_s(p)` with the same tie rule,
   fails on that query; additionally `C_s(p)` must contain a core predicting a
   different recipient response there. This separates added donor information
   from a coefficient change with no predictive consequence.
3. Neither isolated-only, repeated-isolated, nor exact-query memorization
   predicts the correct new mixed response in the corresponding branch.
   Memorization keys include satellite and observation context and return
   unknown on unseen queries.
4. The installed result uses no further satellite updates or supplied labels
   on `x`. The same installed core applies to all evaluation queries.

Keep all nine fixed-core controls across all tasks. Compare the selected core
with each fixed-core prior, without choosing the best prior after evaluation.
Report whether connected evidence changes the core choice and whether a single
fixed core accounts for all primary-teacher success. S2 is especially useful
for detecting an always-complementary-core prior.

Retain input-frozen, NF, fixed-k, and private-block ablation diagnostics from v1.
Report results under two explicit route criteria:

- **Core-dependent:** both core links are necessary for the original gain.
- **V1 route criterion:** additionally, the installed-versus-initial recipient
  contrast survives clamping input interpretation to evidence.

The first is the v2 primary criterion. Failure of the second does not refute
core involvement: clamping input can disrupt a joint core–input mechanism.
This is a disclosed change of estimand, not a retrospective v1 success. Do not
call the first criterion proof of a unique causal route. Report counts under
both so improvements cannot be hidden behind a relaxed route gate.

A primary-teacher witness needs all the conditions in this section on one
branch/query. Report executable-branch success separately from true-block and
all-branch ceilings. The minimal v2 target is at least one executable witness
for each of S1 and C2; publish all 10 conditions per teacher, not just witnesses.
This is not a reliability threshold. A ceiling-only pass is not acquired
transfer by the specified learner.

## 8. Information and provenance checks

Compare before/after core masks, behavioral sign classes, disagreement on held-
out queries (evaluator-only), and donor-induced reductions in uncertainty.
Explain every failure as private-assembly inconsistency, remaining core
ambiguity, wrong prior selection, loss of preservation, no transfer, or failed
causal/donor control. Multiple diagnostics can apply; do not force a unique
cause from an ordered early-exit classification.

Mandatory checks: teacher survival on true blocks; clean filtering and order
independence; complete-set oracle agreement; symmetry; identical private state
before/after core installation; separation of isolated and coupled targets;
absence of held-out access; correct empty-mask handling; faithful repetition
and donor-omission controls; serialization without observations. Compare raw
response sets and test non-singleton targets, not just aggregate accuracies.

Freeze the reviewed contract, source hashes, assignments, targets, and manifest
before scored computation. Preserve all v1 artifacts. Use new v2 paths and
deterministic scientific reports, separate from volatile runtime metadata.
Run the relevant tests and full suite; require byte-identical regeneration.
No computation under this draft has been performed.

## 9. Stopping rule and the dream question

Stop after this grid. Do not retune observations, private-block priors, core
ties, or witness thresholds to obtain a pass. Do not add capacity or use a
held-out-selected private branch.

If C adds useful information but the executable branch fails, the next issue
is representation/acquisition of private structure, not insufficient dreaming.
If all masks remain ambiguous, identify which distinction the allowed evidence
lacks. If the exact reference passes, it becomes a transparent upper comparator
for a later endogenous rule that reorganizes genuinely retained relational
information without the full global hypothesis machinery.

That later study must test preservation during subsequent interference.
Neither an exact filter's retention guarantee nor repeated simulation of a
fixed model demonstrates that dreaming solves preservation.
