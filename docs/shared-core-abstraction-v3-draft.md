# Shared core abstraction v3 — matched views and complementary evidence

**Status:** prospective draft contract, 2026-10-24; not frozen, implemented, or
scored. This turn drafts the experiment only. Preserve all v1/v2 artifacts.

## 1. Question and prior knowledge

Can A and B learn different, individually insufficient constraints on one
relationship from local views of matched events, combine those constraints in
the core, and thereby resolve an unseen response that neither source alone
determines, while preserving established behavior?

[Core abstraction v2](core-abstraction-v2-results.md) establishes donor-dependent
reuse for S1 in 4/10 conditions. In those conditions A's core mask is full and
B identifies the teacher's sign class. That is transfer of B's information, not
resolution requiring two informative sources. In all C2 conditions either
satellite identifies the core alone. C2 transfers and preserves successfully;
its failure is redundancy under the donor-dependence criterion.

These results, the teacher definitions, and the prior query outcomes are known
to the researchers. This design is motivated by them. No v3 subset census or
complementarity computation has been performed during drafting. Held-out means
withheld from acquisition, not historically unknown to researchers.

Do not tune an assignment until C2 passes. Declare and exhaust the finite task
family below, retain all failures, and distinguish a discovered existence
witness from an independently validated learning benchmark. No endogenous
dream rule, new core capacity, or local-learning-rule variant is in scope.

## 2. One relationship, two views

Use the unchanged signed-energy model, teacher vectors S1/C2/S2/T1, canonical
start, domains, and complete reachable-terminal response semantics from the
[v2 contract](core-abstraction-v2-contract.md). S1 and C2 remain primary teachers;
S2 and T1 are specificity controls, not presumed negative for complementarity.

Write a model as `w=(p_A,p_B,c,r)`, with private two-coefficient blocks `p_A,p_B`,
core couplings `c=(c_A,c_B)`, and supplied input-internal coupling `r`. Every
coefficient remains in `{-3,0,3}`. Only `c` changes at core installation.

The shared relationship is the same compatibility contribution in every event:

`E_core(k,a,b) = k^2 - k*(c_A*a+c_B*b)`.

The nine cores have five structural sign classes under `c ~ -c`: no links,
A-only, B-only, matching two-link signs, and opposing two-link signs. Denote
the class set by `Q`. These are structural classes; distinct classes may still
be behaviorally indistinguishable in particular contexts. Do not infer useful
information merely from a reduction in coefficient possibilities.

An **event** is one existing query `x=(e1,e2,D_A,D_B)` under the fully connected
teacher. A's view is its complete marginal `M_A(T,x)`; B's view is `M_B(T,x)`.
Both receive the same query descriptor and event correspondence, but each
receives only its own marginal. Neither receives the other marginal, the joint
set, internal inputs, `k`, or teacher coefficients. Marginals describe all
reachable terminals; they are not two samples from one selected trajectory.

Thus the satellites see different readouts of the **same experimental context**.
For C2 they still respond through different input coordinates and signs. Shared
context does not mean identical stimulation or identical output semantics.

**Supplied alignment:** query identities, input coordinates, mode sign labels,
module identities, and the candidate relation grammar are researcher-defined.
This experiment tests extraction of complementary constraints within that
alignment. It does not discover that two sensory representations denote the
same object, learn a correspondence, or establish semantic abstraction.

## 3. Declared task family and budget

Reuse each of the five frozen splits: 15 axis training queries, 30 within
queries, and 36 mixed queries. For each split enumerate **every nonempty subset**
`J` of its 15 training query IDs. There are `2^15-1 = 32,767` subsets per split.
On a task `J`, both satellites receive their respective marginal on every event
in `J`. They see matched contexts, not alternating disjoint assignments.

The connected budget is `b=|J|` events, or `2b` supplied marginal sets, with
`b=1,...,15`. No mixed query is supplied to either satellite. The same core must
serve all held-out queries after installation. Count events, marginals, set
cardinalities, and work separately.

All subsets are scored. Report results by budget and split, including the full
set `J` and budgets with no witnesses. Do not select training subsets using
held-out scores. Choosing a readable witness after enumeration is permitted
only as evaluator reporting, explicitly labeled an existence witness.

For Stage A there are `4*5*(2^15-1)=655,340` tasks before controls. Stage B repeats
the tasks for the two original isolated-acquisition assignments (40 base runs,
1,310,680 subset tasks). Those assignments affect the private masks only; the
connected events are paired in both conditions. Orders and overlapping subsets
are not independent datasets or estimates of population reliability.

## 4. Stage A — complementary-information ceiling

Use each teacher's true private blocks and `r` as an explicitly oracle-supplied
reference. This isolates whether the present relation family can support the
desired evidence structure; it is not an acquired executable model.

For a fixed private assembly `p`, initialize `C_A=C_B` to all nine cores. An
observation updates only its source's mask:

`C_s <- {c in C_s : M_s((p,c,r),x) = M_s(T,x)}`.

Discard the observation after filtering. Define `Q_s` as the surviving sign
classes and `C_AB=C_A intersect C_B`, with class set `Q_AB`.

**Structural complementarity** on task `J` requires all of:

- `2 <= |Q_A| < 5` and `2 <= |Q_B| < 5`: each source excludes something and
  neither identifies the structural relation alone.
- `|Q_AB|=1`, and its class has both core links nonzero.
- The teacher's class survives. A violation on the true branch is a correctness
  failure, not evidence of contradictory data.

**Behavioral complementarity** additionally requires at least one mixed query
`x` and one recipient `s` such that, for the *same* `(s,x)`:

`Pred_s(C,x) = { M_s((p,c,r),x) : c in C }`,

`|Pred_s(C_A,x)| > 1`, `|Pred_s(C_B,x)| > 1`, and
`Pred_s(C_AB,x) = { M_s(T,x) }`.

Each element of `Pred` is itself a complete response set. Do not replace this
set of predictions with a union of output modes. An empty core mask certifies
nothing. These comparisons belong to the evaluator and never guide filtering.

This predicate rules out both v2 patterns: a full/uninformative source plus an
identifying source, and two individually identifying sources. It also rules
out coefficient ambiguity that has no consequence for the tested response.

Neither single-source tie-break must predict incorrectly: a prior may guess
correctly. The primary question is whether correctness is determined by each
source's information. Report concrete single-source choices and their accuracy
separately; do not manufacture a cooperative advantage with an unlucky tie.

## 5. Capacity versus presentation: declared diagnostic envelope

If matched subsets do not qualify, that alone does not show that this core
family cannot express complementarity. For each teacher/split, independently
enumerate the masks reachable from **all subsets, including empty**, of the 15
allowed A observations, and likewise for B. Combine all attainable mask pairs
whose total marginal budget is at most 30. Their query subsets need not match.

Apply the same structural and behavioral predicates. This is an evaluator-only
envelope over unmatched views of the existing axis training pool, not another
learner or a replacement primary task. Report minimal total marginal budget,
counts, and witnesses, with overlap of the source query sets explicit.

An empty envelope proves only that no allocation of these allowed marginal
observations on that private assembly produces the specified complementarity.
It is not an impossibility theorem for every query, observation type, core
model, or SRI architecture. Do not enlarge the pool after seeing failure.

Archive atomic observation masks and their intersection closure. If each
informative observation already identifies the relevant relation, demonstrate
that implication from these masks: no rearrangement or repetition of those
observations can produce two informative, unresolved sources. More generally,
provide an exact closure certificate for absence, not failed selected searches.

## 6. Stage B — acquired private structure and executable reference

Reproduce v1/v2 isolated acquisition on all 15 original assigned items, for
both assignments and the five frozen orders. Keep the resulting `V_A,V_B`.
This is separate from paired connected events. The total external budget is
15 isolated marginals plus `2b` connected marginals, at most 45; it is not
matched to v2's 30-observation connected arm.

For each `p in V_A × V_B`, retain separate `C_A(p),C_B(p)` masks and update them
only from the corresponding paired observations. Use v2's declared executable
branch rule: first lexicographic `p` with nonempty `C_AB(p)`. If none exists,
retain the original lexicographic private pair with `c0` for defined output and
mark the task inconsistent. The true branch must remain consistent.

Freeze the selected private pair before core installation. Select from its
`C_AB` using v2's minimal-change order: number of nonzero links, then
lexicographic coefficients. Install once; do not update satellites on held-out
queries or choose a branch by its evaluation score.

Apply section 4's complementarity predicates on the selected branch. An
executable pass also requires that branch to equal the teacher's private
blocks; this is evaluator-only. Report compensating wrong-block successes
separately and do not count them toward the target. Report true-private
ceilings and all private-branch diagnostics without choosing a favorable
assembly and calling it the executable learner.

The masks reference global simulations and candidate private assemblies. This
is an exact information reference, not a local structural consolidation law.
No raw external experience store is needed after filtering, but that does not
make the mask representation an established SRI memory mechanism.

## 7. Transfer, preservation, and causal conditions

Use one installed core on all 81 queries. In addition to complementarity,
require all of the following for a full witness:

1. **Transfer:** on the same mixed `(s,x)` witnessing behavioral complementarity,
   the installed response becomes teacher-exact from incorrect at `c0` on the
   same private assembly. That recipient has strictly positive net gain over
   all 36 mixed queries, and neither satellite has any mixed regression.
2. **Acquired connected behavior:** every supplied paired marginal is
   reproduced. Separate this filtering guarantee from old-behavior preservation.
3. **Coupled preservation:** every teacher-correct marginal at `c0` on all 15
   original training queries remains correct. Report the denominator for each
   satellite; a full pass needs at least one such pair per satellite.
4. **Isolated retention:** reproduce the original isolated waking responses in
   their original environments. This is expected from frozen private blocks;
   do not interpret it as active protection against interference.
5. **Core dependence:** on the same witness query, removing both core links,
   or cutting either link separately, loses the new exact response. Reuse the
   same private assembly, evidence, domains, and fresh canonical start.
6. **Both sources matter:** removing either source's constraints restores
   predictive uncertainty on that same response, as defined in section 4.
   Perform this intervention at fixed private assembly. Also rerun each
   source-omitted executable branch rule end to end and report changes in
   private selection separately from core-installation effects.

Record joint response-set accuracy as well as marginals, all within-query
gains/regressions, and v1's cross-context change count as a diagnostic. Do not
require isolated responses to remain unchanged in the coupled context.

Retain v2's NF, input-frozen, fixed-k, and private-block ablations. Core
dependence is primary; survival of the installed-versus-initial contrast under
input freezing is a separately reported stricter route criterion. No control
establishes a unique temporal causal route.

## 8. Required controls

| Control | Purpose and declared interpretation |
| --- | --- |
| No consolidation / unmodified core / core removed | Use `c0` on the same private pair; explicitly report their equivalence here. |
| A only and B only | Report both certified predictions from their masks and v2 minimal-change single-core predictions; do not equate a lucky guess with identification. |
| Duplicate A; duplicate B | Intersect a mask with itself. Repetition cannot add constraints; verify idempotence. |
| Repeated paired observations | One further pass over the same paired set, retaining the same masks. No new labels or distinctions; verify identical state. |
| Memorization only | Store supplied context-tagged marginals and answer unknown for every mixed query. External-history comparator only. |
| All nine fixed cores | Report each prior across all tasks; no best-fixed-core selection per teacher or subset. Count mirror pairs as one class. |
| Full 15-event paired set | Predeclared anchor showing whether extra evidence eliminates the need for combination; redundancy is not failure to learn. |
| Matched-budget unmatched envelope | Section 5: distinguish absence under matched views from absence under any allowed allocation. |

Also report a flat centralized constraint-intersection reference using the
same observations and candidate hypotheses. Its agreement with the two-source
intersection is expected, not a competing architectural success. A pass shows
complementary information and mediated behavioral reuse, not superiority over
ordinary centralized constraint learning.

## 9. Targets and evidence levels

Report four nested outcomes per task: structural complementarity, behavioral
complementarity, preservation-plus-core-dependent transfer on true private
blocks, and executable success satisfying all sections 6–7 conditions.

For each primary teacher, an existence pass requires at least one full witness.
The two-teacher target requires a full executable witness for both S1 and C2.
If only one passes, report a partial result. Do not discard S1 positives to
manufacture a common failure or weaken C2's condition to obtain a joint pass.
Specificity teachers are reported in full but cannot substitute for primaries.

Publish counts and denominators by teacher, split, budget, and private assignment
where applicable; minimal successful budgets; all diagnostic masks; and the
lexicographically first witness only as a readable example. The full event set
may fail complementarity because it is more informative, even while improving
prediction. Do not call that worse learning.

An exhaustive witness search over a frozen task family establishes bounded
existence, not reliability or independently novel generalization evidence.
Stage A alone does not establish acquisition, and even an executable pass does
not establish endogenous dreaming, learned alignment, deeper abstraction, or
preservation during future interference.

## 10. Exact enumeration and correctness gates

Direct subset enumeration is allowed for the 15-query primary pool. An exact
dynamic program may instead process each query once, with include/exclude
transitions, retaining counts by budget and mask state plus canonical subset
witnesses. For Stage B, the state must retain the **whole table** of per-branch
masks; coalescing solely by the true branch can corrupt executable selection.
Coalesce only when all downstream predicates and selector inputs are identical.

For the unmatched envelope, enumerate each source's reachable masks by budget,
then combine mask/budget pairs; multiply multiplicities for independent source
subsets. Do not enumerate only a shortest subset and claim complete budget
counts. Verify compressed enumeration against direct enumeration on the
15-query primary pools and smaller hand-built examples. Counts must sum to
the combinatorial denominators. Resource limits mean incomplete, not absence.

Required checks before scoring:

- Unchanged prior hashes; correct paired query IDs; no mixed or evaluator label
  reaches acquisition, private selection, or the core tie rule.
- Complete response sets against the independent oracle for every response
  used; core-sign symmetry and correct distinction between core masks and
  sets of predicted response sets.
- Teacher-core survival on true private branches; correct empty-mask handling;
  v2 private masks reproduced; order independence and duplicate idempotence.
- Hand-built logical cases for complementary masks, redundant sources,
  uninformative sources, behavioral equivalence despite coefficient ambiguity,
  empty intersections, and lucky single-source guesses. These are test fixtures,
  not claimed positive S1/C2 phenomena.
- Fixed private blocks through core installation; correct same-query link
  cuts; same-branch versus end-to-end source omission kept distinct.
- Persist the installed vector alone, delete raw observations and mask tables,
  reload into a fresh simulator, and reproduce its full response sets.

Record logical and actual mask/table storage, simulation/cache work, acquisition
versus evaluator work, subset-enumeration costs, and memory peaks with their
measurement scope. A cheap installed core does not imply cheap discovery.

## 11. Artifacts, freeze, and stopping rule

Review this draft before implementation. Freeze a final
`docs/shared-core-abstraction-v3-contract.md` and a machine-readable manifest
before any scored subset census. Record known v2 outcomes and the analytical
consequences above. Commit the contract/manifest before scoring; if that order
is not achieved, disclose it rather than presenting hashes as a timestamp proof.

Proposed new files: `src/shared_core_abstraction_v3.py`,
`experiments/shared_core_abstraction_v3.py`,
`tests/test_shared_core_abstraction_v3.py`,
`reports/shared-core-abstraction-v3-{manifest,ceiling,selector,runtime}.json`,
and a results document. Import earlier dynamics without modifying frozen code.
Keep deterministic scientific artifacts separate from volatile runtime metadata.
Run targeted tests and the full suite, and verify byte-identical regeneration.

Complete all primary tasks and the declared envelope, including negative cases.
Stop after the finite census and interpretation. Do not add teachers, expand
the query pool, introduce new readouts, change budgets, enlarge the core, or
retune criteria after inspecting results. A negative complete result is valid.

If complementarity exists, its witnesses become targets for a later separately
specified structural discovery process. If it does not, identify whether the
obstruction is the presentation, allowed observations, candidate relation
family, or executable private acquisition. Any architectural extension needs
a new contract. Dreaming and the previously deferred learning-rule studies
remain out of scope.
