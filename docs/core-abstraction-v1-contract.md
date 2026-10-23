# Acquired core abstraction in S1/C2 — v1 contract

**Status:** prospective; frozen for implementation on 2026-10-23. This revision
replaces the separate Boolean construction with the existing SRI signed-energy
model. S1 and C2 are teachers; A and B are its two satellites. Freeze this
contract's hash in the manifest before any scored computation.

**Pre-implementation review (2026-10-23).** This revision makes the waking
environments, qualification predicates and witness controls exact (sections
4–6), records three consequences of the setup that follow analytically before
any computation (core-sign symmetry; a witness needs both core links; every
Stage-B information class contains all nine core values), and fixes the Stage-B
computation scope (section 7). Teachers, splits, queries, targets and the core
family are unchanged. No Stage-A or Stage-B quantity was computed during review.

## 1. Priority and scope

First ask whether the existing substrate can express a reusable core relation
that enables cross-satellite transfer while preserving acquired behavior. Then
ask what information in learned satellite structure could select that relation.
An endogenous dream discovery rule comes later.

The preservation-signal, plateau-move, and larger-edit investigations from
[local update diagnosis v1](local-update-diagnosis-v1-results.md) remain deferred.
Do not implement them as part of this study or require the waking local learner
to solve preservation independently.

Separate three claims:

1. **Representability:** a core-only change exists with the required behavior.
2. **Acquisition:** a declared structure-only procedure selects such a change
   without evaluator labels or an experience store.
3. **Endogenous discovery:** internal excitation and structural reorganization
   discover it under a declared budget.

The first implementation should establish the representability and information
ceilings below. It must not rename an evaluator-selected core an acquired
abstraction. The draft intentionally does not invent a dream algorithm to bridge
an information or representation gap.

## 2. Preserve the existing model

Use `src/shared_input_mediation.py` and the existing independent oracle unchanged.
State is `z=(i1,i2,k,a,b)`. Input and core coordinates are ternary; satellite
coordinates retain the existing domains `D0`, `D1`, `D2`. A query specifies
immutable evidence `(e1,e2)` and the two domain indices. Keep all 81 queries,
canonical start `(e1,e2,0,0,0)`, and complete reachable-terminal semantics.

Write the persistent coefficients as

`p_A=(w_1A,w_2A)`, `p_B=(w_1B,w_2B)`, `c=(w_KA,w_KB)`, `r=w_12`.

Every coefficient belongs to `{-3,0,3}`. The energy remains

```text
E = 2*((i1-e1)^2 + (i2-e2)^2) + 2*(a^2+b^2) + k^2
    - w_1A*i1*a - w_2A*i2*a - w_1B*i1*b - w_2B*i2*b
    - w_KA*k*a - w_KB*k*b - r*i1*i2.
```

Explore every strictly descending single-coordinate move from the canonical
start, with unchanged tie semantics. Project all reachable terminals to obtain
`Y={(a,b)}` and its marginals `Y_A`, `Y_B`. Never choose one favorable terminal.
A response set with several modes can be exactly correct; ambiguity among
candidate models is a different issue.

Use the existing teacher definitions, in the original seven-coefficient order:

| Teacher | Coefficients | Role |
| --- | --- | --- |
| S1 | `(3,3,3,0,3,-3,0)` | Primary |
| C2 | `(3,0,0,-3,3,-3,-3)` | Primary |
| S2 | `(3,3,3,0,3,3,0)` | Specificity/control teacher |
| T1 | `(3,0,0,-3,3,-3,0)` | Specificity/control teacher |

S2/T1 are negative controls for the previously declared forward input-mediated
chain, not automatically negative for every new transfer criterion. Score their
actual new behavior; do not impose an unproved absence of core transfer.

## 3. What a core abstraction could mean here

Only `c` may change during consolidation. Both satellite blocks and `r` must
remain byte-identical. The transient value `k` is not persistent learning.
There are nine candidate core-coupling pairs; no new core nodes, potentials,
lookup tables, or input couplings are introduced.

The candidate relation is the reusable compatibility contribution

`E_core(k,a,b)=k^2-k*(c_A*a+c_B*b)`.

For descriptive analysis only, minimizing over `k` yields

`Phi_c(a,b)=min(0, 1-|c_A*a+c_B*b|)`.

This describes a context-independent relation between satellite modes. It does
not replace reachable-terminal dynamics: minimizing out `k` need not preserve
reachable paths or response sets. The candidate core family is supplied by the
researcher; its useful member and usefulness must not be assumed.

**Core-sign symmetry.** The energy is invariant under `(k, c) -> (-k, -c)` and
the canonical start has `k=0`, so `c` and `-c` give identical response sets under
every intervention used here except a nonzero `k` clamp (which maps to the
opposite clamp). The nine cores therefore form five behavioral classes:
`(0,0)`, `±(3,0)`, `±(0,3)`, `±(3,3)`, `±(3,-3)`. Verify this exhaustively; report
qualifying cores in full, and never count `c` and `-c` as independent successes.

Call a result a **candidate reusable core relation** only when one fixed `c`
serves all tested queries in a run, preserves acquired responses, and improves
untrained combinations under the controls below. Merely recovering the known
teacher couplings, changing `k`, or lowering energy is insufficient. This very
small family may be incapable of the intended abstraction.

## 4. Controlled satellite acquisition

Use a deliberately transparent construction to separate available satellite
knowledge from an optimized waking rule. This is a new controlled acquisition
protocol, not a rerun of IB and not a claim about the previous B0 learner.

Fix `r` as supplied input infrastructure: `0` for S1/S2/T1 and `-3` for C2.
Report that this gives C2 a known input-internal coupling; it is not learned
core knowledge and must never receive abstraction credit.

For satellite A's waking environment, disconnect the core and zero B's two
input couplings; leave B's coordinate and domain in the dynamics. Reverse the
roles for B. Keep `r` and all other semantics unchanged. The teacher supplies
only the addressed satellite's complete output marginal in that environment.
Thus A acquires its own input-response relationship and B acquires a distinct
one, rather than receiving joint teacher labels or each other's targets.

Exactly, for teacher `T` with blocks `p_A^T`, `p_B^T` and `r`, A's waking vector
for candidate block `p` is `(p_1, p_2, 0, 0, 0, 0, r)` and B's is
`(0, 0, p_1, p_2, 0, 0, r)`. The waking target for A on query `x` is the A-marginal
of the full reachable-terminal set of `explore(x, (p_A^T, 0, 0, 0, 0, r))`, and
symmetrically for B. In A's environment `k` and `b` receive zero field and stay
at `0`; they remain in the dynamics as declared. Waking marginals therefore depend
only on the addressed satellite's block and `r`.

For each of the five existing frozen splits, take its 15 training query IDs in
ascending order. Assign alternating IDs to A and B (8 and 7), and run the swapped
assignment as a second condition. This assignment uses no labels. Both inherit
the original query's domain pair. Use the five frozen orders restricted to each
assignment. Reuse the 30 within and 36 mixed held-out queries unchanged; no
mixed-evidence query may be supplied to either satellite during wake.

For each satellite maintain the set `V_i` of its nine possible two-coefficient
blocks. Initially all nine survive. On its own observation, retain precisely
those blocks reproducing that complete marginal in its waking environment.
Keep a nine-bit candidate mask, not an observation list. This is an exact finite
acquisition reference, not a local plasticity or dreaming mechanism. Blocks are
indexed lexicographically over `(p_1, p_2)` with `-3 < 0 < 3`; assemblies are
ordered by `(index(p_A), index(p_B))`. Run the filter in each of the five frozen
orders restricted to the assignment and verify identical final masks; the
analysis uses that common mask once per (teacher, split, assignment) run. The
teacher's own blocks always survive, so an empty mask is an implementation error.

The evaluator verifies realizability and records uncertainty. After wake,
consider every pair in `V_A × V_B`, at most 81 pairs, as an executable candidate
assembly with initial core `c0=(0,0)`. Do not pick the true private blocks using
teacher identity. A realizable empty version set is a correctness failure.
Order-equivalent final sets are not independent trials.

For each assembly, separately check whether joining the learned satellites at
`c0` preserves their waking responses. Shared input already couples them even
without the core, so this can fail. Such a case is an **assembly interference**
result, not successful acquisition plus preservation. A later core may repair
it, but that is a separate outcome from preserving a correct assembled model.

## 5. Stage A — exact representability ceiling

For every teacher, split, assignment, and surviving private-block pair, enumerate
all nine `c` values. Keep a single `c` fixed across the complete evaluation set.
All candidate scoring is evaluator-only; this is a ceiling search, not learning.

For each candidate, record full joint and marginal response sets on all 81
queries, plus the controls below. Assess two kinds of old behavior explicitly:

- **Original-context retention:** reproduce every waking marginal in its
  original isolated environment. This is a regression check and can be trivial
  because that environment disconnects the core.
- **Retention on assembly:** reproduce those same previously acquired marginals
  when both satellites and the candidate core are connected. This is the
  substantive preservation requirement. It may conflict with joint teacher
  behavior; report the conflict rather than silently changing the target.

Held-out transfer targets are the unchanged full S1/C2 teacher marginals under
ordinary coupled inference. Report joint `Y` accuracy alongside marginals, since
matching both marginals need not preserve joint correlations.

The primary transfer partition is all 36 mixed-evidence queries: their evidence
combinations were supplied to neither satellite. The 30 within queries are a
secondary reuse test. Do not select a successful subset before scoring.

A qualifying candidate must:

1. Start from an assembly that reproduces every waking item at `c0`.
2. Preserve all those acquired marginals after the core-only change, with zero
   losses. Also preserve every teacher-correct marginal at `c0` on the 15
   training queries, including ones not supplied to that satellite.
3. Turn at least one previously incorrect mixed-evidence marginal into the exact
   teacher set, with a strictly positive net gain for that recipient over all
   36 mixed queries. Report all gains and regressions in both satellites.
4. Meet the core dependence and route-specificity conditions in section 6 on
   at least one improved query. The same witness must satisfy all conditions.
5. Use unchanged private blocks and input infrastructure for every query, with
   no additional target labels or satellite updates.

### 5.1 Exact predicates

For one run, assembly `p=(p_A,p_B)` and core `c`, let `w(c)=(p_A,p_B,c,r)` and
`M_s(w,x)` be satellite `s`'s marginal of the full response set. `W_s(x)` is the
waking target of `s` on its own training query `x`; `T_s(x)` and `T_Y(x)` are the
teacher's coupled marginal and joint set. `c0=(0,0)`.

- **Assembled retention** `R(c)`: `M_A(w(c),x)=W_A(x)` for every A training query
  and `M_B(w(c),x)=W_B(x)` for every B training query. Original-context retention
  is the same comparison in the waking vectors and must hold by construction.
- **Assembly interference:** `not R(c0)`.
- **Training preservation** `P(c)`: `R(c)`, and every (satellite, training query)
  pair with `M_s(w(c0),x)=T_s(x)` also has `M_s(w(c),x)=T_s(x)`. `P(c0)` is `R(c0)`.
- **Mixed gain** for recipient `s`: `G_s(c)` = mixed queries correct under `c` and
  not under `c0`; `L_s(c)` = the reverse; net gain `|G_s|-|L_s|`.
- **Transfer** `X(c)`: `c != c0` and some recipient `s` has nonempty `G_s(c)` and
  positive net gain.
- **Core-dependent witness** `(s,x)` with `x in G_s(c)` for such a recipient:
  `M_s` with `c_A` set to zero and with `c_B` set to zero both differ from `T_s(x)`.
  Core removal is `c0` and fails by definition of a gain, so it adds nothing. A
  cut that leaves `c` unchanged cannot remove the gain; hence every witness has
  `c_A != 0` and `c_B != 0`, and only `±(3,3)`, `±(3,-3)` can qualify.
- **Route-specific witness:** additionally, with both inputs clamped to evidence,
  `M_s(w(c),x) != M_s(w(c0),x)`.
- **Qualifying core:** `R(c0)`, `P(c)`, `X(c)` and a route-specific witness.
  **Route-unresolved core:** the same with a core-dependent witness but no
  route-specific one.

Per assembly, report exactly one outcome: `assembly_interference`;
`no_transfer_core` (no `c` satisfies `P` and `X`); `no_core_dependent_witness`;
`route_unresolved_only`; or `qualifying`. Also report `no_preserving_core`
(only `c0` satisfies `P`) and transfer cores that fail preservation, separately.

Teacher blocks always survive and the teacher core (or its sign mirror) is in
the family, so the true assembly can reproduce the teacher everywhere. Report
for every qualifying core whether it is in the teacher's sign class and whether
the assembly is the teacher's own; a qualification achieved only by the true
assembly with the teacher's core is recovery of the known coupling, not more.
Within-query gains and regressions are secondary; also count qualifying cores
with no within-query regression.

For each run, report the complete set of qualifying cores per private-block
pair. Report **possible** success if some surviving assembly qualifies and
**robust** success if every surviving assembly has a qualifying core. These
quantifiers do not imply that one structure-only selector can find a core.

An existence result requires at least one qualifying assembly for each primary
teacher; report every split/assignment and the stronger robust counts. A result
for only one teacher is partial. These are small-witness targets, not reliable
learning thresholds. Report acquisition failure, assembly interference,
no-preserving-core, and no-transfer-core separately, with bounded certificates.
Do not add teachers, coefficients, data, or new core capacity after a failure.

## 6. Controls and route specificity

Evaluate each control from a fresh canonical start, reusing the same query,
private blocks, and witness. Keep domains and evidence unchanged.

| Control | Definition |
| --- | --- |
| Unmodified core | Use `c0=(0,0)` with the assembled private blocks. |
| Core removal | Set both learned core couplings to zero after consolidation; leave `r` intact. This is equivalent to unmodified core here. |
| No consolidation | Retain post-wake private state and `c0`; no further parameter changes. |
| Ordinary repetition | Re-present the same waking feedback to the exact candidate filters for one additional pass; core remains zero. External history is allowed only for this explicitly labeled comparator. Filtering is expected to be idempotent. |
| Memorization only | Store supplied `(satellite, query, waking-environment, marginal)` entries; answer unknown for any unobserved key. It cannot answer a new coupled context by silently reusing an isolated label. |
| Fixed cores | Evaluate all nine fixed pairs across all tasks, without task-specific selection. A universally successful fixed core weakens the acquisition claim. |
| Core fixed in activity | Clamp `k` to each of `-1,0,1`, using the existing intervention semantics. Report whether transient core response is necessary. |
| Individual core-link cuts | Zero each of `c_A`, `c_B` separately on the same improved query. A mediated witness must lose its gain under each cut. |
| Satellite-feedback-removed input | Apply the unchanged NF control, retaining input-internal coupling. Compare learned versus initial core under the same NF dynamics. |
| Input frozen | Clamp input interpretation to evidence in both learned-core and initial-core runs. Compare their recipient response sets and report teacher accuracy separately. |
| Private-structure ablations | Zero each satellite input block in turn, keeping the acquired core fixed. Report which private information the behavior still requires. |

A primary core-mediated witness must lose its gain under core removal and both
individual core-link cuts. In addition, the learned versus initial-core recipient
contrast must persist with input interpretation frozen to evidence. The latter
is a conservative test for a contribution beyond the already-known revisable
input route; it need not remain teacher-correct under that intervention.
If it fails, report **core-dependent but route-unresolved transfer**, not a pure
core-mediated witness. NF and fixed-activity outcomes are additional diagnostics.
Run the per-witness diagnostics (fixed activity, NF, input frozen, private
ablations) on every witness of every qualifying or route-unresolved core, for
both `c` and `c0`, and archive the recipient marginals and teacher correctness.
No consolidation and core removal equal `c0`; ordinary repetition must leave
every mask unchanged (verify) with core `c0`; memorization answers no coupled
query and scores zero on every assembled partition. Report these as such rather
than recomputing identical sets under new names.

Do not require removing the donor satellite after consolidation to leave the
response unchanged: this substrate is a coupled system with persistent donor
participation. Instead distinguish donor structural information used during
consolidation from donor transient activity during inference. A future selector
must be tested with donor structure withheld or replaced; Stage A alone cannot
establish acquired donor-to-core information transfer.

## 7. Stage B — information ceiling before a selector

For the permitted acquisition protocol, group possible environments by identical
final learned state `(V_A,V_B,r)`. A structure-only consolidation procedure must
produce the same output on identical states, regardless of teacher name or
hidden held-out targets. No evaluator-chosen private blocks or candidate cores
may be smuggled into its input.

Compute these equivalence classes over the existing 2,187-vector space, not just
S1/C2, using the same waking intervention definitions and split assignments.
Compare their Stage-A success requirements on the declared evaluation tasks.
Archive indistinguishable environments that require incompatible core choices,
and distinguish ambiguity in private blocks from ambiguity in core relation.
This is evaluator analysis, not an expanded training teacher grid.

There is an important analytical warning already visible from the protocol:
waking data with both core links removed cannot identify their original values.
S1 and S2 have identical private blocks and `r`, so their supplied waking data
and learned states are identical. Whether a single replacement core can satisfy
the new transfer/preservation criteria for both still requires analysis; do not
infer that incompatibility merely from different teacher coefficients.

**Class structure (analytic, to be verified).** A block survives a run's filter
iff it reproduces every waking target, so any teacher whose own blocks lie in
`V_A`, `V_B` produces the same masks. Waking data never involve `c`. Hence every
information class is exactly `V_A × V_B × (all nine cores) × {r}`: each class
contains the zero-core teacher with the same private blocks, whose true assembly
already equals the teacher at `c0` and so can gain nothing. No core choice can be
justified for all members without a declared prior over teacher cores.

**Declared Stage-B computation.** For every split and assignment, compute
`(V_A,V_B,r)` for all 2,187 vectors and the class census (count, sizes, private
mask sizes). For every member `T'` of every class and every assembly in that
class, compute the Stage-A outcome and qualifying cores against `T'`'s targets
(exhaustive; no truncation). Report per class and assembly:

- members with any qualifying core, and the intersection of their qualifying
  sets (an **information-compatible core** if nonempty);
- incompatibility certificates: two members with nonempty, disjoint qualifying
  sets on the same assembly (private blocks identical → core-relation ambiguity;
  different → private-block ambiguity also present);
- for the classes containing S1, C2, S2 and T1, the full 9×9 matrix of member
  core × candidate core qualification on each assembly, which states exactly the
  prior over teacher cores a fixed core would need.

Consequently, a successful evaluator-found core establishes representability,
not acquisition from this isolated information. If the information is
insufficient, report exactly what prior or additional observation would be
needed. Do not feed teacher couplings, held-out labels, B8 witnesses, or archived
histories to the target mechanism. An inductive bias can guess correctly but
must be declared as such.

Stop here for this first study. A structure-only extraction rule needs its own
prospective specification after these ceilings, including how it handles
uncertain private blocks. This staged boundary avoids implementing a selection
rule whose success was determined by the evaluator's answers.

## 8. Verification, accounting, and handoff

The implementing agent should review and freeze a final contract and manifest
before scored computation. Record all prior artifact hashes, teacher vectors,
query assignments, candidate ordering, observation semantics, interventions,
and the precise witness predicates. Preserve all earlier code and artifacts.

Required checks include complete terminal sets against the independent oracle,
correct projection and clamp semantics, core-sign symmetry, exact filter
realizability/order independence, isolated versus assembled context separation,
no held-out access by acquisition, and immutable private blocks during core
search. Existence failures need exhaustive certificates over all nine cores;
truncation is incomplete, never proof of absence. Verify the proposed analytical
core potential without using it as an inference shortcut.

Store raw response sets, private survivor sets, all candidate-core scores,
witness controls, and information-equivalence certificates. Count acquisition,
ceiling search, and evaluation work separately. Report private candidate-mask
storage, executable coefficient storage, temporary search state, and external
comparator memory. Do not charge exhaustive evaluator work to a hypothetical
dream learner or claim cheap discovery from a cheap nine-core representation.

New artifacts: `src/core_abstraction.py`, `experiments/core_abstraction.py`,
`tests/test_core_abstraction.py`, this contract as
`docs/core-abstraction-v1-contract.md`, reports
`reports/core-abstraction-v1-{manifest,ceiling,information,runtime}.json`, and a
results document. Use repository unittest conventions and standard-library
reference code; require deterministic scientific regeneration. Completion status
(`complete`, `stopped_failed_correctness_gate`, `stopped_failed_tests`) is
separate from scientific pass, partial, or obstruction outcomes.

The correctness gate before any scoring: every response set used by Stage A for
the four teachers matches the independent oracle; full-response core-sign
symmetry holds on all 2,187 vectors and 81 queries; waking filters are order
independent and contain the teacher's blocks; original-context retention holds
for every surviving block; and the prior-artifact hashes match. The scope is the
specified finite ceiling study, not a full dream campaign.

## 9. Subsequent endogenous dreaming — deferred

If the substrate admits the intended behavior, use its witnesses and failures
to specify a later `Q_Theta` and structural reorganization rule. The target state
may include acquired satellite structure, a persistent core, and bounded
transient randomness, but no permanent external experience store. Explicitly
identify where any preservation information survives in that structure.

That future study must distinguish internal repetition from discovery, compare
with no dream, wake-only, random-change, and history-access controls, and measure
false self-reinforcement. It should test preservation under subsequent
interfering learning; the present existence ceiling cannot show that dreaming
solves forgetting. No process can recover distinctions absent from its retained
state without additional evidence or assumptions.

The earlier Boolean relation construction is not an implementation target in
this revision. Success here must be demonstrated in the unchanged S1/C2 substrate,
or its obstruction stated before proposing an architectural extension.
