# Shared input interpretation v1 — mechanism contract (no learning)

**Status:** prospective, not implemented. This is a mechanism-feasibility study only.
No learning campaign is in scope. A later learning contract may use a teacher only
if that teacher passes the full gate below. Preserve every earlier contract, code
file and report, including all `learned-reciprocal-participation-v1` artifacts and
`src/reciprocal_participation.py`, unchanged.

**Pre-implementation review (2026-10-23).** This revision clarifies donor versus
recipient input coordinates, adds a feedback-removal control that preserves
input-internal dynamics, and fixes intervention/witness semantics. No teacher
coefficients or v1 gate thresholds were changed. No new teacher was simulated
during this review. This revised file is the implementation authority; record
its hash before evaluation.

## 1. Question and relation to v1

[Learned reciprocal participation v1](learned-reciprocal-participation-v1-results.md)
failed its gate on one condition, feedback necessity. Input revision happened, but
it never changed a response set. In every declared v1 teacher, each input
coordinate couples to exactly one satellite. None of the 225 coefficient vectors
where each input coordinate couples to **at most** one satellite shows an
output-level feedback effect in the declared v1 query space. This is a bounded
finding, not a theorem about arbitrary inputs, penalties or architectures.

The question here is:

> Can feedback from A change a shared input interpretation, which then changes
> B's available responses? Does that causal chain survive the full v1 gate?

There is no router, expert selection, or module dropping. All modules remain in
the dynamics throughout. The v1 failure is kept as a control.

Two candidate mechanisms are declared:

- **S (shared input coordinate).** One input coordinate couples to both A and B.
  This is inside the v1 energy and the 729-vector coefficient family.
- **C (internally coupled input).** Each input coordinate couples to one satellite,
  and the two input coordinates couple to each other through one new term.

## 2. Knowledge at declaration (disclosure)

- **What was known.** The v1 post-gate diagnostic found 312 of 729 v1 coefficient
  vectors whose response set differs from the input-frozen set on some query.
  All 312 have an input coordinate coupled to both satellites.
- **What was not known.** The archived diagnostic reports aggregate counts, not
  the identities of those vectors, and evaluated none against the full gate.
- **How the teachers below were chosen.** By coefficient reasoning about the
  intended chain only. None was run, simulated or looked up before this contract
  was written. Their labels describe coefficients, not expected outcomes.
- **What is not done.** No teacher may be added, replaced, or tuned after any
  evaluation.

## 3. Model

State, domains, evidence, queries (81, in v1 order), canonical start,
strict-descent atomic updates, tie rule, complete reachable-terminal semantics,
caps and truncation are exactly as in v1 §§2–4. Response sets are complete sets Y.
No terminal is ever selected.

For implementation clarity, `z=(i1,i2,k,a,b)`, evidence `e1,e2` and all input/core
coordinates lie in `{-1,0,1}`. A and B each have one of the unchanged domains
`D0={-1,0,1}`, `D1={0,1}`, `D2={-1,0}`. The canonical start is
`(e1,e2,0,0,0)`. Only a single coordinate changes per atomic move; keep every
strictly energy-decreasing move, forbid zero-energy moves, and project all
reachable terminals to `(a,b)` to obtain Y. The full energy is:

```text
E_v1 = 2*((i1-e1)^2 + (i2-e2)^2) + 2*(a^2+b^2) + k^2
       - w_1A*i1*a - w_2A*i2*a - w_1B*i1*b - w_2B*i2*b
       - w_KA*k*a - w_KB*k*b
```

Family S uses the v1 energy unchanged. Family C adds one term:

```text
E_C(z;x) = E_v1(z;x) - w_12 * i1 * i2,      w_12 in {-3, 0, 3}
```

With `w_12 = 0`, `E_C` must equal `E_v1` identically. Test this on all 59,049
v1 (hypothesis, query) cases by comparing reachable sets, terminals and Y with
the frozen v1 implementation. Implement family C in a new module. Do not edit
`src/reciprocal_participation.py`.

Treat S as the new implementation's `w_12=0` case as well as checking it against
v1. A teacher is one fixed coefficient vector; the larger coefficient grid is
used only for correctness verification, not teacher discovery or gate scoring.

## 4. Declared teachers

Coefficient order: `(w_1A, w_2A, w_1B, w_2B, w_KA, w_KB)`, with `w_12` for
family C.

| Teacher | Coefficients | Intended chain (a hypothesis, not a prediction) |
| --- | --- | --- |
| S1 | (3, 3, 3, 0, 3, -3) | A driven by i2; A revises i1; i1 drives B. Complementary core. |
| S2 | (3, 3, 3, 0, 3, 3) | Same input pattern, matching core signs. |
| S3 | (3, 3, -3, 0, 3, -3) | Same, with B reading i1 at the opposite sign. |
| C1 | T1 + `w_12 = 3` | A revises i1; i1 revises i2; i2 drives B. |
| C2 | T1 + `w_12 = -3` | Same chain, opposite internal sign. |
| C3 | T3 + `w_12 = 3` | Crossed: A revises i2; i2 revises i1; i1 drives B. |

Here T1 is `(3,0,0,-3,3,-3)` and T3 is `(0,3,-3,0,-3,3)`. Explicit chain
coordinates for A → input → B are:

| Teachers | Donor coordinate j (receives A feedback) | Recipient coordinate l (influences B) | Chain links |
| --- | --- | --- | --- |
| S1–S3 | i1 | i1 | A–i1, i1–B |
| C1–C2 | i1 | i2 | A–i1, i1–i2, i2–B |
| C3 | i2 | i1 | A–i2, i2–i1, i1–B |

For the reverse chain swap A/B and donor/recipient. The off-chain A–i2 edge in
S remains present in the full-network tests.

The v1 teachers T0–T3 are rerun as **regression controls**. Their conditions 1–7
must reproduce the frozen v1 phase-A report exactly; compare against that file
and do not rewrite it.

**Disclosed confound for C.** Through `w_12`, evidence on one input coordinate
directly biases the other before any satellite moves. That is input-internal
evidence interaction, not satellite feedback. The control and conditions below
separate the two.

## 5. Controls and intervention semantics

Retain all v1 controls, including fully frozen input, disconnected core, fixed
core values, and individual coupling deletion. Add **satellite-feedback-removed
input**, abbreviated NF, for both S and C:

```text
E_I(i1,i2;e) = 2*((i1-e1)^2 + (i2-e2)^2) - w_12*i1*i2
```

In NF, an input-coordinate update uses only the change in E_I. K, A and B still
use their local differences in the full E_C and respond to the current input
state. Thus NF removes A/B → I influence while preserving I → A/B influence and
both directions of the input-internal coupling. It is a directed control; do
not implement it by deleting symmetric energy terms, which would also remove
the forward influence. For S, NF equals fully frozen input from the canonical
start. For C, do not assume that equality.

NF also terminates: for any actual update the pair `(E_I,E_C)` decreases
lexicographically. Input updates strictly decrease E_I; other updates leave E_I
fixed and strictly decrease E_C. This holds on a finite state space and under
the fixed-coordinate interventions below. Use complete terminal sets in NF too.

Every intervention starts a **fresh** exploration. For `do(A=v)`, replace the
canonical starting A coordinate with v and freeze it throughout; leave every
other initial coordinate canonical unless also explicitly clamped. A clamp
value must belong to its unchanged local domain. `do(I=u)` replaces and freezes
both input coordinates at u while evidence stays e. A single-coordinate input
clamp freezes only that coordinate. No trajectory carries over between runs.

An edge ablation sets exactly the named coefficient to zero before exploration;
all other coefficients, domains, evidence, clamps and start rules stay the same.
Full core disconnection removes both core–satellite coefficients. No control
may drop a satellite, its domain constraints, or a difficult terminal outcome.

## 6. Gate

Evaluate all 81 queries for every teacher. Conditions 1–7 are v1 §5 **verbatim**,
including the incompatibility (6), core-contribution (4) and fixed-k requirements.
For family C, "input frozen" fixes both i1 and i2, and "core disconnected" leaves
`w_12` in place. Reproduce v1's legacy input-feedback-removed control only for
regression comparisons; use NF as defined in §5 for new feedback claims.

Two additional conditions:

**3′. Satellite feedback changes B's available responses.** On at least one
canonical query, B's projection `{b : (a,b) in Y}` differs under NF. Keep the
fully frozen comparison as a separate diagnostic. Report active/inactive-set
changes separately from sign-only changes; a difference in either suffices.
This prevents input-internal processing alone from passing feedback necessity.

**8. Causal chain A → input → B.** All parts must hold on a single query x, under
one witness record, with the same pair of A clamp values. Condition 3′ may use a
different query, but report whether its canonical query also supports condition 8.

Notation: `T_v` is the complete terminal-state set under `do(A=v)`;
`U_v^j`, `U_v^l` are its donor and recipient input projections; `B_v` is its B
projection. Enumerate all ordered pairs `v != v'` from D_A, not merely a favorable
preselected pair. The order matters for the fixed-v intervention in (b).

- **(a) A → input.** Require `U_v^j != U_v'^j` and `U_v^l != U_v'^l`. For S
  these are the same test; for C they separately check that the influence reaches
  the recipient coordinate. Archive full terminal states to preserve correlations.
- **(b) recipient input → B.** Choose `r in U_v^l` and `r' in U_v'^l`, with
  `r != r'`. With A fixed at v in both runs, compare `do(i_l=r)` and
  `do(i_l=r')`. Their complete B projections must differ. Other coordinates are
  free. Record terminal witnesses showing where r and r' originated. For C,
  never substitute donor-coordinate values for recipient-coordinate values.
- **(c) Composite effect, carried by satellite feedback into input.** Require
  `B_v != B_v'` in the full dynamics. The contrast must disappear both under
  NF and when both input coordinates are clamped to the same evidence vector e.
  Formally `B_v^NF == B_v'^NF` and `B_v^(I=e) == B_v'^(I=e)`.
  These equalities concern the contrast between A interventions; they do not
  require either controlled set to equal its original full-system counterpart.
- **(d) Individual link necessity for this witness.** Zero each declared chain
  link separately and repeat the same A clamps on x. In every case require
  `B_v^(-edge) == B_v'^(-edge)`. No re-search for another x or clamp pair is
  permitted when validating this witness. Keep all off-chain links intact.
- **(e) Reported, not required.** Repeat the same contrast with the core
  disconnected and with each off-chain coefficient individually zeroed. Also
  clamp I to each of the nine common input vectors u in both A interventions;
  report whether any remaining contrast is a bypass through the core. Only
  u=e is required by (c).

Record all qualifying query/clamp/recipient-value tuples and their count; choose
the lexicographically first only for a readable example. All comparisons are
set comparisons over complete runs, not comparisons of selected trajectories.

Condition 8 establishes an **interventional set-level witness** in this model.
It does not prove a unique temporal route, a natural causal-mediation effect, or
that the input is the only possible route in all contexts. In particular, (d)
deliberately asks for a witness without a redundant bypass. Failure of (d) can
mean parallel pathways, not absence of any input contribution. Report which
subconditions fail rather than equating every gate failure with no interaction.

**Reverse chain B → input → A.** Evaluate it with the same definitions. Report it
fully, including all subconditions; it is not required. Swap source/recipient
roles and use values from the new recipient input coordinate's terminal sets.

**Link-ablation table.** For every teacher and every nonzero coefficient, report
the number of queries whose Y changes when that coefficient alone is zeroed.
Report the input-frozen result alongside it.
Also report NF, with changes to Y, A/B projections and participation signatures
in separate columns. A changed joint correlation need not change either marginal.

**Gate rule.** A single teacher among S1–S3 and C1–C3 must pass 1–7, 3′ and 8.
Report families S and C separately. If none passes, the status is
`chain_refuted_for_declared_teachers` and the study stops. Do not add teachers,
change penalties or scales, add coupling terms, or relax any condition.

## 7. Verification, proofs and outputs

- **Independent oracle.** Build it from the written energy formulas, without
  production helpers. Check family C on all 81 queries × 2,187 coefficient
  vectors, and family S as in v1. Repeat the v1 controlled tests for the new
  module. The 177,147 C cases include the 59,049 zero-internal-coupling S cases;
  do not present the latter as additional independent coverage. Compare reachable
  states, full terminals and Y, not just projected outputs. For the six declared
  teachers, additionally compare every intervention and ablation used to score
  conditions 3′ and 8 against the oracle, including NF's directed update rule.
- **Proofs.** Prove termination and acyclicity for `E_C`, and show that
  `w_12 = 0` reduces to v1. For family C, the local fields of i1 and i2 gain
  `w_12·i2` and `w_12·i1` respectively; prove that every chain-link dependence in
  the energy is present in the corresponding local energy difference. A nonzero
  coefficient does not guarantee a reachable transition crosses its strict
  decrease threshold: distinguish algebraic dependence, reachable witnesses and
  output-level effects. Prove NF's lexicographic termination separately, and
  prove reachable-terminal completeness and domain/evidence preservation.
- **Failure certificates.** Give one for every failed condition, as in v1. For
  condition 8 archive bounded exhaustive candidate counts and subcondition masks
  for each query/A-clamp pair and recipient-value pair, so failure is independently
  checkable. Do not stop evaluating a teacher at its first failed condition.
- **Meaningful fixtures.** Check that NF retains internal input coupling, that
  deleting a symmetric edge differs in definition from removing only feedback,
  that all clamps are within domain and retain immutable evidence, and that
  donor/recipient projections are not interchanged for C. Verify witness-link
  ablations reuse the original query and clamp pair. Check empty projections,
  multiple terminals, ties, zero/exact-boundary caps, and hypothesis/control
  validation. If a requested positive phenomenon is absent, archive its exhaustive
  absence rather than invent a passing fixture. Full uncapped finite runs always
  have at least one terminal; an empty terminal set is an implementation error.
- **Reproducibility.** Require byte-identical regeneration into a temporary
  directory. Record source and contract hashes, interpreter, git revision and
  dirty status. Volatile timings go in a separate runtime file.
  Start timing before preprocessing. Store the six teacher vectors, query order,
  intervention enumeration rules and contract hash in a manifest before scoring.
  Exclude timestamps from deterministic scientific artifacts. Controls must be
  compared with frozen v1 scientific values, excluding volatile provenance.
- **Deliverables.**
  - `src/shared_input_mediation.py`
  - `experiments/shared_input_mediation.py`
  - `tests/test_shared_input_mediation.py`
  - `reports/shared-input-mediation-v1-manifest.json`
  - `reports/shared-input-mediation-v1-gate.json` and `-runtime.json`
  - `docs/shared-input-mediation-v1-results.md`
  - README and open-problems entries, written only after validation.

Implementation order: freeze the manifest; implement dynamics and controls;
validate against the independent oracle and v1 regressions; enumerate all teacher
conditions; regenerate in a temporary directory and compare scientific output;
run the full existing test suite; write the scoped results. Report status
`mechanism_witness_found_for_declared_teachers` if any teacher passes, otherwise
`chain_refuted_for_declared_teachers`. In either case stop: no learning run or new
teacher search is authorized by a successful mechanism gate.

Suggested reproduction interface:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_shared_input_mediation.py' -v
.venv/bin/python -m experiments.shared_input_mediation
.venv/bin/python -m experiments.shared_input_mediation --out-dir /tmp/shared-input-v1-regen
.venv/bin/python -m unittest discover -s tests -v
```

## 8. Scope

Mode-level signed-energy candidate only; two input coordinates; six declared
teachers. A pass establishes the specified interventional witness inside this
candidate. A failure refutes the conjunction for these six teachers and these
interventions, not all 2,187 coefficient vectors or shared-input mediation in
general. The two families have only three hand-selected teachers each; their pass
counts cannot establish that one family is generally superior. This study does not
establish learnability, robustness, recovery or semantics. It also does not show
that energy descent is SRI's foundation. A general transition table could
reproduce any finite result here. Learning, any teacher search, and other
input-coupling forms are out of scope.
