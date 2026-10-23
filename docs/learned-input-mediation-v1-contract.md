# Learned input mediation v1 — learning contract (outputs and mechanism)

**Status:** prospective, not implemented. Preserve every earlier contract, code
file and report unchanged. That includes `src/shared_input_mediation.py`, its
frozen manifest and gate report, and the frozen v1 split manifest.

**Pre-implementation review (2026-10-23).** This revision defines Arm I's learning
rules, direction-specific mechanism controls, query-scoped battery equivalence,
and identification ceilings for both arms. No teacher, hypothesis space, split,
or numerical target was changed. No learning or new mechanism evaluation was
run during the review. Freeze this contract's hash before computation.

## 1. Question

> Can observations teach couplings that recover both unseen response sets and the
> demonstrated input-mediated influence?

Matching a teacher's ordinary responses does not show *how* it produces them.
Agreement on canonical response sets need not imply agreement under feedback
removal, link removal or clamps. Whether that ambiguity actually occurs for these
teachers is a question for the ceiling calculation, not an assumed result. This study
therefore scores learned candidates on two things: held-out outputs, and a
**held-out intervention battery** that the observational learner never receives.
If several candidates fit the observations but disagree on that battery, the
result is **mechanism uncertainty**, and it must be reported as such.

## 2. Knowledge at declaration (disclosure)

- The [shared-input gate](shared-input-mediation-v1-results.md) was run. S1 and C2
  pass it; S2 fails through a core bypass. v1 teacher T1 shows no output-level
  feedback effect. All gate-relevant teacher behavior is therefore known to the
  researchers. "Held out" below means **withheld from the learner**, not unseen
  by the researchers.
- No learning campaign has been run on these teachers. Earlier correctness tests
  did simulate non-teacher hypotheses and some controls. Those tests were not a
  census of mechanism equivalence or a learning campaign. Do not claim the whole
  intervention space is previously unexamined.
- The split manifest is the frozen `reports/learned-reciprocal-participation-v1-splits.json`.
  It was frozen under the v1 contract and never used for learning, because v1
  phase B did not run. It is reused unchanged; no new randomness is introduced.

## 3. Model, hypothesis space, teachers

Dynamics, controls and intervention semantics are exactly those of
`src/shared_input_mediation.py` and the shared-input contract §§3 and 5: the
`E_C` energy, strict-descent atomic updates, complete terminal sets, NF,
fresh-run clamps, and named-coefficient ablations. Import them; do not modify
them.

The hypothesis space is all **2,187** seven-coefficient vectors `(w_1A, w_2A,
w_1B, w_2B, w_KA, w_KB, w_12)` in `{-3,0,3}`, indexed lexicographically. It is the
same for every teacher. The learner is not told which family, S or C, a teacher
belongs to.

| Teacher | Vector | Role |
| --- | --- | --- |
| S1 | (3, 3, 3, 0, 3, -3, 0) | Primary: passed the full gate, shared input |
| C2 | (3, 0, 0, -3, 3, -3, -3) | Second mechanism: passed the full gate, internal input coupling |
| S2 | (3, 3, 3, 0, 3, 3, 0) | Negative control for the forward shared-input chain and full gate; its reverse chain has witnesses |
| T1 | (3, 0, 0, -3, 3, -3, 0) | Negative control: no output-level feedback effect (v1) |

No teacher may be added, removed or replaced after any evaluation.

Coefficient recovery is not the target. In particular, flipping k's sign and
both core-edge coefficients can give identical battery observations; this battery
does not observe k. Distinguish representational symmetry from unresolved
differences in the tested mechanism.

## 4. Intervention battery (evaluation only)

For a hypothesis h and query x, the battery records complete response objects.
A battery **atom** has a stable key identifying its item, query and intervention
arguments, and a set-valued result or a Boolean result. Never merge outputs from
different interventions into one untagged set:

1. **B1** canonical Y. This is the observational output, used for training and
   for output evaluation.
2. **B2** Y under NF.
3. **B3** Y with the input frozen.
4. **B4** Y with the core disconnected.
5. **B5** Y with each of the seven named coefficients zeroed alone (a no-op when
   h already has that coefficient at zero).
6. **B6** for each `v` in D_A, the terminal set under `do(A=v)`, projected to
   `(i1, i2, b)`.
7. **B7** for each `v` in D_B, the terminal set under `do(B=v)`, projected to
   `(i1, i2, a)`.
8. **B8** re-evaluate condition 8(a)–(d), without the report-only (e), on the
   same tuple `(query,v,v',r,r')`, using the frozen shared-input definitions.
   A tuple whose recipient values are absent from h's required terminal
   projections scores false; do not substitute new values or skip it.

Use exactly these four directed chain templates for all hypotheses and teachers:

| Template | Source → target | Donor / recipient | Chain links |
| --- | --- | --- | --- |
| SF | A → B | i1 / i1 | w_1A, w_1B |
| SR | B → A | i1 / i1 | w_1B, w_1A |
| CF | A → B | i1 / i2 | w_1A, w_12, w_2B |
| CR | B → A | i2 / i1 | w_2B, w_12, w_1A |

For every template/query, enumerate all ordered source clamp pairs in its local
domain and all distinct recipient values in `{-1,0,1}`. Define a Boolean B8
**existence atom** stating whether any tuple qualifies on that query. This
enumeration is within each fixed hypothesis; it is not a search for new teachers.

Also define a shared set of B8 **fixed-witness atoms**: the union of all qualifying
tuples of the four teachers under those templates. Compute it before learning,
and compare its native-template entries against the archived shared-input
results (S1/S2: SF/SR; C2: CF/CR). T1 has no archived condition-8 result, so compute
its entries here rather than describing an empty archive as a negative proof.
Evaluate every hypothesis on every atom in this common union, including tuples
originating from other teachers, and include false as well as true teacher labels.
Keep this evaluator-only atom schema fixed for both arms and all checkpoints.

Derive "has any forward witness" as the OR of the SF/CF existence atoms on the
specified query set; report reverse SR/CR separately. Do not call S2 negative
for every direction: its archived SR result is positive.

Two hypotheses are **mechanism-equivalent** on a query set Q when all B2–B7 atoms
and all B8 atoms indexed by queries in Q agree. A global existence flag is derived
from those atoms, not silently added to a partition-specific comparison. The
complete battery equivalence used in target 2 uses all 81 queries. Two hypotheses
are **observationally equivalent** on Q when
B1 agrees on every query in Q.

Compute the battery once for all 2,187 hypotheses on all 81 queries. Keep teacher
labels, all held-out outcomes and B8 records inaccessible to learners. Learners
may simulate their own hypotheses on the current training query using only the
observation/intervention types supplied in their arm; see §6.

Verify every battery entry for the four teachers and for the 50 indices
`{44*j : j=0,...,49}` against the independent shared-input oracle. The largest
sample index is 2156. For B8, independently reconstruct subcondition truth from
oracle terminal sets, not the production chain scorer. In the oracle sample,
compare exhaustive qualifying-tuple sets, not only the existence Boolean.

Cache compact response masks, projections and Boolean atoms, not millions of
full reachable graphs or path records. Reuse pure simulations across intervention
atoms and stream by hypothesis if needed. Record unique inference requests,
cache hits/misses, elapsed time and peak memory. A resource-limited computation
must report `incomplete`, not false witness absence or a successful certificate;
do not shrink the grid or sample away uncomputed atoms. Compressed full battery
artifacts or a deterministic reconstruction interface must support audit of the
per-hypothesis hashes.

## 5. Identification ceiling (computed before learning)

Compute two partitions of the 2,187 hypotheses before learning:

- Arm O's observational signature: all B1 atoms on all 81 queries.
- Arm I's augmented signature: all B1, B6 and B7 atoms on all 81 queries.

For each teacher and each partition, report:

- the teacher's complete-information equivalence class and its size;
- how many mechanism-distinct subclasses it contains, and which battery items
  separate them;
- whether any member gets B8 wrong, separated by direction and template;
- explicit indistinguishable hypothesis pairs and differing battery atoms when
  a class is mechanism-heterogeneous. Prefer the teacher plus a differing member.

If an arm's class is not mechanism-homogeneous, its supplied information cannot
uniquely identify the teacher's battery mechanism within this hypothesis space,
even with all 81 queries. A sound full version-space learner cannot certify it.
A selected model might guess correctly using a prior; that is not identification
from these observations. This is a realizable-class identifiability result,
not a statement that every possible learning algorithm must predict incorrectly.
Report both ceilings before any learning curve. Also report ceiling-based
obstructions to each negative-control target below.

## 6. Protocol and learners

Use the v1 frozen splits as they stand: 15 training queries on axis evidence,
30 within-context held-out queries, and 36 mixed-evidence held-out queries. There
are 5 split seeds × 5 order seeds, with checkpoints at 0, 5, 10 and 15. Orders
share their final version space and must not be counted as independent datasets.

Check query identities, domain order and evidence order against the frozen split
file; import its IDs and orders rather than resampling. Keep evaluator histories
outside learner state. For order-dependent local learning, report all 25 streams
but group them by their five underlying splits.

**Arm O (observational, primary).** Feedback for a training query is its complete
`Y_T(x)` (B1) only. The learners are:

- **full-feedback version space** over 2,187 bits: keep h with `Y_h(x) == f`. An
  empty V is `inconsistent_feedback`; do not reset.
- **positive-only version space:** keep h with `f ⊆ Y_h(x)`.
- **block-local learner:** v1 §7 rules, with blocks `(w_1A,w_2A)`,
  `(w_1B,w_2B)`, `(w_KA,w_KB)` and `(w_12)`, giving 8 + 8 + 8 + 2 = 26
  neighbors. Edits are local in parameter support; credit uses global output
  error.
- **exact-query memory:** answers `unknown` for unseen queries.
- **zero-coupling baseline.**

**Arm I (interventional supervision, secondary).** The same learners and streams.
For each training query, the feedback additionally includes **B6 and B7** (the
satellite clamps) for that query. These are physical interventions on a satellite
state, with access to the resulting **internal input-coordinate readouts** as
well as the other satellite's mode. Arm I therefore has both intervention access
and richer observation than Arm O; an advantage cannot be attributed to
interventions alone. It does not receive K or full trajectories. The NF, freezing,
core-removal and link-ablation outcomes (B2–B5) are
**never** supplied in either arm, because they are architectural surgery rather
than observations. A full-feedback hypothesis survives only if it matches every
supplied object. Arm I asks whether clamp observations resolve the mechanism
uncertainty that arm O leaves.

The teacher's full battery is a labeled upper reference, not a learner.

### Update and prediction rules for both arms

One feedback item is a map from typed observation keys to complete sets: O has
one B1 entry; I has that entry plus one B6 entry per allowed A clamp and one B7
entry per allowed B clamp. Process the map as **one experience**, not several
updates. Both arms use the same 15 query experiences.

- Full-feedback filtering requires equality on every entry. Positive-only
  filtering requires teacher-set containment in the predicted set on every
  entry, in both arms; it must not become equality in Arm I. Empty sets impose
  no positive-only constraint. Its retention guarantee concerns positives only.
- Version-space learners do not select a smallest-ID executable model. Predict
  all-survivor agreement or uncertainty, with certain/possible outputs. Empty V
  is `inconsistent_feedback` and cannot certify anything; preserve failure state.
- The block learner starts at seven zero coefficients. On the current item,
  `loss(h)=sum_key |predicted_set(h,key) symmetric_difference feedback[key]|`.
  Each tuple mismatch has unit weight. Arm I thus has more supervised terms;
  report their counts rather than claiming equal supervision cost. Evaluate the
  26 distinct neighbors and the unchanged vector. Minimize `(loss,
  number_of_changed_coefficients, lexicographic_vector)`, but commit only if loss
  strictly improves. At most one block update per experience, one pass, no
  replay or prior examples retained. No restarts or enlarged neighborhood.
  Predictions are complete terminal output sets, never selected terminals.
- Memory stores the feedback map keyed by the complete query. It answers only
  keys explicitly observed for that query, including B6/B7 in Arm I. All unseen
  query/key combinations, including B2–B5 and B8, return `unknown`. The zero
  baseline remains zero in both arms.

Learner APIs receive `(query, feedback_map)` and their own state. Teacher labels
and evaluation are separate. A simulation service may answer
`(candidate, current_training_query, supplied_observation_key)` for candidate
evaluation. Pure memoization is allowed, but must not expose teacher labels,
hidden intervention keys, held-out results or the evaluator's B8 schema. Count
memoization memory separately from learned state. At checkpoints the evaluator
may inspect model state and simulate held-out cases; none of those results may
flow back into updates.

## 7. Metrics

For every teacher, arm, split, order, checkpoint and partition (seen, within,
mixed):

- **Outputs:** as v1 §9. Certified coverage, accuracy given certification,
  possible-output precision, certain-output recall, survivor count, uncertainty.
  For the local learner: exact-set accuracy, TP/FP/FN, participation-signature
  accuracy, and retention after every update. At post-update steps t=2..15,
  compare against all t−1 earlier feedback maps. Exact seen-feedback retention
  is the fraction of those maps reproduced in full, with an explicit denominator.
  Also report B1-only retention and losses of previously correct items: never
  learning an item differs from forgetting it. Target 4 uses complete-map
  retention for the relevant arm.
- **Mechanism, version-space learners:**
  - for each battery item on held-out queries, whether all survivors agree, and
    whether they agree with the teacher;
  - the fraction of survivors mechanism-equivalent to the teacher;
  - **mechanism uncertainty:** the number of survivors that agree with the teacher
    on every held-out B1 output but differ on B2–B8;
  - B8 summaries: how many survivors reproduce each teacher witness tuple, and how
    many have some chain witness of their own.
- **Mechanism, local learner:** the battery accuracy of its single model per item
  type, whether it reproduces each teacher witness tuple, and whether it passes
  the full shared-input gate (conditions 1–7, 3′ and 8) at the final checkpoint.
  For S1/S2 use SF for condition 8; for C2 use CF. For T1 report both templates
  separately. Condition 3′ always concerns B under NF. Preserve the frozen gate
  rules; do not accept whichever template passes for a primary positive teacher.
- **Negative controls (S2, T1):** the **false mechanism attribution** rate. That
  is the fraction of survivors, or of local models, with **any forward SF or CF
  witness** on all 81 queries, against the teacher's absence. Verify that absence
  in the teacher battery. Report SF, CF and their union separately. Reverse
  witnesses are not false attribution for S2; compare them to actual teacher
  labels. If a negative teacher has a forward witness under this definition,
  mark the negative-control assumption refuted, report the witness, and continue
  the frozen grid without changing labels or choosing a replacement teacher.

For set-valued atoms, certain/possible are intersection/union across V. For
Boolean B8 atoms, record possible truth values in `{false,true}` and certify only
singleton agreement. All-false agreement is a valid negative certificate; an
empty witness list is not by itself mechanism identification. Report witness
sensitivity and false-positive rates separately so numerous false atoms cannot
hide missed positives. For B5, separate teacher-zero coefficient no-ops from
genuine teacher ablations in accuracy summaries.

A mechanism-equivalent survivor fraction is a descriptive fraction under uniform
counting, **not a calibrated probability**. For mechanism uncertainty also report
the denominator (survivors matching every held-out B1 output) and the fraction.
Report output-disagreeing survivors separately; they are uncertain too.

Zero denominators are null. Report distributions over splits and the worst split.
Logical storage: V is 2,187 bits, the local model is 7 ternary coefficients, and
memory is its stored entries. Separate learner work from evaluator and battery
work.

## 8. Correctness gates

- Every sampled battery entry specified in §4 matches the independent oracle.
  This is sample verification of the expanded battery, not exhaustive oracle
  verification of every scored intervention for all 2,187 hypotheses.
- Full-feedback learners retain the teacher, reproduce every observed feedback
  map for every survivor, and have order-independent final V. Positive-only
  learners retain the teacher and all observed positives and are order-independent;
  exact reproduction is not required of that ablation. Neither guarantee is
  imposed on the local learner as an implementation invariant.
- For each set-valued atom, `certain ⊆ teacher ⊆ possible`. For Boolean atoms,
  the teacher truth value belongs to the survivor truth-value set. All certified
  results equal the teacher under clean realizable feedback.
- Frozen shared-input results are reproduced for S1, C2 and S2; T1 reproduces
  the archived v1 conditions 1–7. Its additional battery items are new evaluated
  controls, not purported archived results.
- Learner access obeys §6: only provided training feedback and permitted candidate
  simulations. No hidden teacher identity, gate results, archived witnesses or
  evaluation feedback reaches learning. Splits use no labels.
- Regeneration is deterministic and byte-identical in a temporary directory.

## 9. Prospective targets (no retrospective changes)

For S1 and C2, at checkpoint 15 and pooled over the 5 splits:

Targets 1–3 refer to the full-feedback reference, with positive-only results
alongside as an ablation. Count one final version space per split, since final
V does not depend on order. Local targets apply **separately to each arm** over
its 25 streams; report per-split distributions rather than treating orders as
independent datasets. Pool retention numerator/denominator over all update steps
and streams; pool output counts over final held-out predictions.

1. **Output identification (arm O):** at least 80% certified coverage on each
   held-out partition, with no incorrect certified set.
2. **Mechanism identification (arm O):** in at least 4 of 5 splits, every
   survivor is mechanism-equivalent to the teacher on all 81 queries. If §5 shows
   the observational class is mechanism-heterogeneous, record this target as
   **unattainable by identifiability** rather than as merely failed. Still report
   the measured values.
3. **Mechanism identification (arm I):** the same criterion with clamp
   supervision. Use Arm I's own augmented-information ceiling to label this
   target unattainable when appropriate. Also report whether arm I strictly improves on arm O in
   mechanism-equivalent survivor fraction or in split count.
4. **Local learner:** on each held-out partition, at least 80% exact-set accuracy,
   a strict improvement over zero coupling, and at least 95% retention. Also, in
   at least 13 of 25 runs, the final model reproduces every positive teacher
   fixed-witness tuple (both directions, all declared templates) and passes the
   teacher-specific full gate defined in §7. This is a behavioral recovery target,
   not a claim that the selected coefficients were uniquely identified. Report
   false B8 atoms too; matching positives alone is not mechanism equivalence.
5. **Negative controls:** zero false mechanism attribution among final survivors
   of the full-feedback reference in both arms. For the local learner, report the
   rate; no threshold is set. If an arm's complete-information class contains a
   false-forward-attribution model, its zero-attribution target is unattainable
   by that arm's information. If the negative teacher assumption itself is
   refuted, mark that control target invalid rather than silently redefining it.

Report every condition separately. Reaching an output target while missing a
mechanism target is the central negative outcome this design is meant to detect.
Report it plainly.

## 10. Proofs and tests

- **Proofs:**
  - version-space retention, order independence and certification soundness
    extend to battery items, because each item is a function of h;
  - the entire complete-information equivalence class survives any subset of
    that arm's clean observations. Mechanism heterogeneity therefore prevents
    sound identification across that class and the full-reference target, while
    allowing a selected model to guess correctly;
  - Arm I's survivor set is a subset of Arm O's for the same training prefix and
    filtering rule (full versus full, positive-only versus positive-only). This
    does not guarantee a strict improvement or a better local learner;
  - `k -> -k`, `w_KA -> -w_KA`, `w_KB -> -w_KB` preserves battery observations
    under the declared interventions. Retain this as a known equivalence control,
    rather than treating differing coefficients as a mechanism error.
- **Tests:**
  - battery entries against the oracle;
  - B8 tuple re-evaluation reproduces the frozen teacher results;
  - learner API isolation;
  - contradictory feedback;
  - the block learner's 26 neighbors, single update per item and absence of
    replay;
  - Arm I supplying only B1/B6/B7 and only for training queries; typed-key loss
    accounting and positive-only containment in both arms;
  - memory returning `unknown` for unobserved query/key combinations;
  - correct S2 reverse-chain labels; false as well as true B8 atoms; domain-invalid
    tuple arguments rejected and absent recipient-origin values scored false;
  - per-query B8 existence flags, all-81-query aggregation, and partition-scoped
    equivalence agreeing with direct complete battery comparisons;
  - known core-sign symmetry and explicit indistinguishable/different-mechanism
    certificates when found (do not invent ambiguity if the class is homogeneous);
  - final V independence across orders and the corresponding Arm-I subset checks;
  - deterministic regeneration and unchanged prior scientific/source hashes.

## 11. Deliverables and stopping rule

- Code:
  - `src/mechanism_learning.py`
  - `experiments/mechanism_learning.py`
  - `tests/test_mechanism_learning.py`
- Reports:
  - `reports/learned-input-mediation-v1-manifest.json`, with teacher vectors,
    observation/chain schemas, split-file hash, checkpoints, rules and contract hash;
  - `reports/learned-input-mediation-v1-battery-summary.json`, with per-hypothesis
    battery hashes and full teacher entries;
  - `reports/learned-input-mediation-v1-ceiling.json`, written before learning;
  - `reports/learned-input-mediation-v1-learning.json`;
  - `reports/learned-input-mediation-v1-runtime.json`.
- Results: `docs/learned-input-mediation-v1-results.md`.
- README and open-problems entries after validation.

Order: freeze the new manifest (without rewriting the old split manifest); build
the battery and validate oracle checks; freeze its fixed-witness atom schema and
hash; write both ceiling results; run Arm O, then Arm I. Run both arms in full
regardless of the ceiling result, but never continue through a failed correctness
gate. Stop after this grid.
Do not add teachers, hypothesis families, battery items, intervention types for
training, or learners. Do not change thresholds, and do not search for teachers
that are easier to identify.

Use standard-library reference code and the repository's unittest conventions.
Start timing before preprocessing and separate battery, ceiling, training,
evaluation and test costs. Keep runtime/git metadata separate from deterministic
scientific output; record source/interpreter hashes and prior artifact hashes.
After targeted tests, run the full suite. Support `--out-dir` for regeneration
into a temporary directory and compare the manifest, battery summary, ceilings
and learning report byte-for-byte. Include reproduction commands in the results.
Report completion status separately from each target's pass/fail/unattainable
status; a scientifically negative but complete run is a completed study.

## 12. Scope

The study covers mode-level, realizable, noise-free, complete-set supervision
from four hand-chosen teachers, with no training noise. Success shows that this
outcome supervision, or this clamp-plus-internal-readout supervision, can identify
the specified finite battery equivalence class inside this candidate. Agreement
on a finite battery is not proof of equivalence under all possible interventions.
It does not show autonomous discovery, a biological or SRI
plasticity law (the local learner uses global error), robustness, semantics, or
that energy descent is SRI's foundation.
