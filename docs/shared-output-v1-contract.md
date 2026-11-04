# Shared output v1: contract

Status: **draft, revised 2026-11-04 after stage 0, not frozen. The pilot ran on 2026-11-04 and
its criteria fail ([pilot results](shared-output-v1-pilot.md)): the confirmation does not
proceed as designed.** This is stage 1 of the [shared output brief](shared-output-brief.md).
- **Stage 0** ([results](shared-output-v1-stage0.md)) ran on the existing checkpoints.
- **The confirmation** is frozen only after the pilot below has run and any changes it forces
  have been recorded here.

**Revision history.**
- **First draft.** It proposed a shared scorer on raw item codes. The user rejected it: nothing
  in it explained why a name would inherit relations.
- **This draft.** It derives the readout's structure from the theory (next section) and keeps
  the first draft's scorer only as an ablation (`common-raw`).

## Question

After being taught only X = 3, does the system continue (2, X) with 4 and (4, X) with 2, and
reject wrong completions? Does this survive script relearning?

The readout under test grounds every relational slot in a recalled participant. Is that what
makes the inheritance possible?

## The common readout, from the theory

The brief's theory describes **one neural read**. With a relation, a source and the context
active, the web evokes support across possible fillers:
- a supplied filler **probes** that support;
- an open slot **recalls** a filler, "whose state can participate in another relation."

Three structural commitments follow.

1. **A read returns a support pattern and a recalled state.** Let M hold the participants:
   familiar items whose states m_j are re-read through the current core. A query q gives
   - support a_j = s(q, m_j) for every participant;
   - a probe of a supplied candidate y, s(q, y);
   - a recalled state r = Σ_j softmax(a)_j m_j, or the top participant's state at evaluation.

   **r lies in the same space as the participants,** so it can fill a slot of another query.
   Composition over k steps is k reads, each recalling a real participant. It is not a rollout
   of predicted codes that have never been items.
2. **Identification is the read that meets the raw item.**
   ground(x) = recall of Q(x, ∅, same) over M. It answers "which known participant is this?"
3. **Every relational slot takes a grounded state, never a raw code.**
   - continuation is q = Q(ground(prev), ground(cur), continue);
   - a pair relation is q = Q(ground(a), ∅, r).

   Recall ranks the participants. A probe scores ground(y).

**Why this predicts inheritance.** With (3), a name enters relations only through what it is
identified with. If ground(X) = ground(3), then (2, X) *is* the query (2, 3): substitutivity of
identicals, held by the structure rather than learned from relational examples X never had.
Inheritance is then bounded by identification:

> N1 for names ≈ P(name identified as n) × P(familiar continuation from n is correct).

The comparison of the first draft could not show this. A scorer that reads raw codes needs the
name's code to stand in for n. Stage 0 found that it does not: only 0.26 of names sit nearest
their own count, although 0.46 are matched correctly.

**Evidence already in hand.**
- **Stage 0.** On the existing `g`/`P` checkpoints, with no training through the path,
  grounding a name before `P` continues at 0.56 / 0.50 (ascending / descending), against
  0.39 / 0.27 for the name's own code. It is higher in every seed and direction
  ([stage 0](shared-output-v1-stage0.md)).
- **The step 10 memory pilot.** Recalled dot traces fed through the core answered taps at
  0.94–0.95 ([sense-memory-core stage 1](sense-memory-core-v1-pilot.md)). A recalled
  participant state can take part in a relation the symbol was never taught.

**Gradient rule.** Identification weights receive gradient only from identity evidence:
matching and its acceptance loss. Relational losses reach the participant states m_j through
the core, but are stopped at the identification weights.
- **Why.** Names never enter relational rows. If continuation could reshape how *familiar*
  items are identified, it could learn an identification that names do not share, and
  substitution would fail. Identity comes from evidence of identity.

**What the structure supplies, and the question for the user.** The routing in (3) is supplied
structure. It is substitutivity of identicals, the one logical law built in. Every relation,
including `same` itself, is learned and has no built-in form. **Whether this law belongs in the
"smallest logical consistency system" a core may hold was the user's decision: approved
2026-11-04.**

**What grounding loses.** A grounded state carries what the participant class shares: here,
quantity. A relation about an item's own appearance (script, font) cannot be read from it. In
the theory, context selects which equivalence identifies participants. That is brief stage 2
and is not tested here; in this fixture every relation is about quantity.

## Decisions taken before this draft

- **Host.** The step 8 core: `sparse48` with the `joined-mlp-calibrated` image encoder (frozen
  DINOv2-S and TrOCR-small, fixed block-1 statistics, trained MLP projection;
  [step 8 pilot](sense-core-v1-pilot.md)). It uses the limited-memory main curriculum: English,
  Chinese and Devanagari blocks of 2,000 updates, then a 2,000-update delay, then 200 relearning
  updates on English with rehearsal from the buffer.
- **Single default setting.** There is no provenance, no learned context selection, no ATLAS
  and no attention over context. The two observed items stay as the history, so direction
  comes only from them.
- **Participants are frozen sense codes** (user, 2026-11-04). M stores the raw frozen
  768-dimensional codes of familiar collection exemplars, never images and never core codes.
  They are re-read through the current projection and core at every update and every read.
  This follows the user's decisions of 2026-11-02 that values are sensory traces and that memory
  stores no images.
- **All scores come from the network.** Counts, sequence metadata and relation labels live only
  in the harness.

## Arms

All arms start from the same seeds. They get the same curriculum, loss terms, weights, batches,
candidate draws and participant memory M. **They differ only in the readout.**

| Arm | Readers | Relational slots take | Purpose |
|---|---|---|---|
| `gp` | `g(a, b; r)` for pairs; `P(prev, cur)` with a distance logit plus a learned bias b_c | raw codes | baseline, the current readout |
| `gp-grounded` | the same `g` and `P`; identification is `g(x, m_j; same)` over M | grounded states | grounding with the existing readers |
| `common` | Q and s for every read, identification included | grounded states | **candidate**: one read, grounded |
| `common-raw` | Q and s for every read | raw codes | ablation: sharing without grounding (the first draft) |

**Shapes, Q and s.**
- **Q** is the query encoder. Its input is [slot 1 (16), slot 2 (16), token (8)], with layers
  40 → 64 → 64 → 16. There are four tokens: `next`, `same`, `more`, `continue`. A pair query
  holds ∅, a learned empty vector, in slot 2.
- **s** is the scorer. Its input is [q (16), y (16)], with layers 32 → 64 → 64 → 1.
- **Layout matches the baseline.** Q's input layout is `g`'s and s's input layout is `P`'s. Their
  first layers are built as `g`'s and `P`'s are in `sparse48` (`limited_memory.make_model`: 48
  structure units, one coordinate per operand, RigL on the same schedule), with masks from the
  same named streams.
- **Parameter counts are matched.** By the layer sizes, `g` + `P` have about 14,160 reader
  parameters and Q + s about 14,210; the implementation reports the exact counts.

**Grounding at training time.** Each update re-reads M (72 participants) through the current
core. Every item in a relational row (`next`, `more`, continuation) is grounded once per update
by soft recall, under the gradient rule above. Matching rows are identification evidence and
use raw sources, as now.

**Reference.** The step 8 `joined-mlp-calibrated` results at seeds 300–303 are kept as they
stand. They are a different curriculum: no names and no acceptance losses.

## Curriculum changes (all arms)

1. **Names.** For each seed, 45 pool symbols, 5 per count 1–9. They are the first 45 left after
   `sense_memory_core.symbols`' sets A, B and unseen, as in stage 0 (`shared_output.names`).
   Each name has 8 training renderings and 5 evaluation renderings. Their frozen codes are
   cached (`shared_output names`).
2. **Naming-only teaching, block 3 only.**
   - Each block-3 update adds 9 naming rows, one per count, against the update's 9 candidate
     collections.
   - Names enter only matching, as the identification source.
   - **A name never enters `next`, `more` or continuation rows, as input, target, negative or
     participant.**
   - The runner audits every update and fails on any leak, as in stage 0.
   - There are no naming rows in the delay or in relearning, and names are never rehearsed.
3. **Acceptance supervision.**
   - Every matching and continuation row gets a binary cross-entropy on the same logit that the
     9-way cross-entropy ranks. The 8 negatives are weighted to balance the positive.
   - `next` and `more` keep their binary losses.
   - All weights are 1, untuned.
4. **No mixed-kind histories.** The first draft added them. Stage 0 dropped them for two
   reasons:
   - (collection, taps) histories read as well as (collection, collection) without them (0.98 /
     0.91 against 0.94 / 0.89), and the weaker Devanagari read follows its weaker
     identification.
   - Added form rows would increase the identified relearning disruption.

   **This overrides the first draft's own rule**, which would have kept them; the reason is
   recorded here and in the [stage 0 results](shared-output-v1-stage0.md). In the grounded arms
   the history kind is moot, since slots hold participants.

## Thresholds and abstention

- **Selection.** One threshold per arm, seed, read point and relation (`same`, continuation).
  Each maximizes balanced accuracy on validation single-candidate judgments: training-split
  collection histories that are not in M, and training-split Devanagari forms and taps. Names
  and evaluation renderings are never used.
- **Pooled threshold.** For `common` and `common-raw`, one pooled threshold is also reported.
- **Drift.** A secondary relearn read uses the delay threshold.
- **Recall.** The answer is the top participant. Abstain if its logit is below the threshold.
  Abstention is wrong when a correct participant is available.
- **Identification abstains the same way.** A name below threshold is unknown, and its
  relational queries abstain.

## Stages

- **Stage 0: done** ([results](shared-output-v1-stage0.md)).
- **Pilot.** Four arms, seeds 300–303, full curriculum.
  - **Reads** at the end of block 3 (names just taught), delay and relearn.
  - **Reference check.** `gp` with changes 1–3 off reproduces step 8 `joined-mlp-calibrated`
    exactly at seed 300.
- **Confirmation.** Seeds 424–433 are reserved. Criteria are frozen from the pilot.

## Reads

At each read point, every arm, every seed.

**N1 Naming-only transfer.**
- **Queries:** (evaluation collection n∓1, X_n) should give n±1, for n in 2–8.
- **Reported:** recall and single-candidate acceptance, ascending and descending separately.
  Hard and soft grounding are reported separately for the grounded arms.
- **Controls:**
  - familiar (collection, collection n) and (collection, taps n);
  - (collection, Devanagari n) on trained counts;
  - the substitution bound: the name's slot filled with a familiar n participant.
- **Secondary histories:** (taps, X_n) and pure names (X_{n−1}, X_n), reported separately.
- **Decomposition.** Identification accuracy of names over M, and N1 conditioned on correct
  identification. This tests the bound N1 ≈ identification × familiar continuation for each
  arm.

**N2 Rejection.**
- Single wrong candidates, split into reverse (n∓1), repeat (n) and the rest.
- Sets with the 8 correct participants removed, scored with the same logits restricted to the
  other 64: correct abstention and false acceptance.
- Unidentifiable inputs: pool symbols never taught, which should abstain at identification.

**N3 Retention**, delay against relearn, paired.
- Changes in N1, N2 and name identification.
- English acquisition.
- **The step 8 prior reads:** `transfer8`, matching, composition k = 1–3 and comparison.
- **Memory-present reads** r2, r3, r3′ and r4 on the existing `sense_memory_core` memory states.
  In the grounded arms the recalled value is a participant-space input like any other.
- Collection code spread and collection–taps mismatch.

**Naming interference.** Devanagari and Chinese matching at block 3, against the step 8
reference. Stage 0 found that naming without forms present lowers Devanagari matching (0.47 to
0.20).

**Composition, k ≥ 2.**
- `gp` and `common-raw` roll their predicted state forward.
- The grounded arms recall a participant at each step and feed it forward, abstaining when
  nothing is accepted.
- Both procedures are reported, labelled.

## Pre-declared criteria (pilot)

These are descriptive at four seeds.

- **Gates.**
  - Name identification at block 3 is at least 0.80 in every arm. If it is not, N1 is reported
    conditioned on correct identification, and the unconditioned comparison is not interpreted.
  - Familiar continuation recall at delay is at least 0.60 in every arm.
- **Advance to confirmation** if, at delay:
  - **Transfer.** `common` − `gp` on N1 recall (mean of directions, hard grounding) is at least
    +0.10, paired, and positive in at least 3 of 4 seeds.
  - **Rejection.** On removed-correct sets, `common`'s false acceptance is no more than 0.05
    above `gp`'s.
  - **Prior abilities.** Familiar reads and each N3 prior read are no more than 0.05 below
    `gp`, and so is English matching after relearning.
- **Attribution.**
  - `gp-grounded` − `gp` measures grounding with the existing readers.
  - `common` − `gp-grounded` measures one shared read.
  - `common-raw` − `gp` measures sharing alone.
- **Interpretation.**
  - **Transfer in the grounded arms but not the raw ones:** grounding carries the inheritance.
  - **Grounded transfer well below identification × familiar:** grounded states are not
    readable, a participant-space mismatch.
  - **Transfer at delay that is gone at relearn:** check whether identification fell
    (interference in identification) or continuation from familiar n fell (interference in the
    relation).

## Predictions

1. **[derived] The grounded arms reach roughly identification × familiar continuation.** The
   raw arms are bounded by substitutability.
   - **Stage 0:** grounded 0.56 / 0.50, raw 0.39 / 0.27, identification 0.45 over all counts,
     familiar 0.91 / 0.88. The grounded read exceeds the simple product, so the decomposition
     read (identification restricted to counts 2–8, and conditioning) will check whether the
     bound holds.
2. **[derived] `common` and `gp-grounded` transfer similarly.** Grounding carries the inheritance.
   One shared read adds agreement between recall and probe, and one calibrated threshold, not
   transfer.
3. **[local evidence] Identification is the weak link.** In stage 0 naming was slow (0.46 after
   200 updates) and interfered with Devanagari identification. Block 3 gives 10× the naming
   updates, with Devanagari forms present.
4. **[derived] Relearning affects names through identification.** English relearning moves the
   shared image projection most ([dynamics](relearning-dynamics-v1.md)). Names, like forms, are
   identified through that projection, so N1 at relearn should fall with name identification
   more than with familiar continuation.
5. **[familiar] Thresholds drift through relearning** in every arm.

## Prior work

Search date 2026-11-04. No new search was run. Every source is **[familiar]**, recalled rather
than re-read, unless marked. Shared survey:
[prior work](prior-work.md#shared-output-scorer-and-naming-only-transfer).

- **Grounding a mention in a participant before reasoning.**
  - This is entity linking before knowledge-base reasoning: resolve a mention to an entity, then
    query the entity's relations.
  - Its neural form is key–value memory reading: end-to-end memory networks (Sukhbaatar et al.
    2015) and key–value memory networks (Miller et al. 2016). There a read returns a weighted
    value that can feed the next hop.
  - Modern Hopfield networks (Ramsauer et al. 2020) give attention as associative recall whose
    output lies in the space of the stored patterns.
  - **Known failures:** soft reads blur between participants, and errors compound over hops.
    The design reports hard and soft reads, and composition step by step.
- **Inheritance through an evoked associate.**
  - In psychology: mediated generalization and representation-mediated learning (Holland
    1981), and acquired equivalence (Honey and Hall 1989). A stimulus that evokes another's
    representation inherits that representation's associations. That is the mechanism of
    commitment (3).
  - Stimulus equivalence (Sidman and Tailby 1982) is the behavioural test family.
- **The failure of raw-code transfer.** SCAN's add-primitive split (Lake and Baroni 2018):
  sequence models taught a primitive in isolation fail to compose it. Stage 0 reproduces this
  qualitatively for raw codes.
- **Query encoder plus one candidate scorer.** This is 1-N scoring in knowledge-graph
  completion: ConvE (Dettmers et al. 2018); TransE and NTN for related scoring functions. It is
  a cross-encoder (Humeau et al. 2020) and an energy E(q, y) (LeCun et al. 2006). Scoring a
  candidate next item instead of regressing its code is CPC (van den Oord et al. 2018).
- **Abstention.**
  - Selective classification (Chow 1970; Geifman and El-Yaniv 2017) and open-set rejection
    (Scheirer et al. 2013).
  - **Known failures:** softmax overconfidence off the training distribution (Hendrycks and
    Gimpel 2017), and calibration drift.
  - **Design response:** a binary loss on the ranked logit, validation-only thresholds,
    abstention at identification, and a drift read.
- **What the study adds.** A test, on matched supervision, of whether supplying substitutivity
  through grounded slots yields naming-only inheritance in a learned perceptual core. It also
  separates grounding from scorer sharing, and follows both through continual relearning. The
  components are standard; their combination as the readout of this core is what is tested.

## Supplied versus learned

| Supplied | Learned |
|---|---|
| Relational slots take grounded states (substitutivity); identification is the `same` read | What is the same (identification scores); every relation; participant states via the core |
| The gradient rule: identification weights see only identity evidence | Q, s (or `g`, `P`); the projection and core; ∅ and the tokens |
| M's membership (72 training collections); candidate sets; training targets | — |
| Decision rules: top participant, abstain below a validation threshold | — |

No inverse, symmetry, direction or composition rule is supplied.

## Scaling

- **Identification** costs O(N) scorer calls per item for N participants. A fixed item is
  grounded once per update, or once per read point, and cached.
- **Large N.**
  - **Route:** a cheap first stage (cosine on core codes), then s on the top k. This is
    retrieve-and-rerank, and is consistent with cosine as a cheap first stage.
  - **Memory structure:** at scale, M is a memory network, as in the target architecture's
    long-term memory: kept centres, units and a gate (`src.senses.KeptMemory`). M is that memory
    at small size.
- **Re-reading M** through the current core is O(N) per update. That is the cost of storing
  sensory traces instead of core codes.
- **Names** add nothing to the network. Each costs only its naming rows, and its relations come
  from the participants it is identified with.
- **Composition** over k steps is k reads, O(k · N).

## Computation

**Declared dimensions.**
- Code 16; reader hidden 64; tokens 8.
- 48 structure units per first layer.
- M has 72 participants × 768 float32.
- 45 names × 13 renderings.
- 8,200 updates per run, 16 batch episodes per update.

**Theoretical cost.** These are estimates, multiply-add × 2, dense before masking.
- **Per update:**
  - re-reading M is 72 extra projection and core passes, against the roughly 100–200 items an
    update already encodes;
  - grounding is about 150 distinct items × 72 scorer calls × about 12–13k FLOPs, roughly
    140 MFLOP;
  - continuation scoring in `common` is about 127k FLOPs per row, against about 14.6k in `gp`.
- **Per query:** identification is 72 scorer calls (about 0.9 MFLOP), plus one relational read
  of about 0.9 MFLOP. `gp` costs about 17k FLOPs.

**Storage.**
- Reader parameters about 57 KB float32 per arm, plus Adam's two moments.
- M is 221 KB.
- The name code cache is 1.8 MB per seed. It is harness data, not memory.
- No raw images are retained.

**Measured, in separate runtime files.**
- Per phase: name encoding, training, threshold selection and reads. Wall and CPU seconds,
  aggregate worker CPU, peak RSS and GPU peak memory.
- Counts of updates, scorer calls, M re-reads and threshold candidates.
- The reference check counts as selection cost.
- Warmed, repeated recall timings per arm.
- **Stage 0 measured:** 43 s wall on three CPU workers, and 19 s GPU encoding per seed
  ([results](shared-output-v1-stage0.md#computation)).

**Expected cost.** Step 8 runs took 3–4 minutes each, and grounding adds a few-fold reader cost.
The pilot is 16 runs, about an hour on three workers.

## Not tested

- Context-selected equivalences, provenance and revision, and abstraction (brief stages 2–4).
- Constructing states for unseen participants, and participant memories that grow.
- Kinds of answer other than collections.
- Attention over context.

## Implementation

- **`src/shared_output.py`** converts the sparse48 `GenericSequenceNet` in place, so initial
  weights, masks and every existing read stay as they are.
  - **What it holds:**
    - the participants M (a buffer of raw frozen codes);
    - identification, and grounding with detached identification weights;
    - `relation_logits` (grounded for `next` and `more` in the grounded arms; `same` always
      raw);
    - `cont_logits`;
    - `predict`: `P`'s code, or the recalled participant for the common arms, so the step 8
      composition reads roll it;
    - the acceptance loss and the threshold.
  - **For the common arms,** Q is `g` with its last layer 64 → 16, and s is `P` with its last
    layer 64 → 1. Measured reader parameters: 13,003 for the `gp` arms and 13,025 for the common
    arms, after masking.
- **`experiments/shared_output.py`** has the stages `names`, `stage0`, `reference` and `pilot`.
  - **Names** have their own item kind (`name`). Next negatives, comparison draws and sequence
    rows select by kind or episode, so they cannot reach a name. Every update also asserts that
    names appear only as matching sources.
  - **The update** is `transition_pilot.train_update` with the same draws, plus the acceptance
    losses and continuation through `cont_logits`. Participants are re-read with gradient for
    the grounded arms.
  - **Reads** are added to the block-3, delay and relearn snapshots.
  - **Draws.** Continuation candidates in training are the update's 9 candidates, as before. In
    the grounded arms they are grounded like every relational input.
- **`experiments/shared_output_summary.py`** gives the tables and the pilot criteria.
- **`tests/test_shared_output.py`** has 9 checks:
  - parameter counts and masks across arms;
  - the gradient rule (a relational loss gives no gradient to raw items or identification, and
    some to M);
  - substitution under hard grounding;
  - shapes;
  - threshold ties;
  - acceptance weighting;
  - the naming audit on real batches.
- **Not yet implemented: the memory-present reads** (r2, r3, r3′, r4 on the `sense_memory_core`
  memory states). They will be read from the saved checkpoints, as a separate read stage.
