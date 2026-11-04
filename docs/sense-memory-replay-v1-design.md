# Step 10, stage 2: does latent replay from memory make learning last? (design draft)

Status: **design for the stage 2 pilot, revised 2026-11-04 after external review (*Changes from review*, below). Not a contract. Run: [results](sense-memory-replay-v1-pilot.md).** Follows [stage 1](sense-memory-core-v1-pilot.md) and its [design](sense-memory-core-v1-design.md). Requested after external review of stage 1. The open choices are under *Decisions*.

## Question

Stage 1 showed memory supplying facts that the core interprets across senses, with memory present at every read. Stage 2 asks whether memory makes learning **last in the core itself**. The core trains on its curriculum while memory replays stored experience into its training. Then, **with memory removed at evaluation**:
- **Acquisition:** does the core itself now answer the new associations? Shown a fresh rendering of a symbol, does it give the symbol's count as taps, as dots and in comparison?
- **Retention:** does the core keep its existing cross-sense abilities as well as without replay, or better?

**`no-replay` never learns these associations at all.** Its result shows what acquisition through replay achieves; it is not protection against forgetting after equal initial teaching. Retention of the existing curriculum is a separate result.

Replay is compared with no replay, and with a plain buffer of stored feature pairs under the same storage and training budget. That separates what this memory contributes from what any retained features would.

Replay of stored frozen codes is **rehearsal: latent replay from memory** (user, 2026-11-04). Its bytes, exposures and update cost are reported like any rehearsal's.

## What earlier results predict

- **Without rehearsal, arbitrary pairs fade from the core.** In the [arbitrary pairings study](arbitrary-pairings-v1-results.md), every arm was at 0.12–0.15 on set A after block 5, against chance 0.111. In [part D](number-construction-v1-part-d-results.md), rehearsal removed most forgetting of scripts (Chinese at the end, 0.20 to 0.53). So `no-replay` should be near chance on acquisition, and replay should help.
- **A new symbol taught a count becomes a numeral form.** The core places forms on its number line by count ([shared-core memory, stage 0](shared-core-sparse-memory-v1-contract.md)). Learning a symbol-to-count pairing is exactly that placement, so acquisition does not fight the core's geometry. It may, however, compete for capacity with scripts.
- **Latent replay with a fixed feature extractor works** (Pellegrini et al. 2020; van de Ven et al. 2020). With the extractor frozen, replaying features is equivalent to replaying inputs for everything trained above it. A plain feature buffer is therefore a strong baseline. The open question is whether merged memory units do as well in fewer bytes, or better.
- **Complementary learning systems** predict that interleaving replay with new learning protects old knowledge. They predict no advantage for a fast store over a buffer of the same experiences, except in capacity: storing more experience in fewer bytes.
- **Stage 1:** at relearning, the core reads dots and taps worse (fresh dots on taps, 0.91 to 0.66). That is the retention problem replay should address. Replay of symbol–dot pairs exercises the core's reading of dots, but not of taps.

## Design

### The core and curriculum

The step 8 calibrated core (`joined-mlp-calibrated`), trained exactly as limited-memory main (`run_main`): three scripts in blocks, a delay, then relearning. Seeds are 300–303. With no replay, this must reproduce step 8's `joined-mlp-calibrated` exactly. That is the reference check.

### Facts and memory

- **Facts:** stage 1's, with two sets of 45 pool symbols balanced over counts 1–9. Set A is written during block 2, set B during block 3, each spread evenly over its block's updates, in one order (A then B). Before each set, a passing run of 1,024 other pool symbols goes through the centre rule, as in stage 1.
- **Writes:** 32 per symbol, each a training rendering seen with a different training collection of its count, four times stage 1. Whether storage limits bind is measured, not assumed (below).
- **Memory:** stage 1's kept-centre memory with a frozen gate per centre, raw values, τ 0.85, centres opened by the windowed shift rule.
- **New: raw cue features.** Each unit also keeps the raw frozen code of its cues, as a sum and count like the value. The current address keys are centred, averaged and gated, so they are not raw cue features the core's projection can take. This adds 3 KB per unit, which is counted.

Memory's contents never depend on the core, so the whole write sequence, with its centres and gates, is built once per seed before any core run. During training the same writes are replayed into a live memory at their scheduled updates.

### What a replayed item is, and its objective

A replayed item is a stored pair: a unit's mean raw cue code (the symbol) and mean raw value code (the collection). Both enter the core through the projection, as `FrozenImage.from_codes`, without an image.

The core's matching loss needs the pair's count: it places the positive collection at its count's slot among nine candidate collections. Memory stores no count. So:
- **Objective: a slot chosen by the core.** The data matching loss compares an item with the batch's nine candidate collections, one per count. It replaces the candidate at the item's count with the item's own paired collection, and trains the item to pick that slot. The count only decides which candidate is replaced, so a same-count candidate is never a negative; no numerical coordinate is supervised.
  - For a replayed pair, the slot is chosen by the current core. Before the update, with gradients stopped, the core reads the stored collection against the nine candidates, using the `same` relation, and takes the best slot.
  - The candidate at that slot is replaced by the stored collection, and the cue is trained to pick it, with the same cross-entropy.
  - `buffer-replay` uses the identical procedure.
  - The slot comes from the core at replay time, never from memory, so meaning still comes from the current host.
- **The teacher moves.** The core's reading of stored collections changed substantially over training in stage 1 (0.67–0.71 at block 1, 0.94–0.96 at block 3, 0.72 at relearning on taps). The slot's accuracy against the true count is logged by phase (block 2, block 3, delay, relearning) and by set, not assumed.
- **Diagnostic arm, `memory-replay-true-count`:** the slot is the unit's true count. A merged unit may hold writes of different facts, and so of different counts. Its target is then the count with most writes in the unit, with ties going to the count written first. Replays from units holding more than one fact, or more than one count, are counted and reported separately in every arm.
- **Loss weighting.** Replay enters as its own term, `loss = loss_match + loss_next + loss_cmp + loss_sequence + λ · loss_replay`, where `loss_replay` is the mean cross-entropy over the 16 replayed pairs. λ = 16 / (data matching items in the batch), so a replayed pair weighs the same as one data matching item, and the data terms are unchanged. λ is set per update and logged. The seed-300 smoke run averaged 36 data matching items per batch, so λ is about 0.4–0.6, not the 0.15 this draft first estimated.
- **Passes counted:** 32 extra item passes per update through the projection and core (16 cues, 16 values), with grad. The slot choice reuses those value codes and the batch's candidates, detached, so it adds one 16 × 9 relation evaluation and no item pass.

### Arms

Each replay arm appends **16 replay pairs to every update** from the time its first set is written: the size of the arbitrary pairings study's pair batch. The number of updates is the same in every arm.

| Arm | Stored | What is replayed |
|---|---|---|
| `no-replay` | memory written, never replayed | nothing |
| **`memory-replay`** | memory units (address, raw cue, raw value) | units sampled uniformly |
| `buffer-replay` | a reservoir of raw (cue, value) feature pairs, one per write, with a fixed capacity declared before the run (below) | pairs sampled uniformly |
| `memory-replay-true-count` | as `memory-replay` | as `memory-replay`, labelled with the true count |

**Matched storage.**
- **A fixed budget, declared before any core run.** A buffer whose capacity grows cannot stay a uniform sample of all past writes, because discarded writes cannot be recovered. So the buffer's capacity is fixed: memory's final total occupied bytes for that seed (addresses, raw values, raw cues, counts, centres, gates), divided by 6,144 bytes per pair, rounded down. Memory never depends on the core, so it is built in full before the core runs, and this number is known in advance.
- **Uniform historical sample.** Standard reservoir sampling (Vitter's algorithm R) over the stream of writes. Until it is full, the buffer holds every write; after that, every write so far is kept with equal probability.
- **Bytes over time:** memory's occupied bytes grow with its units, and the buffer's with its items up to capacity. Both are logged at every phase end.
- **Whether storage binds is a measured outcome.** A unit costs about 12.3 KB and a pair 6.1 KB. So memory holds less than the buffer only if it needs fewer than about half as many units as writes, before its shared overhead. If not, the buffer can hold every write.

**Matched training budget.** The same replay items per update, the same updates, and the same projection and core passes per replay item in every replay arm. `no-replay` does 32 fewer item passes per update; that difference is reported.

**What uniform sampling means differs.** Uniform over memory units is not uniform over writes: a symbol split into more units is replayed more. That difference is part of what is tested. Exposures per symbol and per count are logged in every replay arm.

### Reads, all with memory removed

Read at the end of block 2 (after A), block 3 (after B), the delay and relearning.

| Outcome | Read |
|---|---|
| **Acquisition**, primary | fresh evaluation renderings of each set's symbols through the core alone: taps among 9 (across senses), dots among 9, comparison with another symbol of the set. This is stage 1's `core-only` read; chance is 1/9 and 1/2. |
| **Retention**, primary | the step 8 snapshot's taps → collections, comparison on collections and taps, the eight transfer measures, and Devanagari matching (paired, withheld), as paired differences from `no-replay`. Replaying symbol–dot pairs exercises the core's reading of dots, but does not rehearse taps. The taps-retention failure seen in stage 1 is a target to test, not a result this objective guarantees. |
| Retention, secondary | everything else in the step 8 snapshot, reported but not judged |
| A after B | set A's acquisition at block 2 against block 3 and later, in each arm |
| With memory present, secondary | stage 1's end-to-end reads, to check replay has not changed how the core interprets memory |

### Proposed bars

These are for a later contract and are not tested by this pilot:
- **Acquisition:** `memory-replay` reads taps from symbols with memory removed at least 0.3 above `no-replay`, at relearning, on both sets.
- **Retention:** no primary retention measure more than 0.05 below `no-replay`, paired by seed.
- **Against the buffer:** `memory-replay` within 0.05 of `buffer-replay` on acquisition and retention, at equal or lower bytes. That would make it a usable form of rehearsal. Exceeding the buffer would need a mechanism, for example consolidation keeping more of the experience per byte.

## Prior work

These are familiar methods, not re-read for this draft.
- **Latent replay** (Pellegrini et al. 2020, *Latent replay for real-time continual learning*) and **brain-inspired replay** (van de Ven, Siegelmann and Tolias 2020). Replaying stored features at a layer above a frozen part is established, and cheap. The `buffer-replay` arm is this method.
- **Experience replay and reservoir sampling** (Rolnick et al. 2019; Chaudhry et al. 2019, *tiny episodic memories*). Tiny buffers with uniform replay are strong baselines in continual learning. The budget comparison follows their practice of matching memory size.
- **Self-labelling** (pseudo-labels, Lee 2013). It is known to confirm its own errors when the labeller is weak. Here the labeller reads stored dots at 0.94, and the true-count arm measures the cost.
- **Complementary learning systems** (McClelland et al. 1995; Kumaran, Hassabis and McClelland 2016). Interleaved replay from a fast store protects a slow learner.

## Scaling

- **Per update:** 16 replay items through the projection and core. That is O(1) in the number of facts, and the same in every replay arm.
- **Memory:** units grow with distinct experience, not with writes when writes merge; about 12.3 KB per unit with raw cues.
- **Buffer:** pairs grow with writes until the budget, then stay fixed by reservoir sampling.
- **No input is stored or re-encoded,** in either arm.

## Computation plan

- **Frozen codes:** the pool renderings, now 32 training renderings per symbol (about 2,900 per seed, plus stage 1's queries and passing runs).
- **Core runs:** 4 arms × 4 seeds, each about 250 s, as in step 8, with three workers; about 25 minutes of wall time.
- **Reported:** replay exposures, bytes over time per arm, the self-labelling error rate, and the extra item passes.

## Decisions

1. **The slot chosen by the core** is the objective, with the true-count arm as a diagnostic. The in-batch contrastive alternative is not run.
2. **Uniform replay sampling.** Replay selected by similarity is left out.
3. **32 writes per symbol.** Whether storage binds is measured.

## Changes from review (2026-11-04)

- **The pseudo-label is described as choosing the replacement slot.** It is computed before the update with gradients stopped, identically for both replay arms, and its accuracy is logged by phase and set.
- **The true-count target for mixed units is defined,** and mixed units are reported separately.
- **The buffer has a fixed capacity declared before the run,** with reservoir sampling, rather than a growing budget.
- **Units against writes is measured,** with the break-even stated.
- **Exposures per symbol and count are logged,** since uniform over units is not uniform over writes.
- **Loss weighting is a separate term** with λ = 16 / (data matching items). All extra passes are counted.
- **`no-replay` never learns the associations,** and taps retention is a target, not a guarantee. Both are stated.
