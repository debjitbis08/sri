# Step 10, stage 2: does latent replay from memory make learning last? (design draft)

Status: **draft, 2026-11-04. Nothing run, not frozen.** Follows [stage 1](sense-memory-core-v1-pilot.md) and its [design](sense-memory-core-v1-design.md). Requested after external review of stage 1. The open choices are under *Decisions*.

## Question

Stage 1 showed memory supplying facts that the core interprets across senses, with memory present at every read. Stage 2 asks whether memory makes learning **last in the core itself**. The core trains on its curriculum while memory replays stored experience into its training. Then, **with memory removed at evaluation**:
- **Acquisition:** does the core itself now answer the new associations? Shown a fresh rendering of a symbol, does it give the symbol's count as taps, as dots and in comparison?
- **Retention:** does the core keep its existing cross-sense abilities as well as without replay, or better?

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

- **Facts:** stage 1's, with two sets of 45 pool symbols balanced over counts 1–9. Set A is written during block 2, set B during block 3, each spread over its block.
- **Writes:** 32 per symbol, each a training rendering seen with a different training collection of its count. That is four times stage 1, so storage limits bind (below).
- **Memory:** stage 1's kept-centre memory with a frozen gate per centre, raw values, τ 0.85, centres opened by the windowed shift rule.
- **New: raw cue features.** Each unit also keeps the raw frozen code of its cues, as a sum and count like the value. The current address keys are centred, averaged and gated, so they are not raw cue features the core's projection can take. This adds 3 KB per unit, which is counted.

### What a replayed item is, and its objective

A replayed item is a stored pair: a unit's mean raw cue code (the symbol) and mean raw value code (the collection). Both enter the core through the projection, as `FrozenImage.from_codes`, without an image.

The core's matching loss needs the pair's count: it places the positive collection at its count's slot among nine candidate collections. Memory stores no count. So:
- **Objective (proposed):** the core labels the value itself. The current core reads the stored collection against the batch's nine candidate collections, picking the slot whose candidate it matches best (stage 1: 0.94 on stored values). The cue is then trained to match the stored collection at that slot, with the same cross-entropy as data items. The label comes from the core's reading at replay time, never from memory, so meaning still comes from the current host.
- **Diagnostic arm, `memory-replay-true-count`:** the same with the true count. It separates the cost of self-labelling from that of replay.

Self-labelling can confirm its own errors. Its error rate, the share of replayed items labelled wrongly, is logged.

### Arms

Each replay arm appends **16 replay pairs to every update** from the time its first set is written: the size of the arbitrary pairings study's pair batch. The number of updates is the same in every arm.

| Arm | Stored | What is replayed |
|---|---|---|
| `no-replay` | memory written, never replayed | nothing |
| **`memory-replay`** | memory units (address, raw cue, raw value) | units sampled uniformly |
| `buffer-replay` | a reservoir of raw (cue, value) feature pairs, one per write, at memory's total bytes | pairs sampled uniformly |
| `memory-replay-true-count` | as `memory-replay` | as `memory-replay`, labelled with the true count |

**Matched storage.**
- **Bytes:** the buffer's budget is memory's total occupied bytes at each moment, including its addresses (which memory needs for retrieval). A pair costs 6,144 bytes, and a memory unit about 12.3 KB.
- **Expected effect:** at stage 1's 4.1–4.5 units per fact for 8 writes, memory should hold 32 writes per symbol in fewer units than writes. The buffer must then drop writes: a reservoir keeps a uniform sample.
- **If memory's units grow with writes,** the budgets converge and the comparison is between merging and keeping single writes. Units per fact at 32 writes is a measured outcome.

**Matched training budget.** The same replay items per update, the same updates, and the same projection and core passes per replay item in every replay arm. `no-replay` does 16 fewer item passes per update; that difference is reported.

### Reads, all with memory removed

Read at the end of block 2 (after A), block 3 (after B), the delay and relearning.

| Outcome | Read |
|---|---|
| **Acquisition**, primary | fresh evaluation renderings of each set's symbols through the core alone: taps among 9 (across senses), dots among 9, comparison with another symbol of the set. This is stage 1's `core-only` read; chance is 1/9 and 1/2. |
| **Retention**, primary | the step 8 snapshot's taps → collections, comparison on collections and taps, the eight transfer measures, and Devanagari matching (paired, withheld), as paired differences from `no-replay` |
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

Defaults are proposed, and none is needed before implementation:
1. **The self-labelling objective,** with the true-count arm as a diagnostic. The alternative is an in-batch contrastive loss between cues and values, which needs no label but treats same-count pairs as negatives.
2. **Uniform replay sampling.** Replay selected by similarity to the current input, matching recent experience with old as the user described on 2026-11-01, is a natural next arm. It is left out to keep this pilot to one question.
3. **32 writes per symbol,** so that storage limits bind.
