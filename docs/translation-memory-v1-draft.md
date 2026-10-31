# Translation memory v1: design draft (not a contract)

Status: **draft 2026-10-31**, for the researchers' review. Nothing is implemented or frozen. Follows [retention remedies v1](retention-remedies-v1-results.md), whose claim was narrowed to
"some behavior can be transferred into protected weights under an externally managed procedure", and the external review's requirements for a closed memory system recorded there.
Builds on [kinds of memory and learned routing](memory-kinds-notes.md). Items marked **[decide]** need a decision.

## Why a new task

The pairing task's facts were symbol → count: 45 facts shared nine answers the network already represents, so a store could succeed by learning a coarse code-to-count rule instead of
individual facts (the other-symbol cue result), and the only possible "connection to something stored" was the count. The researchers proposed a **translation task with random symbols**:
two invented alphabets drawn from the filtered Unicode pool, a lexicon of arbitrary word pairs, and sentences whose translation needs both the lexicon (memory) and a word-order rule (structure).

## Question

Can a bounded system with its own protected, selectively writable store learn successive batches of arbitrary word translations, keep the earlier batches while later ones are learned,
and use the stored words through its own pathway (including inside sentences), with every store, buffer and cue source counted?

## Task

- **Alphabets.** Source S and target T, disjoint, drawn by a seeded stream from the filtered symbol pool (about 387 symbols after the distance filter), rendered with the existing variable renderer (font fixed per symbol; rotation, size, position, brightness and noise vary); separate training and evaluation renderings. Proposed sizes 150 each **[decide]**.
- **Lexicon facts.** Each source word (one symbol in v1) maps to one target word. Answers are scored against **all** target words (chance 1/|T|, about 0.007), not nine shared values, so each fact must be stored individually.
- **Batches.** The lexicon is taught in successive batches (proposed 6 batches of 20 words **[decide]**), each followed by a delay of other material (context episodes as before and translation of earlier words that are not replayed). One batch is "now"; earlier batches test durability.
- **Connection to something stored (the researchers' hypothesis).** Within each batch, half the new source words map to target words already learned in an earlier batch (*familiar target*: the new fact links a new key to an existing item), half to target words never seen (*new target*: both ends are new). A second hook, *synonym*: a new source word with the same translation as an already-known source word. Prediction to test, not assume: familiar-target and synonym facts are acquired faster and kept longer.
- **Sentences (structure through the system's own pathway).** Short source sentences (proposed three words) translate into target sentences under a fixed word-order rule (e.g. reversal) learned from known words. Test: sentences built from stored words of earlier batches, never seen as sentences. Success needs retrieval of each word and application of the learned order rule by the network, with no evaluator supplying the logic. **[decide: v1 words only, sentences in v2?]**

## System (closed, modular)

The decisions and computations belong to the evaluated system; fixed update rules and architectural constraints are allowed but each contribution is stated.

| Part | Role | Written by | Counted as |
|---|---|---|---|
| Host network (the standard network, image path, readout) | fast acquisition, perception, the order rule | ordinary training | the model |
| Durable store: a bounded addressed key-value memory (N slots, key and value of code width d) | durable lexicon | **consolidation only**; ordinary training cannot modify it (checked bitwise) | N · 2d floats |
| Recent cue buffer: a bounded, decaying store of recent inputs' codes | cues for internal replay | the system, as it experiences inputs (fixed rule: first-in, first-out with capacity C) | C · d floats |
| Consolidation | reads the recent buffer, queries the host for its current answer, writes key → value into the store; preserves earlier slots by addressed writes and an explicit eviction rule | a fixed schedule (every batch end) **[supplied; labeled]** | compute per write |
| Read and route | the store returns a value with a match confidence; a learned gate combines it with the host's answer into the system's answer | the gate learns from the ordinary training signal | inference cost |

What stays externally supplied in v1, labeled as such: the consolidation schedule, and optionally marking a subset for preservation **[decide: mark, or consolidate everything the buffer holds]**. Not supplied: which store answers (the gate does), the cues (the recent buffer), the comparison or order logic (the network).

**Interface drift.** Store keys are codes from a perception path that keeps learning, so old keys drift. v1 measures the damage (store recall on old batches with a changing encoder) and tests one mitigation: re-keying slots during consolidation from cues still in the recent buffer. A drift-free interface is not assumed; a frozen copy of the path is not allowed, since it does not scale.

**Capacity.** No fixed store keeps arbitrarily many independent facts. v1 makes the limit explicit: N smaller than the total lexicon in one condition, so the eviction rule matters, and a capacity curve over N.

## Comparisons

- `host`: host only, no store or buffer.
- `rehearsal-matched`: host plus a replay buffer of stored renderings with the **same bytes** as store plus cue buffer (the fair storage-matched baseline).
- `snapshot-per-batch`: a frozen copy per batch (growing storage), an upper reference that does not scale.
- `store`: the closed system above; variants with and without re-keying, and over N.
- Optional: a MEMIT-style closed-form write into one host layer instead of a separate store (facts written into existing weights; see prior work) **[decide]**.

## Measures and readings (to be prespecified in the contract)

- Retention of each batch after k later batches (forgetting curves by batch age), on evaluation renderings, full-vocabulary scoring.
- Acquisition speed of each new batch (does the system still learn).
- Familiar-target and synonym against new-target facts, on acquisition and retention (the connection hypothesis).
- Sentences from stored words (if included): word accuracy in position and whole-sentence accuracy.
- Routing: gate weight on store for old against recent words; dissociations: silencing the store should hurt old batches and not the current one; silencing the host's fast path the reverse.
- Capacity curve: retained facts against N; eviction outcomes.
- Interface drift: store recall with and without re-keying.
- Costs: persistent and temporary bytes per component, training and consolidation compute, inference overhead; storage-matched comparison against `rehearsal-matched`.

## Scaling (required)

- Store: O(N · d) storage; retrieval with product keys in O(√N · d) instead of O(N · d); a write touches O(1) slots, so interference between facts is bounded by addressing, not by the whole parameter set.
- Recent buffer: O(C · d), bounded and decaying by design; consolidation must keep pace with the arrival rate.
- Gate: O(d) per query.
- Capacity: fixed N forces selection, eviction or compression; growth (adding slots) is a separate condition, not assumed.
- Obstructions carried from the [internet-scale assessment](retention-remedies-v1-contract.md): bits per parameter; key drift under continued perception learning (tested here at small scale); answer spaces (here |T| = 150, still small); routing at scale (tested here by a learned gate); selection bookkeeping.

## Prior work (search 2026-10-31; abstracts only unless tagged)

- **Product-key memory layers** ([Lample et al. 2019](https://arxiv.org/abs/1907.05242)) **[abstract]**: a large structured memory layer with exact fast nearest-neighbor lookup by product keys adds parameters with little compute. Differs: trained end to end by ordinary gradients in language modeling, not written only by consolidation. Hypothesis: the store's addressing and retrieval cost.
- **Fast weights** ([Ba et al. 2016](https://arxiv.org/abs/1610.06258)) **[abstract]**; Hinton and Plaut 1987 **[secondary]**: weights changing faster than slow weights hold recent memories without storing activity copies. Hypothesis: an alternative to the recent cue buffer that would remove its storage, at the cost of enumerating cues from fast weights.
- **Mass-editing memory** ([Meng et al. 2022, MEMIT](https://arxiv.org/abs/2210.07229)) **[abstract]**: thousands of associations written directly into a transformer's weights. Differs: large pretrained language models, edits computed from known subject representations. Hypothesis: a closed-form write into host weights is the "facts in existing weights" alternative to a separate store.
- **Key-value and episodic memories** (memory networks, Weston et al. 2014; Sukhbaatar et al. 2015; neural episodic control, Pritzel et al. 2017; sparse distributed memory, Kanerva 1988) **[familiar]**; **complementary learning systems**, **pseudorehearsal and generative replay**, **distillation**: see [prior work](prior-work.md).
- What the study would add: successive consolidation into a bounded protected store with internally held cues, learned routing and storage-matched baselines, on arbitrary translations with a familiar-connection manipulation. The components are established.
- Known failure modes: key drift; gate collapse onto one source; eviction of useful facts; the recent buffer becoming the real memory (hence its bytes count and its own retention is measured).

## Open decisions

1. Alphabet sizes and batch schedule.
2. v1 words only, or words and sentences.
3. Marking a subset for preservation, or consolidating everything the recent buffer holds.
4. Store type: separate key-value store, MEMIT-style write into host weights, or both.
5. Recent cue buffer or fast weights for cues.
6. Arms: `mixed` only, or also `sparse48` and `mixed-h125`.
