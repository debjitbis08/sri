# Translation memory v1: design draft (not a contract)

Status: **draft 2026-10-31, revised after four external reviews and the [pilot](translation-memory-v1-pilot.md)**, for the researchers' review. Implemented; not frozen. Follows [retention remedies v1](retention-remedies-v1-results.md), whose claim was narrowed to
"some behavior can be transferred into protected weights under an externally managed procedure", and the external review's requirements for a closed memory system recorded there.
Builds on [kinds of memory and learned routing](memory-kinds-notes.md). No **[decide]** items remain (see *Open decisions*).

Revisions:

1. The flat key-value store is a **diagnostic reference**, not the system under test (researchers' decision). Also answered: what the cue buffer retains (no re-keying from stored codes), state-preserving evaluation and a no-exposure retention cohort, the gate's learning opportunity and a confidence-router comparison, exhaustive lookup with explicit allocation and eviction instead of a product-key claim, a fixed-encoder diagnostic, full accounting for the replay comparison, a minimal sentence probe in v1. Familiarity deferred.
2. The system under test is a **bounded sparse associative memory** (separation, completion, reinstatement), motivated, not validated, by the neuroscience findings in [prior work](prior-work.md). Allocated protected blocks become an established comparison.
3. Third review: the buffer holds one entry per word, so it spans batches; every write enters **one reference coordinate space**, fixed at the first consolidation and reached through a map fitted on stored anchor renderings; each event is **written once** with its teacher answer kept; a **phase schedule** per batch; **one exact associative rule** (clipped binary Hebbian, synchronous k-winners-take-all settling) with a **one-step control** on the same weights; an explicit store keyed by the same sparse patterns to isolate storage; overstated guarantees narrowed to hypotheses. The palimpsest rule and `blocks` wait until the single rule and its control are specified and piloted.
4. Proposed pilot settings (external, adopted as proposals, not evidence the configuration works): task size and schedule fixed so only the memory is tuned; C = 40 (two batches); 64 anchors with an identity-centred ridge map; a six-configuration memory grid selected by a storage-fidelity gate, not by recurrence advantage; a logistic gate and a calibrated confidence router; arms `mixed` and `sparse48`. See *Pilot and selection*. Correction: the map is unchanged and current during phases 2–4, not exact.
5. After the [pilot](translation-memory-v1-pilot.md) failed the storage gate and its controls traced the failure to keys that do not carry word identity across renderings or time (fourth review): an **identity objective** and a **dedicated key head**, and a **perception gate** before the memory grid. See *Identity revision*.

## Why a new task

The pairing task's facts were symbol → count: 45 facts shared nine answers the network already represents, so a store could succeed by learning a coarse code-to-count rule instead of
individual facts (the other-symbol cue result), and the only possible "connection to something stored" was the count. The researchers proposed a **translation task with random symbols**:
two invented alphabets drawn from the filtered Unicode pool, a lexicon of arbitrary word pairs, and sentences whose translation needs both the lexicon (memory) and a word-order rule (structure).

## Question

Can a bounded sparse associative memory inside the system keep successive batches of arbitrary word translations in shared connections, retrieve them from imperfect perception,
and reinstate them where the network's own sentence computation uses them, with every store, buffer and cue source counted? Does recurrent completion add anything over one-step retrieval from the same weights, and how does the memory compare with explicit stores on capacity, noisy-cue recall, successive learning, drift and measured computation?

## Task

- **Alphabets.** Source S and target T, disjoint, drawn by a seeded stream from the filtered symbol pool (about 387 symbols after the distance filter), rendered with the existing variable renderer (font fixed per symbol; rotation, size, position, brightness and noise vary); separate training and evaluation renderings. 150 symbols each; the 30 target words never used in v1's lexicon remain distractors in full-vocabulary scoring. 32 of the remaining pool symbols (about 87) supply the anchor set and are never lexicon words.
- **Lexicon facts.** Each source word (one symbol in v1) maps to one target word. Answers are scored against **all** target words (chance 1/|T|, about 0.007), so each fact must be stored individually. In v1 every source word maps to a distinct target word that no earlier word uses (no synonyms; see *Deferred*).
- **Batches.** Six batches of 20 words, 2,000 acquisition updates each (the earlier experiments' update scale), each update carrying 16 word trials and 4 context episodes (the [arbitrary-pairings](arbitrary-pairings-v1-contract.md) mix), identical across arms, with actual exposures recorded per word. After batch 6, 2,000 context-only updates. A word is presented externally only during its own batch (word trials and its own batch's sentences). Afterwards the only external material is unrelated: context episodes and later batches.
- **Retention cohort and clocks.** "No further external presentation" is not "no further internal use": a word still in the buffer trains the gate (see *Gate*). Every retention reading is on words past their batch, and records three ages: updates since the last external presentation, since the word's single memory write, and since it left the buffer. The **primary** retention reading is on words that have left the buffer (at the end: batches 1–4); batches 5–6, still buffered, are reported separately as the recent-memory cohort.
- **Evaluation is state-preserving.** No gradient, optimizer step, buffer insertion, memory write, map refit or running-statistic update. Checked by hashing the full system state before and after each evaluation.
- **Sentences (minimal probe, kept in v1).** Three-word source sentences translate under one fixed word-order rule (reversal). Each batch's sentence phase trains the order rule on sentences of that batch's words only. Test: sentences whose **combinations** were never seen, built from words of earlier batches (each word did take part in its own batch's sentence training). The claim is narrow: composition of remembered words with a learned permutation, not general relational reasoning.

## System under test

The decisions and computations belong to the evaluated system; fixed update rules and architectural constraints are allowed but each contribution is stated.

Pathway: image → host code → **reference map** → **separation** (sparse source pattern) → **retrieval** (settling) → reinstated **target pattern** → answer by nearest target pattern; the routed target pattern feeds the sentence rule.

| Part | Role | Written by | Counted as |
|---|---|---|---|
| Host network (the standard network, image path, readout) | fast acquisition, perception | ordinary training (acquisition phase) | the model |
| Anchor set: K renderings of non-lexicon symbols, each with its code in the reference space | fixes one reference coordinate space | stored once, at the first consolidation | K · (rendering + d) |
| Reference map: a d × d linear map, ridge-regularized | maps current codes into the reference space | refitted at each consolidation on the anchors only | d² floats |
| Separation: a fixed random expansion of the mapped code to M_s units, top k_s kept | distinct sparse source patterns | fixed at initialization | a seed (regenerable) |
| Target codebook: one fixed sparse pattern (M_t units, k_t active) per target index | the reinstated content | fixed at initialization | a seed |
| Associative weights: binary, over the joint pattern [source; target] (M = M_s + M_t), no self-connections | durable lexicon in shared connections | **consolidation only**; unreachable by every other phase (checked bitwise) | M² bits (fewer with sparse connectivity) |
| Cue buffer: C entries, **one per word** | write cues; gate examples | the system, during acquisition | C · (rendering + label + teacher answer + entry step + written flag) |
| Gate | mixes host and memory answers | gate phase only | a few parameters |
| Sentence rule | maps three routed target patterns to three output positions | sentence phase only | small |
| Consolidation | writes each unwritten buffer entry once | fixed schedule (batch end) **[supplied; labeled]** | compute per write |

### Buffer policy

An entry is inserted on a word's **first** presentation in its batch; later presentations of the same word do not insert (the same word is recognized by its label, which the system has during acquisition; this use of supervision is labeled). Capacity C counts **words**, one rendering each. When full, the oldest entry leaves (first in, first out by insertion). C = 40, exactly two batches: at the end of batch b the buffer holds all 20 words of batch b and all 20 of batch b − 1, so the gate trains on equal current and previous cohorts. No insertions happen in the final context-only phase.

### Reference coordinates

The reference space is the host's code space at the **first consolidation** (end of batch 1). At that moment the K anchor renderings are encoded and their codes stored. At every later consolidation the anchors are encoded with the current encoder and a linear map A is fitted from current codes X to the stored anchor codes Y; it replaces the previous map. There is no composition of rolling maps, so error does not accumulate through a chain; it is bounded by how well one linear map fits the drift since batch 1, and by whether anchor drift represents lexicon drift (known failure: anchors cover a different part of code space). K = 64 anchors (32 non-lexicon symbols × 2 renderings).

The fit is **identity-centred ridge**, correcting only the drift the anchors support:

$$\min_A \; \frac{\|AX - Y\|_F^2}{K\, s^2} + \lambda\, \frac{\|A - I\|_F^2}{d},$$

with s² the reference-code scale: the mean squared entry of the reference anchor codes, computed once at the first consolidation and fixed. λ is chosen once, in the pilot (see *Pilot and selection*), then frozen; main runs fit all 64 anchors. Lexicon alignment is an evaluation outcome, never a tuning target.

Every write uses a **freshly encoded** buffered rendering, mapped into the reference space: x_ref = A_t · enc_t(x). Every read maps the same way with the latest map. Batch-1 writes use the identity. The explicit-store references use the same mapped codes. The `no-map` condition uses raw current codes throughout.

### Write once

Each buffer entry is written **exactly once**, at the first consolidation after its insertion. The teacher answer is the host's argmax at that moment; it is recorded in the entry and never recomputed. Entries still in the buffer at later consolidations are not rewritten, so host forgetting cannot write a wrong target over a correct association, and write corruption stays separate from storage interference. Revision (rewriting a stored association) is a later condition.

### Associative rule (one rule in v1)

- **Storage (clipped binary Hebbian, Willshaw-type).** For a joint pattern p = [s; t] (binary, k_s + k_t active), set W_ij = 1 for every i ≠ j with p_i = p_j = 1. Weights never return to 0. Self-connections are 0.
- **Recurrent retrieval.** Start: source part = the cue pattern s̃ (separated mapped code), target part = 0. Each step, synchronously: h = W x; the new source part is the top k_s units of h over the source units, the new target part the top k_t over the target units; ties broken by a fixed seeded unit priority. The source part is **not clamped** after the first step, so a noisy cue can be completed. Stop when the state repeats: a fixed point counts as converged; a cycle (synchronous updates can produce period-2 cycles) is flagged non-converged, and the answer is read from the last state. Cap S from the pilot. Every read runs to 16 steps and records checkpoints 1, 2, 4, 8 and 16, so the recurrence comparison never depends on the chosen cap; the cap sets only the system's answer and its counted compute.
- **One-step control (same weights).** Target part = top k_t of W_ts s̃, one step, no recurrence. Because the target part starts empty, this is the target part of checkpoint 1. Recurrence is credited only where later checkpoints beat it, primarily on noisy cues.
- **Answer.** The target index whose codebook pattern has the largest overlap with the retrieved target part; confidence is that overlap and its margin to the second.
- **Capacity.** Not sized from theory: capacity depends on the retrieval strategy and coding assumptions (Graham and Willshaw 1994, cited by the reviewer), so M, k_s and k_t are chosen by the pilot storage gate (see *Pilot and selection*), with the selection cost reported. The clipped rule overloads abruptly; a capacity condition shows where. The palimpsest rule (bounded weights with decay) is a later condition, and its graceful forgetting is a hypothesis to test, not a property of bounded weights.

### Gate

A single logistic unit, no hidden layer. Inputs (five scalars): the host's maximum softmax probability and normalized entropy, the memory's normalized overlap (best overlap / k_t), its margin to the second, and convergence; output: a mixing weight w. The routed target pattern for the sentence rule is w · (memory target part) + (1 − w) · (host's softmax-weighted codebook patterns). Learning opportunity: the gate phase trains the gate alone on buffered entries with their recorded labels; older entries (the whole previous batch) are where the host and the memory disagree. Compared against a **confidence router** with no learned parameters: the memory answers when its normalized overlap reaches a threshold from {0, 0.25, 0.5, 0.75, 1}, or the host always answers. The threshold is chosen in the pilot and frozen.

### Phase schedule per batch

| Phase | Material | Updates | Frozen |
|---|---|---|---|
| 1. Acquisition | batch b word trials, context episodes | host parameters; buffer insertions | memory, map, gate, sentence rule |
| 2. Consolidation | buffer (unwritten entries), anchors | reference map; associative weights (one write per new entry); recorded teacher answers | host, gate, sentence rule |
| 3. Gate | buffer entries and their labels | gate | everything else |
| 4. Sentences | sentences of batch b words, read through the memory just written | sentence rule | everything else |
| 5. Evaluation | evaluation renderings, all batches; held-out sentence combinations | nothing (state hash checked) | all |

Phase 4 follows phase 2, so current-batch sentences read reinstated targets of associations already written. Because the host encoder is frozen in phases 2–4, the map fitted in phase 2 stays current through them; its fitting error and any failure to generalize from anchors to lexicon words remain. After batch 6, a final block of 2,000 context-only updates (phase 1 with no word trials and no buffer insertions) precedes the last evaluation.

### Pilot and selection

Proposed settings, not evidence that the configuration works. Task size, schedule and buffer are fixed above, so only the memory, the map and the router are selected. Pilot seeds: **299** for every selection, **298** for the confirmation; main runs: **fresh seeds 360–369** (ten). Every selection's cost is reported.

1. **Map regularization.** λ ∈ {0.001, 0.01, 0.1, 1}, chosen by held-out anchor prediction error on pilot seeds with a **symbol-level split** (both renderings of a symbol on the same side). Then frozen.
2. **Memory grid.** M_s = M_t ∈ {256, 512, 1,024}, activity per part about 2% or 4% (k = 5 or 10, 10 or 20, 20 or 41): six configurations. Tested first on exact stored cues, then on independent renderings and controlled corruption. Selected: the **smallest configuration recovering the written target on at least 95% of exact cues**, after all 120 writes. The step cap: the smallest checkpoint within one percentage point of 16-step recall on independent renderings (on exact cues it would trivially be 1). The selection is confirmed on a separate pilot seed. Never selected for the largest recurrence advantage, which is the experiment's question. If no configuration passes the storage gate, the design is revised before freezing.
3. **Router threshold.** Chosen on pilot batches 2–3 (batch 1 has no forgetting to calibrate against), on independent renderings, balancing current and previous words. Calibrated separately for each retrieval method (recurrent, one-step, `flat-sparse`, `flat-store`) under the same budget, since their confidence distributions differ. Frozen before main runs.

### Identity revision (after the pilot)

- **Key head.** A linear head (64 → 32) on the image encoder's output, unit-norm output; the memory's keys, anchors and map use it (map 32 × 33). The host's matching code is unchanged except through the shared encoder.
- **Identity objective.** Each acquisition update adds a supervised contrastive loss (Khosla et al. 2020, temperature 0.1) over two training renderings of each trial's source word (the trial's own and a second from a separate stream; 32 views, positives = same word), weight 1. It trains the key head **and the encoder**, so every comparison (host only, explicit stores, frozen encoder) gets the same change, and its compute counts. Only words being taught are in the loss; whether identity survives for earlier batches is the open question.
- **Perception gate (set before the identity results were seen).** On the pilot seeds, both arms pooled, with the nearest historical single-rendering key (the explicit dense lookup the memory's keys allow at best): **recognition of the word from independent evaluation renderings encoded at the word's write point at least 0.90**, and **at the end of batch 6 at least 0.80**, both pooled over batches and reported by batch age; after the final context block reported only. Without the identity objective the pilot gives 0.24 and 0.11 (`reports/translation-memory-v1-pilot-controls.json`, recognition). If the gate passes, the memory grid and the rest of the selection run as specified on the identity host; if not, the design is revised again before freezing.

What stays externally supplied, labeled as such: the phase schedule, the router threshold, the fixed separation and codebook, the use of labels to deduplicate buffer entries. Not supplied: which source answers (the gate), the cues (the buffer), retrieval (settling), the word order (the network).

## Comparisons

- `host`: host only.
- `one-step`: the system under test read by the one-step control (same weights, same writes); a read mode, not a separate run.
- `flat-sparse` (storage control): an explicit slot per write, keyed by the **same separated source pattern** with the same codebook value; read by largest overlap over all slots. Same buffer, map, writes and gate. It isolates shared-connection storage from slot storage. Allocation: free slot; duplicate (overlap above τ, same value): refresh; eviction when full: least used, ties oldest.
- `flat-store` (system comparison): an explicit slot keyed by the mapped dense code (no separation), exhaustive O(N · d) lookup, the same allocation rules. It changes representation and retrieval as well as storage. A softmax attention over stored patterns (modern Hopfield networks) is in this class.
- `no-map`: raw current codes everywhere.
- `frozen-encoder` (diagnostic): image path frozen after batch 1. It removes drift but also changes acquisition of later batches, so a difference bounds drift's contribution rather than isolating it.
- `rehearsal-matched`: replay of stored renderings and labels, matched on persistent bytes and on replay exposures and compute, with labels, metadata and optimizer state counted on both sides.
- `snapshot-per-batch`: a frozen copy per batch, upper reference.
- Later conditions (after the single rule and control are piloted): palimpsest rule, `blocks` (allocated protected units, PackNet-style), revision writes, writes at experience, perceptual target codes, MEMIT-style writes, fast-weight cues, product-key addressing.

## Implementation (2026-10-31; details the draft left open)

Code: `experiments/translation_memory.py` (host stage), `experiments/translation_store.py` (memory components; checks in `experiments/translation_store_checks.py`), `experiments/translation_memory_pass.py` (memory stage and pilot selection).

- **Two stages.** The host never reads the memory and is frozen in phases 2–4, so its trajectory is identical for every memory variant. The host stage trains it and saves its state at points 1–7; the memory stage replays those states through phases 2–5 for all variants at once (on the CPU). Variants are therefore paired on one host trajectory per arm and seed. Branches of the host stage: `main` and `frozen-encoder`. `rehearsal-matched` is **not yet implemented**: its matching rule (bytes and exposures) needs the selected memory size first.
- **Host.** Blocks 1–3 of the limited-memory curriculum, Option A schedule, as in the earlier studies. A word trial scores the source rendering against 32 target candidates (the update's distinct targets plus distractors from all 150 targets, random training renderings) with the existing matching head; cross-entropy. The host's answer for evaluation and teaching is the matching score against each target word's training rendering 0, over all 150 targets. Codes are the host's 16-dimensional code.
- **Map.** Affine: A is d × (d + 1), the bias column regularized toward 0 by the same term, so a uniform shift of the code space can be corrected. Folds for λ: 4 folds of 8 anchor symbols.
- **Separation.** The mapped code is centred on the reference anchors' mean and divided by s before the random expansion (an uncentred code would select the same units for every input).
- **Gate.** 300 full-batch Adam steps (rate 0.05) per consolidation on the buffer, continuing from the previous batch's gate. Memory scores enter as p_mem = softmax(10 · score).
- **Sentences.** Training: 200 sentences of the batch's words on training renderings, 200 Adam steps (rate 0.1) of the 3 × 3 mixing only. Test: 100 unseen combinations of old words on evaluation renderings per point. The mixing is shared across code units, so it generalizes to any word by construction; the probe tests composition with retrieval, not whether the rule could learn word-specific order.
- **Explicit stores.** `flat-sparse` duplicates: identical pattern; `flat-store` duplicates: cosine ≥ 1 − 10⁻⁶. Capacity 120 slots (no eviction in the main condition).
- **Extras (final point).** Corruption levels 0, 0.2, 0.4, 0.6 of the exact cue's active units; free recall from 200 random sparse starts, a stored pair counted as regenerated when the final target part's best normalized overlap is at least 0.8 with a written target.
- **Selection** runs on both arms pooled; ties in the router go to the earlier option in the order host, 0, 0.25, 0.5, 0.75, 1.

## Measures and readings (to be prespecified in the contract)

- Per word: **teacher accuracy at write** (from the recorded answer), **immediate memory accuracy** (memory alone, right after the write) and **retention** (memory alone and full system) by the three ages.
- Forgetting curves by batch age, full-vocabulary scoring, primary on words outside the buffer.
- Recurrent against one-step: recall on clean and corrupted cues (bits flipped in the source pattern; rendering noise); steps taken; convergence, cycles and spurious states.
- Storage: system under test against `flat-sparse` (same patterns) and `flat-store` (system comparison), on capacity, noisy-cue recall, successive learning, drift and computation.
- Drift: lexicon code change since write, in reference coordinates and raw; anchor map residual; recall with and without the map; the `frozen-encoder` bound.
- Acquisition speed of each batch.
- Capacity: pilot curve of the exact rule (recall against stored pairs, for each M, k_s, k_t); overload point.
- Free recall from random starts: stored pairs regenerated, spurious states (measurement only).
- Routing: gate weight by batch age; learned gate against confidence router. Causal tests: silencing the memory should hurt old batches and not the current one; silencing the host's answer branch (keeping the encoder that feeds the map) should do the reverse.
- Sentences: word accuracy by position and whole-sentence accuracy on unseen combinations of old words, against single-word accuracy on the same words. The fixed codebook keeps stored values constant; whether the sentence rule keeps using old codes correctly is measured (a rule trained only on recent words' patterns may not).
- Computation, counted as work done: map and separation, connections processed per settling step, steps taken, retained weights in bits, writes, gate and sentence phases; dense-implementation operations and active-connection counts both reported; a sparsity saving is claimed only if the implementation exploits it and wall time shows it. Selection (capacity pilot, router threshold) and tuning costs reported per [computation reporting](computation-reporting.md).

## Scaling (required)

- Associative weights: M² bits with full connectivity, M · c with fan-in c. Capacity of this rule at this coding is measured, not assumed; beyond it the clipped rule fails abruptly, which is the case for a forgetting rule later.
- Retrieval: map O(d²), separation O(M_s · d), settling O(S · M²) dense or O(S · active connections) if the implementation follows active units. No comparison against one row per fact. Not O(1): parallel activity is not constant total work.
- Answer readout: overlap with |T| codebook patterns, O(|T| · k_t); the answer-space obstruction at large vocabularies.
- `flat-sparse` and `flat-store`: storage and lookup linear in writes. Product keys would give sub-linear lookup only with shared sub-key codebooks ([prior work](prior-work.md)).
- Anchors O(K · (rendering + d)), buffer O(C · rendering), gate O(1): constant in facts. Whether K anchors keep fitting drift as the system grows is part of the drift question.
- Obstructions carried from the [internet-scale assessment](retention-remedies-v1-contract.md): bits per parameter (bits per associative weight); key drift (here against one anchored reference space); answer spaces (|T| = 150); routing at scale; selection bookkeeping.

## Deferred: familiarity and schema contrast

Three hypotheses to separate in a later contrast, after the core protocol works: **perceptual familiarity** (target symbols seen without association), an **existing association** (synonym), and a **schema** (prior learned structure that new pairs fit). Only the third matches the finding that existing knowledge speeds consolidation. Target exposure matched across conditions; batch 1 all new targets as the bootstrap.

## Prior work (search 2026-10-31; abstracts only unless tagged)

- **Neuroscience hints** (recorded with their limits in [prior work](prior-work.md)): pattern separation and completion (Neunuebel and Knierim 2014) **[read: summary]**; cortical reinstatement (Horner et al. 2015) **[read: excerpts]**; ripple suppression (Girardeau et al. 2009) **[abstract]**; schemas (Tse et al. 2007) **[abstract]**. Analogies and hypotheses only.
- **Retrieval strategy and capacity** ([Graham and Willshaw 1994](https://papers.neurips.cc/paper_files/paper/1994/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)) **[cited by review; not yet read]**: capacity of binary associative memories depends on the retrieval strategy; hence the pilot curve instead of a theoretical size.
- **Associative memories** (Willshaw et al. 1969; Hopfield 1982; Kanerva 1988; Marr 1971; palimpsests, Parisi 1986 and Amit and Fusi 1994; sparse random expansion with winner-take-all, Dasgupta et al. 2017) **[familiar; read before freezing]**. Modern Hopfield networks (Ramsauer et al. 2020) **[familiar]**: attention over stored patterns, in the `flat-store` class.
- **Complementary learning systems** (McClelland et al. 1995; Kumaran et al. 2016) **[familiar]**: roles partly reversed here (the host learns fast; the memory is written at batch end).
- **Drift compensation** (semantic drift compensation, Yu et al. 2020) **[familiar; read before freezing]**; the anchor map is a linear, anchor-based variant.
- **Product keys** ([Lample et al. 2019](https://arxiv.org/abs/1907.05242)) **[read: sections 3.1–3.2]**; **fast weights** (Ba et al. 2016) **[abstract]**; **MEMIT** (Meng et al. 2022) **[abstract]**; **parameter allocation** (PackNet, HAT, progressive networks) **[familiar]**; **gating** (Jacobs et al. 1991) **[familiar]**; **key-value and episodic memories** (Weston et al. 2014; Sukhbaatar et al. 2015; Pritzel et al. 2017) **[familiar]**.
- What the study would add: a bounded binary associative memory written once per event by consolidation in one anchored coordinate space, with recurrent retrieval tested against one-step retrieval on the same weights, routed by a learned gate against a confidence router, feeding a learned sentence rule, with storage isolated by a same-pattern explicit store and computation measured. The components are established.
- Known failure modes: overload of the clipped rule; spurious attractors and cycles; correlated codes defeating separation; anchor drift unrepresentative of lexicon drift; gate collapse; the buffer becoming the real memory (primary readings on words outside it); the sentence rule failing on old words even when single-word recall survives; sparsity savings that exist on paper only.

## Arms

The main comparison runs on `mixed` (continuity with the memory discussion) and `sparse48` (does the added associative memory need the host's dense component). `mixed-h125` is a prespecified follow-up: adding decay now would introduce another mechanism before the basic memory comparison is understood.

## Open decisions

None. Decided 2026-10-31 (researchers deferred to the drafter's judgement): 16 word trials and 4 context episodes per update; s² from the reference anchors; pilot seeds 299 and 298, main seeds 360–369. Next: the literature reads tagged *read before freezing*, then the pilot.
